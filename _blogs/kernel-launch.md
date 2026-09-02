---
title: "Reassessing Megakernel Headroom after CUDA Graphs"
title_zh: "CUDA Graph 之后 MegaKernel 收益空间的再评估"
date: 2026-07-10
permalink: /blog/kernel-launch/
description: "On a single H100, CUDA Graphs reduce the number of host launch API calls to 17–18 per decode step and lower the GPU bubble ratio to approximately 1%–3% for medium-to-large dense models. Therefore, for decode workloads compatible with CUDA Graphs, the headroom available to Megakernels solely from further reducing kernel count is limited."
---


**TL;DR.** On a single H100, CUDA Graphs reduce the number of host launch API calls to 17–18 per decode step and lower the GPU bubble ratio to approximately 1%–3% for medium-to-large dense models. Therefore, for decode workloads compatible with CUDA Graphs, the headroom available to Megakernels solely from further reducing kernel count is limited.

One of the central arguments for Megakernels is that reducing kernel count lowers launch and host-side overhead. We use Nsight Systems traces to quantify GPU activity, launch APIs, and GPU idle time, and ask three questions:

- How should these metrics be defined?
- How large are they across models and workloads?
- Once CUDA Graphs are available, how much theoretical headroom remains for Megakernels?

## 1. From Kernel Launch to GPU Bubble

For a single explicitly synchronized kernel whose stages do not overlap, end-to-end latency can be approximated as:

```text
E2E latency ≈ CPU launch + other launch-path overhead + GPU execution + host-side processing/synchronization
```

In a serving pipeline, CPU launch and GPU execution usually overlap asynchronously. End-to-end latency is therefore determined by the critical path and cannot be obtained by directly summing accumulated component times.

| Component | Definition |
| --- | --- |
| CPU launch overhead | Time spent by the host invoking `cuLaunchKernel`, `cudaLaunchKernel*`, or `cudaGraphLaunch*`, including argument validation, command construction, and enqueueing. |
| Other launch-path overhead | Time spent waiting, parsing, and dispatching commands in the host command buffer, interconnect, and GPU hardware queues. |
| GPU execution time | Useful GPU work such as GEMM, attention, and normalization. |
| Host-side processing/synchronization | Host work that can leave the GPU idle, including D2H copies, sampling, scheduling, metadata updates, and Python runtime execution. This study does not measure this component directly; `other_host_idle_ms` only denotes GPU-idle time outside counted launch API intervals and may also contain GPU-side queueing. |

<p align="center">
  <img src="/images/blog/kernel-launch/kernel_lifeline.drawio.png" alt="Kernel launch and synchronization latency decomposition" style="width: 100%; max-width: 900px; height: auto;">
</p>
<p align="center"><em>Figure 1. Latency decomposition across kernel launch, GPU execution, and host-side processing/synchronization.</em></p>

The CPU produces commands and the GPU consumes them. Their relative rates determine whether launch-path overhead can be hidden by computation:

- When the CPU supplies work more slowly than the GPU consumes it, the GPU idles between adjacent kernels. Launch, framework, and synchronization overhead then appears directly on the critical path. Figure 2 shows a representative gap during eager decode with Qwen3.5-27B.
- When the CPU supplies work faster than the GPU consumes it, queued kernels can overlap with ongoing GPU work. Long-running GEMM and attention kernels readily create this regime; command-buffer backpressure may even block the launch API. A long API duration in this case does not imply an equally large increase in end-to-end latency.

<p align="center">
  <img src="/images/blog/kernel-launch/decode_lifeline.png" alt="Nsight Systems timeline for eager decode" style="width: 100%; max-width: 1100px; height: auto;">
</p>
<p align="center"><em>Figure 2. During eager decode, the GPU waits for the host between adjacent kernels.</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/prefill_command_buffer_full.png" alt="Nsight Systems timeline with a full command buffer during prefill" style="width: 100%; max-width: 1100px; height: auto;">
</p>
<p align="center"><em>Figure 3. During long-prompt prefill, long-running GPU kernels hide substantial launch work and induce command-buffer backpressure.</em></p>

## 2. Measurement Methodology

### 2.1 Metrics

| Metric | Definition and interpretation |
| --- | --- |
| `e2e_ms` | End-to-end latency. |
| `gpu_busy_ms` | Union of all GPU activity intervals, including kernels, memcpy, and memset. Overlapping intervals are counted once, so `gpu_busy_ms <= e2e_ms`. |
| `gpu_bubble_ratio` | Fraction of the end-to-end interval with no GPU activity. It observes system-level idle time but does not by itself identify the causal contribution of launch, scheduling, framework execution, synchronization, or GPU queueing. |
| `unhidden_launch_api_ms` | Time spent in launch operations that is not hidden by GPU activity. |
| `other_host_idle_ms` | Host overhead outside kernel launch. Possible sources include D2H copies, sampling, scheduling, attention metadata planning, the Python runtime, and GPU-side queueing. |

The core relationships are:

```text
gpu_bubble_ms
  = e2e_ms - gpu_busy_ms
  = unhidden_launch_api_ms + other_host_idle_ms

gpu_bubble_ratio
  = gpu_bubble_ms / e2e_ms
```

Our Vast.ai experimental platform does not expose the required hardware counters, so we do not use SM utilization. This blog focuses on system-level idle time rather than the SM efficiency of individual kernels.

### 2.2 Experimental Setup

| Dimension | Configuration |
| --- | --- |
| GPU | Single H100 80 GB |
| Framework | SGLang + FlashInfer |
| Models | Qwen3-0.6B / 1.7B / 8B / 14B / 30B-A3B; Qwen3.5-27B |
| Execution modes | Eager; CUDA Graph |
| Prefill-dominant | BS=1, prompt ∈ {16, 256, 1k, 4k, 8k} |
| Decode | BS=1, prompt=16, decode ∈ {128, 512} |
| Batch decode | BS ∈ {1, 4, 8, 16}, prompt=16, decode ∈ {128, 512} |

## 3. Decode: CUDA Graphs Eliminate Most GPU Bubble

### 3.1 Short Context, BS=1

We first compare eager execution with CUDA Graphs at BS=1 and prompt=16. Figure 4 reports GPU bubble ratio for decode=512, while Figure 5 summarizes end-to-end speedup for decode=128/512.

<p align="center">
  <img src="/images/blog/kernel-launch/fig1_decode_bs1_bubble.png" alt="GPU bubble ratio for batch-one decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>Figure 4. GPU bubble ratio at BS=1, prompt=16, and decode=512.</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig2_decode_bs1_speedup.png" alt="End-to-end speedup from eager to CUDA Graph for batch-one decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>Figure 5. End-to-end eager → CUDA Graph speedup at BS=1, prompt=16, and decode=128/512.</em></p>

The tables below provide per-step metrics for decode=512. Each paired value is **eager → CUDA Graph (CG)**; all times are in milliseconds. Launch API calls are averages normalized by decode step, hence the single decimal place.

| Model | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 84.7% → 14.0% | 5.64× | 8.69 → 1.54 | 374.9 → 17.0 |
| Qwen3-1.7B | 76.3% → 8.6% | 3.81× | 8.73 → 2.29 | 374.9 → 17.0 |
| Qwen3-8B | 45.3% → 2.4% | 1.77× | 11.40 → 6.45 | 513.1 → 17.0 |
| Qwen3-14B | 28.9% → 2.0% | 1.37× | 15.03 → 10.99 | 568.2 → 17.0 |
| Qwen3-30B-A3B | 88.1% → 7.8% | 7.73× | 35.62 → 4.61 | 822.5 → 17.0 |
| Qwen3.5-27B | 54.8% → 1.6% | 2.21× | 43.33 → 19.63 | 975.5 → 18.0 |

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 7.36 → 0.22 | 1.164 → 0.077 | 6.20 → 0.14 |
| Qwen3-1.7B | 6.66 → 0.20 | 1.132 → 0.036 | 5.53 → 0.16 |
| Qwen3-8B | 5.17 → 0.16 | 0.929 → 0.014 | 4.24 → 0.14 |
| Qwen3-14B | 4.34 → 0.22 | 0.677 → 0.014 | 3.66 → 0.21 |
| Qwen3-30B-A3B | 31.37 → 0.36 | 3.050 → 0.066 | 28.32 → 0.29 |
| Qwen3.5-27B | 23.73 → 0.32 | 2.244 → 0.014 | 21.49 → 0.31 |

Our analysis yields the following conclusions:

1. During eager decode, `other_host_idle_ms` is substantially larger than `unhidden_launch_api_ms`. This indicates that the bottleneck is not the raw kernel-launch API time itself, but the scheduler, sampling, synchronization, and framework logic between launch calls that leaves the GPU waiting.
2. The eager-to-CUDA-Graph end-to-end speedup is substantial. Small-model and MoE decode are the most launch/host-bound and therefore achieve the largest speedups. CUDA Graphs provide less benefit for larger dense models, but the gains remain significant.
3. CUDA Graphs reduce host launch API calls from 375–976 to 17–18 per step and substantially lower GPU bubble time. For example, Qwen3-8B has only 2.4% GPU bubble after CUDA Graphs are enabled.
4. The CUDA Graph bubble falls to 1.6%–2.4% for dense models, while ultra-small models and MoE retain 7.8%–14.0%. This partially explains why most current Megakernels are evaluated on—and benefit—ultra-small models, and why work such as MegaMoE is meaningful.

**What about Megakernels?**

**Megakernels can certainly reduce kernel-launch overhead and CPU-launch overhead, but CUDA Graphs already perform well enough.**

1. **First, SGLang already hides kernel-launch overhead effectively behind kernel execution.**
2. **Second, after enabling CUDA Graphs, the number of host launch API calls per decode step is already sufficiently small, and the remaining unhidden kernel-launch and host-synchronization overhead is also sufficiently low.**

**At this point, abandoning CPU-side control and moving scheduling entirely into the GPU runtime to recover roughly 2% host-synchronization overhead may instead reduce serving-system performance by introducing more complex GPU scheduling and synchronization. The tradeoff is unlikely to be worthwhile.**

### 3.2 Long Context, BS=1

We next evaluate `bs1_p8k_d512`: the model generates 512 additional tokens after an 8k-token prompt, and only the decode segment is measured.

<p align="center">
  <img src="/images/blog/kernel-launch/fig6_decode_long_context_bubble.png" alt="GPU bubble ratio for long-context decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>Figure 6. GPU bubble ratio at BS=1, prompt=8k, and decode=512.</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig7_decode_short_vs_long_speedup.png" alt="CUDA Graph speedup for short- and long-context decode" style="width: 100%; max-width: 820px; height: auto;">
</p>
<p align="center"><em>Figure 7. Eager → CUDA Graph speedup for decode=512 with short and 8k-token contexts.</em></p>

Paired values remain **eager → CG**. All timing metrics are reported per step in milliseconds.

| Model | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 81.1% → 10.5% | 4.73× | 9.08 → 1.92 | 381.0 → 17.0 |
| Qwen3-1.7B | 71.9% → 6.7% | 3.33× | 8.93 → 2.68 | 381.0 → 17.0 |
| Qwen3-8B | 41.0% → 2.5% | 1.65× | 12.38 → 7.49 | 521.0 → 17.0 |
| Qwen3-14B | 10.6% → 2.7% | 1.09× | 13.63 → 12.48 | 577.0 → 17.0 |
| Qwen3-30B-A3B | 86.8% → 6.4% | 7.25× | 39.00 → 5.38 | 833.0 → 17.0 |
| Qwen3.5-27B | 42.5% → 1.3% | 1.76× | 36.08 → 20.55 | 979.0 → 18.0 |

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 7.37 → 0.20 | 1.132 → 0.048 | 6.24 → 0.15 |
| Qwen3-1.7B | 6.42 → 0.18 | 1.062 → 0.027 | 5.36 → 0.15 |
| Qwen3-8B | 5.07 → 0.19 | 0.927 → 0.021 | 4.14 → 0.17 |
| Qwen3-14B | 1.45 → 0.34 | 0.242 → 0.125 | 1.21 → 0.21 |
| Qwen3-30B-A3B | 33.86 → 0.35 | 3.269 → 0.064 | 30.60 → 0.28 |
| Qwen3.5-27B | 15.34 → 0.27 | 1.541 → 0.013 | 13.80 → 0.26 |

The results show that long-context KV-cache reads amortize part of the launch/host bubble, but do not eliminate it completely. For larger dense models, increasing the prompt from 16 to 8k makes decode more GPU-execution- or KV-read-bound. For example, the Qwen3-14B speedup falls from 1.37× to 1.09×, indicating that the incremental benefit of CUDA Graphs has already decreased substantially. Small models and MoE, however, remain clearly launch/host-bound: Qwen3-0.6B still achieves a 4.73× speedup, and Qwen3-30B-A3B still achieves 7.25×.

**Our claim remains unchanged: for CUDA Graph decode, even with a long decode context, CUDA Graphs can still substantially reduce kernel-launch and host-side overhead. Consequently, the headroom available to Megakernels from reducing kernel count to lower kernel-launch and host-side overhead is limited.**

### 3.3 Batch Sweep

Does a larger batch naturally hide launch and host-side overhead? Figure 8 reports the batch sweep for decode=128/512.

<p align="center">
  <img src="/images/blog/kernel-launch/fig4_decode_bs_sweep_speedup.png" alt="CUDA Graph speedup across decode batch sizes" style="width: 100%; max-width: 1000px; height: auto;">
</p>
<p align="center"><em>Figure 8. Eager → CUDA Graph speedup for decode=128/512 across batch sizes.</em></p>

The following tables report per-step metrics at BS=8, prompt=16, and decode=512; each paired value is **eager → CG**.

| Model | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 85.4% → 12.9% | 6.00× | 10.15 → 1.69 | 402.9 → 17.0 |
| Qwen3-1.7B | 76.0% → 8.8% | 3.79× | 9.39 → 2.48 | 402.9 → 17.0 |
| Qwen3-8B | 44.8% → 3.4% | 1.75× | 11.98 → 6.84 | 549.1 → 17.0 |
| Qwen3-14B | 15.8% → 2.4% | 1.16× | 13.30 → 11.42 | 608.3 → 17.0 |
| Qwen3-30B-A3B | 87.9% → 8.5% | 7.57× | 39.06 → 5.16 | 870.5 → 17.0 |
| Qwen3.5-27B | 55.5% → 0.5% | 2.28× | 47.79 → 20.95 | 1135.5 → 18.0 |

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 8.67 → 0.22 | 1.404 → 0.060 | 7.27 → 0.16 |
| Qwen3-1.7B | 7.13 → 0.22 | 1.245 → 0.040 | 5.88 → 0.18 |
| Qwen3-8B | 5.37 → 0.23 | 1.003 → 0.020 | 4.37 → 0.21 |
| Qwen3-14B | 2.10 → 0.27 | 0.354 → 0.016 | 1.75 → 0.25 |
| Qwen3-30B-A3B | 34.32 → 0.44 | 3.366 → 0.080 | 30.95 → 0.36 |
| Qwen3.5-27B | 26.53 → 0.10 | 2.779 → 0.004 | 23.75 → 0.09 |

The BS=8 results are broadly consistent with BS=1 at decode=512: for decode, increasing the batch size does not eliminate the value of CUDA Graphs.

## 4. Connection to MPK End-to-End Results

Earlier TPOT comparisons among MPK, vLLM, and SGLang are consistent with the profiling results above.

For decode, MPK does not outperform vLLM or SGLang with CUDA Graphs in our measurements. This is consistent with our observation that CUDA Graphs already reduce kernel-launch and host-side overhead effectively.

<p align="center">
  <img src="/images/blog/kernel-launch/mpk_tpot_all_models.png" alt="TPOT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>Figure 9. Decode TPOT for MPK, vLLM, and SGLang.</em></p>

## 5. MoE: CUDA Graphs Leave a Larger GPU Bubble

Under CUDA Graphs, the residual decode bubble of Qwen3-30B-A3B consistently remains in the 6.7%–9.8% range. We have not investigated the root cause in depth, but two plausible explanations are:

- The execution structure inside the graph differs: routing, dispatch, expert computation, and gather form longer dependency chains, and scheduling or synchronization between nodes may create GPU-idle gaps.
- A3B indicates that only about 3B parameters are active, so its memory traffic and compute volume are also closer to those of an ultra-small model.

| Workload | Case | BS | Prompt | Decode | GPU bubble (%) | GPU bubble/request (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BS=1 decode | bs1_p16_d128 | 1 | 16 | 128 | 9.4% | 56.92 |
| BS=1 decode | bs1_p16_d512 | 1 | 16 | 512 | 8.0% | 190.17 |
| BS=1 decode | bs1_p8k_d512 | 1 | 8192 | 512 | 6.7% | 184.23 |
| Batch decode | bs4_p16_d128 | 4 | 16 | 128 | 9.4% | 62.37 |
| Batch decode | bs4_p16_d512 | 4 | 16 | 512 | 9.0% | 235.16 |
| Batch decode | bs8_p16_d128 | 8 | 16 | 128 | 9.8% | 66.03 |
| Batch decode | bs8_p16_d512 | 8 | 16 | 512 | 8.8% | 232.38 |
| Batch decode | bs16_p16_d128 | 16 | 16 | 128 | 9.6% | 67.56 |
| Batch decode | bs16_p16_d512 | 16 | 16 | 512 | 8.7% | 245.46 |

## 6. Conclusion

**Restricting our assessment to Megakernel's core motivation that “reducing kernel count lowers launch and host-side overhead,” we find that this motivation does not hold.**

The central conclusion is that, for decode, CUDA Graphs reduce host launch API calls to 17–18 per step and lower GPU bubble to approximately 1%–3% for medium-to-large dense models. **Megakernels can certainly reduce kernel-launch overhead and CPU-launch overhead, but CUDA Graphs already perform well enough:**

1. **First, SGLang already hides kernel-launch overhead effectively behind kernel execution.**
2. **Second, after enabling CUDA Graphs, the number of host launch API calls per decode step is already sufficiently small, and the remaining unhidden kernel-launch and host-synchronization overhead is also sufficiently low.**

**At this point, abandoning CPU-side control and moving scheduling entirely into the GPU runtime to recover roughly 2% host-synchronization overhead may instead reduce serving-system performance by introducing more complex GPU scheduling and synchronization. The tradeoff is unlikely to be worthwhile.**

In addition, this evaluation is limited to a single GPU: all results were obtained on one H100 and have not been extended to multi-GPU execution. In theory, single-GPU serving of small models is precisely the scenario with the largest GPU bubble. Multi-GPU serving typically involves larger models and more computation, while GPU bubbles are often caused by communication rather than launch overhead.

## 7. Implications

**The core differentiating value of Megakernels does not lie on the launch path.** We attribute it to:

- communication–computation fusion;
- weight prefetching;
- fine-grained pipelining across kernel boundaries and on-chip residency.

We will explore these directions in future experiments.

<hr class="blog-lang-split">


**TL;DR.** 在单卡 H100 上，CUDA Graph 将每个 decode step 的 host launch API 调用数降至 17–18 次，并把中大型 dense model 的 GPU bubble 压到约 1%–3%。因此，对适用 CUDA Graph 的 decode workload，MegaKernel 仅靠进一步减少 kernel 数量，收益上限有限。

MegaKernel 的一个核心论点是：减少 kernel 数量可以降低 launch 与 host-side 开销。我们用 Nsight Systems trace 量化 GPU activity、launch API 以及 GPU idle，并回答三个问题：
- 这些指标应如何定义？
- 它们在不同模型与 workload 中有多大？
- 在 CUDA Graph 已经可用时，MegaKernel 还剩多少理论空间？

## 1. 从 Kernel Launch 到 GPU Bubble

对单个、显式同步且各阶段未重叠的 kernel，可将端到端延迟近似分解为：

```text
E2E latency ≈ CPU launch + other launch-path overhead + GPU execution + host-side processing/synchronization
```

在实际 serving pipeline 中，CPU launch 与 GPU execution 通常异步重叠，因此端到端时间由 critical-path 决定，不能直接把各项累计时间相加。

| 组成部分                                 | 定义                                                                                                                                                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU launch overhead                  | Host 调用 `cuLaunchKernel`、`cudaLaunchKernel*` 或 `cudaGraphLaunch*`，并完成参数检查、命令封装与入队的时间。                                                                                   |
| Other launch-path overhead           | 命令在 host command buffer、互连与 GPU 硬件队列中的等待、解析和分派时间。                                                                                                                       |
| GPU execution time                   | GEMM、attention、normalization 等有效 GPU 工作。                                                                                                                                |
| Host-side processing/synchronization | D2H copy、sampling、scheduler、metadata 更新及 Python runtime 等可能使 GPU 空闲的 host 工作。本文不直接测量这一项；`other_host_idle_ms` 仅表示未被 launch API 区间覆盖的 GPU idle，其中还可能包含 GPU-side queueing。 |

<p align="center">
  <img src="/images/blog/kernel-launch/kernel_lifeline.drawio.png" alt="Kernel launch and synchronization latency decomposition" style="width: 100%; max-width: 900px; height: auto;">
</p>
<p align="center"><em>图 1. Kernel launch、GPU execution 与 host-side processing/synchronization 的延迟分解。</em></p>

CPU 是命令生产者，GPU 是消费者。两者的相对速率决定 launch-path 开销能否被计算隐藏：
- 当 CPU 供给慢于 GPU 消费时，GPU 会在相邻 kernel 间空转，launch、framework 与同步开销直接暴露在关键路径上。图 2 展示了 Qwen3.5-27B 在 eager decode 中的典型空隙。
- 当 CPU 供给快于 GPU 消费时，队列中的后续 kernel 可与当前 GPU 工作重叠。长时 GEMM/attention 容易形成这种状态；command buffer 甚至会因反压而阻塞 launch API。此时较长的 API 持续时间不等于同等大小的端到端损失。

<p align="center">
  <img src="/images/blog/kernel-launch/decode_lifeline.png" alt="Nsight Systems timeline for eager decode" style="width: 100%; max-width: 1100px; height: auto;">
</p>
<p align="center"><em>图 2. Eager decode 中，GPU 在相邻 kernel 之间等待 host 供给。</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/prefill_command_buffer_full.png" alt="Nsight Systems timeline with a full command buffer during prefill" style="width: 100%; max-width: 1100px; height: auto;">
</p>
<p align="center"><em>图 3. 长 prompt prefill 中，长时 GPU kernel 隐藏了大量 launch 工作，并触发 command-buffer backpressure。</em></p>

## 2. 测量方法

### 2.1 指标

| 指标                       | 定义与解释                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `e2e_ms`                 | 端到端时延                                                                                                                           |
| `gpu_busy_ms`            | Kernel、memcpy、memset 等全部 GPU activity 时间区间的并集；重叠区间只计一次，`gpu_busy_ms <= e2e_ms`。                                                 |
| `gpu_bubble_ratio`       | 端到端窗口内没有 GPU activity 的比例。它观测 system-level idle，但不单独识别 launch、scheduler、framework、同步或 GPU queueing 的因果贡献。                       |
| `unhidden_launch_api_ms` | Launch 操作未被 GPU activity 掩盖的时间。                                                                                                 |
| `other_host_idle_ms`     | Host 在 kernel launch 之外的其他开销，D2H copy、sampling、scheduler、attention metadata planning、Python runtime 与 GPU-side queueing 都是可能来源。 |

核心关系为：

```text
gpu_bubble_ms
  = e2e_ms - gpu_busy_ms
  = unhidden_launch_api_ms + other_host_idle_ms

gpu_bubble_ratio
  = gpu_bubble_ms / e2e_ms
```

我们使用的 Vastai 实验平台不开放所需的硬件计数器，因此未采用 SM utilization；本 blog 关注的也是 system-level idle，而非单个 kernel 的 SM 效率。

### 2.2 实验设置

| 维度               | 配置                                                 |
| ---------------- | -------------------------------------------------- |
| GPU              | 单卡 H100 80 GB                                      |
| Framework        | SGLang + FlashInfer                                |
| Models           | Qwen3-0.6B / 1.7B / 8B / 14B / 30B-A3B；Qwen3.5-27B |
| Execution modes  | Eager；CUDA Graph                                   |
| Prefill-dominant | BS=1，prompt ∈ {16, 256, 1k, 4k, 8k}                |
| Decode           | BS=1，prompt=16，decode ∈ {128, 512}                 |
| Batch decode     | BS ∈ {1, 4, 8, 16}，prompt=16，decode ∈ {128, 512}   |


## 3. Decode：CUDA Graph 消除了大部分 GPU Bubble

### 3.1 短上下文，BS=1

首先比较 BS=1、prompt=16 下的 eager 与 CUDA Graph。图 4 给出 decode=512 的 GPU bubble ratio；图 5 汇总 decode=128/512 的端到端加速比。

<p align="center">
  <img src="/images/blog/kernel-launch/fig1_decode_bs1_bubble.png" alt="GPU bubble ratio for batch-one decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>图 4. BS=1、prompt=16、decode=512 的 GPU bubble ratio。</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig2_decode_bs1_speedup.png" alt="End-to-end speedup from eager to CUDA Graph for batch-one decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>图 5. BS=1、prompt=16、decode=128/512 的 eager → CUDA Graph 端到端加速比。</em></p>

下面展开 decode=512 的每步指标。成对数值均为 **eager → CUDA Graph (CG)**；时间单位为 ms。Launch API 调用数按 decode step 归一化取平均，因此保留一位小数。

| Model         | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| ------------- | -------------: | ----------: | --------------: | -------------------------: |
| Qwen3-0.6B    |  84.7% → 14.0% |       5.64× |     8.69 → 1.54 |               374.9 → 17.0 |
| Qwen3-1.7B    |   76.3% → 8.6% |       3.81× |     8.73 → 2.29 |               374.9 → 17.0 |
| Qwen3-8B      |   45.3% → 2.4% |       1.77× |    11.40 → 6.45 |               513.1 → 17.0 |
| Qwen3-14B     |   28.9% → 2.0% |       1.37× |   15.03 → 10.99 |               568.2 → 17.0 |
| Qwen3-30B-A3B |   88.1% → 7.8% |       7.73× |    35.62 → 4.61 |               822.5 → 17.0 |
| Qwen3.5-27B   |   54.8% → 1.6% |       2.21× |   43.33 → 19.63 |               975.5 → 18.0 |

| Model         | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| ------------- | -------------------: | ----------------------------: | ---------------------------------: |
| Qwen3-0.6B    |          7.36 → 0.22 |                 1.164 → 0.077 |                        6.20 → 0.14 |
| Qwen3-1.7B    |          6.66 → 0.20 |                 1.132 → 0.036 |                        5.53 → 0.16 |
| Qwen3-8B      |          5.17 → 0.16 |                 0.929 → 0.014 |                        4.24 → 0.14 |
| Qwen3-14B     |          4.34 → 0.22 |                 0.677 → 0.014 |                        3.66 → 0.21 |
| Qwen3-30B-A3B |         31.37 → 0.36 |                 3.050 → 0.066 |                       28.32 → 0.29 |
| Qwen3.5-27B   |         23.73 → 0.32 |                 2.244 → 0.014 |                       21.49 → 0.31 |

我们分析得到以下结论：
1. Eager decode 中，`other_host_idle_ms` 显著大于 `unhidden_launch_api_ms`，这说明系统瓶颈不是裸 kernel launch API 时间本身，而是 launch API 之间的 scheduler / sampling / sync / framework 逻辑让 GPU 等待。
2. 从 Eager 到 CUDA Graph，e2e speedup 非常显著，小模型和 MoE decode 最 launch/host-bound，因此 speedup 最大；更大的 dense model 下使用 CUDA Graph 的收益会降低，但依然足够显著。
3. CUDA Graph 将每步 launch API 调用数从 375–976 次降至 17–18 次，CUDA Graph 显著降低了 GPU bubble time。以 Qwen3-8B 为例，开启 CUDA Graph 后的，GPU bubble 仅有 2.4%。
4. Dense model 的 CG bubble 已降至 1.6%–2.4%，但超小模型和 MoE 仍有 7.8%–14.0%。这在一定程度上印证了为什么现在大多数 MegaKernel 仅对超小模型测试并起作用，以及为什么 MegaMoE 这类工作是有意义的。


**What about MegaKernel？**
**MegaKernel 当然可以节省 Kernel Launch Overhead / CPU Launch Overhead，但是 CUDA Graph 已经做的足够好。**
1. **一方面，SGLang 本身就让 Kernel Launch 很好地被 kernel 的执行所掩盖；**
2. **另一方面，使用 CUDA Graph 之后，单次 decode 所启动的 kernel 数量已经足够少了，未能掩盖的 Kernel Launch 以及 host 同步开销也已经足够小了。**
**此时，为了 2% 级别的 host 同步开销，进而将 CPU 侧控制逻辑放弃，完全进入 GPU Runtime 调度，反而可能因为引入更复杂的 GPU 调度/同步逻辑，导致 Serving System 的性能下降，这是得不偿失的。**

### 3.2 长上下文，BS=1

接着考察 `bs1_p8k_d512`：prompt=8k 后继续生成 512 个 token，并只统计 decode 段。

<p align="center">
  <img src="/images/blog/kernel-launch/fig6_decode_long_context_bubble.png" alt="GPU bubble ratio for long-context decode" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>图 6. BS=1、prompt=8k、decode=512 的 GPU bubble ratio。</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig7_decode_short_vs_long_speedup.png" alt="CUDA Graph speedup for short- and long-context decode" style="width: 100%; max-width: 820px; height: auto;">
</p>
<p align="center"><em>图 7. 短上下文与 8k 长上下文 decode=512 的 eager → CUDA Graph 加速比。</em></p>

成对数值仍为 **eager → CG**，时间指标均按 step 统计，单位为 ms。

| Model | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 81.1% → 10.5% | 4.73× | 9.08 → 1.92 | 381.0 → 17.0 |
| Qwen3-1.7B | 71.9% → 6.7% | 3.33× | 8.93 → 2.68 | 381.0 → 17.0 |
| Qwen3-8B | 41.0% → 2.5% | 1.65× | 12.38 → 7.49 | 521.0 → 17.0 |
| Qwen3-14B | 10.6% → 2.7% | 1.09× | 13.63 → 12.48 | 577.0 → 17.0 |
| Qwen3-30B-A3B | 86.8% → 6.4% | 7.25× | 39.00 → 5.38 | 833.0 → 17.0 |
| Qwen3.5-27B | 42.5% → 1.3% | 1.76× | 36.08 → 20.55 | 979.0 → 18.0 |

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 7.37 → 0.20 | 1.132 → 0.048 | 6.24 → 0.15 |
| Qwen3-1.7B | 6.42 → 0.18 | 1.062 → 0.027 | 5.36 → 0.15 |
| Qwen3-8B | 5.07 → 0.19 | 0.927 → 0.021 | 4.14 → 0.17 |
| Qwen3-14B | 1.45 → 0.34 | 0.242 → 0.125 | 1.21 → 0.21 |
| Qwen3-30B-A3B | 33.86 → 0.35 | 3.269 → 0.064 | 30.60 → 0.28 |
| Qwen3.5-27B | 15.34 → 0.27 | 1.541 → 0.013 | 13.80 → 0.26 |


结果显示，长上下文 KV read 会摊薄一部分 launch/host bubble，但不会完全消除。对较大的 dense model，prompt 从 16 增加到 8k 后，decode 更接近 GPU/KV-read-bound：例如 Qwen3-14B 的 speedup 从 1.37x 降到 1.09x，说明 CUDA Graph 的增量空间已经降低了不少。但小模型和 MoE 仍然明显 launch/host-bound：Qwen3-0.6B 仍有 4.73x speedup，Qwen3-30B-A3B 仍有 7.25x speedup。

**claim 依旧是：在 CUDA Graph decode 上，即使 decode 的上下文较长，CUDA Graph 依然可以极大降低 Kernel Launch Overhead 和 Host Overhead，MegaKernel 通过减少 Kernel 的数量来减少 Kernel Launch Overhead + Host Overhead 的收益上限不高。**

### 3.3 Batch Sweep

Batch 增大是否会自然隐藏 launch/host 开销？图 8 给出 decode=128/512 的 batch sweep。

<p align="center">
  <img src="/images/blog/kernel-launch/fig4_decode_bs_sweep_speedup.png" alt="CUDA Graph speedup across decode batch sizes" style="width: 100%; max-width: 1000px; height: auto;">
</p>
<p align="center"><em>图 8. 不同 batch size 下 decode=128/512 的 eager → CUDA Graph 加速比。</em></p>

以下为 BS=8、prompt=16、decode=512 的每步指标；成对数值为 **eager → CG**。

| Model | GPU bubble (%) | E2E speedup | TPOT (ms/token) | Host launch API calls/step |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 85.4% → 12.9% | 6.00× | 10.15 → 1.69 | 402.9 → 17.0 |
| Qwen3-1.7B | 76.0% → 8.8% | 3.79× | 9.39 → 2.48 | 402.9 → 17.0 |
| Qwen3-8B | 44.8% → 3.4% | 1.75× | 11.98 → 6.84 | 549.1 → 17.0 |
| Qwen3-14B | 15.8% → 2.4% | 1.16× | 13.30 → 11.42 | 608.3 → 17.0 |
| Qwen3-30B-A3B | 87.9% → 8.5% | 7.57× | 39.06 → 5.16 | 870.5 → 17.0 |
| Qwen3.5-27B | 55.5% → 0.5% | 2.28× | 47.79 → 20.95 | 1135.5 → 18.0 |

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 8.67 → 0.22 | 1.404 → 0.060 | 7.27 → 0.16 |
| Qwen3-1.7B | 7.13 → 0.22 | 1.245 → 0.040 | 5.88 → 0.18 |
| Qwen3-8B | 5.37 → 0.23 | 1.003 → 0.020 | 4.37 → 0.21 |
| Qwen3-14B | 2.10 → 0.27 | 0.354 → 0.016 | 1.75 → 0.25 |
| Qwen3-30B-A3B | 34.32 → 0.44 | 3.366 → 0.080 | 30.95 → 0.36 |
| Qwen3.5-27B | 26.53 → 0.10 | 2.779 → 0.004 | 23.75 → 0.09 |


BS=8 的观察和 BS=1 decode=512 基本一致，对于 decode 而言，batch 变大不会消除 cudagraph 的价值，

## 4. 与 MPK e2e 结果的对应关系

此前 MPK、vLLM 与 SGLang 的 TPOT 对比与上述 profiling 一致。
针对 decode，在我们的实测中，MPK 干不过 vLLM/SGLang CUDA Graph，这同样和我们观察到的 "`CUDA Graph` 已经能有效地降低 Kernel Launch 和 Host Overhead" 这一现象是相符的，



<p align="center">
  <img src="/images/blog/kernel-launch/mpk_tpot_all_models.png" alt="TPOT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>图 9. MPK、vLLM 与 SGLang 的 decode TPOT。</em></p>


## 5. MoE：CUDA Graph 后仍有较大的 GPU Bubble

Qwen3-30B-A3B 在 CUDA Graph 模式下，decode 的残余 bubble 依然稳定在 6.7%–9.8%；我们没有深入分析，但可以预见的原因是：
- graph 内部的执行结构不同，routing、dispatch、专家计算和 gather 等更长的依赖链，其节点间调度与同步可能产生 GPU 空闲。
- A3B -> 激活参数本来就小，访存量/计算量方面也是超小模型量级。

| Workload     | Case          |  BS | Prompt | Decode | GPU bubble (%) | GPU bubble/request (ms) |
| ------------ | ------------- | --: | -----: | -----: | -------------: | ----------------------: |
| BS=1 decode  | bs1_p16_d128  |   1 |     16 |    128 |           9.4% |                   56.92 |
| BS=1 decode  | bs1_p16_d512  |   1 |     16 |    512 |           8.0% |                  190.17 |
| BS=1 decode  | bs1_p8k_d512  |   1 |   8192 |    512 |           6.7% |                  184.23 |
| Batch decode | bs4_p16_d128  |   4 |     16 |    128 |           9.4% |                   62.37 |
| Batch decode | bs4_p16_d512  |   4 |     16 |    512 |           9.0% |                  235.16 |
| Batch decode | bs8_p16_d128  |   8 |     16 |    128 |           9.8% |                   66.03 |
| Batch decode | bs8_p16_d512  |   8 |     16 |    512 |           8.8% |                  232.38 |
| Batch decode | bs16_p16_d128 |  16 |     16 |    128 |           9.6% |                   67.56 |
| Batch decode | bs16_p16_d512 |  16 |     16 |    512 |           8.7% |                  245.46 |


## 6. 结论

**只针对 MegaKernel 的 ”减少 kernel 数量可以降低 launch 与 host-side 开销“ 这一核心 motivation，我们认为该 motivation 站不住脚。**

核心结论在于：针对 decode，CUDA Graph 已将 host launch API 调用数降至每步 17–18 次，并把中大型 dense model 的 bubble 压至约 1%–3%。**MegaKernel 当然可以节省 Kernel Launch Overhead / CPU Launch Overhead，但是 CUDA Graph 已经做的足够好：**
1. **一方面，SGLang 本身就让 Kernel Launch 很好地被 kernel 的执行所掩盖；**
2. **另一方面，使用 CUDA Graph 之后，单次 decode 所启动的 kernel 数量已经足够少了，未能掩盖的 Kernel Launch 以及 host 同步开销也已经足够小了。**
**此时，为了 2% 级别的 host 同步开销，进而将 CPU 侧控制逻辑放弃，完全进入 GPU Runtime 调度，反而可能因为引入更复杂的 GPU 调度/同步逻辑，导致 Serving System 的性能下降，这是得不偿失的

此外，实验范围仅限单卡，本文所有结果均来自单卡 H100，尚未扩展到多卡。不过，单卡 serve 小模型理论上就是 GPU bubble 最大的场景，多卡场景往往模型较大、计算量较大，GPU bubble 往往由通信造成，而非启动开销。

## 7. 启示 

**MegaKernel 的核心差异化价值不在启动侧。** 我们将其归在：
- 通算融合；
- weight prefetch；
- 打破 kernel 边界的细粒度流水以及片上驻留。

后续，我们将围绕这些方面，展开实验。
