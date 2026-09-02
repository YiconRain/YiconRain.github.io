---
title: "CUDA Graph 之后 MegaKernel 收益空间的再评估"
slug: kernel-launch
lang: zh
date: 2026-09-02
description: "在单卡 H100 上，CUDA Graph 将每个 decode step 的 host launch API 调用数降至 17–18 次，并把中大型 dense model 的 GPU bubble 压到约 1%–3%。对适用 CUDA Graph 的 decode workload，MegaKernel 仅靠进一步减少 kernel 数量，收益上限有限。"
---


**TL;DR.** 在单卡 H100 上，CUDA Graph 将每个 decode step 的 host launch API 调用数降至 17–18 次，并把中大型 dense model 的 GPU bubble 压到约 1%–3%。因此，对适用 CUDA Graph 的 decode workload，MegaKernel 仅靠进一步减少 kernel 数量，收益上限有限。较大的 residual bubble 出现在短 prompt prefill、小模型和 MoE；这些场景在 CUDA Graph 下仍可保留 10% 量级的 GPU 空闲。

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

| 指标                       | 定义与解释                                                                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `e2e_ms`                 | 捕获窗口内 host runtime API 事件的 `max(end) - min(start)`，近似被测 `generate()` 的端到端墙钟时间；prefill-only slice 使用单独的截断边界。                                                               |
| `total_kernel_gpu_ms`    | 所有 kernel duration 的简单求和。多 stream 重叠会被重复计时，因此该值可能大于 `e2e_ms`，不能视为 GPU busy time。                                                                                          |
| `launch_overhead_ms`     | `cudaLaunchKernel*`、`cudaGraphLaunch*` 与 `cuLaunchKernel*` 的 host API duration 之和。队列空闲时单次通常为 2–10 μs；command-buffer backpressure 会把它放大到 ms 甚至 s 级。该指标既可能高估，也可能低估真实关键路径损失。 |
| `gpu_busy_ms`            | Kernel、memcpy、memset 等全部 GPU activity 时间区间的并集；重叠区间只计一次，故 `gpu_busy_ms <= e2e_ms`。                                                                                         |
| `gpu_bubble_ratio`       | 端到端窗口内没有 GPU activity 的比例。它观测 system-level idle，但不单独识别 launch、scheduler、framework、同步或 GPU queueing 的因果贡献。                                                                 |
| `unhidden_launch_api_ms` | Launch API 区间与 GPU idle 区间的交集，是暴露 launch wall time 的代理指标，而非已证明的关键路径因果损失。                                                                                                  |
| `other_host_idle_ms`     | 按 `gpu_bubble_ms - unhidden_launch_api_ms` 定义的 idle time。D2H copy、sampling、scheduler、attention metadata planning、Python runtime 与 GPU-side queueing 都只是可能来源。              |

核心关系为：

```text
gpu_bubble_ms
  = e2e_ms - gpu_busy_ms
  = unhidden_launch_api_ms + other_host_idle_ms

gpu_bubble_ratio
  = gpu_bubble_ms / e2e_ms
```

这里用 GPU activity 区间的并集，而不是可能重复计算并发操作的 duration 之和。我们使用的 vastai 实验平台不开放所需的硬件计数器，因此未采用 SM utilization；本文关注的也是 system-level idle，而非单个 kernel 的 SM 效率。

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

| Model | GPU bubble/step (ms) | Unhidden launch API/step (ms) | Non-launch idle residual/step (ms) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 7.36 → 0.22 | 1.164 → 0.077 | 6.20 → 0.14 |
| Qwen3-1.7B | 6.66 → 0.20 | 1.132 → 0.036 | 5.53 → 0.16 |
| Qwen3-8B | 5.17 → 0.16 | 0.929 → 0.014 | 4.24 → 0.14 |
| Qwen3-14B | 4.34 → 0.22 | 0.677 → 0.014 | 3.66 → 0.21 |
| Qwen3-30B-A3B | 31.37 → 0.36 | 3.050 → 0.066 | 28.32 → 0.29 |
| Qwen3.5-27B | 23.73 → 0.32 | 2.244 → 0.014 | 21.49 → 0.31 |

结果给出三个直接结论：

1. Eager decode 中，`other_host_idle_ms` 显著大于 `unhidden_launch_api_ms`：大部分 GPU idle 发生在 host 未执行所统计 launch API 的区间。仅凭该残差，无法继续区分 scheduler、sampling、同步、framework 或 GPU-side queueing 的贡献。
2. CUDA Graph 将每步 launch API 调用数从 375–976 次降至 17–18 次。以 Qwen3-8B 为例，CG 下暴露的 launch 时间仅为 0.014 ms，约占 6.45 ms TPOT 的 0.2%；other host idle 为 0.14 ms，约占 2%。
3. 中大型 dense model 的 CG bubble 已降至 1.6%–2.4%，但小模型和 MoE 仍有 7.8%–14.0%。因此，“剩余空间有限”适用于前者，不能泛化到所有模型。

对 graph-compatible decode，CG 后的 GPU bubble 是任何消除 GPU idle 的优化都不可能超过的理论时间上界；其中只有经因果归因后确认与 launch 或调度相关的部分，才可能由 MegaKernel 回收。GPU-side scheduler 与同步机制还必须计入自身成本。

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

上下文增长时，KV read 与 GPU 工作变长，bubble ratio 随之下降；这一趋势与固定系统开销被摊薄的解释一致，但 residual 来源尚未分解。Qwen3-14B 的 CG speedup 从短上下文的 1.37× 降至 1.09×；小模型与 MoE 仍获得显著的 eager → CG 加速，Qwen3-0.6B 和 Qwen3-30B-A3B 分别为 4.73× 与 7.25×，表明它们对 graph capture 更敏感。

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

在本测试的 BS 范围内，BS=8 与 BS=1 的趋势一致：小模型和 MoE 仍有显著的 eager → CG 加速，而较大的 dense model 收益较低。增大 batch 本身不足以消除这些场景对 graph capture 的敏感性。

## 4. Prefill：GPU Bubble 随 Prompt Length 变化

Qwen3 系列在 SGLang 中默认使用 piecewise CUDA Graph；Qwen3.5-27B 则回退到不使用 piecewise graph 的路径。该差异是解释结果时的重要边界。

<p align="center">
  <img src="/images/blog/kernel-launch/fig3_prefill_bubble.png" alt="GPU bubble ratio for batch-one prefill" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>图 9. BS=1 prefill 的 GPU bubble ratio。</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig3_prefill_e2e.png" alt="End-to-end latency for batch-one prefill" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>图 10. BS=1 prefill 的端到端延迟。</em></p>

以 BS=1、prompt=8k 的 prefill 测量为例（Qwen3 系列使用 piecewise CUDA Graph，Qwen3.5-27B 为 fallback 路径）：

| Model | Prefill-slice E2E (ms) | Host launch API calls/slice | `cudaGraphLaunch` calls/slice |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 51.11 | 168 | 29 |
| Qwen3-1.7B | 82.69 | 168 | 29 |
| Qwen3-8B | 286.74 | 208 | 37 |
| Qwen3-14B | 483.14 | 228 | 41 |
| Qwen3-30B-A3B | 221.47 | 220 | 49 |
| Qwen3.5-27B | 733.65 | 1,683 | 0 |

| Model | GPU bubble/slice (ms) | GPU bubble (%) | Unhidden launch API/slice (ms) | Non-launch idle residual/slice (ms) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 7.47 | 14.6% | 0.21 | 7.26 |
| Qwen3-1.7B | 8.58 | 10.4% | 0.21 | 8.36 |
| Qwen3-8B | 6.89 | 2.4% | 0.14 | 6.75 |
| Qwen3-14B | 6.69 | 1.4% | 0.15 | 6.54 |
| Qwen3-30B-A3B | 8.09 | 3.7% | 0.18 | 7.92 |
| Qwen3.5-27B | 11.48 | 1.6% | 0.98 | 10.49 |

Prefill bubble 随 prompt length 显著变化：短 prompt 的 observed bubble 较高；prompt 增长时，GEMM 与 attention 变长，bubble ratio 下降。这一趋势与固定系统开销被摊薄一致，但 residual 来源尚未分解。到 prompt=8k，Qwen3-8B、Qwen3-14B 与 Qwen3.5-27B 的 bubble 已降至 1%–3%，而 Qwen3-0.6B/1.7B 仍为 10.4%–14.6%。中短 prompt 与较小模型因而构成待进一步归因的候选优化区间。

## 5. 与 MPK 端到端结果的对应关系

此前 MPK、vLLM 与 SGLang 的 TTFT/TPOT 对比与上述 profiling 一致：

- 在 `prompt_len=16` 的 prefill 中，MPK 的小幅优势与较高 GPU bubble 的观察一致，但现有数据不能单独建立因果关系。该优势随 prompt 增长快速衰减，`prompt_len=32` 时已落后于 vLLM/SGLang；当前对比不能确定唯一原因。
- 在 decode 中，MPK 未能超过 vLLM/SGLang 的 CUDA Graph 路径。这与 CUDA Graph 已显著压低 launch/host overhead 的测量结果一致。

<p align="center">
  <img src="/images/blog/kernel-launch/mpk_prefill_ttft_all.png" alt="TTFT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>图 11. MPK、vLLM 与 SGLang 的 prefill TTFT。</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/mpk_tpot_all_models.png" alt="TPOT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>图 12. MPK、vLLM 与 SGLang 的 decode TPOT。</em></p>

单卡 H100 的 prefill-only slice 中，`prompt=1k` 时所有测试模型仍至少有约 10% bubble，小模型显著更高；`prompt=256` 时各模型均超过约 25%。这些是待归因的 GPU idle 上界；若其中可优化部分确由 launch 或调度造成，中短 prompt prefill 才可能成为 MegaKernel 的有效优化区间。

## 6. MoE：CUDA Graph 后仍有较大的 Residual Bubble

Qwen3-30B-A3B 在 CUDA Graph 模式下仍明显随 workload 变化。Prefill-dominant case 的 bubble 随 prompt 从 16 增至 8k，由 43.0% 降至 3.6%；decode 的残余 bubble 则稳定在 6.7%–9.8%，且在测试的 batch 范围内没有消失。

下表中的 `d0` 行包含 prefill 和随后生成的 1 个 token，数值口径与第 4 节的 prefill-only slice 不同。

| Workload | Case | BS | Prompt | Decode (case ID) | GPU bubble (%) | GPU bubble/request (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prefill | bs1_p16_d0 | 1 | 16 | 0 | 43.0% | 10.67 |
| Prefill | bs1_p256_d0 | 1 | 256 | 0 | 25.7% | 7.41 |
| Prefill | bs1_p1k_d0 | 1 | 1024 | 0 | 19.0% | 7.00 |
| Prefill | bs1_p4k_d0 | 1 | 4096 | 0 | 7.5% | 7.59 |
| Prefill | bs1_p8k_d0 | 1 | 8192 | 0 | 3.6% | 8.04 |
| BS=1 decode | bs1_p16_d128 | 1 | 16 | 128 | 9.4% | 56.92 |
| BS=1 decode | bs1_p16_d512 | 1 | 16 | 512 | 8.0% | 190.17 |
| BS=1 decode | bs1_p8k_d512 | 1 | 8192 | 512 | 6.7% | 184.23 |
| Batch decode | bs4_p16_d128 | 4 | 16 | 128 | 9.4% | 62.37 |
| Batch decode | bs4_p16_d512 | 4 | 16 | 512 | 9.0% | 235.16 |
| Batch decode | bs8_p16_d128 | 8 | 16 | 128 | 9.8% | 66.03 |
| Batch decode | bs8_p16_d512 | 8 | 16 | 512 | 8.8% | 232.38 |
| Batch decode | bs16_p16_d128 | 16 | 16 | 128 | 9.6% | 67.56 |
| Batch decode | bs16_p16_d512 | 16 | 16 | 512 | 8.7% | 245.46 |

这组结果量化了 MoE 与大型 dense model 的差异：CUDA Graph 已显著降低提交开销，但 MoE 的 residual bubble 大部分位于所统计 launch API 区间之外。其根因仍需 event correlation；该 residual bubble 是所有消除 GPU idle 优化的理论上界，而不是可直接获得的加速比。

## 7. 结论：优化目标应是残余 Bubble，而非 Kernel 数量本身

实验不支持把“减少 kernel 数量”视为 MegaKernel 在所有 workload 上的充分动机：

1. **Graph-compatible decode：空间有限。** CUDA Graph 已将 host launch API 调用数降至每步 17–18 次，并把中大型 dense model 的 bubble 压至约 1%–3%。进一步迁移到 GPU runtime 的潜在收益必须覆盖新增调度与同步成本。
2. **Prefill：随 prompt length 变化。** 短 prompt 的 observed bubble 较大；prompt 增长后 GPU kernel 变长、bubble 下降，这与固定系统开销被摊薄的解释一致。
3. **小模型与 MoE：残差更大，但仍需归因。** 即使启用 CUDA Graph，部分 case 仍保留 10% 量级的 bubble。应先定位其来源，再评估 GPU-side scheduling 能否回收其中可优化的部分。
4. **实验范围仅限单卡。** 本文所有结果均来自单卡 H100，尚未扩展到多卡。单卡小模型每步 GPU 工作较少，是本实验中 residual bubble 较高的一类压力场景。多卡 serving 通常面向更大的模型和更高的总计算量，同时引入 collective communication 与跨设备同步；通信可能主导或重塑 GPU idle，但这一假设不能由当前单卡结果直接推出，仍需多卡 trace 验证。
5. **MegaKernel 的差异化价值可能不在启动侧。** CUDA Graph 已把中大型 dense decode 的 bubble 压至约 1%–3%，显著收窄了仅靠减少 host launch 可获得的空间。这表明——但尚未证明——MegaKernel 更值得验证的收益来源可能是通算融合与跨算子 weight prefetch，而非单纯减少 kernel launch。区分这些机制仍需针对性消融。

因此，MegaKernel 不应只最小化 kernel count，而应先定位 CUDA Graph 后 residual GPU bubble 的来源，再通过多卡 trace 与机制级消融，证明通算融合或 weight prefetch 能以低于收益上界的成本回收其中可优化的部分。
