---
title: "17–18 Host Launch API Calls per Decode Step: Reassessing Megakernel Headroom after CUDA Graphs"
slug: kernel-launch
lang: en
date: 2026-09-02
description: "On a single H100, CUDA Graphs reduce host launch API calls to 17–18 per decode step and lower GPU bubble to ~1%–3% for medium-to-large dense models. For graph-compatible decode workloads, a Megakernel has limited headroom if its only advantage is a further reduction in kernel count."
---


**TL;DR.** On a single H100, CUDA Graphs reduce the number of host launch API calls to 17–18 per decode step and lower GPU bubble to approximately 1%–3% for medium-to-large dense models. For graph-compatible decode workloads, a Megakernel therefore has limited headroom if its only advantage is a further reduction in kernel count. Larger residual bubbles appear in short-prompt prefill, small models, and MoE workloads, where roughly 10% of the measured window can remain as unattributed GPU idle time even with CUDA Graphs.

One of the central arguments for Megakernels is that reducing kernel count lowers launch and host-side overhead. This article uses Nsight Systems traces to quantify GPU activity, launch API intervals, and the unattributed idle residual, then asks three questions: 
- How should these metrics be defined? 
- How large are they across models and workloads? 
- Once CUDA Graphs are available, what theoretical headroom remains for a Megakernel? 


## 1. From Kernel Launch to GPU Bubble

For a single explicitly synchronized kernel whose stages do not overlap, end-to-end latency can be approximated as:

```text
E2E latency ≈ CPU launch + other launch-path overhead
            + GPU execution + host-side processing/synchronization
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
| `e2e_ms` | `max(end) - min(start)` over host runtime API events in the captured window, which approximates wall-clock time for the measured `generate()` call; the prefill-only slice uses a separate truncation boundary. |
| `total_kernel_gpu_ms` | Sum of all kernel durations. Overlapping kernels on multiple streams are counted more than once, so this value may exceed `e2e_ms` and must not be interpreted as GPU busy time. |
| `launch_overhead_ms` | Sum of host API durations for `cudaLaunchKernel*`, `cudaGraphLaunch*`, and `cuLaunchKernel*`. An individual call usually takes 2–10 μs when the queue is uncongested, but command-buffer backpressure can increase it to milliseconds or even seconds. This metric can either overestimate or underestimate the true critical-path cost. |
| `gpu_busy_ms` | Union of all GPU activity intervals, including kernels, memcpy, and memset. Overlapping intervals are counted once, so `gpu_busy_ms <= e2e_ms`. |
| `gpu_bubble_ratio` | Fraction of the end-to-end interval with no GPU activity. It observes system-level idle time but does not by itself identify the causal contribution of launch, scheduling, framework execution, synchronization, or GPU queueing. |
| `unhidden_launch_api_ms` | Intersection of launch API intervals with GPU-idle intervals. It is a proxy for exposed launch wall time, not proof of causal critical-path loss. |
| `other_host_idle_ms` | Residual defined as `gpu_bubble_ms - unhidden_launch_api_ms`. D2H copies, sampling, scheduling, attention metadata planning, Python runtime execution, and GPU-side queueing are possible sources, not separately measured attributions. |

The core relationships are:

```text
gpu_bubble_ms
  = e2e_ms - gpu_busy_ms
  = unhidden_launch_api_ms + other_host_idle_ms

gpu_bubble_ratio
  = gpu_bubble_ms / e2e_ms
```

We compute the union of GPU activity intervals rather than summing durations that may double-count concurrent operations. The experimental platform does not expose the required hardware counters, so we do not report SM utilization. Our focus is system-level GPU idle time, not the SM efficiency of an individual kernel.

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

The historically named `bs1_p*_d0` prefill-dominant cases actually execute prefill and generate one token. The figures and tables in Section 4 truncate each trace before the decode tail and report only the prefill slice; the case-level table in Section 6 retains the complete `generate()` path.

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

The results support three direct conclusions:

1. During eager decode, `other_host_idle_ms` is substantially larger than `unhidden_launch_api_ms`: most GPU idle occurs while the host is outside the launch APIs being counted. This residual alone cannot separate the contributions of scheduling, sampling, synchronization, framework execution, or GPU-side queueing.
2. CUDA Graphs reduce host launch API calls per step from 375–976 to 17–18. For Qwen3-8B, the exposed launch time under CG is only 0.014 ms, approximately 0.2% of its 6.45 ms TPOT; other host idle time is 0.14 ms, approximately 2%.
3. CG reduces the bubble of medium-to-large dense models to 1.6%–2.4%, whereas small models and MoE retain 7.8%–14.0%. The claim that little headroom remains therefore applies to the former and must not be generalized to every model.

For graph-compatible decode, the residual bubble after CG is a theoretical time bound that no optimization eliminating GPU idle can exceed. Only the portion causally attributed to launch or scheduling can potentially be recovered by a Megakernel, whose GPU-side scheduling and synchronization costs must also be included.

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

As context grows, KV reads and GPU work become longer while bubble ratio falls. This trend is consistent with fixed system costs being amortized, but the residual has not been decomposed by cause. For Qwen3-14B, CG speedup falls from 1.37× with a short context to 1.09×. Small models and MoE still obtain substantial eager → CG speedups—4.73× for Qwen3-0.6B and 7.25× for Qwen3-30B-A3B—showing greater sensitivity to graph capture.

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

Within the tested batch-size range, BS=8 follows the same trend as BS=1: small models and MoE retain substantial eager → CG speedups, while larger dense models benefit less. Increasing batch size alone does not eliminate sensitivity to graph capture in these cases.

## 4. Prefill: GPU Bubble Varies with Prompt Length

The Qwen3 models use piecewise CUDA Graphs by default in SGLang, whereas Qwen3.5-27B falls back to a path without piecewise graphs. This distinction is an important boundary when interpreting the results.

<p align="center">
  <img src="/images/blog/kernel-launch/fig3_prefill_bubble.png" alt="GPU bubble ratio for batch-one prefill" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>Figure 9. GPU bubble ratio for BS=1 prefill.</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/fig3_prefill_e2e.png" alt="End-to-end latency for batch-one prefill" style="width: 100%; max-width: 780px; height: auto;">
</p>
<p align="center"><em>Figure 10. End-to-end latency for BS=1 prefill.</em></p>

Consider the BS=1, prompt=8k prefill measurements (the Qwen3 models use piecewise CUDA Graphs, whereas Qwen3.5-27B uses the fallback path):

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

Prefill bubble varies strongly with prompt length: the observed bubble is larger for short prompts, then falls as GEMM and attention kernels grow longer. This trend is consistent with fixed system costs being amortized, but the residual has not been decomposed by cause. At prompt=8k, the bubble falls to 1%–3% for Qwen3-8B, Qwen3-14B, and Qwen3.5-27B, but remains 10.4%–14.6% for Qwen3-0.6B/1.7B. Short-to-medium prompts and smaller models therefore form candidate optimization regimes that require further causal attribution.

## 5. Connection to End-to-End MPK Results

Earlier TTFT and TPOT comparisons among MPK, vLLM, and SGLang are consistent with the profiling results above:

- For prefill at `prompt_len=16`, MPK's modest advantage is consistent with the observed high GPU bubble, but the current data do not establish this as a causal relationship. The advantage decays quickly as the prompt grows, and MPK already falls behind vLLM/SGLang at `prompt_len=32`; this comparison does not identify a unique cause.
- During decode, MPK does not outperform the CUDA Graph paths in vLLM/SGLang. This agrees with the observation that CUDA Graphs already reduce launch and host-side overhead substantially.

<p align="center">
  <img src="/images/blog/kernel-launch/mpk_prefill_ttft_all.png" alt="TTFT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>Figure 11. Prefill TTFT for MPK, vLLM, and SGLang.</em></p>

<p align="center">
  <img src="/images/blog/kernel-launch/mpk_tpot_all_models.png" alt="TPOT comparison among MPK, vLLM, and SGLang" style="width: 100%; max-width: 1050px; height: auto;">
</p>
<p align="center"><em>Figure 12. Decode TPOT for MPK, vLLM, and SGLang.</em></p>

In the single-H100 prefill-only slices, every tested model retains at least approximately 10% bubble at `prompt=1k`, with substantially more in the smaller models; at `prompt=256`, every model exceeds approximately 25%. These values are unattributed upper bounds on GPU idle time. Short-to-medium-prompt prefill becomes a viable Megakernel target only if causal analysis shows that an optimizable fraction comes from launch or scheduling.

## 6. MoE: CUDA Graphs Leave a Larger Residual Bubble

Qwen3-30B-A3B continues to exhibit strong workload dependence under CUDA Graph execution. As the prompt grows from 16 to 8k, the bubble of its prefill-dominant case falls from 43.0% to 3.6%. Its residual decode bubble remains between 6.7% and 9.8%, however, and does not disappear over the tested batch-size range.

The `d0` rows below include both prefill and the subsequent generation of one token, so their scope differs from the prefill-only slice in Section 4.

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

These results quantify the difference between MoE and large dense models. CUDA Graphs substantially reduce submission overhead, but most of the MoE residual bubble lies outside the launch API intervals being counted. Its cause still requires event correlation; the residual is a theoretical bound for any optimization that eliminates GPU idle, not a directly attainable speedup.

## 7. Conclusion: Optimize the Residual Bubble, Not Kernel Count Itself

The experiments do not support treating “fewer kernels” as sufficient motivation for a Megakernel across all workloads:

1. **Graph-compatible decode has limited headroom.** CUDA Graphs reduce host launch API calls to 17–18 per step and lower the bubble to approximately 1%–3% for medium-to-large dense models. Any potential gain from further migration into a GPU runtime must offset the additional scheduling and synchronization cost.
2. **Prefill varies with prompt length.** The observed bubble is larger for short prompts; as prompts grow, GPU kernels become longer and the bubble falls, consistent with fixed system costs being amortized.
3. **Small models and MoE retain larger residuals, but attribution is still required.** Even with CUDA Graphs enabled, some cases retain GPU bubble on the order of 10%. The next step is to identify its source, then test whether GPU-side scheduling can recover the optimizable portion.
4. **The evaluation is limited to a single GPU.** All results come from one H100 and have not been extended to multi-GPU execution. Single-GPU serving of small models provides relatively little GPU work per step, making it a stress case with a high residual bubble in this study. Multi-GPU serving typically targets larger models and greater aggregate computation, while also introducing collective communication and cross-device synchronization. Communication may dominate or reshape GPU idle time, but that hypothesis cannot be inferred directly from these single-GPU results and requires multi-GPU traces.
5. **A Megakernel's differentiating value may lie beyond the launch path.** CUDA Graphs reduce the bubble of medium-to-large dense decode to approximately 1%–3%, sharply narrowing the headroom available from fewer host launches alone. This suggests—but does not establish—that the Megakernel benefits most worth testing may come from communication–computation fusion and cross-operator weight prefetching, rather than from kernel-launch reduction itself. Separating these mechanisms requires targeted ablations.

A Megakernel should therefore do more than minimize kernel count: it should first identify the source of the residual GPU bubble after CUDA Graph optimization, then use multi-GPU traces and mechanism-level ablations to show that communication–computation fusion or weight prefetching can recover the optimizable portion at a cost below that benefit ceiling.
