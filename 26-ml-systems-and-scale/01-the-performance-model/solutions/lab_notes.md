# the performance model · lab notes

numbers below are from the reference solution on an Apple M2 laptop (8 cores, unified memory, `mps` GPU), measured while other jobs shared the machine (load average 5–16). a rerun at a load average of 40 measured the GPU at 1.1 TFLOP/s and 39 GB/s, less than half of the numbers below, with the 2048³ matmul at "140% of the roofline": when the machine is that busy, the peak measurement itself is wrong, and every ratio built on it. best of 5 for every timing. your numbers will differ; compare shapes and ratios, and rerun if a number looks off: a busy machine moves individual timings by 10–30%.

## experiment 1: your machine's roofline

**what people predict.** that a 64×64 matmul, at an intensity of 10.7 FLOP/byte, sits somewhere near the roofline (it's "a matmul, so compute-bound"), maybe at half of what's allowed; and that a 2048×2048 one reaches the roof.

**what happens.**

| device | measured bandwidth | measured peak (float32 matmul) | ridge point |
|---|---|---|---|
| cpu | 53 GB/s | about 1,100 GFLOP/s (fastest of 512³–2048³) | about 20 FLOP/B |
| mps | 72 GB/s | 2,730 GFLOP/s (4096³) | 38 FLOP/B |

(Apple's spec for the M2 is 100 GB/s of unified memory bandwidth and about 3.6 TFLOP/s of GPU float32; the measured values are what you can actually get from PyTorch, on a shared machine.)

the CPU's "peak" deserves a warning. about 1.1 TFLOP/s of float32 from an 8-core laptop CPU is roughly ten times what its vector units can do. PyTorch on macOS calls Apple's Accelerate library for matrix products, and Accelerate runs them on the **AMX matrix coprocessor**, a matrix unit next to the CPU cores. a single thread already gets most of it (one thread measured 590 GFLOP/s for a 512³ product on a heavily loaded machine, and adding threads didn't help), and all processes on the machine share it. so on this machine the CPU roofline is really two roofs: one for matrix products (AMX) and a much lower one for everything else (the cores' NEON units). on a Linux or x86 machine, matmul speed scales with cores and threads in the ordinary way, and these numbers will look very different.

| matmul size | cpu, % of roofline | mps, % of roofline |
|---|---|---|
| 16 | 4% | 0% (0.04 GFLOP/s) |
| 64 | 44% | 0.3% (2.5 GFLOP/s) |
| 256 | ~100% | 7% |
| 1024 | ~100% | 80% |
| 2048 | ~100% | 96% |

a 64×64 matmul reaches 44% of its roofline on the CPU and **0.3%** on the GPU. elementwise adds behave the same way: a 16M-element add runs at the bandwidth line on both devices (96–102%), a 4K-element add at 89–107% on the CPU but 0% on the GPU. on reruns, small CPU adds sometimes land *above* the line (up to about 155%). (an earlier version of this lab took the CPU's peak from a single 2048³ matmul and found the 512³ one at 130% "of the roofline": the CPU's matmul speed depends on size, through caches and threading, so the lab now takes the best of three sizes.)

**why.** the small GPU ops aren't memory-bound or compute-bound: they're **overhead-bound**. a 64×64 matmul is 0.5 MFLOP, a fraction of a microsecond of GPU arithmetic, but dispatching it through PyTorch and Metal and waiting for it to come back costs on the order of 100 µs. the roofline doesn't draw this third ceiling; it's a horizontal line in *time*, not in FLOP/s, so on this plot it shows up as points that fall further and further below the roof as the op gets smaller. the CPU has far less per-op overhead (no queue to a separate device), which is why it wins on small ops. the only way to move a 64×64 matmul up to the roof is to stop running it alone: batch many of them into one call (`bmm`), fuse it with its neighbours, or compile the graph so the launches disappear. small adds above the roof aren't breaking physics: a 4K- or 64K-element add's three arrays (48 KB to 768 KB) fit in the CPU's caches, and after the warmup calls they're served from there, at far more than DRAM bandwidth. the roofline drawn here is the DRAM one (bandwidth measured with a 64 MiB copy); each cache level has its own, higher, slanted line. on bandwidth: the CPU and GPU share the same LPDDR5 chips, but one CPU process with a handful of threads can't issue enough simultaneous memory requests to saturate them, while the GPU's thousands of threads get closer.

## experiment 2: a GPT block, op by op

**what people predict.** that the softmax and elementwise ops, with about 1–2% of the FLOPs, take a few percent of the time, maybe 10%.

**what happens.** on `mps` at batch 64: the softmax and the five elementwise ops (two layernorms, GELU, two residual adds) do **1.3%** of the block's FLOPs and take **41%** of its measured time. the softmax alone, at 0.4% of the FLOPs, takes 18% of the time, more than `fc1` (12%), which does 75 times as many FLOPs.

| op | intensity | roofline says | predicted ms | measured ms | measured / predicted |
|---|---|---|---|---|---|
| ln1 | 0.88 | memory | 0.175 | 0.343 | 2.0× |
| q_proj | 47.4 | compute | 0.221 | 0.438 | 2.0× |
| scores | 10.7 | memory | 0.525 | 0.632 | 1.2× |
| softmax | 0.62 | memory | 0.700 | 1.672 | 2.4× |
| attn_v | 10.7 | memory | 0.525 | 0.737 | 1.4× |
| residual1 | 0.08 | memory | 0.262 | 0.346 | 1.3× |
| fc1 | 75.4 | compute | 0.884 | 1.134 | 1.3× |
| gelu | 1.00 | memory | 0.700 | 0.833 | 1.2× |
| fc2 | 75.4 | compute | 0.884 | 1.153 | 1.3× |

(full table in the lab; `k_proj`, `v_proj`, `out_proj`, `ln2` and `residual2` match their twins.)

**why.** time follows bytes, not FLOPs, for every op left of the ridge, and on this GPU that's every op except the six projections. the attention-score matmul is memory-bound despite being a matmul: with a head width of 32, each output score is a dot product of length 32, so the $T \times T$ output dominates the traffic and the intensity is only 10.7. most ops run within 1.2–2.4× of their roofline prediction. the outliers are instructive: the softmax (2.4×) is a reduction, which a GPU kernel can't stream as efficiently as an elementwise op, and the small $(8192, 192) @ (192, 192)$ projections (2×) don't fill the GPU well. a fused attention kernel removes the scores, softmax and weights round trips entirely: about 25% of this block's measured time, which is what the capstone's fused attention buys.

## experiment 3: profiling a training step

**what people predict.** that matrix products dominate the CPU profile (they do "all the work"), and an MFU of 30% or more on the GPU.

**what happens.** on the CPU, at batch 8, `mm`, `bmm` and `addmm` together take 36% of the self time. the next biggest ops are `aten::copy_` (864 calls, 10%), `aten::masked_fill_` (8%: your attention building and applying its causal mask), `gelu_backward` and `gelu` (10% between them), `_softmax` (4%) and hundreds of small `div`, `mul_`, `add_` calls (AdamW's per-parameter arithmetic). MFU counted as $6N$ FLOPs per token:

| device | batch | step time (best of 5) | achieved | MFU |
|---|---|---|---|---|
| cpu | 16 | 138 ms (138–158) | 162 GFLOP/s | about 14% (of the measured 1,125 GFLOP/s peak) |
| mps | 64 | 205 ms (205–208) | 435 GFLOP/s | 15.9% |

**why.** a small transformer is mostly *not* matmuls by time: masks, copies (reshapes and transposes that materialize, `contiguous()`), the elementwise ops of experiment 2 in forward *and* backward, and the optimizer's many small kernels. reasons the MFU is far below a large run's 40–60%, roughly in order: (1) memory-bound ops (softmax, layernorm, GELU, residuals, the naive attention's $T \times T$ tensors), attacked by fusion (topic 02, the capstone's fused attention and `torch.compile`) and fewer bytes per number (topic 03); (2) per-op overhead on small tensors (dozens of kernels per block, plus AdamW's per-parameter kernels), attacked by compilation and fused optimizers; (3) matmuls too small to fill the GPU (width 192), which only a larger model or batch fixes. note that $6N$ ignores the attention FLOPs, so the true utilization is slightly higher, and that the CPU's MFU is relative to its own (much lower) peak.
