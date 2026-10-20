# the performance model

## why this matters

chapter 18's capstone trained a 1.8-million-parameter GPT for 2,000 steps of 64 windows of 128 characters. that's $6 \times 1.8\text{M} \times 16.4\text{M tokens} \approx 1.8 \times 10^{14}$ floating-point operations. a laptop GPU is advertised at a few trillion FLOP/s, so the run "should" take about a minute. chapter 18 measured six minutes, and on the same laptop on busier days this chapter's reruns of it took between 7 and 12. where did the rest of the time go?

not into arithmetic. most of the time a GPU spends on a transformer, it's waiting: for numbers to arrive from memory, for the CPU to tell it what to do next, for a kernel launch. which of those you're waiting for decides what will make you faster. a fused kernel helps if you're waiting for memory and does nothing if you're waiting for the CPU. a bigger batch helps if you're waiting for launches and does nothing if you're already doing arithmetic at full speed. guessing wrong costs days.

this topic gives you the model that lets you stop guessing: count the FLOPs and the bytes of every operation, compare their ratio with your machine's ratio of compute to bandwidth (the **roofline**), and then measure, carefully, to see how close you get. by the end you'll have measured your own machine's roofline and put every operation of a GPT block on it. the rest of the chapter is a list of ways to move operations up and to the right on that plot.

## warm-up: predict first

1. a $(4096 \times 4096) @ (4096 \times 4096)$ matmul and a $4096 \times 4096$ elementwise GELU. how many times more FLOPs does the matmul do? how many times longer will it take on a GPU?
2. your GPU does $3 \times 10^{12}$ FLOP/s and moves $10^{11}$ bytes/s. how many FLOPs must an operation do per byte it reads or writes before arithmetic, rather than memory, limits it?
3. you time `y = model(x)` on a GPU with `time.perf_counter()` before and after. it reports 0.2 ms for a model you know needs 50 ms. what happened?
4. in a GPT block of width 192 with sequences of 128, the softmax over attention scores does about 1% of the block's FLOPs. what fraction of the block's time do you expect it to take?

## prerequisites check

- [ ] you can count the multiply-adds of a matrix product and know why matmul is associative but not cheaper in every order (→ ch 02)
- [ ] you know every op inside your GPT block: layernorm, q/k/v projections, scores, softmax, weighted sum, MLP with GELU, residual adds (→ ch 18 · 03, 04)
- [ ] you can move tensors to a device with `get_device()` and know a forward + backward + optimizer step (→ ch 15 · 07)
- [ ] you remember that ch 18 · 06's online softmax and KV cache were sold as *memory* savings (→ ch 18 · 06)

## core ideas

### idea 1: FLOPs, bytes, and their ratio

every operation on a tensor does two things: arithmetic, and moving data between memory and the processor. a matrix product $C = AB$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$ computes $mn$ dot products of length $k$: $k$ multiplies and $k$ adds each, so $2mnk$ **FLOPs** (floating-point operations; FLOP/s is the rate). at minimum it reads $A$ and $B$ once and writes $C$ once: $(mk + kn + mn)$ elements, times 4 bytes in float32 or 2 in bf16.

the ratio is the **arithmetic intensity**,

$$I = \frac{\text{FLOPs}}{\text{bytes moved}}.$$

for a square float32 matmul it's $2n^3 / 12n^2 = n/6$: it grows with size, because each element read is reused $n$ times. for an elementwise op like GELU, each element is read once, transformed with a handful of FLOPs and written once: $I \approx 8 / 8 = 1$, whatever the size. a matrix–vector product ($m = 1$) reads every weight once and does one multiply-add with it: $I \approx 0.5$. these three numbers, $n/6$, $\sim 1$, $\sim 0.5$, explain most of what follows.

"bytes moved" means bytes moved to and from the memory that limits you. a processor has a **memory hierarchy**: a few kilobytes of registers per core, tens to hundreds of kilobytes of fast on-chip SRAM (CUDA's *shared memory*, Metal's *threadgroup memory*), megabytes of cache, then gigabytes of DRAM. each level is roughly ten times bigger and several times slower than the one above. the minimum-traffic count assumes every input is read from DRAM exactly once, which a good kernel achieves by keeping reused data in the fast levels (topic 02 shows how).

one fact about the hardware this course is tested on: an Apple-silicon Mac has **unified memory**. the CPU and the GPU share one pool of LPDDR memory with one bandwidth. there's no PCIe bus and no host-to-device copy (`x.to("mps")` is cheap compared with a discrete GPU), but there's also no separate high-bandwidth memory for the GPU: the GPU's bandwidth is the system's, about 100 GB/s on an M2 by Apple's spec, against roughly 2,000 GB/s of HBM on a datacenter A100. you'll measure your own in the lab rather than trust either number.

> **spec.** `matmul_flops(m, n, k)` = $2mnk$ for $(m, k) @ (k, n)$. `matmul_bytes(m, n, k, *, bytes_per_element=4)` = $(mk + kn + mn) \cdot$ `bytes_per_element`. `arithmetic_intensity(flops, bytes_moved)` = flops / bytes_moved.

> **▶ build: stage 1.** implement `matmul_flops`, `matmul_bytes` and `arithmetic_intensity`.

### idea 2: the roofline

a machine has two speed limits: a peak arithmetic rate $P$ (FLOP/s) and a memory bandwidth $W$ (bytes/s). an operation with intensity $I$ that streams its data at full bandwidth can do at most $I \cdot W$ FLOP/s. so the attainable rate is

$$\text{attainable}(I) = \min(P,\; I \cdot W).$$

on a log–log plot this is a slanted line (slope 1, the bandwidth) meeting a flat roof (the peak), which gave it the name **roofline** (Williams, Waterman & Patterson, 2009). the corner is the **ridge point** $I^* = P / W$. operations to its left are **memory-bound**: they'd go no faster with a faster ALU, only with fewer bytes. operations to its right are **compute-bound**: they'd go no faster with more bandwidth.

ridge points are large and getting larger, because arithmetic has improved much faster than memory. an A100 (vendor numbers: 312 TFLOP/s in bf16, about 2 TB/s) has $I^* \approx 150$; anything doing fewer than 150 FLOPs per byte wastes most of the chip. a laptop GPU's ridge is lower, but the shape is the same. this is why elementwise ops, normalizations and small-batch inference are memory-bound on every modern accelerator, and why the next five topics mostly reduce bytes: fusing ops so intermediates never touch DRAM (02), using 2-byte numbers instead of 4 (03), storing int8 weights (06).

there's a third regime that the roofline doesn't draw: **overhead-bound**. each PyTorch op costs microseconds of Python, dispatch and kernel-launch time regardless of its size. an op on a small tensor finishes its arithmetic and its memory traffic in less time than it took to launch, and sits far below both limits. the fix is neither fewer FLOPs nor fewer bytes but fewer, bigger ops: larger batches, fused kernels, compilation (He, 2022).

> **spec.** `roofline(intensity, peak_flops, bandwidth)` = $\min(P, I W)$, elementwise when `intensity` is an array. `ridge_point(peak_flops, bandwidth)` = $P / W$. `bound(intensity, peak_flops, bandwidth)` returns `"memory"` if $I < I^*$, else `"compute"`.

> **🔮 predict:** your laptop GPU's ridge point is somewhere between 10 and 50 FLOP/byte. a $(8192 \times 192) @ (192 \times 768)$ float32 matmul (the first MLP layer of chapter 18's model M on one batch): memory- or compute-bound?

> **▶ build: stage 2.** implement `roofline`, `ridge_point` and `bound`.

### idea 3: measuring honestly

the roofline predicts; a stopwatch decides. but timing GPU code is full of traps.

**asynchrony.** when Python calls a GPU op, PyTorch *queues* a kernel and returns immediately. the GPU runs the queue behind your back. `perf_counter()` before and after `model(x)` measures how long it took to queue the work, which is why warm-up question 3 saw 0.2 ms. to time the work, call the device's synchronize (`torch.mps.synchronize()`, `torch.cuda.synchronize()`), which blocks until the queue is empty, before reading the clock. it also has to run once *before* the first measurement, or the first timed interval includes leftover work from earlier.

**warm-up.** the first call of anything is slow: memory gets allocated, Metal shaders and `torch.compile` graphs get compiled, caches are cold. run the function a few times before timing it.

**noise.** other processes share your CPU, your GPU and your memory bandwidth (on the machine this course was tested on, other jobs ran throughout). run several repeats and report the **minimum**, which estimates the undisturbed time (noise only ever adds time), along with the median and the range, so a reader sees the spread. never compare two numbers measured at different times on a busy machine without repeats.

none of this makes a timing portable. your numbers belong to your machine, at that moment; what transfers is the *ratio* between two methods measured back to back.

> **spec.** `benchmark(fn, *, warmup=1, repeats=5, sync=None)`: calls `fn()` `warmup` times untimed; then, if `sync` is given, calls `sync()` once; then `repeats` times: read the clock (`time.perf_counter`), call `fn()`, call `sync()` if given, read the clock. returns `{"best": min, "median": np.median, "times": list of repeats durations in seconds}`.

> **▶ build: stage 3.** implement `benchmark`.

### idea 4: a GPT block on the roofline

now apply the model to the network you built. a pre-LN GPT block with batch $B$, sequence length $T$, width $d$, $H$ heads of width $d_h = d/H$ and MLP width $d_\text{ff}$ (default $4d$) runs fourteen operations in its forward pass:

| op | FLOPs | elements read + written |
|---|---|---|
| `ln1`, `ln2` (layernorm) | $7\,BTd$ | $BTd + BTd$ |
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | matmul $(BT, d) @ (d, d)$ | matmul $(BT, d) @ (d, d)$ |
| `scores` ($QK^\top$ per head) | $BH \times$ matmul $(T, d_h) @ (d_h, T)$ | $BH \times$ matmul $(T, d_h) @ (d_h, T)$ |
| `softmax` | $5\,BHT^2$ | $BHT^2 + BHT^2$ |
| `attn_v` (weights $\times V$ per head) | $BH \times$ matmul $(T, T) @ (T, d_h)$ | $BH \times$ matmul $(T, T) @ (T, d_h)$ |
| `residual1`, `residual2` | $BTd$ | $2BTd + BTd$ |
| `fc1` | matmul $(BT, d) @ (d, d_\text{ff})$ | matmul $(BT, d) @ (d, d_\text{ff})$ |
| `gelu` | $8\,BTd_\text{ff}$ | $BTd_\text{ff} + BTd_\text{ff}$ |
| `fc2` | matmul $(BT, d_\text{ff}) @ (d_\text{ff}, d)$ | matmul $(BT, d_\text{ff}) @ (d_\text{ff}, d)$ |

"matmul $(m, k) @ (k, n)$" means your `matmul_flops(m, n, k)` and `matmul_bytes(m, n, k)`; the elementwise FLOP counts (7 per element for a layernorm, 5 for a softmax, 8 for a tanh-free GELU) are conventions, and any constant of that size leaves the conclusion unchanged. biases are ignored.

add up the projections: $2 \cdot BT \cdot (4d^2 + 2 d\, d_\text{ff}) = 24\,BTd^2$ FLOPs, which is 2 FLOPs per parameter per token (a block has $12d^2$ weights). the two attention matmuls add $4BT^2d$, which matters once $T$ is comparable to $6d$. the backward pass computes two gradients per matmul (with respect to the input and to the weight), each as expensive as the forward, so a training step costs about 3× the forward: **6 FLOPs per parameter per token**. the whole-model estimate

$$C_\text{train} \approx 6ND$$

for $N$ parameters and $D$ training tokens (Kaplan et al., 2020) is what chapter 19's scaling laws and the capstone's budget are written in.

dividing the FLOPs your model needs per step by the measured step time gives the achieved FLOP/s; dividing that by the peak gives **model FLOP utilization** (MFU; Chowdhery et al., 2022). a well-tuned large training run on datacenter GPUs reaches 40–60%. a small model in eager PyTorch on a laptop reaches a few percent, and the next topics explain why.

> **spec.** `transformer_train_flops(n_params, n_tokens)` = $6ND$. `gpt_block_ops(batch, T, d_model, n_heads, *, d_ff=None, bytes_per_element=4)` returns a list of 14 dicts `{"name", "flops", "bytes"}` in the table's order (`ln1, q_proj, k_proj, v_proj, scores, softmax, attn_v, out_proj, residual1, ln2, fc1, gelu, fc2, residual2`), bytes = elements × `bytes_per_element`. `mfu(flops_per_step, seconds_per_step, peak_flops)` = achieved FLOP/s / peak.

> **🔮 predict:** in that table, which ops will land left of your GPU's ridge point at $B = 64$, $T = 128$, $d = 192$?

> **▶ build: stage 4.** implement `transformer_train_flops`, `gpt_block_ops` and `mfu`.

### idea 5: profiling

the calculator tells you what each op *should* cost. a **profiler** tells you what it did cost. `torch.profiler` records every ATen op (the C++ operators behind PyTorch's Python API: `aten::mm`, `aten::softmax`, `aten::native_layer_norm`, …) with its call count, its **total** time (including the ops it calls) and its **self** time (excluding them). sorting by self time answers "where does the time go?" without double counting.

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    step()
for e in prof.key_averages():   # one row per op name
    e.key, e.count, e.self_cpu_time_total   # microseconds
```

two caveats. a profiler costs time itself, so profile a few steps after a warm-up and treat the numbers as proportions, not absolutes. and the CPU profiler sees what the *CPU* does: on a CPU run that's the computation; on a GPU run it's only the time to dispatch each kernel, because the kernels run asynchronously (warm-up question 3 again). seeing GPU time per kernel takes the device's own tools (CUDA's activity in `torch.profiler`, Xcode Instruments' Metal System Trace for `mps`) or synchronized timing of individual ops, which is what the lab does.

> **spec.** `profile_ops(step_fn, *, n_steps=3, warmup=1)`: calls `step_fn()` `warmup` times unprofiled, then `n_steps` times inside `torch.profiler.profile(activities=[ProfilerActivity.CPU])`; returns `[(op_name, self_cpu_time_us, calls), ...]` from `key_averages()`, sorted by self time, largest first.

> **▶ build: stage 5.** implement `profile_ops`.

## worked example

chapter 18's model M: $d = 192$, $H = 6$ ($d_h = 32$), $d_\text{ff} = 768$, trained with $B = 64$, $T = 128$, float32. take a GPU with round numbers $P = 3 \times 10^{12}$ FLOP/s and $W = 10^{11}$ B/s (ridge $I^* = 30$).

**`fc1`**, a $(8192, 192) @ (192, 768)$ matmul: $2 \cdot 8192 \cdot 768 \cdot 192 = 2{,}415{,}919{,}104$ FLOPs; $(8192 \cdot 192 + 192 \cdot 768 + 8192 \cdot 768) \cdot 4 = 32{,}047{,}104$ bytes; $I = 75.4$. compute-bound; best case $2.42 \times 10^9 / 3 \times 10^{12} = 0.81$ ms.

**`scores`**, 384 products $(128, 32) @ (32, 128)$: $384 \cdot 2 \cdot 128 \cdot 128 \cdot 32 = 402{,}653{,}184$ FLOPs; $384 \cdot (4096 + 4096 + 16384) \cdot 4 = 37{,}748{,}736$ bytes; $I = 10.7$. memory-bound, even though it's a matmul: with $d_h = 32$ each score is a dot product of length 32, so each element of the $T \times T$ output is written after very little work. best case $37.7\,\text{MB} / 10^{11} = 0.38$ ms.

**`softmax`** over $384 \times 128 \times 128 = 6{,}291{,}456$ scores: $5 \times$ that $= 31{,}457{,}280$ FLOPs; $2 \times 6{,}291{,}456 \times 4 = 50{,}331{,}648$ bytes; $I = 0.625$. deeply memory-bound; best case $0.50$ ms.

so the softmax, with 1.3% of `fc1`'s FLOPs, needs at least 62% of its time. the attention scores and softmax are exactly what FlashAttention never writes to DRAM (ch 18 · 06), and you can now say why that's worth a custom kernel. your `gpt_block_ops(64, 128, 192, 6)` must reproduce these three rows.

## on paper

1. **intensity of a batched linear layer.** a float32 linear layer $(B, d) @ (d, d)$ with $d = 4096$. write its intensity as a function of $B$, and find the batch size where it crosses a ridge point of 150. what does that say about serving a language model one request at a time?

<details><summary>solution</summary>

FLOPs $2Bd^2$; bytes $4(Bd + d^2 + Bd) = 4d(d + 2B)$; $I = 2Bd^2 / (4d(d + 2B)) = Bd / (2(d + 2B))$. for $B \ll d$, $I \approx B/2$. setting $Bd / (2(d + 2B)) = 150$ with $d = 4096$: $4096B = 300(4096 + 2B)$, $B = 1{,}228{,}800 / 3{,}496 \approx 352$. in bf16 (2 bytes) the intensity doubles and $B \approx 160$. one request at a time ($B = 1$) runs at $I \approx 0.5$, about 300× below the ridge: generation is memory-bound and the GPU's arithmetic is almost idle, so batching many requests is nearly free compute (topic 06).

</details>

2. **6ND.** a linear layer $y = xW$ with $W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}$ on $n$ tokens. count the FLOPs of the forward pass, of $\partial L/\partial x = (\partial L/\partial y) W^\top$, and of $\partial L/\partial W = x^\top (\partial L/\partial y)$. conclude the "6" in $6ND$.

<details><summary>solution</summary>

forward: $(n, d_\text{in}) @ (d_\text{in}, d_\text{out})$, $2n\,d_\text{in}d_\text{out}$. input gradient: $(n, d_\text{out}) @ (d_\text{out}, d_\text{in})$, the same count. weight gradient: $(d_\text{in}, n) @ (n, d_\text{out})$, the same again. total $6 n\, d_\text{in} d_\text{out} = 6 \times (\text{parameters}) \times (\text{tokens})$. since nearly all of a transformer's FLOPs are in such layers, $C \approx 6ND$; attention scores add $\approx 12 L T d$ per token (forward $4BT^2d$ per layer, ×3), which is small while $T \ll 6d$. the first layer's input gradient isn't needed, a minor saving.

</details>

3. **attention vs MLP.** using the table, find the sequence length at which a block's two attention matmuls do as many FLOPs as its six projections. for $d = 192$, which is chapter 18's `block_size` 128 closer to?

<details><summary>solution</summary>

$4BT^2d = 24BTd^2$ gives $T = 6d$. for $d = 192$ that's $T = 1{,}152$: at $T = 128$ attention's matmuls are $128/1152 = 11\%$ of the projections' FLOPs. by FLOPs attention looks cheap; by bytes it isn't: the $T^2$ terms (scores, softmax, weights) are written and read in full at very low intensity, so their share of *time* is far larger than 11% (lab, experiment 2).

</details>

## experiment

the lab measures your machine's roofline: its memory bandwidth from large copies and its peak FLOP/s from large matmuls, on the CPU and (if you have one) the GPU, with a sweep of matmul sizes and elementwise ops plotted against it. then it times every op of chapter 18's model M block on its own and compares each with the roofline's prediction from your `gpt_block_ops`. finally it profiles a full training step of your GPT with `profile_ops` and computes its MFU.

## check your understanding

1. why does a square matmul's intensity grow with $n$ while an elementwise op's doesn't?

<details><summary>answer</summary>

a matmul reads $O(n^2)$ numbers and does $O(n^3)$ work with them: each input element is used in $n$ different dot products. an elementwise op uses each input exactly once. intensity is reuse.

</details>

2. a kernel runs at 40% of peak FLOP/s. is it compute-bound?

<details><summary>answer</summary>

not necessarily. compare its intensity with the ridge. if it's left of the ridge, the bandwidth ceiling there might be 45% of peak, and 40% means it's nearly memory-optimal. if it's right of the ridge, 40% is well short of the roof, and the cause is elsewhere: launch overhead, poor tiling, low occupancy.

</details>

3. why report the minimum of several timings rather than the mean?

<details><summary>answer</summary>

interference (other processes, frequency scaling, page faults) only ever adds time, so the minimum is the best estimate of the undisturbed cost, and it's stable across runs. the mean mixes in the noise. report the median and range too, so readers see how noisy the machine was.

</details>

4. on an Apple-silicon Mac, `x.to("mps")` is much cheaper than `x.to("cuda")` on a desktop. why, and what's the catch?

<details><summary>answer</summary>

unified memory: CPU and GPU share the same DRAM, so there's no PCIe transfer. the catch is that the GPU has no memory of its own: its bandwidth is the system's (about 100 GB/s on an M2 by spec), roughly 20× less than an A100's HBM, so memory-bound ops are relatively more expensive and the ridge sits lower in bandwidth terms.

</details>

5. `torch.profiler` with CPU activity, run on an `mps` training step, says `aten::mm` takes 3% of the time. should you believe it?

<details><summary>answer</summary>

no. on a GPU device the CPU profiler measures dispatch, not execution: queuing a matmul takes about as long as queuing an add. the real distribution needs synchronized per-op timing or a GPU trace (Instruments' Metal System Trace). on a CPU run the same profiler is accurate.

</details>

6. MFU uses the model's FLOPs ($6N$ per token), not the FLOPs the hardware executed. why does that distinction matter for activation checkpointing (topic 05)?

<details><summary>answer</summary>

checkpointing recomputes the forward pass during backward, so the hardware does about $8N$ FLOPs per token instead of $6N$. counting executed FLOPs ("hardware FLOP utilization") would reward that wasted work; MFU counts only the useful work, so it stays an honest measure of training speed.

</details>

7. what makes an op overhead-bound, and which of these fixes help: a faster GPU, a bigger batch, operator fusion?

<details><summary>answer</summary>

its fixed per-call cost (Python, dispatch, launch, a few microseconds) exceeds its arithmetic and memory time, typical for small tensors. a faster GPU doesn't help (the GPU is idle). a bigger batch amortizes the fixed cost over more work. fusion reduces the number of calls. so does `torch.compile` or CUDA graphs.

</details>

## common bugs

- **timing without synchronizing.** GPU timings that are suspiciously fast and don't grow with size are measuring the queue, not the work. sync before reading the clock.
- **timing the first call.** compilation and allocation make the first call 10–1000× slower. warm up.
- **counting multiply-adds as FLOPs.** a multiply-add is 2 FLOPs. vendors quote peak FLOP/s with that convention, so your counts must use it too, or MFU comes out half as large.
- **forgetting the output in the byte count.** a matmul writes $mn$ elements; for skinny shapes like attention scores ($d_h \ll T$) the output dominates the traffic.
- **comparing timings from different moments on a shared machine.** measure the two variants back to back, repeat, and report the spread.
- **reading CPU-profiler numbers for GPU ops as GPU time.** they're dispatch times.

## going deeper

- Williams, Waterman & Patterson (2009), "Roofline: an insightful visual performance model for multicore architectures", *Communications of the ACM* 52(4). [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)
- He (2022), "Making deep learning go brrrr from first principles": compute-, memory- and overhead-bound, with PyTorch examples. [horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)
- Kaplan et al. (2020), "Scaling laws for neural language models", §2.1 derives $C \approx 6ND$ and the per-op FLOP table. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Chowdhery et al. (2022), "PaLM: scaling language modeling with pathways", appendix B defines model FLOP utilization. [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
- Ivanov, Dryden, Ben-Nun, Li & Hoefler (2021), "Data movement is all you need: a case study on optimizing transformers", *MLSys*: most transformer time is in memory-bound ops. [arXiv:2007.00072](https://arxiv.org/abs/2007.00072)
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* (6th ed., 2017), ch. 2 (memory hierarchy) and ch. 4 (GPUs).

## connections

- ← builds on: matmul cost (ch 02), your GPT block and trainer (ch 18 · 03, 04), FlashAttention's memory argument (ch 18 · 06).
- → used in: kernels that move ops up the roofline (02), fewer bytes per number (03), communication as another bandwidth (04, 05), memory-bound decoding (06), and the capstone, where every optimization starts with a profile. chapter 19's scaling laws are written in $6ND$.
