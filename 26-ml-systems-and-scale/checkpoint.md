# 26 · checkpoint: ML systems & scale

pass this to finish the chapter, or to skip it if you already know the material. no notes, no code from the chapter, no internet. a pen, paper and a calculator are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. what is arithmetic intensity, and how does comparing it with a machine's ridge point tell you what limits an operation?

<details><summary>answer</summary>

FLOPs per byte moved to and from memory. the ridge point is peak FLOP/s divided by bandwidth; an op with lower intensity can't use the full arithmetic rate (memory-bound: only fewer bytes help), one with higher intensity is limited by arithmetic (compute-bound: only fewer FLOPs or a faster ALU help).

</details>

2. why must a GPU timing synchronize the device, and why report the minimum of several runs?

<details><summary>answer</summary>

GPU ops are queued asynchronously; without a sync the clock measures only the queuing. noise from other processes only adds time, so the minimum estimates the undisturbed cost (report the median and range too).

</details>

3. what's the third regime besides memory- and compute-bound, and what fixes it?

<details><summary>answer</summary>

overhead-bound: fixed per-op costs (Python, dispatch, kernel launch) exceed the op's work, as for small tensors. bigger batches, fusion and compilation (fewer, larger ops) fix it; a faster GPU doesn't.

</details>

4. why does fusing a softmax speed it up, and why doesn't fusing two large matmuls do much?

<details><summary>answer</summary>

the softmax is memory-bound: eager execution writes and rereads intermediates (about 8 passes of traffic vs 2 fused), so removing traffic removes time. large matmuls are compute-bound; fusion removes some traffic but not arithmetic, which is the limit.

</details>

5. how does tiling raise a matmul's arithmetic intensity?

<details><summary>answer</summary>

each program loads tiles of $A$ and $B$ into fast memory and reuses every loaded element for a whole row or column of its output tile: a tile of side $b$ cuts DRAM loads by about $b$, raising the intensity to about $b/4$ FLOP/byte in float32.

</details>

6. float16 vs bfloat16: what does each trade, and which needs loss scaling?

<details><summary>answer</summary>

float16 (5 exponent, 10 mantissa bits) has more precision but a narrow range (max 65,504, subnormals below $6 \times 10^{-5}$), so small gradients underflow: it needs loss scaling. bfloat16 (8, 7) has float32's range and less precision: no loss scaling needed.

</details>

7. what does a dynamic loss scaler do when a gradient is infinite, and why skip rather than clip?

<details><summary>answer</summary>

skips the optimizer step entirely and halves the scale. an infinite gradient means the scale overflowed something, not a meaningful direction; stepping on it (or on its finite parts) would corrupt the weights and the optimizer's moments.

</details>

8. why do mixed-precision optimizers keep float32 master weights?

<details><summary>answer</summary>

updates smaller than half the 16-bit spacing around a weight round away entirely (in bfloat16, any relative update below about 0.4%), so learning stalls. applying updates to a float32 copy retains them.

</details>

9. how many bytes does each rank send in a ring all-reduce of $n$ bytes over $p$ ranks, and why is that independent of $p$ in the limit?

<details><summary>answer</summary>

$2(p-1)n/p$: the array is split into $p$ chunks pipelined around the ring over $2(p-1)$ steps (reduce-scatter, then all-gather), each step sending one chunk. it approaches $2n$, the bandwidth lower bound, however many ranks there are.

</details>

10. data-parallel training on 8 ranks with local batch 32 is equivalent to what single-process training, and what must hold for that to be exact?

<details><summary>answer</summary>

batch 256: identical initial weights (broadcast), disjoint shards, equal local batch sizes and gradients averaged (not summed) before every step.

</details>

11. what does each ZeRO stage shard, and what does stage 3 cost in communication?

<details><summary>answer</summary>

stage 1 the optimizer state, stage 2 also gradients, stage 3 also the weights. stages 1–2 cost the same as an all-reduce; stage 3 adds an all-gather of the weights in forward and again in backward, about 1.5× the traffic.

</details>

12. what does activation checkpointing trade, and what must it do about dropout?

<details><summary>answer</summary>

memory for compute: only segment inputs are stored and each segment's forward is recomputed during backward (about 33% more compute). the recomputation must replay the same random numbers (restore the RNG state), or the gradients belong to a different network.

</details>

13. why is a transformer MLP split column-parallel then row-parallel?

<details><summary>answer</summary>

column-parallel $W_1$ gives each device complete hidden units, so GELU needs no communication; row-parallel $W_2$ yields partial outputs combined by one all-reduce. the other order would need communication before the nonlinearity.

</details>

14. per-tensor, per-channel, per-group quantization: what's the trade-off?

<details><summary>answer</summary>

finer scale granularity limits an outlier's damage to its channel or group, at the cost of storing more scales (per group of 32 at 16 bits: 0.5 extra bits per weight). per-tensor breaks first at low bit widths.

</details>

15. why does batching raise decoding throughput almost linearly, until when?

<details><summary>answer</summary>

decoding is memory-bound: each step reads all weights once regardless of how many sequences use them, so extra sequences ride along for free until the compute time $2NB/P$ exceeds the memory time (batch around the ridge point), or the KV caches fill memory.

</details>

## 2. by hand

1. **a roofline estimate.** a GPU has $P = 20$ TFLOP/s and $W = 400$ GB/s. for float32 (4 bytes): (a) the ridge point; (b) the intensity and attainable FLOP/s of a $(4096, 1024) @ (1024, 1024)$ matmul; (c) the best-case time of a softmax over $32 \times 8 \times 512 \times 512$ scores (5 FLOPs per element, one read and one write).

<details><summary>solution</summary>

(a) $20 \times 10^{12} / 4 \times 10^{11} = 50$ FLOP/byte. (b) FLOPs $2 \cdot 4096 \cdot 1024 \cdot 1024 = 8.59 \times 10^9$; bytes $4(4096 \cdot 1024 + 1024^2 + 4096 \cdot 1024) = 37.7 \times 10^6$; $I = 228$: compute-bound, 20 TFLOP/s, 0.43 ms. (c) $67.1 \times 10^6$ elements, $2 \times 4 \times 67.1 \times 10^6 = 537$ MB at 400 GB/s: 1.34 ms ($I = 0.625$, memory-bound), three times the matmul's time for 4% of its FLOPs.

</details>

2. **memory for a run.** a 1.3B-parameter model trained with mixed-precision AdamW on 16 data-parallel ranks. per rank, how much memory do weights, gradients and optimizer state take with no sharding, ZeRO-1, ZeRO-2 and ZeRO-3? how many bytes does each rank send per step for the gradient all-reduce (bf16 gradients)?

<details><summary>solution</summary>

16 bytes × 1.3B = 20.8 GB without sharding. ZeRO-1: $(2 + 2 + 12/16) \times 1.3\text{B} = 6.2$ GB. ZeRO-2: $(2 + 14/16) \times 1.3\text{B} = 3.7$ GB. ZeRO-3: $16/16 \times 1.3\text{B} = 1.3$ GB. all-reduce of $2.6$ GB of gradients: $2 \cdot 15/16 \cdot 2.6 = 4.9$ GB sent per rank per step.

</details>

3. **rounding and the bubble.** (a) round 3.3 to float16 and to bfloat16 (give the stored values). (b) a 6-stage GPipe pipeline with 18 micro-batches: what fraction of the time is idle?

<details><summary>solution</summary>

(a) 3.3 is in $[2, 4)$, exponent 1. float16 spacing $2^{1-10} = 2^{-9}$: $3.3 \times 512 = 1689.6 \to 1690$, stored $1690/512 = 3.30078125$. bfloat16 spacing $2^{1-7} = 2^{-6}$: $3.3 \times 64 = 211.2 \to 211$, stored $211/64 = 3.296875$. (b) $(p - 1)/(m + p - 1) = 5/23 \approx 22\%$.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild topic 04's `ring_allreduce` (reduce-scatter then all-gather, chunk by chunk, each rank holding its own copies) and topic 03's `GradScaler` (scale, unscale once per step, skip on inf/NaN, backoff and growth), then check them:

```bash
MLFS_SOLUTIONS=0 pytest 26-ml-systems-and-scale/04-data-parallelism/test_exercise.py -k stage1
MLFS_SOLUTIONS=0 pytest 26-ml-systems-and-scale/03-mixed-precision/test_exercise.py -k stage3
```

the ring check compares your result bit for bit with the ring's summation order, so a direct sum won't pass.

## 4. readiness

move on when you score at least 80% on parts 1 and 2 and both blind-rebuild checks pass. the same checkpoint works as a placement test: if you pass it cold, skim the lessons' spec boxes and go straight to the capstone.
