# writing kernels · lab notes

numbers below are from the reference solution on an Apple M2 laptop (8 cores, `mps` GPU, Metal), measured once while other jobs shared the machine. best of 5–7 for every timing. yours will differ; compare ratios, and rerun anything surprising.

## experiment 1: what fusion saves

**what people predict.** having counted 4× less traffic, most people predict a speedup close to 4× on both devices, or a little less.

**what happens.** on the interpreter, $64 \times 300$: the fused kernel moves 38,400 elements, the five-kernel eager version 153,856, **4.01×**. then the real thing, 16M floats ($4096 \times 4096$):

| device | eager (5 ops) | `torch.softmax` (fused) | eager / fused | fused, effective bandwidth |
|---|---|---|---|---|
| cpu | 15.7 ms | 8.9 ms | 1.8× | 15 GB/s |
| mps | 13.3 ms | 2.5 ms | **5.4×** | 55 GB/s |

**why.** the extra 0.01 is the per-row traffic: the row maxima and sums (64 values each) are each stored by one kernel and loaded by the next, $4 \times 64 = 256$ extra elements on top of $8 \times 19{,}200 = 153{,}600$. on the GPU the fused softmax runs at 55 GB/s, about three quarters of the 72 GB/s copy bandwidth topic 01 measured, so it's memory-bound, and the eager version loses even more than the traffic ratio says: each of its five kernels also pays launch and scheduling overhead, and the composite allocates two intermediate 64 MB tensors. on the CPU the ratio is only 1.8×: the CPU's fused softmax isn't bandwidth-limited (15 GB/s is far below the 53 GB/s the CPU can copy), it's limited by computing 16M exponentials on a few cores, and eager PyTorch's elementwise ops are multithreaded and vectorized, so the extra passes cost less than their bytes suggest. fusion pays most where an op is truly memory-bound and launches are expensive: on GPUs.

## experiment 2: tiling, counted and timed

**what people predict.** that the 16×16 tiled Metal matmul is about 16× faster than the naive one (matching the traffic reduction), and gets within 2× of PyTorch.

**what happens.** on the interpreter, the loads halve with every doubling of the tile side, from 1.05M (tile 4) to 65,536 (tile 64) for a $128^3$ problem, against 4.2M one element at a time and 32,768 for "each input once". on the GPU:

| n | naive Metal | tiled Metal (16×16) | `torch` (`a @ b`) |
|---|---|---|---|
| 256 | 0.31 ms (107 GFLOP/s) | 0.32 ms (105) | 0.17 ms (194) |
| 1024 | 9.3 ms (232) | 5.7 ms (380) | 1.1 ms (1,994) |
| 2048 | 74 ms (232) | 43 ms (397) | 6.4 ms (2,699) |

tiling gives **1.6–1.7×** at the larger sizes, nothing at 256; PyTorch is another **5–7×** faster than the tiled kernel. all three agree to float32 rounding.

**why.** the naive kernel's traffic isn't really $2n^3$ elements from DRAM: neighbouring threads in a threadgroup read the same row of $A$ and neighbouring columns of $B$ at the same time, and the GPU's caches serve most of those reads. so tiling by hand removes less DRAM traffic than the interpreter's count (which has no cache) says; what it does guarantee is reuse that doesn't depend on the cache's luck. at $n = 256$ both kernels are overhead-bound (a fraction of a millisecond, dominated by dispatch). PyTorch's matmul on `mps` calls Apple's Metal Performance Shaders, which use larger, register-blocked tiles (each thread computes several outputs and keeps them in registers), the GPU's SIMD-group matrix instructions, double-buffered loads that overlap memory with arithmetic, and tuning per chip. that's the gap between a correct tiled kernel and a production one, and it's why you write kernels for ops that libraries don't fuse, not for plain matmuls.

## experiment 3: your Metal softmax against PyTorch

**what people predict.** that a 30-line kernel loses to Apple's `torch.softmax` everywhere, but beats the five-op eager version.

**what happens.** 16M floats, rows of different lengths (ms):

| row length | eager (5 ops) | `torch.softmax` | your Metal kernel |
|---|---|---|---|
| 128 | 11.5 | 3.0 | 6.9 |
| 512 | 11.7 | 2.0 | 2.8 |
| 1,024 | 11.6 | 2.1 | 2.2 |
| 4,096 | 11.5 | 2.3 | **2.1** |
| 16,384 | 11.7 | 3.2 | **3.2** |

the Metal kernel beats eager PyTorch by 1.7–5.5× at every length, and ties or slightly beats `torch.softmax` for rows of 1,024 and longer; it's 2.3× slower for rows of 128. (differences under ~5% are within this machine's noise.)

**why.** for long rows, both fused kernels are memory-bound at 60–65 GB/s effective (two passes), close to the copy bandwidth, so there's nothing left to win: a simple kernel that touches DRAM once each way is as good as an expert one. your kernel reads each row three times (max, sum, write), but for rows up to a few thousand floats the second and third reads come from cache. for short rows your kernel wastes most of its 256 threads: a 128-element row leaves half of them idle in the strided loops and still pays two full tree reductions with eight barriers each, per row, for 131,072 rows. a production kernel adapts its threads per row to the row length (or processes several short rows per threadgroup) and reduces within SIMD groups with `simd_max`/`simd_sum`, with no threadgroup memory or barriers. the drop at 16,384 for everyone is the cache: a 64 KB row's repeated reads no longer stay resident.
