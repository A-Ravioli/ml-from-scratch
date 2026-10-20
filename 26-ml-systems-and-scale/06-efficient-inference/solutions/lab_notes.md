# efficient inference · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU with 4 threads for experiments 1–3, `mps` GPU for experiment 4), MNIST with 20,000 training images, one run, other jobs sharing the machine. accuracies are on the 10,000 test images (±0.2 points from sampling alone); experiment 3 gives the spread over 3 seeds.

## experiment 1: accuracy against bits

**what people predict.** that 4-bit per-tensor quantization costs several points, per-channel a little, and that everything breaks somewhere around 4 bits.

**what happens.** the float32 MLP scores 95.89%.

| scheme | 8 bits | 6 | 5 | 4 | 3 | 2 |
|---|---|---|---|---|---|---|
| per tensor | 95.9 | 96.0 | 95.9 | 96.0 | 94.9 | **11.3** |
| per channel | 95.9 | 95.9 | 95.9 | 95.8 | 95.7 | 49.8 |
| per group of 32 | 95.9 | 96.0 | 96.0 | 96.0 | 95.8 | **94.0** |

down to 4 bits, nothing measurable is lost by any scheme (the ±0.1 differences are noise). at 3 bits per-tensor quantization starts to slip (−1 point), and at 2 bits (levels $-1, 0, +1$ times a scale) per-tensor collapses to chance, per-channel keeps half the accuracy, and per-group keeps almost all of it. your `quantize_linears` int8 model scores 95.90% (vs 95.89%), with 273 KB of weights against 1,077 KB, and its forward pass on 512 images takes **0.75 ms against 0.44 ms**: smaller, and slower.

**why.** an MNIST MLP's weights are well behaved (no huge outliers), so even a coarse grid preserves the ranking of activations that classification needs, until the grid is too coarse to represent the weight distribution at all. at 2 bits each weight becomes $-s$, 0 or $+s$; with one $s$ for the whole matrix, set by its largest weight, almost every weight rounds to 0, and the network loses its signal. per-channel scales adapt to each row's range, and groups of 32 adapt to local ranges, which is why group-wise 4-bit (and even 3-bit) formats are the standard for LLMs, whose weights and activations do have outliers that make per-tensor schemes fail much earlier (Dettmers et al., 2022). the int8 model is slower because its forward pass dequantizes the whole weight to float32 in memory on every call, then runs the same float matmul: extra work, same arithmetic. on a memory-bound workload a fused int8 kernel would read a quarter of the bytes and be nearly 4× faster than float32; this MLP on a CPU isn't memory-bound, and has no such kernel.

## experiment 2: pruning

**what people predict.** that half the weights can go with little loss (true), and that a 90%-sparse model is about 10× faster.

**what happens.**

| scope | 0% | 50% | 70% | 80% | 90% | 95% | 98% |
|---|---|---|---|---|---|---|---|
| global | 95.9 | 96.0 | 95.1 | 92.8 | 74.1 | 41.8 | 19.0 |
| per layer | 95.9 | 96.0 | 94.3 | 88.1 | 60.9 | 29.1 | 16.2 |

without any fine-tuning, half the weights can be removed for free, and global pruning holds up better from 70% on (at 90%: 74% against 61%). at 90% global sparsity the three layers keep 9%, 13% and 23% of their weights. multiplying a 90%-sparse $256 \times 256$ weight by a $256 \times 2048$ input takes 0.26 ms dense and **4.1 ms in sparse CSR format, 16× slower** in this run (a reviewer measured 7×; expect somewhere around 5–20× depending on the machine and its load).

**why.** global pruning removes more from the big first layer (784 inputs, many near-zero weights for the blank border pixels of MNIST) and protects the small output layer, where every weight matters; per-layer pruning takes the same fraction from each and damages the output layer earlier. on speed: a dense matmul at this size runs near the CPU's peak with vectorized, cache-friendly loops; a sparse format stores an index per nonzero, reads them irregularly, and can't use the same vector units. unstructured sparsity typically breaks even only above ~95–99% sparsity on CPUs and GPUs, which is why practical pruning is structured (whole neurons, heads, 2:4 patterns) and why the accuracy table, not a speedup, is what magnitude pruning buys here (plus compression, if you store the sparse weights).

## experiment 3: distillation

**what people predict.** that distillation on the same 500 images adds a point or two over labels alone, and the transfer set more.

**what happens.**

| student (784 → 32 → 10) | mean test accuracy | range over 3 seeds |
|---|---|---|
| labels only (500 images) | 86.26% | 85.75–86.87% |
| distilled, same 500 images ($T = 4$, $\alpha = 0.9$) | 85.10% | 84.47–85.44% |
| distilled + 5,000 unlabelled images | **89.41%** | 89.06–89.75% |

on the same 500 images, distillation is about a point *worse* than training on the labels, a gap larger than the seed spread. with the teacher labelling 5,000 extra images, the student gains 3 points.

**why.** soft targets carry more information per example than one-hot labels, but with $\alpha = 0.9$ and only 500 examples, the student mostly learns to imitate a teacher that is itself 4% wrong on held-out data, and 30 epochs of matching soft targets on 500 images fits them closely without seeing anything new; the hard labels alone regularize differently, and here slightly better. (the right $\alpha$, $T$ and epoch count would be tuned on a validation split, which this lab doesn't do; a different setting could reverse the sign.) the transfer set is where distillation earns its reputation: the teacher turns unlabelled images into rich training signal, so the student effectively trains on 5,500 examples instead of 500. that's how distillation is used in practice: a large model labels (or generates) far more data than humans ever could, and the small model learns from all of it.

## experiment 4: batching

**what people predict.** a roughly linear growth in tokens per second with the batch, and that the roofline model predicts the batch-1 speed within a factor of 2.

**what happens.** on `mps` (measured here: 59 GB/s copy bandwidth, 2.5 TFLOP/s peak), model M generating 64 tokens:

| batch | measured tokens/s | roofline tokens/s | measured / roofline |
|---|---|---|---|
| 1 | 64 | 8,063 | 0.008 |
| 4 | 247 | 32,251 | 0.008 |
| 16 | 1,009 | 129,006 | 0.008 |
| 64 | 4,506 | 516,024 | 0.009 |

throughput grows about **70×** from batch 1 to 64 in this run, almost exactly linearly, as the lesson predicts. but the roofline model is far too optimistic: by about 120× in this run, 53–120× across reruns, and not by the same factor at every batch size.

**why.** model M is 7.3 MB: reading its weights takes about 0.12 ms at 59 GB/s, and its arithmetic per token even less. but each decoding step launches dozens of small kernels (four blocks of attention, layernorms, MLPs, the cache's `torch.cat`, sampling) and waits for the result to choose the next token, about 15 ms per step on this machine. that's topic 01's **overhead-bound** regime: a fixed cost per step, independent of the batch, which is exactly why batching is free here too, just for a different reason than the lesson's memory-bound argument. to model it you'd measure the per-step overhead (a batch-1 step's time) and use $\text{tokens/s} = B / \max(t_\text{overhead}, t_\text{memory}, t_\text{compute})$. a 7B model on the same machine (14 GB in 16-bit, 0.24 s per weight read at 59 GB/s) would be firmly memory-bound, the lesson's regime, at about 4 tokens per second per sequence.
