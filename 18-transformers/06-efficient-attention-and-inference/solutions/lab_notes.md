# efficient attention & inference · lab notes

numbers below are from the reference solution on a shared laptop CPU (4 threads, other jobs running), best of 3 runs for the short timings. absolute times will differ on your machine; the ratios are the point.

## experiment 1: generation with and without the cache

**what people predict.** having computed that the cache removes about 99.8% of the token-positions for 1,000 tokens, most people predict a speedup of a few hundred, or at least that it grows in proportion to $n$.

**what happens.**

| new tokens | no cache | cached | speedup |
|---|---|---|---|
| 32 | 0.18 s | 0.04 s | 5.1× |
| 128 | 1.08 s | 0.36 s | 3.0× |
| 512 | 10.9 s | 1.43 s | 7.7× |
| 1,000 | 53.4 s | 5.33 s | 10.0× |

the outputs are identical (greedy, 64 tokens). the speedup is real and grows with length, but at 1,000 tokens it's 10×, not 500× (a reviewer's machine measured 21×: expect 10–20× depending on your CPU and its load). the uncached time grows roughly quadratically (2× the tokens costs about 5× the time from 512 to 1,000); the cached time grows faster than linearly too (3.7× for 2× the tokens). the 128-token row is the noisiest (a shared machine).

**why.** counting token-positions (515,500 against 1,015) assumes every position costs the same, and on this machine it doesn't. a small model (4 layers of width 128) run on one token at a time is dominated by fixed per-call costs: Python and PyTorch dispatch for every module, about 30 small kernel launches per layer, allocating the concatenated cache tensors. processing 500 positions in one call costs far less than 500 times one position, because the matrix products are batched and the overhead is paid once; so the uncached version is much better than its position count suggests. meanwhile the cached step isn't constant: each new token attends over all cached keys ($O(t)$, on paper 1), and `torch.cat` copies the whole cache at every step (a real implementation preallocates it). on a GPU serving a large model, where the work per token is large and the uncached version would recompute billions of multiply-adds per position, the cache's advantage is much closer to the position count, and generation without it is unthinkable.

## experiment 2: what the cache costs in memory

**what people predict.** that the cache is a small fraction of the model, as activations are during inference.

**what happens.** with a batch of 16 (the slider's default):

| model | KV cache, 1 sequence | KV cache, batch of 16 | weights (16-bit) |
|---|---|---|---|
| GPT-2 small (1,024 tokens) | 0.04 GiB | 0.6 GiB | 0.2 GiB |
| Llama-2 7B, MHA (4,096) | 2.00 GiB | **32.0 GiB** | 12.5 GiB |
| Llama-2 70B, GQA with 8 kv heads | 1.25 GiB | 20.0 GiB | 128.5 GiB |
| the 70B shape without GQA (64 kv heads) | 10.00 GiB | 160.0 GiB | 128.5 GiB |

for the 7B model, one full-length conversation's cache is a sixth of the weights, and 16 of them are 2.6× the weights. even GPT-2 small's cache passes its weights at a batch of 7 (its full-context cache is 0.15× its weights per sequence: 6.6 sequences match them). the 70B model's cache is smaller than the 7B model's per sequence, despite 2.5× as many layers, because of GQA; without GQA it would be 8× larger and exceed the weights at a batch of 13.

**why.** the weights are shared by every sequence and every token; the cache is per sequence and per token: $2 L H_{kv} d_h$ numbers for every token of every conversation. serving many long conversations at once is how you use a GPU efficiently (decode is memory-bound, so batching is nearly free compute), and the cache is what limits the batch. GQA shrinks the cache by the ratio of query heads to kv heads while keeping all 64 query heads, so the model still has 64 attention patterns; what's shared is what they read. that's why it's chosen for serving cost: Ainslie et al. (2023) show quality close to full multi-head attention, and the memory saving translates directly into larger batches and longer contexts.

## experiment 3: three ways to attend over long sequences

**what people predict.** that online softmax is faster (it's "FlashAttention"), and that its benefit is speed.

**what happens.**

| $T$ | full: time | full: score matrix | online: time | online: largest block | max difference | linear: time |
|---|---|---|---|---|---|---|
| 512 | 5.2 ms | 4 MiB | 4.9 ms | 2 MiB | 3.9e-07 | 1.1 ms |
| 1,024 | 18.0 ms | 16 MiB | 15.5 ms | 4 MiB | 4.3e-07 | 1.9 ms |
| 2,048 | 66.5 ms | 64 MiB | 56.2 ms | 8 MiB | 4.9e-07 | 2.9 ms |
| 4,096 | 290 ms | 256 MiB | 252 ms | 16 MiB | 4.8e-07 | 4.8 ms |

full and online attention both slow down about 4× per doubling of $T$ (quadratic), and online softmax is only 5–15% faster here; it agrees with the full version to $5 \times 10^{-7}$, rounding error. its largest score block grows linearly (16 MiB instead of 256 MiB at 4,096 tokens). linear attention grows roughly linearly (1.7× per doubling at the larger sizes) and is 60× faster at 4,096.

**why.** online softmax does the same multiply-adds as full attention (plus some cheap rescaling), so it can't change the quadratic time. what it changes is the memory the computation needs: one block of scores instead of the whole matrix. on a CPU, whose caches automatically keep recently used data close, the full version's large matrix costs some time but not a lot (here the blocked version is slightly faster thanks to better cache locality). on a GPU the difference is large, because the full matrix must go to and from main memory several times and attention is memory-bound: FlashAttention's speedups come from that traffic, and its memory saving is what makes 100,000-token contexts possible at all. linear attention is in a different complexity class because it computes a different function: its weights $\phi(q)^\top\phi(k)$ can't be as sharp as a softmax, and a fixed-size state can't retrieve arbitrary past tokens exactly, which is why it hasn't replaced softmax attention despite numbers like these.
