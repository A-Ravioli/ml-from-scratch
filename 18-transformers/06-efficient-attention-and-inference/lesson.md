# efficient attention & inference

## why this matters

training a transformer is beautifully parallel: a whole window of predictions in one pass. using one is not. topic 04's `generate` produces token 500 by running the model over all 499 previous tokens again, recomputing, in every layer, keys and values that haven't changed since the last step. generating $n$ tokens costs on the order of $n^2$ token-passes. at the scale of a deployed LLM, that waste would be the difference between a chat that answers in a second and one that answers in a minute.

attention has a second cost that grows with length: the $T \times T$ score matrix. a 32,000-token context in 32 heads is 32 billion scores per layer, far more than fits in a GPU's fast memory. and once generation is fixed with a cache, a third cost appears: the cache itself, which for a large model serving many users can be bigger than the model.

this topic builds the standard answers. a **KV cache** turns generation from quadratic to linear in token-passes. **online softmax**, the trick inside FlashAttention, computes exact attention one block of keys at a time without ever storing the score matrix. **grouped-query attention** shrinks the cache by sharing keys and values between heads. and **linear attention** replaces the softmax with a kernel so attention becomes a recurrence with a fixed-size state, at a price in quality. chapter 26 measures these on real hardware; chapter 27's Llama uses all but the last.

## warm-up: predict first

1. in a causal transformer, does the key computed for token 10 in layer 3 change when token 11 arrives? what about in a bidirectional (BERT) model?
2. generating 1,000 tokens from a 16-token prompt without a cache: how many token-positions go through the model in total? with a cache?
3. a softmax needs the sum of exponentials over *all* scores before it can output anything. can you compute attention while seeing the keys only a block at a time?
4. a 7-billion-parameter model serves 16 conversations of 4,096 tokens each. which is bigger: the model's weights or its KV cache?
5. $(QK^\top)V$ and $Q(K^\top V)$ give the same matrix. which is cheaper for $T = 4096$ tokens of width 64, and why can't softmax attention use the cheap one?

## prerequisites check

- [ ] your `MultiHeadAttention` calls `cache.update(k, v)` and your causal mask aligns to the end when $T_q < T_k$ (→ topic 01)
- [ ] your `GPT` reads `cache.seq_len` for positions and passes `cache[i]` to block $i$; `GPT.generate` and `sample_next_token` work (→ topic 04)
- [ ] you can compute a softmax stably by subtracting the maximum (→ ch 00, ch 08)
- [ ] you know the cost of a matrix product in multiply-adds, and the associativity of matrix products (→ ch 02)

## core ideas

### idea 1: the KV cache

in a causal transformer, the representation of token $t$ in every layer depends only on tokens $1, \dots, t$. so when token $t + 1$ arrives, nothing about tokens $1, \dots, t$ changes: their keys and values in every layer are exactly what they were at the previous step. the only new work is for the new token: its query, key and value, its attention over all keys so far, and its MLP.

a **KV cache** stores, for each layer, the keys and values of all tokens processed so far, in the heads layout $(B, H, T, d_h)$. at each step the attention module computes the new token's $k$ and $v$, appends them, and attends with the new query over the full cached keys. that's the `cache.update(k, v)` hook your `MultiHeadAttention` has called since topic 01, the reason your causal mask aligns to the end (one query, many keys: the query is the *last* position), and the reason GPT starts positions at `cache.seq_len`. now you build the cache itself: one small object per layer, and one per model that holds them.

caching is exact only because of causality. in BERT, every token attends to every other, so a new token changes the representation of all the old ones, and nothing can be reused.

> **spec.** `LayerKVCache()` starts empty; its `seq_len` (an attribute or property) is the number of cached positions, 0 when empty; `update(k, v)` appends `k`, `v` `(B, H, T_new, d_h)` along dimension $-2$ and returns the full `(k, v)`. `KVCache(n_layers)` holds `n_layers` `LayerKVCache`s; `cache[i]` returns layer $i$'s (always the same object); `seq_len` is layer 0's count; `reset()` empties every layer.

> **🔮 predict:** after a prompt of 5 tokens and 3 generated ones, what is the shape of layer 0's cached keys for a batch of 2 with 4 heads of width 16? what is `cache.seq_len`?

> **▶ build: stage 1.** implement `LayerKVCache` and `KVCache`.

### idea 2: prefill and decode

cached generation has two phases. **prefill**: run the whole prompt through the model in one pass with an empty cache. the causal mask works as usual ($T_q = T_k$), the cache fills with the prompt's keys and values, and the logits at the last position give the first new token. **decode**: feed that single token with the cache. its position is `cache.seq_len`, its query attends to all cached keys plus its own, and its logits give the next token. repeat.

prefill is one big parallel pass and is limited by arithmetic. decode does very little arithmetic per step (one token's worth of matrix-vector products) but must read every weight and the whole cache from memory each time, so it's limited by memory bandwidth. that asymmetry shapes how LLMs are served (chapter 26).

two differences from `GPT.generate`. the cache can't "crop" to the last `block_size` tokens: the cached keys were computed with their positions, and dropping the oldest would leave the rest at the wrong positions. so `generate_cached` refuses to go past `block_size` (models without a hard limit, like chapter 27's RoPE-based Llama, have no `block_size` attribute and aren't limited). and it must work for any causal LM that follows the protocol (`model(idx, cache=cache)` returns `(logits, loss)`, `.blocks` gives the number of layers), so that chapter 27 can reuse it unchanged.

> **spec.** `generate_cached(model, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None)`: creates a `KVCache(len(model.blocks))`, prefills `idx` `(B, T)` in one call, then calls the model with exactly one new token `(B, 1)` and the cache per step, picking tokens with `sample_next_token` from the last position's logits (same options as `GPT.generate`, so with the same generator it returns the same tokens). returns `(B, T + max_new_tokens)`. raises `ValueError` if the model has a `block_size` attribute and `T + max_new_tokens` exceeds it. runs in eval mode under `torch.no_grad()`, restoring the mode.

> **🔮 predict:** your cached generation pushes about 500 times fewer token-positions through the model than `GPT.generate` for 1,000 new tokens. how much faster will it be on your laptop?

> **▶ build: stage 2.** implement `generate_cached`.

### idea 3: attention a block at a time

the score matrix is the memory problem: storing $T \times T$ scores per head, so the softmax can normalize each row, is what limits context length. but you never need the whole row at once. the output for one query is

$$o = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}},$$

two running sums, a numerator and a denominator, and sums can be accumulated a block at a time. the only obstacle is numerical: $e^{s_j}$ overflows for large scores, and the usual fix (subtract the maximum score) seems to need all the scores first.

**online softmax** (Milakov & Gimelshein, 2018) keeps a *running* maximum $m$ and fixes the sums when it changes. process the keys in blocks. for each block, compute its scores $s$, the new maximum $m' = \max(m, \max_j s_j)$, and

$$\ell \leftarrow \ell\, e^{m - m'} + \sum_{j \in \text{block}} e^{s_j - m'}, \qquad a \leftarrow a\, e^{m - m'} + \sum_{j \in \text{block}} e^{s_j - m'} v_j, \qquad m \leftarrow m'.$$

the factor $e^{m - m'} \le 1$ rescales everything accumulated so far as if it had been computed with the new maximum all along (on paper 2). after the last block, $o = a / \ell$: exactly softmax attention, never holding more than one block of scores. FlashAttention (Dao et al., 2022) runs this loop inside one GPU kernel, with blocks of queries and keys sized to fit in on-chip memory, so the score matrix never touches the GPU's main memory at all. the arithmetic is the same as ordinary attention; the speedup comes entirely from memory traffic. on a CPU, which caches small blocks automatically, you'll see the memory saving but not much speedup (lab, experiment 3).

causal masking works blockwise too: mask the scores in each block (end-aligned as always), and a row that has seen only masked keys so far simply has $m = -\infty$, $\ell = 0$ until a real key arrives.

> **spec.** `online_softmax_attention(q, k, v, *, block_size=64, is_causal=False)`: `q` `(..., Tq, d)`, `k`, `v` `(..., Tk, d)`/`(..., Tk, d_v)`. processes keys in chunks of at most `block_size`, keeping a running max, normalizer and output, and never computes scores for more than `block_size` keys at once. `is_causal` aligns to the end, as in topic 01. returns only the output `(..., Tq, d_v)`, equal (up to rounding) to `scaled_dot_product_attention`'s.

> **▶ build: stage 3.** implement `online_softmax_attention`.

### idea 4: sharing keys and values between heads

a KV cache holds $2 \times L \times H \times d_h$ numbers per token: keys and values, every layer, every head. for a model the size of Llama-2-7B, that's 0.5 MB per token in 16-bit, so one 4,096-token conversation needs 2 GiB, and serving many conversations at once quickly needs more memory for caches than for the model.

**multi-query attention** (Shazeer, 2019) keeps all $H$ query heads but only *one* key head and one value head, shared by all queries: the cache shrinks $H$-fold, at some cost in quality. **grouped-query attention** (Ainslie et al., 2023) interpolates: $H_{kv}$ key/value heads, each shared by a group of $H / H_{kv}$ query heads. Llama-2-70B uses 64 query heads and 8 kv heads, an 8× smaller cache, with quality close to full multi-head attention. in code, after projecting to $H_{kv}$ key and value heads, each is **repeated** $H/H_{kv}$ times (in a row: kv head 0 for query heads $0, \dots, H/H_{kv} - 1$, and so on), and attention proceeds as usual. chapter 27 builds the full `GroupedQueryAttention` on top of these pieces; the cache stores the $H_{kv}$ heads, before repetition.

> **spec.** `repeat_kv(x, n_rep)`: `(B, H_kv, T, d_h)` → `(B, H_kv · n_rep, T, d_h)` where output head $h$ is input head $\lfloor h / n_\text{rep} \rfloor$. `kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, *, batch_size=1, bytes_per_value=2)` = $2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot B \cdot \text{bytes}$, an int.

> **▶ build: stage 4.** implement `repeat_kv` and `kv_cache_bytes`.

### idea 5: linear attention

even with online softmax, attention does $T^2 d$ work, because every query meets every key. Katharopoulos et al. (2020) noticed that if the similarity were a dot product of feature maps, $\operatorname{sim}(q, k) = \phi(q)^\top \phi(k)$, instead of $e^{q \cdot k}$, the sums could be reorganized:

$$o_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j)\, v_j}{\sum_j \phi(q_i)^\top \phi(k_j)} = \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j) v_j^\top\right)}{\phi(q_i)^\top \sum_j \phi(k_j)}.$$

the matrix $S = \sum_j \phi(k_j) v_j^\top$ ($d \times d_v$) and the vector $z = \sum_j \phi(k_j)$ are computed once and shared by every query: $O(T d d_v)$ work instead of $O(T^2 d)$. with $\phi(x) = \operatorname{elu}(x) + 1$ (positive, so the weights are positive and normalize to a weighted average) this is a well-defined attention, linear in $T$.

for a causal model the sums run over $j \le i$, which is a recurrence: $S_i = S_{i-1} + \phi(k_i)v_i^\top$, $z_i = z_{i-1} + \phi(k_i)$, and the output uses $S_i, z_i$. a causal linear-attention transformer is an RNN whose hidden state is the matrix $S$: generation needs constant memory and constant time per token, no growing cache. the catch is expressiveness: $\phi(q)^\top \phi(k)$ can't be as sharp as a softmax over large scores (a fixed $d$-dimensional state can't store and retrieve arbitrarily many key–value pairs exactly), and plain linear attention trails softmax attention in quality. the line of work it started (RetNet, RWKV, Mamba and other state-space models) is about recovering that quality while keeping the recurrence.

> **spec.** `linear_attention(q, k, v, *, is_causal=False, eps=1e-6)`: with $\phi = \operatorname{elu} + 1$, returns $\phi(q_i)^\top S / (\phi(q_i)^\top z + \epsilon)$ for every query, with $S, z$ summed over all keys, or over $j \le i$ if `is_causal` (prefix sums), without forming any $T \times T$ matrix; shapes as in topic 01 with $T_q = T_k$ for causal. `linear_attention_step(q, k, v, state=None, *, eps=1e-6)`: one causal step with `q`, `k` `(..., d)`, `v` `(..., d_v)` and `state = (S, z)` (`None` = zeros); adds this token's $\phi(k)v^\top$ and $\phi(k)$ to the state, then returns `(out (..., d_v), (S, z))`.

> **▶ build: stage 5.** implement `linear_attention` and `linear_attention_step`.

## worked example

**online softmax by hand.** one query, four keys with scores $s = (1, 3, 2, 5)$, scalar values $v = (1, 2, 3, 4)$, blocks of two.

| | $m$ | $\ell$ | $a$ |
|---|---|---|---|
| start | $-\infty$ | 0 | 0 |
| block 1 (scores 1, 3) | 3 | $e^{-2} + e^{0} = 1.1353$ | $1 \cdot e^{-2} + 2 \cdot 1 = 2.1353$ |
| block 2 (scores 2, 5) | 5 | $1.1353\,e^{-2} + e^{-3} + e^{0} = 1.2035$ | $2.1353\,e^{-2} + 3e^{-3} + 4 = 4.4384$ |

output $a/\ell = 3.688$. directly: weights $\propto (e^1, e^3, e^2, e^5) = (2.72, 20.09, 7.39, 148.41)$, sum 178.61, output $(2.72 + 40.17 + 22.17 + 593.65)/178.61 = 3.688$. ✓

**cache sizes.** a Llama-2-7B-shaped model (32 layers, 32 kv heads of 128) at 4,096 tokens in 16-bit: $2 \cdot 32 \cdot 32 \cdot 128 \cdot 4096 \cdot 2 = 2{,}147{,}483{,}648$ bytes, 2 GiB per sequence. with 8 kv heads (GQA), 512 MiB.

**generation work.** a prompt of $p = 16$ tokens and $n = 1000$ new ones: without a cache, step $i$ runs the model on $p + i$ positions, $\sum_{i=0}^{n-1}(p + i) = np + n(n-1)/2 = 515{,}500$ token-positions; with a cache, $p + (n - 1) = 1{,}015$. a 508× reduction in token-passes.

your code must reproduce the first two; the lab measures what the third buys.

## on paper

1. **where the speedup goes.** the cached decode step processes one token, yet its cost isn't constant. which part of the step grows with the number of cached tokens $t$, and how does its cost compare with the projections and MLP for $t \ll d$ and $t \gg d$?

<details><summary>solution</summary>

the projections and MLP cost $\approx 12d^2$ multiply-adds per layer for the one token, independent of $t$. attention costs $2td$ (scores against $t$ cached keys, then the weighted sum of $t$ values), plus reading the $2td$ cached numbers from memory. for $t \ll 6d$ the fixed part dominates and steps cost about the same; for long contexts attention (and the memory traffic of reading the cache) dominates, and the step time grows linearly in $t$. the total for $n$ tokens is still quadratic in $n$ in the attention part, but with a constant of $2d$ instead of the full model's $12d^2$ per token.

</details>

2. **online softmax is exact.** show by induction that after processing keys $1, \dots, J$, $m = \max_{j \le J} s_j$, $\ell = \sum_{j \le J} e^{s_j - m}$ and $a = \sum_{j \le J} e^{s_j - m} v_j$.

<details><summary>solution</summary>

before any block, $m = -\infty$, $\ell = a = 0$: the empty sums. suppose the invariant holds after $J$ keys, and a block $B$ arrives. $m' = \max(m, \max_{j \in B} s_j) = \max_{j \le J'} s_j$. then $\ell\,e^{m - m'} = \sum_{j \le J} e^{s_j - m}e^{m - m'} = \sum_{j \le J} e^{s_j - m'}$, and adding $\sum_{j \in B} e^{s_j - m'}$ gives $\sum_{j \le J'} e^{s_j - m'}$. the same algebra holds for $a$. at the end $a/\ell = \sum_j e^{s_j - m}v_j / \sum_j e^{s_j - m}$, and the $e^{-m}$ cancels: exactly softmax attention. every exponent is $\le 0$, so nothing overflows.

</details>

3. **no cache for BERT.** explain why a bidirectional encoder can't reuse cached keys and values when a token is appended, using the dependence of layer-2 keys on layer-1 outputs.

<details><summary>solution</summary>

in layer 1, every token's output is an attention over *all* tokens, so appending token $T + 1$ changes the layer-1 output of every token $t \le T$ (each now also attends to $T + 1$). the layer-2 keys and values are projections of the layer-1 outputs, so they all change too, and so on up the stack. only the first layer's keys and values (projections of the embeddings) are unchanged. in a causal model, token $t$'s layer-1 output doesn't depend on later tokens, so by induction nothing at positions $\le T$ ever changes.

</details>

4. **linear attention's costs.** for $T$ tokens with $d = d_v = 64$, compare the multiply-adds of softmax attention and (non-causal) linear attention, and the memory of the KV cache after $T$ tokens with the recurrent state of causal linear attention. at what $T$ do they break even?

<details><summary>solution</summary>

softmax: $2T^2 d$. linear: $\phi(K)^\top V$ costs $T d d_v$, and $\phi(Q)S$ another $T d d_v$: $2Td^2$. equal at $T = d = 64$; for $T = 4096$ linear attention does 64× less work. memory per head: the cache holds $2Td$ numbers and grows; the state holds $d d_v + d = 4{,}160$ numbers forever. they're equal at $T \approx 32$: beyond a few dozen tokens the recurrent state is smaller, and at 4,096 tokens it's 126× smaller.

</details>

## experiment

the lab times generation with and without your cache for 32 to 1,000 new tokens, and compares the measured speedup with the 500-fold reduction in work. then it uses your `kv_cache_bytes` to put real models' caches next to their weights, with a slider for how many conversations are served at once. finally it times three ways to attend over up to 4,096 tokens: the full score matrix, your online softmax, and linear attention.

## check your understanding

1. why does the KV cache store keys and values but not queries?

<details><summary>answer</summary>

a query is only used once: to compute its own token's attention, at the step that token is processed. keys and values are used by every *later* query. once a token's output is computed, its query is never needed again.

</details>

2. your cached generation matches `GPT.generate` for the first new token and then diverges. name the most likely bug.

<details><summary>answer</summary>

the position of decode tokens: the new token is embedded at position 0 (or at the wrong offset) instead of `cache.seq_len`. the first token comes from the prefill, which is correct; every later one is at the wrong position. a second candidate: a causal mask that isn't end-aligned, which blocks the single new query from all keys but the first.

</details>

3. why can't `generate_cached` crop the context like `GPT.generate` does when it reaches `block_size`?

<details><summary>answer</summary>

the cached keys and values were computed with the positions their tokens had at the time. after dropping the oldest token, every remaining token should move one position earlier, which changes its embedding and so every cached key and value in every layer: the whole cache would have to be recomputed. (models with relative positions, like ALiBi, can slide a window over the cache, which is one of their selling points.)

</details>

4. FlashAttention does the same multiply-adds as ordinary attention. how can it be several times faster on a GPU?

<details><summary>answer</summary>

ordinary attention writes the $T \times T$ scores to the GPU's main memory, reads them back for the softmax, writes the weights, and reads them again for the product with $V$. for long sequences, moving those matrices takes longer than computing them: attention is memory-bound. FlashAttention keeps each block of scores in on-chip memory and never writes the matrix out, so it moves $O(Td)$ bytes instead of $O(T^2)$.

</details>

5. multi-query attention shrinks the cache $H$-fold. why not always use it?

<details><summary>answer</summary>

all heads must then attend over the same keys and read the same values; they differ only in their queries. that's a real loss of capacity (each head can choose *where* to look but not *what* it reads in its own subspace), and quality and training stability suffer somewhat. grouped-query attention with a few kv heads recovers most of the quality at most of the savings.

</details>

## common bugs

- **appending along the wrong axis.** `torch.cat(..., dim=1)` on `(B, H, T, d_h)` adds heads, not tokens. the cache "works" and attention silently attends over garbage.
- **one cache object for all layers.** `[LayerKVCache()] * n` creates $n$ references to the same object; every layer appends to it.
- **feeding the whole sequence each step.** passing `out` instead of the new token to a cached model double-counts the past (and the cache grows by the whole sequence every step).
- **forgetting to rescale the accumulator.** rescaling $\ell$ but not $a$ when the max changes gives outputs that are right when the first block holds the maximum and wrong otherwise. test with the largest score in the last block.
- **`exp(-inf - (-inf))`.** a row whose keys are all masked so far has $m = -\infty$, and $m - m'$ is NaN. substitute 0 for an infinite maximum before subtracting.
- **tiling kv heads instead of repeating.** `x.repeat(1, n_rep, 1, 1)` gives head order $0, 1, 2, 0, 1, 2$; query head $h$ must use kv head $h // n_\text{rep}$.

## going deeper

- Pope et al. (2023), "Efficiently scaling transformer inference", *MLSys*. prefill vs decode, memory bandwidth, and KV-cache memory at scale. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102)
- Milakov & Gimelshein (2018), "Online normalizer calculation for softmax". [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Dao, Fu, Ermon, Rudra & Ré (2022), "FlashAttention: fast and memory-efficient exact attention with IO-awareness", *NeurIPS*. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Shazeer (2019), "Fast transformer decoding: one write-head is all you need". multi-query attention. [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón & Sanghai (2023), "GQA: training generalized multi-query transformer models from multi-head checkpoints", *EMNLP*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Katharopoulos, Vyas, Pappas & Fleuret (2020), "Transformers are RNNs: fast autoregressive transformers with linear attention", *ICML*. [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
- Gu & Dao (2023), "Mamba: linear-time sequence modeling with selective state spaces". where the recurrent line of work went. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

## connections

- ← builds on: the cache hook and end-aligned causal mask (topic 01), RoPE offsets (02), GPT's positions and generate (04), stable softmax (ch 00, 08), RNNs with a carried state (ch 17).
- → used in: the capstone's cached inference and timing; the roofline and memory accounting of chapter 26; chapter 27's Llama with grouped-query attention, its cache and `generate_cached`, and speculative decoding (which verifies several drafted tokens in one cached pass).
