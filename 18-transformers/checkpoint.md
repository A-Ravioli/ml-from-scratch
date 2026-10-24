# 18 · checkpoint: transformers

pass this to finish the chapter, or to skip it if you already know the material. no notes, no code from the chapter, no internet. a pen, paper and a calculator are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. why is the dot product divided by $\sqrt{d}$ before the softmax, and what goes wrong at large $d$ without it?

<details><summary>answer</summary>

with unit-variance entries, $q \cdot k$ has variance $d$, so raw scores grow like $\sqrt d$. a softmax over scores spread by $\pm\sqrt d$ is nearly one-hot, and its Jacobian $\operatorname{diag}(a) - aa^\top$ is nearly zero, so almost no gradient reaches the queries and keys. dividing by $\sqrt d$ keeps the scores at unit scale for any width (topic 01's lab: unscaled attention at $d = 512$ stalls at 67% accuracy on a lookup task the scaled version solves in 25 steps).

</details>

2. self-attention without positional information is permutation-equivariant. what does that mean, and which component breaks the symmetry in a GPT with no positional encoding?

<details><summary>answer</summary>

shuffling the input tokens shuffles the outputs the same way; the model sees a set. in a causal model the mask is tied to positions (position $t$ averages over exactly $t + 1$ values), so it breaks the symmetry and gives a weak positional signal even without encodings.

</details>

3. with fewer queries than keys, how is a causal mask aligned in this course, and why does a KV cache need that convention?

<details><summary>answer</summary>

to the end: the queries are the last $T_q$ positions, so query $i$ sees keys $j \le T_k - T_q + i$. with a cache, one new query attends over all cached keys plus its own; aligning to the top left would let it see only the first key.

</details>

4. what does multi-head attention buy over one head of the same total width, at the same cost?

<details><summary>answer</summary>

several independent attention distributions per token: one head can track the previous token while another finds the subject of the verb. one softmax produces one weighted average, so a single head has to blur relations together. the cost is the same because the width is split across heads, not added.

</details>

5. RoPE rotates queries and keys but not values. why does the score then depend only on relative position, and what can RoPE not express that a learned absolute embedding can?

<details><summary>answer</summary>

rotations are orthogonal and compose by adding angles, so $(R_m q)^\top (R_n k) = q^\top R_{n-m} k$. because values are untouched, a sequence of identical tokens produces identical outputs at every position (any average of identical values is that value): a RoPE model can't count positions in a constant sequence; an additive absolute embedding can.

</details>

6. why did the sinusoidal model of topic 02's lab fall apart past its training length while ALiBi didn't?

<details><summary>answer</summary>

the network had only ever seen $x + \text{PE}(p)$ for $p < 64$; at larger $p$ the slow sinusoids reach values never seen in training, and since the encoding is added to the residual stream, every layer receives unfamiliar inputs. ALiBi only adds a distance penalty to the scores, so a longer sequence just means larger penalties on far tokens, nothing new.

</details>

7. write the pre-LN and post-LN residual updates. which one trains a 24-layer model without warmup, and why?

<details><summary>answer</summary>

pre-LN: $x \leftarrow x + f(\text{LN}(x))$; post-LN: $x \leftarrow \text{LN}(x + f(x))$. pre-LN keeps an untouched identity path from the output to every layer, so gradients are well-scaled from the start. post-LN routes the shortcut through $2L$ layer norms; at initialization its last layers get large effective updates, and without warmup it stalls (topic 03's lab: at the unigram loss, 4.78 bits per character).

</details>

8. where are most of a transformer block's parameters, and what does that part do?

<details><summary>answer</summary>

in the MLP: $8d^2$ of the $12d^2$. it acts on each position independently: attention moves information between positions, the MLP transforms it, and it's thought to store much of a language model's factual knowledge.

</details>

9. what does weight tying share in a GPT, and why is it reasonable?

<details><summary>answer</summary>

the output head's $V \times d$ matrix is the token embedding matrix, so a token's logit is the dot product of the final hidden state with that token's own embedding. tokens that are predicted in similar contexts benefit from similar embeddings anyway; tying halves the vocabulary parameters and trains rarely-seen input tokens through the output softmax on every step.

</details>

10. greedy decoding produces the text a model finds most probable. why is it usually bad text, and what do top-k and top-p fix?

<details><summary>answer</summary>

maximizing each step's probability falls into loops: once a phrase is its own most likely continuation, a deterministic decoder repeats it forever (topic 04's lab: 7% distinct chunks). plain sampling avoids loops but sometimes picks from the long tail of individually unlikely tokens and derails. top-k and top-p remove the tail and keep the randomness; top-p adapts the cut to the model's confidence.

</details>

11. why can't BERT be pretrained by next-token prediction, and what is the 80/10/10 rule for?

<details><summary>answer</summary>

it's bidirectional, so position $t$ can attend to $t + 1$ and copy the answer. masked LM removes the answer from the input. of the 15% selected, 80% become [MASK], 10% a random token, 10% stay: so the model can't assume only [MASK] positions matter (it must model every token), and fine-tuning inputs, which never contain [MASK], aren't out of distribution.

</details>

12. why is a KV cache exact for GPT but impossible for BERT?

<details><summary>answer</summary>

in a causal model, token $t$'s representation in every layer depends only on tokens $\le t$, so nothing about earlier tokens changes when a new one arrives. in a bidirectional model every token attends to every other, so a new token changes all earlier representations from layer 1 up.

</details>

13. FlashAttention does the same multiply-adds as ordinary attention. where does its speed come from?

<details><summary>answer</summary>

memory traffic. online softmax lets it process keys in blocks with a running max and normalizer, so the $T \times T$ score matrix never has to be written to and read from the GPU's main memory. attention at long lengths is memory-bound, so moving $O(Td)$ instead of $O(T^2)$ bytes is the speedup.

</details>

14. what does grouped-query attention share, and what does it save?

<details><summary>answer</summary>

keys and values: $H_{kv} < H$ kv heads, each read by a group of $H/H_{kv}$ query heads. it shrinks the KV cache (and the k/v projections) by $H/H_{kv}$, which at serving time is often bigger than the model itself (topic 06: 16 conversations of 4,096 tokens on a 7B MHA model need 32 GiB of cache against 12.5 GiB of weights).

</details>

15. a ViT and a ResNet of equal size are trained on 1,000 images. which wins, and what are the convolution's two built-in assumptions that explain it?

<details><summary>answer</summary>

the ResNet, by about 9 points in topic 07's lab (and by 8–10 points with 5,000 or 20,000 images at the same short budget). locality (filters combine neighbouring pixels) and translation equivariance (one filter applied everywhere). the ViT has to learn both from data, which costs examples; with hundreds of millions of images it learns them and more, and overtakes.

</details>

## 2. by hand

1. **attention.** queries, keys and values for three tokens, $d = 4$: $q_1 = k_1 = (1, 1, 1, 1)$, $q_2 = k_2 = (1, 1, -1, -1)$, $q_3 = k_3 = 0$; $v_1 = (1, 0)$, $v_2 = (0, 1)$, $v_3 = (1, 1)$. compute the causal attention output for token 2, and the unmasked output for token 1.

<details><summary>solution</summary>

causal, token 2: scores against keys 1 and 2 are $(0, 4)/2 = (0, 2)$, weights $(1, e^2)/(1 + e^2) = (0.119, 0.881)$, output $(0.119, 0.881)$. unmasked, token 1: scores $(4, 0, 0)/2 = (2, 0, 0)$, weights $(e^2, 1, 1)/(e^2 + 2) = (0.787, 0.107, 0.107)$, output $0.787(1, 0) + 0.107(0, 1) + 0.107(1, 1) = (0.893, 0.213)$.

</details>

2. **online softmax.** one query; four keys with scores $(1, 3, 2, 5)$ and scalar values $(1, 2, 3, 4)$, processed in two blocks of two. give $m$, $\ell$, $a$ after each block and the output.

<details><summary>solution</summary>

block 1: $m = 3$, $\ell = e^{-2} + 1 = 1.1353$, $a = e^{-2} + 2 = 2.1353$. block 2: $m = 5$; rescale by $e^{-2}$: $\ell = 1.1353 e^{-2} + e^{-3} + 1 = 1.2035$, $a = 2.1353 e^{-2} + 3e^{-3} + 4 = 4.4384$. output $a/\ell = 3.688$, equal to the direct softmax average.

</details>

3. **sizes.** a GPT with vocabulary 65, `block_size` 256, $d = 128$, 4 layers, learned positions and tied weights: count its parameters. then the KV cache, in bytes, of a model with 32 layers, 8 kv heads of 128 dimensions, at 4,096 tokens in 16-bit, for a batch of 4.

<details><summary>solution</summary>

embeddings $65 \cdot 128 + 256 \cdot 128 = 41{,}088$; each block $4(128^2 + 128) + 8 \cdot 128^2 + 5 \cdot 128 + 4 \cdot 128 = 198{,}272$, four of them $793{,}088$; final norm 256. total 834,432. cache: $2 \cdot 32 \cdot 8 \cdot 128 \cdot 4096 \cdot 4 \cdot 2 = 2{,}147{,}483{,}648$ bytes = 2 GiB.

</details>

## 3. blind rebuild (60 minutes)

in a fresh copy of `01-self-attention/exercise.py` (or an empty marimo notebook with the same names), with no notes and nothing from this chapter open, implement:

- `causal_mask(T, *, device=None)`
- `scaled_dot_product_attention(q, k, v, *, mask=None, is_causal=False, dropout_p=0.0)`: boolean masks (True = attend), end-aligned `is_causal`, float masks as additive bias, empty rows giving zeros, inverted dropout on the weights, returning `(out, weights)`
- `split_heads`, `merge_heads`
- `MultiHeadAttention(d_model, n_heads, *, dropout=0.0, bias=True)` with `q_proj`, `k_proj`, `v_proj`, `out_proj`, cross-attention through `context`, the `cache.update(k, v)` hook and `save_weights`

then run topic 01's checks against your notebook:

```bash
pytest 18-transformers/01-self-attention/test_exercise.py
```

if you finish early, add topic 06's `LayerKVCache`, `KVCache` and `generate_cached` and run `pytest 18-transformers/06-efficient-attention-and-inference/test_exercise.py -k "stage1 or stage2"` (they use the course's reference GPT).

## 4. readiness

move on when you score at least 80% on parts 1 and 2 together (15 concept questions plus 3 problems, with partial credit for parts of a problem) and the rebuild passes every topic-01 test. the same checkpoint works as a placement test: if you pass it cold, you can skip this chapter and use the reference `mlfs.transformer` (`MLFS_REFERENCE=transformer`).
