# self-attention

## why this matters

at the end of chapter 17 you trained an encoder–decoder that translated dates. without attention it learned the year at once and the day only after thousands of steps, because everything the decoder knew about the input had to squeeze through one hidden vector, and the day was written furthest from it. Bahdanau attention fixed that: at every output step the decoder could look back at every input position and take a weighted average of what it found.

look at what was left after that fix. the encoder was still a GRU, so token 50 could only learn about token 1 through 49 applications of the same recurrence, and the gradient that would teach that connection shrank on the way (topic 04's lab measured it). the recurrence was also strictly sequential: a GPU that can multiply a 4096 × 4096 matrix in a millisecond sat idle while the RNN processed one time step after another.

Vaswani et al. (2017) asked the obvious question: if attention is what made the model work, why keep the recurrence at all? let *every* position look at every other position directly, in one step, in parallel. the distance between any two tokens becomes 1, and a whole sequence becomes a few large matrix multiplications. that operation, **self-attention**, is the core of every transformer: GPT, BERT, the vision transformer, the diffusion transformer and the LLMs of chapter 27. this topic builds it, with exactly the conventions (masks, shapes, a cache hook) that the rest of the course depends on.

## warm-up: predict first

1. two random vectors in $\mathbb{R}^d$ with independent standard-normal entries. how does the typical size of their dot product grow with $d$?
2. a softmax over 64 scores whose standard deviation is 30. roughly what does the output look like?
3. you shuffle the words of a sentence and feed it to self-attention with no positional information. what happens to the output?
4. one attention head computes one weighted average of the values per query. what can't a single weighted average do that two can?
5. attention over $T$ tokens compares every pair. for a document of 8,000 tokens and a model width of 1,000, which costs more: the attention scores, or the linear projections?

## prerequisites check

- [ ] you can compute a softmax and its Jacobian (→ ch 08 · softmax regression, ch 03)
- [ ] you know that the variance of a sum of independent terms is the sum of their variances (→ ch 04)
- [ ] you've built Bahdanau's additive attention with a masked softmax (→ ch 17 · topic 06)
- [ ] you can write an `nn.Module` and use batched matrix products with broadcasting (→ ch 15 · topic 07, ch 00)
- [ ] you know inverted dropout (→ ch 15 · topic 03)

## core ideas

### idea 1: attention is a soft dictionary lookup

a Python dictionary maps a key to a value: you present a query, it finds the key that equals the query, and returns that key's value. attention is the differentiable version. every position holds a **key** $k_j$ (what it offers to be found by) and a **value** $v_j$ (what it hands over when found). a **query** $q_i$ is compared with every key, the comparisons become weights by a softmax, and the result is the weighted average of the values:

$$\text{out}_i = \sum_j a_{ij} v_j, \qquad a_{ij} = \frac{\exp(s_{ij})}{\sum_{j'} \exp(s_{ij'})}, \qquad s_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}}.$$

Bahdanau scored with a small MLP; the transformer uses the plain dot product, because it's a single matrix multiplication for all pairs at once. stacking queries as rows of $Q \in \mathbb{R}^{T_q \times d}$, keys as rows of $K \in \mathbb{R}^{T_k \times d}$ and values as rows of $V \in \mathbb{R}^{T_k \times d_v}$:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V.$$

**why divide by $\sqrt d$?** if the entries of $q$ and $k$ are independent with mean 0 and variance 1, then $q \cdot k$ is a sum of $d$ independent terms of variance 1, so its standard deviation is $\sqrt d$ (on paper 1). at $d = 64$ the raw scores are spread over about $\pm 16$, and $e^{16}$ against $e^{0}$ means one key takes almost all the weight. a saturated softmax has a Jacobian close to zero, so the gradient that would teach the queries and keys which tokens to attend to vanishes. dividing by $\sqrt d$ keeps the scores at unit scale for any width. the lab measures how much this matters.

**masks.** often a query must not see some keys: a language model predicting token $t + 1$ must not look at tokens after $t$, and padding tokens added to make a batch rectangular must be ignored. a mask is a boolean array broadcastable to $(T_q, T_k)$ where **True means "may attend"**. blocked scores are set to $-\infty$ before the softmax, so their weights are exactly 0. the **causal mask** is lower-triangular: query $i$ sees keys $j \le i$.

two conventions need care, and the rest of the course depends on both. first, when there are *fewer queries than keys*, the causal mask is aligned to the end: the queries are the *last* $T_q$ tokens of the sequence, so query $i$ sees keys $j \le T_k - T_q + i$. that is what lets topic 06's KV cache feed one new token at a time while the keys of all earlier tokens are kept. (PyTorch's `is_causal` aligns to the top left instead; the two agree when $T_q = T_k$.) second, a query whose every key is blocked would compute $0/0$: such rows return zero weights and a zero output, as `F.scaled_dot_product_attention` does.

two more details. a *float* mask is an additive bias on the scores, which is how topic 02's ALiBi will tell attention about distance (and `-inf` entries block). and attention dropout zeroes random weights after the softmax (inverted, so survivors are scaled by $1/(1-p)$); the weights you return are the ones before dropout.

> **spec.** `causal_mask(T, *, device=None)` returns a `(T, T)` bool tensor, True on and below the diagonal. `scaled_dot_product_attention(q, k, v, *, mask=None, is_causal=False, dropout_p=0.0)`: `q` is `(..., Tq, d)`, `k` is `(..., Tk, d)`, `v` is `(..., Tk, d_v)`, with any number of leading batch dimensions. scores are $qk^\top/\sqrt{d}$ with $d$ = `q.shape[-1]`. a bool `mask` (broadcastable to `(..., Tq, Tk)`, True = attend) and `is_causal` (aligned to the end) are combined with AND; a float `mask` is added to the scores. rows with no allowed key give zero weights and zero output. dropout with probability `dropout_p` is applied to the weights used for the output, whenever `dropout_p > 0`. returns `(out (..., Tq, d_v), weights (..., Tq, Tk))`, weights before dropout. it must match `F.scaled_dot_product_attention` wherever their conventions agree.

> **🔮 predict:** in the worked example below, query 3 is the zero vector. what are its weights, whatever the keys are?

> **▶ build: stage 1.** implement `causal_mask` and `scaled_dot_product_attention`.

### idea 2: many heads instead of one

one attention computes, for each query, *one* weighted average. that is a real limitation: if a verb needs to know both its subject and its object, a single softmax has to split its weight between them and returns a blur of both. the fix is to run several attentions side by side, each with its own queries, keys and values, so one **head** can find the subject while another finds the object, and concatenate their outputs.

the trick that makes this free is to *split* the model width rather than add to it. with $d_{\text{model}} = 512$ and $H = 8$ heads, each head works in $d_h = 64$ dimensions: project the input to 512-dimensional queries, keys and values as before, then view channels $0..63$ as head 0, $64..127$ as head 1, and so on. the total arithmetic is the same as one 512-wide attention; you've just computed 8 softmaxes instead of 1. each head scales its scores by $\sqrt{d_h}$, its own width.

in code, the split is a reshape and a transpose: $(B, T, d) \to (B, T, H, d_h) \to (B, H, T, d_h)$. the head axis moves next to the batch axis, and your stage-1 function, which treats every leading dimension as a batch dimension, attends in all heads at once. merging is the inverse. the order of operations matters: reshaping $(B, T, d)$ directly into $(B, H, T, d_h)$ has the right shape and silently mixes tokens across heads.

> **spec.** `split_heads(x, n_heads)`: `(B, T, d)` → `(B, n_heads, T, d // n_heads)`, where head $h$ of token $t$ holds channels $h d_h, \dots, (h + 1)d_h - 1$ of token $t$. `merge_heads(x)`: `(B, H, T, d_h)` → `(B, T, H d_h)`, its exact inverse.

> **▶ build: stage 2.** implement `split_heads` and `merge_heads`.

### idea 3: the multi-head attention module

where do $Q$, $K$ and $V$ come from? in **self-attention**, all three are linear projections of the same sequence $X \in \mathbb{R}^{T \times d}$: $Q = XW_Q$, $K = XW_K$, $V = XW_V$. the token at position $i$ asks a question ($q_i$), advertises what it is about ($k_i$) and offers content ($v_i$), and these are three different things, which is why there are three matrices. after the heads are merged, an output projection $W_O$ mixes what the heads found:

$$\operatorname{MHA}(X) = \operatorname{merge}\big(\operatorname{Attention}(\operatorname{split}(XW_Q), \operatorname{split}(XW_K), \operatorname{split}(XW_V))\big) W_O.$$

in **cross-attention**, queries come from one sequence and keys and values from another (the `context`): a translation decoder's tokens query the encoded source sentence, exactly like chapter 17's decoder, and an image generator's patches will query a text prompt in chapter 22. nothing else changes; $T_q$ and $T_k$ simply differ.

the module also carries two hooks the rest of the course uses. a **cache**: any object with a method `update(k, v)` that stores the new keys and values (in the heads layout $(B, H, T, d_h)$) and returns everything stored so far. the module calls it after projecting and before attending, so that when you generate text one token at a time the keys and values of earlier tokens aren't recomputed. you'll build the real cache in topic 06; here it's a protocol, and the test uses a ten-line stand-in. and **save_weights**: when set, the module keeps the last attention weights (detached), so that labs can plot them and chapter 28 can study them.

with $T$ tokens, the module does four $d \times d$ projections ($4Td^2$ multiply-adds) and two $T \times T$ products in attention ($2T^2 d$). attention is quadratic in sequence length, both in time and in the memory for the $T \times T$ weights: that is the cost topic 06 attacks.

> **spec.** `MultiHeadAttention(d_model, n_heads, *, dropout=0.0, bias=True)` has submodules `q_proj`, `k_proj`, `v_proj`, `out_proj` (each `nn.Linear(d_model, d_model, bias=bias)`), raises `ValueError` if `n_heads` doesn't divide `d_model`, and an attribute `save_weights = False`. `forward(x, context=None, *, mask=None, is_causal=False, cache=None)`: `x` is `(B, T, d)`; queries come from `x`, keys and values from `context` `(B, S, d)` if given, else from `x`. if `cache` is given, `k, v = cache.update(k, v)` (heads layout) before attending. `mask` is broadcastable to `(B, H, Tq, Tk)` (so a key-padding mask is passed as `(B, 1, 1, Tk)`). attention dropout uses `dropout` in training mode and 0 in eval mode. if `save_weights`, store the weights `(B, H, Tq, Tk)`, detached, as `last_weights`. returns `(B, T, d)`.

> **🔮 predict:** you run causal self-attention on a prompt of 3 tokens with an empty cache, then on token 4 alone with the same cache. what is $T_q$, what is $T_k$, and which keys may token 4 see?

> **▶ build: stage 3.** implement `MultiHeadAttention`.

## worked example

three tokens, $d = 4$ (so $\sqrt d = 2$), keys equal to queries:

$$Q = K = \begin{pmatrix} 1 & 1 & 1 & 1 \\ 1 & 1 & -1 & -1 \\ 0 & 0 & 0 & 0 \end{pmatrix}, \qquad V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}.$$

the dot products of row 1 with the three keys are $(4, 0, 0)$; divided by 2, the scores are $(2, 0, 0)$. the weights are $(e^2, 1, 1)/(e^2 + 2) = (0.7870, 0.1065, 0.1065)$, and the output is $0.7870\,(1, 0) + 0.1065\,(0, 1) + 0.1065\,(1, 1) = (0.8935, 0.2130)$. row 2 is the mirror image: scores $(0, 2, 0)$, output $(0.2130, 0.8935)$. row 3 is the zero vector, so all its scores are 0 and its weights uniform, $(1/3, 1/3, 1/3)$, whatever the keys: output $(2/3, 2/3)$.

with `is_causal=True`: query 1 sees only key 1 (weight 1, output $(1, 0)$); query 2 sees keys 1 and 2 with scores $(0, 2)$, weights $(1, e^2)/(1 + e^2) = (0.1192, 0.8808)$; query 3 sees all three, unchanged. without the $\sqrt d$, query 1's scores would be $(4, 0, 0)$ and its top weight 0.965 instead of 0.787. your code must reproduce these numbers.

## on paper

1. **the variance of a dot product.** let $q, k \in \mathbb{R}^d$ have independent entries with mean 0 and variance 1. show that $\mathbb{E}[q \cdot k] = 0$ and $\operatorname{Var}[q \cdot k] = d$. what scale factor makes the variance 1?

<details><summary>solution</summary>

$q \cdot k = \sum_{i=1}^d q_i k_i$. each term has mean $\mathbb{E}[q_i]\mathbb{E}[k_i] = 0$ by independence, and variance $\mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1$. the terms are independent of each other, so the variances add: $\operatorname{Var}[q \cdot k] = d$. dividing by $\sqrt d$ divides the variance by $d$, giving 1. in a trained model the entries aren't independent or unit-variance, but the scaling keeps the scores at the same order of magnitude whatever width you choose, so the same learning rate and initialization work across model sizes.

</details>

2. **the softmax's gradient.** for $a = \operatorname{softmax}(s)$, show that $\partial a_i / \partial s_j = a_i(\delta_{ij} - a_j)$, i.e. $J = \operatorname{diag}(a) - aa^\top$. what is $J$ when $a$ is one-hot? when $a$ is uniform over $n$ entries?

<details><summary>solution</summary>

$a_i = e^{s_i}/Z$ with $Z = \sum_k e^{s_k}$. for $j = i$: $\partial a_i/\partial s_i = (e^{s_i} Z - e^{s_i} e^{s_i})/Z^2 = a_i - a_i^2$. for $j \ne i$: $\partial a_i / \partial s_j = -e^{s_i}e^{s_j}/Z^2 = -a_i a_j$. together, $a_i(\delta_{ij} - a_j)$. one-hot $a = e_m$: $\operatorname{diag}(e_m) - e_m e_m^\top = 0$, so no gradient reaches any score; a nearly one-hot softmax has a nearly zero Jacobian. uniform $a = \mathbf{1}/n$: $J = \frac1n I - \frac{1}{n^2}\mathbf{1}\mathbf{1}^\top$, entries of order $1/n$, and every score gets gradient. unscaled scores at large $d$ push attention toward the one-hot case.

</details>

3. **permutation equivariance.** let $P$ be a $T \times T$ permutation matrix. show that unmasked self-attention satisfies $\operatorname{MHA}(PX) = P\operatorname{MHA}(X)$. which step breaks with a causal mask?

<details><summary>solution</summary>

projections act row by row: $(PX)W = P(XW)$, so $Q' = PQ$, $K' = PK$, $V' = PV$. the scores become $Q'K'^\top = PQK^\top P^\top$, which permutes rows and columns the same way. the softmax is taken row by row and doesn't care about the order of entries within a row, so $\operatorname{softmax}(PSP^\top) = P\operatorname{softmax}(S)P^\top$. then $(PAP^\top)(PV) = PAV$ because $P^\top P = I$. splitting heads, merging and the output projection all act per token, so they commute with $P$ too. with a causal mask $M$, the masked scores are $PSP^\top + M$, and $M \ne PMP^\top$ for a non-identity $P$: the mask is tied to positions, not to tokens. so a transformer with no positional information sees a *set*, not a sequence, unless it's causal.

</details>

4. **where the time goes.** count multiply-adds for one `MultiHeadAttention` forward on $T$ tokens of width $d$ (ignore biases and the softmax). for which $T$ do the attention products cost as much as the projections? evaluate for $d = 1024$.

<details><summary>solution</summary>

projections: four products of $(T \times d)$ by $(d \times d)$, $4Td^2$. scores $QK^\top$: across all heads, $T \cdot T \cdot d_h \cdot H = T^2 d$. weights times values: another $T^2 d$. equal when $2T^2 d = 4Td^2$, i.e. $T = 2d$. at $d = 1024$ that's 2,048 tokens: shorter sequences are dominated by the projections (and the MLP of topic 03, which is bigger still), longer ones by attention. the memory for the $T \times T$ weights, $H T^2$ numbers per layer, becomes a problem sooner, which is why topic 06 computes attention without ever storing them.

</details>

## experiment

the lab measures what the $\sqrt d$ does: the size of the scores, how peaked the attention is, and how much gradient reaches the scores, from $d = 4$ to $d = 1024$, with and without the scaling. then it trains one attention layer on a lookup task that needs sharp attention, scaled and unscaled, at three widths, to see whether an adaptive optimizer rescues the unscaled version. finally it shuffles the input of your attention to find out what attention knows about order, with and without the causal mask.

## check your understanding

1. why are keys and values separate projections? what would you lose if $V = K$?

<details><summary>answer</summary>

the key decides *whether* a token is found and the value decides *what* it contributes. with $V = K$, a token could only hand over the same features it's matched on. a pronoun's key might say "I'm a noun phrase in the subject position", while its value carries the entity's identity: different information. separate projections also let $d_v$ differ from $d$.

</details>

2. the causal mask makes the softmax ignore future tokens. why fill their scores with $-\infty$ rather than multiply the weights by 0 after the softmax?

<details><summary>answer</summary>

zeroing after the softmax leaves the remaining weights summing to less than 1, and worse, the normalizer $Z$ still contains the future scores, so the output (and its gradient) still depends on future tokens: information leaks. filling with $-\infty$ removes them from $Z$ too: the softmax renormalizes over the allowed keys only.

</details>

3. your attention returns zeros for a query that can attend to nothing. when does that happen in practice, and why does it matter that it's 0 and not NaN?

<details><summary>answer</summary>

with a key-padding mask, padding *queries* in a batch of short sentences can end up with no allowed keys if they're also masked as queries (or with a combined causal + padding mask on left-padded sequences). their outputs are discarded later, but a NaN would spread: through the output projection into every other number of that example in the next layer's normalization, and into the gradients of every parameter. a zero is harmless.

</details>

4. a model has $d_{\text{model}} = 768$ and 12 heads. how many numbers does one layer's attention-weight tensor hold for a batch of 8 sequences of 1,024 tokens?

<details><summary>answer</summary>

$B \cdot H \cdot T^2 = 8 \cdot 12 \cdot 1024^2 \approx 10^8$ numbers, 400 MB in float32, per layer, and training keeps them for the backward pass. this is the memory that FlashAttention (topic 06) avoids storing.

</details>

5. multi-head attention with $H$ heads of width $d/H$ costs the same as one head of width $d$. why isn't one head of width $d$ just as good?

<details><summary>answer</summary>

one head produces one distribution over the keys per query, so it can pick out one "thing" (or a blur of several). $H$ heads give $H$ independent distributions: different heads can track different relations (the previous token, the matching bracket, the subject of the verb). the price is that each head's scores live in a lower-dimensional space. empirically several heads beat one at equal cost, though many heads can be pruned after training (Michel et al., 2019).

</details>

6. you pass `mask=torch.tensor([[1., 1., 0.]])` meaning "attend to the first two keys". what does your function do?

<details><summary>answer</summary>

it's a float tensor, so it's an *additive bias*: the scores of the first two keys get $+1$ and the third gets $+0$. nothing is blocked. boolean masks block; float masks shift. convert with `mask.bool()`, or build the mask with comparisons, which return bools.

</details>

## common bugs

- **softmax over the wrong axis.** `softmax(scores, dim=-2)` normalizes over queries, so each *key's* column sums to 1. outputs are wrong but the shapes are fine. normalize over the last dimension, the keys.
- **splitting heads with a bare `view`.** `x.view(B, H, T, dh)` has the right shape and mixes tokens across heads. view as `(B, T, H, dh)`, then transpose.
- **mask conventions.** this course (and `F.scaled_dot_product_attention`) uses True = attend; `nn.MultiheadAttention`'s `attn_mask` and `key_padding_mask` use True = *blocked*. mixing them up masks exactly the tokens you wanted to keep, and training still runs.
- **NaN from an empty row.** a query whose keys are all masked gives softmax $0/0$. symptom: the loss is NaN only for some batches (the ones with heavy padding).
- **forgetting the scale, or scaling by $d$.** both train, worse, and the oracle test catches them. each head scales by $\sqrt{d_h}$, not $\sqrt{d_{\text{model}}}$.
- **a 0/1 float mask.** added, not applied (question 6).

## going deeper

- Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (2017), "Attention is all you need", *NeurIPS*. section 3.2 defines scaled dot-product and multi-head attention, and footnote 4 is on paper 1. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau, Cho & Bengio (2015), "Neural machine translation by jointly learning to align and translate", *ICLR*. the additive attention of chapter 17. [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Luong, Pham & Manning (2015), "Effective approaches to attention-based neural machine translation", *EMNLP*. dot-product scoring for seq2seq. [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Phuong & Hutter (2022), "Formal algorithms for transformers". precise pseudocode for every transformer component, a good companion to this chapter. [arXiv:2207.09238](https://arxiv.org/abs/2207.09238)
- Michel, Levy & Neubig (2019), "Are sixteen heads really better than one?", *NeurIPS*. [arXiv:1905.10650](https://arxiv.org/abs/1905.10650)
- Haviv, Ram, Press, Izsak & Levy (2022), "Transformer language models without positional encodings still learn positional information", *Findings of EMNLP*. the lab's third experiment. [arXiv:2203.16634](https://arxiv.org/abs/2203.16634)
- Prince (2023), *Understanding Deep Learning*, chapter 12 (transformers), sections 12.1–12.4. [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/)

## connections

- ← builds on: Bahdanau attention and seq2seq (ch 17 · topic 06), softmax and its gradient (ch 03, ch 08), variance of sums (ch 04), PyTorch modules and inverted dropout (ch 15).
- → used in: positions (topic 02), the transformer block (03), GPT (04), BERT (05), the KV cache and FlashAttention-style tiling (06), the vision transformer (07); cross-attention in conditional generation (ch 21–22) and multimodal models (ch 25); grouped-query attention in LLMs (ch 27); attention patterns in interpretability (ch 28).
