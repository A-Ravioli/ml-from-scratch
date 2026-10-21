# modern LLM architecture

## why this matters

the GPT you built in chapter 18 is GPT-2 (2019). open the code of Llama, Mistral, Qwen or Gemma and you'll find the same skeleton (a stack of residual blocks, each attention then MLP, a causal mask, next-token loss) with a handful of components swapped out. none of the swaps is dramatic on its own; together they're the difference between a 2019 model and a 2024 one at the same size. **RMSNorm** replaces LayerNorm (cheaper, no mean, no bias). **SwiGLU** replaces the GELU MLP (a gated unit that trains better per parameter). **rotary embeddings** replace learned positions (relative positions inside attention, so nothing to learn and better behaviour at new lengths). **grouped-query attention** shares keys and values between heads, which cuts the KV cache, the thing that limits how many users a server can batch (chapter 26 · 06), by 4–8× at almost no cost in quality. and **mixture-of-experts** layers grow the parameter count without growing the compute per token: Mixtral 8x7B (Jiang et al., 2024) holds 47B parameters but uses 13B per token.

in this topic you build each piece, assemble them into `Llama`, and make it work with the KV cache and `generate_cached` you wrote in chapter 18. this is the model the rest of the chapter fine-tunes, aligns, decodes, evaluates and (in chapter 28) dissects.

## warm-up: predict first

1. a GELU MLP has two weight matrices of size $d \times 4d$. a SwiGLU MLP has three. what hidden width $d_\text{ff}$ keeps the parameter count the same?
2. a 7B model has 32 layers and 32 heads of width 128. how many bytes of KV cache does one token take in 16-bit? one 4,096-token conversation?
3. if 8 query heads share one key/value head, how much quality do you expect to lose?
4. an MoE layer routes each token to its 2 best experts out of 8. left alone, will the router spread tokens evenly?

## prerequisites check

- [ ] you can write multi-head causal attention and explain the $1/\sqrt{d_h}$ (→ ch 18 · 01)
- [ ] you built `RotaryEmbedding`, `repeat_kv`, `kv_cache_bytes`, `KVCache` and `generate_cached` (→ ch 18 · 02, 06)
- [ ] you built `RMSNorm` (→ ch 15 · 07) and know why pre-norm trains more stably than post-norm (→ ch 18 · 03)
- [ ] you know the roofline argument for why decoding is memory-bound (→ ch 26 · 01, 06)

## core ideas

### idea 1: SwiGLU

the transformer MLP expands each position to a wider hidden layer, applies a nonlinearity, and projects back. Shazeer (2020) tried **gated linear units** in its place: two parallel projections, one passed through a nonlinearity and used as a gate on the other,

$$\text{SwiGLU}(x) = W_\text{down}\big(\text{SiLU}(W_\text{gate} x) \odot W_\text{up} x\big), \qquad \text{SiLU}(z) = z\,\sigma(z).$$

the gate lets the layer multiply features together instead of only thresholding them, and in Shazeer's experiments GLU variants beat ReLU and GELU MLPs at equal parameters and compute. PaLM and every Llama since use SwiGLU. there's no satisfying theory of *why* it works better; Shazeer's paper ends by attributing it to "divine benevolence". this is an empirical field, and the lab measures the combined effect yourself.

three matrices instead of two: to keep the parameter count of a $4d$-wide GELU MLP ($8d^2$), choose $d_\text{ff} = \tfrac{2}{3} \cdot 4d = 8d/3$, since $3 \cdot d \cdot 8d/3 = 8d^2$. it's rounded up to a hardware-friendly multiple (Llama rounds to 256: 11,008 for $d = 4096$; here, to 8). no biases anywhere: Llama drops them from every linear layer, which saves a little memory and changes nothing measurable.

> **spec.** `SwiGLU(d_model, d_ff=None)`: bias-free `nn.Linear` layers `w_gate`, `w_up` ($d \to d_\text{ff}$) and `w_down` ($d_\text{ff} \to d$); default $d_\text{ff}$ = `int(8 * d_model / 3)` rounded up to a multiple of 8. `forward(x)` = `w_down(silu(w_gate(x)) * w_up(x))` on `(..., d_model)`.

> **▶ build: stage 1.** implement `SwiGLU`.

### idea 2: rotary, grouped-query attention

the attention of a Llama differs from chapter 18's in two ways. **positions** enter through rotary embeddings (Su et al., 2021; your `RotaryEmbedding`): queries and keys are rotated by an angle proportional to their position before the dot product, so $q_i \cdot k_j$ depends on $i - j$ only. there's no position embedding at the input; values are not rotated.

and **keys and values are shared**. in multi-head attention (MHA) every one of $H$ heads has its own $W_q$, $W_k$, $W_v$. during generation the keys and values of every past token are cached (chapter 18 · 06), and the cache, not the weights, dominates memory at long contexts and large batches: a 7B model caches $2 \cdot 32 \text{ layers} \cdot 32 \text{ heads} \cdot 128 \cdot 2 \text{ bytes} = 0.5$ MiB per token, 2 GiB per 4,096-token sequence. Shazeer (2019) proposed **multi-query attention** (MQA): all query heads share a single key/value head, cutting the cache by $H$. it costs some quality. **grouped-query attention** (GQA; Ainslie et al., 2023) interpolates: $H_{kv}$ key/value heads, each shared by a group of $H / H_{kv}$ consecutive query heads. Llama 2 70B uses 64 query heads and 8 key/value heads: an 8× smaller cache with quality close to MHA.

implementation: project $q$ to $H$ heads and $k, v$ to $H_{kv}$ heads, rotate $q$ and $k$ at their absolute positions, append $k, v$ to the cache **before** expanding them (the cache stores $H_{kv}$ heads; that's the point), then `repeat_kv` so key/value head $j$ is used by query heads $j \cdot n_\text{rep}, \dots, (j+1) n_\text{rep} - 1$, and attend causally. one subtlety: with a cache, the $T$ new queries sit at positions $\text{offset}, \dots, \text{offset} + T - 1$ and the keys at $0, \dots, \text{offset} + T - 1$, so query $i$ may see keys up to $\text{offset} + i$: the causal mask is aligned to the **end** of the key sequence. PyTorch's `is_causal=True` aligns it to the start, which is only right when $T$ equals the number of keys.

> **spec.** `GroupedQueryAttention(d_model, n_heads, n_kv_heads, *, rope_base=10000.0, max_seq_len=2048, dropout=0.0)`: `ValueError` unless `n_heads` divides `d_model` and `n_kv_heads` divides `n_heads`; attributes `n_heads`, `n_kv_heads`, `head_dim`; bias-free `q_proj` ($d \to H d_h$), `k_proj`, `v_proj` ($d \to H_{kv} d_h$), `o_proj` ($H d_h \to d$); `rope = RotaryEmbedding(head_dim, base=rope_base, max_seq_len=max_seq_len)`. `forward(x, *, cache=None)`, $x$ of shape $(B, T, d)$: rotate $q, k$ at offset `cache.seq_len` (0 without a cache, read before updating), `k, v = cache.update(k, v)` with the $H_{kv}$-head tensors, `repeat_kv`, causal attention aligned to the end, attention dropout with probability `dropout` in training mode only, `o_proj`. always causal. attribute `save_weights` (default `False`): when set to `True`, the attention is computed with an explicit softmax and `last_weights` (B, H, Tq, Tk) is stored, detached, after the end-aligned causal mask and before dropout, the same semantics as chapter 18's `MultiHeadAttention`; the output is unchanged. (`F.scaled_dot_product_attention` is unlocked since chapter 26.)

> **🔮 predict:** with 4 query heads, going from 4 key/value heads to 1, how much validation loss do you lose on Shakespeare?

> **▶ build: stage 2.** implement `GroupedQueryAttention`.

### idea 3: mixture of experts

a dense MLP applies all its parameters to every token. a **mixture-of-experts** layer (Shazeer et al., 2017) holds $E$ expert MLPs and a **router**, a linear map from the token to $E$ logits. each token is sent to its top $k$ experts by router probability, and their outputs are combined with gate weights:

$$p = \text{softmax}(W_r x), \qquad y = \sum_{e \in \text{top-}k(p)} g_e\, \text{expert}_e(x), \qquad g_e = \frac{p_e}{\sum_{e' \in \text{top-}k} p_{e'}}.$$

(Mixtral renormalizes the chosen probabilities this way; Switch Transformers use $k = 1$ and the raw probability.) the compute per token is $k$ experts' worth, the parameters are $E$ experts' worth: capacity grows with $E$ while FLOPs stay fixed. the router learns through the gates: $g_e$ multiplies the expert's output, so the loss's gradient reaches $W_r$.

an MoE must actually be sparse: each expert runs only on the tokens routed to it (gather its rows, run the expert, scatter the weighted outputs back with `index_add_`). running every expert on every token and masking afterwards gives the same numbers at $E/k$ times the cost.

left alone, routers **collapse**: an expert that happens to get more tokens early gets more gradient, gets better, gets chosen more. some experts starve, the effective capacity shrinks, and on real hardware the busy experts' devices become the bottleneck. Switch Transformers (Fedus et al., 2022) add an auxiliary **load-balancing loss**:

$$\mathcal{L}_\text{aux} = E \sum_{e=1}^{E} f_e P_e,$$

where $f_e$ is the fraction of routing slots assigned to expert $e$ (here: of the $N k$ (token, slot) pairs) and $P_e$ the mean router probability of $e$ over the batch. it equals 1 when both are uniform and grows when the router concentrates. $f_e$ comes from an argmax and carries no gradient; $P_e$ does, so the loss pushes probability away from overloaded experts. it's added to the LM loss with a small coefficient (0.01 in Switch).

> **spec.** `MoE(d_model, d_ff, n_experts, *, top_k=2)`: `ValueError` unless $1 \le$ `top_k` $\le$ `n_experts`; attributes `n_experts`, `top_k`; `router` = bias-free `nn.Linear(d_model, n_experts)`; `experts` = `nn.ModuleList` of `SwiGLU(d_model, d_ff)`. `forward(x)`, $x$ of shape $(B, T, d)$, returns `(out, aux_loss)`: softmax router probabilities (in float32), top-`k` experts per token, gates renormalized over the chosen $k$, each expert evaluated **only on its tokens**; `aux_loss` = $E \sum_e f_e P_e$ as above (a differentiable scalar). after every forward, `self.last_counts` is an int64 tensor `(n_experts,)` of (token, slot) assignments per expert.

> **🔮 predict:** train an MoE with the auxiliary coefficient set to 0. how many of 8 experts end up doing most of the work?

> **▶ build: stage 3.** implement `MoE`.

### idea 4: the Llama block

the block is chapter 18's pre-norm block with the new parts: $h = x + \text{attn}(\text{RMSNorm}_1(x))$, then $\text{out} = h + \text{mlp}(\text{RMSNorm}_2(h))$. the residual stream itself is never normalized; each sublayer reads a normalized copy and adds its output back, which keeps gradients flowing through the identity path of a deep stack (Xiong et al., 2020). RMSNorm (Zhang & Sennrich, 2019) divides by the root mean square without subtracting the mean and has no bias: one reduction instead of two, and in practice no loss of quality. when the block is an MoE block, it keeps the layer's auxiliary loss so the model can add it to the training loss.

> **spec.** `LlamaBlock(d_model, n_heads, n_kv_heads, *, d_ff=None, n_experts=0, top_k=2, max_seq_len=2048)`: submodules `norm1`, `norm2` (`mlfs.torchnn.RMSNorm`), `attn` (`GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len=max_seq_len)`), `mlp` (`SwiGLU(d_model, d_ff)`, or `MoE(d_model, d_ff, n_experts, top_k=top_k)` if `n_experts > 0`). `forward(x, *, cache=None)` is the pre-norm computation above, passing `cache` to `attn`; afterwards `self.aux_loss` holds the MoE's auxiliary loss (a tensor, not detached), or `0.0` for a dense block.

> **▶ build: stage 4.** implement `LlamaBlock`.

### idea 5: the model

`Llama` is a token embedding, a stack of blocks, a final RMSNorm and a head (tied to the embedding by default, as in chapter 18). there is no position embedding: RoPE inside every attention layer carries position. initialization follows GPT-2: $\mathcal{N}(0, 0.02)$ for every weight, and the two matrices that write into the residual stream in each block (`attn.o_proj` and the MLP's `w_down`, every expert's in an MoE) at $0.02/\sqrt{2L}$, so the residual stream's variance doesn't grow with depth. the loss is cross-entropy plus the auxiliary losses:

$$\mathcal{L} = \text{CE} + \lambda_\text{aux} \sum_{\ell} \mathcal{L}_{\text{aux}, \ell}.$$

the model follows the causal-LM protocol of chapter 18 (pre-shifted targets, `-100` ignored, `hidden_states`, `.blocks`), and the cache protocol: given a `KVCache`, block $i$ gets `cache[i]`, and the total length (cached plus new) may not exceed `block_size`. that's all `generate_cached` needs, so your chapter 18 decoding loop runs your Llama unchanged.

> **spec.** `Llama(vocab_size, *, block_size=256, n_layers=4, n_heads=4, n_kv_heads=2, d_model=128, d_ff=None, n_experts=0, top_k=2, aux_loss_coef=0.01, tie_weights=True)`: attributes `vocab_size`, `block_size`, `n_layers`, `aux_loss_coef`; submodules `tok_emb`, `blocks` (`nn.ModuleList` of `LlamaBlock`, each with `max_seq_len=block_size`), `norm` (RMSNorm), `lm_head` (bias-free; `lm_head.weight is tok_emb.weight` if `tie_weights`); initialization as above. `forward(idx, targets=None, *, cache=None)` returns `(logits (B, T, V), loss)`; `ValueError` if cached plus new tokens exceed `block_size`; loss as above (`None` without targets). note that the loss includes the auxiliary term in eval mode too: compare MoE models by cross-entropy from the logits. `hidden_states(idx)`: the final-norm output $(B, T, d)$. `generate(idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None)`: like `GPT.generate`, cropping the context to `block_size`, sampling with `sample_next_token`, without a cache, and restoring the training mode.

> **▶ build: stage 5.** implement `Llama`.

## worked example

**SwiGLU width.** $d = 128$: $\lfloor 8 \cdot 128/3 \rfloor = 341$, rounded up to a multiple of 8: 344. the three matrices hold $3 \cdot 128 \cdot 344 = 132{,}096$ parameters against $8 \cdot 128^2 = 131{,}072$ for a $4d$ GELU MLP without biases.

**parameter count.** a Llama with $V = 17$, $d = 32$, 2 layers, 4 heads of width 8, 2 key/value heads, default $d_\text{ff} = 88$, tied: embedding $17 \cdot 32 = 544$; per block $q$ and $o$ $2 \cdot 32^2 = 2{,}048$, $k$ and $v$ $2 \cdot 32 \cdot 16 = 1{,}024$, SwiGLU $3 \cdot 32 \cdot 88 = 8{,}448$, norms 64, total 11,584; two blocks and the final norm: $544 + 23{,}168 + 32 = 23{,}744$. with 4 key/value heads it's 2,048 more.

**the auxiliary loss.** two experts, top-1, four tokens with router probabilities for expert 0 of 0.9, 0.9, 0.9, 0.4. three tokens choose expert 0: $f = (0.75, 0.25)$; $P = (0.775, 0.225)$; $\mathcal{L}_\text{aux} = 2(0.75 \cdot 0.775 + 0.25 \cdot 0.225) = 1.275$, above the balanced value 1.

**the KV cache.** a 7B-class model (32 layers, $d_h = 128$, 16-bit) with 32, 8 and 1 key/value heads: 0.5 MiB, 128 KiB and 16 KiB per token; for 4,096 tokens, 2 GiB, 512 MiB and 64 MiB. your model and `kv_cache_bytes` must reproduce these numbers.

## on paper

1. **RoPE is relative.** let $R_\theta$ rotate each pair of channels by angles proportional to the position. show that $(R_{i} q)^\top (R_{j} k) = q^\top R_{j - i} k$, so attention scores depend on $i - j$ only, and explain why rotating the values would break the output.

<details><summary>solution</summary>

rotations are orthogonal and compose by adding angles: $R_i^\top = R_{-i}$ and $R_{-i} R_j = R_{j-i}$, so $(R_i q)^\top R_j k = q^\top R_i^\top R_j k = q^\top R_{j-i} k$. the values are what gets averaged and passed on; rotating them by their own positions would mix position-dependent rotations into the output, which the following layers would have to undo. positions belong in the scores (which token to look at), not in the content.

</details>

2. **the gradient of the auxiliary loss.** with $P_e = \frac{1}{N}\sum_n p_{n,e}$ and $f_e$ treated as a constant, show that $\partial \mathcal{L}_\text{aux}/\partial p_{n,e} = E f_e / N$. explain how this lowers the probability of overloaded experts.

<details><summary>solution</summary>

$\mathcal{L}_\text{aux} = E \sum_e f_e \frac{1}{N}\sum_n p_{n,e}$ is linear in each $p_{n,e}$ with coefficient $E f_e / N$. gradient descent lowers each $p_{n,e}$ in proportion to its expert's load $f_e$; since the probabilities of a token sum to 1 (through the softmax), mass moves from heavily loaded experts to lightly loaded ones. $f_e$ itself (an argmax count) has no gradient, which is why the loss needs the probabilities.

</details>

3. **when does GQA pay?** for decoding at batch $B$ and context $T$, compare the bytes read per step for the weights ($2N$ in 16-bit) and for the KV cache. for a 7B model with $B = 32$, $T = 4096$, with 32 versus 8 key/value heads, which dominates?

<details><summary>solution</summary>

weights: 14 GB per step. cache: $B \cdot T \cdot 0.5$ MiB = 64 GiB with 32 heads (MHA), 16 GiB with 8. with MHA the cache reads are 4–5× the weight reads, so decoding time is dominated by the cache, and GQA's 4× smaller cache makes each step about 2.6× faster (14 + 69 GB read per step against 14 + 17 GB) and lets 4× more sequences fit in memory. at batch 1 and short contexts the weights dominate and GQA barely matters.

</details>

## experiment

the lab trains your Llama and your chapter 18 GPT at the same size on tiny Shakespeare, each at three learning rates, and compares their validation losses; trains the Llama with 4, 2 and 1 key/value heads and sets the losses beside the cache sizes (yours and a 7B-class model's); and trains an MoE Llama with three load-balancing coefficients, tracking how evenly your router spreads tokens over training.

## check your understanding

1. why is SwiGLU's default width $8d/3$ and not $4d$?

<details><summary>answer</summary>

it has three matrices instead of two; $8d/3$ keeps the parameter count (and FLOPs) equal to a $4d$ two-matrix MLP, so comparisons are at equal cost.

</details>

2. where do you append to the KV cache in GQA: before or after `repeat_kv`? why?

<details><summary>answer</summary>

before: the cache stores the $H_{kv}$ distinct heads. appending after repeating stores $n_\text{rep}$ identical copies of each, giving back all the memory GQA saves.

</details>

3. with a cache holding 5 tokens and 2 new tokens, which keys may the second new token attend to? what goes wrong with `is_causal=True`?

<details><summary>answer</summary>

all 7: the 5 cached and both new ones (it's at position 6). `is_causal=True` with 2 queries and 7 keys builds a top-left-aligned mask, letting the queries see only keys 0 and 0–1: wrong. align the mask to the end (`tril(diagonal=Tk - T)`).

</details>

4. why does a router collapse without the auxiliary loss?

<details><summary>answer</summary>

positive feedback: experts receive gradient only from the tokens routed to them, so a popular expert improves faster, gets higher router logits, and attracts still more tokens. unpopular experts stop learning and become even less attractive.

</details>

5. an MoE with 8 experts and top-2 routing has how many times the parameters of the dense layer with one expert's width, and how many times the FLOPs?

<details><summary>answer</summary>

8× the expert parameters (plus a tiny router) and 2× the FLOPs: a token passes through 2 experts.

</details>

6. why does the Llama have no position embedding, and what would happen if you added one anyway?

<details><summary>answer</summary>

RoPE inside every attention layer already provides (relative) position. an extra absolute embedding would still work but adds parameters, fixes a maximum length, and partially undoes RoPE's relative-position property.

</details>

## common bugs

- **rotating the values** or forgetting to rotate the keys. scores become position-blind or the output position-scrambled.
- **`is_causal=True` with a cache and more than one new token.** the mask aligns to the wrong end; outputs differ from the uncached forward.
- **caching after `repeat_kv`**, or repeating heads in the wrong order (`k.repeat(1, n_rep, 1, 1)` interleaves 0,1,0,1 instead of 0,0,1,1).
- **a dense MoE**: computing every expert on every token. correct and $E/k$ times too slow.
- **detaching the auxiliary loss** or the gates: the router never learns, or never balances.
- **reporting MoE validation loss from `model(x, y)`**: it includes $\lambda_\text{aux} \sum \mathcal{L}_\text{aux}$; compare cross-entropies.

## going deeper

- Touvron et al. (2023), "LLaMA: open and efficient foundation language models". [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Shazeer (2020), "GLU variants improve transformer". [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Su et al. (2021), "RoFormer: enhanced transformer with rotary position embedding". [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Zhang & Sennrich (2019), "Root mean square layer normalization", *NeurIPS*. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- Shazeer (2019), "Fast transformer decoding: one write-head is all you need". [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- Ainslie et al. (2023), "GQA: training generalized multi-query transformer models from multi-head checkpoints", *EMNLP*. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Shazeer et al. (2017), "Outrageously large neural networks: the sparsely-gated mixture-of-experts layer", *ICLR*. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
- Fedus, Zoph & Shazeer (2022), "Switch transformers: scaling to trillion parameter models with simple and efficient sparsity", *JMLR*. [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)
- Jiang et al. (2024), "Mixtral of experts". [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)

## connections

- ← builds on: attention, RoPE, the KV cache and `generate_cached` (ch 18), RMSNorm (ch 15 · 07), the memory-bound decoding argument (ch 26 · 06).
- → used in: every later topic of this chapter (fine-tuning, preference learning, speculative decoding and evaluation all run on your `Llama`), the capstone's assistant, and chapter 28's interpretability tools, which hook into `blocks`.
