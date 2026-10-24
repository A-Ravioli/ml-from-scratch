# the transformer block

## why this matters

attention on its own is not a model. it moves information between positions: each token's output is a weighted average of other tokens' values. but averaging is nearly linear, and a stack of attentions alone can't compute much. it also has nowhere to *store* knowledge: "Paris is the capital of France" has to live in some weights. and a deep stack of anything is hard to train unless the gradient has an easy path back to the first layer.

the transformer block solves all three with parts you already know. a small MLP applied to every position separately does the per-token computation and holds most of the parameters. a **residual connection** around each part makes the block compute a *change* to its input rather than a replacement, so a 96-layer model is a sum of 96 small edits to one running vector. and **layer normalization** keeps that vector at a sane scale. where exactly the normalization goes turned out to matter: the original transformer normalized *after* each residual addition (post-LN) and needed a carefully tuned learning-rate warmup to train at all; GPT-2 moved it *before* each branch (pre-LN), and almost every model since has followed. the lab shows the difference on a 24-layer model.

this topic builds the block (with optional cross-attention, which makes it a decoder block), then stacks blocks into the full encoder–decoder transformer of Vaswani et al. (2017), the architecture chapter 17's seq2seq model was waiting for.

## warm-up: predict first

1. a block computes $x + f(x)$. if $f$'s output layer is initialized to zero, what does the block compute? what does a stack of 100 such blocks compute?
2. in a transformer with $d = 512$, which has more parameters: the attention of a block, or its MLP?
3. attention mixes information across positions. does the MLP?
4. with post-LN, $x \mapsto \text{LN}(x + f(x))$, is there any path from the output of block 24 back to block 1 that doesn't pass through a layer norm?
5. you add 24 residual branches to a vector, each with output variance about 1. what happens to the vector's size?

## prerequisites check

- [ ] you can implement layer normalization and say what it does to a vector's mean and scale (→ ch 15 · topic 02)
- [ ] you know what learning-rate warmup is for (→ ch 15 · topic 04)
- [ ] you've seen why identity shortcuts help deep networks: residual blocks (→ ch 16 · topic 04, if you've done it; idea 2 recaps what you need)
- [ ] your `MultiHeadAttention` does self- and cross-attention with masks and a cache, and `sinusoidal_positions` works (→ topics 01–02)
- [ ] you know teacher forcing and encoder–decoder models (→ ch 17 · topic 06)

## core ideas

### idea 1: the position-wise MLP

after attention has gathered information from other positions into each token, each token is processed on its own by a two-layer MLP, the same one at every position:

$$\operatorname{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2, \qquad W_1 \in \mathbb{R}^{d_{ff} \times d},\ W_2 \in \mathbb{R}^{d \times d_{ff}}.$$

the hidden width is conventionally $d_{ff} = 4d$, and the activation $\phi$ is the GELU, $\phi(x) = x\,\Phi(x)$ with $\Phi$ the standard normal CDF, a smooth ReLU that GPT and BERT use (the original transformer used ReLU). "position-wise" means it acts on the last dimension only: a $(B, T, d)$ input is just $BT$ independent vectors. this division of labour is the transformer's whole design: **attention communicates, the MLP computes**.

the MLP is also where most of the parameters are. attention has four $d \times d$ matrices, $4d^2$; the MLP has two $d \times 4d$ matrices, $8d^2$. two thirds of a block's weights are in its MLP, and interpretability work (Geva et al., 2021) suggests that's where much of a language model's factual knowledge is stored: the first layer's rows act as keys that detect patterns, the second layer's columns as the values written back.

> **spec.** `FeedForward(d_model, d_ff=None, *, dropout=0.0, activation='gelu')` has `fc1 = nn.Linear(d_model, d_ff)` and `fc2 = nn.Linear(d_ff, d_model)`, with `d_ff = 4 * d_model` by default; `activation` is `'gelu'` (exact, `nn.GELU()`) or `'relu'`, anything else raises `ValueError`. `forward(x)`: fc1 → activation → fc2 → dropout, on `(..., d_model)`.

> **🔮 predict:** with $d_{ff} = 4d$, what fraction of a block's weight matrices is in the MLP? does the answer depend on the number of heads?

> **▶ build: stage 1.** implement `FeedForward`.

### idea 2: the residual stream, and where the norm goes

think of the block's input $x$ as a vector that flows from the embedding to the output head, the **residual stream**. each sublayer reads the stream, computes something, and *adds* its result back: $x \leftarrow x + f(x)$. nothing ever overwrites the stream, so information written by layer 2 is still available to layer 30, and a sublayer that has nothing useful to contribute can learn to output zero and get out of the way. the backward pass sees the same structure. differentiating $x_{l+1} = x_l + f_l(x_l)$ gives $\partial x_{l+1}/\partial x_l = I + \partial f_l/\partial x_l$, and multiplying these out from layer $L$ down to $l$ always contains the identity term (on paper 1): the gradient reaches every layer directly, not only through all the layers above it. this is the same reason ResNets train at depth 152.

the stream needs normalization somewhere, and there are two choices. **post-LN**, the original:

$$x \leftarrow \text{LN}(x + \text{Attn}(x)), \qquad x \leftarrow \text{LN}(x + \text{FFN}(x)).$$

**pre-LN**, GPT-2 onwards:

$$x \leftarrow x + \text{Attn}(\text{LN}_1(x)), \qquad x \leftarrow x + \text{FFN}(\text{LN}_2(x)).$$

they look like a cosmetic reshuffle. they're not. in pre-LN the stream itself is never normalized; each branch sees a normalized copy, and the identity path from top to bottom is clean. in post-LN, every residual addition is followed by a norm, so the "shortcut" passes through $2L$ layer norms, and its gradient is rescaled at each one. Xiong et al. (2020) showed that at initialization the post-LN gradient is largest near the output and the effective step size of Adam is too big there: training diverges or stalls unless the learning rate warms up from near zero. pre-LN's gradients are well-behaved from the start, and it trains without warmup. two costs of pre-LN: since the stream is never normalized, its scale grows with depth (on paper 3), so the stack needs one **final layer norm** before the output head; and a well-tuned post-LN model is sometimes slightly better, which is why the question isn't completely closed.

a **decoder block** has three sublayers: causal self-attention, **cross-attention** (queries from the stream, keys and values from the encoder's output), and the MLP, each with its own norm and residual. dropout, when used, goes on each branch's output before it's added.

> **spec.** `TransformerBlock(d_model, n_heads, *, d_ff=None, dropout=0.0, norm_first=True, cross_attention=False)` has submodules `ln1`, `attn` (your `MultiHeadAttention`), `ln2`, `mlp` (your `FeedForward`), and with `cross_attention=True` also `ln_cross` and `cross_attn`. all norms are `nn.LayerNorm(d_model)`. `forward(x, context=None, *, mask=None, is_causal=False, cache=None, context_mask=None)`: `x` is `(B, T, d)`; `mask`, `is_causal` and `cache` go to the self-attention; `context` `(B, S, d)` is what cross-attention attends to, and `context_mask` (bool, broadcastable to `(B, H, T, S)`, True = attend, e.g. source padding as `(B, 1, 1, S)`) is cross-attention's mask. a cross-attention block called without a context, or a block without cross-attention given a `context` or `context_mask`, raises `ValueError`. dropout (`dropout`) applies to the output of each attention branch and inside the MLP. pre-LN (`norm_first=True`): `x = x + drop(attn(ln1(x)))`, then `x = x + drop(cross_attn(ln_cross(x), context))` if present, then `x = x + mlp(ln2(x))`. post-LN: `x = ln1(x + drop(attn(x)))`, `x = ln_cross(x + drop(cross_attn(x, context)))`, `x = ln2(x + mlp(x))`. it must match `nn.TransformerEncoderLayer` / `nn.TransformerDecoderLayer` with GELU and copied weights.

> **🔮 predict:** a 24-layer post-LN language model and a 24-layer pre-LN one, same learning rate ($10^{-3}$), no warmup. after 200 steps, how far apart are they? (lab, experiment 2.)

> **▶ build: stage 2.** implement `TransformerBlock`.

### idea 3: the encoder–decoder transformer

stack the blocks and you have the model of "attention is all you need". the **encoder** embeds the source tokens, adds sinusoidal positions, and runs $N$ blocks of bidirectional self-attention: every source token sees every other one, except padding (a key-padding mask, `(B, 1, 1, S)`, from a `(B, S)` boolean `src_mask`). its output is the **memory**, one vector per source token. the **decoder** embeds the target tokens shifted right (teacher forcing, as in chapter 17), adds positions, and runs $N$ decoder blocks: causal self-attention over the target so far, then cross-attention into the memory, with the same source-padding mask, so that no target position reads a padding token. a linear head turns each position into logits over the target vocabulary. with pre-LN, each stack ends with a final layer norm (`enc_norm`, `dec_norm`); with post-LN, the last block already ends in one, and those are identities.

compared with chapter 17's GRU seq2seq, nothing is sequential during training: the whole target is processed in one pass, every decoder position attends to every source position directly, and the encoder's view of the source is bidirectional at every layer. at inference you still generate one token at a time (and could reuse chapter 17's `beam_search` through a step function).

> **spec.** `EncoderDecoder(src_vocab_size, tgt_vocab_size, *, d_model=128, n_heads=4, n_layers=2, d_ff=None, dropout=0.0, max_len=256, norm_first=True)` has `src_emb`, `tgt_emb` (`nn.Embedding`), `encoder` and `decoder` (`nn.ModuleList`s of `n_layers` of your `TransformerBlock`, the decoder's with cross-attention), `enc_norm`, `dec_norm` and `head`. embeddings get `sinusoidal_positions` added (no $\sqrt d$ scaling: PyTorch's embeddings are already initialized at unit scale). `encode(src, src_mask=None)`: `(B, S)` int64 and an optional `(B, S)` bool mask, True = real token → memory `(B, S, d)`. `decode(tgt_in, memory, memory_mask=None)` → logits `(B, T, tgt_vocab_size)`, causal in `tgt_in`, with `memory_mask` `(B, S)` (True = real source token) given to every decoder block as `context_mask`, so no target position attends to source padding. `forward(src, tgt_in, *, src_mask=None)` = `decode(tgt_in, encode(src, src_mask), src_mask)`.

> **▶ build: stage 3.** implement `EncoderDecoder`.

### idea 4: watching the gradient

to see the difference between pre-LN and post-LN, you need to see how much gradient each block receives. after a backward pass, the gradient of block $l$ is the concatenation of the gradients of all its parameters, and its size is the 2-norm of that concatenation: $\sqrt{\sum_{p \in \text{block } l} \|\nabla_p\|^2}$, the square root of the summed squares (not the sum of per-parameter norms). chapter 15's `DiagnosticsLogger` tracked this per parameter; here you want it per block.

> **spec.** `block_grad_norms(blocks)`: `blocks` is an iterable of modules (e.g. `model.blocks`). returns a list of Python floats, one per block: the 2-norm over all of that block's parameter gradients, skipping parameters whose `.grad` is `None`; 0.0 for a block with no gradients at all.

> **▶ build: stage 4.** implement `block_grad_norms`.

## worked example

**parameter count of one block**, $d = 512$, 8 heads, $d_{ff} = 2048$:

| part | parameters |
|---|---|
| attention: 4 projections of $512 \times 512$ plus 4 biases of 512 | $4(512^2 + 512) = 1{,}050{,}624$ |
| MLP: $512 \times 2048 + 2048$ and $2048 \times 512 + 512$ | $2{,}099{,}712$ |
| 2 layer norms, a weight and a bias of 512 each | $2{,}048$ |
| **total** | **3,152,384** |

the number of heads doesn't appear: heads split the width, they don't add to it. a decoder block adds a second attention and a third norm: $3{,}152{,}384 + 1{,}050{,}624 + 1{,}024 = 4{,}204{,}032$. a 6-layer encoder plus 6-layer decoder (the original "base" model) is $6 \times 3.15\text{M} + 6 \times 4.20\text{M} \approx 44\text{M}$ parameters in its blocks; its shared vocabulary embedding of $37{,}000 \times 512$ adds 19M, close to the 65M the paper reports. your code must reproduce the per-block numbers.

## on paper

1. **the identity path.** for a pre-LN stack $x_{l+1} = x_l + f_l(x_l)$, $l = 0, \dots, L - 1$, write $\partial x_L / \partial x_0$ as a product, and expand it to show it contains $I$. for post-LN, $x_{l+1} = \text{LN}(x_l + f_l(x_l))$, what multiplies the identity term?

<details><summary>solution</summary>

pre-LN: $\frac{\partial x_L}{\partial x_0} = \prod_{l = L - 1}^{0} (I + J_l)$ with $J_l = \partial f_l/\partial x_l$. expanding gives $I + \sum_l J_l + \sum_{l > m} J_l J_m + \dots$: the identity is there regardless of what the $J_l$ are, so the gradient of the loss reaches $x_0$ at least as $\partial \mathcal{L}/\partial x_L$ itself. post-LN: $\frac{\partial x_L}{\partial x_0} = \prod_l N_l (I + J_l)$ with $N_l$ the layer norm's Jacobian at that point. the "identity" term is now $\prod_l N_l$, a product of $L$ (here $2L$, with two norms per block) projections-and-rescalings. LN's Jacobian scales by $\sqrt d / \|x - \bar x\|$ and projects out the mean direction; its size depends on the activations, so the product can shrink or grow with depth, and the shortcut is no longer a clean highway.

</details>

2. **where the compute goes.** count the multiply-adds per token for one block's matrix multiplications (projections and MLP, ignoring attention's $T^2$ part). how does that compare with the parameter count? use it to estimate the training FLOPs per token of a model with $N$ parameters.

<details><summary>solution</summary>

each weight matrix entry is used once per token in the forward pass: one multiply-add per parameter, so $\approx 12d^2$ multiply-adds per token per block ($4d^2$ attention, $8d^2$ MLP), $2 \times 12d^2$ FLOPs. the backward pass costs about twice the forward (gradients with respect to both inputs and weights). total: $\approx 6$ FLOPs per parameter per token, the $6ND$ rule that chapter 19's scaling laws and chapter 26's performance model use.

</details>

3. **the stream grows.** in a pre-LN stack, suppose each branch's output at initialization has independent entries of variance $\sigma^2$, uncorrelated with the stream. what is the variance of the stream after $L$ blocks (2 branches each), starting from variance $v_0$? why does this make a final layer norm necessary?

<details><summary>solution</summary>

variances of uncorrelated sums add: $v_L = v_0 + 2L\sigma^2$, growing linearly with depth. the head would see inputs whose scale depends on depth (and drifts during training), and the last blocks' contributions would be relatively smaller than the first ones'. a final LN rescales the stream to unit variance before the head. GPT-2 also scaled the initialization of each branch's output layer by $1/\sqrt{2L}$ to keep $v_L$ independent of depth, which topic 04's GPT uses.

</details>

## experiment

the lab builds a 24-layer character language model from your blocks, once pre-LN and once post-LN. first it looks at the gradient each block receives at initialization. then it trains three versions for 200 steps at the same learning rate: pre-LN, post-LN, and post-LN with a learning-rate warmup. one of them learns almost nothing; find out which, and what it does learn.

## check your understanding

1. "attention communicates, the MLP computes." what would a transformer with no MLPs be unable to do well? and one with no attention?

<details><summary>answer</summary>

without MLPs, each layer outputs weighted averages of linear projections, so per-token computation is limited to these near-linear mixtures (plus layer norms): no rich nonlinear feature detectors, little capacity to store facts. without attention, every position is processed independently: the model is a bag of per-token MLPs and can't use context at all, a unigram model however deep it is.

</details>

2. why does dropout go on each branch's output before the residual addition, and not on the stream after it?

<details><summary>answer</summary>

dropping entries of the stream itself would destroy information written by every earlier layer, at every block: 24 blocks of 10% dropout would leave very little intact. dropping a branch's output only perturbs that layer's contribution, while the stream keeps everything else.

</details>

3. in the encoder–decoder, why does the decoder's self-attention need a causal mask but its cross-attention doesn't?

<details><summary>answer</summary>

self-attention over the target would otherwise let position $t$ see target tokens after $t$, which during training are the answers it must predict (teacher forcing feeds the whole target at once). cross-attention looks at the source sentence, which is fully known at inference time, so every target position may see all of it.

</details>

4. the pre-LN block's two layer norms each have a weight and bias. in a stack, what does the final layer norm add that the blocks' norms don't provide?

<details><summary>answer</summary>

the blocks' norms only normalize the *inputs* to their branches; the stream itself, which is what the head reads, is never normalized. the final norm rescales it (on paper 3).

</details>

5. a colleague says post-LN is "obviously" better because each block's output is normalized. what does the lab show, and what's the standard fix if you want post-LN anyway?

<details><summary>answer</summary>

at depth, post-LN without warmup stalls at the unigram loss while pre-LN trains. the fix is a learning-rate warmup (the original transformer warmed up for 4,000 steps), and/or careful initialization (e.g. DeepNet's scaled residuals, Wang et al., 2022). pre-LN avoids the problem by construction.

</details>

## common bugs

- **normalizing the stream in pre-LN.** `x = ln1(x); x = x + attn(x)` normalizes the residual stream itself: it's neither pre-LN nor post-LN, and the oracle test fails. the norm goes on the branch input only.
- **sharing one LayerNorm for both branches.** each sublayer has its own `ln`, with its own weights.
- **the tanh GELU.** `F.gelu(x, approximate="tanh")` is GPT-2's approximation; PyTorch's encoder layer uses the exact erf form, and the oracle tests compare against that (the difference is about $10^{-3}$).
- **forgetting the final norm** in a pre-LN stack: the model trains, a bit worse, and its loss is sensitive to depth.
- **counting heads in the parameters.** splitting into heads is a reshape; the projections are $d \times d$ regardless.
- **the key-padding mask as `(B, S)`.** it must broadcast over heads and queries: `(B, 1, 1, S)`.

## going deeper

- Vaswani et al. (2017), "Attention is all you need", *NeurIPS*. section 3.1 (encoder and decoder stacks), 3.3 (position-wise FFN), 5.3 (warmup). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang & Liu (2020), "On layer normalization in the transformer architecture", *ICML*. why post-LN needs warmup, pre-LN doesn't. [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Liu, Liu, Gao, Chen & Han (2020), "Understanding the difficulty of training transformers", *EMNLP*. post-LN's dependence on its residual branches. [arXiv:2004.08249](https://arxiv.org/abs/2004.08249)
- Ba, Kiros & Hinton (2016), "Layer normalization". [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
- Hendrycks & Gimpel (2016), "Gaussian error linear units (GELUs)". [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Geva, Schuster, Berant & Levy (2021), "Transformer feed-forward layers are key-value memories", *EMNLP*. [arXiv:2012.14913](https://arxiv.org/abs/2012.14913)
- Elhage et al. (2021), "A mathematical framework for transformer circuits". the residual-stream view. [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)

## connections

- ← builds on: multi-head attention and its cache hook (topic 01), sinusoidal positions (02), layer norm, dropout and warmup (ch 15), seq2seq with teacher forcing (ch 17), residual connections (ch 16 · topic 04).
- → used in: GPT stacks causal blocks (topic 04), BERT stacks bidirectional ones (05), the cache runs through the blocks (06), the ViT stacks them over patches (07); the R1 milestone can reproduce a translation result with your `EncoderDecoder`; DiT (ch 22), MAE and CLIP (ch 25), Llama's RMSNorm/SwiGLU variant of the block (ch 27), and residual-stream interpretability (ch 28).
