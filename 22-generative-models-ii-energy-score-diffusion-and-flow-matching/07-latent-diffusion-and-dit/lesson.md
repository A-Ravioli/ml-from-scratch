# latent diffusion & DiT

## why this matters

every model so far in this chapter works directly on pixels, and every sampling step runs the whole network over every pixel. that's fine for 28 × 28 digits and painful for 32 × 32 CIFAR images; for a 512 × 512 photograph it's 786,432 numbers per sample, a thousand times over. yet most of those numbers carry detail you'd never miss: the exact grain of the noise in a patch of sky. a good autoencoder can throw most of it away and still reconstruct an image a person can't tell from the original.

**latent diffusion** (Rombach, Blattmann, Lorenz, Esser & Ommer, 2022) splits the work accordingly: train an autoencoder once, then train the diffusion model in its compact latent space, where every step is many times cheaper, and decode only at the end. Stable Diffusion is this recipe. the second idea of this topic replaces the U-Net with a transformer: the **diffusion transformer** (DiT; Peebles & Xie, 2023) cuts the latent into patches, treats them as tokens like a ViT (chapter 18), and steers every block with the time and class through **adaptive layer norm**. DiTs scale with compute as predictably as language models do, which is why the large image and video generators of the last two years (Stable Diffusion 3, Sora-style video models, Flux) are latent DiTs trained with flow matching. you'll build the DiT and the latent pipeline around chapter 21's `ConvVAE`.

## warm-up: predict first

1. a 32 × 32 latent cut into 2 × 2 patches is how many tokens? with 1 × 1 patches, how much more does self-attention cost?
2. a transformer block whose last layer is initialized to zero, inside a residual connection: what function does it compute at initialization?
3. latents from an autoencoder have a standard deviation of 5. what goes wrong if you diffuse them as they are?
4. a latent diffusion model is perfect. what limits the quality of its samples?
5. which costs more per sampling step: a U-Net on 28 × 28 pixels or a small transformer on 49 latent tokens?

## prerequisites check

- [ ] patch embeddings, ViTs, `MultiHeadAttention`, `FeedForward`, sinusoidal positions (→ ch 18 · topics 01–03, 07)
- [ ] layer norm and why its scale and shift are learnable (→ ch 15 · topic 02)
- [ ] the `ConvVAE`, its spatial latents and the KL weight $\beta$ (→ ch 21 · topic 03)
- [ ] `TimeEmbedding`, the null-label convention, `ddpm_loss` / `flow_matching_loss` and their samplers (→ topics 03–06)
- [ ] `mlfs.torchnn.make_loader` (→ ch 15 · topic 07)

## core ideas

### idea 1: images as sequences of patches

a transformer works on a set of tokens, so first turn the latent $(C, H, W)$ into a sequence: cut it into non-overlapping $p \times p$ patches, read them in raster order (row by row), and flatten each into a vector of $p^2 C$ numbers. a linear layer (a convolution with kernel and stride $p$: chapter 18's `PatchEmbedding`) maps each patch vector to a token of width $d$. the patch size sets the cost: $N = (H/p)(W/p)$ tokens, and self-attention costs $O(N^2 d)$, so halving $p$ quadruples the tokens and multiplies attention cost by 16 (on paper 1). DiT's experiments found that smaller patches (more tokens, more compute, same parameters) give steadily better FID: compute, not parameter count, is what matters.

to undo it, the model's output tokens (each predicting $p^2 C$ numbers) are **unpatchified**: reshaped back into patches and stitched in raster order. patchify and unpatchify must be exact inverses with an agreed ordering inside each patch; here it's (row in patch, column in patch, channel).

attention doesn't know where a token came from (permuting tokens permutes the output), so positions are added to the tokens. DiT uses fixed 2-D sine-cosine embeddings: half of each token's channels encode its row with chapter 18's 1-D sinusoidal positions, the other half its column.

> **spec · stage 1.**
> - `patchify(x, patch_size)`: `(B, C, H, W)` → `(B, N, p·p·C)`, patches in raster order, each flattened in the order (row in patch, column in patch, channel); `ValueError` if $p$ doesn't divide $H$ and $W$.
> - `unpatchify(tokens, patch_size, channels)`: the exact inverse for square images, `(B, N, p·p·C)` → `(B, C, \sqrt N p, \sqrt N p)`.
> - `pos_embed_2d(grid_size, d_model)`: `(grid_size², d_model)`; the row for grid position $(r, c)$ (raster order) is the concatenation of `mlfs.transformer.sinusoidal_positions(grid_size, d_model // 2)` at $r$ and at $c$.

> **▶ build: stage 1.** implement `patchify`, `unpatchify` and `pos_embed_2d` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: adaLN-Zero

how should the noise level and the class reach a transformer? DiT compared three ways: append them as extra tokens, attend to them with cross-attention, or let them set the **scale and shift of every layer norm**. the last, adaptive layer norm, won, and with one more trick it won clearly. from the conditioning vector $c$ (time embedding plus class embedding), a small network produces six vectors per block, $(\beta_1, \gamma_1, \alpha_1, \beta_2, \gamma_2, \alpha_2)$:

$$h \leftarrow h + \alpha_1 \odot \operatorname{Attn}\big(\operatorname{LN}(h)\odot(1 + \gamma_1) + \beta_1\big), \qquad h \leftarrow h + \alpha_2 \odot \operatorname{MLP}\big(\operatorname{LN}(h)\odot(1 + \gamma_2) + \beta_2\big),$$

where LN has no learnable parameters of its own (the conditioning supplies them). the trick, **adaLN-Zero**, is to initialize the layer producing these vectors to zero. then every gate $\alpha$ starts at 0 and every block starts as the identity map: the network begins as a trivial function and grows into its depth, the same idea as zero-initializing the last layer of a residual branch (Goyal et al., 2017). the gates still receive gradient, so training moves them away from zero (on paper 2). one consequence you can check: a block with modulation but no positions inside is **permutation-equivariant** over tokens; all spatial awareness comes from the positional embedding added at the input.

> **spec · stage 2.** `DiTBlock(d_model, n_heads)`, an `nn.Module` with two `nn.LayerNorm(d_model, elementwise_affine=False)`, an `mlfs.transformer.MultiHeadAttention(d_model, n_heads)`, an `mlfs.transformer.FeedForward(d_model)`, and a modulation network SiLU → `nn.Linear(d_model, 6 * d_model)` with zero-initialized weight and bias; `forward(x, c)` maps `x` `(B, N, d)` and `c` `(B, d)` to `(B, N, d)` with the update above.

> **▶ build: stage 2.** implement `DiTBlock`.

### idea 3: the diffusion transformer

the full model: patch embedding plus the fixed 2-D positions (a buffer, not a parameter); the conditioning vector $c$ = `TimeEmbedding(d)` of $t$ plus a class embedding (a table with $K + 1$ rows, the last one the null label, so classifier-free guidance works unchanged); a stack of DiT blocks; a final layer norm with its own adaptive shift and scale; a linear layer to $p^2 C$ numbers per token; unpatchify. the adaptive final layer and the output layer are zero-initialized too, so a fresh DiT outputs exactly zero, which as a noise or velocity prediction means a starting loss of about 1 (for unit-scale data) instead of something huge.

the DiT doesn't care what objective it serves: it has the chapter's `model(x, t, y)` signature, so `ddpm_loss`, `flow_matching_loss`, every sampler and `cfg_model` work with it as with `TimeUNet`.

> **spec · stage 3.** `DiT(input_size=8, patch_size=2, in_channels=4, *, d_model=256, n_layers=6, n_heads=4, n_classes=None)`: `forward(x, t, y=None)` maps `(B, in_channels, input_size, input_size)` to the same shape; `mlfs.transformer.PatchEmbedding` plus `pos_embed_2d` as a buffer; `self.blocks`, an `nn.ModuleList` of `n_layers` `DiTBlock`s; the conditioning, null label and `ValueError` as for `TimeMLP`; the fixed positions registered as a buffer (not a parameter); attributes `ada_out` (SiLU → `nn.Linear(d_model, 2 * d_model)`, the final shift and scale, applied as in the blocks after a non-affine LayerNorm) and `out` (`nn.Linear(d_model, p * p * in_channels)`), both zero-initialized, so a fresh model returns exactly zero. this is `mlfs.diffusion.DiT`.

> **▶ build: stage 3.** implement `DiT`.

> **🔮 predict:** a 4-layer DiT with width 128 on 7 × 7 latents (49 tokens), against a `TimeUNet` with base width 16 on 28 × 28 pixels: which has more parameters, and which is faster per training step?

### idea 4: the latent pipeline

with an autoencoder in hand (chapter 21's `ConvVAE` maps $(C, H, W)$ images to $(4, H/4, W/4)$ latents), latent diffusion is three steps:

1. **encode the dataset once**: run the encoder over every image and keep the **means** $\mu(x)$ of $q(z \mid x)$, not samples (the sampled noise would only add variance that the diffusion model then has to learn). the encoder is frozen, so this is done once, and diffusion training never touches pixels.
2. **normalize**: divide the latents by their overall standard deviation, so they have unit scale. the forward process is designed for data of scale about 1: latents with standard deviation 5 would still be clearly visible at the "pure noise" end of the schedule, and latents with scale 0.1 would be destroyed in the first few steps (on paper 3). Stable Diffusion's famous constant 0.18215 is exactly this factor for its autoencoder.
3. **train and sample** any diffusion or flow model on the normalized latents; to produce images, multiply samples by the standard deviation and decode.

the autoencoder's KL weight $\beta$ decides how this goes. a large $\beta$ makes a smooth, nearly Gaussian latent space that's easy to model but loses detail; Rombach et al. use a tiny KL weight ($\approx 10^{-6}$), so the latents carry nearly all the information and look like a compressed image rather than a Gaussian. whatever you choose, the decoder sets a ceiling: a perfect latent model can't produce better images than the autoencoder's reconstructions, so the FID of the reconstructions is the floor for the FID of the samples.

> **spec · stage 4.**
> - `encode_latents(vae, loader, *, device='cpu')`: run `vae.encode(x)` → `(mu, logvar)` over a loader of `(x, y)` batches (eval mode, restored afterwards; no gradients); returns `(latents, labels)`: the means concatenated on the CPU, `(N, C, h, w)`, and the labels as int64 `(N,)`, in loader order. this is `mlfs.diffusion.encode_latents`.
> - `decode_latents(vae, z, *, scale=1.0, batch_size=256, device='cpu')`: `vae.decode(z * scale)` in batches of `batch_size` (eval mode, restored; no gradients), on the CPU.

> **▶ build: stage 4.** implement `encode_latents` and `decode_latents`.

> **🔮 predict:** two `ConvVAE`s, trained with KL weights $\beta = 10^{-3}$ and $\beta = 0.1$, each followed by the same latent DiT. which pipeline produces the better samples, and why?

## worked example

**patchify by hand.** a 2-channel 4 × 4 image whose entries are $0, 1, \dots, 31$ in `(C, H, W)` order, patch size 2: there are 4 patches of $2 \cdot 2 \cdot 2 = 8$ numbers. patch 1 is the top-right one (grid row 0, column 1), and its vector lists, for each (row, column) inside the patch, both channels: $(x_{0,0,2}, x_{1,0,2}, x_{0,0,3}, x_{1,0,3}, x_{0,1,2}, x_{1,1,2}, x_{0,1,3}, x_{1,1,3}) = (2, 18, 3, 19, 6, 22, 7, 23)$. the checks compare your `patchify` with this vector and require `unpatchify(patchify(x))` to return `x` exactly.

**token budgets.** CIFAR-10 through the `ConvVAE` gives $4 \times 8 \times 8$ latents: 256 numbers instead of 3,072 (12× fewer). with $p = 2$ that's 16 tokens; with $p = 1$, 64 tokens and 16 times the attention cost. MNIST gives $4 \times 7 \times 7$ latents; 7 is prime, so the patch size must be 1 (49 tokens) or 7 (a single token, useless).

**the start of training.** a fresh DiT outputs 0. for flow matching on unit-scale latents, the target $x_1 - x_0$ has variance 2 per coordinate (data plus noise), so the first loss is about 2; for $\epsilon$-prediction the target is unit noise and the first loss is about 1.

## on paper

1. **patch size and cost.** for a latent of side $L$, patch size $p$, width $d$ and $n$ blocks, count the tokens and estimate the multiply-adds of attention ($\sim N^2 d$ per block) and of the MLPs ($\sim 8 N d^2$ per block). when does attention dominate?

<details><summary>solution</summary>

$N = (L/p)^2$. attention scores and the weighted sum cost about $2N^2 d$ multiply-adds per block; the projections $4Nd^2$; the MLP (width $4d$) $8Nd^2$. attention dominates when $N \gtrsim 6d$, so for $d = 256$ above about 1,500 tokens: a 64 × 64 latent with $p = 2$ (1,024 tokens) is near the crossover; with $p = 1$ (4,096 tokens) attention dominates. the parameter count doesn't depend on $p$ (except for the tiny embedding layers), which is why DiT's "smaller patches" scale compute at fixed size.

</details>

2. **zero init still learns.** for the block $h' = h + \alpha\,f(h)$ with $\alpha = W c$ and $W = 0$ at initialization, show that $h' = h$ and that $\partial\mathcal{L}/\partial W \ne 0$ in general, while $\partial\mathcal{L}/\partial(\text{parameters of } f) = 0$ at that point.

<details><summary>solution</summary>

with $W = 0$, $\alpha = 0$ and $h' = h$. $\partial\mathcal{L}/\partial\alpha = (\partial\mathcal{L}/\partial h')\odot f(h)$, generally nonzero, so $\partial\mathcal{L}/\partial W = (\partial\mathcal{L}/\partial\alpha)\,c^\top \ne 0$: the gate opens on the first step. the gradient reaching $f$'s parameters is multiplied by $\alpha = 0$, so they don't move on the first step, and only start learning once the gate is open. the network starts shallow and deepens itself.

</details>

3. **why normalize latents.** latents with standard deviation $\sigma_z$ are diffused with the default linear schedule. what is the signal-to-noise ratio at the last step for $\sigma_z = 5$, and at which step does the SNR fall below 1 for $\sigma_z = 0.1$? (use $\bar\alpha_{999} \approx 4 \times 10^{-5}$, and $\bar\alpha = 0.5$ at step 259 and $\bar\alpha = 0.99$ near step 30.)

<details><summary>solution</summary>

SNR $= \bar\alpha\sigma_z^2/(1 - \bar\alpha)$. at the last step with $\sigma_z = 5$: $4 \times 10^{-5}\cdot 25 = 10^{-3}$, 25 times the intended $4 \times 10^{-5}$ (amplitude ratio 3%, not 0.6%): the sampler starts from noise that "should" contain a visible trace of the data. with $\sigma_z = 0.1$: SNR $< 1$ when $\bar\alpha/(1 - \bar\alpha) < 100$, i.e. $\bar\alpha < 0.99$, around step 30 instead of 259: almost the whole chain is spent on noise.

</details>

4. **the decoder's ceiling.** let $D$ be the decoder and $E$ the encoder's mean. show that if a latent model samples exactly from the distribution of $E(x)$, the decoded samples have the distribution of the reconstructions $D(E(x))$, not of $x$.

<details><summary>solution</summary>

decoding is a deterministic map: if $z \sim p_{E(x)}$ (the push-forward of the data through $E$), then $D(z) \sim$ the push-forward through $D \circ E$, the reconstructions. any information $E$ discards and $D$ can't restore is missing from every sample; the reconstruction error is a floor on the sample error, whatever the latent model.

</details>

## experiment

`lab.py` builds a latent generator for digits with your code:

1. a `ConvVAE` and its latents: reconstruction quality, the latents' scale, and what normalization does.
2. latent flow matching with your DiT against pixel flow matching with a `TimeUNet`: cost per step, cost per sample, and FID, with the reconstruction FID as the floor.
3. the autoencoder's KL weight against the quality of the latent model.

## check your understanding

1. why encode the means $\mu(x)$ rather than samples $z \sim q(z \mid x)$?

<details><summary>answer</summary>

samples add the encoder's posterior noise, which carries no information about the image; the diffusion model would have to learn to reproduce that noise. with a small KL weight the posterior is narrow anyway, so the means lose almost nothing. (Stable Diffusion samples, but with a tiny variance; the choice matters little there.)

</details>

2. why is the adaptive layer norm's own affine transform turned off?

<details><summary>answer</summary>

the conditioning network supplies the scale $(1 + \gamma)$ and shift $\beta$ per example; a second, unconditional affine transform after the norm would be redundant (it can be absorbed into the conditioning's bias), so it's dropped.

</details>

3. a DiT block is permutation-equivariant. where does spatial structure come from?

<details><summary>answer</summary>

only from the positional embedding added to the patch tokens at the input (and from the fixed raster order in which unpatchify reassembles the output). without positions, the model couldn't tell a digit from the same patches shuffled.

</details>

4. what makes latent diffusion cheaper: fewer sampling steps, or cheaper steps?

<details><summary>answer</summary>

cheaper steps: the sampler is the same, but each network call processes 12–48× fewer numbers. the one-off cost of training the autoencoder and encoding the dataset is amortized over every later training run and every sample.

</details>

5. your latent model's FID is 5 points worse than the autoencoder's reconstruction FID. where would you invest compute: the autoencoder or the latent model?

<details><summary>answer</summary>

the gap between the two is the latent model's share; the reconstruction FID is the autoencoder's share. compare their sizes: if reconstructions are already near the real-data floor, the latent model is the bottleneck; if the reconstruction FID is large, no latent model can beat it and the autoencoder needs work (a lower KL weight, more capacity, a perceptual or adversarial loss as Rombach et al. used).

</details>

## common bugs

- **patchify and unpatchify disagree on the order.** symptom: the model trains (it can learn any fixed permutation) but output patches come back scrambled at their borders, or the unit check `unpatchify(patchify(x)) == x` fails.
- **positional embeddings as a trainable parameter by accident.** symptom: harmless, but not the spec; register a buffer so it moves with `.to(device)` and isn't trained.
- **forgetting to zero-initialize the modulation.** symptom: early training is unstable and slower; a fresh DiT outputs large values.
- **diffusing unnormalized latents.** symptom: samples decode to washed-out or saturated images; divide by the latents' std before training and multiply after sampling.
- **encoding with the VAE in train mode, or with gradients.** symptom: memory blows up, or batch-dependent latents if the encoder has batch norm.
- **judging the latent model by its own loss.** symptom: a low loss and bad images; evaluate decoded samples (FID) against the reconstruction floor.

## going deeper

- Rombach, Blattmann, Lorenz, Esser & Ommer (2022), "High-resolution image synthesis with latent diffusion models", *CVPR*. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- Peebles & Xie (2023), "Scalable diffusion models with transformers", *ICCV*. DiT and adaLN-Zero. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- Esser et al. (2024), "Scaling rectified flow transformers for high-resolution image synthesis", *ICML*. latent flow matching with a multimodal DiT. [arXiv:2403.03206](https://arxiv.org/abs/2403.03206)
- Perez, Strub, de Vries, Dumoulin & Courville (2018), "FiLM: visual reasoning with a general conditioning layer", *AAAI*. the idea behind adaptive normalization. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
- Goyal et al. (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour". zero-initialized residual branches. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Bao et al. (2023), "All are worth words: a ViT backbone for diffusion models", *CVPR*. U-ViT. [arXiv:2209.12152](https://arxiv.org/abs/2209.12152)

## connections

- ← builds on: ViTs, attention and positional encodings (ch 18), the `ConvVAE` (ch 21 · topic 03), layer norm (ch 15), the chapter's objectives, samplers and guidance (topics 03–06).
- → used in: the chapter capstone's stretch goal (latent CIFAR-10), multimodal models that condition DiTs on text tokens (ch 25), large-scale training and its systems costs (ch 26), and the R2 milestone if you reproduce a latent-diffusion result.
