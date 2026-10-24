# masked modeling

## why this matters

BERT (ch 18 · topic 05) learned language by hiding 15% of the words in a sentence and predicting them. for years the same idea didn't work nearly as well for images: masked pixels or patches were easy to fill in by smearing their neighbours, and contrastive methods (topics 02–03) dominated. then He et al. (2022) changed three things and made masked image modeling competitive with, and on fine-tuning better than, everything else: hide most of the image (75%, not 15%), don't feed the hidden patches to the encoder at all, and reconstruct raw pixels with a small separate decoder. the result, the **masked autoencoder** (MAE), is simple, needs no augmentations beyond crops, no negatives and no momentum networks, and trains 3× faster than its encoder size suggests because the encoder only ever sees a quarter of each image.

in this topic you build an MAE from your chapter 18 transformer blocks: patchify, random masking, the asymmetric encoder–decoder, and the masked loss. in the lab you find out how much of an image should be hidden, and what an MAE "imagines" behind the mask.

## warm-up: predict first

1. BERT masks 15% of tokens. why might the right fraction be much higher for image patches?
2. the encoder never sees the masked patches. how does the decoder know *where* to put its predictions?
3. you mask 75% of the patches. roughly how much cheaper is the encoder's forward pass than on the full image?
4. why compute the loss only on the masked patches, not on all of them?
5. an MAE is trained to output pixels with a squared-error loss. will its reconstructions be sharp?

## prerequisites check

- [ ] you have `TransformerBlock`, `sinusoidal_positions` and the ViT's patch embedding (→ ch 18 · topics 02, 03, 07)
- [ ] you know BERT's masked-language-model objective and `mask_tokens` (→ ch 18 · topic 05)
- [ ] you know autoencoders and reconstruction losses (→ ch 11 · topic 03, ch 14)
- [ ] you can use `torch.gather` and `torch.argsort` along a dimension (→ ch 15)
- [ ] you have the evaluation harness (→ topic 01)

## core ideas

### idea 1: an image as a sequence of patches

a transformer works on a sequence of tokens. a ViT turns an image into one by cutting it into non-overlapping $p \times p$ patches, $N = (H/p)(W/p)$ of them, and flattening each into a vector of $p^2C$ numbers. MAE needs that flattening as an explicit operation, `patchify`, because its *targets* are flattened patches, and it needs the inverse, `unpatchify`, to turn predictions back into pictures.

order matters, because the loss compares prediction and target coordinate by coordinate. patches are numbered row by row over the image grid. inside a patch, MAE's code orders the values by row $i$, then column $j$, then channel $c$ (channel fastest), i.e. `einsum('nchpwq->nhwpqc')`. this is *not* the order of chapter 18's `patchify`, which put the channel first ($(c, i, j)$, the layout `Conv2d` weights use). either is fine as long as `patchify` and `unpatchify` agree; we follow MAE.

> **spec.** `patchify(imgs, patch_size)`: `(B, C, H, W)` → `(B, N, p·p·C)`, patches in row-major order over the grid, each flattened in $(i, j, c)$ order; `ValueError` if $H$ or $W$ isn't divisible by $p$. `unpatchify(patches, patch_size, channels=3)`: the exact inverse for square images, `(B, N, p·p·C)` → `(B, C, H, W)` with $H = W = p\sqrt N$.

> **▶ build: stage 1.** implement `patchify` and `unpatchify`.

### idea 2: random masking, with a way back

MAE masks a random subset of patches per image, uniformly, a fresh draw every time an image is seen. the implementation trick makes this batched and loop-free: draw one random number per patch, **sort** the patches by it, keep the first $N_\text{keep}$ (a random subset, since the sort order is random), and remember the permutation so you can undo it later:

1. `noise = torch.rand(B, N)`; `ids_shuffle = argsort(noise)` (ascending: the smallest noise values are kept);
2. `ids_restore = argsort(ids_shuffle)`, the inverse permutation;
3. `x_visible = gather(x, ids_shuffle[:, :N_keep])`;
4. the binary `mask` (1 = hidden) is 0 for the first $N_\text{keep}$ positions *in shuffled order*, gathered back to the original order with `ids_restore`.

later, the decoder appends placeholder tokens for the hidden patches after the visible ones and applies `ids_restore` to put every token back where it belongs.

how many to keep: $N_\text{keep} = \operatorname{round}(N(1 - r))$ for mask ratio $r$. MAE's reference code uses `int(N * (1 - r))`, which truncates; with $N = 10$ and $r = 0.9$ it computes `int(0.9999999999999998)` = 0 because $1 - 0.9$ isn't exactly $0.1$ in floating point. round to the nearest integer instead (ties up), and refuse to mask everything.

> **spec.** `random_masking(x, mask_ratio, *, generator=None)`: `x` `(B, N, D)`. $N_\text{keep} = \lfloor N(1 - r) + \frac12\rfloor$, `ValueError` if that's 0. draw `noise = torch.rand(B, N, generator=generator)` on the CPU (move it to `x`'s device afterwards), then follow steps 1–4. returns `(x_visible (B, N_keep, D), mask (B, N) float, 1 = masked, ids_restore (B, N) int64)`.

> **▶ build: stage 2.** implement `random_masking`.

### idea 3: the asymmetric encoder

MAE's central design choice is that the encoder sees **only the visible patches**. no mask tokens, no zeros in their place: the masked patches are dropped from the sequence. that's possible because a transformer doesn't care how long its input is (positions come from positional embeddings, added *before* masking, so every visible token still knows where it sits). and it's what makes MAE cheap: at 75% masking the encoder processes a quarter of the tokens, so its MLPs cost a quarter and its attention a sixteenth (on paper 1).

the encoder is a ViT without a [CLS] token: a linear **patch embedding** applied to `patchify(x)`, fixed sinusoidal positional embeddings (MAE uses 2-D sin-cos; we use your 1-D `sinusoidal_positions` over the patch index, which works fine at this size), a stack of pre-LN `TransformerBlock`s and a final LayerNorm.

for **evaluation**, the encoder runs on *all* patches, no masking, and its output tokens are averaged into one feature vector (`encode`). there's a quiet train/test mismatch here: the encoder was trained on sequences of $N/4$ tokens and is evaluated on $N$. it works, but it's one reason MAE features need fine-tuning to shine: on ImageNet, MAE's linear-probe accuracy is well below contrastive methods like MoCo v3, while its fine-tuning accuracy is higher (He et al., 2022).

> **spec.** `MAE(img_size=32, patch_size=4, in_channels=3, *, encoder_dim=192, encoder_depth=6, encoder_heads=3, decoder_dim=128, decoder_depth=2, decoder_heads=4, mask_ratio=0.75, norm_pix_loss=False)`: `ValueError` if `img_size % patch_size`. encoder attributes: `patch_embed = nn.Linear(p·p·C, encoder_dim)`; buffer `pos_embed = sinusoidal_positions(N, encoder_dim)[None]`; `blocks`, an `nn.ModuleList` of `encoder_depth` `TransformerBlock(encoder_dim, encoder_heads)`; `norm = nn.LayerNorm(encoder_dim)`. decoder attributes: `decoder_embed = nn.Linear(encoder_dim, decoder_dim)`; `mask_token`, a parameter `(1, 1, decoder_dim)` (normal init, std 0.02); buffer `decoder_pos_embed = sinusoidal_positions(N, decoder_dim)[None]`; `decoder_blocks` (`decoder_depth` blocks with `decoder_heads` heads); `decoder_norm`; `decoder_pred = nn.Linear(decoder_dim, p·p·C)`. `encode(x)` returns the mean over tokens of `norm(blocks(patch_embed(patchify(x)) + pos_embed))`, shape `(B, encoder_dim)`.

> **🔮 predict:** before any training, how useful are `encode`'s features from a randomly initialised MAE for a linear probe on 10-class MNIST (chance 0.10)? and after training at 75% masking, the encoder is fed all 49 patches although it only ever saw 12 at a time: will that hurt? (the lab prints both.)

> **▶ build: stage 3.** implement `MAE.__init__` and `MAE.encode`.

### idea 4: the decoder and the masked loss

the decoder's job is to turn the encoder's visible tokens back into a full image's worth of predictions. project the encoded tokens to the (narrower) decoder width, append one copy of a learned **mask token** for every hidden patch, unshuffle with `ids_restore` so the sequence is in patch order again, add the decoder's positional embeddings (so each mask token knows *which* patch it stands for), run a few transformer blocks, and project every token to $p^2C$ pixel values. the decoder is deliberately small (MAE's default is 8 blocks of width 512 for a ViT-L encoder of width 1024 and depth 24); it's thrown away after pretraining, and keeping it shallow pushes the semantic work into the encoder.

the loss is the mean squared error per patch, **averaged over the masked patches only**:

$$\mathcal L = \frac{\sum_{k} m_k \cdot \frac{1}{p^2C}\|\hat y_k - y_k\|^2}{\sum_k m_k}.$$

the visible patches are excluded because reconstructing them is trivial (the encoder saw them), and including them dilutes the signal: He et al. report that the masked-only loss gives better representations. their best variant also normalises each target patch by its own mean and variance (`norm_pix_loss`), so the model is graded on structure and contrast within a patch rather than on its absolute brightness.

**BERT, revisited.** BERT and MAE are the same idea, but the right masking ratio differs by 5×. language is dense: each word carries a lot of meaning, and predicting 15% of them is already a hard, semantic task. images are redundant: a hidden patch can usually be interpolated from its neighbours with no understanding at all. masking 75% removes most neighbours, so the model has to reason about whole objects. BERT also *replaces* masked words with a `[MASK]` token (and sometimes a random word), keeping the sequence length; MAE *drops* them from the encoder, which is cheaper and avoids the encoder ever seeing a token that doesn't exist at test time.

> **spec.** `MAE.forward(x)` returns `(loss, pred, mask)`. `patches = patchify(x)`; tokens `patch_embed(patches) + pos_embed`; `random_masking(tokens, mask_ratio)` with torch's global generator (the only random draw in `forward`); the encoder blocks and `norm` on the visible tokens; `decoder_embed`; append `mask_token` for the hidden patches; unshuffle with `ids_restore`; add `decoder_pos_embed`; `decoder_blocks`, `decoder_norm`, `decoder_pred` → `pred` `(B, N, p·p·C)`. the target is `patches`, or with `norm_pix_loss` each patch normalised as $(y - \bar y)/\sqrt{\operatorname{var}(y) + 10^{-6}}$ (unbiased variance over the patch's values). `loss` as above; `mask` from `random_masking`.

> **🔮 predict:** you train MAEs at 25%, 50%, 75% and 90% masking on MNIST and probe their features. which ratio wins? and which one reconstructs best?

> **▶ build: stage 4.** implement `MAE.forward`.

### idea 5: pretraining

no augmentations beyond random crops (we skip even those), no second view: each step masks a batch, reconstructs, and steps. MAE trains with AdamW, $\beta = (0.9, 0.95)$ (a smaller $\beta_2$ than the default, common for transformers: the second-moment estimate adapts faster to changing gradient scales), weight decay 0.05, and long schedules (800–1,600 epochs on ImageNet; the loss keeps improving the representation long after reconstructions look good). because every epoch draws fresh masks, the model effectively sees a new task every time: there are about $9 \times 10^{10}$ ways to keep 12 of MNIST's 49 patches.

> **spec.** `train_mae(model, images, *, epochs=1, batch_size=256, lr=1e-3, weight_decay=0.05, seed=0, device='cpu')`: `torch.manual_seed(seed)` first; move the model to `device`; `torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)`. each epoch: `model.train()`, a fresh permutation from `np.random.default_rng(seed)` (one generator per run), `N // bs` full batches with `bs = min(batch_size, N)`; per batch, step on `model(xb)[0]`. returns `{'loss': [...]}`, one float per step.

> **▶ build: stage 5.** implement `train_mae`.

## worked example

**patchify.** one image with $C = 2$ channels of $2 \times 2$ pixels, channel 0 $= \begin{pmatrix}1 & 2\\ 3 & 4\end{pmatrix}$ and channel 1 $= \begin{pmatrix}5 & 6\\ 7 & 8\end{pmatrix}$, with $p = 2$: one patch, flattened in $(i, j, c)$ order: $(1, 5, 2, 6, 3, 7, 4, 8)$. chapter 18's order would give $(1, 2, 3, 4, 5, 6, 7, 8)$.

**masking.** $N = 5$ patches, ratio $0.6$, so $N_\text{keep} = \lfloor 5 \cdot 0.4 + 0.5\rfloor = 2$. with noise $(0.7, 0.1, 0.9, 0.4, 0.3)$: `ids_shuffle` $= (1, 4, 3, 0, 2)$ (patches by increasing noise), so patches 1 and 4 are kept; `ids_restore` $= (3, 0, 4, 2, 1)$ (patch 0 sits at position 3 of the shuffled order, and so on); the shuffled mask $(0, 0, 1, 1, 1)$ gathered with `ids_restore` gives `mask` $= (1, 0, 1, 1, 0)$: patches 1 and 4 visible, as they should be.

**the loss.** two patches of 4 values, targets $(0, 0, 0, 0)$ and $(1, 1, 1, 1)$, predictions $(0.5, 0.5, 0.5, 0.5)$ for both, mask $(1, 0)$: the per-patch errors are $0.25$ and $0.25$, but only the first is masked, so $\mathcal L = 0.25$. with mask $(1, 1)$ it would also be $0.25$; with predictions $(0, 0, 0, 0)$ and $(0, 0, 0, 0)$ and mask $(0, 1)$, $\mathcal L = 1$.

the checks reproduce `random_masking` with exactly these steps on seeded noise.

## on paper

1. **what masking saves.** for a transformer block on $T$ tokens of width $d$, attention costs $\approx 2T^2d$ multiply-adds for the scores and weighted sum, and the projections plus MLP about $12Td^2$. with mask ratio $r$, compare the encoder's cost on $T = N(1 - r)$ tokens with the full $T = N$, for $r = 0.75$, $N = 196$, $d = 1024$.

<details><summary>solution</summary>

the $Td^2$ terms scale by $(1 - r) = 1/4$ and the $T^2d$ terms by $(1 - r)^2 = 1/16$. with $N = 196$ and $d = 1024$: full cost $2 \cdot 196^2 \cdot 1024 + 12 \cdot 196 \cdot 1024^2 \approx 0.079 + 2.47$ GFLOPs (in multiply-adds) per block; masked, $\approx 0.005 + 0.62$. the MLP dominates at this width, so the saving is close to 4×, and MAE's paper reports a 3× or more wall-clock speedup including the decoder. at long sequence lengths (small patches, video) the attention term dominates and the saving approaches 16×.

</details>

2. **why the reconstructions are blurry.** for a fixed visible context $v$, the hidden patch $Y$ is uncertain, with conditional distribution $p(y \mid v)$. show that the prediction minimising $\mathbb E[\|\hat y - Y\|^2 \mid v]$ is $\hat y = \mathbb E[Y \mid v]$. what does that look like when two different strokes are equally plausible?

<details><summary>solution</summary>

$\mathbb E\|\hat y - Y\|^2 = \|\hat y - \mathbb E Y\|^2 + \mathbb E\|Y - \mathbb E Y\|^2$ (expand around the mean; the cross term vanishes), minimised by $\hat y = \mathbb E[Y \mid v]$. with two equally likely strokes, the optimal prediction is their pixel-wise average: two half-intensity ghost strokes, a grey blur. an MAE is therefore a poor generator, but the conditional mean still requires knowing *which* completions are plausible, and that's the knowledge that ends up in the encoder.

</details>

3. **norm_pix_loss.** show that with per-patch normalised targets, the loss is unchanged if a hidden patch's pixels are all multiplied by $a > 0$ and shifted by $b$. what does the model no longer need to predict, and why might that help the representation?

<details><summary>solution</summary>

$(ay + b - \overline{ay + b})/\sqrt{\operatorname{var}(ay + b)} = a(y - \bar y)/(a\sqrt{\operatorname{var}(y)}) = (y - \bar y)/\sqrt{\operatorname{var}(y)}$ (ignoring the tiny $\epsilon$): the target is the same. the model no longer has to predict a patch's average brightness and contrast, low-frequency information that is easy to guess from neighbours and says little about content; its capacity goes to the patch's internal structure (edges, texture), which He et al. found gives better features.

</details>

## experiment

`lab.py` investigates, with your code, on unlabelled MNIST:

1. how much should you hide? MAEs at four mask ratios (two seeds each): kNN and linear-probe accuracy of their features, and their reconstruction error next to a trivial baseline.
2. what does an MAE paint in? reconstructions at each ratio.

## check your understanding

1. why must the positional embeddings be added *before* masking?

<details><summary>answer</summary>

after masking, the visible tokens are a shuffled subset; their position in the sequence says nothing about where they are in the image. adding the positional embedding first stamps each token with its true location, so the encoder can relate visible patches spatially, and the decoder's own positional embeddings tell each mask token which hole it fills.

</details>

2. what goes wrong if the decoder forgets to unshuffle with `ids_restore`?

<details><summary>answer</summary>

the visible tokens sit in shuffled order and the mask tokens all at the end, so the decoder's positional embedding for position $k$ is added to the wrong token, and prediction $k$ is compared with the target of a different patch. training still lowers the loss a little (the decoder can learn averages), but the reconstructions are scrambled and the representation suffers.

</details>

3. why does MAE replace masked patches with nothing in the encoder, while BERT replaces them with `[MASK]`?

<details><summary>answer</summary>

dropping is cheaper (the encoder processes only visible tokens) and avoids a train/test mismatch in the *encoder*: at test time there are no `[MASK]` tokens, so an encoder trained on them sees inputs it never sees in use. BERT can't easily drop tokens because its encoder output is used at every position, including masked ones, to predict the word there; MAE delegates that to the separate decoder.

</details>

4. an MAE trained at 75% masking gets a lower linear-probe accuracy than a SimCLR model of the same size, but beats it after fine-tuning. what might explain the difference?

<details><summary>answer</summary>

MAE's features are trained to support pixel reconstruction, which keeps a lot of low-level, spatially detailed information in a form that isn't linearly separable by class; contrastive features are explicitly trained to be invariant and to cluster semantically similar images. fine-tuning can reorganise MAE's rich features for the task; a linear probe can't. the mean pooling over all patches at evaluation time (versus a quarter of them in training) adds a mismatch on top.

</details>

5. why compute `n_keep` by rounding rather than with `int()`?

<details><summary>answer</summary>

`int` truncates, and $N(1 - r)$ is computed in floating point: $10 \times (1 - 0.9) = 0.9999999999999998$, so `int` keeps zero patches (and the encoder gets an empty sequence) when the intended number is 1. rounding to the nearest integer gives the intended count whenever $N(1 - r)$ is meant to be an integer.

</details>

## common bugs

- **channel-first patches** (ch 18's order) with MAE-order `unpatchify`, or vice versa: reconstructions look like coloured noise even when the loss is low.
- **forgetting to unshuffle** before the decoder: scrambled reconstructions.
- **loss over all patches:** lower numbers, weaker features.
- **mask tokens in the encoder** (zeroing or replacing instead of dropping): it works, but loses MAE's speed and adds a train/test mismatch.
- **`int(N * (1 - r))`:** off by one for some ratios, zero visible patches for others.
- **a generator on the wrong device:** `torch.rand(..., generator=cpu_gen, device='mps')` raises; draw on the CPU and move.

## going deeper

- He, Chen, Xie, Li, Dollár & Girshick (2022), *Masked autoencoders are scalable vision learners*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Devlin, Chang, Lee & Toutanova (2019), *BERT: pre-training of deep bidirectional transformers for language understanding*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- Bao, Dong, Piao & Wei (2022), *BEiT: BERT pre-training of image transformers*: predicting discrete visual tokens instead of pixels. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
- Xie et al. (2022), *SimMIM: a simple framework for masked image modeling*. [arXiv:2111.09886](https://arxiv.org/abs/2111.09886)
- Assran et al. (2023), *Self-supervised learning from images with a joint-embedding predictive architecture* (I-JEPA): predicting masked regions in representation space. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)
- Vincent, Larochelle, Bengio & Manzagol (2008), *Extracting and composing robust features with denoising autoencoders*. [ICML 2008](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)

## connections

- ← builds on: transformer blocks, positional encodings, ViT patches and BERT's masking (ch 18), autoencoders (ch 11, ch 14), the evaluation harness (topic 01).
- → used in: masked prediction underlies modern vision and multimodal pretraining; the capstone can swap in your MAE as its pretraining method (a stretch goal), so you can compare masked and contrastive pretraining on the same label-efficiency curve.
