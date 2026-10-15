# vision transformers

## why this matters

chapter 16 built convolutional networks around two assumptions about images: what matters is local (a 3×3 filter looks at neighbouring pixels) and the same pattern means the same thing anywhere (one filter slides over the whole image). these assumptions, the convolution's **inductive biases**, are why a ResNet learns CIFAR-10 from 50,000 images.

in 2020, Dosovitskiy et al. asked what happens if you throw them away. cut the image into 16×16-pixel patches, treat each patch as a "word", and feed the sequence to a standard transformer encoder, the one you built in topics 01–05, with no convolutions at all. the answer was a surprise in two directions. trained on ImageNet's 1.3 million images, the vision transformer (ViT) was *worse* than a ResNet of similar size. trained on 300 million images, it was better, and cheaper to train. with enough data, learning the structure of images beats having it built in.

this topic builds the ViT (patches, a [CLS] token, learned positions, your blocks), and the lab reproduces the first half of that story at laptop scale: the ViT against your chapter-16 ResNet, with 1,000 to 20,000 CIFAR-10 images. it then looks at what the trained ViT's attention heads do, and whether it even needs to know where its patches came from. the same patchify-then-transform recipe runs chapter 22's diffusion transformer and chapter 25's MAE and CLIP.

## warm-up: predict first

1. a 32×32 image cut into 4×4 patches: how many tokens? and a 224×224 image in 16×16 patches?
2. a patch embedding is "a linear layer on each flattened patch". why is it usually written as a convolution?
3. a ViT and a ResNet of the same size, trained on 1,000 images. which wins, and by how much?
4. remove the position embeddings, so the ViT sees an unordered set of patches. how much accuracy does it lose on CIFAR-10?
5. every ViT head can attend to the whole image from layer 1. after training, will layer 1's heads look locally or globally?

## prerequisites check

- [ ] you can reshape and permute image tensors in NCHW layout, and you know what a strided convolution computes (→ ch 16 · topic 01)
- [ ] you know the ResNet you're comparing against: `mlfs.cnn.ResNet` and its training recipe (→ ch 16 · topics 04–05)
- [ ] your `TransformerBlock` works without a mask, and your attention can save its weights (→ topics 01, 03)
- [ ] you've used a [CLS] token and learned position embeddings (→ topics 02, 05)

## core ideas

### idea 1: an image is a sequence of patches

a transformer eats a sequence of vectors. an image of $H \times W$ pixels with $C$ channels becomes one by cutting it into non-overlapping $p \times p$ patches, $N = HW/p^2$ of them, flattening each into a vector of $Cp^2$ numbers, and projecting each with the same linear map to the model width $d$. read row by row across the patch grid, these are the tokens.

the choice of $p$ sets the sequence length, and so the cost of attention: a 224×224 image in 16×16 patches is $14 \times 14 = 196$ tokens (the original ViT-B/16, "an image is worth 16×16 words"), in 8×8 patches 784 tokens, 16 times the attention cost. for CIFAR's 32×32 images, 4×4 patches give 64 tokens.

the projection applied to every patch is a linear map with weight $d \times Cp^2$. that's exactly a convolution with $d$ output channels, a $p \times p$ kernel and **stride $p$**: the kernel visits each patch once and never overlaps. writing it as `nn.Conv2d(C, d, kernel_size=p, stride=p)` gives you the patch extraction and the projection in one fast operation; flattening the output's spatial grid row by row gives the tokens in the same order as explicit patchifying. the test checks the two against each other, which pins down the flattening order: each patch flattens as (channel, row, column), the order a conv weight $(d, C, p, p)$ flattens in.

> **spec.** `patchify(x, patch_size)`: `(B, C, H, W)` → `(B, N, C p^2)`, patches in row-major order over the grid, each flattened in (channel, row, column) order; `ValueError` if $H$ or $W$ isn't divisible by $p$. `PatchEmbedding(img_size, patch_size, in_channels, d_model)` has `proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)`, an attribute `n_patches` $= (\text{img\_size}/p)^2$, and raises `ValueError` if `img_size` isn't divisible by `patch_size`. `forward(x)`: `(B, C, H, W)` → `(B, n_patches, d_model)`, equal to `patchify(x, p) @ proj.weight.reshape(d, -1).T + proj.bias`.

> **🔮 predict:** halving the patch size from 8 to 4 on a 32×32 image: how many tokens before and after, and by what factor does the cost of the attention scores grow?

> **▶ build: stage 1.** implement `patchify` and `PatchEmbedding`.

### idea 2: the vision transformer

the rest is BERT with a classification head. prepend a learned **[CLS] token** to the patch tokens, add a learned **position embedding** (one vector per token, $N + 1$ in all), run a stack of pre-LN transformer blocks with no mask (every patch sees every other), apply a final layer norm, and classify from the [CLS] token's output. the alternative, **mean pooling** over the patch tokens, works as well or better for small models and needs no [CLS] token; both are common.

the ViT's position embeddings are learned, and nothing tells them that patch 9 is below patch 1 in an 8-wide grid; they have to discover the 2-D layout from data. Dosovitskiy et al. found that after training, each position's embedding is most similar to those in the same row and column: the layout is learned. they also found that hand-designed 2-D or relative encodings help little, and, more surprisingly, that removing positions entirely costs only a few points of accuracy (lab, experiment 2).

what the ViT lacks is the convolution's two priors. **locality**: a convolution's first layer sees 3×3 pixels; a ViT's first layer can attend from any patch to any other, and must learn that neighbours usually matter most. **translation equivariance**: a convolution applies the same filter everywhere; a ViT's patch embedding is shared across positions, but everything after it can treat positions differently (through the position embeddings), so "an eye at the top left" and "an eye at the bottom right" must be learned separately unless the data teaches the model to generalize. priors are a substitute for data: with little data they're invaluable; with a lot, they can become a ceiling.

> **spec.** `ViT(img_size=32, patch_size=4, in_channels=3, num_classes=10, *, d_model=192, n_layers=6, n_heads=3, d_ff=None, dropout=0.0, pool='cls')` has `patch_embed` (your `PatchEmbedding`), `cls_token` (a `(1, 1, d)` parameter if `pool='cls'`, else `None`), `pos_emb` (a `(1, N + 1, d)` parameter, `(1, N, d)` without [CLS]; [CLS] and positions initialized from a truncated normal with std 0.02), dropout with rate `dropout` on the tokens after the positions are added, `blocks` (pre-LN `TransformerBlock`s, no mask), `norm` (final LayerNorm) and `head` (`nn.Linear(d, num_classes)`, or `nn.Identity()` if `num_classes == 0`); `pool` is `'cls'` or `'mean'` (else `ValueError`). `forward_features(x)` → tokens after the final norm, `(B, N (+1), d)`, [CLS] first; `forward(x)` → `head` of the [CLS] token or of the mean of the tokens.

> **🔮 predict:** with the position embeddings fixed at zero, the ViT gives the same output for an image and for the same image with its patches shuffled. why, exactly? (the test checks it.)

> **▶ build: stage 2.** implement `ViT`.

### idea 3: how far does each head look?

a convolution's receptive field is fixed by its architecture; a ViT's is learned, separately for every head. to see what it learned, Dosovitskiy et al. measured the **mean attention distance**: for each head, the average over queries of $\sum_k a_{qk}\, \|\text{pos}_q - \text{pos}_k\|$, the distance in pixels between a patch and the patches it attends to, weighted by attention. it's the attention analogue of a receptive field. at initialization, attention is nearly uniform, and the distance is the average distance between two patches of the image (16.5 pixels for a 32×32 image in 4×4 patches). after training, a head that attends to its immediate neighbours scores 4–6 pixels, a head that looks everywhere scores 15 or more.

to compute it, turn on your attention modules' `save_weights`, run a batch, and read each block's weights. use only patch-to-patch attention: drop the [CLS] row and column and renormalize each row over the patches. distances are between patch positions (grid coordinates times $p$), so a neighbouring patch is $p$ pixels away.

> **spec.** `attention_distance(model, x)`: `model` a `ViT`, `x` a batch of images. with `save_weights` turned on in every block's `attn`, run the model in eval mode without gradients; for every layer and head, average over images and patch queries the attention-weighted Euclidean distance (in pixels, between patch grid positions times `patch_size`) to the patch keys, with the [CLS] query and key removed and each row renormalized over the patches. returns a CPU tensor `(n_layers, n_heads)`. restores the model's mode and turns `save_weights` off.

> **▶ build: stage 3.** implement `attention_distance`.

## worked example

**the default `ViT()`**: 32×32×3 images, 4×4 patches, $d = 192$, 6 layers, 3 heads.

| part | parameters |
|---|---|
| patch embedding: conv $192 \times 3 \times 4 \times 4$ + bias | $9{,}216 + 192 = 9{,}408$ |
| [CLS] token | 192 |
| positions: $65 \times 192$ (64 patches + [CLS]) | 12,480 |
| 6 blocks of $4(d^2 + d) + 8d^2 + 5d + 4d$ (topic 03) | $6 \times 444{,}864 = 2{,}669{,}184$ |
| final norm, head $192 \times 10 + 10$ | $384 + 1{,}930$ |
| **total** | **2,693,578** |

97% of the parameters are in the blocks, which don't know the input is an image. the same formula with $224 \times 224$ images, 16×16 patches, $d = 768$, 12 layers and 1,000 classes gives ViT-B/16's 86.6M. your code must reproduce the first number.

**uniform attention distance** on a 4×4 grid of 4-pixel patches (a 16×16 image): the average Euclidean distance between two patch centres (including a patch and itself) is 8.03 pixels; on CIFAR's 8×8 grid it's 16.5. a model whose attention ignores content scores exactly that, which the test uses.

## on paper

1. **the patch embedding is a linear map on patches.** show that a convolution with kernel size $p$ and stride $p$ applied to a $C \times H \times W$ image computes, at output position $(r, c)$, $W_{\text{flat}} \cdot \operatorname{vec}(\text{patch}_{r,c}) + b$, and give the shape of $W_{\text{flat}}$.

<details><summary>solution</summary>

a cross-correlation at output $(r, c)$ with stride $p$ reads input pixels $(\cdot, rp + i, cp + j)$ for $i, j < p$: exactly patch $(r, c)$, and different outputs read disjoint patches. its value for output channel $o$ is $\sum_{c', i, j} W[o, c', i, j]\, x[c', rp + i, cp + j] + b_o$, a dot product of the flattened weight with the patch flattened in the same $(c', i, j)$ order. stacking over $o$: $W_{\text{flat}} \in \mathbb{R}^{d \times Cp^2}$, the conv weight reshaped. every patch uses the same $W_{\text{flat}}$, which is the one piece of weight sharing a ViT has.

</details>

2. **attention cost against patch size.** for a 224×224 image, compare the number of tokens and the cost of one attention layer's score matrix ($N^2 d$) for $p = 32, 16, 8$. why did the original paper's best models use small patches anyway?

<details><summary>solution</summary>

$N = (224/p)^2$: 49, 196, 784 tokens. the score cost scales as $N^2$: $2{,}401d$, $38{,}416d$, $614{,}656d$, a factor 16 per halving of $p$. the MLPs and projections scale as $N d^2$, only 4× per halving. smaller patches give the model finer detail (each token is a smaller region) and more tokens to compute with, and accuracy improved with smaller patches at a known compute price; the paper's naming (ViT-B/16, ViT-L/14) reports it.

</details>

3. **why positions can be removed cheaply.** without position embeddings, show that the ViT's output is invariant to any permutation of the patches (with either pooling). what kinds of image features survive this, and which don't?

<details><summary>solution</summary>

the patch embedding acts on each patch separately, so permuting patches permutes the tokens. the blocks are permutation-equivariant without a mask (topic 01, on paper 3), so the final tokens are permuted the same way. the [CLS] token sits outside the permutation and attends over all tokens with weights that are permuted along with them, so its output is unchanged; mean pooling is a symmetric function, also unchanged. so the model sees a *set* of patches. features local to a patch (colours, textures, small parts like a wheel or an eye, at 4×4 pixels) survive; arrangements (the sky above the ground, the shape of the outline) don't. on CIFAR-10, the set of local features is already quite informative.

</details>

## experiment

the lab trains your ViT and your chapter-16 ResNet, of about the same size and with the same recipe, on 1,000 and 5,000 CIFAR-10 images (20,000 behind a button). then it trains the ViT once more without position embeddings. finally it measures the attention distance of every head of the trained ViT, layer by layer, against an untrained one.

## check your understanding

1. why does a ViT need position embeddings at all, when a CNN doesn't have them?

<details><summary>answer</summary>

attention is permutation-equivariant (topic 01), so without positions the ViT sees a set of patches. a CNN's convolutions are tied to the pixel grid: a filter's output at a location depends on that location's neighbourhood, so spatial arrangement is built into the computation (and zero padding at the borders even leaks absolute position).

</details>

2. a ViT trained on 32×32 images is applied to 64×64 images with the same patch size. what breaks, and what's the usual fix?

<details><summary>answer</summary>

there are now 256 patches instead of 64, but only 65 learned position embeddings. the standard fix is to interpolate the position embeddings: reshape them to their 8×8 grid, resize it to 16×16 with bilinear interpolation, and fine-tune briefly at the new resolution (as the ViT paper does for high-resolution fine-tuning).

</details>

3. why might mean pooling work better than [CLS] pooling for a small ViT trained from scratch?

<details><summary>answer</summary>

mean pooling sends gradient from the classification loss to every patch token directly, while [CLS] pooling relies on attention learning to collect the information into one token, an extra thing to learn from limited data. with big pretraining both work; for small models mean pooling is often slightly better and simpler.

</details>

4. the ViT has 97% of its parameters in blocks that "don't know the input is an image". in what sense is that a strength?

<details><summary>answer</summary>

the same architecture (and code, and hardware optimizations, and scaling behaviour) works for text, images, audio and their combinations, so one model can take patches and words in the same sequence (chapter 25's multimodal models). the image-specific part is a thin tokenizer at the front. what the model knows about images, it learned; with enough data that turns out to be more than what was built into CNNs.

</details>

5. Dosovitskiy et al. report that a ViT pretrained on 300 million images beats ResNets. what does the lab's 20,000-image result let you conclude about that claim?

<details><summary>answer</summary>

nothing directly: the lab tests the small-data regime, where the ResNet's priors should win, and that's consistent with the paper (whose ViT also lost on ImageNet alone). a gap that doesn't shrink at this budget says nothing against the trend either, and extrapolating from 20,000 images to 300 million is exactly the kind of claim chapter 19's scaling laws exist to make carefully.

</details>

## common bugs

- **patchify with a bare `reshape`.** `x.reshape(B, N, C*p*p)` mixes pixels from different patches; you need the reshape to $(B, C, H/p, p, W/p, p)$, a permute, then a reshape.
- **flattening the conv output channels-first.** `proj(x).flatten(2)` is $(B, d, N)$; forgetting the `.transpose(1, 2)` gives $d$ "tokens" of width $N$.
- **positions for the wrong number of tokens.** `pos_emb` sized $N$ while a [CLS] token is prepended: either a shape error, or (with broadcasting tricks) the [CLS] token getting patch 0's position.
- **a causal mask.** the ViT is an encoder; with `is_causal=True` the first patch can see only itself and the [CLS] token (first) sees nothing else, so [CLS] pooling learns nothing useful.
- **forgetting to turn `save_weights` off.** every later forward pass stores attention maps, which costs memory and holds references to old tensors.
- **comparing a ViT and a ResNet with different recipes.** ViTs need stronger regularization and augmentation than ResNets; a fair comparison uses the same recipe for both, or tunes each, and says which.

## going deeper

- Dosovitskiy et al. (2021), "An image is worth 16x16 words: transformers for image recognition at scale", *ICLR*. section 3.1, figures 3–4 (pretraining data size), figure 7 (attention distance), appendix D.4 (position embeddings). [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron, Cord, Douze, Massa, Sablayrolles & Jégou (2021), "Training data-efficient image transformers & distillation through attention" (DeiT), *ICML*. ViTs on ImageNet alone, with strong augmentation. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Beyer, Zhai & Kolesnikov (2022), "Better plain ViT baselines for ImageNet-1k". mean pooling, simple recipes. [arXiv:2205.01580](https://arxiv.org/abs/2205.01580)
- Raghu, Unterthiner, Kornblith, Zhang & Dosovitskiy (2021), "Do vision transformers see like convolutional neural networks?", *NeurIPS*. [arXiv:2108.08810](https://arxiv.org/abs/2108.08810)
- Liu, Mao, Wu, Feichtenhofer, Darrell & Xie (2022), "A ConvNet for the 2020s" (ConvNeXt), *CVPR*. the CNN's answer, from chapter 16. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

## connections

- ← builds on: convolutions, ResNets and the CIFAR recipe (ch 16), attention and its permutation equivariance (topic 01), learned positions (02), the block (03), [CLS] pooling (05).
- → used in: the R1 milestone (reproducing a ViT data-scaling result is a candidate); patchified latents in chapter 22's diffusion transformer; MAE and CLIP image encoders in chapter 25; attention-map attribution in chapter 28.
