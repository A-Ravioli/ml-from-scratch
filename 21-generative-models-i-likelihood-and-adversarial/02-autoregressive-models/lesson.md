# autoregressive models

## why this matters

you want a model that assigns an exact probability to any image, so that you can grade it by test likelihood (topic 01), and that can also draw new images. a joint distribution over 784 binary pixels has $2^{784}$ outcomes; no table can hold it, and most ways of parameterizing it give you either probabilities you can't normalize or samples you can't draw. the **chain rule of probability** dissolves the problem:

$$p(x) = p(x_1)\,p(x_2 \mid x_1)\,p(x_3 \mid x_1, x_2) \cdots p(x_D \mid x_{<D}) = \prod_{i=1}^{D} p(x_i \mid x_{<i}).$$

no approximation, any distribution can be written this way. if a network outputs each conditional as a normalized distribution (a sigmoid for a bit, a softmax over 256 levels for a pixel), the product is automatically a normalized joint, the log-likelihood is a sum of $D$ cross-entropies, and sampling is $D$ easy steps: draw $x_1$, then $x_2$ given it, and so on. that's the whole idea behind **autoregressive models**, from NADE and MADE through PixelCNN and WaveNet to every GPT (chapter 18).

the engineering problem is speed. computing $D$ conditionals with $D$ separate networks would be hopeless. the trick of this topic is **masking**: one network computes all $D$ conditionals in a single forward pass, with connections removed so that output $i$ cannot see inputs $i, i+1, \dots$. you'll build masked linear layers and MADE for vectors, masked convolutions and PixelCNN for images, and measure the price: training and likelihood take one pass, but sampling takes $D$.

## warm-up: predict first

1. an autoregressive model computes the likelihood of a 28 × 28 image. how many forward passes does it need? how many to *sample* one?
2. a fully connected layer feeds output $i$ from every input. what's the simplest way to stop output $i$ from seeing inputs $\ge i$, in a single layer?
3. with two hidden layers, can you mask each layer independently and still guarantee output $i$ never sees input $i$?
4. a PixelCNN predicts pixel $(r, c)$ from the pixels above it and to its left. after stacking many masked $3 \times 3$ convolutions, does it see *every* earlier pixel?
5. the chain rule holds for any ordering of the pixels. will a model trained in a random pixel order reach the same likelihood as one trained in raster order?

## prerequisites check

- [ ] conditional probability and the chain rule (→ ch 04 · topic 01)
- [ ] cross-entropy as negative log-likelihood, sigmoid and softmax outputs (→ ch 08 · topic 04; ch 05 · topic 05)
- [ ] convolutions, padding and receptive fields (→ ch 16 · topics 01–02); residual connections (→ ch 16 · topic 04)
- [ ] bits per dimension (→ topic 01)

## core ideas

### idea 1: masked layers

the chain rule needs a function $f: \mathbb{R}^D \to \mathbb{R}^D$ whose $i$-th output depends only on $x_{<i}$. in other words, its Jacobian must be **strictly lower triangular**: $\partial f_i / \partial x_j = 0$ for every $j \ge i$. for a single linear layer this is easy: multiply the weight matrix elementwise by a fixed 0/1 **mask** that is 1 only below the diagonal. the masked entries contribute nothing to the output, and because the mask sits inside the forward pass, they also receive zero gradient, so training never brings them back.

that last point matters. zeroing the forbidden weights once at initialization is not enough: the first optimizer step would make them nonzero again (their gradients aren't zero unless the mask is applied in the forward pass). the mask must be applied **every forward pass**, as `weight * mask`, and stored as a **buffer** (`register_buffer`) so it moves with `.to(device)` and is saved with the model, but is never trained.

> **spec · stage 1.** `MaskedLinear(in_features, out_features, bias=True)`, an `nn.Linear` subclass with a buffer `mask` of ones, shape `(out_features, in_features)`. `set_mask(mask)` copies a 0/1 mask (tensor or ndarray) into the buffer; `forward(x)` computes `F.linear(x, weight * mask, bias)`. this is `mlfs.generative.MaskedLinear`.

> **▶ build: stage 1.** implement `MaskedLinear` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: MADE

a single masked layer is a linear model, too weak to be useful. stack several with nonlinearities, and you can't mask each layer as "lower triangular" anymore: hidden units aren't ordered like the inputs. MADE (Germain, Gregor, Murray & Larochelle, 2015) solves this with **degrees**. every unit gets an integer degree that says "I may depend on inputs $1, \dots, m$" (1-indexed):

- input $x_d$ has degree $m(x_d) = d$, for $d = 1, \dots, D$.
- each hidden unit $k$ gets a degree $m(k)$ between 1 and $D - 1$ (a hidden unit of degree $D$ would see everything, which no output may use).
- a connection from unit $j$ in one layer to unit $k$ in the next exists iff $m(k) \ge m(j)$: a unit may listen to units that see no more than it does.
- the **output** for dimension $d$ (degree $d$) connects to hidden unit $k$ iff $d > m(k)$, **strictly**.

follow any path from input $x_j$ to output $d$: degrees can only stay equal or grow along hidden layers, so every hidden unit on the path has degree $\ge j$, and the last one has degree $< d$. so a path exists only if $j < d$, which is exactly the autoregressive property (on paper 1). nothing about it depends on the weights: it holds before, during and after training.

the hidden degrees are drawn at random. MADE draws each layer's degrees uniformly between the smallest degree in the previous layer and $D - 1$, so that no unit is created that can't receive any input. with enough units per layer, every allowed connection is present.

for discrete data with more than two values, or for a Gaussian per dimension, you want several numbers per conditional. **`n_outputs_per_dim`** repeats the output degrees: output unit $o \cdot D + d$ has degree $d + 1$, and the output is reshaped to $(B, n_\text{out}, D)$.

> **spec · stage 2.** `MADE(in_dim, hidden_dims, *, n_outputs_per_dim=1, seed=0)`, an `nn.Module` built from `MaskedLinear` layers with ReLUs between them. input degrees $1..D$; each hidden layer's degrees drawn by `np.random.default_rng(seed)` uniformly from the integers in $[\min(\text{previous degrees}), D - 1]$ (so the same seed gives the same masks); hidden masks $m(k) \ge m(j)$; output masks $(d + 1) > m(k)$ for output unit $o \cdot D + d$. `forward(x)`: `(B, D)` → `(B, n_outputs_per_dim, D)`, entry `[:, o, d]` from output unit $o \cdot D + d$, depending only on `x[:, :d]`. `D = 1` must work. this is `mlfs.generative.MADE`.

> **▶ build: stage 2.** implement `MADE`.

> **🔮 predict:** output 0 of a MADE (the first dimension) is connected to no input at all. what does it compute, and does it change from one input image to the next?

### idea 3: one pass to score, $D$ passes to sample

for binary data, a MADE with one output per dimension produces a logit $\ell_i$ for each $p(x_i = 1 \mid x_{<i}) = \sigma(\ell_i)$. the exact negative log-likelihood is a sum of binary cross-entropies:

$$-\log p(x) = -\sum_{i=1}^{D} \Big[x_i \log \sigma(\ell_i) + (1 - x_i)\log\big(1 - \sigma(\ell_i)\big)\Big],$$

all computed in **one** forward pass on the observed $x$: each $\ell_i$ only looks at the observed $x_{<i}$, which is exactly what the chain rule asks for. training is ordinary maximum likelihood, as parallel as any classifier.

**sampling** can't be parallel. $x_2$'s conditional needs the *sampled* $x_1$, which doesn't exist until you've drawn it. so: start from zeros, run the network, draw $x_1$ from $\sigma(\ell_1)$, write it in, run the network again (now $\ell_2$ sees the real $x_1$; the zeros at positions $\ge 2$ are never looked at), draw $x_2$, and so on: $D$ forward passes per batch of samples. for a 28 × 28 image that's 784 passes, each over the whole image. this asymmetry (fast likelihood, slow sampling) is the signature of autoregressive models. transformer language models pay it too, and cache intermediate results to make each step cheaper (chapter 18).

the **ordering** is a modeling choice. the chain rule holds for every ordering of the dimensions, so in principle any order gives the same family of distributions. in practice, a network has to *learn* each conditional, and some orderings make the conditionals much easier than others. the lab measures how much it matters.

> **spec · stage 3.**
> - `made_nll(model, x)`: binary `x` `(B, D)` (float 0/1) → exact NLL in nats `(B,)` under a MADE whose `[:, 0, :]` outputs are Bernoulli logits.
> - `made_sample(model, n, *, generator=None)`: `n` binary samples `(n, D)` (float), filled in one dimension per forward pass (exactly `D` passes, under `no_grad`), $x_i \sim \text{Bernoulli}(\sigma(\ell_i))$ drawn with `torch.bernoulli(..., generator=generator)`. the model's dimension is `model.in_dim`.

> **▶ build: stage 3.** implement `made_nll` and `made_sample`.

### idea 4: masked convolutions

images have more structure than a flat vector: nearby pixels predict each other best, and the same local rule applies everywhere. PixelCNN (van den Oord, Kalchbrenner & Kavukcuoglu, 2016) keeps the autoregressive idea and uses **convolutions** with masked kernels. order the pixels in **raster scan**: row by row, left to right. pixel $(r, c)$ may depend on every pixel in rows above, and on pixels to its left in its own row.

a convolution's output at $(r, c)$ is a weighted sum over a $k \times k$ window centred on $(r, c)$. to make it causal, zero the kernel at the centre row's right half and at every row below. there are two variants:

- **type A** also zeroes the **centre**: the output at $(r, c)$ cannot see input $(r, c)$. used only in the **first** layer, where the input is the image itself, because a pixel must not predict itself.
- **type B** keeps the centre. used in every later layer: by then the feature at $(r, c)$ already summarizes only pixels *before* $(r, c)$ (thanks to layer 1), so it's safe for the next layer to look at it.

for a $5 \times 5$ kernel, the spatial masks are

$$A = \begin{pmatrix} 1&1&1&1&1\\ 1&1&1&1&1\\ 1&1&0&0&0\\ 0&0&0&0&0\\ 0&0&0&0&0 \end{pmatrix}, \qquad B = \begin{pmatrix} 1&1&1&1&1\\ 1&1&1&1&1\\ 1&1&1&0&0\\ 0&0&0&0&0\\ 0&0&0&0&0 \end{pmatrix},$$

the same for every pair of input and output channels. padding $k/2$ keeps the image size. a stack of one type-A layer followed by any number of type-B layers is causal.

> **spec · stage 4.** `MaskedConv2d(mask_type, in_channels, out_channels, kernel_size)`, an `nn.Conv2d` subclass with `padding=kernel_size // 2` and a buffer `mask` of shape `(out, in, k, k)` as above (`ValueError` for a type other than `"A"` or `"B"`); store `self.mask_type`; `forward` convolves with `weight * mask`. this is `mlfs.generative.MaskedConv2d`.

> **▶ build: stage 4.** implement `MaskedConv2d`.

### idea 5: PixelCNN

the model: one $7 \times 7$ type-A masked convolution, a stack of $3 \times 3$ type-B masked convolutions with ReLUs (residual connections, ch 16 · topic 04, help it train), and two $1 \times 1$ convolutions that output, for every pixel and channel, **a softmax over the 256 intensity levels**. a categorical output is a striking choice: it ignores that level 127 is close to 128, and has to learn that from data. van den Oord et al. found it works better than a continuous density, because it can represent any shape, including the sharp spikes at 0 and 255 that MNIST is full of.

with $C$ channels per pixel, a full PixelCNN also orders the channels (red before green before blue) inside each pixel. here, to keep the masks simple, all channels of a pixel are predicted **independently given the previous pixels**: still an exact, normalized likelihood, just a slightly weaker model. the network maps $x \in [0, 1]$ (rescaled to $[-1, 1]$ inside) to logits of shape $(B, L, C, H, W)$ with $L$ levels; the target level of a pixel value $v$ is $\operatorname{round}(v (L - 1))$, so 8-bit MNIST scaled by $1/255$ gives back the original integers, and binarized MNIST uses $L = 2$.

stacked masked convolutions have a famous flaw: a **blind spot**. a type-B $3 \times 3$ layer extends the receptive field one pixel up, left or up-left, and to the *right* only together with a step up. so in the row $k$ above the current pixel, the visible pixels reach at most $k$ columns to the right (plus the first layer's own reach), whatever the depth: a $45°$ boundary, and the triangle of pixels above and to the right of it is never seen (on paper 3). the gated PixelCNN (van den Oord et al., 2016b) fixed it by splitting the network into a vertical and a horizontal stack. the lab draws the blind spot of your model.

> **spec · stage 5.**
> - `PixelCNN(in_channels=1, *, n_levels=256, hidden_channels=64, n_layers=7)`, an `nn.Module`: a type-A first layer and only type-B masked convolutions after it (the internal design is yours; the reference uses a $7 \times 7$ type-A layer, `n_layers - 1` residual $3 \times 3$ type-B layers and a $1 \times 1$ head). `forward(x)`: `(B, C, H, W)` in $[0, 1]$ → logits `(B, n_levels, C, H, W)`, where the logits of pixel $(r, c)$ depend only on pixels before it in raster order (all channels). attributes `n_levels`, `in_channels`. `sample(n, img_size, *, generator=None, device="cpu")`: `img_size` an int (square) or `(H, W)`; exactly one forward pass per pixel in raster order, under `no_grad` and in eval mode (restored after), levels drawn with `torch.multinomial(probs, 1, generator=generator)` on CPU probabilities; returns images in $[0, 1]$ (level $/ (L - 1)$). this is `mlfs.generative.PixelCNN`.
> - `pixelcnn_nll(model, x)`: exact NLL in nats per image `(B,)`: the cross-entropy of the logits against the levels $\operatorname{round}(x (L - 1))$, summed over channels and pixels.

> **▶ build: stage 5.** implement `PixelCNN` and `pixelcnn_nll`.

> **🔮 predict:** sampling a $14 \times 14$ image takes $t$ seconds with your PixelCNN. how long does a $28 \times 28$ image take: $2t$, $4t$, or more?

## worked example

**a MADE by hand.** $D = 3$, one hidden layer of 4 units with degrees $(1, 2, 1, 2)$. the hidden mask ($m(k) \ge m(j)$, rows are hidden units, columns inputs $x_1, x_2, x_3$) and the output mask ($d > m(k)$, rows are outputs of degree 1, 2, 3):

$$M^{(1)} = \begin{pmatrix} 1&0&0\\ 1&1&0\\ 1&0&0\\ 1&1&0 \end{pmatrix}, \qquad M^{(2)} = \begin{pmatrix} 0&0&0&0\\ 1&0&1&0\\ 1&1&1&1 \end{pmatrix}.$$

their product $M^{(2)} M^{(1)} = \begin{pmatrix} 0&0&0\\ 2&0&0\\ 4&2&0 \end{pmatrix}$ counts the paths from each input to each output: strictly lower triangular. output 1 sees nothing (it learns the marginal $p(x_1)$ through its bias), output 2 sees $x_1$, output 3 sees $x_1$ and $x_2$, and $x_3$ feeds nobody.

**a likelihood by hand.** the model outputs $\sigma(\ell_1) = 0.8$, then (given $x_1 = 1$) $\sigma(\ell_2) = 0.3$, then (given $x_1 = 1, x_2 = 0$) $\sigma(\ell_3) = 0.9$. the vector $x = (1, 0, 1)$ has probability $0.8 \times 0.7 \times 0.9 = 0.504$: an NLL of $0.685$ nats, or $0.685 / (3 \ln 2) = 0.33$ bits/dim. the probabilities of all 8 vectors sum to 1 for *any* weights, a fact the checks test.

**masked kernels.** for a $3 \times 3$ kernel, type A is $\begin{pmatrix} 1&1&1\\ 1&0&0\\ 0&0&0 \end{pmatrix}$ and type B is $\begin{pmatrix} 1&1&1\\ 1&1&0\\ 0&0&0 \end{pmatrix}$: four and five live taps out of nine. the checks reproduce the $5 \times 5$ masks above.

## on paper

1. **why MADE's masks work.** prove that with MADE's degree rules, $\partial f_d / \partial x_j = 0$ whenever $j \ge d$, for any weights and any number of hidden layers.

<details><summary>solution</summary>

$f_d$ depends on $x_j$ only through paths $x_j \to h^{(1)}_{k_1} \to \dots \to h^{(L)}_{k_L} \to f_d$ whose every connection is unmasked (the derivative is a sum over such paths of products of weights and activation derivatives; a masked connection contributes a factor 0). along an unmasked path, $m(k_1) \ge j$, $m(k_{l+1}) \ge m(k_l)$, and $d > m(k_L)$. chaining: $d > m(k_L) \ge \dots \ge m(k_1) \ge j$, so $d > j$. if $j \ge d$, no unmasked path exists and the derivative is identically 0.

</details>

2. **the chain rule sums to one.** show that $\sum_{x \in \{0,1\}^D} \prod_i p(x_i \mid x_{<i}) = 1$ for any conditionals.

<details><summary>solution</summary>

sum over $x_D$ first: $\sum_{x_D} p(x_D \mid x_{<D}) = 1$ for every fixed $x_{<D}$, which removes the last factor. then sum over $x_{D-1}$, and so on: after $D$ steps the sum is 1. the argument needs nothing about the conditionals except that each is normalized, which a sigmoid (or softmax) output guarantees.

</details>

3. **the blind spot.** consider a type-A $3 \times 3$ layer followed by $n$ type-B $3 \times 3$ layers. which input pixels in the row directly above the current pixel $(r, c)$ can influence its output, and which pixels in row $r - 1$ to the right of column $c + 1$ are never seen?

<details><summary>solution</summary>

track the set of input offsets $(\Delta r, \Delta c)$ (relative to the output pixel) that a unit can see. layer A: $\{(-1, -1), (-1, 0), (-1, +1), (0, -1)\}$. each type-B layer composes with offsets $\{(-1, -1), (-1, 0), (-1, +1), (0, -1), (0, 0)\}$. reaching $(-1, c')$ for $c' > 1$ requires moving right by more than 1 in one row up, but every right step $(-1, +1)$ also moves up a row. a path to $(-1, +2)$ would need two right steps and a net change of only one row, which needs a step $(+1, \cdot)$ that no mask allows. so in row $r - 1$, only columns up to $c + 1$ are visible; in row $r - k$, columns up to $c + k$ (with enough layers): the visible region is bounded by a $45°$ diagonal, and the triangle to the upper right of it is blind no matter the depth.

</details>

4. **sampling cost.** a PixelCNN forward pass on an $H \times W$ image costs about $c\,HW$ operations. what does sampling one image cost, and how does it scale when you double the side length?

<details><summary>solution</summary>

$HW$ passes of cost $c\,HW$ each: $c\,(HW)^2$. doubling the side multiplies $HW$ by 4, so sampling costs 16 times more (density evaluation only 4 times more). caching the activations that don't change between steps (as fast PixelCNN++ and transformer KV-caches do) brings each step's cost down to the work that actually changes.

</details>

## experiment

`lab.py` uses your MADE and PixelCNN on MNIST:

1. the cost of sampling versus scoring, as the image size grows.
2. the same MADE trained on binarized digits in raster order, reversed order and a random order: does the order change the likelihood?
3. the gradient receptive field of your PixelCNN, and its blind spot.

## check your understanding

1. why does an autoregressive model give an exact, normalized likelihood without computing any normalizing constant?

<details><summary>answer</summary>

each factor $p(x_i \mid x_{<i})$ is normalized by construction (a sigmoid or softmax output), and a product of normalized conditionals is a normalized joint (on paper 2). the normalizing constant over $2^D$ or $256^D$ outcomes is never needed.

</details>

2. why must the mask be applied in every forward pass, not just at initialization?

<details><summary>answer</summary>

if the forbidden weights are zeroed once, their gradients are still nonzero (the forward pass used them as zero but the gradient of the loss with respect to each weight entry is computed regardless), so the first optimizer step brings them back, breaking the autoregressive property. multiplying by the mask inside the forward pass makes their gradient exactly zero.

</details>

3. why does the first PixelCNN layer use mask A and the others mask B?

<details><summary>answer</summary>

the first layer sees the raw image, so its output at $(r, c)$ must not include input $(r, c)$. after it, the feature at $(r, c)$ depends only on pixels before $(r, c)$, so later layers may include the centre (B) without leaking the current pixel. using A everywhere would be safe but would shrink the receptive field needlessly; using B in the first layer would let each pixel see itself and the model would learn the identity.

</details>

4. a trained MADE has a test NLL of 88 nats on binarized MNIST, a VAE has an ELBO of $-95$ nats. which is the better density model?

<details><summary>answer</summary>

MADE's 88 is an exact NLL; the VAE's 95 is an upper bound on its NLL (topic 03), so its true NLL is somewhere below 95, possibly below 88. you need a tighter estimate for the VAE (importance sampling, topic 03) before comparing. the safe conclusion is that MADE is at least as good as the VAE's bound suggests.

</details>

5. why is the categorical (256-way softmax) output a reasonable choice for 8-bit pixels, despite ignoring the order of intensities?

<details><summary>answer</summary>

it can represent any distribution over the 256 levels: multimodal, skewed, or spiked at 0 and 255 (very common in MNIST). the cost is that nearness of levels has to be learned from data and the output layer is large. PixelCNN++ replaced it with a discretized mixture of logistics, which builds in the ordering and trains faster.

</details>

6. where in an image does PixelCNN's blind spot hurt most?

<details><summary>answer</summary>

pixels just above and to the right of the current pixel (rows above, beyond the diagonal) are invisible. for strokes that run up and to the right, like the upper part of a 7 or the diagonal of a 2, the model can't see where a nearby stroke is heading, so it must guess from less context. the gated PixelCNN's two stacks remove the blind spot.

</details>

## common bugs

- **masking only at initialization.** symptom: the Jacobian test passes before training and fails after one optimizer step.
- **hidden degrees up to $D$.** symptom: units of degree $D$ that no output can use (wasted), or, with a non-strict output rule, outputs that see their own input: suspiciously low training loss and garbage samples.
- **a non-strict output mask ($d \ge m(k)$).** symptom: output $d$ sees $x_d$; the training NLL drops toward 0 while sampling produces noise.
- **type A in every layer, or type B in the first.** symptom: a tiny receptive field and poor likelihood; or a near-zero loss and broken samples.
- **sampling from the wrong pixel's logits.** symptom: samples look like shifted or repeated copies; the empirical conditionals don't match the model's.
- **levels from `x * 256` instead of `round(x * (L - 1))`.** symptom: level 256 is out of range for white pixels (an index error), or every level is off by a fraction.

## going deeper

- Larochelle & Murray (2011), "The neural autoregressive distribution estimator", *AISTATS*. NADE. [proceedings.mlr.press/v15/larochelle11a](https://proceedings.mlr.press/v15/larochelle11a.html)
- Germain, Gregor, Murray & Larochelle (2015), "MADE: masked autoencoder for distribution estimation", *ICML*. [arXiv:1502.03509](https://arxiv.org/abs/1502.03509)
- van den Oord, Kalchbrenner & Kavukcuoglu (2016), "Pixel recurrent neural networks", *ICML*. PixelRNN and PixelCNN, masks A and B. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
- van den Oord et al. (2016b), "Conditional image generation with PixelCNN decoders", *NeurIPS*. the gated PixelCNN and the blind spot. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
- Salimans, Karpathy, Chen & Kingma (2017), "PixelCNN++: improving the PixelCNN with discretized logistic mixture likelihood and other modifications", *ICLR*. [arXiv:1701.05517](https://arxiv.org/abs/1701.05517)
- Uria, Côté, Gregor, Murray & Larochelle (2016), "Neural autoregressive distribution estimation", *JMLR* 17. orderings and order-agnostic training. [arXiv:1605.02226](https://arxiv.org/abs/1605.02226)

## connections

- ← builds on: the chain rule of probability (ch 04), cross-entropy (ch 05, ch 08), convolutions and residual blocks (ch 16), bits per dimension (topic 01), character-level language models (ch 17).
- → used in: masked autoregressive flows (topic 07 builds on the same masks), the PixelCNN prior over VQ-VAE codes (topic 04), the chapter capstone (the autoregressive family), transformer language models and their KV caches (ch 18), and autoregressive audio models in the audio elective.
