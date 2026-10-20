# pooling & receptive fields

## why this matters

a $3 \times 3$ convolution sees 9 pixels. a cat on a $224 \times 224$ photo covers tens of thousands. somewhere between the first layer and the last, the network has to go from "is there an edge here?" to "is there a cat in this image?", and every output unit can only answer questions about the part of the image it can see. how big is that part? after ten $3 \times 3$ layers, a unit can depend on a $21 \times 21$ patch: still too small. so real networks also **downsample**: they shrink the feature maps with pooling or strided convolutions, and every later layer's $3 \times 3$ window then covers a much larger patch of the original image.

this topic is about that bookkeeping and about its fine print. you'll build the pooling layers (and their backward passes, on top of your `im2col`/`col2im`), the arithmetic that tells you exactly which input pixels a unit can depend on, and a measurement of which pixels it *actually* depends on. the two are not the same: in the lab, a 32-layer network whose units can see a $65 \times 65$ patch turns out to put 90% of its sensitivity in the central $27 \times 27$. and you'll test a claim you've probably heard, that convolutional networks are translation-invariant.

## warm-up: predict first

1. a $2 \times 2$ max pooling layer halves a $32 \times 32$ feature map to $16 \times 16$. how many parameters does it have?
2. you stack five $3 \times 3$ convolutions (stride 1). how wide is the patch of input one output can depend on? and if a $2 \times 2$ pooling layer comes first?
3. during backpropagation through a max pooling layer, which inputs receive a gradient?
4. a unit's receptive field is $65 \times 65$ pixels. are all 4,225 pixels equally important to it?
5. you shift an image by one pixel. does a CNN's prediction stay exactly the same?

## prerequisites check

- [ ] your `im2col`, `col2im` and `as_pair` from topic 01, and the convolution output size formula (→ topic 01)
- [ ] the backward pass of a max (a subgradient) and of a mean (→ ch 14 · topic 03)
- [ ] `torch.autograd.grad` and evaluation mode (`model.eval()`, batch norm's running statistics) (→ ch 15 · topics 02 and 07)
- [ ] the central limit theorem: sums of many independent variables look Gaussian, with variance adding up (→ ch 04 · topic 05)

## core ideas

### idea 1: pooling, and what it buys

a **pooling** layer summarizes each window of a feature map with one number. **max pooling** keeps the largest value ("is this feature present anywhere in the window?"); **average pooling** keeps the mean ("how much of it is there?"). like a convolution, it slides a $k \times k$ window with a stride, but it has no weights, and it works on each channel separately: channel $c$ of the output only sees channel $c$ of the input. the usual choice is $k = s = 2$, non-overlapping windows that halve both spatial dimensions; when `stride` is left out it defaults to the kernel size. with no padding, the output size is your `conv_output_size` with $p = 0$: windows that don't fit are dropped, so a $5 \times 5$ map pooled $2 \times 2$ becomes $2 \times 2$ and its last row and column are never read.

why downsample at all? three reasons.

- **compute and memory.** halving $H$ and $W$ divides the cost of every later layer by 4, which is what makes it affordable to double the number of channels at the same time (topic 03's VGG does exactly this).
- **receptive field.** after a stride-2 layer, adjacent units of the next layer are 2 input pixels apart, so its $3 \times 3$ windows span twice as much of the image (idea 3).
- **small-shift tolerance.** a feature that moves within a max-pooling window doesn't change the window's max, so pooling makes the output less sensitive to exactly where things are.

that last point deserves care. a convolution is **translation-equivariant**: shift the input and the output shifts by the same amount (away from the borders). a stride-$s$ layer is only equivariant to shifts by multiples of $s$: shift the input by one pixel and a $2 \times 2$ pool's output changes in a way that isn't any shift of the original output. global average pooling at the end of a network (topic 03) is **invariant** to shifts, but only to shifts of the features it receives, which the strided layers in between have already scrambled. so a CNN is *approximately* invariant to small shifts, and the experiment in the lab measures how approximately.

> **🔮 predict:** a small CNN with three stride-2 steps and global average pooling classifies MNIST digits at about 94%. you move every test digit by one pixel diagonally. what fraction of its predictions change?

implementation, using your topic 01 code: reshape $x$ from $(N, C, H, W)$ to $(N C, 1, H, W)$ so every channel is its own one-channel image, run `im2col` with the pooling kernel and stride to get $(N C, k_h k_w, L)$, take the max (or mean) over axis 1, and reshape to $(N, C, H_\text{out}, W_\text{out})$.

> **▶ build: stage 1.** implement `max_pool2d(x, kernel_size, stride=None)` and `avg_pool2d` with `im2col` (imported from your topic 01 code via `mlfs.cnn`).

### idea 2: where pooling sends its gradients

average pooling is linear: each output is $\frac{1}{k_h k_w}$ times the sum of its window, so each window's upstream gradient is shared equally, $\frac{1}{k_h k_w}$ to each of its pixels. max pooling is piecewise linear: the output equals one input (the window's maximum) and doesn't depend on the others at all, so the whole gradient goes to the **winner** and the rest of the window gets zero. (at a tie the max isn't differentiable; any convex combination of the tied inputs is a valid subgradient, and libraries pick one, here the first in window order.)

both fit the column machinery of topic 01. build the gradient with respect to the columns, then send it back to the pixels with `col2im`, which adds up contributions from overlapping windows:

- **max**: `cols.argmax(axis=1)` picks each window's winner; put `dout` there and zeros elsewhere (`np.put_along_axis` does it in one call), then `col2im`.
- **average**: broadcast `dout / (kh * kw)` to every row of the columns, then `col2im`.

with overlapping windows ($k > s$), a pixel that wins several windows collects several gradients, and with $k < s$ some pixels are never read and get exactly zero. a pixel that wins no window gets nothing either, which is a real effect in training: in a $2 \times 2$ max pool, three quarters of the activations get no gradient on any given step.

> **▶ build: stage 2.** implement `max_pool2d_backward(dout, x, kernel_size, stride=None)` and `avg_pool2d_backward(dout, x_shape, kernel_size, stride=None)`.

### idea 3: receptive field arithmetic

the **receptive field** of a unit is the set of input pixels that can influence it. for a stack of convolutions and poolings it's a square (in each axis, an interval), and its width follows from two numbers per layer, tracked from the input upwards:

- $r_l$, the receptive field width of layer $l$'s units, in input pixels;
- $j_l$, the **jump**: the distance, in input pixels, between two adjacent units of layer $l$.

the input itself has $r_0 = 1$ and $j_0 = 1$. layer $l$, with kernel $k_l$, stride $s_l$ and dilation $d_l$, combines $k_l$ units of layer $l - 1$ spaced $d_l$ apart. those units are $d_l j_{l-1}$ input pixels apart, so the first and last tap's centres are $d_l (k_l - 1) j_{l-1}$ pixels apart, and each tap itself sees $r_{l-1}$ pixels:

$$r_l = r_{l-1} + d_l (k_l - 1)\, j_{l-1}, \qquad j_l = j_{l-1}\, s_l.$$

the jump is updated *after* the field, because the stride of layer $l$ affects how far apart *its* outputs are, not its inputs. unrolled, $r_L = 1 + \sum_l d_l (k_l - 1) \prod_{m < l} s_m$: every layer adds its kernel's span, multiplied by the product of all strides *below* it. that formula explains the design of most CNNs:

- stacked stride-1 $3 \times 3$ convolutions grow the field by 2 per layer: linearly, and slowly.
- every stride-2 layer doubles what all *later* layers add: downsampling early buys a large field cheaply, at the price of resolution.
- dilations $1, 2, 4, \dots$ grow it exponentially without losing resolution (topic 01, on paper 4).

padding doesn't change the field's width (it adds zeros, which carry no information); it only moves where the field sits relative to the image. for networks built from `nn.Conv2d` and pooling layers, `rf_layers(model)` collects each layer's $(k, s, d)$ in order, so `receptive_field(rf_layers(model))` reads the field off a real model.

> **🔮 predict:** a stack of 32 stride-1 $3 \times 3$ convolutions has a $65 \times 65$ receptive field. picture how strongly the centre output depends on each of those 4,225 input pixels. is the map flat, a pyramid, a bell?

> **▶ build: stage 3.** implement `receptive_field(layers)` (layers given as `(kernel_size, stride, dilation)`) and `rf_layers(model)`.

### idea 4: the effective receptive field

the receptive field says which pixels *can* matter. how much each one *does* matter is a different question, and the natural measurement is a gradient: the **effective receptive field** (Luo et al., 2016) of a unit is the map $|\partial y / \partial x_{ij}|$ over input pixels. for a linear stack of convolutions you can compute it by hand: the centre output's gradient with respect to the input is the kernel convolved with itself $L$ times (each layer spreads the gradient by one more kernel). with all-ones $3 \times 3$ kernels, one layer gives a flat $3 \times 3$ box, two layers give a pyramid with weights $1, 2, 3, 2, 1$ per axis (the number of paths from each pixel to the centre), and $L$ layers give the distribution of a **sum of $L$ independent steps** of $-1, 0, +1$. by the central limit theorem that's a Gaussian whose standard deviation grows like $\sqrt{L}$, while the theoretical field grows like $L$. the effective field is a shrinking fraction of the theoretical one: most of the field is technically visible and practically ignored.

with random weights and ReLUs the paths no longer have equal weight, but the same averaging argument applies, and Luo et al. found Gaussian-shaped effective fields in trained networks too, with training widening them somewhat.

to measure it: feed a batch of random Gaussian inputs (`n_samples` of them, from a seeded `torch.Generator`, in the model's dtype and on its device), run the model in **eval mode** (so batch norm uses its running statistics and every sample is independent), take the output at the spatial centre, sum it over channels and samples, and differentiate with respect to the *input only* with `torch.autograd.grad(out, x)`, which leaves no gradients behind on the model's parameters. take absolute values (negative sensitivity is sensitivity too), sum over samples and input channels, and normalize the map to sum to 1. restore the model's mode afterwards. to summarize a map with one number, `erf_radius(erf, mass)` returns the smallest $r$ such that the $(2r + 1) \times (2r + 1)$ square around the centre pixel $(H/\!/2, W/\!/2)$ holds at least `mass` of it (use an odd input size so that the centre is unambiguous).

> **▶ build: stage 4.** implement `effective_receptive_field(model, input_size, *, in_channels=1, n_samples=8, seed=0)` and `erf_radius(erf, mass=0.9)`.

## worked example

**pooling by hand.** take the single-channel image

$$x = \begin{pmatrix} 1 & 3 & 2 & 0 \\ 4 & 2 & 1 & 5 \\ 0 & 1 & 3 & 3 \\ 2 & 6 & 1 & 1 \end{pmatrix}.$$

$2 \times 2$ max pooling (stride 2) gives $\begin{pmatrix} 4 & 5 \\ 6 & 3 \end{pmatrix}$, average pooling $\begin{pmatrix} 2.5 & 2 \\ 2.25 & 2 \end{pmatrix}$. with $3 \times 3$ windows and stride 1 the windows overlap and max pooling gives $\begin{pmatrix} 4 & 5 \\ 6 & 6 \end{pmatrix}$: the 6 wins two windows.

**backward.** with an upstream gradient of 1 for every output, max pooling's input gradient is 1 at each block's winner (the 4, the 5, the 6, and the *first* 3 of the bottom-right block, at row 2, column 2) and 0 elsewhere. with the overlapping $3 \times 3$ windows, the 6 collects a gradient of 2. average pooling gives $\frac14$ to every pixel.

**a receptive field.** the stack conv3, conv3, pool2, conv3, conv3, pool2 (all convs stride 1):

| layer | $(k, s, d)$ | $r$ | $j$ |
|---|---|---|---|
| input | | 1 | 1 |
| conv | (3, 1, 1) | $1 + 2 \cdot 1 = 3$ | 1 |
| conv | (3, 1, 1) | 5 | 1 |
| pool | (2, 2, 1) | $5 + 1 \cdot 1 = 6$ | 2 |
| conv | (3, 1, 1) | $6 + 2 \cdot 2 = 10$ | 2 |
| conv | (3, 1, 1) | 14 | 2 |
| pool | (2, 2, 1) | $14 + 1 \cdot 2 = 16$ | 4 |

so each unit after the second pooling sees a $16 \times 16$ patch, and neighbouring units are 4 pixels apart. order matters: pool-then-conv, $[(2, 2, 1), (3, 1, 1)]$, gives $(6, 2)$, while conv-then-pool gives $(4, 2)$. ResNet's ImageNet stem, a $7 \times 7$ stride-2 conv then a $3 \times 3$ stride-2 max pool, gives $(11, 4)$ after just two layers.

**an effective receptive field.** two all-ones $3 \times 3$ convolutions (no nonlinearity): the centre output is reached from the centre pixel by $3 \times 3 = 9$ paths and from a corner of its $5 \times 5$ field by 1, so the normalized map is $\frac{1}{81}\,(1, 2, 3, 2, 1)^\top (1, 2, 3, 2, 1)$. its central $3 \times 3$ holds $\frac{(2 + 3 + 2)^2}{81} = \frac{49}{81} \approx 0.60$ of the mass, so its 50% radius is 1 and its 90% radius is 2.

the tests check every number in this section.

## on paper

1. **the receptive field recursion, unrolled.** prove $r_L = 1 + \sum_{l=1}^{L} d_l (k_l - 1) \prod_{m=1}^{l-1} s_m$ from the recursion, and use it to find the field of $L$ stride-1 $3 \times 3$ convs, and of the same stack with a stride-2 layer at the very beginning.

<details><summary>solution</summary>

by induction, $j_{l-1} = \prod_{m < l} s_m$, and $r_l - r_{l-1} = d_l (k_l - 1) j_{l-1}$; summing the differences from $r_0 = 1$ gives the formula. $L$ stride-1 $3 \times 3$ convs: $r = 1 + 2L$. with a stride-2 $3 \times 3$ conv first and $L - 1$ stride-1 convs after it: $r = 1 + 2 + 2 \cdot 2(L - 1) = 4L - 1$, nearly twice as wide, at the cost of a quarter of the resolution.

</details>

2. **the effective field of a uniform stack.** in 1-D, $L$ layers of the kernel $(1, 1, 1)$ with no nonlinearity. show that the centre output's gradient with respect to input pixel $t$ (relative to the centre) is proportional to the number of ways to write $t$ as a sum of $L$ steps in $\{-1, 0, 1\}$, and find the standard deviation of the normalized map. for $L = 32$, compare a 90% central interval with the theoretical radius 32.

<details><summary>solution</summary>

the gradient flows back through $L$ layers, each spreading it to offsets $-1, 0, +1$ with weight 1, so the weight of offset $t$ is the number of paths, i.e. of step sequences summing to $t$: the $L$-fold convolution of $(1, 1, 1)$. normalized, it's the distribution of a sum of $L$ independent uniform $\{-1, 0, 1\}$ steps, each with variance $\frac{2}{3}$, so the standard deviation is $\sqrt{2L/3}$: 4.6 for $L = 32$. by the CLT, 90% of the mass lies within about $1.645 \times 4.6 \approx 7.6$ pixels of the centre, a quarter of the theoretical radius of 32, and the ratio shrinks like $1/\sqrt{L}$.

</details>

3. **two small kernels or one big one.** compare one $5 \times 5$ convolution with two stacked $3 \times 3$ convolutions, and one $7 \times 7$ with three $3 \times 3$, all $C \to C$ channels: receptive field, number of weights, and number of nonlinearities.

<details><summary>solution</summary>

two $3 \times 3$ layers have $r = 5$, like one $5 \times 5$, with $2 \cdot 9 C^2 = 18 C^2$ weights instead of $25 C^2$, and two ReLUs instead of one. three $3 \times 3$ layers reach $r = 7$ with $27 C^2$ weights instead of $49 C^2$ (45% fewer), and three nonlinearities. this is the argument of the VGG paper (topic 03) for building networks only from $3 \times 3$ convolutions.

</details>

4. **a shift by one pixel.** a 1-D signal $x = (0, 0, 1, 0, 0, 0)$ goes through max pooling with $k = s = 2$. compute the output for $x$ and for $x$ shifted right by one and by two pixels. which shifts produce a shifted output?

<details><summary>solution</summary>

$x$: windows $(0, 0), (1, 0), (0, 0)$ → $(0, 1, 0)$. shifted by one, $(0, 0, 0, 1, 0, 0)$: windows $(0, 0), (0, 1), (0, 0)$ → $(0, 1, 0)$, the same output: within-window shifts are absorbed. shifted by two, $(0, 0, 0, 0, 1, 0)$ → $(0, 0, 1)$: the output shifted by one. now take $x = (0, 1, 0, 0, 0, 0)$ → $(1, 0, 0)$; shifted by one, $(0, 0, 1, 0, 0, 0)$ → $(0, 1, 0)$: the *same* one-pixel shift now moves the output. whether a small shift changes the output depends on where the feature sits relative to the pooling grid; only shifts by multiples of the stride are reliably equivariant. that's aliasing, and it's why CNN outputs can change under one-pixel shifts (Azulay & Weiss, 2019; Zhang, 2019).

</details>

## experiment

`lab.py` measures how far networks look and how much they care where things are:

1. the theoretical and effective receptive fields of stacks of 2 to 32 random $3 \times 3$ convolutions.
2. three small CNNs that downsample with max pooling, average pooling or strided convolutions, trained on MNIST and tested on shifted digits.
3. three 8-layer stacks that grow their receptive field in different ways (plain, strided, dilated): theoretical versus effective.

## check your understanding

1. why does a max pooling layer have no parameters, yet change what a network learns?

<details><summary>answer</summary>

it's a fixed function (a max over each window), so there's nothing to learn. but it changes the architecture's assumptions: it throws away exact positions within each window, shrinks later layers' cost, grows their receptive field, and routes gradients only to the winning activations, so earlier layers learn features whose *presence* matters more than their exact location.

</details>

2. in a $2 \times 2$ max pool, what fraction of the input activations receive a nonzero gradient on one step, and does that slow learning?

<details><summary>answer</summary>

one per window, so a quarter (fewer if some windows' upstream gradients are zero). the others get exactly zero for that example. it doesn't stall learning in practice, because different examples and positions pick different winners and every weight of the previous convolution is shared across all positions, so it collects gradient through the winners everywhere.

</details>

3. a classmate computes receptive fields by multiplying kernel sizes: two $3 \times 3$ layers see $9 \times 9$. what's wrong?

<details><summary>answer</summary>

fields add, they don't multiply: each layer adds $d(k - 1)$ times the current jump. two stride-1 $3 \times 3$ layers see $1 + 2 + 2 = 5$ pixels. multiplication happens only to the *jump*, via the strides.

</details>

4. why does `effective_receptive_field` run the model in eval mode and differentiate with `torch.autograd.grad` rather than `.backward()`?

<details><summary>answer</summary>

in training mode batch norm normalizes with the batch's statistics, which couples the samples (the gradient of one sample's output would leak into the others' inputs) and updates the running statistics, changing the model. `.backward()` would also accumulate gradients into every parameter's `.grad`, a side effect on the model; `autograd.grad(out, x)` computes only the gradient with respect to the input.

</details>

5. why is the effective receptive field roughly Gaussian, and why does its width grow like $\sqrt{L}$?

<details><summary>answer</summary>

the gradient from the centre output reaches an input pixel along every path through the layers, and each layer moves the path by a small random offset within its kernel. the input offset is the sum of $L$ such offsets, so (central limit theorem) its distribution, weighted by the path products, is approximately Gaussian with variance proportional to $L$: standard deviation $\propto \sqrt{L}$, against a theoretical field $\propto L$.

</details>

6. you need a segmentation network whose output pixels can use context from 100 pixels away, without downsampling. what are your options, and what does the effective receptive field warn you about?

<details><summary>answer</summary>

dilated convolutions with growing rates (receptive field exponential in depth), very large kernels, or a downsample–upsample architecture with skip connections (topic 06's U-Net). the ERF warns that a theoretical field of 100 pixels doesn't mean the network *uses* pixels 100 away: the sensitivity concentrates near the centre, so you want a theoretical field comfortably larger than the context you need (and to check the effective one).

</details>

## common bugs

- **stride defaulting to 1.** symptom: max pooling returns an almost full-size map. in PyTorch and here, `stride=None` means `stride = kernel_size`.
- **pooling across channels.** symptom: matches torch on single-channel inputs, not on multi-channel ones. reshape to $(N C, 1, H, W)$ so each channel is pooled on its own, and reshape back in the same order.
- **spreading max-pool gradients over the window.** symptom: the backward matches average pooling's; the gradient must go to the argmax only.
- **assigning instead of adding overlapping gradients.** symptom: correct for $k = s$, wrong for overlapping windows. `col2im` adds.
- **updating the jump before the field.** symptom: receptive fields too large after every strided layer. the stride of layer $l$ only affects layers above it.
- **leaving gradients or training mode behind.** symptom: a model whose `.grad`s are populated or whose batch-norm statistics moved after you only *measured* it. differentiate with respect to the input with `autograd.grad`, in eval mode, and restore the mode.

## going deeper

- Luo, Li, Urtasun & Zemel (2016), "Understanding the effective receptive field in deep convolutional neural networks", *NeurIPS*. [arXiv:1701.04128](https://arxiv.org/abs/1701.04128)
- Araujo, Norris & Sim (2019), "Computing receptive fields of convolutional neural networks", *Distill*. receptive field arithmetic for arbitrary architectures. [doi:10.23915/distill.00021](https://doi.org/10.23915/distill.00021)
- Azulay & Weiss (2019), "Why do deep convolutional networks generalize so poorly to small image transformations?", *JMLR* 20(184). [arXiv:1805.12177](https://arxiv.org/abs/1805.12177)
- Zhang (2019), "Making convolutional networks shift-invariant again", *ICML*. anti-aliased (blurred) downsampling. [arXiv:1904.11486](https://arxiv.org/abs/1904.11486)
- Springenberg, Dosovitskiy, Brox & Riedmiller (2015), "Striving for simplicity: the all convolutional net", *ICLR workshop*. replacing pooling with strided convolutions. [arXiv:1412.6806](https://arxiv.org/abs/1412.6806)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, section 9.3 (pooling). [deeplearningbook.org](https://www.deeplearningbook.org)

## connections

- ← builds on: your `im2col`/`col2im`, `as_pair` and the output-size formula (topic 01), subgradients of a max (ch 14 · topic 03), evaluation mode and autograd (ch 15 · topics 02 and 07), the central limit theorem (ch 04 · topic 05).
- → used in: VGG's conv–conv–pool pattern and global average pooling (topic 03), ResNet's strided stages and stem (topic 04), dilated convolutions and U-Net's encoder–decoder (topic 06), patch sizes in vision transformers (ch 18), shift robustness and adversarial perturbations (ch 28).
