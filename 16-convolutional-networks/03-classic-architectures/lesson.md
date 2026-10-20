# classic architectures

## why this matters

you have the parts: convolutions that detect local patterns anywhere in an image, and pooling that trades resolution for reach. how should they be put together? for almost twenty years the answer evolved through a handful of famous networks, and each one is still worth knowing because its design decisions are still in every network you'll use. **LeNet-5** (LeCun et al., 1998) read handwritten digits on bank cheques with 60,000 parameters, and its pattern (convolve, pool, repeat, then classify) is the template of the whole field. **AlexNet** (Krizhevsky, Sutskever & Hinton, 2012) won ImageNet by a landslide with the same template scaled up, plus ReLUs, dropout, augmentation and GPUs, and started the deep-learning era. **VGG** (Simonyan & Zisserman, 2015) asked the cleanest question: what if every convolution is $3 \times 3$ and we just go deeper?

this topic builds LeNet and a small VGG with `nn.Conv2d` and `nn.MaxPool2d`, which you've earned, and a profiler that answers two questions you should ask of every network: where are its parameters, and where is its compute? (they are in very different places.) in the lab you'll see how much of a CNN's advantage comes from its *assumptions* about images, by breaking them, and what happens when you stack more layers without batch norm.

## warm-up: predict first

1. LeNet-5 takes $32 \times 32$ images, but MNIST digits are $28 \times 28$. what would you do?
2. VGG halves the resolution and doubles the number of channels at every stage. what happens to each layer's compute?
3. a convolution followed by batch norm: why is the convolution's bias pointless?
4. VGG-16 has 138 million parameters. roughly what fraction do you think are in its convolutional layers?
5. you shuffle the pixels of every MNIST image with the same fixed permutation. which suffers more, an MLP or a CNN?

## prerequisites check

- [ ] convolution and pooling, their output sizes and parameter counts (→ topics 01 and 02)
- [ ] receptive fields and the effect of downsampling on them (→ topic 02)
- [ ] batch norm: what it computes per channel, its running statistics, train vs eval mode (→ ch 15 · topic 02)
- [ ] `nn.Module`, `nn.Sequential`, forward hooks, `count_parameters` (→ ch 15 · topics 05 and 07)

## core ideas

### idea 1: LeNet-5, the template

LeNet-5 alternates convolutions and subsampling, then finishes with fully connected layers:

| layer | operation | output shape |
|---|---|---|
| input | a digit, zero-padded from $28 \times 28$ to $32 \times 32$ | $1 \times 32 \times 32$ |
| `conv1` | $5 \times 5$ conv, 6 filters, no padding | $6 \times 28 \times 28$ |
| | ReLU, $2 \times 2$ max pool | $6 \times 14 \times 14$ |
| `conv2` | $5 \times 5$ conv, 16 filters | $16 \times 10 \times 10$ |
| | ReLU, $2 \times 2$ max pool | $16 \times 5 \times 5$ |
| `fc1`, `fc2`, `fc3` | flatten to 400, then $400 \to 120 \to 84 \to$ classes, ReLU between | classes |

three things about this design are worth seeing. **local features first, global decisions last**: the convolutions build a $16 \times 5 \times 5$ grid of features, each of which summarizes a patch of the digit (on paper 1: a $16 \times 16$ patch), and only then do dense layers combine all positions. **channels grow as resolution shrinks** ($1 \to 6 \to 16$ channels as $32 \to 14 \to 5$ pixels), so the representation trades "where" for "what". and **the padding has a purpose**: the original paper padded digits so that strokes near the border fall in the middle of $5 \times 5$ windows, where the feature detectors can see them whole.

the 1998 network differed in details you don't need: a scaled tanh instead of ReLU, average pooling with a trainable coefficient instead of max pooling, `conv2` connected to only some of `conv1`'s channels, and a radial-basis output layer. the modern reading, which you'll build, uses ReLU and max pooling. pad in `forward` (with `F.pad`, splitting the padding evenly between the sides) only when the input is smaller than $32 \times 32$, so the same network also accepts $32 \times 32$ colour images (`in_channels=3`), and return raw logits.

> **▶ build: stage 1.** implement `LeNet(num_classes=10, in_channels=1)` with submodules `conv1`, `conv2`, `fc1`, `fc2`, `fc3`.

> **🔮 predict:** LeNet has 61,706 parameters. an MLP with one hidden layer of 77 units has 61,225. on MNIST, which is more accurate after two epochs, and by how much?

### idea 2: AlexNet, VGG and the "just use $3 \times 3$" rule

**AlexNet** kept LeNet's template and scaled it to $224 \times 224$ colour photos and 1,000 classes: five convolutional layers (the first $11 \times 11$ with stride 4), three max pools, three fully connected layers of 4,096 units, 60 million parameters. what made it work was a set of ingredients you know from chapter 15: **ReLU** activations (trained several times faster than tanh), **dropout** in the fully connected layers, **data augmentation** (random crops and flips), weight decay, and two GPUs. its depth was limited by what could be trained.

**VGG** simplified the recipe to a few rules and pushed depth to 16–19 layers:

1. every convolution is $3 \times 3$ with stride 1 and padding 1 (resolution preserved). two $3 \times 3$ layers see what one $5 \times 5$ sees with 28% fewer weights and an extra nonlinearity; three replace a $7 \times 7$ with 45% fewer (topic 02, on paper 3).
2. layers come in **blocks**: a few convolutions, then a $2 \times 2$ max pool.
3. after every pool, the number of channels **doubles** (64, 128, 256, 512). halving $H$ and $W$ divides a layer's cost by 4; doubling $C_\text{in}$ and $C_\text{out}$ multiplies it by 4, so every block costs about the same compute (on paper 1).

your **`MiniVGG(num_classes=10, in_channels=3, *, widths=(32, 64, 128), batch_norm=True)`** is VGG for $32 \times 32$ images, with two updates from later work. each block is two `conv3x3 → BatchNorm2d → ReLU` layers and a max pool, stored in `self.features` (an `nn.Sequential`). batch norm (Ioffe & Szegedy, 2015) is what lets plain stacks like this train reliably at any depth; it goes *between* the convolution and the ReLU, and since it subtracts each channel's mean, a bias in the convolution would be cancelled exactly: use `bias=False` when `batch_norm=True` and `bias=True` otherwise. the head is **global average pooling** (`self.pool = nn.AdaptiveAvgPool2d(1)`, Lin et al., 2014) followed by a single `nn.Linear` named `self.fc`: each channel of the last feature map is averaged over all positions, so the head has $C \times k$ weights instead of VGG's $25{,}088 \times 4{,}096$, and the network accepts any input size.

> **▶ build: stage 2.** implement `MiniVGG`.

### idea 3: where the parameters are and where the compute goes

two budgets describe a network. **parameters** decide the size of the file you ship and, with optimizer state, the memory for training. **multiply–accumulates** (MACs) decide the time per image. for a layer they're very different numbers:

- a convolution with $C_\text{in}$ inputs, $C_\text{out}$ filters of size $k \times k$ (and $g$ groups) has $C_\text{out} \frac{C_\text{in}}{g} k^2$ weights, plus $C_\text{out}$ biases, but uses each weight at every output position: $\text{MACs} = C_\text{out} H_\text{out} W_\text{out} \cdot \frac{C_\text{in}}{g} k^2$, i.e. (number of output entries) × (inputs per output).
- a linear layer $n_\text{in} \to n_\text{out}$ has $n_\text{in} n_\text{out}$ weights and uses each one once per example: $\text{MACs} = n_\text{out} \cdot n_\text{in}$.

so convolutions are cheap in parameters and expensive in compute, and fully connected layers the opposite. the profiler makes this concrete. **`profile_model(model, input_shape)`** runs one zero input of shape $(1, *\text{input\_shape})$ through the model under `torch.no_grad()` in eval mode (so batch norm statistics don't move), with a **forward hook** (chapter 15, topic 05) on every `nn.Conv2d` and `nn.Linear`, and returns one dict per layer in forward order: `name` (from `model.named_modules()`), `out_shape` (without the batch dimension), `params` (the layer's own parameters, `module.parameters(recurse=False)`) and `macs`. remove the hooks and restore the model's mode afterwards, even if the forward pass fails.

> **🔮 predict:** VGG-16 has 13 convolutional layers and 3 fully connected ones. which group holds most of its 138 million parameters, and which does most of its 15 billion MACs?

> **▶ build: stage 3.** implement `profile_model(model, input_shape)`.

## worked example

**LeNet-5, layer by layer**, for a $28 \times 28$ digit (padded to 32):

| layer | output | parameters | MACs |
|---|---|---|---|
| `conv1` | $6 \times 28 \times 28$ | $6 \cdot 25 + 6 = 156$ | $6 \cdot 28 \cdot 28 \cdot 25 = 117{,}600$ |
| `conv2` | $16 \times 10 \times 10$ | $16 \cdot 6 \cdot 25 + 16 = 2{,}416$ | $16 \cdot 10 \cdot 10 \cdot 150 = 240{,}000$ |
| `fc1` | 120 | $400 \cdot 120 + 120 = 48{,}120$ | $48{,}000$ |
| `fc2` | 84 | $120 \cdot 84 + 84 = 10{,}164$ | $10{,}080$ |
| `fc3` | 10 | $84 \cdot 10 + 10 = 850$ | $840$ |
| total | | **61,706** | **416,520** |

the two convolutions hold 4% of the parameters and do 86% of the work; `fc1` alone holds 78% of the parameters. with three input channels and three classes, `conv1` gains $6 \cdot 2 \cdot 25 = 300$ weights and `fc3` loses $7 \cdot 84 + 7 = 595$ parameters: 61,411.

**MiniVGG with `widths=(8, 16)`**, 3 input channels, 5 classes, batch norm: the four convolutions have $3 \cdot 8 \cdot 9$, $8 \cdot 8 \cdot 9$, $8 \cdot 16 \cdot 9$ and $16 \cdot 16 \cdot 9$ weights (no biases), each batch norm 2 parameters per channel ($\gamma$, $\beta$), and the head $16 \cdot 5 + 5$: $232 + 592 + 1{,}184 + 2{,}336 + 85 = 4{,}429$ parameters. the default `MiniVGG()` has 288,746 (287,850 in its convolution and linear layers, plus 896 batch-norm parameters).

the tests check both tables.

## on paper

1. **VGG's compute per block.** a $3 \times 3$ convolution from $C$ to $C$ channels on an $H \times W$ map costs $9 H W C^2$ MACs. show that halving $H$ and $W$ while doubling $C$ keeps the cost constant, and compute the receptive field of LeNet's `conv2` outputs and of one unit after its second pool.

<details><summary>solution</summary>

$9 \cdot \frac{H}{2} \cdot \frac{W}{2} \cdot (2C)^2 = 9 H W C^2$. (the first conv of a block, $C \to 2C$, costs half as much.) LeNet: conv1 $(5, 1, 1)$ gives $r = 5$; pool $(2, 2, 1)$: $r = 6$, $j = 2$; conv2 $(5, 1, 1)$: $r = 6 + 4 \cdot 2 = 14$; pool: $r = 14 + 1 \cdot 2 = 16$, $j = 4$. each of the 400 features entering `fc1` summarizes a $16 \times 16$ patch of the padded $32 \times 32$ image.

</details>

2. **the bias before batch norm.** a convolution computes $z = w * x + b$ per channel and batch norm outputs $\gamma \frac{z - \mu_B}{\sigma_B} + \beta$. show that the output doesn't depend on $b$, in training mode and in eval mode.

<details><summary>solution</summary>

in training mode $\mu_B = \overline{w * x} + b$ (the bias is the same for every position and example of the channel) and $\sigma_B$ is unchanged by adding a constant, so $z - \mu_B = w * x - \overline{w * x}$: $b$ cancels. in eval mode the running mean was accumulated from the same $\mu_B$s, so it contains $b$ too, and cancels it the same way. the bias's gradient is exactly zero; $\beta$ does its job.

</details>

3. **VGG-16's first fully connected layer.** the last pool outputs $512 \times 7 \times 7$ for a $224 \times 224$ input, followed by $\text{fc} \colon 25{,}088 \to 4{,}096$. how many parameters is that, what fraction of 138.4 million, and what happens to it for a $448 \times 448$ input?

<details><summary>solution</summary>

$25{,}088 \cdot 4{,}096 + 4{,}096 = 102{,}764{,}544$, 74% of the network in one layer, used once per image (103 million MACs of 15.5 billion, under 1%). a $448 \times 448$ input gives a $512 \times 14 \times 14$ map, so the layer would need $100{,}352$ inputs: it no longer fits. global average pooling (MiniVGG) has no such problem, and so does treating fc6 as a $7 \times 7$ convolution, which slides over larger inputs and produces a map of predictions ("fully convolutional" networks).

</details>

4. **a permutation-equivariant MLP.** let $P$ be a fixed $784 \times 784$ permutation matrix and an MLP's first layer compute $W x$. show that training on permuted inputs $P x$ is equivalent to training on the originals with the first weight matrix replaced by $W P$, and argue why the MLP's test accuracy is the same in distribution (with an init whose distribution is permutation-invariant).

<details><summary>solution</summary>

$W(Px) = (WP)x$, so a network with weights $W$ on permuted data computes exactly what a network with weights $WP$ computes on the original data, and the gradient steps correspond as well ($\nabla_W L(WPx)$ is $\nabla_{W'} L(W'x)$ at $W' = WP$, times $P^\top$). if $W$'s entries are i.i.d., $WP$ has the same distribution as $W$, so the two training runs are statistically identical: same accuracy distribution (and with Adam, which is elementwise, the correspondence holds step by step). a CNN has no such symmetry: its weights are tied to *neighbourhoods*, which a permutation destroys.

</details>

## experiment

`lab.py` asks three questions of your networks:

1. where VGG-16, MiniVGG and LeNet keep their parameters and spend their compute, with your profiler.
2. LeNet against an MLP of the same size, on MNIST and on MNIST with its pixels shuffled.
3. how test accuracy on CIFAR-10 changes with depth, with and without batch norm.

## check your understanding

1. what are the three assumptions about images that a CNN builds in, and which part of the architecture encodes each?

<details><summary>answer</summary>

locality (useful features are computed from nearby pixels: small kernels), translation equivariance (the same feature is useful anywhere: weight sharing across positions), and hierarchy (complex features are compositions of simpler ones over larger regions: stacking layers with downsampling, so receptive fields grow).

</details>

2. LeNet pads $28 \times 28$ digits to $32 \times 32$. what would change if it didn't pad?

<details><summary>answer</summary>

conv1 would output $24 \times 24$, the pools and conv2 would give $16 \times 4 \times 4 = 256$ features instead of 400, so fc1 would need a different size. and strokes near the border would only ever be seen by the edge of a window. padding keeps the arithmetic of the original design and lets the border be seen by full windows.

</details>

3. why did AlexNet's fully connected layers need dropout, while its convolutional layers mostly didn't?

<details><summary>answer</summary>

almost all of its parameters (and so most of its capacity to memorize) are in the fully connected layers, which see the whole image and have no weight sharing. the convolutional layers have few parameters, each used at thousands of positions per image, so they're constrained by the data far more.

</details>

4. why do VGG networks double the channel count after each pooling layer?

<details><summary>answer</summary>

to keep each block's compute roughly constant (on paper 1) and to keep the representation's capacity from collapsing: the spatial grid shrinks by 4 per block, so more channels per position are needed to carry what the network has learned about *what* is there as it forgets *where* exactly.

</details>

5. your profiler reports MACs for one input. how would you estimate the time per training step for a batch of 256?

<details><summary>answer</summary>

the forward pass costs about $256 \times$ the MACs, the backward pass about twice the forward (one product for the input gradient, one for the weight gradient), so a step is roughly $3 \times 256 \times \text{MACs} \times 2$ FLOPs; divide by the hardware's achieved throughput. memory-bound layers (batch norm, ReLU, pooling) add time the MAC count doesn't show.

</details>

6. global average pooling throws away all positional information at the end of the network. why doesn't that hurt classification?

<details><summary>answer</summary>

by the last block every position's features already summarize a large receptive field (often most of the image), and a class is defined by what is present, not where. the averaging also acts as a structural regularizer (no huge dense layer to overfit) and makes the network accept any input size. tasks that do need positions (detection, segmentation, topic 06) keep the spatial map instead.

</details>

## common bugs

- **padding inside `conv1` instead of the input.** symptom: LeNet works on $28 \times 28$ digits but crashes on $32 \times 32$ images (the flattened size is 576, not 400). pad only inputs smaller than 32.
- **a ReLU on the logits.** symptom: the loss plateaus; negative logits are clamped to zero, so the network can't push wrong classes down. the last layer is linear.
- **BN after the ReLU, or conv biases with BN.** symptom: a structure that differs from the spec; the bias is wasted (it cancels), and BN after ReLU normalizes a skewed distribution. the order is conv → BN → ReLU.
- **flattening without global pooling.** symptom: `MiniVGG` breaks for any input size other than the one you hard-coded. use `AdaptiveAvgPool2d(1)`.
- **profiling in training mode.** symptom: running statistics change when you only wanted to count things, and dropout layers change the outputs. profile in eval mode under `no_grad`, and remove the hooks.
- **counting MACs for the whole batch.** symptom: numbers proportional to the batch size. profile one example.

## going deeper

- LeCun, Bottou, Bengio & Haffner (1998), "Gradient-based learning applied to document recognition", *Proceedings of the IEEE* 86(11): 2278–2324. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- Krizhevsky, Sutskever & Hinton (2012), "ImageNet classification with deep convolutional neural networks", *NeurIPS*. [papers.nips.cc/paper/4824](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- Simonyan & Zisserman (2015), "Very deep convolutional networks for large-scale image recognition", *ICLR*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Lin, Chen & Yan (2014), "Network in network", *ICLR*. $1 \times 1$ convolutions and global average pooling. [arXiv:1312.4400](https://arxiv.org/abs/1312.4400)
- Ioffe & Szegedy (2015), "Batch normalization", *ICML*. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)
- Zhang, Lipton, Li & Smola (2023), *Dive into Deep Learning*, chapters 7–8 (convolutional networks, modern CNNs). [d2l.ai](https://d2l.ai)

## connections

- ← builds on: convolution (topic 01), pooling and receptive fields (topic 02), batch norm, dropout and augmentation (ch 15 · topics 02–03), forward hooks (ch 15 · topic 05), `MLP` and `count_parameters` (ch 15 · topic 07), the bias–variance view of inductive bias (ch 07).
- → used in: the degradation problem and residual networks (topic 04), parameter and compute budgets of efficient convnets (topic 05), U-Net's VGG-like encoder (topic 06), the capstone, FLOP accounting for training systems (ch 26).
