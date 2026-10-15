# residual networks

## why this matters

in 2015, a team at Microsoft Research trained plain convolutional networks of 20 and 56 layers on CIFAR-10, with batch norm and careful initialization, and found something that shouldn't happen: the 56-layer network had higher **training** error than the 20-layer one. not test error, where overfitting could explain it; training error. that's strange, because the deeper network can represent everything the shallower one can: copy the 20 layers and make the other 36 compute the identity. a solution at least as good exists, and the optimizer doesn't find it. they called it the **degradation problem**, and their fix (He, Zhang, Ren & Sun, 2016) was almost embarrassingly simple: let each pair of layers compute a *correction* to its input, $x + F(x)$, instead of a replacement. if the identity is what's needed, the layers just have to output zero. their residual networks trained at 152 layers on ImageNet and over 1,000 on CIFAR, won every major vision competition that year, and the idea is now in nearly every deep architecture, transformers included (chapter 18).

this topic builds the residual block, the ResNet around it (the backbone that chapters 18, 21, 25 and 28 reuse, and that you'll train to 93% on CIFAR-10 in the capstone) and the bottleneck block that makes very deep ResNets affordable. in the lab you'll watch the degradation problem happen, and then delete layers from a trained ResNet to see that it barely notices.

## warm-up: predict first

1. a 34-layer plain network and an 18-layer one are trained identically. which reaches the lower *training* loss?
2. a residual block computes $\operatorname{relu}(x + F(x))$. what does it compute if all of $F$'s weights are zero?
3. a block halves the resolution and doubles the channels. how can you add its input to its output?
4. the gradient through a plain layer is multiplied by its Jacobian $J$. what is it multiplied by through a residual block?
5. you delete one block from the middle of a trained 34-layer ResNet. how much accuracy do you lose?

## prerequisites check

- [ ] `MiniVGG`'s conv → BN → ReLU pattern, global average pooling, parameter counts (→ topic 03)
- [ ] strides, receptive fields and why stages double their width (→ topics 02–03)
- [ ] kaiming initialization and why depth makes signals and gradients explode or vanish (→ ch 15 · topic 01)
- [ ] batch norm in training and eval mode (→ ch 15 · topic 02); the chain rule for a sum, $\partial(a + b) = \partial a + \partial b$ (→ ch 14 · topic 02)

## core ideas

### idea 1: the residual block

call the function a stack of layers should compute $H(x)$. a plain stack learns $H$ directly; a **residual** stack learns $F(x) = H(x) - x$ and outputs $x + F(x)$. the two can represent the same functions, so why should it matter? because of what's *easy*. if a block's best contribution is small (a refinement of the features it receives, or nothing at all), the residual block only has to push its weights towards zero, which weight decay and small initializations already do. the plain block has to learn the identity map through two convolutions, two batch norms and two ReLUs, which is a specific, hard-to-find point in weight space. He et al.'s hypothesis was that deep networks' optimal layers are close to identities, and that making identity the default is what fixes degradation.

the **basic block** of ResNet-18 and ResNet-34 is two $3 \times 3$ convolutions with batch norm, plus a shortcut:

$$y = \operatorname{relu}\big(\underbrace{\operatorname{bn}_2(\operatorname{conv}_2(\operatorname{relu}(\operatorname{bn}_1(\operatorname{conv}_1(x)))))}_{F(x)} + \operatorname{shortcut}(x)\big).$$

details that matter:

- `conv1` carries the block's **stride**; `conv2` always has stride 1. both are $3 \times 3$ with padding 1 and **no bias** (batch norm follows, topic 03).
- the final ReLU comes **after** the addition. putting it on $F(x)$ before adding would make the branch only able to add, never subtract.
- the **shortcut** is the identity (an empty `nn.Sequential`, no parameters) when input and output shapes match. when the block changes the shape (stride 2, or a different channel count), $x$ can't be added directly, so the shortcut is a $1 \times 1$ convolution with the block's stride, followed by batch norm: a **projection**.
- with `residual=False` the block is **plain**: no addition and no shortcut at all, just $\operatorname{relu}(\operatorname{bn}_2(\dots))$. it has the same layers as the residual block (minus projections), which is what makes the lab's comparison fair.

why it trains: the gradient of a residual block's output with respect to its input is

$$\frac{\partial y}{\partial x} = D\,\Big(I + \frac{\partial F}{\partial x}\Big),$$

where $D$ is the final ReLU's 0/1 mask. through $L$ blocks, the backward pass multiplies $L$ such factors. in a plain network it multiplies $L$ Jacobians $\partial F_l/\partial x$, and any systematic shrinking or growing compounds exponentially. with residual blocks each factor is the identity plus a correction, so the product contains a term that is (nearly) the identity, a direct gradient path from the loss to every layer. a related trick: initialize each block's last batch-norm scale $\gamma$ to zero, and every residual block starts as exactly the identity (Goyal et al., 2017).

> **▶ build: stage 1.** implement `BasicBlock(in_channels, out_channels, stride=1, *, residual=True)` with submodules `conv1`, `bn1`, `conv2`, `bn2`, `shortcut`.

> **🔮 predict:** the plain 34-layer network has batch norm after every convolution, so every layer's activations have unit scale at initialization. will the gradient at its first layers be about as large as at its last layers?

### idea 2: the ResNet

a ResNet is a **stem**, four **stages** of blocks, global average pooling and a linear head:

| part | ResNet-18 for CIFAR, $32 \times 32$ input | output |
|---|---|---|
| stem | $3 \times 3$ conv, 64 channels, stride 1, BN, ReLU | $64 \times 32 \times 32$ |
| stage 1 | 2 basic blocks, 64 channels | $64 \times 32 \times 32$ |
| stage 2 | 2 blocks, 128 channels, the first with stride 2 | $128 \times 16 \times 16$ |
| stage 3 | 2 blocks, 256 channels, stride 2 first | $256 \times 8 \times 8$ |
| stage 4 | 2 blocks, 512 channels, stride 2 first | $512 \times 4 \times 4$ |
| head | global average pool, linear | 512 → classes |

the pattern is VGG's: each stage halves the resolution and doubles the width. the block counts name the network: $(2, 2, 2, 2)$ is ResNet-18 (16 convolutions in blocks, plus the stem and the head), $(3, 4, 6, 3)$ is ResNet-34. **`ResNet(block_counts=(2, 2, 2, 2), num_classes=10, *, in_channels=3, base_width=64, residual=True, small_input=True)`** builds stage $i$ with width `base_width * 2**i`, stride 2 in the first block of every stage after the first, and passes `residual` to every block. two stems:

- `small_input=True` (CIFAR): a $3 \times 3$, stride-1 convolution, BN, ReLU. $32 \times 32$ images are already small.
- `small_input=False` (ImageNet): a $7 \times 7$ stride-2 convolution with padding 3, BN, ReLU, and a $3 \times 3$ stride-2 max pool with padding 1, which divide a $224 \times 224$ photo by 4 before the first stage (topic 02: a receptive field of 11 after two layers).

store them as `self.stem`, `self.layers` (an `nn.Sequential` with one `nn.Sequential` of blocks per stage), `self.pool` (`nn.AdaptiveAvgPool2d(1)`) and `self.fc`, and set `self.feature_dim`, the width of the last stage (512 for ResNet-18). initialize every convolution with kaiming-normal (`mode="fan_out"`, ReLU gain), as torchvision does. then two methods: **`forward_features(x)`** returns the pooled, flattened features of shape $(B, \text{feature\_dim})$, and **`forward(x)`** returns `self.fc(self.forward_features(x))`. that split is part of the course's contract: chapter 18 compares this ResNet with a vision transformer, chapter 21 computes FID scores from `forward_features`, chapter 25 trains self-supervised encoders through it and replaces `fc`, and chapter 28 attacks and explains these models. finally, **`resnet18(num_classes=10, *, in_channels=3, residual=True)`** is `ResNet((2, 2, 2, 2), ...)` with base width 64 and the CIFAR stem.

> **▶ build: stage 2.** implement `ResNet` (with `forward_features`) and `resnet18`.

### idea 3: bottlenecks, and a network of many paths

a basic block at 256 channels costs $2 \times 9 \times 256^2 \approx 1.18$ million weights. ResNet-50 and deeper use a cheaper **bottleneck** block: a $1 \times 1$ convolution **reduces** the channels by a factor of 4, a $3 \times 3$ convolution works at the reduced width, and a $1 \times 1$ convolution **expands** back:

$$x\;(256) \xrightarrow{1 \times 1} 64 \xrightarrow{3 \times 3} 64 \xrightarrow{1 \times 1} 256, \qquad \text{plus } x.$$

each convolution is followed by batch norm, the first two by ReLU, and the last ReLU comes after the addition, as before. the block has $256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256 = 69{,}632$ weights, 17 times fewer than the basic block at the same width, so ResNet-50 can be twice as deep as ResNet-34 at a similar cost. **`Bottleneck(in_channels, mid_channels, stride=1, *, expansion=4, residual=True)`** outputs `mid_channels * expansion` channels (store it as `self.out_channels`), puts the stride in the $3 \times 3$ convolution (`conv2`), and uses a projection shortcut when the shape changes.

there's a second way to see what residual connections do. unroll two blocks: $x_2 = x_1 + F_1(x_1)$ and $x_3 = x_2 + F_2(x_2)$, so the input reaches the output through the identity, through $F_1$ alone, through $F_2$ alone, and through both. with $L$ blocks there are $2^L$ such paths, of lengths from 0 to $L$, most of them of length about $L/2$ (Veit, Wilber & Belongie, 2016). a deep ResNet behaves like a large ensemble of shallower networks that share weights, and deleting one block removes only the paths through it, half of them, all others intact. in a plain network there is exactly one path, and deleting any layer breaks it. the lab tests this.

> **🔮 predict:** you remove one of the 13 non-downsampling blocks from a trained 34-layer ResNet, and from a trained plain 34-layer network. how much test accuracy does each lose?

> **▶ build: stage 3.** implement `Bottleneck`.

## worked example

**parameter counts.** a basic block at 64 channels with an identity shortcut: two $3 \times 3$ convolutions, $2 \cdot 64 \cdot 64 \cdot 9 = 73{,}728$ weights, and two batch norms, $4 \cdot 64 = 256$: **73,984**. the first block of stage 2 (64 → 128, stride 2): `conv1` $64 \cdot 128 \cdot 9 = 73{,}728$, `conv2` $128 \cdot 128 \cdot 9 = 147{,}456$, two BNs $512$, and the projection $64 \cdot 128 + 256 = 8{,}448$: **230,144**.

**ResNet-18 for CIFAR** adds up to **11,173,962** parameters: the stem ($1{,}728 + 128$), eight basic blocks (three of them with projections), and the head ($512 \cdot 10 + 10$). its plain version, without the three projections, has **11,000,138**. with the ImageNet stem and 1,000 classes it has **11,689,512**, exactly torchvision's `resnet18`. a 32 × 32 image goes through the shapes in idea 2's table, ending in 512 features.

**bottleneck vs basic.** `BasicBlock(256, 256)` has $2 \cdot 9 \cdot 256^2 + 4 \cdot 256 = 1{,}180{,}672$ parameters; `Bottleneck(256, 64)` has $69{,}632$ weights plus $2 \cdot (64 + 64 + 256) = 768$ batch-norm parameters: **70,400**.

**the identity at initialization.** set `bn2.weight` to zero in an identity block and $F(x) = \operatorname{bn}_2(\dots) = 0 \cdot \hat z + \beta = 0$ (with $\beta = 0$ at init), so the block outputs $\operatorname{relu}(x)$.

the tests check the network totals, the bottleneck's count and the identity property.

## on paper

1. **the gradient through residual blocks.** for blocks without the final ReLU, $x_{l+1} = x_l + F_l(x_l)$, write $\partial x_L/\partial x_0$ as a product, expand it, and explain why the gradient reaching $x_0$ can't vanish unless the corrections conspire to cancel the identity.

<details><summary>solution</summary>

$\frac{\partial x_L}{\partial x_0} = \prod_{l=L-1}^{0} (I + J_l)$ with $J_l = \partial F_l/\partial x_l$. expanding gives $I + \sum_l J_l + \sum_{l > m} J_l J_m + \dots$, a sum over all $2^L$ subsets of blocks. the identity term passes the gradient $\partial L/\partial x_L$ to $x_0$ unchanged, so $\partial L/\partial x_0 = \partial L/\partial x_L + (\text{terms through the blocks})$: unless the block terms cancel the identity, the gradient at the input is at least as large as some direct copy of the gradient at the output. a plain network has only the single longest term $\prod_l J_l$, which shrinks or grows exponentially if the $J_l$ are systematically smaller or larger than 1 in norm.

</details>

2. **the construction argument.** show that for any 18-layer plain network there's a 34-layer plain network computing the same function (with batch norm, assume you may set its parameters freely), and explain why that makes the degradation problem an optimization problem.

<details><summary>solution</summary>

take the 18-layer network's layers and insert extra blocks that compute the identity: with ReLU activations, a conv whose kernel is the identity (a 1 at the centre tap of matching in/out channels) followed by a BN that undoes its own normalization ($\gamma = \sigma$, $\beta = \mu$, in eval mode) and a ReLU passes nonnegative inputs through unchanged (the inputs after a ReLU are nonnegative). so the deeper network's best training loss is at most the shallower one's. if gradient descent finds a *worse* solution for the deeper network, it isn't a representation limit and it isn't overfitting (the training loss is worse): the optimizer fails to find the good solutions. residual blocks make "do nothing" the default ($F = 0$) instead of a fine-tuned point.

</details>

3. **bottleneck cost.** compare the MACs of `BasicBlock(256, 256)` and `Bottleneck(256, 64)` on a $14 \times 14$ feature map, and the number of $3 \times 3$ convolutions each network of 50 layers would contain.

<details><summary>solution</summary>

basic: $2 \cdot 9 \cdot 256^2 \cdot 196 \approx 231$ million MACs. bottleneck: $(256 \cdot 64 + 9 \cdot 64^2 + 64 \cdot 256) \cdot 196 \approx 13.6$ million, 17 times less (the ratio of the weight counts, since all three convolutions keep the resolution). ResNet-50 has 16 bottleneck blocks ($3 + 4 + 6 + 3$), so 16 $3 \times 3$ convolutions; a basic-block network of 50 layers would have 48. most of ResNet-50's layers are cheap $1 \times 1$ convolutions.

</details>

4. **the lengths of the paths.** in a ResNet with $n$ residual blocks unrolled into $2^n$ paths, how many paths have length $k$ (pass through $k$ blocks)? what is the mean length for $n = 16$, and what fraction of paths survive deleting one block?

<details><summary>solution</summary>

a path chooses, for each block, the branch or the shortcut, so $\binom{n}{k}$ paths have length $k$: a binomial distribution with mean $n/2 = 8$ and standard deviation $\sqrt{n}/2 = 2$ for $n = 16$. 95% of the paths have lengths between 4 and 12; the network's "effective depth" is much less than 34. deleting block $l$ removes the paths that go through $F_l$, exactly half; the other $2^{15}$, none of which uses $F_l$, are untouched. (in a trained network the gradient also concentrates on short paths, Veit et al. found.)

</details>

## experiment

`lab.py` compares plain and residual networks of 18 and 34 layers:

1. gradient norms at every block at initialization.
2. training curves: does the deeper plain network fit the training data better or worse?
3. deleting single blocks from the trained 34-layer networks.

## check your understanding

1. the degradation problem is about training error. why does that rule out overfitting as its cause?

<details><summary>answer</summary>

overfitting means fitting the training data *better* while generalizing worse. a deeper network with higher training error fits the training data worse, even though it could represent the shallower network's solution exactly. it's an optimization failure.

</details>

2. why does the final ReLU come after the addition, not on the residual branch before it?

<details><summary>answer</summary>

a ReLU on $F(x)$ would make the branch's output nonnegative, so a block could only increase each activation, never decrease it: the output $x + \operatorname{relu}(F(x))$ can only grow with depth. after the addition, the branch can make corrections in either direction and the ReLU acts on the combined signal. (pre-activation ResNets, He et al. 2016b, move BN and ReLU *before* the convolutions so the identity path has no ReLU at all.)

</details>

3. when does a block need a projection shortcut, and why is it followed by batch norm?

<details><summary>answer</summary>

when the output shape differs from the input's: a stride of 2 (different resolution) or a different channel count. the $1 \times 1$ stride-$s$ convolution maps $x$ to the right shape. batch norm puts the projected shortcut on the same scale as the branch's batch-normalized output, so neither dominates the sum.

</details>

4. why does `ResNet` expose `forward_features` separately from `forward`?

<details><summary>answer</summary>

the pooled features are what other tasks reuse: transfer learning trains a new head on them (topic 05), FID compares their statistics between real and generated images (ch 21), self-supervised methods train the encoder through them (ch 25), attribution methods explain them (ch 28). a stable `forward_features` / `fc` split lets all of that work without knowing the network's internals.

</details>

5. the CIFAR stem has no max pool and stride 1. what would the ImageNet stem do to a $32 \times 32$ image?

<details><summary>answer</summary>

it divides the resolution by 4 before the first block: stage 1 would work at $8 \times 8$ and stage 4 at $1 \times 1$, throwing away most of the spatial detail of an already tiny image (and making the last stages' $3 \times 3$ convolutions mostly padding). the CIFAR stem keeps $32 \times 32$ for stage 1 and ends at $4 \times 4$.

</details>

6. in the unrolled view, a 34-layer ResNet is an ensemble of paths of mostly about half its depth. does that make the depth useless?

<details><summary>answer</summary>

no: the longer paths still exist and contribute, and the blocks are trained jointly so that their corrections compose. but it explains why ResNets degrade gracefully (deleting a block removes some paths, not the network) and why their effective depth, the path lengths that carry most of the gradient, is smaller than the nominal depth. very deep ResNets gain less from depth than the layer count suggests, which is part of why later work scaled width and resolution too (topic 05).

</details>

## common bugs

- **the ReLU before the addition.** symptom: the forward pass differs from the spec, and a block with a zeroed branch returns $x$ instead of $\operatorname{relu}(x)$ (negative inputs pass through).
- **the stride on the wrong convolution.** symptom: shapes are right but `conv1`'s stride is 1 and `conv2`'s is 2; the first convolution then does 4× the work at full resolution. the stride goes in `conv1` (and, in a bottleneck, in the $3 \times 3$ `conv2`).
- **projecting every shortcut.** symptom: extra parameters (and slower training); identity shortcuts must be parameter-free.
- **projections in the plain network.** symptom: `residual=False` blocks still have a shortcut; the plain network must have no shortcut at all.
- **an unpooled `forward_features`.** symptom: features of shape $(B, 512 \cdot 4 \cdot 4)$, or the right shape but not averaged. later chapters rely on pooled $(B, \text{feature\_dim})$ features.
- **the wrong stem for the data.** symptom: CIFAR accuracy stuck 10–20 points too low with the ImageNet stem. `small_input=True` for small images.

## going deeper

- He, Zhang, Ren & Sun (2016), "Deep residual learning for image recognition", *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- He, Zhang, Ren & Sun (2016b), "Identity mappings in deep residual networks", *ECCV*. pre-activation blocks. [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)
- Veit, Wilber & Belongie (2016), "Residual networks behave like ensembles of relatively shallow networks", *NeurIPS*. [arXiv:1605.06431](https://arxiv.org/abs/1605.06431)
- Goyal et al. (2017), "Accurate, large minibatch SGD: training ImageNet in 1 hour". zero-initialized residual branches. [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- Balduzzi et al. (2017), "The shattered gradients problem: if resnets are the answer, then what is the question?", *ICML*. [arXiv:1702.08591](https://arxiv.org/abs/1702.08591)
- Zagoruyko & Komodakis (2016), "Wide residual networks", *BMVC*. width versus depth. [arXiv:1605.07146](https://arxiv.org/abs/1605.07146)

## connections

- ← builds on: `MiniVGG`'s blocks and stages (topic 03), receptive fields and strides (topic 02), kaiming init and signal propagation (ch 15 · topic 01), batch norm (ch 15 · topic 02), the chain rule for sums (ch 14).
- → used in: depthwise and ConvNeXt blocks, transfer learning with `forward_features` and `fc` (topic 05), U-Net's skip connections (topic 06), the capstone (your ResNet-18 on CIFAR-10), residual streams in transformers (ch 18), signal propagation theory (ch 19), FID features (ch 21), SSL backbones (ch 25), attacks and attribution (ch 28).
