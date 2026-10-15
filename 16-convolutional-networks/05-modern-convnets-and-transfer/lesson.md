# modern convnets & transfer

## why this matters

two problems remain after ResNet. the first is cost: a ResNet-50 does 4 billion multiply–accumulates per image, too many for a phone, a drone or a browser, and most of that work is in $3 \times 3$ convolutions that mix every input channel with every output channel at every pixel. **depthwise separable convolutions** split that work into a cheap spatial step and a cheap channel-mixing step, and are the reason MobileNets and EfficientNets run on phones. the same building block, scaled up with ideas borrowed from transformers, gave **ConvNeXt** (Liu et al., 2022), a pure convnet that matches vision transformers.

the second problem is data. a ResNet trained from scratch needs tens of thousands of labelled images per task; a hospital with 300 labelled scans, or you with 50 photos of your plants, doesn't have them. but the features a network learns on a large dataset (edges, textures, parts, shapes) are useful far beyond the classes it was trained on, so you can take a **pretrained backbone**, put a new head on it, and train only what you must. that's **transfer learning**, and it's how most vision models are built in practice. you'll build both blocks and the routine that adapts a pretrained network to a new task, by **linear probing** (train only the head) or **fine-tuning** (train everything, gently), and in the lab you'll find out when each wins, what adapting costs the old task, and why "fewer multiplications" doesn't always mean "faster".

## warm-up: predict first

1. a standard $3 \times 3$ convolution from 128 to 128 channels has 147,456 weights. how many does a depthwise separable one with the same input and output have?
2. if a layer does 8× fewer multiply–accumulates, will it run 8× faster?
3. you have 5 labelled images per class for a new task and a backbone pretrained on related images. train only a new head, or fine-tune everything?
4. you freeze every parameter of a backbone with `requires_grad=False` and train a head on top in `train()` mode. can the backbone's behaviour still change?
5. after fine-tuning a pretrained network on a new task, how well does its backbone still do on the original task?

## prerequisites check

- [ ] convolution with groups, parameter and MAC counts, `profile_model` (→ topics 01 and 03)
- [ ] residual blocks, bottlenecks, `ResNet.fc` and `forward_features` (→ topic 04)
- [ ] layer norm, GELU, AdamW and optimizer parameter groups (→ ch 15 · topics 02–04 and 07)
- [ ] batch norm's running statistics, updated in train mode and used in eval mode (→ ch 15 · topic 02)

## core ideas

### idea 1: depthwise separable convolutions

a standard convolution does two jobs at once: it filters **spatially** (a $k \times k$ neighbourhood) and it **mixes channels** (every output channel looks at all input channels). its cost per output position is $k^2 C_\text{in} C_\text{out}$. a **depthwise separable** convolution does the two jobs one after the other:

1. a **depthwise** convolution: one $k \times k$ filter per input channel, applied to that channel only. in PyTorch it's `nn.Conv2d(C, C, k, groups=C)`: with `groups=C`, the channels are split into $C$ groups of one, each convolved separately. cost $k^2 C_\text{in}$ per position.
2. a **pointwise** convolution: a $1 \times 1$ convolution that mixes channels at each pixel, $C_\text{in} \to C_\text{out}$. cost $C_\text{in} C_\text{out}$ per position.

the ratio of costs is

$$\frac{k^2 C_\text{in} + C_\text{in} C_\text{out}}{k^2 C_\text{in} C_\text{out}} = \frac{1}{C_\text{out}} + \frac{1}{k^2},$$

about $\frac19$ for $3 \times 3$ kernels and wide layers: 8–9 times fewer multiply–accumulates and weights. it's an approximation, not an equivalent (a standard convolution can mix space and channels jointly in ways the factorized one can't), but MobileNet (Howard et al., 2017) showed that networks built from it lose only a little accuracy for a large saving.

**`DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=1)`** is MobileNet v1's block: `depthwise` ($k \times k$, `groups=in_channels`, padding $k /\!/ 2$, the stride, no bias) → `bn1` → ReLU → `pointwise` ($1 \times 1$, no bias) → `bn2` → ReLU.

one catch, which the lab measures: hardware speed isn't MAC count. a depthwise convolution does only $k^2$ multiply–accumulates for every input value it reads, while a standard convolution does $k^2 C_\text{out}$. on CPUs and GPUs, which can do far more arithmetic per second than they can move bytes, the depthwise step is limited by memory bandwidth, not arithmetic, so it runs far below the hardware's peak.

> **▶ build: stage 1.** implement `DepthwiseSeparableConv`.

> **🔮 predict:** at 128 channels, the depthwise separable block does 8.4 times fewer MACs than a standard $3 \times 3$ convolution. how much faster will it run on your CPU?

### idea 2: ConvNeXt, a convnet for the 2020s

in 2020–21, vision transformers (chapter 18) overtook ResNets on large-scale benchmarks. ConvNeXt asked how much of that was attention itself and how much was the transformer's *recipe*, and modernized a ResNet step by step until it matched them. its block borrows three ideas:

- **large depthwise kernels**: a $7 \times 7$ depthwise convolution does the spatial mixing (like attention's mixing across positions), cheaply, with a large receptive field (topic 02).
- **an inverted bottleneck**: channel mixing happens in an MLP that *expands* the channels 4× and projects back (MobileNet v2's design, and the transformer's feed-forward block), the opposite of ResNet's bottleneck, which reduces.
- **transformer-style details**: layer norm instead of batch norm, GELU instead of ReLU, a single normalization and activation per block, and **layer scale**, a learnable per-channel factor $\gamma$ initialized to $10^{-6}$ that multiplies the branch's output, so each block starts as (almost) the identity, the same idea as zero-initialized residual branches (topic 04).

$$y = x + \gamma \odot \operatorname{pw}_2\big(\operatorname{GELU}(\operatorname{pw}_1(\operatorname{LN}(\operatorname{dw}_{7 \times 7}(x))))\big).$$

the two $1 \times 1$ "pointwise convolutions" are implemented as `nn.Linear` layers on the channels-last tensor: after the depthwise convolution, permute $(B, C, H, W) \to (B, H, W, C)$, so that `nn.LayerNorm(dim, eps=1e-6)` normalizes each pixel's channel vector and the linear layers act on channels at every pixel; multiply by $\gamma$, permute back, and add the input. **`ConvNeXtBlock(dim, *, expansion=4, layer_scale_init=1e-6)`** has submodules `dwconv` (with bias, padding 3, `groups=dim`), `norm`, `pwconv1`, `pwconv2` and the parameter `gamma`.

> **▶ build: stage 2.** implement `ConvNeXtBlock`.

### idea 3: transfer learning, replacing and freezing

a network trained on a large dataset learns a hierarchy of features: edges and colour blobs in the first layers (topic 01's lab), textures and parts in the middle, object-specific features at the top. Yosinski et al. (2014) showed that the early and middle features transfer well between tasks, and the late ones less so. to reuse them, keep the **backbone** (everything up to `forward_features`) and replace the **head**:

- **`replace_head(model, num_classes)`** swaps `model.fc` (or `model.head`, for models that name it that way) for a fresh `nn.Linear` with the same `in_features` and `num_classes` outputs, on the same device and with the same dtype, and returns the model.
- **`freeze_backbone(model, *, head_name="fc")`** sets `requires_grad=False` for every parameter outside the submodule named `head_name` and `requires_grad=True` inside it, and returns the model; if the model has no submodule called `head_name`, it raises `ValueError` (silently freezing everything is a bug you'd only notice when nothing trains). it doesn't touch the modules' train/eval mode. "inside" means the parameter's name is `head_name` itself or starts with `head_name + "."`: a backbone layer called `fc_pre` is not the head.

freezing stops the *weights* from changing, but it doesn't make the backbone a fixed function. batch norm layers keep **running statistics** that are updated on every forward pass in training mode, whatever `requires_grad` says. a "frozen" backbone run in `train()` mode silently re-estimates its statistics on the new data, and with them changes its features. a truly frozen backbone runs in **eval mode**.

> **▶ build: stage 3.** implement `replace_head` and `freeze_backbone`.

### idea 4: linear probing and fine-tuning

with a pretrained backbone and a new head, there are two standard ways to train.

- **linear probing**: freeze the backbone (in eval mode) and train only the head, a linear classifier on fixed features. it fits a few thousand parameters, so it needs little data and can't damage the backbone; but if the pretrained features aren't quite right for the new task, it can't fix them.
- **fine-tuning**: train everything, starting from the pretrained weights. it can adapt every feature, so with enough data it usually wins; with very little data it can overfit, and it changes the backbone, so the original task suffers (the lab measures how much). a common refinement is a **smaller learning rate for the backbone** than for the new head: the head starts from random weights and needs large steps, while the backbone is already good and needs small adjustments.

**`fine_tune(model, train_loader, num_classes, *, mode="full", epochs=5, lr=1e-3, backbone_lr_mult=0.1, weight_decay=0.0, head_name="fc", device="cpu")`** does both with AdamW:

1. `replace_head(model, num_classes)` and move the model to the device;
2. `mode="linear"`: `freeze_backbone`, one parameter group with the head's parameters at `lr`, and in every epoch the whole model in `eval()` mode with only the head in `train()` mode;
3. `mode="full"`: every parameter trainable, two parameter groups (the backbone at `lr * backbone_lr_mult`, the head at `lr`), and the model in `train()` mode;
4. train for `epochs` epochs on `train_loader` with cross-entropy, and return `{"train_loss": [...]}`, the mean loss per example of each epoch. any other `mode` raises `ValueError`.

> **🔮 predict:** with 5 labelled images per class, will a linear probe or full fine-tuning give the better target accuracy? and at 500 per class, how close does a network trained from scratch get?

> **▶ build: stage 4.** implement `fine_tune`.

## worked example

**depthwise separable vs standard, 16 → 32 channels.** a standard $3 \times 3$ convolution has $16 \cdot 32 \cdot 9 = 4{,}608$ weights. `DepthwiseSeparableConv(16, 32)` has $16 \cdot 9 = 144$ depthwise and $16 \cdot 32 = 512$ pointwise weights, plus $2 \cdot 16 + 2 \cdot 32 = 96$ batch-norm parameters: **752**. the weight ratio is $656 / 4{,}608 = 0.142 = \frac{1}{32} + \frac{1}{9}$. at 128 channels on a $32 \times 32$ map: standard $147{,}456 \cdot 1{,}024 = 151.0$ million MACs, separable $(1{,}152 + 16{,}384) \cdot 1{,}024 = 18.0$ million, **8.4 times fewer**.

**a ConvNeXt block at `dim=16`.** depthwise $7 \times 7$: $49 \cdot 16 + 16 = 800$; layer norm $2 \cdot 16 = 32$; `pwconv1` $16 \cdot 64 + 64 = 1{,}088$; `pwconv2` $64 \cdot 16 + 16 = 1{,}040$; `gamma` 16: **2,976** parameters, two thirds of them in the MLP. at initialization $\gamma = 10^{-6}$, so the block's output differs from its input by about $10^{-6}$ times the branch's.

**what a linear probe trains.** the lab's small ResNet (one block per stage, base width 16) has a 128-dimensional `forward_features`; a probe for 5 classes trains $128 \cdot 5 + 5 = 645$ parameters, against 308,181 for full fine-tuning.

the tests check the parameter counts and the identity at initialization.

## on paper

1. **the cost ratio.** derive the MAC ratio of a depthwise separable to a standard convolution, and evaluate it for $k = 3$ with $C_\text{out} = 32$ and $C_\text{out} = 512$, and for $k = 7$ with $C_\text{out} = 512$.

<details><summary>solution</summary>

per output position, standard: $k^2 C_\text{in} C_\text{out}$; separable: $k^2 C_\text{in} + C_\text{in} C_\text{out}$; ratio $\frac{1}{C_\text{out}} + \frac{1}{k^2}$. $k = 3$: $0.142$ for $C_\text{out} = 32$, $0.113$ for 512 (8.9× fewer). $k = 7$: $0.0224$, 45× fewer: large depthwise kernels are nearly free compared with large standard ones, which is why ConvNeXt can afford $7 \times 7$.

</details>

2. **arithmetic intensity.** a depthwise $3 \times 3$ convolution on a $C \times H \times W$ float32 map reads $4 C H W$ bytes and writes as many; a standard $3 \times 3$ convolution $C \to C$ does the same memory traffic (plus its weights). compute MACs per byte for each, for $C = 128$. a CPU core does about 50 billion MACs per second and moves about 20 GB per second from memory: which one is limited by what?

<details><summary>solution</summary>

depthwise: $9 C H W$ MACs over $8 C H W$ bytes, about 1.1 MACs per byte. standard: $9 C^2 H W$ MACs, about $1.1 C = 144$ MACs per byte. the machine balances at $50 / 20 = 2.5$ MACs per byte: the depthwise convolution (1.1) waits for memory and runs at under half of peak at best (in practice much less, since its access pattern is less cache-friendly), while the standard one (144) is limited by arithmetic. so 8× fewer MACs can give a much smaller speedup, and the pointwise convolution, a plain matrix product, does most of the useful work.

</details>

3. **why a frozen backbone's batch norm must be in eval mode.** a batch norm layer in training mode updates $\hat\mu \leftarrow (1 - m)\hat\mu + m\,\mu_B$ with momentum $m = 0.1$. after training a probe for 300 steps on target data, how much weight do the original running statistics keep?

<details><summary>solution</summary>

$(1 - m)^{300} = 0.9^{300} \approx 2 \times 10^{-14}$: none. the running mean and variance are now estimates for the *target* data, so the backbone, evaluated afterwards in eval mode, normalizes source images with target statistics. its features (and anything built on them, like the source head) shift, although no weight changed. and with small batches, the running estimates are also noisy.

</details>

4. **how many numbers does each method fit?** with $n$ images per class and 5 classes, a linear probe fits $128 \cdot 5 + 5$ parameters and full fine-tuning about 308,000. for $n = 5$, compare each with the number of labels.

<details><summary>solution</summary>

25 labels. the probe has 645 parameters, 26 per label: underdetermined, but the features are fixed and weight decay or early stopping keep it sensible. fine-tuning has 12,000 parameters per label: it can memorize the 25 images in a few steps, and the only thing that keeps it near a good solution is that it *starts* near one (the pretrained weights) and moves slowly (a small learning rate, few steps). that's why the backbone learning rate and the number of steps matter so much in the low-data regime.

</details>

## experiment

`lab.py` adapts networks to a new task within CIFAR-10:

1. the parameters, MACs and measured CPU time of a standard convolution, your depthwise separable block and your ConvNeXt block.
2. a backbone pretrained on five CIFAR classes, adapted to the other five with 5, 50 or 500 labels per class: linear probe, fine-tune (two ways), or train from scratch.
3. what each way of adapting does to the backbone's performance on the *original* task.

## check your understanding

1. why does a depthwise convolution need a pointwise convolution after it?

<details><summary>answer</summary>

the depthwise step never mixes channels: output channel $c$ depends only on input channel $c$. without a channel-mixing step, a stack of depthwise convolutions would process each channel independently forever, and features couldn't combine (a "horizontal edge in red" detector couldn't be built from red and edge information). the $1 \times 1$ convolution provides the mixing.

</details>

2. what does `groups` do in `nn.Conv2d`, and what are `groups=1` and `groups=in_channels`?

<details><summary>answer</summary>

it splits the input and output channels into `groups` equal groups and convolves each group of inputs only into its own group of outputs; the weight has shape $(C_\text{out}, C_\text{in}/g, k, k)$. `groups=1` is a standard convolution; `groups=in_channels` (with `out_channels = in_channels`) is a depthwise convolution. values in between (ResNeXt's grouped convolutions) trade cost against mixing.

</details>

3. ConvNeXt's MLP expands the channels 4×; ResNet's bottleneck reduces them 4×. why can ConvNeXt afford the expansion?

<details><summary>answer</summary>

its spatial filtering is depthwise, which is cheap at any width, so the expensive part of the block is the two $1 \times 1$ layers, which are plain matrix products that run at full hardware speed. ResNet's bottleneck reduces the channels to make its *dense* $3 \times 3$ convolution affordable; ConvNeXt has no dense $3 \times 3$ convolution to protect.

</details>

4. what does layer scale do at initialization, and why is that useful?

<details><summary>answer</summary>

with $\gamma = 10^{-6}$ the residual branch's contribution is negligible, so every block starts as (almost) the identity and a deep stack starts as a shallow function of its input. as in topic 04, this keeps the signal and gradients well-behaved at the start of training; the network then grows each block's contribution as it learns.

</details>

5. you fine-tune a pretrained ResNet with batch size 4. what goes wrong with batch norm, and what can you do?

<details><summary>answer</summary>

batch statistics over 4 images are very noisy, so training-mode normalization is noisy and the running statistics drift towards a poor estimate. options: keep the batch-norm layers in eval mode (use the pretrained statistics, only train the affine parameters or nothing), freeze them entirely, replace them with group norm, or accumulate a larger effective batch.

</details>

6. when would you choose a linear probe even though fine-tuning gives higher accuracy on the new task?

<details><summary>answer</summary>

when the backbone must keep serving other tasks (one frozen backbone, many cheap heads), when compute or memory for training is limited (no backward pass through the backbone), when you need the features to stay comparable over time, or when the new data are so few or so noisy that fine-tuning is unstable. it's also the standard way to *evaluate* how good a representation is (chapter 25).

</details>

## common bugs

- **forgetting `groups`.** symptom: the "depthwise" convolution has $C^2 k^2$ weights and costs as much as a standard one.
- **normalizing over the wrong axes in ConvNeXt.** symptom: the block's output differs from the spec; layer norm must normalize each pixel's channel vector, so apply it channels-last (or use a channels-first layer norm explicitly).
- **matching the head by prefix.** symptom: a backbone layer called `fc_pre` stays trainable. compare names with `head_name` exactly or with `head_name + "."`.
- **a "frozen" backbone in training mode.** symptom: no weights changed, but the backbone's running statistics did, and its accuracy on the original task dropped. run frozen parts in eval mode.
- **one learning rate for a random head and a pretrained backbone.** symptom: either the head learns too slowly or the backbone is wrecked in the first steps; use separate parameter groups.
- **the new head on the wrong device or dtype.** symptom: a device or dtype mismatch error on the first forward pass after `replace_head`; create the new layer like the old one.

## going deeper

- Howard et al. (2017), "MobileNets: efficient convolutional neural networks for mobile vision applications". [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
- Sandler et al. (2018), "MobileNetV2: inverted residuals and linear bottlenecks", *CVPR*. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)
- Liu et al. (2022), "A ConvNet for the 2020s", *CVPR*. ConvNeXt. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Tan & Le (2019), "EfficientNet: rethinking model scaling for convolutional neural networks", *ICML*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Yosinski, Clune, Bengio & Lipson (2014), "How transferable are features in deep neural networks?", *NeurIPS*. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
- Kumar et al. (2022), "Fine-tuning can distort pretrained features and underperform out-of-distribution", *ICLR*. linear probing, fine-tuning, and LP-FT. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- Williams et al. (2009), "Roofline: an insightful visual performance model for multicore architectures", *Communications of the ACM* 52(4). arithmetic intensity. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785)

## connections

- ← builds on: grouped convolution and MAC counting (topics 01 and 03), residual blocks and `fc`/`forward_features` (topic 04), layer norm, GELU, AdamW and parameter groups (ch 15), batch norm's running statistics (ch 15 · topic 02).
- → used in: backbones for detection and segmentation (topic 06), vision transformers and their MLP blocks (ch 18), FID features from a pretrained network (ch 21), linear-probe evaluation of self-supervised representations (ch 25), parameter-efficient fine-tuning and LoRA for large models (ch 27), roofline analysis of kernels (ch 26).
