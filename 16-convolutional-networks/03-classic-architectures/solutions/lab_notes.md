# classic architectures · lab notes

numbers are for real MNIST and CIFAR-10, measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 10; the whole lab took 110 s). accuracies vary by a point or so with the seed; the comparisons don't. offline (the synthetic stand-ins that `verify.py` uses), experiment 1 is unchanged (it uses no data) and experiments 2–3 run on smooth blobs, where every model does well.

## experiment 1: where do the parameters live, and where does the time go?

**what people usually predict:** parameters and compute are spread roughly in proportion, or most of both sit in the convolutions because there are more of them.

**what happens:**

| network | conv + linear parameters | in fully connected layers | MACs per image | in conv layers |
|---|---|---|---|---|
| LeNet-5 | 61,706 | 95.8% | 0.4 M | 85.9% |
| MiniVGG (32, 64, 128) | 287,850 (288,746 with batch norm) | 0.4% | 38.6 M | 100.0% |
| VGG-16 | 138,357,544 | 89.4% | 15,470 M | 99.2% |

VGG-16 keeps 89% of its parameters in its three fully connected layers (74% in the first one alone) and does 99% of its compute in its 13 convolutions. LeNet is the same picture in miniature. your MiniVGG, with global average pooling and a single small linear layer, has essentially all its parameters *and* compute in convolutions, and 480 times fewer parameters than VGG-16.

**why:** a weight's compute cost is the number of times it's used per image. a convolutional weight is used at every output position: in VGG-16's second layer, $224 \times 224 = 50{,}176$ times. a fully connected weight is used exactly once. so the 103 million weights of `fc14` cost 103 million MACs, while the 37,000 weights of `conv2` cost 1.85 billion. the fully connected layers' size also depends on the input resolution: at $448 \times 448$ the last feature map is $14 \times 14 \times 512$ and the original VGG-16's `fc14` would need four times as many inputs (that network doesn't run on larger images without changes; torchvision's `vgg16` inserts an `AdaptiveAvgPool2d(7)` before the classifier for exactly this reason). MiniVGG's global average pooling removes that dependence, which is why it took over (Network in Network, then ResNet, topic 04). for training speed, the MACs (and memory traffic) matter; for the size of the checkpoint and the optimizer state, the parameters do. VGG-16 is the worst of both worlds: a large file *and* slow.

## experiment 2: what does a convolutional network assume?

**what people usually predict:** LeNet wins on the original digits; on permuted digits both suffer, or LeNet still wins because it's "better at images".

**what happens:** two epochs on 20,000 digits, tested on 5,000:

| pixels | model | parameters | test accuracy |
|---|---|---|---|
| original | LeNet | 61,706 | **0.9668** |
| original | MLP (77 hidden) | 61,225 | 0.9234 |
| permuted | LeNet | 61,706 | 0.8902 |
| permuted | MLP (77 hidden) | 61,225 | **0.9238** |

on real digits LeNet beats the same-sized MLP by 4.3 points. shuffle the pixels and LeNet loses 7.7 points, falling well *below* the MLP, while the MLP's accuracy doesn't change (0.9234 vs 0.9238: noise).

**why:** the MLP treats its 784 inputs as an unordered set: permuting them just permutes the columns of its first weight matrix, and since its initialization and Adam's update are both symmetric under that relabelling, training on permuted digits is the same experiment in different coordinates (on paper 4). LeNet assumes that *neighbouring* pixels belong together (local $5 \times 5$ kernels) and that the same pattern matters anywhere (weight sharing). on real digits those assumptions are true and save it from having to learn them, which is worth 4 points with the same number of parameters. after the permutation, neighbouring pixels are unrelated, so its $5 \times 5$ windows see random pixel subsets, its max pooling merges unrelated positions, and its strong assumptions become wrong ones: it still learns something (89%, because the fully connected layers can partly undo the damage), but less than a model that assumed nothing. on tabular data (patient records, where column order is arbitrary) a CNN would be in exactly this position: inductive bias helps only when it matches the data.

## experiment 3: depth vs accuracy on CIFAR-10

**what people usually predict:** accuracy rises with depth in both cases, a little less without batch norm.

**what happens:** 2 epochs on 6,000 images, tested on 2,000 (± about 0.011):

| conv layers | widths | with batch norm (final train loss) | without batch norm (final train loss) |
|---|---|---|---|
| 2 | (8,) | 0.316 (1.89) | 0.241 (2.04) |
| 4 | (8, 16) | 0.270 (1.72) | 0.227 (2.14) |
| 8 | (8, 16, 32, 64) | **0.346** (1.71) | **0.098** (2.30) |

without batch norm, *every* depth is worse than with it, and the 8-layer network doesn't train at all: its training loss is $2.30 = \ln 10$, the loss of a uniform guess, and its test accuracy is chance. that's the robust result. with batch norm, the deepest network is best and the training loss falls with depth, but the test accuracies at this tiny budget don't rise monotonically (the 4-layer net dips below the 2-layer one here, by more than the test set's noise): two epochs are too few for a reliable depth ranking. an earlier, longer version of this experiment (3 epochs on 10,000 images, with a 6-layer row) gave 0.359, 0.458, 0.448 and 0.478 with batch norm and 0.309, 0.311, 0.287 and 0.098 without: the same failure, and a clearer benefit of depth.

**why:** with batch norm, each extra block doubles the receptive field and the capacity, and the final features summarize more of the image (after 2 layers and a pool each feature summarizes a $6 \times 6$ patch before the global average; after 8 layers and four pools, a $76 \times 76$ field that covers the whole $32 \times 32$ image). the short budget limits how much the deeper ones can show, and makes the ranking among the batch-norm networks noisy. without batch norm the deep network fails to *train*, not to generalize: look at its training loss. PyTorch's default convolution initialization (kaiming-uniform with $a = \sqrt 5$) has variance $\frac{1}{3 \cdot \text{fan\_in}}$, a sixth of the $\frac{2}{\text{fan\_in}}$ that ReLUs need (chapter 15, topic 01), so the activations shrink by a factor of about $\sqrt 6$ per layer and the gradients reaching the early layers are tiny; with SGD at learning rate 0.05 the 8-layer network never leaves its starting point. batch norm rescales every layer's output to unit variance, so the initialization's scale stops mattering. the VGG authors (who had neither batch norm nor kaiming init) trained their 16- and 19-layer networks by initializing them from a trained 11-layer one; today you'd use kaiming init and batch norm, and for really deep networks, the residual connections of topic 04.
