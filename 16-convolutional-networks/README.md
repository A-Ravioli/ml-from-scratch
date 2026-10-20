# 16 · convolutional networks

> exploit spatial structure, and train modern vision models.

an MLP sees an image as 3,072 unrelated numbers: move a cat three pixels and, as far as its first layer is concerned, it's a different input. convolutional networks build in what you know about images (nearby pixels belong together, and a pattern means the same thing wherever it appears), and that single assumption is why they took over computer vision. this chapter builds them from the ground up. you'll write the convolution itself in NumPy, first as loops and then as the one matrix product every library uses, with its backward pass; pooling and the arithmetic of how far each unit can see (and a measurement of how far it actually looks); LeNet and VGG and a profiler that shows where their parameters and their compute go; residual networks, the one-line fix that made 100-layer networks trainable, as the backbone later chapters reuse; depthwise separable and ConvNeXt blocks and the routine that adapts a pretrained network to a new task; and the core of detection and segmentation: IoU, non-maximum suppression, anchors and a U-Net. the capstone trains your ResNet-18 to **92.9% on CIFAR-10** in 35 minutes on a laptop GPU. on the way you'll meet random filters that beat your hand-made ones, a 32-layer network that ignores most of what it can see, a CNN whose predictions change when a digit moves one pixel, a plain 34-layer network that fits its training data worse than an 18-layer one (and collapses when you delete any layer), a block with 8× fewer multiplications that's barely faster, a crowd in which no NMS threshold works, and a standard anchor grid that can't see anything under 16 pixels.

**you will build:** `conv_output_size`, `conv2d_naive`, `im2col`, `col2im`, `conv2d`, `conv2d_backward` in NumPy (topic 01); `max_pool2d`, `avg_pool2d` and their backward passes, `receptive_field`, `rf_layers`, `effective_receptive_field`, `erf_radius` (02); `LeNet`, `MiniVGG`, `profile_model` (03); `BasicBlock`, `ResNet` with `forward_features`, `resnet18`, `Bottleneck` (04); `DepthwiseSeparableConv`, `ConvNeXtBlock`, `replace_head`, `freeze_backbone`, `fine_tune` (05); `box_iou`, `nms`, `make_anchors`, `match_anchors`, `dice_loss`, `DoubleConv`, `UNet` (06). they make up **`mlfs.cnn`**: topics 01–02 in NumPy (checked against PyTorch), topics 03–06 in PyTorch. `ResNet`, `resnet18` and `forward_features` are reused by chapters 18 (ViT vs ResNet), 21 (FID features), 25 (self-supervised backbones) and 28 (attacks and attribution).
**prerequisites:** chapter 15 (`mlfs.torchnn`'s `fit`, `evaluate`, `make_loader`, `get_device`; `mlfs.train`'s `CosineSchedule`; batch norm, augmentation, weight decay, label smoothing, kaiming init), chapter 14 (backpropagation and VJPs), chapter 00's capstone (sliding hand-made filters over images), vectorization and strided slicing (ch 00 · topic 02), the bootstrap (ch 05), and precision, recall and class imbalance (ch 07).
**time · scaffolding:** about 75 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the conventions (shapes, orders, defaults, which layer gets the stride) are in the lessons. hints have two levels. the capstone gives you data loading and the experiment cells; the pipeline is yours.
**newly unlocked library components:** from chapter 17 on, and inside this chapter from the topic after you build them: `torch.nn.Conv2d`, `F.conv2d`, `F.unfold`/`F.fold` (topic 01); `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d` and their functional forms (02); `torchvision.models.vgg*`/`alexnet` and FLOP counters such as `torch.utils.flop_counter` (03); `torchvision.models.resnet*` and their blocks (04); grouped convolutions (`groups=`), `torchvision`'s MobileNet, EfficientNet and ConvNeXt models, and pretrained weights for transfer learning (05); `torchvision.ops.box_iou`, `nms` and anchor generators, `nn.ConvTranspose2d`, and `torchvision`'s detection and segmentation models (06). inside `mlfs`, `mlfs.cnn`'s NumPy convolution stays the reference for what `nn.Conv2d` computes.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [convolution](01-convolution/) | `conv_output_size`, `conv2d_naive`, `im2col`, `col2im`, `conv2d`, `conv2d_backward` → `mlfs.cnn` | a tiny CNN trained with your NumPy conv learns Sobel-like edge filters, yet 8 frozen *random* filters beat your 8 hand-made ones (0.917 vs 0.878) and come within 4.5 points of learned ones |
| 02 | [pooling & receptive fields](02-pooling-and-receptive-fields/) | `max_pool2d`, `avg_pool2d` (+ backward), `receptive_field`, `rf_layers`, `effective_receptive_field`, `erf_radius` → `mlfs.cnn` | a 32-layer stack puts 90% of its sensitivity in the central 27×27 of its 65×65 field (radius grows like √L); moving a digit one pixel changes 6–13% of a CNN's predictions |
| 03 | [classic architectures](03-classic-architectures/) | `LeNet`, `MiniVGG`, `profile_model` → `mlfs.cnn` | VGG-16 keeps 89% of its parameters in fully connected layers that do 1% of its compute; shuffling the pixels drops LeNet below a same-sized MLP, which doesn't notice; without batch norm, an 8-layer VGG doesn't train at all |
| 04 | [residual networks](04-residual-networks/) | `BasicBlock`, `ResNet`, `resnet18`, `Bottleneck` → `mlfs.cnn` | the plain 34-layer net's gradients *explode* 45× towards the input; it fits its training data worse than plain-18, and deleting any one block sends it most of the way to chance while the residual-34 loses 4.5 points on average |
| 05 | [modern convnets & transfer](05-modern-convnets-and-transfer/) | `DepthwiseSeparableConv`, `ConvNeXtBlock`, `replace_head`, `freeze_backbone`, `fine_tune` → `mlfs.cnn` | 8.4× fewer MACs buys only a 1.2–2× speedup, and the ConvNeXt block is 2–5× slower; a "frozen" backbone probed in `train()` mode on dusk images loses 14–18 points on its original task, as much as full fine-tuning, without a single weight changing |
| 06 | [detection & segmentation](06-detection-and-segmentation/) | `box_iou`, `nms`, `make_anchors`, `match_anchors`, `dice_loss`, `DoubleConv`, `UNet` → `mlfs.cnn` | in a crowd no NMS threshold reaches 95% recall at 90% precision; a standard anchor grid matches no object under 16 pixels; Dice alone gives the worst masks, while adding it to cross-entropy finds more tiny objects |

## capstone

*CIFAR-10 with your ResNet*: per-channel normalization, batched random-crop and flip augmentation on the device, your ResNet-18, SGD with Nesterov momentum and weight decay on weights only, your warmup + cosine schedule, label smoothing and your `fit`, then an ablation table with paired bootstrap intervals. the reference reaches **0.929 test accuracy** (95% CI [0.924, 0.934]; a second seed: 0.930) after 20 epochs in 35 minutes on an Apple M2 GPU; in the ablations (a narrower network, 10 epochs), dropping the learning-rate schedule costs 17 points and augmentation 3, while label smoothing and the residual connections are each worth under a point at this budget, within the seed-to-seed noise, and weight decay about 1.2 points. the brief documents the path to 94–95% (longer schedules on a GPU). small CPU config: about a minute on the offline synthetic stand-in.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a convolution and its three gradients, ResNet-18's receptive field, IoU, NMS and Dice), and a 60-minute blind rebuild of convolution with im2col.

## notation

| symbol | meaning |
|---|---|
| $N$, $C$, $H$, $W$ | batch size, channels, height, width (`NCHW` layout) |
| $C_\text{in}$, $C_\text{out}$ | input and output channels of a layer |
| $k$, $k_h \times k_w$ | kernel size |
| $s$, $p$, $d$ | stride, padding, dilation |
| $H_\text{out}$, $W_\text{out}$, $L$ | output height and width, $L = H_\text{out} W_\text{out}$ output positions |
| $w$, $W$ | a kernel $(C_\text{out}, C_\text{in}, k_h, k_w)$; its flattening $C_\text{out} \times C_\text{in} k_h k_w$ |
| $X_n = \operatorname{im2col}(x)_n$ | the column matrix of image $n$, $(C_\text{in} k_h k_w) \times L$ |
| $G$ | upstream gradient $\partial L/\partial y$ |
| $g$ | groups of a convolution (`groups=`) |
| $r_l$, $j_l$ | receptive field and jump of layer $l$, in input pixels |
| MACs | multiply–accumulates (about half the FLOPs) |
| $F(x)$ | a residual branch; a block outputs $\operatorname{relu}(x + F(x))$ |
| $\gamma$ (layer scale) | ConvNeXt's per-channel factor on the residual branch |
| xyxy | a box as $(x_1, y_1, x_2, y_2)$, top-left and bottom-right corners |
| IoU | intersection over union of two boxes or masks |
| $a$, $r$ | anchor size (area $a^2$) and aspect ratio (height / width) |
| $p_i$, $t_i$, $\epsilon$ | predicted probability and target of pixel $i$; Dice smoothing |

## how this connects

- builds on: your autograd and VJPs (ch 14), the training recipe and `mlfs.torchnn` (ch 15), chapter 00's image filters and vectorization, the bootstrap (ch 05), class imbalance and precision/recall (ch 07).
- used later in: sequence models' 1-D convolutions (ch 17); patch embeddings and vision transformers, whose MLP blocks ConvNeXt borrowed (ch 18); signal propagation in deep residual networks (ch 19); convolutional encoders and decoders, FID computed from `forward_features` (ch 21); U-Nets as the denoisers of diffusion models (ch 22); ResNet backbones for self-supervised learning and linear-probe evaluation (ch 25); FLOPs, memory and kernels (ch 26); adversarial attacks and attribution maps on these models (ch 28).

## core references

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, chapter 9 (convolutional networks). [deeplearningbook.org](https://www.deeplearningbook.org)
- Zhang, Lipton, Li & Smola (2023), *Dive into Deep Learning*, chapters 7, 8 and 14 (convolutions, modern CNNs, computer vision). [d2l.ai](https://d2l.ai)
- Dumoulin & Visin (2016), "A guide to convolution arithmetic for deep learning". [arXiv:1603.07285](https://arxiv.org/abs/1603.07285)
- Simonyan & Zisserman (2015), "Very deep convolutional networks for large-scale image recognition", *ICLR*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- He, Zhang, Ren & Sun (2016), "Deep residual learning for image recognition", *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Liu et al. (2022), "A ConvNet for the 2020s", *CVPR*. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
