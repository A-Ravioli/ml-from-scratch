# pooling & receptive fields · lab notes

numbers are for real MNIST (12,000 training and 5,000 test digits, standardized pixels), measured with the reference solution on an Apple M2 laptop CPU (load average about 10 from other jobs; the lab took 60 s). experiments 1 and 3 don't use data at all (random inputs through random networks), so they give the same numbers offline. experiment 2's accuracies vary by a point or two with the seed and the data order; the shape of the curves doesn't. offline (the synthetic stand-in that `verify.py` uses) experiment 2 runs on smooth blobs and its numbers say nothing about digits.

## experiment 1: how far does a deep network actually look?

**what people usually predict:** the effective field is the theoretical one, or a bit smaller, say radius 25–30 for the 32-layer stack.

**what happens:** with ReLUs (the "linear" option gives identical radii):

| depth L | theoretical rf | theoretical radius | 90% ERF radius | ratio |
|---|---|---|---|---|
| 2 | 5 | 2 | 2 | 1.00 |
| 4 | 9 | 4 | 4 | 1.00 |
| 8 | 17 | 8 | 6 | 0.75 |
| 16 | 33 | 16 | 9 | 0.56 |
| 32 | 65 | 32 | 13 | 0.41 |

the 32-layer stack puts 90% of its sensitivity within 13 pixels of the centre, a $27 \times 27$ square inside its $65 \times 65$ field (two other initializations give 12 and 13). the radius grows like $\sqrt{L}$ (6 → 9 → 13 as $L$ doubles), not like $L$; the maps are round, bell-shaped blobs, not squares.

**why:** the gradient of the centre output reaches an input pixel along every path through the network, and each layer moves a path by an offset of $-1$, $0$ or $+1$ in each direction. pixels near the centre are reached by vastly more paths than pixels at the edge of the field, which are reached only by the single path that steps outward at every layer. the distribution of a sum of $L$ independent small offsets is (central limit theorem) a Gaussian whose width grows like $\sqrt{L}$, so the fraction of the theoretical field that matters shrinks like $1/\sqrt{L}$. random weights and ReLUs change the path weights but not the averaging argument, which is why linear and ReLU stacks give the same radii. for a network that must relate two objects 50 pixels apart, a theoretical field of 100 is not enough of a guarantee: the far end of the field carries almost no gradient at initialization, which is one reason architectures downsample (strides) or use dilation and attention instead of just stacking more layers.

## experiment 2: is a CNN translation-invariant?

**what people usually predict:** convolutions plus global average pooling are invariant, so nothing changes for small shifts; maybe a little for large ones.

**what happens:** test accuracy after diagonal shifts of $s$ pixels:

| downsampling | shift 0 | shift 1 | shift 2 | shift 3 | shift 4 | shift 6 | predictions changed by a 1-px shift |
|---|---|---|---|---|---|---|---|
| max | 0.932 | 0.909 | 0.834 | 0.727 | 0.605 | 0.337 | 6.1% |
| avg | 0.877 | 0.851 | 0.730 | 0.521 | 0.302 | 0.136 | 8.3% |
| stride | 0.887 | 0.828 | 0.630 | 0.414 | 0.290 | 0.209 | 13.2% |

a single-pixel shift changes 6–13% of the predictions; four pixels cost 33–60 points of accuracy, and six pixels push all three networks most of the way to chance. max pooling is the most robust (and the most accurate unshifted), the strided convolution the least robust to small shifts. (an earlier run with 20,000 training digits gave 5.4%, 6.9% and 13.1% for the one-pixel changes: the same ranking.)

**why:** two effects. first, **aliasing**: every stride-2 step keeps one sample out of each $2 \times 2$ block, so a one-pixel shift changes *which* samples survive (on paper 4). after three such steps (a jump of 8), only shifts by multiples of 8 move the final feature map by a clean shift; anything else produces a genuinely different feature map, which the global average pool then averages into a different vector. max pooling suffers least because it keeps the window's strongest response wherever it sits within the window, so small shifts are partly absorbed; a learned stride-2 convolution samples one fixed position per window and is hit hardest (this is the effect Zhang, 2019, fixes by blurring before subsampling). second, **training data**: MNIST digits are centred by their centre of mass, so the networks never saw a digit 4–6 pixels off-centre, partly cut off by the border and next to zero padding where its strokes used to be. global pooling is only invariant to shifts of the features, and features of a digit touching the border are different features. the practical lesson: equivariance is a property of single convolutions; whole CNNs are only approximately invariant, and the invariance they do have comes as much from **augmentation** (random crops, which the capstone uses) as from the architecture.

## experiment 3: three ways to see farther

**what people usually predict:** the dilated stack uses its large theoretical field less well (gaps between the taps), or the strided one does (fewer units).

**what happens:**

| stack | theoretical rf | jump | theoretical radius | 50% ERF radius | 90% ERF radius | 90% radius / theoretical |
|---|---|---|---|---|---|---|
| plain | 17 | 1 | 8 | 3 | 6 | 0.75 |
| strided | 51 | 4 | 25 | 11 | 19 | 0.76 |
| dilated | 61 | 1 | 30 | 15 | 24 | 0.80 |

all three use about the same fraction of their theoretical field (0.75–0.80), so the dilated and strided stacks see 3–4 times farther *effectively* as well, for the same 8 layers and the same weights. the dilated map is a smooth blob with no visible gaps; the strided map is coarser (its input gradient comes through only $33 \times 33$ output positions).

**why:** the fraction is set by how many layers the paths pass through (8 here, against 32 in experiment 1's 0.41), not by how the field grows: each layer still adds one random offset, just a larger one (scaled by the dilation or by the jump). strides are the cheapest way to grow the field, because every later layer's span is multiplied by the jump, and they reduce compute by $4\times$ per stride-2 layer, but the price is resolution: the strided stack produces a $33 \times 33$ output for a $129 \times 129$ input, fine for classification and a problem for segmentation. dilation keeps full resolution; mixing rates $1, 2, 4, 8$ fills in the gaps between taps (each rate covers the holes left by the larger ones). if every layer used dilation 8, the centre output would only ever see pixels whose offsets are multiples of 8 (a "gridding" artifact), which is why dilated networks for segmentation mix rates (Yu & Koltun, 2016; Wang et al., 2018).

## references

- Yu & Koltun (2016), "Multi-scale context aggregation by dilated convolutions", *ICLR*. [arXiv:1511.07122](https://arxiv.org/abs/1511.07122)
- Wang et al. (2018), "Understanding convolution for semantic segmentation", *WACV*. hybrid dilation rates against gridding. [arXiv:1702.08502](https://arxiv.org/abs/1702.08502)
