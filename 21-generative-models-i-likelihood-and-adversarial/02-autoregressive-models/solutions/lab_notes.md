# autoregressive models · lab notes

numbers are for real MNIST (10,000 training and 2,000 test digits, shrunk to $14 \times 14$ and binarized with a fixed seed), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running (load average about 20; the lab took 57 s). timings depend on the machine; their *ratios* are what matter. offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end, but its numbers say nothing about MNIST.

## experiment 1: scoring is one pass, sampling is many

**what people usually predict:** a $28 \times 28$ sample takes $4t$ (four times the pixels), or, after on paper 4, $16t$.

**what happens:** the PixelCNN (32 channels, 5 layers) trained for 2 epochs in 36 s and reached a test NLL of **41.4 nats per image, 0.305 bits/dim** (exact, on these binarized $14 \times 14$ digits). timings:

| image | pixels | one forward pass (score) | one sample | sample time / 14×14's | sample / score |
|---|---|---|---|---|---|
| 7 × 7 | 49 | 0.41 ms | 0.02 s | 0.2 | 43 |
| 14 × 14 | 196 | 0.44 ms | 0.10 s | 1.0 | 222 |
| 21 × 21 | 441 | 0.56 ms | 0.28 s | 2.9 | 500 |
| 28 × 28 | 784 | 0.88 ms | 0.65 s | 6.7 | 740 |

doubling the side made sampling **6.7×** slower, between the two predictions, and sampling a $28 \times 28$ image cost **740 forward passes' worth** of time against one for scoring it. timings are machine- and load-dependent: a reviewer's rerun measured 4.4× and 669, so expect something in the range 4–7× and 600–800.

**why:** a sample needs one forward pass per pixel ($HW$ passes), and each pass costs $a + c\,HW$: a fixed overhead $a$ (Python, kernel launches, memory allocation) plus work proportional to the image. at these tiny sizes the overhead dominates: the forward pass at $28 \times 28$ is only twice as slow as at $14 \times 14$, not four times. so the measured scaling is closer to $HW$ than to $(HW)^2$ here, and it approaches the quadratic law only once the work per pass dominates (larger images, more channels). the ratio of sample to score time grows like the number of pixels either way: scoring is one parallel pass over the observed image; sampling must wait for each pixel before computing the next, because each conditional needs the *sampled* values before it. caching the activations that don't change between steps (fast PixelCNN++, transformer KV caches in chapter 18) cuts the work per step, but not the number of sequential steps.

## experiment 2: does the order matter?

**what people usually predict:** raster order wins clearly (it follows the image's structure), and a random order is much worse because early pixels have no neighbours to predict from.

**what happens:** test NLL of the MADE (one hidden layer of 500 units, 8 epochs; mean of 2 seeds, ± half their difference):

| order | test NLL (nats / image) | ± | bits/dim |
|---|---|---|---|
| raster (row by row) | 42.8 | 0.1 | 0.315 |
| reversed raster | 42.7 | 0.1 | 0.314 |
| random permutation | **40.9** | 0.1 | **0.301** |

the **random order wins**, by 1.9 nats per image, against a seed-to-seed spread of 0.1. it isn't a lucky permutation: three different random permutations gave 40.95, 40.89 and 41.15 nats (seed 0), all better than raster's 42.66. with 20 epochs instead of 8, the gap narrows but remains: raster 39.99, random 39.12–39.30. raster and its reverse are indistinguishable.

**why:** the chain rule makes all orders equivalent in principle, so the differences are about how hard the conditionals are to *learn* with a limited network and budget. in raster order, a pixel in the middle of a digit conditions on the rows above and its left neighbours, and must *extrapolate* the stroke downward with everything below it hidden. in a random order, half the pixels come after the halfway point, and each of those sees a random half of the image all around it: an *interpolation* problem, which is much easier (the late pixels are nearly determined by their neighbours on all sides). the early pixels of a random order are predicted from little context, so they are expensive; the measurement says the cheap late pixels more than pay for them. a MADE is fully connected and has no notion of locality, so raster order gives it no architectural advantage. a convolutional model like PixelCNN is a different story: its masks bake the raster order into the architecture, and its locality bias is what makes raster natural for it. MADE's authors exploited the order-agnostic view by training one network over many random orderings and ensembling them (Germain et al., 2015; Uria et al., 2016).

## experiment 3: the blind spot

**what people usually predict:** after enough layers, the centre pixel sees every pixel above it and to its left.

**what happens:** with 6 layers (a $7 \times 7$ type-A layer and five $3 \times 3$ type-B layers), the centre of a $15 \times 15$ image may use 112 pixels and actually depends on **102**: a staircase of **10 pixels** above and to the right is blind. adding layers enlarges the visible region upward and leftward, but the staircase to the upper right stays (move the slider).

**why:** a type-B $3 \times 3$ layer moves information one pixel up, left, or diagonally, and to the right only together with a step up (the kernel's top row). so after any number of layers, a pixel $k$ rows above the centre is visible only if it lies at most $k$ columns to the right (plus the first layer's reach, 3 columns for the $7 \times 7$ type-A kernel): the visible region is bounded by a $45°$ line, and everything to the right of it in the rows above is invisible however deep the stack (on paper 3). the $7 \times 7$ first layer pushes the line three pixels to the right, which hides the problem right next to the current pixel but not further up. strokes that rise to the right, like the top of a 7 or the diagonal of a 2, run straight through the blind spot. the gated PixelCNN (van den Oord et al., 2016b) removes it by combining a vertical stack (which sees all rows above, without the staircase) with a horizontal stack for the current row.

## references

- Uria, Côté, Gregor, Murray & Larochelle (2016), "Neural autoregressive distribution estimation", *JMLR* 17. orderings and order-agnostic training. [arXiv:1605.02226](https://arxiv.org/abs/1605.02226)
- van den Oord et al. (2016b), "Conditional image generation with PixelCNN decoders", *NeurIPS*. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328)
