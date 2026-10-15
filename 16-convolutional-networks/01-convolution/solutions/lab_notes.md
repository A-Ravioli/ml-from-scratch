# convolution · lab notes

numbers are for real MNIST (20,000 training and 5,000 test images, standardized pixels), measured with the reference solution on an Apple M2 laptop CPU while other jobs were running. experiment 1's timings depend on the machine and its load; compare ratios, not milliseconds. offline (the synthetic stand-in that `verify.py` uses) the lab runs end to end and the rankings in experiment 3 can change: the synthetic "digits" are smooth blobs that any filter bank separates.

## experiment 1: loops, a matrix product, and PyTorch

**what people usually predict:** the matrix version is 100–1,000× faster than the loops, and im2col's memory is "a bit more" than the input.

**what happens:** one forward pass, 16 → 16 channels, $3 \times 3$, batch 8, float64:

| size | naive (ms) | im2col (ms) | torch (ms) | naive / im2col | im2col memory / input |
|---|---|---|---|---|---|
| 8 | 0.6 | 0.11 | 0.19 | 5× | 9.0× |
| 16 | 2.5 | 0.28 | 0.24 | 9× | 9.0× |
| 32 | 10.3 | 0.83 | 0.96 | 12× | 9.0× |
| 64 | 44.7 | 14.8 | 4.8 | 3× | 9.0× |

the im2col version is 5–12× faster than the loops, not 100×, and on this run the gap shrank to 3× at $64 \times 64$. im2col's matrix is exactly 9 times the size of the input. PyTorch matches your im2col version on small images and was 3× faster on the largest. these timings are machine- and load-dependent: on another M-series machine a reviewer measured ratios of 3, 14, 15 and 20× that *grow* with size, with PyTorch slower than the im2col version at 16 and 32 pixels. what holds everywhere: one-to-two orders of magnitude, not three, and the 9× memory.

**why:** your "naive" version isn't the textbook seven nested loops: it loops over the $H_\text{out} W_\text{out}$ output positions but vectorizes over the batch, the channels and the kernel with one `tensordot` per position. its cost is a fixed Python and NumPy overhead per position (a few microseconds), times 4,096 positions at $64 \times 64$. the im2col version runs a few dozen Python lines regardless of size, so at small sizes it wins by the overhead it removes. at $64 \times 64$ the column matrix is $8 \times 144 \times 4{,}096$ float64 numbers, 38 MB, and on a machine with smaller caches or more memory contention the time goes into *moving memory*: writing the matrix, then streaming it through the matrix product, which no longer fits in the CPU's caches. that's the $k^2$ memory cost of idea 2 showing up as time. the ratio is $k_h k_w$ because every pixel is copied once per kernel tap (minus the few windows that hang over the zero padding, which the padded input already accounts for); a $7 \times 7$ kernel would make it 49×. PyTorch doesn't materialize the whole matrix: its CPU kernels work on blocks that stay in cache (and pick among several algorithms per shape), which is why it can pull ahead where your version becomes memory-bound. (with seven nested Python loops instead of one `tensordot` per position, the naive version would be about 1,000× slower again.)

## experiment 2: what does a convolutional layer learn?

**what people usually predict:** either clean, recognizable edge detectors, or noise.

**what happens:** after 2 epochs (11 s with your NumPy convolution) the network scores **0.962** on the test set. the eight learned $5 \times 5$ filters are smooth, mostly two-signed patterns: four of them are oriented edge detectors whose nearest hand-made filter is a Sobel (x or y, either sign), one is a diagonal edge, two are centre-weighted blobs nearest to the gaussian, and one is a nearly flat, low-contrast filter nearest to the box blur. their best $|\cos|$ with the hand-made bank: 0.41, 0.48, 0.43, 0.32, 0.50, 0.63, 0.40, 0.55 (median 0.46), against about 0.33 for the best match of a random $5 \times 5$ filter (0.16 for a single random pair). no learned filter looks like the laplacian or the sharpen kernel.

**why:** the filters resemble edge detectors because edges are where the information is: digits are strokes, and a stroke's identity is mostly its orientation and position. nothing told the network about edges; gradient descent found that oriented, two-signed filters followed by a ReLU and an average pool produce features that make the linear classifier's job easy. they're smoother and larger than Sobel (which is a $3 \times 3$ filter padded to $5 \times 5$, so the cosine can't be close to 1) because strokes are 2–3 pixels wide and the network matched its filters to that scale. the blob-like filters with a large mean act like "is there ink here?" detectors, which the ReLU turns into a soft mask. filters like the laplacian or sharpen, which respond to fine detail and noise, aren't useful at this resolution after a $4 \times 4$ average pool. in deep networks trained on natural images, the first layer famously learns oriented edges and colour blobs, very much like Gabor filters (Krizhevsky et al., 2012; Zeiler & Fergus, 2014): the same story at scale.

## experiment 3: do the filters need to be learned at all?

**what people usually predict:** learned > hand-made > random, with random far behind (near chance or a weak linear model).

**what happens:** with 8 filters each:

| filters | trained? | test accuracy (± 1 s.e.) |
|---|---|---|
| 8 learned | yes | **0.962** ± 0.003 |
| 8 hand-made | frozen | 0.878 ± 0.005 |
| 8 random | frozen | **0.917** ± 0.004 |
| majority class | | 0.114 |

random noise beats your hand-made filters by 4 points, and is only 4.5 points behind learned ones. other random banks (seeds 3–5) give 0.908–0.917, and the hand-made bank with a different head initialization 0.894: the ranking is stable. with 32 frozen filters (the dropdown), random noise reaches 0.932, within 3 points of the 8 learned filters, while the hand-made bank padded with 24 small random filters reaches only 0.892.

**why:** most of the work is done by the architecture, not the filter values. any bank of local filters followed by a ReLU and pooling produces a few hundred nonlinear, spatially organized features, and the trained linear layer on top picks a good combination: random features plus a learned readout are a surprisingly strong baseline (Jarrett et al., 2009; Saxe et al., 2011). the hand-made bank loses to random noise because it is *redundant*: box blur and gaussian are similar (cosine 0.70), the two diagonals are sums and differences of sobel x and y (cosine ±0.67), and laplacian and sharpen are almost the same filter with opposite signs (cosine −0.93), both emphasizing single-pixel detail that the pooling averages away. after the ReLU, which keeps one polarity of each filter, the bank covers only a handful of distinct directions of the 25-dimensional patch space. random filters are diverse by construction, so 8 of them cover more directions of the $5 \times 5$ patch space. learning still wins: it tailors the filters to strokes and to each other, with no redundancy. the gap grows with depth and harder images, because deeper layers' useful features are compositions that random weights almost never produce: random features stop being competitive somewhere between MNIST and CIFAR-10, which is part of what topics 03 and 04 show.

## references

- Jarrett, Kavukcuoglu, Ranzato & LeCun (2009), "What is the best multi-stage architecture for object recognition?", *ICCV*. random filters with the right nonlinearity and pooling work surprisingly well. [doi:10.1109/ICCV.2009.5459469](https://doi.org/10.1109/ICCV.2009.5459469)
- Saxe et al. (2011), "On random weights and unsupervised feature learning", *ICML*. [icml.cc/2011/papers/551_icmlpaper.pdf](https://icml.cc/2011/papers/551_icmlpaper.pdf)
- Zeiler & Fergus (2014), "Visualizing and understanding convolutional networks", *ECCV*. [arXiv:1311.2901](https://arxiv.org/abs/1311.2901)
