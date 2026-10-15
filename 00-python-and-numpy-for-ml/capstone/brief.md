# 00 · capstone: image filters from scratch

## goal

every photo app can blur, sharpen and find edges, and every one of those effects is the same operation underneath: **convolution**, sliding a small grid of weights (a *kernel*) across an image and taking a weighted sum at every position. you'll build it yourself, vectorized with NumPy so it runs in milliseconds instead of seconds, and then build the classic filters on top of it: a box blur, a Gaussian blur, unsharp-mask sharpening, the Sobel edge detector, and a complete blur-differentiate-threshold edge detector.

this capstone pulls together every topic in the chapter. images are arrays and the whole computation is broadcasting and slicing (topic 02); a blur kernel must sum to exactly 1 and PSNR lives in log space (topic 03); you'll look at everything with your own `image_grid` and `plot_curves` (topic 04); and you'll check your work against an independent reference, with tolerances, and evaluate on held-out images with error bars (topic 05). it also sets up chapter 16: a convolutional neural network is exactly this convolution, with kernels that are *learned* instead of designed by hand.

## data

two real datasets, both loaded through `mlfs.datasets`, plus a synthetic image with known edges:

```python
from mlfs import datasets
faces = datasets.olivetti_faces()[0]       # float64 (400, 64, 64) grayscale faces in [0, 1]
photos = datasets.cifar10()["X_test"]      # uint8 (10000, 32, 32, 3) small color photos
```

the first call downloads each file once and caches it under `~/.cache/mlfs/` (the CIFAR-10 archive is about 170 MB; if you'd rather not download it, everything works with the faces alone). with `MLFS_OFFLINE=1`, or with no network, you get small synthetic stand-ins with the same shapes, so the notebook always runs. the stand-ins are noisy blobs rather than photographs, so some results (notably the best amount of blur) differ from the real data. the tests never use either: they generate their own synthetic images.

the notebook also provides `make_shapes`, which draws rectangles and a disk on a gentle gradient and returns the exact edge map, so edge detection can be scored objectively.

## configurations

| config | data | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | synthetic stand-ins and synthetic shapes | 8 images of 64 × 64 (faces) and 32 × 32 × 3 (photos); test images up to 512 × 512 | a few seconds |
| **full** (your write-up) | real Olivetti faces and CIFAR-10 photos | 16 images of each | a few seconds once downloaded |

the notebook picks the config automatically from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `correlate2d(image, kernel)`: "valid" cross-correlation, vectorized over pixels, for grayscale and color images
   - `filter2d(image, kernel, *, padding="reflect")`: same-size convolution with zero, edge or reflect padding
   - `box_kernel(size)`, `gaussian_kernel(size, sigma)`, `unsharp_mask(image, *, sigma=1.0, amount=1.0)`
   - `sobel_edges(image)`: the gradient magnitude
   - `detect_edges(image, *, sigma=1.0, threshold=0.2)`: a boolean edge map
2. the write-up section at the end of the notebook, answering its five prompts (correctness, denoising with uncertainty, edges, padding, speed).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `correlate2d` on a 512 × 512 image with a 7 × 7 kernel | under 0.5 s (enforced by `test_stage1_is_vectorized`) |
| edge F1 (1-pixel tolerance) on clean synthetic shapes, `sigma=1, threshold=0.25` | ≥ 0.85 (enforced by `test_stage5_clean_edges_threshold`) |
| edge F1 on synthetic shapes with noise of std 0.25, `sigma=1.5, threshold=0.35` | ≥ 0.75, and at least 0.2 above the same detector without blur (enforced by `test_stage5_noisy_edges_threshold`) |
| denoising on real faces (full config), σ chosen on the tuning faces | a PSNR gain over the noisy input, reported on the held-out faces with its standard error |
| speed | report the speedup of your `correlate2d` over the provided pixel-loop version |

for reference, the reference solution in the small config gets: noisy input 20.0 dB; on the synthetic shapes image the best blur is σ = 0.75 at 25.3 dB (+5.4 dB); edge F1 0.93 on the clean shapes image, 0.36 on the noisy one without blur, 0.80 with σ = 1.5; and a speedup of about 30× at 32 × 32 rising to about 100× at 128 × 128 over the pixel loops. on the offline *face* stand-ins only the lightest blur helps (σ = 0.5, about +1.8 dB), and anything wider makes things worse, because the stand-ins are mostly pixel noise to begin with; that's expected, and a good thing to explain in your write-up if you work offline. on the real faces (full config, noise std 0.1), the reference picks σ = 1.0 on the tuning faces and gains **about +7 dB** on the report faces (+6.9 ± 0.2 dB). with milder noise (std 0.05) the best σ drops to 0.75 and the gain to about +4 dB.

## stretch goals

- **separable filters.** a Gaussian kernel is an outer product $g g^\top$, so blurring with it equals blurring the rows with $g$ and then the columns. implement it and measure the speedup for large $\sigma$: what's the cost per pixel of each version, as a function of the kernel width?
- **your own photo.** load one with `matplotlib.image.imread` (convert to float in [0, 1]), and apply every filter. what changes with image size?
- **Canny-style thinning.** keep only the edge pixels whose magnitude is a local maximum along the gradient direction (`np.arctan2(gy, gx)`), and compare the F1 with your thresholded edges.
- **median filtering.** add salt-and-pepper noise (random pixels set to 0 or 1) and compare a Gaussian blur with a 3 × 3 median filter (which isn't a convolution: use `np.lib.stride_tricks.sliding_window_view`). why does the median win?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Gonzalez & Woods (2018), *Digital Image Processing*, 4th ed., Pearson, chapter 3 (spatial filtering: smoothing, sharpening and unsharp masking, gradients and the Sobel operator).
- Szeliski (2022), *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, §3.2–3.3 (linear filtering, separable filters, padding) and §7.2 (edge detection). [free online](https://szeliski.org/Book/)
- Canny (1986), "A computational approach to edge detection", *IEEE TPAMI* 8(6): 679–698. why you blur before differentiating, derived from first principles. [doi:10.1109/TPAMI.1986.4767851](https://doi.org/10.1109/TPAMI.1986.4767851)
- Samaria & Harter (1994), "Parameterisation of a stochastic model for human face identification", *Proceedings of the 2nd IEEE Workshop on Applications of Computer Vision*: 138–142. the Olivetti/ORL faces. [doi:10.1109/ACV.1994.341300](https://doi.org/10.1109/ACV.1994.341300)
- Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images*, technical report, University of Toronto. the CIFAR-10 dataset. [pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
