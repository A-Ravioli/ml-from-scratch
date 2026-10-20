import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.utils import image_grid, plot_curves  # yours, from topic 04

    # small CPU config (offline: synthetic stand-ins, used by tests and verify) vs full config
    OFFLINE = datasets.is_offline()
    N_FACES = 8 if OFFLINE else 16


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 00 · capstone: image filters from scratch

    read [`brief.md`](brief.md) first. you'll build the core of an image-processing library:
    a vectorized 2-D convolution, then blurring, sharpening and edge detection on top of it.
    the data loading, the displays and the evaluation are written for you. the functions in
    the stages below are yours.

    | stage | you build |
    |---|---|
    | 1 | `correlate2d`: slide a kernel over an image, vectorized |
    | 2 | `filter2d`: true convolution, same-size output, three padding modes |
    | 3 | `box_kernel`, `gaussian_kernel`, `unsharp_mask`: blur and sharpen |
    | 4 | `sobel_edges`: the gradient magnitude of an image |
    | 5 | `detect_edges`: blur, differentiate, threshold |

    when they're green, scroll down to run the experiments on real images and write up what
    you find.
    """)
    return


@app.function
def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """stage 1: 'valid' cross-correlation of an image with a small kernel, vectorized.

    image: (H, W) grayscale or (H, W, C) color; kernel: (kh, kw) with kh <= H and kw <= W.
    returns a float array (H - kh + 1, W - kw + 1) (or (..., C): each channel separately) with
        out[i, j] = sum over a, b of kernel[a, b] * image[i + a, j + b]
    (no flipping, no padding). raises ValueError if the kernel isn't 2-D or is larger than
    the image. loop over the kh * kw kernel entries only, never over pixels.
    """
    # step 1: convert both to float arrays; check kernel.ndim == 2 and that it fits in the image
    # step 2: the output size: oh = H - kh + 1, ow = W - kw + 1; start from zeros (keep any channel axis)
    # step 3: for each kernel entry (a, b), add kernel[a, b] times the image shifted by (a, b):
    # step 3: the slice image[a:a + oh, b:b + ow] lines up pixel (i + a, j + b) with output (i, j)
    raise NotImplementedError("stage1: implement correlate2d")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", correlate2d=correlate2d)
    return


@app.function
def filter2d(image: np.ndarray, kernel: np.ndarray, *, padding: str = "reflect") -> np.ndarray:
    """stage 2: same-size 2-D CONVOLUTION of an image with an odd-sized kernel.

    convolution flips the kernel (kernel[::-1, ::-1]) and then correlates. the image is first
    padded by kh // 2 rows and kw // 2 columns on each side (spatial axes only), so the
    output has the image's shape. padding: "zero" (np.pad mode "constant"), "edge" (repeat
    the border pixel) or "reflect" (mirror without repeating the border pixel).
    raises ValueError for an even kernel size or an unknown padding mode.
    """
    # step 1: convert the kernel; check both of its sizes are odd and the padding name is known
    # step 2: pad with np.pad: ((ph, ph), (pw, pw)) for the rows and columns, plus (0, 0) for a channel axis
    # step 3: correlate the padded image with the FLIPPED kernel (your correlate2d)
    raise NotImplementedError("stage2: implement filter2d")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", filter2d=filter2d)
    return


@app.function
def box_kernel(size: int) -> np.ndarray:
    """stage 3: the (size, size) averaging kernel: every entry 1 / size**2 (odd size >= 1)."""
    # step 1: np.full((size, size), 1 / size**2)
    raise NotImplementedError("stage3: implement box_kernel")


@app.function
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """stage 3: a (size, size) Gaussian blur kernel with standard deviation sigma (in pixels).

    entry (a, b), with offsets i = a - size // 2 and j = b - size // 2 from the center, is
    proportional to exp(-(i**2 + j**2) / (2 * sigma**2)); the entries sum to 1.
    """
    # step 1: the offsets from the center, as a 1-D array: np.arange(size) - size // 2
    # step 2: a 1-D Gaussian g = exp(-r**2 / (2 sigma**2)); the 2-D kernel is the outer product g g^T
    # step 3: divide by the sum so that blurring keeps the average brightness
    raise NotImplementedError("stage3: implement gaussian_kernel")


@app.function
def unsharp_mask(image: np.ndarray, *, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """stage 3: sharpen by adding back the detail that a blur removes.

    blurred = filter2d(image, gaussian_kernel(size, sigma)) with size = 2 * ceil(3 * sigma) + 1
    and the default "reflect" padding; returns image + amount * (image - blurred), a float
    array of the image's shape. (no clipping: values may leave [0, 1].)
    """
    # step 1: size = 2 * math.ceil(3 * sigma) + 1 (wide enough to hold the Gaussian's bulk)
    # step 2: blur with your filter2d and gaussian_kernel
    # step 3: return image + amount * (image - blurred)
    raise NotImplementedError("stage3: implement unsharp_mask")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", box_kernel=box_kernel, gaussian_kernel=gaussian_kernel,
               unsharp_mask=unsharp_mask, filter2d=filter2d)
    return


@app.function
def sobel_edges(image: np.ndarray) -> np.ndarray:
    """stage 4: the Sobel gradient magnitude of a grayscale image (H, W), same shape, float.

    gx = filter2d(image, SX) and gy = filter2d(image, SX.T) with "reflect" padding, where
        SX = [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    and the result is sqrt(gx**2 + gy**2). a step from 0 to 1 gives a magnitude of 4 next to it.
    """
    # step 1: build SX as a float array; SY is its transpose
    # step 2: gx and gy with your filter2d (default "reflect" padding)
    # step 3: return the magnitude, np.hypot(gx, gy) (or the square root of the sum of squares)
    raise NotImplementedError("stage4: implement sobel_edges")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", sobel_edges=sobel_edges)
    return


@app.function
def detect_edges(image: np.ndarray, *, sigma: float = 1.0, threshold: float = 0.2) -> np.ndarray:
    """stage 5: a boolean edge map (H, W) of a grayscale image.

    1. blur: if sigma > 0, filter2d with gaussian_kernel(2 * ceil(3 * sigma) + 1, sigma);
       sigma == 0 means no blur.
    2. magnitude = sobel_edges(blurred), scaled to [0, 1] by dividing by its maximum
       (an image with no edges at all, maximum 0, has no edge pixels).
    3. edge pixels are those with scaled magnitude > threshold.
    """
    # step 1: blur (skip it when sigma == 0)
    # step 2: the Sobel magnitude, divided by its maximum (if the maximum is 0, return all False)
    # step 3: threshold
    raise NotImplementedError("stage5: implement detect_edges")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", detect_edges=detect_edges)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## given: evaluation helpers

    these are written for you. `psnr` measures how close a restored image is to the clean
    one (in decibels: higher is better; +6 dB means the typical pixel error is halved). `edge_f1` scores an
    edge map against the true edges, forgiving a 1-pixel misplacement. `make_shapes` draws a
    synthetic image whose true edges are known exactly.
    """)
    return


@app.function
def psnr(estimate: np.ndarray, clean: np.ndarray, peak: float = 1.0) -> float:
    """peak signal-to-noise ratio in dB: 10 log10(peak^2 / mean squared error)."""
    mse = float(np.mean((np.asarray(estimate, dtype=float) - clean) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(peak**2 / mse)


@app.function
def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """grow a boolean mask by r pixels in every direction (a (2r+1) x (2r+1) square)."""
    m = np.pad(mask, r)
    H, W = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for a in range(2 * r + 1):
        for b in range(2 * r + 1):
            out |= m[a:a + H, b:b + W]
    return out


@app.function
def edge_f1(pred: np.ndarray, truth: np.ndarray, tol: int = 1) -> float:
    """F1 score of a predicted edge map against true edges, with `tol` pixels of slack."""
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if not pred.any() or not truth.any():
        return 0.0
    precision = np.mean(dilate(truth, tol)[pred])
    recall = np.mean(dilate(pred, tol)[truth])
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


@app.function
def make_shapes(size: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """a synthetic grayscale image (rectangles and a disk on a gradient) and its true edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = 0.2 + 0.1 * xx / size
    regions = np.zeros((size, size), dtype=int)
    for k in range(3):
        y0, x0 = rng.integers(4, size // 2, size=2)
        h, w = rng.integers(size // 6, size // 3, size=2)
        regions[y0:y0 + h, x0:x0 + w] = k + 1
    cy, cx, r = rng.integers(size // 3, 2 * size // 3, size=2).tolist() + [size // 6]
    regions[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 4
    levels = np.array([0.0, 0.5, 0.3, 0.7, 0.6])
    img = img + levels[regions]
    edges = np.zeros((size, size), dtype=bool)
    edges[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    edges[:-1, :] |= regions[:-1, :] != regions[1:, :]
    return img, edges


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    grayscale faces from the Olivetti set (64 × 64) and small color photos from CIFAR-10
    (32 × 32). offline, `mlfs.datasets` substitutes synthetic stand-ins with the same
    shapes, so everything below still runs. shown with your `image_grid`.
    """)
    return


@app.cell
def _():
    faces = datasets.olivetti_faces()[0][:N_FACES]          # (N, 64, 64) floats in [0, 1]
    photos = datasets.cifar10()["X_test"][:N_FACES]          # (N, 32, 32, 3) uint8
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 3.2))
    image_grid(faces, nrow=8, ax=_a1)
    image_grid(photos, nrow=8, ax=_a2)
    _a1.set_title("faces" + (" (synthetic stand-in)" if OFFLINE else ""), fontsize=9)
    _a2.set_title("photos" + (" (synthetic stand-in)" if OFFLINE else ""), fontsize=9)
    _fig
    return faces, photos


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## blur and sharpen

    move the sliders: a Gaussian blur of width $\sigma$, and an unsharp mask of the same width
    with strength `amount`.
    """)
    return


@app.cell
def _():
    sigma_slider = mo.ui.slider(0.5, 4.0, value=1.5, step=0.25, label="σ (pixels)")
    amount_slider = mo.ui.slider(0.0, 3.0, value=1.5, step=0.25, label="sharpening amount")
    mo.hstack([sigma_slider, amount_slider])
    return amount_slider, sigma_slider


@app.cell
def _(amount_slider, faces, photos, sigma_slider):
    _s = sigma_slider.value
    _k = gaussian_kernel(2 * math.ceil(3 * _s) + 1, _s)
    _face = faces[0]
    _photo = photos[0].astype(float) / 255
    _row = [_face, filter2d(_face, _k), unsharp_mask(_face, sigma=_s, amount=amount_slider.value)]
    _row_c = [_photo, filter2d(_photo, _k), unsharp_mask(_photo, sigma=_s, amount=amount_slider.value)]
    _fig, _axes = plt.subplots(2, 3, figsize=(9, 6))
    for _j, _title in enumerate(["original", f"gaussian blur σ={_s}", f"unsharp mask, amount {amount_slider.value}"]):
        _axes[0, _j].imshow(np.clip(_row[_j], 0, 1), cmap="gray", vmin=0, vmax=1)
        _axes[1, _j].imshow(np.clip(_row_c[_j], 0, 1))
        _axes[0, _j].set_title(_title, fontsize=9)
    for _ax in _axes.ravel():
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## denoising: how much blur?

    add Gaussian noise (standard deviation 0.1) to each face, then blur it with increasing
    $\sigma$ and measure the PSNR against the clean face. the noisy image itself is the
    baseline. $\sigma$ is chosen on the first half of the faces (the "tuning" faces) and the
    result is reported on the other half, so the reported number isn't flattered by the
    choice. the synthetic shapes image (smooth regions, sharp edges) is shown for comparison.
    """)
    return


@app.cell
def _(faces):
    _rng = np.random.default_rng(0)
    _sigmas = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
    _images = list(faces) + [make_shapes(64, seed=0)[0]]
    _noisy_psnr, _curves = [], []
    for _clean in _images:
        _noisy = _clean + 0.1 * _rng.normal(size=_clean.shape)
        _noisy_psnr.append(psnr(_noisy, _clean))
        _curves.append([psnr(filter2d(_noisy, gaussian_kernel(2 * math.ceil(3 * _s) + 1, _s)), _clean)
                        for _s in _sigmas])
    _curves, _noisy_psnr = np.array(_curves), np.array(_noisy_psnr)
    _half = len(faces) // 2
    _tune, _report = _curves[:_half], _curves[_half:len(faces)]
    _best = int(np.argmax(_tune.mean(axis=0)))
    _gain = _report[:, _best] - _noisy_psnr[_half:len(faces)]
    _ax = plot_curves({"tuning faces (mean)": (np.array(_sigmas), _tune.mean(axis=0)),
                       "report faces (mean)": (np.array(_sigmas), _report.mean(axis=0)),
                       "shapes image": (np.array(_sigmas), _curves[-1])},
                      xlabel="blur σ (pixels)", ylabel="PSNR (dB)", title="denoising by blurring")
    _ax.axhline(_noisy_psnr[:len(faces)].mean(), color="gray", ls=":")
    _ax.figure.tight_layout()
    mo.vstack([_ax.figure, mo.md(
        f"noisy input (dotted line): **{_noisy_psnr[:len(faces)].mean():.2f} dB**. σ chosen on the tuning faces: "
        f"**{_sigmas[_best]}**. on the {len(_report)} report faces it gives **{_report[:, _best].mean():.2f} dB**, a gain of "
        f"**{_gain.mean():+.2f} ± {_gain.std(ddof=1) / np.sqrt(len(_gain)):.2f} dB** (mean ± standard error) over the noisy input. "
        f"shapes image: best σ = {_sigmas[int(np.argmax(_curves[-1]))]}, gain {_curves[-1].max() - _noisy_psnr[-1]:+.2f} dB.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## edges: why blur first?

    edge detection on a synthetic image with known edges, clean and with noise, with and
    without the blur step. the score is `edge_f1` (1.0 is perfect).
    """)
    return


@app.cell
def _():
    _img, _truth = make_shapes(96, seed=1)
    _noisy = _img + 0.25 * np.random.default_rng(1).normal(size=_img.shape)
    _cases = [("clean, no blur", _img, 0.0), ("noisy, no blur", _noisy, 0.0), ("noisy, σ = 1.5", _noisy, 1.5)]
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.3))
    _axes[0].imshow(_noisy, cmap="gray")
    _axes[0].set_title("noisy input", fontsize=9)
    _rows = []
    for _ax, (_name, _x, _s) in zip(_axes[1:], _cases):
        _e = detect_edges(_x, sigma=_s, threshold=0.35)
        _ax.imshow(_e, cmap="gray")
        _f1 = edge_f1(_e, _truth)
        _ax.set_title(f"{_name}: F1 {_f1:.2f}", fontsize=9)
        _rows.append(f"| {_name} | {_f1:.3f} |")
    for _ax in _axes:
        _ax.axis("off")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| case | edge F1 |\n|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## speed: vectorized versus loops

    the same correlation written with four nested Python loops (over pixels and kernel
    entries), against your vectorized `correlate2d`.
    """)
    return


@app.function
def correlate2d_loops(image, kernel):
    """'valid' cross-correlation with a loop over every output pixel (for comparison only)."""
    kh, kw = kernel.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(kernel * image[i:i + kh, j:j + kw])
    return out


@app.cell
def _():
    _rng = np.random.default_rng(2)
    _rows = []
    for _n in (32, 64, 128):
        _img, _k = _rng.normal(size=(_n, _n)), gaussian_kernel(5, 1.0)
        _t0 = time.perf_counter()
        _slow = correlate2d_loops(_img, _k)
        _t1 = time.perf_counter()
        _fast = correlate2d(_img, _k)
        _t2 = time.perf_counter()
        assert np.allclose(_slow, _fast)
        _rows.append(f"| {_n} × {_n} | {(_t1 - _t0) * 1e3:.1f} ms | {(_t2 - _t1) * 1e3:.3f} ms | {(_t1 - _t0) / (_t2 - _t1):,.0f}× |")
    mo.md("| image | pixel loops | yours | speedup |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **correctness.** how do you know your `filter2d` is right? describe one check beyond
       the provided tests (for example, a property such as "blurring a constant image changes
       nothing", or a comparison with a hand-computed pixel).
    2. **denoising.** report the PSNR of the noisy input and of the best blur, with the spread
       across faces. why does too much blur make things worse again? which $\sigma$ would you
       pick, and how did you avoid choosing it on the same images you report?
    3. **edges.** explain the F1 scores of the three edge maps. why does noise hurt the
       Sobel filter so much, and why does blurring first help? what does the threshold trade
       off?
    4. **padding.** show one image where the padding mode visibly matters, and explain what
       each mode does at the border.
    5. **speed.** report the speedup of your vectorized correlation and explain where it comes
       from. how does the cost of each version grow with the image size and the kernel size?
    """)
    return


if __name__ == "__main__":
    app.run()
