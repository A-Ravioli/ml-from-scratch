"""Checks for capstone.py (image filters from scratch): functional tests plus quality thresholds.

All images here are synthetic and generated deterministically; nothing is downloaded.
SciPy's filters serve as an independent reference (your notebook doesn't use SciPy).
"""
import ast
import inspect
import math
import textwrap
import time
import warnings

import numpy as np
import pytest
from scipy import ndimage, signal


def _shapes(size=64, seed=0):
    """Piecewise-constant test image (rectangles and a disk) and its true edge map."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    regions = np.zeros((size, size), dtype=int)
    for k in range(3):
        y0, x0 = rng.integers(4, size // 2, size=2)
        h, w = rng.integers(size // 6, size // 3, size=2)
        regions[y0:y0 + h, x0:x0 + w] = k + 1
    cy, cx = rng.integers(size // 3, 2 * size // 3, size=2)
    regions[(yy - cy) ** 2 + (xx - cx) ** 2 <= (size // 6) ** 2] = 4
    img = 0.2 + 0.1 * xx / size + np.array([0.0, 0.5, 0.3, 0.7, 0.6])[regions]
    edges = np.zeros((size, size), dtype=bool)
    edges[:, :-1] |= regions[:, :-1] != regions[:, 1:]
    edges[:-1, :] |= regions[:-1, :] != regions[1:, :]
    return img, edges


def _f1(pred, truth, tol=1):
    grow = lambda m: ndimage.binary_dilation(m, structure=np.ones((2 * tol + 1, 2 * tol + 1)))
    pred, truth = np.asarray(pred, dtype=bool), np.asarray(truth, dtype=bool)
    if not pred.any():
        return 0.0
    p, r = grow(truth)[pred].mean(), grow(pred)[truth].mean()
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


_ASYM = np.array([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0], [0.5, 0.0, -2.0]])  # flipping changes it


# ---------------------------------------------------------------- stage 1: correlate2d


def test_stage1_hand_example(nb):
    img = np.arange(16.0).reshape(4, 4)
    k = np.array([[1.0, 0.0], [0.0, -1.0]])
    got = np.asarray(nb.correlate2d(img, k))
    assert got.shape == (3, 3), f"a (4, 4) image and a (2, 2) kernel give a (3, 3) 'valid' output, got {got.shape}"
    assert np.allclose(got, -5.0), (
        f"out[i, j] = img[i, j] - img[i + 1, j + 1] = -5 everywhere for this image, got {got.tolist()}. "
        + ("you flipped the kernel: correlation uses it as is (convolution flips it)." if np.allclose(got, 5.0) else ""))


def test_stage1_matches_scipy(nb):
    rng = np.random.default_rng(0)
    img = rng.normal(size=(20, 17))
    for k in (_ASYM, rng.normal(size=(5, 2)), np.ones((1, 1)), rng.normal(size=(20, 17))):
        got = np.asarray(nb.correlate2d(img, k))
        want = signal.correlate2d(img, k, mode="valid")
        assert got.shape == want.shape and np.allclose(got, want), (
            f"correlate2d with a {k.shape} kernel disagrees with the definition out[i, j] = "
            "sum_ab kernel[a, b] * image[i + a, j + b] (shape or values)")


def test_stage1_color_images(nb):
    rng = np.random.default_rng(1)
    img = rng.uniform(size=(12, 10, 3))
    got = np.asarray(nb.correlate2d(img, _ASYM))
    assert got.shape == (10, 8, 3), f"an (H, W, 3) image should give an (H-kh+1, W-kw+1, 3) result, got {got.shape}"
    for c in range(3):
        assert np.allclose(got[..., c], signal.correlate2d(img[..., c], _ASYM, mode="valid")), (
            f"channel {c} is wrong: filter each color channel separately (slices like image[a:a+oh, b:b+ow] "
            "keep the channel axis automatically)")


def test_stage1_rejects_bad_kernels(nb):
    with pytest.raises(ValueError):
        nb.correlate2d(np.ones((4, 4)), np.ones((5, 3)))
    with pytest.raises(ValueError):
        nb.correlate2d(np.ones((4, 4)), np.ones(3))


def _kernel_offset_correlate(image, kernel):
    """the vectorized method: one whole-image multiply-add per kernel entry (the timing baseline)."""
    kh, kw = kernel.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for a in range(kh):
        for b in range(kw):
            out += kernel[a, b] * image[a:a + oh, b:b + ow]
    return out


def _best_time(fn, *args, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def _loop_depth(node, depth=0):
    """the deepest nesting of for/while loops (comprehensions count as loops) under node."""
    loops = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    here = depth + isinstance(node, loops)
    return max([here] + [_loop_depth(child, here) for child in ast.iter_child_nodes(node)])


def test_stage1_is_vectorized(nb):
    hint = ("loop over the kh * kw kernel entries only (two nested loops at most), and for each one add "
            "kernel[a, b] times a shifted slice of the WHOLE image. never loop over rows, columns or pixels.")
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(nb.correlate2d)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    if tree is not None:
        depth = _loop_depth(tree)
        assert depth <= 2, f"correlate2d has loops nested {depth} deep. {hint}"
    rng = np.random.default_rng(2)
    k = rng.normal(size=(3, 3))
    for shape in ((8000, 16), (16, 8000)):   # a loop over rows (or columns) shows up in one of these
        img = rng.normal(size=shape)
        t_ref, t = _best_time(_kernel_offset_correlate, img, k), _best_time(nb.correlate2d, img, k)
        assert t <= 20 * t_ref + 0.005, (
            f"on a {shape[0]} x {shape[1]} image, correlate2d took {t * 1e3:.1f} ms where the vectorized method "
            f"takes {t_ref * 1e3:.1f} ms ({t / t_ref:.0f}x slower). {hint}")


# ---------------------------------------------------------------- stage 2: filter2d


def test_stage2_same_size_and_flipped(nb):
    rng = np.random.default_rng(3)
    img = rng.normal(size=(15, 13))
    got = np.asarray(nb.filter2d(img, _ASYM, padding="zero"))
    assert got.shape == img.shape, f"filter2d should keep the image's shape {img.shape}, got {got.shape}"
    want = signal.convolve2d(img, _ASYM, mode="same", boundary="fill")
    if np.allclose(got, signal.correlate2d(img, _ASYM, mode="same", boundary="fill")):
        raise AssertionError("filter2d computed a correlation: convolution flips the kernel first, kernel[::-1, ::-1]")
    assert np.allclose(got, want), "filter2d with zero padding disagrees with a same-size convolution"


def test_stage2_padding_modes(nb):
    rng = np.random.default_rng(4)
    img = rng.uniform(size=(11, 9))
    k = rng.normal(size=(5, 3))
    for padding, mode in (("zero", "constant"), ("edge", "nearest"), ("reflect", "mirror")):
        got = np.asarray(nb.filter2d(img, k, padding=padding))
        want = ndimage.convolve(img, k, mode=mode, cval=0.0)
        assert np.allclose(got, want), (
            f"padding={padding!r} is wrong near the borders. pad by kh // 2 rows and kw // 2 columns with "
            f"np.pad(..., mode={ {'zero': 'constant', 'edge': 'edge', 'reflect': 'reflect'}[padding]!r}).")
    assert np.allclose(nb.filter2d(img, k), ndimage.convolve(img, k, mode="mirror")), "the default padding is 'reflect'"


def test_stage2_color_and_errors(nb):
    img = np.random.default_rng(5).uniform(size=(8, 9, 3))
    got = np.asarray(nb.filter2d(img, _ASYM, padding="edge"))
    assert got.shape == img.shape, f"a color image keeps its shape {img.shape}, got {got.shape}"
    assert np.allclose(got[..., 2], ndimage.convolve(img[..., 2], _ASYM, mode="nearest")), (
        "pad only the two spatial axes of a color image: np.pad(..., ((ph, ph), (pw, pw), (0, 0)))")
    with pytest.raises(ValueError):
        nb.filter2d(img, np.ones((2, 3)))
    with pytest.raises(ValueError):
        nb.filter2d(img, np.ones((3, 3)), padding="wrap")


# ---------------------------------------------------------------- stage 3: blur and sharpen


def test_stage3_kernels(nb):
    b = np.asarray(nb.box_kernel(3))
    assert b.shape == (3, 3) and np.allclose(b, 1 / 9), f"box_kernel(3) is a 3 x 3 array of 1/9, got {b.tolist()}"
    g = np.asarray(nb.gaussian_kernel(7, 1.5))
    assert g.shape == (7, 7) and math.isclose(g.sum(), 1.0), "a blur kernel must sum to 1 (or it changes the brightness)"
    r = np.arange(7) - 3
    want = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * 1.5**2))
    assert np.allclose(g, want / want.sum()), (
        "gaussian_kernel(size, sigma)[a, b] should be proportional to exp(-(i^2 + j^2) / (2 sigma^2)) with "
        "i, j the offsets from the CENTER (a - size // 2, b - size // 2)")
    assert np.argmax(g) == 24 and np.allclose(g, g.T) and np.allclose(g, g[::-1]), "the Gaussian peaks at the center and is symmetric"


def test_stage3_blur_reduces_noise(nb):
    rng = np.random.default_rng(6)
    clean, _ = _shapes(64, seed=2)
    noisy = clean + 0.1 * rng.normal(size=clean.shape)
    blurred = np.asarray(nb.filter2d(noisy, nb.gaussian_kernel(7, 1.0)))
    before, after = np.mean((noisy - clean) ** 2), np.mean((blurred - clean) ** 2)
    assert after < 0.4 * before, (
        f"a sigma = 1 Gaussian blur should cut the squared error of a noisy piecewise-constant image by more "
        f"than half: before {before:.5f}, after {after:.5f}")
    const = np.full((10, 10), 0.37)
    assert np.allclose(nb.filter2d(const, nb.gaussian_kernel(5, 2.0)), 0.37), "blurring a constant image must change nothing"


def test_stage3_unsharp_mask(nb):
    rng = np.random.default_rng(7)
    img = rng.uniform(size=(16, 16))
    want = img + 1.5 * (img - ndimage.convolve(img, nb.gaussian_kernel(9, 1.2), mode="mirror"))
    got = np.asarray(nb.unsharp_mask(img, sigma=1.2, amount=1.5))
    assert got.shape == img.shape and np.allclose(got, want), (
        "unsharp_mask should return image + amount * (image - blurred), blurring with "
        "gaussian_kernel(2 * ceil(3 * sigma) + 1, sigma) and 'reflect' padding (a 9 x 9 kernel for sigma = 1.2)")
    assert np.allclose(nb.unsharp_mask(img, amount=0.0), img), "amount = 0 should return the image unchanged"
    step = np.tile(np.r_[np.zeros(8), np.ones(8)], (5, 1))
    s = np.asarray(nb.unsharp_mask(step, sigma=1.0, amount=1.0))
    assert s.max() > 1.05 and s.min() < -0.05, (
        "sharpening a step edge should overshoot on both sides (values above 1 and below 0): that's what makes "
        "edges look crisper")


# ---------------------------------------------------------------- stage 4: Sobel


def test_stage4_step_edge(nb):
    step = np.tile(np.r_[np.zeros(6), np.ones(6)], (8, 1))
    mag = np.asarray(nb.sobel_edges(step))
    assert mag.shape == step.shape, f"sobel_edges should keep the shape {step.shape}, got {mag.shape}"
    assert np.allclose(mag[:, 5:7], 4.0), (
        f"a step from 0 to 1 should give magnitude 4 on the two columns next to it (1 + 2 + 1 times a "
        f"difference of 1), got {np.round(mag[:, 5:7], 3).tolist()[0]}")
    assert np.allclose(mag[:, :4], 0.0) and np.allclose(mag[:, 8:], 0.0), (
        "flat regions (including the image border, with reflect padding) should have zero gradient")


def test_stage4_matches_scipy(nb):
    img, _ = _shapes(40, seed=3)
    img = img + 0.05 * np.random.default_rng(8).normal(size=img.shape)
    want = np.hypot(ndimage.sobel(img, axis=0, mode="mirror"), ndimage.sobel(img, axis=1, mode="mirror"))
    got = np.asarray(nb.sobel_edges(img))
    assert np.allclose(got, want), (
        "sobel_edges should be sqrt(gx^2 + gy^2) with gx = filter2d(image, SX), gy = filter2d(image, SX.T) and "
        "SX = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]")
    assert np.allclose(nb.sobel_edges(img.T), got.T), "rotating the image by transposing should transpose the edge map"


# ---------------------------------------------------------------- stage 5: edge detection


def test_stage5_output_and_flat_image(nb):
    img, _ = _shapes(48, seed=4)
    e = np.asarray(nb.detect_edges(img))
    assert e.shape == img.shape and e.dtype == bool, f"detect_edges should return a boolean (H, W) map, got {e.dtype} {e.shape}"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 0 / 0 would warn "invalid value encountered"
        flat = np.asarray(nb.detect_edges(np.full((20, 20), 0.5)))
    assert flat.dtype == bool and not flat.any(), "a constant image has no edges (and dividing by a maximum of 0 must not give nan)"
    loose, strict = np.asarray(nb.detect_edges(img, threshold=0.1)), np.asarray(nb.detect_edges(img, threshold=0.5))
    assert strict.sum() < loose.sum(), "a higher threshold should keep fewer edge pixels"
    assert np.all(strict <= loose), "every pixel above the high threshold is also above the low one"
    step = np.tile(np.r_[np.zeros(6), np.ones(6)], (8, 1))   # its scaled magnitude is exactly 1 or 0
    assert not np.asarray(nb.detect_edges(step, sigma=0.0, threshold=1.0)).any(), (
        "edge pixels are those STRICTLY above the threshold (scaled magnitude > threshold): with threshold 1.0 "
        "nothing can pass")
    assert np.asarray(nb.detect_edges(step, sigma=0.0, threshold=0.99)).sum() == 16, (
        "on a step image without blur, exactly the two columns next to the step (16 pixels) are edges")


def test_stage5_clean_edges_threshold(nb):
    scores = []
    for seed in range(3):
        img, truth = _shapes(64, seed=10 + seed)
        scores.append(_f1(nb.detect_edges(img, sigma=1.0, threshold=0.25), truth))
    assert min(scores) >= 0.85, (
        f"on clean synthetic shapes, detect_edges(sigma=1, threshold=0.25) should reach an edge F1 of at least "
        f"0.85 (1-pixel tolerance); got {np.round(scores, 3).tolist()}")


def test_stage5_noisy_edges_threshold(nb):
    rng = np.random.default_rng(9)
    blurred, raw = [], []
    for seed in range(3):
        img, truth = _shapes(64, seed=20 + seed)
        noisy = img + 0.25 * rng.normal(size=img.shape)
        blurred.append(_f1(nb.detect_edges(noisy, sigma=1.5, threshold=0.35), truth))
        raw.append(_f1(nb.detect_edges(noisy, sigma=0.0, threshold=0.35), truth))
    assert min(blurred) >= 0.75, (
        f"with strong noise (std 0.25), blurring first (sigma = 1.5) should keep the edge F1 at least 0.75; got "
        f"{np.round(blurred, 3).tolist()}")
    assert np.mean(blurred) > np.mean(raw) + 0.2, (
        f"blurring should help on noisy images: F1 with sigma = 1.5 averaged {np.mean(blurred):.3f}, without "
        f"blur {np.mean(raw):.3f}. with sigma = 0, skip the blur; otherwise blur with "
        "gaussian_kernel(2 * ceil(3 * sigma) + 1, sigma) BEFORE the Sobel filter.")
