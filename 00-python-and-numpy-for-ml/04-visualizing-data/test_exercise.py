"""Checks for exercise.py (visualizing data).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Plots can't be compared by eye here, so the tests inspect the matplotlib objects your
functions create: the lines, labels, images, text and points on the Axes.
"""
import ast
import inspect
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure


def _fresh_ax():
    """An Axes on a figure that isn't managed by pyplot (nothing pops up, nothing leaks)."""
    return Figure().subplots()


def _close(ax):
    if ax is not None and hasattr(ax, "figure"):
        plt.close(ax.figure)


# ---------------------------------------------------------------- stage 1: plot_curves


def test_stage1_draws_one_labeled_line_per_curve(nb):
    ax = _fresh_ax()
    out = nb.plot_curves({"train": [3.0, 2.0, 1.5], "val": [3.5, 2.5, 2.4]}, ax=ax)
    assert out is ax, "plot_curves should draw on the Axes you pass in and return that same Axes"
    lines = ax.get_lines()
    assert len(lines) == 2, f"two curves should give two lines, got {len(lines)}"
    assert [ln.get_label() for ln in lines] == ["train", "val"], (
        f"each line should be labeled with its dict key; got {[ln.get_label() for ln in lines]}. "
        "pass label=... to ax.plot.")
    assert np.array_equal(lines[0].get_xdata(), [0, 1, 2]) and np.array_equal(lines[0].get_ydata(), [3.0, 2.0, 1.5]), (
        "a curve given as y values only should be plotted against x = 0, 1, 2, ...")
    legend = ax.get_legend()
    assert legend is not None and [t.get_text() for t in legend.get_texts()] == ["train", "val"], (
        "add a legend (ax.legend()) so the reader can tell the curves apart")


def test_stage1_x_y_pairs(nb):
    ax = _fresh_ax()
    nb.plot_curves({"eval": (np.array([0, 10, 20]), np.array([1.0, 0.5, 0.25])), "two points": [0.5, 0.3]}, ax=ax)
    lines = ax.get_lines()
    assert np.array_equal(lines[0].get_xdata(), [0, 10, 20]) and np.array_equal(lines[0].get_ydata(), [1.0, 0.5, 0.25]), (
        "a curve given as an (x, y) tuple should use those x values")
    assert np.array_equal(lines[1].get_xdata(), [0, 1]) and np.array_equal(lines[1].get_ydata(), [0.5, 0.3]), (
        "[0.5, 0.3] is a sequence of two y VALUES (its items are numbers, not sequences), not an (x, y) pair: "
        "only a tuple whose two items are both sequences counts as a pair.")
    ax2 = _fresh_ax()
    nb.plot_curves({"tuple of numbers": (0.9, 0.8)}, ax=ax2)
    assert np.array_equal(ax2.get_lines()[0].get_ydata(), [0.9, 0.8]), "(0.9, 0.8) holds two y values, not a pair"
    with pytest.raises(ValueError):
        nb.plot_curves({"bad": ([0, 1, 2], [1.0, 2.0])}, ax=_fresh_ax())


def test_stage1_labels_scale_and_title(nb):
    ax = _fresh_ax()
    nb.plot_curves({"a": [1.0, 2.0]}, ax=ax)
    assert ax.get_xlabel() == "step" and ax.get_ylabel() == "value", (
        f"the default axis labels are 'step' and 'value', got {ax.get_xlabel()!r} and {ax.get_ylabel()!r}")
    assert ax.get_yscale() == "linear" and ax.get_title() == "", "by default: linear y axis, no title"
    ax = _fresh_ax()
    nb.plot_curves({"a": [1.0, 0.1, 0.01]}, ax=ax, xlabel="epoch", ylabel="loss", logy=True, title="training")
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("epoch", "loss"), "xlabel and ylabel should label the axes"
    assert ax.get_yscale() == "log", "logy=True should put the y axis on a log scale: ax.set_yscale('log')"
    assert ax.get_title() == "training", "title should become the Axes title: ax.set_title(title)"


def test_stage1_creates_axes_when_none_given(nb):
    ax = nb.plot_curves({"a": [1.0, 2.0, 3.0]})
    try:
        assert isinstance(ax, Axes), f"with ax=None, plot_curves should create a figure and return its Axes, got {type(ax).__name__}"
        assert len(ax.get_lines()) == 1
    finally:
        _close(ax)


# ---------------------------------------------------------------- stage 2: histogram


def test_stage2_hand_example(nb):
    counts, edges = nb.histogram(np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0]), bins=3)
    counts, edges = np.asarray(counts), np.asarray(edges)
    assert np.allclose(edges, [0.0, 1.0, 2.0, 3.0]), f"3 equal bins from 0 to 3 have edges [0, 1, 2, 3], got {edges.tolist()}"
    assert counts.tolist() == [1, 2, 4], (
        f"bins [0, 1), [1, 2), [2, 3] hold 1, 2 and 4 values, got {counts.tolist()}. "
        + ("the value 3 equals the last edge and must be counted in the LAST bin (it's closed on the right)."
           if counts.tolist() == [1, 2, 3] else
           "a value on an edge belongs to the bin on its RIGHT (except the very last edge)."
           if counts.tolist() == [3, 3, 1] else ""))
    assert counts.dtype.kind in "iu", f"counts should be integers, got dtype {counts.dtype}"


def test_stage2_matches_numpy(nb):
    rng = np.random.default_rng(0)
    for x, bins in ((rng.normal(size=1000), 10), (rng.exponential(size=333), 7), (rng.integers(0, 5, 50), 5),
                    (np.array([2.5]), 3)):
        counts, edges = (np.asarray(a) for a in nb.histogram(x, bins))
        want_c, want_e = np.histogram(x, bins)
        assert edges.shape == (bins + 1,) and np.allclose(edges, want_e), (
            f"edges should be np.linspace(min, max, bins + 1) ({bins + 1} values); got {edges.shape[0]} values "
            f"from {edges[0]:.4g} to {edges[-1]:.4g}, expected {want_e[0]:.4g} to {want_e[-1]:.4g}")
        assert np.array_equal(counts, want_c), (
            f"counts differ from np.histogram(x, {bins}): got {counts.tolist()}, expected {want_c.tolist()}")
        assert counts.sum() == x.size, "every value must land in exactly one bin"


def test_stage2_constant_data(nb):
    counts, edges = (np.asarray(a) for a in nb.histogram(np.array([5.0, 5.0, 5.0]), bins=4))
    assert np.allclose(edges, [4.5, 4.75, 5.0, 5.25, 5.5]), (
        f"when every value is the same, widen the range to (value - 0.5, value + 0.5); got edges {edges.tolist()}")
    assert counts.tolist() == [0, 0, 3, 0], f"the three 5s belong in the bin [5.0, 5.25), got {counts.tolist()}"


def test_stage2_rejects_bad_input(nb):
    with pytest.raises(ValueError):
        nb.histogram(np.array([]), 3)
    with pytest.raises(ValueError):
        nb.histogram(np.array([1.0, 2.0]), 0)


def _best_time(fn, *args, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def test_stage2_is_vectorized(nb):
    hint = "find every value's bin at once with np.searchsorted, then count them with np.bincount."
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(nb.histogram)))
    except (OSError, TypeError, SyntaxError):
        tree = None
    if tree is not None:
        loops = [f"{type(n).__name__} on line {n.lineno}" for n in ast.walk(tree)
                 if isinstance(n, (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]
        assert not loops, f"histogram contains a Python loop ({', '.join(loops)}): no for, while or comprehensions. {hint}"
    x = np.random.default_rng(1).normal(size=1_000_000)
    t_np, t = _best_time(np.histogram, x, 50), _best_time(nb.histogram, x, 50)
    assert t <= 20 * t_np + 0.005, (
        f"histogram took {t * 1e3:.0f} ms for a million values, {t / t_np:.0f}x np.histogram's "
        f"{t_np * 1e3:.0f} ms on the same data. {hint}")


# ---------------------------------------------------------------- stage 3: image grids


def test_stage3_grid_layout(nb):
    imgs = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)  # 5 images of 2 x 3 pixels
    grid = np.asarray(nb.make_grid(imgs, nrow=3, padding=1, pad_value=-1.0))
    assert grid.shape == (7, 13), (
        f"5 images with nrow=3 make 2 rows of 3 tiles: (2*2 + 3*1, 3*3 + 4*1) = (7, 13) with padding 1; "
        f"got {grid.shape}")
    assert np.array_equal(grid[1:3, 1:4], imgs[0]), "image 0 belongs at the top left, just inside the padding"
    assert np.array_equal(grid[1:3, 5:8], imgs[1]), "image 1 is the second tile of the first row"
    assert np.array_equal(grid[4:6, 1:4], imgs[3]), "image 3 starts the second row (nrow=3 images per row)"
    assert np.all(grid[4:6, 9:12] == -1.0), "the unused 6th tile should be filled with pad_value"
    assert np.all(grid[0] == -1.0) and np.all(grid[:, 4] == -1.0), "the padding rows and columns hold pad_value"


def test_stage3_grid_formats(nb):
    rng = np.random.default_rng(2)
    nhwc = rng.uniform(size=(4, 5, 6, 3))
    g = np.asarray(nb.make_grid(nhwc, nrow=2, padding=0))
    assert g.shape == (10, 12, 3) and np.array_equal(g[:5, 6:], nhwc[1]), (
        f"(N, H, W, 3) color images should give an (h, w, 3) grid; got shape {g.shape}")
    nchw = nhwc.transpose(0, 3, 1, 2)
    g2 = np.asarray(nb.make_grid(nchw, nrow=2, padding=0))
    assert g2.shape == (10, 12, 3) and np.allclose(g2, g), (
        "(N, 3, H, W) images (channels first, as in PyTorch) should give the same grid as (N, H, W, 3). "
        "transpose them with (0, 2, 3, 1).")
    g3 = np.asarray(nb.make_grid(nhwc[..., :1], nrow=2, padding=0))
    assert g3.shape == (10, 12), f"single-channel images should give a 2-D grid, got shape {g3.shape}"
    g4 = np.asarray(nb.make_grid(nchw[:, :1], nrow=4, padding=0))
    assert g4.shape == (5, 24), f"(N, 1, H, W) images should give a 2-D grid, got shape {g4.shape}"
    with pytest.raises(ValueError):
        nb.make_grid(rng.uniform(size=(4, 5)))


def test_stage3_grid_scaling_and_defaults(nb):
    imgs = np.full((3, 2, 2), 51, dtype=np.uint8)
    g = np.asarray(nb.make_grid(imgs, nrow=8, padding=1))
    assert g.dtype.kind == "f", "the grid should be a float array"
    assert np.allclose(g[1:3, 1:3], 0.2), "uint8 images (0..255) should be scaled to [0, 1]: 51 -> 0.2"
    assert g.shape == (4, 10), f"3 images with nrow=8 fit in one row: shape (4, 10), got {g.shape}"
    assert np.allclose(g[0], 0.2), "by default the padding uses the largest pixel value"


def test_stage3_image_grid_shows_the_grid(nb):
    imgs = np.random.default_rng(3).uniform(size=(6, 4, 4))
    ax = _fresh_ax()
    out = nb.image_grid(imgs, nrow=3, ax=ax)
    assert out is ax, "image_grid should draw on the given Axes and return it"
    assert len(ax.images) == 1, f"image_grid should draw ONE image (the tiled grid) with ax.imshow, found {len(ax.images)}"
    shown = np.asarray(ax.images[0].get_array())
    assert np.allclose(shown, nb.make_grid(imgs, nrow=3)), "the image drawn should be make_grid(images, nrow=nrow)"
    assert ax.images[0].get_cmap().name == "gray", "grayscale grids use cmap='gray' by default"
    assert not ax.axison, "hide the axes around an image grid: ax.axis('off')"
    ax2 = _fresh_ax()
    nb.image_grid(imgs, ax=ax2, cmap="viridis")
    assert ax2.images[0].get_cmap().name == "viridis", "pass cmap through to imshow"


def test_stage3_image_grid_titles(nb):
    imgs = np.random.default_rng(4).uniform(size=(4, 8, 8))
    ax = _fresh_ax()
    nb.image_grid(imgs, nrow=2, ax=ax, titles=["a", "b", "c", "d"])
    texts = sorted(t.get_text() for t in ax.texts)
    assert texts == ["a", "b", "c", "d"], f"each title should be drawn with ax.text; found texts {texts}"
    with pytest.raises(ValueError):
        nb.image_grid(imgs, ax=_fresh_ax(), titles=["only one"])
    ax3 = nb.image_grid(imgs)
    try:
        assert isinstance(ax3, Axes), "with ax=None, image_grid should create a figure and return its Axes"
    finally:
        _close(ax3)


# ---------------------------------------------------------------- stage 4: decision boundaries


def test_stage4_meshgrid(nb):
    X = np.array([[0.0, 10.0], [2.0, 14.0], [1.0, 12.0]])
    xx, yy = (np.asarray(a) for a in nb.make_meshgrid(X, resolution=5, padding=1.0))
    assert xx.shape == yy.shape == (5, 5), f"resolution=5 should give two (5, 5) arrays, got {xx.shape}, {yy.shape}"
    assert np.allclose(xx[0], [-1.0, 0.0, 1.0, 2.0, 3.0]), (
        f"xx should run from min - padding = -1 to max + padding = 3 along each ROW, got {xx[0].tolist()}")
    assert np.allclose(yy[:, 0], [9.0, 10.5, 12.0, 13.5, 15.0]), (
        f"yy should run from 9 to 15 down each COLUMN, got {yy[:, 0].tolist()}. use np.meshgrid(xs, ys).")
    assert np.allclose(xx[:, 2], 1.0) and np.allclose(yy[3], 13.5), "xx is constant down columns, yy along rows"


class _Spy:
    """A fake classifier that records what it was asked: label = 1 if x0 + x1 > 0 else 0."""

    def __init__(self, labels=(0, 1)):
        self.calls = []
        self.labels = np.asarray(labels)

    def __call__(self, P):
        self.calls.append(np.array(P, copy=True))
        return np.where(P[:, 0] + P[:, 1] > 0, self.labels[1], self.labels[0])


def _scatter(ax):
    found = [c for c in ax.collections if isinstance(c, PathCollection)]
    assert found, "no scatter plot of the data found: draw the points with ax.scatter"
    return found[-1]


def test_stage4_predict_called_once_on_the_whole_grid(nb):
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(40, 2)), rng.integers(0, 2, 40)
    spy = _Spy()
    ax = _fresh_ax()
    out = nb.plot_decision_boundary(spy, X, y, ax=ax, resolution=30, padding=0.25)
    assert out is ax, "plot_decision_boundary should draw on the given Axes and return it"
    assert len(spy.calls) == 1, (
        f"predict was called {len(spy.calls)} times. call it ONCE with every grid point stacked into an "
        "(m, 2) array: classifiers are vectorized, and calling one per point is painfully slow.")
    P = spy.calls[0]
    assert P.shape == (900, 2), f"with resolution=30, predict should get all 30*30 = 900 points as a (900, 2) array, got {P.shape}"
    lo, hi = X.min(axis=0) - 0.25, X.max(axis=0) + 0.25
    assert np.allclose(P.min(axis=0), lo) and np.allclose(P.max(axis=0), hi), (
        f"the grid should span the data plus padding: from {np.round(lo, 3)} to {np.round(hi, 3)}, got "
        f"{np.round(P.min(axis=0), 3)} to {np.round(P.max(axis=0), 3)}")


def test_stage4_draws_regions_and_points(nb):
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(30, 2)), rng.integers(0, 2, 30)
    ax = _fresh_ax()
    nb.plot_decision_boundary(_Spy(), X, y, ax=ax, resolution=40)
    region = [c for c in ax.collections if not isinstance(c, PathCollection)] + list(ax.images)
    assert region, "the predicted regions should be filled in (ax.contourf), underneath the points"
    sc = _scatter(ax)
    assert np.allclose(sc.get_offsets(), X), "the scatter should show the data points X"
    c = np.asarray(sc.get_array())
    assert c.shape == (30,) and len(np.unique(c)) == 2, (
        "color each point by its class: pass c=class_index to ax.scatter (one value per point)")
    for cls in (0, 1):
        assert len(np.unique(c[y == cls])) == 1, "points of the same class must get the same color"


def test_stage4_regions_match_the_predictions(nb):
    rng = np.random.default_rng(11)
    X = np.column_stack([rng.uniform(-1, 3, 50), rng.uniform(-2, 0, 50)])   # different ranges in x and y
    y = (X[:, 0] > 0.3).astype(int)
    ax = _fresh_ax()
    calls = []
    original = ax.contourf

    def recording_contourf(*args, **kwargs):
        calls.append([np.asarray(a) for a in args[:3]])
        return original(*args, **kwargs)

    ax.contourf = recording_contourf
    nb.plot_decision_boundary(lambda P: (P[:, 0] > 0.3).astype(int), X, y, ax=ax, resolution=25)
    assert calls, "draw the predicted regions with ax.contourf(xx, yy, Z, ...)"
    xx, yy, Z = calls[0]
    assert xx.shape == yy.shape == Z.shape == (25, 25), (
        f"contourf should get the grid xx, yy and the predictions Z, all (25, 25); got {xx.shape}, {yy.shape}, {Z.shape}")
    want = (xx > 0.3).astype(float)
    assert np.array_equal(np.asarray(Z, dtype=float), want), (
        "the colored regions don't match the classifier: Z[i, j] must be the prediction at the grid point "
        "(xx[i, j], yy[i, j]). here the classifier is 'x > 0.3', so Z should change along each ROW. reshape the "
        "predictions with .reshape(xx.shape) (no transpose), and map labels to class indices with "
        "np.searchsorted(classes, Z).")


def test_stage4_string_labels_and_bad_predict(nb):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 2))
    y = np.where(X.sum(axis=1) > 0, "dog", "cat")
    ax = _fresh_ax()
    nb.plot_decision_boundary(_Spy(labels=("cat", "dog")), X, y, ax=ax, resolution=20)
    c = np.asarray(_scatter(ax).get_array(), dtype=float)
    assert np.array_equal(c[y == "cat"], np.zeros((y == "cat").sum())) and np.all(c[y == "dog"] == 1), (
        "labels can be strings: map them to class indices 0, 1, ... in sorted order with "
        "np.unique(y, return_inverse=True) before coloring")
    with pytest.raises(ValueError):
        nb.plot_decision_boundary(lambda P: np.zeros(3), X, y, ax=_fresh_ax(), resolution=10)
    ax2 = nb.plot_decision_boundary(lambda P: P[:, 0] - P[:, 1], X, y, resolution=10)  # scores, not labels
    try:
        assert isinstance(ax2, Axes), "with ax=None, plot_decision_boundary should create a figure and return its Axes"
    finally:
        _close(ax2)
