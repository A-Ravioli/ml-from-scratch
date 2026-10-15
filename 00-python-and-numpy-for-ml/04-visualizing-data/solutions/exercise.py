import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # visualizing data · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks your code live as you edit. the checks can't look at a
    picture, so they inspect the matplotlib objects your functions create: which lines,
    labels, images and points are on the axes. a test cell under each stage draws an example
    so you can see the picture too.

    | stage | you build |
    |---|---|
    | 1 | `plot_curves`: training curves on labeled axes → `mlfs.utils` |
    | 2 | `histogram`: bin counts from scratch, the numbers behind every histogram |
    | 3 | `make_grid`, `image_grid`: many images tiled into one picture → `mlfs.utils` |
    | 4 | `make_meshgrid`, `plot_decision_boundary`: what a classifier does everywhere → `mlfs.utils` |

    `plot_curves`, `image_grid` and `plot_decision_boundary` become part of `mlfs.utils`, and
    you'll use them in nearly every chapter: loss curves from chapter 06 on, image samples
    from chapter 07 on, decision boundaries in chapters 08–10. you can also run the checks
    from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def plot_curves(curves: dict, *, ax=None, xlabel: str = "step", ylabel: str = "value",
                logy: bool = False, title: str | None = None):
    """stage 1: draw one labeled line per curve, with axis labels and a legend. returns the Axes.

    curves: a dict mapping a label to EITHER a sequence of y values (plotted against
    x = 0, 1, 2, ...) OR an (x, y) pair: a tuple of length 2 whose two items are both
    sequences of the same length (raise ValueError if their lengths differ).
    ax: the matplotlib Axes to draw on; if None, make a new figure with plt.subplots().
    logy=True puts the y axis on a log scale. title, if given, is set as the Axes title.
    """
    # step 1: if ax is None, create one: _, ax = plt.subplots(figsize=(6, 3.5))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    for label, curve in curves.items():
        # step 2: decide whether `curve` is an (x, y) pair: a tuple of length 2 whose items
        # step 2: both have np.ndim(...) == 1. otherwise it's just the y values
        if isinstance(curve, tuple) and len(curve) == 2 and np.ndim(curve[0]) == 1 and np.ndim(curve[1]) == 1:
            x, y = np.asarray(curve[0]), np.asarray(curve[1])
            if len(x) != len(y):
                raise ValueError(f"curve {label!r}: x has {len(x)} values but y has {len(y)}")
        else:
            # step 3: for y values alone, use x = np.arange(len(y))
            y = np.asarray(curve)
            x = np.arange(len(y))
        # step 4: draw it with ax.plot(x, y, label=str(label))
        ax.plot(x, y, label=str(label))
    # step 5: set the axis labels, the log scale (ax.set_yscale("log")) and the title if asked
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    # step 6: add a legend if there was at least one curve, and return ax
    if curves:
        ax.legend()
    return ax


@app.cell(hide_code=True)
def _():
    checks.run(1, plot_curves=plot_curves)
    return


@app.cell
def _():
    # try it: two made-up training curves
    _steps = np.arange(1, 301)
    try:
        _out = plot_curves({"train": 2.0 / np.sqrt(_steps), "validation": (_steps[::10], 2.2 / _steps[::10] ** 0.4)},
                           ylabel="loss", logy=True, title="a log-scale loss plot").figure
    except RuntimeError as _e:  # "not implemented yet" is a RuntimeError; other errors show in full
        _out = mo.md(f"*finish stage 1 to see a picture here* ({_e})")
    _out
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def histogram(x: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """stage 2: counts of the values of x in `bins` equal-width bins, like np.histogram(x, bins).

    the bin edges are np.linspace(lo, hi, bins + 1) with lo, hi = x.min(), x.max() (if all
    values are equal, use lo - 0.5 and hi + 0.5 instead). bin i holds edges[i] <= v < edges[i+1],
    except the LAST bin, which also includes its right edge (so x.max() is counted).
    returns (counts, edges): an int array (bins,) and a float array (bins + 1,).
    raises ValueError if x is empty or bins < 1. no Python loops (the checks read your code).
    """
    # step 1: x = np.asarray(x, dtype=float).ravel(); raise ValueError if it's empty or bins < 1
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or bins < 1:
        raise ValueError(f"need at least one value and one bin, got {x.size} values and bins={bins}")
    # step 2: lo, hi = x.min(), x.max(); if lo == hi, widen them to lo - 0.5, hi + 0.5
    lo, hi = x.min(), x.max()
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    # step 3: edges = np.linspace(lo, hi, bins + 1)
    edges = np.linspace(lo, hi, bins + 1)
    # step 4: the bin of each value: np.searchsorted(edges, x, side="right") - 1 gives the index
    # step 4: of the last edge <= value; values equal to the last edge belong to the last bin
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.minimum(idx, bins - 1)
    # step 5: count how many values fall in each bin: np.bincount(idx, minlength=bins)
    counts = np.bincount(idx, minlength=bins)
    return counts, edges


@app.cell(hide_code=True)
def _():
    checks.run(2, histogram=histogram)
    return


@app.cell
def _():
    # try it: the same data, two bin counts
    _x = np.random.default_rng(0).normal(size=500)
    try:
        _hists = [(_b, *histogram(_x, _b)) for _b in (8, 40)]
        _fig, _axes = plt.subplots(1, 2, figsize=(8, 2.5))
        for _ax, (_b, _counts, _edges) in zip(_axes, _hists):
            _ax.stairs(_counts, _edges, fill=True)
            _ax.set_title(f"{_b} bins", fontsize=9)
        _out = _fig
    except RuntimeError as _e:  # "not implemented yet" is a RuntimeError; other errors show in full
        _out = mo.md(f"*finish stage 2 to see a picture here* ({_e})")
    _out
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def make_grid(images: np.ndarray, *, nrow: int = 8, padding: int = 1,
              pad_value: float | None = None) -> np.ndarray:
    """stage 3: tile a batch of images into one big image, `nrow` images per row.

    images: (N, H, W) grayscale, (N, H, W, C) or (N, C, H, W) with C in {1, 3}. if the last
    axis has size 1 or 3 the layout is NHWC; otherwise, if axis 1 has size 1 or 3, it's NCHW.
    uint8 images are scaled to [0, 1]; anything else is converted to float as is.
    the grid has ncols = min(nrow, N) columns and ceil(N / ncols) rows of tiles, with
    `padding` pixels of pad_value (default: the largest pixel value) around and between tiles.
    returns a float array (rows*H + (rows+1)*padding, ncols*W + (ncols+1)*padding) for one
    channel, or the same with a trailing axis of 3 for color. raises ValueError for other shapes.
    """
    # step 1: imgs = np.asarray(images); remember whether it's uint8, then convert to float
    imgs = np.asarray(images)
    scale = 255.0 if imgs.dtype == np.uint8 else 1.0
    imgs = imgs.astype(float) / scale
    # step 2: bring everything to NHWC: (N, H, W) gets a channel axis (imgs[..., None]);
    # step 2: NCHW (axis 1 in {1, 3}, last axis not) is transposed with (0, 2, 3, 1)
    if imgs.ndim == 3:
        imgs = imgs[..., None]
    elif imgs.ndim == 4 and imgs.shape[-1] not in (1, 3):
        if imgs.shape[1] not in (1, 3):
            raise ValueError(f"can't tell where the channels are in shape {imgs.shape}")
        imgs = imgs.transpose(0, 2, 3, 1)
    elif imgs.ndim != 4:
        raise ValueError(f"images must be (N, H, W), (N, H, W, C) or (N, C, H, W), got shape {imgs.shape}")
    n, h, w, c = imgs.shape
    if n == 0:
        raise ValueError("no images to tile")
    # step 3: ncols = min(nrow, n); nrows = math.ceil(n / ncols)
    ncols = min(nrow, n)
    nrows = math.ceil(n / ncols)
    # step 4: start from a canvas full of the pad value, of shape
    # step 4: (nrows*h + (nrows+1)*padding, ncols*w + (ncols+1)*padding, c)
    fill = imgs.max() if pad_value is None else pad_value
    grid = np.full((nrows * h + (nrows + 1) * padding, ncols * w + (ncols + 1) * padding, c), fill, dtype=float)
    # step 5: copy image k into tile (row, col) = divmod(k, ncols): its top-left corner is at
    # step 5: (padding + row*(h + padding), padding + col*(w + padding))
    for k in range(n):
        row, col = divmod(k, ncols)
        top, left = padding + row * (h + padding), padding + col * (w + padding)
        grid[top:top + h, left:left + w] = imgs[k]
    # step 6: drop the channel axis for single-channel images (grid[..., 0])
    return grid[..., 0] if c == 1 else grid


@app.function
def image_grid(images: np.ndarray, *, nrow: int = 8, ax=None, cmap: str = "gray",
               titles: list[str] | None = None):
    """stage 3: show a batch of images as one tiled picture (via make_grid). returns the Axes.

    single-channel grids are drawn with ax.imshow(grid, cmap=cmap); color grids ignore cmap
    (values outside [0, 1] are rescaled into it first). the axis lines and ticks are hidden.
    titles: optional list of N strings, each drawn with ax.text at the top of its tile
    (raise ValueError if the count doesn't match). ax=None makes a new figure.
    """
    # step 1: grid = make_grid(images, nrow=nrow); count the images (len(images))
    grid = make_grid(images, nrow=nrow)
    n = len(images)
    if titles is not None and len(titles) != n:
        raise ValueError(f"got {len(titles)} titles for {n} images")
    # step 2: if ax is None, make a figure whose shape follows the grid's aspect ratio
    if ax is None:
        _, ax = plt.subplots(figsize=(min(12.0, 1.2 * min(nrow, n)), min(12.0, 1.2 * math.ceil(n / min(nrow, n)))))
    # step 3: imshow: grayscale (grid.ndim == 2) with cmap; color after rescaling into [0, 1]
    if grid.ndim == 2:
        ax.imshow(grid, cmap=cmap)
    else:
        lo, hi = grid.min(), grid.max()
        if lo < 0 or hi > 1:
            grid = (grid - lo) / (hi - lo if hi > lo else 1.0)
        ax.imshow(grid)
    # step 4: hide the axes with ax.axis("off")
    ax.axis("off")
    # step 5: titles: tile k sits at (row, col) = divmod(k, ncols); put the text at the top
    # step 5: center of the tile (the same corner arithmetic as make_grid, with padding 1)
    if titles is not None:
        img = np.asarray(images)
        h, w = (img.shape[2], img.shape[3]) if (img.ndim == 4 and img.shape[-1] not in (1, 3)) else img.shape[1:3]
        ncols = min(nrow, n)
        for k, t in enumerate(titles):
            row, col = divmod(k, ncols)
            ax.text(1 + col * (w + 1) + w / 2, 1 + row * (h + 1), str(t), ha="center", va="top",
                    fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.5, "pad": 1, "lw": 0})
    return ax


@app.cell(hide_code=True)
def _():
    checks.run(3, make_grid=make_grid, image_grid=image_grid)
    return


@app.cell
def _():
    # try it: 12 random "images" (smooth blobs), with titles
    _rng = np.random.default_rng(1)
    _imgs = np.cumsum(np.cumsum(_rng.normal(size=(12, 16, 16)), axis=1), axis=2)
    try:
        _out = image_grid(_imgs, nrow=6, titles=[f"#{i}" for i in range(12)]).figure
    except RuntimeError as _e:  # "not implemented yet" is a RuntimeError; other errors show in full
        _out = mo.md(f"*finish stage 3 to see a picture here* ({_e})")
    _out
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def make_meshgrid(X: np.ndarray, *, resolution: int = 200,
                  padding: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """stage 4: a grid of points covering the 2-D data X (n, 2), with a margin of `padding`.

    returns (xx, yy), each (resolution, resolution), from np.meshgrid of
    np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, resolution) and the same
    for column 1. xx varies along each row (axis 1), yy down each column (axis 0).
    """
    # step 1: X = np.asarray(X, dtype=float)
    X = np.asarray(X, dtype=float)
    # step 2: one np.linspace per column, from min - padding to max + padding
    xs = np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, resolution)
    ys = np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, resolution)
    # step 3: return np.meshgrid(xs, ys)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy


@app.function
def plot_decision_boundary(predict, X: np.ndarray, y: np.ndarray, *, ax=None,
                           resolution: int = 200, padding: float = 0.5):
    """stage 4: color the plane by a classifier's prediction, and draw the data on top.

    predict: a function taking an (m, 2) array of points and returning (m,) labels or scores.
    it is called ONCE, on every point of make_meshgrid(X, ...) at the same time (as an
    (resolution**2, 2) array). raise ValueError if it doesn't return one value per point.
    X: (n, 2) points; y: (n,) labels of any sortable type (ints, strings, ...).
    if every prediction is one of the labels in y, the regions get one color per class
    (ax.contourf); otherwise the predictions are treated as scores and shaded continuously.
    the points are drawn with ax.scatter, colored by class index (np.unique(y, return_inverse=True)),
    in the same colors as their regions. returns the Axes; ax=None makes a new figure.
    """
    # step 1: convert X and y to arrays; build the grid with make_meshgrid
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    xx, yy = make_meshgrid(X, resolution=resolution, padding=padding)
    # step 2: stack the grid into an (m, 2) array of points and call predict ONCE on all of them
    points = np.column_stack([xx.ravel(), yy.ravel()])
    Z = np.asarray(predict(points))
    if Z.shape != (len(points),):
        raise ValueError(f"predict should return one value per point, shape ({len(points)},), got {Z.shape}")
    # step 3: class indices for the data: classes, y_idx = np.unique(y, return_inverse=True)
    classes, y_idx = np.unique(y, return_inverse=True)
    k = len(classes)
    colors = ListedColormap(plt.cm.tab10(np.arange(k) % 10))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    # step 4: if every prediction is one of the classes (np.isin), map predictions to class
    # step 4: indices (np.searchsorted(classes, Z)) and draw them with ax.contourf, levels at
    # step 4: -0.5, 0.5, ..., k - 0.5 so each class gets its own color; else shade the scores
    if np.all(np.isin(Z, classes)):
        Z_idx = np.searchsorted(classes, Z).reshape(xx.shape)
        ax.contourf(xx, yy, Z_idx, levels=np.arange(k + 1) - 0.5, cmap=colors, alpha=0.3)
    else:
        ax.contourf(xx, yy, Z.astype(float).reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.4)
    # step 5: scatter the data, colored by y_idx with the same colormap (vmin=-0.5, vmax=k-0.5)
    ax.scatter(X[:, 0], X[:, 1], c=y_idx, cmap=colors, vmin=-0.5, vmax=k - 0.5, edgecolors="k", s=20)
    return ax


@app.cell(hide_code=True)
def _():
    checks.run(4, make_meshgrid=make_meshgrid, plot_decision_boundary=plot_decision_boundary)
    return


@app.cell
def _():
    # try it: a "classifier" that predicts by the nearest of three centers
    _rng = np.random.default_rng(2)
    _centers = np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.0]])
    _y = _rng.integers(0, 3, size=90)
    _X = _centers[_y] + _rng.normal(size=(90, 2)) * 0.7
    _nearest = lambda P: np.argmin(((P[:, None, :] - _centers[None]) ** 2).sum(-1), axis=1)
    try:
        _out = plot_decision_boundary(_nearest, _X, _y).figure
    except RuntimeError as _e:  # "not implemented yet" is a RuntimeError; other errors show in full
        _out = mo.md(f"*finish stage 4 to see a picture here* ({_e})")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `plot_curves`, `image_grid` and `plot_decision_boundary` are
    in `mlfs.utils` (try `from mlfs.utils import image_grid` in a scratch cell). then open
    the lab: `marimo edit lab.py`. it shows how the same data can tell four different stories
    depending on how you plot it.
    """)
    return


if __name__ == "__main__":
    app.run()
