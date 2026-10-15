import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()


@app.function
def mnist_subset(n_train, n_test, seed=0):
    """n_train / n_test MNIST digits as float rows in [0, 1] (784 features), drawn at random."""
    d = datasets.mnist()
    rng = np.random.default_rng(seed)
    tr = rng.choice(len(d["X_train"]), size=min(n_train, len(d["X_train"])), replace=False)
    te = rng.choice(len(d["X_test"]), size=min(n_test, len(d["X_test"])), replace=False)
    flat = lambda a: a.reshape(len(a), 784).astype(float) / 255.0  # noqa: E731
    return flat(d["X_train"][tr]), d["y_train"][tr], flat(d["X_test"][te]), d["y_test"][te]


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # k-nearest neighbors · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    {"**offline mode:** MNIST is replaced by a small synthetic stand-in, so experiments 1 and 3 say little about real digits. run the lab without `MLFS_OFFLINE` for the real numbers." if OFFLINE else "MNIST is loaded from the cache (`~/.cache/mlfs`)."}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: do distances concentrate?

    points uniform in the unit cube $[0, 1]^d$, for $d$ from 1 to 1,000. for 20 random queries,
    your `relative_contrast` measures how much farther the farthest point is than the nearest,
    relative to the nearest. then the same measurement on real data: each of 20 MNIST digits
    against the other digits of the sample (784 pixels, scaled to $[0, 1]$).
    """)
    return


@app.cell
def _():
    n_points = mo.ui.slider(200, 2000, step=200, value=1000, label="points per dataset")
    n_points
    return (n_points,)


@app.cell
def _():
    p1 = ui.predict(
        "for uniform data in 784 dimensions, what will the relative contrast be (2? 0.5? 0.05?)? "
        "and for real 784-pixel digits: similar, or very different?"
    )
    p1
    return (p1,)


@app.cell
def _(n_points, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _n = n_points.value
    _dims = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 784, 1000])
    _rc = []
    for _d in _dims:
        _X = _rng.random((_n, _d))
        _rc.append(np.median([ex.relative_contrast(_X, _rng.random(_d)) for _ in range(20)]))
    _Xm = mnist_subset(_n + 20, 0, seed=1)[0]
    _rc_mnist = np.median([ex.relative_contrast(np.delete(_Xm, _i, axis=0), _Xm[_i]) for _i in range(20)])
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.loglog(_dims, _rc, "o-", label="uniform in $[0, 1]^d$")
    _ax.loglog(_dims, _rc[4] * np.sqrt(_dims[4] / _dims), "k:", lw=1, label=r"$\propto 1/\sqrt{d}$")
    _ax.plot([784], [_rc_mnist], "r*", ms=14, label="MNIST digits (784 pixels)")
    _ax.set(xlabel="dimension d", ylabel="relative contrast (median of 20 queries)",
            title=f"{_n} points per dataset")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"uniform: d = 2 → **{_rc[1]:.1f}**, d = 100 → **{_rc[6]:.3f}**, d = 784 → **{_rc[9]:.3f}**. "
        f"MNIST (784 pixels): **{_rc_mnist:.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the contrast fall like $1/\sqrt d$ for uniform data (on paper 2)?
    the MNIST digits have as many coordinates as the $d = 784$ cube. why is their contrast so
    different, and what does that say about using $k$-NN on images?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how much does your KD-tree skip?

    4,000 points from a standard normal in $d$ dimensions, a KD-tree with leaf size 16, and 100
    queries from the same distribution, $k = 5$. your tree counts every distance it computes
    in `n_distances_`; brute force computes $4{,}000$ per query. we plot the fraction.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "in 2-D the tree computes a few percent of the distances. at what dimension will it be "
        "computing more than half of them: 5? 10? 50? never?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(2)
    _dims = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]
    _frac = []
    for _d in _dims:
        _X = _rng.normal(size=(4000, _d))
        _tree = ex.KDTree(_X, leaf_size=16)
        _tree.query(_rng.normal(size=(100, _d)), k=5)
        _frac.append(_tree.n_distances_ / (100 * 4000))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.plot(_dims, _frac, "o-")
    _ax.axhline(1.0, color="gray", ls=":", label="brute force")
    _ax.axvline(np.log2(4000 / 16), color="C3", ls="--", lw=1, label=r"$d = \log_2(n / \mathrm{leaf\ size})$ (tree depth)")
    _ax.set(xlabel="dimension d", ylabel="fraction of distances computed", ylim=(0, 1.05))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(" | ".join(f"d={d}: {f:.1%}" for d, f in zip(_dims, _frac)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the tree is still exact in 20-D. why does it compute almost every distance
    there? relate the pruning test $|q_{\text{dim}} - s| \ge r$ to experiment 1: how big is a
    single coordinate's gap compared with the current $k$-th best distance $r$? and how many
    times can 4,000 points be halved before the leaves are reached?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: k-NN on handwritten digits

    5,000 random MNIST training digits and 1,000 test digits, raw pixels in $[0, 1]$. your
    `KNeighborsClassifier` "trains" by storing the images. we measure test accuracy for several
    $k$, with uniform and distance weights. the majority-class baseline is about 10%.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what test accuracy will 1-NN reach on raw pixels with 5,000 training digits, and will "
        "k = 50 do better or worse than k = 1?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _Xtr, _ytr, _Xte, _yte = mnist_subset(5000, 1000, seed=3)
    _ks = [1, 3, 5, 10, 20, 50]
    _acc = {"uniform": [], "distance": []}
    for _w in _acc:
        for _k in _ks:
            _acc[_w].append(ex.KNeighborsClassifier(_k, weights=_w).fit(_Xtr, _ytr).score(_Xte, _yte))
    _base = np.mean(_yte == np.bincount(_ytr).argmax())
    _se = np.sqrt(_acc["uniform"][0] * (1 - _acc["uniform"][0]) / len(_yte))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _w, _a in _acc.items():
        _ax.plot(_ks, _a, "o-", label=f"weights = {_w}")
    _ax.set_xscale("log")
    _ax.set(xlabel="k", ylabel="test accuracy (1,000 digits)", title=f"majority-class baseline: {_base:.1%}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"1-NN: **{_acc['uniform'][0]:.1%}** (± {_se:.1%} standard error) · best uniform: "
        f"**{max(_acc['uniform']):.1%}** at k = {_ks[int(np.argmax(_acc['uniform']))]} · k = 50 uniform: "
        f"**{_acc['uniform'][-1]:.1%}**, distance: **{_acc['distance'][-1]:.1%}**")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** nothing was learned, yet accuracy is far above the baseline. what does that
    tell you about where digits lie in pixel space? why does accuracy *fall* with large $k$,
    and why does distance weighting help a little at every $k > 1$? what would you expect with
    60,000 training digits instead of 5,000?
    """)
    return


if __name__ == "__main__":
    app.run()
