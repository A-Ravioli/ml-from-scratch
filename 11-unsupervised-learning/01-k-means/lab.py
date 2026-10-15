import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # k-means · lab

    every experiment here runs **your** `KMeans` and `silhouette_score`. for each one: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes under half a
    minute.

    the adjusted Rand index (ARI) used to compare a clustering with the true groups is 1 for
    a perfect match and about 0 for a random one; you'll build it in topic 05, so here it's a
    small helper.
    """)
    return


@app.cell
def _():
    def moons(n=400, noise=0.08, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, np.pi, n)
        lab = rng.integers(0, 2, n)
        X = np.where(lab[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
        return X + noise * rng.standard_normal((n, 2)), lab

    def rings(n=400, noise=0.05, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, 2 * np.pi, n)
        lab = rng.integers(0, 2, n)
        r = np.where(lab == 0, 1.0, 0.4)
        return np.c_[r * np.cos(t), r * np.sin(t)] + noise * rng.standard_normal((n, 2)), lab

    def grid_blobs(side=5, spread=0.6, n_per=40, seed=0):
        rng = np.random.default_rng(seed)
        centers = np.array([(10.0 * i, 10.0 * j) for i in range(side) for j in range(side)])
        X = np.concatenate([c + spread * rng.standard_normal((n_per, 2)) for c in centers])
        return X, np.repeat(np.arange(side * side), n_per)

    def inertia_of(X, labels):
        """the k-means objective of a given labeling, each cluster at its own mean."""
        return float(sum(((X[labels == c] - X[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels)))

    def ari(a, b):
        """adjusted Rand index from the contingency table (you build it properly in topic 05)."""
        _, ia = np.unique(a, return_inverse=True)
        _, ib = np.unique(b, return_inverse=True)
        C = np.zeros((ia.max() + 1, ib.max() + 1))
        np.add.at(C, (ia, ib), 1)
        comb = lambda m: (m * (m - 1) / 2).sum()  # noqa: E731
        idx, ra, rb, tot = comb(C), comb(C.sum(1)), comb(C.sum(0)), comb(np.array([len(a)]))
        exp = ra * rb / tot
        return float((idx - exp) / (0.5 * (ra + rb) - exp))

    return ari, grid_blobs, inertia_of, moons, rings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: shapes k-means can't see

    two interlocking moons, and two concentric rings, 400 points each, two true groups.
    your `KMeans(2)` (10 restarts) clusters them. besides the ARI against the true groups, we
    compute the inertia of k-means's answer **and the inertia of the true labeling** (each
    true group scored around its own mean).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how well will k-means recover the moons and the rings (ARI)? and is the true labeling's inertia "
        "higher or lower than k-means's?"
    )
    p1
    return (p1,)


@app.cell
def _(ari, inertia_of, moons, p1, rings):
    ui.gate(p1)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.4))
    _rows = []
    for _ax, (_name, _make) in zip(_axes, [("two moons", moons), ("two rings", rings)]):
        _X, _y = _make()
        _km = ex.KMeans(2, rng=0).fit(_X)
        _ax.scatter(_X[:, 0], _X[:, 1], c=_km.labels_, cmap="coolwarm", s=8)
        _ax.scatter(*_km.cluster_centers_.T, c="k", marker="x", s=120, linewidths=3)
        _ax.set_aspect("equal")
        _ax.set_title(f"{_name}: k-means, ARI {ari(_y, _km.labels_):.2f}")
        _rows.append(f"| {_name} | {ari(_y, _km.labels_):.3f} | {_km.inertia_:.1f} | {inertia_of(_X, _y):.1f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| data | ARI | inertia of k-means's answer | inertia of the true labeling |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** is k-means failing to *optimize* here, or optimizing the wrong thing? what
    shape is every k-means cluster (on paper 4), and why does that rule out a crescent or a
    ring? what would you have to change, the algorithm or the representation of the data,
    to cluster the rings correctly?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how lucky is one run?

    a 5 × 5 grid of round blobs, 40 points each, spacing 10 and standard deviation 0.6: the
    blobs are separated by more than 15 standard deviations, so the right answer is obvious
    to the eye. we run your `KMeans(25, n_init=1)` 200 times from uniformly random starts and
    200 times from k-means++ starts, and count how often a single run finds the 25 blobs
    exactly (ARI > 0.99). then 50 fits with `n_init=10` for each.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "out of 200 single runs, how many find all 25 blobs, with random starts? with k-means++?"
    )
    p2
    return (p2,)


@app.cell
def _(ari, grid_blobs, p2):
    ui.gate(p2)
    _X, _y = grid_blobs()
    _res = {}
    for _init in ("random", "k-means++"):
        _fits = [ex.KMeans(25, init=_init, n_init=1, rng=_r).fit(_X) for _r in range(200)]
        _res[_init] = (np.array([ari(_y, _f.labels_) for _f in _fits]), np.array([_f.inertia_ for _f in _fits]), _fits)
    _ten = {_init: np.mean([ari(_y, ex.KMeans(25, init=_init, n_init=10, rng=1000 + _r).fit(_X).labels_) > 0.99
                            for _r in range(50)]) for _init in ("random", "k-means++")}
    _best = min(_res["k-means++"][1].min(), _res["random"][1].min())
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _bins = np.linspace(_best * 0.95, max(_res["random"][1].max(), _res["k-means++"][1].max()) * 1.02, 40)
    for _init in ("random", "k-means++"):
        _axes[0].hist(_res[_init][1], bins=_bins, alpha=0.6, label=_init)
    _axes[0].axvline(_best, color="k", ls="--", lw=1, label="best found")
    _axes[0].set(xlabel="final inertia (one run)", ylabel="runs (of 200)", title="200 single runs each")
    _axes[0].legend(fontsize=8)
    for _ax, _init in zip(_axes[1:], ("random", "k-means++")):
        _f = _res[_init][2][int(np.argsort(_res[_init][1])[100])]  # a typical (median-inertia) run
        _ax.scatter(_X[:, 0], _X[:, 1], c=_f.labels_ % 10, cmap="tab10", s=5)
        _ax.scatter(*_f.cluster_centers_.T, c="k", marker="x", s=60)
        _ax.set_aspect("equal")
        _ax.set_title(f"a median run from {_init} starts")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"single runs finding all 25 blobs: random starts **{np.sum(_res['random'][0] > 0.99)}/200**, k-means++ "
        f"**{np.sum(_res['k-means++'][0] > 0.99)}/200**. median ARI: random {np.median(_res['random'][0]):.3f}, "
        f"k-means++ {np.median(_res['k-means++'][0]):.3f}. with `n_init=10`: random **{_ten['random']:.0%}**, "
        f"k-means++ **{_ten['k-means++']:.0%}** of 50 fits.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the median runs, what does a typical mistake look like (where are the
    centres)? why can't Lloyd's algorithm repair it, even though the blobs are so far apart?
    k-means++ "spreads the centres out": why does it still fail so often? use the restart
    numbers to estimate the probability that one k-means++ run succeeds, and check that
    $1 - (1 - p)^{10}$ matches the `n_init=10` rate.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: choosing k when clusters come in pairs

    four round blobs (standard deviation 1, 60 points each) at the corners of a rectangle:
    two pairs, 20 apart, and within each pair the blobs are `d` apart. for $k = 2, \dots, 8$
    we fit your `KMeans` and plot the inertia (the elbow) and your silhouette score. move the
    slider after you've answered.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with d = 6 (the four blobs are clearly separate when you plot them), which k will the silhouette pick?"
    )
    d_slider = mo.ui.slider(2, 20, value=6, step=1, label="distance d between the blobs of a pair")
    mo.vstack([p3, d_slider])
    return d_slider, p3


@app.cell
def _(d_slider, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    _d = d_slider.value
    _centers = np.array([(0, 0), (_d, 0), (0, 20), (_d, 20)], dtype=float)
    _X = np.concatenate([_c + _rng.standard_normal((60, 2)) for _c in _centers])
    _ks = list(range(2, 9))
    _fits = [ex.KMeans(_k, rng=0).fit(_X) for _k in _ks]
    _sil = [ex.silhouette_score(_X, _f.labels_) for _f in _fits]
    _best = _ks[int(np.argmax(_sil))]
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].scatter(_X[:, 0], _X[:, 1], c=_fits[_ks.index(_best)].labels_, cmap="tab10", s=8)
    _axes[0].set_aspect("equal")
    _axes[0].set_title(f"d = {_d}: the silhouette's choice, k = {_best}")
    _axes[1].plot(_ks, [_f.inertia_ for _f in _fits], "o-")
    _axes[1].set(xlabel="k", ylabel="inertia", title="the elbow")
    _axes[2].plot(_ks, _sil, "o-", color="C1")
    _axes[2].set(xlabel="k", ylabel="silhouette score", title="silhouette")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"silhouette by k: " + ", ".join(f"k={_k}: {_s:.3f}" for _k, _s in zip(_ks, _sil))
                           + f". the silhouette picks **k = {_best}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for small `d`, why does the silhouette prefer 2 clusters even though there are 4 visible blobs?
    work out roughly what $a(i)$ and $b(i)$ are for a point at $k = 2$ and $k = 4$. at what
    `d` does the choice flip? where does the elbow plot bend, and does it help? which answer
    is "right" depends on what for?
    """)
    return


if __name__ == "__main__":
    app.run()
