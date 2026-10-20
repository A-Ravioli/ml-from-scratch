import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.unsupervised import GaussianMixture, KMeans  # yours, from topics 01 and 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # graph & density clustering · lab

    every experiment here runs **your** clustering methods from this topic (plus your
    `KMeans` and `GaussianMixture`), scored by your `adjusted_rand_score`. for each one:
    write down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about ten
    seconds.
    """)
    return


@app.cell
def _():
    def make(name, n=400, seed=0):
        """four 2-D data sets (standardized), with their true labels."""
        rng = np.random.default_rng(seed)
        if name == "two moons":
            t = rng.uniform(0, np.pi, n)
            y = np.arange(n) % 2
            X = np.where(y[:, None] == 0, np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
            X = X + 0.06 * rng.standard_normal((n, 2))
        elif name == "two rings":
            t = rng.uniform(0, 2 * np.pi, n)
            y = np.arange(n) % 2
            r = np.where(y == 0, 1.0, 0.4)
            X = np.c_[r * np.cos(t), r * np.sin(t)] + 0.05 * rng.standard_normal((n, 2))
        elif name == "varied density":
            y = np.arange(n) % 3
            X = np.array([[0, 0], [4, 0], [2, 5]])[y] + np.array([0.3, 0.8, 1.6])[y, None] * rng.standard_normal((n, 2))
        else:  # stretched
            y = np.arange(n) % 3
            X = np.array([[0, 0], [0, 2.5], [0, 5]])[y] + rng.standard_normal((n, 2)) * [3, 0.35]
            X = X @ np.array([[np.cos(0.6), np.sin(0.6)], [-np.sin(0.6), np.cos(0.6)]])
        return (X - X.mean(0)) / X.std(0), y

    DATASETS = ["two moons", "two rings", "varied density", "stretched"]
    return DATASETS, make


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: every method on every shape

    four standardized 2-D data sets of 400 points: two moons, two rings, three blobs of very
    different density, and three stretched, tilted blobs. six methods, each told the true
    number of clusters where it needs one: your `KMeans`, `GaussianMixture` (full
    covariances), `SpectralClustering` (15-nearest-neighbour graph), `DBSCAN`
    (`min_samples=5`, `eps` from the dropdown), and `AgglomerativeClustering` with single and
    Ward linkage. we score each by ARI.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("which method wins on the moons? on the varied-density blobs? is there one method that's never bad?")
    eps_pick = mo.ui.dropdown(["0.1", "0.15", "0.2", "0.3"], value="0.2", label="DBSCAN eps (on standardized data)")
    mo.vstack([p1, eps_pick])
    return eps_pick, p1


@app.cell
def _(DATASETS, eps_pick, make, p1):
    ui.gate(p1)
    _eps = float(eps_pick.value)
    _methods = {
        "k-means": lambda X, k: KMeans(k, rng=0).fit(X).labels_,
        "GMM": lambda X, k: GaussianMixture(k, rng=0).fit(X).predict(X),
        "spectral": lambda X, k: ex.SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=15, rng=0).fit_predict(X),
        "DBSCAN": lambda X, k: ex.DBSCAN(_eps, min_samples=5).fit_predict(X),
        "single": lambda X, k: ex.AgglomerativeClustering(k, linkage="single").fit_predict(X),
        "Ward": lambda X, k: ex.AgglomerativeClustering(k, linkage="ward").fit_predict(X),
    }
    _fig, _axes = plt.subplots(len(DATASETS), len(_methods), figsize=(17, 11))
    _table = {}
    for _r, _name in enumerate(DATASETS):
        _X, _y = make(_name)
        _k = len(np.unique(_y))
        for _c, (_m, _fn) in enumerate(_methods.items()):
            _lab = _fn(_X, _k)
            _a = ex.adjusted_rand_score(_y, _lab)
            _table[(_name, _m)] = _a
            _ax = _axes[_r, _c]
            _ax.scatter(_X[:, 0], _X[:, 1], c=np.where(_lab < 0, -1, _lab % 10), cmap="tab10", vmin=-1, vmax=9, s=4)
            _ax.set_xticks([])
            _ax.set_yticks([])
            _ax.set_title(f"{_m}: {_a:.2f}", fontsize=10)
            if _c == 0:
                _ax.set_ylabel(_name)
    _fig.tight_layout()
    _md = "| data | " + " | ".join(_methods) + " |\n|---|" + "---|" * len(_methods) + "\n"
    for _name in DATASETS:
        _md += f"| {_name} | " + " | ".join(f"{_table[(_name, _m)]:.2f}" for _m in _methods) + " |\n"
    mo.vstack([_fig, mo.md(_md + f"\n(ARI; DBSCAN with eps = {_eps}; noise points count as their own label.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each method, say what it believes a cluster is (a centre, an
    ellipse, a weakly connected part of a graph, a dense region, a chain, a compact group)
    and use that to explain where it wins and where it fails. why does single linkage get
    0.00 on the varied-density blobs while succeeding on the moons? change DBSCAN's `eps`:
    is there one value that works everywhere?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: DBSCAN's density threshold

    the varied-density blobs (standard deviations 0.3, 0.8 and 1.6 before standardizing).
    we sweep `eps` from 0.03 to 0.6 with `min_samples=5` and record the ARI, the number of
    clusters found and the fraction of points labelled noise. on the right, the
    **k-distance plot**: every point's distance to its 5th nearest neighbour, sorted, the
    usual tool for choosing `eps`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("what's the best ARI DBSCAN can reach on these three blobs, for any eps?")
    p2
    return (p2,)


@app.cell
def _(make, p2):
    ui.gate(p2)
    _X, _y = make("varied density")
    _eps = np.linspace(0.03, 0.6, 39)
    _ari, _nc, _noise = [], [], []
    for _e in _eps:
        _lab = ex.DBSCAN(_e, min_samples=5).fit_predict(_X)
        _ari.append(ex.adjusted_rand_score(_y, _lab))
        _nc.append(len(set(_lab) - {-1}))
        _noise.append(np.mean(_lab == -1))
    _D = np.sqrt(((_X[:, None] - _X[None]) ** 2).sum(-1))
    _kd = np.sort(np.sort(_D, 1)[:, 5])  # column 0 is the point itself
    _best = int(np.argmax(_ari))
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    _axes[0].plot(_eps, _ari, "o-", ms=3, label="ARI")
    _axes[0].plot(_eps, _noise, "s-", ms=3, label="noise fraction")
    _axes[0].set(xlabel="eps", title="DBSCAN on varied-density blobs")
    _axes[0].legend()
    _axes[1].plot(_eps, _nc, "o-", ms=3, color="C2")
    _axes[1].set(xlabel="eps", ylabel="clusters found (true: 3)")
    _axes[2].plot(_kd)
    _axes[2].set(xlabel="points, sorted", ylabel="distance to 5th neighbour", title="k-distance plot")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"best ARI **{_ari[_best]:.3f}** at eps = {_eps[_best]:.3f} ({_nc[_best]} clusters, {_noise[_best]:.0%} noise). "
        f"5th-neighbour distances: 10th percentile {np.percentile(_kd, 10):.3f}, median {np.median(_kd):.3f}, "
        f"90th percentile {np.percentile(_kd, 90):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at the best `eps`, which cluster is still wrong, and how (noise, or merged
    with a neighbour)? why can no single `eps` be right for all three blobs? what does the
    k-distance plot look like when densities differ, and why is there no clean knee?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a thin bridge

    two round blobs of 150 points each (standard deviation 1), 8 apart, and a straight line
    of `m` extra points from one blob to the other. we cluster into 2 groups with your
    single-linkage and Ward-linkage `AgglomerativeClustering` and score the ARI on the 300
    blob points. move the slider after you've answered.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how many bridge points does it take to break single linkage? and Ward?")
    m_slider = mo.ui.slider(0, 20, value=6, label="bridge points m")
    mo.vstack([p3, m_slider])
    return m_slider, p3


@app.cell
def _(m_slider, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _blobs = np.r_[_rng.standard_normal((150, 2)), _rng.standard_normal((150, 2)) + [8, 0]]
    _y = np.repeat([0, 1], 150)

    def _run(m):
        bridge = np.c_[np.linspace(1.5, 6.5, m), np.zeros(m)] if m else np.zeros((0, 2))
        X = np.r_[_blobs, bridge]
        out = {}
        for link in ("single", "ward"):
            lab = ex.AgglomerativeClustering(2, linkage=link).fit_predict(X)
            out[link] = (ex.adjusted_rand_score(_y, lab[:300]), lab, X)
        return out

    _curve = {m: _run(m) for m in range(0, 21, 2)}
    _now = _run(m_slider.value)
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    for _ax, _link in zip(_axes[:2], ("single", "ward")):
        _a, _lab, _X = _now[_link]
        _ax.scatter(_X[:, 0], _X[:, 1], c=_lab, cmap="coolwarm", s=6)
        _ax.set_aspect("equal")
        _ax.set_title(f"{_link} linkage, m = {m_slider.value}: ARI {_a:.2f}")
    _ms = list(_curve)
    for _link in ("single", "ward"):
        _axes[2].plot(_ms, [_curve[_m][_link][0] for _m in _ms], "o-", label=_link)
    _axes[2].set(xlabel="bridge points m", ylabel="ARI on the blobs")
    _axes[2].legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("ARI by number of bridge points: " + ", ".join(
        f"m={_m}: single {_curve[_m]['single'][0]:.2f} / Ward {_curve[_m]['ward'][0]:.2f}" for _m in _ms))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** without a bridge, what are the two clusters single linkage merges last,
    and at what height? once the bridge spans the gap with steps smaller than the typical
    distance inside a blob, what does single linkage's *last* merge separate instead? why is
    Ward unaffected? which real data look like "blobs with bridges"?
    """)
    return


if __name__ == "__main__":
    app.run()
