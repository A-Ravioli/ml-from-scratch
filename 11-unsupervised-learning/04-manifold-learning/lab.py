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
    # manifold learning · lab

    every experiment here runs **your** `MDS`, `Isomap` and `TSNE`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. exact t-SNE is $O(n^2)$ per iteration, so the
    whole lab takes about a minute.
    """)
    return


@app.cell
def _():
    def knn_accuracy(Y, y, k=5):
        """fraction of points whose k nearest neighbours in Y mostly share their label."""
        D = ((Y[:, None] - Y[None]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, 1)[:, :k]
        return float(np.mean([np.bincount(y[r]).argmax() == y[i] for i, r in enumerate(nn)]))

    return (knn_accuracy,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: perplexity

    top row: five Gaussian clusters in 10 dimensions (60 points each, standard deviation 1,
    centres about 10 apart). bottom row: 300 points of pure Gaussian noise in 10 dimensions,
    no clusters at all. your `TSNE` with perplexity 2, 10, 30 and 100 (1,000 iterations,
    same seed). for the clusters we also report the 5-nearest-neighbour accuracy of the
    embedding (how often a point's plot neighbours share its cluster).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which perplexity gives the cleanest five clusters? what does t-SNE show for pure noise at perplexity 2?"
    )
    p1
    return (p1,)


@app.cell
def _(knn_accuracy, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _C = _rng.normal(0, 7, (5, 10))
    _Xc = np.concatenate([_c + _rng.standard_normal((60, 10)) for _c in _C])
    _yc = np.repeat(np.arange(5), 60)
    _Xn = _rng.standard_normal((300, 10))
    _perps = [2, 10, 30, 100]
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 7.5))
    _rows = []
    for _j, _perp in enumerate(_perps):
        _tc = ex.TSNE(perplexity=float(_perp), rng=0).fit(_Xc)
        _tn = ex.TSNE(perplexity=float(_perp), rng=0).fit(_Xn)
        _axes[0, _j].scatter(*_tc.embedding_.T, c=_yc, cmap="tab10", s=6)
        _axes[0, _j].set_title(f"5 clusters, perplexity {_perp}")
        _axes[1, _j].scatter(*_tn.embedding_.T, c="0.3", s=6)
        _axes[1, _j].set_title(f"pure noise, perplexity {_perp}")
        _rows.append(f"| {_perp} | {knn_accuracy(_tc.embedding_, _yc):.3f} | {_tc.kl_divergence_:.3f} | {_tn.kl_divergence_:.3f} |")
    for _ax in _axes.ravel():
        _ax.set_xticks([])
        _ax.set_yticks([])
        _ax.set_aspect("equal")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| perplexity | 5-NN accuracy (clusters) | KL (clusters) | KL (noise) |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does a perplexity of 2 ask each point to do, and why does that
    fragment both the real clusters and the noise? why is the KL so different across
    perplexities, and why can't you choose the perplexity by minimizing it? what's a
    reasonable way to choose it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what sizes and distances mean in a t-SNE plot

    **left:** two clusters of 100 points in 10 dimensions, far apart; one has standard
    deviation 1, the other 0.1, so it's ten times tighter. **right:** three clusters
    (standard deviation 1) with centres on a line at 0, 10 and 50: A and B are close, C is
    five times further from A than B is (and four times further from B than A is). your `TSNE` with perplexity 30. we measure the
    spread of each cluster in the plot and the distances between plotted cluster centres.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "in the plot, how much bigger will the wide cluster look than the tight one? and how much further will C be "
        "from B than A is from B?"
    )
    perp2 = mo.ui.dropdown(["5", "30", "50"], value="30", label="perplexity (after you've answered)")
    mo.vstack([p2, perp2])
    return p2, perp2


@app.cell
def _(p2, perp2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _Xa = np.concatenate([_rng.standard_normal((100, 10)), 30 + 0.1 * _rng.standard_normal((100, 10))])
    _ya = np.repeat([0, 1], 100)
    _perp = float(perp2.value)
    _Ea = ex.TSNE(perplexity=_perp, rng=0).fit_transform(_Xa)
    _spread = lambda E, m: float(E[m].std(0).mean())  # noqa: E731
    _data_ratio = _spread(_Xa, _ya == 0) / _spread(_Xa, _ya == 1)
    _plot_ratio = _spread(_Ea, _ya == 0) / _spread(_Ea, _ya == 1)
    _c = np.zeros((3, 10))
    _c[1, 0], _c[2, 0] = 10, 50
    _Xb = np.concatenate([_cc + _rng.standard_normal((100, 10)) for _cc in _c])
    _yb = np.repeat([0, 1, 2], 100)
    _Eb = ex.TSNE(perplexity=_perp, rng=0).fit_transform(_Xb)
    _m = np.array([_Eb[_yb == _k].mean(0) for _k in range(3)])
    _d = lambda a, b: float(np.linalg.norm(_m[a] - _m[b]))  # noqa: E731
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _axes[0].scatter(*_Ea.T, c=_ya, cmap="coolwarm", s=8)
    _axes[0].set_title("wide (blue, sd 1) and tight (red, sd 0.1) clusters")
    _axes[1].scatter(*_Eb.T, c=_yb, cmap="viridis", s=8)
    for _k, _name in enumerate("ABC"):
        _axes[1].annotate(_name, _m[_k], fontsize=16, weight="bold")
    _axes[1].set_title("clusters at 0 (A), 10 (B) and 50 (C)")
    for _ax in _axes:
        _ax.set_aspect("equal")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"spread ratio wide/tight: **{_data_ratio:.1f}** in the data, **{_plot_ratio:.2f}** in the plot. "
        f"distances between plotted centres: A–B **{_d(0, 1):.1f}**, B–C **{_d(1, 2):.1f}**, A–C **{_d(0, 2):.1f}**; "
        f"ratio B–C / A–B: **4.0** in the data, **{_d(1, 2) / _d(0, 1):.2f}** in the plot; A–C / A–B: **5.0** in the data, "
        f"**{_d(0, 2) / _d(0, 1):.2f}** in the plot.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each point's $\beta_i$ is set to give it 30 effective neighbours. what
    does that do to the tight cluster's affinities compared with the wide one's? why does
    t-SNE place C roughly as far from B as from A (what does the KL charge for getting a
    large distance wrong when $p_{ij} \approx 0$)? what *can* you read from a t-SNE plot?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the swiss roll three ways

    800 points on a swiss roll (a rolled-up 2-D sheet in 3-D), coloured by position along
    the roll. we embed it in 2-D with your `MDS` (straight-line distances), your
    `Isomap` (8 neighbours) and your `TSNE` (perplexity 30), and measure how well each
    embedding preserves the order along the roll: the rank correlation of the best embedding
    coordinate with the true arc length, and the fraction of each point's 10 nearest
    neighbours on the sheet that are also among its 10 nearest in the embedding.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "rank MDS, Isomap and t-SNE by how well they preserve the order along the roll, and by how well they "
        "preserve local neighbourhoods."
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _t = 1.5 * np.pi * (1 + 2 * _rng.uniform(size=800))
    _h = _rng.uniform(0, 15, 800)
    _X = np.c_[_t * np.cos(_t), _h, _t * np.sin(_t)] + 0.05 * _rng.standard_normal((800, 3))
    _arc = 0.5 * (_t * np.sqrt(1 + _t**2) + np.arcsinh(_t))  # arc length along r = t
    _sheet = np.c_[_arc, _h]

    def _rank_corr(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return abs(np.corrcoef(ra, rb)[0, 1])

    def _nbrs(Z, k=10):
        D = ((Z[:, None] - Z[None]) ** 2).sum(-1)
        np.fill_diagonal(D, np.inf)
        return np.argsort(D, 1)[:, :k]

    _true_nn = _nbrs(_sheet)
    _embs = {"MDS": ex.MDS(2).fit_transform(_X), "Isomap": ex.Isomap(2, n_neighbors=8).fit_transform(_X),
             "t-SNE": ex.TSNE(perplexity=30.0, rng=0).fit_transform(_X)}
    _fig, _axes = plt.subplots(1, 4, figsize=(17, 4))
    _axes[0].scatter(_X[:, 0], _X[:, 2], c=_arc, cmap="Spectral", s=5)
    _axes[0].set_title("the roll (seen from the side)")
    _rows = []
    for _ax, (_name, _E) in zip(_axes[1:], _embs.items()):
        _ax.scatter(*_E.T, c=_arc, cmap="Spectral", s=5)
        _ax.set_title(_name)
        _rc = max(_rank_corr(_E[:, 0], _arc), _rank_corr(_E[:, 1], _arc))
        _nn = _nbrs(_E)
        _keep = np.mean([len(set(_a) & set(_b)) / 10 for _a, _b in zip(_nn, _true_nn)])
        _rows.append(f"| {_name} | {_rc:.3f} | {_keep:.3f} |")
    for _ax in _axes:
        _ax.set_aspect("equal")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| method | rank correlation with arc length | 10-NN preserved |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't MDS on straight-line distances unroll the sheet, whatever you
    do? Isomap preserves the global order: why? t-SNE may keep local neighbourhoods well
    and still break the global order (look at where the colours jump): which part of its
    objective allows that? which method would you trust for which question?
    """)
    return


if __name__ == "__main__":
    app.run()
