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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # dimensionality reduction · lab

    every experiment here runs **your** `PCA`, `ProbabilisticPCA` and `KernelPCA`. for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about
    half a minute.

    experiment 1 uses the Olivetti faces (400 photos of 40 people, 64 × 64 pixels) from
    `mlfs.datasets`, downloaded once and cached; offline, a small synthetic stand-in is used
    and the numbers will differ from the notes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: explained variance versus held-out reconstruction

    we flatten the faces to 4,096-dimensional vectors, hold out a random 20% of the photos,
    and fit your `PCA` on the rest. for each $k$ we measure the relative reconstruction
    error $\sum\|x - \hat x\|^2 / \sum\|x - \bar x\|^2$ on the training photos and on the
    held-out ones. we repeat over 5 random splits.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "keep the components explaining 95% of the training variance. what fraction of a held-out face's "
        "variance will they capture? and with every component kept?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _faces, _ = datasets.olivetti_faces()
    _X = _faces.reshape(len(_faces), -1)
    _n_test = len(_X) // 5
    _curves, _at95, _atall, _k95s = [], [], [], []
    for _split in range(5):
        _perm = np.random.default_rng(_split).permutation(len(_X))
        _Xte, _Xtr = _X[_perm[:_n_test]], _X[_perm[_n_test:]]
        _p = ex.PCA().fit(_Xtr)
        _Zte = (_Xte - _p.mean_) @ _p.components_.T  # all scores; truncate per k below
        _den_te = ((_Xte - _p.mean_) ** 2).sum()
        _ks = np.unique(np.linspace(1, _p.n_components_, 40).astype(int))
        _tr = 1 - np.cumsum(_p.explained_variance_ratio_)[_ks - 1]  # training error = discarded variance
        _te = np.array([((_Xte - _p.mean_ - _Zte[:, :_k] @ _p.components_[:_k]) ** 2).sum() / _den_te for _k in _ks])
        _curves.append((_ks, _tr, _te))
        _k95 = int(np.searchsorted(np.cumsum(_p.explained_variance_ratio_), 0.95) + 1)
        _k95s.append(_k95)
        _at95.append(1 - ((_Xte - _p.mean_ - _Zte[:, :_k95] @ _p.components_[:_k95]) ** 2).sum() / _den_te)
        _atall.append(1 - _te[-1])
        if _split == 0:
            _show = (_Xte[:3], _p, _Zte[:3])
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    for _i, (_ks, _tr, _te) in enumerate(_curves):
        _axes[0].plot(_ks, _tr, color="C0", alpha=0.7, label="training" if _i == 0 else None)
        _axes[0].plot(_ks, _te, color="C3", alpha=0.7, label="held out" if _i == 0 else None)
    _axes[0].set(xlabel="components kept, k", ylabel="relative reconstruction error", yscale="log",
                 title="5 random 80/20 splits", ylim=(1e-3, 1.2))
    _axes[0].legend()
    _imgs, _p, _Z = _show
    _side = int(np.sqrt(_imgs.shape[1]))
    _cols = [10, 50, _k95s[0], _p.n_components_]
    _axes[1].axis("off")
    _grid = []
    for _img, _z in zip(_imgs, _Z):
        _row = [_img] + [_p.mean_ + _z[:_k] @ _p.components_[:_k] for _k in _cols]
        _grid.append(np.hstack([_r.reshape(_side, _side) for _r in _row]))
    _axes[1].imshow(np.vstack(_grid), cmap="gray", vmin=0, vmax=1)
    _axes[1].set_title("held-out faces: original, then k = " + ", ".join(map(str, _cols)))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"components for 95% of the training variance: **{np.mean(_k95s):.0f}** (range {min(_k95s)}–{max(_k95s)}). "
        f"fraction of held-out variance they capture: **{np.mean(_at95):.3f} ± {np.std(_at95, ddof=1):.3f}** "
        f"(mean ± sd over 5 splits). with every component ({_p.n_components_}): training **1.000**, held out "
        f"**{np.mean(_atall):.3f} ± {np.std(_atall, ddof=1):.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the training curve exactly "1 − cumulative explained variance
    ratio"? why does the held-out error still decrease with every component, and why does
    it level off instead of reaching 0? what is in the part of a held-out face that no
    combination of training components can produce? what would happen to the gap if the
    held-out photos were of people PCA had never seen?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many dimensions?

    synthetic data with a known answer: $x = Wz + \varepsilon$ with **5** latent dimensions
    ($W$ is $30 \times 5$, with columns of decreasing size) and noise of standard deviation
    0.7 in all 30 dimensions. we fit on 200 points and evaluate on 2,000 fresh ones, for
    $k = 1, \dots, 12$: the held-out reconstruction error of your `PCA`, and the held-out
    mean log-likelihood of your `ProbabilisticPCA`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which k minimizes the held-out reconstruction error? which k maximizes the held-out PPCA log-likelihood?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _W = _rng.normal(0, 1, (30, 5)) * np.array([3, 2.5, 2, 1.5, 1.0])

    def _draw(n):
        return _rng.standard_normal((n, 5)) @ _W.T + 0.7 * _rng.standard_normal((n, 30))

    _Xtr, _Xte = _draw(200), _draw(2000)
    _ks = list(range(1, 13))
    _rec, _ll_tr, _ll_te, _evr = [], [], [], []
    for _k in _ks:
        _p = ex.PCA(_k).fit(_Xtr)
        _rec.append(((_Xte - _p.inverse_transform(_p.transform(_Xte))) ** 2).sum(1).mean())
        _evr.append(_p.explained_variance_ratio_.sum())
        _m = ex.ProbabilisticPCA(_k).fit(_Xtr)
        _ll_tr.append(_m.score_samples(_Xtr).mean())
        _ll_te.append(_m.score_samples(_Xte).mean())
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].plot(_ks, _rec, "o-")
    _axes[0].set(xlabel="k", ylabel="held-out squared error per point", title="PCA reconstruction")
    _axes[1].plot(_ks, _ll_tr, "o-", label="training")
    _axes[1].plot(_ks, _ll_te, "s-", label="held out")
    _axes[1].set(xlabel="k", ylabel="mean log-likelihood", title="probabilistic PCA")
    _axes[1].legend()
    for _ax in _axes:
        _ax.axvline(5, color="0.6", ls=":")
    _fig.tight_layout()
    _table = ("| k | " + " | ".join(map(str, _ks)) + " |\n|---|" + "---|" * len(_ks) + "\n"
              "| held-out error | " + " | ".join(f"{_v:.2f}" for _v in _rec) + " |\n"
              "| held-out log-lik | " + " | ".join(f"{_v:.3f}" for _v in _ll_te) + " |\n"
              "| explained variance | " + " | ".join(f"{_v:.3f}" for _v in _evr) + " |")
    mo.vstack([_fig, mo.md(_table + f"\n\nheld-out error is smallest at **k = {_ks[int(np.argmin(_rec))]}**; held-out "
                                    f"log-likelihood is largest at **k = {_ks[int(np.argmax(_ll_te))]}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't the held-out reconstruction error ever increase with $k$, even
    on data PCA never saw (think of nested subspaces)? what changes at $k = 5$ in both
    curves? why does the PPCA likelihood go *down* after 5, slowly; what does adding a noise
    direction to $W$ cost on new data?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: kernel PCA and the RBF width

    two noisy concentric rings (radii 1 and 0.3, 200 points each). we run your `KernelPCA`
    with the RBF kernel for several $\gamma$ and ask, for each of the first four components,
    how well a single threshold on it separates the rings (1.0 = perfectly, 0.5 = chance).
    plain PCA is the baseline. pick a $\gamma$ to plot after you've answered.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "as gamma goes from 0.1 to 1,000, does the separation get better, worse, or something else? "
        "which component carries it?"
    )
    gamma_pick = mo.ui.dropdown(["0.1", "0.3", "1", "3", "10", "30", "100", "1000"], value="3", label="gamma to plot")
    mo.vstack([p3, gamma_pick])
    return gamma_pick, p3


@app.cell
def _(gamma_pick, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(0)
    _t = _rng.uniform(0, 2 * np.pi, 400)
    _y = (np.arange(400) < 200).astype(int)
    _r = np.where(_y == 1, 1.0, 0.3)
    _X = np.c_[_r * np.cos(_t), _r * np.sin(_t)] + 0.05 * _rng.standard_normal((400, 2))

    def _sep(z):
        best = 0.5
        for thr in np.unique(z):
            a = np.mean((z > thr) == _y)
            best = max(best, a, 1 - a)
        return best

    _rows = [f"| PCA (linear) | " + " | ".join(f"{_sep(_z):.3f}" for _z in ex.PCA(2).fit_transform(_X).T) + " | – | – |"]
    _gammas = [0.1, 0.3, 1, 3, 10, 30, 100, 1000]
    for _g in _gammas:
        _Z = ex.KernelPCA(4, gamma=_g).fit_transform(_X)
        _rows.append(f"| RBF, γ = {_g} | " + " | ".join(f"{_sep(_z):.3f}" for _z in _Z.T) + " |")
    _g = float(gamma_pick.value)
    _Z = ex.KernelPCA(4, gamma=_g).fit_transform(_X)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].scatter(_X[:, 0], _X[:, 1], c=_y, cmap="coolwarm", s=6)
    _axes[0].set_aspect("equal")
    _axes[0].set_title("the data")
    _axes[1].scatter(_Z[:, 0], _Z[:, 1], c=_y, cmap="coolwarm", s=6)
    _axes[1].set(xlabel="component 1", ylabel="component 2", title=f"kernel PCA, γ = {_g}")
    _axes[2].scatter(_Z[:, 2], _Z[:, 3], c=_y, cmap="coolwarm", s=6)
    _axes[2].set(xlabel="component 3", ylabel="component 4", title=f"kernel PCA, γ = {_g}")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("best single-threshold accuracy on each component:\n\n| | comp. 1 | comp. 2 | comp. 3 | comp. 4 |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for small $\gamma$, which component separates the rings, and why that one
    (expand $e^{-\gamma\|x - x'\|^2}$ for small $\gamma$: which features of $x$ appear)? why
    does a middle $\gamma$ move the separation to component 1? what do the components
    describe when $\gamma$ is huge, and why is that useless? how would you choose $\gamma$
    without labels?
    """)
    return


if __name__ == "__main__":
    app.run()
