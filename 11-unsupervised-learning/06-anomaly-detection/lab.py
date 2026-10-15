import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.metrics import f1_score, precision_score, recall_score, roc_auc_score  # yours, from ch 07
    from mlfs.unsupervised import GaussianMixture  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # anomaly detection · lab

    every experiment here runs **your** detectors from `exercise.py`, evaluated with your
    `mlfs.metrics` from chapter 07. the true anomalies are known here (we planted them), so
    we can score the detectors; in real use you'd have only the ranking. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes
    about fifteen seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the contamination rate

    1,900 normal points (a 2-D Gaussian) and 100 anomalies (5%) scattered uniformly around
    them, at least 2.5 standard deviations from the centre. we fit your `IsolationForest` (100 trees,
    seed 0) with `contamination` set to 1%, 2%, 5%, 10% and 20%, and report the AUC of
    `score_samples` and the precision, recall and F1 of `predict`. we repeat the fit with 10
    seeds to see how much the numbers move.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as the assumed contamination goes from 1% to 20% (the truth is 5%), what happens to the AUC? to precision "
        "and recall?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _inl = _rng.standard_normal((1900, 2))
    _out = []
    while len(_out) < 100:
        _p = _rng.uniform(-5, 5, 2)
        if np.linalg.norm(_p) > 2.5:
            _out.append(_p)
    _X = np.r_[_inl, np.array(_out)]
    _y = np.r_[np.zeros(1900), np.ones(100)].astype(int)
    _rates = [0.01, 0.02, 0.05, 0.1, 0.2]
    _res = {_c: [] for _c in _rates}
    for _seed in range(10):
        _f = ex.IsolationForest(100, contamination=0.05, rng=_seed).fit(_X)
        _s = _f.score_samples(_X)
        for _c in _rates:
            _pred = (_s > np.quantile(_s, 1 - _c)).astype(int)  # what fit(contamination=c) would set as threshold_
            _res[_c].append((roc_auc_score(_y, _s), precision_score(_y, _pred), recall_score(_y, _pred), f1_score(_y, _pred)))
    _f0 = ex.IsolationForest(100, contamination=0.05, rng=0).fit(_X)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _sc = _axes[0].scatter(_X[:, 0], _X[:, 1], c=_f0.score_samples(_X), cmap="magma_r", s=6)
    _axes[0].scatter(_X[_y == 1, 0], _X[_y == 1, 1], facecolors="none", edgecolors="C0", s=40, lw=0.8, label="planted anomalies")
    _axes[0].set_aspect("equal")
    _axes[0].legend(fontsize=8, loc="upper right")
    _fig.colorbar(_sc, ax=_axes[0], label="anomaly score")
    _m = {_c: np.mean(_res[_c], axis=0) for _c in _rates}
    for _j, _name in enumerate(["AUC", "precision", "recall", "F1"]):
        _axes[1].plot([100 * _c for _c in _rates], [_m[_c][_j] for _c in _rates], "o-", label=_name)
    _axes[1].axvline(5, color="0.6", ls=":")
    _axes[1].set(xscale="log", xlabel="assumed contamination (%)", ylabel="mean over 10 seeds", xticks=[1, 2, 5, 10, 20],
                 xticklabels=["1", "2", "5", "10", "20"])
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {100 * _c:g}% | " + " | ".join(f"{np.mean(np.array(_res[_c])[:, _j]):.3f} ± {np.std(np.array(_res[_c])[:, _j], ddof=1):.3f}"
                                                 for _j in range(4)) + " |" for _c in _rates]
    mo.vstack([_fig, mo.md("| contamination | AUC | precision | recall | F1 |\n|---|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\n(mean ± sd over 10 forests with different seeds.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the AUC the same in every row? at 1% and at 20%, which kind of
    error dominates, and why is that forced by the number of points you flag? what would you
    report to someone who doesn't know the true anomaly rate?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: masking, and the subsample size

    1,000 normal points (a 2-D standard Gaussian) and **100 anomalies in a tight cluster**
    (standard deviation 0.15) centred at $(3, 3)$, about four standard deviations out. we fit
    your `IsolationForest` with `max_samples` from 16 to all 1,100 points and report the AUC.
    as a comparison, your `knn_distance_scores` with $k = 5$ and $k = 150$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which max_samples gives the best AUC: 16, 256, or all 1,100 points? and how does the 5-NN distance score do?"
    )
    psi_pick = mo.ui.dropdown(["16", "64", "256", "1100"], value="16", label="max_samples for the left map")
    mo.vstack([p2, psi_pick])
    return p2, psi_pick


@app.cell
def _(p2, psi_pick):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _X = np.r_[_rng.standard_normal((1000, 2)), [3.0, 3.0] + 0.15 * _rng.standard_normal((100, 2))]
    _y = np.r_[np.zeros(1000), np.ones(100)].astype(int)
    _psis = [16, 32, 64, 128, 256, 512, 1100]
    _auc = {}
    for _psi in _psis:
        _a = [roc_auc_score(_y, ex.IsolationForest(100, max_samples=_psi, rng=_seed).fit(_X).score_samples(_X)) for _seed in range(3)]
        _auc[_psi] = (np.mean(_a), np.std(_a, ddof=1))
    _knn = {_k: roc_auc_score(_y, ex.knn_distance_scores(_X, _k)) for _k in (5, 150)}
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for _ax, _psi in zip(_axes[:2], (int(psi_pick.value), 1100)):
        _s = ex.IsolationForest(100, max_samples=_psi, rng=0).fit(_X).score_samples(_X)
        _sc = _ax.scatter(_X[:, 0], _X[:, 1], c=_s, cmap="magma_r", s=6)
        _ax.set_aspect("equal")
        _ax.set_title(f"max_samples = {_psi}")
        _fig.colorbar(_sc, ax=_ax)
    _axes[2].errorbar(_psis, [_auc[_p][0] for _p in _psis], yerr=[_auc[_p][1] for _p in _psis], fmt="o-", capsize=3)
    _axes[2].set(xscale="log", xlabel="max_samples (log scale)", ylabel="AUC (mean ± sd, 3 seeds)", title="isolation forest")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "AUC by max_samples: " + ", ".join(f"{_p}: **{_auc[_p][0]:.3f}**" for _p in _psis)
        + f". k-NN distance score: k = 5 **{_knn[5]:.3f}**, k = 150 **{_knn[150]:.3f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with all 1,100 points in every tree, why are the clustered anomalies hard
    to isolate (what protects each of them)? what changes when a tree sees only 16 random
    points, about 1.5 of which are anomalies? why is the 5-NN score *worse than random*
    here, and what does $k = 150$ fix? what's the general name for this failure?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: two modes, anomalies in the middle

    normal data in two blobs (500 points each, at $(-4, 0)$ and $(4, 0)$) and 15 anomalies
    in a small clump at the origin, exactly between them. four detectors: your
    `gaussian_scores` (one Gaussian fitted to everything), your `GaussianMixture` with two
    components (score $-\log p(x)$), your `knn_distance_scores` ($k = 10$) and your
    `IsolationForest`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("rank the four detectors by AUC on the anomalies in the middle.")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _X = np.r_[[-4.0, 0.0] + _rng.standard_normal((500, 2)), [4.0, 0.0] + _rng.standard_normal((500, 2)),
               0.3 * _rng.standard_normal((15, 2))]
    _y = np.r_[np.zeros(1000), np.ones(15)].astype(int)
    _scores = {
        "one Gaussian (Mahalanobis)": ex.gaussian_scores(_X, _X),
        "GMM, 2 components": -GaussianMixture(2, rng=0).fit(_X).score_samples(_X),
        "10-NN distance": ex.knn_distance_scores(_X, 10),
        "isolation forest": ex.IsolationForest(100, rng=0).fit(_X).score_samples(_X),
    }
    _fig, _axes = plt.subplots(1, 4, figsize=(17, 3.8))
    _rows = []
    for _ax, (_name, _s) in zip(_axes, _scores.items()):
        _r = np.argsort(np.argsort(-_s))  # rank 0 = most anomalous
        _ax.scatter(_X[:, 0], _X[:, 1], c=np.argsort(np.argsort(_s)), cmap="magma_r", s=5)
        _ax.scatter(_X[_y == 1, 0], _X[_y == 1, 1], facecolors="none", edgecolors="C0", s=40, lw=0.8)
        _ax.set_aspect("equal")
        _auc = roc_auc_score(_y, _s)
        _ax.set_title(f"{_name}\nAUC {_auc:.3f}", fontsize=10)
        _rows.append(f"| {_name} | {_auc:.3f} | {np.median(_r[_y == 1]):.0f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| detector | AUC | median rank of the 15 anomalies (0 = most anomalous of 1,015) |\n|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where is the single Gaussian's mean, and what does that do to the
    Mahalanobis distance of the points in the middle? why does the 2-component GMM fix it?
    the $k$-NN score and the isolation forest don't assume any shape: why are they still
    short of perfect (think about what a clump of 15 points does, as in experiment 2)?
    """)
    return


if __name__ == "__main__":
    app.run()
