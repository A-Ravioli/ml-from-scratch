import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.metrics import log_loss  # yours, from ch 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gradient boosting · lab

    every experiment here runs **your** booster from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    experiments 1 and 2 use California housing (median house value of a district, in
    \$100,000, from 8 numeric features): 5,000 random districts for training and 5,000
    others for testing, so that each fit takes seconds. offline (`MLFS_OFFLINE=1`) you get
    the small synthetic stand-in, and the numbers mean little. the lab takes about a minute.
    """)
    return


@app.cell
def _():
    X_all, y_all, names = datasets.california_housing()
    _idx = np.random.default_rng(0).permutation(len(y_all))
    _n = min(5000, len(y_all) // 2)
    tr, te = _idx[:_n], _idx[_n:2 * _n]
    X_tr, y_tr, X_te, y_te = X_all[tr], y_all[tr], X_all[te], y_all[te]
    mo.md(f"{len(y_all):,} districts available · train {len(y_tr):,} · test {len(y_te):,}")
    return X_te, X_tr, y_te, y_tr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: learning rate versus number of trees

    depth-4 trees, $\lambda = 1$, all 256 bins. for learning rates 1, 0.3, 0.1 and 0.03 we
    boost up to 2,000 trees and record the test RMSE after every tree (`staged_predict`).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which learning rate reaches the lowest test RMSE, and after roughly how many trees? does learning_rate=1 "
        "end up as good as the others if you give it enough trees?"
    )
    p1
    return (p1,)


@app.cell
def _(X_te, X_tr, p1, y_te, y_tr):
    ui.gate(p1)
    _settings = [(1.0, 200), (0.3, 600), (0.1, 2000), (0.03, 2000)]
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _rows = []
    for _lr, _M in _settings:
        _m = ex.GradientBoostingRegressor(n_estimators=_M, learning_rate=_lr, max_depth=4).fit(X_tr, y_tr)
        _rmse = np.array([np.sqrt(np.mean((p - y_te) ** 2)) for p in _m.staged_predict(X_te)])
        _ax.semilogx(np.arange(1, _M + 1), _rmse, label=f"learning_rate {_lr}")
        _b = int(np.argmin(_rmse))
        _rows.append(f"| {_lr} | {_rmse[_b]:.4f} | {_b + 1} | {_rmse[-1]:.4f} ({_M} trees) |")
    _ax.set(xlabel="number of trees", ylabel="test RMSE ($100k)", ylim=(None, 0.9))
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| learning rate | best test RMSE | at tree | at the last tree |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does a smaller learning rate reach a *lower* minimum, not just the
    same minimum more slowly? why does learning_rate=1 overfit so quickly, and does it ever
    recover? how does the number of trees at the minimum scale with the learning rate? what
    does this mean for how you'd tune the two in practice?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many bins?

    the same data, learning rate 0.1, depth 4, 400 trees; only `n_bins` changes, from 2 (one
    possible threshold per feature) to 256.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 400 trees, how many bins per feature do you need to get within 1% of the 256-bin test RMSE?"
    )
    p2
    return (p2,)


@app.cell
def _(X_te, X_tr, p2, y_te, y_tr):
    ui.gate(p2)
    _bins = [2, 4, 8, 16, 32, 64, 128, 256]
    _rmse = []
    for _b in _bins:
        _m = ex.GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=4, n_bins=_b).fit(X_tr, y_tr)
        _rmse.append(np.sqrt(np.mean((_m.predict(X_te) - y_te) ** 2)))
    _rmse = np.array(_rmse)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.semilogx(_bins, _rmse, "o-", base=2)
    _ax.axhline(_rmse[-1] * 1.01, color="gray", ls=":", label="256 bins + 1%")
    _ax.set(xlabel="n_bins", ylabel="test RMSE ($100k)", xticks=_bins, xticklabels=[str(b) for b in _bins])
    _ax.legend()
    _fig.tight_layout()
    _ok = _bins[int(np.argmax(_rmse <= _rmse[-1] * 1.01))]
    mo.vstack([_fig, mo.md(
        "| n_bins | " + " | ".join(map(str, _bins)) + " |\n|---|" + "---|" * len(_bins) + "\n| test RMSE | "
        + " | ".join(f"{r:.4f}" for r in _rmse) + " |\n\n"
        f"the first bin count within 1% of 256 bins: **{_ok}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the folklore says boosting barely cares about the bin count once there are
    a few dozen bins. what did California show instead, and which features need the fine
    thresholds (think about where house prices jump)? why are 2 bins so bad? on what kind of
    data would 16 bins be enough?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what does the second derivative buy?

    a synthetic, imbalanced classification problem: 6 features, 4,000 rows, about 8%
    positives. we run the classifier's boosting loop by hand with your `HistGradientTree`,
    twice with the same learning rate 0.3 and depth 3: once with the true hessians
    $h = p(1 - p)$ (Newton leaves, your classifier) and once with $h = 1$ (plain gradient
    leaves: each leaf moves by the mean of $t - p$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "Newton leaves vs gradient leaves, same learning rate: after 100 trees, how far apart are the test log "
        "losses? does the gradient version catch up eventually?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(1)
    _X = _rng.normal(size=(8000, 6))
    _logit = 1.5 * _X[:, 0] - _X[:, 1] ** 2 + _X[:, 2] * _X[:, 3] - 3.0
    _t = (_rng.random(8000) < 1 / (1 + np.exp(-_logit))).astype(float)
    _Xtr, _ttr, _Xte, _tte = _X[:4000], _t[:4000], _X[4000:], _t[4000:]
    _edges = ex.quantile_bin_edges(_Xtr, 256)
    _ctr, _cte = ex.bin_data(_Xtr, _edges), ex.bin_data(_Xte, _edges)
    _rate = _ttr.mean()
    _curves = {}
    for _name in ("Newton: h = p(1 − p)", "gradient: h = 1"):
        _F = np.full(4000, np.log(_rate / (1 - _rate)))
        _Fte = np.full(4000, _F[0])
        _ll = []
        for _ in range(400):
            _p = 0.5 * (1 + np.tanh(_F / 2))
            _h = _p * (1 - _p) if _name.startswith("Newton") else np.ones(4000)
            _tree = ex.HistGradientTree(max_depth=3, min_child_weight=1e-3, reg_lambda=1.0).fit(_ctr, _p - _ttr, _h)
            _F = _F + 0.3 * _tree.predict(_ctr)
            _Fte = _Fte + 0.3 * _tree.predict(_cte)
            _ll.append(log_loss(_tte.astype(int), 0.5 * (1 + np.tanh(_Fte / 2))))
        _curves[_name] = np.array(_ll)
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    for _name, _ll in _curves.items():
        _ax.semilogx(np.arange(1, 401), _ll, label=_name)
    _ax.set(xlabel="number of trees", ylabel="test log loss")
    _ax.legend()
    _fig.tight_layout()
    _n, _g = _curves["Newton: h = p(1 − p)"], _curves["gradient: h = 1"]
    _target = 1.01 * min(_n.min(), _g.min())

    def _first(c):
        return str(int(np.argmax(c <= _target)) + 1) if np.any(c <= _target) else "never (in 400)"

    mo.vstack([_fig, mo.md(
        f"positives: {_ttr.mean():.1%} of the training rows. after 10 / 100 / 400 trees: Newton "
        f"{_n[9]:.4f} / {_n[99]:.4f} / {_n[-1]:.4f}, gradient {_g[9]:.4f} / {_g[99]:.4f} / {_g[-1]:.4f}. "
        f"best: Newton **{_n.min():.4f}** (tree {int(np.argmin(_n)) + 1}), gradient **{_g.min():.4f}** "
        f"(tree {int(np.argmin(_g)) + 1}). trees needed to get within 1% of the better minimum "
        f"({_target:.4f}): Newton **{_first(_n)}**, gradient **{_first(_g)}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with only 8% positives, most rows have $p$ near 0.08. how big is $h$ there,
    and how does dividing by $\sum h$ instead of the row count change a leaf's step? why does
    that matter most for leaves full of rare positives? is the gradient version wrong, or just
    slow? and why does the Newton version start to overfit after its minimum (look at the
    `min_child_weight` we used)?
    """)
    return


if __name__ == "__main__":
    app.run()
