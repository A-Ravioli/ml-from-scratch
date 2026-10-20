import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.kernels import KNeighborsClassifier  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    OFFLINE = datasets.is_offline()


@app.function
def moons(n, noise, rng):
    """two interleaving half circles with Gaussian noise: X (n, 2), y in {0, 1}."""
    y = rng.integers(0, 2, size=n)
    t = rng.uniform(0, np.pi, size=n)
    X = np.where(y[:, None] == 0, np.column_stack([np.cos(t), np.sin(t)]),
                 np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]))
    return X + noise * rng.normal(size=(n, 2)), y


@app.function
def boundary(ax, model, X, y, title, lim=((-1.7, 2.7), (-1.5, 1.8))):
    """decision_function contours (0 = boundary, ±1 = margins), the data, and circled support vectors."""
    gx, gy = np.meshgrid(np.linspace(*lim[0], 150), np.linspace(*lim[1], 120))
    f = model.decision_function(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    ax.contourf(gx, gy, f, levels=[-1e9, 0, 1e9], colors=["#dbe8f5", "#f8dfcf"])
    ax.contour(gx, gy, f, levels=[-1, 0, 1], colors="k", linestyles=[":", "-", ":"], linewidths=[0.8, 1.2, 0.8])
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, "C1", "C0"), s=8)
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=40, facecolors="none", edgecolors="k", linewidths=0.6)
    ax.set(xticks=[], yticks=[], xlim=lim[0], ylim=lim[1])
    ax.set_title(title, fontsize=8.5)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # support vector machines · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    {"**offline mode:** MNIST in experiment 3 is a synthetic stand-in; run without `MLFS_OFFLINE` for the real digits." if OFFLINE else ""}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: sweeping C and γ

    200 training points from two noisy moons, 1,000 test points from the same distribution.
    your `SVC` with an RBF kernel, for every combination of $C \in \{0.1, 1, 100\}$ and
    $\gamma \in \{0.1, 1, 30\}$. solid line: the boundary $f = 0$; dotted: the margins
    $f = \pm 1$; circled: the support vectors.
    """)
    return


@app.cell
def _():
    noise = mo.ui.slider(0.1, 0.5, step=0.05, value=0.25, label="moon noise")
    noise
    return (noise,)


@app.cell
def _():
    p1 = ui.predict(
        "which corner of the C × γ grid overfits most, and which underfits? where will almost every "
        "training point be a support vector?"
    )
    p1
    return (p1,)


@app.cell
def _(noise, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X, _y = moons(200, noise.value, _rng)
    _Xt, _yt = moons(1000, noise.value, _rng)
    _fig, _axes = plt.subplots(3, 3, figsize=(11, 8.5))
    _rows = []
    for _r, _C in enumerate([0.1, 1.0, 100.0]):
        for _c, _g in enumerate([0.1, 1.0, 30.0]):
            _m = ex.SVC(_C, gamma=_g, rng=0).fit(_X, _y)
            _tr, _te = _m.score(_X, _y), _m.score(_Xt, _yt)
            _rows.append((_C, _g, _tr, _te, len(_m.support_)))
            boundary(_axes[_r, _c], _m, _X, _y,
                     f"C={_C:g}, γ={_g:g}: train {_tr:.0%}, test {_te:.0%}, {len(_m.support_)} SV")
    _fig.tight_layout()
    _best = max(_rows, key=lambda r: r[3])
    mo.vstack([_fig, mo.md(
        f"best test accuracy: **{_best[3]:.1%}** at C = {_best[0]:g}, γ = {_best[1]:g} ({_best[4]} support vectors). "
        "all nine: " + "; ".join(f"({C:g}, {g:g}) {te:.1%}/{sv} SV" for C, g, _, te, sv in _rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** read each row and each column. what does $\gamma$ do to the shape of the
    boundary, and what does $C$ do to the width of the margin? why is the number of support
    vectors largest where it is? which settings have train accuracy far above test accuracy?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: only the support vectors matter

    one SVM ($C = 1$, $\gamma = 1$, a tight `tol` of $10^{-5}$) on 200 moon points. we refit it
    on its support vectors alone, and compare the two decision functions on a grid. then we remove a single support
    vector and refit. for contrast, the same deletions for your 15-nearest-neighbour classifier
    from topic 01.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "you delete every point that isn't a support vector (more than half the data) and refit. how much "
        "does the SVM's decision function change? and the kNN classifier's predictions?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _X, _y = moons(200, 0.25, _rng)
    _G = np.column_stack([a.ravel() for a in np.meshgrid(np.linspace(-1.6, 2.6, 80), np.linspace(-1.2, 1.7, 60))])
    _full = ex.SVC(1.0, gamma=1.0, tol=1e-5, rng=0).fit(_X, _y)
    _sv = _full.support_
    _only = ex.SVC(1.0, gamma=1.0, tol=1e-5, rng=0).fit(_X[_sv], _y[_sv])
    _free = _sv[_full.alpha_[_sv] < 1.0 - 1e-8]
    _drop = _free[np.argmax(_full.alpha_[_free])] if len(_free) else _sv[0]
    _keep = np.setdiff1d(np.arange(len(_X)), [_drop])
    _minus = ex.SVC(1.0, gamma=1.0, tol=1e-5, rng=0).fit(_X[_keep], _y[_keep])
    _f0 = _full.decision_function(_G)
    _d_only = np.max(np.abs(_only.decision_function(_G) - _f0))
    _d_minus = np.max(np.abs(_minus.decision_function(_G) - _f0))
    _s_only = np.mean(np.sign(_only.decision_function(_G)) != np.sign(_f0))
    _s_minus = np.mean(np.sign(_minus.decision_function(_G)) != np.sign(_f0))
    _knn = KNeighborsClassifier(15).fit(_X, _y).predict(_G)
    _knn_only = np.mean(KNeighborsClassifier(15).fit(_X[_sv], _y[_sv]).predict(_G) != _knn)
    _knn_minus = np.mean(KNeighborsClassifier(15).fit(_X[_keep], _y[_keep]).predict(_G) != _knn)
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.8))
    boundary(_axes[0], _full, _X, _y, f"all 200 points ({len(_sv)} SV)")
    boundary(_axes[1], _only, _X[_sv], _y[_sv], f"refit on the {len(_sv)} SVs only")
    boundary(_axes[2], _minus, _X[_keep], _y[_keep], "one free SV removed")
    _axes[2].plot(*_X[_drop], "rx", ms=10, mew=2)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"SVM: refitting on the {len(_sv)} support vectors changes the decision function by at most **{_d_only:.3f}** "
        f"({_s_only:.2%} of grid predictions flip); removing the one free support vector with the largest α changes it "
        f"by up to **{_d_minus:.3f}** ({_s_minus:.2%} flip). kNN (k = 15): "
        f"**{_knn_only:.1%}** of grid predictions change on the support vectors only, **{_knn_minus:.1%}** "
        "after removing the one point.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the refit on the support vectors (essentially) identical? use the
    KKT conditions. why does one support vector matter when 100 other points don't? why does
    kNN behave so differently?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: 4s versus 9s

    1,000 MNIST training digits that are 4s or 9s (pixels in $[0, 1]$), and every 4 and 9 in
    the test set. your `SVC` with an RBF kernel, $\gamma = 0.02$, for several $C$. we report
    the test error with its standard error, and the fraction of training points that are
    support vectors, which bounds the leave-one-out error (on paper 3).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "4 and 9 are among the most confused digits. what test error will the SVM reach with 1,000 training "
        "digits? will the support-vector bound be close to it, or far above?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _d = datasets.mnist()
    _rng = np.random.default_rng(2)
    _tr = np.flatnonzero(np.isin(_d["y_train"], [4, 9]))
    _tr = _rng.choice(_tr, size=min(1000, len(_tr)), replace=False)
    _te = np.flatnonzero(np.isin(_d["y_test"], [4, 9]))
    _Xtr = _d["X_train"][_tr].reshape(len(_tr), -1) / 255.0
    _Xte = _d["X_test"][_te].reshape(len(_te), -1) / 255.0
    _ytr, _yte = _d["y_train"][_tr], _d["y_test"][_te]
    _Cs = [0.1, 1.0, 10.0, 100.0]
    _res = []
    for _C in _Cs:
        _m = ex.SVC(_C, gamma=0.02, rng=0).fit(_Xtr, _ytr)
        _err = 1 - _m.score(_Xte, _yte)
        _res.append((_C, _err, np.sqrt(_err * (1 - _err) / len(_yte)), len(_m.support_) / len(_tr), 1 - _m.score(_Xtr, _ytr)))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.semilogx(_Cs, [r[1] for r in _res], "o-", label="test error")
    _ax.fill_between(_Cs, [r[1] - 2 * r[2] for r in _res], [r[1] + 2 * r[2] for r in _res], alpha=0.2)
    _ax.semilogx(_Cs, [r[3] for r in _res], "s--", label="#SV / n (bound on leave-one-out error)")
    _ax.semilogx(_Cs, [r[4] for r in _res], "^:", label="training error")
    _ax.set(xlabel="C", ylabel="error", title=f"4 vs 9: {len(_tr)} training, {len(_te)} test digits", ylim=(0, None))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(" · ".join(
        f"C = {C:g}: test error **{e:.1%}** ± {s:.1%}, SV {f:.0%}, train error {t:.1%}" for C, e, s, f, t in _res))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does $C$ trade training error against the number of support vectors?
    the support-vector bound is valid but loose: why? why does the test error flatten for large
    $C$ while the training error goes to zero?
    """)
    return


if __name__ == "__main__":
    app.run()
