import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.model_selection import KFold  # yours, from ch 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def sparse_problem(n, rng, d=10, corr=0.3, noise=1.0):
    """standardized features with pairwise correlation `corr`; only the first three matter."""
    C = corr * np.ones((d, d)) + (1 - corr) * np.eye(d)
    X = rng.normal(size=(n, d)) @ np.linalg.cholesky(C).T
    X = (X - X.mean(0)) / X.std(0)
    w = np.zeros(d)
    w[:3] = [3.0, -2.0, 1.5]
    y = X @ w + noise * rng.normal(size=n)
    return X, y - y.mean(), w


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: ridge paths and lasso paths

    100 rows, 10 standardized features with pairwise correlation 0.3. only the first three
    matter: $w = (3, -2, 1.5, 0, \dots, 0)$, noise $\sigma = 1$. you compute the whole ridge
    path with your `ridge_path` and the whole lasso path with your `lasso_path`, over 60
    values of $\alpha$ each, and count how many coefficients are *exactly* zero along the way.
    (the two penalties are on different scales, so each gets its own range of $\alpha$.)
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "along the ridge path, at what alpha does the first coefficient become exactly 0? along "
        "the lasso path, in what order do the 10 coefficients become nonzero as alpha decreases?"
    )
    p1
    return (p1,)


@app.cell
def _():
    where = mo.ui.slider(0, 59, value=30, label="position along the lasso path", show_value=True)
    where
    return (where,)


@app.cell
def _(p1, where):
    ui.gate(p1)
    _X, _y, _w = sparse_problem(100, np.random.default_rng(0))
    _ra = np.geomspace(1e-2, 1e5, 60)
    _R = np.asarray(ex.ridge_path(_X, _y, _ra))
    _amax = ex.lasso_alpha_max(_X, _y)
    _la = np.geomspace(_amax, 1e-3 * _amax, 60)
    _L = np.asarray(ex.lasso_path(_X, _y, _la, max_iters=5000, tol=1e-9))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for _j in range(10):
        _style = dict(lw=2) if _j < 3 else dict(lw=1, color="gray", alpha=0.6)
        _a1.semilogx(_ra, _R[:, _j], **_style)
        _a2.semilogx(_la, _L[:, _j], **_style)
    _a1.set(xlabel=r"ridge $\alpha$", ylabel="coefficient", title="ridge path")
    _a2.set(xlabel=r"lasso $\alpha$", title="lasso path")
    _a2.axvline(_la[where.value], color="k", ls=":")
    for _a in (_a1, _a2):
        _a.invert_xaxis()
        _a.axhline(0, color="k", lw=0.5)
    _fig.tight_layout()
    _entry = [int(np.argmax(_L[:, _j] != 0)) if np.any(_L[:, _j] != 0) else 60 for _j in range(10)]
    _order = np.argsort(_entry, kind="stable")
    _here = _L[where.value]
    mo.vstack([_fig, mo.md(
        f"exactly-zero coefficients anywhere on the ridge path: **{int(np.sum(_R == 0))}** of {_R.size}. "
        f"order in which lasso coefficients become nonzero: {[int(j) for j in _order]}. "
        f"at the dotted line ($\\alpha$ = {_la[where.value]:.3g}): **{int(np.sum(_here != 0))}** nonzero "
        f"coefficients, on features {np.flatnonzero(_here).tolist()}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does ridge shrink every coefficient smoothly towards 0 without ever
    reaching it, while the lasso sets them to exactly 0, one at a time? use the soft-threshold
    update: what has to be true about a feature's correlation with the residual for its
    coefficient to stay at 0? why do the three real features enter first, and why do some
    irrelevant ones still enter at small $\alpha$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: two features that say the same thing

    $x_1$ and $x_2$ are near-copies of each other (correlation 0.995), and $y = x_1 + x_2 +
    \varepsilon$, plus 5 irrelevant features. you draw 40 bootstrap resamples of 100 rows and
    fit, on each, your `Lasso` ($\alpha = 0.1$), your `Ridge` ($\alpha = 10$) and your
    `ElasticNet` ($\alpha = 0.1$, `l1_ratio` = 0.5), and plot the two coefficients
    $(w_1, w_2)$ that each model gives on each resample.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the truth is w1 = w2 = 1. what will the lasso's (w1, w2) look like across the 40 "
        "resamples? and ridge's?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n = 100
    _x1 = _rng.normal(size=_n)
    _X = np.column_stack([_x1, _x1 + 0.1 * _rng.normal(size=_n), _rng.normal(size=(_n, 5))])
    _y = _X[:, 0] + _X[:, 1] + _rng.normal(size=_n)
    _models = {"lasso": lambda: ex.Lasso(alpha=0.1, max_iters=5000, tol=1e-8),
               "ridge": lambda: ex.Ridge(alpha=10.0),
               "elastic net": lambda: ex.ElasticNet(alpha=0.1, l1_ratio=0.5, max_iters=5000, tol=1e-8)}
    _coefs = {k: [] for k in _models}
    for _b in range(40):
        _idx = _rng.integers(0, _n, _n)
        for _k, _make in _models.items():
            _coefs[_k].append(_make().fit(_X[_idx], _y[_idx]).coef_[:2])
    _fig, _ax = plt.subplots(figsize=(5.2, 5.0))
    for _k, _mk in zip(_models, ("o", "s", "^")):
        _c = np.array(_coefs[_k])
        _ax.scatter(_c[:, 0], _c[:, 1], marker=_mk, label=_k, alpha=0.7)
    _ax.plot([0, 2], [2, 0], "k:", lw=1, label="$w_1 + w_2 = 2$")
    _ax.plot(1, 1, "k*", ms=12, label="truth")
    _ax.set(xlabel="$w_1$", ylabel="$w_2$", xlim=(-0.3, 2.3), ylim=(-0.3, 2.3), aspect="equal")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _k in _models:
        _c = np.array(_coefs[_k])
        _rows.append(f"| {_k} | {np.mean(np.any(_c == 0, axis=1)):.0%} | {_c[:, 0].std():.2f} | "
                     f"{np.mean(_c.sum(axis=1)):.2f} |")
    mo.vstack([_fig, mo.md("| model | resamples with $w_1$ or $w_2$ exactly 0 | std of $w_1$ | mean of $w_1 + w_2$ |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the lasso tend to keep one of the twins and drop the other, and
    why does *which* one it keeps change from resample to resample? why does ridge split the
    weight evenly? the sum $w_1 + w_2$ is about the same for all three models: what does that
    say about their predictions, versus their coefficients? if you had to report "which
    features matter", which model would you trust?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the Bayesian view, measured

    the lesson says ridge is the posterior mean when $w \sim \mathcal N(0, \tau^2 I)$ and the
    noise has variance $\sigma^2$, with $\alpha = \sigma^2/\tau^2$. here the data really come
    from that model: $n = 40$ rows, $d = 30$ features, $\sigma = 1$, and a fresh $w$ and
    dataset on each of 60 draws. for each draw you compute, with your `ridge_path`, the test
    error of every $\alpha$ on a grid, and the $\alpha$ that 5-fold cross-validation (your
    `KFold`) would pick.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with tau = 0.3 the theory says alpha = 1 / 0.09 = 11. which alpha minimizes the average "
        "test error? how widely does cross-validation's choice scatter around it?"
    )
    p3
    return (p3,)


@app.cell
def _():
    tau = mo.ui.dropdown({"0.1": 0.1, "0.3": 0.3, "1.0": 1.0}, value="0.3", label="prior scale tau")
    tau
    return (tau,)


@app.cell
def _(p3, tau):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _n, _d, _sigma, _t = 40, 30, 1.0, tau.value
    _alphas = np.geomspace(1e-2, 1e4, 49)
    _test_err = np.zeros((60, len(_alphas)))
    _picked = []
    for _r in range(60):
        _w = _t * _rng.normal(size=_d)
        _X = _rng.normal(size=(_n, _d))
        _y = _X @ _w + _sigma * _rng.normal(size=_n)
        _Xt = _rng.normal(size=(500, _d))
        _yt = _Xt @ _w + _sigma * _rng.normal(size=500)
        _coefs = np.asarray(ex.ridge_path(_X, _y, _alphas))
        _test_err[_r] = np.mean((_Xt @ _coefs.T - _yt[:, None]) ** 2, axis=0)
        _cv = np.zeros(len(_alphas))
        for _tr, _va in KFold(5, shuffle=True, rng=_r).split(_X):
            _c = np.asarray(ex.ridge_path(_X[_tr], _y[_tr], _alphas))
            _cv += np.mean((_X[_va] @ _c.T - _y[_va, None]) ** 2, axis=0)
        _picked.append(_alphas[np.argmin(_cv)])
    _mean_err = _test_err.mean(axis=0)
    _theory = _sigma**2 / _t**2
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    _a1.semilogx(_alphas, _mean_err, "o-", ms=3)
    _a1.axvline(_theory, color="k", ls=":", label=r"$\sigma^2/\tau^2$")
    _a1.set(xlabel=r"$\alpha$", ylabel="test MSE (mean of 60 draws)", title="average test error")
    _a1.legend()
    _a2.hist(np.log10(_picked), bins=np.linspace(-2, 4, 25))
    _a2.axvline(np.log10(_theory), color="k", ls=":")
    _a2.set(xlabel=r"$\log_{10}$ of the $\alpha$ picked by 5-fold CV", ylabel="draws", title="CV's choice")
    _fig.tight_layout()
    _best = _alphas[np.argmin(_mean_err)]
    _oracle = np.mean(_test_err.min(axis=1))
    _cv_err = np.mean([_test_err[_r, np.searchsorted(_alphas, _picked[_r])] for _r in range(60)])
    _theory_err = np.interp(np.log(_theory), np.log(_alphas), _mean_err)
    mo.vstack([_fig, mo.md(
        f"$\\sigma^2/\\tau^2$ = **{_theory:.3g}**; the $\\alpha$ with the lowest average test MSE = **{_best:.3g}**; "
        f"median CV choice = **{np.median(_picked):.3g}** (middle half {np.quantile(_picked, 0.25):.3g} to "
        f"{np.quantile(_picked, 0.75):.3g}).\n\n"
        f"test MSE: least squares ($\\alpha$ = 0.01) {_mean_err[0]:.2f} · $\\alpha = \\sigma^2/\\tau^2$ {_theory_err:.2f} · "
        f"CV's choice {_cv_err:.2f} · best $\\alpha$ per draw (cheating) {_oracle:.2f} · noise floor {_sigma**2:.2f}")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is $\alpha = \sigma^2/\tau^2$ the best choice *on average* (what does the
    posterior mean minimize)? why does unregularized least squares do so badly with 40 rows and
    30 features? cross-validation knows nothing about $\tau$ or $\sigma$: how does it find
    roughly the right $\alpha$, and why does its choice scatter so much while costing so little
    test error? change $\tau$ and check that the optimum moves the way the theory says.
    """)
    return


if __name__ == "__main__":
    app.run()
