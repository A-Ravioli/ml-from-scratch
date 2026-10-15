import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui
    from mlfs.linear import Lasso, LinearRegression, LogisticRegression  # yours, from ch 08
    from mlfs.trees import (  # yours, from ch 10
        DecisionTreeRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # double machine learning · lab

    every experiment here runs **your** DML code from `exercise.py`, with **your** part II
    models (the lasso from ch 08, trees and gradient boosting from ch 10) as the nuisance
    learners. for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare with `solutions/lab_notes.md` afterwards. the whole lab takes about a
    minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: regularization bias

    a high-dimensional partially linear model: $n = 200$ units, $p = 180$ correlated
    covariates ($\mathrm{corr}(x_j, x_k) = 0.5^{|j-k|}$), and confounding spread over many of
    them with decaying weights $\beta_j = 1/j^2$:

    $$d = 1.5\,x^\top\beta + v,\qquad y = 0.5\,d + 3\,x^\top\beta + u.$$

    with almost as many covariates as units, plain OLS is barely possible, so you reach for
    the lasso. three estimators of $\theta = 0.5$: OLS on $(d, x)$; the **naive plug-in**
    (your `naive_plugin` with a lasso for $g$); and **DML** (your `dml_plr`, a lasso for each
    nuisance, 5 folds). first 80 datasets at lasso penalty $\alpha = 0.1$, then 30 each at
    three other penalties.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the naive plug-in and DML use the same lasso. will they give similar answers? which is biased, "
        "and does the bias depend on the penalty alpha? how does OLS compare?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    n1, p_dim = 200, 180
    _S = 0.5 ** np.abs(np.subtract.outer(np.arange(p_dim), np.arange(p_dim)))
    _L = np.linalg.cholesky(_S)
    _beta = 1.0 / np.arange(1, p_dim + 1) ** 2

    def plr_world(rng):
        X = rng.normal(size=(n1, p_dim)) @ _L.T
        d = 1.5 * X @ _beta + rng.normal(size=n1)
        return X, 0.5 * d + 3 * X @ _beta + rng.normal(size=n1), d

    def run1(alpha, reps, seed):
        las = lambda: Lasso(alpha, max_iters=200, tol=1e-4)  # noqa: E731
        rng = np.random.default_rng(seed)
        out = {"OLS": [], "naive plug-in (lasso)": [], "DML (lasso)": []}
        cover = []
        for r in range(reps):
            X, y, d = plr_world(rng)
            A = np.column_stack([np.ones(n1), d, X])
            out["OLS"].append(np.linalg.lstsq(A, y, rcond=None)[0][1])
            out["naive plug-in (lasso)"].append(ex.naive_plugin(X, y, d, las, n_iters=10))
            o = ex.dml_plr(X, y, d, las, las, n_folds=5, rng=r)
            out["DML (lasso)"].append(o["theta"])
            cover.append(abs(o["theta"] - 0.5) <= 1.96 * o["se"])
        return {k: np.array(v) for k, v in out.items()}, np.mean(cover)

    res1, cov1 = run1(0.1, 80, 0)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _bins = np.linspace(-0.4, 1.4, 46)
    for _k, _v in res1.items():
        _ax.hist(_v, bins=_bins, alpha=0.55,
                 label=f"{_k}: bias {_v.mean() - 0.5:+.3f} ± {_v.std(ddof=1) / np.sqrt(len(_v)):.3f}, sd {_v.std(ddof=1):.3f}")
    _ax.axvline(0.5, color="k", ls="--", lw=1)
    _ax.set(xlabel="estimate of θ (truth 0.5)", ylabel="datasets", title="80 datasets, lasso α = 0.1")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _a in [0.05, 0.2, 0.4]:
        _r, _c = run1(_a, 30, 1)
        _rows.append(f"| {_a} | {_r['naive plug-in (lasso)'].mean() - 0.5:+.3f} | {_r['DML (lasso)'].mean() - 0.5:+.3f} "
                     f"| {_c:.0%} |")
    _rows.insert(1, f"| 0.1 | {res1['naive plug-in (lasso)'].mean() - 0.5:+.3f} | {res1['DML (lasso)'].mean() - 0.5:+.3f} "
                    f"| {cov1:.0%} |")
    _rng_ols = np.random.default_rng(999)  # OLS is cheap: pin down its bias on many more datasets
    _ols = []
    for _ in range(1000):
        _X, _y, _d = plr_world(_rng_ols)
        _ols.append(np.linalg.lstsq(np.column_stack([np.ones(n1), _d, _X]), _y, rcond=None)[0][1])
    _ols = np.array(_ols)
    mo.vstack([_fig, mo.md(
        f"bias ± its standard error over the datasets. OLS on 1,000 further datasets: bias **{_ols.mean() - 0.5:+.3f} "
        f"± {_ols.std(ddof=1) / np.sqrt(1000):.3f}**, sd {_ols.std(ddof=1):.3f}.\n\n"
        "| lasso α | naive bias | DML bias | DML 95% CI coverage |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both estimators use the same lasso, and the lasso shrinks every
    coefficient towards zero. why does that shrinkage go straight into the naive estimate
    (which part of $g$ does the lasso miss, and how is it correlated with $d$?) but only
    as a *product* of two errors into DML? OLS is unbiased (check the 1,000-dataset line): why
    is it still a poor choice here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: why cross-fit?

    a low-dimensional nonlinear world, $x \in \mathbb R^3$, $h(x) = \sin 2x_0 + x_1^2/2 + x_2$:

    $$d = h(x) + v,\qquad y = 0.5\,d + 2h(x) + u,\qquad n = 400.$$

    the nuisance models are your decision trees, from heavily regularized (at least 50
    units per leaf) to fully grown (1 per leaf). each is used two ways: **in-sample**
    (fit both nuisances on all the data and partial out with in-sample residuals) and
    **cross-fitted** (your `dml_plr`, 5 folds). 20 datasets per setting.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "as the trees get more flexible (fewer units per leaf), what happens to the in-sample estimate and to the "
        "cross-fitted one? which setting gives the best cross-fitted estimate?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def tree_world(rng, n=400):
        X = rng.normal(size=(n, 3))
        h = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + X[:, 2]
        d = h + rng.normal(size=n)
        return X, 0.5 * d + 2 * h + rng.normal(size=n), d

    leaves = [50, 20, 10, 5, 2, 1]
    res2 = {}
    for _leaf in leaves:
        _mk = lambda: DecisionTreeRegressor(min_samples_leaf=_leaf)  # noqa: E731
        _rng = np.random.default_rng(1)
        _ins, _cf, _rd, _oof = [], [], [], []
        for _r in range(20):
            _X, _y, _d = tree_world(_rng)
            _ry = _y - _mk().fit(_X, _y).predict(_X)
            _rdd = _d - _mk().fit(_X, _d).predict(_X)
            _rd.append(np.mean(_rdd**2))
            _oof.append(np.mean((_d - ex.cross_fit_predict(_mk, _X, _d, n_folds=5, rng=_r)) ** 2))
            _ins.append(ex.partialling_out(_ry, _rdd)[0] if np.sum(_rdd**2) > 1e-12 else np.nan)
            _cf.append(ex.dml_plr(_X, _y, _d, _mk, _mk, n_folds=5, rng=_r)["theta"])
        res2[_leaf] = (np.array(_ins), np.array(_cf), np.mean(_rd), np.mean(_oof))
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _x = np.arange(len(leaves))
    def _ms(v):
        v = v[~np.isnan(v)]
        return (v.mean(), v.std(ddof=1)) if len(v) > 1 else (np.nan, np.nan)

    for _k, (_lab, _mk2) in enumerate([("in-sample residuals", "o-"), ("cross-fitted (DML)", "s-")]):
        _st = np.array([_ms(res2[l][_k]) for l in leaves])
        _ax.errorbar(_x, _st[:, 0], yerr=_st[:, 1], fmt=_mk2, capsize=3, label=_lab)
    _ax.axhline(0.5, color="k", ls="--", lw=1, label="truth 0.5")
    _ax.set_xticks(_x, [str(l) for l in leaves])
    _ax.set(xlabel="min samples per leaf (← more regularized | more flexible →)", ylabel="estimate of θ (mean ± sd)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| min per leaf | in-sample mean ± sd | cross-fitted mean ± sd | in-sample mean of (d - m̂)² "
        "| out-of-fold mean of (d - m̂)² |\n|---|---|---|---|---|\n"
        + "\n".join(("| {} | {} | {:.3f} ± {:.3f} | {:.3f} | {:.3f} |").format(
            l, "undefined: every residual is 0" if np.isnan(_ms(res2[l][0])[0]) else "{:.3f} ± {:.3f}".format(*_ms(res2[l][0])),
            res2[l][1].mean(), res2[l][1].std(ddof=1), res2[l][2], res2[l][3]) for l in leaves))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the true variance of $v = d - m(x)$ is 1. what do the last two columns say
    the in-sample and out-of-fold residuals are doing as the trees grow, and what happens to
    an estimator that divides by $\sum\tilde d^2$? why are heavily regularized trees bad for
    both versions, why is 2 per leaf the best cross-fitted setting, and why does 1 per leaf
    fall *below* the truth (on paper 3: what does the $-\theta\,\mathbb E[\delta_m^2]$ part of the
    error do when $\hat m$ is very noisy)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a semi-synthetic study

    the covariates are **real**: 1,500 California districts per replicate from
    `mlfs.datasets.california_housing` (log-transformed counts, standardized). treatment and
    outcome are simulated from them, so the true effect is known (the IHDP recipe of
    Hill, 2011):

    $$\operatorname{logit} e = 1.2\,\text{inc} - 0.6\,\text{age}^2 + 0.5\tanh(\text{lat}) + 0.2,\qquad y = 1.5\,\text{inc} + \sin(2\,\text{lat}) + 0.5\,\text{age}^2 + 0.3\,\text{rooms}\cdot\text{inc} + \tau\,d + \varepsilon,$$

    with a heterogeneous effect $\tau = 1 + 0.5\tanh(\text{inc})$ and the sample ATE as the
    target. the estimators: the naive difference, OLS on $(d, x)$, your `dml_irm` with
    linear/logistic nuisances, and your `dml_irm` with gradient boosting. 10 replicates.
    (offline, the loader returns a synthetic stand-in with the same columns; the numbers in
    the lab notes come from the real data.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "rank the four estimators by bias. will DML with linear/logistic nuisances do better than plain OLS? "
        "what coverage will the boosted DML's 95% intervals have?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X0, _, _names = datasets.california_housing()
    _Z = _X0.copy()
    for _j in [3, 4, 5, 6]:  # rooms, bedrooms, population, households are heavy-tailed counts
        _Z[:, _j] = np.sign(_Z[:, _j]) * np.log1p(np.abs(_Z[:, _j]))
    Zc = (_Z - _Z.mean(0)) / _Z.std(0)
    res3 = {}
    for _r in range(10):
        _rng = np.random.default_rng(100 + _r)
        _X = Zc[_rng.choice(len(Zc), min(1500, len(Zc)), replace=False)]
        _inc, _age, _lat, _rooms = _X[:, 7], _X[:, 2], _X[:, 1], _X[:, 3]
        _e = 1 / (1 + np.exp(-(1.2 * _inc - 0.6 * _age**2 + 0.5 * np.tanh(_lat) + 0.2)))
        _d = (_rng.random(len(_X)) < _e).astype(float)
        _tau = 1 + 0.5 * np.tanh(_inc)
        _y = 1.5 * _inc + np.sin(2 * _lat) + 0.5 * _age**2 + 0.3 * _rooms * _inc + _tau * _d + _rng.normal(size=len(_X))
        _ate = _tau.mean()
        _A = np.column_stack([np.ones(len(_X)), _d, _X])
        _o_lin = ex.dml_irm(_X, _y, _d, LinearRegression, LogisticRegression, n_folds=5, rng=_r)
        _o_gb = ex.dml_irm(_X, _y, _d, lambda: GradientBoostingRegressor(60, max_depth=3, rng=0),
                           lambda: GradientBoostingClassifier(60, max_depth=3, rng=0), n_folds=5, rng=_r)
        for _k, (_th, _se) in {"naive difference": (_y[_d == 1].mean() - _y[_d == 0].mean(), np.nan),
                               "OLS on (d, x)": (np.linalg.lstsq(_A, _y, rcond=None)[0][1], np.nan),
                               "DML-IRM, linear + logistic": (_o_lin["theta"], _o_lin["se"]),
                               "DML-IRM, gradient boosting": (_o_gb["theta"], _o_gb["se"])}.items():
            res3.setdefault(_k, []).append((_th - _ate, _se))
    _rows = []
    for _k, _v in res3.items():
        _v = np.array(_v)
        _cov = "—" if np.isnan(_v[0, 1]) else f"{np.mean(np.abs(_v[:, 0]) <= 1.96 * _v[:, 1]):.0%}"
        _se = "—" if np.isnan(_v[0, 1]) else f"{np.mean(_v[:, 1]):.3f}"
        _rows.append(f"| {_k} | {_v[:, 0].mean():+.3f} | {_v[:, 0].std(ddof=1):.3f} | {_se} | {_cov} |")
    mo.md(f"covariates: {len(Zc):,} districts available, 1,500 drawn per replicate.\n\n"
          "| estimator | bias | sd over replicates | mean reported se | 95% CI coverage |\n|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is DML with linear nuisances no better than OLS here (what does
    double robustness need, and does either linear model have it)? what does the boosted
    version buy, and what does it cost in standard error? ten replicates is a small study:
    how precisely do you know the coverage you measured?
    """)
    return


if __name__ == "__main__":
    app.run()
