import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.trees import DecisionTreeRegressor  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bagging & random forests · lab

    every experiment here runs **your** forests from `exercise.py` (built from your topic-01
    trees). for each one: write down a prediction, lock it in, run it, then explain what
    happened. compare your explanations with `solutions/lab_notes.md` afterwards.

    the data is Friedman's benchmark #1: 10 features uniform on $[0, 1]$, target
    $10\sin(\pi x_1x_2) + 20(x_3 - 0.5)^2 + 10x_4 + 5x_5$ plus standard normal noise.
    features 6–10 are pure noise. 300 training rows, 1,000 test rows. the whole lab takes
    about a minute.
    """)
    return


@app.cell
def _():
    def friedman(n, rng, d=10, noise=1.0):
        X = rng.uniform(size=(n, d))
        f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
        return X, f + noise * rng.normal(size=n), f

    X_te, y_te, f_te = friedman(1000, np.random.default_rng(0))
    return X_te, f_te, friedman, y_te


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the variance floor

    we draw **8 independent training sets** and fit a 60-tree forest on each, twice: plain
    bagging (`max_features=1.0`) and a random forest (`max_features=0.33`), both with
    `min_samples_leaf=3`. the average of a forest's first $b$ trees is a $b$-tree forest, so
    for every $b$ we can measure how much the prediction at a test point varies across the 8
    training sets. from the same trees we estimate a single tree's variance $\sigma^2$ and
    the correlation $\rho$ between two trees of the same forest, and draw
    $\rho\sigma^2 + (1 - \rho)\sigma^2/b$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how does the forest's variance fall as trees are added: like 1/b forever, or does it level off? which has "
        "the lower floor, bagging or the random forest, and which has more variable individual trees?"
    )
    p1
    return (p1,)


@app.cell
def _(X_te, f_te, friedman, p1, y_te):
    ui.gate(p1)
    _R, _B = 8, 60
    _bs = np.arange(1, _B + 1)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.3))
    _rows = []
    for _name, _mf, _c in [("bagging", 1.0, "C0"), ("random forest", 0.33, "C3")]:
        _P = np.zeros((_R, _B, len(X_te)))
        _oob = None
        for _r in range(_R):
            _X, _y, _ = friedman(300, np.random.default_rng(10 + _r))
            _rf = ex.RandomForestRegressor(n_estimators=_B, min_samples_leaf=3, max_features=_mf, rng=_r,
                                           oob_score=(_r == 0)).fit(_X, _y)
            if _r == 0:
                _oob = (_rf.oob_score_, _rf.score(X_te, y_te))
            _P[_r] = [_t.predict(X_te) for _t in _rf.estimators_]
        _cum = np.cumsum(_P, axis=1) / _bs[None, :, None]           # b-tree forests
        _var = _cum.var(axis=0, ddof=1).mean(axis=1)                 # across training sets
        _sigma2 = _P.var(axis=0, ddof=1).mean()                      # one tree
        _cov = np.mean([((_P[:, i] - _P[:, i].mean(0)) * (_P[:, i + 1] - _P[:, i + 1].mean(0))).sum(0).mean()
                        / (_R - 1) for i in range(0, _B, 2)])
        _rho = _cov / _sigma2
        _mse = ((_cum - f_te) ** 2).mean(axis=(0, 2)) + 1.0         # + noise variance = expected test MSE
        _axes[0].loglog(_bs, _var, "o", ms=3, color=_c, label=f"{_name} (measured)")
        _axes[0].loglog(_bs, _rho * _sigma2 + (1 - _rho) * _sigma2 / _bs, "-", color=_c,
                        label=f"ρσ² + (1−ρ)σ²/b, ρ = {_rho:.3f}")
        _axes[0].axhline(_rho * _sigma2, color=_c, ls=":", lw=1)
        _axes[1].semilogx(_bs, _mse, "-", color=_c, label=_name)
        _rows.append(f"| {_name} | {_sigma2:.2f} | {_rho:.3f} | {_rho * _sigma2:.2f} | {_var[-1]:.2f} | "
                     f"{_mse[0]:.2f} → {_mse[-1]:.2f} | {_oob[0]:.3f} / {_oob[1]:.3f} |")
    _axes[0].set(xlabel="number of trees b", ylabel="variance across training sets", title="variance of the forest")
    _axes[0].legend(fontsize=7)
    _axes[1].set(xlabel="number of trees b", ylabel="expected test MSE", title="test error (noise variance 1)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| | σ² (one tree) | ρ | floor ρσ² | variance at b = 60 | test MSE, b = 1 → 60 | OOB R² / test R² (one forest) |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does the floor come from, and why can't more trees get under it?
    why does the random forest have *more* variable trees but a *lower* floor? does the test
    error ever go back up as trees are added? how close is the OOB $R^2$ to the test $R^2$,
    and which way is it off?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: weaker trees, better forest?

    one training set, 50-tree forests with `max_features` from 1 to 10. for each we measure
    the average single tree's test MSE, the average correlation between two trees' errors
    $h_b(x) - f(x)$ over the test points, and the forest's test MSE (against the noisy
    targets).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "as max_features drops from 10 to 1, single trees get worse. does the forest get worse too, or better? "
        "where is it best?"
    )
    p2
    return (p2,)


@app.cell
def _(X_te, f_te, friedman, p2, y_te):
    ui.gate(p2)
    _X, _y, _ = friedman(300, np.random.default_rng(99))
    _ms = [1, 2, 3, 4, 6, 8, 10]
    _tree, _corr, _forest = [], [], []
    for _m in _ms:
        _rf = ex.RandomForestRegressor(n_estimators=50, min_samples_leaf=3, max_features=_m, rng=0).fit(_X, _y)
        _P = np.array([_t.predict(X_te) for _t in _rf.estimators_])
        _tree.append(((_P - y_te) ** 2).mean())
        _corr.append(np.corrcoef(_P - f_te)[np.triu_indices(50, 1)].mean())
        _forest.append(((_P.mean(0) - y_te) ** 2).mean())
    _fig, _ax = plt.subplots(figsize=(7.5, 4.2))
    _ax.plot(_ms, _tree, "o-", label="single tree (mean)")
    _ax.plot(_ms, _forest, "s-", label="forest of 50")
    _ax.set(xlabel="max_features", ylabel="test MSE")
    _ax2 = _ax.twinx()
    _ax2.plot(_ms, _corr, "^--", color="C2", label="error correlation ρ")
    _ax2.set_ylabel("correlation between trees' errors", color="C2")
    _ax.legend(loc="upper center", fontsize=8)
    _fig.tight_layout()
    _md = "| max_features | " + " | ".join(map(str, _ms)) + " |\n|---|" + "---|" * len(_ms) + "\n"
    _md += "| single tree MSE | " + " | ".join(f"{v:.1f}" for v in _tree) + " |\n"
    _md += "| error correlation | " + " | ".join(f"{v:.2f}" for v in _corr) + " |\n"
    _md += "| forest MSE | " + " | ".join(f"{v:.2f}" for v in _forest) + " |\n"
    mo.vstack([_fig, mo.md(_md)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the forest's error is roughly (average tree error) × (correlation +
    a $1/B$ term). use the table to explain the shape of the forest's curve. why is
    `max_features=1` so bad *on this data* (how many of the 10 features carry signal)?
    on a dataset where every feature is informative, where would you expect the optimum?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what is worth bagging?

    one training set; for `max_depth` = 1, 2, 4, 8 and unlimited we compare one tree with a
    bag of 50 trees of the same depth (bootstrap only, `max_features=1.0`).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "bagging 50 stumps (depth 1): by what percentage does the test MSE fall compared with one stump? and for "
        "fully grown trees?"
    )
    p3
    return (p3,)


@app.cell
def _():
    n_bag = mo.ui.slider(5, 100, value=50, step=5, label="trees per bag")
    n_bag
    return (n_bag,)


@app.cell
def _(X_te, friedman, n_bag, p3, y_te):
    ui.gate(p3)
    _X, _y, _ = friedman(300, np.random.default_rng(99))
    _depths = [1, 2, 4, 8, None]
    _single, _bagged = [], []
    for _d in _depths:
        _single.append(((DecisionTreeRegressor(max_depth=_d).fit(_X, _y).predict(X_te) - y_te) ** 2).mean())
        _bag = ex.RandomForestRegressor(n_estimators=n_bag.value, max_depth=_d, max_features=1.0, rng=0).fit(_X, _y)
        _bagged.append(((_bag.predict(X_te) - y_te) ** 2).mean())
    _fig, _ax = plt.subplots(figsize=(7.5, 4))
    _x = np.arange(len(_depths))
    _ax.bar(_x - 0.2, _single, 0.4, label="one tree")
    _ax.bar(_x + 0.2, _bagged, 0.4, label=f"bag of {n_bag.value}")
    _ax.set(xticks=_x, xticklabels=[str(d) if d else "full" for d in _depths], xlabel="max_depth",
            ylabel="test MSE")
    _ax.legend()
    _fig.tight_layout()
    _md = "| max_depth | one tree | bagged | reduction |\n|---|---|---|---|\n" + "\n".join(
        f"| {d if d else 'full'} | {s:.2f} | {b:.2f} | {100 * (1 - b / s):.0f}% |"
        for d, s, b in zip(_depths, _single, _bagged))
    mo.vstack([_fig, mo.md(_md)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does bagging barely help stumps? which single tree is best, and
    which bag is best: why aren't they the same depth? what does this tell you about the
    kind of base learner bagging wants, and about the opposite strategy (topics 03–04),
    which combines many *shallow* trees?
    """)
    return


if __name__ == "__main__":
    app.run()
