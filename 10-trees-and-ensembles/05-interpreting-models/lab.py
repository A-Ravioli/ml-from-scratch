import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.trees import GradientBoostingRegressor, RandomForestClassifier  # yours, from topics 02 and 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # interpreting models · lab

    every experiment here runs **your** inspection tools from `exercise.py` on **your**
    forests and boosters from topics 02 and 04. for each one: write down a prediction, lock
    it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the lab takes about half a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: which feature matters?

    six features, 500 training rows: $x_1$ is a coin flip that matters ($P(y = 1)$ is 0.7
    when $x_1 = 1$ and 0.3 otherwise). the other five are **pure noise**, independent of
    $y$: a coin flip, a 4-level and a 10-level and a 20-level integer, and a continuous
    standard normal. a random forest (100 trees, 3 features per split) is fitted, and we
    compare its impurity importances (`feature_importances_`) with permutation importance on
    2,000 fresh test rows, and on the training rows.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the forest's impurity importances: which feature comes out on top? where does x1, the only one that "
        "matters, rank?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def _data(n, rng):
        X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n), rng.integers(0, 4, n),
                             rng.integers(0, 10, n), rng.integers(0, 20, n), rng.normal(size=n)]).astype(float)
        y = (rng.random(n) < np.where(X[:, 0] == 1, 0.7, 0.3)).astype(int)
        return X, y

    _rng = np.random.default_rng(0)
    _X, _y = _data(500, _rng)
    _Xt, _yt = _data(2000, _rng)
    _rf = RandomForestClassifier(n_estimators=100, max_features=3, rng=0).fit(_X, _y)
    _imp = _rf.feature_importances_
    _pt = ex.permutation_importance(_rf, _Xt, _yt, n_repeats=5, rng=0)
    _ptr = ex.permutation_importance(_rf, _X, _y, n_repeats=5, rng=0)
    _names = ["x1 (signal)", "coin", "4 levels", "10 levels", "20 levels", "continuous"]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for _ax, _vals, _err, _title in [
        (_axes[0], _imp, None, "impurity importance (training)"),
        (_axes[1], _ptr["importances_mean"], _ptr["importances_std"], "permutation importance, training rows"),
        (_axes[2], _pt["importances_mean"], _pt["importances_std"], "permutation importance, test rows"),
    ]:
        _ax.barh(_names, _vals, xerr=_err, color=["C3"] + ["C0"] * 5)
        _ax.axvline(0, color="k", lw=0.5)
        _ax.set_title(_title, fontsize=10)
    _axes[0].invert_yaxis()
    _fig.tight_layout()
    _md = "| feature | impurity | permutation (train) | permutation (test) |\n|---|---|---|---|\n" + "\n".join(
        f"| {n} | {a:.3f} | {b:.3f} | {c:.3f} ± {s:.3f} |" for n, a, b, c, s in
        zip(_names, _imp, _ptr["importances_mean"], _pt["importances_mean"], _pt["importances_std"]))
    mo.vstack([_fig, mo.md(_md + f"\n\nforest test accuracy {_rf.score(_Xt, _yt):.3f} (the best possible is 0.70).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does impurity importance grow with the number of distinct values a
    noise feature has? (count the thresholds each feature offers a fully grown tree, and
    think about what a split on noise does to *training* impurity.) why does permutation
    importance on the training rows show the same bias, and why does it vanish on test rows?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: twins

    $y = x_1 + x_2 + 0.3\varepsilon$ with independent standard normal $x_1, x_2$, plus
    $x_3 = x_1 + 0.05\cdot$noise (a near copy of $x_1$) and a noise feature $x_4$. we boost
    (200 trees, depth 3) on 1,000 rows and compute permutation importance on 2,000 test rows.
    then we compare with **drop-column importance**: refit without a feature (or two) and
    measure the loss in test $R^2$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "x1 and its near copy x3 carry the same information as x2. how does the permutation importance of x1 "
        "compare with x2's? what happens to test R^2 if you remove x1 and x3 together?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _data(n, rng):
        x1, x2, x4 = rng.normal(size=(3, n))
        X = np.column_stack([x1, x2, x1 + 0.05 * rng.normal(size=n), x4])
        return X, x1 + x2 + 0.3 * rng.normal(size=n)

    _rng = np.random.default_rng(1)
    _X, _y = _data(1000, _rng)
    _Xt, _yt = _data(2000, _rng)

    def _fit(cols):
        return GradientBoostingRegressor(n_estimators=200, max_depth=3).fit(_X[:, cols], _y).score(_Xt[:, cols], _yt)

    _gb = GradientBoostingRegressor(n_estimators=200, max_depth=3).fit(_X, _y)
    _base = _gb.score(_Xt, _yt)
    _pi = ex.permutation_importance(_gb, _Xt, _yt, n_repeats=5, rng=0)
    _drops = {"x1": [1, 2, 3], "x2": [0, 2, 3], "x3": [0, 1, 3], "x4": [0, 1, 2], "x1 and x3": [1, 3]}
    _dc = {k: _base - _fit(v) for k, v in _drops.items()}
    _names = ["x1", "x2", "x3 (≈ x1)", "x4 (noise)"]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _axes[0].barh(_names, _pi["importances_mean"], xerr=_pi["importances_std"])
    _axes[0].invert_yaxis()
    _axes[0].set_title("permutation importance (test R² drop)", fontsize=10)
    _axes[1].barh(list(_dc), list(_dc.values()), color="C1")
    _axes[1].invert_yaxis()
    _axes[1].set_title("drop-column importance (refit, test R² drop)", fontsize=10)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"full model test R² **{_base:.3f}**. permutation: x1 **{_pi['importances_mean'][0]:.3f}**, x2 "
        f"**{_pi['importances_mean'][1]:.3f}**, x3 {_pi['importances_mean'][2]:.3f}, x4 {_pi['importances_mean'][3]:.3f}. "
        f"drop-column: " + ", ".join(f"{k} {v:.3f}" for k, v in _dc.items()) + ".")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is x1's permutation importance so much smaller than x2's, although they
    matter equally for $y$? where did the rest of x1's importance go? what do the refits say
    that permutation can't, and what question does each method answer? (also: when you
    shuffle x1 but not x3, what kind of rows does the model get to see?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: an effect that averages away

    $y = x_1x_2 + x_3$ with $x_1, x_2, x_3$ uniform on $[-1, 1]$ (2,000 rows), boosted with
    300 trees of depth 3. we draw the partial dependence of a feature, its ICE curves (200
    rows), and its KernelSHAP values for 200 rows, with 100 background rows.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what does the partial dependence plot of x1 look like? and what do x1's SHAP values look like for "
        "individual rows?"
    )
    p3
    return (p3,)


@app.cell
def _():
    which = mo.ui.dropdown({"x1": 0, "x2": 1, "x3": 2}, value="x1", label="feature")
    which
    return (which,)


@app.cell
def _():
    _rng = np.random.default_rng(2)
    X3 = _rng.uniform(-1, 1, size=(2000, 3))
    y3 = X3[:, 0] * X3[:, 1] + X3[:, 2] + 0.05 * _rng.normal(size=2000)
    gb3 = GradientBoostingRegressor(n_estimators=300, max_depth=3).fit(X3, y3)
    return X3, gb3


@app.cell
def _(X3, gb3, p3, which):
    ui.gate(p3)
    _j = which.value
    _rows = X3[:200]
    _grid = np.linspace(-1, 1, 21)
    _ice = ex.ice_curves(gb3, _rows, _j, _grid)
    _, _pd = ex.partial_dependence(gb3, X3, _j, grid=_grid)
    _bg = X3[1000:1100]
    _phi = np.array([ex.kernel_shap(gb3.predict, _bg, _x, rng=0) for _x in _rows])
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _other = 1 if _j == 0 else 0
    _c = np.where(_rows[:, _other] > 0, "C3", "C0")
    for _i in range(len(_rows)):
        _axes[0].plot(_grid, _ice[_i] - _ice[_i].mean(), color=_c[_i], alpha=0.15, lw=0.8)
    _axes[0].plot(_grid, _pd - _pd.mean(), "k", lw=3, label="partial dependence")
    _axes[0].set(xlabel=f"x{_j + 1}", ylabel="centred prediction",
                 title=f"ICE curves (red: x{_other + 1} > 0, blue: x{_other + 1} < 0)")
    _axes[0].legend()
    _axes[1].scatter(_rows[:, _j], _phi[:, _j], c=_c, s=12)
    _axes[1].set(xlabel=f"x{_j + 1}", ylabel=f"SHAP value of x{_j + 1}", title="KernelSHAP, one dot per row")
    _fig.tight_layout()
    _pd_range = _pd.max() - _pd.min()
    mo.vstack([_fig, mo.md(
        f"x{_j + 1}: partial dependence range **{_pd_range:.3f}**; mean |SHAP| **{np.abs(_phi[:, _j]).mean():.3f}**; "
        f"mean |SHAP| of every feature: {np.round(np.abs(_phi).mean(0), 3).tolist()}. efficiency check: largest "
        f"|Σφ + E f − f(x)| = {np.abs(_phi.sum(1) + gb3.predict(_bg).mean() - gb3.predict(_rows)).max():.1e}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is x1's partial dependence nearly flat when x1 is half of the model?
    what do the two families of ICE curves show? how do the SHAP values split the interaction
    $x_1x_2$ between the two features (compare the pictures for x1 and x2)? switch to x3:
    why do all three views agree there?
    """)
    return


if __name__ == "__main__":
    app.run()
