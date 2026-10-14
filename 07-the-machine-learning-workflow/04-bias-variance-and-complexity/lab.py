import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.metrics import get_scorer  # yours, from topic 03
    from mlfs.model_selection import KFold  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    SIGMA = 0.3  # noise standard deviation in every experiment
    X_GRID = np.linspace(-1, 1, 101).reshape(-1, 1)


@app.function
def f_true(x):
    """the function behind the data."""
    return np.sin(np.pi * np.asarray(x, dtype=float))


@app.function
def make_sampler(n):
    """sample_data(gen) for bias_variance_decomposition: n points, x uniform on [-1, 1]."""
    def sample(gen):
        x = gen.uniform(-1, 1, size=n)
        return x.reshape(-1, 1), f_true(x) + SIGMA * gen.normal(size=n)
    return sample


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bias–variance & complexity · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.

    the data throughout: $y = \sin(\pi x) + \varepsilon$ with $x$ uniform on $[-1, 1]$ and
    noise $\varepsilon \sim \mathcal N(0, 0.3^2)$, so the irreducible error is $0.09$.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: sweeping the degree

    for each polynomial degree from 0 to 12, your `bias_variance_decomposition` refits
    `PolynomialRegressor` on 200 fresh training sets of $n$ points and measures bias²,
    variance and their sum with the noise, averaged over $[-1, 1]$. the dashed line is what
    you'd see in practice from a *single* dataset: your `validation_curve` (5-fold CV,
    MSE).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with n = 30 points, which degree has the lowest expected error? how large is the "
        "variance at degree 12 compared with the noise (0.09)?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n1 = mo.ui.slider(15, 200, value=30, step=5, label="training points n")
    n1
    return (n1,)


@app.cell
def _(n1, p1):
    ui.gate(p1)
    _degrees = np.arange(13)
    _res = [ex.bias_variance_decomposition(ex.PolynomialRegressor(int(d)), make_sampler(n1.value), X_GRID,
                                           f_true(X_GRID[:, 0]), n_rounds=200, noise_var=SIGMA**2, rng=0)
            for d in _degrees]
    _b = np.array([r["bias2"] for r in _res])
    _v = np.array([r["variance"] for r in _res])
    _e = np.array([r["expected_error"] for r in _res])
    _X1, _y1 = make_sampler(n1.value)(np.random.default_rng(1))
    _, _val = ex.validation_curve(ex.PolynomialRegressor(), _X1, _y1, "degree", list(_degrees),
                                  cv=KFold(5, shuffle=True, rng=0), scoring=get_scorer("neg_mean_squared_error"))
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.semilogy(_degrees, _b, "o-", label="bias²")
    _ax.semilogy(_degrees, _v, "s-", label="variance")
    _ax.semilogy(_degrees, _e, "k^-", label="expected error = bias² + variance + noise")
    _ax.semilogy(_degrees, -_val.mean(axis=1), "k--", alpha=0.6, label="5-fold CV MSE on ONE dataset")
    _ax.axhline(SIGMA**2, color="gray", ls=":", label="noise σ² = 0.09")
    _ax.set(xlabel="polynomial degree", ylabel="squared error (log scale)", title=f"n = {n1.value}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _best = int(np.argmin(_e))
    mo.vstack([_fig, mo.md(
        f"lowest expected error at degree **{_best}** ({_e[_best]:.3f}); CV picks degree "
        f"**{int(np.argmax(_val.mean(axis=1)))}**. at degree 12: bias² {_b[12]:.3g}, variance **{_v[12]:.3g}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does bias² drop only when the degree becomes odd (0 → 1, 2 → 3,
    4 → 5)? why does the variance explode at high degree even though the noise is small? (and
    why does the *measured* bias² climb again at degree 10+, when a richer model should only
    lower it? compare it with variance / 200, and reread the end of idea 2.) does the single-dataset CV curve find the same best degree? move $n$ to 100:
    what changes, and why?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: where does the variance live?

    the same setup, but now we look at the individual fits. each thin line is your
    `PolynomialRegressor` fitted to a different training set of 30 points; the bottom panel
    shows the variance of the predictions at each $x$ (over 200 training sets).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for degree 10 with 30 points, is the variance of the fitted curve spread evenly over "
        "[-1, 1], or concentrated somewhere? where?"
    )
    p2
    return (p2,)


@app.cell
def _():
    deg2 = mo.ui.slider(0, 14, value=10, label="degree")
    deg2
    return (deg2,)


@app.cell
def _(deg2, p2):
    ui.gate(p2)
    _gen = np.random.default_rng(3)
    _sample = make_sampler(30)
    _preds = np.array([ex.PolynomialRegressor(deg2.value).fit(*_sample(_gen)).predict(X_GRID) for _ in range(200)])
    _fig, (_a1, _a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for _p in _preds[:25]:
        _a1.plot(X_GRID[:, 0], _p, color="C0", alpha=0.3, lw=1)
    _a1.plot(X_GRID[:, 0], _preds.mean(axis=0), "C3", lw=2, label="mean of 200 fits")
    _a1.plot(X_GRID[:, 0], f_true(X_GRID[:, 0]), "k--", lw=1.5, label="true f")
    _a1.set(ylim=(-2.5, 2.5), ylabel="y", title=f"degree {deg2.value}: 25 of 200 fits")
    _a1.legend(fontsize=8, loc="upper left")
    _var = _preds.var(axis=0)
    _a2.semilogy(X_GRID[:, 0], _var, "C1")
    _a2.axhline(SIGMA**2, color="gray", ls=":")
    _a2.set(xlabel="x", ylabel="variance of the fit")
    _fig.tight_layout()
    _mid = _var[np.abs(X_GRID[:, 0]) <= 0.5].mean()
    _edge = _var[np.abs(X_GRID[:, 0]) >= 0.9].mean()
    mo.vstack([_fig, mo.md(f"mean variance for |x| ≤ 0.5: **{_mid:.3g}**; for |x| ≥ 0.9: **{_edge:.3g}** "
                            f"({_edge / _mid:.0f}× larger)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are the edges so much worse than the middle? (think about how many
    data points constrain the polynomial near $x = 1$, and what a high power like $x^{10}$
    does there.) what does this say about trusting a flexible model where the data is thin,
    for example when *extrapolating*?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: learning curves

    400 points, and your `learning_curve` with 5 shuffled folds: training and validation MSE
    as the training set grows from 10% to 100% of each fold (32 to 320 points), for degrees
    1, 3 and 9.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for degree 1 (a straight line), what happens to the validation error as you go from "
        "32 to 320 training points? and for degree 9?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _X, _y = make_sampler(400)(np.random.default_rng(4))
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    _rows = []
    for _ax, _d in zip(_axes, (1, 3, 9)):
        _sizes, _tr, _va = ex.learning_curve(ex.PolynomialRegressor(_d), _X, _y,
                                             train_sizes=np.linspace(0.1, 1.0, 10), cv=KFold(5, shuffle=True, rng=0),
                                             scoring=get_scorer("neg_mean_squared_error"))
        _ax.plot(_sizes, -_tr.mean(axis=1), "o-", label="training MSE")
        _ax.plot(_sizes, -_va.mean(axis=1), "s-", label="validation MSE")
        _ax.axhline(SIGMA**2, color="gray", ls=":", label="noise σ²")
        _ax.set(xlabel="training points", title=f"degree {_d}", yscale="log")
        _rows.append(f"| {_d} | {-_va.mean(axis=1)[0]:.3f} | {-_va.mean(axis=1)[-1]:.3f} | {-_tr.mean(axis=1)[-1]:.3f} |")
    _axes[0].set_ylabel("MSE (log scale)")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| degree | validation MSE at 32 points | at 320 points | training MSE at 320 |\n"
                            "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the line's two curves meet at a plateau far above the noise level. what is
    that plateau, in bias–variance terms, and why can't more data move it? for degree 9, what
    closes the gap between training and validation error? if your model's learning curves
    look like degree 1's, should you collect more data or change the model?
    """)
    return


if __name__ == "__main__":
    app.run()
