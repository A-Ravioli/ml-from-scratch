import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.linear import LinearRegression  # yours, from topic 01

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def rentals(n, rng):
    """daily bike rentals: temperature (standardized), rain (0/1) and weekend (0/1) act
    multiplicatively on the expected count: mu = exp(2 + 0.6 temp - 0.8 rain + 0.3 weekend)."""
    temp = rng.normal(size=n)
    rain = (rng.random(n) < 0.3).astype(float)
    weekend = (rng.random(n) < 2 / 7).astype(float)
    X = np.column_stack([temp, rain, weekend])
    mu = np.exp(2.0 + 0.6 * temp - 0.8 * rain + 0.3 * weekend)
    return X, rng.poisson(mu).astype(float), mu


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # generalized linear models · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: modelling count data

    daily bike rentals at a small station: on average about 7 a day, more when it's warm, fewer
    when it rains, a few more on weekends, and the effects multiply (a rainy day loses the
    same *fraction* of riders whatever the temperature). you fit topic 01's
    `LinearRegression` and your `GLM("poisson")` on 300 days and evaluate on 2,000 new ones.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what fraction of the linear model's test predictions will be negative (impossible "
        "counts)? which model will have the lower test RMSE, and by how much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X, _y, _ = rentals(300, _rng)
    _Xt, _yt, _mut = rentals(2000, _rng)
    _ls = LinearRegression().fit(_X, _y)
    _glm = ex.GLM("poisson").fit(_X, _y)
    _p_ls, _p_glm = _ls.predict(_Xt), _glm.predict(_Xt)
    _fam = ex.Poisson()
    _rmse = lambda p: float(np.sqrt(np.mean((_yt - p) ** 2)))
    _dev = lambda p: float(np.mean(_fam.deviance(_yt, np.maximum(p, 0.1))))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _temps = np.linspace(-2.5, 2.5, 100)
    for _rain, _ls_style in ((0.0, "-"), (1.0, "--")):
        _grid = np.column_stack([_temps, np.full(100, _rain), np.zeros(100)])
        _a1.plot(_temps, _ls.predict(_grid), "C0" + _ls_style, label=f"linear, rain={int(_rain)}")
        _a1.plot(_temps, _glm.predict(_grid), "C1" + _ls_style, label=f"poisson GLM, rain={int(_rain)}")
        _a1.plot(_temps, np.exp(2.0 + 0.6 * _temps - 0.8 * _rain), "k" + _ls_style, lw=0.8, alpha=0.6)
    _a1.axhline(0, color="gray", lw=0.5)
    _a1.set(xlabel="temperature (standardized)", ylabel="expected rentals (weekday)",
            title="fits (thin black: the truth)")
    _a1.legend(fontsize=7)
    _a2.scatter(_p_ls, _yt - _p_ls, s=4, alpha=0.4, label="linear")
    _a2.scatter(_p_glm, _yt - _p_glm, s=4, alpha=0.4, label="poisson GLM")
    _a2.axhline(0, color="k", lw=0.5)
    _a2.set(xlabel="predicted count", ylabel="residual (test)", title="residuals")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| model | negative predictions | test RMSE | mean poisson deviance | $D^2$ |\n|---|---|---|---|---|\n"
        f"| linear regression | {np.mean(_p_ls < 0):.1%} | {_rmse(_p_ls):.2f} | {_dev(_p_ls):.3f} | "
        f"{ex.d2_score(_yt, np.maximum(_p_ls, 0.1), _fam):.3f} |\n"
        f"| poisson GLM | {np.mean(_p_glm < 0):.1%} | {_rmse(_p_glm):.2f} | {_dev(_p_glm):.3f} | "
        f"{ex.d2_score(_yt, _p_glm, _fam):.3f} |\n"
        f"| the true means | 0.0% | {_rmse(_mut):.2f} | {_dev(_mut):.3f} | {ex.d2_score(_yt, _mut, _fam):.3f} |\n\n"
        f"(negative linear predictions are clipped at 0.1 for the deviance.) GLM coefficients: "
        f"{np.round(_glm.coef_, 3).tolist()} (truth [0.6, -0.8, 0.3]); rain multiplies rentals by "
        f"{np.exp(_glm.coef_[1]):.2f} (truth {np.exp(-0.8):.2f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the linear model says rain costs the same *number* of rentals on a cold day
    as on a hot one. what does the log link say instead, and which one matches how the data
    were made? why does the linear model predict negative counts, and where? in the residual
    plot, why do the residuals fan out as the prediction grows, and which model's variance
    assumption agrees with that? what is the diagonal edge of blue points at the lower left?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when the variance is wrong

    the slope is $0.5$ in $\log\mu = 1 + 0.5x$, and you fit your Poisson `GLM` to 300
    simulated datasets of 200 points each. from each fit you build the textbook 95% confidence
    interval for the slope, $\hat w \pm 1.96\,\text{se}$, with the standard error from the
    inverse Fisher information $(A^\top \operatorname{diag}(\hat\mu) A)^{-1}$, and count how
    often it covers 0.5. in the second row, the counts are *overdispersed*: each one's mean is
    multiplied by an independent gamma factor with mean 1, so their variance is
    $\mu + \mu^2$ instead of $\mu$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for the overdispersed counts, will the slope estimates be biased? what fraction of the "
        "'95%' intervals will contain the true slope?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _rows = []
    for _label, _shape in (("poisson counts", None), ("overdispersed counts", 1.0)):
        _est, _cov_naive, _cov_fixed, _phis, _ses = [], 0, 0, [], []
        for _r in range(300):
            _x = _rng.normal(size=200)
            _A = np.column_stack([np.ones(200), _x])
            _mu = np.exp(1.0 + 0.5 * _x)
            _lam = _mu if _shape is None else _mu * _rng.gamma(_shape, 1 / _shape, 200)
            _y = _rng.poisson(_lam).astype(float)
            _m = ex.GLM("poisson").fit(_x[:, None], _y)
            _muh = _m.predict(_x[:, None])
            _se = float(np.sqrt(np.linalg.inv((_A.T * _muh) @ _A)[1, 1]))
            _phi = ex.pearson_dispersion(_y, _muh, ex.Poisson(), 2)
            _w = float(_m.coef_[0])
            _est.append(_w)
            _ses.append(_se)
            _phis.append(_phi)
            _cov_naive += abs(_w - 0.5) < 1.96 * _se
            _cov_fixed += abs(_w - 0.5) < 1.96 * _se * np.sqrt(_phi)
        _rows.append(f"| {_label} | {np.mean(_est):.3f} | {np.std(_est):.3f} | {np.mean(_ses):.3f} | "
                     f"**{_cov_naive / 300:.0%}** | {np.median(_phis):.2f} | {_cov_fixed / 300:.0%} |")
    mo.md("| data | mean slope | actual spread of the slope | average model-based se | coverage of the 95% CI "
          "| median Pearson dispersion | coverage with se × √dispersion |\n|---|---|---|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are the slope estimates still centered on 0.5 when the variance
    assumption is wrong (what does the estimating equation $A^\top(y - \mu) = 0$ require)?
    the standard error comes from the Poisson assumption $\operatorname{Var}(y) = \mu$: by what
    factor is it off, and how does the Pearson dispersion measure that factor? what other
    fixes would you consider (a negative binomial model, the bootstrap, "sandwich" standard
    errors)?
    """)
    return


if __name__ == "__main__":
    app.run()
