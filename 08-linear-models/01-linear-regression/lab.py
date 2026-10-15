import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def sine_data(n, rng, noise=0.25):
    """x uniform on [0, 1], y = sin(2 pi x) + Gaussian noise."""
    x = np.sort(rng.uniform(0, 1, n))
    return x, np.sin(2 * np.pi * x) + noise * rng.normal(size=n)


@app.function
def rmse(y, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(y_pred)) ** 2)))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # linear regression · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: polynomial bases overfit

    15 noisy points from $y = \sin(2\pi x) + \varepsilon$, $\sigma = 0.25$, on $[0, 1]$. you fit
    `LinearRegression(solver="qr")` on `polynomial_basis(x, degree)` for every degree from 1 to
    14, and measure the RMSE on the 15 training points and on 2,000 fresh test points from the
    same distribution. the noise alone gives an RMSE of 0.25: no model can do better on average.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which degree gives the lowest test RMSE? what are the training RMSE and the largest "
        "coefficient at degree 14 (15 parameters for 15 points)?"
    )
    p1
    return (p1,)


@app.cell
def _():
    degree = mo.ui.slider(1, 14, value=3, label="degree to plot", show_value=True)
    degree
    return (degree,)


@app.cell
def _(degree, p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _x, _y = sine_data(15, _rng)
    _xt, _yt = sine_data(2000, _rng)
    _degrees = list(range(1, 15))
    _train, _test, _big = [], [], []
    for _d in _degrees:
        _m = ex.LinearRegression(solver="qr").fit(ex.polynomial_basis(_x, _d), _y)
        _train.append(rmse(_y, _m.predict(ex.polynomial_basis(_x, _d))))
        _test.append(rmse(_yt, _m.predict(ex.polynomial_basis(_xt, _d))))
        _big.append(float(np.max(np.abs(_m.coef_))))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    _a1.semilogy(_degrees, np.maximum(_train, 1e-16), "o-", label="train")
    _a1.semilogy(_degrees, _test, "s-", label="test")
    _a1.axhline(0.25, color="gray", ls=":", label="noise level")
    _a1.set(xlabel="degree", ylabel="RMSE (log scale)", title="error versus degree")
    _a1.legend()
    _grid = np.linspace(0, 1, 400)
    _m = ex.LinearRegression(solver="qr").fit(ex.polynomial_basis(_x, degree.value), _y)
    _a2.plot(_grid, np.sin(2 * np.pi * _grid), color="gray", ls=":", label="truth")
    _a2.plot(_grid, _m.predict(ex.polynomial_basis(_grid, degree.value)),
             label=f"degree {degree.value}")
    _a2.plot(_x, _y, "ko", ms=4)
    _a2.set(ylim=(-2, 2), xlabel="x", ylabel="y", title="the fit")
    _a2.legend(loc="lower left")
    _fig.tight_layout()
    _best = int(np.argmin(_test))
    mo.vstack([_fig, mo.md(
        f"predicting the training mean everywhere: test RMSE {rmse(_yt, np.full(2000, _y.mean())):.3f}. "
        f"lowest test RMSE: degree **{_degrees[_best]}** ({_test[_best]:.3f}). degree 14: train RMSE "
        f"{_train[-1]:.1e}, test RMSE {_test[-1]:.1f}, largest |coefficient| **{_big[-1]:.1e}** "
        f"(degree 3: {_big[2]:.1f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the training error keep falling all the way to (numerically) zero,
    while the test error first falls and then explodes? where along $x$ does the degree-14
    curve go wrong, and why there? what do coefficients in the trillions tell you about how
    the fit uses its basis functions, and why isn't the degree-14 training error exactly 0? (chapter 07, topic 04 measured the variance
    part of this.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: normal equations versus QR

    the same monomial basis, now on 100 points in $[0, 1]$, with a *noiseless* target
    $y = 1 + x + x^2 + \dots + x^d$, so the exact answer is all ones and every bit of error is
    rounding. for each degree you solve the least-squares problem with your
    `solve_normal_equations` and your `solve_qr` (on the design matrix with a ones column)
    and measure the relative error of the coefficients, next to the condition number of $X$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the condition number of X grows about 5x per degree. at what degree does each solver "
        "lose all its digits (relative error near 1)?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _x = np.linspace(0, 1, 100)
    _rows = []
    _deg = list(range(1, 15))
    _err = {"normal": [], "qr": []}
    _conds = []
    for _d in _deg:
        _X = np.column_stack([np.ones(100), ex.polynomial_basis(_x, _d)])
        _w = np.ones(_d + 1)
        _y = _X @ _w
        _conds.append(float(np.linalg.cond(_X)))
        for _name, _solve in (("normal", ex.solve_normal_equations), ("qr", ex.solve_qr)):
            try:
                _got = np.asarray(_solve(_X, _y))
                _e = float(np.linalg.norm(_got - _w) / np.linalg.norm(_w))
            except np.linalg.LinAlgError:
                _e = np.nan
            _err[_name].append(_e)
        _n = _err["normal"][-1]
        _rows.append(f"| {_d} | {_conds[-1]:.1e} | {'cholesky failed' if np.isnan(_n) else f'{_n:.1e}'} | "
                     f"{_err['qr'][-1]:.1e} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogy(_deg, _err["normal"], "o-", label="normal equations (Cholesky)")
    _ax.semilogy(_deg, _err["qr"], "s-", label="QR")
    _ax.semilogy(_deg, np.array(_conds) * 2.2e-16, "k:", label=r"cond(X) $\cdot\ \epsilon$")
    _ax.semilogy(_deg, np.array(_conds) ** 2 * 2.2e-16, "k--", label=r"cond(X)$^2 \cdot\ \epsilon$")
    _ax.set(xlabel="degree", ylabel="relative coefficient error", ylim=(1e-17, 1e3))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| degree | cond(X) | normal equations | QR |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which dashed line does each solver follow, and why does forming $X^\top X$
    square the condition number? the normal equations' error stops growing once it reaches
    about 1: what does an error of 1 mean for the coefficients you get back? (if cholesky had
    raised `LinAlgError`, what would that say about $X^\top X$ in floating point, given that
    the exact $X^\top X$ is positive definite?) would scaling the columns help here?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: gradient descent and the shape of the bowl

    two independent standard-normal features, the second multiplied by a scale factor
    $s$. you run your `solve_gd` on the centered data with the largest stable-ish step,
    $\eta = 1/L$, where $L$ is the largest eigenvalue of $X^\top X / n$, until the gradient
    norm drops below $10^{-8}$ (capped at 100,000 steps). then you standardize the features
    and run it again.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with the second feature multiplied by 10, how many more gradient steps than with s = 1 "
        "will gradient descent need? 10x? 100x?"
    )
    p3
    return (p3,)


@app.cell
def _():
    scale = mo.ui.dropdown({"1": 1.0, "3": 3.0, "10": 10.0, "30": 30.0}, value="10", label="scale factor s")
    scale
    return (scale,)


@app.function
def gd_steps(X, y, max_iters=100_000):
    """centre, take lr = 1 / L, run solve_gd; returns (steps, condition number of X^T X / n)."""
    Xc, yc = X - X.mean(0), y - y.mean()
    eig = np.linalg.eigvalsh(Xc.T @ Xc / len(y))
    res = ex.solve_gd(Xc, yc, lr=1.0 / eig[-1], max_iters=max_iters, tol=1e-8)
    return res.n_iters, float(eig[-1] / eig[0]), res.converged


@app.cell
def _(p3, scale):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _Z = _rng.normal(size=(200, 2))
    _y = _Z @ np.array([1.0, -2.0]) + 0.1 * _rng.normal(size=200)
    _rows = []
    for _s in (1.0, 3.0, 10.0, 30.0):
        _X = _Z * np.array([1.0, _s])
        _k, _kappa, _ok = gd_steps(_X, _y)
        _ks, _kappas, _ = gd_steps((_X - _X.mean(0)) / _X.std(0), _y)
        _mark = " ◀" if _s == scale.value else ""
        _rows.append(f"| {_s:g}{_mark} | {_kappa:,.0f} | {_k:,}{'' if _ok else ' (cap)'} | {_kappas:.2f} | {_ks} |")
    mo.md("| scale s | condition number | GD steps (raw) | condition number (standardized) | GD steps (standardized) |\n"
          "|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does the number of steps grow with $s$, and with the condition number
    $\kappa$? use the convergence rate from chapter 06 (error shrinks like $(1 - 1/\kappa)^k$) to
    predict the ratio between two rows, and compare. why does standardizing make $s$
    irrelevant? the exact solvers don't care about $s$ at all here: what is it that gradient
    descent "feels" that they don't?
    """)
    return


if __name__ == "__main__":
    app.run()
