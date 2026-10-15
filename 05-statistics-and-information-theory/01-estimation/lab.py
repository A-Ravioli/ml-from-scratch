import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    # unit-variance samplers with different tails. kurtosis = E[(X - mu)^4] / sigma^4.
    SAMPLERS = {
        "uniform (kurtosis 1.8)": lambda s, r: math.sqrt(12.0) * (r.random(s) - 0.5),
        "normal (kurtosis 3)": lambda s, r: r.normal(size=s),
        "laplace (kurtosis 6)": lambda s, r: r.laplace(0.0, 1 / math.sqrt(2.0), s),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # estimation · lab

    every experiment here runs **your** harness from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: divide by $n - 1$, $n$, or $n + 1$?

    every statistics course tells you to estimate a variance by dividing the sum of squared
    deviations by $n - 1$, because that makes it unbiased. here your `evaluate` measures the
    MSE of `scaled_variance(x, c)` for a range of divisors $c$, with $n = 10$ draws from three
    distributions that all have variance 1 but different tails. the dashed curve is your exact
    Gaussian formula, `scaled_variance_mse`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for normal data with n = 10, which divisor has the smallest MSE: 9, 10 or 11? "
        "does the answer change for light tails (uniform) or heavy tails (laplace)?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _n = 10
    _cs = np.arange(6, 25)
    _fig, _ax = plt.subplots(figsize=(7.5, 3.8))
    _rows = []
    for _k, (_label, _s) in enumerate(SAMPLERS.items()):
        _rng = np.random.default_rng(_k)
        _mse = np.array([ex.evaluate(lambda x, c=_c: ex.scaled_variance(x, c), _s, 1.0, _n, 40_000, rng=_rng)["mse"]
                         for _c in _cs])
        _ax.plot(_cs, _mse, "o-", ms=3, label=_label)
        _best = int(_cs[np.argmin(_mse)])
        _at = {c: _mse[list(_cs).index(c)] for c in (9, 10, 11)}
        _rows.append(f"| {_label} | {_at[9]:.3f} | {_at[10]:.3f} | {_at[11]:.3f} | {_best} |")
    _ax.plot(_cs, ex.scaled_variance_mse(_n, _cs), "k--", lw=1, label="exact, normal")
    for _c in (9, 10, 11):
        _ax.axvline(_c, color="gray", lw=0.5, ls=":")
    _ax.set(xlabel="divisor c", ylabel="MSE of the variance estimate", yscale="log",
            title="n = 10 draws, true variance 1")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| distribution | MSE, c = 9 (unbiased) | c = 10 (MLE) | c = 11 | best c on the grid |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** dividing by a larger $c$ shrinks every estimate toward 0. why does a
    little shrinkage *lower* the MSE even though it adds bias? why does the best divisor grow
    with the kurtosis? (hint: what does a heavy tail do to the variance of the sum of squared
    deviations?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Stein's paradox

    you observe one draw $X \sim \mathcal{N}(\theta, I_d)$ of a $d$-dimensional mean, and want
    to estimate all $d$ coordinates of $\theta$. the obvious estimate is $X$ itself: it's
    unbiased, it's the MLE, and its total MSE is exactly $d$. the **James–Stein** estimator
    shrinks it toward the origin by a data-dependent amount:

    $$\hat\theta_{JS} = \Big(1 - \frac{d - 2}{\lVert X\rVert^2}\Big)X.$$

    the coordinates can be completely unrelated quantities (a batting average, the price of
    wheat, the number of cats in a city), rescaled so the noise has variance 1. your
    `bias_variance_mse` scores each coordinate; the plot shows the total MSE as the true
    $\lVert\theta\rVert$ grows.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "for d = 10, for which values of ||theta|| does James–Stein beat the MLE's total MSE of 10? "
        "and how can shrinking unrelated quantities toward an arbitrary point possibly help?"
    )
    p2
    return (p2,)


@app.cell
def _():
    dim = mo.ui.slider(steps=[1, 2, 3, 5, 10, 50], value=10, label="dimension d", show_value=True)
    dim
    return (dim,)


@app.cell
def _(dim, p2):
    ui.gate(p2)
    _d = dim.value
    _rng = np.random.default_rng(11)
    _norms = np.linspace(0, 3 * math.sqrt(_d) + 3, 13)
    _reps = 4000
    _mle, _js = [], []
    for _r in _norms:
        _theta = np.full(_d, _r / math.sqrt(_d))  # any direction gives the same risk
        _x = _theta + _rng.normal(size=(_reps, _d))
        _shrink = 1 - (_d - 2) / np.sum(_x**2, axis=1, keepdims=True)
        _est = _shrink * _x
        _mle.append(sum(ex.bias_variance_mse(_x[:, j], _theta[j])["mse"] for j in range(_d)))
        _js.append(sum(ex.bias_variance_mse(_est[:, j], _theta[j])["mse"] for j in range(_d)))
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _ax.plot(_norms, _mle, "o-", ms=3, label="MLE: X")
    _ax.plot(_norms, _js, "o-", ms=3, label="James–Stein")
    _ax.axhline(_d, color="gray", lw=0.6, ls=":")
    _ax.set(xlabel="true ||theta||", ylabel="total MSE (sum over coordinates)", title=f"d = {_d}")
    _ax.set_ylim(0, 1.6 * _d)  # for d = 1 the James-Stein risk is enormous near 0: see the table
    _ax.legend()
    _rows = "\n".join(f"| {_r:.2f} | {_a:.2f} | {_b:.2f} |" for _r, _a, _b in zip(_norms, _mle, _js))
    mo.vstack([_fig, mo.md("| true ‖θ‖ | MLE total MSE | James–Stein total MSE |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for $d \ge 3$ the James–Stein curve sits below the MLE's curve *everywhere*. so the
    MLE is **inadmissible**: another estimator is at least as good for every $\theta$ and
    strictly better for some. where is the gain largest, and why does it fade (but never turn
    into a loss) as $\lVert\theta\rVert$ grows? what goes wrong for $d = 1$ and $d = 2$? does
    James–Stein improve every *individual* coordinate, or only the total?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: consistency, and how fast

    five estimators of $\theta = 1$ from $n$ draws of Uniform$(0, \theta)$: twice the mean,
    the max, the corrected max, twice the median, and twice the *first* draw. your
    `mse_curve` measures each one's MSE for $n$ from 4 to 1024, and `convergence_rate` fits
    the slope on a log–log plot.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "which of these estimators are consistent? the 'standard' rate is MSE ~ 1/n. "
        "does any of them beat it?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    _u = lambda s, r: r.random(s)
    _ests = {
        "2 · mean": ex.theta_mom,
        "max (MLE)": ex.theta_max,
        "(n+1)/n · max": ex.theta_unbiased_max,
        "2 · median": lambda x: 2 * np.median(x, axis=-1),
        "2 · first draw": lambda x: 2 * x[:, 0],
    }
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _rows = []
    for _k, (_label, _e) in enumerate(_ests.items()):
        _m = ex.mse_curve(_e, _u, 1.0, _ns, 2000, rng=100 + _k)
        _ax.loglog(_ns, _m, "o-", ms=3, label=_label)
        _rows.append(f"| {_label} | {_m[0]:.2e} | {_m[-1]:.2e} | {ex.convergence_rate(_ns, _m):+.2f} |")
    _ax.set(xlabel="n", ylabel="MSE")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| estimator | MSE at n = 4 | MSE at n = 1024 | fitted rate |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** "twice the first draw" is unbiased for every $n$. why is it useless? why do
    the max-based estimators converge like $1/n^2$ while the mean and median manage only
    $1/n$? (think about how much one draw near $\theta$ tells you, and about what the max of
    $n$ uniforms looks like.) why is the median worse than the mean here, when for heavy-tailed
    data it's often better?
    """)
    return


if __name__ == "__main__":
    app.run()
