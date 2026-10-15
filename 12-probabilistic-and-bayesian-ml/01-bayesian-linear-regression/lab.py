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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bayesian linear regression · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: where is the model unsure?

    40 noisy points from $y = \sin(2.5x)$, in two clumps: $x \in [-1, -0.35]$ and
    $x \in [0.35, 1]$, with a gap in the middle. we fit two Bayesian regressions, each with
    $\alpha$ and $\beta$ set by evidence maximization (`optimize=True`): one on a cubic
    polynomial basis, one on 12 Gaussian bumps of width 0.25 spread over $[-1, 1]$ (plus the
    constant). then we draw the predictive mean $\pm 2$ standard deviations on $[-3, 3]$, far
    beyond the data.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for each basis: is the predictive standard deviation at x = 3 (far outside the data) larger or smaller "
        "than at x = 0 (in the gap)? by roughly how much?"
    )
    p1
    return (p1,)


@app.cell
def _():
    _rng = np.random.default_rng(0)
    x1 = np.concatenate([_rng.uniform(-1, -0.35, 20), _rng.uniform(0.35, 1, 20)])
    y1 = np.sin(2.5 * x1) + _rng.normal(0, 0.15, 40)
    centers1 = np.linspace(-1, 1, 12)
    return centers1, x1, y1


@app.cell
def _(centers1, p1, x1, y1):
    ui.gate(p1)
    _grid = np.linspace(-3, 3, 400)
    _bases = {
        "cubic polynomial": lambda x: ex.polynomial_basis(x, 3),
        "12 Gaussian bumps": lambda x: ex.gaussian_basis(x, centers1, 0.25),
    }
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    _rows = []
    for _ax, (_name, _phi) in zip(_axes, _bases.items()):
        _m = ex.BayesianLinearRegression(optimize=True).fit(_phi(x1), y1)
        _mu, _sd = _m.predict(_phi(_grid), return_std=True)
        _ax.fill_between(_grid, _mu - 2 * _sd, _mu + 2 * _sd, alpha=0.25, label="±2 sd")
        _ax.plot(_grid, _mu, label="predictive mean")
        _ax.plot(_grid, np.sin(2.5 * _grid), "k:", lw=1, label="truth")
        _ax.plot(x1, y1, "k.", ms=4)
        _ax.set(title=_name, xlabel="x", ylim=(-4, 4))
        _sd_at = {v: float(_m.predict(_phi(np.array([v])), return_std=True)[1][0]) for v in (0.0, 0.7, 3.0)}
        _rows.append(f"| {_name} | {_sd_at[0.7]:.3f} | {_sd_at[0.0]:.3f} | {_sd_at[3.0]:.3f} | "
                     f"{1 / np.sqrt(_m.beta_):.3f} |")
    _axes[0].legend(fontsize=8, loc="lower left")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| basis | sd at x = 0.7 (data) | sd at x = 0 (gap) | sd at x = 3 (far) | noise sd $1/\\sqrt{\\beta}$ |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the predictive variance is $1/\beta + \phi(x)^\top S_N\,\phi(x)$. what happens to
    $\phi(x)$ for each basis as $x$ moves far from the data, and so to the second term? which
    model's error bars would you trust at $x = 3$, and is that because it *knows* it's
    extrapolating? what would you have to change about the Gaussian-bump model to make its
    uncertainty grow away from the data?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the evidence picks a degree

    30 points from a cubic, $y = x^3 - x + \varepsilon$ with noise sd 0.2 (move the slider to
    change the number of points). for each polynomial degree 0–9 we fit with evidence
    maximization and record three numbers: the training RMSE, the test RMSE on 2,000 fresh
    points, and the log evidence $\log p(y \mid \hat\alpha, \hat\beta)$ of the training data.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("which degree has the highest evidence? the lowest training error? the lowest test error?")
    n2 = mo.ui.slider(10, 200, value=30, step=5, label="training points")
    mo.vstack([p2, n2])
    return n2, p2


@app.cell
def _(n2, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _x = _rng.uniform(-1.2, 1.2, n2.value)
    _y = _x**3 - _x + _rng.normal(0, 0.2, n2.value)
    _xt = _rng.uniform(-1.2, 1.2, 2000)
    _yt = _xt**3 - _xt + _rng.normal(0, 0.2, 2000)
    _degs = np.arange(10)
    _tr, _te, _ev = [], [], []
    for _d in _degs:
        _m = ex.BayesianLinearRegression(optimize=True, max_iters=500).fit(ex.polynomial_basis(_x, _d), _y)
        _tr.append(np.sqrt(np.mean((_m.predict(ex.polynomial_basis(_x, _d)) - _y) ** 2)))
        _te.append(np.sqrt(np.mean((_m.predict(ex.polynomial_basis(_xt, _d)) - _yt) ** 2)))
        _ev.append(_m.log_evidence(ex.polynomial_basis(_x, _d), _y))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _axes[0].plot(_degs, _tr, "o-", label="train RMSE")
    _axes[0].plot(_degs, _te, "s-", label="test RMSE")
    _axes[0].axhline(0.2, color="k", ls=":", lw=1, label="noise sd")
    _axes[0].set(xlabel="degree", ylabel="RMSE", yscale="log")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_degs, _ev, "o-", color="C2")
    _axes[1].axvline(int(np.argmax(_ev)), color="C2", ls="--", lw=1)
    _axes[1].set(xlabel="degree", ylabel="log evidence")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"highest evidence at degree **{int(np.argmax(_ev))}**; lowest training RMSE at degree "
        f"**{int(np.argmin(_tr))}**; lowest test RMSE at degree **{int(np.argmin(_te))}**. "
        f"log evidence by degree: {', '.join(f'{e:.1f}' for e in _ev)}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** training error (almost) only goes down as the degree grows. why does the evidence go
    *down* after the true degree, even though the fit gets better? (think of the evidence as
    the probability the model assigned to *this* dataset before seeing it: what does a
    degree-9 model spread its probability over?) the evidence used only the training data.
    how does it compare with the test error as a model selector, and what happens with 10
    points? with 200?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: one point at a time

    a straight line $y = w_0 + w_1 x$, true weights $(-0.3, 0.5)$, noise sd 0.2, prior
    $\alpha = 2$. we feed the points in one at a time with your `sequential_posterior` and
    show the posterior density over $(w_0, w_1)$ and six lines drawn from it, after 0, 1, 2
    and 20 points.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("after the FIRST data point, what shape is the posterior over (w0, w1): a smaller round blob, "
                    "an ellipse, or something else?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _x = _rng.uniform(-1, 1, 20)
    _y = -0.3 + 0.5 * _x + _rng.normal(0, 0.2, 20)
    _beta, _alpha = 25.0, 2.0
    _m, _S = np.zeros(2), np.eye(2) / _alpha
    _snap = {0: (_m, _S)}
    for _i in range(20):
        _m, _S = ex.sequential_posterior(_m, _S, np.array([[1.0, _x[_i]]]), _y[_i:_i + 1], _beta)
        if _i + 1 in (1, 2, 20):
            _snap[_i + 1] = (_m, _S)
    _w0, _w1 = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    _W = np.stack([_w0, _w1], -1)
    _fig, _axes = plt.subplots(2, 4, figsize=(13, 6))
    _lines = np.linspace(-1, 1, 2)
    _info = []
    for _k, (_n, (_mm, _SS)) in enumerate(_snap.items()):
        _d = _W - _mm
        _q = np.einsum("...i,ij,...j->...", _d, np.linalg.inv(_SS), _d)
        _axes[0, _k].contourf(_w0, _w1, np.exp(-0.5 * _q), levels=12, cmap="Blues")
        _axes[0, _k].plot(-0.3, 0.5, "r+", ms=12, mew=2)
        _axes[0, _k].set(title=f"{_n} point" + ("" if _n == 1 else "s"), xlabel="w0", ylabel="w1", aspect="equal")
        _ws = _rng.multivariate_normal(_mm, _SS, 6)
        for _w in _ws:
            _axes[1, _k].plot(_lines, _w[0] + _w[1] * _lines, "C1", lw=1)
        _axes[1, _k].plot(_x[:_n], _y[:_n], "ko", ms=4)
        _axes[1, _k].set(xlim=(-1, 1), ylim=(-1.5, 1.5), xlabel="x")
        _ev = np.linalg.eigvalsh(_SS)
        _info.append(f"| {_n} | {np.sqrt(_ev[0]):.3f} | {np.sqrt(_ev[1]):.3f} | {np.sqrt(_ev[1] / _ev[0]):.1f} |")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| points | narrowest sd | widest sd | elongation |\n|---|---|---|---|\n" + "\n".join(_info))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** one point $(x_1, y_1)$ says $w_0 + w_1 x_1 \approx y_1$. what set of weights is
    that, and why does the posterior collapse along one direction while staying as wide as
    the prior along another? what do the six sampled lines after one point have in common?
    the first two points happen to sit fairly close together (at $x \approx -0.83$ and $-0.52$): why does the
    second one barely narrow the wide direction, and where would you put a second point to
    shrink it most?
    """)
    return


if __name__ == "__main__":
    app.run()
