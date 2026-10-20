import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def std_normal_logpdf(z):
        z = np.atleast_1d(z)
        return float(-0.5 * z @ z - 0.5 * z.size * np.log(2 * np.pi))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # integration & change of variables · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: does a transformed Gaussian still integrate to 1?

    $Z \sim \mathcal N(0, I_2)$ and $X = T(Z)$ for one of three invertible maps. we compute
    $p_X$ on a grid with your `pushforward_log_density` and integrate it with your
    `midpoint_rule`, once with the Jacobian term and once with it deliberately left out
    (just $p_Z(T^{-1}(x))$).

    - **scale**: $T(z) = (2z_1,\ z_2 + z_1)$, a linear map.
    - **banana**: $T(z) = (z_1,\ z_2 + 0.5z_1^2)$, a nonlinear shear.
    - **radial**: $T(z) = z\,(1 + \|z\|)$, which pushes mass outwards.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for each map, what do you get when you integrate the density WITHOUT the Jacobian term? "
        "is it always wrong?"
    )
    p1
    return (p1,)


@app.cell
def _():
    which = mo.ui.dropdown(["scale", "banana", "radial"], value="banana", label="map T")
    which
    return (which,)


@app.cell
def _(p1, which):
    ui.gate(p1)

    def _radial_inv(x):
        s = np.linalg.norm(x)
        r = (-1 + np.sqrt(1 + 4 * s)) / 2  # solve s = r (1 + r)
        return x / (1 + r)

    _maps = {
        "scale": (lambda z: np.array([2 * z[0], z[1] + z[0]]),
                  lambda x: np.array([x[0] / 2, x[1] - x[0] / 2]), 12.0),
        "banana": (lambda z: np.array([z[0], z[1] + 0.5 * z[0] ** 2]),
                   lambda x: np.array([x[0], x[1] - 0.5 * x[0] ** 2]), 12.0),
        "radial": (lambda z: z * (1 + np.linalg.norm(z)), _radial_inv, 30.0),
    }
    _T, _Tinv, _R = _maps[which.value]
    _with = lambda X: np.exp(ex.pushforward_log_density(std_normal_logpdf, _Tinv, X))
    _without = lambda X: np.exp([std_normal_logpdf(_Tinv(x)) for x in X])
    _lo, _hi = [-_R, -_R], [_R, _R]
    _I_with = ex.midpoint_rule(_with, _lo, _hi, 300)
    _I_without = ex.midpoint_rule(_without, _lo, _hi, 300)
    _z = np.random.default_rng(0).normal(size=(3000, 2))
    _x = np.array([_T(z) for z in _z])
    _g = np.linspace(-6, 6, 121)
    _GX, _GY = np.meshgrid(_g, _g)
    _P = _with(np.stack([_GX.ravel(), _GY.ravel()], axis=1)).reshape(_GX.shape)
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _ax.contourf(_GX, _GY, _P, levels=20, cmap="Blues")
    _ax.scatter(_x[:, 0], _x[:, 1], s=1, c="C3", alpha=0.3, label="samples T(z)")
    _ax.set(aspect="equal", xlim=(-6, 6), ylim=(-6, 6), title=f"{which.value}: your p_X with samples of X")
    _ax.legend(fontsize=8, loc="lower right")
    mo.vstack([_fig, mo.md(
        f"∫ p_X with the Jacobian term: **{_I_with:.4f}** · without it: **{_I_without:.4f}** "
        f"(midpoint rule, 300 × 300 cells on [−{_R:g}, {_R:g}]²).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compute $\det J_{T^{-1}}$ for each map by hand. why does the "wrong"
    density integrate to exactly 1 for one of them? for the linear map, the wrong integral is
    a simple number: which one? for normalizing flows (chapter 21), designers *choose* maps
    whose Jacobian determinant is cheap. why is a shear like the banana map such a popular
    building block?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: grids versus random points as the dimension grows

    the integral of $e^{x_1 + \dots + x_d}$ over the unit cube $[0, 1]^d$ is $(e - 1)^d$.
    with a fixed budget of about $2^{16} = 65{,}536$ function evaluations, we compare your
    `midpoint_rule` (with $n = \lfloor 65536^{1/d}\rfloor$ cells per axis) and your
    `monte_carlo_integrate` (65,536 random points), for $d = 1$ to $8$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the midpoint rule's error falls like h^2, Monte Carlo's like 1/sqrt(n). with the same "
        "number of evaluations, in which dimension does Monte Carlo start to win?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _f = lambda X: np.exp(np.sum(X, axis=1))
    _budget = 2**16
    _rows, _ds, _e_grid, _e_mc, _se = [], list(range(1, 9)), [], [], []
    for _d in _ds:
        _truth = (math.e - 1) ** _d
        _n = int(np.floor(_budget ** (1 / _d) + 1e-9))
        _grid = ex.midpoint_rule(_f, [0.0] * _d, [1.0] * _d, _n)
        _mc, _s = ex.monte_carlo_integrate(_f, [0.0] * _d, [1.0] * _d, _budget, rng=_d)
        _e_grid.append(abs(_grid - _truth) / _truth)
        _e_mc.append(abs(_mc - _truth) / _truth)
        _se.append(_s / _truth)
        _rows.append(f"| {_d} | {_n} ({_n ** _d:,} pts) | {_e_grid[-1]:.1e} | {_e_mc[-1]:.1e} | {_se[-1]:.1e} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogy(_ds, _e_grid, "o-", label="midpoint rule")
    _ax.semilogy(_ds, _e_mc, "s-", label="Monte Carlo (actual error)")
    _ax.semilogy(_ds, _se, "k:", label="Monte Carlo (its standard error)")
    _ax.set(xlabel="dimension d", ylabel="relative error", title=f"about {_budget:,} evaluations each")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| d | grid | midpoint error | Monte Carlo error | MC standard error |\n|---|---|---|---|---|\n"
        + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with $N$ evaluations in $d$ dimensions, the grid has $N^{1/d}$ cells per
    axis, so $h \propto N^{-1/d}$ and the error is about $N^{-2/d}$. Monte Carlo's error is
    about $\sigma N^{-1/2}$ in any dimension. where do the two cross? does Monte Carlo's
    standard error correctly report its own error? why is high-dimensional integration
    (posterior expectations, chapter 12) done by sampling?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: where is the peak of a transformed density?

    $Z \sim \mathcal N(0, \sigma^2)$ and $X = e^Z$ (a lognormal). the most likely value of $Z$
    is $0$, and $e^0 = 1$. we draw samples of $X$, and draw your
    `pushforward_log_density` of $X$ on top.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "where is the peak (the mode) of the density of X = e^Z? at 1 = e^(mode of Z), or somewhere else? "
        "what happens as sigma grows?"
    )
    p3
    return (p3,)


@app.cell
def _():
    sigma = mo.ui.slider(0.1, 1.5, value=1.0, step=0.1, label="σ")
    sigma
    return (sigma,)


@app.cell
def _(p3, sigma):
    ui.gate(p3)
    _s = sigma.value
    _log_pz = lambda z: float(-0.5 * (z[0] / _s) ** 2 - np.log(_s * np.sqrt(2 * np.pi)))
    _xs = np.exp(_s * np.random.default_rng(1).normal(size=20000))
    _grid = np.linspace(0.01, 5, 400)
    _dens = np.exp(ex.pushforward_log_density(_log_pz, np.log, _grid[:, None]))
    _mode = _grid[int(np.argmax(_dens))]
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.hist(_xs[_xs < 5], bins=120, density=True, alpha=0.4, label="samples of X = exp(Z)")
    _ax.plot(_grid, _dens, "C3", label="your p_X")
    _ax.axvline(1.0, color="k", ls=":", label="exp(mode of Z) = 1")
    _ax.axvline(_mode, color="C3", ls="--", label=f"mode of X ≈ {_mode:.3f}")
    _ax.set(xlabel="x", xlim=(0, 5))
    _ax.legend(fontsize=8)
    _mass = ex.midpoint_rule(lambda X: np.exp(ex.pushforward_log_density(_log_pz, np.log, X)), [1e-9], [60.0], 6000)
    mo.vstack([_fig, mo.md(f"mode of X: **{_mode:.3f}** (theory: e^(−σ²) = {np.exp(-_s**2):.3f}); "
                           f"median of the samples: {np.median(_xs):.3f}; ∫ p_X ≈ {_mass:.4f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** write $\log p_X(x) = \log p_Z(\log x) - \log x$ and maximize it. why does
    the $-\log x$ term move the peak? the median *does* transform ($e^{\text{median of }Z} = 1$):
    why do medians and other quantiles survive a monotone map while modes don't? what does
    this mean for "the most likely parameter value" when you reparameterize a model (for
    example, a variance versus a log-variance)?
    """)
    return


if __name__ == "__main__":
    app.run()
