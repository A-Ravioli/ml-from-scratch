import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.optim import gradient_descent  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # momentum & acceleration · lab

    every experiment here runs **your** code from `exercise.py` (and your `gradient_descent`
    from topic 02). for each one: write down a prediction, lock it in, run it, then explain
    what happened. compare your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the race

    a 50-dimensional quadratic $f(x) = \tfrac12x^\top Ax$ with eigenvalues spread on a log
    scale from $\mu = 1$ to $L = 1{,}000$ ($\kappa = 1{,}000$), started from a random point.
    four racers, each run until $\|\nabla f\|$ has shrunk by $10^8$ (at most 20,000 steps):

    - gradient descent with step $1/L$ (your `gradient_descent`);
    - heavy ball with `tuned_momentum(L, mu)`;
    - Nesterov with `tuned_momentum(L, mu, nesterov=True)` (constant momentum);
    - Nesterov with the $(k - 1)/(k + 2)$ schedule and step $1/L$ (it doesn't know $\mu$).

    for the first three we also print the rate your `momentum_spectral_radius` predicts.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many iterations does each racer need? by what factor does tuned momentum beat gradient descent, "
        "and where does the schedule version (which doesn't know mu) finish?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _eigs = np.geomspace(1.0, 1000.0, 50)
    _A = np.diag(_eigs)
    _f = lambda x: float(0.5 * x @ _A @ x)
    _g = lambda x: _A @ x
    _x0 = np.random.default_rng(0).normal(size=50)
    _tol = 1e-8 * np.linalg.norm(_g(_x0))
    _L, _mu = 1000.0, 1.0
    _hb = ex.tuned_momentum(_L, _mu)
    _ne = ex.tuned_momentum(_L, _mu, nesterov=True)
    _runs = {
        "gradient descent, 1/L": (gradient_descent(_f, _g, _x0, lr=1 / _L, tol=_tol, max_iters=20_000),
                                  ex.momentum_spectral_radius(_eigs, 1 / _L, 0.0)),
        "heavy ball, tuned": (ex.heavy_ball(_f, _g, _x0, lr=_hb[0], momentum=_hb[1], tol=_tol, max_iters=20_000),
                              ex.momentum_spectral_radius(_eigs, *_hb)),
        "Nesterov, tuned constant": (ex.nesterov(_f, _g, _x0, lr=_ne[0], momentum=_ne[1], tol=_tol, max_iters=20_000),
                                     ex.momentum_spectral_radius(_eigs, *_ne, nesterov=True)),
        "Nesterov, (k−1)/(k+2)": (ex.nesterov(_f, _g, _x0, lr=1 / _L, tol=_tol, max_iters=20_000), None),
    }
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _rows = []
    for _name, (_r, _rho) in _runs.items():
        _h = np.maximum(np.array(_r.history), 1e-300)
        _ax.semilogy(_h, label=_name, lw=1.2)
        _tail = _h[-min(200, len(_h) - 1):]
        _meas = np.exp(np.mean(np.diff(np.log(_tail)))) if len(_tail) > 2 else float("nan")
        _pred = "—" if _rho is None else f"{_rho:.5f} (so {_rho ** 2:.5f} for f)"
        _rows.append(f"| {_name} | {_r.n_iters:,} | {'yes' if _r.converged else 'no'} | {_meas:.5f} | {_pred} |")
    _ax.set(xscale="log", xlabel="iteration", ylabel="f − f*", title="κ = 1,000, 50-D quadratic")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| method | iterations | converged | measured f-factor per step (last 200) | predicted rate (distance) |\n"
        "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the iteration counts with $\kappa\ln(1/\epsilon)$ and
    $\sqrt\kappa\ln(1/\epsilon)$. why is the measured factor for $f$ the *square* of the
    predicted rate? the schedule version doesn't know $\mu$: what does its curve look like,
    and why is it slower than the tuned constant at the end, even though it's "optimal"?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: ripples, and restarting the momentum

    a 2-D quadratic with $\kappa = 10^4$, solved by Nesterov with the $(k - 1)/(k + 2)$
    schedule (step $1/L$) for 3,000 steps. the schedule pushes the momentum towards 1, which
    is right for convex functions but too much for a strongly convex one. a simple fix is a
    **restart**: every $K$ steps, throw the momentum away and start the schedule again from the
    current point. we do that by calling your `nesterov` repeatedly for $K$ steps at a time.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "without restarts, is f(x_k) monotone? after 3,000 steps, roughly how far is f from f*? "
        "what will restarting every ~200 steps do?"
    )
    p2
    return (p2,)


@app.cell
def _():
    period = mo.ui.slider(20, 1000, value=200, step=20, label="restart every K steps")
    period
    return (period,)


@app.cell
def _(p2, period):
    ui.gate(p2)
    _A = np.diag([1.0, 1e4])
    _f = lambda x: float(0.5 * x @ _A @ x)
    _g = lambda x: _A @ x
    _x0 = np.array([1.0, 0.01])
    _n = 3000
    _plain = ex.nesterov(_f, _g, _x0, lr=1e-4, max_iters=_n, tol=0.0).history
    _K = int(period.value)
    _restarted = [_f(_x0)]
    _x = _x0
    while len(_restarted) <= _n:
        _r = ex.nesterov(_f, _g, _x, lr=1e-4, max_iters=min(_K, _n + 1 - len(_restarted)), tol=0.0)
        _restarted += _r.history[1:]
        _x = _r.x
    _gd = gradient_descent(_f, _g, _x0, lr=1e-4, max_iters=_n, tol=0.0).history
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.semilogy(np.maximum(_gd, 1e-300), label="gradient descent, 1/L")
    _ax.semilogy(np.maximum(_plain, 1e-300), label="Nesterov, schedule")
    _ax.semilogy(np.maximum(_restarted[: _n + 1], 1e-300), label=f"Nesterov, restart every {_K}")
    _lo = max(min(np.min(_plain[: _n + 1]), np.min(_restarted[: _n + 1])), 1e-300) / 10
    _ax.set(xlabel="iteration", ylabel="f − f*", title="κ = 10⁴", ylim=(_lo, 10))
    _ax.legend(fontsize=8)
    _ups = int(np.sum(np.diff(_plain) > 0))
    mo.vstack([_fig, mo.md(
        f"after {_n} steps: gradient descent **{_gd[-1]:.2e}**, Nesterov **{_plain[-1]:.2e}** "
        f"(f went *up* on {_ups} of its steps), restarted every {_K}: **{_restarted[_n]:.2e}**. "
        f"√κ = {np.sqrt(1e4):.0f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where do the ripples come from (what happens to the momentum as
    $k$ grows, and what does an underdamped oscillator do)? why is the schedule's long-run
    behaviour no better than linear with a poor rate? why does restarting help, and why is
    the best restart period on the order of $\sqrt\kappa$? try a very short and a very long
    period with the slider.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: optimal for quadratics, broken elsewhere

    a 1-D function that is strongly convex ($\mu = 1$) and smooth ($L = 25$), but not a
    quadratic: three quadratic pieces glued together, with gradient $25x$ for $x < 1$,
    $x + 24$ for $1 \le x < 2$ and $25x - 24$ for $x \ge 2$. its minimum is at 0. on any
    quadratic with curvature in $[1, 25]$, heavy ball with `tuned_momentum(25, 1)` is
    guaranteed the rate $(\sqrt{25} - 1)/(\sqrt{25} + 1) = 2/3$. we run your `heavy_ball` and
    your `nesterov` with their tuned parameters from $x_0 = 3.3$, and from 201 starts in $[-5, 5]$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the function is smooth and strongly convex with kappa = 25. does tuned heavy ball converge from x0 = 3.3? "
        "from what fraction of starts in [-5, 5]?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)

    def _f(x):
        v = x[0]
        return float(12.5 * v * v if v < 1 else (0.5 * v * v + 24 * v - 12 if v < 2 else 12.5 * v * v - 24 * v + 36))

    def _g(x):
        v = x[0]
        return np.array([25 * v if v < 1 else (v + 24 if v < 2 else 25 * v - 24)])

    _hb = ex.tuned_momentum(25.0, 1.0)
    _ne = ex.tuned_momentum(25.0, 1.0, nesterov=True)
    _path_hb, _path_ne = [3.3], [3.3]
    _r_hb = ex.heavy_ball(_f, _g, np.array([3.3]), lr=_hb[0], momentum=_hb[1], max_iters=60, tol=1e-12,
                          callback=lambda x, k: _path_hb.append(x[0]))
    _r_ne = ex.nesterov(_f, _g, np.array([3.3]), lr=_ne[0], momentum=_ne[1], max_iters=60, tol=1e-12,
                        callback=lambda x, k: _path_ne.append(x[0]))
    _starts = np.linspace(-5, 5, 201)
    _ok_hb = np.mean([ex.heavy_ball(_f, _g, np.array([s]), lr=_hb[0], momentum=_hb[1], max_iters=500, tol=1e-9).converged
                      for s in _starts])
    _ok_ne = np.mean([ex.nesterov(_f, _g, np.array([s]), lr=_ne[0], momentum=_ne[1], max_iters=500, tol=1e-9).converged
                      for s in _starts])
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.plot(_path_hb, "o-", ms=3, label="heavy ball, tuned")
    _ax.plot(_path_ne, "s-", ms=3, label="Nesterov, tuned")
    _ax.axhspan(1, 2, color="0.9", label="middle piece (curvature 1)")
    _ax.set(xlabel="iteration", ylabel="x_k", title="from x₀ = 3.3")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"tuned heavy ball: lr = {_hb[0]:.4f}, momentum = {_hb[1]:.4f}; predicted rate on quadratics "
        f"**{ex.momentum_spectral_radius([1.0, 25.0], *_hb):.4f}**. from 3.3: converged **{_r_hb.converged}**, last "
        f"iterates {np.round(_path_hb[-6:], 3).tolist()}. Nesterov from 3.3: converged **{_r_ne.converged}** in "
        f"{_r_ne.n_iters} steps.\n\nfraction of 201 starts in [−5, 5] that converge within 500 steps: heavy ball "
        f"**{_ok_hb:.0%}**, Nesterov **{_ok_ne:.0%}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the rate $2/3$ was derived for quadratics. which step of that derivation
    fails when the curvature depends on where you are? follow the cycle: which piece is each
    of its three points on, and what curvature does each step "see"? why does Nesterov's
    look-ahead gradient save it here (and why does its theory cover every smooth strongly
    convex function)? what does this say about trusting quadratic analyses of optimizers for
    neural networks?
    """)
    return


if __name__ == "__main__":
    app.run()
