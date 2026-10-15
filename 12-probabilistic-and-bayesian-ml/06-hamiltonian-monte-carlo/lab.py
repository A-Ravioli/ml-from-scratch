import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.infer import effective_sample_size, metropolis_hastings  # yours, from topic 05

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hamiltonian monte carlo · lab

    every experiment here runs **your** `hmc` and `leapfrog` from `exercise.py`, and your
    topic 05 Metropolis sampler and ESS from `mlfs.infer`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: HMC versus random-walk Metropolis, at equal cost

    two targets: a 2-D Gaussian with correlation $\rho = 0.99$ (a narrow diagonal ridge, sd
    0.1 across it), and a 100-D standard Gaussian. each sampler gets about the same budget of
    **40,000 density or gradient evaluations**: Metropolis takes 40,000 steps (one density
    evaluation each); HMC with $L$ leapfrog steps takes $40{,}000/(L + 1)$ iterations. both
    are tuned sensibly (Metropolis near its optimal acceptance, HMC with a step size safely
    inside the stability limit). we report the smallest ESS over the coordinates, for $x$ and
    for $x^2$, and the bar chart shows the ESS of $x$ per 1,000 evaluations. (topic 05's
    estimator can't report more than $n\log_{10}n$: an anti-correlated chain hits that cap.)
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "HMC pays L + 1 gradients per sample. on the 100-D Gaussian, at equal cost, which gives more effective "
        "samples, and by roughly what factor?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _budget = 40_000
    _rho = 0.99
    _P = np.linalg.inv(np.array([[1, _rho], [_rho, 1]]))
    _targets = {
        "2-D, ρ = 0.99": (lambda x: -0.5 * x @ _P @ x, lambda x: -_P @ x, 2, 0.14, 0.08, 20),
        "100-D iid": (lambda x: -0.5 * float(x @ x), lambda x: -x, 100, 0.24, 0.25, 6),
    }
    _rows, _bars = [], {}
    for _name, (_lp, _g, _d, _s_mh, _eps, _L) in _targets.items():
        _x, _acc_mh = metropolis_hastings(_lp, np.zeros(_d), _budget, proposal_scale=_s_mh, burn_in=1000, rng=0)
        _ess_mh = float(np.min(effective_sample_size(_x)))
        _n = _budget // (_L + 1)
        _y, _acc_h = ex.hmc(_lp, np.zeros(_d), _n, grad_log_prob=_g, step_size=_eps, n_leapfrog=_L, burn_in=50, rng=1)
        _ess_h = float(np.min(effective_sample_size(_y)))
        _cap = _n * np.log10(_n)  # the largest ESS topic 05's estimator can report (tau floored at 1 / log10 n)
        _sq_mh = float(np.min(effective_sample_size(_x**2)))
        _sq_h = float(np.min(effective_sample_size(_y**2)))
        _bars[_name] = (1000 * _ess_mh / _budget, 1000 * _ess_h / _budget)
        _capped = " (at the cap)" if _ess_h > 0.99 * _cap else ""
        _rows.append(f"| {_name} | {_acc_mh:.2f} | {_ess_mh:.0f} / {_sq_mh:.0f} | {_acc_h:.2f} (L = {_L}, ε = {_eps}) | "
                     f"{_n} | {_ess_h:.0f}{_capped} / {_sq_h:.0f} | **{_ess_h / _ess_mh:.0f}x** / **{_sq_h / _sq_mh:.0f}x** |")
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    _xs = np.arange(len(_bars))
    _ax.bar(_xs - 0.2, [v[0] for v in _bars.values()], 0.4, label="random-walk Metropolis")
    _ax.bar(_xs + 0.2, [v[1] for v in _bars.values()], 0.4, label="HMC")
    _ax.set_xticks(_xs, list(_bars))
    _ax.set(ylabel="min ESS per 1,000 evaluations", yscale="log")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| target | MH acceptance | MH min ESS (x / x²) | HMC acceptance | HMC samples | HMC min ESS (x / x²) | ratio (x / x²) |\n"
                           "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** a Metropolis step moves a distance of about the proposal scale, in a random
    direction; an HMC trajectory moves a distance of about $L\varepsilon$, in a *consistent*
    direction. how many steps does a random walk need to cover a distance $D$ with steps of
    size $s$, and how many does a trajectory need? why does the gap widen with dimension (how
    must each sampler's step size scale with $d$ to keep its acceptance up)? what does HMC need
    that Metropolis doesn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the stability cliff

    HMC on a standard Gaussian with $L = 10$ leapfrog steps, for step sizes from 0.05 to 2.6,
    in $d = 1$ and in $d = 100$. we record the acceptance rate (500 iterations each) and the
    energy error $|H(q', p') - H(q, p)|$ of one trajectory from a typical starting point.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("in 1-D, how does the acceptance rate change as the step size goes from 1.5 to 2.1? gradually, "
                    "or suddenly?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _eps = np.round(np.concatenate([np.arange(0.05, 1.9, 0.15), np.arange(1.9, 2.61, 0.05)]), 3)
    _res = {}
    for _d in (1, 100):
        _acc, _err = [], []
        _rng = np.random.default_rng(2)
        _q0 = _rng.standard_normal(_d)
        _p0 = _rng.standard_normal(_d)
        for _k, _e in enumerate(_eps):
            _, _a = ex.hmc(lambda x: -0.5 * float(x @ x), _q0, 500, grad_log_prob=lambda x: -x, step_size=_e,
                           n_leapfrog=10, rng=10 + _k)
            _acc.append(_a)
            with np.errstate(over="ignore", invalid="ignore"):
                _q, _p = ex.leapfrog(_q0, _p0, lambda x: -x, _e, 10)
                _err.append(abs(0.5 * (_q @ _q + _p @ _p) - 0.5 * (_q0 @ _q0 + _p0 @ _p0)))
        _res[_d] = (np.array(_acc), np.array(_err))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for _d, _c in [(1, "C0"), (100, "C3")]:
        _axes[0].plot(_eps, _res[_d][0], "o-", ms=3, color=_c, label=f"d = {_d}")
        _axes[1].semilogy(_eps, np.maximum(_res[_d][1], 1e-16), "o-", ms=3, color=_c, label=f"d = {_d}")
    for _a in _axes:
        _a.axvline(2.0, color="k", ls=":", lw=1)
        _a.set(xlabel="step size ε")
        _a.legend(fontsize=8)
    _axes[0].set(ylabel="acceptance rate")
    _axes[1].set(ylabel="|energy error| of one trajectory")
    _fig.tight_layout()
    _i = {e: k for k, e in enumerate(_eps)}
    _pick = [0.5, 0.95, 1.7, 1.85, 1.95, 2.0, 2.05, 2.2]
    _rows = "\n".join(f"| {e} | {_res[1][0][_i[e]]:.2f} | {_res[100][0][_i[e]]:.2f} |" for e in _pick if e in _i)
    mo.vstack([_fig, mo.md("| ε | acceptance, d = 1 | acceptance, d = 100 |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** one leapfrog step on $U(q) = q^2/2$ is a linear map of $(q, p)$. what are its
    eigenvalues, and what happens to them at $\varepsilon = 2$? why does that turn a small,
    bounded energy error into an exponentially growing one? why does the 100-D chain lose
    acceptance *before* the cliff, and how should the step size shrink with $d$ to keep the
    acceptance rate fixed? what does "a divergent transition" in Stan's output mean, in these
    terms?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how long should a trajectory be?

    HMC on a 1-D standard Gaussian with a small step size (0.1), varying the trajectory length
    $T = L\varepsilon$ from 0.1 to 7. for each $T$ we run 600 iterations and measure the ESS
    of $x$ and of $x^2$ per iteration.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("is a longer trajectory always better? what happens at T = 2π ≈ 6.3?")
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _Ls = np.unique(np.round(np.linspace(1, 70, 30)).astype(int))
    _e1, _e2 = [], []
    for _k, _L in enumerate(_Ls):
        _s, _ = ex.hmc(lambda x: -0.5 * float(x @ x), np.array([0.5]), 600, grad_log_prob=lambda x: -x,
                       step_size=0.1, n_leapfrog=int(_L), burn_in=20, rng=100 + _k)
        _e1.append(effective_sample_size(_s[:, 0]) / 600)
        _e2.append(effective_sample_size(_s[:, 0] ** 2) / 600)
    _T = _Ls * 0.1
    _fig, _ax = plt.subplots(figsize=(8, 3.8))
    _ax.semilogy(_T, _e1, "o-", ms=3, label="ESS of x per iteration")
    _ax.semilogy(_T, _e2, "s-", ms=3, label="ESS of x² per iteration")
    for _m, _lab in [(np.pi / 2, "π/2"), (np.pi, "π"), (2 * np.pi, "2π")]:
        _ax.axvline(_m, color="k", ls=":", lw=1)
        _ax.text(_m, 1.02, _lab, transform=_ax.get_xaxis_transform(), ha="center", fontsize=8)
    _ax.axhline(1, color="0.5", lw=0.8)
    _ax.set(xlabel="trajectory length T = L ε", ylabel="ESS / iteration")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _near = lambda t: int(np.argmin(np.abs(_T - t)))
    _rows = "\n".join(f"| {_T[_near(t)]:.1f} | {_e1[_near(t)]:.2f} | {_e2[_near(t)]:.2f} |"
                      for t in (0.3, np.pi / 2, np.pi, 4.7, 2 * np.pi))
    mo.vstack([_fig, mo.md("| T | ESS(x) / iter | ESS(x²) / iter |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for $U(q) = q^2/2$ the exact dynamics are a rotation in the $(q, p)$ plane with
    period $2\pi$. where does a trajectory of length $\pi/2$ take a point, and why does that make
    the next draw independent of the current one? what does length $\pi$ do to $x$, and to $x^2$?
    why can the ESS of $x$ exceed the number of iterations? a real posterior has many different
    scales at once: why does this make a single fixed $T$ a bad idea, and how do NUTS (stop at a
    U-turn) and random jittering of $L$ address it?
    """)
    return


if __name__ == "__main__":
    app.run()
