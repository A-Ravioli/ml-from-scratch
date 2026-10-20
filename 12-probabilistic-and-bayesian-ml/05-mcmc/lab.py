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
    # markov chain monte carlo · lab

    every experiment here runs **your** samplers and diagnostics from `exercise.py`. for each
    one: write down a prediction, lock it in, run it, then explain what happened. compare
    your explanations with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: step size, acceptance, mixing

    random-walk Metropolis on a standard normal target, in $d = 1$ and in $d = 20$
    dimensions. for 25 proposal scales between 0.05 and 20 we run 6,000 steps (after 500 of
    burn-in) and record the acceptance rate and the effective sample size per step (averaged
    over the coordinates), with your `metropolis_hastings` and `effective_sample_size`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which acceptance rate gives the best mixing (the most effective samples per step) in 1-D? in 20-D? "
        "is a higher acceptance rate always better?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _scales = np.logspace(np.log10(0.05), np.log10(20), 25)
    _res = {}
    for _d in (1, 20):
        _acc, _eff = [], []
        for _k, _s in enumerate(_scales):
            _x, _a = ex.metropolis_hastings(lambda x: -0.5 * float(x @ x), np.zeros(_d), 6000,
                                            proposal_scale=_s, burn_in=500, rng=_k)
            _acc.append(_a)
            _eff.append(float(np.mean(ex.effective_sample_size(_x))) / 6000)  # averaged over coordinates
        _res[_d] = (np.array(_acc), np.array(_eff))
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _ax2 = _axes[0].twinx()
    _lines = []
    for _d, _c in [(1, "C0"), (20, "C3")]:
        _acc, _eff = _res[_d]
        _axes[0].semilogx(_scales, _acc, "o-", color=_c, ms=3, label=f"acceptance, d = {_d}")
        _ax2.loglog(_scales, np.maximum(_eff, 1e-5), "s--", color=_c, ms=3, alpha=0.7, label=f"ESS / step, d = {_d}")
        _axes[1].plot(_acc, _eff, "o", color=_c, label=f"d = {_d}")
        _best = int(np.argmax(_eff))
        _lines.append(f"| {_d} | {_scales[_best]:.2f} | {_acc[_best]:.2f} | {_eff[_best]:.3f} | "
                      f"{_acc[0]:.2f} | {_eff[0]:.4f} |")
    _axes[0].set(xlabel="proposal scale", ylabel="acceptance rate (solid)")
    _ax2.set(ylabel="ESS per step (dashed, log scale)")
    _axes[0].legend(fontsize=7, loc="center left")
    _ax2.legend(fontsize=7, loc="upper right")
    _axes[1].set(xlabel="acceptance rate", ylabel="ESS per step", yscale="log")
    _axes[1].axvline(0.44, color="C0", ls=":", lw=1)
    _axes[1].axvline(0.234, color="C3", ls=":", lw=1)
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| d | best scale | its acceptance | its ESS / step | acceptance at scale 0.05 | ESS / step there |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_lines))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at a tiny scale nearly every proposal is accepted. why is that the *worst*
    regime for mixing, and how does the chain behave there? what goes wrong at a huge scale?
    why is the best acceptance rate lower in 20 dimensions, and why does the best scale shrink
    with $d$ (compare it with $2.38/\sqrt{d}$)? what is ESS per step at the optimum in 20-D
    compared with 1-D, and what does that predict for $d = 1000$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a chain that looks fine

    a bimodal target: an equal mixture of $\mathcal N(-4, 1)$ and $\mathcal N(4, 1)$, whose
    true mean is 0. four Metropolis chains of 4,000 steps (scale 0.7), started at $-6$, $-3$,
    $3$ and $6$. we report each chain's mean and ESS, and your `r_hat` across the four.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("each chain alone has an ESS of several hundred. is chain 4's estimate of the mean (truth: 0) trustworthy? "
                    "what will R-hat say?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _logp(x):
        return float(np.logaddexp(-0.5 * (x[0] + 4) ** 2, -0.5 * (x[0] - 4) ** 2))

    _starts = [-6.0, -3.0, 3.0, 6.0]
    _chains = np.array([ex.metropolis_hastings(_logp, np.array([s]), 4000, proposal_scale=0.7, burn_in=200,
                                               rng=10 + k)[0][:, 0] for k, s in enumerate(_starts)])
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for _k, _c in enumerate(_chains):
        _axes[0].plot(_c, lw=0.6, label=f"chain {_k + 1} (start {_starts[_k]:+.0f})")
    _axes[0].set(xlabel="step", ylabel="x")
    _axes[0].legend(fontsize=7, ncol=2)
    _grid = np.linspace(-8, 8, 300)
    _axes[1].hist(_chains.ravel(), bins=60, density=True, alpha=0.5, label="all four chains")
    _axes[1].plot(_grid, 0.5 * (np.exp(-0.5 * (_grid + 4) ** 2) + np.exp(-0.5 * (_grid - 4) ** 2)) / np.sqrt(2 * np.pi),
                  "k", lw=1, label="target")
    _axes[1].legend(fontsize=7)
    _fig.tight_layout()
    _rows = "\n".join(f"| {k + 1} | {c.mean():+.3f} | {ex.effective_sample_size(c):.0f} | {np.mean(c > 0):.2f} |"
                      for k, c in enumerate(_chains))
    mo.vstack([_fig, mo.md(
        "| chain | mean | ESS | fraction of time in the right mode |\n|---|---|---|---|\n" + _rows
        + f"\n\nR-hat across the four chains: **{ex.r_hat(_chains):.2f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each chain's ESS measures how well it explores *the region it's in*. why
    can't a single chain's diagnostics tell you that it never visited the other mode? why does
    R-hat catch it? the target puts half its mass in each mode: what would you need for one
    random-walk chain to estimate that half correctly? would a chain started at 0 do better?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: Gibbs in a narrow valley

    Gibbs sampling a standard bivariate normal with correlation $\rho$ (slider), using its
    exact conditionals $x_1 \mid x_2 \sim \mathcal N(\rho x_2, 1 - \rho^2)$ and vice versa.
    we run 5,000 sweeps with your `gibbs_sampler` for several $\rho$ (ESS averaged over 4
    seeds) and plot the first 30 moves for the slider's value.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("with rho = 0.99, how many sweeps does it take to get one effectively independent draw of x1?")
    rho3 = mo.ui.slider(0.0, 0.995, value=0.95, step=0.005, label="ρ")
    mo.vstack([p3, rho3])
    return p3, rho3


@app.cell
def _(p3, rho3):
    ui.gate(p3)

    def _run(rho, n=5000, seed=0):
        sd = np.sqrt(1 - rho**2)
        conds = [lambda x, rng: rho * x[1] + sd * rng.standard_normal(),
                 lambda x, rng: rho * x[0] + sd * rng.standard_normal()]
        return ex.gibbs_sampler(conds, np.array([2.0, 2.0]), n, rng=seed)

    _rhos = [0.0, 0.5, 0.9, 0.95, 0.99, 0.995]
    _ess = [np.mean([ex.effective_sample_size(_run(r, seed=k)[:, 0]) for k in range(4)]) / 5000 for r in _rhos]
    _theory = [(1 - r**2) / (1 + r**2) for r in _rhos]
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    _path = _run(rho3.value, n=30, seed=1)
    _zig = [np.array([2.0, 2.0])]
    for _p in _path:
        _zig.append(np.array([_p[0], _zig[-1][1]]))
        _zig.append(_p.copy())
    _zig = np.array(_zig)
    _t = np.linspace(0, 2 * np.pi, 200)
    for _r in (1, 2):
        _L = np.linalg.cholesky(np.array([[1, rho3.value], [rho3.value, 1]]))
        _e = _r * (_L @ np.stack([np.cos(_t), np.sin(_t)]))
        _axes[0].plot(_e[0], _e[1], "k:", lw=0.8)
    _axes[0].plot(_zig[:, 0], _zig[:, 1], "C0-", lw=0.8)
    _axes[0].plot(_path[:, 0], _path[:, 1], "C0.", ms=4)
    _axes[0].set(aspect="equal", xlim=(-3, 3), ylim=(-3, 3), title=f"30 sweeps, ρ = {rho3.value}", xlabel="x1", ylabel="x2")
    _axes[1].semilogy(_rhos, _ess, "o-", label="measured ESS / sweep")
    _axes[1].semilogy(_rhos, _theory, "k:", label="(1 − ρ²) / (1 + ρ²)")
    _axes[1].set(xlabel="ρ", ylabel="ESS per sweep (x1)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {r} | {e:.4f} | {1 / e:.0f} |" for r, e in zip(_rhos, _ess))
    mo.vstack([_fig, mo.md("| ρ | ESS per sweep | sweeps per independent draw |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each Gibbs update moves parallel to an axis. why does that make the sampler
    crawl along a narrow diagonal valley? the $x_1$ chain is an AR(1) process with
    coefficient $\rho^2$: show that, and derive the dotted line from the AR(1) formula
    $\text{ESS}/n = (1 - \phi)/(1 + \phi)$. what single change of variables would make Gibbs
    perfect here, and what does that suggest for real models with correlated parameters?
    """)
    return


if __name__ == "__main__":
    app.run()
