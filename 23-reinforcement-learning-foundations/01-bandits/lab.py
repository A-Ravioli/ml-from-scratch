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
    # bandits · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. each experiment averages many seeded runs, so
    the numbers are reproducible; the whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: regret curves

    five Bernoulli arms with means $(0.3, 0.4, 0.45, 0.5, 0.6)$, 3,000 rounds, 60 seeded runs
    per agent (the bandit's seed is shared across agents, so they face the same reward
    streams). the agents: greedy ($\varepsilon = 0$, random ties), ε-greedy with
    $\varepsilon = 0.1$ and $0.01$, UCB with $c = 1$, and Thompson sampling with a uniform
    prior. the plots show the average cumulative pseudo-regret, on a linear and on a log
    time axis.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the five agents by regret after 3,000 rounds. which curves are straight lines on the linear plot, "
        "which on the log-time plot? does UCB (logarithmic regret, in theory) beat epsilon-greedy (linear) here?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    means1 = [0.3, 0.4, 0.45, 0.5, 0.6]
    T1, R1 = 3000, 60
    makers = {
        "greedy (ε = 0)": lambda s: ex.EpsilonGreedy(5, 0.0, rng=s),
        "ε-greedy, ε = 0.1": lambda s: ex.EpsilonGreedy(5, 0.1, rng=s),
        "ε-greedy, ε = 0.01": lambda s: ex.EpsilonGreedy(5, 0.01, rng=s),
        "UCB, c = 1": lambda s: ex.UCB(5, c=1.0),
        "Thompson": lambda s: ex.ThompsonSampling(5, rng=s),
    }
    curves1 = {}
    for _name, _make in makers.items():
        curves1[_name] = np.array([ex.run_bandit(ex.BernoulliBandit(means1, rng=1000 + _s), _make(_s), T1)[1]
                                   for _s in range(R1)])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _t = np.arange(1, T1 + 1)
    for _name, _regs in curves1.items():
        for _ax in _axes:
            _ax.plot(_t, _regs.mean(0), label=_name, lw=1.4)
    _axes[0].set(xlabel="round t", ylabel="average cumulative regret", title="linear time")
    _axes[1].set(xscale="log", xlabel="round t (log scale)", title="log time")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _name, _regs in curves1.items():
        _late = _regs[:, -1] - _regs[:, 1999]  # regret added in the last 1,000 rounds
        _rows.append(f"| {_name} | {_regs[:, 99].mean():.1f} | {_regs[:, 999].mean():.1f} | {_regs[:, -1].mean():.1f} "
                     f"| {_late.mean():.1f} | {np.mean(_late > 50):.0%} |")
    mo.vstack([_fig, mo.md(
        "| agent | regret at 100 | at 1,000 | at 3,000 | added in rounds 2,001–3,000 | runs adding > 50 there |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which agents have linear regret, and *why* is each linear (what does it
    keep doing wrong forever)? the last column counts runs that still lose more than 0.05
    per round at the end: what are those runs doing? UCB's guarantee is $O(\ln T)$, yet
    ε-greedy with $\varepsilon = 0.1$ is level with it or ahead at 3,000 rounds: what's
    hiding in the $O$? use the slider below to try other settings.
    """)
    return


@app.cell
def _():
    eps_s = mo.ui.slider(0.0, 0.3, value=0.05, step=0.01, label="ε for ε-greedy")
    c_s = mo.ui.slider(0.0, 3.0, value=0.5, step=0.1, label="c for UCB")
    mo.hstack([eps_s, c_s])
    return c_s, eps_s


@app.cell
def _(c_s, eps_s, p1):
    ui.gate(p1)
    _means = [0.3, 0.4, 0.45, 0.5, 0.6]
    _eg = np.mean([ex.run_bandit(ex.BernoulliBandit(_means, rng=1000 + s), ex.EpsilonGreedy(5, eps_s.value, rng=s),
                                 3000)[1][-1] for s in range(20)])
    _ucb = np.mean([ex.run_bandit(ex.BernoulliBandit(_means, rng=1000 + s), ex.UCB(5, c=c_s.value), 3000)[1][-1]
                    for s in range(20)])
    mo.md(f"20 runs, 3,000 rounds: ε-greedy (ε = {eps_s.value:.2f}) regret **{_eg:.1f}**, "
          f"UCB (c = {c_s.value:.1f}) regret **{_ucb:.1f}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: which gap is hardest?

    two arms, means $0.5$ and $0.5 + \Delta$, for eight gaps $\Delta$ from $0.005$ to
    $0.45$. Thompson sampling and UCB ($c = 1$) each play 2,000 rounds, 30 runs per gap. for
    reference the plot also shows the trivial bound $\Delta T$ (always playing the bad arm)
    and the asymptotic Lai–Robbins rate $\frac{\Delta}{\mathrm{KL}}\ln T$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how does the regret after 2,000 rounds depend on the gap? is a tiny gap (0.505 vs 0.5) the hardest "
        "problem, the easiest, or neither?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    T2 = 2000
    gaps = np.geomspace(0.005, 0.45, 8)
    _ts, _ucb, _lr = [], [], []
    for _d in gaps:
        _m = [0.5, 0.5 + _d]
        _ts.append(np.mean([ex.run_bandit(ex.BernoulliBandit(_m, rng=s), ex.ThompsonSampling(2, rng=s + 99), T2)[1][-1]
                            for s in range(30)]))
        _ucb.append(np.mean([ex.run_bandit(ex.BernoulliBandit(_m, rng=s), ex.UCB(2, c=1.0), T2)[1][-1]
                             for s in range(30)]))
        _lr.append(ex.lai_robbins_constant(_m) * np.log(T2))
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.loglog(gaps, _ts, "o-", label="Thompson")
    _ax.loglog(gaps, _ucb, "s-", label="UCB, c = 1")
    _ax.loglog(gaps, gaps * T2, "k:", label="ΔT (always the bad arm)")
    _ax.loglog(gaps, _lr, "k--", label="Lai–Robbins (Δ/KL) ln T")
    _ax.set(xlabel="gap Δ", ylabel="regret after 2,000 rounds", ylim=(1, 2000))
    _ax.legend(fontsize=8)
    _peak = gaps[int(np.argmax(_ts))]
    mo.vstack([_fig, mo.md(
        "| gap Δ | " + " | ".join(f"{d:.3f}" for d in gaps) + " |\n|---|" + "---|" * len(gaps) + "\n"
        "| Thompson | " + " | ".join(f"{v:.1f}" for v in _ts) + " |\n"
        "| UCB | " + " | ".join(f"{v:.1f}" for v in _ucb) + " |\n\n"
        f"Thompson's regret peaks at Δ ≈ **{_peak:.3f}**; 1/√T = {1 / np.sqrt(T2):.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is regret small for *both* tiny and large gaps? which of the two
    reference lines is the right description on each side of the peak, and why does the
    Lai–Robbins line blow up as $\Delta \to 0$ while the measured regret doesn't? roughly
    where should the peak be for a horizon $T$ (how many pulls does it take to tell two
    means $\Delta$ apart)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: beating the lower bound?

    two arms with means $0.45$ and $0.55$, 20,000 rounds, 100 runs of Thompson sampling. the
    Lai–Robbins theorem says no reasonable algorithm's regret can grow slower than
    $C\ln T$ with $C = \sum_a\Delta_a/\mathrm{KL}(\mu_a, \mu^*)$, computed by your
    `lai_robbins_constant`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "is Thompson sampling's average regret above or below the curve C ln t? what is the slope of regret "
        "against ln t over the last decade (rounds 2,000 to 20,000), compared with C?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _m = [0.45, 0.55]
    _T = 20_000
    _runs = np.array([ex.run_bandit(ex.BernoulliBandit(_m, rng=s), ex.ThompsonSampling(2, rng=s + 7), _T)[1]
                      for s in range(100)])
    _regs = _runs.mean(0)
    _C = ex.lai_robbins_constant(_m)
    _t = np.arange(1, _T + 1)
    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.semilogx(_t, _regs, label="Thompson (average of 100 runs)")
    _ax.semilogx(_t, _C * np.log(_t), "k--", label=f"C ln t, C = {_C:.2f}")
    _ax.set(xlabel="round t (log scale)", ylabel="cumulative regret")
    _ax.legend(fontsize=8)
    _se = lambda v: v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731
    _s1 = (_runs[:, 1999] - _runs[:, 199]) / np.log(10)
    _s2 = (_runs[:, -1] - _runs[:, 1999]) / np.log(10)
    mo.vstack([_fig, mo.md(
        f"regret at 20,000 rounds: **{_regs[-1]:.1f} ± {_se(_runs[:, -1]):.1f}** (mean ± s.e. over 100 runs), versus "
        f"C ln T = **{_C * np.log(_T):.1f}**. slope of regret against ln t: **{_s1.mean():.2f} ± {_se(_s1):.2f}** over "
        f"rounds 200–2,000, **{_s2.mean():.2f} ± {_se(_s2):.2f}** over 2,000–20,000 (C = {_C:.2f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** does this contradict the theorem? read its statement carefully (what is
    the limit taken over, and what does $\liminf$ allow at finite $T$?). how many pulls of
    the worse arm does each picture imply, and how many does the bound's $\ln T/\mathrm{KL}$
    ask for? what does this tell you about using asymptotic bounds to choose an algorithm
    for a 20,000-round experiment?
    """)
    return


if __name__ == "__main__":
    app.run()
