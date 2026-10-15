import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import ndtri

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def bern(p, u):
    """a Bernoulli(p) draw from standard normal noise u: P(u < Phi^-1(p)) = p."""
    return (u < ndtri(np.clip(p, 1e-12, 1 - 1e-12))).astype(float)


@app.function
def true_effect(scm, x, y, n, seed):
    """E[y | do(x=1)] - E[y | do(x=0)] by simulation, with the same noise in both worlds."""
    return float(scm.sample(n, do={x: 1.0}, rng=seed)[y].mean() - scm.sample(n, do={x: 0.0}, rng=seed)[y].mean())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # structural causal models & do-calculus · lab

    every experiment here runs **your** code from `exercise.py`: your `SCM` simulates the
    worlds, your graph functions check adjustment sets, and your adjustment formulas
    estimate effects. because you wrote the world, you know the true causal effect exactly
    (run it twice with `do` and the same seed), so you can grade every estimate. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. the whole lab takes under a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: healthier, and harmed

    a simulated population. $Z$ is health-consciousness (half the people have it), $X$ is
    taking a daily vitamin, $Y$ is a health score:

    $$Z \sim \mathrm{Bern}(0.5),\quad X \sim \mathrm{Bern}\big(0.5 + 0.45\,s\,(2Z - 1)\big),\quad Y := -0.2X + Z + 0.5\,U_Y.$$

    the vitamin is mildly **harmful** ($-0.2$). the knob $s \in [0, 1]$ sets how strongly
    health-consciousness drives vitamin-taking. for five values of $s$ the lab draws
    20,000 people and compares the naive difference
    $\mathbb E[Y \mid X{=}1] - \mathbb E[Y \mid X{=}0]$, your backdoor estimate adjusting for
    $Z$, and the truth.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at full confounding (s = 1), what's the naive difference in health between vitamin-takers and others? "
        "positive or negative, and roughly how big compared with the true effect -0.2?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def vitamin_world(s):
        parents = {"Z": [], "X": ["Z"], "Y": ["X", "Z"]}
        mech = {"Z": lambda pa, u: bern(0.5, u),
                "X": lambda pa, u: bern(0.5 + 0.45 * s * (2 * pa["Z"] - 1), u),
                "Y": lambda pa, u: -0.2 * pa["X"] + pa["Z"] + 0.5 * u}
        return ex.SCM(parents, mech)

    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows1 = []
    for _s in strengths:
        _m = vitamin_world(_s)
        _d = _m.sample(20_000, rng=1)
        _naive = _d["Y"][_d["X"] == 1].mean() - _d["Y"][_d["X"] == 0].mean()
        _bd = ex.backdoor_adjust(_d, "X", "Y", ["Z"], 1) - ex.backdoor_adjust(_d, "X", "Y", ["Z"], 0)
        rows1.append((_s, _naive, _bd, true_effect(_m, "X", "Y", 20_000, 2)))
    _r = np.array(rows1)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.plot(_r[:, 0], _r[:, 1], "o-", label="naive E[Y|X=1] - E[Y|X=0]")
    _ax.plot(_r[:, 0], _r[:, 2], "s-", label="backdoor adjustment for Z")
    _ax.plot(_r[:, 0], _r[:, 3], "k--", label="true effect (simulated do)")
    _ax.axhline(0, color="gray", lw=0.6)
    _ax.set(xlabel="confounding strength s", ylabel="estimated effect of the vitamin")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| s | naive | backdoor | truth |\n|---|---|---|---|\n"
        + "\n".join(f"| {a:.2f} | {b:+.3f} | {c:+.3f} | {d:+.3f} |" for a, b, c, d in rows1))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** derive the naive difference at $s = 1$ by hand: it's
    $-0.2 + \big(P(Z{=}1 \mid X{=}1) - P(Z{=}1 \mid X{=}0)\big)$. why does that formula hold
    here? at which $s$ does the naive estimate change sign, and does the backdoor estimate
    care about $s$ at all?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: adjust for everything?

    a common rule of thumb: "adjust for every variable measured before treatment". this world
    has a confounder $Z$, an M-structure through a pre-treatment variable $B$ (its causes
    $U_1$ and $U_2$ are **unobserved**), a mediator $M$ and a collider $C$:

    $$U_1 \to B \leftarrow U_2,\quad U_1 \to X,\quad U_2 \to Y,\quad Z \to X,\quad Z \to Y,\quad X \to M \to Y,\quad X \to Y,\quad X \to C \leftarrow Y.$$

    the total effect of $X$ on $Y$ is $0.5$ (direct) $+\ 0.6 \times 1.0$ (through $M$)
    $= 1.1$. for each candidate adjustment set your `is_backdoor_adjustment` gives a verdict
    and your `backdoor_adjust` gives an estimate (50,000 draws, mean ± sd over 10 seeds).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which of these sets give an unbiased estimate: {}, {Z}, {Z, B}, {Z, M}, {Z, C}, {Z, B, M, C}? "
        "is adding the pre-treatment variable B to {Z} harmless, helpful or harmful?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    parents2 = {"U1": [], "U2": [], "Z": [], "B": ["U1", "U2"], "X": ["Z", "U1"], "M": ["X"],
                "Y": ["X", "M", "Z", "U2"], "C": ["X", "Y"]}
    mech2 = {"U1": lambda pa, u: bern(0.5, u),
             "U2": lambda pa, u: bern(0.5, u),
             "Z": lambda pa, u: bern(0.5, u),
             "B": lambda pa, u: bern(0.05 + 0.45 * pa["U1"] + 0.45 * pa["U2"], u),
             "X": lambda pa, u: bern(0.1 + 0.4 * pa["Z"] + 0.45 * pa["U1"], u),
             "M": lambda pa, u: bern(0.2 + 0.6 * pa["X"], u),
             "Y": lambda pa, u: 0.5 * pa["X"] + pa["M"] + pa["Z"] + 2.0 * pa["U2"] + u,
             "C": lambda pa, u: (pa["X"] + pa["Y"] + u > 2.5).astype(float)}
    world2 = ex.SCM(parents2, mech2)
    truth2 = true_effect(world2, "X", "Y", 200_000, 0)
    candidates = [[], ["Z"], ["Z", "B"], ["Z", "B", "U1"], ["Z", "M"], ["Z", "C"], ["Z", "B", "M", "C"]]
    rows2 = []
    for _zs in candidates:
        _ok = ex.is_backdoor_adjustment(parents2, "X", "Y", set(_zs))
        _est = []
        for _seed in range(10):
            _d = world2.sample(50_000, rng=100 + _seed)
            _est.append(ex.backdoor_adjust(_d, "X", "Y", _zs, 1) - ex.backdoor_adjust(_d, "X", "Y", _zs, 0))
        rows2.append(("{" + ", ".join(_zs) + "}", _ok, np.mean(_est), np.std(_est, ddof=1)))
    mo.md(
        f"true total effect (simulated do): **{truth2:.3f}**\n\n"
        "| adjustment set | backdoor criterion | estimate (mean ± sd) | bias |\n|---|---|---|---|\n"
        + "\n".join(f"| {name} {'(U1 is unobserved!)' if 'U1' in name else ''} | {'valid' if ok else 'invalid'} "
                    f"| {m:.3f} ± {s:.3f} | {m - truth2:+.3f} |" for name, ok, m, s in rows2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for every invalid set, name the path it leaves open or the path it
    opens. why is $\{Z, B\}$ biased when $B$ was measured *before* treatment? why does
    adding $U_1$ repair it, and why can't you do that in practice? which way does
    adjusting for the mediator move the estimate, and why by about the amount it does?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the front door

    now the confounder is unobserved: $U \to X$, $U \to Y$, and $X$ acts on $Y$ only
    through $M$:

    $$U \sim \mathrm{Bern}(0.5),\ X \sim \mathrm{Bern}(0.2 + 0.6U),\ M \sim \mathrm{Bern}(0.1 + 0.7X),\ Y := 2M + 3U + \delta X + U_Y.$$

    with $\delta = 0$ the true effect is $2 \times 0.7 = 1.4$. no backdoor set exists among
    the observed $\{X, M, Y\}$, but the front-door criterion holds. the lab runs your
    `frontdoor_adjust` at four sample sizes (40 seeds each), then lets you add a direct
    effect $\delta$, which breaks the front-door assumption.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with 500 samples, is the front-door estimate much noisier (sd over seeds) than the naive one? "
        "and once you add a direct effect delta = 1 (true effect 2.4), what will the front-door estimate converge to?"
    )
    p3
    return (p3,)


@app.cell
def _():
    delta = mo.ui.slider(0.0, 2.0, value=0.0, step=0.25, label="direct effect δ of X on Y")
    delta
    return (delta,)


@app.cell
def _(delta, p3):
    ui.gate(p3)

    def fd_world(dlt):
        parents = {"U": [], "X": ["U"], "M": ["X"], "Y": ["M", "U", "X"]}
        mech = {"U": lambda pa, u: bern(0.5, u),
                "X": lambda pa, u: bern(0.2 + 0.6 * pa["U"], u),
                "M": lambda pa, u: bern(0.1 + 0.7 * pa["X"], u),
                "Y": lambda pa, u: 2 * pa["M"] + 3 * pa["U"] + dlt * pa["X"] + u}
        return ex.SCM(parents, mech)

    _w = fd_world(delta.value)
    _truth = true_effect(_w, "X", "Y", 200_000, 0)
    sizes = [500, 2000, 8000, 32000]
    res3 = {}
    for _n in sizes:
        _fd, _nv = [], []
        for _seed in range(40):
            _d = _w.sample(_n, rng=1000 * _n + _seed)
            _fd.append(ex.frontdoor_adjust(_d, "X", "Y", "M", 1) - ex.frontdoor_adjust(_d, "X", "Y", "M", 0))
            _nv.append(_d["Y"][_d["X"] == 1].mean() - _d["Y"][_d["X"] == 0].mean())
        res3[_n] = (np.array(_fd), np.array(_nv))
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _k, (_lab, _mk) in enumerate([("front-door", "o-"), ("naive", "s-")]):
        _mu = [res3[n][_k].mean() for n in sizes]
        _sd = [res3[n][_k].std(ddof=1) for n in sizes]
        _ax.errorbar(sizes, _mu, yerr=_sd, fmt=_mk, capsize=3, label=f"{_lab} (mean ± sd, 40 seeds)")
    _ax.axhline(_truth, color="k", ls="--", label=f"truth {_truth:.2f}")
    _ax.set(xscale="log", xlabel="sample size n", ylabel="estimated effect", title=f"δ = {delta.value:.2f}")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"truth **{_truth:.3f}**.\n\n| n | front-door mean ± sd | naive mean ± sd |\n|---|---|---|\n"
        + "\n".join(f"| {n:,} | {res3[n][0].mean():.3f} ± {res3[n][0].std(ddof=1):.3f} "
                    f"| {res3[n][1].mean():.3f} ± {res3[n][1].std(ddof=1):.3f} |" for n in sizes))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the naive estimate is precise *and* wrong; how would you tell, from its
    error bars alone? the front-door estimate splits the data into cells $(m, x')$: what
    would make it much noisier than here? with $\delta > 0$, the front-door estimate
    still converges, just to the wrong number. which part of the effect does it capture,
    and which assumption in the derivation (on paper 3) does the direct edge break? could
    you detect the violation from the data alone?
    """)
    return


if __name__ == "__main__":
    app.run()
