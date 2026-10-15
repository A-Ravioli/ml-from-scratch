import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def _bern(p):
        return p * (1 - p) * (p**2 + (1 - p) ** 2)  # E|X - p|^3 for Bernoulli(p)

    # name: (sampler, mean, sd, E|X - mu|^3 or None)
    DISTS = {
        "uniform(0, 1)": (lambda s, r: r.random(s), 0.5, math.sqrt(1 / 12), 1 / 32),
        "exponential(1)": (lambda s, r: r.exponential(1.0, s), 1.0, 1.0, 12 / math.e - 2),
        "bernoulli(0.5)": (lambda s, r: (r.random(s) < 0.5).astype(float), 0.5, 0.5, _bern(0.5)),
        "bernoulli(0.01)": (lambda s, r: (r.random(s) < 0.01).astype(float), 0.01, math.sqrt(0.0099), _bern(0.01)),
        "lognormal(0, 1)": (lambda s, r: np.exp(r.normal(size=s)), math.exp(0.5),
                            math.sqrt((math.e - 1) * math.e), None),
        "cauchy (no mean, no variance)": (lambda s, r: r.standard_cauchy(s), 0.0, 1.0, None),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # limit theorems · lab

    every experiment here runs **your** simulator from `exercise.py`. for each one: write down
    a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: do running averages settle?

    six independent paths of 200,000 draws each, with your `running_mean`, for three
    distributions: a normal, a Pareto with tail exponent 1.5 (finite mean 3, infinite
    variance), and a standard Cauchy (a bell-shaped curve centered at 0 with no mean at all).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after 200,000 draws, how close is each path's running average to the center (0, 3, 0)? "
        "will the Cauchy paths agree with each other?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _n = 200_000
    _cases = [("normal(0, 1)", lambda: _rng.normal(size=_n), 0.0),
              ("pareto, tail 1.5 (mean 3)", lambda: 1.0 + _rng.pareto(1.5, _n), 3.0),
              ("cauchy", lambda: _rng.standard_cauchy(_n), 0.0)]
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.4))
    _rows = []
    for (_label, _draw, _center), _ax in zip(_cases, _axes):
        _finals = []
        for _ in range(6):
            _rm = ex.running_mean(_draw())
            _ax.semilogx(np.arange(1, _n + 1), _rm, lw=0.8)
            _finals.append(_rm[-1])
        _ax.axhline(_center, color="k", lw=0.6, ls=":")
        _ax.set(title=_label, xlabel="n")
        _lo, _hi = np.percentile(_finals, [0, 100])
        _ax.set_ylim(_center - 3 * max(1.0, _hi - _lo), _center + 3 * max(1.0, _hi - _lo))
        _rows.append(f"| {_label} | " + ", ".join(f"{_f:+.3f}" for _f in _finals) + " |")
    _axes[0].set_ylabel("running average")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| distribution | final running average of each path |\n|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** look at the *shape* of the Cauchy paths: long flat stretches, then sudden
    jumps. what causes a jump, and why do they never stop? the Pareto paths do approach 3,
    but how does their convergence look different from the normal's, and which theorem still
    holds for them?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how fast does the CLT kick in?

    for each distribution, your `clt_distances` measures the KS distance between the
    standardized mean of $n$ draws and $\mathcal{N}(0, 1)$, with 4,000 repetitions per $n$
    (distances below about 0.015 are indistinguishable from noise at that sample size). the
    Berry–Esseen bound is shown where $\mathbb{E}|X - \mu|^3$ is finite and known. the
    histogram shows one distribution in detail.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the distributions by how many draws their mean needs to look normal. how far from "
        "normal is the mean of 64 bernoulli(0.01) draws? of 64 Cauchy draws?"
    )
    p2
    return (p2,)


@app.cell
def _():
    which = mo.ui.dropdown(options=list(DISTS), value="bernoulli(0.01)", label="histogram of")
    n_hist = mo.ui.slider(steps=[1, 4, 16, 64, 256, 1024], value=64, label="n", show_value=True)
    mo.hstack([which, n_hist])
    return n_hist, which


@app.cell
def _(n_hist, p2, which):
    ui.gate(p2)
    _ns = [1, 4, 16, 64, 256]
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    _rows = []
    for _k, (_label, (_s, _mu, _sd, _rho)) in enumerate(DISTS.items()):
        _d = ex.clt_distances(_s, _mu, _sd, _ns, 4000, rng=_k)
        _a1.loglog(_ns, _d, "o-", ms=3, label=_label)
        _be = "" if _rho is None else " / ".join(f"{_b:.2f}" for _b in ex.berry_esseen_bound(np.array(_ns), _sd, _rho))
        _rows.append(f"| {_label} | " + " | ".join(f"{_x:.3f}" for _x in _d) + f" | {_be} |")
    _a1.axhline(0.87 / math.sqrt(4000), color="gray", ls=":", lw=0.8)
    _a1.set(xlabel="n", ylabel="KS distance to N(0, 1)")
    _a1.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _s, _mu, _sd, _ = DISTS[which.value]
    _n = n_hist.value
    _z = math.sqrt(_n) * (ex.sample_means(_s, _n, 4000, rng=99) - _mu) / _sd
    _z = _z[np.abs(_z) < 8]
    _a2.hist(_z, bins=80, density=True, alpha=0.6, range=(-6, 6))
    _t = np.linspace(-6, 6, 300)
    _a2.plot(_t, stats.norm.pdf(_t), "k-", lw=1)
    _a2.set(title=f"{which.value}, n = {_n}: standardized means", xlabel="z", xlim=(-6, 6))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| distribution | n=1 | n=4 | n=16 | n=64 | n=256 | Berry–Esseen bounds |\n"
                           "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** use $\rho/\sigma^3$ to explain the ranking of the finite-variance
    distributions. why does the mean of 64 bernoulli(0.01) draws look so un-normal (what does
    its histogram consist of)? for the Cauchy, the distance *grows* with $n$: why does the
    $\sqrt n$ scaling make things worse for it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the CLT in the tails

    the CLT says $P(Z_n > t) \to 1 - \Phi(t)$. for $t = 3$ the normal answer is 0.00135, and
    people use it to say things like "a three-sigma event". here $Z_n$ is the standardized mean
    of $n$ exponential draws. we estimate both tails with your `sample_means` (200,000
    repetitions) and compute them exactly (the sum is a Gamma variable).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for the mean of n = 30 exponentials (a common rule of thumb says n ≥ 30 is 'large'), how "
        "far off is the normal approximation for P(Z > 3)? for P(Z < -3)? what n gets both within 10%?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _normal_tail = stats.norm.sf(3.0)
    _rows = []
    _rng = np.random.default_rng(3)
    for _n in (1, 10, 30, 100, 1000, 10_000, 100_000):
        _up = stats.gamma(_n).sf(_n + 3 * math.sqrt(_n))
        _lo = stats.gamma(_n).cdf(_n - 3 * math.sqrt(_n))
        if _n <= 100:
            _z = math.sqrt(_n) * (ex.sample_means(DISTS["exponential(1)"][0], _n, 200_000, rng=_rng) - 1.0)
            _sim = f"{np.mean(_z > 3):.5f} / {np.mean(_z < -3):.5f}"
        else:
            _sim = "(too many draws)"
        _rows.append(f"| {_n:,} | {_up:.5f} ({_up / _normal_tail:.2f}×) | {_lo:.5f} ({_lo / _normal_tail:.2f}×) | {_sim} |")
    mo.md(f"normal approximation for either tail: **{_normal_tail:.5f}**.\n\n"
          "| n | exact P(Z > 3) (ratio to normal) | exact P(Z < -3) (ratio) | simulated, upper / lower |\n"
          "|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** Berry–Esseen bounds the *absolute* error of the CDF by about
    $0.4748 \cdot 2.4/\sqrt n$. why is that guarantee useless for a probability of size 0.001?
    why is the right tail too heavy and the left tail too light for the exponential? what
    does this mean for "p < 0.001" claims computed from normal approximations?
    """)
    return


if __name__ == "__main__":
    app.run()
