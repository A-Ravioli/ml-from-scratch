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

    # name: (sampler, true mean)
    DISTS = {
        "normal(0, 1)": (lambda s, r: r.normal(size=s), 0.0),
        "exponential(1)": (lambda s, r: r.exponential(1.0, s), 1.0),
        "lognormal(0, 1)": (lambda s, r: np.exp(r.normal(size=s)), math.exp(0.5)),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # confidence intervals & the bootstrap · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: do 95% intervals cover 95% of the time?

    your `coverage` runs three interval methods for the **mean** on 1,000 simulated datasets
    each: the normal interval (`normal_ci`), the percentile bootstrap and BCa (`bootstrap_ci`,
    1,000 resamples). choose the data distribution below. the Monte Carlo standard error of each
    coverage number is under 1 percentage point.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for lognormal data (very right-skewed) with n = 10, what fraction of nominal 95% intervals "
        "contain the true mean, for each method? which method is best, and does the bootstrap fix things?"
    )
    p1
    return (p1,)


@app.cell
def _():
    dist = mo.ui.dropdown(options=list(DISTS), value="lognormal(0, 1)", label="data distribution")
    dist
    return (dist,)


@app.cell
def _(dist, p1):
    ui.gate(p1)
    _sampler, _mu = DISTS[dist.value]
    _methods = {
        "normal": lambda x: ex.normal_ci(x),
        "percentile bootstrap": lambda x, g=np.random.default_rng(0): ex.bootstrap_ci(x, np.mean, n_boot=1000, rng=g),
        "BCa bootstrap": lambda x, g=np.random.default_rng(1): ex.bootstrap_ci(x, np.mean, n_boot=1000, method="bca", rng=g),
    }
    _ns = [10, 30, 100]
    _cov = {m: [] for m in _methods}
    _rows = []
    for _n in _ns:
        _cells = []
        for _k, (_m, _f) in enumerate(_methods.items()):
            _c, _w = ex.coverage(_f, _sampler, _mu, _n, 1000, rng=10 * _n + _k)
            _cov[_m].append(_c)
            _cells.append(f"{_c:.3f} (width {_w:.2f})")
        _rows.append(f"| {_n} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _m in _methods:
        _ax.plot(_ns, _cov[_m], "o-", label=_m)
    _ax.axhline(0.95, color="k", ls="--", lw=0.8, label="nominal 0.95")
    _ax.set(xscale="log", xlabel="n", ylabel="actual coverage", title=f"mean of {dist.value} data", ylim=(0.7, 1.0))
    _ax.set_xticks(_ns, [str(_n) for _n in _ns])
    _ax.legend(fontsize=8)
    # which side do the misses fall on? (normal interval, n = 10)
    _x10 = np.asarray(_sampler((4000, 10), np.random.default_rng(99)))
    _cis = np.array([ex.normal_ci(_row) for _row in _x10])
    _below, _above = np.mean(_cis[:, 1] < _mu), np.mean(_cis[:, 0] > _mu)
    mo.vstack([_fig, mo.md("| n | " + " | ".join(_methods) + " |\n|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\nnormal interval at n = 10: entirely **below** the true mean {_below:.3f} of the time, "
                           f"entirely **above** it {_above:.3f} of the time (nominal: 0.025 each).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does every method undercover for skewed data at small $n$, and why are
    the misses so lopsided (think about which side of the truth a sample mean of lognormal
    data usually falls on, and what its sample standard deviation looks like then)? the
    percentile bootstrap is no better than the normal interval, and a little narrower: why?
    (recall which variance the bootstrap estimates.) BCa barely helps here: why can't any
    method that only sees 10 points fix this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: when the bootstrap fails

    estimate the upper end $\theta = 1$ of Uniform$(0, \theta)$ with the sample **max**
    (topic 01's MLE). your bootstrap gives percentile and BCa intervals for it, and `coverage`
    measures how often they contain $\theta$. the histogram compares the bootstrap
    distribution of the max, from one dataset, with its true sampling distribution.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with n = 50 draws, what's the actual coverage of the 95% bootstrap intervals for theta "
        "based on the max?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _u = lambda s, r: r.random(s)
    _n = 50
    _pct = ex.coverage(lambda x, g=np.random.default_rng(2): ex.bootstrap_ci(x, np.max, n_boot=1000, rng=g),
                       _u, 1.0, _n, 300, rng=3)
    _bca = ex.coverage(lambda x, g=np.random.default_rng(4): ex.bootstrap_ci(x, np.max, n_boot=1000, method="bca", rng=g),
                       _u, 1.0, _n, 300, rng=5)
    _x = np.random.default_rng(6).random(_n)
    _reps = ex.bootstrap_replicates(_x, np.max, 5000, rng=7)
    _true = np.random.default_rng(8).random((5000, _n)).max(axis=1)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _bins = np.linspace(0.85, 1.0, 61)
    _ax.hist(_true, bins=_bins, alpha=0.5, density=True, label="true sampling distribution of the max")
    _ax.hist(_reps, bins=_bins, alpha=0.5, density=True, label="bootstrap distribution (one dataset)")
    _ax.axvline(1.0, color="k", lw=1, ls="--", label="theta = 1")
    _ax.axvline(_x.max(), color="C1", lw=1, label="this dataset's max")
    _ax.set(xlabel="max of 50 draws", yticks=[])
    _ax.legend(fontsize=7, loc="upper left")
    _share = np.mean(_reps == _x.max())
    mo.vstack([_fig, mo.md(
        f"| method | actual coverage | mean width |\n|---|---|---|\n"
        f"| percentile bootstrap | {_pct[0]:.3f} | {_pct[1]:.4f} |\n| BCa bootstrap | {_bca[0]:.3f} | {_bca[1]:.4f} |\n\n"
        f"fraction of bootstrap replicates exactly equal to the sample max: **{_share:.3f}** "
        f"(theory: 1 - (1 - 1/n)^n = {1 - (1 - 1 / _n) ** _n:.3f})")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can no bootstrap interval built from replicates of the max *ever*
    contain $\theta$? what does the spike in the bootstrap histogram mean, and why does the
    bootstrap's core assumption ("the sample is to the population as a resample is to the
    sample") break for the max but not for the mean? how would you build a correct interval
    here? (hint: topic 01 worked out the distribution of $\max_i X_i / \theta$.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: paired data, resampled two ways

    two models are scored on the same 200 test examples (per-example losses, correlated: an
    example that's hard for one is hard for the other). model B is truly better by 0.05 on
    average. you want an interval for the mean difference A − B. **paired**: resample
    examples, keeping each example's two scores together (`bootstrap_ci((a, b), ...)`).
    **unpaired**: resample A's scores and B's scores independently, as if they came from
    different test sets. the slider sets the correlation between the two models' losses.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with correlation 0.9, how much wider is the unpaired interval than the paired one? "
        "what are their actual coverages?"
    )
    p3
    return (p3,)


@app.cell
def _():
    corr = mo.ui.slider(start=0.0, stop=0.95, step=0.05, value=0.9, label="correlation of the losses", show_value=True)
    corr
    return (corr,)


@app.cell
def _(corr, p3):
    ui.gate(p3)
    _rho, _n, _gap = corr.value, 200, 0.05

    def _draw(rng):
        _z = rng.normal(size=(2, _n))
        _a = 1.0 + _z[0]
        _b = 1.0 - _gap + _rho * _z[0] + math.sqrt(1 - _rho**2) * _z[1]
        return _a, _b

    def _paired(a, b, rng):
        return ex.bootstrap_ci((a, b), lambda x, y: np.mean(x - y), n_boot=1000, rng=rng)

    def _unpaired(a, b, rng):
        _d = ex.bootstrap_replicates(a, np.mean, 1000, rng=rng) - ex.bootstrap_replicates(b, np.mean, 1000, rng=rng)
        return tuple(np.quantile(_d, [0.025, 0.975]))

    _rng = np.random.default_rng(9)
    _res = {"paired": [0, 0.0, 0], "unpaired": [0, 0.0, 0]}
    _reps = 300
    for _ in range(_reps):
        _a, _b = _draw(_rng)
        for _name, _f in (("paired", _paired), ("unpaired", _unpaired)):
            _lo, _hi = _f(_a, _b, _rng)
            _res[_name][0] += _lo <= _gap <= _hi
            _res[_name][1] += _hi - _lo
            _res[_name][2] += _lo > 0  # "B is significantly better"
    _rows = "\n".join(f"| {_k} | {_v[0] / _reps:.3f} | {_v[1] / _reps:.4f} | {_v[2] / _reps:.3f} |" for _k, _v in _res.items())
    mo.md(f"correlation {_rho}, {_reps} simulated test sets of {_n} examples:\n\n"
          "| bootstrap | coverage of the true gap 0.05 | mean width | fraction of intervals excluding 0 |\n"
          "|---|---|---|---|\n" + _rows)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** both methods estimate the same difference. why is the unpaired interval so
    much wider when the correlation is high, and what does that do to your ability to tell the
    two models apart? write $\operatorname{Var}(\bar a - \bar b)$ in terms of the two variances
    and the covariance. which method is *wrong* here, and in which direction is it wrong?
    """)
    return


if __name__ == "__main__":
    app.run()
