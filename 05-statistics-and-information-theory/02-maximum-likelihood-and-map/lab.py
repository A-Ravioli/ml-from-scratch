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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # maximum likelihood & MAP · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much is a prior worth?

    a coin has true bias $p$ (the slider). you estimate it from $n$ flips with the posterior
    mean of a Beta prior centered at **0.5**, with "strength" $m = a + b$: Beta$(m/2, m/2)$.
    $m = 0$ is the MLE. the prior is wrong unless $p = 0.5$. the plot shows the root mean
    squared error of each estimate (your `beta_bernoulli_posterior` and `beta_mean`, 3,000
    simulated coins per point) as $n$ grows.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with true p = 0.3, at n = 10 flips, which has the smaller error: the MLE, or a prior "
        "centered at the wrong value 0.5 with strength m = 10? at n = 1000?"
    )
    p1
    return (p1,)


@app.cell
def _():
    true_p = mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.3, label="true p", show_value=True)
    true_p
    return (true_p,)


@app.cell
def _(p1, true_p):
    ui.gate(p1)
    _p = true_p.value
    _ns = [1, 3, 10, 30, 100, 300, 1000]
    _ms = [0, 2, 10, 50, 200]
    _rng = np.random.default_rng(0)
    _reps = 3000
    _rmse = {m: [] for m in _ms}
    for _n in _ns:
        _flips = (_rng.random((_reps, _n)) < _p).astype(int)
        for _m in _ms:
            if _m == 0:
                _est = np.array([ex.mle_bernoulli(_f) for _f in _flips])
            else:
                _est = np.array([ex.beta_mean(*ex.beta_bernoulli_posterior(_f, _m / 2, _m / 2)) for _f in _flips])
            _rmse[_m].append(float(np.sqrt(np.mean((_est - _p) ** 2))))
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    for _m in _ms:
        _ax.loglog(_ns, _rmse[_m], "o-", ms=3, label="MLE (m = 0)" if _m == 0 else f"prior at 0.5, m = {_m}")
    _ax.set(xlabel="number of flips n", ylabel="RMSE of the estimate of p", title=f"true p = {_p}")
    _ax.legend(fontsize=8)
    _rows = "\n".join(f"| {_m} | " + " | ".join(f"{_r:.3f}" for _r in _rmse[_m]) + " |" for _m in _ms)
    mo.vstack([_fig, mo.md("| prior strength m | " + " | ".join(f"n = {_n}" for _n in _ns) + " |\n|---|"
                           + "---|" * len(_ns) + "\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the posterior mean is $\frac{a + k}{m + n}$: a weighted average of the
    prior mean (weight $m$) and the MLE (weight $n$). use that to explain when the wrong prior
    helps and when it hurts. roughly at what $n$ does each prior stop helping? what happens
    when you move the slider to $p = 0.5$, or to $p = 0.05$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can maximum likelihood be infinitely wrong?

    a rare event has probability $p = 0.05$ (a click, a typo, a fraud). after $n$ observations
    you predict the probability $q$ that the next one is a 1, and you're scored by the
    **log loss** on fresh data, $-[p\log q + (1 - p)\log(1 - q)]$: the expected number of
    nats of surprise per observation (topic 05 calls it a cross-entropy). the MLE $q = k/n$
    competes with the posterior mean under a uniform prior, $q = (k + 1)/(n + 2)$ (Laplace's
    rule of succession), using your `mle_bernoulli` and `beta_mean`. 4,000 simulated datasets
    per $n$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what's the expected log loss of the MLE's prediction after n = 20 observations? "
        "after n = 100? how does Laplace's rule compare?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _p = 0.05
    _best = -(_p * math.log(_p) + (1 - _p) * math.log(1 - _p))
    _rng = np.random.default_rng(1)
    _rows = []
    for _n in (5, 20, 50, 100, 200, 1000):
        _flips = (_rng.random((4000, _n)) < _p).astype(int)
        _q_mle = np.array([ex.mle_bernoulli(_f) for _f in _flips])
        _q_lap = np.array([ex.beta_mean(*ex.beta_bernoulli_posterior(_f, 1.0, 1.0)) for _f in _flips])
        with np.errstate(divide="ignore", invalid="ignore"):
            _loss_mle = -(_p * np.log(_q_mle) + (1 - _p) * np.log1p(-_q_mle))
        _loss_lap = -(_p * np.log(_q_lap) + (1 - _p) * np.log1p(-_q_lap))
        _inf = np.mean(~np.isfinite(_loss_mle))
        _rows.append(f"| {_n} | {_inf:.3f} | {(1 - _p) ** _n:.3f} | {np.mean(_loss_mle):.4f} | "
                     f"{np.mean(_loss_mle[np.isfinite(_loss_mle)]):.4f} | {np.mean(_loss_lap):.4f} |")
    mo.md(f"the best possible expected log loss (knowing $p$) is the entropy, **{_best:.4f}** nats.\n\n"
          "| n | fraction of datasets where the MLE predicts q = 0 | exact P(no 1s) = 0.95ⁿ | MLE: mean log loss "
          "| MLE: mean over the finite ones | Laplace: mean log loss |\n|---|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the MLE's expected log loss infinite at *every* $n$, even
    $n = 1000$? why is the mean over the "finite" datasets misleading too? what does
    Laplace's $+1/+2$ correspond to as a prior, and why does it cure the problem for any $n$?
    where else in ML have you seen "add a small count before taking a log" (or will you)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: do priors matter in the end?

    three people analyze the same coin (true $p = 0.3$) with very different priors: uniform,
    a confident wrong prior Beta(40, 10) (mean 0.8), and a strange two-humped prior
    (a mixture of Beta(30, 70) and Beta(70, 30), peaks near 0.3 and 0.7). your `grid_posterior` computes each
    posterior, and `credible_interval` its 95% interval, after $n = 0, 10, 100, 1000$ flips.
    the dashed curve is a normal density centered at the MLE with standard deviation
    $\sqrt{\hat p(1 - \hat p)/n}$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "after 1000 flips, how different are the three 95% credible intervals? does the wrong "
        "prior Beta(40, 10) still pull the posterior toward 0.8?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _grid = np.linspace(0.0005, 0.9995, 2000)
    _priors = {
        "uniform": lambda g: np.zeros_like(g),
        "Beta(40, 10)": lambda g: stats.beta(40, 10).logpdf(g),
        "two humps": lambda g: np.logaddexp(stats.beta(30, 70).logpdf(g), stats.beta(70, 30).logpdf(g)),
    }
    _flips = (np.random.default_rng(2).random(1000) < 0.3).astype(int)
    _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.2))
    _rows = []
    for _ax, _n in zip(_axes, (0, 10, 100, 1000)):
        _k = int(_flips[:_n].sum())
        _cells = []
        for _label, _lp in _priors.items():
            _post = ex.grid_posterior(lambda g: _k * np.log(g) + (_n - _k) * np.log1p(-g), _lp, _grid)
            _ax.plot(_grid, _post / (_grid[1] - _grid[0]), lw=1.2, label=_label)
            _lo, _hi = ex.credible_interval(_grid, _post)
            _cells.append(f"({_lo:.3f}, {_hi:.3f})")
        if _n > 0:
            _ph = _k / _n
            _ax.plot(_grid, stats.norm(_ph, math.sqrt(_ph * (1 - _ph) / _n)).pdf(_grid), "k--", lw=0.8)
        _ax.axvline(0.3, color="gray", lw=0.6, ls=":")
        _ax.set(title=f"n = {_n} ({_k} heads)", xlabel="p", yticks=[])
        _rows.append(f"| {_n} | " + " | ".join(_cells) + " |")
    _axes[0].legend(fontsize=7)
    _axes[-1].set_xlim(0.2, 0.4)
    mo.vstack([_fig, mo.md("| n | uniform | Beta(40, 10) | two humps |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do the posteriors forget their priors as $n$ grows, and why do they
    all approach the same *Gaussian*, whatever shape the prior had? (look at what grows with
    $n$ in the log-posterior and what doesn't.) which prior needed the most data to be
    overruled, and how many flips is a Beta(40, 10) prior "worth"? what would happen with a
    prior that puts zero probability on $p < 0.5$?
    """)
    return


if __name__ == "__main__":
    app.run()
