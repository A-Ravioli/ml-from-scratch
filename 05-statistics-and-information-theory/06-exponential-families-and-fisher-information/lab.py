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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # exponential families & fisher information · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: who reaches the bound?

    the Cramér–Rao bound says no **unbiased** estimator can have variance below
    $1/(n\,I(\theta))$. here are four estimators, each compared with the bound for what it
    estimates, computed from your `fisher_information` and `cramer_rao_bound` (for $\sigma^2$
    the bound is carried over from natural parameters with the delta method, as in the lesson).
    the table shows each one's variance and MSE as a multiple of the bound, over 20,000
    simulated datasets.

    - the Poisson rate $\lambda = 3$, estimated by $\bar x$;
    - a Gaussian variance $\sigma^2 = 2$ (mean unknown), by the unbiased $s^2$ (divide by $n - 1$);
    - the same, by the MLE $\hat\sigma^2$ (divide by $n$);
    - the same, by topic 01's $S^2_{n+1}$ (divide by $n + 1$).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "at n = 10, which of these estimators has variance at the bound, which above, and can any be below it?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _reps = 20_000
    _lam, _s2 = 3.0, 2.0
    _pois, _gauss = ex.poisson_family(), ex.gaussian_family()
    _eta_g = np.array([0.0, -1 / (2 * _s2)])
    _rows = []
    _ratios = {}
    for _n in (3, 10, 30, 100):
        # Poisson: bound for lambda = (d lambda / d eta)^2 * CRB(eta), with lambda = e^eta
        _crb_eta = ex.poisson_family().cramer_rao_bound([math.log(_lam)], _n)[0, 0]
        _crb_lam = _lam**2 * _crb_eta
        _xbar = _rng.poisson(_lam, (_reps, _n)).mean(axis=1)
        # Gaussian: sigma^2 = -1 / (2 eta2), gradient (0, 1 / (2 eta2^2)); delta method on the 2x2 bound
        _gr = np.array([0.0, 1 / (2 * _eta_g[1] ** 2)])
        _crb_s2 = float(_gr @ _gauss.cramer_rao_bound(_eta_g, _n) @ _gr)
        _x = _rng.normal(0.0, math.sqrt(_s2), (_reps, _n))
        _ss = np.sum((_x - _x.mean(axis=1, keepdims=True)) ** 2, axis=1)
        _cases = [("poisson x̄", _xbar, _lam, _crb_lam), ("s² (n−1)", _ss / (_n - 1), _s2, _crb_s2),
                  ("MLE σ̂² (n)", _ss / _n, _s2, _crb_s2), ("S²ₙ₊₁ (n+1)", _ss / (_n + 1), _s2, _crb_s2)]
        _cells = []
        for _name, _est, _true, _crb in _cases:
            _v, _m = _est.var() / _crb, np.mean((_est - _true) ** 2) / _crb
            _ratios.setdefault(_name, []).append(_v)
            _cells.append(f"{_v:.2f} / {_m:.2f}")
        _rows.append(f"| {_n} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    for _name, _r in _ratios.items():
        _ax.plot([3, 10, 30, 100], _r, "o-", label=_name)
    _ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Cramér–Rao bound")
    _ax.set(xscale="log", xlabel="n", ylabel="variance / bound")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("variance / bound and MSE / bound (1.00 = exactly at the bound)\n\n"
                           "| n | poisson x̄ | s² (n−1) | MLE σ̂² (n) | S²ₙ₊₁ (n+1) |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does $\bar x$ sit *exactly* on the bound at every $n$ for the Poisson,
    while $s^2$ only gets there as $n \to \infty$? the MLE $\hat\sigma^2$ and $S^2_{n+1}$ have
    variance (and even MSE) **below** the "lower bound". what assumption of the theorem do they
    break, and what does that tell you about the bound's limits?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the MLE in natural parameters

    now use your generic `fit` on 600 Gaussian datasets ($\mu = 1$, $\sigma^2 = 2$) for each
    sample size, and compare the spread of the fitted natural parameters
    $\hat\eta = (\hat\mu/\hat\sigma^2,\ -1/(2\hat\sigma^2))$ with the Cramér–Rao matrix from
    your `cramer_rao_bound`. the plot shows the fitted $\hat\eta$ for $n = 10$ and $n = 200$
    with the bound's 95% ellipse around the truth.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the MLE is 'asymptotically efficient'. at n = 10, is the variance of eta-hat-2 near the bound, "
        "a bit above it, or far above it? is the MLE of eta unbiased?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _fam = ex.gaussian_family()
    _eta = np.array([1 / 2.0, -1 / (2 * 2.0)])
    _rng = np.random.default_rng(1)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _rows = []
    _t = np.linspace(0, 2 * np.pi, 200)
    for _ax, _n in zip(_axes, (10, 200)):
        _fits = np.array([_fam.fit(_rng.normal(1.0, math.sqrt(2.0), _n)) for _ in range(600)])
        _crb = _fam.cramer_rao_bound(_eta, _n)
        _cov = np.cov(_fits.T)
        _bias = _fits.mean(axis=0) - _eta
        _rows.append(f"| {_n} | {_bias[0]:+.4f} / {_bias[1]:+.4f} | {_cov[0, 0] / _crb[0, 0]:.2f} | {_cov[1, 1] / _crb[1, 1]:.2f} |")
        _L = np.linalg.cholesky(_crb)
        _ell = _eta[:, None] + math.sqrt(5.991) * _L @ np.stack([np.cos(_t), np.sin(_t)])
        _ax.scatter(_fits[:, 0], _fits[:, 1], s=3, alpha=0.4, label="fitted eta-hat")
        _ax.plot(_ell[0], _ell[1], "k-", lw=1.2, label="95% ellipse of the bound")
        _ax.plot(*_eta, "r+", ms=12, mew=2, label="true eta")
        _ax.set(title=f"n = {_n}", xlabel="eta1 = mu / sigma^2", ylabel="eta2 = -1 / (2 sigma^2)")
    _axes[0].legend(fontsize=7)
    mo.vstack([_fig, mo.md("| n | bias of eta-hat (eta1 / eta2) | var(eta1-hat) / bound | var(eta2-hat) / bound |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the MLE of $\eta_2 = -1/(2\sigma^2)$ biased, even though the MLE of
    the mean parameter $E[X^2]$ is unbiased? (think about $1/\hat\sigma^2$ and Jensen's
    inequality.) why is its variance so far above the bound at $n = 10$, and why does the cloud
    shrink onto the ellipse at $n = 200$? what does "asymptotically efficient" promise, and what
    doesn't it?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: Rao–Blackwell, and a bound nobody reaches

    estimate $P(X = 0) = e^{-\lambda}$ from $n = 10$ Poisson draws with $\lambda$ set by the
    slider. three estimators:

    - **naive**: $\mathbb{1}[x_1 = 0]$, unbiased but uses one observation;
    - **Rao–Blackwellized**: $E[\mathbb{1}[x_1 = 0] \mid S] = (1 - 1/n)^S$ with $S = \sum x_i$, the
      sufficient statistic; still unbiased;
    - **plug-in MLE**: $e^{-\bar x}$.

    the bound for $g(\lambda) = e^{-\lambda}$ is $g'(\lambda)^2 \cdot \mathrm{CRB}(\lambda)$, from your
    `cramer_rao_bound`. 50,000 simulated datasets.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how much smaller is the Rao-Blackwellized estimator's variance than the naive one's? does it reach "
        "the Cramér-Rao bound?"
    )
    p3
    return (p3,)


@app.cell
def _():
    lam3 = mo.ui.slider(start=0.5, stop=4.0, step=0.5, value=2.0, label="lambda", show_value=True)
    lam3
    return (lam3,)


@app.cell
def _(lam3, p3):
    ui.gate(p3)
    _lam, _n = lam3.value, 10
    _crb_eta = ex.poisson_family().cramer_rao_bound([math.log(_lam)], _n)[0, 0]
    _crb_g = (math.exp(-_lam) * _lam) ** 2 * _crb_eta  # g = exp(-e^eta): dg/deta = -e^-lambda * lambda
    _x = np.random.default_rng(2).poisson(_lam, (50_000, _n))
    _S = _x.sum(axis=1)
    _target = math.exp(-_lam)
    _ests = {"naive 1[x₁ = 0]": (_x[:, 0] == 0).astype(float),
             "Rao–Blackwell (1 − 1/n)^S": (1 - 1 / _n) ** _S,
             "plug-in MLE e^(−x̄)": np.exp(-_x.mean(axis=1))}
    _rows = [f"| {_k} | {_e.mean() - _target:+.5f} | {_e.var():.3e} | {_e.var() / _crb_g:.3f} | "
             f"{np.mean((_e - _target) ** 2) / _crb_g:.3f} |" for _k, _e in _ests.items()]
    mo.md(f"target $e^{{-\\lambda}}$ = {_target:.4f}, Cramér–Rao bound = {_crb_g:.3e}\n\n"
          "| estimator | bias | variance | variance / bound | MSE / bound |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does conditioning on the sufficient statistic keep the estimator
    unbiased and can only reduce its variance? the Rao–Blackwellized estimator is the best
    unbiased estimator there is (it's a function of a complete sufficient statistic), yet it
    sits above the bound. why can't *any* unbiased estimator of $e^{-\lambda}$ reach it, when
    $\bar x$ reaches the bound for $\lambda$ itself? (which functions of the parameter have an
    efficient estimator in an exponential family?)
    """)
    return


if __name__ == "__main__":
    app.run()
