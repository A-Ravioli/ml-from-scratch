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


@app.function
def simulate(n, gamma, rng):
    """X ~ N(0, I_2); logit e(x) = gamma (x0 - 0.5 x1); y(0) = x0 + x1 + eps; y(1) = y(0) + 2 + x0. ATE = 2."""
    X = rng.normal(size=(n, 2))
    e = 1 / (1 + np.exp(-gamma * (X[:, 0] - 0.5 * X[:, 1])))
    t = (rng.random(n) < e).astype(float)
    y0 = X[:, 0] + X[:, 1] + rng.normal(size=n)
    y = np.where(t == 1, y0 + 2 + X[:, 0], y0)
    return X, t, y


@app.function
def estimates(X, t, y):
    """the four estimators of experiment 1, with estimated propensities and linear outcome models."""
    e = ex.propensity_scores(X, t)
    ec = ex.propensity_scores(X, t, clip=0.05)
    mu0, mu1 = ex.outcome_models(X, y, t)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    return {
        "IPW (Horvitz–Thompson)": ex.ipw_ate(y, t, e),
        "IPW (Hájek)": ex.ipw_ate(y, t, e, normalized=True),
        "IPW, clipped at 0.05": ex.ipw_ate(y, t, ec),
        "AIPW": ex.aipw_ate(y, t, e, mu0, mu1)[0],
    }, w.max() / w.sum(), ex.effective_sample_size(w[t == 1])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # potential outcomes & weighting · lab

    every experiment here runs **your** estimators from `exercise.py` on simulated data
    where you know the ATE. for each one: write down a prediction, lock it in, run it, then
    explain what happened. compare with `solutions/lab_notes.md` afterwards. the whole lab
    takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: when the weights explode

    two covariates, $\operatorname{logit} e(x) = \gamma\,(x_0 - 0.5x_1)$, and a treatment
    effect $2 + x_0$ (so the ATE is 2, but the units most likely to be treated benefit
    most). $\gamma$ controls **overlap**: at $\gamma = 0.5$ everyone has a decent chance of
    either treatment; at $\gamma = 4$ many units are almost certain to get one or the other.
    for each $\gamma$, 200 datasets of $n = 1{,}000$; every estimator uses your
    *estimated* propensity scores and linear outcome models.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as overlap gets worse (gamma from 0.5 to 4), how does the spread (sd) of the Horvitz-Thompson IPW estimate change: "
        "2x, 10x, more? which of the four estimators stays reliable? does clipping at 0.05 help?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    gammas = [0.5, 1.0, 2.0, 3.0, 4.0]
    res1, maxshare, ess = {}, {}, {}
    for _g in gammas:
        _rng = np.random.default_rng(int(10 * _g))
        _runs = [estimates(*simulate(1000, _g, _rng)) for _ in range(200)]
        res1[_g] = {k: np.array([r[0][k] for r in _runs]) for k in _runs[0][0]}
        maxshare[_g] = np.median([r[1] for r in _runs])
        ess[_g] = np.median([r[2] for r in _runs])
    names1 = list(res1[gammas[0]])
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))
    for _k in names1:
        _axes[0].plot(gammas, [res1[g][_k].std(ddof=1) for g in gammas], "o-", label=_k)
    _axes[0].set(yscale="log", xlabel="γ (worse overlap →)", ylabel="sd over 200 datasets", title="spread")
    _axes[0].legend(fontsize=8)
    _axes[1].hist(res1[4.0][names1[0]], bins=40, alpha=0.6, label=names1[0])
    _axes[1].hist(res1[4.0]["AIPW"], bins=40, alpha=0.6, label="AIPW")
    _axes[1].axvline(2, color="k", ls="--", lw=1)
    _axes[1].set(xlabel="estimate", ylabel="datasets", title="γ = 4: 200 estimates of an ATE of 2")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    _rows = []
    for _g in gammas:
        _cells = " | ".join(f"{res1[_g][k].mean() - 2:+.2f} / {res1[_g][k].std(ddof=1):.2f}" for k in names1)
        _rows.append(f"| {_g} | {_cells} | {100 * maxshare[_g]:.1f}% | {ess[_g]:.0f} |")
    mo.vstack([_fig, mo.md(
        "bias / sd of each estimator; the last columns are medians over datasets.\n\n"
        "| γ | " + " | ".join(names1) + " | largest weight's share | ESS of the ~500 treated |\n"
        "|---|" + "---|" * (len(names1) + 2) + "\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does one unit's weight $1/e$ matter so much more at $\gamma = 4$?
    relate the sd to the effective sample size. clipping reduces the spread: what does it
    cost, and why is the cost larger here because the effect is $2 + x_0$? why does AIPW
    suffer less than IPW even though it divides by the same propensity scores?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: doubly robust

    now the truth is nonlinear. $x \sim \mathcal N(0, I_2)$,
    $\operatorname{logit} e(x) = x_0 + x_0^2 - 1$, $y(0) = x_0 + 2x_0^2 + x_1 + \varepsilon$,
    $y(1) = y(0) + 2$. a "right" model gets the features $(x_0, x_1, x_0^2)$, a "wrong" one
    only $(x_0, x_1)$. the lab tries every combination of right and wrong propensity and
    outcome models (100 datasets of $n = 2{,}000$ each).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which estimators are unbiased when ONLY the propensity model is right? when only the outcome model is right? "
        "and when both are wrong, is AIPW better or worse than plain IPW or regression?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(2)
    _acc = {}
    for _ in range(100):
        _X = _rng.normal(size=(2000, 2))
        _e = 1 / (1 + np.exp(-(_X[:, 0] + _X[:, 0] ** 2 - 1)))
        _t = (_rng.random(2000) < _e).astype(float)
        _y = _X[:, 0] + 2 * _X[:, 0] ** 2 + _X[:, 1] + _rng.normal(size=2000) + 2 * _t
        _feats = {"right": np.column_stack([_X, _X[:, 0] ** 2]), "wrong": _X}
        _es = {k: ex.propensity_scores(F, _t) for k, F in _feats.items()}
        _mus = {k: ex.outcome_models(F, _y, _t) for k, F in _feats.items()}
        for _pk in ("right", "wrong"):
            _acc.setdefault(("IPW (Hájek)", _pk, "—"), []).append(ex.ipw_ate(_y, _t, _es[_pk], normalized=True))
            _acc.setdefault(("regression", "—", _pk), []).append(np.mean(_mus[_pk][1] - _mus[_pk][0]))
            for _ok in ("right", "wrong"):
                _acc.setdefault(("AIPW", _pk, _ok), []).append(ex.aipw_ate(_y, _t, _es[_pk], *_mus[_ok])[0])
    rows2 = sorted(_acc.items(), key=lambda kv: (kv[0][0] != "AIPW", kv[0]))
    mo.md(
        "| estimator | propensity model | outcome model | bias | sd |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {k[0]} | {k[1]} | {k[2]} | {np.mean(v) - 2:+.3f} | {np.std(v, ddof=1):.3f} |" for k, v in rows2))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** use the bias decomposition from the lesson (on paper 3) to say why AIPW
    is unbiased when *either* model is right. when both are wrong, the two errors multiply:
    is AIPW then guaranteed to beat the single-model estimators? what does that suggest
    about how much to trust "doubly robust" in practice?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: do the error bars tell the truth?

    your `aipw_ate` returns a standard error. the usual 95% interval is
    $\hat\tau \pm 1.96\,\widehat{\mathrm{se}}$. back in the world of experiment 1, the lab
    checks how often that interval actually contains the true ATE of 2 (400 datasets per
    $\gamma$ and sample size $n$, both models correctly specified).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the AIPW 95% interval should cover the truth 95% of the time. at gamma = 4 (poor overlap), what coverage do you expect with n = 200, and with n = 1,000?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    rows3 = []
    for _n in [200, 1000]:
        for _g in [0.5, 2.0, 4.0]:
            _rng = np.random.default_rng(100 + int(10 * _g) + _n)
            _hits, _ses, _ests = [], [], []
            for _ in range(400):
                _X, _t, _y = simulate(_n, _g, _rng)
                _e = ex.propensity_scores(_X, _t)
                _est, _se = ex.aipw_ate(_y, _t, _e, *ex.outcome_models(_X, _y, _t))
                _hits.append(abs(_est - 2) <= 1.96 * _se)
                _ses.append(_se)
                _ests.append(_est)
            _cov = np.mean(_hits)
            rows3.append((_n, _g, _cov, np.sqrt(_cov * (1 - _cov) / 400), np.median(_ses), np.std(_ests, ddof=1),
                          np.mean(np.abs(np.array(_ests) - 2) > 1)))
    mo.md(
        "| n | γ | coverage of the 95% interval | median reported se | actual sd of the estimates | datasets off by > 1 |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(f"| {n:,} | {g} | {c:.1%} ± {s:.1%} | {m:.3f} | {a:.3f} | {f:.1%} |" for n, g, c, s, m, a, f in rows3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the median reported se with the actual sd across datasets. why
    can the standard error be wrong when a few units carry huge weights (think about what
    the sample standard deviation of heavy-tailed scores estimates, from one dataset)? what
    would you report instead, or change about the estimand, in a study with this overlap?
    """)
    return


if __name__ == "__main__":
    app.run()
