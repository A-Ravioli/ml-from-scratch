import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    EPSILONS = [0.2, 0.1, 0.05, 0.02, 0.01]
    DELTA = 0.05
    M_GRID = np.unique(np.round(np.geomspace(2, 4000, 90)).astype(int))  # ~8% apart


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PAC learning · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.

    the setting throughout: $x \sim \text{Uniform}[0, 1]$, target threshold $t^\star = 0.5$,
    your `erm_threshold` as the learner, and $\delta = 0.05$. a "measured" sample complexity is
    the first sample size on a grid (about 8% apart) at which at most 5% of 2,000 simulated
    samples gave an $\varepsilon$-bad threshold.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: theory against simulation, no noise

    for five tolerances $\varepsilon$ we compare four numbers: the sample size measured by
    simulation with your `empirical_sample_complexity`; the exact answer, the smallest $m$ with
    $(1 - \varepsilon)^m \le \delta$; the bound $\ln(1/\delta)/\varepsilon$ from idea 3; and your
    `finite_class_sample_complexity` for two *discretized* threshold classes, a grid of 1,000
    thresholds and every float32 value ($|\mathcal H| = 2^{32}$).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how close will the measured sample complexity be to ln(1/delta)/epsilon? how many times larger is the "
        "finite-class bound for 2^32 float32 thresholds?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rows, _res = [], {}
    for _k, _eps in enumerate(EPSILONS):
        _meas = ex.empirical_sample_complexity(_eps, DELTA, M_GRID, n_trials=2000, rng=100 + _k)
        _exact = math.ceil(math.log(DELTA) / math.log(1 - _eps))
        _bound = math.log(1 / DELTA) / _eps
        _grid = ex.finite_class_sample_complexity(1000, _eps, DELTA)
        _f32 = ex.finite_class_sample_complexity(2**32, _eps, DELTA)
        _res[_eps] = (_meas, _exact, _bound, _grid, _f32)
        _rows.append(f"| {_eps} | {_meas} | {_exact} | {_bound:.1f} | {_grid} | {_f32} | {_f32 / _exact:.1f}× |")
    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _e = np.array(EPSILONS)
    for _j, (_lab, _sty) in enumerate([("measured (your simulation)", "ko"), ("exact", "C0-"),
                                       ("ln(1/δ)/ε", "C0:"), ("finite class, 1,000 thresholds", "C1--"),
                                       ("finite class, 2^32 thresholds", "C3--")]):
        _ax.loglog(_e, [_res[v][_j] for v in EPSILONS], _sty, label=_lab, ms=6)
    _ax.set(xlabel="ε (tolerance)", ylabel="sample size for δ = 0.05", title="realizable thresholds")
    _ax.set_xticks(_e, [str(v) for v in EPSILONS])
    _ax.minorticks_off()
    _ax.invert_xaxis()
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| ε | measured | exact | ln(1/δ)/ε | finite, 1,000 | finite, 2^32 | 2^32 bound / exact |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the bound $\ln(1/\delta)/\varepsilon$ match the exact sample
    complexity so closely here (compare it with on paper 3)? the finite-class bound for $2^{32}$
    thresholds is a *valid* bound for the float32 class, yet it's many times larger: which term
    is responsible, and why does the factor stay the same for every $\varepsilon$? what does that
    say about counting hypotheses as a measure of complexity?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: label noise

    now each label is flipped with probability $\eta$ (slider; 0.1 by default), and failure
    means excess risk above $\varepsilon$. we measure the sample complexity for four tolerances
    and compare it with the agnostic finite-class bound for the 1,000-threshold grid,
    $\ln(2|\mathcal H|/\delta)/(2\varepsilon^2)$. the dashed line on the right has slope $-2$
    (the agnostic rate), the dotted one slope $-1$ (the realizable rate).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 10% label noise and epsilon = 0.02, the agnostic bound says about 13,000 examples. what will the simulation "
        "measure? when epsilon halves, does the measured sample size double or quadruple?"
    )
    eta2 = mo.ui.slider(0.0, 0.4, value=0.1, step=0.05, label="label noise η")
    mo.vstack([p2, eta2])
    return eta2, p2


@app.cell
def _(eta2, p2):
    ui.gate(p2)
    _eps_list = [0.2, 0.1, 0.05, 0.02]
    _grid = np.unique(np.round(np.geomspace(2, 20000, 120)).astype(int))
    _meas = [ex.empirical_sample_complexity(e, DELTA, _grid, noise=eta2.value, n_trials=2000, rng=200 + k)
             for k, e in enumerate(_eps_list)]
    _agn = [ex.finite_class_sample_complexity(1000, e, DELTA, realizable=False) for e in _eps_list]
    _ok = [(e, m) for e, m in zip(_eps_list, _meas) if m is not None]
    _slope = (np.polyfit(np.log([e for e, _ in _ok]), np.log([m for _, m in _ok]), 1)[0]
              if len(_ok) >= 2 else float("nan"))
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    _e = np.array(_eps_list)
    _a1.loglog(_e, _agn, "C1--", label="agnostic bound, 1,000 thresholds")
    _a1.loglog([e for e, _ in _ok], [m for _, m in _ok], "ko-", label=f"measured, η = {eta2.value}")
    _a1.loglog(_e, [math.ceil(math.log(DELTA) / math.log(1 - e)) for e in _eps_list], "C0-", lw=1,
               label="exact, no noise (experiment 1)")
    _a1.set(xlabel="ε", ylabel="sample size for δ = 0.05")
    for _a in (_a1, _a2):
        _a.set_xticks(_e, [str(v) for v in _eps_list])
        _a.minorticks_off()
    _a1.invert_xaxis()
    _a1.legend(fontsize=8)
    if _ok:
        _e0, _m0 = _ok[0]
        _a2.loglog([e for e, _ in _ok], [m for _, m in _ok], "ko-", label="measured")
        _a2.loglog(_e, _m0 * (_e / _e0) ** -2.0, "C1--", label="slope −2 (1/ε²)")
        _a2.loglog(_e, _m0 * (_e / _e0) ** -1.0, "C0:", label="slope −1 (1/ε)")
        _a2.set(xlabel="ε", ylabel="measured sample size", title=f"fitted slope {_slope:.2f}")
        _a2.set_xticks(_e, [str(v) for v in _eps_list])
        _a2.minorticks_off()
        _a2.invert_xaxis()
        _a2.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {e} | {m} | {a} | " + (f"{a / m:.0f}×" if m else "–") + " |"
                      for e, m, a in zip(_eps_list, _meas, _agn))
    mo.vstack([_fig, mo.md(
        "| ε | measured | agnostic bound (1,000 thresholds) | bound / measured |\n|---|---|---|---|\n" + _rows
        + f"\n\nfitted log-log slope of the measured sample complexity against ε: **{_slope:.2f}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the agnostic bound is a valid upper bound, so the measurements must lie
    below it, but by how much, and why does the gap grow as $\varepsilon$ shrinks? the slope of
    the measured curve is close to $-1$, not $-2$. look at the excess risk of a threshold,
    $(1 - 2\eta)|t - t^\star|$: why does a risk that grows *linearly* away from the best
    hypothesis make the problem nearly as easy as the realizable one? what kind of
    distribution would force the $1/\varepsilon^2$ rate? move the slider: what happens as
    $\eta \to 0.5$, and why?
    """)
    return


if __name__ == "__main__":
    app.run()
