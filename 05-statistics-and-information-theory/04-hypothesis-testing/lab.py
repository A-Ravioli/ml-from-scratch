import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    from statistics import NormalDist

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    N_PERM = 200  # permutations per test in this lab: p-values are multiples of 1/201


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # hypothesis testing · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a p-hacking simulator

    a researcher runs a study of a treatment that does **nothing**: treatment and control are
    the same distribution. they test with your `permutation_test` at $\alpha = 0.05$, and they
    have some freedom in how they analyze the data. each "researcher degree of freedom" below
    sounds reasonable on its own:

    - **many outcomes**: measure several outcome variables and report whichever is significant.
    - **optional stopping**: test after every 10 participants per group, and stop as soon as
      $p \le 0.05$ (up to 100 per group).
    - **outlier rule**: if the result isn't significant, try again after dropping values more
      than 2 standard deviations from their group's mean.

    each row simulates between 200 and 1,000 null studies and counts how many "discover" an effect.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 10 outcomes, what fraction of these null studies report a significant effect? with optional "
        "stopping alone? with all three habits combined?"
    )
    p1
    return (p1,)


@app.cell
def _():
    n_outcomes = mo.ui.slider(start=1, stop=20, step=1, value=10, label="number of outcomes", show_value=True)
    n_outcomes
    return (n_outcomes,)


@app.cell
def _(n_outcomes, p1):
    ui.gate(p1)

    def _p(x, y, rng):
        return ex.permutation_test(x, y, n_permutations=N_PERM, rng=rng)[1]

    def _trim(v):
        return v[np.abs(v - v.mean()) <= 2 * v.std()]

    def _study(rng, outcomes, stopping, outliers):
        # data for the whole study: (outcome, group, participant); stop early if allowed
        _data = rng.normal(size=(outcomes, 2, 100))
        _looks = range(10, 101, 10) if stopping else [30]
        for _n in _looks:
            for _k in range(outcomes):
                _x, _y = _data[_k, 0, :_n], _data[_k, 1, :_n]
                if _p(_x, _y, rng) <= 0.05:
                    return True
                if outliers and _p(_trim(_x), _trim(_y), rng) <= 0.05:
                    return True
        return False

    _K = n_outcomes.value
    _configs = [  # (label, outcomes, optional stopping, outlier rule, number of simulated studies)
        ("honest: 1 outcome, n = 30, no tricks", 1, False, False, 1000),
        (f"{_K} outcomes", _K, False, False, 400),
        ("optional stopping", 1, True, False, 1000),
        ("outlier rule", 1, False, True, 1000),
        (f"all three ({_K} outcomes)", _K, True, True, 200),
    ]
    _rows = []
    for _i, (_label, _k, _s, _o, _reps) in enumerate(_configs):
        _rng = np.random.default_rng(100 + _i)
        _rate = sum(_study(_rng, _k, _s, _o) for _ in range(_reps)) / _reps
        _se = math.sqrt(max(_rate * (1 - _rate), 1e-12) / _reps)
        _rows.append(f"| {_label} | {_rate:.3f} ± {_se:.3f} | {_reps:,} |")
    _theory = 1 - 0.95**_K
    mo.md("| analysis | fraction of null studies with a 'discovery' (± Monte Carlo SE) | studies simulated |\n"
          "|---|---|---|\n" + "\n".join(_rows)
          + f"\n\nfor {_K} independent outcomes, $1 - 0.95^{{{_K}}}$ = **{_theory:.3f}**.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each test, alone, rejects a true null only 5% of the time. why do these
    habits push the false-positive rate so far above 5%? why does optional stopping inflate it
    even though every look uses a valid test? which of the three could Bonferroni or
    Benjamini–Hochberg fix, and which can't they fix? what's the cure for the others?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: power, and the winner's curse

    now the treatment *does* something: it shifts the outcome by $\delta = 0.3$ standard
    deviations. your `rejection_rate` measures the **power** of the permutation test (how
    often it detects the effect) for different group sizes. for the studies that do reach
    significance, the table also reports the *average estimated effect* among them. the
    histogram shows the distribution of p-values with no effect and with an effect.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with 20 per group and a true effect of 0.3, how often is the result significant? among the "
        "significant studies, what's the average estimated effect?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _delta = 0.3
    _rows = []
    for _n in (20, 50, 100, 200):
        _sig_effects = []

        def _sim(rng, n=_n):
            return rng.normal(_delta, 1.0, n), rng.normal(0.0, 1.0, n)

        def _test(x, y, rng, store=_sig_effects):
            _obs, _p = ex.permutation_test(x, y, n_permutations=N_PERM, rng=rng)
            if _p <= 0.05:
                store.append(_obs)
            return _p

        _power = ex.rejection_rate(_test, _sim, 200, 0.05, rng=_n)
        _rows.append(f"| {_n} | {_power:.3f} | {np.mean(_sig_effects):.3f} ({np.mean(_sig_effects) / _delta:.1f}× the truth) |")
    _rng = np.random.default_rng(1)
    _p_null = [ex.permutation_test(_rng.normal(size=30), _rng.normal(size=30), n_permutations=N_PERM, rng=_rng)[1]
               for _ in range(500)]
    _p_alt = [ex.permutation_test(_rng.normal(0.5, 1, 30), _rng.normal(size=30), n_permutations=N_PERM, rng=_rng)[1]
              for _ in range(500)]
    _fig, _ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    _ax[0].hist(_p_null, bins=20, range=(0, 1), color="C0")
    _ax[0].set(title="no effect (n = 30 per group)", xlabel="p-value", ylabel="number of studies")
    _ax[1].hist(_p_alt, bins=20, range=(0, 1), color="C1")
    _ax[1].set(title="effect 0.5 sd (n = 30 per group)", xlabel="p-value")
    for _a in _ax:
        _a.axvline(0.05, color="k", lw=0.8, ls="--")
    mo.vstack([mo.md("| n per group | power | mean estimated effect among significant studies (truth 0.3) |\n"
                     "|---|---|---|\n" + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are p-values *uniformly* distributed when there's no effect (so that
    $p = 0.04$ and $p = 0.84$ are equally likely)? why do significant results from small studies
    overestimate the effect so badly? what does that mean for a literature (or a leaderboard)
    that only reports significant results?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a thousand tests at once

    a screen tests 1,000 hypotheses, of which 50 are real effects (of varying size) and 950
    are nulls, think of 1,000 features checked for association with a label, or 1,000 genes.
    each test is a $z$-test with a normal-approximation p-value (permutation p-values from 200
    shuffles can't go below 1/201, far above Bonferroni's threshold $0.05/1000$). compare
    no correction, your `bonferroni`, and your `benjamini_hochberg`, averaged over 100 screens.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with no correction at 0.05, how many of the 'discoveries' are false? how many of the 50 real "
        "effects does Bonferroni find? and BH at 0.05?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _m, _m1 = 1000, 50
    _nd = NormalDist()
    _res = {"no correction": [], "bonferroni": [], "benjamini-hochberg": []}
    for _ in range(100):
        _effects = np.zeros(_m)
        _effects[:_m1] = _rng.uniform(1.0, 4.0, _m1)  # z-scale effect sizes
        _z = _effects + _rng.normal(size=_m)
        _p = np.array([_nd.cdf(-_v) for _v in _z])  # one-sided p-values
        for _name, _mask in (("no correction", _p <= 0.05), ("bonferroni", ex.bonferroni(_p, 0.05)),
                             ("benjamini-hochberg", ex.benjamini_hochberg(_p, 0.05))):
            _mask = np.asarray(_mask)
            _true, _false = _mask[:_m1].sum(), _mask[_m1:].sum()
            _res[_name].append((_true, _false, _false / max(1, _true + _false), _false > 0))
    _rows = []
    for _name, _v in _res.items():
        _a = np.array(_v, dtype=float)
        _rows.append(f"| {_name} | {_a[:, 0].mean():.1f} of 50 | {_a[:, 1].mean():.1f} | {_a[:, 2].mean():.3f} | {_a[:, 3].mean():.2f} |")
    mo.md("| procedure | real effects found | false discoveries | false discovery proportion | P(at least one false) |\n"
          "|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does each procedure control: the false positive rate per test, the
    family-wise error rate, or the false discovery rate? why is Bonferroni so conservative
    with 1,000 tests, and why does BH find so many more real effects while keeping its
    promise? when would you still want Bonferroni?
    """)
    return


if __name__ == "__main__":
    app.run()
