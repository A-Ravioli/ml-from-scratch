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
    # probability by simulation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the birthday problem

    $k$ people, each with a birthday drawn uniformly from 365 days. what's the probability
    that at least two of them share a birthday? your `mc_estimate` simulates 10,000 rooms
    for the $k$ on the slider, and 2,000 rooms for each $k$ on the curve (with error bars of
    ±2 standard errors), against the exact answer from the lesson's on-paper problem 2.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with k = 23 people, what's the chance of a shared birthday? how many people make "
        "it 90%?"
    )
    p1
    return (p1,)


@app.cell
def _():
    k_people = mo.ui.slider(2, 80, value=23, step=1, label="people in the room, k")
    k_people
    return (k_people,)


@app.cell
def _(k_people, p1):
    ui.gate(p1)

    def _shared(days):
        """1.0 for every room (row) in which some birthday repeats."""
        return np.any(np.diff(np.sort(days, axis=1), axis=1) == 0, axis=1).astype(float)

    def _exact(k):
        return 1 - np.prod((365 - np.arange(k)) / 365)

    def _rooms(k):
        return lambda n, rng: rng.integers(0, 365, size=(n, k))

    _k = k_people.value
    _est, _se = ex.mc_estimate(_shared, _rooms(_k), 10_000, rng=2)
    _ks = np.arange(2, 81, 3)
    _curve = [ex.mc_estimate(_shared, _rooms(k), 2_000, rng=int(k)) for k in _ks]
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.errorbar(_ks, [c[0] for c in _curve], yerr=[2 * c[1] for c in _curve], fmt="o", ms=3,
                 capsize=2, label="your mc_estimate (2,000 rooms each, ±2 se)")
    _ax.plot(np.arange(2, 81), [_exact(k) for k in range(2, 81)], "-", color="gray", label="exact")
    _ax.axvline(_k, color="C1", ls=":")
    _ax.set(xlabel="people in the room k", ylabel="P(shared birthday)")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"k = {_k}: simulated **{_est:.3f} ± {_se:.3f}** (10,000 rooms), exact **{_exact(_k):.3f}**. "
        f"the exact probability first passes 90% at k = {next(k for k in range(2, 366) if _exact(k) > 0.9)}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the probability climb so much faster than intuition suggests?
    (count the *pairs* of people in a room of 23.) is the simulated value within its error
    bars of the exact one? would you have trusted the simulation without the error bar?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does one more digit cost?

    we run your `estimate_pi` 20 times (different seeds) for each number of darts from 10
    to a million, and plot the typical error $|\hat\pi - \pi|$ and your reported standard
    error.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what slope will the error have on a log-log plot? how many darts do you need for "
        "3 correct decimal places (error below 0.0005)?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _ns = np.unique(np.geomspace(10, 1_000_000, 11).astype(int))
    _err, _se = [], []
    for _n in _ns:
        _runs = [ex.estimate_pi(int(_n), rng=1000 * int(_n) + _s) for _s in range(20)]
        _err.append(np.mean([abs(e - np.pi) for e, _ in _runs]))
        _se.append(np.mean([s for _, s in _runs]))
    _slope = np.polyfit(np.log(_ns), np.log(_err), 1)[0]
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.loglog(_ns, _err, "o-", label="average |error| over 20 runs")
    _ax.loglog(_ns, _se, "s--", label="average reported standard error")
    _ax.loglog(_ns, 1.64 / np.sqrt(_ns), ":", color="gray", label=r"$1.64/\sqrt{n}$")
    _ax.axhline(5e-4, color="C3", lw=0.8, label="3 decimal places")
    _ax.set(xlabel="number of darts n", ylabel="error")
    _ax.legend(fontsize=8)
    _need = (1.64 * 0.8 / 5e-4) ** 2  # mean |error| is about 0.8 se for bell-shaped errors
    mo.vstack([_fig, mo.md(
        f"fitted slope: **{_slope:.2f}**. to make the typical error 0.0005 you need about "
        f"**{_need:,.0f}** darts, and each further digit costs **100×** more.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what slope do you see, and how does it follow from $\sigma/\sqrt n$?
    compare with topic 03, where the trapezoid rule's error fell like $1/n^2$. why would
    anyone use Monte Carlo for integrals, then? (think about dimensions.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: do error bars tell the truth?

    the rule of thumb says the interval *estimate ± 1.96 standard errors* contains the true
    value 95% of the time. we test it: estimate a probability $p$ with your `mc_estimate`
    from $n$ simulated trials, build the interval, check whether it contains $p$, and repeat
    2,000 times. we try a common event ($p = 0.5$) and a rare one ($p = 0.01$), with few and
    many trials.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for each of the four cases (p = 0.5 or 0.01, n = 50 or 5000), what fraction of the "
        "'95%' intervals contain the true p?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _rows = []
    for _p in (0.5, 0.01):
        for _n in (50, 5000):
            _hits, _zero = 0, 0
            for _ in range(2000):
                _est, _se = ex.mc_estimate(lambda u, p=_p: (u < p).astype(float),
                                           lambda n, rng: rng.random(n), _n, rng=_rng)
                _hits += abs(_est - _p) <= 1.96 * _se
                _zero += _est == 0
            _rows.append(f"| {_p} | {_n} | **{_hits / 2000:.1%}** | {_zero / 2000:.1%} |")
    mo.md("| true p | trials n | intervals containing p | runs that saw the event zero times |\n"
          "|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** which case breaks, and why? what does `mean_and_se` return when the event
    never happens in the $n$ trials, and what interval does that make? the rule of thumb
    relies on the average having a bell-shaped distribution: when is that a bad
    assumption? (chapter 05 builds intervals that behave better here.)
    """)
    return


if __name__ == "__main__":
    app.run()
