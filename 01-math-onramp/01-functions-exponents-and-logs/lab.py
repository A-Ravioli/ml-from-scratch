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
    # functions, exponents & logs · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how many terms does the series need?

    for each whole number $x$ from $-20$ to $20$, we add more and more terms of your
    `exp_series(x, n_terms)` and record the first `n_terms` whose relative error,
    $|\text{series} - e^x| / e^x$, is below a target. we give up at 300 terms. for comparison,
    the dashed line is what your range-reduced `exp` needs: the series only ever sees
    $|r| \le \frac12$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many terms does 1e-10 accuracy take at x = 1? at x = 20? at x = -20? "
        "and with range reduction (|r| <= 1/2)?"
    )
    p1
    return (p1,)


@app.cell
def _():
    target = mo.ui.dropdown(options={"1e-6": 1e-6, "1e-10": 1e-10, "1e-14": 1e-14},
                            value="1e-10", label="target relative error")
    target
    return (target,)


@app.cell
def _(p1, target):
    ui.gate(p1)

    def _terms_needed(xs, tol, max_terms=300):
        """first n_terms with relative error below tol, for every x (max_terms + 1 = never)."""
        _need = np.full(len(xs), max_terms + 1)
        for _n in range(1, max_terms + 1):
            _err = np.abs(np.asarray(ex.exp_series(xs, _n)) - np.exp(xs)) / np.exp(xs)
            _need = np.where((_need > max_terms) & (_err < tol), _n, _need)
        return _need

    _xs = np.arange(-20, 21).astype(float)
    _tol = target.value
    _raw = _terms_needed(_xs, _tol)
    _reduced = _terms_needed(np.linspace(-0.5, 0.5, 21), _tol).max()
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ok = _raw <= 300
    _ax.plot(_xs[_ok], _raw[_ok], "o", ms=4, label="raw series exp_series(x, n)")
    _ax.plot(_xs[~_ok], np.full((~_ok).sum(), 300), "x", color="C3", ms=6, label="never (gave up at 300)")
    _ax.axhline(_reduced, ls="--", color="C2", label=f"with range reduction: {_reduced} terms")
    _ax.set(xlabel="x", ylabel="terms needed", title=f"terms for relative error < {_tol:g}")
    _ax.legend(fontsize=8)
    _never = _xs[~_ok]
    mo.vstack([_fig, mo.md(
        f"x = 1: **{_raw[21]}** terms · x = 20: **{_raw[40]}** · x = -20: "
        f"**{'never' if _raw[0] > 300 else _raw[0]}** · range-reduced: **{_reduced}**. "
        + (f"the raw series never gets there for x <= {_never.max():g}." if len(_never) else ""))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the number of terms grow with $|x|$? why does the raw series
    *never* reach the target for very negative $x$, no matter how many terms you add? (look
    at the size of the biggest term compared with the answer.) why does range reduction make
    the count small and the same for every $x$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: bisection against Newton

    two ways to compute $\log 10 = 2.302585\ldots$ using your `exp`: your `bisect_inverse`
    on the bracket $[0, 5]$, and your `newton_log_steps` from $x_0 = 2$. we record the error
    of each method's guess after every step.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "after 5 steps, roughly how many correct digits does each method have? "
        "how many steps does each need for 12 digits?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _true = np.log(10.0)
    _seen = []

    def _exp_recorded(x):
        _seen.append(x)
        return float(ex.exp(x))

    ex.bisect_inverse(_exp_recorded, 10.0, 0.0, 5.0, tol=1e-13)
    _bis = np.abs(np.array(_seen[2:]) - _true)  # the first two calls check the bracket ends
    _newt = np.abs(np.asarray(ex.newton_log_steps(10.0, 2.0, 6)) - _true)[1:]
    _fig, _ax = plt.subplots(figsize=(7, 3.8))
    _ax.semilogy(np.arange(1, len(_bis) + 1), _bis + 1e-17, ".-", label="bisection")
    _ax.semilogy(np.arange(1, len(_newt) + 1), _newt + 1e-17, "o-", label="Newton")
    _ax.axhline(1e-12, color="gray", ls=":", lw=0.8)
    _ax.set(xlabel="step", ylabel="error |x - log 10|", ylim=(1e-17, 10))
    _ax.legend()
    _b12 = int(np.argmax(_bis < 1e-12)) + 1
    _n12 = int(np.argmax(_newt < 1e-12)) + 1
    mo.vstack([_fig, mo.md(
        f"after 5 steps: bisection error **{_bis[4]:.1e}**, Newton error **{_newt[4]:.1e}**. "
        f"steps to reach 1e-12: bisection **{_b12}**, Newton **{_n12}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on this log-scale plot, bisection is (roughly) a straight line and Newton
    curves sharply downwards. what does a straight line on a log scale mean? what is its
    slope, in digits per step? why does Newton's curve bend, and what does the lesson's
    $t_{\text{new}} \approx t^2/2$ predict for the step after an error of $10^{-4}$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: who wins, and when?

    an exponential $b^n$ against a power $n^a$. both get astronomically big, so we compare
    their **logs** with your `log`: $n \log b$ against $a \log n$. turn the knobs and find
    the crossover. (below that, a fixed comparison: $\log n$ against $n^{0.1}$.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for which n does 1.01^n overtake n^5? and for which n does n^0.1 overtake log n?"
    )
    p3
    return (p3,)


@app.cell
def _():
    base = mo.ui.slider(1.001, 1.2, value=1.01, step=0.001, label="base b of b^n")
    power = mo.ui.slider(1, 20, value=5, step=1, label="power a of n^a")
    mo.hstack([base, power])
    return base, power


@app.cell
def _(base, p3, power):
    ui.gate(p3)
    def _last_crossing(x, ahead):
        """x where `ahead` switches from False to True for the LAST time (None if it never does)."""
        _up = np.flatnonzero(ahead[1:] & ~ahead[:-1]) + 1
        return x[_up[-1]] if len(_up) else None

    _n = np.geomspace(2, 1e9, 4000)
    _log_n = np.asarray(ex.log(_n))
    _exp_side = _n * float(ex.log(base.value))   # log(b^n)
    _pow_side = power.value * _log_n             # log(n^a)
    _cross = _last_crossing(_n, _exp_side > _pow_side)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _axes[0].semilogx(_n, _exp_side, label=f"log({base.value:g}^n)")
    _axes[0].semilogx(_n, _pow_side, label=f"log(n^{power.value})")
    if _cross is not None:
        _axes[0].axvline(_cross, color="gray", ls=":")
    _axes[0].set(xlabel="n", ylabel="log of the value", ylim=(0, 3 * max(_pow_side[-1], 10)))
    _axes[0].legend(fontsize=8)
    # log n against n^0.1, for n up to 1e30. both are drawn on a log scale, so the crossing is visible.
    # (the curves also touch near n = 3; we want the last crossing, after which n^0.1 stays ahead.)
    _m = np.geomspace(3, 1e30, 4000)
    _lm = np.asarray(ex.log(_m))
    _axes[1].loglog(_m, _lm, label="log n")
    _axes[1].loglog(_m, np.asarray(ex.exp(0.1 * _lm)), label="n^0.1")
    _c2 = _last_crossing(_m, 0.1 * _lm > np.asarray(ex.log(_lm)))
    _axes[1].axvline(_c2, color="gray", ls=":")
    _axes[1].set(xlabel="n", ylabel="value (log scale)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        (f"{base.value:g}^n overtakes n^{power.value} at about **n = {_cross:,.0f}**. "
         if _cross is not None else f"{base.value:g}^n hasn't overtaken n^{power.value} by n = 1e9. ")
        + f"n^0.1 overtakes log n at about **n = {_c2:.1e}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do we compare logs instead of the numbers themselves? (try
    `1.01 ** 100000` in a cell.) how does the crossover move when you make the base closer
    to 1, or the power bigger? if "exponential always beats polynomial", why might an
    algorithm that costs $1.001^n$ steps still be the better choice in practice?
    """)
    return


if __name__ == "__main__":
    app.run()
