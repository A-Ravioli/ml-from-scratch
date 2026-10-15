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
    # integrals · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell
def _():
    NS = 2 ** np.arange(1, 12)  # 2, 4, ..., 2048 strips

    def rule_errors(f, a, b, exact):
        """errors of the four rules for every n in NS, and each rule's measured order."""
        rules = {
            "left": lambda n: ex.riemann_sum(f, a, b, n, rule="left"),
            "midpoint": lambda n: ex.riemann_sum(f, a, b, n, rule="midpoint"),
            "trapezoid": lambda n: ex.trapezoid(f, a, b, n),
            "simpson": lambda n: ex.simpson(f, a, b, n),
        }
        out = {}
        for name, rule in rules.items():
            errs = np.array([abs(rule(n) - exact) for n in NS])
            # measure the order only where the error is well above rounding (and skip n = 2)
            keep = (errs > 1e-13) & (NS >= 4)
            order = ex.convergence_order(NS[keep], errs[keep]) if keep.sum() >= 2 else float("nan")
            out[name] = (errs, order)
        return out

    def plot_errors(results, title):
        fig, ax = plt.subplots(figsize=(7, 4))
        for (name, (errs, order)), mk in zip(results.items(), "o^sd"):
            ax.loglog(NS, errs + 1e-17, mk + "-", ms=4, label=f"{name} (measured order {order:.2f})")
        ax.set(xlabel="number of strips n", ylabel="|error|", title=title, ylim=(1e-17, 10))
        ax.legend(fontsize=8)
        return fig

    return NS, plot_errors, rule_errors


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: convergence rates

    we integrate a smooth function with all four of your rules, for $n = 2, 4, \ldots, 2048$
    strips, and measure each rule's order with your `convergence_order`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what order (slope) will each rule show: left, midpoint, trapezoid, Simpson? "
        "will Simpson's error keep falling all the way to n = 2048?"
    )
    p1
    return (p1,)


@app.cell
def _():
    smooth = mo.ui.dropdown(options=["e^x on [0, 1]", "1 / (1 + x^2) on [0, 1]"],
                            value="e^x on [0, 1]", label="function")
    smooth
    return (smooth,)


@app.cell
def _(NS, p1, plot_errors, rule_errors, smooth):
    ui.gate(p1)
    _f, _exact = {
        "e^x on [0, 1]": (np.exp, np.e - 1),
        "1 / (1 + x^2) on [0, 1]": (lambda x: 1 / (1 + x**2), np.pi / 4),
    }[smooth.value]
    _res = rule_errors(_f, 0.0, 1.0, _exact)
    _simp = _res["simpson"][0]
    mo.vstack([plot_errors(_res, smooth.value), mo.md(
        "measured orders: " + ", ".join(f"{k} **{v[1]:.2f}**" for k, v in _res.items())
        + f". Simpson's smallest error: **{_simp.min():.1e}** at n = {NS[int(np.argmin(_simp))]}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** match each measured order with the lesson's error formulas. why are the
    midpoint and trapezoid lines parallel, with the midpoint line lower? why does Simpson's
    line stop falling and turn ragged near $10^{-15}$–$10^{-16}$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a function that repeats itself

    $f(x) = e^{\sin x}$ repeats every $2\pi$. we integrate it over exactly one period,
    $[0, 2\pi]$. (the exact answer is $2\pi I_0(1) = 7.95492652\ldots$, where $I_0$ is a
    Bessel function; `np.i0` computes it.)
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which rule wins here, and by how much? how many strips does the trapezoid rule "
        "need for 10 correct digits? and Simpson?"
    )
    p2
    return (p2,)


@app.cell
def _(NS, p2, plot_errors, rule_errors):
    ui.gate(p2)
    _res = rule_errors(lambda x: np.exp(np.sin(x)), 0.0, 2 * np.pi, 2 * np.pi * np.i0(1.0))
    _need = {k: next((int(n) for n, e in zip(NS, v[0]) if e < 1e-10), None) for k, v in _res.items()}
    _rows = "\n".join(f"| {k} | {v[0][2]:.1e} | {_need[k]} |" for k, v in _res.items())
    mo.vstack([plot_errors(_res, "e^(sin x) over one full period [0, 2π]"),
               mo.md("| rule | error at n = 8 | strips for error < 1e-10 |\n|---|---|---|\n" + _rows)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why are the left-rectangle and trapezoid errors *identical* here? the
    trapezoid rule's error formula says "$\propto h^2$". why is it so much better than that
    here? (hint: the formula's derivation adds up the strip errors. for a periodic function
    over a full period, what happens to the error terms at the two ends?) why does Simpson's
    rule, the "better" method, lose?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: a function with an infinitely steep start

    $\sqrt x$ on $[0, 1]$ (exact area $\frac23$) is continuous, but its slope
    $\frac{1}{2\sqrt x}$ is infinite at $x = 0$, so its derivatives don't exist there.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what orders will the four rules show on sqrt(x)? does Simpson keep its big lead "
        "over the trapezoid rule?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, plot_errors, rule_errors):
    ui.gate(p3)
    _res = rule_errors(np.sqrt, 0.0, 1.0, 2.0 / 3.0)
    _t, _s = _res["trapezoid"][0][-1], _res["simpson"][0][-1]
    mo.vstack([plot_errors(_res, "sqrt(x) on [0, 1]"), mo.md(
        "measured orders: " + ", ".join(f"{k} **{v[1]:.2f}**" for k, v in _res.items())
        + f". at n = 2048: trapezoid error {_t:.1e}, Simpson error {_s:.1e} "
        f"(Simpson better by a factor of only **{_t / _s:.1f}**).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the error formulas contain $f''$ and $f''''$. what happens to them near
    $x = 0$ for $\sqrt x$? which strip do you think contributes most of the error? how could
    you rescue the accuracy (think about where to put the strips, or about substituting
    $x = t^2$)?
    """)
    return


if __name__ == "__main__":
    app.run()
