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
    # derivatives · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how small should the step be?

    we estimate a derivative we know exactly, using your `derivative` (forward and central)
    and your `second_derivative`, for 57 step sizes from $h = 10^{-1}$ down to $10^{-15}$,
    and plot the error of each.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which h gives the smallest error for the forward difference? for the central "
        "difference? for the second derivative? what happens at h = 1e-15?"
    )
    p1
    return (p1,)


@app.cell
def _():
    func = mo.ui.dropdown(
        options=["sin(x) at x = 1", "exp(x) at x = 2", "x^3 at x = 1.5"],
        value="sin(x) at x = 1", label="function")
    func
    return (func,)


@app.cell
def _(func, p1):
    ui.gate(p1)
    _f, _df, _d2f, _x = {
        "sin(x) at x = 1": (np.sin, np.cos(1.0), -np.sin(1.0), 1.0),
        "exp(x) at x = 2": (np.exp, np.exp(2.0), np.exp(2.0), 2.0),
        "x^3 at x = 1.5": (lambda t: t**3, 3 * 1.5**2, 6 * 1.5, 1.5),
    }[func.value]
    _hs = np.logspace(-1, -15, 57)
    _errs = {
        "forward": [abs(float(ex.derivative(_f, _x, h=_h, method="forward")) - _df) for _h in _hs],
        "central": [abs(float(ex.derivative(_f, _x, h=_h, method="central")) - _df) for _h in _hs],
        "second derivative": [abs(float(ex.second_derivative(_f, _x, h=_h)) - _d2f) for _h in _hs],
    }
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for (_name, _e), _mk in zip(_errs.items(), "osd"):
        _e = np.array(_e) + 1e-17  # exact zeros can't go on a log scale
        _ax.loglog(_hs, _e, _mk + "-", ms=3, label=_name)
        _i = int(np.argmin(_e))
        _rows.append(f"| {_name} | {_hs[_i]:.0e} | {_e[_i]:.1e} | {_e[-1]:.1e} |")
    _ax.invert_xaxis()
    _ax.set(xlabel="step h (shrinking →)", ylabel="absolute error", ylim=(1e-17, 1e10))
    _ax.legend()
    mo.vstack([_fig, mo.md("| method | best h | error at best h | error at h = 1e-15 |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on the right side of each curve (large $h$), what slope do you see, and
    which part of the error is it? on the left side (tiny $h$), why does the error *grow*,
    and why does the second derivative's error grow fastest? use the lesson's
    "truncation + rounding" formulas to predict each best $h$ and compare with the table.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the slope at a corner

    $|x|$ has slope $-1$ to the left of $0$ and $+1$ to the right, with a sharp corner at
    $0$, where it has no derivative at all. what do your finite differences say there? and
    what does the central difference say a tiny distance *away* from the corner?
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "at x = 0, what do the forward, backward and central differences of |x| give? "
        "at x = 0.000003 (with the default h = 1e-5), what does the central difference give?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rows = []
    for _x in (0.0, 3e-6, 2e-5):
        _vals = [float(ex.derivative(np.abs, _x, method=_m)) for _m in ("forward", "backward", "central")]
        _rows.append(f"| {_x:g} | " + " | ".join(f"{_v:+.3f}" for _v in _vals) + " | +1 or undefined |")
    _xs = np.linspace(-4e-5, 4e-5, 401)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    _ax.plot(_xs, np.asarray(ex.derivative(np.abs, _xs)), label="central difference, h = 1e-5")
    _ax.plot(_xs, np.sign(_xs), "--", color="gray", label="true slope of |x|")
    _ax.axvspan(-1e-5, 1e-5, color="C1", alpha=0.15, label="within h of the corner")
    _ax.set(xlabel="x", ylabel="estimated slope")
    _ax.legend(fontsize=8)
    mo.vstack([mo.md("| x | forward | backward | central | true slope |\n|---|---|---|---|---|\n"
                     + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the central difference give exactly $0$ at the corner, a value
    that is *nobody's* slope? why is it wrong at $x = 0.000003$, where $|x|$ is perfectly
    smooth? in ML, the ReLU function $\max(0, x)$ has the same corner. what does this mean
    for checking the derivative of a ReLU network at a point where some input is near $0$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: can a derivative check miss a bug?

    the derivative of $\sin(x^2)$ is $\cos(x^2) \cdot 2x$. we run your `check_derivative`
    on two classic chain-rule bugs, "forgot the inner derivative" ($\cos(x^2)$) and
    "evaluated the outer derivative at $x$" ($\cos(x) \cdot 2x$), and on the derivative your
    `chain_rule` builds, at four different sets of check points.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the checks use the points (a) {0, 1}, (b) {0.5, 1.2533}, (c) {0}, (d) five random "
        "points. which point sets catch which bug?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _f = lambda x: np.sin(x**2)
    _derivs = {
        "forgot inner: cos(x²)": lambda x: np.cos(x**2),
        "outer at x: cos(x)·2x": lambda x: np.cos(x) * 2 * x,
        "your chain_rule": ex.chain_rule(np.sin, np.cos, lambda x: x**2, lambda x: 2 * x),
    }
    _sets = {
        "(a) {0, 1}": [0.0, 1.0],
        "(b) {0.5, √(π/2)}": [0.5, np.sqrt(np.pi / 2)],
        "(c) {0}": [0.0],
        "(d) 5 random points": list(np.random.default_rng(0).uniform(-2, 2, 5)),
    }
    _rows = []
    for _sname, _xs in _sets.items():
        _cells = []
        for _dname, _d in _derivs.items():
            _e = ex.check_derivative(_f, _d, _xs)
            _flag = "" if _dname == "your chain_rule" else (" ✅ caught" if _e > 1e-6 else " ❌ missed")
            _cells.append(f"{_e:.1e}{_flag}")
        _rows.append(f"| {_sname} | " + " | ".join(_cells) + " |")
    mo.md("| check points | " + " | ".join(_derivs) + " |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each "missed", show that the buggy formula and the true derivative
    really are equal at those points. what rules will you follow from now on when you choose
    where to check a derivative?
    """)
    return


if __name__ == "__main__":
    app.run()
