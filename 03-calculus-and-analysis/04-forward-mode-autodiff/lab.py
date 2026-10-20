import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.numgrad import numerical_gradient

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # forward-mode autodiff · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what does a forward-mode gradient cost?

    $f(x) = \sum_i \log(1 + x_i^2) + \sum_i x_ix_{i+1}$, a function of $n$ inputs that costs
    $O(n)$ to evaluate. we time four ways of getting information about its derivative as $n$
    grows: your `grad` (all $n$ partials), your `jvp` (one directional derivative), a
    finite-difference gradient (`mlfs.numgrad.numerical_gradient` on plain floats), and the
    hand-derived gradient. finally your `jacobian` of the "opposite" shape: a map with **one**
    input and $n$ outputs.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how does the time of each method grow with n? at n = 320, is your exact forward-mode "
        "grad faster or slower than the finite-difference gradient, and by roughly how much?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)

    def _f(x):
        return np.sum(np.log(1 + x**2)) + np.sum(x[:-1] * x[1:])

    def _g_exact(x):
        g = 2 * x / (1 + x**2)
        g[:-1] += x[1:]
        g[1:] += x[:-1]
        return g

    def _clock(fn, reps=1):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    _ns = [10, 20, 40, 80, 160, 320]
    _rows, _times = [], {k: [] for k in ("grad", "jvp", "finite differences", "hand-derived", "jacobian, 1 input → n outputs")}
    _rng = np.random.default_rng(0)
    for _n in _ns:
        _x = _rng.normal(size=_n)
        _v = _rng.normal(size=_n)
        _t = np.linspace(0, 1, _n)
        _times["grad"].append(_clock(lambda: ex.grad(_f, _x)))
        _times["jvp"].append(_clock(lambda: ex.jvp(_f, _x, _v), 5))
        _times["finite differences"].append(_clock(lambda: numerical_gradient(_f, _x), 3))
        _times["hand-derived"].append(_clock(lambda: _g_exact(_x), 50))
        _times["jacobian, 1 input → n outputs"].append(_clock(lambda: ex.jacobian(lambda s: np.sin(s[0] * _t), np.array([0.3])), 5))
    _err = np.abs(ex.grad(_f, _x) - _g_exact(_x)).max()
    _err_fd = np.abs(numerical_gradient(_f, _x) - _g_exact(_x)).max()
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _k, _ts in _times.items():
        _ax.loglog(_ns, _ts, "o-", ms=3, label=_k)
    _ax.set(xlabel="number of inputs n", ylabel="seconds")
    _ax.legend(fontsize=8)
    _last = {k: v[-1] for k, v in _times.items()}
    mo.vstack([_fig, mo.md(
        f"at n = {_ns[-1]}: grad **{_last['grad'] * 1e3:.0f} ms**, jvp **{_last['jvp'] * 1e3:.2f} ms** "
        f"(grad / jvp = **{_last['grad'] / _last['jvp']:.0f}×**), finite differences "
        f"**{_last['finite differences'] * 1e3:.1f} ms**, hand-derived **{_last['hand-derived'] * 1e6:.0f} µs**. "
        f"max error vs the hand-derived gradient: your grad **{_err:.1e}**, finite differences **{_err_fd:.1e}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does `grad` grow like $n^2$ here while `jvp` grows like $n$? why is
    the exact method *slower* than finite differences on this machine, even though both make
    about $n$ passes? (what does one arithmetic operation on a `Dual` cost compared with one
    on a float inside a NumPy array?) and why is the one-input, $n$-output Jacobian cheap?
    which way of multiplying Jacobians would make a gradient cost one pass?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: autodiff differentiates programs, not formulas

    your `derivative` never sees a formula: it runs code. four programs:

    - **(a) a loop.** `babylonian(a, k)`: $k$ steps of $x \leftarrow \tfrac12(x + a/x)$ from
      $x = 1$, which approximates $\sqrt a$. we differentiate it with respect to $a$ at $a = 2$.
    - **(b) a branch.** `square(x)`: returns `1.0` if `x == 1` else `x * x`. as a function it's
      exactly $x^2$ everywhere.
    - **(c) a rewrite.** `np.sqrt(x * x)`, which equals $|x|$, at $x = 0$.
    - **(d) the same rewrite** at $x = 10^{-200}$.
    - **(e) a derivative inside a derivative.** $\frac{d}{dx}\Big[x\cdot\frac{d}{dy}(x + y)\Big]$ at
      $x = 1$, written as `derivative(lambda x: x * derivative(lambda y: x + y, 1.0), 1.0)`. the
      inner derivative is 1, so the answer is $\frac{d}{dx}x = 1$.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "(a) does the derivative of babylonian(a, k) approach 1/(2√2) as k grows, and how fast? "
        "(b) what does your derivative say for square at x = 1? (c), (d): what do you get for sqrt(x*x)? "
        "(e) does the nested derivative give 1?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _babylonian(a, k):
        x = 1.0 + 0.0 * a  # the starting guess, as a constant of the same type as a
        for _ in range(k):
            x = 0.5 * (x + a / x)
        return x

    _true_v, _true_d = np.sqrt(2.0), 1 / (2 * np.sqrt(2.0))
    _rows = []
    for _k in range(0, 7):
        _v = _babylonian(2.0, _k)
        _d = ex.derivative(lambda a: _babylonian(a, _k), 2.0)
        _rows.append(f"| {_k} | {abs(_v - _true_v):.1e} | {_d:.15f} | {abs(_d - _true_d):.1e} |")
    _square = lambda x: 1.0 if x == 1 else x * x
    with np.errstate(all="ignore"):
        _b = ex.derivative(_square, 1.0)
        _b2 = ex.derivative(_square, 1.0 + 1e-9)
        _c = ex.derivative(lambda x: np.sqrt(x * x), 0.0)
        _dd = ex.derivative(lambda x: np.sqrt(x * x), 1e-200)
    _e = ex.derivative(lambda x: x * ex.derivative(lambda y: x + y, 1.0), 1.0)
    mo.md(
        "**(a)** derivative of the Babylonian loop with respect to $a$ (true value $1/(2\\sqrt2) = 0.353553390593274$):\n\n"
        "| k | value error | derivative | derivative error |\n|---|---|---|---|\n" + "\n".join(_rows) +
        f"\n\n**(b)** `square` at 1: **{_b}**; at $1 + 10^{{-9}}$: **{_b2:.6f}**.  "
        f"**(c)** `sqrt(x*x)` at 0: **{_c}**.  **(d)** at $10^{{-200}}$: **{_dd}**.  "
        f"**(e)** the nested derivative: **{_e}** (the true value is 1)."
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** (a) the loop's *derivative* converges too: at what rate, compared with the
    values? differentiate the update $x \leftarrow \tfrac12(x + a/x)$ by hand to see the
    recursion your Duals are running. (b) which branch did the Dual take, and what is the
    derivative of the constant `1.0`? (c) and (d): trace `sqrt` on `x * x` through your
    rules. what intermediate value breaks? what would you tell someone who writes
    `np.sqrt(np.sum(x**2))` for a norm inside a loss? (e) inside the inner call, both `x` and `y`
    carry a dual part of 1: which infinitesimal does each belong to, and what does `x + y` do with
    them? how does the capstone's `hessian` avoid this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: exact derivatives where finite differences give up

    the logistic map $x \mapsto r\,x(1 - x)$ with $r = 3.9$ is chaotic. let $F_k(x_0)$ be the
    result of $k$ steps from $x_0 = 0.2$. we compute $F_k'(x_0)$ three ways: your `derivative`,
    central finite differences with several step sizes, and the chain rule by hand,
    $F_k'(x_0) = \prod_{i<k} r(1 - 2x_i)$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for k = 40 steps, do finite differences (h = 1e-6, 1e-8, 1e-10) agree with your exact derivative? "
        "to how many digits?"
    )
    p3
    return (p3,)


@app.cell
def _():
    steps = mo.ui.slider(5, 60, value=40, step=5, label="number of steps k")
    steps
    return (steps,)


@app.cell
def _(p3, steps):
    ui.gate(p3)
    _r, _x0, _k = 3.9, 0.2, steps.value

    def _F(x):
        for _ in range(_k):
            x = _r * x * (1 - x)
        return x

    _xs = [_x0]
    for _ in range(_k - 1):
        _xs.append(_r * _xs[-1] * (1 - _xs[-1]))
    _chain = float(np.prod([_r * (1 - 2 * _xi) for _xi in _xs]))
    _dual = ex.derivative(_F, _x0)
    _rows = [f"| your `derivative` (dual numbers) | {_dual:.6e} | {abs(_dual - _chain) / abs(_chain):.1e} |"]
    for _h in (1e-4, 1e-6, 1e-8, 1e-10):
        _fd = (_F(_x0 + _h) - _F(_x0 - _h)) / (2 * _h)
        _rows.append(f"| finite difference, h = {_h:.0e} | {_fd:.6e} | {abs(_fd - _chain) / abs(_chain):.1e} |")
    mo.md(f"$F_{{{_k}}}'(0.2)$ by the chain rule: **{_chain:.6e}**\n\n"
          "| method | derivative | relative error vs chain rule |\n|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the derivative is about $e^{\lambda k}$ for some $\lambda > 0$. what does
    that mean for the interval $[x_0 - h, x_0 + h]$ after $k$ steps, and for the assumption
    behind a finite difference (that $F$ is nearly linear on that interval)? why don't dual
    numbers need that assumption? is the exact derivative *useful* here?
    """)
    return


if __name__ == "__main__":
    app.run()
