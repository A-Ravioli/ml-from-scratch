import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # matrix calculus · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how small should the step $h$ be?

    take the mean logistic loss $f(w)$ on 50 random points in 5-D. your `grad_logistic` gives
    the exact gradient. we compare it with finite differences for step sizes $h$ from $10^{-1}$
    down to $10^{-15}$: your central `numerical_gradient`, and a one-sided forward difference
    $(f(w + he_i) - f(w))/h$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which h gives the most accurate central difference? the most accurate forward "
        "difference? what happens at h = 1e-15?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(50, 5))
    _y = (_rng.random(50) < 0.5).astype(float)
    _w = _rng.normal(size=5)

    def _loss(v):
        _z = _X @ v
        return np.mean(np.logaddexp(0.0, _z) - _y * _z)

    _g = ex.grad_logistic(_X, _y, _w)
    _hs = np.logspace(-1, -15, 57)
    _err_c, _err_f = [], []
    for _h in _hs:
        _gc = ex.numerical_gradient(_loss, _w, h=_h)
        _gf = np.array([(_loss(_w + _h * _e) - _loss(_w)) / _h for _e in np.eye(5)])
        _err_c.append(np.linalg.norm(_gc - _g) / np.linalg.norm(_g) + 1e-17)
        _err_f.append(np.linalg.norm(_gf - _g) / np.linalg.norm(_g) + 1e-17)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.loglog(_hs, _err_c, "o-", ms=3, label="central (your numerical_gradient)")
    _ax.loglog(_hs, _err_f, "s-", ms=3, label="forward")
    _ax.invert_xaxis()
    _ax.set(xlabel="step h (shrinking →)", ylabel="relative error", title="the finite-difference U-curve")
    _ax.legend()
    _best_c, _best_f = _hs[int(np.argmin(_err_c))], _hs[int(np.argmin(_err_f))]
    mo.vstack([_fig, mo.md(f"best h: central **{_best_c:.0e}** (error {min(_err_c):.1e}), "
                           f"forward **{_best_f:.0e}** (error {min(_err_f):.1e})")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** two errors fight here. truncation error (from Taylor's theorem) shrinks
    with $h$: like $h$ for forward, $h^2$ for central. rounding error *grows* as $h$ shrinks:
    $f$ is only known to about $\varepsilon|f|$, and you divide by $h$. balance the two to
    predict the best $h$ for each method. which slopes do you see on each side of the
    minimum?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: can a gradient check miss a bug?

    a classic bug: writing the gradient of $x^\top Ax$ as $2Ax$. that's right for symmetric
    $A$ and wrong otherwise. we run your `check_gradient` on the buggy version and on your
    correct `grad_quadratic`, at four kinds of test point.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the buggy gradient is checked at (a) A = I, x = ones; (b) A symmetric random; "
        "(c) A random, x = 0; (d) A random, x random. which checks catch the bug?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _n = 5
    _M = _rng.normal(size=(_n, _n))
    _cases = {
        "(a) A = I, x = ones": (np.eye(_n), np.ones(_n)),
        "(b) A symmetric, x random": (_M + _M.T, _rng.normal(size=_n)),
        "(c) A random, x = 0": (_M, np.zeros(_n)),
        "(d) A random, x random": (_M, _rng.normal(size=_n)),
    }
    _rows = []
    for _name, (_A, _x) in _cases.items():
        _f = lambda v, A=_A: v @ A @ v
        _bug = ex.check_gradient(_f, lambda v, A=_A: 2 * A @ v, _x)
        _ok = ex.check_gradient(_f, lambda v, A=_A: ex.grad_quadratic(A, v), _x)
        _caught = "✅ caught" if _bug > 1e-5 else "❌ missed"
        _rows.append(f"| {_name} | {_bug:.1e} | {_caught} | {_ok:.1e} |")
    mo.md("| test point | error of buggy 2Ax | verdict | error of your grad_quadratic |\n"
          "|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each missed case, show that $2Ax = (A + A^\top)x$ there. what rules
    would you adopt for choosing gradient-check inputs from now on?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: what does a numerical gradient cost?

    the gradient of $\|Ax - b\|^2$ for $A$ of shape $(n, n)$, by your `numerical_gradient`
    (2n evaluations of $f$) and by your `grad_least_squares` (one formula). we time both as
    $n$ grows.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how does each time grow with n? extrapolate: how long would a numerical gradient take "
        "for a model with one million parameters (where one loss evaluation takes 10 ms)?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _ns = [25, 50, 100, 200, 400]
    _t_num, _t_ana = [], []
    for _n in _ns:
        _A, _b, _x = _rng.normal(size=(_n, _n)), _rng.normal(size=_n), _rng.normal(size=_n)
        _f = lambda v, A=_A, b=_b: float(np.sum((A @ v - b) ** 2))
        _t0 = time.perf_counter()
        _gn = ex.numerical_gradient(_f, _x)
        _t1 = time.perf_counter()
        for _ in range(20):
            _ga = ex.grad_least_squares(_A, _b, _x)
        _t2 = time.perf_counter()
        _t_num.append(_t1 - _t0)
        _t_ana.append((_t2 - _t1) / 20)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.loglog(_ns, _t_num, "o-", label="numerical_gradient")
    _ax.loglog(_ns, _t_ana, "s-", label="grad_least_squares")
    _ax.set(xlabel="number of parameters n", ylabel="seconds")
    _ax.legend()
    _ratio = _t_num[-1] / _t_ana[-1]
    mo.vstack([_fig, mo.md(f"at n = {_ns[-1]} the numerical gradient is **{_ratio:,.0f}×** slower. "
                           f"for 10⁶ parameters at 10 ms per loss: 2 × 10⁶ × 0.01 s ≈ "
                           f"**{2e6 * 0.01 / 3600:.1f} hours per gradient**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** how does the ratio between the two change as n grows? why is the numerical gradient's cost (number of
    parameters) × (cost of one loss evaluation)? what would you need to get *every* partial
    derivative for roughly the cost of one or two loss evaluations? (chapter 03 builds forward
    mode; chapter 14 builds reverse mode, which is exactly that.)
    """)
    return


if __name__ == "__main__":
    app.run()
