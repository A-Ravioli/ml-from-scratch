import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.autodiff import numerical_hessian
    from mlfs.numgrad import numerical_gradient

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # taylor approximation · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: more terms, better approximation?

    $f(x) = 1/(1 + x^2)$ is as smooth as a function can be: it has derivatives of every order
    at every real $x$, all bounded, and no kinks or poles on the real line. we evaluate its
    Taylor polynomials around a center $x_0$ with your `taylor_poly_1d`, using the exact
    derivatives $f^{(k)}(x) = (-1)^k k!\,\operatorname{Im}\big[(x - i)^{-(k+1)}\big]$, and
    compare with $e^x$ around the same center. move the slider to add terms.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "around x0 = 0, does the Taylor polynomial of 1/(1 + x^2) get closer to f at x = 1.5 as the "
        "degree grows? does the answer change if you expand around x0 = 1 instead?"
    )
    p1
    return (p1,)


@app.cell
def _():
    degree = mo.ui.slider(2, 40, value=10, step=2, label="degree K")
    center = mo.ui.dropdown(["0", "1"], value="0", label="center x0")
    mo.hstack([degree, center], justify="start")
    return center, degree


@app.cell
def _(center, degree, p1):
    ui.gate(p1)
    _x0, _K = float(center.value), degree.value
    _derivs = [(-1) ** k * math.factorial(k) * ((_x0 - 1j) ** (-(k + 1))).imag for k in range(_K + 1)]
    _exp_derivs = [math.exp(_x0)] * (_K + 1)
    _x = np.linspace(-2.5, 2.5, 501)
    _f = 1 / (1 + _x**2)
    _T = ex.taylor_poly_1d(_derivs, _x0, _x)
    _E = ex.taylor_poly_1d(_exp_derivs, _x0, _x)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _a1.plot(_x, _f, "k", lw=2, label="1/(1+x²)")
    _a1.plot(_x, np.clip(_T, -2, 3), "C3", label=f"Taylor, degree {_K}")
    _R = math.hypot(_x0, 1.0)
    for _edge in (_x0 - _R, _x0 + _R):
        _a1.axvline(_edge, color="gray", ls=":", lw=1)
    _a1.set(ylim=(-0.5, 1.5), title=f"1/(1+x²) around x0 = {_x0:g}")
    _a1.legend(fontsize=8)
    _a2.semilogy(_x, np.abs(_T - _f) + 1e-17, "C3", label="1/(1+x²)")
    _a2.semilogy(_x, np.abs(_E - np.exp(_x)) / np.exp(_x) + 1e-17, "C0", label="exp (relative)")
    _a2.set(ylim=(1e-17, 1e6), title="error of the Taylor polynomial")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    _i = int(np.argmin(np.abs(_x - 1.5)))
    mo.vstack([_fig, mo.md(
        f"at x = 1.5: f = {_f[_i]:.6f}, degree-{_K} Taylor polynomial = **{_T[_i]:.6g}** "
        f"(error {abs(_T[_i] - _f[_i]):.1e}). the dotted lines are at x0 ± {_R:.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** inside the dotted lines the error shrinks as $K$ grows; outside it
    explodes. the distance from $x_0$ to the dotted lines is $\sqrt{x_0^2 + 1}$, the distance
    from $x_0$ to the points $\pm i$ in the complex plane. why would *complex* numbers decide
    where a real Taylor series converges? (look at the size of the derivatives, and at the
    Lagrange bound.) why doesn't $e^x$ have this problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how far can you trust a quadratic model?

    $f(x, y) = (1 - x)^2 + 10(y - x^2)^2$, a gentler Rosenbrock function with a curved
    valley along $y = x^2$. at a center $x_0$ we build your `taylor_model` (order 2), then use
    your `trust_radius` (`tol = 0.25`) along 36 directions to map out the region where the
    model's predicted decrease is within 25% of the actual one. the slider moves the center.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "what shape is the region where the quadratic model is trustworthy: a disk, an ellipse, "
        "something else? is it bigger in the curved valley or on its steep walls?"
    )
    p2
    return (p2,)


@app.cell
def _():
    where = mo.ui.slider(-1.5, 1.5, value=-0.5, step=0.25, label="center x coordinate (y = x² + 0.3)")
    where
    return (where,)


@app.cell
def _(p2, where):
    ui.gate(p2)
    _f = lambda v: (1 - v[0]) ** 2 + 10 * (v[1] - v[0] ** 2) ** 2
    _x0 = np.array([where.value, where.value**2 + 0.3])
    _m = ex.taylor_model(_f, _x0)
    _radii = np.logspace(-3, 0.5, 60)
    _angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    _dirs = np.stack([np.cos(_angles), np.sin(_angles)], axis=1)
    _rs = np.array([ex.trust_radius(_f, _m, _x0, _d, _radii) for _d in _dirs])
    _poly = _x0 + _rs[:, None] * _dirs
    _g = np.linspace(-2, 2, 201)
    _X, _Y = np.meshgrid(_g, np.linspace(-0.5, 3.5, 201))
    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.contour(_X, _Y, (1 - _X) ** 2 + 10 * (_Y - _X**2) ** 2, levels=np.geomspace(0.05, 200, 18),
                colors="lightgray", linewidths=0.7)
    _ax.fill(_poly[:, 0], _poly[:, 1], color="C2", alpha=0.4, label="trust region (tol 0.25)")
    _ax.plot(*_x0, "ko")
    _ax.plot(_g, _g**2, "C0:", lw=1, label="valley floor y = x²")
    _ax.set(aspect="equal", xlim=(-2, 2), ylim=(-0.5, 3.5))
    _ax.legend(fontsize=8, loc="upper center")
    _H = numerical_hessian(_f, _x0)
    _lam, _V = np.linalg.eigh(_H)
    mo.vstack([_fig, mo.md(
        f"center {np.round(_x0, 3)}: trust radius ranges from **{_rs.min():.3f}** to **{_rs.max():.3f}** "
        f"(ratio {(_rs.max() / max(_rs.min(), 1e-12)):.0f}×). Hessian eigenvalues {np.round(_lam, 2)}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the model's error is governed by the third derivatives (the next Taylor
    term), while the predicted decrease is governed by the gradient and Hessian. along which
    directions is the predicted decrease large compared with the cubic error? why does that
    make the region lopsided, and sometimes not even contain the center's neighbourhood
    symmetrically? what would a trust-region optimizer (chapter 06) do with this information?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: where does Newton's method go?

    one Newton step jumps to the stationary point of the quadratic model:
    $x \leftarrow x - H^{-1}\nabla f$. we run 12 Newton steps (gradient and Hessian from your
    numerical tools) from a grid of starting points for $f(x, y) = x^3 - 3x + y^2$, which has a
    minimum at $(1, 0)$ and a saddle at $(-1, 0)$ (topic 02's worked example), and check the
    first step's actual change in $f$ against the change your `quadratic_model` predicted.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "what fraction of starting points in [-2, 2] x [-2, 2] does Newton's method take to the minimum? "
        "does the first step always decrease f?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _f = lambda v: v[0] ** 3 - 3 * v[0] + v[1] ** 2
    _grid = np.linspace(-1.9, 1.9, 20)
    _end = np.zeros((20, 20))
    _ups, _model_ups, _n = 0, 0, 0
    for _i, _yy in enumerate(_grid):
        for _j, _xx in enumerate(_grid):
            _x = np.array([_xx, _yy])
            for _k in range(12):
                _g = numerical_gradient(_f, _x)
                _H = numerical_hessian(_f, _x)
                try:
                    _step = -np.linalg.solve(_H, _g)
                except np.linalg.LinAlgError:
                    break
                if _k == 0:
                    _m = ex.quadratic_model(_x, _f(_x), _g, _H)
                    _model_ups += int(_m(_x + _step) > _m(_x))  # the model itself predicts an increase
                    _ups += int(_f(_x + _step) > _f(_x))
                    _n += 1
                _x = _x + _step
            _end[_i, _j] = 1 if np.allclose(_x, [1, 0], atol=1e-6) else (-1 if np.allclose(_x, [-1, 0], atol=1e-6) else 0)
    _fig, _ax = plt.subplots(figsize=(5, 4.5))
    _ax.imshow(_end, origin="lower", extent=(-2, 2, -2, 2), cmap="coolwarm_r", vmin=-1, vmax=1)
    _ax.plot([1], [0], "k*", ms=12)
    _ax.plot([-1], [0], "kx", ms=10, mew=2)
    _ax.set(title="blue: ends at the minimum · red: ends at the saddle", xlabel="x0", ylabel="y0")
    mo.vstack([_fig, mo.md(
        f"ended at the minimum: **{np.mean(_end == 1):.0%}** · at the saddle: **{np.mean(_end == -1):.0%}** · "
        f"elsewhere: **{np.mean(_end == 0):.0%}**. first steps that *increased* f: **{_ups} of {_n}**; "
        f"first steps where your quadratic model itself predicted an increase: **{_model_ups} of {_n}**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** Newton's method solves $\nabla f = 0$ using the quadratic model. what does
    the model look like when the Hessian has a negative eigenvalue, and where is its
    stationary point? in this example the $x$- and $y$-updates decouple: write down the
    $x$-update and find which starting $x$ lead to which root. what would you change to make
    Newton a *minimizer*? (the capstone does this.)
    """)
    return


if __name__ == "__main__":
    app.run()
