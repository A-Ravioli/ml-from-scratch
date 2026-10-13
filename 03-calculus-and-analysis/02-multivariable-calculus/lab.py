import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
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
    # multivariable calculus · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: gradients and level sets

    $f(x, y) = \sin x \cos y + 0.1(x^2 + y^2)$ on $[-3, 3]^2$. we draw some of its level sets
    (contour lines), then walk along each one: at every point we compare the direction of the
    curve with the direction of steepest ascent from your `steepest_direction`, and record the
    slope it reports.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "what angle does the steepest-ascent direction make with the contour line through the "
        "same point? where the contour lines are packed close together, is the slope large or small?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _f = lambda v: np.sin(v[0]) * np.cos(v[1]) + 0.1 * (v[0] ** 2 + v[1] ** 2)
    _g = np.linspace(-3, 3, 241)
    _X, _Y = np.meshgrid(_g, _g)
    _Z = np.sin(_X) * np.cos(_Y) + 0.1 * (_X**2 + _Y**2)
    _fig, _ax = plt.subplots(figsize=(6, 5.5))
    _cs = _ax.contour(_X, _Y, _Z, levels=np.linspace(-0.6, 1.6, 12), cmap="viridis", linewidths=0.8)
    _cosines, _pts, _arrows = [], [], []
    for _segs in _cs.allsegs:
        for _seg in _segs:
            if len(_seg) < 9:
                continue
            for _k in range(4, len(_seg) - 4, 12):
                _t = _seg[_k + 1] - _seg[_k - 1]  # tangent to the level set
                _t = _t / np.linalg.norm(_t)
                try:
                    _u, _slope = ex.steepest_direction(_f, _seg[_k])
                except ValueError:
                    continue
                _cosines.append(abs(float(_t @ _u)))
                _pts.append(_seg[_k])
                _arrows.append(np.asarray(_u) * _slope)
    _pts, _arrows = np.array(_pts), np.array(_arrows)
    _ax.quiver(_pts[:, 0], _pts[:, 1], _arrows[:, 0], _arrows[:, 1], color="C3", angles="xy",
               scale_units="xy", scale=2.5, width=0.004)
    _ax.set(aspect="equal", title="level sets and slope × steepest-ascent direction")
    _ang = np.degrees(np.arccos(np.clip(_cosines, 0, 1)))
    mo.vstack([_fig, mo.md(
        f"{len(_cosines)} points on the contour lines. angle between the curve and the steepest-ascent "
        f"direction: mean **{_ang.mean():.2f}°**, min **{_ang.min():.2f}°**, max **{_ang.max():.2f}°**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** use the directional derivative to show that the gradient is perpendicular
    to the level set: what is $\nabla f \cdot t$ for a vector $t$ tangent to a curve on which
    $f$ is constant? the angles aren't exactly 90°: where does the small error come from?
    and why do long arrows sit where the contour lines bunch together?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: does "steepest" depend on your units?

    the bowl $f(x, y) = (x - 1)^2 + (y - 1)^2$ has its minimum at $(1, 1)$. now measure $y$ in
    different units: $u = s\,y$ (with $s = 10$, $y$ in meters becomes $u$ in decimeters). in the
    new coordinates the *same* function is $g(x, u) = (x - 1)^2 + (u/s - 1)^2$. from the
    physical point $(-1, -1)$ we ask your `steepest_direction` for the steepest-descent
    direction in each coordinate system, map both back to $(x, y)$, and take 25 gradient-descent
    steps in each system.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "do the two coordinate systems agree on the direction of steepest descent at (-1, -1)? "
        "where does gradient descent in (x, u) coordinates end up after 25 steps?"
    )
    p2
    return (p2,)


@app.cell
def _():
    scale = mo.ui.slider(1, 30, value=10, label="unit scale s (u = s·y)")
    scale
    return (scale,)


@app.cell
def _(p2, scale):
    ui.gate(p2)
    _s = float(scale.value)
    _f = lambda v: (v[0] - 1) ** 2 + (v[1] - 1) ** 2
    _g = lambda w: (w[0] - 1) ** 2 + (w[1] / _s - 1) ** 2
    _start = np.array([-1.0, -1.0])
    _to_phys = lambda w: np.array([w[0], w[1] / _s])
    _u_f, _ = ex.steepest_direction(_f, _start)
    _u_g, _ = ex.steepest_direction(_g, np.array([_start[0], _s * _start[1]]))
    _d_f = -np.asarray(_u_f)
    _d_g = -_to_phys(np.asarray(_u_g))
    _d_g = _d_g / np.linalg.norm(_d_g)
    # Newton's direction -H^{-1} grad in each system, mapped back
    _w0 = np.array([_start[0], _s * _start[1]])
    _n_f = -np.linalg.solve(ex.numerical_hessian(_f, _start), numerical_gradient(_f, _start))
    _n_g = -_to_phys(np.linalg.solve(ex.numerical_hessian(_g, _w0), numerical_gradient(_g, _w0)))
    _paths = {}
    for _name, _fun, _x0, _back in (("GD in (x, y)", _f, _start, lambda w: w),
                                     ("GD in (x, u)", _g, _w0, _to_phys)):
        _w = _x0.copy()
        _trail = [_back(_w)]
        for _ in range(25):
            _w = _w - 0.1 * numerical_gradient(_fun, _w)
            _trail.append(_back(_w))
        _paths[_name] = np.array(_trail)
    _fig, _ax = plt.subplots(figsize=(5.5, 5))
    _gr = np.linspace(-1.5, 1.5, 101)
    _X, _Y = np.meshgrid(_gr, _gr)
    _ax.contour(_X, _Y, (_X - 1) ** 2 + (_Y - 1) ** 2, levels=12, colors="lightgray", linewidths=0.7)
    for _name, _p in _paths.items():
        _ax.plot(_p[:, 0], _p[:, 1], "o-", ms=2.5, label=_name)
    _ax.annotate("", _start + 0.8 * _d_f, _start, arrowprops=dict(color="C0", width=1.5))
    _ax.annotate("", _start + 0.8 * _d_g, _start, arrowprops=dict(color="C1", width=1.5))
    _ax.plot(1, 1, "k*", ms=10)
    _ax.set(aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xlabel="x", ylabel="y (physical)")
    _ax.legend(fontsize=8, loc="lower right")
    _angle = np.degrees(np.arccos(np.clip(_d_f @ _d_g, -1, 1)))
    _newton_angle = np.degrees(np.arccos(np.clip(_n_f @ _n_g / np.linalg.norm(_n_f) / np.linalg.norm(_n_g), -1, 1)))
    mo.vstack([_fig, mo.md(
        f"steepest descent in (x, y): **{np.round(_d_f, 3)}**; in (x, u), mapped back: **{np.round(_d_g, 3)}**: "
        f"**{_angle:.1f}°** apart. after 25 steps: (x, y)-GD at {np.round(_paths['GD in (x, y)'][-1], 3)}, "
        f"(x, u)-GD at {np.round(_paths['GD in (x, u)'][-1], 3)}. "
        f"Newton's direction $-H^{{-1}}\\nabla f$ in the two systems differs by **{_newton_angle:.2e}°**.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with $y = u/s$, use the chain rule to write $\partial g/\partial u$ in terms
    of $\partial f/\partial y$. why does that make the $(x, u)$ gradient, mapped back, lean
    towards $x$? "steepest" means "largest slope per unit step length": which step length
    changed? why is Newton's direction immune?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how big should the Hessian's step be?

    $f(v) = \log\sum_i e^{v_i}$ (logsumexp) in 4-D, whose Hessian is known exactly:
    $\operatorname{diag}(p) - pp^\top$ with $p = \operatorname{softmax}(v)$. we compare it with
    your `numerical_hessian` for steps $h$ from $10^{-1}$ to $10^{-8}$, and with a second
    route: your `numerical_jacobian` applied to the *exact* gradient $p$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the gradient's default step is h = 1e-6. what step gives the most accurate "
        "numerical_hessian, and how many correct digits do you get at h = 1e-6?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _v = np.array([0.3, -1.2, 0.8, 0.1])
    _lse = lambda v: np.max(v) + np.log(np.sum(np.exp(v - np.max(v))))
    _soft = lambda v: np.exp(v - _lse(v))
    _p = _soft(_v)
    _H = np.diag(_p) - np.outer(_p, _p)
    _hs = np.logspace(-1, -8, 29)
    _e_hess = [np.abs(ex.numerical_hessian(_lse, _v, h=_h) - _H).max() / np.abs(_H).max() + 1e-17 for _h in _hs]
    _e_jac = [np.abs(ex.numerical_jacobian(_soft, _v, h=_h) - _H).max() / np.abs(_H).max() + 1e-17 for _h in _hs]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.loglog(_hs, _e_hess, "o-", ms=3, label="numerical_hessian (second differences of f)")
    _ax.loglog(_hs, _e_jac, "s-", ms=3, label="numerical_jacobian of the exact gradient")
    _ax.invert_xaxis()
    _ax.set(xlabel="step h (shrinking →)", ylabel="relative error")
    _ax.legend(fontsize=8)
    _i6 = int(np.argmin(np.abs(_hs - 1e-6)))
    mo.vstack([_fig, mo.md(
        f"best h for second differences: **{_hs[int(np.argmin(_e_hess))]:.0e}** (error {min(_e_hess):.1e}); "
        f"at h = 1e-6 the error is **{_e_hess[_i6]:.1e}**. best h for differencing the exact gradient: "
        f"**{_hs[int(np.argmin(_e_jac))]:.0e}** (error {min(_e_jac):.1e}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for the second difference, truncation error is $O(h^2)$ but rounding error
    is about $\varepsilon|f|/h^2$ (you divide by $h^2$, not $h$). balance the two: where should
    the minimum be, and how small is the error there? why does differencing an *exact*
    gradient do so much better? (topic 04 gives you exact gradients; the capstone uses them.)
    """)
    return


if __name__ == "__main__":
    app.run()
