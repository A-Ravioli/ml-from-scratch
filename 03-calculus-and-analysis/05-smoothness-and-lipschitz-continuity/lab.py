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
    # smoothness & lipschitz continuity · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the knife edge at $2/L$

    a logistic-regression loss on 200 synthetic points in 2-D (plus an $\ell_2$ penalty),
    $f(w) = \frac1n\sum_i\log(1 + e^{-y_ix_i^\top w}) + \tfrac\lambda2\|w\|^2$. we estimate its
    smoothness constant $L$ with your `smoothness_estimate` on 300 random points, then run your
    `gradient_descent` for 200 steps with step sizes just below and just above $2/L$, and ask
    your `largest_stable_step` where the threshold really is.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with the estimated L, does gradient descent converge at step 1.9/L? at 2.1/L? at 3/L? "
        "how does the true stability threshold compare with 2/L_est?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(200, 2)) @ np.array([[3.0, 0.0], [1.0, 0.5]])
    _y = np.where(_X @ np.array([1.0, -2.0]) + 0.5 * _rng.normal(size=200) > 0, 1.0, -1.0)
    _lam = 0.1

    def _f(w):
        return np.mean(np.logaddexp(0.0, -_y * (_X @ w))) + 0.5 * _lam * w @ w

    def _g(w):
        s = 1 / (1 + np.exp(_y * (_X @ w)))
        return -(_X.T @ (_y * s)) / len(_y) + _lam * w

    _L = ex.smoothness_estimate(_g, _rng.normal(size=(300, 2)) * 3)
    _w0 = np.array([4.0, 4.0])
    _w_ref = ex.gradient_descent(_g, _w0, 1.0 / _L, 3000)[-1]
    _f_star = _f(_w_ref)  # the minimum value, from a long, safe run
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _rows = []
    for _c in (1.0, 1.9, 2.1, 3.0, 5.0):
        with np.errstate(all="ignore"):
            _ws = ex.gradient_descent(_g, _w0, _c / _L, 200)
        _vals = np.array([_f(w) for w in _ws])
        _ax.semilogy(np.maximum(_vals - _f_star, 1e-16), label=f"step {_c}/L_est")
        _rows.append(f"| {_c}/L | {_vals[-1]:.6f} | {np.linalg.norm(_g(_ws[-1])):.1e} |")
    _ax.set(xlabel="iteration", ylabel="f(w) − f*", title="logistic loss under gradient descent")
    _ax.legend(fontsize=8)
    _steps = np.linspace(0.5, 40, 80) / _L
    _thr = ex.largest_stable_step(_g, _w0, _steps, n_iters=200)
    _H = np.diag([1.0, 1.0]) * _lam + (_X.T * 0.25) @ _X / len(_y)
    _w_star = ex.gradient_descent(_g, _w0, 1.0 / _L, 2000)[-1]
    _L_star = ex.hessian_smoothness(_f, _w_star)
    mo.vstack([_fig, mo.md(
        f"estimated L = **{_L:.3f}** (the global bound λmax(XᵀX)/4n + λ = {np.linalg.eigvalsh(_H).max():.3f}); "
        f"largest stable step found: **{_thr * _L:.1f}/L_est**. the local smoothness at the minimum "
        f"(your `hessian_smoothness`) is **{_L_star:.3f}**, so 2/L_local = **{2 / _L_star * _L:.1f}/L_est**.\n\n"
        "| step | f after 200 steps | ‖∇f‖ |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for a quadratic the threshold is exactly $2/L$. why can gradient descent
    on the logistic loss survive steps well above $2/L_{\text{est}}$? (where along the path
    is the curvature largest, and where does the run spend its time?) what does that tell
    you about $2/L$ as a guarantee versus a prediction?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: estimating $L$ by sampling, in high dimensions

    a quadratic $f(x) = \tfrac12x^\top Ax$ in $d$ dimensions, where $A$ has one large eigenvalue
    (100) and all the others equal to 1, in a random orientation. so $L = 100$ exactly. we
    estimate $L$ two ways: your `smoothness_estimate` on 100 random points (about 5,000 pairs),
    and your `hessian_smoothness` at a single point. then we run gradient descent with
    step $1.9/L$ for each estimate.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "in d = 50 dimensions, how close does the sampled estimate (5,000 pairs!) get to L = 100? "
        "what happens to gradient descent if you trust it?"
    )
    p2
    return (p2,)


@app.cell
def _():
    dim = mo.ui.slider(2, 100, value=50, step=2, label="dimension d")
    dim
    return (dim,)


@app.cell
def _(dim, p2):
    ui.gate(p2)
    _d = dim.value
    _rng = np.random.default_rng(1)
    _Q, _ = np.linalg.qr(_rng.normal(size=(_d, _d)))
    _eigs = np.ones(_d)
    _eigs[0] = 100.0
    _A = _Q @ np.diag(_eigs) @ _Q.T
    _g = lambda v: _A @ v
    _f = lambda v: 0.5 * v @ _A @ v
    _L_sample = ex.smoothness_estimate(_g, _rng.normal(size=(100, _d)))
    _L_hess = ex.hessian_smoothness(_f, _rng.normal(size=_d))
    _x0 = _rng.normal(size=_d)
    _rows = []
    for _name, _Lhat in (("sampled pairs", _L_sample), ("Hessian eigenvalue", _L_hess)):
        with np.errstate(all="ignore"):
            _xs = ex.gradient_descent(_g, _x0, 1.9 / _Lhat, 100)
        _rows.append(f"| {_name} | {_Lhat:.2f} | {1.9 / _Lhat:.4f} | {_f(_xs[-1]):.3e} |")
    _rand_u = _rng.normal(size=(2000, _d))
    _typ = np.linalg.norm(_rand_u @ _A, axis=1) / np.linalg.norm(_rand_u, axis=1)
    mo.md(f"true L = 100, d = {_d}. a random direction u gives ‖Au‖/‖u‖ ≈ **{np.median(_typ):.1f}** (median); "
          f"the best of 2,000 random directions gives {_typ.max():.1f}.\n\n"
          "| estimate from | L̂ | step 1.9/L̂ | f after 100 GD steps (start ≈ "
          f"{_f(_x0):.1f}) |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for a random unit direction $u$, compute $\|Au\|^2 = \sum_i\lambda_i^2(q_i^\top u)^2$
    and its typical size in $d$ dimensions. why does taking the max over thousands of random
    pairs barely help? why does the Hessian's eigenvalue get it exactly, from a single point?
    which estimate would you trust for setting a learning rate, and why is an *under*estimate
    of $L$ the dangerous direction?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: Lipschitz but not smooth

    $f(x) = |x_1| + |x_2|$ is Lipschitz (with constant $\sqrt2$), but its gradient jumps at the
    axes, so it has no smoothness constant. its smooth cousin, the Huber function with width
    $\delta$, agrees with it away from the axes and is $(1/\delta)$-smooth. we run your
    `gradient_descent` with the same fixed step on both, and estimate the smoothness of each
    with your `smoothness_estimate` on points near the minimum.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "with a fixed step of 0.15, how close does gradient descent get to the minimum of |x1| + |x2| "
        "after 200 steps? and on the Huber version?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _delta = 0.1
    _g_abs = lambda v: np.sign(v)
    _g_hub = lambda v: np.clip(v / _delta, -1.0, 1.0)
    _x0 = np.array([1.03, -0.71])
    _xa = ex.gradient_descent(_g_abs, _x0, 0.15, 200)
    _xh = ex.gradient_descent(_g_hub, _x0, 0.15, 200)
    _pts = np.random.default_rng(2).uniform(-0.05, 0.05, size=(60, 2))
    _La = ex.smoothness_estimate(_g_abs, _pts)
    _Lh = ex.smoothness_estimate(_g_hub, _pts)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogy(np.linalg.norm(_xa, axis=1) + 1e-17, label="|x1| + |x2|")
    _ax.semilogy(np.linalg.norm(_xh, axis=1) + 1e-17, label=f"Huber, δ = {_delta}")
    _ax.set(xlabel="iteration", ylabel="distance to the minimum", title="fixed step 0.15")
    _ax.legend()
    mo.vstack([_fig, mo.md(
        f"after 200 steps: distance **{np.linalg.norm(_xa[-1]):.3f}** for |x|, **{np.linalg.norm(_xh[-1]):.1e}** for Huber. "
        f"smoothness estimated near 0: |x| **{_La:.1f}** (it grows as the points get closer), Huber **{_Lh:.2f}** "
        f"(theory: 1/δ = {1 / _delta:.0f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** near the minimum, what is the gradient of $|x_1| + |x_2|$, and what does a
    fixed step do with it? why does the estimated smoothness constant blow up as the sample
    points get closer to the axes? the Huber function's step limit $2/L = 2\delta$: is a step
    of 0.15 inside it? what would you need to change to make gradient descent converge on the
    non-smooth function?
    """)
    return


if __name__ == "__main__":
    app.run()
