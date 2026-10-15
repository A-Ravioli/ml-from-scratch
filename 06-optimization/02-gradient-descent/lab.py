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
    # gradient descent · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the condition number and the zig-zag

    quadratics $f(x) = \tfrac12x^\top Ax$ in 20 dimensions, with eigenvalues spread
    evenly on a log scale from $\mu = 1$ to $L = \kappa$, started from a random point. we run
    your `gradient_descent` with the fixed step $1/L$ and your `steepest_descent_quadratic`
    (exact line search), and count the iterations until $\|\nabla f\| \le 10^{-8}\|\nabla f(x_0)\|$,
    for $\kappa$ from 2 to 1,000. on the left, the first 30 steps of both methods on the 2-D
    quadratic $\tfrac12(x_1^2 + \kappa x_2^2)$ from $x_0 = (1, 1/\kappa)$ (the slider picks $\kappa$).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "going from kappa = 10 to kappa = 1000, by what factor does the iteration count grow, for the fixed "
        "step 1/L and for the exact line search? which of the two needs fewer iterations?"
    )
    p1
    return (p1,)


@app.cell
def _():
    kappa = mo.ui.slider(2, 50, value=10, step=1, label="κ for the path plot")
    kappa
    return (kappa,)


@app.cell
def _(kappa, p1):
    ui.gate(p1)

    def _count_exact(A, x0, tol, cap):
        xs = ex.steepest_descent_quadratic(A, np.zeros(len(A)), x0, cap)
        g = np.linalg.norm(xs @ A, axis=1)
        hit = np.flatnonzero(g <= tol)
        return int(hit[0]) if hit.size else cap

    _rng = np.random.default_rng(0)
    _Q, _ = np.linalg.qr(_rng.normal(size=(20, 20)))
    _z0 = _rng.normal(size=20)
    _kappas = [2, 5, 10, 30, 100, 300, 1000]
    _fixed, _exact = [], []
    for _k in _kappas:
        _A = _Q @ np.diag(np.geomspace(1.0, float(_k), 20)) @ _Q.T
        _f = lambda x, A=_A: float(0.5 * x @ A @ x)
        _g = lambda x, A=_A: A @ x
        _tol = 1e-8 * np.linalg.norm(_A @ _z0)
        _fixed.append(ex.gradient_descent(_f, _g, _z0, lr=1.0 / _k, tol=_tol, max_iters=200_000).n_iters)
        _exact.append(_count_exact(_A, _z0, _tol, 20 * _k + 200))
    _slope_f = np.polyfit(np.log(_kappas[2:]), np.log(_fixed[2:]), 1)[0]
    _slope_e = np.polyfit(np.log(_kappas[2:]), np.log(_exact[2:]), 1)[0]

    _K = float(kappa.value)
    _A = np.diag([1.0, _K])
    _path_fixed = []
    _start = np.array([1.0, 1.0 / _K])
    ex.gradient_descent(lambda x: float(0.5 * x @ _A @ x), lambda x: _A @ x, _start, lr=1.0 / _K,
                        max_iters=30, tol=0.0, callback=lambda x, k: _path_fixed.append(x))
    _path_fixed = np.vstack([_start, _path_fixed])
    _path_exact = ex.steepest_descent_quadratic(_A, np.zeros(2), _start, 30)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    _t = np.linspace(-1.2, 1.2, 300)
    _GX, _GY = np.meshgrid(_t, _t)
    _a1.contour(_GX, _GY, 0.5 * (_GX**2 + _K * _GY**2), levels=np.geomspace(1e-3, 1.0, 12),
                colors="0.75", linewidths=0.7)
    _a1.plot(_path_fixed[:, 0], _path_fixed[:, 1], "o-", ms=3, label="fixed step 1/L")
    _a1.plot(_path_exact[:, 0], _path_exact[:, 1], "s-", ms=3, label="exact line search")
    _a1.set_aspect("equal")
    _a1.set(xlim=(-0.05, 1.05), ylim=(-0.25, 0.25), title=f"first 30 steps, κ = {_K:.0f}")
    _a1.legend(fontsize=8, loc="lower left")
    _a2.loglog(_kappas, _fixed, "o-", label=f"fixed 1/L (slope {_slope_f:.2f})")
    _a2.loglog(_kappas, _exact, "s-", label=f"exact line search (slope {_slope_e:.2f})")
    _a2.set(xlabel="condition number κ", ylabel="iterations to shrink ‖∇f‖ by 1e8", title="20-D, random start")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| κ | fixed 1/L | exact line search |\n|---|---|---|\n"
        + "\n".join(f"| {k} | {a} | {b} |" for k, a, b in zip(_kappas, _fixed, _exact)))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why do the exact-line-search steps come out at right angles, and why
    does that make it crawl along the valley floor? from the log-log slopes, how does the
    iteration count scale with $\kappa$? the fixed step $1/L$ never zig-zags here: why not
    (what does it do to $x_2$ in one step)? which is faster, and does it change the
    $O(\kappa)$ scaling?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: measured rates against the theory

    one quadratic, $A = \operatorname{diag}(1, 20)$ ($\kappa = 20$), started from the worst case
    $x_0 = (1, 1/20)$ for exact line search. four ways of choosing the step: the textbook
    $1/L$, the best fixed step $2/(\mu + L)$, your backtracking (`line_search=True`,
    $\alpha_0 = 1$), and your exact line search. for each we measure the asymptotic
    per-step factor by which $f - f^*$ shrinks (geometric mean over iterations 20–60).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "rank the four step rules from fastest to slowest. the theorem for step 1/L promises f - f* shrinks by at "
        "least 1 - 1/kappa = 0.95 per step: will the measured factor match it?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _A = np.diag([1.0, 20.0])
    _mu, _L = 1.0, 20.0
    _kap = _L / _mu
    _f = lambda x: float(0.5 * x @ _A @ x)
    _g = lambda x: _A @ x
    _x0 = np.array([1.0, 1.0 / 20])
    _runs = {
        "fixed 1/L": ex.gradient_descent(_f, _g, _x0, lr=1 / _L, max_iters=60, tol=0.0).history,
        "fixed 2/(μ+L)": ex.gradient_descent(_f, _g, _x0, lr=2 / (_mu + _L), max_iters=60, tol=0.0).history,
        "backtracking, α₀ = 1": ex.gradient_descent(_f, _g, _x0, lr=1.0, line_search=True, max_iters=60, tol=0.0).history,
        "exact line search": [_f(x) for x in ex.steepest_descent_quadratic(_A, np.zeros(2), _x0, 60)],
    }
    _theory = {
        "fixed 1/L": f"≤ 1 − 1/κ = {1 - 1 / _kap:.4f} (theorem); (1 − 1/κ)² = {(1 - 1 / _kap) ** 2:.4f} (quadratic)",
        "fixed 2/(μ+L)": f"((κ−1)/(κ+1))² = {((_kap - 1) / (_kap + 1)) ** 2:.4f}",
        "backtracking, α₀ = 1": "≤ 1 − 2ρ(1−c)μ/L ≈ 0.95 (theorem)",
        "exact line search": f"≤ ((κ−1)/(κ+1))² = {((_kap - 1) / (_kap + 1)) ** 2:.4f} (Kantorovich)",
    }
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _rows = []
    for _name, _h in _runs.items():
        _h = np.maximum(np.array(_h), 1e-300)
        _ax.semilogy(_h, ls="--" if _name.startswith("exact") else "-", lw=2.5 if _name.startswith("fixed 2") else 1.5,
                     label=_name)
        _rate = np.exp(np.mean(np.diff(np.log(_h[20:61]))))
        _rows.append(f"| {_name} | **{_rate:.4f}** | {_theory[_name]} |")
    _ax.set(xlabel="iteration", ylabel="f − f*", title="κ = 20, worst-case start for exact line search")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| step rule | measured factor per step | theory |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the measured factor for $1/L$ the *square* of the theorem's
    $1 - 1/\kappa$? (what does the step do to each eigen-coordinate, and how does $f$ depend
    on the coordinates?) why are the best fixed step and the exact line search tied? what
    did backtracking do to beat them on this start? look at its curve: is its rate constant?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: when the minimum is at infinity

    logistic regression (no intercept, no penalty) on 8 points that a line through the
    origin separates perfectly: positives at $(1, \pm0.5)$, $(2, \pm1.5)$, $(4, 3)$, $(3, 2.5)$
    and negatives at their mirror images. the widest-margin separator through the origin
    points along $(1, 0)$, set by the two points at $(1, \pm0.5)$. we run your
    `gradient_descent` with a fixed step for 100,000 iterations and record the loss, the
    weight norm $\|w_k\|$ and the angle between $w_k$ and $(1, 0)$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the loss has no minimizer (it keeps falling as ||w|| grows). after 100,000 steps: is the loss tiny? "
        "how big is ||w||? does the DIRECTION of w settle, and where?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _P = np.array([[1.0, 0.5], [1.0, -0.5], [2.0, 1.5], [2.0, -1.5], [4.0, 3.0], [3.0, 2.5]])
    _X = np.vstack([_P, -_P])
    _y = np.r_[np.ones(6), -np.ones(6)]

    def _loss(w):
        return float(np.mean(np.logaddexp(0.0, -_y * (_X @ w))))

    def _grad(w):
        m = _y * (_X @ w)
        return -(_X.T @ (_y * 0.5 * (1 - np.tanh(0.5 * m)))) / len(_y)

    _L = np.linalg.eigvalsh(_X.T @ _X / len(_y)).max() / 4
    _marks = set(np.unique(np.geomspace(1, 100_000, 60).astype(int)).tolist())
    _rec = []

    def _cb(w, k):
        if k in _marks:
            _rec.append((k, _loss(w), np.linalg.norm(w), np.degrees(np.arctan2(w[1], w[0]))))

    _res = ex.gradient_descent(_loss, _grad, np.zeros(2), lr=1.0 / _L, max_iters=100_000, tol=0.0, callback=_cb)
    _rec = np.array(_rec)
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _axes[0].loglog(_rec[:, 0], _rec[:, 1])
    _axes[0].set(xlabel="iteration k", ylabel="loss", title="loss → 0 like 1/k")
    _axes[1].semilogx(_rec[:, 0], _rec[:, 2])
    _axes[1].set(xlabel="iteration k (log scale)", ylabel="‖w_k‖", title="‖w‖ grows like log k")
    _axes[2].semilogx(_rec[:, 0], _rec[:, 3])
    _axes[2].axhline(0, color="k", lw=0.8, ls="--")
    _axes[2].set(xlabel="iteration k (log scale)", ylabel="angle of w_k (degrees)", title="direction → max margin")
    _fig.tight_layout()
    _pick = [np.argmin(np.abs(_rec[:, 0] - k)) for k in (100, 1_000, 10_000, 100_000)]
    mo.vstack([_fig, mo.md(
        f"step 1/L = {1 / _L:.2f}. converged: **{_res.converged}** (the gradient never reaches 0).\n\n"
        "| k | loss | ‖w‖ | angle to (1, 0) |\n|---|---|---|---|\n"
        + "\n".join(f"| {int(_rec[i, 0]):,} | {_rec[i, 1]:.2e} | {_rec[i, 2]:.2f} | {_rec[i, 3]:.2f}° |" for i in _pick))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the logistic loss have no minimizer on separable data, and why
    doesn't that make gradient descent diverge? as $\|w\|$ grows, which training points
    still contribute to the gradient (look at the weights $\sigma(-y_iw^\top x_i)$)? why
    does that steer the direction towards the maximum-margin separator, and why so slowly?
    what would a small ridge penalty $\tfrac\alpha2\|w\|^2$ change?
    """)
    return


if __name__ == "__main__":
    app.run()
