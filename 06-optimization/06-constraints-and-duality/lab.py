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
    # constraints & duality · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a support vector machine, solved through its dual

    40 points in the plane with labels $y_i = \pm1$, separable by a line through the origin.
    the hard-margin SVM (without a bias term) looks for the widest margin:

    $$\min_w\tfrac12\|w\|^2\quad\text{subject to}\quad y_i\,w^\top x_i \ge 1\ \text{ for every } i,$$

    a QP with $Q = I$, $c = 0$, $G = -\operatorname{diag}(y)X$, $h = -\mathbf 1$. we maximize its
    dual with your `projected_gradient_descent` (on $-g$, over $\lambda \ge 0$), tracking the dual
    value $g(\lambda_k)$ and the primal value $f(x(\lambda_k))$ with your `qp_dual`, and finish
    with your `solve_qp_dual` and `kkt_residuals`.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "the dual has one multiplier per training point (40). at the optimum, how many of them are nonzero? "
        "which points? do the primal and dual values meet?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _u = np.array([np.cos(0.6), np.sin(0.6)])
    _P = _rng.normal(size=(400, 2)) * 1.5
    _m = _P @ _u
    _P = _P[np.abs(_m) > 0.5][:40]
    _y = np.sign(_P @ _u)
    _Q, _c = np.eye(2), np.zeros(2)
    _G, _h = -(_y[:, None] * _P), -np.ones(len(_y))
    _L = float(np.linalg.eigvalsh(_G @ _G.T)[-1])
    _trace = []

    def _cb(lam, k):
        if k % 10 == 0 or k < 10:
            g_val, x = ex.qp_dual(_Q, _c, _G, _h, lam)
            _trace.append((k, g_val, 0.5 * x @ x, max(0.0, float(np.max(_G @ x - _h)))))

    ex.projected_gradient_descent(lambda lam: -ex.qp_dual(_Q, _c, _G, _h, lam)[0],
                                  lambda lam: -(_G @ ex.qp_dual(_Q, _c, _G, _h, lam)[1] - _h),
                                  lambda lam: ex.project_box(lam, 0.0, np.inf), np.zeros(len(_y)),
                                  lr=1 / _L, max_iters=3000, tol=0.0, callback=_cb)
    _w, _lam, _gap = ex.solve_qp_dual(_Q, _c, _G, _h)
    _kkt = ex.kkt_residuals(_Q, _c, _G, _h, _w, _lam)
    _sv = _lam > 1e-6
    _tr = np.array(_trace)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _a1.plot(_tr[:, 0], _tr[:, 1], label="dual g(λ_k)")
    _a1.plot(_tr[:, 0], _tr[:, 2], label="primal f(x(λ_k)) (may be infeasible)")
    _a1.axhline(0.5 * _w @ _w, color="k", ls=":", lw=0.8)
    _a1.set(xscale="log", xlabel="iteration", ylabel="objective", title="the two objectives meet")
    _a1.legend(fontsize=8)
    _a2.scatter(*_P[_y > 0].T, s=15, c="C0")
    _a2.scatter(*_P[_y < 0].T, s=15, c="C3")
    _a2.scatter(*_P[_sv].T, s=120, facecolors="none", edgecolors="k", label="λ > 0 (support vectors)")
    _t = np.linspace(-5, 5, 2)
    _n = np.array([-_w[1], _w[0]]) / np.linalg.norm(_w)
    for _off, _ls in ((0, "-"), (1, "--"), (-1, "--")):
        _base = _off * _w / (_w @ _w)
        _a2.plot(*(_base[:, None] + _n[:, None] * _t), "k", ls=_ls, lw=0.8)
    _a2.set_aspect("equal")
    _a2.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), title="w·x = 0 and the margins w·x = ±1")
    _a2.legend(fontsize=8, loc="lower right")
    _fig.tight_layout()
    _recon = (_lam * _y) @ _P
    mo.vstack([_fig, mo.md(
        f"nonzero multipliers: **{int(_sv.sum())} of {len(_y)}**, λ = {np.round(_lam[_sv], 4).tolist()}. "
        f"w = {np.round(_w, 4)}, margin width 2/‖w‖ = **{2 / np.linalg.norm(_w):.3f}**. "
        f"Σ λᵢ yᵢ xᵢ = {np.round(_recon, 4)}. duality gap **{_gap:.1e}**; KKT residuals "
        + ", ".join(f"{k} {v:.1e}" for k, v in _kkt.items()) + ".")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the primal value $f(x(\lambda_k))$ *below* the optimum early on (and
    why doesn't that contradict weak duality)? stationarity says $w = \sum_i\lambda_iy_ix_i$:
    which points determine $w$, and why do all the others have $\lambda_i = 0$ (complementary
    slackness)? what would happen to the classifier if you deleted a point that isn't a
    support vector? a support vector?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the constraint set does to the solution

    least squares, $\min_w\tfrac12\|Aw - b\|^2$ with 50 noisy observations and 20 weights (the
    data come from a mixture of 4 of the 20 columns, plus noise),
    solved three ways with your code: unconstrained (`gradient_descent` would do; we use
    `np.linalg.lstsq` for the reference), constrained to the probability simplex
    (`project_simplex`), and constrained to a Euclidean ball (`project_l2_ball`) of the same
    "budget" as the simplex solution's norm.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many of the 20 weights will be EXACTLY zero for the simplex-constrained solution? for the "
        "ball-constrained one?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _A = _rng.normal(size=(50, 20))
    _w_true = np.zeros(20)
    _w_true[:4] = _rng.dirichlet(np.ones(4))
    _b = _A @ _w_true + 0.5 * _rng.normal(size=50)
    _f = lambda w: float(0.5 * np.sum((_A @ w - _b) ** 2))
    _g = lambda w: _A.T @ (_A @ w - _b)
    _lr = 1 / np.linalg.eigvalsh(_A.T @ _A).max()
    _w_ls = np.linalg.lstsq(_A, _b, rcond=None)[0]
    _r_s = ex.projected_gradient_descent(_f, _g, ex.project_simplex, np.ones(20) / 20, lr=_lr, max_iters=20_000, tol=1e-9)
    _rad = float(np.linalg.norm(_r_s.x))
    _r_b = ex.projected_gradient_descent(_f, _g, lambda w: ex.project_l2_ball(w, _rad), np.zeros(20), lr=_lr,
                                         max_iters=20_000, tol=1e-9)
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _idx = np.arange(20)
    _ax.bar(_idx - 0.27, _w_ls, 0.27, label="unconstrained")
    _ax.bar(_idx, _r_s.x, 0.27, label="simplex")
    _ax.bar(_idx + 0.27, _r_b.x, 0.27, label=f"ball, radius {_rad:.2f}")
    _ax.axhline(0, color="k", lw=0.6)
    _ax.set(xlabel="weight index", ylabel="value", title="least-squares weights under three constraints")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [("unconstrained", _w_ls, _f(_w_ls), "—"), ("simplex", _r_s.x, _r_s.fun, _r_s.converged),
             ("ball", _r_b.x, _r_b.fun, _r_b.converged)]
    mo.vstack([_fig, mo.md(
        "| constraint | exact zeros | ‖w‖₂ | loss | converged |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {n} | {int(np.sum(w == 0))} of 20 | {np.linalg.norm(w):.3f} | {v:.3f} | {c} |" for n, w, v, c in _rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where do the exact zeros come from (look at the form of `project_simplex`)?
    why does the ball, with the same norm budget, give none? the simplex is the $\ell_1$ ball
    restricted to $w \ge 0$: what does that suggest about $\ell_1$ penalties (the lasso,
    chapter 08)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: multipliers are prices

    a random QP with 4 variables and 6 inequality constraints. we solve it with your
    `solve_qp_dual`, then move each constraint in turn ($h_i \to h_i \pm \epsilon$, with
    $\epsilon = 10^{-4}$), re-solve, and compare the measured rate of change of the optimal
    value (a central difference) with the prediction $-\lambda_i$. the same for an equality-constrained QP with your
    `solve_eq_qp`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "for the constraints with lambda_i = 0, how much does loosening them change the optimum? and for the "
        "others: how close is the measured change to -lambda_i * epsilon?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(9)
    _M = _rng.normal(size=(4, 4))
    _Q = _M @ _M.T + np.eye(4)
    _c = _rng.normal(size=4) * 8
    _G = _rng.normal(size=(6, 4))
    _h = _rng.uniform(0.2, 1.0, size=6)
    _val = lambda x: float(0.5 * x @ _Q @ x + _c @ x)
    _x, _lam, _ = ex.solve_qp_dual(_Q, _c, _G, _h)
    _eps = 1e-4
    _rows = []
    def _opt_h(i, delta):
        h2 = _h.copy()
        h2[i] += delta
        return _val(ex.solve_qp_dual(_Q, _c, _G, h2)[0])

    for _i in range(6):
        _fd = (_opt_h(_i, _eps) - _opt_h(_i, -_eps)) / (2 * _eps)  # central difference
        _rows.append(f"| inequality {_i} | {_lam[_i]:.4f} | {'active' if _lam[_i] > 1e-6 else 'slack'} | "
                     f"{_fd:+.5f} | {-_lam[_i]:+.5f} |")
    _A = _rng.normal(size=(2, 4))
    _b = _rng.normal(size=2)
    _xe, _le = ex.solve_eq_qp(_Q, _c, _A, _b)
    for _i in range(2):
        _bp, _bm = _b.copy(), _b.copy()
        _bp[_i] += _eps
        _bm[_i] -= _eps
        _fd = (_val(ex.solve_eq_qp(_Q, _c, _A, _bp)[0]) - _val(ex.solve_eq_qp(_Q, _c, _A, _bm)[0])) / (2 * _eps)
        _rows.append(f"| equality {_i} | {_le[_i]:+.4f} | (any sign) | {_fd:+.5f} | {-_le[_i]:+.5f} |")
    mo.md("| constraint | λ | status | measured ∂f*/∂h (central difference) | predicted −λ |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the sensitivity of a slack constraint exactly zero? why is the
    prediction for active constraints accurate to about $\epsilon\lambda$-squared order, and
    what would break it for a large $\epsilon$ (think about the active set changing)? why can
    equality multipliers have either sign, while inequality multipliers can't be negative?
    """)
    return


if __name__ == "__main__":
    app.run()
