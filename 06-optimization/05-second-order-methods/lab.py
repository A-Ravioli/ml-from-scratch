import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.optim import gradient_descent  # yours, from topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # second-order methods · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.function
def logistic_problem(n, d, seed, alpha=1e-3):
    """a ridge-regularized logistic regression with correlated, badly scaled features.
    returns (f, grad, hess) for the mean negative log-likelihood + alpha/2 ||w||^2."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d))
    X = (Z + 0.5 * Z[:, :1]) * np.geomspace(1.0, 0.1, d)
    w_true = rng.normal(size=d) * 2
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ w_true))).astype(float)

    def f(w):
        z = X @ w
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * alpha * w @ w)

    def grad(w):
        p = 0.5 * (1 + np.tanh(0.5 * (X @ w)))
        return X.T @ (p - y) / n + alpha * w

    def hess(w):
        p = 0.5 * (1 + np.tanh(0.5 * (X @ w)))
        return (X.T * (p * (1 - p))) @ X / n + alpha * np.eye(d)

    return f, grad, hess


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: iterations versus seconds

    logistic regression on 2,000 synthetic examples with $d$ correlated, badly scaled
    features (plus a small ridge penalty), for $d$ from 10 to 400. four of your optimizers run
    to $\|\nabla f\| \le 10^{-6}$ from $w = 0$: gradient descent with a line search
    (`lr=50`, so backtracking chooses the step; at most 5,000 steps), Newton (with the exact
    Hessian, solved by your `mlfs.linalg.solve`), BFGS and L-BFGS. we record iterations and
    wall-clock time (these depend on your machine; compare them with each other).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the four methods by iterations, and separately by wall-clock time, at d = 10 and at d = 400. "
        "where does Newton's time per iteration start to hurt?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _dims = [10, 50, 100, 200, 400]
    _rows = []
    _times = {m: [] for m in ("gradient descent", "Newton", "BFGS", "L-BFGS")}
    for _d in _dims:
        _f, _g, _h = logistic_problem(2000, _d, seed=_d)
        _w0 = np.zeros(_d)
        _runs = {
            "gradient descent": lambda: gradient_descent(_f, _g, _w0, lr=50.0, line_search=True, max_iters=5000),
            "Newton": lambda: ex.newton(_f, _g, _h, _w0, tol=1e-6),
            "BFGS": lambda: ex.bfgs(_f, _g, _w0, tol=1e-6, max_iters=2000),
            "L-BFGS": lambda: ex.lbfgs(_f, _g, _w0, tol=1e-6, max_iters=2000),
        }
        _cells = []
        for _name, _run in _runs.items():
            _t0 = time.perf_counter()
            _r = _run()
            _sec = time.perf_counter() - _t0
            _times[_name].append(_sec)
            _cells.append(f"{_r.n_iters}{'' if _r.converged else '*'} / {_sec * 1e3:.0f} ms")
        _rows.append(f"| {_d} | " + " | ".join(_cells) + " |")
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _name, _ts in _times.items():
        _ax.loglog(_dims, np.array(_ts) * 1e3, "o-", label=_name)
    _ax.set(xlabel="number of features d", ylabel="wall-clock time (ms)", title="time to ‖∇f‖ ≤ 1e-6")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| d | gradient descent | Newton | BFGS | L-BFGS |\n|---|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\neach cell: iterations / time. * = not converged within the cap.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does Newton need the fewest iterations at every $d$, and about the
    same number at every $d$? what does one Newton iteration cost as a function of $d$ (the
    Hessian, and the solve)? and one BFGS iteration? read the slopes of the time plot. which
    method would you pick at $d = 10$, and at $d = 10^5$?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: why does full BFGS lose to L-BFGS?

    in experiment 1, L-BFGS (10 pairs) needed fewer iterations than BFGS at every size,
    although BFGS keeps *all* the curvature information. here we take the 20-feature
    logistic regression apart. we solve it with your `bfgs`, your `lbfgs` with
    `history_size=10`, and your `lbfgs` with `history_size=100000` (it never forgets, so it
    has exactly the same pairs as BFGS). then two controls, where theory makes a clear
    prediction:

    - the same three methods on a **quadratic with the Hessian frozen** at the minimizer,
      $\tfrac12(w - w^*)^\top\nabla^2f(w^*)(w - w^*)$: the Hessian never changes, so no
      curvature information can go stale;
    - BFGS on that quadratic with an **exact** line search (built here from your
      `bfgs_update`), where theory says BFGS finishes in about $d = 20$ steps.

    we also record the scale $\gamma = s^\top y/y^\top y$ of every step: BFGS uses only the
    first one, to seed its matrix $\gamma_0I$; L-BFGS re-seeds with the newest one at every
    iteration.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "which wins on the logistic loss: BFGS, L-BFGS with 10 pairs, or L-BFGS that never forgets? and on the "
        "frozen quadratic, where old curvature can't go stale?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _f, _g, _h = logistic_problem(2000, 20, seed=7)
    _ws = ex.newton(_f, _g, _h, np.zeros(20), tol=1e-10).x
    _Hs = _h(_ws)
    _fq = lambda w: float(0.5 * (w - _ws) @ _Hs @ (w - _ws))
    _gq = lambda w: _Hs @ (w - _ws)
    _methods = {
        "BFGS": lambda F, G, cb: ex.bfgs(F, G, np.zeros(20), tol=1e-8, max_iters=3000, callback=cb),
        "L-BFGS, 10 pairs": lambda F, G, cb: ex.lbfgs(F, G, np.zeros(20), tol=1e-8, max_iters=3000, callback=cb),
        "L-BFGS, never forgets": lambda F, G, cb: ex.lbfgs(F, G, np.zeros(20), history_size=100_000, tol=1e-8,
                                                          max_iters=3000, callback=cb),
    }
    _rows, _curves, _gam = [], {}, []
    for _pname, (_F, _G) in (("logistic loss", (_f, _g)), ("frozen quadratic", (_fq, _gq))):
        _cells = []
        for _mname, _run in _methods.items():
            _xs = [np.zeros(20)]
            _r = _run(_F, _G, lambda x, k: _xs.append(np.array(x)))
            _cells.append(str(_r.n_iters))
            if _pname == "logistic loss":
                _curves[_mname] = [np.linalg.norm(_G(x)) for x in _xs]
                if _mname == "BFGS":
                    _gam = [float((b - a) @ (_G(b) - _G(a)) / np.sum((_G(b) - _G(a)) ** 2))
                            for a, b in zip(_xs[:-1], _xs[1:])]
        _rows.append(f"| {_pname} | " + " | ".join(_cells) + " |")
    def _bfgs_seeded(F, G, gamma):
        # your bfgs, except that the starting matrix is gamma I (no rescale from the first step)
        x, H, k = np.zeros(20), gamma * np.eye(20), 0
        fx, gx = F(x), G(x)
        while np.linalg.norm(gx) > 1e-8 and k < 3000:
            d = -H @ gx
            a, f_new, g_new = ex.wolfe_line_search(F, G, x, fx, gx, d)
            if a == 0.0:
                break
            s_, y_ = a * d, g_new - gx
            if s_ @ y_ > 1e-10 * np.linalg.norm(s_) * np.linalg.norm(y_):
                H = ex.bfgs_update(H, s_, y_)
            x, fx, gx, k = x + s_, f_new, g_new, k + 1
        return k

    _gmed = float(np.median(_gam[1:]))
    _seeded = [str(_bfgs_seeded(_f, _g, _gmed)), str(_bfgs_seeded(_fq, _gq, _gmed))]
    _x, _Hi, _k = np.zeros(20), np.eye(20), 0
    while np.linalg.norm(_gq(_x)) > 1e-8 and _k < 3000:
        _gk = _gq(_x)
        _d = -_Hi @ _gk
        _s = -(_gk @ _d) / (_d @ _Hs @ _d) * _d  # the exact minimizer along d
        _y = _Hs @ _s
        if _k == 0:
            _Hi = (_s @ _y) / (_y @ _y) * np.eye(20)
        _Hi = ex.bfgs_update(_Hi, _s, _y)
        _x, _k = _x + _s, _k + 1
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for _mname, _c in _curves.items():
        _a1.semilogy(_c, label=_mname)
    _a1.set(xlabel="iteration", ylabel="‖∇f‖", title="logistic loss, 20 features")
    _a1.legend(fontsize=8)
    _a2.semilogy(np.arange(1, len(_gam) + 1), _gam, ".", ms=4)
    _a2.axhline(_gam[0], color="C3", ls="--", label=f"γ₀ = {_gam[0]:.3g} (BFGS's seed)")
    _a2.set(xlabel="step", ylabel="γ = sᵀy / yᵀy", title="the scale of each step (BFGS run)")
    _a2.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "iterations to ‖∇f‖ ≤ 1e-8:\n\n| problem | BFGS | L-BFGS, 10 pairs | L-BFGS, never forgets |\n|---|---|---|---|\n"
        + "\n".join(_rows)
        + f"\n\nBFGS with an **exact** line search on the frozen quadratic: **{_k}** iterations. "
        f"the first step's scale γ₀ = {_gam[0]:.3g}; the median γ over the later steps is {_gmed:.3g}.\n\n"
        f"the clinching control: BFGS seeded with **{_gmed:.3g}·I** instead of γ₀·I needs **{_seeded[0]}** "
        f"iterations on the logistic loss and **{_seeded[1]}** on the frozen quadratic.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the never-forgetting L-BFGS has *the same pairs* as BFGS. what is the
    only difference between the matrices the two apply to $g$? (BFGS seeds with $\gamma_0I$
    once; L-BFGS with $\gamma_kI$ at every iteration.) the frozen quadratic rules out "stale
    curvature" as the explanation: what's left? what does an exact line search change, and
    why does the theory's "about $d$ steps" assume one?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how much memory does L-BFGS need?

    a 300-feature logistic regression, solved by your `lbfgs` with `history_size` from 1 to
    50 and with unlimited memory (`history_size=100000`), against your `bfgs` and gradient
    descent with a line search, all to $\|\nabla f\| \le 10^{-6}$.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "how many iterations does L-BFGS need with history_size 1, 5, 20 and 50, compared with full BFGS? "
        "is there a sweet spot?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _f, _g, _h = logistic_problem(2000, 300, seed=11)
    _w0 = np.zeros(300)
    _ms = [1, 2, 3, 5, 10, 20, 50, 100_000]
    _its = []
    _evals = []
    for _m in _ms:
        _calls = [0]

        def _gc(w):
            _calls[0] += 1
            return _g(w)

        _r = ex.lbfgs(_f, _gc, _w0, history_size=_m, tol=1e-6, max_iters=3000)
        _its.append(_r.n_iters)
        _evals.append(_calls[0])
    _rb = ex.bfgs(_f, _g, _w0, tol=1e-6, max_iters=3000)
    _rg = gradient_descent(_f, _g, _w0, lr=50.0, line_search=True, max_iters=5000)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.semilogx(_ms[:-1], _its[:-1], "o-", label="L-BFGS")
    _ax.axhline(_its[-1], color="C2", ls=":", label=f"L-BFGS, never forgets ({_its[-1]})")
    _ax.axhline(_rb.n_iters, color="C1", ls="--", label=f"BFGS ({_rb.n_iters})")
    _ax.set(xlabel="history_size m", ylabel="iterations to ‖∇f‖ ≤ 1e-6", title="300 features",
            ylim=(0, max(max(_its), _rb.n_iters) * 1.15))
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| history_size | iterations | gradient evaluations |\n|---|---|---|\n"
        + "\n".join(f"| {'∞ (100000)' if m == 100_000 else m} | {i} | {e} |" for m, i, e in zip(_ms, _its, _evals))
        + f"\n\nfull BFGS: **{_rb.n_iters}** iterations; gradient descent with a line search: **{_rg.n_iters}**"
        f"{'' if _rg.converged else ' (not converged)'} iterations.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where do the returns diminish, and why does even $m = 1$ beat gradient
    descent by so much? here more memory keeps helping, all the way to "never forget": so
    what does full BFGS do differently (experiment 2)? what does each extra pair cost per
    iteration?
    """)
    return


if __name__ == "__main__":
    app.run()
