import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import notebooks, ui
    from mlfs.linear import LinearRegression  # yours, from topic 01
    from mlfs.optim import gradient_descent  # yours, from ch 06

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.function
def gd_trace(A, y, pen, lr, n_steps):
    """run your gradient_descent for exactly n_steps (tol = 0) and record ||theta|| and ||grad||."""
    norms, grads = [], []

    def record(theta, k):
        norms.append(float(np.linalg.norm(theta[1:])))
        grads.append(float(np.linalg.norm(ex.logistic_gradient(theta, A, y, pen))))

    gradient_descent(lambda t: ex.logistic_loss(t, A, y, pen), lambda t: ex.logistic_gradient(t, A, y, pen),
                     np.zeros(A.shape[1]), lr=lr, max_iters=n_steps, tol=0.0, callback=record)
    return np.array(norms), np.array(grads)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # logistic regression · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: when a line separates the classes perfectly

    100 points in 2-D, labelled by which side of a line they're on: no noise, so the classes
    are separable. you fit your `LogisticRegression` with $\alpha = 0$ using IRLS at three
    tolerances, then run gradient descent for up to 100,000 steps and watch the size of the
    weights, and finally add a tiny penalty, $\alpha = 0.01$.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with alpha = 0 on separable data, what is ||w|| when IRLS reports convergence? does the "
        "answer depend on tol? how does ||w|| grow with the number of gradient-descent steps?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(100, 2))
    _y = (_X[:, 0] + 0.5 * _X[:, 1] > 0).astype(float)
    _rows = []
    for _tol in (1e-4, 1e-8, 1e-12):
        _m = ex.LogisticRegression(alpha=0.0, tol=_tol, max_iters=200).fit(_X, _y)
        _rows.append(f"| IRLS, $\\alpha$ = 0, tol = {_tol:g} | {_m.n_iter_} | {_m.converged_} | "
                     f"{np.linalg.norm(_m.coef_):.1f} | {_m.predict_proba(_X).min():.1e} |")
    _m = ex.LogisticRegression(alpha=0.01).fit(_X, _y)
    _rows.append(f"| IRLS, $\\alpha$ = 0.01 | {_m.n_iter_} | {_m.converged_} | {np.linalg.norm(_m.coef_):.1f} | "
                 f"{_m.predict_proba(_X).min():.1e} |")
    _A = np.column_stack([np.ones(100), _X])
    _norms, _ = gd_trace(_A, _y, np.zeros(3), 1.0, 100_000)
    _norms_r, _ = gd_trace(_A, _y, np.array([0.0, 0.01, 0.01]), 1.0, 100_000)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _k = np.arange(1, len(_norms) + 1)
    _ax.semilogx(_k, _norms, label=r"GD, $\alpha = 0$")
    _ax.semilogx(_k, _norms_r, label=r"GD, $\alpha = 0.01$")
    _ax.set(xlabel="gradient steps (log scale)", ylabel=r"$\|w\|$")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([
        mo.md("| fit | steps | converged_ | $\\|w\\|$ | most extreme probability (closest to 0) |\n|---|---|---|---|---|\n"
              + "\n".join(_rows)),
        _fig,
        mo.md(f"GD with $\\alpha$ = 0: $\\|w\\|$ = {_norms[99]:.1f} after 100 steps, {_norms[999]:.1f} after 1,000, "
              f"{_norms[9_999]:.1f} after 10,000, {_norms[-1]:.1f} after 100,000. with $\\alpha$ = 0.01: "
              f"{_norms_r[-1]:.2f}."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on separable data, what happens to the loss if you take any separating $w$
    and double it? so where is the minimum? why does IRLS still say `converged_ = True`, and
    why does its answer depend on `tol`? gradient descent's $\|w\|$ keeps growing, but it
    needs ten times as many steps for each further gain: look at the size of the gradient
    when every margin is large. what does the penalty change, and what does it do to the
    probabilities?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: Newton versus gradient descent

    500 points, 5 standardized features with pairwise correlation 0.8 (realistic: features
    often overlap), overlapping classes, $\alpha = 10^{-3}$. you track the gradient norm of
    the objective, step by step, for your `irls` and for gradient descent with the classic
    safe step size $\eta = 1/L$, where $L = \lambda_{\max}(A^\top A)/(4n)$ bounds the
    curvature of the loss everywhere.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "how many steps does each method need to bring the gradient norm below 1e-8? (gradient "
        "descent's steps are much cheaper: is it worth it?)"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _C = 0.8 * np.ones((5, 5)) + 0.2 * np.eye(5)
    _X = _rng.normal(size=(500, 5)) @ np.linalg.cholesky(_C).T
    _y = (_rng.random(500) < 1 / (1 + np.exp(-(_X @ np.array([2.0, -1.0, 0.5, 0.0, 1.5]))))).astype(float)
    _A = np.column_stack([np.ones(500), _X])
    _pen = np.concatenate([[0.0], np.full(5, 1e-3)])
    _L = np.linalg.eigvalsh(_A.T @ _A / 500).max() / 4
    _newton = []
    for _k in range(0, 9):
        _theta, _, _ = ex.irls(_A, _y, _pen, max_iters=_k, tol=0.0)
        _newton.append(float(np.linalg.norm(ex.logistic_gradient(_theta, _A, _y, _pen))))
    _, _gd = gd_trace(_A, _y, _pen, 1.0 / _L, 3000)
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    _ax.semilogy(range(len(_newton)), np.maximum(_newton, 1e-17), "o-", label="IRLS (Newton)")
    _ax.semilogy(np.arange(1, len(_gd) + 1), _gd, label=r"gradient descent, $\eta = 1/L$")
    _ax.set(xlabel="steps", ylabel="gradient norm", xscale="symlog", ylim=(1e-17, 1))
    _ax.legend()
    _fig.tight_layout()
    _first = lambda g: next((i for i, v in enumerate(g) if v < 1e-8), None)
    _gd_hit = _first(_gd)
    mo.vstack([_fig, mo.md(
        f"IRLS gradient norms: {', '.join(f'{v:.0e}' for v in _newton)}.\n\n"
        f"below $10^{{-8}}$: IRLS after **{_first(_newton)}** steps; gradient descent after "
        f"**{'more than 3,000' if _gd_hit is None else f'{_gd_hit + 1:,}'}** steps. "
        f"($\\eta = 1/L$ = {1 / _L:.2f} here.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does "the number of correct digits doubles every step" look like on
    this plot, and in which steps does it happen for IRLS? why is gradient descent's line
    straight (linear convergence), and what sets its slope? each Newton step costs a
    $p \times p$ solve and an $O(np^2)$ Hessian: with $p = 6$ that's nothing. when would you
    prefer gradient descent (or L-BFGS) anyway?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: least squares as a classifier

    a tempting shortcut: code the classes as 0 and 1, fit your `LinearRegression` (topic 01),
    and predict class 1 where the fit exceeds 0.5. two overlapping clusters of 100 points
    each, first on their own, then with 40 extra class-1 points added far away on the
    *correct* side. compare with your `LogisticRegression` ($\alpha = 10^{-3}$).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the 40 extra points are easy: any sensible boundary already classifies them correctly. "
        "how much does adding them change each model's accuracy on the original 200 points?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _rng = np.random.default_rng(2)
    _X0 = _rng.normal(size=(100, 2)) + np.array([-1.0, 0.0])
    _X1 = _rng.normal(size=(100, 2)) + np.array([1.0, 0.0])
    _Xb = np.vstack([_X0, _X1])
    _yb = np.repeat([0.0, 1.0], 100)
    _far = _rng.normal(size=(40, 2)) * 0.5 + np.array([8.0, 3.0])
    _Xf, _yf = np.vstack([_Xb, _far]), np.concatenate([_yb, np.ones(40)])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _rows = []
    _g1, _g2 = np.meshgrid(np.linspace(-4, 10, 300), np.linspace(-4, 6, 220))
    _grid = np.column_stack([_g1.ravel(), _g2.ravel()])
    for _ax, (_X, _y, _title) in zip(_axes, ((_Xb, _yb, "original 200 points"), (_Xf, _yf, "plus 40 easy points"))):
        _ls = LinearRegression().fit(_X, _y)
        _lr = ex.LogisticRegression(alpha=1e-3).fit(_X, _y)
        _acc_ls = np.mean((_ls.predict(_Xb) > 0.5) == _yb)
        _acc_lr = np.mean(_lr.predict(_Xb) == _yb)
        _rows.append(f"| {_title} | {_acc_ls:.3f} | {_acc_lr:.3f} |")
        _ax.scatter(_X[:, 0], _X[:, 1], c=_y, cmap="coolwarm", s=10)
        _ax.contour(_g1, _g2, _ls.predict(_grid).reshape(_g1.shape), levels=[0.5], colors="k", linestyles="--")
        _ax.contour(_g1, _g2, _lr.decision_function(_grid).reshape(_g1.shape), levels=[0.0], colors="g")
        _ax.set(title=_title, xlim=(-4, 10), ylim=(-4, 6))
    _axes[0].plot([], [], "k--", label="least squares (fit = 0.5)")
    _axes[0].plot([], [], "g-", label="logistic (p = 0.5)")
    _axes[0].legend(loc="lower right", fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("accuracy on the original 200 points:\n\n| training set | least squares | logistic |\n"
                           "|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** least squares wants the fit to *equal* 1 on the far points, not just exceed
    0.5. what does it have to do to the line to get closer to that, and what happens to the
    boundary as a result? what does the cross-entropy charge for a point that's already
    classified correctly with a large margin, and so how much do the far points pull on the
    logistic boundary?
    """)
    return


if __name__ == "__main__":
    app.run()
