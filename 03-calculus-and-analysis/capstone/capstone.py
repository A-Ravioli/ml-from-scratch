import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.autodiff import Dual, grad, jacobian, numerical_jacobian, value_and_grad  # yours, topics 02 and 04

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    FRACTAL_RES = 40 if SMALL else 200


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 · capstone: an autodiff-powered newton solver

    read [`brief.md`](brief.md) first. you'll build Newton's method twice, for solving
    $F(x) = 0$ and for minimizing $f(x)$, with every derivative computed **exactly** by your
    forward-mode library `mlfs.autodiff`, and then measure how it converges, where it fails,
    and how it compares with first-order methods.

    | stage | you build |
    |---|---|
    | 1 | `newton_root`: Newton's method for systems, with Jacobians from your `jacobian` |
    | 2 | `hessian`: exact Hessians from nested dual numbers |
    | 3 | `newton_minimize`: damped Newton with a Hessian fix and a line search |
    | 4 | `estimate_order`: a robust order-of-convergence estimate from an error sequence |

    when they're green, scroll down: the experiments run on your functions, and the
    write-up prompts are at the end.
    """)
    return


@app.function
def newton_root(F, x0: np.ndarray, *, tol: float = 1e-10, max_iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """stage 1: solve F(x) = 0 for F: (n,) -> (n,) by Newton's method.

    each step solves J(x_k) s = -F(x_k), with J from mlfs.autodiff.jacobian (exact, one
    forward pass per input), and sets x_{k+1} = x_k + s. stop at the first iterate with
    ||F(x_k)||_2 <= tol, or after max_iters steps. returns (x_final (n,), iterates (k + 1, n)),
    where iterates[0] = x0 and iterates[-1] = x_final. a singular Jacobian raises
    np.linalg.LinAlgError (from np.linalg.solve).
    """
    raise NotImplementedError("stage1: implement newton_root")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", newton_root=newton_root)
    return


@app.function
def hessian(f, x: np.ndarray) -> np.ndarray:
    """stage 2: the exact Hessian of f: (n,) -> float at x, shape (n, n), from nested Duals.

    for each pair i <= j, evaluate f once on inputs Dual(Dual(x_k, [k == i]), Dual([k == j], 0))
    (two independent infinitesimals, on x_i and on x_j); the .dual.dual part of the result is
    d^2 f / dx_i dx_j. fill both H[i, j] and H[j, i]: n(n + 1)/2 evaluations, exactly symmetric.
    if f ignores its input (returns a plain number), that entry is 0.
    """
    raise NotImplementedError("stage2: implement hessian")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", hessian=hessian)
    return


@app.function
def newton_minimize(f, x0: np.ndarray, *, tol: float = 1e-10, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """stage 3: minimize f: (n,) -> float by damped Newton; returns (x_final, iterates (k + 1, n)).

    at each iterate: g = gradient (mlfs.autodiff.value_and_grad), H = your hessian. stop when
    ||g||_2 <= tol. otherwise make H positive definite: try np.linalg.cholesky(H + lam I) with
    lam = 0, then lam = 1e-8 * max(1, max|H_ii|), multiplying lam by 10 until it succeeds
    (raise np.linalg.LinAlgError after 100 tries, e.g. if H contains nan).
    the direction is d = -(H + lam I)^{-1} g (a descent direction). then backtrack (Armijo):
    t = 1, halve t while f(x + t d) > f(x) + 1e-4 t g.d, at most 60 halvings. if no t passes,
    STOP (return without stepping): f is then flat to rounding level along d. otherwise step to
    x + t d. so f never increases, and near a minimum (H positive definite, t = 1) it is pure
    Newton. choose tol above the gradient's rounding floor (about 1e-10 to 1e-12 relative to
    the size of f's terms), or the line search, not tol, will end the run.
    """
    raise NotImplementedError("stage3: implement newton_minimize")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", newton_minimize=newton_minimize)
    return


@app.function
def estimate_order(errors: np.ndarray, *, floor: float = 1e-13) -> float:
    """stage 4: a robust estimate of the order of convergence q from errors e_k = |x_k - x*|.

    drop every error <= floor (they're at the rounding level), compute the estimates
    q_k = log(e_{k+2}/e_{k+1}) / log(e_{k+1}/e_k) on what's left (topic 01), and return the
    median of the LAST three (or of all of them, if there are fewer). raises ValueError if
    fewer than 3 errors remain.
    """
    raise NotImplementedError("stage4: implement estimate_order")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", estimate_order=estimate_order)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A (given): how fast does Newton converge?

    three root-finding problems solved with your `newton_root`, and the cubic again with
    finite-difference Jacobians (`numerical_jacobian`, with a good step and a crude one)
    instead of your exact one. the error is measured against a reference root, and your
    `estimate_order` reads off the order.
    """)
    return


@app.cell
def _():
    _r1 = 2.0945514815423265
    _circle = lambda v: np.array([v[0] ** 2 + v[1] ** 2 - 4.0, v[0] * v[1] - 1.0], dtype=object)
    _circle_root = newton_root(_circle, np.array([2.0, 0.5]), tol=0.0, max_iters=12)[0]
    _problems = {
        "x³ − 2x − 5 (simple root)": (lambda v: v**3 - 2 * v - 5, np.array([3.0]), np.array([_r1]), "exact"),
        "circle ∩ hyperbola (2-D)": (_circle, np.array([2.0, 0.5]), _circle_root, "exact"),
        "(x − 1)²(x + 2) (double root)": (lambda v: (v - 1) ** 2 * (v + 2), np.array([2.5]), np.array([1.0]), "exact"),
        "x³ − 2x − 5, finite-difference J (h = 1e-6)": (lambda v: v**3 - 2 * v - 5, np.array([3.0]), np.array([_r1]), 1e-6),
        "x³ − 2x − 5, finite-difference J (h = 0.3)": (lambda v: v**3 - 2 * v - 5, np.array([3.0]), np.array([_r1]), 0.3),
    }

    def _newton_fd(F, x0, iters, h):
        x = np.array(x0, dtype=float)
        xs = [x.copy()]
        for _ in range(iters):
            Fx = np.asarray(F(x), dtype=float)
            if np.linalg.norm(Fx) == 0:
                break
            x = x + np.linalg.solve(numerical_jacobian(lambda z: np.asarray(F(z), dtype=float), x, h=h), -Fx)
            xs.append(x.copy())
        return np.array(xs)

    _rows = []
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _name, (_F, _x0, _root, _kind) in _problems.items():
        if _kind == "exact":
            _its = newton_root(_F, _x0, tol=0.0, max_iters=60 if "double" in _name else 12)[1]
        else:
            _its = _newton_fd(_F, _x0, 25, _kind)
        _err = np.linalg.norm(_its - _root, axis=1)
        _ax.semilogy(np.maximum(_err, 1e-17), "o-", ms=3, label=_name)
        try:
            _q = estimate_order(_err)
        except ValueError:
            _q = float("nan")
        _first = np.flatnonzero(_err < 1e-10)
        _rows.append(f"| {_name} | {_first[0] if _first.size else '> ' + str(len(_err) - 1)} | {_q:.2f} |")
    _ax.set(xlabel="iteration", ylabel="error", title="Newton's method: error per iteration")
    _ax.legend(fontsize=7)
    mo.vstack([_fig, mo.md("| problem | iterations to error < 1e-10 | estimated order |\n|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B (given): Newton versus gradient descent on Rosenbrock

    $f(x, y) = (1 - x)^2 + 100(y - x^2)^2$ from $(-1.2, 1)$. your `newton_minimize` against
    gradient descent with the same backtracking line search (and gradients from your
    `grad`), both run until $\|\nabla f\| \le 10^{-8}$ or a cap on iterations.
    """)
    return


@app.cell
def _():
    _rosen = lambda v: (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2
    _x0 = np.array([-1.2, 1.0])
    _t0 = time.perf_counter()
    _xn, _its_n = newton_minimize(_rosen, _x0, tol=1e-8)
    _t_newton = time.perf_counter() - _t0

    def _gd(f, x0, tol, cap):
        x = np.array(x0, dtype=float)
        vals = [f(x)]
        for _ in range(cap):
            fx, g = value_and_grad(f, x)
            if np.linalg.norm(g) <= tol:
                break
            t = 1.0
            while f(x - t * g) > fx - 1e-4 * t * (g @ g) and t > 1e-12:
                t *= 0.5
            x = x - t * g
            vals.append(f(x))
        return x, np.array(vals)

    _cap = 2000 if SMALL else 20000
    _t0 = time.perf_counter()
    _xg, _vals_g = _gd(_rosen, _x0, 1e-8, _cap)
    _t_gd = time.perf_counter() - _t0
    _vals_n = np.array([_rosen(v) for v in _its_n])
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _ax.semilogy(np.maximum(_vals_n, 1e-30), "o-", ms=3, label=f"damped Newton ({len(_its_n) - 1} its)")
    _ax.semilogy(np.maximum(_vals_g, 1e-30), "-", label=f"gradient descent ({len(_vals_g) - 1} its)")
    _ax.set(xscale="symlog", xlabel="iteration", ylabel="f(x) − f*", title="Rosenbrock from (−1.2, 1)")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"damped Newton: **{len(_its_n) - 1}** iterations, ends at {np.round(_xn, 10)}, {_t_newton * 1e3:.0f} ms. "
        f"gradient descent: **{len(_vals_g) - 1}** iterations (cap {_cap}), ends at {np.round(_xg, 6)}, "
        f"f = {_vals_g[-1]:.2e}, {_t_gd * 1e3:.0f} ms.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C (given): basins of attraction

    $z^3 = 1$ has three complex roots. written as a real system,
    $F(x, y) = (x^3 - 3xy^2 - 1,\ 3x^2y - y^3)$. we start your `newton_root` from every point
    of a grid and color each start by the root it reaches (black: no convergence in 40
    steps). small config: a 40 × 40 grid; full config: 200 × 200.
    """)
    return


@app.cell
def _():
    _F = lambda v: np.array([v[0] ** 3 - 3 * v[0] * v[1] ** 2 - 1, 3 * v[0] ** 2 * v[1] - v[1] ** 3], dtype=object)
    _roots = np.array([[1.0, 0.0], [-0.5, np.sqrt(3) / 2], [-0.5, -np.sqrt(3) / 2]])
    _g = np.linspace(-1.5, 1.5, FRACTAL_RES)
    _img = np.full((FRACTAL_RES, FRACTAL_RES), -1)
    _iters = np.zeros((FRACTAL_RES, FRACTAL_RES))
    _t0 = time.perf_counter()
    for _i, _yy in enumerate(_g):
        for _j, _xx in enumerate(_g):
            try:
                _x, _its = newton_root(_F, np.array([_xx, _yy]), tol=1e-10, max_iters=40)
            except np.linalg.LinAlgError:
                continue
            _d = np.linalg.norm(_roots - _x, axis=1)
            if _d.min() < 1e-6:
                _img[_i, _j] = int(np.argmin(_d))
                _iters[_i, _j] = len(_its) - 1
    _secs = time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _cmap = plt.matplotlib.colors.ListedColormap(["black", "C0", "C1", "C2"])
    _ax.imshow(_img + 1, origin="lower", extent=(-1.5, 1.5, -1.5, 1.5), cmap=_cmap, vmin=0, vmax=3)
    _ax.plot(_roots[:, 0], _roots[:, 1], "w*", ms=10)
    _ax.set(title=f"basins of Newton's method for z³ = 1 ({FRACTAL_RES}×{FRACTAL_RES})")
    _ok = _img >= 0
    mo.vstack([_fig, mo.md(
        f"converged: **{_ok.mean():.1%}** of starts; median iterations **{np.median(_iters[_ok]):.0f}**, "
        f"max {_iters[_ok].max():.0f}. share of each root: {', '.join(f'{np.mean(_img == r):.1%}' for r in range(3))}. "
        f"{_secs:.1f} s.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment D (given): maximum likelihood by Newton

    logistic regression on 300 synthetic points in 3-D (with an intercept), fitted by
    minimizing the mean negative log-likelihood plus a small ridge penalty with your
    `newton_minimize`. every gradient and Hessian comes from your dual numbers; no
    derivative is written by hand.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    _X = np.hstack([np.ones((300, 1)), _rng.normal(size=(300, 3))])
    _w_true = np.array([-0.5, 2.0, -1.0, 0.5])
    _y = (_rng.random(300) < 1 / (1 + np.exp(-_X @ _w_true))).astype(float)

    def _nll(w):
        z = _X @ w
        # log(1 + e^z) - y z, written with Dual-friendly operations (no np.logaddexp)
        return np.sum(np.log(1 + np.exp(-np.abs(z))) + np.maximum(z, 0.0) - _y * z) / len(_y) + 0.005 * (w @ w)

    _w, _its = newton_minimize(_nll, np.zeros(4), tol=1e-10)
    _gn = [np.linalg.norm(grad(_nll, w)) for w in _its]
    _rows = "\n".join(f"| {k} | {g:.2e} |" for k, g in enumerate(_gn))
    mo.md(f"fitted weights **{np.round(_w, 4)}** (true {_w_true}); **{len(_its) - 1}** Newton steps.\n\n"
          "| step | ‖∇f‖ |\n|---|---|\n" + _rows)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **convergence orders.** from experiment A: which problems converge quadratically, which
       linearly, and why? quote your estimated orders, and explain the double root with the
       analysis from topic 01. what happens to the order with a finite-difference Jacobian,
       and at what error does it set in?
    2. **Newton versus gradient descent.** from experiment B: iterations *and* wall-clock time
       for each. which is faster by each measure, and why? estimate the cost of one Newton
       iteration in function evaluations (gradient plus Hessian via your dual numbers) for
       $n$ inputs, and say when you'd stop preferring Newton.
    3. **safeguards.** describe one start where pure Newton (no Hessian fix, no line search)
       goes wrong for minimization, and what each of your two safeguards does about it.
       show it with numbers.
    4. **basins.** describe the picture from experiment C. where do the basins meet, and what
       happens to starts near those boundaries? what does it say about "Newton converges if
       you start close enough"?
    5. **exactness.** where in this capstone did *exact* derivatives (your dual numbers)
       matter, compared with finite differences from topic 02? give one place where they
       made no visible difference and one where they did.
    """)
    return


if __name__ == "__main__":
    app.run()
