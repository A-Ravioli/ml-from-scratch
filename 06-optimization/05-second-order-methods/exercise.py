import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections import deque

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import solve  # yours, from ch 02 · topic 03
    from mlfs.optim import (  # yours, from topics 02 and 03
        OptimResult,
        backtracking_line_search,
        gradient_descent,
        heavy_ball,
        nesterov,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # second-order methods · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end, `mlfs.optim` has
    Newton's method, the two quasi-Newton methods that real libraries use (BFGS and
    L-BFGS), and a single `minimize` entry point for every deterministic optimizer in the
    chapter.

    | stage | you build |
    |---|---|
    | 1 | `newton`: damped Newton with a line search and safe fallbacks → `mlfs.optim` |
    | 2 | `wolfe_line_search`, `bfgs_update`, `bfgs`: a line search that keeps curvature positive, and an inverse Hessian learned from gradient differences → `mlfs.optim` |
    | 3 | `two_loop`, `lbfgs`: the same, from the last few steps only, in $O(mn)$ memory → `mlfs.optim` |
    | 4 | `minimize`: one interface for gd, heavy ball, Nesterov, Newton, BFGS and L-BFGS → `mlfs.optim` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def newton(f, grad, hess, x0, *, damping=0.0, line_search=True, max_iters=100, tol=1e-08, callback=None):
    """stage 1: (damped) Newton's method for minimizing f.

    x0 may have any shape: work on the flattened vector internally (n = x0.size), so
    grad(x) is flattened and hess(x) must return the (n, n) Hessian with respect to the
    flattened x (row-major, like x.ravel()). f, grad, hess and callback always receive x in
    x0's shape, and the result's x has x0's shape too.
    at each iterate with g = grad(x) (stop, converged, when ||g||_2 <= tol):
      H = hess(x) (n, n); solve (H + damping * I) d = -g with mlfs.linalg.solve.
      if the solve fails (np.linalg.LinAlgError: singular), use d = -g instead.
      line_search=True: if d is not a descent direction (g . d >= 0), use d = -g instead;
          alpha = backtracking_line_search(f, x, f(x), g, d, alpha0=1.0); stop if alpha == 0.
      line_search=False: alpha = 1 (pure Newton: it may head for a saddle or a maximum).
      x <- x + alpha * d.
    same loop conventions as gradient_descent (x0 copied; history = [f(x0), ...];
    callback(x, k) after step k, its return value ignored; stop if f is not finite; at most
    max_iters steps; one grad and one hess call per step). returns OptimResult.
    """
    raise NotImplementedError("stage1: implement newton")


@app.cell(hide_code=True)
def _():
    checks.run(1, newton=newton)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def wolfe_line_search(f, grad, x, fx, gx, direction, *, alpha0=1.0, c1=0.0001, c2=0.9, max_iters=50):
    """stage 2: a step satisfying the (weak) Wolfe conditions, found by expanding and bisecting.

    with slope = gx . direction (< 0), a step alpha is accepted when BOTH hold:
        sufficient decrease (Armijo):  f(x + alpha d) <= fx + c1 alpha slope
        curvature:                     grad(x + alpha d) . d >= c2 slope
    keep a bracket lo = 0, hi = inf and start at alpha = alpha0. at each trial: if Armijo
    fails (or f is not finite), hi = alpha; else if the curvature condition fails (the step
    is too short), lo = alpha; else return. next alpha = (lo + hi) / 2 if hi is finite, else
    2 alpha. at most max_iters trials (one f call each, plus one grad call when Armijo holds).
    returns (alpha, f(x + alpha d), grad(x + alpha d)) on success, (0.0, None, None) if no
    trial passes. x, gx and direction are 1-D arrays (n,) (bfgs and lbfgs call it on
    flattened vectors).
    """
    raise NotImplementedError("stage2: implement wolfe_line_search")


@app.function
def bfgs_update(Hinv: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """stage 2: the BFGS update of an inverse-Hessian approximation.

    Hinv (n, n) symmetric positive definite, s = x_new - x (n,), y = g_new - g (n,), with
    s . y > 0. with rho = 1 / (y . s), returns the NEW matrix
        (I - rho s y^T) Hinv (I - rho y s^T) + rho s s^T,
    which satisfies the secant condition Hinv_new @ y == s and stays symmetric positive
    definite. Hinv is not modified.
    """
    raise NotImplementedError("stage2: implement bfgs_update")


@app.function
def bfgs(f, grad, x0, *, max_iters=500, tol=1e-06, callback=None):
    """stage 2: BFGS with a dense inverse-Hessian approximation and a Wolfe line search.

    start with Hinv = I. at each iterate (stop, converged, when ||g||_2 <= tol):
      d = -Hinv @ g; if g . d >= 0 (not a descent direction), reset Hinv = I and d = -g.
      alpha, f_new, g_new = wolfe_line_search(f, grad, x, f(x), g, d); stop if alpha == 0.
      s = alpha d (so x_new = x + s), y = g_new - g. the Wolfe step guarantees s . y > 0;
      still, skip the update unless s . y > 1e-10 ||s|| ||y|| (rounding).
      before the FIRST update, rescale Hinv = (s . y)/(y . y) * I; then
      Hinv = bfgs_update(Hinv, s, y).
    x0 may have any shape: work on the flattened vector (n = x0.size) and give f, grad and
    callback x in x0's shape; the result's x has x0's shape. same loop conventions as
    gradient_descent (history, callback(x, k) with its return value ignored, max_iters, stop
    if f is not finite). reuse f_new and g_new from the line search: no extra calls of f or
    grad. returns OptimResult.
    """
    raise NotImplementedError("stage2: implement bfgs")


@app.cell(hide_code=True)
def _():
    checks.run(2, wolfe_line_search=wolfe_line_search, bfgs_update=bfgs_update, bfgs=bfgs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def two_loop(g: np.ndarray, s_hist, y_hist) -> np.ndarray:
    """stage 3: the L-BFGS two-loop recursion: returns H g, where H is the inverse-Hessian
    approximation built from the pairs (s_i, y_i) (oldest first), without forming any matrix.

    H is what you'd get by starting from gamma * I, with gamma = (s . y)/(y . y) of the NEWEST
    pair (gamma = 1 with no pairs), and applying bfgs_update with each pair, oldest first.
      q = g;  for the pairs newest -> oldest: a_i = rho_i s_i . q;  q -= a_i y_i
      r = gamma q;  for the pairs oldest -> newest: b = rho_i y_i . r;  r += (a_i - b) s_i
    with rho_i = 1 / (y_i . s_i). returns r (n,), a new array (g is not modified).
    """
    raise NotImplementedError("stage3: implement two_loop")


@app.function
def lbfgs(f, grad, x0, *, history_size=10, max_iters=500, tol=1e-06, callback=None):
    """stage 3: L-BFGS: BFGS with only the last history_size pairs (s, y), via two_loop.

    the same loop as your bfgs (with wolfe_line_search), except: d = -two_loop(g, S, Y);
    if g . d >= 0, clear the history and use d = -g; after each step, if
    s . y > 1e-10 ||s|| ||y||, append (s, y), dropping the oldest pair beyond history_size.
    memory O(history_size * n): never form an n x n matrix. like bfgs, x0 may have any shape
    (flatten internally; f, grad, callback and the result see x0's shape). returns
    OptimResult.
    """
    raise NotImplementedError("stage3: implement lbfgs")


@app.cell(hide_code=True)
def _():
    checks.run(3, two_loop=two_loop, lbfgs=lbfgs, bfgs_update=bfgs_update, wolfe_line_search=wolfe_line_search)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def minimize(f, grad, x0, *, method="lbfgs", hess=None, max_iters=1000, tol=1e-06, callback=None, **options):
    """stage 4: one entry point for every deterministic minimizer in mlfs.optim.

    method in {'gd', 'heavy_ball', 'nesterov', 'newton', 'bfgs', 'lbfgs'} calls
    gradient_descent, heavy_ball, nesterov (from mlfs.optim), newton, bfgs or lbfgs with
    max_iters, tol and callback passed through, and every extra keyword in **options
    forwarded unchanged (lr, momentum, line_search, damping, history_size, ...).
    'newton' needs hess: raise ValueError if it's None. an unknown method raises ValueError
    naming the valid ones. every method accepts x0 of any shape (e.g. a (d, k) weight
    matrix) and returns x in that shape. returns the method's OptimResult.
    """
    raise NotImplementedError("stage4: implement minimize")


@app.cell(hide_code=True)
def _():
    checks.run(4, minimize=minimize)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.optim import minimize` gives you every
    deterministic optimizer of the chapter behind one call. open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
