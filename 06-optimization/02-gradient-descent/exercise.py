import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gradient descent · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. this topic starts **`mlfs.optim`**,
    the optimizer library you'll train models with for the rest of the course: the result
    record, the line search and gradient descent you write here are what later chapters call.

    | stage | you build |
    |---|---|
    | 1 | `OptimResult`, `gradient_descent` (fixed step): the result record and the basic loop → `mlfs.optim` |
    | 2 | `backtracking_line_search`: the Armijo rule → `mlfs.optim` |
    | 3 | `gradient_descent(..., line_search=True)`: gradient descent that picks its own step |
    | 4 | `exact_step`, `steepest_descent_quadratic`: the best possible step on a quadratic, and its zig-zag |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class OptimResult:
    """the result of a deterministic optimizer (every minimizer in mlfs.optim returns one).

    attributes: x (final iterate, float ndarray), fun (f(x), float), n_iters (int, number of
    steps taken), converged (bool), history (list of floats: f at every iterate, starting
    with f(x0), so len(history) == n_iters + 1).
    """

    def __init__(self, x, fun, n_iters, converged, history):
        """stage 1: store the five arguments as same-named attributes, converted:
        x -> float ndarray, fun -> float, n_iters -> int, converged -> bool, history -> list
        of floats.
        """
        raise NotImplementedError("stage1: implement OptimResult.__init__")

    def __repr__(self):
        return (f"OptimResult(fun={self.fun:.6g}, n_iters={self.n_iters}, converged={self.converged}, "
                f"x={np.array2string(self.x, precision=4, threshold=6)})")


@app.function
def gradient_descent(f, grad, x0, *, lr=0.01, line_search=False, max_iters=1000, tol=1e-06, callback=None):
    """stage 1 (line_search=False) and stage 3 (line_search=True): minimize f by gradient descent.

    f(x) -> float, grad(x) -> ndarray shaped like x; x0 is copied (never modified).
    loop: g = grad(x); stop (converged) if ||g||_2 <= tol; otherwise take a step
        line_search=False: x <- x - lr * g
        line_search=True:  alpha = backtracking_line_search(f, x, f(x), g, -g, alpha0=lr),
                           x <- x - alpha * g; if alpha == 0 (no acceptable step), stop.
    after each step k = 1, 2, ...: append f(x) to history and call callback(x, k) if given
    (its return value is ignored: there's no early stopping through the callback).
    also stop (not converged) after max_iters steps, or as soon as f(x) is not finite
    (diverged). call grad at most once per iterate, and f at most once per accepted iterate
    plus whatever the line search needs.
    returns OptimResult(x, f(x), n_iters = steps taken, converged = ||grad(x)|| <= tol at the
    returned x, history = [f(x0), f(x1), ...]).
    """
    if line_search:
        raise NotImplementedError("stage3: implement gradient_descent with line_search=True")
    raise NotImplementedError("stage1: implement gradient_descent with line_search=False")


@app.cell(hide_code=True)
def _():
    checks.run(1, OptimResult=OptimResult, gradient_descent=gradient_descent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def backtracking_line_search(f, x, fx, gx, direction, *, alpha0=1.0, rho=0.5, c=0.0001, max_iters=50):
    """stage 2: the Armijo backtracking line search.

    try alpha = alpha0, alpha0 * rho, alpha0 * rho^2, ... (at most max_iters trials) and
    return the first (largest) alpha with the sufficient-decrease condition
        f(x + alpha * direction) <= fx + c * alpha * (gx . direction).
    fx = f(x) and gx = grad f(x) are passed in (don't recompute them): one call of f per
    trial. returns a float; 0.0 if no trial passes (the caller should stop).
    """
    raise NotImplementedError("stage2: implement backtracking_line_search")


@app.cell(hide_code=True)
def _():
    checks.run(2, backtracking_line_search=backtracking_line_search)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    stage 3 is the `line_search=True` branch of `gradient_descent` above: go back and fill
    it in. its checks run here.
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, gradient_descent=gradient_descent, backtracking_line_search=backtracking_line_search)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def exact_step(A: np.ndarray, g: np.ndarray) -> float:
    """stage 4: the exact line-search step for a quadratic f(x) = x^T A x / 2 - b^T x.

    A (n, n) symmetric positive definite, g (n,) the gradient at the current point.
    returns the alpha minimizing f(x - alpha g), which is (g . g) / (g . A g), as a float.
    if g is exactly zero, return 0.0.
    """
    raise NotImplementedError("stage4: implement exact_step")


@app.function
def steepest_descent_quadratic(A: np.ndarray, b: np.ndarray, x0: np.ndarray, n_iters: int) -> np.ndarray:
    """stage 4: n_iters steps of gradient descent with exact line search on x^T A x / 2 - b^T x.

    A (n, n) symmetric positive definite, b (n,), x0 (n,) (not modified). each step:
    g = A x - b, x <- x - exact_step(A, g) * g. returns every iterate, shape (n_iters + 1, n),
    starting with x0.
    """
    raise NotImplementedError("stage4: implement steepest_descent_quadratic")


@app.cell(hide_code=True)
def _():
    checks.run(4, exact_step=exact_step, steepest_descent_quadratic=steepest_descent_quadratic)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your `OptimResult`, `backtracking_line_search` and
    `gradient_descent` are `mlfs.optim` (try `from mlfs.optim import gradient_descent` in a
    new notebook). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
