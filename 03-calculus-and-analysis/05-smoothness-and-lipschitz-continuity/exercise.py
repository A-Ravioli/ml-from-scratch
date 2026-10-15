import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import numerical_hessian  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # smoothness & lipschitz continuity · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you can estimate how
    fast a function (or its gradient) can change, and use that number to predict the
    largest step size gradient descent can take without blowing up.

    | stage | you build |
    |---|---|
    | 1 | `lipschitz_estimate`: the largest slope between any two sample points |
    | 2 | `smoothness_estimate`, `strong_convexity_estimate`: how fast the gradient changes, at most and at least |
    | 3 | `descent_lemma_gap`, `hessian_smoothness`: the quadratic upper bound, and $L$ from the Hessian |
    | 4 | `gradient_descent`, `largest_stable_step`: run GD and find its stability threshold |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def lipschitz_estimate(f, points: np.ndarray) -> float:
    """stage 1: the largest ||f(x_i) - f(x_j)|| / ||x_i - x_j|| over all pairs of rows i < j.

    f maps (n,) to a float or to an array (m,); points is (k, n) with k >= 2. pairs of
    identical points are skipped. vectorize over pairs (evaluate f once per point, then use
    broadcasting); no Python loop over pairs. this is a LOWER bound on the true constant.
    """
    raise NotImplementedError("stage1: implement lipschitz_estimate")


@app.cell(hide_code=True)
def _():
    checks.run(1, lipschitz_estimate=lipschitz_estimate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def smoothness_estimate(grad_f, points: np.ndarray) -> float:
    """stage 2: an estimate of the smoothness constant L of f: the Lipschitz constant of its
    gradient, max over pairs of ||grad_f(x_i) - grad_f(x_j)|| / ||x_i - x_j||.

    grad_f maps (n,) -> (n,); points (k, n). (a lower bound on the true L, like stage 1.)
    """
    raise NotImplementedError("stage2: implement smoothness_estimate")


@app.function
def strong_convexity_estimate(grad_f, points: np.ndarray) -> float:
    """stage 2: min over pairs i < j of (grad_f(x_i) - grad_f(x_j)) . (x_i - x_j) / ||x_i - x_j||^2.

    for a mu-strongly convex f every pair gives at least mu, so this is an UPPER bound on mu.
    a negative value proves f is not convex. identical points are skipped; vectorized.
    """
    raise NotImplementedError("stage2: implement strong_convexity_estimate")


@app.cell(hide_code=True)
def _():
    checks.run(2, smoothness_estimate=smoothness_estimate, strong_convexity_estimate=strong_convexity_estimate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def descent_lemma_gap(f, grad_f, L: float, x: np.ndarray, y: np.ndarray) -> float:
    """stage 3: how far the quadratic upper bound at x sits above f at y:

        f(x) + grad_f(x) . (y - x) + (L/2) ||y - x||^2 - f(y).

    if f is L-smooth this is >= 0 for every x, y; a negative value proves L is too small.
    """
    raise NotImplementedError("stage3: implement descent_lemma_gap")


@app.function
def hessian_smoothness(f, x: np.ndarray) -> float:
    """stage 3: the local smoothness constant at x: max |eigenvalue| of the Hessian of f at x.

    use numerical_hessian (from mlfs.autodiff) and np.linalg.eigvalsh. note the absolute
    value: a large NEGATIVE curvature also limits the step size.
    """
    raise NotImplementedError("stage3: implement hessian_smoothness")


@app.cell(hide_code=True)
def _():
    checks.run(3, descent_lemma_gap=descent_lemma_gap, hessian_smoothness=hessian_smoothness)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def gradient_descent(grad_f, x0: np.ndarray, step: float, n_iters: int) -> np.ndarray:
    """stage 4: n_iters steps of x <- x - step * grad_f(x) from x0; returns every iterate,
    shape (n_iters + 1, n), starting with x0. x0 is not modified.
    """
    raise NotImplementedError("stage4: implement gradient_descent")


@app.function
def largest_stable_step(grad_f, x0: np.ndarray, steps, *, n_iters: int = 100, tol: float = 1e-3) -> float:
    """stage 4: the largest step in `steps` for which gradient descent converges.

    a step counts as stable if, after n_iters iterations of your gradient_descent from x0,
    the last iterate is finite and ||grad_f(last)|| <= tol * ||grad_f(x0)|| (RELATIVE to the
    start): real progress, so both divergence and endless oscillation fail. try EVERY step,
    independently (steps needn't be sorted; don't stop at the first failure). returns 0.0 if
    no step is stable.
    """
    raise NotImplementedError("stage4: implement largest_stable_step")


@app.cell(hide_code=True)
def _():
    checks.run(4, gradient_descent=gradient_descent, largest_stable_step=largest_stable_step)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
