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
    # smoothness & lipschitz continuity · build (reference solution)

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
    P = np.asarray(points, dtype=float)
    F = np.array([np.atleast_1d(np.asarray(f(p), dtype=float)) for p in P])
    dx = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    df = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
    mask = np.triu(dx > 0, k=1)
    return float(np.max(df[mask] / dx[mask]))


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
    return lipschitz_estimate(grad_f, points)


@app.function
def strong_convexity_estimate(grad_f, points: np.ndarray) -> float:
    """stage 2: min over pairs i < j of (grad_f(x_i) - grad_f(x_j)) . (x_i - x_j) / ||x_i - x_j||^2.

    for a mu-strongly convex f every pair gives at least mu, so this is an UPPER bound on mu.
    a negative value proves f is not convex. identical points are skipped; vectorized.
    """
    P = np.asarray(points, dtype=float)
    G = np.array([np.asarray(grad_f(p), dtype=float) for p in P])
    D = P[:, None, :] - P[None, :, :]
    num = np.sum((G[:, None, :] - G[None, :, :]) * D, axis=-1)
    den = np.sum(D * D, axis=-1)
    mask = np.triu(den > 0, k=1)
    return float(np.min(num[mask] / den[mask]))


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y - x
    return float(f(x) + np.asarray(grad_f(x), dtype=float) @ d + 0.5 * L * (d @ d) - f(y))


@app.function
def hessian_smoothness(f, x: np.ndarray) -> float:
    """stage 3: the local smoothness constant at x: max |eigenvalue| of the Hessian of f at x.

    use numerical_hessian (from mlfs.autodiff) and np.linalg.eigvalsh. note the absolute
    value: a large NEGATIVE curvature also limits the step size.
    """
    return float(np.max(np.abs(np.linalg.eigvalsh(numerical_hessian(f, np.asarray(x, dtype=float))))))


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
    xs = [np.array(x0, dtype=float)]
    for _ in range(n_iters):
        xs.append(xs[-1] - step * np.asarray(grad_f(xs[-1]), dtype=float))
    return np.array(xs)


@app.function
def largest_stable_step(grad_f, x0: np.ndarray, steps, *, n_iters: int = 100, tol: float = 1e-3) -> float:
    """stage 4: the largest step in `steps` for which gradient descent converges.

    a step counts as stable if, after n_iters iterations of your gradient_descent from x0,
    the last iterate is finite and ||grad_f(last)|| <= tol * ||grad_f(x0)|| (RELATIVE to the
    start): real progress, so both divergence and endless oscillation fail. try EVERY step,
    independently (steps needn't be sorted; don't stop at the first failure). returns 0.0 if
    no step is stable.
    """
    g0 = np.linalg.norm(grad_f(np.asarray(x0, dtype=float)))
    best = 0.0
    for s in steps:
        with np.errstate(over="ignore", invalid="ignore"):
            last = gradient_descent(grad_f, x0, s, n_iters)[-1]
            ok = bool(np.all(np.isfinite(last))) and np.linalg.norm(grad_f(last)) <= tol * g0
        if ok:
            best = max(best, float(s))
    return best


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
