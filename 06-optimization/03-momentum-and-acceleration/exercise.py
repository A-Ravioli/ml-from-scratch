import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.optim import OptimResult  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # momentum & acceleration · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. by the end you have the two
    classic accelerated methods in `mlfs.optim`, a way to predict their speed on any
    quadratic before running them, and the function that proves no first-order method can
    do much better.

    | stage | you build |
    |---|---|
    | 1 | `heavy_ball`: gradient descent plus a velocity → `mlfs.optim` |
    | 2 | `nesterov`: look ahead, then step → `mlfs.optim` |
    | 3 | `momentum_spectral_radius`, `tuned_momentum`: the rate on a quadratic, and the parameters that optimize it |
    | 4 | `worst_case_quadratic`, `first_order_lower_bound`: the function no first-order method can solve fast |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def heavy_ball(f, grad, x0, *, lr=0.01, momentum=0.9, max_iters=1000, tol=1e-06, callback=None):
    """stage 1: Polyak's heavy-ball method.

        x_{k+1} = x_k - lr * grad(x_k) + momentum * (x_k - x_{k-1}),   with x_{-1} = x_0

    (so the first step is plain gradient descent). same loop conventions as your
    gradient_descent: x0 is copied; stop (converged) when ||grad(x_k)||_2 <= tol, after
    max_iters steps, or as soon as f(x) is not finite; one grad call per iterate; history
    = [f(x0), f(x1), ...]; callback(x, k) after step k = 1, 2, ... (its return
    value is ignored); each iterate a new array.
    returns OptimResult (from mlfs.optim).
    """
    raise NotImplementedError("stage1: implement heavy_ball")


@app.cell(hide_code=True)
def _():
    checks.run(1, heavy_ball=heavy_ball)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def nesterov(f, grad, x0, *, lr=0.01, momentum=None, max_iters=1000, tol=1e-06, callback=None):
    """stage 2: Nesterov's accelerated gradient method.

    step k = 1, 2, ... turns x_{k-1} (and x_{k-2}, with x_{-1} = x_0) into x_k:
        beta = (k - 1) / (k + 2)  if momentum is None (the schedule for convex f), else momentum
        y    = x_{k-1} + beta * (x_{k-1} - x_{k-2})     (look ahead along the last step)
        x_k  = y - lr * grad(y)                         (a gradient step FROM y)
    (momentum=0.0 is a constant 0, i.e. plain gradient descent: test `is None`.) the
    stopping test uses the gradient at the iterate x_k, so there are two grad calls per
    step (at y and at x_k). otherwise the same conventions as heavy_ball.
    returns OptimResult.
    """
    raise NotImplementedError("stage2: implement nesterov")


@app.cell(hide_code=True)
def _():
    checks.run(2, nesterov=nesterov)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def momentum_spectral_radius(eigs, lr: float, momentum: float, *, nesterov: bool = False) -> float:
    """stage 3: the asymptotic convergence rate of a momentum method on a quadratic.

    eigs: the Hessian's eigenvalues (any iterable of floats). along an eigenvector with
    eigenvalue lam, the error obeys e_{k+1} = a e_k - b e_{k-1} with
        heavy ball: a = 1 + momentum - lr * lam,       b = momentum
        nesterov:   a = (1 + momentum) (1 - lr * lam), b = momentum (1 - lr * lam)
    whose rate is the larger |root| of z^2 - a z + b = 0 (the roots may be complex).
    returns the max over the eigenvalues, a float. the method converges iff it's < 1.
    vectorized over eigs (use complex square roots).
    """
    raise NotImplementedError("stage3: implement momentum_spectral_radius")


@app.function
def tuned_momentum(L: float, mu: float, *, nesterov: bool = False) -> tuple[float, float]:
    """stage 3: the textbook (lr, momentum) for an L-smooth, mu-strongly convex function.

    with kappa = L / mu:
        heavy ball: lr = 4 / (sqrt(L) + sqrt(mu))^2,  momentum = ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^2
        nesterov:   lr = 1 / L,                        momentum = (sqrt(kappa) - 1) / (sqrt(kappa) + 1)
    returns (lr, momentum) as floats.
    """
    raise NotImplementedError("stage3: implement tuned_momentum")


@app.cell(hide_code=True)
def _():
    checks.run(3, momentum_spectral_radius=momentum_spectral_radius, tuned_momentum=tuned_momentum,
               heavy_ball=heavy_ball, nesterov=nesterov)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def worst_case_quadratic(n: int, L: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """stage 4: Nesterov's worst-case function f(x) = x^T A x / 2 - b^T x in n dimensions.

    A = (L / 4) T, where T (n, n) is tridiagonal with 2 on the diagonal and -1 next to it;
    b = (L / 4) e_1 (only the first entry nonzero). f is convex and L-smooth (||T|| < 4), and
    its minimizer is x*_i = 1 - i / (n + 1) for i = 1..n. returns (A, b) as float arrays.
    """
    raise NotImplementedError("stage4: implement worst_case_quadratic")


@app.function
def first_order_lower_bound(L: float, R: float, k: int) -> float:
    """stage 4: Nesterov's lower bound 3 L R^2 / (32 (k + 1)^2).

    for any method whose iterates stay in x0 + span(grad f(x_0), ..., grad f(x_{k-1})), on the
    worst-case quadratic with n = 2k + 1 and R = ||x0 - x*||, f(x_k) - f* is at least this.
    returns a float.
    """
    raise NotImplementedError("stage4: implement first_order_lower_bound")


@app.cell(hide_code=True)
def _():
    checks.run(4, worst_case_quadratic=worst_case_quadratic, first_order_lower_bound=first_order_lower_bound,
               heavy_ball=heavy_ball, nesterov=nesterov)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `heavy_ball` and `nesterov` are part of `mlfs.optim`. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
