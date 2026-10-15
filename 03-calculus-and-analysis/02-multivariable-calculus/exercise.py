import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.numgrad import numerical_gradient  # yours, from ch 02 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # multivariable calculus · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `numerical_jacobian` and
    `numerical_hessian` become the first half of **`mlfs.autodiff`**: the finite-difference
    reference that topic 04's exact derivatives, and every Newton method after it, are
    checked against. gradients you already have: `numerical_gradient` from `mlfs.numgrad`.

    | stage | you build |
    |---|---|
    | 1 | `directional_derivative`, `steepest_direction`: the slope along any direction, and the direction of the steepest one |
    | 2 | `numerical_jacobian`: all first derivatives of a vector-valued map |
    | 3 | `numerical_hessian`: all second derivatives of a scalar function |
    | 4 | `classify_critical_point`: minimum, maximum, saddle or degenerate, from the Hessian |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def directional_derivative(f, x: np.ndarray, v: np.ndarray, *, h: float = 1e-6) -> float:
    """stage 1: the derivative of f at x along the vector v, by a central difference.

    f maps (n,) -> float; x, v are (n,). returns the float (f(x + h v) - f(x - h v)) / (2h),
    which approximates grad f(x) . v. v is NOT normalized: doubling v doubles the result.
    """
    raise NotImplementedError("stage1: implement directional_derivative")


@app.function
def steepest_direction(f, x: np.ndarray, *, h: float = 1e-6) -> tuple[np.ndarray, float]:
    """stage 1: the unit direction of steepest ascent of f at x, and the slope along it.

    uses numerical_gradient (from mlfs.numgrad) with step h. returns (u (n,), slope) with
    u = g / ||g|| and slope = ||g|| for g = grad f(x). raises ValueError if g == 0 (every
    direction is flat, so there is no steepest one).
    """
    raise NotImplementedError("stage1: implement steepest_direction")


@app.cell(hide_code=True)
def _():
    checks.run(1, directional_derivative=directional_derivative, steepest_direction=steepest_direction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def numerical_jacobian(f, x: np.ndarray, *, h: float = 1e-6) -> np.ndarray:
    """stage 2: the Jacobian of f: (n,) -> (m,) at x by central differences, shape (m, n).

    entry [i, j] approximates d f_i / d x_j, so column j is (f(x + h e_j) - f(x - h e_j)) / (2h).
    a scalar-valued f gives a (1, n) Jacobian. x is not modified (works for integer x too).
    """
    raise NotImplementedError("stage2: implement numerical_jacobian")


@app.cell(hide_code=True)
def _():
    checks.run(2, numerical_jacobian=numerical_jacobian)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def numerical_hessian(f, x: np.ndarray, *, h: float = 1e-4) -> np.ndarray:
    """stage 3: the Hessian of f: (n,) -> float at x by second differences, shape (n, n), symmetric.

    diagonal:     H[i, i] = (f(x + h e_i) - 2 f(x) + f(x - h e_i)) / h^2
    off-diagonal: H[i, j] = (f(x + h e_i + h e_j) - f(x + h e_i - h e_j)
                             - f(x - h e_i + h e_j) + f(x - h e_i - h e_j)) / (4 h^2), and H[j, i] = H[i, j].
    the default h = 1e-4 is larger than the gradient's 1e-6 on purpose (see the lesson).
    """
    raise NotImplementedError("stage3: implement numerical_hessian")


@app.cell(hide_code=True)
def _():
    checks.run(3, numerical_hessian=numerical_hessian)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def classify_critical_point(f, x: np.ndarray, *, tol: float = 1e-6) -> str:
    """stage 4: classify a critical point x of f: (n,) -> float from the eigenvalues of its Hessian.

    uses numerical_hessian and np.linalg.eigvalsh. with s = tol * max(1, max |eigenvalue|):
      "saddle"     if some eigenvalue > s and some < -s (whatever the others are),
      "minimum"    if every eigenvalue > s,
      "maximum"    if every eigenvalue < -s,
      "degenerate" otherwise (some eigenvalue is ~0, so second derivatives can't decide).
    x is assumed to be a critical point (gradient ~0); this isn't checked.
    """
    raise NotImplementedError("stage4: implement classify_critical_point")


@app.cell(hide_code=True)
def _():
    checks.run(4, classify_critical_point=classify_critical_point)
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
