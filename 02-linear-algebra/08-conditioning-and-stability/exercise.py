import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import lu, lu_solve, solve_triangular, svd  # yours, from topics 03 and 06


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # conditioning & stability · build

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. `cond` and `cond_estimate` join
    `mlfs.linalg`. they use your `svd` (topic 06) and your `lu` / `lu_solve` /
    `solve_triangular` (topic 03). if those aren't finished, launch with
    `MLFS_REFERENCE=linalg marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `matrix_norm`: the 1-, 2-, ∞- and Frobenius norms of a matrix |
    | 2 | `cond`: the 2-norm condition number from the SVD |
    | 3 | `cond_estimate`: a cheap estimate from one LU and a few solves |
    | 4 | `backward_error`: how wrong is a computed solution, really? |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def matrix_norm(A: np.ndarray, ord=2) -> float:
    """stage 1: an induced or entrywise norm of a matrix A (m, n), as a float.

    ord=1: max absolute column sum. ord=np.inf: max absolute row sum.
    ord="fro": sqrt of the sum of squares. ord=2: the largest singular value (use svd).
    raises ValueError for any other ord.
    """
    raise NotImplementedError("stage1: implement matrix_norm")


@app.cell(hide_code=True)
def _():
    checks.run(1, matrix_norm=matrix_norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def cond(A: np.ndarray) -> float:
    """stage 2: the 2-norm condition number s_max / s_min of A (m, n), from your svd.

    returns np.inf if the smallest singular value is exactly zero.
    """
    raise NotImplementedError("stage2: implement cond")


@app.cell(hide_code=True)
def _():
    checks.run(2, cond=cond)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def cond_estimate(A: np.ndarray, *, max_iters: int = 100, rng=None) -> float:
    """stage 3: estimate cond(A) for square A (n, n) without an SVD.

    factor P A = L U once (return np.inf if U has an exactly zero pivot). then, from random
    unit vectors drawn from np.random.default_rng(rng):
      sigma_max^2: power iteration on A^T A (v <- A^T (A v), normalized), up to max_iters steps
      sigma_min^2: inverse iteration on A^T A (solve A^T y = v, then A x = y, with the LU factors)
    either loop may stop early once its estimate stops changing. returns sqrt(lam_max / lam_min).
    """
    raise NotImplementedError("stage3: implement cond_estimate")


@app.cell(hide_code=True)
def _():
    checks.run(3, cond_estimate=cond_estimate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def backward_error(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """stage 4: the normwise relative backward error of x as a solution of A x = b.

    eta = ||b - A x|| / (||A|| ||x|| + ||b||) with 2-norms (||A|| = matrix_norm(A, 2)).
    eta is the size of the smallest relative change to A and b that makes x an exact solution.
    """
    raise NotImplementedError("stage4: implement backward_error")


@app.cell(hide_code=True)
def _():
    checks.run(4, backward_error=backward_error)
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
