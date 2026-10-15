import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import lu, lu_solve, qr  # yours, from topics 03 and 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # eigenvalues & eigenvectors · build

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. these functions join
    `mlfs.linalg`, and they use your `lu`, `lu_solve` (topic 03) and `qr` (topic 04). if those
    aren't finished, launch with `MLFS_REFERENCE=linalg marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `power_iteration`: the dominant eigenpair |
    | 2 | `inverse_iteration`: the eigenpair nearest a shift |
    | 3 | `qr_algorithm`: all eigenvalues at once (unshifted) |
    | 4 | `eigh`: every eigenpair of a symmetric matrix, by Jacobi rotations |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def power_iteration(A: np.ndarray, *, max_iters: int = 1000, tol: float = 1e-10,
                    rng=None) -> tuple[float, np.ndarray]:
    """stage 1: the eigenvalue of largest magnitude of A (n, n) and a unit eigenvector (n,).

    start from a random unit vector drawn from np.random.default_rng(rng) (rng: None, int seed
    or Generator), repeat v <- A v / ||A v||, estimate the eigenvalue by the Rayleigh quotient
    v^T A v. stop after max_iters steps, or once ||A v - lam v|| <= tol * ||A v||.
    returns (float eigenvalue, unit eigenvector).
    """
    raise NotImplementedError("stage1: implement power_iteration")


@app.cell(hide_code=True)
def _():
    checks.run(1, power_iteration=power_iteration)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def inverse_iteration(A: np.ndarray, mu: float = 0.0, *, max_iters: int = 1000,
                      tol: float = 1e-10, rng=None) -> tuple[float, np.ndarray]:
    """stage 2: the eigenvalue of A (n, n) closest to the shift mu, and a unit eigenvector (n,).

    power iteration with (A - mu I)^{-1}: factor A - mu I ONCE with lu, then each step solves
    with lu_solve. if A - mu I is exactly singular (mu is an eigenvalue), nudge mu slightly.
    eigenvalue by the Rayleigh quotient with A; same stopping rule and rng as power_iteration.
    """
    raise NotImplementedError("stage2: implement inverse_iteration")


@app.cell(hide_code=True)
def _():
    checks.run(2, inverse_iteration=inverse_iteration)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def qr_algorithm(A: np.ndarray, *, max_iters: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """stage 3: all eigenvalues of A (n, n), assumed real, by the unshifted QR algorithm.

    repeat Q, R = qr(A_k); A_{k+1} = R Q. stop after max_iters steps, or once the strictly
    lower-triangular part of A_k has Frobenius norm <= tol * ||A||_F.
    returns the diagonal of the final A_k as an (n,) array sorted DESCENDING.
    """
    raise NotImplementedError("stage3: implement qr_algorithm")


@app.cell(hide_code=True)
def _():
    checks.run(3, qr_algorithm=qr_algorithm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def eigh(A: np.ndarray, *, max_iters: int = 10000, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """stage 4: eigenvalues and eigenvectors of a symmetric A (n, n), by cyclic Jacobi rotations.

    returns (w, V): w (n,) eigenvalues sorted ASCENDING (like np.linalg.eigh), V (n, n) with
    orthonormal columns and A @ V == V @ diag(w). each sweep rotates every pair (p, q), p < q,
    once to zero A[p, q]. max_iters caps the number of sweeps; stop when the off-diagonal part
    has Frobenius norm <= tol * ||A||_F. does not modify A.
    """
    raise NotImplementedError("stage4: implement eigh")


@app.cell(hide_code=True)
def _():
    checks.run(4, eigh=eigh)
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
