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
    # solving linear systems · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. everything you write here
    becomes `mlfs.linalg`, which later chapters import to fit regressions, sample Gaussians,
    and take Newton steps.

    | stage | you build |
    |---|---|
    | 1 | `solve_triangular`: forward and back substitution |
    | 2 | `lu`: Gaussian elimination with partial pivoting, as `P A = L U` |
    | 3 | `lu_solve`, `solve`: solve `A x = b` from the factors |
    | 4 | `det`: the determinant, almost for free |
    | 5 | `cholesky`, `cho_solve`: the symmetric positive definite special case |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def solve_triangular(T: np.ndarray, b: np.ndarray, *, lower: bool = True,
                     unit_diagonal: bool = False) -> np.ndarray:
    """stage 1: solve T x = b for a triangular T (n, n).

    b is (n,) or (n, k) (k right-hand sides); x has the same shape as b.
    lower=True: forward substitution using only the lower triangle of T.
    lower=False: back substitution using only the upper triangle.
    unit_diagonal=True: treat the diagonal as all ones and never read it.
    raises np.linalg.LinAlgError on a zero diagonal entry. does not modify T or b.
    """
    T = np.asarray(T, dtype=float)
    x = np.array(b, dtype=float)  # a copy: we overwrite it row by row
    n = T.shape[0]
    rows = range(n) if lower else range(n - 1, -1, -1)
    for i in rows:
        if lower:
            x[i] -= T[i, :i] @ x[:i]
        else:
            x[i] -= T[i, i + 1:] @ x[i + 1:]
        if not unit_diagonal:
            if T[i, i] == 0:
                raise np.linalg.LinAlgError(f"zero pivot in row {i}: the matrix is singular")
            x[i] /= T[i, i]
    return x


@app.cell(hide_code=True)
def _():
    checks.run(1, solve_triangular=solve_triangular)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def lu(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stage 2: LU factorization with partial pivoting of a square A (n, n).

    returns (P, L, U), all (n, n) float arrays, with P @ A == L @ U:
      P: a permutation matrix
      L: unit lower-triangular (ones on the diagonal), every |L[i, j]| <= 1
      U: upper-triangular
    at step k, the pivot is the row (at or below k) with the largest |entry| in column k;
    ties go to the earliest row. singular matrices still factor (U gets a zero pivot).
    does not modify A.
    """
    U = np.array(A, dtype=float)
    n = U.shape[0]
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]  # multipliers computed so far travel with their rows
            perm[[k, p]] = perm[[p, k]]
        if U[k, k] == 0:  # the whole column below is zero: nothing to eliminate
            continue
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])
        U[k + 1:, k] = 0.0  # exact zeros, not 1e-17s
    P = np.eye(n)[perm]
    return P, L, U


@app.cell(hide_code=True)
def _():
    checks.run(2, lu=lu)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def lu_solve(factors: tuple[np.ndarray, np.ndarray, np.ndarray], b: np.ndarray) -> np.ndarray:
    """stage 3: solve A x = b given factors = (P, L, U) = lu(A).

    b is (n,) or (n, k); x has the same shape. two triangular solves, no new factorization.
    """
    P, L, U = factors
    y = solve_triangular(L, P @ np.asarray(b, dtype=float), lower=True, unit_diagonal=True)
    return solve_triangular(U, y, lower=False)


@app.function
def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """stage 3: solve A x = b for square A (n, n); b (n,) or (n, k). raises LinAlgError if singular."""
    return lu_solve(lu(A), b)


@app.cell(hide_code=True)
def _():
    checks.run(3, lu=lu, lu_solve=lu_solve, solve=solve)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def det(A: np.ndarray) -> float:
    """stage 4: the determinant of square A (n, n), from its LU factors.

    det(A) = det(P) * prod(diag(U)), with det(P) = +1 or -1 by the parity of the permutation.
    returns a Python float.
    """
    P, _, U = lu(A)
    perm = np.argmax(P, axis=1)  # row i of P A is row perm[i] of A
    sign = 1.0
    seen = np.zeros(len(perm), dtype=bool)
    for start in range(len(perm)):  # a cycle of length c takes c - 1 swaps
        length, j = 0, start
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0 and length > 0:
            sign = -sign
    return float(sign * np.prod(np.diag(U)))


@app.cell(hide_code=True)
def _():
    checks.run(4, det=det)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def cholesky(A: np.ndarray) -> np.ndarray:
    """stage 5: the Cholesky factor of a symmetric positive definite A (n, n).

    returns lower-triangular L (n, n) with positive diagonal and L @ L.T == A.
    raises np.linalg.LinAlgError if A is not symmetric (to a RELATIVE tolerance: a 1e-12-scale
    matrix must be judged on its own scale) or not positive definite.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or not np.allclose(A, A.T, rtol=1e-10, atol=1e-14 * np.abs(A).max()):
        raise np.linalg.LinAlgError("cholesky needs a symmetric matrix")  # checked RELATIVE to the size of A
    L = np.zeros((n, n))
    for j in range(n):
        d = A[j, j] - L[j, :j] @ L[j, :j]
        if d <= 0:
            raise np.linalg.LinAlgError(f"not positive definite: pivot {j} would be {d:.3g}")
        L[j, j] = np.sqrt(d)
        L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
    return L


@app.function
def cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """stage 5: solve (L @ L.T) x = b given the lower Cholesky factor L (n, n); b (n,) or (n, k)."""
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y, lower=False)


@app.cell(hide_code=True)
def _():
    checks.run(5, cholesky=cholesky, cho_solve=cho_solve)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your code is `mlfs.linalg` (try `from mlfs.linalg import solve`
    in a scratch cell). then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
