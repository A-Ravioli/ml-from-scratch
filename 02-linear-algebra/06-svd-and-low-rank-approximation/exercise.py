import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import qr  # yours, from topic 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # SVD & low-rank approximation · build

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. these functions join
    `mlfs.linalg`. `svd` uses your Householder `qr` from topic 04 (if that isn't finished,
    launch with `MLFS_REFERENCE=linalg marimo edit exercise.py`). this chapter's eigenfaces
    capstone runs on the `svd` you write here.

    | stage | you build |
    |---|---|
    | 1 | `jacobi_svd_square`: the SVD of a square, full-rank matrix by one-sided Jacobi rotations |
    | 2 | `svd(A)`: any shape and any rank (transpose, QR first, QR of the rotated columns) |
    | 3 | `svd(A, full_matrices=True)`: complete orthonormal bases |
    | 4 | `truncated_svd`, `low_rank_approx`, `pinv`: what the SVD is for |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def jacobi_svd_square(A: np.ndarray, *, max_sweeps: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stage 1: the SVD of a square, full-rank A (n, n) by one-sided Jacobi: (U, s, Vt).

    start from W = A (a float copy) and V = I. sweep over all column pairs p < q: with
    alpha = ||w_p||^2, beta = ||w_q||^2, gamma = w_p . w_q, skip the pair if
    |gamma| <= n * eps * sqrt(alpha * beta) (already orthogonal) or if min(alpha, beta) <=
    (eps * ||A||_F)^2 (a numerically zero column); otherwise rotate columns p, q of BOTH W and
    V so that w_p . w_q = 0. stop after a sweep with no rotation (or max_sweeps sweeps).
    then s = column norms of W, sorted DESCENDING (permute W and V the same way), U = W / s,
    Vt = V.T. columns of U for exactly-zero s are left as zeros. does not modify A.
    """
    raise NotImplementedError("stage1: implement jacobi_svd_square")


@app.cell(hide_code=True)
def _():
    checks.run(1, jacobi_svd_square=jacobi_svd_square)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def svd(A: np.ndarray, *, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stage 2 (full_matrices=False) and stage 3 (full_matrices=True): the SVD A = U diag(s) Vt.

    A (m, n), any shape and any rank; r = min(m, n). s (r,) non-negative and sorted descending.
    full_matrices=False: U (m, r) and Vt (r, n) with orthonormal columns / rows.
    full_matrices=True:  U (m, m) and Vt (n, n) orthogonal; the extra columns of U (rows of Vt)
                         complete the bases.
    does not modify A.
    """
    if full_matrices:
        raise NotImplementedError("stage3: implement svd with full_matrices=True")
    raise NotImplementedError("stage2: implement svd with full_matrices=False")


@app.cell(hide_code=True)
def _():
    checks.run(2, svd=svd)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, svd=svd)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def truncated_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """stage 4: the top-k part of the SVD: U (m, k), s (k,), Vt (k, n).

    raises ValueError unless 1 <= k <= min(m, n).
    """
    raise NotImplementedError("stage4: implement truncated_svd")


@app.function
def low_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """stage 4: the best rank-k approximation of A (Eckart-Young), same shape as A."""
    raise NotImplementedError("stage4: implement low_rank_approx")


@app.function
def pinv(A: np.ndarray, *, rcond: float = 1e-15) -> np.ndarray:
    """stage 4: the Moore-Penrose pseudoinverse of A (m, n), shape (n, m).

    A+ = V diag(1/s) U^T, where singular values s <= rcond * max(s) (RELATIVE to the largest)
    are treated as zero: their reciprocal is replaced by 0, not 1/s.
    """
    raise NotImplementedError("stage4: implement pinv")


@app.cell(hide_code=True)
def _():
    checks.run(4, truncated_svd=truncated_svd, low_rank_approx=low_rank_approx, pinv=pinv)
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
