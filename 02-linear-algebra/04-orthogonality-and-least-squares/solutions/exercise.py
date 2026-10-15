import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import solve_triangular  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # orthogonality & least squares · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit. these functions join
    `mlfs.linalg`. stage 4 uses your `solve_triangular` from topic 03 through `mlfs.linalg`
    (if topic 03 isn't finished yet, launch with `MLFS_REFERENCE=linalg marimo edit exercise.py`
    to borrow the reference).

    | stage | you build |
    |---|---|
    | 1 | `gram_schmidt(A, modified=False)`: classical Gram–Schmidt, `A = Q R` |
    | 2 | `gram_schmidt(A, modified=True)`: the numerically better ordering |
    | 3 | `qr`: Householder QR, reduced and complete |
    | 4 | `lstsq`: least squares through QR |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def gram_schmidt(A: np.ndarray, *, modified: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """stage 1 (modified=False) and stage 2 (modified=True): Gram-Schmidt QR.

    A (m, n) with full column rank (m >= n). returns the reduced factorization
    (Q (m, n) with orthonormal columns, R (n, n) upper-triangular with positive diagonal),
    A == Q @ R. modified=False is classical GS: all of column j's coefficients come from the
    original a_j. modified=True removes each new q_j from all remaining columns right away.
    raises np.linalg.LinAlgError if a column is exactly dependent on earlier ones.
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    if modified:
        V = A.copy()  # V[:, j:] holds the not-yet-normalized remainders
        for j in range(n):
            R[j, j] = np.linalg.norm(V[:, j])
            if R[j, j] == 0:
                raise np.linalg.LinAlgError(f"column {j} depends on the earlier columns")
            Q[:, j] = V[:, j] / R[j, j]
            R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
            V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    else:
        for j in range(n):
            R[:j, j] = Q[:, :j].T @ A[:, j]
            v = A[:, j] - Q[:, :j] @ R[:j, j]
            R[j, j] = np.linalg.norm(v)
            if R[j, j] == 0:
                raise np.linalg.LinAlgError(f"column {j} depends on the earlier columns")
            Q[:, j] = v / R[j, j]
    return Q, R


@app.cell(hide_code=True)
def _():
    checks.run(1, gram_schmidt=gram_schmidt)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, gram_schmidt=gram_schmidt)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def qr(A: np.ndarray, *, mode: str = "reduced") -> tuple[np.ndarray, np.ndarray]:
    """stage 3: Householder QR of A (m, n), any shape, A == Q @ R.

    with k = min(m, n):
      mode="reduced":  Q (m, k) with orthonormal columns, R (k, n) upper-triangular
      mode="complete": Q (m, m) orthogonal, R (m, n) upper-triangular
    diag(R) may be negative. raises ValueError for any other mode. does not modify A.
    """
    if mode not in ("reduced", "complete"):
        raise ValueError(f"mode must be 'reduced' or 'complete', got {mode!r}")
    R = np.array(A, dtype=float)
    m, n = R.shape
    reflectors = []
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x == 0:  # nothing below (or on) the diagonal to zero out
            reflectors.append(None)
            continue
        v = x.copy()
        v[0] += norm_x if x[0] >= 0 else -norm_x  # same sign as x[0]: no cancellation
        v /= np.linalg.norm(v)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        reflectors.append(v)
    k_out = min(m, n)
    Q = np.eye(m) if mode == "complete" else np.eye(m, k_out)
    for k in reversed(range(len(reflectors))):  # Q = H_0 H_1 ... applied to the identity
        v = reflectors[k]
        if v is not None:
            Q[k:] -= 2.0 * np.outer(v, v @ Q[k:])
    R = np.triu(R)
    if mode == "complete":
        return Q, R
    return Q, R[:k_out]


@app.cell(hide_code=True)
def _():
    checks.run(3, qr=qr)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """stage 4: the least-squares solution argmin_x ||A x - b||_2, computed via QR.

    A (m, n) with m >= n and full column rank; b (m,) or (m, k). returns x (n,) or (n, k).
    """
    Q, R = qr(A)
    return solve_triangular(R, Q.T @ np.asarray(b, dtype=float), lower=False)


@app.cell(hide_code=True)
def _():
    checks.run(4, lstsq=lstsq)
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
