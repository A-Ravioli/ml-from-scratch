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
    # eigenvalues & eigenvectors · build (reference solution)

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
    A = np.asarray(A, dtype=float)
    rng = np.random.default_rng(rng)
    v = rng.normal(size=A.shape[0])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(max_iters):
        w = A @ v
        lam = float(v @ w)  # Rayleigh quotient (v is a unit vector)
        norm_w = np.linalg.norm(w)
        if np.linalg.norm(w - lam * v) <= tol * max(norm_w, np.finfo(float).tiny):
            break
        v = w / norm_w
    return lam, v


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
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    rng = np.random.default_rng(rng)
    shift = float(mu)
    factors = lu(A - shift * np.eye(n))
    while np.any(np.diag(factors[2]) == 0):  # exactly singular: move off the eigenvalue
        shift += 1e-10 * (1.0 + abs(shift))
        factors = lu(A - shift * np.eye(n))
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    lam = float(v @ A @ v)
    for _ in range(max_iters):
        w = lu_solve(factors, v)
        v = w / np.linalg.norm(w)
        Av = A @ v
        lam = float(v @ Av)
        if np.linalg.norm(Av - lam * v) <= tol * max(np.linalg.norm(Av), np.finfo(float).tiny):
            break
    return lam, v


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
    Ak = np.array(A, dtype=float)
    scale = max(np.linalg.norm(Ak), np.finfo(float).tiny)
    for _ in range(max_iters):
        if np.linalg.norm(np.tril(Ak, -1)) <= tol * scale:
            break
        Q, R = qr(Ak)
        Ak = R @ Q
    return np.sort(np.diag(Ak))[::-1]


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
    A = np.array(A, dtype=float)
    A = (A + A.T) / 2  # make exactly symmetric
    n = A.shape[0]
    V = np.eye(n)
    scale = np.linalg.norm(A)
    negligible = np.finfo(float).eps ** 2 * scale  # far below anything that affects the result
    for _ in range(max_iters):
        if np.linalg.norm(A - np.diag(np.diag(A))) <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p, q]
                if abs(apq) <= negligible:  # also avoids overflow in tau below
                    continue
                tau = (A[q, q] - A[p, p]) / (2.0 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + np.hypot(1.0, tau))
                c = 1.0 / np.hypot(1.0, t)
                s = t * c
                # A <- J^T A J with J the rotation in the (p, q) plane: columns, then rows
                ap, aq = A[:, p].copy(), A[:, q].copy()
                A[:, p], A[:, q] = c * ap - s * aq, s * ap + c * aq
                ap, aq = A[p, :].copy(), A[q, :].copy()
                A[p, :], A[q, :] = c * ap - s * aq, s * ap + c * aq
                A[p, q] = A[q, p] = 0.0  # exactly, not 1e-17
                vp, vq = V[:, p].copy(), V[:, q].copy()
                V[:, p], V[:, q] = c * vp - s * vq, s * vp + c * vq
    w = np.diag(A).copy()
    order = np.argsort(w)
    return w[order], V[:, order]


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
