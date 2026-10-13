"""Checks for exercise.py (conditioning and stability).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np
import pytest

A_WORKED = np.array([[1.0, 1.0], [1.0, 1.0001]])   # nearly singular, cond ~ 4e4
B_WORKED = np.array([2.0, 2.0001])                   # exact solution (1, 1)
N_WORKED = np.array([[1.0, -2.0], [3.0, 4.0]])       # for the norm examples


def _hilbert(n):
    i = np.arange(n)
    return 1.0 / (i[:, None] + i[None, :] + 1)


def _with_singular_values(n, s, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return (U * np.asarray(s, dtype=float)) @ V.T


# ---------------------------------------------------------------- stage 1: matrix norms


def test_stage1_hand_examples(nb):
    cases = {1: 6.0, np.inf: 7.0, "fro": np.sqrt(30.0), 2: np.linalg.norm(N_WORKED, 2)}
    hints = {1: "the 1-norm is the largest absolute COLUMN sum (|-2| + |4| = 6)",
             np.inf: "the inf-norm is the largest absolute ROW sum (|3| + |4| = 7)",
             "fro": "the Frobenius norm is sqrt(sum of squares) = sqrt(30)",
             2: "the 2-norm is the largest singular value (use your svd), not the Frobenius norm"}
    for ord_, want in cases.items():
        got = nb.matrix_norm(N_WORKED, ord_)
        assert np.isclose(got, want), f"matrix_norm([[1,-2],[3,4]], {ord_!r}) should be {want:.6f}, got {got}. {hints[ord_]}."


def test_stage1_default_is_two_norm_and_rectangular(nb):
    A = np.random.default_rng(0).normal(size=(7, 4))
    assert np.isclose(nb.matrix_norm(A), np.linalg.norm(A, 2)), "the default ord is 2: the largest singular value"
    for ord_ in (1, np.inf, "fro"):
        assert np.isclose(nb.matrix_norm(A, ord_), np.linalg.norm(A, ord_)), f"ord={ord_!r} wrong on a (7, 4) matrix"


def test_stage1_rejects_unknown_ord(nb):
    with pytest.raises(ValueError):
        nb.matrix_norm(N_WORKED, "nuc")


# ---------------------------------------------------------------- stage 2: condition number


def test_stage2_worked_example(nb):
    c = nb.cond(A_WORKED)
    assert np.isclose(c, 40002.0, rtol=1e-6), (
        f"cond([[1, 1], [1, 1.0001]]) = sigma_max / sigma_min = 2.00005 / 4.999875e-5 = 40002, got {c}")


def test_stage2_scale_does_not_matter(nb):
    Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(5, 5)))
    assert np.isclose(nb.cond(Q), 1.0), "an orthogonal matrix has condition number exactly 1"
    assert np.isclose(nb.cond(1e-10 * np.eye(4)), 1.0), (
        "1e-10 * I has a tiny determinant but condition number 1: cond measures distance to "
        "singularity relative to size, not size itself")


def test_stage2_singular_and_rectangular(nb):
    Z = np.array([[1.0, 0.0, 2.0], [3.0, 0.0, 1.0], [0.0, 0.0, 5.0]])
    assert nb.cond(Z) == np.inf, "a matrix with an exactly zero singular value has condition number inf"
    A = np.random.default_rng(2).normal(size=(9, 4))
    assert np.isclose(nb.cond(A), np.linalg.cond(A)), "for a tall full-rank matrix, cond = s_max / s_min of its reduced SVD"


def test_stage2_hilbert(nb):
    c = nb.cond(_hilbert(8))
    assert np.isclose(c, np.linalg.cond(_hilbert(8)), rtol=1e-5), (
        f"the 8x8 Hilbert matrix has condition number about 1.53e10, got {c:.3e}")


# ---------------------------------------------------------------- stage 3: condition estimation


def test_stage3_prescribed_condition_numbers(nb):
    for kappa, seed in ((1e3, 3), (1e8, 4)):
        s = np.logspace(0, -np.log10(kappa), 10)
        A = _with_singular_values(10, s, seed)
        est = nb.cond_estimate(A, rng=0)
        assert np.isclose(est, kappa, rtol=1e-6), (
            f"for a matrix with singular values from 1 to {1 / kappa:.0e}, cond_estimate should return "
            f"about {kappa:.0e}, got {est:.4e}. sigma_max^2 is the top eigenvalue of A^T A (power "
            "iteration with v <- A^T (A v)); sigma_min^2 is the bottom one (inverse iteration: solve "
            "A^T A x = v with your LU factors of A).")


def test_stage3_random_matrix_and_reproducible(nb):
    A = np.random.default_rng(5).normal(size=(30, 30))
    est = nb.cond_estimate(A, rng=1)
    assert np.isclose(est, np.linalg.cond(A), rtol=1e-3), (
        f"on a random 30x30 matrix the estimate should be within 0.1% of the true {np.linalg.cond(A):.4e}, got {est:.4e}")
    assert nb.cond_estimate(A, max_iters=3, rng=7) == nb.cond_estimate(A, max_iters=3, rng=np.random.default_rng(7)), (
        "the same seed (int or Generator) must give the same estimate")


def test_stage3_singular_gives_inf(nb):
    est = nb.cond_estimate(np.array([[0.0, 0.0], [0.0, 1.0]]), rng=0)
    assert est == np.inf, f"a matrix whose LU has an exactly zero pivot is singular: return inf, got {est}"


# ---------------------------------------------------------------- stage 4: backward error


def test_stage4_worked_example(nb):
    eta = nb.backward_error(A_WORKED, np.array([0.0, 2.0]), B_WORKED)
    assert np.isclose(eta, 1.4644e-5, rtol=1e-3), (
        f"x = (0, 2) has residual (0, -1e-4); eta = ||r|| / (||A|| ||x|| + ||b||) = 1e-4 / (2.00005 * 2 + 2.8285) "
        f"= 1.464e-5, got {eta}. use 2-norms throughout.")


def test_stage4_backward_stable_solve(nb):
    H = _hilbert(11)
    x_true = np.ones(11)
    b = H @ x_true
    x = np.linalg.solve(H, b)  # LAPACK's pivoted LU, the same algorithm as your solve
    eta = nb.backward_error(H, x, b)
    assert eta < 1e-14, (
        f"LU with partial pivoting is backward stable: even on the 11x11 Hilbert matrix (cond ~5e14) the "
        f"backward error of its solution should be ~1e-17, got {eta:.1e}")
    assert np.isclose(nb.backward_error(H, x_true, b), np.linalg.norm(b - H @ x_true)
                      / (np.linalg.norm(H, 2) * np.linalg.norm(x_true) + np.linalg.norm(b)))


def test_stage4_uses_the_two_norm(nb):
    eta = nb.backward_error(np.eye(4), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(4))
    assert np.isclose(eta, 1.0), (
        f"for A = I (4x4), x = e1, b = 0: ||r|| = 1, ||A||_2 = 1, ||x|| = 1, ||b|| = 0, so eta = 1; got {eta}. "
        + ("0.5 means you used the Frobenius norm of A (||I||_F = 2): use matrix_norm(A, 2)." if np.isclose(eta, 0.5) else ""))


def test_stage4_scale_invariant_and_zero_solution(nb):
    rng = np.random.default_rng(6)
    A, x, b = rng.normal(size=(5, 5)), rng.normal(size=5), rng.normal(size=5)
    e1, e2 = nb.backward_error(A, x, b), nb.backward_error(1e6 * A, x, 1e6 * b)
    assert np.isclose(e1, e2), "scaling A and b by the same factor must not change the (relative) backward error"
    assert np.isclose(nb.backward_error(A, np.zeros(5), b), 1.0), (
        "x = 0 has residual b, so its backward error is ||b|| / ||b|| = 1 (as wrong as possible)")
