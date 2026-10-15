"""Checks for exercise.py (orthogonality and least squares).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np
import pytest

# the lesson's worked example: fit a line c + d t through (0, 6), (1, 0), (2, 0)
A_LINE = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
B_LINE = np.array([6.0, 0.0, 0.0])
S3, S2 = np.sqrt(3.0), np.sqrt(2.0)
Q_LINE = np.array([[1 / S3, -1 / S2], [1 / S3, 0.0], [1 / S3, 1 / S2]])
R_LINE = np.array([[S3, S3], [0.0, S2]])


def _with_condition(m, n, kappa, seed):
    """An (m, n) matrix with singular values spaced logarithmically from 1 down to 1/kappa."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return (U * np.logspace(0, -np.log10(kappa), n)) @ V.T


def _orth_error(Q):
    Q = np.asarray(Q)
    return np.abs(Q.T @ Q - np.eye(Q.shape[1])).max()


def _check_qr(A, Q, R, label, *, positive_diag=False):
    Q, R = np.asarray(Q, dtype=float), np.asarray(R, dtype=float)
    m, n = A.shape
    assert Q.shape == (m, n) and R.shape == (n, n), (
        f"{label}: for A of shape {A.shape}, expected Q {(m, n)} and R {(n, n)}, got {Q.shape} and {R.shape}")
    assert np.allclose(R, np.triu(R)), f"{label}: R should be upper-triangular"
    assert np.allclose(Q @ R, A, atol=1e-10), f"{label}: Q @ R should reproduce A (max error {np.abs(Q @ R - A).max():.2e})"
    assert _orth_error(Q) < 1e-10, f"{label}: the columns of Q should be orthonormal (Q.T @ Q = I), error {_orth_error(Q):.2e}"
    if positive_diag:
        assert np.all(np.diag(R) > 0), f"{label}: Gram-Schmidt gives R with a positive diagonal (R[j, j] = ||v_j||)"


# ---------------------------------------------------------------- stage 1: classical Gram-Schmidt


def test_stage1_worked_example(nb):
    Q, R = nb.gram_schmidt(A_LINE, modified=False)
    _check_qr(A_LINE, Q, R, "worked example", positive_diag=True)
    assert np.allclose(Q, Q_LINE) and np.allclose(R, R_LINE), (
        f"for A = [[1,0],[1,1],[1,2]], expected Q = (1,1,1)/sqrt3 and (-1,0,1)/sqrt2 and "
        f"R = [[sqrt3, sqrt3], [0, sqrt2]]. got R = {np.round(np.asarray(R), 4).tolist()}")


def test_stage1_random_tall(nb):
    for seed, shape in ((0, (10, 4)), (1, (6, 6)), (2, (30, 1))):
        A = np.random.default_rng(seed).normal(size=shape)
        Q, R = nb.gram_schmidt(A, modified=False)
        _check_qr(A, Q, R, f"random {shape}", positive_diag=True)


def test_stage1_does_not_modify_input(nb):
    A = np.random.default_rng(3).normal(size=(5, 3))
    A0 = A.copy()
    nb.gram_schmidt(A, modified=False)
    assert np.array_equal(A, A0), "gram_schmidt changed its input; work on a float copy"


# ---------------------------------------------------------------- stage 2: modified Gram-Schmidt


def test_stage2_worked_example_and_random(nb):
    Q, R = nb.gram_schmidt(A_LINE)  # modified=True is the default
    _check_qr(A_LINE, Q, R, "worked example (modified)", positive_diag=True)
    assert np.allclose(R, R_LINE), "in exact arithmetic modified GS gives the same R as classical GS"
    A = np.random.default_rng(4).normal(size=(12, 5))
    Q, R = nb.gram_schmidt(A, modified=True)
    _check_qr(A, Q, R, "random (12, 5), modified", positive_diag=True)


def test_stage2_modified_keeps_orthogonality(nb):
    A = _with_condition(40, 10, 1e8, seed=1)
    Qm, Rm = nb.gram_schmidt(A, modified=True)
    Qc, Rc = nb.gram_schmidt(A, modified=False)
    em, ec = _orth_error(Qm), _orth_error(Qc)
    assert np.allclose(np.asarray(Qm) @ np.asarray(Rm), A, atol=1e-12), "modified GS: Q @ R should still reproduce A"
    assert em < 1e-6, (
        f"on a matrix with condition number 1e8, modified GS should keep max|Q^T Q - I| around 1e-8, "
        f"got {em:.1e}. are you projecting out q_j from the CURRENT (already-updated) remaining "
        "columns, rather than from the original columns of A?")
    assert ec > 1e-5, (
        f"with modified=False you should run CLASSICAL GS, which loses orthogonality here "
        f"(about 1e-3); yours gives {ec:.1e}. classical GS computes all the coefficients for column j "
        "from the ORIGINAL a_j: R[:j, j] = Q[:, :j].T @ A[:, j].")


# ---------------------------------------------------------------- stage 3: Householder QR


def test_stage3_reduced_and_complete_shapes(nb):
    rng = np.random.default_rng(5)
    for shape in ((7, 4), (5, 5), (4, 7)):
        A = rng.normal(size=shape)
        m, n = shape
        k = min(m, n)
        Q, R = (np.asarray(M) for M in nb.qr(A))
        assert Q.shape == (m, k) and R.shape == (k, n), (
            f"qr(A) of shape {shape} in reduced mode should give Q {(m, k)} and R {(k, n)}, got {Q.shape}, {R.shape}")
        assert np.allclose(Q @ R, A) and _orth_error(Q) < 1e-12 and np.allclose(R, np.triu(R)), (
            f"reduced qr of a {shape} matrix: need Q @ R = A, orthonormal Q, upper-triangular R")
        Qc, Rc = (np.asarray(M) for M in nb.qr(A, mode="complete"))
        assert Qc.shape == (m, m) and Rc.shape == (m, n), (
            f"complete mode on {shape} should give Q {(m, m)} and R {(m, n)}, got {Qc.shape}, {Rc.shape}")
        assert np.allclose(Qc @ Rc, A) and _orth_error(Qc) < 1e-12 and np.allclose(Rc, np.triu(Rc)), (
            f"complete qr of a {shape} matrix: need Q @ R = A, square orthogonal Q, upper-triangular R")


def test_stage3_agrees_with_gram_schmidt_up_to_signs(nb):
    Q, R = (np.asarray(M) for M in nb.qr(A_LINE))
    assert np.allclose(np.abs(R), np.abs(R_LINE)), (
        f"Householder and Gram-Schmidt give the same R up to the sign of each row; expected |R| = "
        f"{np.round(np.abs(R_LINE), 4).tolist()}, got {np.round(R, 4).tolist()}")


def test_stage3_orthogonal_even_when_ill_conditioned(nb):
    A = _with_condition(40, 10, 1e12, seed=2)
    Q, R = (np.asarray(M) for M in nb.qr(A))
    err = _orth_error(Q)
    assert err < 1e-13, (
        f"Householder QR should give an orthonormal Q to machine precision whatever the condition "
        f"number, but max|Q^T Q - I| = {err:.1e} on a matrix with condition number 1e12. "
        "are you building Q by Gram-Schmidt instead of from the reflections?")
    assert np.allclose(Q @ R, A, atol=1e-14)


def test_stage3_handles_zero_column_and_bad_mode(nb):
    A = np.array([[1.0, 0.0, 2.0], [2.0, 0.0, 1.0], [2.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    Q, R = (np.asarray(M) for M in nb.qr(A))
    assert np.allclose(Q @ R, A) and _orth_error(Q) < 1e-12, (
        "qr should still work when a column is zero below the diagonal (skip that reflection)")
    with pytest.raises(ValueError):
        nb.qr(A, mode="economic")


def test_stage3_already_triangular_inputs(nb):
    for n in (1, 3, 5):
        Q, R = (np.asarray(M) for M in nb.qr(np.eye(n)))
        assert np.all(np.isfinite(Q)) and np.all(np.isfinite(R)), (
            f"qr(np.eye({n})) produced NaN or inf. when x = R[k:, k] already points along e1, "
            "v = x - ||x|| e1 is zero and normalizing it divides by zero: choose the sign so you ADD, "
            "v[0] += sign(x[0]) * ||x|| (with sign(0) = +1).")
        assert np.allclose(Q @ R, np.eye(n)) and _orth_error(Q) < 1e-12, f"qr(np.eye({n})): need Q R = I with orthonormal Q"
    U = np.triu(np.random.default_rng(15).uniform(1.0, 2.0, size=(4, 4)))  # positive diagonal
    Q, R = (np.asarray(M) for M in nb.qr(U))
    assert np.all(np.isfinite(Q)) and np.allclose(Q @ R, U) and _orth_error(Q) < 1e-12, (
        "qr of an upper-triangular matrix with a positive diagonal must work (every column is already "
        "'reflected'); check the Householder sign choice")


def test_stage3_does_not_modify_input(nb):
    A = np.random.default_rng(6).normal(size=(5, 3))
    A0 = A.copy()
    nb.qr(A)
    assert np.array_equal(A, A0), "qr changed its input; work on a float copy"


# ---------------------------------------------------------------- stage 4: least squares


def test_stage4_worked_example(nb):
    x = np.asarray(nb.lstsq(A_LINE, B_LINE))
    assert np.allclose(x, [5.0, -3.0]), (
        f"the best line through (0, 6), (1, 0), (2, 0) is 5 - 3t, so x = [5, -3]; got {np.round(x, 6).tolist()}")


def test_stage4_matches_numpy_and_residual_is_orthogonal(nb):
    rng = np.random.default_rng(7)
    A = rng.normal(size=(20, 5))
    b, B = rng.normal(size=20), rng.normal(size=(20, 3))
    x = np.asarray(nb.lstsq(A, b))
    assert x.shape == (5,) and np.allclose(x, np.linalg.lstsq(A, b, rcond=None)[0]), (
        "lstsq(A, b) should match np.linalg.lstsq")
    assert np.abs(A.T @ (b - A @ x)).max() < 1e-10, (
        "at the least-squares solution the residual b - A x is orthogonal to every column of A")
    X = np.asarray(nb.lstsq(A, B))
    assert X.shape == (5, 3) and np.allclose(X, np.linalg.lstsq(A, B, rcond=None)[0]), (
        "lstsq with a (m, k) right-hand side should solve each column, returning (n, k)")


def test_stage4_square_system_is_exact(nb):
    A = np.random.default_rng(8).normal(size=(6, 6))
    b = np.arange(6.0)
    assert np.allclose(A @ np.asarray(nb.lstsq(A, b)), b), "for square invertible A, lstsq solves A x = b exactly"


def test_stage4_accurate_on_ill_conditioned_fit(nb):
    t = np.linspace(0, 1, 30)
    A = np.vander(t, 12, increasing=True)  # condition number ~1e8
    x_true = np.random.default_rng(9).normal(size=12)
    x = np.asarray(nb.lstsq(A, A @ x_true))
    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    assert err < 1e-6, (
        f"fitting a degree-11 polynomial (condition number ~1e8) to exact data should recover the "
        f"coefficients to ~1e-8 with QR, but the relative error is {err:.1e}. if it's around 0.1, you "
        "solved the normal equations A^T A x = A^T b, which squares the condition number.")
