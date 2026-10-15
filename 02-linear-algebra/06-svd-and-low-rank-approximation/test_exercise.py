"""Checks for exercise.py (SVD and low-rank approximation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np
import pytest

A_WORKED = np.array([[3.0, 0.0], [4.0, 5.0]])  # singular values 3*sqrt(5) and sqrt(5)
S_WORKED = np.array([3 * np.sqrt(5.0), np.sqrt(5.0)])


def _with_singular_values(m, n, s, seed):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, len(s))))
    V, _ = np.linalg.qr(rng.normal(size=(n, len(s))))
    return (U * np.asarray(s, dtype=float)) @ V.T


def _orth_cols(M):
    M = np.asarray(M)
    return np.abs(M.T @ M - np.eye(M.shape[1])).max()


def _check_reduced(A, U, s, Vt, label):
    U, s, Vt = (np.asarray(x, dtype=float) for x in (U, s, Vt))
    m, n = A.shape
    r = min(m, n)
    assert U.shape == (m, r) and s.shape == (r,) and Vt.shape == (r, n), (
        f"{label}: for A of shape {A.shape} the reduced SVD has U {(m, r)}, s {(r,)}, Vt {(r, n)}; "
        f"got {U.shape}, {s.shape}, {Vt.shape}")
    assert np.all(s >= 0) and np.all(np.diff(s) <= 1e-12 * max(1, s[0])), (
        f"{label}: singular values must be non-negative and sorted descending, got {np.round(s, 6).tolist()}")
    assert np.allclose((U * s) @ Vt, A, atol=1e-12 * max(1, np.abs(A).max())), (
        f"{label}: U @ diag(s) @ Vt should reproduce A")
    assert _orth_cols(U) < 1e-12, f"{label}: the columns of U must be orthonormal (error {_orth_cols(U):.1e})"
    assert _orth_cols(Vt.T) < 1e-12, f"{label}: the rows of Vt must be orthonormal (error {_orth_cols(Vt.T):.1e})"


# ---------------------------------------------------------------- stage 1: Jacobi SVD of a square matrix


def _check_square(A, U, s, Vt, label):
    U, s, Vt = (np.asarray(x, dtype=float) for x in (U, s, Vt))
    n = A.shape[0]
    assert U.shape == (n, n) and s.shape == (n,) and Vt.shape == (n, n), (
        f"{label}: expected U ({n}, {n}), s ({n},), Vt ({n}, {n}); got {U.shape}, {s.shape}, {Vt.shape}")
    assert np.all(np.diff(s) <= 1e-12 * max(1, s[0])), f"{label}: sort s DESCENDING, got {np.round(s, 6).tolist()}"
    assert np.allclose((U * s) @ Vt, A, atol=1e-12 * max(1, np.abs(A).max())), (
        f"{label}: U @ diag(s) @ Vt should reproduce A. rotate the columns of V together with those of "
        "W (W = A V must stay true), and permute U, s and V together when you sort")
    assert _orth_cols(Vt.T) < 1e-12, f"{label}: V must be orthogonal (it's a product of rotations)"
    assert _orth_cols(U) < 1e-12, (
        f"{label}: the columns of U = W / s must be orthonormal (error {_orth_cols(U):.1e}). keep sweeping "
        "until a whole sweep makes no rotation.")


def test_stage1_worked_example(nb):
    U, s, Vt = nb.jacobi_svd_square(A_WORKED)
    _check_square(A_WORKED, U, s, Vt, "worked example")
    assert np.allclose(s, S_WORKED), (
        f"the singular values of [[3,0],[4,5]] are 3*sqrt5 = 6.708 and sqrt5 = 2.236, got {np.round(s, 4).tolist()}. "
        "one rotation with zeta = (beta - alpha) / (2 gamma) = 0, t = 1 makes the two columns orthogonal.")
    v1 = np.asarray(Vt)[0]
    assert abs(abs(v1 @ np.array([1, 1]) / np.sqrt(2)) - 1) < 1e-10, "the first right singular vector is ±(1, 1)/sqrt2"


def test_stage1_random_and_already_orthogonal(nb):
    A = np.random.default_rng(12).normal(size=(7, 7))
    U, s, Vt = nb.jacobi_svd_square(A)
    _check_square(A, U, s, Vt, "random 7x7")
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False)), "singular values disagree with NumPy"
    D = np.diag([-2.0, 1.0, 3.0])  # columns already orthogonal: no rotations needed, only sorting
    U, s, Vt = nb.jacobi_svd_square(D)
    _check_square(D, U, s, Vt, "diag(-2, 1, 3)")
    assert np.allclose(s, [3.0, 2.0, 1.0]), (
        f"diag(-2, 1, 3) has singular values (3, 2, 1): norms of the columns, sorted. got {np.round(s, 6).tolist()}")


def test_stage1_accurate_small_singular_values(nb):
    s_true = np.logspace(0, -12, 10)
    A = _with_singular_values(10, 10, s_true, seed=13)
    s = np.asarray(nb.jacobi_svd_square(A)[1])
    err = np.abs(s - s_true).max()
    assert err < 1e-14, (
        f"singular values from 1 down to 1e-12 should each be accurate to about 1e-16 (absolute), but the "
        f"worst error is {err:.1e}. check your stopping rule: skip a pair only when "
        "|gamma| <= n * eps * sqrt(alpha * beta), RELATIVE to the two columns' own sizes.")


def test_stage1_does_not_modify_input(nb):
    A = np.random.default_rng(14).normal(size=(4, 4))
    A0 = A.copy()
    nb.jacobi_svd_square(A)
    assert np.array_equal(A, A0), "jacobi_svd_square changed its input; rotate a float copy"


# ---------------------------------------------------------------- stage 2: the reduced SVD of any matrix


def test_stage2_worked_example(nb):
    U, s, Vt = nb.svd(A_WORKED)
    _check_reduced(A_WORKED, U, s, Vt, "worked example")
    assert np.allclose(s, S_WORKED), (
        f"the singular values of [[3,0],[4,5]] are 3*sqrt5 = 6.708 and sqrt5 = 2.236, got {np.round(s, 4).tolist()}")
    v1 = np.asarray(Vt)[0]
    assert abs(abs(v1 @ np.array([1, 1]) / np.sqrt(2)) - 1) < 1e-10, "the first right singular vector is ±(1, 1)/sqrt2"


def test_stage2_shapes_and_values(nb):
    rng = np.random.default_rng(0)
    for shape in ((8, 5), (5, 8), (6, 6), (1, 4), (4, 1)):
        A = rng.normal(size=shape)
        U, s, Vt = nb.svd(A)
        _check_reduced(A, U, s, Vt, f"random {shape}")
        assert np.allclose(s, np.linalg.svd(A, compute_uv=False)), f"random {shape}: singular values disagree with NumPy"


def test_stage2_rank_deficient(nb):
    A = _with_singular_values(7, 5, [4.0, 2.0, 1.0], seed=1)  # rank 3
    U, s, Vt = nb.svd(A)
    _check_reduced(A, U, s, Vt, "rank-3 matrix of shape (7, 5)")
    assert np.allclose(s, [4, 2, 1, 0, 0], atol=1e-12), (
        f"expected singular values [4, 2, 1, 0, 0], got {np.round(s, 12).tolist()}. even for the zero singular "
        "values, U needs orthonormal columns: you can't get them by dividing A v by sigma = 0.")
    Z = np.zeros((3, 2))
    U, s, Vt = nb.svd(Z)
    _check_reduced(Z, U, s, Vt, "the zero matrix")


def test_stage2_zero_column_first(nb):
    A = np.array([[0.0, 3.0], [0.0, 4.0]])
    U, s, Vt = nb.svd(A)
    _check_reduced(A, U, s, Vt, "[[0, 3], [0, 4]]")
    assert np.allclose(s, [5.0, 0.0]), (
        f"[[0, 3], [0, 4]] has singular values (5, 0), got {np.round(np.asarray(s), 6).tolist()}. if you got "
        "(4, 0), the QR of W saw the zero column FIRST: sort W's columns by decreasing norm before the QR.")
    B = _with_singular_values(4, 3, [3.0, 1.0], seed=11)  # 4x3, rank 2
    U, s, Vt = nb.svd(B)
    _check_reduced(B, U, s, Vt, "a rank-2 matrix of shape (4, 3)")
    assert np.allclose(s, [3.0, 1.0, 0.0], atol=1e-12), f"expected singular values (3, 1, 0), got {np.round(s, 12).tolist()}"


def test_stage2_accurate_small_singular_values(nb):
    s_true = np.logspace(0, -12, 10)
    A = _with_singular_values(20, 10, s_true, seed=2)
    s = np.asarray(nb.svd(A)[1])
    err = np.abs(s - s_true).max()
    assert err < 1e-14, (
        f"singular values from 1 down to 1e-12 should each be accurate to about 1e-16 (absolute), "
        f"but the worst error is {err:.1e}. if it's around 1e-8 or worse, you computed s as the square "
        "roots of the eigenvalues of A^T A, which squares the condition number: orthogonalize the "
        "columns of A directly (one-sided Jacobi) instead.")


def test_stage2_does_not_modify_input(nb):
    A = np.random.default_rng(3).normal(size=(4, 3))
    A0 = A.copy()
    nb.svd(A)
    assert np.array_equal(A, A0), "svd changed its input; work on a float copy"


# ---------------------------------------------------------------- stage 3: full_matrices=True


def test_stage3_full_shapes(nb):
    rng = np.random.default_rng(4)
    for shape in ((7, 4), (4, 7), (5, 5)):
        A = rng.normal(size=shape)
        m, n = shape
        r = min(m, n)
        U, s, Vt = (np.asarray(x) for x in nb.svd(A, full_matrices=True))
        assert U.shape == (m, m) and s.shape == (r,) and Vt.shape == (n, n), (
            f"full_matrices=True on {shape}: expected U {(m, m)}, s {(r,)}, Vt {(n, n)}; got "
            f"{U.shape}, {s.shape}, {Vt.shape}")
        assert _orth_cols(U) < 1e-12 and _orth_cols(Vt.T) < 1e-12, (
            f"full_matrices=True on {shape}: U and Vt must be square orthogonal matrices")
        assert np.allclose((U[:, :r] * s) @ Vt[:r], A), (
            f"full_matrices=True on {shape}: the first r columns/rows must still give A = U diag(s) Vt")


def test_stage3_extra_columns_span_the_null_spaces(nb):
    A = np.random.default_rng(5).normal(size=(6, 3))
    U, s, Vt = (np.asarray(x) for x in nb.svd(A, full_matrices=True))
    assert np.abs(A.T @ U[:, 3:]).max() < 1e-12, (
        "the last m - n columns of the full U are an orthonormal basis of the left null space: "
        "A^T U[:, n:] should be zero")
    U, s, Vt = (np.asarray(x) for x in nb.svd(A.T, full_matrices=True))
    assert np.abs(A.T @ Vt[3:].T).max() < 1e-12, (
        "for a wide matrix, the last n - m rows of the full Vt span its null space: A Vt[m:].T = 0")


# ---------------------------------------------------------------- stage 4: truncation, Eckart-Young and the pseudoinverse


def test_stage4_worked_example_rank_one(nb):
    A1 = np.asarray(nb.low_rank_approx(A_WORKED, 1))
    assert np.allclose(A1, [[1.5, 1.5], [4.5, 4.5]]), (
        f"the best rank-1 approximation of [[3,0],[4,5]] is [[1.5,1.5],[4.5,4.5]], got {np.round(A1, 6).tolist()}")
    assert np.isclose(np.linalg.norm(A_WORKED - A1, 2), np.sqrt(5.0)), "its 2-norm error should be sigma_2 = sqrt5"


def test_stage4_truncated_shapes_and_bounds(nb):
    A = np.random.default_rng(6).normal(size=(9, 6))
    U, s, Vt = (np.asarray(x) for x in nb.truncated_svd(A, 3))
    assert U.shape == (9, 3) and s.shape == (3,) and Vt.shape == (3, 6), (
        f"truncated_svd(A, 3) on a (9, 6) matrix should give (9, 3), (3,), (3, 6); got {U.shape}, {s.shape}, {Vt.shape}")
    assert np.allclose(s, np.linalg.svd(A, compute_uv=False)[:3]), "truncated_svd should keep the LARGEST k singular values"
    for bad in (0, 7):
        with pytest.raises(ValueError):
            nb.truncated_svd(A, bad)


def test_stage4_eckart_young(nb):
    rng = np.random.default_rng(7)
    A = rng.normal(size=(10, 7))
    s = np.linalg.svd(A, compute_uv=False)
    for k in (1, 3, 6):
        Ak = np.asarray(nb.low_rank_approx(A, k))
        assert Ak.shape == A.shape, "low_rank_approx returns a matrix of the same shape as A"
        assert np.linalg.matrix_rank(Ak) == k, f"low_rank_approx(A, {k}) should have rank {k}"
        assert np.isclose(np.linalg.norm(A - Ak, 2), s[k]), (
            f"Eckart-Young: the 2-norm error of the best rank-{k} approximation is sigma_{k + 1} = {s[k]:.4f}, "
            f"got {np.linalg.norm(A - Ak, 2):.4f}")
        assert np.isclose(np.linalg.norm(A - Ak), np.sqrt(np.sum(s[k:] ** 2))), (
            "the Frobenius error should be sqrt(sum of the discarded sigma_i^2)")
        for _ in range(5):  # no random rank-k matrix does better
            B = rng.normal(size=(10, k)) @ rng.normal(size=(k, 7))
            assert np.linalg.norm(A - B) >= np.linalg.norm(A - Ak) - 1e-12
    assert np.allclose(nb.low_rank_approx(A, 7), A), "keeping every singular value should reproduce A"


def test_stage4_pinv_worked_examples(nb):
    P = np.asarray(nb.pinv(A_WORKED))
    assert np.allclose(P, np.array([[5.0, 0.0], [-4.0, 3.0]]) / 15), (
        "for an invertible matrix the pseudoinverse is the inverse: pinv([[3,0],[4,5]]) = [[5,0],[-4,3]]/15")
    B = np.ones((2, 2))
    P = np.asarray(nb.pinv(B))
    assert np.allclose(P, B / 4), (
        f"pinv([[1,1],[1,1]]) = [[1,1],[1,1]]/4 (invert sigma_1 = 2, leave sigma_2 = 0 alone); got {np.round(P, 6).tolist()}. "
        "if you see huge numbers, you inverted a zero (or 1e-16) singular value.")


def test_stage4_penrose_conditions(nb):
    A = _with_singular_values(6, 4, [3.0, 1.0], seed=8)  # rank 2
    P = np.asarray(nb.pinv(A))
    assert P.shape == (4, 6), f"pinv of a (6, 4) matrix is (4, 6), got {P.shape}"
    assert np.allclose(A @ P @ A, A), "Penrose 1: A A+ A = A"
    assert np.allclose(P @ A @ P, P), "Penrose 2: A+ A A+ = A+"
    assert np.allclose((A @ P).T, A @ P) and np.allclose((P @ A).T, P @ A), "Penrose 3-4: A A+ and A+ A are symmetric"
    assert np.allclose(P, np.linalg.pinv(A)), "pinv disagrees with np.linalg.pinv"


def test_stage4_least_squares_and_rcond(nb):
    rng = np.random.default_rng(9)
    A, b = rng.normal(size=(12, 4)), rng.normal(size=12)
    x = np.asarray(nb.pinv(A)) @ b
    assert np.allclose(x, np.linalg.lstsq(A, b, rcond=None)[0]), "for full-rank tall A, pinv(A) @ b is the least-squares solution"
    C = 1e6 * _with_singular_values(5, 5, [1.0, 1e-10, 1e-10, 1e-10, 1e-10], seed=10)  # s = 1e6, 1e-4, ...
    P_loose = np.asarray(nb.pinv(C, rcond=1e-8))
    assert np.abs(P_loose).max() < 1e-4, (
        "with rcond=1e-8, singular values below 1e-8 * s_max = 1e-2 must be treated as zero, so pinv(C) has "
        "entries around 1e-6. the cutoff is RELATIVE: rcond times the largest singular value, not rcond itself.")
    P_tight = np.asarray(nb.pinv(C, rcond=1e-12))
    assert np.abs(P_tight).max() > 1e3, "with rcond=1e-12 the 1e-4 singular values are kept and inverted (entries ~1e4)"
    Z = np.asarray(nb.pinv(np.zeros((3, 2))))
    assert Z.shape == (2, 3) and np.all(Z == 0), "the pseudoinverse of a zero matrix is the zero matrix (transposed shape)"
