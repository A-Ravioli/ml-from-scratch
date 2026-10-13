"""Checks for exercise.py (solving linear systems).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import numpy as np
import pytest

LinAlgError = np.linalg.LinAlgError

# the lesson's worked example
A_WORKED = np.array([[2.0, 1.0, 1.0], [4.0, -6.0, 0.0], [-2.0, 7.0, 2.0]])
B_WORKED = np.array([5.0, -2.0, 9.0])
S_WORKED = np.array([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]])


def _random_lower(n, seed):
    rng = np.random.default_rng(seed)
    return np.tril(rng.normal(size=(n, n))) + 3 * np.eye(n)


def _random_spd(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n))
    return X @ X.T + n * np.eye(n)


# ---------------------------------------------------------------- stage 1: triangular solves


def test_stage1_lower_hand_example(nb):
    L = np.array([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, -1.0, 5.0]])
    b = np.array([2.0, 7.0, 13.0])
    got = np.asarray(nb.solve_triangular(L, b, lower=True))
    want = np.array([1.0, 2.0, 2.2])
    assert got.shape == (3,), f"b has shape (3,), so x should too; got shape {got.shape}"
    assert np.allclose(got, want), (
        f"forward substitution on L = {L.tolist()}, b = {b.tolist()} should give {want.tolist()}, "
        f"got {np.round(got, 6).tolist()}. x[0] = 2/2 = 1, then x[1] = (7 - 1*1)/3 = 2, then "
        "x[2] = (13 - 4*1 + 1*2)/5 = 2.2.")


def test_stage1_upper_hand_example(nb):
    U = np.array([[4.0, -6.0, 0.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]])
    y = np.array([-2.0, 6.0, 2.0])
    got = np.asarray(nb.solve_triangular(U, y, lower=False))
    assert np.allclose(got, [1.0, 1.0, 2.0]), (
        f"back substitution on the worked example's U should give [1, 1, 2], got "
        f"{np.round(got, 6).tolist()}. with lower=False, start from the LAST row and work up.")


def test_stage1_matches_numpy_vector_and_matrix_rhs(nb):
    rng = np.random.default_rng(0)
    L = _random_lower(8, 1)
    b, B = rng.normal(size=8), rng.normal(size=(8, 3))
    for lower, T in ((True, L), (False, L.T)):
        got_b = np.asarray(nb.solve_triangular(T, b, lower=lower))
        got_B = np.asarray(nb.solve_triangular(T, B, lower=lower))
        assert np.allclose(T @ got_b, b), f"lower={lower}: T @ x should equal b (vector right-hand side)"
        assert got_B.shape == (8, 3), (
            f"lower={lower}: b of shape (8, 3) holds 3 right-hand sides, so x should be (8, 3); "
            f"got {got_B.shape}")
        assert np.allclose(T @ got_B, B), (
            f"lower={lower}: T @ X should equal B for a (n, k) right-hand side. "
            "write the substitution so each row update works on whole rows of x, not scalars.")


def test_stage1_uses_only_its_triangle(nb):
    rng = np.random.default_rng(2)
    T = rng.normal(size=(5, 5)) + 4 * np.eye(5)
    b = rng.normal(size=5)
    got_lo = np.asarray(nb.solve_triangular(T, b, lower=True))
    got_up = np.asarray(nb.solve_triangular(T, b, lower=False))
    assert np.allclose(np.tril(T) @ got_lo, b), (
        "with lower=True, entries above the diagonal must be ignored (treat them as zero). "
        "your solve seems to read the upper triangle too.")
    assert np.allclose(np.triu(T) @ got_up, b), (
        "with lower=False, entries below the diagonal must be ignored (treat them as zero).")


def test_stage1_unit_diagonal_ignores_stored_diagonal(nb):
    L = np.array([[7.0, 0.0, 0.0], [2.0, 7.0, 0.0], [-1.0, 3.0, 7.0]])
    b = np.array([1.0, 4.0, 10.0])
    got = np.asarray(nb.solve_triangular(L, b, lower=True, unit_diagonal=True))
    L_unit = L.copy()
    np.fill_diagonal(L_unit, 1.0)
    assert np.allclose(L_unit @ got, b), (
        "unit_diagonal=True means 'pretend the diagonal is all ones and never read it'. "
        f"expected {np.linalg.solve(L_unit, b).tolist()}, got {np.round(got, 6).tolist()}.")


def test_stage1_zero_pivot_raises(nb):
    U = np.array([[1.0, 2.0], [0.0, 0.0]])
    with pytest.raises(LinAlgError):
        nb.solve_triangular(U, np.array([1.0, 1.0]), lower=False)


def test_stage1_does_not_modify_inputs(nb):
    L = _random_lower(4, 3)
    b = np.arange(1.0, 5.0)
    L0, b0 = L.copy(), b.copy()
    nb.solve_triangular(L, b, lower=True)
    assert np.array_equal(L, L0) and np.array_equal(b, b0), (
        "solve_triangular changed its inputs. work on a copy of b (np.array(b, dtype=float)).")


# ---------------------------------------------------------------- stage 2: LU with partial pivoting


def _check_plu(A, P, L, U, label):
    n = A.shape[0]
    P, L, U = (np.asarray(M, dtype=float) for M in (P, L, U))
    assert P.shape == L.shape == U.shape == (n, n), f"{label}: P, L, U should all be ({n}, {n})"
    assert np.array_equal(np.sort(P, axis=None)[-n:], np.ones(n)) and np.allclose(P.sum(0), 1) \
        and np.allclose(P.sum(1), 1), f"{label}: P should be a permutation matrix (one 1 per row and column)"
    assert np.allclose(L, np.tril(L)) and np.allclose(np.diag(L), 1), (
        f"{label}: L should be lower-triangular with ones on the diagonal")
    assert np.allclose(U, np.triu(U)), f"{label}: U should be upper-triangular (zeros below the diagonal)"
    err = np.abs(P @ A - L @ U).max()
    assert err < 1e-10 * max(1, np.abs(A).max()), (
        f"{label}: P @ A should equal L @ U, but they differ by {err:.2e}. if this is large, check "
        "that when you swap rows of U you also swap the already-computed part of L (columns :k).")


def test_stage2_worked_example(nb):
    P, L, U = nb.lu(A_WORKED)
    _check_plu(A_WORKED, P, L, U, "worked example")
    assert np.allclose(U, [[4, -6, 0], [0, 4, 1], [0, 0, 1]]), (
        f"for the worked example, U should be [[4,-6,0],[0,4,1],[0,0,1]], got {np.round(U, 6).tolist()}. "
        "partial pivoting puts the row with the largest |entry| in the pivot column on top: "
        "row 2 (the 4) goes first. on ties, keep the earliest row.")
    assert np.allclose(L, [[1, 0, 0], [0.5, 1, 0], [-0.5, 1, 1]]), (
        f"for the worked example, L should be [[1,0,0],[0.5,1,0],[-0.5,1,1]], got {np.round(L, 6).tolist()}")


def test_stage2_random_matrices(nb):
    for seed in range(3):
        A = np.random.default_rng(seed).normal(size=(7, 7))
        P, L, U = nb.lu(A)
        _check_plu(A, P, L, U, f"random 7x7 (seed {seed})")


def test_stage2_actually_pivots(nb):
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    P, L, U = nb.lu(A)
    _check_plu(A, P, L, U, "[[0, 1], [1, 1]]")
    A = np.random.default_rng(5).normal(size=(9, 9))
    _, L, _ = nb.lu(A)
    assert np.abs(np.asarray(L)).max() <= 1 + 1e-12, (
        f"with partial pivoting every multiplier satisfies |L[i, k]| <= 1, but max |L| is "
        f"{np.abs(np.asarray(L)).max():.3f}. pick the pivot as the LARGEST |entry| in the column "
        "at or below the diagonal, not just the first nonzero one.")


def test_stage2_does_not_modify_input_and_handles_singular(nb):
    A = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]])
    A0 = A.copy()
    P, L, U = nb.lu(A)
    assert np.array_equal(A, A0), "lu changed its input. factor a copy: U = np.array(A, dtype=float)."
    _check_plu(A0, P, L, U, "a singular matrix")


# ---------------------------------------------------------------- stage 3: solving


def test_stage3_worked_example(nb):
    x = np.asarray(nb.solve(A_WORKED, B_WORKED))
    assert np.allclose(x, [1.0, 1.0, 2.0]), (
        f"the worked example's solution is x = [1, 1, 2], got {np.round(x, 6).tolist()}. "
        "remember to permute b first: L U x = P b.")


def test_stage3_matches_numpy(nb):
    rng = np.random.default_rng(10)
    A = rng.normal(size=(12, 12))
    b, B = rng.normal(size=12), rng.normal(size=(12, 4))
    assert np.allclose(nb.solve(A, b), np.linalg.solve(A, b)), "solve(A, b) disagrees with np.linalg.solve"
    X = np.asarray(nb.solve(A, B))
    assert X.shape == (12, 4) and np.allclose(X, np.linalg.solve(A, B)), (
        "solve(A, B) with a (n, k) right-hand side should return (n, k), one solution per column")


def test_stage3_factor_once_solve_many(nb):
    rng = np.random.default_rng(11)
    A = rng.normal(size=(6, 6))
    factors = nb.lu(A)
    for _ in range(3):
        b = rng.normal(size=6)
        x = np.asarray(nb.lu_solve(factors, b))
        assert np.allclose(A @ x, b), "lu_solve(lu(A), b) should solve A x = b for any b"


def test_stage3_needs_pivoting(nb):
    A = np.array([[1e-20, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    x = np.asarray(nb.solve(A, b))
    assert np.allclose(x, [1.0, 1.0], atol=1e-12), (
        f"A = [[1e-20, 1], [1, 1]], b = [1, 2] has solution ~[1, 1], got {x.tolist()}. "
        "if x[0] came out 0, elimination used the tiny 1e-20 as a pivot: your lu isn't pivoting.")


def test_stage3_singular_raises(nb):
    A = np.array([[1.0, 2.0], [2.0, 4.0]])
    with pytest.raises(LinAlgError):
        nb.solve(A, np.array([1.0, 1.0]))


# ---------------------------------------------------------------- stage 4: determinant


def test_stage4_worked_example_sign(nb):
    d = nb.det(A_WORKED)
    assert np.isclose(d, -16.0), (
        f"det of the worked example is -16, got {d}. "
        + ("you got +16: you forgot that each row swap flips the sign (det P = ±1)."
           if np.isclose(d, 16.0) else "det = det(P) * prod(diag(U)), where det(P) = (-1)^(number of swaps)."))


def test_stage4_matches_numpy(nb):
    for seed in range(6):
        A = np.random.default_rng(seed).normal(size=(5, 5))
        got, want = nb.det(A), np.linalg.det(A)
        assert np.isclose(got, want, rtol=1e-9), f"seed {seed}: det should be {want:.6f}, got {got:.6f}"


def test_stage4_permutations_and_singular(nb):
    P3 = np.eye(4)[[1, 2, 0, 3]]   # a 3-cycle: even, det = +1
    P2 = np.eye(4)[[1, 0, 3, 2]]   # two swaps: even, det = +1
    P1 = np.eye(4)[[0, 1, 3, 2]]   # one swap: odd, det = -1
    for M, want in ((P3, 1.0), (P2, 1.0), (P1, -1.0)):
        got = nb.det(M)
        assert np.isclose(got, want), (
            f"det of the permutation matrix {M.argmax(1).tolist()} is {want:+.0f}, got {got}. "
            "count the row swaps your lu actually made, not the number of rows out of place.")
    assert np.isclose(nb.det(np.array([[1.0, 2.0], [2.0, 4.0]])), 0.0), "a singular matrix has det 0"


def test_stage4_multiplicative(nb):
    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    assert np.isclose(nb.det(A @ B), nb.det(A) * nb.det(B), rtol=1e-9), "det(AB) should equal det(A) det(B)"


# ---------------------------------------------------------------- stage 5: Cholesky


def test_stage5_worked_example(nb):
    L = np.asarray(nb.cholesky(S_WORKED))
    assert np.allclose(L, [[2, 0, 0], [1, 2, 0], [1, 1, 2]]), (
        f"the worked example's Cholesky factor is [[2,0,0],[1,2,0],[1,1,2]], got {np.round(L, 6).tolist()}")
    x = np.asarray(nb.cho_solve(L, np.array([8.0, 10.0, 11.0])))
    assert np.allclose(x, [1.0, 1.0, 1.0]), f"cho_solve(L, [8, 10, 11]) should give [1, 1, 1], got {x.tolist()}"


def test_stage5_matches_numpy(nb):
    S = _random_spd(10, 4)
    L = np.asarray(nb.cholesky(S))
    assert np.allclose(L, np.tril(L)), "cholesky should return a LOWER-triangular L"
    assert np.allclose(L @ L.T, S), "L @ L.T should reproduce A"
    assert np.allclose(L, np.linalg.cholesky(S)), (
        "your L reproduces A but isn't the standard factor: the diagonal of L must be positive")
    B = np.random.default_rng(5).normal(size=(10, 2))
    assert np.allclose(nb.cho_solve(L, B), np.linalg.solve(S, B)), "cho_solve(L, B) should solve (L L^T) X = B"


def test_stage5_rejects_non_spd(nb):
    with pytest.raises(LinAlgError):
        nb.cholesky(np.array([[1.0, 2.0], [2.0, 1.0]]))  # symmetric, eigenvalues 3 and -1
    with pytest.raises(LinAlgError):
        nb.cholesky(np.array([[2.0, 1.0], [0.0, 2.0]]))  # not symmetric
    with pytest.raises(LinAlgError):
        nb.cholesky(np.array([[1.0, 1.0], [1.0, 1.0]]))  # positive SEMI-definite only
    with pytest.raises(LinAlgError):
        nb.cholesky(1e-12 * np.array([[2.0, 1.0], [0.0, 2.0]]))  # tiny but clearly NOT symmetric
    S = 1e-12 * _random_spd(4, 6)
    assert np.allclose(np.asarray(nb.cholesky(S)) @ np.asarray(nb.cholesky(S)).T, S, rtol=1e-10, atol=0), (
        "a tiny SPD matrix (entries ~1e-12) must still factor. judge symmetry RELATIVE to the matrix's own "
        "size: np.allclose's default atol=1e-8 makes every 1e-12-scale matrix look symmetric.")
