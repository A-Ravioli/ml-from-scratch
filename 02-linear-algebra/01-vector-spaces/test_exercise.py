"""Checks for exercise.py (vector spaces).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np
import pytest

# the lesson's worked example: column 2 is twice column 1, so rank 2 with one free column
A_WORKED = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [1.0, 2.0, 4.0]])
R_WORKED = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])


def _low_rank(m, n, r, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, r)) @ rng.normal(size=(r, n))


def _check_rref_structure(R, pivots, label):
    R = np.asarray(R, dtype=float)
    pivots = list(pivots)
    assert pivots == sorted(pivots) and len(set(pivots)) == len(pivots), (
        f"{label}: pivots should be strictly increasing column indices, got {pivots}")
    for i, c in enumerate(pivots):
        col = np.zeros(R.shape[0])
        col[i] = 1.0
        assert np.allclose(R[:, c], col, atol=1e-9), (
            f"{label}: pivot column {c} should be the unit vector e_{i} (a 1 in row {i}, zeros "
            "elsewhere, ABOVE the pivot too: that's the 'reduced' in reduced row echelon form)")
        assert np.allclose(R[i, :c], 0, atol=1e-9), f"{label}: row {i} must be zero left of its pivot"
    assert np.allclose(R[len(pivots):], 0, atol=1e-9), f"{label}: rows below the last pivot must be zero"


# ---------------------------------------------------------------- stage 1: reduced row echelon form


def test_stage1_worked_example(nb):
    R, pivots = nb.rref(A_WORKED)
    R = np.asarray(R)
    assert tuple(pivots) == (0, 2), (
        f"the worked example has pivots in columns 0 and 2 (column 1 is 2x column 0), got {tuple(pivots)}")
    assert np.allclose(R, R_WORKED), (
        f"rref of the worked example is [[1,2,0],[0,0,1],[0,0,0]], got {np.round(R, 6).tolist()}")


def test_stage1_full_rank_square_reduces_to_identity(nb):
    A = np.random.default_rng(0).normal(size=(5, 5))
    R, pivots = nb.rref(A)
    assert tuple(pivots) == (0, 1, 2, 3, 4) and np.allclose(R, np.eye(5), atol=1e-9), (
        "an invertible matrix row-reduces to the identity, with a pivot in every column")


def test_stage1_rank_deficient_structure(nb):
    for (m, n, r, seed) in ((6, 5, 2, 1), (4, 7, 3, 2), (5, 5, 5, 3)):
        A = _low_rank(m, n, r, seed)
        R, pivots = nb.rref(A)
        _check_rref_structure(R, pivots, f"a rank-{r} matrix of shape {(m, n)}")
        assert len(pivots) == r, f"a rank-{r} matrix has {r} pivots, got {len(pivots)}"


def test_stage1_zero_matrix_and_no_mutation(nb):
    R, pivots = nb.rref(np.zeros((3, 4)))
    assert len(pivots) == 0 and np.allclose(R, 0), "the zero matrix has no pivots and stays zero"
    A = A_WORKED.copy()
    nb.rref(A)
    assert np.array_equal(A, A_WORKED), "rref changed its input; work on a float copy"
    R, _ = nb.rref(np.array([[2, 4], [1, 3]]))
    assert np.allclose(R, np.eye(2)), (
        "rref of the INTEGER matrix [[2,4],[1,3]] should be the identity; integer arrays truncate "
        "when you divide in place, so convert with np.array(A, dtype=float)")


# ---------------------------------------------------------------- stage 2: rank and span


def test_stage2_rank_examples(nb):
    assert nb.rank(A_WORKED) == 2, f"the worked example has rank 2, got {nb.rank(A_WORKED)}"
    assert nb.rank(np.eye(4)) == 4 and nb.rank(np.zeros((3, 3))) == 0
    for (m, n, r, seed) in ((8, 6, 3, 4), (3, 9, 2, 5), (7, 7, 7, 6)):
        A = _low_rank(m, n, r, seed)
        assert nb.rank(A) == r, f"rank of a random rank-{r} {(m, n)} matrix should be {r}, got {nb.rank(A)}"
        assert nb.rank(A.T) == r, "row rank equals column rank: rank(A.T) == rank(A)"


def test_stage2_rank_depends_on_tolerance(nb):
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    assert nb.rank(A) == 1, (
        "with the default tol=1e-10, [[1, 1], [1, 1 + 1e-12]] counts as rank 1: after elimination the "
        "second pivot is 1e-12, below the tolerance")
    assert nb.rank(A, tol=1e-14) == 2, "with tol=1e-14 the same matrix has rank 2: pass tol through to rref"


def test_stage2_in_span(nb):
    rng = np.random.default_rng(7)
    V = rng.normal(size=(5, 3))
    assert nb.in_span(V, V @ np.array([1.0, -2.0, 0.5])), "a combination of the columns is in their span"
    assert not nb.in_span(V, rng.normal(size=5)), "a random vector in R^5 is almost surely NOT in a 3-D subspace"
    assert nb.in_span(V, np.zeros(5)), "the zero vector is in every span"
    assert nb.in_span(np.eye(3), rng.normal(size=3)), "the columns of the identity span all of R^3"
    D = A_WORKED  # dependent columns: rank 2, column space = span of columns 0 and 2
    assert nb.in_span(D, D @ np.array([1.0, 1.0, -1.0])), (
        "x = A @ c is in the column space even when A's columns are dependent. compare rank([V | x]) "
        "with rank(V), not with the number of columns of V.")
    assert not nb.in_span(D, np.array([0.0, 0.0, 1.0])), "(0, 0, 1) is not in the worked example's column space"


def test_stage1_picks_the_largest_pivot(nb):
    A = np.array([[1e-8, 1.0, 1.0], [1.0, 1.0, 2.0]])  # [V | x]: the solution of V c = x is ~(1, 1)
    R, pivots = nb.rref(A)
    exact = np.linalg.solve(A[:, :2], A[:, 2])
    err = np.abs(np.asarray(R)[:, 2] - exact).max()
    assert tuple(pivots) == (0, 1) and err < 1e-12, (
        f"rref of [[1e-8, 1, 1], [1, 1, 2]] should end in the column {exact.tolist()} to ~1e-16, but the "
        f"error is {err:.1e}. if it's around 1e-8, you used the first entry above tol as the pivot; "
        "partial pivoting picks the LARGEST |entry| in the column, here the 1 in row 1.")


# ---------------------------------------------------------------- stage 3: bases and coordinates


def test_stage3_is_basis(nb):
    assert nb.is_basis(np.eye(3)), "the standard basis is a basis"
    assert nb.is_basis(np.random.default_rng(8).normal(size=(4, 4))), "4 random vectors in R^4 form a basis"
    assert not nb.is_basis(A_WORKED), "the worked example's columns are dependent: not a basis"
    assert not nb.is_basis(np.random.default_rng(9).normal(size=(3, 2))), (
        "2 vectors can't span R^3: too few for a basis, even though they're independent")
    assert not nb.is_basis(np.random.default_rng(10).normal(size=(3, 4))), (
        "4 vectors in R^3 must be dependent: too many for a basis, even though they span")


def test_stage3_coordinates(nb):
    V = np.array([[1.0, 1.0], [1.0, -1.0]])
    c = np.asarray(nb.coordinates(V, np.array([3.0, 1.0])))
    assert np.allclose(c, [2.0, 1.0]), (
        f"(3, 1) = 2 * (1, 1) + 1 * (1, -1), so its coordinates in this basis are [2, 1], got {c.tolist()}")
    rng = np.random.default_rng(11)
    V = rng.normal(size=(6, 6))
    x = rng.normal(size=6)
    c = np.asarray(nb.coordinates(V, x))
    assert c.shape == (6,) and np.allclose(V @ c, x), "coordinates c must satisfy V @ c == x"
    with pytest.raises(ValueError):
        nb.coordinates(A_WORKED, np.ones(3))


# ---------------------------------------------------------------- stage 4: null space


def test_stage4_worked_example(nb):
    N = np.asarray(nb.null_space(A_WORKED))
    assert N.shape == (3, 1), f"the worked example has rank 2, so its null space is 3 - 2 = 1 dimensional; got shape {N.shape}"
    assert np.allclose(N[:, 0], [-2.0, 1.0, 0.0]), (
        f"set the free variable x1 = 1 and solve for the pivots: x = (-2, 1, 0); got {N[:, 0].tolist()}")


def test_stage4_rank_nullity(nb):
    for (m, n, r, seed) in ((5, 8, 3, 12), (7, 4, 2, 13), (6, 6, 6, 14)):
        A = _low_rank(m, n, r, seed)
        N = np.asarray(nb.null_space(A))
        assert N.shape == (n, n - r), (
            f"rank-nullity: a rank-{r} matrix with {n} columns has a null space of dimension {n - r}; "
            f"expected shape {(n, n - r)}, got {N.shape}")
        if n > r:
            assert np.abs(A @ N).max() < 1e-8 * max(1, np.abs(A).max()), "every column of N must satisfy A @ v = 0"
            assert np.linalg.matrix_rank(N) == n - r, "the columns of N must be independent (a basis)"
