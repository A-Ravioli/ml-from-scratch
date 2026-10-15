"""Checks for exercise.py (linear maps and matmul).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
"""
import numpy as np

# the lesson's worked example
A_WORKED = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
B_WORKED = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])
C_WORKED = np.array([[5.0, 2.0, -1.0], [11.0, 4.0, -3.0], [17.0, 6.0, -5.0]])


def _pairs():
    rng = np.random.default_rng(0)
    yield rng.normal(size=(4, 7)), rng.normal(size=(7, 3))   # rectangular, all sizes distinct
    yield rng.normal(size=(1, 5)), rng.normal(size=(5, 1))   # row times column: a 1x1 result
    yield rng.normal(size=(6, 1)), rng.normal(size=(1, 2))   # column times row: an outer product
    yield rng.normal(size=(5, 5)), rng.normal(size=(5, 5))


def _check_matmul(fn, name):
    got = np.asarray(fn(A_WORKED, B_WORKED))
    assert got.shape == (3, 3), f"{name}: a (3, 2) matrix times a (2, 3) matrix is (3, 3), got {got.shape}"
    assert np.allclose(got, C_WORKED), (
        f"{name} on the worked example should give {C_WORKED.tolist()}, got {np.round(got, 6).tolist()}")
    for A, B in _pairs():
        got = np.asarray(fn(A, B))
        assert got.shape == (A.shape[0], B.shape[1]), (
            f"{name}: ({A.shape[0]}, {A.shape[1]}) @ ({B.shape[0]}, {B.shape[1]}) should have shape "
            f"{(A.shape[0], B.shape[1])}, got {got.shape}")
        assert np.allclose(got, A @ B), f"{name} disagrees with A @ B for shapes {A.shape} and {B.shape}"


# ---------------------------------------------------------------- stage 1: entries


def test_stage1_entries(nb):
    _check_matmul(nb.matmul_entries, "matmul_entries")


def test_stage1_integer_inputs(nb):
    A = np.array([[1, 2], [3, 4]])
    got = np.asarray(nb.matmul_entries(A, A))
    assert np.allclose(got, [[7, 10], [15, 22]]), f"[[1,2],[3,4]] squared is [[7,10],[15,22]], got {got.tolist()}"


# ---------------------------------------------------------------- stage 2: columns and rows


def test_stage2_columns(nb):
    _check_matmul(nb.matmul_columns, "matmul_columns")


def test_stage2_rows(nb):
    _check_matmul(nb.matmul_rows, "matmul_rows")


# ---------------------------------------------------------------- stage 3: outer products


def test_stage3_outer(nb):
    _check_matmul(nb.matmul_outer, "matmul_outer")


def test_stage3_rank_one_terms(nb):
    A = np.array([[1.0], [2.0]])
    B = np.array([[3.0, 4.0, 5.0]])
    got = np.asarray(nb.matmul_outer(A, B))
    assert np.allclose(got, [[3, 4, 5], [6, 8, 10]]), (
        "a (2, 1) times (1, 3) product is a single outer product: column of A times row of B")


# ---------------------------------------------------------------- stage 4: linear maps


def test_stage4_matrix_of_known_maps(nb):
    rot = lambda v: np.array([-v[1], v[0]])  # rotate by 90 degrees
    M = np.asarray(nb.matrix_of(rot, 2))
    assert np.allclose(M, [[0, -1], [1, 0]]), (
        f"rotation by 90 degrees sends e1 to (0, 1) and e2 to (-1, 0); those are the COLUMNS of "
        f"its matrix [[0, -1], [1, 0]], got {M.tolist()}")
    A = np.random.default_rng(1).normal(size=(4, 3))
    M = np.asarray(nb.matrix_of(lambda v: A @ v, 3))
    assert M.shape == (4, 3) and np.allclose(M, A), (
        f"matrix_of(lambda v: A @ v, 3) should recover A (shape (4, 3)), got shape {M.shape}")


def test_stage4_composition_is_product(nb):
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    f, g = (lambda v: A @ v), (lambda v: B @ v)
    M = np.asarray(nb.matrix_of(lambda v: f(g(v)), 2))
    assert np.allclose(M, A @ B), "the matrix of f(g(x)) is (matrix of f) @ (matrix of g): composition is multiplication"


def test_stage4_is_linear(nb):
    A = np.random.default_rng(3).normal(size=(3, 3))
    assert nb.is_linear(lambda v: A @ v, 3, rng=0), "x -> A x is linear"
    assert nb.is_linear(lambda v: np.array([v[1], -2 * v[0]]), 2, rng=0), "(x, y) -> (y, -2x) is linear"
    assert not nb.is_linear(lambda v: v + 1.0, 3, rng=0), (
        "x -> x + 1 is NOT linear (it's affine): it doesn't send 0 to 0, and f(2x) != 2 f(x)")
    assert not nb.is_linear(lambda v: v**2, 3, rng=0), "x -> x^2 (entrywise) is not linear"
    assert not nb.is_linear(lambda v: np.abs(v), 3, rng=0), (
        "x -> |x| is not linear: |x + y| != |x| + |y| for vectors of mixed signs. check f(a x + b y) == a f(x) + b f(y) "
        "at random x, y, a, b, not just scaling by positive numbers")


# ---------------------------------------------------------------- stage 5: blocked matmul


def test_stage5_blocked_matches(nb):
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(50, 37)), rng.normal(size=(37, 45))
    for block in (1, 8, 16, 64):
        got = np.asarray(nb.matmul_blocked(A, B, block=block))
        assert got.shape == (50, 45) and np.allclose(got, A @ B), (
            f"matmul_blocked with block={block} on (50, 37) @ (37, 45) disagrees with A @ B. sizes that "
            "aren't multiples of the block leave smaller tiles at the edges: slicing past the end of an "
            "array is fine in NumPy, so A[i:i + block] handles them.")


def test_stage5_worked_example_and_default(nb):
    got = np.asarray(nb.matmul_blocked(A_WORKED, B_WORKED, block=2))
    assert np.allclose(got, C_WORKED), "with block=2, the 3x2 @ 2x3 worked example needs ragged edge tiles"
    A = np.random.default_rng(5).normal(size=(70, 70))
    assert np.allclose(nb.matmul_blocked(A, A), A @ A), "the default block size should work too"
