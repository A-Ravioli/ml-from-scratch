"""Checks for exercise.py (numpy arrays & vectorization).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import ast
import inspect
import textwrap
import time
import tracemalloc
import warnings

import numpy as np
import pytest

# the lesson's worked example
X_WORKED = np.array([[0.0, 0.0], [3.0, 4.0]])
Y_WORKED = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
D_WORKED = np.array([[0.0, 1.0, 3.0], [5.0, np.sqrt(20.0), 4.0]])


def _best_time(fn, *args, repeats=3):
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


_LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _python_loops(fn):
    """The loops (for, while, comprehensions) in fn's own source, or None if it can't be read."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return None
    return [f"{type(n).__name__} on line {n.lineno}" for n in ast.walk(tree) if isinstance(n, _LOOPS)]


def _assert_no_loops(fn, name, hint):
    loops = _python_loops(fn)
    if loops:
        raise AssertionError(
            f"{name} contains a Python loop ({', '.join(loops)}). this topic's functions must be "
            f"vectorized: no for or while loops, and no comprehensions (they're loops too). {hint}")


def _assert_as_fast_as(fn, baseline, args, name, hint, factor=20):
    """fn must run within factor x baseline (+5 ms), best of 3, on the same input."""
    t_base = _best_time(baseline, *args)
    t = _best_time(fn, *args)
    assert t <= factor * t_base + 0.005, (
        f"{name} took {t * 1e3:.1f} ms where whole-array NumPy takes {t_base * 1e3:.1f} ms on the same input "
        f"({t / t_base:.0f}x slower). {hint}")


def _brute_distances(X, Y):
    out = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            out[i, j] = np.sqrt(np.sum((X[i] - Y[j]) ** 2))
    return out


# ---------------------------------------------------------------- stage 1: elementwise ops and reductions


def test_stage1_batch_dot_hand_example(nb):
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    got = np.asarray(nb.batch_dot(A, B))
    assert got.shape == (2,), (
        f"batch_dot of two (2, 2) arrays should have shape (2,), one number per row; got shape {got.shape}. "
        + ("a (2, 2) result usually means you computed A @ B.T (every row against every row)."
           if got.shape == (2, 2) else "a single number means you summed over ALL entries: use axis=1."))
    assert np.allclose(got, [17.0, 53.0]), (
        f"row 0: 1*5 + 2*6 = 17, row 1: 3*7 + 4*8 = 53, so expected [17, 53], got {got.tolist()}. "
        + ("[26, 44] means you summed down the columns (axis=0) instead of across each row (axis=1)."
           if np.allclose(got, [26.0, 44.0]) else ""))


def test_stage1_batch_dot_matches_numpy(nb):
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(50, 7)), rng.normal(size=(50, 7))
    got = np.asarray(nb.batch_dot(A, B))
    assert np.allclose(got, np.einsum("ij,ij->i", A, B)), "batch_dot disagrees with the row-by-row dot products"
    got_int = np.asarray(nb.batch_dot([[1, 2]], [[3, 4]]))
    assert got_int.dtype.kind == "f" and np.allclose(got_int, [11.0]), (
        f"batch_dot([[1, 2]], [[3, 4]]) should return a FLOAT array [11.0], got {got_int!r}. "
        "convert the inputs with np.asarray(..., dtype=float).")


def test_stage1_batch_dot_rejects_shape_mismatch(nb):
    A = np.ones((3, 2))
    with pytest.raises(ValueError):
        nb.batch_dot(A, np.ones((1, 2)))  # broadcasting would silently "work" here: check shapes first
    with pytest.raises(ValueError):
        nb.batch_dot(A, np.ones((3, 3)))


def test_stage1_row_norms(nb):
    X = np.array([[3.0, 4.0], [0.0, 0.0], [-6.0, 8.0]])
    got = np.asarray(nb.row_norms(X))
    assert got.shape == (3,), f"row_norms of a (3, 2) array should have shape (3,), got {got.shape}"
    assert np.allclose(got, [5.0, 0.0, 10.0]), (
        f"the rows (3, 4), (0, 0), (-6, 8) have lengths 5, 0, 10; got {got.tolist()}. "
        + ("those are SQUARED lengths: take np.sqrt." if np.allclose(got, [25.0, 0.0, 100.0]) else ""))


def test_stage1_no_python_loops(nb):
    hint = "use A * B and np.sum(..., axis=1): one elementwise product, one reduction."
    _assert_no_loops(nb.batch_dot, "batch_dot", hint)
    rng = np.random.default_rng(1)
    A, B = rng.normal(size=(200_000, 3)), rng.normal(size=(200_000, 3))
    _assert_as_fast_as(nb.batch_dot, lambda a, b: np.einsum("ij,ij->i", a, b), (A, B), "batch_dot", hint)


# ---------------------------------------------------------------- stage 2: slicing and fancy indexing


def test_stage2_split_features_target(nb):
    data = np.arange(12.0).reshape(4, 3)
    X, y = nb.split_features_target(data)
    X, y = np.asarray(X), np.asarray(y)
    assert X.shape == (4, 2) and np.array_equal(X, data[:, :2]), (
        f"X should be every column but the last, shape (4, 2); got shape {X.shape}")
    assert y.shape == (4,), (
        f"y should be the last column as a 1-D array of shape (4,), got shape {y.shape}. "
        + ("data[:, -1:] keeps a column of shape (4, 1): index with -1, not the slice -1:."
           if y.shape == (4, 1) else ""))
    assert np.array_equal(y, [2.0, 5.0, 8.0, 11.0]), f"y should be [2, 5, 8, 11], got {y.tolist()}"


def test_stage2_split_returns_views(nb):
    data = np.arange(12.0).reshape(4, 3)
    X, y = nb.split_features_target(data)
    assert np.shares_memory(X, data) and np.shares_memory(y, data), (
        "X and y should be VIEWS into data (basic slicing never copies), but at least one is a copy. "
        "don't call .copy() or np.array(...), and use slices like data[:, :-1] rather than a list of "
        "column indices (a list is fancy indexing, which always copies).")
    data[0, -1] = 100.0
    assert y[0] == 100.0, "a view sees changes to the original array"


def test_stage2_one_hot_hand_example(nb):
    got = np.asarray(nb.one_hot(np.array([2, 0, 1, 2]), 3))
    want = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert got.shape == (4, 3), f"one_hot of 4 labels with 3 classes should have shape (4, 3), got {got.shape}"
    assert got.dtype.kind == "f", f"one_hot should return floats, got dtype {got.dtype}"
    assert np.array_equal(got, want), (
        f"one_hot([2, 0, 1, 2], 3) should be {want.astype(int).tolist()}, got {got.tolist()}. "
        "set out[np.arange(n), labels] = 1: row i gets its 1 in column labels[i].")
    assert np.asarray(nb.one_hot(np.array([], dtype=int), 4)).shape == (0, 4), "no labels -> shape (0, n_classes)"


def test_stage2_one_hot_rejects_bad_labels(nb):
    with pytest.raises(ValueError):
        nb.one_hot(np.array([0, 3]), 3)
    with pytest.raises(ValueError):
        nb.one_hot(np.array([0, -1]), 3)  # -1 would silently mean "the last column" in fancy indexing


def test_stage2_moving_average_hand_example(nb):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    got = np.asarray(nb.moving_average(x, 2))
    assert got.shape == (4,), (
        f"5 values with window 2 give 5 - 2 + 1 = 4 averages, got shape {got.shape}. "
        "put a 0 in front of the cumulative sum so that c[k] is the sum of the first k values.")
    assert np.allclose(got, [1.5, 2.5, 3.5, 4.5]), f"expected [1.5, 2.5, 3.5, 4.5], got {got.tolist()}"
    assert np.allclose(nb.moving_average(x, 1), x), "window 1 should return x itself"
    assert np.allclose(nb.moving_average(x, 5), [3.0]), "window 5 on 5 values is a single average, [3.0]"


def test_stage2_moving_average_matches_brute_force(nb):
    x = np.random.default_rng(2).normal(size=200)
    x0 = x.copy()
    got = np.asarray(nb.moving_average(x, 7))
    want = np.array([x[i:i + 7].mean() for i in range(200 - 6)])
    assert got.shape == want.shape and np.allclose(got, want), "moving_average disagrees with the mean of each window"
    assert np.array_equal(x, x0), "moving_average changed its input"
    for bad in (0, 201):
        with pytest.raises(ValueError):
            nb.moving_average(x, bad)


def test_stage2_no_python_loops(nb):
    _assert_no_loops(nb.one_hot, "one_hot",
                     "set every 1 at once with fancy indexing: out[np.arange(n), labels] = 1.0.")
    hint = "take ONE cumulative sum, then subtract two slices of it: c[window:] - c[:-window]."
    _assert_no_loops(nb.moving_average, "moving_average", hint)
    x = np.random.default_rng(3).normal(size=1_000_000)

    def baseline(x, w):
        c = np.concatenate([[0.0], np.cumsum(x)])
        return (c[w:] - c[:-w]) / w

    _assert_as_fast_as(nb.moving_average, baseline, (x, 50), "moving_average", hint)


# ---------------------------------------------------------------- stage 3: broadcasting


def test_stage3_standardize_hand_example(nb):
    X = np.array([[1.0, 10.0, 0.0], [3.0, 10.0, 4.0]])
    got = np.asarray(nb.standardize(X))
    want = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 1.0]])
    assert got.shape == X.shape, f"standardize should keep the shape {X.shape}, got {got.shape}"
    assert np.all(np.isfinite(got)), (
        "standardize produced nan or inf. the middle column is constant (std 0): divide it by 1 "
        "instead, so it becomes all zeros.")
    assert np.allclose(got, want), (
        f"each column should become (column - its mean) / its std: expected {want.tolist()}, got "
        f"{np.round(got, 4).tolist()}. "
        + ("you divided by the ddof=1 std (n - 1); use NumPy's default ddof=0."
           if np.allclose(np.abs(got[:, 0]), 1 / np.sqrt(2)) else ""))


def test_stage3_standardize_columns_independently(nb):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 5)) * np.array([1.0, 10.0, 0.1, 5.0, 2.0]) + np.array([0.0, 100.0, -3.0, 7.0, 1.0])
    X0 = X.copy()
    Z = np.asarray(nb.standardize(X))
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-12) and np.allclose(Z.std(axis=0), 1.0), (
        f"every COLUMN should end with mean 0 and std 1; got column means {np.round(Z.mean(axis=0), 4).tolist()} "
        f"and stds {np.round(Z.std(axis=0), 4).tolist()}. reduce with axis=0 (down the rows), not axis=1 "
        "or over the whole array.")
    assert np.array_equal(X, X0), (
        "standardize changed its input. X -= ... modifies the caller's array in place: compute a new "
        "array with X - mean instead.")


def test_stage3_standardize_integer_input(nb):
    X = np.array([[1, 2], [3, 6]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = np.asarray(nb.standardize(X))
    assert got.dtype.kind == "f" and np.allclose(got, [[-1, -1], [1, 1]]), (
        f"standardize([[1, 2], [3, 6]]) should give [[-1, -1], [1, 1]] as floats, got {got!r}")
    assert np.array_equal(X, [[1, 2], [3, 6]]), "standardize changed its integer input"


def test_stage3_standardize_constant_columns_exactly(nb):
    X = np.column_stack([np.full(7, 0.1), np.arange(7.0), np.full(7, 1e8 + 0.1), np.full(7, 3.3)])
    assert X[:, 0].std() != 0, "test setup: the std of a column of 0.1s is a tiny nonzero number, not 0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Z = np.asarray(nb.standardize(X))
    assert np.all(Z[:, [0, 2, 3]] == 0.0), (
        f"constant columns (all 0.1, all 1e8 + 0.1, all 3.3) must become exactly 0, got first rows "
        f"{np.round(Z[:3, [0, 2, 3]], 3).tolist()}. their computed std isn't 0 but ~1e-17 (0.1 isn't exact in "
        "binary, so the mean isn't exactly 0.1): std == 0 misses them and you divide rounding noise by "
        "rounding noise. detect constant columns exactly with X.max(axis=0) == X.min(axis=0).")
    assert np.allclose(Z[:, 1].std(), 1.0), "the non-constant column must still be standardized"


def test_stage3_pairwise_differences(nb):
    rng = np.random.default_rng(5)
    X, Y = rng.normal(size=(4, 3)), rng.normal(size=(6, 3))
    D = np.asarray(nb.pairwise_differences(X, Y))
    assert D.shape == (4, 6, 3), (
        f"X (4, 3) and Y (6, 3) should give differences of shape (4, 6, 3), got {D.shape}. "
        "reshape X to (4, 1, 3) and Y to (1, 6, 3) with None, then subtract.")
    assert np.allclose(D[2, 5], X[2] - Y[5]), (
        "D[i, j] should be X[i] - Y[j]. "
        + ("you computed Y[j] - X[i]: swap the order." if np.allclose(D[2, 5], Y[5] - X[2]) else ""))
    want = np.array([[X[i] - Y[j] for j in range(6)] for i in range(4)])
    assert np.allclose(D, want), "some entries of pairwise_differences are wrong"
    with pytest.raises(ValueError):
        nb.pairwise_differences(X, rng.normal(size=(6, 2)))
    with pytest.raises(ValueError):
        nb.pairwise_differences(X, rng.normal(size=(6, 1)))  # would broadcast silently: check the columns
    _assert_no_loops(nb.pairwise_differences, "pairwise_differences",
                     "reshape X to (n, 1, d) and Y to (1, m, d) with None, then subtract once.")


# ---------------------------------------------------------------- stage 4: pairwise distances


def test_stage4_worked_example(nb):
    got = np.asarray(nb.pairwise_distances(X_WORKED, Y_WORKED))
    assert got.shape == (2, 3), f"X has 2 rows and Y has 3, so the result should be (2, 3); got {got.shape}"
    if np.allclose(got, D_WORKED ** 2):
        raise AssertionError("you returned SQUARED distances with squared=False: take the square root.")
    assert np.allclose(got, D_WORKED), (
        f"expected {np.round(D_WORKED, 4).tolist()} (the lesson's worked example), got {np.round(got, 4).tolist()}. "
        "check the formula ||x||^2 - 2 x.y + ||y||^2: the norms of X go down the rows (shape (n, 1)) "
        "and the norms of Y go across the columns (shape (1, m)).")
    got_sq = np.asarray(nb.pairwise_distances(X_WORKED, Y_WORKED, squared=True))
    assert np.allclose(got_sq, D_WORKED ** 2), "with squared=True, return the squared distances (no square root)"


def test_stage4_matches_brute_force(nb):
    rng = np.random.default_rng(6)
    X, Y = rng.normal(size=(30, 5)), rng.normal(size=(20, 5)) + 1.0
    got = np.asarray(nb.pairwise_distances(X, Y))
    assert got.shape == (30, 20), f"expected shape (30, 20), got {got.shape}"
    assert np.allclose(got, _brute_distances(X, Y), rtol=1e-9, atol=1e-9), (
        "pairwise_distances disagrees with the double-loop definition on random points")
    got_int = np.asarray(nb.pairwise_distances(np.array([[0, 0], [1, 1]]), np.array([[3, 4]])))
    assert got_int.dtype.kind == "f" and np.allclose(got_int, [[5.0], [np.sqrt(13.0)]]), (
        f"integer inputs should give float distances [[5], [3.606]], got {got_int!r}")


def test_stage4_default_y_is_x(nb):
    X = np.random.default_rng(7).normal(size=(15, 4))
    D = np.asarray(nb.pairwise_distances(X))
    assert D.shape == (15, 15), f"pairwise_distances(X) compares X with itself: shape (15, 15), got {D.shape}"
    assert np.allclose(D, _brute_distances(X, X), atol=1e-7), "pairwise_distances(X) should equal pairwise_distances(X, X)"


def test_stage4_self_distances_exact_zero_diagonal_and_symmetry(nb):
    X_far = 1e3 + np.random.default_rng(17).normal(size=(40, 6))   # far from the origin: rounding shows
    X_near = np.random.default_rng(18).normal(size=(50, 7))
    for X, squared in ((X_far, False), (X_far, True), (X_near, False), (X_near, True)):
        D = np.asarray(nb.pairwise_distances(X, squared=squared))
        assert np.all(np.diag(D) == 0.0), (
            f"with Y omitted, every point is at distance EXACTLY 0 from itself, but the largest diagonal entry "
            f"is {np.abs(np.diag(D)).max():.3g} (squared={squared}). rounding in ||x||^2 - 2 x.x + ||x||^2 doesn't "
            "cancel exactly: when Y is None, set the diagonal with np.fill_diagonal(d2, 0.0).")
        assert np.array_equal(D, D.T), (
            f"with Y omitted the matrix must be EXACTLY symmetric, but D and D.T differ by up to "
            f"{np.abs(D - D.T).max():.3g} (squared={squared}): X @ X.T rounds slightly differently above and "
            "below the diagonal. average the two halves, d2 = (d2 + d2.T) / 2, before the square root.")


def test_stage4_never_nan_when_points_coincide(nb):
    X = 1e4 + np.random.default_rng(8).normal(size=(20, 3))   # points far from the origin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D = np.asarray(nb.pairwise_distances(X, X.copy()))
        D2 = np.asarray(nb.pairwise_distances(X, X.copy(), squared=True))
    assert np.all(np.isfinite(D)), (
        f"{np.sum(~np.isfinite(D))} distances came out nan. with coordinates near 1e4, rounding makes "
        "||x||^2 - 2 x.y + ||y||^2 slightly NEGATIVE for identical points, and sqrt of a negative is nan. "
        "clip at 0 first: np.maximum(d2, 0.0).")
    assert np.all(D >= 0) and np.all(D2 >= 0), "distances (and squared distances) must never be negative: clip at 0"
    assert np.allclose(np.diag(D), 0.0, atol=1e-2), "identical points should be at (nearly) zero distance"


def test_stage4_rejects_bad_shapes(nb):
    with pytest.raises(ValueError):
        nb.pairwise_distances(np.ones((3, 2)), np.ones((4, 3)))
    with pytest.raises(ValueError):
        nb.pairwise_distances(np.ones(3), np.ones(3))


def test_stage4_is_vectorized(nb):
    hint = ("use X @ Y.T for all the dot products at once, plus the squared norms as a column and a row: "
            "no loops over points.")
    _assert_no_loops(nb.pairwise_distances, "pairwise_distances", hint)
    rng = np.random.default_rng(9)
    X, Y = rng.normal(size=(600, 10)), rng.normal(size=(500, 10))

    def baseline(X, Y):
        d2 = (X * X).sum(1)[:, None] - 2 * X @ Y.T + (Y * Y).sum(1)[None, :]
        return np.sqrt(np.maximum(d2, 0.0))

    _assert_as_fast_as(nb.pairwise_distances, baseline, (X, Y), "pairwise_distances", hint)


def test_stage4_never_builds_the_difference_array(nb):
    rng = np.random.default_rng(10)
    n, m, d = 300, 300, 120
    X, Y = rng.normal(size=(n, d)), rng.normal(size=(m, d))
    tracemalloc.start()
    try:
        nb.pairwise_distances(X, Y)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    limit = 15 * n * m * 8   # a handful of (n, m) arrays
    assert peak < limit, (
        f"pairwise_distances used {peak / 1e6:.1f} MB of scratch memory for {n} x {m} points in {d} "
        f"dimensions; the expansion needs a few (n, m) arrays, under {limit / 1e6:.1f} MB. an (n, m, d) array of "
        f"differences takes {n * m * d * 8 / 1e6:.0f} MB: don't broadcast X[:, None] - Y[None] here. use "
        "||x||^2 - 2 x.y + ||y||^2 with one matrix product.")
