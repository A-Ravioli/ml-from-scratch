"""Checks for exercise.py (vectors & matrices, a first look).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. NumPy's own
operators (`@`, np.linalg) are the oracles here; your code shouldn't use them.
"""
import ast
import inspect
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pytest

FORBIDDEN_CALLS = {"dot", "matmul", "inner", "einsum", "vdot", "tensordot"}


def _shortcuts(fn):
    """the built-in product shortcuts used in fn's code (docstrings and comments don't count)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        pytest.skip(f"can't read the source of {getattr(fn, '__name__', fn)} to check it")
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", None) == "NotImplementedError"):
            raise NotImplementedError("not started")  # an untouched stub: report it as todo, not a pass
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            found.add("@")
        elif isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_CALLS or node.attr == "linalg"):
            found.add(f".{node.attr}")
    return found


def _assert_built_by_hand(nb, *names):
    for name in names:
        used = _shortcuts(getattr(nb, name))
        assert not used, (
            f"{name} uses {', '.join(sorted(used))}. the point of this stage is to BUILD the product: multiply "
            "entry by entry with * and add with np.sum (or a loop). you may use @ freely after this topic.")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")  # don't leak figures from a failing plot test


RNG = np.random.default_rng(0)
A_RECT = RNG.normal(size=(3, 4))      # non-square on purpose: a transposed matrix can't sneak through
B_RECT = RNG.normal(size=(4, 2))
V4 = RNG.normal(size=4)


# ---------------------------------------------------------------- stage 1: dot and norm


def test_stage1_dot_worked_example(nb):
    got = nb.dot([1, 2, 3], [4, -5, 6])
    assert np.isclose(got, 12.0), f"dot([1, 2, 3], [4, -5, 6]) = 1*4 + 2*(-5) + 3*6 = 12, got {got}"
    assert isinstance(got, float), f"dot should return a float, got {type(got).__name__}"


def test_stage1_dot_matches_numpy(nb):
    u, v = RNG.normal(size=7), RNG.normal(size=7)
    got = nb.dot(u, v)
    assert np.isclose(got, u @ v), (
        f"dot of two random 7-vectors should be {u @ v:.6f}, got {got:.6f}. multiply entry by entry, "
        "then add ALL the products.")


def test_stage1_dot_rejects_mismatched_lengths(nb):
    with pytest.raises(ValueError):
        nb.dot([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        nb.dot([1.0, 2.0, 3.0], [2.0])  # NumPy would happily "broadcast" this: check the lengths yourself


def test_stage1_built_by_hand(nb):
    _assert_built_by_hand(nb, "dot", "norm")


def test_stage1_norm(nb):
    assert np.isclose(nb.norm([3, 4]), 5.0), f"norm([3, 4]) = sqrt(9 + 16) = 5, got {nb.norm([3, 4])}"
    v = np.array([1.0, -2.0, 2.0, 4.0])
    got = nb.norm(v)
    assert np.isclose(got, 5.0), (
        f"norm([1, -2, 2, 4]) = sqrt(1 + 4 + 4 + 16) = 5, got {got}. "
        + ("25 means you forgot the square root." if np.isclose(got, 25) else ""))


# ---------------------------------------------------------------- stage 2: angles and projections


def test_stage2_angle_examples(nb):
    cases = [([1, 0], [1, 1], np.pi / 4), ([1, 0], [0, 3], np.pi / 2), ([2, 0], [-5, 0], np.pi),
             ([1, 2, 2], [2, 4, 4], 0.0)]
    for u, v, want in cases:
        got = nb.angle(u, v)
        assert np.isclose(got, want, atol=1e-7), (
            f"angle({u}, {v}) should be {want:.6f} radians ({np.degrees(want):.0f} degrees), got {got:.6f}. "
            + ("that looks like degrees: return radians." if np.isclose(got, np.degrees(want)) else
               "divide the dot product by BOTH lengths before taking arccos."))


def test_stage2_angle_survives_rounding(nb):
    # for some of these, u.v / (|u| |v|) rounds to 1.0000000000000002, just outside arccos's domain
    for v in ([1.0, 1.0, 1.0], [0.1, 0.7], [0.1, 0.2, 0.3], [3.0, 7.0]):
        for k in (1.0, 3.0):
            got = nb.angle(v, k * np.asarray(v))
            assert np.isfinite(got) and abs(got) < 1e-7, (
                f"the angle between v = {v} and {k:g} v is 0, got {got}. if you got nan, rounding pushed the "
                "cosine just above 1: np.clip it into [-1, 1] before np.arccos.")


def test_stage2_project(nb):
    got = np.asarray(nb.project([2.0, 3.0], [1.0, 0.0]))
    assert np.allclose(got, [2.0, 0.0]), f"the shadow of (2, 3) on the x-axis is (2, 0), got {got}"
    got = np.asarray(nb.project([1.0, 2.0], [2.0, 0.0]))
    assert np.allclose(got, [1.0, 0.0]), (
        f"project([1, 2], [2, 0]) = (u.v / v.v) v = (2 / 4) * [2, 0] = [1, 0], got {got}. "
        "divide by v.v (the SQUARED length), so the length of v doesn't matter.")


def test_stage2_project_leftover_is_perpendicular(nb):
    u, v = RNG.normal(size=5), RNG.normal(size=5)
    p = np.asarray(nb.project(u, v))
    assert p.shape == (5,), f"project should return a vector shaped like v, (5,), got {p.shape}"
    assert abs((u - p) @ v) < 1e-10, "what's left over, u - project(u, v), should be perpendicular to v"
    assert np.allclose(p, (u @ v) / (v @ v) * v), "project(u, v) should be (u.v / v.v) v"


# ---------------------------------------------------------------- stage 3: matvec and matmul


def test_stage3_matvec_worked_example(nb):
    got = np.asarray(nb.matvec([[1, 2], [3, 4]], [10, 1]))
    assert np.allclose(got, [12, 34]), (
        f"[[1, 2], [3, 4]] times [10, 1] = 10 * (column [1, 3]) + 1 * (column [2, 4]) = [12, 34], got {got}. "
        + ("[13, 24] means you combined ROWS instead of columns." if np.allclose(got, [13, 24]) else ""))


def test_stage3_matvec_rectangular(nb):
    got = np.asarray(nb.matvec(A_RECT, V4))
    assert got.shape == (3,), f"a (3, 4) matrix times a (4,) vector is a (3,) vector, got shape {got.shape}"
    assert np.allclose(got, A_RECT @ V4), f"A v should be {A_RECT @ V4}, got {got}"
    with pytest.raises(ValueError):
        nb.matvec(A_RECT, np.ones(3))


def test_stage3_matmul(nb):
    got = np.asarray(nb.matmul([[1, 2], [3, 4]], [[0, 1], [1, 0]]))
    assert np.allclose(got, [[2, 1], [4, 3]]), (
        f"multiplying on the right by [[0, 1], [1, 0]] swaps the columns: expected [[2, 1], [4, 3]], got {got.tolist()}")
    got = np.asarray(nb.matmul(A_RECT, B_RECT))
    assert got.shape == (3, 2), f"(3, 4) times (4, 2) gives (3, 2), got {got.shape}"
    assert np.allclose(got, A_RECT @ B_RECT), "column j of A B should be A times column j of B"
    with pytest.raises(ValueError):
        nb.matmul(A_RECT, A_RECT)


def test_stage3_built_by_hand(nb):
    _assert_built_by_hand(nb, "matvec", "matmul")


def test_stage3_matmul_order_matters(nb):
    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    B = np.array([[0.0, -1.0], [1.0, 0.0]])
    AB, BA = np.asarray(nb.matmul(A, B)), np.asarray(nb.matmul(B, A))
    assert np.allclose(AB, A @ B) and np.allclose(BA, B @ A), (
        f"matmul(A, B) should be A B = {(A @ B).tolist()} and matmul(B, A) should be B A = {(B @ A).tolist()}; "
        f"got {AB.tolist()} and {BA.tolist()}")


# ---------------------------------------------------------------- stage 4: a gallery of transformations


def test_stage4_rotation(nb):
    R = np.asarray(nb.rotation(np.pi / 2))
    assert R.shape == (2, 2), f"rotation should return a (2, 2) matrix, got {R.shape}"
    assert np.allclose(R @ [1, 0], [0, 1], atol=1e-12), (
        f"a quarter turn counterclockwise sends (1, 0) to (0, 1); your matrix sends it to {R @ [1, 0]}. "
        "column 0 is (cos t, sin t).")
    assert np.allclose(R @ [0, 1], [-1, 0], atol=1e-12), (
        f"a quarter turn counterclockwise sends (0, 1) to (-1, 0); yours gives {R @ [0, 1]}. "
        "column 1 is (-sin t, cos t). (if you got (1, 0), your rotation is clockwise.)")
    t = 0.7
    R = np.asarray(nb.rotation(t))
    v = np.array([0.3, -1.2])
    assert np.isclose(np.linalg.norm(R @ v), np.linalg.norm(v)), "rotations don't change lengths"
    assert np.allclose(np.asarray(nb.rotation(0.3)) @ np.asarray(nb.rotation(0.4)), np.asarray(nb.rotation(0.7))), (
        "rotating by 0.4 and then by 0.3 should be the same as rotating by 0.7")


def test_stage4_scaling_and_shear(nb):
    S = np.asarray(nb.scaling(2.0, -3.0))
    assert np.allclose(S, [[2, 0], [0, -3]]), f"scaling(2, -3) should be [[2, 0], [0, -3]], got {S.tolist()}"
    H = np.asarray(nb.shear(0.5))
    assert np.allclose(H @ [0, 2], [1, 2]), (
        f"shear(0.5) moves (x, y) sideways by 0.5 y: (0, 2) -> (1, 2); got {H @ [0, 2]}. "
        "the columns are (1, 0) and (k, 1).")
    assert np.allclose(H @ [3, 0], [3, 0]), "a horizontal shear leaves points on the x-axis where they are"


# ---------------------------------------------------------------- stage 5: the visualizer


def _circle(n):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])


def test_stage5_transform_points(nb):
    got = np.asarray(nb.transform_points(np.diag([2.0, 3.0]), [[1, 1], [1, 0]]))
    assert np.allclose(got, [[2, 3], [2, 0]]), (
        f"scaling by (2, 3) sends the points (1, 1) and (1, 0) to (2, 3) and (2, 0), got {got.tolist()}")
    A = np.array([[0.5, -1.2], [2.0, 0.3]])  # non-symmetric, so A and A.T differ
    P = RNG.normal(size=(6, 2))
    got = np.asarray(nb.transform_points(A, P))
    assert got.shape == (6, 2), f"6 points in, 6 points out: expected shape (6, 2), got {got.shape}"
    assert np.allclose(got, P @ A.T), (
        "row i of the result should be A times row i of P. "
        + ("you applied A transposed: multiply A by the points as COLUMNS (P.T), then transpose back."
           if np.allclose(got, P @ A) else ""))


def test_stage5_plot_draws_circle_and_image(nb):
    A = np.array([[1.5, 0.4], [-0.3, 0.8]])
    fig, ax = plt.subplots()
    out = nb.plot_transform(A, ax=ax, n=50)
    assert out is ax, "plot_transform(A, ax=ax) should draw on the ax you give it and return that same ax"
    lines = [np.column_stack([ln.get_xdata(), ln.get_ydata()]) for ln in ax.get_lines()]
    lines = [np.asarray(xy, dtype=float) for xy in lines]
    circ = _circle(50)
    assert any(xy.shape == circ.shape and np.allclose(xy, circ) for xy in lines), (
        "draw the unit circle as one line through the n points (cos t, sin t), t = np.linspace(0, 2 pi, n)")
    img = circ @ A.T
    assert any(xy.shape == img.shape and np.allclose(xy, img) for xy in lines), (
        "draw the image of the circle as one line through the n transformed points (transform_points(A, circle))")
    assert ax.get_aspect() == 1.0, "call ax.set_aspect('equal') so that circles look round"
    plt.close(fig)


def test_stage5_plot_makes_its_own_axes(nb):
    ax = nb.plot_transform(np.eye(2))
    assert hasattr(ax, "plot") and hasattr(ax, "get_lines"), (
        f"with ax=None, plot_transform should create an Axes (plt.subplots()) and return it, got {type(ax).__name__}")
    assert len(ax.get_lines()) >= 2, "expected at least two lines (the circle and its image)"
    plt.close(ax.figure)


# ---------------------------------------------------------------- stage 6: areas and determinants


def test_stage6_det2(nb):
    assert np.isclose(nb.det2([[2, 1], [1, 2]]), 3.0), f"det of [[2, 1], [1, 2]] is 2*2 - 1*1 = 3, got {nb.det2([[2, 1], [1, 2]])}"
    M = np.array([[0.3, -1.4], [2.2, 0.9]])
    got = nb.det2(M)
    assert np.isclose(got, np.linalg.det(M)), (
        f"det of [[a, b], [c, d]] is a d - b c = {np.linalg.det(M):.6f}, got {got:.6f}. "
        + ("you computed b c - a d: the sign is backwards." if np.isclose(got, -np.linalg.det(M)) else ""))
    assert np.isclose(nb.det2([[0, 1], [1, 0]]), -1.0), "swapping the axes flips the plane over: det = -1"


def test_stage6_polygon_area(nb):
    sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert np.isclose(nb.polygon_area(sq), 1.0), f"the unit square has area 1, got {nb.polygon_area(sq)}"
    assert np.isclose(nb.polygon_area(sq[::-1]), -1.0), (
        f"the same square traced clockwise has SIGNED area -1, got {nb.polygon_area(sq[::-1])}")
    tri = [[1, 1], [5, 1], [2, 4]]
    got = nb.polygon_area(tri)
    assert np.isclose(got, 6.0), (
        f"the triangle (1,1), (5,1), (2,4) has area base * height / 2 = 4 * 3 / 2 = 6, got {got}. "
        "don't forget the wrap-around term from the last corner back to the first (np.roll).")


def test_stage6_area_scales_by_det(nb):
    A = np.array([[1.2, -0.7], [0.4, 0.9]])
    poly = _circle(400)[:-1]  # a 399-gon: almost a circle
    before = nb.polygon_area(poly)
    after = nb.polygon_area(np.asarray(nb.transform_points(A, poly)))
    assert np.isclose(after, nb.det2(A) * before, rtol=1e-10), (
        f"a matrix multiplies every area by its determinant: expected {nb.det2(A) * before:.6f}, got {after:.6f}")
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    flipped = nb.polygon_area(np.asarray(nb.transform_points(M, poly)))
    assert np.isclose(flipped, -before, rtol=1e-10), "a mirror (det -1) turns counterclockwise into clockwise: the signed area flips"
