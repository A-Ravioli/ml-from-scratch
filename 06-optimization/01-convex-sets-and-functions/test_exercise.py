"""Checks for exercise.py (convex sets & functions).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The functions
used are small enough that the exact answers are known by hand: polyhedra and annuli, the
double well x^4 - 2x^2 (the lesson's worked example), quadratics, and log-sum-exp.
"""
import numpy as np


class _Counter:
    """wraps f and counts calls; raises a teaching error past a budget (so loops can't run away)."""

    def __init__(self, f, budget=100_000):
        self.f, self.calls, self.budget = f, 0, budget

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.budget, f"f was called more than {self.budget} times: is a loop running away?"
        return self.f(x)


def _well(v):
    return float(v[0] ** 4 - 2 * v[0] ** 2)


def _well_grad(v):
    return np.array([4 * v[0] ** 3 - 4 * v[0]])


def _lse(v):
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))


def _lse_grad(v):
    e = np.exp(v - np.max(v))
    return e / e.sum()


# ---------------------------------------------------------------- stage 1: convex sets


def test_stage1_polyhedron_membership(nb):
    # the triangle x >= 0, y >= 0, x + y <= 1, written as A x <= b
    A = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    b = np.array([0.0, 0.0, 1.0])
    X = np.array([[0.2, 0.2], [0.5, 0.5], [0.6, 0.6], [-0.1, 0.5], [1.0, 0.0], [0.0, 0.0]])
    got = np.asarray(nb.in_polyhedron(A, b, X))
    want = np.array([True, True, False, False, True, True])
    assert got.shape == (6,), f"one bool per row of X: shape (6,), got {got.shape}"
    assert got.dtype == bool, f"return a bool array, got dtype {got.dtype}"
    assert np.array_equal(got, want), (
        f"for the triangle x, y >= 0, x + y <= 1: expected {want.tolist()}, got {got.tolist()}. a point is inside only "
        "if ALL rows of A x <= b hold (np.all over the constraints), and boundary points (A x = b) are inside.")


def test_stage1_polyhedron_tolerance_and_shape(nb):
    A = np.array([[1.0, 2.0, -1.0]])
    b = np.array([3.0])
    X = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, -1e-14], [1.0, 1.0, -1e-6]])
    got = np.asarray(nb.in_polyhedron(A, b, X))
    assert got.tolist() == [True, True, False], (
        f"x = (1, 1, 0) is on the boundary x + 2y - z = 3, and (1, 1, -1e-14) is outside by only 1e-14 (rounding "
        f"level): both count as inside with tol = 1e-12; (1, 1, -1e-6) is outside. expected [True, True, False], got "
        f"{got.tolist()}. use A x <= b + tol, with A x computed as X @ A.T for all rows at once.")


def test_stage1_convex_sets_have_no_violation(nb):
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.5, 1.5, size=(40, 2))
    disk = lambda x: bool(x @ x <= 1.0)
    assert nb.set_convexity_violation(disk, pts) is None, (
        "the unit disk is convex: no chord between two of its points leaves it, so return None")
    A = np.array([[1.0, 1.0], [-1.0, 2.0], [0.0, -1.0]])
    b = np.array([1.0, 2.0, 1.0])
    poly = lambda x: bool(nb.in_polyhedron(A, b, x[None, :])[0])
    assert nb.set_convexity_violation(poly, pts) is None, "a polyhedron is convex: expected None"


def test_stage1_annulus_violation(nb):
    annulus = lambda x: bool(0.5 <= np.hypot(x[0], x[1]) <= 1.0)
    pts = np.array([[0.0, 0.0], [0.75, 0.0], [0.0, 0.8], [-0.75, 0.0]])
    got = nb.set_convexity_violation(annulus, pts)
    assert got is not None, "the annulus 0.5 <= |x| <= 1 isn't convex: some chord crosses the hole"
    i, j, t = got
    assert (i, j) == (1, 3) and np.isclose(t, 0.25), (
        f"row 0 (the origin) is NOT in the annulus, so skip it. among member rows 1, 2, 3, the pair (1, 2) stays "
        f"inside for t = 0.25, 0.5, 0.75 (the closest point of that chord is at distance 0.55), and (1, 3) at "
        f"t = 0.25 lands at (-0.375, 0), inside the hole. return the first violation in loop order i, j, t as "
        f"(1, 3, 0.25), with indices into the ORIGINAL points; got {got}")


def test_stage1_union_of_disks_needs_the_right_t(nb):
    two_disks = lambda x: bool(np.hypot(x[0] + 1, x[1]) <= 0.6 or np.hypot(x[0] - 1, x[1]) <= 0.6)
    pts = np.array([[-1.0, 0.0], [1.0, 0.0]])
    got = nb.set_convexity_violation(two_disks, pts, ts=(0.1, 0.5))
    assert got == (0, 1, 0.5), (
        f"two disjoint disks: the chord from (-1, 0) to (1, 0) leaves the set at t = 0.5 (the origin), but not at "
        f"t = 0.1 (the point (0.8, 0), inside the right disk). test the ts you were given, in order; expected "
        f"(0, 1, 0.5), got {got}")
    lopsided = lambda x: bool(np.hypot(x[0] + 1, x[1]) <= 0.95 or np.hypot(x[0] - 1, x[1]) <= 0.1)
    got = nb.set_convexity_violation(lopsided, pts, ts=(0.75,))
    assert got is None, (
        f"with t = 0.75 the tested point is 0.75 x_0 + 0.25 x_1 = (-0.5, 0), inside the big left disk, so there's no "
        f"violation to report; got {got}. t weights the FIRST point of the pair: z = t x_i + (1 - t) x_j.")


# ---------------------------------------------------------------- stage 2: Jensen's inequality


def test_stage2_jensen_gap_hand_values(nb):
    x, y = np.array([-1.0]), np.array([1.0])
    got = nb.jensen_gap(_well, x, y, 0.5)
    assert np.isclose(got, -1.0), (
        f"for f = x^4 - 2x^2, the chord from (-1, -1) to (1, -1) has height -1 at the midpoint, while f(0) = 0: "
        f"the gap is -1 - 0 = -1 (negative: f is not convex); got {got}")
    q = lambda v: float(v @ v)
    got = nb.jensen_gap(q, np.array([2.0, 0.0]), np.array([0.0, 0.0]), 0.25)
    assert np.isclose(got, 0.75), (
        f"for f = ||x||^2, x = (2, 0), y = 0, t = 0.25: t f(x) + (1 - t) f(y) = 1 and f(0.25 x + 0.75 y) = 0.25, so "
        f"the gap is 0.75; got {got}" + (" (did you swap t and 1 - t?)" if np.isclose(got, 2.75) or np.isclose(got, -1.25) else ""))
    assert isinstance(got, float), "return a float"


def test_stage2_t_belongs_to_x(nb):
    f = lambda v: float(np.exp(v[0]))
    got = nb.jensen_gap(f, np.array([2.0]), np.array([0.0]), 0.25)
    want = 0.25 * np.exp(2.0) + 0.75 - np.exp(0.5)
    assert np.isclose(got, want), (
        f"the weight t goes with x: t f(x) + (1 - t) f(y) - f(t x + (1 - t) y). for e^x, x = 2, y = 0, t = 0.25 that's "
        f"{want:.6f}; got {got:.6f}")


def test_stage2_min_gap_worked_example(nb):
    pts = np.array([[-1.0], [0.0], [1.0], [2.0]])
    gap, where = nb.min_jensen_gap(_well, pts)
    assert np.isclose(gap, -1.0) and tuple(where) == (0, 2, 0.5), (
        f"for x^4 - 2x^2 at -1, 0, 1, 2 the most negative gap is -1, for the pair (0, 2) at t = 0.5 (the chord "
        f"across the hump); got gap {gap}, at {where}")
    gap, where = nb.min_jensen_gap(_well, pts, ts=(0.25, 0.75))
    assert np.isclose(gap, -0.5625) and tuple(where) == (0, 2, 0.25), (
        f"with ts = (0.25, 0.75) the pair (0, 2) gives -0.5625 at BOTH t values (the hump is symmetric): on ties keep "
        f"the first in loop order i, j, t, i.e. (0, 2, 0.25); got {gap}, {where}")


def test_stage2_convex_functions_and_call_budget(nb):
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(12, 3))
    f = _Counter(_lse)
    gap, _ = nb.min_jensen_gap(f, pts)
    assert gap >= -1e-12, f"log-sum-exp is convex, so every Jensen gap is >= 0 (up to rounding); got {gap}"
    budget = 12 + 66 * 3
    assert f.calls <= budget, (
        f"evaluate f once per row and once per (pair, t): at most 12 + 66 * 3 = {budget} calls; you made {f.calls}. "
        "reuse f(x_i) and f(x_j) instead of recomputing them for every pair.")


def test_stage2_detects_nonconvexity_with_given_ts(nb):
    # sin is concave on [0, pi]: every chord lies below it
    pts = np.array([[0.5], [2.5]])
    gap, where = nb.min_jensen_gap(lambda v: float(np.sin(v[0])), pts, ts=(0.9,))
    want = 0.9 * np.sin(0.5) + 0.1 * np.sin(2.5) - np.sin(0.7)
    assert np.isclose(gap, want) and tuple(where) == (0, 1, 0.9), (
        f"with ts = (0.9,) only t = 0.9 is tested: the gap is {want:.6f} at (0, 1, 0.9); got {gap}, {where}")


# ---------------------------------------------------------------- stage 3: first-order condition


def test_stage3_first_order_gap_hand_values(nb):
    got = nb.first_order_gap(_well, _well_grad, np.array([0.0]), np.array([1.0]))
    assert np.isclose(got, -1.0), (
        f"the tangent of x^4 - 2x^2 at 0 is the line y = 0, and f(1) = -1 lies below it: the gap is -1; got {got}")
    got = nb.first_order_gap(_well, _well_grad, np.array([2.0]), np.array([0.0]))
    assert np.isclose(got, 40.0), (
        f"at x = 2: f = 8 and f' = 24, so at y = 0 the gap is f(y) - f(x) - f'(x)(y - x) = 0 - 8 + 48 = 40; got {got}"
        + (" (that's the gradient at y, not at x)" if np.isclose(got, -8.0) else ""))


def test_stage3_quadratic_gap_is_bregman(nb):
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda v: float(0.5 * v @ A @ v)
    g = lambda v: A @ v
    x, y = np.array([1.0, -2.0]), np.array([0.5, 3.0])
    got = nb.first_order_gap(f, g, x, y)
    d = y - x
    assert np.isclose(got, 0.5 * d @ A @ d), (
        f"for f = x^T A x / 2 the first-order gap is exactly (y - x)^T A (y - x) / 2 = {0.5 * d @ A @ d}; got {got}. "
        "use the gradient at x (the tangent point), not at y.")


def test_stage3_ordered_pairs(nb):
    pts = np.array([[-1.0], [0.0]])
    gap, where = nb.min_first_order_gap(_well, _well_grad, pts)
    assert np.isclose(gap, -1.0) and tuple(where) == (1, 0), (
        f"with rows -1 and 0: the tangent at -1 (slope 0, height -1) lies below f(0) = 0 (gap +1), but the tangent "
        f"at 0 (the line y = 0) lies above f(-1) = -1 (gap -1). the order matters: test every ORDERED pair i != j. "
        f"expected (-1, (1, 0)), got ({gap}, {where})")


def test_stage3_matches_brute_force(nb):
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(9, 2)) * 1.5
    f = lambda v: float(np.sin(v[0]) * v[1] + 0.1 * v[1] ** 2)
    g = lambda v: np.array([np.cos(v[0]) * v[1], np.sin(v[0]) + 0.2 * v[1]])
    gaps = {(i, j): f(pts[j]) - f(pts[i]) - g(pts[i]) @ (pts[j] - pts[i])
            for i in range(9) for j in range(9) if i != j}
    want = min(gaps, key=gaps.get)
    gap, where = nb.min_first_order_gap(f, g, pts)
    assert np.isclose(gap, gaps[want]) and tuple(where) == want, (
        f"brute force over all ordered pairs gives gap {gaps[want]:.6f} at {want}; got {gap:.6f} at {where}. "
        "gap[i, j] = f(x_j) - f(x_i) - grad(x_i) . (x_j - x_i): row i is the tangent point.")


def test_stage3_convex_function_nonnegative(nb):
    pts = np.random.default_rng(3).normal(size=(15, 4)) * 3
    gap, _ = nb.min_first_order_gap(_lse, _lse_grad, pts)
    assert gap >= -1e-12, f"log-sum-exp is convex: it lies above every tangent plane, so the min gap is >= 0; got {gap}"
    sq = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    gap, (i, j) = nb.min_first_order_gap(lambda v: float(v @ v), lambda v: 2 * v, sq)
    assert i != j and np.isclose(gap, 1.0), (
        f"for ||x||^2 the gap is ||x_j - x_i||^2, smallest (1.0) for the closest pair of DIFFERENT points; got "
        f"{gap} at ({i}, {j}). exclude the diagonal i = j (its gap is always 0).")


# ---------------------------------------------------------------- stage 4: second order, and the verdict


def test_stage4_hessian_eigenvalue_worked_example(nb):
    pts = np.array([[-1.0], [0.0], [1.0], [2.0]])
    lam, k = nb.min_hessian_eigenvalue(_well, pts)
    assert np.isclose(lam, -4.0, atol=1e-5) and k == 1, (
        f"f'' = 12x^2 - 4 is 8, -4, 8, 44 at the four points: the smallest is -4, at row 1; got ({lam}, {k})"
        + (" (that's the largest; take the minimum)" if np.isclose(lam, 44, atol=1e-3) else ""))


def test_stage4_hessian_eigenvalue_multivariate(nb):
    Q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
    A = Q @ np.diag([5.0, 1.0, -0.5]) @ Q.T
    f = lambda v: float(0.5 * v @ A @ v + 0.25 * v[0] ** 4)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lam, k = nb.min_hessian_eigenvalue(f, pts)
    want = np.linalg.eigvalsh(A + np.diag([3.0, 0.0, 0.0]))[0]
    assert k == 0 and np.isclose(lam, -0.5, atol=1e-5), (
        f"the Hessian is A + diag(3 x_1^2, 0, 0): at the origin its smallest eigenvalue is -0.5, at (1, 0, 0) it's "
        f"{want:.4f}. expected (-0.5, 0), got ({lam}, {k}). use the smallest eigenvalue of each Hessian "
        "(np.linalg.eigvalsh sorts ascending), then the smallest over the points.")


def test_stage4_verdict_and_witness(nb):
    pts = np.array([[-1.0], [0.0], [1.0], [2.0]])
    verdict, witness = nb.check_convexity(_well, pts)
    assert verdict is False or verdict == False, f"x^4 - 2x^2 is not convex: the verdict should be False, got {verdict}"  # noqa: E712
    x, y, t = witness
    assert np.allclose(x, [-1.0]) and np.allclose(y, [1.0]) and np.isclose(t, 0.5), (
        f"the witness is the pair of POINTS and t of the most negative gap: (array([-1.]), array([1.]), 0.5); "
        f"got {witness}")
    x[0] = 99.0
    assert pts[0, 0] == -1.0, "return copies of the rows, so a caller editing the witness can't change your points"


def test_stage4_convex_passes_and_relative_tolerance(nb):
    pts = np.random.default_rng(5).normal(size=(10, 2))
    verdict, witness = nb.check_convexity(lambda v: float(v @ v), pts)
    assert verdict and witness is None, f"||x||^2 is convex: expected (True, None), got ({verdict}, {witness})"
    big = lambda v: 1e12 + float(v @ v)  # convex, but evaluated at 1e12 with rounding steps of ~1.2e-4
    near = 0.03 * pts
    verdict, witness = nb.check_convexity(big, near)
    assert verdict, (
        "f = 1e12 + ||x||^2 is convex, but at that size f is rounded to steps of about 1.2e-4, and on these close "
        "points one Jensen gap comes out at -1.2e-4: pure rounding. the tolerance must be RELATIVE: report a "
        "violation only if gap < -tol * (1 + max |f|).")
    verdict, _ = nb.check_convexity(lambda v: -float(v @ v), pts)
    assert not verdict, "-||x||^2 is concave, so every chord lies below it: the verdict must be False"
