"""Checks for exercise.py (smoothness & lipschitz continuity).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Most checks use
quadratics, whose constants are known exactly (L and mu are the extreme Hessian eigenvalues),
with sample points chosen so the exact answer is attained by one particular pair.
"""
import numpy as np


def _quad(eigs, seed=0):
    """a random symmetric matrix with the given eigenvalues, and its eigenvectors."""
    n = len(eigs)
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return Q @ np.diag(eigs) @ Q.T, Q


# ---------------------------------------------------------------- stage 1: Lipschitz estimates


def test_stage1_all_pairs_not_neighbours(nb):
    pts = np.array([[0.0], [10.0], [1.0], [1.1]])
    got = nb.lipschitz_estimate(lambda v: float(v[0] ** 2), pts)
    # slopes of x^2 between a and b are |a + b|; the largest pair is (10, 1.1): 11.1
    assert np.isclose(got, 11.1), (
        f"for f = x^2 at the points 0, 10, 1, 1.1 the slope between a and b is |a + b|, largest for the pair "
        f"(10, 1.1): 11.1; got {got}"
        + (". 11.0 means you only compared neighbouring rows: use every pair i < j." if np.isclose(got, 11.0) else ""))
    assert isinstance(got, float), "return a float"


def test_stage1_lower_bound_and_converges(nb):
    f = lambda v: float(np.sin(3 * v[0]))
    coarse = nb.lipschitz_estimate(f, np.linspace(0, 2, 6)[:, None])
    fine = nb.lipschitz_estimate(f, np.linspace(0, 2, 400)[:, None])
    assert coarse <= 3.0 + 1e-12 and fine <= 3.0 + 1e-12, (
        f"sin(3x) has Lipschitz constant 3, and a sampled estimate can never exceed it; got {coarse}, {fine}")
    assert fine > 2.99 and coarse < fine, (
        f"with 400 points on [0, 2] the estimate should approach 3 (got {fine:.4f}), and beat 6 points ({coarse:.4f})")


def test_stage1_vector_valued_linear_map(nb):
    A, Q = _quad([4.0, 1.0, 0.5], seed=1)
    B = A @ np.diag([1.0, 2.0, 3.0])  # non-symmetric
    U, s, Vt = np.linalg.svd(B)
    rng = np.random.default_rng(2)
    pts = np.vstack([rng.normal(size=(10, 3)), np.zeros(3), Vt[0]])  # includes a pair along the top singular direction
    got = nb.lipschitz_estimate(lambda v: B @ v, pts)
    assert np.isclose(got, s[0], rtol=1e-10), (
        f"for the linear map x -> B x the slope between x and y is ||B (x - y)|| / ||x - y||, at most the largest "
        f"singular value {s[0]:.6f}, attained by the pair (0, v_1); got {got:.6f}. for vector outputs use the "
        "norm of the difference of the outputs.")


def test_stage1_duplicate_points_are_skipped(nb):
    pts = np.array([[1.0, 2.0], [1.0, 2.0], [0.0, 0.0]])
    got = nb.lipschitz_estimate(lambda v: float(v @ np.array([3.0, 4.0])), pts)
    assert np.isfinite(got) and np.isclose(got, 11 / np.sqrt(5)), (
        f"identical points give 0/0: skip them (and the diagonal i = j). expected {11 / np.sqrt(5):.6f}, got {got}")


# ---------------------------------------------------------------- stage 2: smoothness and strong convexity


def test_stage2_quadratic_exact(nb):
    A, Q = _quad([10.0, 3.0, 1.0], seed=3)
    grad = lambda v: A @ v
    pts = np.vstack([np.zeros(3), Q[:, 0], Q[:, 2], np.random.default_rng(4).normal(size=(6, 3))])
    L = nb.smoothness_estimate(grad, pts)
    mu = nb.strong_convexity_estimate(grad, pts)
    assert np.isclose(L, 10.0, rtol=1e-10), (
        f"for f = x^T A x / 2 the gradient is A x, so L is the largest eigenvalue 10, attained by the pair "
        f"(0, top eigenvector); got {L}")
    assert np.isclose(mu, 1.0, rtol=1e-10), (
        f"mu is the SMALLEST curvature (g_i - g_j).(x_i - x_j) / ||x_i - x_j||^2 over pairs: 1 here, along the bottom "
        f"eigenvector; got {mu}" + (" (that's the largest, not the smallest)" if np.isclose(mu, 10.0) else ""))


def test_stage2_bounds_on_random_points(nb):
    A, _ = _quad([5.0, 2.0, 0.5, 0.2], seed=5)
    grad = lambda v: A @ v
    pts = np.random.default_rng(6).normal(size=(30, 4))
    L = nb.smoothness_estimate(grad, pts)
    mu = nb.strong_convexity_estimate(grad, pts)
    assert 0.2 - 1e-12 <= mu <= L <= 5.0 + 1e-12, (
        f"sampled estimates must satisfy mu_true <= mu_est <= L_est <= L_true, i.e. 0.2 <= {mu:.4f} <= {L:.4f} <= 5")


def test_stage2_nonconvex_gives_negative(nb):
    grad = lambda v: np.array([2 * v[0], -2 * v[1]])  # f = x^2 - y^2
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    mu = nb.strong_convexity_estimate(grad, pts)
    assert np.isclose(mu, -2.0), (
        f"x^2 - y^2 curves DOWN along y with curvature -2; the minimum over pairs must report it, got {mu}. "
        "use the dot product (g_i - g_j).(x_i - x_j), not a norm (a norm is never negative), and divide by the "
        "SQUARED distance.")


# ---------------------------------------------------------------- stage 3: the descent lemma, L from the Hessian


def test_stage3_descent_lemma_hand_values(nb):
    f, g = (lambda v: float(v @ v)), (lambda v: 2 * v)
    x, y = np.array([1.0, -2.0]), np.array([0.5, 1.0])
    gap2 = nb.descent_lemma_gap(f, g, 2.0, x, y)
    gap3 = nb.descent_lemma_gap(f, g, 3.0, x, y)
    d2 = float((y - x) @ (y - x))
    assert np.isclose(gap2, 0.0, atol=1e-12), (
        f"f = ||x||^2 is 2-smooth and its quadratic upper bound with L = 2 is exact: the gap is 0; got {gap2}")
    assert np.isclose(gap3, 0.5 * d2), f"with L = 3 the gap is (3 - 2)/2 ||y - x||^2 = {0.5 * d2}; got {gap3}"


def test_stage3_descent_lemma_detects_small_L(nb):
    A, Q = _quad([8.0, 1.0], seed=7)
    f, g = (lambda v: 0.5 * v @ A @ v), (lambda v: A @ v)
    rng = np.random.default_rng(8)
    gaps_ok = [nb.descent_lemma_gap(f, g, 8.0, rng.normal(size=2), rng.normal(size=2)) for _ in range(50)]
    assert min(gaps_ok) >= -1e-12, "with the true L = 8 the gap must be >= 0 for every pair of points"
    x = rng.normal(size=2)
    bad = nb.descent_lemma_gap(f, g, 4.0, x, x + Q[:, 0])
    assert bad < 0, f"with L = 4 < 8, the bound fails along the top eigenvector: the gap should be negative, got {bad}"


def test_stage3_hessian_smoothness(nb):
    A, _ = _quad([-5.0, 2.0, 0.5], seed=9)
    got = nb.hessian_smoothness(lambda v: 0.5 * v @ A @ v, np.array([0.3, -0.1, 0.7]))
    assert np.isclose(got, 5.0, rtol=1e-6), (
        f"the Hessian has eigenvalues -5, 2, 0.5: the local smoothness is the largest ABSOLUTE eigenvalue, 5; got {got}"
        + (" (you took the largest eigenvalue, 2)" if np.isclose(got, 2.0, rtol=1e-4) else ""))
    f = lambda v: np.log(1 + np.exp(v[0])) + 0.25 * v[1] ** 2  # Hessian diag(s(1-s), 0.5)
    got = nb.hessian_smoothness(f, np.zeros(2))
    assert np.isclose(got, 0.5, rtol=1e-6), f"at 0 the Hessian is diag(1/4, 1/2), so the answer is 0.5; got {got}"


# ---------------------------------------------------------------- stage 4: gradient descent and its threshold


def _gq(v):
    return np.array([1.0, 10.0]) * v  # f = (x^2 + 10 y^2) / 2: L = 10


def test_stage4_gradient_descent_worked_example(nb):
    x0 = np.array([1.0, 1.0])
    xs = np.asarray(nb.gradient_descent(_gq, x0, 0.21, 3), dtype=float)
    assert xs.shape == (4, 2), f"3 iterations from x0 give 4 iterates (x0 included): shape (4, 2), got {xs.shape}"
    want = np.array([[1.0, 1.0], [0.79, -1.1], [0.6241, 1.21], [0.493039, -1.331]])
    assert np.allclose(xs, want), (
        f"with step 0.21 the x-coordinate shrinks by 1 - 0.21 = 0.79 per step and y by 1 - 2.1 = -1.1: "
        f"expected {want.tolist()}, got {np.round(xs, 6).tolist()}")
    assert x0.tolist() == [1.0, 1.0], "gradient_descent must not modify x0"


def test_stage4_threshold_is_two_over_L(nb):
    steps = [0.05, 0.1, 0.15, 0.19, 0.21, 0.25]
    got = nb.largest_stable_step(_gq, np.array([1.0, 1.0]), steps)
    assert np.isclose(got, 0.19), (
        f"for L = 10 gradient descent is stable for steps below 2/L = 0.2 and diverges above: among {steps} the "
        f"largest stable one is 0.19; got {got}")
    got = nb.largest_stable_step(_gq, np.array([1.0, 1.0]), [0.05, 0.25, 0.19, 0.3])
    assert np.isclose(got, 0.19), (
        f"steps needn't be sorted: try every one (0.25 failing doesn't mean 0.19 will); still 0.19, got {got}")
    got = nb.largest_stable_step(_gq, np.array([1.0, 1.0]), [0.19, 0.05, 0.25])
    assert np.isclose(got, 0.19), f"keep the LARGEST stable step, not the last stable one tried: 0.19, got {got}"
    got = nb.largest_stable_step(_gq, np.array([1e4, 1e4]), [0.05, 0.19, 0.21])
    assert np.isclose(got, 0.19), (
        f"starting 10,000 times farther away must not change the answer (0.19); got {got}. the test is RELATIVE: "
        "||grad(last)|| <= tol * ||grad(x0)||, not ||grad(last)|| <= tol.")


def test_stage4_oscillation_is_not_stable(nb):
    got = nb.largest_stable_step(_gq, np.array([1.0, 1.0]), [0.1, 0.2])
    assert np.isclose(got, 0.1), (
        f"at step exactly 2/L = 0.2 the y-coordinate flips sign forever (factor 1 - 0.2 * 10 = -1): bounded, "
        f"but the gradient never shrinks, so it's not stable. expected 0.1, got {got}. require "
        "||grad(last)|| <= tol * ||grad(x0)||, not just 'smaller than at the start'.")


def test_stage4_nothing_stable_and_overflow(nb):
    got = nb.largest_stable_step(_gq, np.array([1.0, 1.0]), [0.5, 1.0, 5.0], n_iters=400)
    assert got == 0.0, (
        f"every step here is above 2/L = 0.2, so none is stable: return 0.0; got {got}. (steps of 5.0 overflow to "
        "inf/nan after a few hundred iterations: treat non-finite iterates as unstable.)")
