"""Checks for exercise.py (constraints & duality).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Projections are
checked by hand examples and by the variational inequality that characterizes a projection.
QPs are checked against hand solutions and against brute force over active sets (tiny problems,
so enumerating every subset of constraints is cheap). Every f/grad handed to your loop is wrapped
in a call budget, so a loop that ignores max_iters fails instead of hanging.
"""
from itertools import combinations

import numpy as np

from mlfs.testing import bounded


class _Budget:
    def __init__(self, fn, limit, what="grad"):
        self.fn, self.limit, self.what, self.calls = fn, limit, what, 0

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.limit, f"{self.what} was called more than {self.limit} times: does your loop respect max_iters?"
        return self.fn(x)


def _brute_force_qp(Q, c, G, h):
    """solve min 1/2 x^T Q x + c^T x s.t. G x <= h by trying every active set (tiny problems only)."""
    n, m = Q.shape[0], G.shape[0]
    best = None
    for k in range(0, min(n, m) + 1):
        for act in combinations(range(m), k):
            act = list(act)
            if act:
                K = np.block([[Q, G[act].T], [G[act], np.zeros((k, k))]])
                try:
                    z = np.linalg.solve(K, np.concatenate([-c, h[act]]))
                except np.linalg.LinAlgError:
                    continue
                x, lam_a = z[:n], z[n:]
            else:
                x, lam_a = np.linalg.solve(Q, -c), np.zeros(0)
            if np.all(G @ x <= h + 1e-9) and np.all(lam_a >= -1e-9):
                lam = np.zeros(m)
                lam[act] = lam_a
                val = 0.5 * x @ Q @ x + c @ x
                if best is None or val < best[2] - 1e-12:
                    best = (x, lam, val)
    return best


def _random_qp(seed, n=3, m=4):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    Q = M @ M.T + np.eye(n)
    c = rng.normal(size=n) * 3
    G = rng.normal(size=(m, n))
    h = rng.uniform(0.1, 1.0, size=m)  # x = 0 is strictly feasible
    return Q, c, G, h


# ---------------------------------------------------------------- stage 1: projections


def test_stage1_box_and_ball(nb):
    got = np.asarray(nb.project_box(np.array([-2.0, 0.5, 3.0]), 0.0, 1.0))
    assert got.tolist() == [0.0, 0.5, 1.0], f"clip each coordinate into [0, 1]: (0, 0.5, 1); got {got.tolist()}"
    got = np.asarray(nb.project_box(np.array([5.0, -5.0]), np.array([-1.0, -np.inf]), np.array([1.0, 0.0])))
    assert got.tolist() == [1.0, -5.0], f"bounds can be arrays, and -inf means no lower bound: (1, -5); got {got.tolist()}"
    v = np.array([3.0, 4.0])
    got = np.asarray(nb.project_l2_ball(v, radius=2.0))
    assert np.allclose(got, [1.2, 1.6]), f"(3, 4) has norm 5: scale it onto the sphere of radius 2: (1.2, 1.6); got {got}"
    inside = np.array([0.3, -0.4])
    got = np.asarray(nb.project_l2_ball(inside))
    assert np.allclose(got, inside) and got is not inside, (
        "a point inside the unit ball is its own projection: return it unchanged (as a new array)")
    assert v.tolist() == [3.0, 4.0], "projections must not modify their input"


def test_stage1_simplex_hand_values(nb):
    cases = [([0.5, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]), ([2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
             ([1.0, 0.5, -1.0], [0.75, 0.25, 0.0]), ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
             ([-1.0, -2.0], [1.0, 0.0])]
    for v, want in cases:
        got = np.asarray(nb.project_simplex(np.array(v)))
        assert np.allclose(got, want), (
            f"projection of {v} onto the simplex should be {np.round(want, 4).tolist()}, got {np.round(got, 4).tolist()}. "
            "it has the form max(v - theta, 0): for (1, 0.5, -1), theta = 0.25 gives (0.75, 0.25, 0).")


def test_stage1_simplex_is_the_closest_point(nb):
    rng = np.random.default_rng(0)
    for trial in range(20):
        v = rng.normal(size=6) * rng.uniform(0.1, 5)
        p = np.asarray(nb.project_simplex(v))
        assert np.all(p >= 0) and np.isclose(p.sum(), 1.0), f"the projection must lie on the simplex; got {p}"
        Z = rng.dirichlet(np.ones(6), size=200)  # random points of the simplex
        worst = np.max((Z - p) @ (v - p))
        assert worst <= 1e-10, (
            f"p is the projection of v iff (v - p) . (z - p) <= 0 for every z in the set (the angle at p is obtuse); "
            f"found {worst:.3g} > 0 for v = {np.round(v, 3)}, p = {np.round(p, 3)}")
        assert np.allclose(nb.project_simplex(v + 3.7), p), "projecting v + c 1 gives the same point (theta absorbs c)"


# ---------------------------------------------------------------- stage 2: projected gradient descent


def test_stage2_box_constrained_quadratic(nb):
    t = np.array([2.0, -0.5, 0.3])
    f = lambda x: float(np.sum((x - t) ** 2))
    g = lambda x: 2 * (x - t)
    res = nb.projected_gradient_descent(f, _Budget(g, 500), lambda x: nb.project_box(x, 0.0, 1.0), np.zeros(3),
                                        lr=0.1, tol=1e-8)
    assert res.converged and np.allclose(res.x, [1.0, 0.0, 0.3], atol=1e-6), (
        f"the closest point of the box [0, 1]^3 to (2, -0.5, 0.3) is (1, 0, 0.3); got {res}")
    assert np.linalg.norm(g(res.x)) > 1, "(the gradient there isn't small: constrained minima have nonzero gradients)"
    assert len(res.history) == res.n_iters + 1 and res.history[0] == f(np.zeros(3)), "history = [f(x0), ...]"


def test_stage2_stopping_rule_and_first_steps(nb):
    t = np.array([0.6, 0.2])
    f = lambda x: float(np.sum((x - t) ** 2))
    g = lambda x: 2 * (x - t)
    seen = []
    res = nb.projected_gradient_descent(f, _Budget(g, 10), lambda x: nb.project_l2_ball(x, 0.5), np.array([3.0, 0.0]),
                                        lr=0.25, max_iters=3, tol=0.0, callback=lambda x, k: seen.append(np.array(x)))
    # step 1 from (3, 0): x - 0.25 * 2 (x - t) = (1.8, 0.1), projected onto the ball of radius 0.5
    want1 = np.array([1.8, 0.1]) * 0.5 / np.hypot(1.8, 0.1)
    assert np.allclose(seen[0], want1), (
        f"one step: x - lr * grad = (1.8, 0.1), then project onto the ball of radius 0.5: {np.round(want1, 5)}; got "
        f"{seen[0]}. x0 may be infeasible: use it as is (don't project it first).")
    assert res.n_iters == 3 and not res.converged, "tol = 0 never passes: stop after max_iters = 3 steps"
    res = nb.projected_gradient_descent(f, _Budget(g, 200), lambda x: nb.project_l2_ball(x, 0.5), np.zeros(2),
                                        lr=0.25, tol=1e-3)
    assert res.converged and res.n_iters < 50, f"should converge quickly; got {res}"
    assert np.linalg.norm(res.x) <= 0.5 + 1e-12, "the result must be feasible"


def test_stage2_tolerance_is_relative_to_lr(nb):
    xs = []
    res = nb.projected_gradient_descent(lambda x: float(0.5 * x @ x), _Budget(lambda x: x, 100), lambda x: x,
                                        np.array([1.0]), lr=0.5, tol=1e-3, callback=lambda x, k: xs.append(x[0]))
    assert res.converged and res.n_iters == 11 and np.isclose(res.x[0], 0.5**11), (
        f"x halves every step, so step k moves by 0.5^k. the test ||x_new - x|| <= tol * lr = 5e-4 first passes at "
        f"step 11, and that step is still taken: n_iters 11, x = 0.5^11. got n_iters {res.n_iters}, x = {res.x}. "
        "(tol is on the gradient mapping (x - x_new)/lr, so it's multiplied by lr.)")


def test_stage2_simplex_least_squares(nb):
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 5))
    w_true = np.array([0.6, 0.4, 0.0, 0.0, 0.0])
    b = A @ w_true
    f = lambda w: float(0.5 * np.sum((A @ w - b) ** 2))
    g = lambda w: A.T @ (A @ w - b)
    lr = 1 / np.linalg.eigvalsh(A.T @ A).max()
    res = nb.projected_gradient_descent(f, _Budget(g, 20_000), nb.project_simplex, np.ones(5) / 5, lr=lr,
                                        max_iters=10_000, tol=1e-10)
    assert res.converged and np.allclose(res.x, w_true, atol=1e-6), (
        f"least squares over the simplex with a planted solution (0.6, 0.4, 0, 0, 0) on the simplex: projected gradient "
        f"descent should recover it; got {np.round(res.x, 6)}")


# ---------------------------------------------------------------- stage 3: equality-constrained QP


def test_stage3_hand_example(nb):
    x, lam = nb.solve_eq_qp(np.eye(2), np.zeros(2), np.array([[1.0, 1.0]]), np.array([1.0]))
    assert np.allclose(x, [0.5, 0.5]), f"the point of x1 + x2 = 1 closest to the origin is (0.5, 0.5); got {x}"
    assert np.allclose(lam, [-0.5]), (
        f"with L = 1/2 ||x||^2 + lam (x1 + x2 - 1), stationarity x + lam (1, 1) = 0 gives lam = -0.5; got {lam}. "
        "the KKT right-hand side is [-c; b].")


def test_stage3_random_against_numpy(nb):
    rng = np.random.default_rng(2)
    for trial in range(5):
        n, m = 5, 2
        M = rng.normal(size=(n, n))
        Q = M @ M.T + 0.5 * np.eye(n)
        c, A, b = rng.normal(size=n), rng.normal(size=(m, n)), rng.normal(size=m)
        x, lam = nb.solve_eq_qp(Q, c, A, b)
        assert np.allclose(A @ x, b), "the solution must satisfy A x = b"
        assert np.allclose(Q @ x + c + A.T @ lam, 0, atol=1e-9), "stationarity: Q x + c + A^T lam = 0"
        K = np.block([[Q, A.T], [A, np.zeros((m, m))]])
        z = np.linalg.solve(K, np.r_[-c, b])
        assert np.allclose(np.r_[x, lam], z), "(x, lam) should solve the KKT system [[Q, A^T], [A, 0]] [x; lam] = [-c; b]"


def test_stage3_semidefinite_q_is_fine(nb):
    # Q is singular, but the constraint pins down the flat direction: the KKT matrix is still invertible
    Q = np.diag([2.0, 0.0])
    x, lam = nb.solve_eq_qp(Q, np.array([-2.0, 1.0]), np.array([[0.0, 1.0]]), np.array([3.0]))
    assert np.allclose(x, [1.0, 3.0]) and np.allclose(lam, [-1.0]), (
        f"min x1^2 - 2 x1 + x2 s.t. x2 = 3: x = (1, 3), lam = -1. Q = diag(2, 0) is singular, so you can't solve with "
        f"Q alone or with a Cholesky factorization (the KKT matrix is indefinite): use LU (mlfs.linalg.solve). got "
        f"{x}, {lam}")


# ---------------------------------------------------------------- stage 4: the dual function and KKT


def test_stage4_dual_hand_example(nb):
    # min 1/2 x^2  s.t.  x >= 1, written as -x <= -1
    Q, c, G, h = np.array([[1.0]]), np.zeros(1), np.array([[-1.0]]), np.array([-1.0])
    for lam, want_g in ((0.0, 0.0), (0.5, 0.375), (1.0, 0.5), (2.0, 0.0)):
        g_val, x = nb.qp_dual(Q, c, G, h, np.array([lam]))
        assert np.isclose(g_val, want_g) and np.allclose(x, [lam]), (
            f"for min x^2/2 s.t. x >= 1: x(lam) = lam and g(lam) = -lam^2/2 + lam; at lam = {lam}: g = {want_g}, "
            f"x = {lam}; got g = {g_val}, x = {x}")


def test_stage4_weak_duality(nb):
    Q, c, G, h = _random_qp(4)
    rng = np.random.default_rng(4)
    best = _brute_force_qp(Q, c, G, h)
    for _ in range(30):
        lam = rng.exponential(size=4)
        g_val, _ = nb.qp_dual(Q, c, G, h, lam)
        assert g_val <= best[2] + 1e-9, (
            f"weak duality: g(lam) <= f(x) for every lam >= 0 and feasible x. g = {g_val:.6f} exceeds the optimal "
            f"value {best[2]:.6f}: check the sign of h^T lam and the factor 1/2")
    g_star, x_star = nb.qp_dual(Q, c, G, h, best[1])
    assert np.isclose(g_star, best[2]) and np.allclose(x_star, best[0]), (
        "at the optimal multipliers the dual value equals the primal optimum (strong duality) and x(lam*) = x*")


def test_stage4_kkt_residuals(nb):
    Q, c, G, h = _random_qp(5)
    x, lam, _ = _brute_force_qp(Q, c, G, h)
    r = nb.kkt_residuals(Q, c, G, h, x, lam)
    assert set(r) == {"stationarity", "primal", "dual", "complementarity"}, f"return the four named residuals; got {set(r)}"
    assert all(0 <= v < 1e-9 for v in r.values()), f"at the optimum every residual is 0 (and none is negative); got {r}"
    r0 = nb.kkt_residuals(Q, c, G, h, np.zeros(3), np.array([1.0, 1.0, 0.0, 0.0]))
    assert r0["primal"] == 0.0 and np.isclose(r0["complementarity"], max(h[0], h[1])), (
        f"x = 0 is strictly feasible (primal residual 0, not the negative max slack), and with lam = (1, 1, 0, 0) the "
        f"worst complementarity violation is max(|1 * h_0|, |1 * h_1|) = {max(h[0], h[1]):.4f}: take the max over "
        f"constraints, not the sum. got {r0}")
    bad = nb.kkt_residuals(Q, c, G, h, x + 10 * G[0], -np.abs(lam) - 0.5)
    assert bad["primal"] > 1 and np.isclose(bad["dual"], np.max(np.abs(lam) + 0.5)) and bad["stationarity"] > 1, (
        f"a point far outside the feasible set with negative multipliers must show primal, dual and stationarity "
        f"violations; got {bad}")
    slack = G @ x - h
    j = int(np.argmin(np.abs(slack) < 1e-9))  # an inactive constraint (slack < 0)
    lam2 = lam.copy()
    lam2[j] = 2.0
    r2 = nb.kkt_residuals(Q, c, G, h, x, lam2)
    assert np.isclose(r2["complementarity"], abs(2.0 * slack[j])), (
        f"a positive multiplier on an inactive constraint (slack {slack[j]:.3f}) violates complementary slackness by "
        f"|lam_j (G x - h)_j| = {abs(2 * slack[j]):.4f}; got {r2['complementarity']}")


# ---------------------------------------------------------------- stage 5: solving through the dual


def test_stage5_matches_brute_force(nb):
    for seed in (4, 6, 9, 11, 14):
        Q, c, G, h = _random_qp(seed)
        x_bf, lam_bf, val = _brute_force_qp(Q, c, G, h)
        x, lam, gap = bounded(nb.solve_qp_dual, Q, c, G, h)
        assert np.allclose(x, x_bf, atol=1e-6) and np.allclose(lam, lam_bf, atol=1e-5), (
            f"seed {seed}: the dual route should find x* = {np.round(x_bf, 5)} and lam* = {np.round(lam_bf, 5)} (by "
            f"brute force over active sets); got x = {np.round(x, 5)}, lam = {np.round(lam, 5)}")
        assert abs(gap) < 1e-7, f"at the optimum the duality gap f(x*) - g(lam*) is 0 (strong duality); got {gap:.2e}"
        assert np.all(lam >= 0), "multipliers of inequality constraints are nonnegative: project onto lam >= 0"


def test_stage5_inactive_constraints_and_options(nb):
    Q, c = np.eye(2), np.array([-0.2, -0.1])
    G, h = np.eye(2), np.array([1.0, 1.0])  # the unconstrained minimizer (0.2, 0.1) is feasible
    x, lam, gap = bounded(nb.solve_qp_dual, Q, c, G, h)
    assert np.allclose(x, [0.2, 0.1]) and np.allclose(lam, 0), (
        f"when the unconstrained minimizer is feasible, no constraint is active and every multiplier is 0; got x = "
        f"{x}, lam = {lam}")
    Q, c, G, h = _random_qp(5)
    x, lam, gap = bounded(nb.solve_qp_dual, Q, c, G, h, max_iters=3)
    x_bf = _brute_force_qp(Q, c, G, h)[0]
    assert not np.allclose(x, x_bf, atol=1e-6), "max_iters must be passed through (3 steps can't be enough here)"
    x1, lam1, _ = bounded(nb.solve_qp_dual, Q, c, G, h, max_iters=1)
    L = np.linalg.eigvalsh(G @ np.linalg.solve(Q, G.T)).max()
    want = np.maximum(0.0, (G @ np.linalg.solve(Q, -c) - h) / L)
    assert np.allclose(lam1, want), (
        f"one projected ascent step from lam = 0 with the default lr = 1/L (L = largest eigenvalue of G Q^-1 G^T = "
        f"{L:.4f}) gives lam = max(0, (G x(0) - h) / L) = {np.round(want, 5)}; got {np.round(lam1, 5)}")
    g_val, _ = nb.qp_dual(Q, c, G, h, lam)
    assert np.isclose(gap, 0.5 * x @ Q @ x + c @ x - g_val), (
        f"gap is f(x(lam)) - g(lam) at the returned lam (here after only 3 steps): expected "
        f"{0.5 * x @ Q @ x + c @ x - g_val:.6f}, got {gap:.6f}")
