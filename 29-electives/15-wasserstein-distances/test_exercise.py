"""Checks for exercise.py (Wasserstein distances).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The 1-D closed form is
checked against scipy.stats.wasserstein_distance and an independent LP; the Kantorovich LP against
scipy's linear_sum_assignment and hand examples; the transportation simplex against HiGHS and its own
optimality certificate (dual feasibility, complementary slackness, strong duality); the Gaussian closed
forms against ch 21's Fréchet distance and their defining equations. All data are synthetic.
"""
import itertools

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment, linprog
from scipy.stats import wasserstein_distance

from mlfs.generative import frechet_distance
from mlfs.testing import bounded


def _b(fn, *args, **kwargs):
    """run a solver call under a line budget, so a loop without an exit fails instead of hanging."""
    return bounded(fn, *args, budget=2_000_000, what=getattr(fn, "__name__", "the function"),
                   hint="check that every loop (the north-west walk, the tree searches, the pivots) makes progress "
                        "and has an exit: e.g. a breadth-first search must pop from its queue, not peek",
                   **kwargs)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _lp(a, b, C):
    """an independent LP oracle (HiGHS) for min <C, P> over the transport polytope."""
    n, m = C.shape
    A = np.zeros((n + m, n * m))
    for i in range(n):
        A[i, i * m:(i + 1) * m] = 1
    for j in range(m):
        A[n + j, j::m] = 1
    res = linprog(C.ravel(), A_eq=A, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    return res.x.reshape(n, m), res.fun


# ---------------------------------------------------------------- stage 1: the 1-D closed form


def test_stage1_worked_example(nb):
    u, wu = [0.0, 2.0], [0.5, 0.5]
    v, wv = [1.0, 3.0], [0.25, 0.75]
    w1 = nb.wasserstein_1d(u, v, 1, u_weights=wu, v_weights=wv)
    w2 = nb.wasserstein_1d(u, v, 2, u_weights=wu, v_weights=wv)
    assert np.isclose(w1, 1.5), (
        f"the lesson's worked example: W1(½δ0 + ½δ2, ¼δ1 + ¾δ3) = ¼·1 + ¼·3 + ½·1 = 1.5, got {w1}. the quantile "
        "functions change value at t = ¼ and t = ½: integrate |F⁻¹(t) − G⁻¹(t)| piece by piece")
    assert np.isclose(w2, np.sqrt(3.0)), (
        f"W2 of the same pair is sqrt(¼·1 + ¼·9 + ½·1) = sqrt(3) = 1.732, got {w2}. if you got 3.0, you returned "
        "W_p^p instead of W_p; if you got 1.5, you integrated |F − G| (that's only W1)")


def test_stage1_matches_scipy_w1(nb):
    g = _rng(1)
    for _ in range(10):
        n, m = g.integers(1, 15, size=2)
        u, v = g.normal(size=n) * 2, g.normal(size=m) + g.normal()
        wu, wv = g.random(n) + 0.01, g.random(m) + 0.01
        got = nb.wasserstein_1d(u, v, 1, u_weights=wu, v_weights=wv)
        want = wasserstein_distance(u, v, wu, wv)
        assert np.isclose(got, want), (
            f"W1 with {n} vs {m} unsorted, weighted points: scipy.stats.wasserstein_distance gives {want:.6f}, "
            f"you gave {got:.6f}. sort each sample *together with its weights*, and normalize the weights to sum 1")
    u, v = g.normal(size=9), g.normal(size=4)
    assert np.isclose(nb.wasserstein_1d(u, v), wasserstein_distance(u, v)), "without weights every point gets 1/n"


def test_stage1_matches_lp_for_p_above_one(nb):
    g = _rng(2)
    for p in (2, 3, 1.5):
        n, m = 6, 9
        u, v = g.normal(size=n), g.normal(size=m) * 1.5 + 0.3
        wu, wv = g.random(n) + 0.1, g.random(m) + 0.1
        a, b = wu / wu.sum(), wv / wv.sum()
        _, cost = _lp(a, b, np.abs(u[:, None] - v[None, :]) ** p)
        got = nb.wasserstein_1d(u, v, p, u_weights=wu, v_weights=wv)
        assert np.isclose(got, cost ** (1 / p), rtol=1e-6), (
            f"p = {p}: the transport LP gives W_p = {cost ** (1 / p):.6f}, you gave {got:.6f}. for p > 1 the "
            "closed form is (∫₀¹ |F⁻¹(t) − G⁻¹(t)|^p dt)^(1/p), over the merged breakpoints of both quantile functions")


def test_stage1_weights_are_normalized_and_order_free(nb):
    g = _rng(3)
    u, v, wu, wv = g.normal(size=8), g.normal(size=5), g.random(8) + 0.1, g.random(5) + 0.1
    base = nb.wasserstein_1d(u, v, 2, u_weights=wu, v_weights=wv)
    scaled = nb.wasserstein_1d(u, v, 2, u_weights=7 * wu, v_weights=0.1 * wv)
    assert np.isclose(base, scaled), "weights are relative: multiplying all of u's weights by 7 must not change W_p"
    perm = g.permutation(8)
    shuffled = nb.wasserstein_1d(u[perm], v, 2, u_weights=wu[perm], v_weights=wv)
    assert np.isclose(base, shuffled), "the order in which points are listed is irrelevant (with their weights)"
    zero = nb.wasserstein_1d(np.r_[u, 50.0], v, 2, u_weights=np.r_[wu, 0.0], v_weights=wv)
    assert np.isclose(base, zero), "a point with weight 0 carries no mass and must not change the distance"
    assert np.isclose(nb.wasserstein_1d(u, u + 0.7, 3), 0.7), "a pure translation by 0.7 has W_p = 0.7 for every p"


# ---------------------------------------------------------------- stage 2: the Kantorovich LP


def test_stage2_cost_matrix(nb):
    g = _rng(4)
    X, Y = g.normal(size=(5, 3)), g.normal(size=(7, 3))
    D = np.sqrt(((X[:, None] - Y[None]) ** 2).sum(-1))
    for p in (1, 2, 3):
        got = np.asarray(nb.cost_matrix(X, Y, p))
        assert got.shape == (5, 7), f"(n, m) = (5, 7), got {got.shape}"
        assert np.allclose(got, D ** p), f"p = {p}: entry (i, j) is ||x_i − y_j||^p with the Euclidean norm"
    assert np.allclose(nb.cost_matrix(X, Y), D ** 2), "the default is p = 2, the squared Euclidean cost"
    u, v = g.normal(size=4), g.normal(size=6)
    assert np.allclose(nb.cost_matrix(u, v, 1), np.abs(u[:, None] - v[None])), "1-D inputs (n,) are points on a line"


def test_stage2_transport_constraints(nb):
    g = _rng(5)
    for n, m in [(2, 3), (4, 1), (3, 5)]:
        A = np.asarray(nb.transport_constraints(n, m))
        assert A.shape == (n + m, n * m), f"n + m = {n + m} constraints on n·m = {n * m} variables, got {A.shape}"
        P = g.random((n, m))
        assert np.allclose(A @ P.ravel(), np.r_[P.sum(1), P.sum(0)]), (
            f"(n, m) = ({n}, {m}): the first n rows of A @ P.ravel() must be the row sums of P, the last m its column "
            "sums. P.ravel() is row-major: entry (i, j) is at index i·m + j")


def test_stage2_lp_matches_assignment(nb):
    g = _rng(6)
    for n in (4, 9, 16):
        X, Y = g.normal(size=(n, 2)), g.normal(size=(n, 2)) + 1
        C = ((X[:, None] - Y[None]) ** 2).sum(-1)
        a = np.full(n, 1 / n)
        P, cost = nb.kantorovich_lp(a, a, C)
        P = np.asarray(P)
        r, c = linear_sum_assignment(C)
        assert P.shape == (n, n) and np.all(P >= -1e-12), "P is an (n, m) array of nonnegative masses"
        assert np.allclose(P.sum(1), a, atol=1e-9) and np.allclose(P.sum(0), a, atol=1e-9), "P's marginals are a and b"
        assert np.isclose(cost, C[r, c].mean()), (
            f"n = {n} uniform points on each side: the optimum is a permutation (an assignment) with cost "
            f"{C[r, c].mean():.6f}, you got {cost:.6f}")
        assert np.isclose(cost, np.sum(P * C)), "the returned cost is <C, P> for the returned plan"


def test_stage2_lp_hand_example_and_mass(nb):
    a, b = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    C = np.abs(np.array([0.0, 2.0])[:, None] - np.array([1.0, 3.0])[None, :])
    P, cost = nb.kantorovich_lp(a, b, C)
    assert np.allclose(P, [[0.25, 0.25], [0.0, 0.5]], atol=1e-9) and np.isclose(cost, 1.5), (
        f"the worked example: P = [[¼, ¼], [0, ½]] with cost 1.5, got P = {np.round(P, 4).tolist()}, cost {cost}")
    P3, cost3 = nb.kantorovich_lp(3 * a, 3 * b, C)
    assert np.isclose(cost3, 4.5) and np.allclose(P3, 3 * np.asarray(P), atol=1e-9), (
        "masses need not sum to 1, only to the same total: tripling both triples the plan and the cost")
    with pytest.raises(ValueError):
        nb.kantorovich_lp(a, 2 * b, C)  # balanced OT: unequal total masses have no feasible plan


def test_stage2_lp_matches_1d_closed_form(nb):
    g = _rng(7)
    u, v = g.normal(size=7), g.normal(size=10) + 0.5
    wu, wv = g.random(7) + 0.1, g.random(10) + 0.1
    a, b = wu / wu.sum(), wv / wv.sum()
    _, cost = nb.kantorovich_lp(a, b, np.asarray(nb.cost_matrix(u, v, 2)))
    _, want = _lp(a, b, (u[:, None] - v[None]) ** 2)
    assert np.isclose(cost, want), f"7 × 10 weighted points, squared cost: the LP optimum is {want:.6f}, got {cost:.6f}"


# ---------------------------------------------------------------- stage 3: the transportation simplex


def test_stage3_northwest_corner(nb):
    P, basis = _b(nb.northwest_corner, [0.5, 0.5], [0.25, 0.75])
    assert np.allclose(P, [[0.25, 0.25], [0.0, 0.5]]), (
        f"north-west corner of a = (½, ½), b = (¼, ¾): fill (0,0) with ¼, move right, (0,1) gets ¼, move down, "
        f"(1,1) gets ½. got {np.round(np.asarray(P), 4).tolist()}")
    assert [tuple(map(int, c)) for c in basis] == [(0, 0), (0, 1), (1, 1)], f"basic cells in visiting order, got {basis}"
    P, basis = _b(nb.northwest_corner, [0.5, 0.5], [0.5, 0.5])
    assert [tuple(map(int, c)) for c in basis] == [(0, 0), (1, 0), (1, 1)] and np.allclose(P, np.eye(2) / 2), (
        "degenerate case a = b = (½, ½): after (0,0) gets ½ both row 0 and column 0 are used up; the rule moves DOWN "
        f"on a tie, so (1, 0) enters the basis with mass 0, then (1, 1). got basis {basis}")
    g = _rng(8)
    a, b = g.random(5) + 0.1, g.random(7) + 0.1
    b *= a.sum() / b.sum()
    P, basis = _b(nb.northwest_corner, a, b)
    P = np.asarray(P)
    assert np.allclose(P.sum(1), a) and np.allclose(P.sum(0), b), "the north-west corner plan is feasible"
    cells = [tuple(map(int, c)) for c in basis]
    assert len(cells) == 5 + 7 - 1 and cells[0] == (0, 0) and cells[-1] == (4, 6), (
        f"a staircase of n + m − 1 = 11 cells from (0, 0) to (n−1, m−1), got {len(cells)} from {cells[0]} to {cells[-1]}")
    steps = {(i2 - i1, j2 - j1) for (i1, j1), (i2, j2) in zip(cells, cells[1:])}
    assert steps <= {(1, 0), (0, 1)}, "every step goes one cell down or one cell right"


def _check_optimal(a, b, C, out, what):
    P, f, g = (np.asarray(x, dtype=float) for x in out)
    _, want = _lp(a, b, C)
    tol = 1e-8 * (1 + np.abs(C).max())
    assert P.shape == C.shape and np.all(P >= -1e-12), f"{what}: P must be a nonnegative (n, m) plan"
    assert np.allclose(P.sum(1), a, atol=1e-9) and np.allclose(P.sum(0), b, atol=1e-9), f"{what}: marginals a and b"
    assert np.isclose(np.sum(P * C), want, rtol=1e-8, atol=1e-10), (
        f"{what}: HiGHS finds cost {want:.8f}, your plan costs {np.sum(P * C):.8f}")
    assert f.shape == a.shape and g.shape == b.shape, "f has one entry per row, g one per column"
    assert np.all(C - f[:, None] - g[None, :] >= -tol), (
        f"{what}: dual infeasible: some f_i + g_j > C_ij (by {-(C - f[:, None] - g[None, :]).min():.2e}). at the "
        "optimum every reduced cost C_ij − f_i − g_j is >= 0")
    assert np.all(np.abs((C - f[:, None] - g[None, :])[P > 1e-12]) <= tol), (
        f"{what}: complementary slackness: wherever P_ij > 0, f_i + g_j must equal C_ij")
    assert np.isclose(a @ f + b @ g, want, rtol=1e-8, atol=1e-10), (
        f"{what}: strong duality: <a, f> + <b, g> = {a @ f + b @ g:.8f} should equal the optimal cost {want:.8f}")
    assert np.isclose(f[0], 0.0, atol=1e-12), f"potentials are normalized with f[0] = 0, got f[0] = {f[0]}"
    assert np.count_nonzero(P > 1e-12) <= len(a) + len(b) - 1, (
        f"{what}: a simplex solution is a vertex, with at most n + m − 1 nonzero cells")


def test_stage3_simplex_is_optimal_with_certificate(nb):
    g = _rng(9)
    for k in range(8):
        n, m = g.integers(2, 9, size=2)
        X, Y = g.normal(size=(n, 2)), g.normal(size=(m, 2)) * 1.3 + 0.5
        a, b = g.random(n) + 0.05, g.random(m) + 0.05
        a, b = a / a.sum(), b / b.sum()
        C = ((X[:, None] - Y[None]) ** 2).sum(-1)
        out = bounded(nb.transport_simplex, a, b, C, what="transport_simplex",
                      hint="every pivot must strictly follow the rules; a missing exit leaves it pivoting forever")
        _check_optimal(a, b, C, out, f"random {n}×{m} problem #{k}")


def test_stage3_degenerate_problems_terminate(nb):
    g = _rng(10)
    n = 10
    X, Y = g.normal(size=(n, 2)), g.normal(size=(n, 2))
    C = ((X[:, None] - Y[None]) ** 2).sum(-1)
    a = np.full(n, 1 / n)
    out = bounded(nb.transport_simplex, a, a, C, budget=3_000_000, what="transport_simplex on an assignment problem",
                  hint="uniform equal-size marginals make most pivots degenerate (θ = 0); without an anti-cycling "
                       "rule (Bland's: smallest index enters, smallest index leaves on ties) the simplex can cycle")
    _check_optimal(a, a, C, out, "10×10 assignment problem")
    r, c = linear_sum_assignment(C)
    assert np.isclose(np.sum(np.asarray(out[0]) * C), C[r, c].mean()), "the assignment optimum"
    Ci = g.integers(0, 3, size=(6, 7)).astype(float)  # many ties and many optimal plans
    ai, bi = np.full(6, 7.0), np.full(7, 6.0)
    out = bounded(nb.transport_simplex, ai, bi, Ci, budget=3_000_000, what="transport_simplex with tied costs")
    _check_optimal(ai, bi, Ci, out, "integer costs in {0, 1, 2}, integer masses")


def test_stage3_max_iters_and_mass(nb):
    a, b = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    C_sorted = np.abs(np.array([0.0, 2.0])[:, None] - np.array([1.0, 3.0])[None, :]) ** 2
    P, _, _ = _b(nb.transport_simplex, a, b, C_sorted, max_iters=0)
    assert np.allclose(P, [[0.25, 0.25], [0.0, 0.5]]), (
        "for sorted 1-D points and a convex cost the north-west corner plan is already optimal, so max_iters=0 "
        "(no pivots) must return it")
    C_rev = C_sorted[:, ::-1].copy()  # the same points with the targets listed in reverse: NW corner is wrong
    with pytest.raises(RuntimeError):
        _b(nb.transport_simplex, a, b[::-1].copy(), C_rev, max_iters=0)
    with pytest.raises(ValueError):
        _b(nb.transport_simplex, a, 2 * b, C_sorted)


def test_stage3_brute_force_vertices(nb):
    g = _rng(11)
    n = 5
    C = g.random((n, n))
    a = np.full(n, 1 / n)
    best = min(C[np.arange(n), list(p)].mean() for p in itertools.permutations(range(n)))
    P, f, gg = _b(nb.transport_simplex, a, a, C)
    assert np.isclose(np.sum(np.asarray(P) * C), best), (
        "by Birkhoff's theorem the optimum of a uniform n × n problem is at a permutation: brute force over all 120 "
        f"permutations gives {best:.6f}, you got {np.sum(np.asarray(P) * C):.6f}")


# ---------------------------------------------------------------- stage 4: Monge maps and Gaussians


def test_stage4_barycentric_map(nb):
    P = np.array([[0.25, 0.25], [0.0, 0.5]])
    Y = np.array([1.0, 3.0])
    T = np.asarray(nb.barycentric_map(P, Y))
    assert T.shape == (2,) and np.allclose(T, [2.0, 3.0]), (
        f"the worked example: x = 0 sends half its mass to 1 and half to 3, so T(0) = 2; T(2) = 3. got {T}")
    g = _rng(12)
    perm = g.permutation(6)
    Y2 = g.normal(size=(6, 3))
    Pperm = np.eye(6)[perm] / 6
    assert np.allclose(nb.barycentric_map(Pperm, Y2), Y2[perm]), "a permutation plan maps x_i to y_{σ(i)} exactly"
    Pr = g.random((4, 5)) * 3  # total mass 3·(something): rows need not sum to 1/n
    T = np.asarray(nb.barycentric_map(Pr, Y2[:5]))
    assert np.allclose(T, (Pr / Pr.sum(1, keepdims=True)) @ Y2[:5]), (
        "divide by each row's own mass sum_j P_ij, not by 1/n: T(x_i) is the mean of the targets x_i is sent to")
    assert np.allclose(Pr.sum(1) @ T, Pr.sum(0) @ Y2[:5]), "mass-weighted, the images average to the targets' mean"


def test_stage4_gaussian_w2_closed_forms(nb):
    got = nb.gaussian_w2([1.0], [[4.0]], [-2.0], [[1.0]])
    assert np.isclose(got, np.sqrt(9 + 1)), (
        f"1-D: W2(N(1, 4), N(−2, 1))² = (1 − (−2))² + (2 − 1)² = 10, so W2 = 3.162; got {got}. (if you got 10, "
        "return the distance, not its square)")
    D1, D2 = np.diag([1.0, 4.0, 9.0]), np.diag([4.0, 1.0, 1.0])
    want = np.sqrt(1 + ((np.sqrt(np.diag(D1)) - np.sqrt(np.diag(D2))) ** 2).sum())
    assert np.isclose(nb.gaussian_w2(np.zeros(3), D1, np.r_[1.0, 0, 0], D2), want), (
        "diagonal (commuting) covariances: ||m1 − m2||² + Σ_k (σ1k − σ2k)²")
    g = _rng(13)
    for _ in range(5):
        A1, A2 = g.normal(size=(4, 4)), g.normal(size=(4, 4))
        S1, S2 = A1 @ A1.T + 0.1 * np.eye(4), A2 @ A2.T + 0.1 * np.eye(4)
        m1, m2 = g.normal(size=4), g.normal(size=4)
        got = nb.gaussian_w2(m1, S1, m2, S2)
        want = np.sqrt(frechet_distance(m1, S1, m2, S2))
        assert np.isclose(got, want, rtol=1e-6), (
            f"non-commuting covariances: ch 21's Fréchet distance is W2², sqrt = {want:.6f}, got {got:.6f}. the cross "
            "term is tr((S1^½ S2 S1^½)^½): symmetric square roots, not sqrt(S1 S2) elementwise or by products")
        assert np.isclose(nb.gaussian_w2(m1, S1, m1, S1), 0.0, atol=1e-6), "a Gaussian is at distance 0 from itself"
        assert np.isclose(got, nb.gaussian_w2(m2, S2, m1, S1), rtol=1e-6), "W2 is symmetric"


def test_stage4_gaussian_w2_singular_covariances(nb):
    got = nb.gaussian_w2(np.zeros(2), np.diag([1.0, 0.0]), np.zeros(2), np.diag([4.0, 1.0]))
    assert np.isfinite(got) and np.isclose(got, np.sqrt(2.0)), (
        f"a degenerate Gaussian (covariance diag(1, 0), all its mass on a line) is allowed: W2² = (1 − 2)² + (0 − 1)² "
        f"= 2, got {got}. the covariances are only positive SEMI-definite: no Cholesky, and clip eigenvalues that "
        "rounding makes slightly negative to 0 before the square root")
    for th in (0.3, 0.6, 1.1, 2.0):
        u = np.array([np.cos(th), np.sin(th)])
        S_line, S_full = 2.0 * np.outer(u, u), np.array([[1.5, 0.4], [0.4, 0.8]])
        want = np.sqrt(2.0 + np.trace(S_full) - 2 * np.sqrt(2.0 * u @ S_full @ u))
        for args in [(S_line, S_full), (S_full, S_line)]:
            got = nb.gaussian_w2(np.zeros(2), args[0], np.ones(2), args[1])
            want_m = np.sqrt(want ** 2 + 2.0)
            assert np.isfinite(got) and np.isclose(got, want_m, rtol=1e-6), (
                f"rank-one covariance along angle {th}: W2 = {want_m:.6f}, got {got}. rounding makes some eigenvalues "
                "of a singular PSD matrix (or of S1^½ S2 S1^½) slightly negative: clip them to 0 before sqrt")


def test_stage4_gaussian_monge_map(nb):
    g = _rng(14)
    A1, A2 = g.normal(size=(3, 3)), g.normal(size=(3, 3))
    S1, S2 = A1 @ A1.T + 0.2 * np.eye(3), A2 @ A2.T + 0.2 * np.eye(3)
    m1, m2 = g.normal(size=3), g.normal(size=3)
    A, t = (np.asarray(x, dtype=float) for x in nb.gaussian_monge_map(m1, S1, m2, S2))
    assert A.shape == (3, 3) and t.shape == (3,), f"A (d, d) and t (d,), got {A.shape}, {t.shape}"
    assert np.allclose(A, A.T, atol=1e-8), (
        "the optimal map is the gradient of a convex function (Brenier), so its matrix is SYMMETRIC positive definite. "
        "with rows, x @ A has covariance Aᵀ S1 A, so A = S1^−½ S2^½ pushes the right covariance forward, but it isn't "
        "symmetric and isn't optimal")
    assert np.all(np.linalg.eigvalsh((A + A.T) / 2) > 0), "A is positive definite"
    assert np.allclose(A.T @ S1 @ A, S2, atol=1e-8), "T(x) = x @ A + t pushes N(m1, S1) to covariance Aᵀ S1 A = S2"
    assert np.allclose(m1 @ A + t, m2), "T maps the mean m1 to m2"
    IA = np.eye(3) - A
    cost = np.sum((m1 - (m1 @ A + t)) ** 2) + np.trace(IA.T @ S1 @ IA)
    assert np.isclose(cost, frechet_distance(m1, S1, m2, S2), rtol=1e-6), (
        "E||x − T(x)||² under N(m1, S1) must equal W2², i.e. the map is optimal, not just mass-preserving")
    A_1d, t_1d = nb.gaussian_monge_map([1.0], [[4.0]], [-2.0], [[1.0]])
    assert np.allclose(np.asarray(A_1d).ravel(), [0.5]) and np.allclose(np.asarray(t_1d).ravel(), [-2.5]), (
        "1-D: T(x) = m2 + (σ2/σ1)(x − m1) = −2 + ½(x − 1)")
