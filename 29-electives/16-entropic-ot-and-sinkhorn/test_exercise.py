"""Checks for exercise.py (entropic OT & Sinkhorn).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Sinkhorn's plans are
checked against the structure every entropic plan must have (diagonal scalings of exp(-C/eps), so
fixed cross-ratios), against an independent log-domain implementation, and against the exact LP
(scipy's HiGHS) as eps -> 0, with the bound <P_eps, C> - W <= eps (H(a) + H(b)). The objectives are
checked by hand, by weak and strong duality, and by the monotone ascent of the dual along the
iterations. The divergence is checked by exact identities (zero on equal inputs, |t|^2 for a
translation at every eps, the squared mean difference as eps -> infinity). All data are synthetic.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp

from mlfs.testing import bounded


def _b(fn, *args, _budget=3_000_000, **kwargs):
    """run a Sinkhorn-style call under a line budget, so a loop without an exit fails instead of hanging."""
    return bounded(fn, *args, budget=_budget, what=getattr(fn, "__name__", "the function"),
                   hint="every Sinkhorn loop must stop after max_iters iterations, even if tol is never reached",
                   **kwargs)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _problem(n=6, m=8, seed=0, shift=0.5, uniform=False):
    g = _rng(seed)
    X, Y = g.normal(size=(n, 2)), g.normal(size=(m, 2)) + shift
    a, b = (np.full(n, 1 / n), np.full(m, 1 / m)) if uniform else (g.random(n) + 0.2, g.random(m) + 0.2)
    return X, Y, a / a.sum(), b / b.sum(), ((X[:, None] - Y[None]) ** 2).sum(-1)


def _oracle(a, b, C, eps, iters=5000):
    """independent log-domain Sinkhorn (SciPy's logsumexp), run to convergence."""
    g = np.zeros(len(b))
    for _ in range(iters):
        f = -eps * logsumexp(np.log(b)[None] + (g[None] - C) / eps, axis=1)
        g = -eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps, axis=0)
        P = np.exp(np.log(a)[:, None] + np.log(b)[None] + (f[:, None] + g[None] - C) / eps)
        if np.abs(P.sum(1) - a).sum() < 1e-12:
            break
    return P, f, g


def _lp(a, b, C):
    n, m = C.shape
    A = np.vstack([np.kron(np.eye(n), np.ones((1, m))), np.kron(np.ones((1, n)), np.eye(m))])
    res = linprog(C.ravel(), A_eq=A, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    return res.x.reshape(n, m), res.fun


def _cross_ratio_error(P, C, eps):
    """entropic plans are diag(u) exp(-C/eps) diag(v): log P + C/eps must be a_i + b_j (double-centred to 0)."""
    L = np.log(P) + C / eps
    L = L - L.mean(0, keepdims=True) - L.mean(1, keepdims=True) + L.mean()
    return np.abs(L).max()


# ---------------------------------------------------------------- stage 1: Sinkhorn's scalings


def test_stage1_worked_example(nb):
    C = np.array([[0.0, 1.0], [1.0, 0.0]])
    P = np.asarray(_b(nb.sinkhorn, [0.5, 0.5], [0.5, 0.5], C, 1.0))
    p = 0.5 / (1 + np.exp(-1.0))
    assert P.shape == (2, 2) and np.allclose(P, [[p, 0.5 - p], [0.5 - p, p]], atol=1e-8), (
        f"the lesson's worked example: with C = [[0, 1], [1, 0]], uniform marginals and eps = 1, P = [[p, ½ − p], "
        f"[½ − p, p]] with p = ½·σ(1/eps) = {p:.5f}; got {np.round(P, 5).tolist()}")


def test_stage1_constant_cost_gives_independent_plan(nb):
    g = _rng(1)
    a, b = g.random(4) + 0.1, g.random(6) + 0.1
    a, b = a / a.sum(), b / b.sum()
    P = np.asarray(_b(nb.sinkhorn, a, b, np.full((4, 6), 3.0), 0.7))
    assert np.allclose(P, np.outer(a, b), atol=1e-10), (
        "when every route costs the same, the entropic plan is the independent coupling a bᵀ: nothing favours any "
        "pairing, and entropy spreads the mass as evenly as the marginals allow")


def test_stage1_marginals_and_scaling_structure(nb):
    for seed, eps, m in [(2, 1.0, 8), (3, 0.5, 8), (4, 2.0, 6)]:
        _, _, a, b, C = _problem(m=m, seed=seed)
        P = np.asarray(bounded(nb.sinkhorn, a, b, C, eps, max_iters=20000, what="sinkhorn",
                               hint="the loop must stop after max_iters iterations even if tol is never reached"))
        assert P.shape == C.shape and np.all(P > 0), "an entropic plan is (n, m) and strictly positive everywhere"
        assert np.abs(P.sum(1) - a).sum() <= 1e-8, (
            f"rows must sum to a within tol = 1e-9 (L1) after at most 20,000 iterations; off by {np.abs(P.sum(1) - a).sum():.2e}")
        assert np.allclose(P.sum(0), b, atol=1e-12), (
            "after the last v-update the columns sum to b exactly: update u from v, then v from u, then form P")
        err = _cross_ratio_error(P, C, eps)
        assert err < 1e-6, (
            f"eps = {eps}: log P + C/eps must split as α_i + β_j (P = diag(u) exp(−C/eps) diag(v)), off by {err:.2e}. "
            "check the kernel: exp(−C / eps), not exp(C / eps) or exp(−C)")
        want, _, _ = _oracle(a, b, C, eps)
        assert np.allclose(P, want, atol=1e-8), "the plan must match an independent Sinkhorn run to convergence"


def test_stage1_one_iteration_and_tol(nb):
    _, _, a, b, C = _problem(seed=5)
    eps = 0.8
    K = np.exp(-C / eps)
    u = a / K.sum(1)
    v = b / (K.T @ u)
    want = u[:, None] * K * v[None, :]
    got = np.asarray(_b(nb.sinkhorn, a, b, C, eps, max_iters=1))
    assert np.allclose(got, want, rtol=1e-10), (
        "max_iters=1 is one full iteration from v = 1: u = a / (K v), then v = b / (Kᵀ u); P = diag(u) K diag(v)")
    loose = np.asarray(_b(nb.sinkhorn, a, b, C, eps, tol=10.0))
    assert np.allclose(loose, want, rtol=1e-10), (
        "with tol = 10 the row-marginal error (L1) is below tol after the first iteration, so it must stop there")
    tight = np.asarray(_b(nb.sinkhorn, a, b, C, eps, max_iters=3))
    assert not np.allclose(tight, want), "max_iters=3 must run more than one iteration"


def _naive_iterates(a, b, C, eps, k):
    K = np.exp(-C / eps)
    v, out = np.ones(len(b)), []
    for _ in range(k):
        u = a / (K @ v)
        v = b / (K.T @ u)
        out.append(u[:, None] * K * v[None, :])
    return out


def _log_iterates(a, b, C, eps, k):
    g, out = np.zeros(len(b)), []
    for _ in range(k):
        f = -eps * logsumexp(np.log(b)[None] + (g[None] - C) / eps, axis=1)
        g = -eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps, axis=0)
        out.append(np.exp(np.log(a)[:, None] + np.log(b)[None] + (f[:, None] + g[None] - C) / eps))
    return out


def _boundary_tol(its, a):
    """a tol at which stopping on the L1 and on the max row error differ, and the iteration the L1 rule stops at."""
    l1 = np.array([np.abs(P.sum(1) - a).sum() for P in its])
    mx = np.array([np.abs(P.sum(1) - a).max() for P in its])
    for k0 in range(len(its)):  # a tol at which the two stopping rules pick different iterations
        tol = (mx[k0] + l1[k0]) / 2
        if (l1 <= tol).any() and np.argmax(l1 <= tol) != np.argmax(mx <= tol):
            return tol, int(np.argmax(l1 <= tol))
    raise AssertionError("test setup: no tol separates the L1 and max stopping rules")


def test_stage1_stops_on_the_l1_row_error(nb):
    _, _, a, b, C = _problem(n=6, m=8, seed=18)
    its = _naive_iterates(a, b, C, 0.5, 60)
    tol, k = _boundary_tol(its, a)
    got = np.asarray(_b(nb.sinkhorn, a, b, C, 0.5, tol=tol))
    assert np.allclose(got, its[k], rtol=1e-10, atol=1e-14), (
        f"stop as soon as the L1 row error sum_i |(P 1)_i − a_i| is <= tol. with tol = {tol:.3g} that happens after "
        f"iteration {k + 1}; the largest single row error drops below tol earlier, so if you stopped earlier you "
        "measured the max, not the sum")


# ---------------------------------------------------------------- stage 2: the log domain


def test_stage2_matches_scaling_version(nb):
    for seed, eps in [(6, 1.0), (7, 0.5)]:
        _, _, a, b, C = _problem(seed=seed)
        P, f, g = (np.asarray(x) for x in _b(nb.sinkhorn_log, a, b, C, eps, max_iters=5000))
        want, _, _ = _oracle(a, b, C, eps)
        assert P.shape == C.shape and f.shape == a.shape and g.shape == b.shape, "returns (P (n, m), f (n,), g (m,))"
        assert np.allclose(P, want, atol=1e-8), f"eps = {eps}: the converged plan must equal Sinkhorn's"
        assert np.allclose(P, np.outer(a, b) * np.exp((f[:, None] + g[None, :] - C) / eps), rtol=1e-8), (
            "P_ij = a_i b_j exp((f_i + g_j − C_ij) / eps): with this convention f, g are the dual potentials")


def test_stage2_fixed_point_and_first_iteration(nb):
    _, _, a, b, C = _problem(seed=8)
    eps = 0.5
    P, f, g = (np.asarray(x) for x in _b(nb.sinkhorn_log, a, b, C, eps))
    g_from_f = -eps * logsumexp(np.log(a)[:, None] + (f[:, None] - C) / eps, axis=0)
    assert np.allclose(g, g_from_f, atol=1e-10), (
        "g is the last update: g_j = −eps log Σ_i a_i exp((f_i − C_ij)/eps), so this equation holds exactly")
    f1 = -eps * logsumexp(np.log(b)[None] - C / eps, axis=1)
    g1 = -eps * logsumexp(np.log(a)[:, None] + (f1[:, None] - C) / eps, axis=0)
    P1, F1, G1 = (np.asarray(x) for x in _b(nb.sinkhorn_log, a, b, C, eps, max_iters=1))
    assert np.allclose(F1, f1) and np.allclose(G1, g1), (
        "max_iters=1 from g = 0: f = −eps LSE_j(log b_j + (g_j − C_ij)/eps), then g from f the same way. (did you "
        "forget the log-weights log a, log b inside the log-sum-exps?)")
    P10, F10, G10 = (np.asarray(x) for x in _b(nb.sinkhorn_log, a, b, C, eps, tol=1e3))
    assert np.allclose(F10, f1) and np.allclose(G10, g1), "a huge tol stops after the first iteration"


def test_stage2_stops_on_the_l1_row_error(nb):
    _, _, a, b, C = _problem(n=6, m=8, seed=19)
    its = _log_iterates(a, b, C, 0.5, 60)
    tol, k = _boundary_tol(its, a)
    got = np.asarray(_b(nb.sinkhorn_log, a, b, C, 0.5, tol=tol)[0])
    assert np.allclose(got, its[k], rtol=1e-10, atol=1e-14), (
        f"stop as soon as the L1 row error sum_i |(P 1)_i − a_i| is <= tol. with tol = {tol:.3g} that happens after "
        f"iteration {k + 1} (counting from g = 0); stopping on the largest single row error stops earlier")


def test_stage2_stable_at_small_eps(nb):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    t = np.array([6.0, 3.0])
    a = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    C = ((X[:, None] - (X + t)[None]) ** 2).sum(-1)  # a weighted cloud and its translate: the optimal plan is diag(a)
    eps = 0.05
    assert np.all(np.exp(-np.diag(C) / eps) == 0.0)  # the naive kernel has underflowed exactly where the plan lives
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        P, f, g = (np.asarray(x) for x in bounded(
            nb.sinkhorn_log, a, a, C, eps, max_iters=2000, tol=1e-9, budget=10_000_000, what="sinkhorn_log",
            hint="the loop must stop after max_iters iterations even if tol is never reached"))
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(f)) and np.all(np.isfinite(g)), (
        "with costs up to 76 and eps = 0.05, exp(−C/eps) underflows: work with potentials and log-sum-exp only, "
        "never forming exp(−C/eps) on its own")
    assert np.allclose(P.sum(0), a, atol=1e-12), "after the last g-update the columns sum to b exactly"
    assert np.abs(P.sum(1) - a).sum() <= 1e-3, (
        f"after 2,000 iterations the rows should sum to a within 1e-3 (L1), got {np.abs(P.sum(1) - a).sum():.2e}")
    assert np.abs(P - np.diag(a)).sum() <= 2e-3, (
        "a translated cloud is matched to itself: the plan must be close to diag(a). (if your plan sits off the "
        "diagonal, some exp underflowed to 0 where it mattered)")
    Ha = -(a * np.log(a)).sum()
    gap = np.sum(P * C) - t @ t
    assert -1e-2 <= gap <= 2 * eps * Ha + 1e-2, (
        f"<P_eps, C> must be close to W2² = |t|² = {t @ t} (within eps (H(a) + H(b))), got {np.sum(P * C):.4f}")


def test_stage2_converges_to_the_lp(nb):
    _, _, a, b, C = _problem(n=6, m=7, seed=10)
    P_lp, w = _lp(a, b, C)
    Ha, Hb = -(a * np.log(a)).sum(), -(b * np.log(b)).sum()
    prev = np.inf
    for eps in (1.0, 0.3, 0.1):
        P, _, _ = bounded(nb.sinkhorn_log, a, b, C, eps, max_iters=20000, tol=1e-10, budget=20_000_000,
                          what="sinkhorn_log")
        gap = np.sum(np.asarray(P) * C) - w
        assert -1e-7 <= gap <= eps * (Ha + Hb) + 1e-7, (
            f"eps = {eps}: 0 <= <P_eps, C> − W <= eps (H(a) + H(b)) = {eps * (Ha + Hb):.4f}, got {gap:.5f}")
        assert gap <= prev + 1e-7, "the transport cost of the entropic plan decreases towards the LP's as eps shrinks"
        prev = gap


# ---------------------------------------------------------------- stage 3: primal and dual objectives


def test_stage3_hand_values(nb):
    p = 0.5 / (1 + np.exp(-1.0))
    P2 = np.array([[p, 0.5 - p], [0.5 - p, p]])
    h, C2 = np.array([0.5, 0.5]), np.array([[0.0, 1.0], [1.0, 0.0]])
    c = 0.5 * np.log(4 * p)
    assert np.isclose(nb.entropic_primal(P2, h, h, C2, 1.0), 0.3798855, atol=1e-6), (
        "the lesson's worked example: <C, P> = 0.26894 plus KL(P‖abᵀ) = 0.11094 gives 0.37989")
    assert np.isclose(nb.entropic_dual(np.full(2, c), np.full(2, c), h, h, C2, 1.0), 0.3798855, atol=1e-6), (
        "the worked example's potentials f = g = ½ log(4p) = 0.18994 give the same dual value 2c = 0.37989")
    a, b = np.array([0.25, 0.75]), np.array([0.5, 0.2, 0.3])
    C = np.array([[0.0, 1.0, 2.0], [3.0, 1.0, 0.5]])
    ab = np.outer(a, b)
    got = nb.entropic_primal(ab, a, b, C, 0.4)
    assert np.isclose(got, np.sum(ab * C)), f"P = a bᵀ has KL(P‖a bᵀ) = 0, so the primal is <C, a bᵀ>, got {got}"
    got2 = nb.entropic_primal(2 * ab, a, b, C, 0.4)
    want2 = 2 * np.sum(ab * C) + 0.4 * (2 * np.log(2) - 1)
    assert np.isclose(got2, want2), (
        f"P = 2 a bᵀ: KL(P‖a bᵀ) = Σ (P log(P/ab) − P + ab) = 2 log 2 − 1, so primal = {want2:.6f}; got {got2}. "
        "keep the −P + ab terms of the generalized KL")
    Pz = np.array([[0.25, 0.0, 0.0], [0.25, 0.2, 0.3]])
    want_z = np.sum(Pz * C) + 0.4 * (np.sum(Pz[Pz > 0] * np.log(Pz[Pz > 0] / ab[Pz > 0])) - 1 + 1)
    assert np.isclose(nb.entropic_primal(Pz, a, b, C, 0.4), want_z), "0 log 0 = 0: zero entries contribute nothing"
    f0, g0 = np.zeros(2), np.zeros(3)
    want_d = -0.4 * np.sum(ab * (np.exp(-C / 0.4) - 1))
    assert np.isclose(nb.entropic_dual(f0, g0, a, b, C, 0.4), want_d), (
        "dual at f = g = 0: −eps Σ a_i b_j (exp(−C_ij/eps) − 1)")
    f1, g1 = np.array([0.1, -0.2]), np.array([0.3, 0.0, -0.1])
    want_d1 = f1 @ a + g1 @ b - 0.4 * np.sum(ab * (np.exp((f1[:, None] + g1[None] - C) / 0.4) - 1))
    assert np.isclose(nb.entropic_dual(f1, g1, a, b, C, 0.4), want_d1), "<f, a> + <g, b> − eps Σ ab (e^{…} − 1)"


def test_stage3_weak_and_strong_duality(nb):
    _, _, a, b, C = _problem(seed=11)
    g = _rng(12)
    P_lp, _ = _lp(a, b, C)
    for eps in (0.5, 1.0):
        P, f, gg = (np.asarray(x) for x in _b(nb.sinkhorn_log, a, b, C, eps, max_iters=20000, tol=1e-11,
                                                     _budget=20_000_000))
        primal, dual = nb.entropic_primal(P, a, b, C, eps), nb.entropic_dual(f, gg, a, b, C, eps)
        assert abs(primal - dual) < 1e-7 * (1 + abs(primal)), (
            f"eps = {eps}: at Sinkhorn's fixed point primal and dual must agree, got {primal:.8f} vs {dual:.8f}")
        assert np.isclose(dual, f @ a + gg @ b, atol=1e-8), (
            "at convergence the mass term vanishes: the dual equals <f, a> + <g, b>")
        for lam in (0.0, 0.3, 0.8):
            Q = lam * P_lp + (1 - lam) * np.outer(a, b)  # feasible plans
            fr, gr = g.normal(size=len(a)), g.normal(size=len(b))
            assert nb.entropic_dual(fr, gr, a, b, C, eps) <= nb.entropic_primal(Q, a, b, C, eps) + 1e-10, (
                "weak duality: any dual value is <= any feasible primal value")
            assert nb.entropic_primal(Q, a, b, C, eps) >= primal - 1e-8, "Sinkhorn's plan minimizes the primal"


def test_stage3_dual_ascends(nb):
    _, _, a, b, C = _problem(seed=13)
    eps = 0.3
    vals = []
    for k in range(1, 16):
        _, f, g = _b(nb.sinkhorn_log, a, b, C, eps, max_iters=k, tol=0.0)
        vals.append(nb.entropic_dual(f, g, a, b, C, eps))
    assert np.all(np.diff(vals) >= -1e-12), (
        f"each Sinkhorn update maximizes the dual exactly over f (or g) with the other fixed, so the dual can only go "
        f"up: got {np.round(vals, 6).tolist()}")
    assert vals[-1] > vals[0], "and it should actually improve over 15 iterations"


# ---------------------------------------------------------------- stage 4: the debiased Sinkhorn divergence


def test_stage4_zero_and_translation(nb):
    g = _rng(14)
    X = g.normal(size=(8, 2))
    w = g.random(8) + 0.2
    w /= w.sum()
    t = np.array([0.7, -0.4])
    for eps in (1.0, 3.0):
        s0 = _b(nb.sinkhorn_divergence, X, X, eps, a=w, b=w, max_iters=20000, tol=1e-11, _budget=20_000_000)
        assert abs(s0) < 1e-8, f"eps = {eps}: S(μ, μ) = 0 exactly (the debiasing terms cancel OT(μ, μ)), got {s0}"
        st = _b(nb.sinkhorn_divergence, X, X + t, eps, a=w, b=w, max_iters=20000, tol=1e-11, _budget=20_000_000)
        assert np.isclose(st, t @ t, atol=1e-6), (
            f"eps = {eps}: for squared costs a translation by t gives S = |t|² = {t @ t:.4f} at EVERY eps (the "
            f"cross term is linear, so it doesn't change the plan); got {st:.6f}. use the converged entropic value "
            "<f, a> + <g, b> of each problem, and subtract HALF of each self-term")


def test_stage4_limits(nb):
    X, Y, a, b, C = _problem(n=6, m=5, seed=15)
    big = _b(nb.sinkhorn_divergence, X, Y, 1e6, a=a, b=b)
    want = np.sum((a @ X - b @ Y) ** 2)
    assert np.isclose(big, want, rtol=1e-3, atol=1e-4), (
        f"eps → ∞ with squared costs: the plans become independent and S tends to |mean_a(X) − mean_b(Y)|² = "
        f"{want:.5f}, got {big:.5f}. (are the weights a, b used?)")
    _, w2 = _lp(a, b, C)
    eps = 0.02
    bound = eps * (-(a * np.log(a)).sum() - (b * np.log(b)).sum()) + 2e-3
    small = _b(nb.sinkhorn_divergence, X, Y, eps, a=a, b=b, max_iters=5000, tol=1e-8)
    assert abs(small - w2) < bound, (
        f"eps → 0: |S − W2²| <= eps (H(a) + H(b)) = {bound:.4f}, with W2² = {w2:.5f} from the exact LP; got {small:.5f}")
    C1 = np.sqrt(C)
    _, w1 = _lp(a, b, C1)
    small1 = _b(nb.sinkhorn_divergence, X, Y, eps, a=a, b=b, p=1, max_iters=5000, tol=1e-8)
    assert abs(small1 - w1) < bound, (
        f"p = 1 uses the cost |x − y|: as eps → 0, S → W1 = {w1:.5f} (within {bound:.4f}), got {small1:.5f}")


def test_stage4_matches_oracle_and_is_nonnegative(nb):
    g = _rng(16)
    for k in range(4):
        X, Y = g.normal(size=(7, 2)), g.normal(size=(9, 2)) + g.normal(size=2)
        eps = (1.0, 2.0, 3.0, 1.5)[k]
        a, b = np.full(7, 1 / 7), np.full(9, 1 / 9)

        def ot(U, V, wu, wv):
            Cuv = ((U[:, None] - V[None]) ** 2).sum(-1)
            _, f, gg = _oracle(wu, wv, Cuv, eps)
            return f @ wu + gg @ wv

        want = ot(X, Y, a, b) - ot(X, X, a, a) / 2 - ot(Y, Y, b, b) / 2
        got = _b(nb.sinkhorn_divergence, X, Y, eps, max_iters=5000, tol=1e-11)
        assert np.isclose(got, want, atol=1e-6), (
            f"eps = {eps}: S = OT_eps(X, Y) − ½ OT_eps(X, X) − ½ OT_eps(Y, Y) with OT_eps the converged entropic "
            f"value (the dual <f, a> + <g, b>, not the transport part <P, C>): want {want:.6f}, got {got:.6f}")
        assert got >= -1e-9, "the debiased divergence is nonnegative"
        assert np.isclose(got, _b(nb.sinkhorn_divergence, Y, X, eps, max_iters=5000, tol=1e-11), atol=1e-6), "symmetric"
    u, v = g.normal(size=5), g.normal(size=4) + 1
    s1d = _b(nb.sinkhorn_divergence, u, v, 1.0)
    s2d = _b(nb.sinkhorn_divergence, u[:, None], v[:, None], 1.0)
    assert np.isclose(s1d, s2d), "1-D arrays are points on a line"


def test_stage4_forwards_iteration_controls(nb):
    X, Y, a, b, _ = _problem(n=5, m=6, seed=17, uniform=True)
    eps = 0.4

    def ot1(U, V, wu, wv):
        Cuv = ((U[:, None] - V[None]) ** 2).sum(-1)
        _, f, g = _b(nb.sinkhorn_log, wu, wv, Cuv, eps, max_iters=1)
        return nb.entropic_dual(f, g, wu, wv, Cuv, eps)

    want = ot1(X, Y, a, b) - ot1(X, X, a, a) / 2 - ot1(Y, Y, b, b) / 2
    got = _b(nb.sinkhorn_divergence, X, Y, eps, max_iters=1)
    assert np.isclose(got, want, rtol=1e-9), (
        "max_iters is passed on to every sinkhorn_log call: with max_iters=1 each OT term is the dual value after "
        "one iteration")
    got_tol = _b(nb.sinkhorn_divergence, X, Y, eps, tol=1e3)
    assert np.isclose(got_tol, want, rtol=1e-9), (
        "tol is passed on to every sinkhorn_log call too: with tol=1e3 each one stops after its first iteration, so "
        "the divergence must equal the one-iteration value")
