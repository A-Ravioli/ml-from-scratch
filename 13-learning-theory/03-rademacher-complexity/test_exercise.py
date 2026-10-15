"""Checks for exercise.py (rademacher complexity).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

Monte Carlo estimators are checked against EXACT values (random-walk expectations computed
from the binomial distribution, and closed forms for the identity design) with tolerances of
4.5 Monte Carlo standard errors, plus exact invariances that hold draw by draw for a fixed seed
(scaling, duplicated hypotheses). Bounds are checked by hand values and against the estimates
they bound.
"""
import math
import time

import numpy as np

from mlfs.testing import bounded


def _abs_walk(n):
    """mean and sd of |S_n| / n, S_n a sum of n independent random signs (exact, from the binomial)."""
    k = np.arange(n + 1)
    p = np.array([math.comb(n, int(j)) for j in k], dtype=float) / 2.0**n
    v = np.abs(2 * k - n) / n
    m = float(p @ v)
    return m, float(np.sqrt(p @ v**2 - m**2))


def _threshold_behaviours(x):
    """all distinct +-1 labelings of x by thresholds with either orientation."""
    xs = np.sort(x)
    cuts = np.concatenate([[xs[0] - 1], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1]])
    rows = [np.where(x >= c, 1.0, -1.0) for c in cuts]
    rows += [-r for r in rows]
    return np.unique(np.array(rows), axis=0)


# ---------------------------------------------------------------- stage 1: Monte Carlo for a finite class


def test_stage1_exact_cases(nb):
    import itertools
    allpm = np.array(list(itertools.product((-1.0, 1.0), repeat=6)))
    got = nb.empirical_rademacher(allpm, n_draws=200, rng=0)
    assert np.isclose(got, 1.0), (
        f"the class of ALL 64 sign patterns on 6 points always contains sigma itself, so the max correlation is exactly "
        f"1 for every draw; got {got}. normalize by n: (1/n) sum_i sigma_i h_i")
    n, draws = 10, 4000
    m, sd = _abs_walk(n)
    got = bounded(nb.empirical_rademacher, np.vstack([np.ones(n), -np.ones(n)]), n_draws=draws, rng=1,
                  what="empirical_rademacher")
    assert abs(got - m) < 4.5 * sd / math.sqrt(draws), (
        f"H = {{all +1, all -1}} on 10 points: the max is |sum sigma_i| / 10, whose exact mean is {m:.4f}; got {got:.4f}")
    one = nb.empirical_rademacher(np.ones(25), n_draws=draws, rng=2)
    assert abs(one) < 4.5 / math.sqrt(25 * draws), (
        f"a single hypothesis has Rademacher complexity 0 (E sigma_i = 0); got {one:.4f}. a 1-D predictions array is ONE "
        "hypothesis, and there's no absolute value in the definition")


def test_stage1_exact_invariances(nb):
    rng = np.random.default_rng(3)
    P = np.sign(rng.normal(size=(7, 30)))
    base = nb.empirical_rademacher(P, n_draws=300, rng=9)
    assert np.isclose(nb.empirical_rademacher(2.5 * P, n_draws=300, rng=9), 2.5 * base), (
        "scaling every hypothesis by 2.5 scales the complexity by exactly 2.5 (same seed, same signs)")
    assert np.isclose(nb.empirical_rademacher(np.vstack([P, P, P[:3]]), n_draws=300, rng=9), base), (
        "duplicate hypotheses can't change max_h, so the estimate (with the same seed) must not change")
    assert isinstance(base, float), f"return a Python float, got {type(base).__name__}"


def test_stage1_is_vectorized(nb):
    rng = np.random.default_rng(4)
    P = np.sign(rng.normal(size=(1500, 200)))
    got = bounded(nb.empirical_rademacher, P, n_draws=1000, rng=0, what="empirical_rademacher", budget=200_000,
                  hint="draw a whole (n_draws, n) matrix of signs (or large chunks of it) and use one matrix product")
    assert 0.1 < got < 0.4, f"1,500 random sign hypotheses on 200 points: about sqrt(2 ln 1500 / 200) * 0.8 = 0.24; got {got}"

    def ref():
        s = 2.0 * rng.integers(0, 2, size=(1000, 200)) - 1
        return np.mean(np.max(s @ P.T, axis=1)) / 200

    def best(fn):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_ref, t_you = best(ref), best(lambda: nb.empirical_rademacher(P, n_draws=1000, rng=0))
    assert t_you <= 20 * t_ref + 0.05, (
        f"empirical_rademacher took {t_you:.3f} s; one (draws, n) @ (n, hypotheses) product takes {t_ref:.3f} s. "
        "avoid Python loops over draws AND hypotheses")


# ---------------------------------------------------------------- stage 2: Massart


def test_stage2_massart(nb):
    P = np.array([[1, 1, 1, 1], [1, -1, 1, -1]], dtype=float)
    got = nb.massart_bound(P)
    want = 2 * math.sqrt(2 * math.log(2)) / 4
    assert np.isclose(got, want), (
        f"two hypotheses with norm 2 on 4 points: max ||h|| sqrt(2 ln N) / n = 2 sqrt(2 ln 2) / 4 = {want:.4f}; got {got}")
    assert np.isclose(nb.massart_bound(np.vstack([P, P, P])), want), (
        "N counts DISTINCT hypotheses (np.unique(..., axis=0)): copies don't make a class richer")
    assert nb.massart_bound(np.ones((3, 5))) == 0.0, "a single distinct hypothesis: ln 1 = 0, bound 0"
    Q = np.array([[3.0, 0, 0, 0], [1.0, 1.0, 0, 0]])
    assert np.isclose(nb.massart_bound(Q), 3 * math.sqrt(2 * math.log(2)) / 4), (
        "r is the LARGEST row norm (here 3, not the mean of 3 and sqrt 2)")
    rng = np.random.default_rng(5)
    for N, n in [(2, 40), (30, 50), (500, 100)]:
        Q = np.sign(rng.normal(size=(N, n)))
        b = nb.massart_bound(Q)
        est = nb.empirical_rademacher(Q, n_draws=3000, rng=6)
        assert est <= b, f"Massart's lemma is an upper bound: with N = {N}, n = {n} the estimate {est:.4f} exceeds {b:.4f}"


# ---------------------------------------------------------------- stage 3: norm balls


def test_stage3_identity_design_is_exact(nb):
    n, B = 16, 3.0
    X = np.eye(n)
    cases = [(2, B / math.sqrt(n), "||sigma||_2 = sqrt(n)"), (1, B / n, "||sigma||_inf = 1"),
             (np.inf, B, "||sigma||_1 = n")]
    for p, want, why in cases:
        got = nb.norm_ball_rademacher(X, B, norm=p, n_draws=50, rng=0)
        assert np.isclose(got, want), (
            f"X = I_16, B = 3, norm = {p}: sum_i sigma_i x_i = sigma and the sup over the ball is B times the DUAL norm "
            f"of sigma ({why}), so the complexity is exactly {want:.4f}; got {got}")
    assert np.isclose(nb.l2_rademacher_bound(X, B), B / math.sqrt(n)), "Jensen's bound is exact for the identity"


def test_stage3_monte_carlo_and_bounds(nb):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 5)) * np.array([3.0, 1.0, 0.5, 0.2, 0.1])
    got = nb.norm_ball_rademacher(X, 2.0, n_draws=4000, rng=8)
    s = 2.0 * rng.integers(0, 2, size=(40000, 30)) - 1
    vals = 2.0 * np.linalg.norm(s @ X, axis=1) / 30
    assert abs(got - vals.mean()) < 4.5 * vals.std() * math.sqrt(1 / 4000 + 1 / 40000), (
        f"L2 ball, B = 2: a reference simulation gives {vals.mean():.4f}, you got {got:.4f}")
    jensen = nb.l2_rademacher_bound(X, 2.0)
    assert vals.mean() <= jensen <= 1.4 * vals.mean(), (
        f"Jensen's bound B sqrt(sum ||x_i||^2) / n = {jensen:.4f} must lie above the true value {vals.mean():.4f} "
        "(and it should be within tens of percent of it here)")
    W = rng.normal(size=(3000, 5))
    W = 2.0 * W / np.linalg.norm(W, axis=1, keepdims=True)
    finite = nb.empirical_rademacher(W @ X.T, n_draws=2000, rng=9)
    se = vals.std() * math.sqrt(1 / 2000 + 1 / 4000)  # two independent Monte Carlo estimates
    assert finite <= got + 4.5 * se, (
        f"3,000 vectors on the sphere of radius 2 are a subset of the ball, so their complexity ({finite:.4f}) can't "
        f"exceed the ball's ({got:.4f})")
    l1 = nb.norm_ball_rademacher(X, 2.0, norm=1, n_draws=2000, rng=10)
    linf = nb.norm_ball_rademacher(X, 2.0, norm=np.inf, n_draws=2000, rng=10)
    assert l1 < got < linf, (
        f"for the same B the L1 ball is inside the L2 ball, which is inside the L-infinity ball, so their complexities "
        f"must be ordered: {l1:.4f} < {got:.4f} < {linf:.4f}? use the dual norm: 1 -> inf, 2 -> 2, inf -> 1")
    assert np.isclose(nb.norm_ball_rademacher(X, 6.0, n_draws=500, rng=11),
                      3 * nb.norm_ball_rademacher(X, 2.0, n_draws=500, rng=11)), "the complexity is linear in B"


# ---------------------------------------------------------------- stage 4: the bound


def test_stage4_bound(nb):
    got = nb.rademacher_bound(0.1, 0.05, 1000, 0.05)
    want = 0.1 + 0.1 + 3 * math.sqrt(math.log(40) / 2000)
    assert np.isclose(got, want), (
        f"0.1 + 2 * 0.05 + 3 sqrt(ln(2 / 0.05) / (2 * 1000)) = {want:.5f}; got {got}. the data-dependent bound has "
        "TWICE the empirical Rademacher complexity and 3 sqrt(ln(2/delta) / (2n))")
    assert nb.rademacher_bound(0.0, 0.0, 100, 0.01) > nb.rademacher_bound(0.0, 0.0, 100, 0.1) > \
        nb.rademacher_bound(0.0, 0.0, 400, 0.1), "the bound grows as delta shrinks and shrinks with n"
    assert np.isclose(nb.rademacher_bound(0.0, 0.0, 400, 0.1) * 2, nb.rademacher_bound(0.0, 0.0, 100, 0.1)), (
        "the confidence term scales like 1/sqrt(n)")


# ---------------------------------------------------------------- stage 5: random labels


def test_stage5_memorizer_and_ignorer(nb):
    X = np.random.default_rng(12).normal(size=(40, 3))
    calls = []

    def memorize(Xa, sigma):
        calls.append((Xa is X, np.asarray(sigma).shape, set(np.unique(sigma))))
        return sigma

    got = bounded(nb.rademacher_via_erm, memorize, X, n_draws=20, rng=0, what="rademacher_via_erm")
    assert np.isclose(got, 1.0), f"a learner that memorizes the labels has correlation exactly 1; got {got}"
    assert len(calls) == 20, f"fit_predict must be called once per draw (n_draws = 20); it was called {len(calls)} times"
    assert all(c[0] for c in calls) and all(c[1] == (40,) for c in calls), "pass X itself and sigma of shape (n,)"
    assert all(c[2] <= {-1.0, 1.0} for c in calls), "sigma holds random signs -1 and +1 (not 0/1 labels)"
    ign = nb.rademacher_via_erm(lambda Xa, s: np.ones(len(Xa)), X, n_draws=2000, rng=1)
    assert abs(ign) < 4.5 / math.sqrt(40 * 2000), f"a learner that ignores the labels has correlation ~0; got {ign:.4f}"


def test_stage5_erm_matches_the_class_complexity(nb):
    rng = np.random.default_rng(13)
    x = rng.random(30)

    def majority(Xa, s):
        return np.full(len(Xa), 1.0 if np.sum(s) >= 0 else -1.0)

    m, sd = _abs_walk(31)
    got = nb.rademacher_via_erm(majority, np.zeros(31), n_draws=3000, rng=2)
    assert abs(got - m) < 4.5 * sd / math.sqrt(3000), (
        f"majority vote is the ERM for the class {{all +1, all -1}}, whose complexity on 31 points is {m:.4f}; got {got:.4f}")
    H = _threshold_behaviours(x)

    def erm_thresholds(Xa, s):
        return H[np.argmax(H @ s)]

    got = nb.rademacher_via_erm(erm_thresholds, x, n_draws=1500, rng=3)
    s = 2.0 * rng.integers(0, 2, size=(30000, 30)) - 1
    vals = np.max(s @ H.T, axis=1) / 30
    assert abs(got - vals.mean()) < 4.5 * vals.std() * math.sqrt(1 / 1500 + 1 / 30000), (
        f"with an exact ERM for two-sided thresholds, fitting random signs measures the class's Rademacher complexity "
        f"({vals.mean():.4f} by direct simulation); got {got:.4f}")
