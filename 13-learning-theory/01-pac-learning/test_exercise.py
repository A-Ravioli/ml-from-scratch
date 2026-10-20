"""Checks for exercise.py (PAC learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.

The calculator is checked against hand-computed values and its defining property (it is the
smallest m that makes the union bound <= delta). ERM is checked against a brute-force O(n^2)
reference on data with ties. Monte Carlo estimates are accepted within 4.5 binomial standard
errors of the exact failure probability (so a correct estimator fails about once in 150,000
runs), and Monte Carlo loops run under mlfs.testing.bounded so a runaway loop fails instead of
hanging.
"""
import math
import time

import numpy as np

from mlfs.testing import bounded


def _brute_erm(x, y):
    cands = np.append(np.unique(x), np.inf)
    errs = [int(np.sum((x >= c) != y)) for c in cands]
    return float(cands[int(np.argmin(errs))]), min(errs)


def _binom_ok(p_hat, p, n, k=4.5):
    se = math.sqrt(max(p * (1 - p), 1e-4) / n)
    return abs(p_hat - p) <= k * se + 1e-12, se


# ---------------------------------------------------------------- stage 1: the calculator


def test_stage1_worked_example(nb):
    got = nb.finite_class_sample_complexity(1000, 0.1, 0.05)
    assert got == 100 and isinstance(got, int), (
        f"|H| = 1000, epsilon = 0.1, delta = 0.05 (realizable): ln(1000 / 0.05) / 0.1 = 99.03, so the smallest integer "
        f"m is 100; got {got!r}. round UP (math.ceil) and return a Python int")
    agn = nb.finite_class_sample_complexity(1000, 0.1, 0.05, realizable=False)
    assert agn == 530, (
        f"agnostic: ln(2 * 1000 / 0.05) / (2 * 0.1^2) = 529.8, so m = 530; got {agn!r}. the agnostic rate has "
        "epsilon SQUARED in the denominator and a factor 2 inside the log (two-sided Hoeffding)")


def test_stage1_is_the_smallest_sufficient_m(nb):
    for H, eps, delta in [(1, 0.2, 0.1), (7, 0.03, 0.01), (10**6, 0.01, 0.001), (365, 0.15, 0.2)]:
        m = nb.finite_class_sample_complexity(H, eps, delta)
        assert H * math.exp(-eps * m) <= delta * (1 + 1e-12) and H * math.exp(-eps * (m - 1)) > delta, (
            f"|H| = {H}, epsilon = {eps}, delta = {delta}: m must be the smallest integer with |H| e^(-epsilon m) <= "
            f"delta; got {m} ({H * math.exp(-eps * m):.4g} vs delta)")
        a = nb.finite_class_sample_complexity(H, eps, delta, realizable=False)
        assert 2 * H * math.exp(-2 * eps**2 * a) <= delta * (1 + 1e-12) and 2 * H * math.exp(-2 * eps**2 * (a - 1)) > delta, (
            f"agnostic |H| = {H}, epsilon = {eps}, delta = {delta}: the smallest m with 2|H| e^(-2 epsilon^2 m) <= delta; "
            f"got {a}")


def test_stage1_huge_classes_and_shape(nb):
    # a float32 linear model on 784 pixels + bias is a finite class with 2^(32 * 785) members
    H = 2 ** (32 * 785)
    m = nb.finite_class_sample_complexity(H, 0.05, 0.05)
    want = math.ceil((32 * 785 * math.log(2) + math.log(20)) / 0.05)
    assert m == want, (
        f"|H| = 2^(32*785) (every float32 weight vector): expected {want}, got {m}. np.log(H) or H / delta overflow for a "
        "number this big; math.log(H) works on Python ints of any size, and ln(|H|/delta) = ln|H| + ln(1/delta)")
    m1 = nb.finite_class_sample_complexity(500, 0.1, 0.05)
    m2 = nb.finite_class_sample_complexity(1000, 0.1, 0.05)
    assert m2 - m1 in (6, 7, 8), f"doubling |H| should add only about ln(2) / epsilon = 6.9 examples; it added {m2 - m1}"
    assert nb.finite_class_sample_complexity(1000, 0.05, 0.05) > m2 > nb.finite_class_sample_complexity(1000, 0.1, 0.5), (
        "m must grow as epsilon or delta shrink")
    for bad in [(10, 0.0, 0.1), (10, 0.1, 1.0), (10, 1.5, 0.1)]:
        try:
            nb.finite_class_sample_complexity(*bad)
        except ValueError:
            continue
        raise AssertionError(f"finite_class_sample_complexity{bad} should raise ValueError: epsilon and delta live in (0, 1)")


# ---------------------------------------------------------------- stage 2: ERM over thresholds


def test_stage2_hand_examples(nb):
    x = np.array([0.9, 0.1, 0.4, 0.7, 0.35])
    y = np.array([1, 0, 1, 1, 0])
    t = nb.erm_threshold(x, y)
    assert t == 0.4 and isinstance(t, float), (
        f"separable sample: negatives at 0.1, 0.35, positives at 0.4, 0.7, 0.9. the smallest training-error-0 "
        f"candidate threshold is 0.4; got {t!r}")
    t0 = nb.erm_threshold(np.array([0.2, 0.5, 0.6]), np.array([0, 0, 0]))
    assert t0 == np.inf, f"no positive examples: the only zero-error candidate is t = +inf (predict 0 everywhere); got {t0}"
    t1 = nb.erm_threshold(np.array([0.2, 0.5, 0.6]), np.array([1, 1, 1]))
    assert t1 == 0.2, f"all examples positive: the smallest candidate, t = min(x) = 0.2, has zero error; got {t1}"


def test_stage2_matches_brute_force_with_noise_and_ties(nb):
    rng = np.random.default_rng(0)
    for trial in range(40):
        n = int(rng.integers(1, 40))
        x = rng.integers(0, 12, size=n) / 12.0  # lots of ties
        y = (x >= rng.random()) ^ (rng.random(n) < 0.25)
        want, best_err = _brute_erm(x, y)
        got = nb.erm_threshold(x, y)
        got_err = int(np.sum((x >= got) != y))
        assert got_err == best_err, (
            f"trial {trial}: your threshold {got} makes {got_err} training errors, but {want} makes only {best_err}. "
            "candidates are every distinct x value and +inf; error(t) = #(positives with x < t) + #(negatives with x >= t)")
        assert got == want, (
            f"trial {trial}: several thresholds tie at {best_err} errors; return the SMALLEST ({want}), got {got}")


def test_stage2_is_n_log_n(nb):
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.random(n)
    y = (x >= 0.3) ^ (rng.random(n) < 0.1)
    got = bounded(nb.erm_threshold, x, y, what="erm_threshold", budget=300_000,
                  hint="for 4,000 points an O(n^2) double loop takes 16 million steps: sort once and count with "
                       "np.searchsorted or cumulative sums")
    assert got == _brute_erm(x, y)[0], "wrong threshold on a 4,000-point noisy sample"

    def ref(x, y):
        c = np.append(np.unique(x), np.inf)
        p, q = np.sort(x[y]), np.sort(x[~y])
        e = np.searchsorted(p, c) + len(q) - np.searchsorted(q, c)
        return c[np.argmin(e)]

    def best(fn):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn(x, y)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    t_ref, t_you = best(ref), best(nb.erm_threshold)
    assert t_you <= 40 * t_ref + 0.02, (
        f"erm_threshold took {t_you * 1e3:.1f} ms on 4,000 points; a sort-and-count version takes {t_ref * 1e3:.2f} ms. "
        "an (n, n) comparison matrix is O(n^2): sort, then count with np.searchsorted")


# ---------------------------------------------------------------- stage 3: true risk, exact failure probability


def test_stage3_threshold_risk(nb):
    cases = [((0.7, 0.5), 0.2), ((0.3, 0.5), 0.2), ((np.inf, 0.4), 0.6), ((-2.0, 0.25), 0.25), ((0.5, 0.5), 0.0)]
    for (t, ts), want in cases:
        got = nb.threshold_risk(t, ts)
        assert np.isclose(got, want), (
            f"threshold_risk({t}, {ts}) should be {want}: under U[0, 1] the two classifiers disagree exactly on the interval "
            f"between the thresholds, clipped to [0, 1]; got {got}")
    got = nb.threshold_risk(0.7, 0.5, noise=0.1)
    assert np.isclose(got, 0.26), (
        f"with label noise 0.1: on the disagreement interval (mass 0.2) you're wrong 90% of the time, elsewhere 10%: "
        f"0.1 + (1 - 2 * 0.1) * 0.2 = 0.26; got {got}")
    assert np.isclose(nb.threshold_risk(0.5, 0.5, noise=0.1), 0.1), "the best threshold still errs on every flipped label"
    rng = np.random.default_rng(2)
    x = rng.random(400_000)
    y = (x >= 0.35) ^ (rng.random(x.size) < 0.15)
    mc = np.mean((x >= 0.6) != y)
    assert abs(nb.threshold_risk(0.6, 0.35, noise=0.15) - mc) < 0.004, "threshold_risk disagrees with a simulation"


def test_stage3_exact_failure_probability(nb):
    got = nb.threshold_failure_probability(10, 0.1)
    assert np.isclose(got, 0.9**10), (
        f"the ERM threshold has risk > 0.1 exactly when none of the 10 examples lands in [t*, t* + 0.1): probability "
        f"0.9^10 = {0.9**10:.4f}; got {got}")
    assert nb.threshold_failure_probability(50, 0.1, t_star=0.95) == 0.0, (
        "with t* = 0.95 and epsilon = 0.1, even t = +inf has risk 0.05 <= epsilon: failure is impossible")
    assert nb.threshold_failure_probability(10, 0.25, t_star=0.75) == 0.0, (
        "at the boundary t* + epsilon = 1 the worst possible risk is 1 - t* = epsilon, which is not MORE than epsilon: "
        "the failure probability is 0 (use >= in the edge case)")
    # the formula must describe YOUR erm_threshold: simulate it with an independent random stream
    rng = np.random.default_rng(3)
    m, eps, trials = 20, 0.1, 4000
    fails = 0
    for _ in range(trials):
        x = rng.random(m)
        t = nb.erm_threshold(x, x >= 0.5)
        fails += abs(min(t, 1.0) - 0.5) > eps
    exact = nb.threshold_failure_probability(m, eps)
    ok, se = _binom_ok(fails / trials, exact, trials)
    assert ok, (f"simulating your erm_threshold gives a failure rate of {fails / trials:.4f}, but "
                f"threshold_failure_probability says {exact:.4f} (Monte Carlo SE {se:.4f})")


# ---------------------------------------------------------------- stage 4: Monte Carlo


def test_stage4_matches_the_exact_probability(nb):
    for m, eps, trials in [(20, 0.1, 3000), (30, 0.05, 3000), (5, 0.3, 2000)]:
        p = bounded(nb.estimate_failure_probability, m, eps, n_trials=trials, rng=11, what="estimate_failure_probability",
                    budget=3_000_000)
        exact = (1 - eps) ** m
        ok, se = _binom_ok(p, exact, trials)
        assert ok, (
            f"m = {m}, epsilon = {eps}: your estimate {p:.4f} is {abs(p - exact) / se:.1f} standard errors from the exact "
            f"{exact:.4f}. in each trial draw x ~ U[0, 1]^m, label it 1[x >= t*], run erm_threshold, and count "
            "threshold_risk(t_hat) - threshold_risk(t*) > epsilon")
        assert abs(p * trials - round(p * trials)) < 1e-9, "the estimate is (number of failures) / n_trials"
    assert nb.estimate_failure_probability(40, 0.1, t_star=0.95, n_trials=200, rng=0) == 0.0
    a = nb.estimate_failure_probability(15, 0.1, n_trials=300, rng=5)
    b = nb.estimate_failure_probability(15, 0.1, n_trials=300, rng=5)
    assert a == b, "the same rng must give the same estimate (use as_rng(rng) once and draw everything from it)"


def test_stage4_noise_means_excess_risk(nb):
    p = bounded(nb.estimate_failure_probability, 2000, 0.05, noise=0.3, n_trials=100, rng=1,
                what="estimate_failure_probability", budget=3_000_000)
    assert p < 0.1, (
        f"with label noise 0.3 even the best threshold has risk 0.3. failure means EXCESS risk > epsilon: "
        f"threshold_risk(t_hat) - threshold_risk(t*) > epsilon. with 2,000 examples ERM is very close to t*, so the "
        f"failure rate should be near 0; got {p:.2f}")
    # a small noisy sample: compare with an independent reference simulation
    noisy = nb.estimate_failure_probability(30, 0.05, noise=0.2, n_trials=2000, rng=2)
    rng = np.random.default_rng(3)
    ref_trials, ref_fails = 4000, 0
    for _ in range(ref_trials):
        x = rng.random(30)
        y = (x >= 0.5) ^ (rng.random(30) < 0.2)
        t, _ = _brute_erm(x, y)
        ref_fails += 0.6 * abs(min(t, 1.0) - 0.5) > 0.05
    ref = ref_fails / ref_trials
    se = math.sqrt(ref * (1 - ref) * (1 / 2000 + 1 / ref_trials))
    assert abs(noisy - ref) < 4.5 * se, (
        f"30 examples, label noise 0.2, epsilon = 0.05: a reference simulation fails {ref:.3f} of the time, yours "
        f"{noisy:.3f} ({abs(noisy - ref) / se:.1f} standard errors apart). flip each label independently with "
        "probability `noise`, and measure excess risk (1 - 2 noise) |t_hat - t*| (that's what threshold_risk(t_hat) - "
        "threshold_risk(t*) gives)")


# ---------------------------------------------------------------- stage 5: empirical sample complexity


def test_stage5_noiseless(nb):
    m = bounded(nb.empirical_sample_complexity, 0.1, 0.1, range(60, 4, -1), n_trials=1500, rng=0,
                what="empirical_sample_complexity", budget=20_000_000)
    exact = math.ceil(math.log(0.1) / math.log(0.9))
    assert m is not None and 19 <= m <= 27, (
        f"epsilon = delta = 0.1: the exact sample complexity is {exact} (the smallest m with 0.9^m <= 0.1), so the "
        f"Monte Carlo answer should be close to it; got {m}. scan the grid in INCREASING order and return the first m "
        "whose estimate is <= delta")
    assert isinstance(m, int), f"return the grid value as an int, got {type(m).__name__}"
    none = nb.empirical_sample_complexity(0.05, 0.01, [2, 4, 8], n_trials=300, rng=1)
    assert none is None, f"no grid value is large enough here: return None; got {none}"


def test_stage5_noise_raises_the_sample_complexity(nb):
    grid = np.unique(np.round(np.geomspace(5, 1500, 45)).astype(int))  # about 14% apart
    clean = bounded(nb.empirical_sample_complexity, 0.05, 0.1, grid, n_trials=400, rng=4,
                    what="empirical_sample_complexity", budget=40_000_000)
    noisy = bounded(nb.empirical_sample_complexity, 0.05, 0.1, grid, noise=0.3, n_trials=400, rng=5,
                    what="empirical_sample_complexity", budget=80_000_000)
    assert clean is not None and noisy is not None, f"both searches should succeed on this grid; got {clean}, {noisy}"
    assert 30 <= clean <= 65, (
        f"noiseless, epsilon = 0.05, delta = 0.1: the exact answer is 45 (0.95^m <= 0.1); got {clean}")
    # absolute threshold: noiseless answers land in 40-59 and noisy ones in 87-128 across seeds, so a ratio is
    # flaky at the edges, while a solution that ignores `noise` stays near the noiseless range (30-65)
    assert noisy >= 75, (
        f"label noise 0.3 makes the excess risk (1 - 2 * 0.3) |t - t*| and the labels near t* less informative, so ERM "
        f"needs clearly more examples than without noise: got {noisy} (noise 0.3) vs {clean} (no noise). pass `noise` "
        "on to estimate_failure_probability")
    assert noisy <= 600, (
        f"even with noise 0.3, thresholds need only a few hundred examples here (the agnostic finite-class bound says "
        f"thousands); got {noisy}")
