"""Checks for exercise.py (concentration inequalities).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import math

import numpy as np
from scipy import stats


def _f(x):
    return float(np.asarray(x))


# ---------------------------------------------------------------- stage 1: Markov and Chebyshev


def test_stage1_markov_values(nb):
    assert abs(_f(nb.markov_bound(2.0, 10.0)) - 0.2) < 1e-12, "Markov: P(X >= 10) <= E[X]/10 = 2/10 = 0.2"
    assert _f(nb.markov_bound(2.0, 1.0)) == 1.0, (
        f"for t below the mean, mean/t > 1, which says nothing: clip to 1. got {_f(nb.markov_bound(2.0, 1.0))}")
    got = np.asarray(nb.markov_bound(3.0, np.array([1.0, 3.0, 6.0, 30.0])))
    assert got.shape == (4,) and np.allclose(got, [1.0, 1.0, 0.5, 0.1]), (
        f"markov_bound(3, [1, 3, 6, 30]) should be [1, 1, 0.5, 0.1] (elementwise, clipped), got {got.tolist()}")


def test_stage1_markov_is_tight(nb):
    """the two-point distribution X in {0, t} with P(X = t) = mean / t meets Markov with equality."""
    mean, t = 1.5, 6.0
    p_top = mean / t
    assert abs(_f(nb.markov_bound(mean, t)) - p_top) < 1e-12, (
        f"X = 6 with probability 0.25 (else 0) has mean 1.5 and P(X >= 6) = 0.25, exactly the Markov bound "
        f"1.5/6. your bound gives {_f(nb.markov_bound(mean, t))}")


def test_stage1_chebyshev_values(nb):
    assert abs(_f(nb.chebyshev_bound(4.0, 4.0)) - 0.25) < 1e-12, (
        f"Chebyshev with variance 4 and t = 4 (two standard deviations): 4 / 16 = 0.25, got {_f(nb.chebyshev_bound(4.0, 4.0))}. "
        "divide by t SQUARED.")
    got = np.asarray(nb.chebyshev_bound(1.0, np.array([0.5, 1.0, 2.0, 10.0])))
    assert np.allclose(got, [1.0, 1.0, 0.25, 0.01]), f"chebyshev_bound(1, [0.5, 1, 2, 10]) should be [1, 1, .25, .01], got {got.tolist()}"


def test_stage1_bounds_hold_for_exponential(nb):
    x = np.random.default_rng(0).exponential(2.0, size=200_000)  # mean 2, variance 4
    for t in (3.0, 6.0, 10.0):
        emp = np.mean(x >= t)
        assert emp <= _f(nb.markov_bound(2.0, t)) + 0.005, f"Markov must hold: P(X >= {t}) ≈ {emp:.4f}"
        emp2 = np.mean(np.abs(x - 2.0) >= t)
        assert emp2 <= _f(nb.chebyshev_bound(4.0, t)) + 0.005, f"Chebyshev must hold: P(|X - 2| >= {t}) ≈ {emp2:.4f}"


# ---------------------------------------------------------------- stage 2: Hoeffding


def test_stage2_hoeffding_values(nb):
    got = _f(nb.hoeffding_bound(100, 0.1))
    assert abs(got - 2 * math.exp(-2)) < 1e-12, (
        f"Hoeffding for 100 draws in [0, 1] and t = 0.1: 2 exp(-2 * 100 * 0.01) = 2e^-2 ≈ 0.2707, got {got:.4f}. "
        + ("you dropped the factor 2 in front (the bound is two-sided)." if abs(got - math.exp(-2)) < 1e-9 else ""))
    wide = _f(nb.hoeffding_bound(100, 0.4, a=-1.0, b=1.0))
    assert abs(wide - 2 * math.exp(-2 * 100 * 0.16 / 4)) < 1e-12, (
        f"for draws in [-1, 1] the range is 2, and the exponent divides by (b - a)^2 = 4: expected "
        f"{2 * math.exp(-8):.6f}, got {wide:.6f}")
    assert _f(nb.hoeffding_bound(1, 0.01)) == 1.0, "a bound above 1 says nothing: clip to 1"
    arr = np.asarray(nb.hoeffding_bound(np.array([10, 100, 1000]), 0.1))
    assert arr.shape == (3,) and np.all(np.diff(arr) < 0), "hoeffding_bound should work on an array of n, decreasing in n"


def test_stage2_sample_size(nb):
    n = nb.hoeffding_sample_size(0.05, 0.05)
    assert isinstance(n, int), f"hoeffding_sample_size should return a Python int, got {type(n).__name__}"
    assert n == 738, (
        f"to get within 0.05 with probability 0.95 (draws in [0, 1]): n >= ln(2/0.05) / (2 * 0.05^2) = 737.8, "
        f"so n = 738. got {n}. round UP.")
    for t, delta, a, b in ((0.01, 0.01, 0.0, 1.0), (0.2, 0.1, -3.0, 5.0), (0.1, 0.5, 0.0, 1.0)):
        n = nb.hoeffding_sample_size(t, delta, a, b)
        assert _f(nb.hoeffding_bound(n, t, a, b)) <= delta < _f(nb.hoeffding_bound(n - 1, t, a, b)), (
            f"t={t}, delta={delta}, [{a}, {b}]: n = {n} should be the SMALLEST n with bound <= delta")


def test_stage2_hoeffding_holds_for_coins(nb):
    rng = np.random.default_rng(1)
    means = (rng.random((40_000, 50)) < 0.3).mean(axis=1)
    for t in (0.1, 0.15, 0.2):
        emp = np.mean(np.abs(means - 0.3) >= t)
        assert emp <= _f(nb.hoeffding_bound(50, t)) + 0.003, f"Hoeffding must hold: P(|mean - 0.3| >= {t}) ≈ {emp:.4f}"


# ---------------------------------------------------------------- stage 3: Bernstein


def test_stage3_bernstein_value(nb):
    var, c = 0.01 * 0.99, 0.99
    got = _f(nb.bernstein_bound(100, 0.1, var, c))
    want = 2 * math.exp(-100 * 0.01 / (2 * var + 2 * c * 0.1 / 3))
    assert abs(got - want) < 1e-15 + 1e-9 * want, (
        f"Bernstein for 100 Bernoulli(0.01) draws, t = 0.1: 2 exp(-n t^2 / (2 var + 2 c t / 3)) = {want:.3e}, got {got:.3e}")
    assert _f(nb.bernstein_bound(5, 0.01, 0.25, 1.0)) == 1.0, "clip to 1"


def test_stage3_beats_hoeffding_for_small_variance(nb):
    p = 0.01
    for n, t in ((500, 0.02), (2000, 0.01)):
        bern = _f(nb.bernstein_bound(n, t, p * (1 - p), 1 - p))
        hoef = _f(nb.hoeffding_bound(n, t))
        exact = float(stats.binom(n, p).sf(math.ceil(n * (p + t)) - 1) + stats.binom(n, p).cdf(math.floor(n * (p - t))))
        assert exact <= bern, f"Bernstein must hold: exact {exact:.3e} > your bound {bern:.3e}"
        assert bern < hoef / 5, (
            f"n={n}, t={t}, Bernoulli(0.01): Bernstein should be far tighter than Hoeffding when the variance is "
            f"small; got Bernstein {bern:.3e} vs Hoeffding {hoef:.3e}")


def test_stage3_elementwise(nb):
    got = np.asarray(nb.bernstein_bound(np.array([100, 400]), np.array([0.1, 0.05]), 0.25, 0.5))
    want = 2 * np.exp(-np.array([100, 400]) * np.array([0.01, 0.0025]) / (0.5 + np.array([0.1, 0.05]) / 3))
    assert got.shape == (2,) and np.allclose(got, np.minimum(1, want)), "bernstein_bound should work elementwise on arrays of n and t"


# ---------------------------------------------------------------- stage 4: Chernoff, and the truth


def test_stage4_chernoff_value(nb):
    got = _f(nb.chernoff_bernoulli_bound(100, 0.5, 0.1))
    kl = 0.6 * math.log(1.2) + 0.4 * math.log(0.8)
    assert abs(got - math.exp(-100 * kl)) < 1e-12, (
        f"Chernoff for 100 fair coins, P(mean >= 0.6) <= exp(-100 KL(0.6 || 0.5)) = exp(-100 * {kl:.5f}) "
        f"= {math.exp(-100 * kl):.4f}; got {got:.4f}. KL(q || p) = q log(q/p) + (1 - q) log((1 - q)/(1 - p)).")
    exact = float(stats.binom(100, 0.5).sf(59))
    assert exact <= got <= _f(nb.hoeffding_bound(100, 0.1)) / 2 + 1e-12, (
        "Chernoff must hold, and be at least as tight as one-sided Hoeffding exp(-2 n t^2)")


def test_stage4_chernoff_edges(nb):
    got = _f(nb.chernoff_bernoulli_bound(10, 0.3, 0.7))
    assert abs(got - 0.3**10) < 1e-15, (
        f"P(mean of 10 Bernoulli(0.3) >= 1) = 0.3^10 exactly, and Chernoff gives exactly that (KL(1 || p) = log(1/p)); "
        f"got {got}. use 0 log 0 = 0 for the second term.")
    assert _f(nb.chernoff_bernoulli_bound(10, 0.3, 0.8)) == 0.0, "mean >= 1.1 is impossible: the bound is 0"
    arr = np.asarray(nb.chernoff_bernoulli_bound(50, 0.2, np.array([0.05, 0.1, 0.3, 0.8, 0.9])))
    assert arr.shape == (5,) and np.all(np.isfinite(arr)) and arr[-1] == 0.0 and np.all(np.diff(arr) <= 0), (
        f"chernoff_bernoulli_bound should work on an array of t, finite and non-increasing, 0 beyond p + t > 1; got {arr.tolist()}")


def test_stage4_chernoff_holds(nb):
    for n, p, t in ((200, 0.05, 0.05), (1000, 0.3, 0.05), (50, 0.5, 0.2)):
        k = math.ceil(n * (p + t) - 1e-9)
        exact = float(stats.binom(n, p).sf(k - 1))
        ch = _f(nb.chernoff_bernoulli_bound(n, p, t))
        assert exact <= ch, f"Chernoff must hold: n={n}, p={p}, t={t}: exact {exact:.3e} > bound {ch:.3e}"


def test_stage4_empirical_tail(nb):
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert abs(nb.empirical_tail(s, 2.0, 1.0) - 0.8) < 1e-12, (
        f"|s - 2| = [2, 1, 0, 1, 2]: four of five are >= 1 (the boundary counts), so 0.8; got {nb.empirical_tail(s, 2.0, 1.0)}")
    assert isinstance(nb.empirical_tail(s, 2.0, 1.0), float), "a scalar t should give a Python float"
    got = np.asarray(nb.empirical_tail(s, 2.0, np.array([0.0, 1.5, 2.0, 2.5])))
    assert got.shape == (4,) and np.allclose(got, [1.0, 0.4, 0.4, 0.0]), (
        f"for t = [0, 1.5, 2, 2.5] the two-sided tails are [1, .4, .4, 0], got {got.tolist()}")
    one = np.asarray(nb.empirical_tail(s, 2.0, np.array([1.0, 2.0]), two_sided=False))
    assert np.allclose(one, [0.4, 0.2]), f"one-sided, s - 2 >= t for t = [1, 2] gives [.4, .2], got {one.tolist()}"
