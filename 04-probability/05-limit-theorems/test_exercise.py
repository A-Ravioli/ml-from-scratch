"""Checks for exercise.py (limit theorems).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Simulation results are compared with exact answers (from SciPy) using tolerances of about
4-5 Monte Carlo standard errors: a correct simulator passes whatever random numbers it draws,
and the bugs these tests look for miss by far more.
"""
import math

import numpy as np
import pytest
from scipy import stats


def _uniform12(size, rng):
    return 12.0 * rng.random(size)  # mean 6, sd sqrt(12)


def _expo(size, rng):
    return rng.exponential(1.0, size)


def _coin(size, rng):
    return (rng.random(size) < 0.5).astype(float)


def _exact_ks(cdf_of_z):
    """sup_t |F(t) - Phi(t)| on a fine grid."""
    t = np.linspace(-8, 8, 160_001)
    return float(np.max(np.abs(cdf_of_z(t) - stats.norm.cdf(t))))


# ---------------------------------------------------------------- stage 1: averages


def test_stage1_running_mean_hand_example(nb):
    got = np.asarray(nb.running_mean([2, 4, 6, 8, 0]))
    assert got.shape == (5,), f"running_mean of 5 numbers should have shape (5,), got {got.shape}"
    assert np.allclose(got, [2, 3, 4, 5, 4]), (
        f"running means of [2, 4, 6, 8, 0] are [2, 3, 4, 5, 4], got {got.tolist()}. divide the running SUM by "
        "the number of terms so far (1, 2, 3, ...), not by n.")


def test_stage1_running_mean_long(nb):
    x = np.random.default_rng(0).normal(size=100_000)
    got = np.asarray(nb.running_mean(x))
    assert abs(got[-1] - x.mean()) < 1e-12 and abs(got[999] - x[:1000].mean()) < 1e-12, (
        "the last running mean should equal the mean of everything, and entry k the mean of x[:k+1]")


def test_stage1_sample_means_shape_and_constant(nb):
    m = np.asarray(nb.sample_means(lambda size, rng: np.full(size, 3.5), 7, 11, rng=0))
    assert m.shape == (11,), f"sample_means(..., n=7, n_reps=11) returns one mean per repetition: shape (11,), got {m.shape}"
    assert np.allclose(m, 3.5), "the mean of a constant sampler is that constant"


def test_stage1_sample_means_distribution(nb):
    m = np.asarray(nb.sample_means(_uniform12, 12, 40_000, rng=1))
    se_mean = math.sqrt(12 / 12 / 40_000)
    assert abs(m.mean() - 6) < 5 * se_mean, f"means of Uniform(0, 12) draws should average 6, got {m.mean():.4f}"
    var = m.var()
    assert abs(var - 1.0) < 0.04, (
        f"the mean of n = 12 draws with variance 12 has variance 12 / 12 = 1, got {var:.4f}. "
        + ("you averaged over the wrong axis: each ROW (n draws) gives one mean." if var < 0.1 else ""))


def test_stage1_sample_means_reproducible(nb):
    a = np.asarray(nb.sample_means(_expo, 5, 10, rng=42))
    b = np.asarray(nb.sample_means(_expo, 5, 10, rng=42))
    assert np.array_equal(a, b), "the same seed should give the same means: pass np.random.default_rng(rng) to the sampler"
    c = np.asarray(nb.sample_means(_expo, 5, 10, rng=43))
    assert not np.array_equal(a, c), (
        "seeds 42 and 43 gave identical means: use the rng you were given (np.random.default_rng(rng)), "
        "don't create your own fixed-seed generator")
    g = np.random.default_rng(7)
    d, e = np.asarray(nb.sample_means(_expo, 5, 10, rng=g)), np.asarray(nb.sample_means(_expo, 5, 10, rng=g))
    assert not np.array_equal(d, e), "two calls with the same Generator should continue its stream, not repeat it"


# ---------------------------------------------------------------- stage 2: distance to the normal


def test_stage2_normal_cdf(nb):
    assert abs(nb.normal_cdf(0.0) - 0.5) < 1e-15, "Phi(0) = 1/2"
    assert isinstance(nb.normal_cdf(1.0), float), "a scalar input should give a Python float"
    z = np.array([[-3.0, -1.0], [1.959963984540054, 5.0]])
    got = np.asarray(nb.normal_cdf(z))
    assert got.shape == (2, 2), f"normal_cdf should keep the input's shape (2, 2), got {got.shape}"
    assert np.allclose(got, stats.norm.cdf(z), atol=1e-14), (
        f"Phi at [[-3, -1], [1.96, 5]] should be {np.round(stats.norm.cdf(z), 6).tolist()}, got "
        f"{np.round(got, 6).tolist()}. Phi(z) = (1 + erf(z / sqrt 2)) / 2: did you forget the sqrt 2?")


def test_stage2_ks_hand_examples(nb):
    d1 = nb.ks_distance(np.array([0.0]))
    assert abs(d1 - 0.5) < 1e-12, (
        f"for the single point 0, F_n jumps from 0 to 1 at 0 where Phi = 1/2: the distance is 1/2, got {d1}")
    d2 = nb.ks_distance(np.array([1.0, -1.0]))
    want = stats.norm.cdf(1.0) - 0.5
    assert abs(d2 - want) < 1e-12, (
        f"for the points (1, -1) the distance is Phi(1) - 1/2 = {want:.4f}, attained just BEFORE the jump at 1; "
        f"got {d2:.4f}. check both sides of every jump: i/n - Phi(z_i) and Phi(z_i) - (i-1)/n, with z sorted.")


def test_stage2_ks_matches_scipy(nb):
    rng = np.random.default_rng(2)
    for z in (rng.normal(size=500), rng.normal(0.3, 1.0, size=400), rng.standard_t(3, size=300)):
        got, want = nb.ks_distance(z), stats.kstest(z, "norm").statistic
        assert abs(got - want) < 1e-12, f"ks_distance disagrees with scipy.stats.kstest: {got:.6f} vs {want:.6f}"


# ---------------------------------------------------------------- stage 3: the CLT, measured and bounded


def test_stage3_uniform_needs_standardizing(nb):
    d = np.asarray(nb.clt_distances(_uniform12, 6.0, math.sqrt(12.0), [1, 4], 20_000, rng=3))
    exact1 = _exact_ks(lambda t: np.clip((t + math.sqrt(3)) / (2 * math.sqrt(3)), 0, 1))
    assert d.shape == (2,), f"one distance per n: shape (2,), got {d.shape}"
    assert abs(d[0] - exact1) < 0.03, (
        f"n = 1: a standardized Uniform is at KS distance {exact1:.4f} from N(0, 1), but you measured {d[0]:.4f}. "
        "standardize as z = sqrt(n) (mean - mu) / sigma, with sigma the standard DEVIATION.")
    assert d[1] < 0.03, f"the mean of 4 uniforms is already very close to normal (distance ~0.01), you measured {d[1]:.4f}"


def test_stage3_exponential_matches_exact(nb):
    ns = [1, 4, 16, 64]
    d = np.asarray(nb.clt_distances(_expo, 1.0, 1.0, ns, 20_000, rng=4))
    exact = [_exact_ks(lambda t, n=n: stats.gamma(n).cdf(n + t * math.sqrt(n))) for n in ns]
    for n, got, want in zip(ns, d, exact):
        assert abs(got - want) < 0.03, (
            f"n = {n}: the standardized mean of n Exponential(1) draws is at exact KS distance {want:.4f} from "
            f"N(0, 1) (it's a scaled Gamma(n)); you measured {got:.4f}. "
            + ("did you forget the sqrt(n)?" if got > want + 0.15 else ""))


def test_stage3_berry_esseen(nb):
    rho = 12 / math.e - 2  # E|X - 1|^3 for Exponential(1)
    b = nb.berry_esseen_bound(16, 1.0, rho)
    assert abs(b - 0.4748 * rho / 4) < 1e-12, f"Berry–Esseen: C rho / (sigma^3 sqrt n) = {0.4748 * rho / 4:.5f} for n = 16, got {b}"
    arr = np.asarray(nb.berry_esseen_bound(np.array([1, 4, 100]), 2.0, 8.0, C=1.0))
    assert np.allclose(arr, 8.0 / (8.0 * np.sqrt([1, 4, 100]))), "it should work on an array of n, and use sigma CUBED"
    for n in (1, 4, 16, 64):
        exact = _exact_ks(lambda t, n=n: stats.gamma(n).cdf(n + t * math.sqrt(n)))
        assert exact <= nb.berry_esseen_bound(n, 1.0, rho), "the bound must hold for the exponential"


# ---------------------------------------------------------------- stage 4: the weak law


def test_stage4_matches_binomial_tails(nb):
    ns = [8, 16, 32, 64, 128]
    reps = 20_000
    p = np.asarray(nb.deviation_probability(_coin, 0.5, 0.125, ns, reps, rng=5))
    assert p.shape == (5,), f"one probability per n: shape (5,), got {p.shape}"
    for n, got in zip(ns, p):
        k = np.arange(n + 1)
        exact = float(stats.binom(n, 0.5).pmf(k)[np.abs(k / n - 0.5) > 0.125].sum())
        weak = float(stats.binom(n, 0.5).pmf(k)[np.abs(k / n - 0.5) >= 0.125].sum())
        tol = 5 * math.sqrt(exact * (1 - exact) / reps) + 1e-3
        assert abs(got - exact) < tol, (
            f"n = {n}: P(|mean - 1/2| > 1/8) for a fair coin is exactly {exact:.4f}, you estimated {got:.4f}. "
            + ("that's P(... >= 1/8): the definition uses a STRICT inequality, and with a coin the mean can "
               "land exactly on 1/2 ± 1/8." if abs(got - weak) < tol else ""))
    assert p[-1] < p[0], "the probability of a large deviation must shrink as n grows: that's the weak law"


def test_stage4_continuous_case(nb):
    p = np.asarray(nb.deviation_probability(_expo, 1.0, 0.2, [25, 100, 400], 20_000, rng=6))
    exact = [float(stats.gamma(n).cdf(0.8 * n) + stats.gamma(n).sf(1.2 * n)) for n in (25, 100, 400)]
    assert np.allclose(p, exact, atol=0.015), (
        f"P(|mean of n Exp(1) draws - 1| > 0.2) for n = 25, 100, 400 is {np.round(exact, 4).tolist()}, "
        f"you estimated {np.round(p, 4).tolist()}")


def test_stage4_uses_the_rng(nb):
    a = np.asarray(nb.deviation_probability(_expo, 1.0, 0.3, [10, 20], 2_000, rng=1))
    b = np.asarray(nb.deviation_probability(_expo, 1.0, 0.3, [10, 20], 2_000, rng=1))
    c = np.asarray(nb.deviation_probability(_expo, 1.0, 0.3, [10, 20], 2_000, rng=2))
    assert np.array_equal(a, b), "the same seed should give the same estimates"
    assert not np.array_equal(a, c), (
        "seeds 1 and 2 gave identical estimates: pass the given rng (normalized once) to sample_means, "
        "don't re-seed with a fixed number")
    d = np.asarray(nb.clt_distances(_expo, 1.0, 1.0, [4], 2_000, rng=1))
    e = np.asarray(nb.clt_distances(_expo, 1.0, 1.0, [4], 2_000, rng=2))
    assert not np.array_equal(d, e), "clt_distances: seeds 1 and 2 gave identical results; use the rng you were given"


@pytest.mark.parametrize("bad_mu", [0.0])
def test_stage4_uses_mu(nb, bad_mu):
    p = np.asarray(nb.deviation_probability(_coin, bad_mu, 0.125, [64], 5_000, rng=7))
    assert p[0] > 0.99, "measured against the wrong center (mu = 0), the deviation probability should be ~1: use mu"
