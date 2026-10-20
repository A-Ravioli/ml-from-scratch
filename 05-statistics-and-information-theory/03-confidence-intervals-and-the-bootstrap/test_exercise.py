"""Checks for exercise.py (confidence intervals & the bootstrap).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Randomized checks use fixed seeds and tolerances of several Monte Carlo standard errors.
Deterministic pieces (the BCa formula, the link between bootstrap_ci and its replicates) are
checked exactly.
"""
import math

import numpy as np
from scipy import stats


def _normal(size, rng):
    return rng.normal(0.0, 1.0, size)


def _var(a):
    return float(np.var(a, ddof=1))


def _oracle_bca(theta_hat, reps, jack, alpha):
    reps = np.asarray(reps, dtype=float)
    B = len(reps)
    p0 = (np.sum(reps < theta_hat) + np.sum(reps <= theta_hat)) / (2 * B)
    p0 = min(max(p0, 1 / (2 * B)), 1 - 1 / (2 * B))
    z0 = stats.norm.ppf(p0)
    d = np.mean(jack) - np.asarray(jack, dtype=float)
    den = 6 * np.sum(d**2) ** 1.5
    a = np.sum(d**3) / den if den > 0 else 0.0
    zs = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
    levels = stats.norm.cdf(z0 + (z0 + zs) / (1 - a * (z0 + zs)))
    return tuple(np.quantile(reps, levels))


# ---------------------------------------------------------------- stage 1: the normal interval


def test_stage1_hand_example(nb):
    lo, hi = nb.normal_ci(np.array([4.0, 6.0, 5.0, 9.0]))
    # mean 6, s = sqrt(14/3) = 2.160, se = 1.080
    half = 1.959963984540054 * math.sqrt(14 / 3) / 2
    assert abs(lo - (6 - half)) < 1e-9 and abs(hi - (6 + half)) < 1e-9, (
        f"(4, 6, 5, 9): mean 6, s = sqrt(14/3) = 2.160 (ddof=1), se = s / sqrt(4) = 1.080, so the 95% interval is "
        f"6 ± 1.96 · 1.080 = ({6 - half:.3f}, {6 + half:.3f}); got ({lo:.3f}, {hi:.3f}). "
        + ("you used ddof=0 for s." if abs(hi - 6 - 1.959963984540054 * math.sqrt(3.5) / 2) < 1e-6 else ""))


def test_stage1_alpha(nb):
    x = np.random.default_rng(0).normal(10, 2, 30)
    lo90, hi90 = nb.normal_ci(x, alpha=0.1)
    lo99, hi99 = nb.normal_ci(x, alpha=0.01)
    se = np.std(x, ddof=1) / math.sqrt(30)
    assert abs((hi90 - lo90) / 2 - 1.6448536269514722 * se) < 1e-9, (
        "a 90% interval (alpha = 0.1) uses z = 1.645, the 95th percentile of N(0, 1): z = inv_cdf(1 - alpha / 2)")
    assert abs((hi99 - lo99) / 2 - 2.5758293035489004 * se) < 1e-9, "a 99% interval uses z = 2.576"
    assert abs((lo90 + hi90) / 2 - x.mean()) < 1e-9, "the interval is centered at the sample mean"


# ---------------------------------------------------------------- stage 2: coverage


def test_stage2_small_sample_undercoverage(nb):
    reps = 20_000
    cov, width = nb.coverage(nb.normal_ci, _normal, 0.0, 5, reps, rng=1)
    exact = 2 * stats.t(4).cdf(1.959963984540054) - 1  # 0.8786: the z interval with only 5 points
    assert abs(cov - exact) < 5 * math.sqrt(exact * (1 - exact) / reps), (
        f"with n = 5 normal draws, mean ± 1.96 s/sqrt(n) covers the true mean only {exact:.4f} of the time (s is "
        f"noisy, so the t distribution applies), but you measured {cov:.4f}")
    c4 = math.sqrt(2 / 4) * math.gamma(2.5) / math.gamma(2.0)  # E[s] = c4 * sigma for n = 5
    want = 2 * 1.959963984540054 * c4 / math.sqrt(5)
    assert abs(width - want) < 0.02, f"the mean width should be 2 · 1.96 · E[s] / sqrt(5) = {want:.3f}, got {width:.3f}"


def test_stage2_rows_and_closed_interval(nb):
    seen = []

    def ci(row):
        seen.append(len(row))
        return (0.0, 0.0)

    cov, width = nb.coverage(ci, _normal, 0.0, 7, 50, rng=2)
    assert seen == [7] * 50, (
        f"ci_fn should be called once per dataset (50 times), each time on a row of n = 7 points; got "
        f"{len(seen)} calls with lengths {sorted(set(seen))}")
    assert cov == 1.0 and width == 0.0, (
        "an interval (0, 0) contains theta = 0: count lo <= theta <= hi with <=, not <")
    cov, width = nb.coverage(lambda row: (-0.5, 0.5), _normal, 0.0, 5, 40, rng=4)
    assert cov == 1.0 and abs(width - 1.0) < 1e-12, (
        f"40 intervals of width 1: the MEAN width is exactly 1 (divide the total by n_reps), got {width}")
    cov, _ = nb.coverage(lambda row: (row.min(), row.max()), _normal, 100.0, 5, 200, rng=3)
    assert cov == 0.0, "an interval far from theta covers it 0% of the time"


def test_stage2_uses_the_rng(nb):
    f = lambda row: (row.min(), row.max())  # data-dependent width, so different data give a different mean width
    a = nb.coverage(f, _normal, 0.0, 4, 300, rng=5)
    b = nb.coverage(f, _normal, 0.0, 4, 300, rng=5)
    c = nb.coverage(f, _normal, 0.0, 4, 300, rng=6)
    assert a == b, "the same seed should give the same coverage"
    assert a != c, "seeds 5 and 6 gave identical results: pass as_rng(rng) to the sampler"


# ---------------------------------------------------------------- stage 3: the percentile bootstrap


def test_stage3_replicates_basics(nb):
    x = np.arange(10.0)
    r = np.asarray(nb.bootstrap_replicates(x, np.mean, 200, rng=0))
    assert r.shape == (200,), f"one replicate per bootstrap sample: shape (200,), got {r.shape}"
    assert np.array_equal(r, np.asarray(nb.bootstrap_replicates(x, np.mean, 200, rng=0))), "same seed, same replicates"
    assert not np.array_equal(r, np.asarray(nb.bootstrap_replicates(x, np.mean, 200, rng=1))), (
        "seeds 0 and 1 gave identical replicates: use the rng you were given")
    g = np.random.default_rng(9)
    assert not np.array_equal(nb.bootstrap_replicates(x, np.mean, 50, rng=g), nb.bootstrap_replicates(x, np.mean, 50, rng=g)), (
        "two calls with the same Generator should continue its stream")
    lens = []
    nb.bootstrap_replicates(x, lambda a: lens.append(len(a)) or 0.0, 20, rng=0)
    assert lens == [10] * 20, f"every bootstrap sample has the same size n = 10 as the data; got sizes {sorted(set(lens))}"


def test_stage3_resamples_every_row(nb):
    x = np.arange(10.0)
    seen = set()
    nb.bootstrap_replicates(x, lambda a: seen.update(a.tolist()) or 0.0, 300, rng=5)
    assert seen == set(x.tolist()), (
        f"over 300 bootstrap samples of (0, 1, ..., 9), every value should be drawn at least once; never drawn: "
        f"{sorted(set(x.tolist()) - seen)}. indices must run over 0 .. n-1 inclusive: rng.integers(0, n, size=n).")
    r = np.asarray(nb.bootstrap_replicates(x, np.mean, 2000, rng=6))
    assert abs(r.mean() - 4.5) < 0.1, (
        f"bootstrap means of (0, ..., 9) should average the sample mean 4.5 (within 0.1), got {r.mean():.3f}. "
        "an off-by-one in the index range shifts it.")


def test_stage3_with_replacement(nb):
    n = 100
    r = np.asarray(nb.bootstrap_replicates(np.arange(float(n)), lambda a: len(np.unique(a)), 2000, rng=1))
    want = n * (1 - (1 - 1 / n) ** n)  # 63.4: a bootstrap sample contains about 63% of the distinct points
    assert abs(r.mean() - want) < 1.0, (
        f"a bootstrap sample of 100 distinct points should contain about {want:.1f} distinct values on average, got "
        f"{r.mean():.1f}. " + ("you resampled WITHOUT replacement (a permutation): draw n indices with replacement."
                               if r.mean() > 99 else ""))


def test_stage3_tuples_are_resampled_jointly(nb):
    rng = np.random.default_rng(2)
    x = rng.normal(size=60)
    y = x + 3.0
    r = np.asarray(nb.bootstrap_replicates((x, y), lambda a, b: np.mean(b - a), 300, rng=3))
    assert np.allclose(r, 3.0), (
        "with data = (x, y), use ONE set of indices per replicate for both arrays, so pairs stay together. "
        "here y = x + 3, so every replicate of mean(y - x) must be exactly 3.")
    r = np.asarray(nb.bootstrap_replicates((x, -x, x), lambda a, b, c: np.sum(a + b) + np.sum(c - a), 50, rng=4))
    assert np.allclose(r, 0.0), "three arrays: each replicate passes all three, resampled with the same indices"


def test_stage3_percentile_ci(nb):
    x = np.random.default_rng(5).normal(3.0, 2.0, 400)
    lo, hi = nb.bootstrap_ci(x, np.mean, n_boot=4000, rng=6)
    se = np.std(x) / math.sqrt(400)  # the bootstrap's own standard error uses ddof 0
    assert abs(lo - (x.mean() - 1.96 * se)) < 0.2 * se and abs(hi - (x.mean() + 1.96 * se)) < 0.2 * se, (
        f"for the mean of 400 normal points, the percentile interval should be about mean ± 1.96 se = "
        f"({x.mean() - 1.96 * se:.3f}, {x.mean() + 1.96 * se:.3f}), got ({lo:.3f}, {hi:.3f}). "
        "the endpoints are the alpha/2 and 1 - alpha/2 quantiles of the replicates.")
    reps = np.asarray(nb.bootstrap_replicates(x[:50], _var, 500, rng=7))
    for alpha in (0.05, 0.2):
        got = nb.bootstrap_ci(x[:50], _var, n_boot=500, alpha=alpha, rng=7)
        want = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        assert np.allclose(got, want, rtol=0, atol=1e-12), (
            f"bootstrap_ci (alpha = {alpha}) should return np.quantile(replicates, [alpha/2, 1 - alpha/2]) of "
            f"bootstrap_replicates(data, statistic, n_boot, rng=rng): expected {np.round(want, 5)}, got {np.round(got, 5)}")


def test_stage3_bad_method(nb):
    try:
        nb.bootstrap_ci(np.arange(5.0), np.mean, n_boot=10, method="basic", rng=0)
    except ValueError:
        return
    raise AssertionError("method='basic' isn't supported: raise ValueError for anything but 'percentile' and 'bca'")


# ---------------------------------------------------------------- stage 4: BCa


def test_stage4_jackknife(nb):
    got = np.asarray(nb.jackknife(np.array([1.0, 2.0, 3.0, 6.0]), np.mean))
    assert got.shape == (4,) and np.allclose(got, [11 / 3, 10 / 3, 3.0, 2.0]), (
        f"leave-one-out means of (1, 2, 3, 6) are (11/3, 10/3, 3, 2), got {np.round(got, 4).tolist()}")
    x = np.array([1.0, 2.0, 3.0])
    got = np.asarray(nb.jackknife((x, 10 * x), lambda a, b: np.sum(b) - np.sum(a)))
    assert np.allclose(got, [45.0, 36.0, 27.0]), (
        "with a tuple, remove row i from EVERY array: sum(10x) - sum(x) without row i is 9 * (6 - x_i)")


def test_stage4_bca_formula(nb):
    rng = np.random.default_rng(8)
    reps = rng.gamma(3.0, 1.0, 5000)
    jack = rng.gamma(2.0, 1.0, 40)  # skewed: a != 0
    for theta_hat, alpha in ((2.7, 0.05), (3.4, 0.1), (2.0, 0.01)):
        got = nb.bca_interval(theta_hat, reps, jack, alpha)
        want = _oracle_bca(theta_hat, reps, jack, alpha)
        assert np.allclose(got, want, rtol=0, atol=1e-10), (
            f"bca_interval(theta_hat={theta_hat}, alpha={alpha}) should be {np.round(want, 6)}, got {np.round(got, 6)}. "
            "check z0 = Phi^-1(p0), a = sum(d^3) / (6 sum(d^2)^1.5) with d = mean(jack) - jack, and the adjusted "
            "levels Phi(z0 + (z0 + z) / (1 - a (z0 + z))).")


def test_stage4_bca_ties_and_degenerate(nb):
    reps = np.array([1.0, 2.0, 2.0, 2.0, 3.0] * 200)
    lo, hi = nb.bca_interval(2.0, reps, np.ones(10), 0.5)
    want = np.quantile(reps, [0.25, 0.75])
    assert np.allclose((lo, hi), want), (
        f"replicates tied with theta_hat count half: p0 = (#< + #<=) / (2B) = 0.5 here, so z0 = 0; with constant "
        f"jackknife values a = 0 (not nan), and BCa reduces to the percentile interval {want}. got ({lo}, {hi})")
    lo, hi = nb.bca_interval(-5.0, np.linspace(0, 1, 101), np.array([0.0, 1.0, 5.0]), 0.05)
    assert np.isfinite(lo) and np.isfinite(hi), (
        "theta_hat below every replicate gives p0 = 0 and z0 = -inf: clip p0 to [1/(2B), 1 - 1/(2B)] first")


def test_stage4_bootstrap_ci_bca(nb):
    x = np.random.default_rng(9).exponential(1.0, 40)
    got = nb.bootstrap_ci(x, _var, n_boot=2000, method="bca", rng=10)
    want = nb.bca_interval(_var(x), nb.bootstrap_replicates(x, _var, 2000, rng=10), nb.jackknife(x, _var), 0.05)
    assert np.allclose(got, want, rtol=0, atol=1e-12), (
        "bootstrap_ci(method='bca') should return bca_interval(statistic(data), replicates, jackknife(data, statistic), "
        f"alpha) with the same replicates as the percentile method: expected {np.round(want, 5)}, got {np.round(got, 5)}")
    got_g = nb.bootstrap_ci(x, _var, n_boot=500, method="bca", rng=np.random.default_rng(10))
    want_g = nb.bca_interval(_var(x), nb.bootstrap_replicates(x, _var, 500, rng=np.random.default_rng(10)),
                             nb.jackknife(x, _var), 0.05)
    assert np.allclose(got_g, want_g, rtol=0, atol=1e-12), (
        "with a Generator as rng, bootstrap_ci(method='bca') must use the replicates it drew ONCE; drawing a second "
        "set continues the generator's stream and gives a different interval")
    pct = nb.bootstrap_ci(x, _var, n_boot=2000, method="percentile", rng=10)
    assert got[0] > pct[0] and got[1] > pct[1], (
        f"for the variance of skewed data the percentile interval {np.round(pct, 3)} sits too low; BCa should shift "
        f"BOTH ends up. got {np.round(got, 3)}. check the signs of z0 and a.")


def test_stage4_matches_scipy(nb):
    x = np.random.default_rng(11).lognormal(0.0, 0.8, 30)
    ours = nb.bootstrap_ci(x, np.mean, n_boot=10_000, method="bca", rng=12)
    ref = stats.bootstrap((x,), np.mean, n_resamples=10_000, method="BCa", random_state=13).confidence_interval
    width = ref.high - ref.low
    assert abs(ours[0] - ref.low) < 0.06 * width and abs(ours[1] - ref.high) < 0.06 * width, (
        f"your BCa interval {np.round(ours, 4)} disagrees with scipy.stats.bootstrap's ({ref.low:.4f}, {ref.high:.4f}) "
        "by more than Monte Carlo noise explains")
