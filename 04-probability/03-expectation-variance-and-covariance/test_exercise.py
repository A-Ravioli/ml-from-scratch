"""Checks for exercise.py (expectation, variance & covariance).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.
"""
import math

import numpy as np
import pytest

DIE = np.arange(1, 7)
FAIR = np.full(6, 1 / 6)
OFFSET_DATA = 1e9 + np.array([4.0, 7.0, 13.0, 16.0])  # the lesson's worked example: variance 22.5


# ---------------------------------------------------------------- stage 1: expectation


def test_stage1_die_moments(nb):
    m = nb.expectation(DIE, FAIR)
    assert isinstance(m, float) and abs(m - 3.5) < 1e-12, f"E[X] for a fair die is 3.5, got {m!r}"
    m2 = nb.expectation(DIE, FAIR, lambda x: x**2)
    assert abs(m2 - 91 / 6) < 1e-12, (
        f"E[X^2] for a fair die is (1 + 4 + 9 + 16 + 25 + 36)/6 = 91/6 ≈ 15.1667, got {m2:.4f}. "
        "LOTUS: apply f to the VALUES and weight by the probabilities, sum f(x_i) p_i.")
    assert abs(m2 - m**2 - 35 / 12) < 1e-12, "Var = E[X^2] - E[X]^2 = 35/12 for a fair die"


def test_stage1_lotus_not_f_of_mean(nb):
    vals, probs = np.array([0.0, 10.0]), np.array([0.9, 0.1])
    got = nb.expectation(vals, probs, np.exp)
    want = 0.9 + 0.1 * math.exp(10)
    assert abs(got - want) < 1e-9 * want, (
        f"E[exp(X)] with P(0) = .9, P(10) = .1 is .9 + .1 e^10 ≈ {want:.1f}, got {got:.4f}. "
        + ("you computed exp(E[X]) = e^1: E[f(X)] is NOT f(E[X]) unless f is linear."
           if abs(got - math.e) < 1e-6 else ""))


def test_stage1_linearity(nb):
    rng = np.random.default_rng(0)
    vals = rng.normal(size=8)
    probs = rng.random(8)
    probs /= probs.sum()
    lhs = nb.expectation(vals, probs, lambda x: 3 * x + 2 * x**2 - 1)
    rhs = 3 * nb.expectation(vals, probs) + 2 * nb.expectation(vals, probs, lambda x: x**2) - 1
    assert abs(lhs - rhs) < 1e-12, "linearity: E[3X + 2X^2 - 1] should equal 3E[X] + 2E[X^2] - 1"


def test_stage1_mc_estimate_formula(nb):
    x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    mean, se = nb.mc_estimate(x)
    s = math.sqrt(32 / 7)  # sum of squared deviations is 32, n - 1 = 7
    assert abs(mean - 5.0) < 1e-12, f"the sample mean of {x.tolist()} is 5, got {mean}"
    assert abs(se - s / math.sqrt(8)) < 1e-12, (
        f"the standard error is s / sqrt(n) with s = sqrt(32/7) (ddof=1): {s / math.sqrt(8):.6f}, got {se:.6f}. "
        + ("you used ddof=0 (dividing by n)." if abs(se - 2 / math.sqrt(8)) < 1e-9 else
           "you returned the standard deviation, not the standard error: divide by sqrt(n)."
           if abs(se - s) < 1e-9 else ""))


def test_stage1_mc_error_bars_cover(nb):
    """95% intervals (mean ± 1.96 SE) for E[U^2] = 1/3 should cover about 95% of the time."""
    rng = np.random.default_rng(1)
    hits = 0
    for _ in range(400):
        m, se = nb.mc_estimate(rng.random(200) ** 2)
        hits += abs(m - 1 / 3) <= 1.96 * se
    assert 0.90 <= hits / 400 <= 0.99, (
        f"mean ± 1.96 SE covered the true value 1/3 in {hits}/400 experiments; it should be about 95%. "
        "check that the standard error shrinks like 1/sqrt(n).")


# ---------------------------------------------------------------- stage 2: variance, two ways


def test_stage2_both_agree_on_nice_data(nb):
    x = np.array([4.0, 7.0, 13.0, 16.0])
    for fn in ("naive_variance", "two_pass_variance"):
        got = getattr(nb, fn)(x)
        assert abs(got - 22.5) < 1e-9, f"{fn}([4, 7, 13, 16]) should be 22.5 (population variance), got {got}"


def test_stage2_naive_formula_fails_with_an_offset(nb):
    x = 1e8 + np.array([0.0, 1.0, 2.0])
    got = nb.naive_variance(x)
    assert abs(got - 2 / 3) > 0.1, (
        f"naive_variance(1e8 + [0, 1, 2]) returned {got}, which is close to the true 2/3. naive_variance is "
        "meant to be the textbook formula mean(x**2) - mean(x)**2, computed exactly as written: with values "
        "near 1e8, x**2 ~ 1e16 is stored with a spacing of 2, so the formula should give garbage like 2.0. "
        "(you've implemented a stable method: keep it for two_pass_variance.)")


def test_stage2_two_pass_survives_the_offset(nb):
    got = nb.two_pass_variance(OFFSET_DATA)
    assert abs(got - 22.5) < 1e-6, (
        f"two_pass_variance(1e9 + [4, 7, 13, 16]) should be 22.5, got {got}. subtract the mean FIRST, then "
        "square: the deviations are small numbers, so nothing cancels.")
    assert abs(nb.two_pass_variance(OFFSET_DATA, ddof=1) - 30.0) < 1e-6, (
        "with ddof=1, divide by n - 1 = 3: the unbiased sample variance is 90 / 3 = 30")


def test_stage2_matches_numpy(nb):
    x = np.random.default_rng(2).normal(5.0, 3.0, size=1000)
    assert abs(nb.two_pass_variance(x) - np.var(x)) < 1e-10, "two_pass_variance(x) should match np.var(x)"
    assert abs(nb.two_pass_variance(x, ddof=1) - np.var(x, ddof=1)) < 1e-10, "ddof=1 should match np.var(x, ddof=1)"


# ---------------------------------------------------------------- stage 3: RunningStats


def test_stage3_empty_and_one_at_a_time(nb):
    rs = nb.RunningStats()
    assert rs.count == 0, f"a new RunningStats has count 0, got {rs.count}"
    x = np.random.default_rng(3).normal(2.0, 1.5, size=500)
    for v in x:
        rs.update(v)
    assert rs.count == 500, f"after 500 single updates count should be 500, got {rs.count}"
    assert abs(float(rs.mean) - x.mean()) < 1e-12, f"running mean {float(rs.mean)} != np.mean {x.mean()}"
    assert abs(float(rs.var) - x.var()) < 1e-10, (
        f"running variance {float(rs.var):.10f} != np.var {x.var():.10f} (population, ddof=0). m2 is the running "
        "SUM of squared deviations; var = m2 / count.")
    assert abs(float(rs.std) - x.std()) < 1e-10, "std should be sqrt(var)"


def test_stage3_update_returns_self(nb):
    rs = nb.RunningStats()
    assert rs.update(1.0) is rs, "update should return self, so calls can be chained"


def test_stage3_batches_of_any_size(nb):
    x = np.random.default_rng(4).exponential(size=1000)
    rs = nb.RunningStats()
    i = 0
    for k in (1, 7, 0, 100, 2, 390, 500):
        rs.update(x[i:i + k])
        i += k
    assert rs.count == 1000, f"batches of sizes 1, 7, 0, 100, 2, 390, 500 make 1000 samples; count is {rs.count}"
    assert abs(float(rs.mean) - x.mean()) < 1e-12 and abs(float(rs.var) - x.var()) < 1e-10, (
        f"after uneven batches: mean {float(rs.mean):.6f} (want {x.mean():.6f}), var {float(rs.var):.6f} "
        f"(want {x.var():.6f}). the merge needs the cross term delta^2 * count * n_b / n, using the count "
        "BEFORE the update.")


def test_stage3_order_does_not_matter(nb):
    x = np.random.default_rng(5).normal(size=300)
    a, b = nb.RunningStats(), nb.RunningStats()
    a.update(x[:50]).update(x[50:])
    b.update(x[200:]).update(x[:200])
    assert abs(float(a.var) - float(b.var)) < 1e-12 and abs(float(a.mean) - float(b.mean)) < 1e-12, (
        "the same samples fed in a different order must give the same mean and variance")


def test_stage3_vector_samples(nb):
    X = np.random.default_rng(6).normal([0.0, 10.0, -5.0], [1.0, 2.0, 0.5], size=(400, 3))
    rs = nb.RunningStats(shape=(3,))
    rs.update(X[0])            # one sample of shape (3,)
    rs.update(X[1:150])        # a batch (149, 3)
    for row in X[150:]:
        rs.update(row)
    assert np.asarray(rs.mean).shape == (3,), f"mean should have shape (3,), got {np.asarray(rs.mean).shape}"
    assert np.allclose(rs.mean, X.mean(axis=0)) and np.allclose(rs.var, X.var(axis=0)), (
        "per-coordinate mean and variance should match X.mean(axis=0), X.var(axis=0). a sample of shape (3,) "
        "is ONE sample, a (k, 3) array is k samples.")


def test_stage3_shape_errors(nb):
    rs = nb.RunningStats(shape=(3,))
    with pytest.raises(ValueError):
        rs.update(np.zeros(6))
        pytest.fail("a (6,) array is neither one (3,) sample nor a (k, 3) batch: raise ValueError (do not reshape it)")
    with pytest.raises(ValueError):
        rs.update(np.zeros((4, 3, 1)))
        pytest.fail("a (4, 3, 1) array is not a (k, 3) batch: raise ValueError")


def test_stage3_stable_with_a_huge_offset(nb):
    rng = np.random.default_rng(7)
    x = 1e9 + rng.normal(size=10_000)
    rs = nb.RunningStats()
    for chunk in np.array_split(x, 37):
        rs.update(chunk)
    for v in x[:100]:  # and some single updates on top
        rs.update(v)
    full = np.concatenate([x, x[:100]])
    want = np.var(full - 1e9)
    assert abs(float(rs.var) - want) < 1e-6 * want, (
        f"data near 1e9 with variance ~1: your running variance is {float(rs.var):.8f}, the truth is "
        f"{want:.8f}. if you keep running sums of x and x^2, that's the naive formula again: track the mean "
        "and the sum of squared DEVIATIONS (m2) instead.")


# ---------------------------------------------------------------- stage 4: covariance


def test_stage4_matches_numpy(nb):
    X = np.random.default_rng(8).normal(size=(200, 4)) @ np.array(
        [[1.0, 0.5, 0, 0], [0, 1.0, 0.3, 0], [0, 0, 1.0, -0.4], [0, 0, 0, 2.0]])
    C = np.asarray(nb.covariance(X))
    assert C.shape == (4, 4), f"covariance of X (200, 4) should be (4, 4), got {C.shape}"
    assert np.allclose(C, np.cov(X, rowvar=False)), (
        "covariance(X) should match np.cov(X, rowvar=False): center each COLUMN, then Xc.T @ Xc / (n - 1). "
        + ("yours is (n-1)/n times too small: the default is ddof=1."
           if np.allclose(C, np.cov(X, rowvar=False, ddof=0)) else ""))
    C0 = np.asarray(nb.covariance(X, ddof=0))
    assert np.allclose(C0, np.cov(X, rowvar=False, ddof=0)), "ddof=0 should divide by n"
    assert np.allclose(C, C.T), "a covariance matrix is symmetric"


def test_stage4_worked_example(nb):
    X = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [6.0, 4.0]])
    C = np.asarray(nb.covariance(X))
    want = np.array([[14 / 3, 7 / 3], [7 / 3, 10 / 3]])
    assert np.allclose(C, want), (
        f"for the lesson's 4 points the covariance is [[14/3, 7/3], [7/3, 10/3]], got {np.round(C, 4).tolist()}")


def test_stage4_stable_with_an_offset(nb):
    rng = np.random.default_rng(9)
    Z = rng.normal(size=(1000, 2))
    X = 1e9 + Z
    C = np.asarray(nb.covariance(X))
    want = np.cov(Z, rowvar=False)
    assert np.allclose(C, want, atol=1e-6), (
        f"columns near 1e9 with unit variance: got {np.round(C, 4).tolist()}, want {np.round(want, 4).tolist()}. "
        "E[XY] - E[X]E[Y] cancels catastrophically here: center first (two passes).")


def test_stage4_one_dimensional_input(nb):
    x = np.array([4.0, 7.0, 13.0, 16.0])
    C = np.asarray(nb.covariance(x))
    assert C.shape == (1, 1) and abs(C[0, 0] - 30.0) < 1e-9, (
        f"a 1-D input is one variable: covariance([4, 7, 13, 16]) is [[30]] (ddof=1), got {C.tolist()}")


def test_stage4_correlation(nb):
    rng = np.random.default_rng(10)
    x = rng.normal(size=500)
    X = np.column_stack([x, 2 * x + 0.5 * rng.normal(size=500), -3 * x, rng.normal(size=500) * 100])
    R = np.asarray(nb.correlation(X))
    assert np.allclose(np.diag(R), 1), "a correlation matrix has ones on the diagonal"
    assert np.allclose(R, np.corrcoef(X, rowvar=False)), (
        "correlation should match np.corrcoef: divide cov_ij by sd_i * sd_j")
    assert abs(R[0, 2] + 1) < 1e-12, "x and -3x are perfectly negatively correlated: R[0, 2] = -1"
