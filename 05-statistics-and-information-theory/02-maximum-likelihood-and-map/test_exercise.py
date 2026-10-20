"""Checks for exercise.py (maximum likelihood & MAP).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook. Log-likelihoods and posteriors are checked against
scipy.stats; every random dataset uses a fixed seed.
"""
import math

import numpy as np
from scipy import stats

COIN = np.array([1, 0, 1, 1, 0, 1, 1, 1])  # the lesson's worked example: 6 heads in 8 flips


# ---------------------------------------------------------------- stage 1: log-likelihoods


def test_stage1_bernoulli_hand_and_scipy(nb):
    got = nb.bernoulli_loglik(np.array([1, 0, 1, 1]), 0.75)
    want = 3 * math.log(0.75) + math.log(0.25)
    assert isinstance(got, float), "return a Python float"
    assert abs(got - want) < 1e-12, (
        f"three ones and a zero at p = 0.75: 3 log 0.75 + log 0.25 = {want:.5f}, got {got}. "
        "remember the zeros contribute log(1 - p).")
    x = np.random.default_rng(0).integers(0, 2, 50)
    for p in (0.1, 0.5, 0.93):
        assert abs(nb.bernoulli_loglik(x, p) - stats.bernoulli(p).logpmf(x).sum()) < 1e-9, (
            f"bernoulli_loglik disagrees with scipy at p = {p}")


def test_stage1_bernoulli_edges(nb):
    assert nb.bernoulli_loglik(np.array([1, 1, 1]), 1.0) == 0.0, (
        "three heads at p = 1 have probability 1, log-likelihood 0. the zeros' term (n - k) log(1 - p) "
        "has n - k = 0, and 0 * log 0 counts as 0, not nan.")
    assert nb.bernoulli_loglik(np.array([0, 0]), 0.0) == 0.0, "two tails at p = 0: log-likelihood 0"
    assert nb.bernoulli_loglik(np.array([1, 0, 1]), 1.0) == -math.inf, (
        "a tail is impossible at p = 1, so the log-likelihood is -inf (not nan, and no exception)")


def test_stage1_normal_and_poisson(nb):
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 3.0, 40)
    for mu, s2 in ((2.0, 9.0), (0.0, 1.0), (-1.5, 0.25)):
        got, want = nb.normal_loglik(x, mu, s2), stats.norm(mu, math.sqrt(s2)).logpdf(x).sum()
        assert abs(got - want) < 1e-8 * abs(want), (
            f"normal_loglik(mu={mu}, sigma2={s2}) = {got:.4f}, scipy says {want:.4f}. "
            "the second argument is the VARIANCE, and the normalizer is -log(2 pi sigma2)/2 per point.")
    k = rng.poisson(3.0, 30)
    for lam in (0.5, 3.0, 7.0):
        got, want = nb.poisson_loglik(k, lam), stats.poisson(lam).logpmf(k).sum()
        assert abs(got - want) < 1e-9 * abs(want), (
            f"poisson_loglik(lam={lam}) = {got:.4f}, scipy says {want:.4f}. did you forget the -log(x!) terms?")
    assert nb.poisson_loglik(np.array([0, 0, 0]), 0.0) == 0.0, "all-zero counts at rate 0 have probability 1"


# ---------------------------------------------------------------- stage 2: closed-form MLEs


def test_stage2_hand_examples(nb):
    assert abs(nb.mle_bernoulli(COIN) - 0.75) < 1e-12, f"6 heads in 8 flips: p̂ = 0.75, got {nb.mle_bernoulli(COIN)}"
    mu, s2 = nb.mle_normal(np.array([4.0, 6.0, 5.0, 9.0]))
    assert abs(mu - 6.0) < 1e-12 and abs(s2 - 3.5) < 1e-12, (
        f"(4, 6, 5, 9) has MLE mu = 6 and sigma^2 = (4 + 0 + 1 + 9) / 4 = 3.5, got ({mu}, {s2}). "
        + ("you divided by n - 1: the MLE divides by n." if abs(s2 - 14 / 3) < 1e-9 else ""))
    assert abs(nb.mle_poisson(np.array([2, 0, 3, 1])) - 1.5) < 1e-12, "the Poisson MLE is the mean count"
    r = nb.mle_exponential(np.array([0.5, 1.5, 1.0, 3.0]))
    assert abs(r - 1 / 1.5) < 1e-12, (
        f"the exponential's MLE rate is 1 / mean = 1 / 1.5 = 0.667, got {r}. "
        + ("you returned the mean (that's the MLE of the SCALE, 1/rate)." if abs(r - 1.5) < 1e-9 else ""))


def test_stage2_maximize_the_likelihood(nb):
    rng = np.random.default_rng(2)
    checks = [
        ("mle_bernoulli", rng.integers(0, 2, 37), lambda x, p: stats.bernoulli(p).logpmf(x).sum()),
        ("mle_poisson", rng.poisson(4.2, 25), lambda x, l: stats.poisson(l).logpmf(x).sum()),
        ("mle_exponential", rng.exponential(0.3, 25), lambda x, r: stats.expon(scale=1 / r).logpdf(x).sum()),
    ]
    for name, x, ll in checks:
        est = getattr(nb, name)(x)
        for h in (1e-3, -1e-3):
            assert ll(x, est) >= ll(x, est * (1 + h)), (
                f"{name}: moving the estimate by 0.1% INCREASED the log-likelihood, so it isn't the maximizer")
    x = rng.normal(-1.0, 2.0, 30)
    mu, s2 = nb.mle_normal(x)
    best = stats.norm(mu, math.sqrt(s2)).logpdf(x).sum()
    for dm, ds in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
        assert best >= stats.norm(mu + dm, math.sqrt(s2 * (1 + ds))).logpdf(x).sum(), (
            "mle_normal: a nearby (mu, sigma^2) has a higher likelihood")


# ---------------------------------------------------------------- stage 3: beta-bernoulli


def test_stage3_update_hand_example(nb):
    a, b = nb.beta_bernoulli_posterior(COIN, 2.0, 2.0)
    assert (a, b) == (8.0, 4.0), (
        f"Beta(2, 2) prior plus 6 heads and 2 tails gives Beta(2 + 6, 2 + 2) = Beta(8, 4), got Beta({a}, {b}). "
        "heads add to a, tails to b.")
    assert abs(nb.beta_mean(8.0, 4.0) - 2 / 3) < 1e-12, "the mean of Beta(8, 4) is 8 / 12"
    assert abs(nb.beta_mode(8.0, 4.0) - 0.7) < 1e-12, (
        f"the mode (MAP) of Beta(8, 4) is (8 - 1) / (8 + 4 - 2) = 0.7, got {nb.beta_mode(8.0, 4.0)}")


def test_stage3_sequential_equals_batch(nb):
    x = np.random.default_rng(3).integers(0, 2, 30)
    a, b = 0.5, 3.0
    for chunk in np.array_split(x, 7):
        a, b = nb.beta_bernoulli_posterior(chunk, a, b)
    a2, b2 = nb.beta_bernoulli_posterior(x, 0.5, 3.0)
    assert abs(a - a2) < 1e-12 and abs(b - b2) < 1e-12, (
        "updating on the data in 7 chunks (each posterior becoming the next prior) must give the same "
        f"posterior as updating on all of it at once: got Beta({a}, {b}) vs Beta({a2}, {b2})")
    assert nb.beta_bernoulli_posterior(np.array([], dtype=int), 2.0, 5.0) == (2.0, 5.0), "no data: posterior = prior"


def test_stage3_mode_matches_scipy_and_edges(nb):
    for a, b in ((3.0, 7.0), (1.0, 4.0), (5.0, 1.0), (2.5, 2.5)):
        grid = np.linspace(0, 1, 200_001)
        want = grid[np.argmax(stats.beta(a, b).pdf(grid))]
        assert abs(nb.beta_mode(a, b) - want) < 1e-4, f"mode of Beta({a}, {b}) should be {want:.4f}, got {nb.beta_mode(a, b)}"
    assert abs(nb.beta_mode(4.0, 1.0) - 1.0) < 1e-12, "Beta(4, 1) (uniform prior, 3 heads): MAP = 1 = the MLE"
    for a, b in ((1.0, 1.0), (0.5, 3.0), (2.0, 0.9)):
        try:
            nb.beta_mode(a, b)
        except ValueError:
            continue
        raise AssertionError(f"beta_mode({a}, {b}) should raise ValueError: the formula only applies for a, b >= 1, a + b > 2")


# ---------------------------------------------------------------- stage 4: normal-normal


def test_stage4_hand_example(nb):
    m, v = nb.normal_posterior(np.array([4.0, 6.0, 5.0, 9.0]), 4.0, 0.0, 1.0)
    assert abs(v - 0.5) < 1e-12, (
        f"precisions add: 1/1 (prior) + 4/4 (data) = 2, so the posterior variance is 0.5; got {v}")
    assert abs(m - 3.0) < 1e-12, (
        f"the posterior mean is (0/1 + 24/4) / 2 = 3: halfway between the prior mean 0 and the sample mean 6, "
        f"because prior and data have equal precision. got {m}")


def test_stage4_limits_and_sequential(nb):
    x = np.random.default_rng(4).normal(3.0, 2.0, 50)
    m, v = nb.normal_posterior(x, 4.0, -10.0, 1e12)
    assert abs(m - x.mean()) < 1e-6 and abs(v - 4.0 / 50) < 1e-9, (
        "with a nearly flat prior (tau2 huge) the posterior is N(x̄, sigma2 / n): the MLE with its sampling variance")
    m, v = nb.normal_posterior(x, 4.0, -10.0, 1e-12)
    assert abs(m + 10.0) < 1e-6, "with a nearly certain prior (tau2 tiny) the posterior mean stays at mu0"
    m1, v1 = nb.normal_posterior(x[:20], 4.0, 1.0, 2.0)
    m2, v2 = nb.normal_posterior(x[20:], 4.0, m1, v1)
    m3, v3 = nb.normal_posterior(x, 4.0, 1.0, 2.0)
    assert abs(m2 - m3) < 1e-12 and abs(v2 - v3) < 1e-12, (
        "updating on 20 points and then (with that posterior as the prior) on the other 30 must equal one update on all 50")


# ---------------------------------------------------------------- stage 5: grid posteriors


def test_stage5_grid_matches_beta(nb):
    grid = np.linspace(0.0005, 0.9995, 1000)
    k, n = 6, 8
    post = np.asarray(nb.grid_posterior(lambda p: k * np.log(p) + (n - k) * np.log1p(-p),
                                        lambda p: stats.beta(2, 2).logpdf(p), grid))
    assert post.shape == grid.shape and abs(post.sum() - 1) < 1e-12, "the grid posterior must sum to 1"
    mean = float(np.sum(grid * post))
    assert abs(mean - 8 / 12) < 1e-4, (
        f"a Beta(2, 2) prior and 6 heads in 8 flips give the Beta(8, 4) posterior, mean 0.6667; your grid "
        f"posterior has mean {mean:.4f}. add the log-prior to the log-likelihood before normalizing.")
    assert abs(grid[np.argmax(post)] - 0.7) < 1e-3, "the grid posterior's peak (the MAP) should be at 0.7"


def test_stage5_no_underflow(nb):
    x = np.random.default_rng(5).normal(1.0, 1.0, 20_000)
    grid = np.linspace(0.95, 1.05, 201)
    post = np.asarray(nb.grid_posterior(lambda mu: -0.5 * ((x[:, None] - mu) ** 2).sum(axis=0),
                                        lambda mu: np.zeros_like(mu), grid))
    assert np.all(np.isfinite(post)) and abs(post.sum() - 1) < 1e-9, (
        "with 20,000 data points the log-likelihoods are about -10,000: exp() of them is 0, and 0/0 is nan. "
        "subtract the max (or use logsumexp) before exponentiating.")
    sd = float(np.sqrt(np.sum(post * (grid - np.sum(grid * post)) ** 2)))
    assert abs(sd - 1 / math.sqrt(20_000)) < 3e-4, f"the posterior sd should be about 1/sqrt(n) = 0.0071, got {sd:.4f}"


def test_stage5_credible_interval(nb):
    lo, hi = nb.credible_interval(np.arange(5.0), np.array([0.1, 0.2, 0.4, 0.2, 0.1]), 0.6)
    assert (lo, hi) == (1.0, 3.0), (
        f"cumulative probabilities (0.1, 0.3, 0.7, 0.9, 1.0): the first to reach 0.2 is at 1, the first to reach 0.8 "
        f"is at 3, so the 60% interval is (1, 3); got ({lo}, {hi})")
    lo, hi = nb.credible_interval(np.arange(4.0), np.full(4, 0.25), 0.5)
    assert (lo, hi) == (0.0, 2.0), (
        f"cumulative probabilities (0.25, 0.5, 0.75, 1): the 25% tail is REACHED at 0 and the 75% point at 2, so the "
        f"50% interval is (0, 2); got ({lo}, {hi}). 'reaches' means cdf >= the tail (searchsorted's default side='left').")
    grid = np.linspace(0, 1, 20_001)
    post = stats.beta(8, 4).pdf(grid)
    post /= post.sum()
    lo, hi = nb.credible_interval(grid, post)
    want = stats.beta(8, 4).ppf([0.025, 0.975])
    assert abs(lo - want[0]) < 2e-4 and abs(hi - want[1]) < 2e-4, (
        f"the 95% equal-tailed interval of Beta(8, 4) is ({want[0]:.4f}, {want[1]:.4f}), got ({lo:.4f}, {hi:.4f}). "
        "the tails are (1 - level) / 2 each.")
