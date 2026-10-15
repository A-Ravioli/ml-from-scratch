"""Checks for exercise.py (distributions & sampling).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Statistical checks use fixed seeds and thresholds at very small p-values (about 1e-5), so a
correct sampler passes (whatever transform it uses), while a wrong one fails by a wide margin.
"""
import math
from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
from scipy import stats

N = 100_000
KS_C = 2.3  # Kolmogorov–Smirnov: a correct sampler has sqrt(n) * D < 2.3 except with p ≈ 5e-5

FORBIDDEN = ("normal", "standard_normal", "exponential", "standard_exponential", "gamma",
             "standard_gamma", "beta", "poisson", "binomial", "choice", "integers", "uniform",
             "multinomial", "geometric", "lognormal", "chisquare", "standard_cauchy",
             "multivariate_normal", "permutation", "shuffle")


BUDGET = 20_000_000  # uniforms one test may use; far above what any correct sampler here needs


class OnlyUniform(np.random.Generator):
    """A Generator that records any sampling method other than .random() that gets called, and
    stops a runaway loop: past `budget` uniforms it raises instead of letting a check hang."""

    def __init__(self, seed, budget=BUDGET):
        super().__init__(np.random.PCG64(seed))
        self.used = set()
        self.n_random = 0
        self.budget = budget

    def random(self, *args, **kwargs):
        out = super().random(*args, **kwargs)
        self.n_random += np.size(out)
        if self.n_random > self.budget:
            raise AssertionError(
                f"your code has drawn more than {self.budget:,} uniforms for this check, far more than a "
                "correct sampler needs: is a while-loop never finishing (e.g. a rejection test that never "
                "accepts, or a Knuth loop whose stopping condition is never met)?")
        return out


for _name in FORBIDDEN:
    def _make(name):
        def method(self, *args, **kwargs):
            self.used.add(name)
            return getattr(np.random.Generator, name)(self, *args, **kwargs)
        return method
    setattr(OnlyUniform, _name, _make(_name))


class Extreme(np.random.Generator):
    """A Generator whose .random() always returns the same extreme value (0.0 or the largest
    double below 1), to check the edge cases that a real generator hits once in 2^53 draws."""

    def __init__(self, value):
        super().__init__(np.random.PCG64(0))
        self.value = value

    def random(self, size=None, dtype=np.float64, out=None):
        return self.value if size is None else np.full(size, self.value)


LARGEST_BELOW_1 = 1.0 - 2.0**-53  # what rng.random() returns at its very top


@contextmanager
def _no_global_numpy_random():
    """Make np.random.<sampler> (the legacy global functions) raise inside the block."""
    patches = [mock.patch.object(np.random, name, side_effect=AssertionError(
        f"you called np.random.{name}: build every sampler from rng.random() only"))
        for name in FORBIDDEN if hasattr(np.random, name)]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _only_uniform(call, label):
    """Run call(rng) with a spying generator; fail if anything but rng.random() was used."""
    rng = OnlyUniform(0)
    with _no_global_numpy_random():
        out = call(rng)
    assert not rng.used, (
        f"{label} called rng.{sorted(rng.used)[0]}(...). the rule of this notebook: the ONLY source "
        "of randomness is rng.random(). build everything else from it (and from your earlier samplers).")
    assert rng.n_random > 0, f"{label} never called rng.random(): where does its randomness come from?"
    return out


def _ks(x, cdf, label, hint=""):
    x = np.asarray(x, dtype=float).ravel()
    d = stats.kstest(x, cdf).statistic
    crit = KS_C / math.sqrt(len(x))
    assert d < crit, (
        f"{label}: your {len(x):,} draws don't follow the target distribution. the largest gap between "
        f"their empirical CDF and the true CDF is {d:.4f}; a correct sampler stays below {crit:.4f}. "
        f"(sample mean {x.mean():.4f}, std {x.std():.4f}.) {hint}")


def _chi2(counts, probs, label, hint=""):
    counts = np.asarray(counts, dtype=float)
    expected = np.asarray(probs, dtype=float) * counts.sum()
    mask = expected > 0
    assert np.all(counts[~mask] == 0), (
        f"{label}: you drew a category whose probability is 0 (categories {np.flatnonzero(~mask & (counts > 0)).tolist()}).")
    chi2 = float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]))
    crit = stats.chi2.ppf(1 - 1e-5, mask.sum() - 1)
    assert chi2 < crit, (
        f"{label}: the frequencies don't match the probabilities (chi-square {chi2:.1f}, a correct sampler "
        f"stays below {crit:.1f}). observed {np.round(counts / counts.sum(), 4).tolist()} vs expected "
        f"{np.round(np.asarray(probs, dtype=float), 4).tolist()}. {hint}")


def _check_size_conventions(f, label, is_int=False):
    f(50, OnlyUniform(9, budget=1_000_000))  # guarded first call: a runaway loop fails here instead of hanging
    one = f(None, 1)
    kind = (int, np.integer) if is_int else (float,)
    assert isinstance(one, kind) and np.ndim(one) == 0, (
        f"{label} with size=None should return a single Python {'int' if is_int else 'float'}, "
        f"got {type(one).__name__} {one!r}")
    arr = np.asarray(f((3, 4), 2))
    assert arr.shape == (3, 4), f"{label} with size=(3, 4) should return shape (3, 4), got {arr.shape}"
    v = np.asarray(f(7, 3))
    assert v.shape == (7,), f"{label} with size=7 should return shape (7,), got {v.shape}"
    if is_int:
        assert np.issubdtype(v.dtype, np.integer), f"{label} should return integers, got dtype {v.dtype}"
    a, b = np.asarray(f(50, 123)), np.asarray(f(50, 123))
    assert np.array_equal(a, b), f"{label}: the same int seed must give the same draws (use np.random.default_rng(rng))"
    g = np.random.default_rng(5)
    c, d = np.asarray(f(50, g)), np.asarray(f(50, g))
    assert not np.array_equal(c, d), (
        f"{label}: two calls with the same Generator gave identical draws. pass the Generator itself "
        "through (np.random.default_rng(g) returns g unchanged), don't re-seed it.")


# ---------------------------------------------------------------- stage 1: inverse CDF


def test_stage1_uniform(nb):
    _check_size_conventions(lambda size, rng: nb.uniform(2.0, 5.0, size, rng=rng), "uniform")
    x = _only_uniform(lambda rng: nb.uniform(2.0, 5.0, N, rng=rng), "uniform")
    assert x.min() >= 2.0 and x.max() < 5.0, f"Uniform(2, 5) draws must lie in [2, 5), got [{x.min()}, {x.max()}]"
    _ks(x, stats.uniform(2.0, 3.0).cdf, "Uniform(2, 5)", "rescale u in [0, 1) as low + (high - low) * u.")
    with pytest.raises(ValueError):
        nb.uniform(1.0, 1.0)


def test_stage1_inverse_cdf_feeds_open_interval(nb):
    seen = []

    def spy(u):
        seen.append(np.asarray(u, dtype=float).copy())
        return u

    x = _only_uniform(lambda rng: nb.inverse_cdf_sample(spy, 20_000, rng=rng), "inverse_cdf_sample")
    u = np.concatenate([s.ravel() for s in seen])
    assert u.size == 20_000, f"inv_cdf should be called on all 20,000 uniforms (vectorized), it saw {u.size}"
    assert np.all((u > 0) & (u < 1)), "inv_cdf must only ever see u strictly between 0 and 1"
    _ks(x, stats.uniform().cdf, "inverse_cdf_sample with the identity", "with inv_cdf(u) = u you should get uniforms back.")
    one = nb.inverse_cdf_sample(lambda u: 2 * u, rng=0)
    assert isinstance(one, float), f"size=None should return a Python float, got {type(one).__name__}"
    for value in (0.0, LARGEST_BELOW_1):
        seen.clear()
        nb.inverse_cdf_sample(spy, 4, rng=Extreme(value))
        u = np.concatenate([s.ravel() for s in seen])
        assert np.all((u > 0) & (u < 1)), (
            f"when rng.random() returns {value!r} (it can: it returns k / 2^53 for k = 0 .. 2^53 - 1), "
            f"your inv_cdf received {float(u[0])!r}. it must only ever see the OPEN interval (0, 1): replace "
            "an exact 0 by a tiny positive number such as 2^-54. (careful: adding 2^-54 to every u rounds "
            "the top value up to exactly 1.0, and 1 - rng.random() can be exactly 1.0 too.)")


def test_stage1_inverse_cdf_logistic(nb):
    # the logistic distribution has inverse CDF log(u / (1 - u))
    x = nb.inverse_cdf_sample(lambda u: np.log(u / (1 - u)), N, rng=1)
    assert np.all(np.isfinite(x)), "got inf/nan: some u was exactly 0 or 1. shift rng.random() into (0, 1)."
    _ks(x, stats.logistic.cdf, "inverse_cdf_sample(logistic inverse CDF)")


def test_stage1_exponential(nb):
    _check_size_conventions(lambda size, rng: nb.exponential(2.0, size, rng=rng), "exponential")
    x = _only_uniform(lambda rng: nb.exponential(4.0, N, rng=rng), "exponential")
    assert np.all(x >= 0) and np.all(np.isfinite(x)), "exponential draws must be finite and >= 0"
    _ks(x, stats.expon(scale=1 / 4.0).cdf, "Exponential(rate=4)",
        f"the mean should be 1/rate = 0.25. if yours is near {4.0:.0f}, you multiplied by the rate "
        "instead of dividing. if your draws are negative, check the sign: F^-1(u) = -log(1 - u) / rate.")
    with pytest.raises(ValueError):
        nb.exponential(0.0)


# ---------------------------------------------------------------- stage 2: discrete


def test_stage2_bernoulli(nb):
    _check_size_conventions(lambda size, rng: nb.bernoulli(0.3, size, rng=rng), "bernoulli", is_int=True)
    x = _only_uniform(lambda rng: nb.bernoulli(0.3, N, rng=rng), "bernoulli")
    assert set(np.unique(x)) <= {0, 1}, f"bernoulli draws must be 0 or 1, got values {np.unique(x)[:5]}"
    _chi2(np.bincount(x, minlength=2), [0.7, 0.3], "Bernoulli(0.3)",
          "P(1) should be p: return 1 when u < p.")
    assert np.all(nb.bernoulli(0.0, 100, rng=0) == 0) and np.all(nb.bernoulli(1.0, 100, rng=0) == 1), (
        "p = 0 must always give 0 and p = 1 must always give 1")
    with pytest.raises(ValueError):
        nb.bernoulli(1.5)


def test_stage2_categorical(nb):
    probs = np.array([0.1, 0.0, 0.25, 0.4, 0.25])
    _check_size_conventions(lambda size, rng: nb.categorical(probs, size, rng=rng), "categorical", is_int=True)
    x = _only_uniform(lambda rng: nb.categorical(probs, N, rng=rng), "categorical")
    assert x.min() >= 0 and x.max() <= 4, f"categories must be in 0..4, got range [{x.min()}, {x.max()}]"
    _chi2(np.bincount(x, minlength=5), probs, "categorical([.1, 0, .25, .4, .25])",
          "build the CDF with np.cumsum and find where each u falls (np.searchsorted). an off-by-one "
          "shifts every frequency to the next category.")
    tail = np.array([0.5, 0.5, 0.0])
    assert np.all(nb.categorical(tail, 20_000, rng=3) < 2), (
        "probs = [.5, .5, 0]: category 2 has probability 0 and must never be drawn, even when the "
        "cumulative sum rounds to slightly less than 1.")
    with pytest.raises(ValueError):
        nb.categorical([0.5, 0.6])
    with pytest.raises(ValueError):
        nb.categorical([1.2, -0.2])


def test_stage2_categorical_rounding_in_cdf(nb):
    # ten categories of 0.1: the float cumsum ends at 0.9999999999999999, not 1
    probs = np.full(10, 0.1)
    x = nb.categorical(probs, N, rng=4)
    assert x.max() <= 9, (
        f"drew category {x.max()} from 10 categories. np.cumsum([0.1] * 10)[-1] is 0.9999999999999999, so "
        "a u above it falls off the end. scale u by cdf[-1] (or clip the index).")
    _chi2(np.bincount(x, minlength=10), probs, "categorical(10 x 0.1)")
    first = np.asarray(nb.categorical(np.array([0.0, 1.0]), 3, rng=Extreme(0.0)))
    assert np.all(first == 1), (
        f"probs = [0, 1] with u = 0.0 (rng.random() can return exactly 0): category 0 has probability 0, but "
        f"you drew {first.tolist()}. with np.searchsorted(cdf, u) use side='right', so an empty interval "
        "(cdf[i-1] == cdf[i]) is skipped.")
    top = np.asarray(nb.categorical(probs, 3, rng=Extreme(LARGEST_BELOW_1)))
    assert np.all(top == 9), (
        f"when rng.random() returns its largest value 1 - 2^-53, the draw from 10 x 0.1 should be the last "
        f"category, 9; got {top.tolist()}. np.cumsum([0.1] * 10)[-1] equals 1 - 2^-53 too, so the plain "
        "searchsorted index is 10, off the end. scale u by cdf[-1] (or clip the index).")


@pytest.mark.parametrize("lam", [0.5, 4.0, 25.0, 200.0])
def test_stage2_poisson(nb, lam):
    x = _only_uniform(lambda rng: nb.poisson(lam, N, rng=rng), f"poisson({lam})")
    assert np.issubdtype(np.asarray(x).dtype, np.integer) and x.min() >= 0, "Poisson draws are non-negative ints"
    se = math.sqrt(lam / N)
    assert abs(x.mean() - lam) < 6 * se, (
        f"Poisson({lam}): the sample mean should be {lam} ± {6 * se:.3f}, got {x.mean():.4f}. "
        "Knuth: multiply uniforms until the product drops to exp(-lam) or below; the count is the number "
        "of factors BEFORE that happens (an off-by-one gives mean lam + 1).")
    kmax = int(lam + 10 * math.sqrt(lam) + 10)
    counts = np.bincount(np.minimum(x, kmax), minlength=kmax + 1)
    p = stats.poisson.pmf(np.arange(kmax + 1), lam)
    p[-1] += stats.poisson.sf(kmax, lam)
    keep = p * N >= 5  # pool rare values so the chi-square test is valid
    _chi2(np.append(counts[keep], counts[~keep].sum()), np.append(p[keep], p[~keep].sum()), f"Poisson({lam})")


def test_stage2_poisson_conventions(nb):
    _check_size_conventions(lambda size, rng: nb.poisson(3.0, size, rng=rng), "poisson", is_int=True)
    _check_size_conventions(lambda size, rng: nb.poisson(80.0, size, rng=rng), "poisson (lam > 30)", is_int=True)
    assert np.all(nb.poisson(0.0, 50, rng=0) == 0), "Poisson(0) is always 0"
    with pytest.raises(ValueError):
        nb.poisson(-1.0)


# ---------------------------------------------------------------- stage 3: Box–Muller


def test_stage3_box_muller_worked_example(nb):
    z1, z2 = nb.box_muller(0.25, 0.125)
    r = math.sqrt(-2 * math.log(0.25))
    assert abs(z1 - r * math.cos(math.pi / 4)) < 1e-12 and abs(z2 - r * math.sin(math.pi / 4)) < 1e-12, (
        f"box_muller(0.25, 0.125): r = sqrt(-2 ln 0.25) = {r:.4f} and the angle is 2π · 0.125 = π/4, so "
        f"z1 = z2 = {r * math.cos(math.pi / 4):.4f}. got ({z1:.4f}, {z2:.4f}).")
    z1, z2 = nb.box_muller(np.array([1.0, math.exp(-2)]), np.array([0.0, 0.25]))
    assert np.allclose(z1, [0.0, 0.0]) and np.allclose(z2, [0.0, 2.0]), (
        "box_muller must work elementwise on arrays: u1 = [1, e^-2], u2 = [0, 1/4] gives "
        f"z1 = [0, 0], z2 = [0, 2]; got {np.round(z1, 4).tolist()}, {np.round(z2, 4).tolist()}")


def test_stage3_normal_distribution(nb):
    x = _only_uniform(lambda rng: nb.normal(1.0, 3.0, N, rng=rng), "normal")
    assert np.all(np.isfinite(x)), "normal returned inf/nan: u1 hit 0. use 1 - rng.random(), which is in (0, 1]."
    _ks(x, stats.norm(1.0, 3.0).cdf, "Normal(loc=1, scale=3)",
        "scale is the standard DEVIATION: return loc + scale * z. a std of 0.707 means you dropped the 2 in "
        "sqrt(-2 log u1).")


def test_stage3_normal_pairs_are_independent(nb):
    rng = OnlyUniform(7)
    x = nb.normal(0.0, 1.0, 2 * 50_000, rng=rng)
    assert rng.n_random <= 2 * 50_000 + 2, (
        f"100,000 normals used {rng.n_random:,} uniforms. Box–Muller makes TWO normals from two "
        "uniforms: use both z1 and z2.")
    xs = np.sort(np.asarray(x))
    assert len(np.unique(xs)) == len(xs), "your normal draws contain exact duplicates: are you reusing uniforms?"
    # whatever order you put z1 and z2 in, the draws must be independent: check squares of neighbours
    r_adj = np.corrcoef(x[:-1] ** 2, x[1:] ** 2)[0, 1]
    half = len(x) // 2
    r_half = np.corrcoef(x[:half] ** 2, x[half:2 * half] ** 2)[0, 1]
    assert abs(r_adj) < 0.03 and abs(r_half) < 0.03, (
        f"your normals are dependent: corr(x_i^2, x_(i+1)^2) = {r_adj:.3f}, corr of squares across halves "
        f"= {r_half:.3f}. z1 and z2 from one pair share r, so they must come from the SAME r but different "
        "angles (cos and sin of one angle), never from the same u1 used twice for different pairs.")


def test_stage3_normal_conventions(nb):
    _check_size_conventions(lambda size, rng: nb.normal(0.0, 1.0, size, rng=rng), "normal")
    odd = np.asarray(nb.normal(0.0, 1.0, (3, 5), rng=0))
    assert odd.shape == (3, 5), f"15 draws (an odd number) should still come back as shape (3, 5), got {odd.shape}"
    for value in (0.0, LARGEST_BELOW_1):
        z = np.asarray(nb.normal(0.0, 1.0, 4, rng=Extreme(value)))
        assert np.all(np.isfinite(z)), (
            f"when rng.random() returns {value!r}, your normal gave {z.tolist()}. log(u1) needs u1 > 0: "
            "use u1 = 1 - rng.random(), which lies in (0, 1].")
    with pytest.raises(ValueError):
        nb.normal(0.0, -1.0)


# ---------------------------------------------------------------- stage 4: rejection sampling


def _capped(sample, cap):
    """wrap a proposal sampler so a rejection loop that never finishes fails instead of hanging."""
    used = [0]

    def wrapped(n, r):
        used[0] += n
        if used[0] > cap:
            raise AssertionError(
                f"rejection_sample has asked for more than {cap:,} proposals, dozens of times what this check "
                "needs: does your loop ever stop (are you counting accepted draws correctly)?")
        return sample(n, r)
    return wrapped


def _beta25(nb, size, rng, log_M=math.log(2.5)):
    # target Beta(2, 5), unnormalized: x (1 - x)^4 on (0, 1); proposal Uniform(0, 1); max of target ~0.08192
    def log_target(x):
        return np.log(30 * x) + 4 * np.log1p(-x)  # normalized Beta(2, 5) density, max ≈ 2.4576

    return nb.rejection_sample(log_target, _capped(lambda n, r: r.random(n), 100 * size),
                               lambda x: np.zeros_like(x), log_M, size, rng=rng)


def test_stage4_exact_count_and_distribution(nb):
    x = _only_uniform(lambda rng: _beta25(nb, 30_000, rng), "rejection_sample")
    x = np.asarray(x)
    assert x.shape == (30_000,), f"rejection_sample must return exactly `size` = 30,000 draws, got shape {x.shape}"
    d = stats.kstest(x, stats.beta(2, 5).cdf).statistic
    crit = 2.0 / math.sqrt(30_000)
    assert d < crit, (
        f"your accepted draws don't follow Beta(2, 5): KS distance {d:.4f} (should be < {crit:.4f}), mean "
        f"{x.mean():.4f} (should be 2/7 = 0.2857). accept x when u < target(x) / (M proposal(x)), i.e. "
        "log u < log_target - log_M - log_proposal.")


def test_stage4_acceptance_rate_is_one_over_M(nb):
    """With a normalized target, about 1/M of proposals are accepted: check the uniform budget."""
    rng = OnlyUniform(11)
    calls = []

    def proposal(n, r):
        calls.append(n)
        if sum(calls) > 2_000_000:
            raise AssertionError("rejection_sample asked for 100 proposals per draw: does your loop ever stop?")
        return r.random(n)

    def log_target(x):
        return np.log(30 * x) + 4 * np.log1p(-x)
    nb.rejection_sample(log_target, proposal, lambda x: np.zeros_like(x), math.log(5.0), 20_000, rng=rng)
    proposed = sum(calls)
    assert 20_000 * 5.0 * 0.95 <= proposed <= 20_000 * 5.0 * 4, (
        f"with M = 5 and a normalized target, about 5 proposals are needed per accepted draw, so ~100,000 "
        f"proposals for 20,000 draws; you made {proposed:,}. are you accepting with probability "
        "target / (M proposal)?")


def _capped_square(f):
    return _capped(f, 500_000)


def test_stage4_two_dimensional_draws(nb):
    # uniform on the unit disk by rejection from the square [-1, 1]^2
    def log_target(x):
        return np.where(np.sum(x**2, axis=1) <= 1, 0.0, -np.inf)

    @_capped_square
    def sample_square(n, r):
        return 2 * r.random((n, 2)) - 1
    pts = np.asarray(nb.rejection_sample(log_target, sample_square, lambda x: np.zeros(len(x)),
                                         0.0, 5_000, rng=OnlyUniform(2)))
    assert pts.shape == (5_000, 2), f"2-D proposals of shape (n, 2) should give accepted draws (5000, 2), got {pts.shape}"
    assert np.all(np.sum(pts**2, axis=1) <= 1), "every accepted point must lie inside the disk (target = 0 outside)"
    frac_inner = np.mean(np.sum(pts**2, axis=1) <= 0.25)
    assert abs(frac_inner - 0.25) < 0.025, (
        f"for a uniform point in the unit disk, P(radius < 1/2) = 1/4; got {frac_inner:.3f}")


def _halfnormal_by_exponential(nb, size, rng, calls):
    """target: the half-normal density sqrt(2/pi) exp(-x^2/2) on x > 0. proposal: Exponential(1),
    density exp(-x). the ratio peaks at x = 1, where it equals sqrt(2e/pi) ≈ 1.3155 = M."""
    def log_target(x):
        return 0.5 * math.log(2 / math.pi) - 0.5 * x**2

    def sample_q(n, r):
        calls.append(n)
        if sum(calls) > 100 * size:
            raise AssertionError("rejection_sample asked for 100 proposals per draw: does your loop ever stop?")
        return -np.log1p(-r.random(n))  # Exponential(1) by inversion

    def log_q(x):
        return -x

    return nb.rejection_sample(log_target, sample_q, log_q, 0.5 * math.log(2 * math.e / math.pi), size, rng=rng)


def test_stage4_non_uniform_proposal(nb):
    calls = []
    x = np.asarray(_only_uniform(lambda rng: _halfnormal_by_exponential(nb, 40_000, rng, calls),
                                 "rejection_sample"))
    assert x.shape == (40_000,) and np.all(x > 0), f"expected 40,000 positive draws, got shape {x.shape}"
    d = stats.kstest(x, stats.halfnorm.cdf).statistic
    crit = KS_C / math.sqrt(len(x))
    assert d < crit, (
        f"half-normal target, Exponential(1) proposal: your accepted draws are at KS distance {d:.4f} from the "
        f"half-normal (should be < {crit:.4f}; your mean {x.mean():.4f}, the true mean sqrt(2/pi) = 0.7979). "
        "with a non-uniform proposal the log_proposal term matters: accept when "
        "log u < log_target(x) - log_M - log_proposal(x) (a MINUS before log_proposal).")
    M = math.sqrt(2 * math.e / math.pi)
    proposed = sum(calls)
    assert 40_000 * M * 0.95 <= proposed <= 40_000 * M * 4, (
        f"with M = {M:.4f} and a normalized target, about {M:.2f} proposals are needed per accepted draw "
        f"(~{40_000 * M:,.0f} in all); you made {proposed:,}.")


def test_stage4_rejects_a_bad_envelope(nb):
    with pytest.raises(ValueError):
        _beta25(nb, 1_000, OnlyUniform(0), log_M=math.log(1.5))  # Beta(2, 5) peaks at 2.46 > 1.5 x uniform
        pytest.fail("Beta(2, 5) peaks at 2.46 but M = 1.5: the envelope doesn't cover the target. raise ValueError.")


# ---------------------------------------------------------------- stage 5: gamma and beta


@pytest.mark.parametrize("shape,scale", [(1.0, 1.0), (2.5, 2.0), (9.0, 0.5), (0.3, 1.0)])
def test_stage5_gamma_distribution(nb, shape, scale):
    x = _only_uniform(lambda rng: nb.gamma(shape, scale, N, rng=rng), f"gamma({shape}, {scale})")
    x = np.asarray(x)
    assert x.shape == (N,) and np.all(x > 0) and np.all(np.isfinite(x)), "gamma draws must be positive and finite"
    hint = (f"the mean should be shape * scale = {shape * scale}. " +
            ("for shape < 1, draw from Gamma(shape + 1) and multiply by u^(1/shape)."
             if shape < 1 else "check the Marsaglia–Tsang acceptance test and return d * v (times scale)."))
    _ks(x, stats.gamma(shape, scale=scale).cdf, f"Gamma(shape={shape}, scale={scale})", hint)


def test_stage5_gamma_is_efficient(nb):
    rng = OnlyUniform(3)
    with _no_global_numpy_random():
        nb.gamma(3.0, 1.0, 20_000, rng=rng)
    per_draw = rng.n_random / 20_000
    assert per_draw <= 4.0, (
        f"gamma(3) used {per_draw:.2f} uniforms per draw. Marsaglia–Tsang accepts over 95% of proposals, so it "
        "needs about 2 per draw (one normal, one acceptance uniform). about 6 per draw means the acceptance rate "
        "is ~1/3: check that c = 1 / sqrt(9 d), with d = shape - 1/3.")


def test_stage5_gamma_conventions(nb):
    _check_size_conventions(lambda size, rng: nb.gamma(3.0, 1.0, size, rng=rng), "gamma")
    _check_size_conventions(lambda size, rng: nb.gamma(0.5, 1.0, size, rng=rng), "gamma (shape < 1)")
    with pytest.raises(ValueError):
        nb.gamma(0.0)
    with pytest.raises(ValueError):
        nb.gamma(2.0, -1.0)


@pytest.mark.parametrize("a,b", [(2.0, 5.0), (0.5, 0.5), (30.0, 3.0)])
def test_stage5_beta_distribution(nb, a, b):
    x = _only_uniform(lambda rng: nb.beta(a, b, N, rng=rng), f"beta({a}, {b})")
    x = np.asarray(x)
    assert np.all((x > 0) & (x < 1)), f"Beta draws lie strictly in (0, 1), got range [{x.min()}, {x.max()}]"
    _ks(x, stats.beta(a, b).cdf, f"Beta({a}, {b})",
        f"the mean should be a / (a + b) = {a / (a + b):.4f}. use X / (X + Y) with X ~ Gamma(a), Y ~ Gamma(b), "
        "two INDEPENDENT draws.")


def test_stage5_beta_conventions(nb):
    _check_size_conventions(lambda size, rng: nb.beta(2.0, 3.0, size, rng=rng), "beta")
    with pytest.raises(ValueError):
        nb.beta(0.0, 1.0)
