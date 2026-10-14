"""Checks for exercise.py (hypothesis testing).

Every test takes `nb`, the notebook under test (your exercise.py, or the reference when
MLFS_SOLUTIONS=1), and is named `test_stageN_*` so `pytest -k stageN` works. The same
functions run live inside the notebook.

Permutation p-values are compared with exact p-values from full enumeration, within about 5
Monte Carlo standard errors; error rates are compared with their targets the same way.
"""
import itertools
import math

import numpy as np
from scipy import stats

X_SMALL = np.array([1.2, 3.4, 2.2, 5.0])
Y_SMALL = np.array([0.1, 0.5, 1.9])


def _exact_unpaired(x, y, alternative="two-sided"):
    pooled = np.concatenate([x, y])
    obs = x.mean() - y.mean()
    diffs = []
    for idx in itertools.combinations(range(len(pooled)), len(x)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        diffs.append(pooled[mask].mean() - pooled[~mask].mean())
    diffs = np.array(diffs)
    tol = 1e-12
    if alternative == "greater":
        return np.mean(diffs >= obs - tol)
    if alternative == "less":
        return np.mean(diffs <= obs + tol)
    return np.mean(np.abs(diffs) >= abs(obs) - tol)


def _exact_paired(d):
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    means = (signs * d).mean(axis=1)
    return np.mean(np.abs(means) >= abs(d.mean()) - 1e-12)


# ---------------------------------------------------------------- stage 1: p-values


def test_stage1_hand_example(nb):
    null = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 2.5, 4.0])
    for alt, want in (("greater", (1 + 3) / 8), ("less", (1 + 5) / 8), ("two-sided", (1 + 4) / 8)):
        got = nb.p_value(2.0, null, alt)
        assert abs(got - want) < 1e-12, (
            f"observed 2.0 against the null sample (-3, -1, 0, 0.5, 2, 2.5, 4), alternative '{alt}': "
            f"p = (1 + #as-extreme) / (1 + 7) = {want}, got {got}. "
            + ("count values EQUAL to the observed one as extreme (>=, not >)." if alt == "greater" and abs(got - 3 / 8) < 1e-9 else "")
            + ("two-sided counts |null| >= |observed|: -3, 2, 2.5 and 4." if alt == "two-sided" else ""))


def test_stage1_plus_one_and_ties(nb):
    got = nb.p_value(100.0, np.zeros(999))
    assert abs(got - 1 / 1000) < 1e-15, (
        f"no null value is as extreme as 100, but a Monte Carlo p-value is never 0: (1 + 0) / (1 + 999) = 0.001, got {got}")
    got = nb.p_value(0.1 + 0.2, np.array([0.3, 0.0, 0.0, 0.0]), "greater")
    assert abs(got - 2 / 5) < 1e-12, (
        "0.3 is a TIE with the observed 0.1 + 0.2 = 0.30000000000000004, not a smaller value: compare with a tiny tolerance "
        f"(null >= observed - 1e-12 * max(1, |observed|)). expected p = 2/5, got {got}")
    try:
        nb.p_value(1.0, np.zeros(3), "bigger")
    except ValueError:
        return
    raise AssertionError("alternative='bigger' isn't valid: raise ValueError")


# ---------------------------------------------------------------- stage 2: the null distribution


def test_stage2_unpaired_shapes_and_splits(nb):
    sizes = []

    def stat(a, b):
        sizes.append((len(a), len(b)))
        return float(np.sum(a) + np.sum(b))

    null = np.asarray(nb.permutation_null(X_SMALL, Y_SMALL, stat, n_permutations=50, rng=0))
    assert null.shape == (50,), f"one statistic per permutation: shape (50,), got {null.shape}"
    assert set(sizes) == {(4, 3)}, f"each permutation splits the pooled 7 values into 4 (like x) and 3 (like y); got {set(sizes)}"
    assert np.allclose(null, X_SMALL.sum() + Y_SMALL.sum()), (
        "a permutation rearranges the POOLED values: the total of x' and y' must always equal the original total")


def test_stage2_unpaired_matches_enumeration(nb):
    null = np.asarray(nb.permutation_null(X_SMALL, Y_SMALL, n_permutations=20_000, rng=1))
    pooled = np.concatenate([X_SMALL, Y_SMALL])
    exact = sorted({round(pooled[list(i)].mean() - np.delete(pooled, list(i)).mean(), 9)
                    for i in itertools.combinations(range(7), 4)})
    assert set(np.round(null, 9)) <= set(exact), (
        "a null value isn't the mean difference of any split of the pooled data into groups of 4 and 3: "
        "shuffle the pooled array and split it, don't resample with replacement")
    assert abs(null.mean()) < 0.05, f"the default statistic mean(x') - mean(y') has mean 0 under shuffling, got {null.mean():.3f}"
    sd = np.std([pooled[list(i)].mean() - np.delete(pooled, list(i)).mean() for i in itertools.combinations(range(7), 4)])
    assert abs(null.std() - sd) < 0.05 * sd, f"the null's spread should be {sd:.3f}, got {null.std():.3f}"


def test_stage2_paired_swaps_within_pairs(nb):
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=12), rng.normal(size=12)
    calls = []

    def same_pairs(a, b):
        calls.append(1)
        assert np.allclose(np.asarray(a) + np.asarray(b), x + y) and np.allclose(np.sort(np.stack([a, b]), axis=0),
                                                                                  np.sort(np.stack([x, y]), axis=0)), (
            "paired: each permutation may only swap x[i] and y[i] WITHIN pair i, so position i of (x', y') must hold "
            "the same two values as position i of (x, y). shuffling the order of x' (or across pairs) breaks this.")
        return 0.0

    nb.permutation_null(x, y, same_pairs, n_permutations=30, paired=True, rng=3)
    assert len(calls) == 30, f"the statistic should be called once per permutation (30), got {len(calls)}"
    d = x - y
    null = np.asarray(nb.permutation_null(x, y, n_permutations=20_000, paired=True, rng=4))
    sd = math.sqrt(np.sum(d**2)) / len(d)  # sd of mean(s * d) for random signs s
    assert abs(null.mean()) < 5 * sd / math.sqrt(20_000) and abs(null.std() - sd) < 0.03 * sd, (
        f"under random sign flips, mean(x' - y') has mean 0 and sd sqrt(sum d^2)/n = {sd:.4f}; got mean "
        f"{null.mean():.4f}, sd {null.std():.4f}. each pair should be swapped independently with probability 1/2.")
    try:
        nb.permutation_null(x, y[:5], paired=True, n_permutations=5, rng=0)
    except ValueError:
        pass
    else:
        raise AssertionError("paired=True with different lengths (12 and 5) should raise ValueError")


def test_stage2_uses_the_rng(nb):
    a = np.asarray(nb.permutation_null(X_SMALL, Y_SMALL, n_permutations=40, rng=5))
    b = np.asarray(nb.permutation_null(X_SMALL, Y_SMALL, n_permutations=40, rng=5))
    c = np.asarray(nb.permutation_null(X_SMALL, Y_SMALL, n_permutations=40, rng=6))
    assert np.array_equal(a, b) and not np.array_equal(a, c), "same seed -> same null; different seeds -> different null"


# ---------------------------------------------------------------- stage 3: the test, its size and power


def test_stage3_matches_exact_unpaired(nb):
    B = 20_000
    for alt in ("two-sided", "greater", "less"):
        obs, p = nb.permutation_test(X_SMALL, Y_SMALL, n_permutations=B, alternative=alt, rng=7)
        assert abs(obs - (X_SMALL.mean() - Y_SMALL.mean())) < 1e-12, "observed should be mean(x) - mean(y)"
        exact = _exact_unpaired(X_SMALL, Y_SMALL, alt)
        assert abs(p - exact) < 5 * math.sqrt(exact * (1 - exact) / B) + 1e-3, (
            f"alternative '{alt}': the exact permutation p-value (all 35 splits) is {exact:.4f}, you got {p:.4f}")


def test_stage3_matches_exact_paired(nb):
    rng = np.random.default_rng(8)
    x = rng.normal(0.0, 3.0, 8)  # a large per-pair component shared by x and y
    y = x - rng.normal(0.5, 0.7, 8)
    exact = _exact_paired(x - y)
    obs, p = nb.permutation_test(x, y, paired=True, n_permutations=20_000, rng=9)
    assert abs(p - exact) < 5 * math.sqrt(exact * (1 - exact) / 20_000) + 1e-3, (
        f"paired: the exact sign-flip p-value (all 256 sign patterns) is {exact:.4f}, you got {p:.4f}")
    _, p_unpaired = nb.permutation_test(x, y, n_permutations=5000, rng=10)
    assert p < p_unpaired, "here x and y share a large common component: the paired test should be more powerful"


def test_stage3_custom_statistic_and_floor(nb):
    x, y = np.arange(20.0) + 100, np.arange(20.0)
    x[-1] = 1000.0  # an outlier: the mean difference is no longer 100, the median difference still is
    obs, p = nb.permutation_test(x, y, lambda a, b: float(np.median(a) - np.median(b)), n_permutations=499, rng=11)
    assert obs == 100.0, f"use the statistic you were given: the median difference is 100, got {obs}"
    assert abs(p - 1 / 500) < 1e-12, (
        f"perfectly separated groups of 20: (almost) no shuffle is as extreme, so p = 1/(1 + 499) = 0.002 exactly, got {p}")


def test_stage3_size_and_power(nb):
    def null_world(rng):
        return rng.exponential(1.0, 15), rng.exponential(1.0, 15)

    def shifted(rng):
        return rng.normal(1.0, 1.0, 20), rng.normal(0.0, 1.0, 20)

    def test(x, y, rng):
        return nb.permutation_test(x, y, n_permutations=199, rng=rng)[1]

    size = nb.rejection_rate(test, null_world, 400, 0.05, rng=12)
    assert size <= 0.05 + 5 * math.sqrt(0.05 * 0.95 / 400), (
        f"when x and y come from the same distribution, a permutation test at alpha = 0.05 rejects at most 5% of "
        f"the time (here even with skewed data); you measured {size:.3f}")
    power = nb.rejection_rate(test, shifted, 300, 0.05, rng=13)
    exact_t = 1 - stats.nct(38, 1.0 * math.sqrt(10)).cdf(stats.t(38).ppf(0.975))  # t-test power, about 0.87
    assert abs(power - exact_t) < 0.08, (
        f"a shift of one standard deviation with 20 per group should be detected about {exact_t:.2f} of the time; "
        f"you measured {power:.3f}")


def test_stage3_rejection_rate_counts(nb):
    calls = []

    def sim(rng):
        calls.append(rng)
        return None, None

    ps = iter([0.01, 0.05, 0.051, 0.5, 0.0])
    rate = nb.rejection_rate(lambda x, y, rng: next(ps), sim, 5, 0.05, rng=0)
    assert abs(rate - 3 / 5) < 1e-12, f"p-values (0.01, 0.05, 0.051, 0.5, 0.0) at alpha = 0.05: 3 of 5 have p <= alpha, got {rate}"
    assert len(calls) == 5 and all(c is calls[0] for c in calls) and isinstance(calls[0], np.random.Generator), (
        "simulate should get the SAME np.random.Generator (as_rng(rng)) in every experiment")


# ---------------------------------------------------------------- stage 4: multiple comparisons


def test_stage4_hand_example(nb):
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.20])
    bonf = np.asarray(nb.bonferroni(p, 0.05))
    assert bonf.dtype == bool and bonf.tolist() == [True, False, False, True, False], (
        f"Bonferroni at 0.05 with m = 5 rejects p <= 0.01: [True, False, False, True, False], got {bonf.tolist()}")
    bh = np.asarray(nb.benjamini_hochberg(p, 0.05))
    assert bh.dtype == bool and bh.tolist() == [True, True, True, True, False], (
        "BH: sorted p = (0.005, 0.01, 0.03, 0.04, 0.2) against k * 0.05 / 5 = (0.01, 0.02, 0.03, 0.04, 0.05): the "
        f"largest k with p_(k) <= threshold is 4, so reject the 4 smallest, in ORIGINAL order; got {bh.tolist()}")


def test_stage4_bonferroni_threshold_and_edges(nb):
    got = np.asarray(nb.bonferroni(np.array([0.011, 0.5, 0.5, 0.5, 0.5]), 0.05))
    assert not got.any(), (
        f"with m = 5 the Bonferroni threshold is 0.05 / 5 = 0.01, so p = 0.011 is NOT rejected; got {got.tolist()}. "
        "divide alpha by the number of p-values, m.")
    for fn in ("bonferroni", "benjamini_hochberg"):
        empty = np.asarray(getattr(nb, fn)(np.array([]), 0.05))
        assert empty.shape == (0,) and empty.dtype == bool, f"{fn} of no p-values is an empty bool mask"
        try:
            getattr(nb, fn)(np.array([0.01, np.nan, 0.2]), 0.05)
        except ValueError:
            continue
        raise AssertionError(f"{fn}: a NaN p-value is a bug upstream; raise ValueError instead of silently deciding")


def test_stage4_step_up_not_step_down(nb):
    p = np.array([0.04, 0.02, 0.03, 0.035])
    bh = np.asarray(nb.benjamini_hochberg(p, 0.05))
    assert bh.all(), (
        "sorted p = (0.02, 0.03, 0.035, 0.04) vs thresholds (0.0125, 0.025, 0.0375, 0.05): the smallest p fails "
        "its threshold, but the LARGEST k that passes is 4, so BH rejects all four. find the largest passing k, "
        "don't stop at the first failure.")
    assert not np.asarray(nb.benjamini_hochberg(np.array([0.3, 0.9, 0.06]), 0.05)).any(), "nothing passes: reject nothing"


def test_stage4_matches_scipy_and_controls_fdr(nb):
    rng = np.random.default_rng(14)
    fdps, bonf_fwer = [], []
    for _ in range(200):
        z = rng.normal(size=500)
        z[:50] += 3.0  # 50 real effects, 450 nulls
        p = stats.norm.sf(z)
        bh = np.asarray(nb.benjamini_hochberg(p, 0.1))
        assert np.array_equal(bh, stats.false_discovery_control(p) <= 0.1), "your BH disagrees with scipy's"
        fdps.append(bh[50:].sum() / max(1, bh.sum()))
        bonf_fwer.append(np.asarray(nb.bonferroni(p, 0.1))[50:].any())
    fdr = float(np.mean(fdps))
    assert fdr <= 0.1 * 450 / 500 + 0.02, f"BH at q = 0.1 should keep the false discovery rate near 0.09; got {fdr:.3f}"
    assert np.mean(bonf_fwer) <= 0.1 + 5 * math.sqrt(0.09 / 200), "Bonferroni should keep P(any false rejection) <= 0.1"
