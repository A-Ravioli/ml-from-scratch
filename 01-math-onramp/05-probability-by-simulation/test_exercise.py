"""Checks for exercise.py (probability by simulation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Random tests use
fixed seeds, and tolerances of several standard errors, so a correct implementation passes
(essentially) always and a wrong one fails.
"""
import numpy as np


def _uniform(n, rng):
    return rng.random(n)


# ---------------------------------------------------------------- stage 1: counting and dice


def test_stage1_two_dice_by_counting(nb):
    sums, probs = (np.asarray(a) for a in nb.dice_sum_pmf(2))
    assert np.array_equal(sums, np.arange(2, 13)), f"two dice can sum to 2, 3, ..., 12; got sums {sums}"
    want = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36
    assert np.allclose(probs, want), (
        f"P(sum = 2, ..., 12) for two dice should be [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1] / 36, got "
        f"{np.round(probs * 36, 3)} / 36. count every ordered pair: (1, 2) and (2, 1) are different outcomes.")


def test_stage1_one_and_three_dice(nb):
    sums, probs = (np.asarray(a) for a in nb.dice_sum_pmf(1))
    assert np.array_equal(sums, np.arange(1, 7)) and np.allclose(probs, 1 / 6), (
        f"one die: sums 1..6, each with probability 1/6; got {sums}, {probs}")
    sums, probs = (np.asarray(a) for a in nb.dice_sum_pmf(3))
    assert len(sums) == 16 and sums[0] == 3 and sums[-1] == 18, f"three dice sum to 3, ..., 18 (16 values), got {sums}"
    assert np.isclose(probs.sum(), 1.0), f"probabilities must add up to 1, got {probs.sum()}"
    assert np.isclose(probs[sums == 10][0], 27 / 216), (
        f"27 of the 216 outcomes of three dice sum to 10, so P(10) = 0.125; got {probs[sums == 10][0]}")


def test_stage1_roll_dice_shape_and_faces(nb):
    rolls = np.asarray(nb.roll_dice(60_000, 2, rng=0))
    assert rolls.shape == (60_000, 2), f"roll_dice(60000, 2) should have shape (60000, 2), got {rolls.shape}"
    faces, counts = np.unique(rolls, return_counts=True)
    assert np.array_equal(faces, np.arange(1, 7)), (
        f"the faces should be exactly 1..6, got {faces}. "
        + ("6 never comes up: rng.integers(1, 6) excludes 6; use rng.integers(1, 7)." if 6 not in faces else ""))
    freq = counts / rolls.size
    assert np.allclose(freq, 1 / 6, atol=0.006), f"each face should come up about 1/6 = 0.1667 of the time, got {np.round(freq, 4)}"


def test_stage1_roll_dice_uses_rng(nb):
    a = np.asarray(nb.roll_dice(50, 3, rng=42))
    b = np.asarray(nb.roll_dice(50, 3, rng=42))
    c = np.asarray(nb.roll_dice(50, 3, rng=43))
    assert np.array_equal(a, b), "the same seed must give the same rolls: draw through np.random.default_rng(rng)"
    assert not np.array_equal(a, c), "different seeds should give different rolls"
    g = np.random.default_rng(7)
    first = np.asarray(nb.roll_dice(20, 1, rng=g))
    second = np.asarray(nb.roll_dice(20, 1, rng=g))
    assert not np.array_equal(first, second), (
        "passing the same Generator twice should continue its stream (different rolls), not restart it")


# ---------------------------------------------------------------- stage 2: frequencies and expectations


def test_stage2_empirical_pmf_worked_example(nb):
    got = np.asarray(nb.empirical_pmf([1, 2, 2, 3, 3, 3], [1, 2, 3, 4]))
    assert np.allclose(got, [1 / 6, 2 / 6, 3 / 6, 0]), (
        f"in [1, 2, 2, 3, 3, 3], the values 1, 2, 3, 4 appear 1, 2, 3, 0 times out of 6: expected "
        f"[0.167, 0.333, 0.5, 0], got {np.round(got, 3)}")


def test_stage2_frequencies_approach_probabilities(nb):
    rng = np.random.default_rng(1)
    s = rng.integers(1, 7, size=(200_000, 2)).sum(axis=1)
    got = np.asarray(nb.empirical_pmf(s, np.arange(2, 13)))
    want = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36
    assert np.abs(got - want).max() < 0.004, (
        f"with 200,000 throws, the frequencies of the sums should be within 0.004 of the exact probabilities; "
        f"worst gap {np.abs(got - want).max():.4f}")


def test_stage2_expectation_and_variance_of_a_die(nb):
    faces, p = np.arange(1, 7), np.full(6, 1 / 6)
    e = nb.expectation(faces, p)
    assert np.isclose(e, 3.5), f"a fair die's expected value is (1 + ... + 6) / 6 = 3.5, got {e}"
    v = nb.variance(faces, p)
    assert np.isclose(v, 35 / 12), (
        f"a fair die's variance is 35/12 = 2.9167, got {v}. "
        + ("15.17 is E[X^2]: subtract the mean before squaring." if np.isclose(v, 91 / 6) else ""))


def test_stage2_weighted_not_plain_average(nb):
    values, probs = np.array([0.0, 10.0]), np.array([0.9, 0.1])
    e = nb.expectation(values, probs)
    assert np.isclose(e, 1.0), (
        f"a 10% chance of 10 and a 90% chance of 0 has expected value 0.1 * 10 = 1, got {e}. "
        + ("5 is the plain average: weight each value by its probability." if np.isclose(e, 5.0) else ""))
    assert np.isclose(nb.variance(values, probs), 9.0), "its variance is 0.9 * 1^2 + 0.1 * 9^2 = 9"


# ---------------------------------------------------------------- stage 3: mean and standard error


def test_stage3_worked_example(nb):
    m, se = nb.mean_and_se([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(m, 2.5), f"the mean of [1, 2, 3, 4] is 2.5, got {m}"
    want = np.sqrt(5 / 3) / 2
    assert np.isclose(se, want), (
        f"s = sqrt(((1.5)^2 + (0.5)^2 + (0.5)^2 + (1.5)^2) / 3) = 1.291, so se = 1.291 / sqrt(4) = {want:.4f}; "
        f"got {se:.4f}. "
        + ("0.559 means you divided by n instead of n - 1 (use np.std(x, ddof=1))." if np.isclose(se, np.sqrt(1.25) / 2)
           else "1.291 means you returned s itself; divide by sqrt(n)." if np.isclose(se, np.sqrt(5 / 3)) else ""))


def test_stage3_se_shrinks_like_one_over_sqrt_n(nb):
    rng = np.random.default_rng(2)
    x = rng.normal(5.0, 2.0, size=40_000)
    m1, se1 = nb.mean_and_se(x[:400])
    m2, se2 = nb.mean_and_se(x)
    assert abs(se1 - 2.0 / np.sqrt(400)) < 0.02 and abs(se2 - 2.0 / np.sqrt(40_000)) < 0.001, (
        f"for draws with standard deviation 2, the standard error with n = 400 should be about 0.1 and with "
        f"n = 40,000 about 0.01; got {se1:.4f} and {se2:.4f}")
    assert abs(m2 - 5.0) < 4 * se2, f"the mean of 40,000 draws should be within a few se of 5, got {m2}"


# ---------------------------------------------------------------- stage 4: Monte Carlo


def test_stage4_estimate_matches_mean_and_se_of_the_draws(nb):
    est, se = nb.mc_estimate(lambda u: u**2, _uniform, 5000, rng=3)
    u = np.random.default_rng(3).random(5000)
    v = u**2
    assert np.isclose(est, v.mean()) and np.isclose(se, v.std(ddof=1) / np.sqrt(5000)), (
        f"with rng=3 and n=5000, mc_estimate should average f over sample(5000, np.random.default_rng(3)): "
        f"expected ({v.mean():.6f}, {v.std(ddof=1) / np.sqrt(5000):.6f}), got ({est:.6f}, {se:.6f}). "
        "apply f to the draws, then use your mean_and_se.")


def test_stage4_passes_n_and_a_generator(nb):
    seen = {}

    def sample(n, rng):
        seen["n"], seen["rng"] = n, rng
        return rng.random(n)

    g = np.random.default_rng(11)
    nb.mc_estimate(lambda u: u, sample, 123, rng=g)
    assert seen.get("n") == 123, f"mc_estimate(..., n=123) should call sample(123, rng), got n = {seen.get('n')}"
    assert seen.get("rng") is g, (
        "when you pass a Generator, sample must receive that same Generator (np.random.default_rng(g) returns g "
        "itself); got " + type(seen.get("rng")).__name__)
    nb.mc_estimate(lambda u: u, sample, 10)
    assert isinstance(seen["rng"], np.random.Generator), (
        f"with rng=None, sample should still get a np.random.Generator (default_rng(None)), got {type(seen['rng']).__name__}")


def test_stage4_true_value_within_error_bars(nb):
    # E[e^U] for U uniform on [0, 1] is e - 1
    est, se = nb.mc_estimate(np.exp, _uniform, 20_000, rng=4)
    assert abs(est - (np.e - 1)) < 4 * se, (
        f"E[e^U] = e - 1 = {np.e - 1:.5f}; your estimate {est:.5f} is more than 4 standard errors ({se:.5f}) away")
    assert 0.002 < se < 0.005, f"the standard error for n = 20,000 should be about 0.0035, got {se:.5f}"


def test_stage4_default_n_and_reproducible(nb):
    a = nb.mc_estimate(lambda u: u, _uniform, rng=5)
    b = nb.mc_estimate(lambda u: u, _uniform, rng=5)
    assert a == b, "the same seed must give exactly the same estimate"
    assert abs(a[1] - np.sqrt(1 / 12) / 100) < 1e-4, (
        f"with the default n = 10,000, the se for a uniform draw is about 0.0029, got {a[1]:.5f}")


# ---------------------------------------------------------------- stage 5: pi from darts


def test_stage5_pi_within_error_bars(nb):
    est, se = nb.estimate_pi(40_000, rng=6)
    assert abs(est - np.pi) < 4 * se, (
        f"your estimate {est:.4f} is more than 4 standard errors ({se:.4f}) from pi. "
        + ("about 0.785 means you forgot the factor 4." if abs(est - np.pi / 4) < 0.05 else
           "darts should be uniform on [-1, 1] x [-1, 1], and count those with x^2 + y^2 <= 1."))
    want_se = np.sqrt(np.pi * (4 - np.pi) / 40_000)
    assert abs(se - want_se) < 0.001, (
        f"each dart scores 4 or 0, so the standard error is about sqrt(pi (4 - pi) / n) = {want_se:.4f}; got {se:.4f}")


def test_stage5_reproducible_and_uses_square(nb):
    a = nb.estimate_pi(1000, rng=7)
    assert a == nb.estimate_pi(1000, rng=7), "the same seed must give the same estimate"
    assert a != nb.estimate_pi(1000, rng=9), "different seeds should give different estimates: pass rng on to mc_estimate"
    est, _ = nb.estimate_pi(200_000, rng=8)
    assert abs(est - np.pi) < 0.02, (
        f"with 200,000 darts the estimate should be within 0.02 of pi, got {est:.4f}. "
        + ("about 4.0 or 1.0: darts on [0, 1] x [0, 1] work too only if you count the quarter circle; "
           "use the square [-1, 1] x [-1, 1]." if abs(est - np.pi) > 0.5 else ""))
