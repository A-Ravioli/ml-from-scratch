"""Checks for capstone.py: the fair-comparison toolkit every elective report uses. hand examples,
oracles (SciPy, your mlfs.stats, brute-force definitions) and calibration checks that simulate
many experiments with no true effect."""
import itertools

import numpy as np
import pytest
from scipy import stats as sps

from mlfs.stats import bonferroni, bootstrap_ci, permutation_test


# ---------------------------------------------------------------- stage 1: pairing


def test_stage1_pairs_by_seed_not_by_order(nb):
    method = {3: 0.9, 0: 0.5, 7: 0.4}
    baseline = {0: 0.1, 7: 0.3, 3: 0.2}
    seeds, d = nb.paired_differences(method, baseline)
    assert list(seeds) == [0, 3, 7], f"seeds should be the shared seeds in sorted order [0, 3, 7], got {list(seeds)}"
    assert np.allclose(d, [0.4, 0.7, 0.1]), (
        f"d should be method - baseline seed by seed: [0.5-0.1, 0.9-0.2, 0.4-0.3] = [0.4, 0.7, 0.1], got {d}. "
        "if you zipped the dicts' values, you paired seed 3 with seed 0: dict order isn't seed order."
    )


def test_stage1_sequences_are_indexed_by_seed(nb):
    seeds, d = nb.paired_differences([0.8, 0.6, 0.9], np.array([0.5, 0.5, 0.5]))
    assert list(seeds) == [0, 1, 2], f"for sequences, the seeds are the indices 0..n-1, got {list(seeds)}"
    d = np.asarray(d)
    assert d.dtype.kind == "f" and np.allclose(d, [0.3, 0.1, 0.4]), f"expected d = [0.3, 0.1, 0.4], got {d}"


@pytest.mark.parametrize("method, baseline, why", [
    ({0: 1.0, 1: 2.0}, {0: 1.0, 2: 2.0}, "seed sets differ"),
    ({0: 1.0, 1: 2.0}, {0: 1.0, 1: 2.0, 2: 3.0}, "the baseline has an extra seed"),
    ([1.0, 2.0, 3.0], [1.0, 2.0], "the lengths differ"),
    ({0: 1.0, 1: 2.0}, [1.0, 2.0], "one is a dict and one a sequence"),
    ([1.0, np.nan], [1.0, 2.0], "a score is nan (a crashed run)"),
    ([], [], "there are no scores"),
])
def test_stage1_refuses_what_it_cant_pair(nb, method, baseline, why):
    with pytest.raises(ValueError):
        nb.paired_differences(method, baseline)
        pytest.fail(f"paired_differences should raise ValueError when {why}: silently dropping or truncating "
                    "seeds is how an unpaired comparison sneaks into a report")


# ---------------------------------------------------------------- stage 2: intervals


def test_stage2_t_interval_by_hand(nb):
    lo, hi = nb.paired_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean 3, s = sqrt(2.5), t_{0.975, 4} = 2.7764: half-width 2.7764 * 1.5811 / sqrt(5) = 1.9632
    assert np.isclose(lo, 1.0368, atol=1e-3) and np.isclose(hi, 4.9632, atol=1e-3), (
        f"the 95% t interval of [1, 2, 3, 4, 5] is (1.037, 4.963), got ({lo:.4f}, {hi:.4f}). "
        "(1.614, 4.386) means you used 1.96 instead of the t quantile with n - 1 = 4 degrees of freedom; "
        "(1.244, 4.756) means you used the ddof=0 standard deviation."
    )


@pytest.mark.parametrize("n, alpha", [(3, 0.05), (8, 0.05), (12, 0.1), (30, 0.01)])
def test_stage2_t_interval_matches_scipy(nb, n, alpha):
    d = np.random.default_rng(n).normal(0.2, 0.5, size=n)
    lo, hi = nb.paired_ci(d, alpha=alpha)
    want = sps.t.interval(1 - alpha, n - 1, loc=d.mean(), scale=sps.sem(d))
    assert np.allclose((lo, hi), want, atol=1e-9), f"n={n}, alpha={alpha}: expected {want}, got ({lo}, {hi})"


def test_stage2_bootstrap_delegates_to_your_stats_module(nb):
    d = np.random.default_rng(1).normal(size=9)
    got = nb.paired_ci(d, method="bootstrap", n_boot=2000, alpha=0.1, rng=3)
    want = bootstrap_ci(d, np.mean, n_boot=2000, alpha=0.1, rng=3)
    assert np.allclose(got, want), (
        f"method='bootstrap' should return mlfs.stats.bootstrap_ci(d, np.mean, n_boot=..., alpha=..., rng=rng) = "
        f"{want}, got {got}. use your chapter-05 bootstrap; don't write a second one."
    )


def test_stage2_edge_cases(nb):
    lo, hi = nb.paired_ci([0.25, 0.25, 0.25])
    assert np.isclose(lo, 0.25) and np.isclose(hi, 0.25), (
        f"identical differences give a zero-width interval (0.25, 0.25), got ({lo}, {hi})")
    with pytest.raises(ValueError):
        nb.paired_ci([1.0])
        pytest.fail("one seed has no spread: paired_ci should raise ValueError")
    with pytest.raises(ValueError):
        nb.paired_ci([1.0, 2.0], method="z")
        pytest.fail("an unknown method should raise ValueError")


def test_stage2_t_interval_covers_95_percent_with_five_seeds(nb):
    rng = np.random.default_rng(2)
    R, hits = 3000, 0
    for _ in range(R):
        d = 0.3 + rng.normal(size=5)
        lo, hi = nb.paired_ci(d)
        hits += lo <= 0.3 <= hi
    cover = hits / R
    assert 0.935 <= cover <= 0.965, (
        f"over {R} simulated 5-seed experiments your 95% interval covered the true mean {cover:.3f} of the time "
        "(should be 0.95 ± 0.012). about 0.88 means a normal quantile (1.96) instead of t with 4 dof."
    )


# ---------------------------------------------------------------- stage 3: sign-flip test


def test_stage3_exact_by_hand(nb):
    d = [1.0, 2.0, 3.0]
    # the 8 sign patterns give sums 6, 4, 2, 0, 0, -2, -4, -6
    for alt, want in [("two-sided", 2 / 8), ("greater", 1 / 8), ("less", 8 / 8)]:
        got = nb.sign_flip_test(d, alternative=alt)
        assert np.isclose(got, want), (
            f"exact sign-flip p for d = [1, 2, 3], alternative={alt!r}: {want}, got {got}. enumerate all 2**3 "
            "sign patterns, count those at least as extreme as the observed one (itself included), divide by 8. "
            "0.333 means you used the Monte Carlo formula (1 + k) / (1 + B), which isn't exact."
        )


def test_stage3_five_seeds_cannot_reach_005(nb):
    d = [0.37, 0.41, 0.35, 0.39, 0.40]  # a huge, perfectly consistent effect
    got = nb.sign_flip_test(d)
    assert np.isclose(got, 2 / 32), (
        f"with 5 seeds, all positive, the exact two-sided p is 2/32 = 0.0625 (the observed pattern and its mirror), "
        f"got {got}")
    assert np.isclose(nb.min_p_value(5), 0.0625) and np.isclose(nb.min_p_value(5, "greater"), 1 / 32), (
        f"min_p_value(5) should be 2/2**5 = 0.0625 and min_p_value(5, 'greater') 1/32, got "
        f"{nb.min_p_value(5)} and {nb.min_p_value(5, 'greater')}")
    assert np.isclose(nb.min_p_value(20), 2 / 2**20), "min_p_value(20) should be 2 / 2**20"


def test_stage3_ties_count_as_extreme(nb):
    # d in tenths; flipping 0.1, 0.2 and -0.3 gives exactly the observed sum 0.3, but floats disagree slightly
    tenths = np.array([3, 1, 2, -3, 4, -1])
    d = tenths / 10
    sums = np.array([np.dot(s, tenths) for s in itertools.product([1, -1], repeat=6)])  # exact integers
    t = tenths.sum()
    for alt, want in [("two-sided", np.mean(np.abs(sums) >= abs(t))), ("greater", np.mean(sums >= t)),
                      ("less", np.mean(sums <= t))]:
        got = nb.sign_flip_test(d, alternative=alt)
        assert np.isclose(got, want), (
            f"alternative={alt!r}: patterns that tie the observed mean must count as extreme; exact p = {want:.4f}, "
            f"got {got:.4f}. compare with a small tolerance (e.g. null >= t - 1e-12 * max(1, |t|)), since "
            "0.3 - 0.1 - 0.2 + 0.3 isn't exactly 0.3 in floating point."
        )


def test_stage3_is_exact_hence_deterministic_for_small_n(nb):
    d = np.random.default_rng(4).normal(0.3, 1.0, size=9)
    p1 = nb.sign_flip_test(d, rng=0)
    p2 = nb.sign_flip_test(d, rng=12345)
    assert p1 == p2, (f"with 2**9 = 512 <= n_permutations patterns the test is exact: the rng shouldn't matter, "
                      f"got {p1} and {p2}")
    assert np.isclose(p1 * 512, round(p1 * 512)), f"an exact p over 512 patterns is a multiple of 1/512, got {p1}"


def test_stage3_large_n_uses_your_permutation_test(nb):
    d = np.random.default_rng(5).normal(0.1, 1.0, size=16)  # 2**16 > 2000 patterns: Monte Carlo
    got = nb.sign_flip_test(d, n_permutations=2000, rng=7)
    want = permutation_test(d, np.zeros(16), n_permutations=2000, paired=True, rng=7)[1]
    assert np.isclose(got, want), (
        f"when 2**n > n_permutations, return mlfs.stats.permutation_test(d, zeros, paired=True, "
        f"n_permutations=..., rng=rng)'s p-value ({want}), got {got}")
    big = nb.sign_flip_test(np.random.default_rng(6).normal(2.0, 1.0, size=20), alternative="greater",
                            n_permutations=2000, rng=0)
    assert big < 0.01, f"20 differences with mean ~2 and sd 1 are overwhelming evidence ('greater'), got p = {big}"


def test_stage3_size_under_the_null(nb):
    rng = np.random.default_rng(8)
    R = 3000
    rejections = sum(nb.sign_flip_test(rng.standard_t(3, size=8)) <= 0.05 for _ in range(R))
    rate = rejections / R
    # exact size at n = 8: P(p <= 0.05) = 12/256 = 0.047
    assert 0.03 <= rate <= 0.065, (
        f"with no true effect (symmetric heavy-tailed differences, 8 seeds), p <= 0.05 should happen about 4.7% of "
        f"the time, got {rate:.3f}. far above: your null isn't centred (flip signs of d, don't permute values); "
        "far below: you aren't counting the observed pattern."
    )


def test_stage3_power(nb):
    rng = np.random.default_rng(9)
    R = 400
    power = np.mean([nb.sign_flip_test(rng.normal(1.0, 1.0, size=10)) <= 0.05 for _ in range(R)])
    assert power > 0.7, f"a 1-sd effect over 10 seeds should be detected most of the time (~0.85), got {power:.2f}"


def test_stage3_bad_alternative(nb):
    with pytest.raises(ValueError):
        nb.sign_flip_test([1.0, 2.0], alternative="bigger")
        pytest.fail("an unknown alternative should raise ValueError")


# ---------------------------------------------------------------- stage 4: Holm


def _holm_oracle(p, alpha):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = [max(min(1.0, (m - j) * p[order[j]]) for j in range(i + 1)) for i in range(m)]
    adj = np.empty(m)
    adj[order] = adj_sorted
    reject = np.zeros(m, bool)
    for k, i in enumerate(order):  # step down: stop at the first failure
        if p[i] > alpha / (m - k):
            break
        reject[i] = True
    return reject, adj


def test_stage4_by_hand(nb):
    reject, adj = nb.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02]), (
        f"Holm-adjusted p for [0.01, 0.04, 0.03, 0.005]: sorted 0.005×4 = 0.02, 0.01×3 = 0.03, 0.03×2 = 0.06, "
        f"0.04×1 = 0.04 → raised to the running max 0.06; in input order [0.03, 0.06, 0.06, 0.02]. got {adj}. "
        "0.04 for the second entry means you forgot the running maximum."
    )
    assert list(np.asarray(reject, bool)) == [True, False, False, True], (
        f"at alpha = 0.05 Holm rejects the two smallest only, got {reject}")


def test_stage4_matches_definition(nb):
    rng = np.random.default_rng(10)
    for _ in range(50):
        m = int(rng.integers(1, 12))
        p = rng.uniform(0, 0.2, size=m) ** rng.uniform(0.5, 2)
        if rng.random() < 0.3:
            p[: m // 2] = p[0]  # ties
        reject, adj = nb.holm(p, alpha=0.05)
        want_r, want_a = _holm_oracle(p, 0.05)
        assert np.allclose(adj, want_a) and np.array_equal(np.asarray(reject, bool), want_r), (
            f"p = {np.round(p, 4)}: expected adjusted {np.round(want_a, 4)}, reject {want_r}; got "
            f"{np.round(adj, 4)}, {reject}")


def test_stage4_never_weaker_than_bonferroni(nb):
    rng = np.random.default_rng(11)
    for _ in range(100):
        p = rng.uniform(0, 0.05, size=6)
        reject, _ = nb.holm(p)
        bon = np.asarray(bonferroni(p, 0.05), bool)
        assert np.all(np.asarray(reject, bool)[bon]), (
            f"Holm must reject everything Bonferroni rejects: p = {np.round(p, 4)}, Bonferroni {bon}, yours {reject}")


def test_stage4_controls_familywise_error(nb):
    rng = np.random.default_rng(12)
    R = 4000
    fwer = np.mean([np.any(nb.holm(rng.uniform(size=5), alpha=0.05)[0]) for _ in range(R)])
    assert fwer <= 0.06, (
        f"with 5 true nulls, Holm should make any false rejection at most 5% of the time, got {fwer:.3f}. "
        "testing every p at alpha (no correction) gives about 23%.")
    assert fwer >= 0.035, f"FWER {fwer:.3f} is far below 0.05: check the thresholds alpha/m, alpha/(m-1), ..."


# ---------------------------------------------------------------- stage 5: budgets


def test_stage5_memory_cost_units(nb):
    gen = [(10 * 2 * 784, "float32")]
    cases = [(gen, "floats", 15_680), (gen, "bytes", 62_720), ([(156_800, "uint8")], "bytes", 156_800),
             ([(156_800, np.uint8)], "floats", 156_800), ([(10, "float64"), (3, "int16")], "bytes", 86),
             ([(15_680, "float32"), (23_520, "float32")], "floats", 39_200), ([], "bytes", 0)]
    for comps, unit, want in cases:
        got = nb.memory_cost(comps, unit)
        assert got == want, (
            f"memory_cost({comps}, {unit!r}) should be {want}, got {got}. 'floats' counts stored values whatever "
            "their dtype; 'bytes' multiplies each count by np.dtype(dtype).itemsize.")
    with pytest.raises(ValueError):
        nb.memory_cost(gen, "megabytes")
        pytest.fail("an unknown unit should raise ValueError")


def _continual_methods():
    return {
        "fine-tuning": {"memory": [], "scores": [0.196, 0.195, 0.196, 0.196, 0.196]},
        "gaussian": {"memory": [(15_680, "float32")], "scores": [0.791, 0.802, 0.765, 0.780, 0.760]},
        "reservoir50_f32": {"memory": [(39_200, "float32")], "scores": [0.561, 0.563, 0.583, 0.556, 0.564]},
        "reservoir200_u8": {"memory": [(156_800, "uint8")], "scores": [0.794, 0.811, 0.829, 0.777, 0.808]},
    }


def test_stage5_the_unit_decides_the_winner(nb):
    t_f = nb.budget_table(_continual_methods(), 39_200, unit="floats")
    t_b = nb.budget_table(_continual_methods(), 156_800, unit="bytes")
    assert t_f["winner"] == "gaussian", (
        f"budget 39,200 floats: the 200 uint8 images count 156,800 values and don't fit, so the Gaussian "
        f"generator (0.780) wins; got {t_f['winner']!r}")
    assert t_b["winner"] == "reservoir200_u8", (
        f"budget 156,800 bytes: 200 uint8 images take exactly 156,800 bytes and fit (cost <= budget), and their "
        f"0.804 beats the generator's 0.780; got {t_b['winner']!r}. check that a cost equal to the budget fits.")
    names = [r["name"] for r in t_f["rows"]]
    assert names == ["gaussian", "reservoir50_f32", "fine-tuning", "reservoir200_u8"], (
        f"rows should list fitting methods first, best mean first, then the rest: got {names}")
    rows = {r["name"]: r for r in t_b["rows"]}
    assert rows["gaussian"]["cost"] == 62_720 and rows["reservoir50_f32"]["cost"] == 156_800, (
        f"costs in bytes: gaussian 62,720, reservoir50_f32 156,800; got {rows['gaussian']['cost']}, "
        f"{rows['reservoir50_f32']['cost']}")
    g = rows["gaussian"]
    assert np.isclose(g["mean"], 0.7796) and np.isclose(g["sd"], np.std([0.791, 0.802, 0.765, 0.780, 0.760],
                                                                        ddof=1)) and g["n"] == 5 and g["fits"], (
        f"each row needs mean, sd (ddof=1), n and fits; got {g}")


def test_stage5_lower_is_better_and_nothing_fits(nb):
    methods = {"a": {"memory": [(100, "float32")], "scores": [3.0, 3.2]},
               "b": {"memory": [(10, "float32")], "scores": [2.0, 2.4]},
               "c": {"memory": [(10, "float64")], "scores": [1.0, 1.1]}}
    t = nb.budget_table(methods, 40, unit="bytes", higher_is_better=False)
    assert t["winner"] == "b" and [r["name"] for r in t["rows"]] == ["b", "c", "a"], (
        f"lower is better, budget 40 bytes: only b (40 bytes) fits; c (80 bytes) has the lowest error but doesn't. "
        f"expected winner 'b' and rows [b, c, a], got {t['winner']!r}, {[r['name'] for r in t['rows']]}")
    t = nb.budget_table(methods, 5, unit="floats")
    assert t["winner"] is None and not any(r["fits"] for r in t["rows"]), (
        f"no method fits in 5 floats: winner should be None, got {t['winner']!r}")


# ---------------------------------------------------------------- stage 6: portfolio entries


def test_stage6_entry_fields(nb):
    method = {2: 0.70, 0: 0.80, 1: 0.75, 3: 0.72}
    base = {0: 0.60, 1: 0.62, 2: 0.65, 3: 0.61}
    e = nb.portfolio_entry("blk", "q?", "M", method, "B", base, metric="acc", rng=0)
    d = np.array([0.20, 0.13, 0.05, 0.11])
    assert e["n"] == 4 and list(e["seeds"]) == [0, 1, 2, 3], f"n = 4 seeds [0, 1, 2, 3], got n={e['n']}, {e['seeds']}"
    assert np.isclose(e["diff"], d.mean()), f"diff = mean(method - baseline) = {d.mean():.4f}, got {e['diff']}"
    assert np.isclose(e["method_mean"], 0.7425) and np.isclose(e["baseline_mean"], 0.62), (
        f"method_mean 0.7425, baseline_mean 0.62, got {e['method_mean']}, {e['baseline_mean']}")
    assert np.allclose(e["ci"], nb.paired_ci(d)), f"ci should be paired_ci(d) (t, 95%), got {e['ci']}"
    assert np.isclose(e["p"], nb.sign_flip_test(d)) and np.isclose(e["p"], 2 / 16), (
        f"p should be sign_flip_test(d) = 2/16 (all four differences positive), got {e['p']}")
    assert np.isclose(e["min_p"], nb.min_p_value(4)), f"min_p should be min_p_value(4) = 0.125, got {e['min_p']}"
    assert e["verdict"] == "better", f"the interval {e['ci']} lies above 0 and higher is better: 'better', got {e['verdict']!r}"
    for key in ("block", "question", "metric", "method", "baseline", "higher_is_better", "ci_method", "alpha",
                "budget", "notes"):
        assert key in e, f"the entry is missing the key {key!r}"


@pytest.mark.parametrize("shift, hib, want", [
    (0.5, True, "better"), (-0.5, True, "worse"), (0.5, False, "worse"), (-0.5, False, "better"), (0.0, True,
                                                                                                 "inconclusive"),
])
def test_stage6_verdict(nb, shift, hib, want):
    rng = np.random.default_rng(13)
    base = rng.normal(size=12)
    noise = np.array([0.05, -0.05] * 6)
    e = nb.portfolio_entry("blk", "q", "M", base + shift + noise, "B", base, metric="m", higher_is_better=hib, rng=0)
    assert e["verdict"] == want, (
        f"method = baseline + ({shift:+}) with {'higher' if hib else 'lower'} is better: verdict should be {want!r}, "
        f"got {e['verdict']!r} (ci {e['ci']}). the verdict reads the interval for method − baseline: entirely on "
        "the good side of 0 → better, entirely on the bad side → worse, else inconclusive.")


def test_stage6_refuses_unpaired_scores(nb):
    with pytest.raises(ValueError):
        nb.portfolio_entry("blk", "q", "M", {0: 1.0, 1: 2.0}, "B", {0: 1.0, 5: 2.0}, metric="m")
        pytest.fail("portfolio_entry should pass the scores through paired_differences, which refuses mismatched seeds")


def test_stage6_false_claims_under_the_null(nb):
    rng = np.random.default_rng(14)
    R, claims = 1000, 0
    for _ in range(R):
        base = rng.normal(size=10)
        method = base + rng.normal(scale=0.3, size=10)  # no true effect
        claims += nb.portfolio_entry("b", "q", "M", method, "B", base, metric="m", rng=0)["verdict"] != "inconclusive"
    rate = claims / R
    assert 0.025 <= rate <= 0.08, (
        f"with no true effect, a 95% interval should claim 'better' or 'worse' about 5% of the time, got {rate:.3f}")


def test_stage6_render(nb):
    method = {0: 0.30, 1: 0.32, 2: 0.29, 3: 0.40, 4: 0.31}  # seed 3 goes the other way: p = 4/32, min_p = 2/32
    base = {0: 0.40, 1: 0.41, 2: 0.42, 3: 0.39, 4: 0.40}
    e = nb.portfolio_entry("time series (05–08)", "does it forecast better?", "SSM", method, "seasonal naive", base,
                           metric="MASE", higher_is_better=False, budget="10k parameters each", notes="a caveat.",
                           rng=0)
    assert np.isclose(e["p"], 0.125) and e["verdict"] == "better", (
        f"for this entry p = 4/32 = 0.125 and the verdict is 'better', got {e['p']}, {e['verdict']!r}")
    text = nb.render_entry(e)
    lo, hi = e["ci"]
    must = ["### time series (05–08)", "does it forecast better?", "SSM", "seasonal naive (baseline)", "MASE",
            "n = 5", f"{e['diff']:+.3f}", f"95% CI [{lo:+.3f}, {hi:+.3f}]", f"{e['p']:.3g}", f"{e['min_p']:.3g}",
            "**verdict:** better", "lower is better", "**budget:** 10k parameters each", "a caveat."]
    missing = [s for s in must if s not in text]
    assert not missing, f"render_entry's markdown is missing {missing} (see brief.md for the format). got:\n{text}"
    e2 = nb.portfolio_entry("b", "q", "M", [1.0, 2.0, 3.0], "B", [0.0, 0.0, 0.5], metric="m", rng=0)
    text2 = nb.render_entry(e2)
    assert "**budget:**" not in text2, "with budget=None, leave the budget line out"
    assert "higher is better" in text2, "say which direction is better"
