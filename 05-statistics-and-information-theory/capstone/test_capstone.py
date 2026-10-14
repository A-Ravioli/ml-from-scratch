"""Checks for capstone.py (the model-comparison toolkit): functional tests plus behavioral thresholds.

All data here are synthetic: labels and the predictions of imaginary classifiers whose accuracies
and error correlations are set exactly, drawn with fixed seeds. Nothing is downloaded. Random
checks use tolerances of several Monte Carlo standard errors, or thresholds the reference clears
by a wide margin.
"""
import math

import numpy as np
from scipy import stats


def _pair(n, acc_a, acc_b, shared, seed):
    """labels plus two classifiers' predictions. with probability `shared` an example is 'hard' or
    'easy' for BOTH models together, which correlates their errors."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    u_common, u_a, u_b = rng.random(n), rng.random(n), rng.random(n)
    use_common = rng.random(n) < shared
    ra = np.where(use_common, u_common < acc_a, u_a < acc_a)
    rb = np.where(use_common, u_common < acc_b, u_b < acc_b)
    return y, np.where(ra, y, 1 - y), np.where(rb, y, 1 - y)


def _acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


# ---------------------------------------------------------------- stage 1: a metric with an interval


def test_stage1_accuracy(nb):
    assert nb.accuracy(np.array([1, 0, 1, 1]), np.array([1, 1, 1, 0])) == 0.5, "2 of 4 predictions match: accuracy 0.5"
    assert nb.accuracy(np.array(["a", "b"]), np.array(["a", "b"])) == 1.0, "labels can be any comparable values"


def test_stage1_metric_ci_width_and_center(nb):
    y, p, _ = _pair(2000, 0.8, 0.8, 0.0, seed=0)
    acc = _acc(y, p)
    est, lo, hi = nb.metric_ci(nb.accuracy, y, p, n_boot=2000, rng=1)
    assert abs(est - acc) < 1e-12, f"the estimate is the metric on the full test set, {acc:.4f}; got {est}"
    se = math.sqrt(acc * (1 - acc) / 2000)
    assert lo < acc < hi, f"the interval ({lo:.4f}, {hi:.4f}) should contain the observed accuracy {acc:.4f}"
    assert abs((hi - lo) - 2 * 1.96 * se) < 0.15 * 2 * 1.96 * se, (
        f"for 2000 examples at accuracy {acc:.3f}, a 95% interval is about ±1.96 · sqrt(p(1-p)/n) wide: width "
        f"{2 * 1.96 * se:.4f}; got {hi - lo:.4f}. resample EXAMPLES: pass (y_true, y_out) as a tuple to bootstrap_ci.")
    _, lo90, hi90 = nb.metric_ci(nb.accuracy, y, p, alpha=0.1, n_boot=2000, rng=1)
    assert hi90 - lo90 < hi - lo, "alpha = 0.1 gives a 90% interval, narrower than the 95% one"


def test_stage1_metric_ci_coverage(nb):
    hits, reps = 0, 150
    rng = np.random.default_rng(2)
    for _ in range(reps):
        y = rng.integers(0, 2, 300)
        p = np.where(rng.random(300) < 0.7, y, 1 - y)
        _, lo, hi = nb.metric_ci(nb.accuracy, y, p, n_boot=400, method="percentile", rng=rng)
        hits += lo <= 0.7 <= hi
    assert hits / reps >= 0.88, (
        f"95% intervals for an accuracy of 0.7 on 300 examples should contain 0.7 about 95% of the time; yours did "
        f"{hits / reps:.2f} of the time")


def test_stage1_any_metric(nb):
    y = np.random.default_rng(3).integers(0, 2, 500)
    p = y.copy()
    p[:50] = 1 - p[:50]
    est, lo, hi = nb.metric_ci(lambda t, q: float(np.mean((t == 1) & (q == 1)) / np.mean(t == 1)), y, p, n_boot=500,
                               method="percentile", rng=4)
    want = np.mean((y == 1) & (p == 1)) / np.mean(y == 1)
    assert abs(est - want) < 1e-12 and lo <= est <= hi, "metric_ci should work for any metric(y_true, y_out), e.g. recall"


# ---------------------------------------------------------------- stage 2: paired differences


def test_stage2_identical_models(nb):
    y, p, _ = _pair(500, 0.75, 0.75, 0.0, seed=5)
    diff, lo, hi = nb.paired_difference_ci(nb.accuracy, y, p, p.copy(), n_boot=300, method="percentile", rng=6)
    assert diff == 0.0 and lo == 0.0 and hi == 0.0, (
        f"two models with identical predictions differ by exactly 0 on every resample, got ({diff}, {lo}, {hi}). "
        "resample the examples jointly: one index set for y_true, out_a and out_b.")


def test_stage2_sign_and_pairing(nb):
    y, a, b = _pair(3000, 0.86, 0.80, 0.9, seed=7)
    diff, lo, hi = nb.paired_difference_ci(nb.accuracy, y, a, b, n_boot=2000, rng=8)
    assert abs(diff - (_acc(y, a) - _acc(y, b))) < 1e-12, "diff = metric(a) - metric(b), in that order"
    assert lo < diff < hi and lo > 0, (
        f"model a is truly 6 points better and the errors are strongly correlated: the paired interval should exclude 0. "
        f"got ({lo:.4f}, {hi:.4f})")
    ra, rb = (a == y).astype(float), (b == y).astype(float)
    unpaired = 2 * 1.96 * math.sqrt(ra.var() / 3000 + rb.var() / 3000)
    paired = 2 * 1.96 * (ra - rb).std() / math.sqrt(3000)
    assert abs((hi - lo) - paired) < 0.15 * paired and hi - lo < 0.6 * unpaired, (
        f"the paired interval should be about {paired:.4f} wide (from the per-example differences), much narrower than "
        f"the {unpaired:.4f} an unpaired analysis gives; got {hi - lo:.4f}. did you resample a and b independently?")


def test_stage2_covers_the_truth(nb):
    hits = 0
    rng = np.random.default_rng(9)
    for i in range(100):
        y, a, b = _pair(400, 0.78, 0.74, 0.7, seed=100 + i)
        _, lo, hi = nb.paired_difference_ci(nb.accuracy, y, a, b, n_boot=400, method="percentile", rng=rng)
        hits += lo <= 0.04 <= hi
    assert hits >= 85, f"95% paired intervals should contain the true difference 0.04 about 95 times in 100; got {hits}"


# ---------------------------------------------------------------- stage 3: tests


def test_stage3_mcnemar_hand_example(nb):
    y = np.zeros(20, dtype=int)
    a, b = y.copy(), y.copy()
    b[:10] = 1      # a right, b wrong: 10
    a[10:12] = 1    # a wrong, b right: 2
    bb, cc, p = nb.mcnemar_test(y, a, b)
    want = 2 * (1 + 12 + 66) / 2**12  # 2 P(Bin(12, 1/2) <= 2) = 0.0386
    assert (bb, cc) == (10, 2), f"10 examples have a right and b wrong, 2 the reverse; got b = {bb}, c = {cc}"
    assert abs(p - want) < 1e-12, (
        f"exact McNemar: p = 2 P(Binomial(12, 1/2) <= 2) = {want:.4f}, got {p}. only the 12 discordant examples matter.")
    assert nb.mcnemar_test(y, a, a)[2] == 1.0, "no discordant examples: p = 1"


def test_stage3_mcnemar_matches_scipy(nb):
    for seed, (pa, pb) in enumerate(((0.8, 0.8), (0.85, 0.78), (0.6, 0.62))):
        y, a, b = _pair(600, pa, pb, 0.3, seed=20 + seed)
        bb, cc, p = nb.mcnemar_test(y, a, b)
        want = stats.binomtest(min(bb, cc), bb + cc, 0.5).pvalue
        assert abs(p - min(1.0, want)) < 1e-9, f"McNemar's exact p should equal the two-sided binomial test, {want:.6f}; got {p}"


def test_stage3_permutation_identical_and_separated(nb):
    y, p, _ = _pair(300, 0.7, 0.7, 0.0, seed=10)
    diff, pval = nb.paired_permutation_test(nb.accuracy, y, p, p.copy(), n_permutations=199, rng=11)
    assert diff == 0.0 and pval == 1.0, f"identical models: difference 0 and p = 1 exactly, got ({diff}, {pval})"
    wrong = 1 - y
    diff, pval = nb.paired_permutation_test(nb.accuracy, y, y.copy(), wrong, n_permutations=199, rng=12)
    assert diff == 1.0 and abs(pval - 1 / 200) < 1e-12, (
        f"a perfect model vs an always-wrong one: no swap is as extreme, so p = 1/(1 + 199); got ({diff}, {pval})")
    diff, pval = nb.paired_permutation_test(nb.accuracy, y, wrong, y.copy(), n_permutations=199, rng=12)
    assert diff == -1.0 and abs(pval - 1 / 200) < 1e-12, (
        f"the same comparison the other way round: difference -1 and, two-sided, p = 1/200 again; got ({diff}, {pval})")


def test_stage3_size_and_power(nb):
    rejections = 0
    rng = np.random.default_rng(13)
    for i in range(150):
        y, a, b = _pair(300, 0.75, 0.75, 0.5, seed=200 + i)
        rejections += nb.paired_permutation_test(nb.accuracy, y, a, b, n_permutations=199, rng=rng)[1] <= 0.05
    assert rejections / 150 <= 0.05 + 5 * math.sqrt(0.05 * 0.95 / 150), (
        f"two equally accurate models should be declared different at most ~5% of the time; got {rejections / 150:.3f}")
    detected = 0
    for i in range(60):
        y, a, b = _pair(2000, 0.80, 0.77, 0.8, seed=400 + i)
        detected += nb.paired_permutation_test(nb.accuracy, y, a, b, n_permutations=199, rng=rng)[1] <= 0.05
    assert detected / 60 >= 0.8, (
        f"a 3-point improvement on 2000 examples, with correlated errors, should be detected at least 80% of the time "
        f"with a PAIRED test (the reference: about 99%); you detected it {detected / 60:.2f} of the time")


# ---------------------------------------------------------------- stage 4: many models


def test_stage4_structure(nb):
    y, a, b = _pair(400, 0.8, 0.7, 0.5, seed=30)
    rows = nb.compare_models(nb.accuracy, y, {"base": b, "m1": a, "m2": b.copy()}, "base",
                             n_boot=200, n_permutations=99, rng=0)
    assert [r["model"] for r in rows] == ["m1", "m2"], "one row per non-baseline model, in the dict's order"
    assert set(rows[0]) >= {"model", "score", "diff", "lo", "hi", "p", "reject"}, f"missing keys in {sorted(rows[0])}"
    r1, r2 = rows
    assert abs(r1["score"] - _acc(y, a)) < 1e-12 and abs(r1["diff"] - (_acc(y, a) - _acc(y, b))) < 1e-12, (
        "score is the model's metric and diff = score - baseline score")
    assert r2["diff"] == 0.0 and r2["p"] == 1.0 and r2["reject"] is False, "a copy of the baseline: diff 0, p 1, not rejected"
    for bad in ({"baseline": "nope"}, {"correction": "holm"}):
        try:
            nb.compare_models(nb.accuracy, y, {"base": b, "m1": a}, bad.get("baseline", "base"),
                              correction=bad.get("correction", "bh"), n_boot=10, n_permutations=9, rng=0)
        except ValueError:
            continue
        raise AssertionError(f"compare_models with {bad} should raise ValueError")


def test_stage4_corrections(nb):
    rng = np.random.default_rng(40)
    n = 1500
    y = rng.integers(0, 2, n)

    def model(acc):  # predictions that are right on each example independently with probability acc
        return np.where(rng.random(n) < acc, y, 1 - y)

    outputs = {"base": model(0.75)}
    for k in range(8):  # 8 models exactly as good as the baseline
        outputs[f"same{k}"] = model(0.75)
    for k in range(3):  # 3 models somewhat better: with 11 comparisons, BH and Bonferroni disagree about them
        outputs[f"good{k}"] = model(0.82)
    outputs["better"] = model(0.85)
    rows = {c: nb.compare_models(nb.accuracy, y, outputs, "base", correction=c, n_boot=200, n_permutations=199, rng=44)
            for c in ("none", "bonferroni", "bh")}
    ps = np.array([r["p"] for r in rows["bh"]])
    from mlfs.stats import benjamini_hochberg as _bh, bonferroni as _bonf  # the reference implementations
    assert [r["reject"] for r in rows["bonferroni"]] == list(_bonf(np.array([r["p"] for r in rows["bonferroni"]]), 0.05)), (
        "correction='bonferroni' should apply bonferroni to all the p-values together")
    assert [r["reject"] for r in rows["bh"]] == list(_bh(ps, 0.05)), "correction='bh' should apply benjamini_hochberg"
    assert sum(r["reject"] for r in rows["bh"]) > sum(r["reject"] for r in rows["bonferroni"]), (
        "here BH should reject more hypotheses than Bonferroni (with 199 permutations no p-value is below 0.005, "
        "and Bonferroni needs p <= 0.05 / 12)")
    assert [r["reject"] for r in rows["none"]] == [p <= 0.05 for p in (r["p"] for r in rows["none"])], (
        "correction='none' rejects p <= alpha")
    assert rows["bh"][-1]["reject"], "the clearly better model must be detected"
    assert rows["bh"][-1]["lo"] > 0, "its paired interval should exclude 0"


def test_stage4_threads_the_generator_and_alpha_boundary(nb):
    rng = np.random.default_rng(50)
    n = 400
    y = rng.integers(0, 2, n)

    def model(acc):
        return np.where(rng.random(n) < acc, y, 1 - y)

    outputs = {"base": model(0.75), "m1": model(0.78), "m2": model(0.77), "m3": model(0.79)}
    a = nb.compare_models(nb.accuracy, y, outputs, "base", n_boot=50, n_permutations=499, rng=1)
    b = nb.compare_models(nb.accuracy, y, outputs, "base", n_boot=50, n_permutations=499, rng=2)
    assert any(ra["p"] != rb["p"] for ra, rb in zip(a, b)), (
        "seeds 1 and 2 gave identical permutation p-values for every model: pass the shared generator "
        "(as_rng(rng)) to paired_permutation_test too, not a fixed seed")
    better = {"base": outputs["base"], "perfect": y.copy()}
    rows = nb.compare_models(nb.accuracy, y, better, "base", correction="none", n_boot=50, n_permutations=19, rng=3)
    assert abs(rows[0]["p"] - 0.05) < 1e-12 and rows[0]["reject"], (
        f"with 19 permutations a perfect model gets p = 1/20 = 0.05 exactly, and correction='none' rejects p <= alpha "
        f"(not <): got p = {rows[0]['p']}, reject = {rows[0]['reject']}")
