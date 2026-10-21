"""Checks for 30 · 03 experimental rigor (mlfs.research).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. run_seeds for call order, default
seeds, both result shapes and its refusals; mean_ci against hand-computed t intervals, coverage and the bootstrap
oracle; compare_runs for pairing, the two-sample bootstrap's width, rng use and n_permutations; ablation_table
against an exact hand-built table; expected_best against the order-statistics formula and Monte Carlo.
"""
import numpy as np
import pytest
from scipy import stats as sps

from mlfs.stats import bootstrap_ci, permutation_test
from mlfs.utils import as_rng


# ---------------------------------------------------------------- stage 1: run_seeds


def test_stage1_calls_each_seed_once_in_order(nb):
    calls = []

    def fn(seed):
        calls.append(seed)
        return 10.0 + seed

    out = nb.run_seeds(fn, seeds=(7, 3, 11))
    assert calls == [7, 3, 11], (
        f"fn should be called once per seed, in the order given: expected [7, 3, 11], got {calls}. "
        "don't sort, deduplicate or re-run seeds."
    )
    assert isinstance(out, np.ndarray) and out.dtype.kind == "f" and out.shape == (3,), (
        f"scalar results should come back as a float ndarray of shape (n_seeds,), got {type(out).__name__} "
        f"{getattr(out, 'dtype', '')} {getattr(out, 'shape', '')}"
    )
    np.testing.assert_allclose(out, [17.0, 13.0, 21.0], err_msg="results must stay in seed order")


def test_stage1_default_seeds_are_zero_to_four(nb):
    calls = []
    out = nb.run_seeds(lambda s: calls.append(s) or s)
    assert calls == [0, 1, 2, 3, 4], f"the default seeds are (0, 1, 2, 3, 4), fn was called with {calls}"
    np.testing.assert_allclose(out, [0, 1, 2, 3, 4])


def test_stage1_accepts_numpy_scalars_and_any_iterable(nb):
    out = nb.run_seeds(lambda s: np.float32(s) / 2, seeds=range(3))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
    out = nb.run_seeds(lambda s: np.array(s * 2), seeds=iter([1, 2]))
    np.testing.assert_allclose(out, [2.0, 4.0], err_msg="0-d arrays are scalars; any iterable of seeds works")
    out = nb.run_seeds(lambda s: int(s), seeds=[4, 5])
    assert out.dtype.kind == "f", f"integer results should become floats, got dtype {out.dtype}"


def test_stage1_dict_results_are_collected_per_metric(nb):
    out = nb.run_seeds(lambda s: {"acc": 0.9 + s / 100, "loss": float(s)}, seeds=(2, 0, 1))
    assert isinstance(out, dict) and set(out) == {"acc", "loss"}, (
        f"dict results should give a dict metric -> per-seed values with the same keys, got {out!r}"
    )
    np.testing.assert_allclose(out["acc"], [0.92, 0.90, 0.91], err_msg="values must follow the seed order")
    np.testing.assert_allclose(out["loss"], [2.0, 0.0, 1.0])
    out = nb.run_seeds(lambda s: {"z": 1.0, "a": 2.0} if s == 0 else {"a": 2.0, "z": 1.0}, seeds=(0, 1))
    assert list(out) == ["z", "a"], f"keys come in the first result's order (not sorted), got {list(out)}"
    assert all(isinstance(v, np.ndarray) and v.dtype.kind == "f" for v in out.values()), (
        "each metric should be a float ndarray (n_seeds,)"
    )


def test_stage1_keeps_crashed_runs(nb):
    out = nb.run_seeds(lambda s: float("nan") if s == 1 else 1.0, seeds=(0, 1, 2))
    assert out.shape == (3,) and np.isnan(out[1]), (
        f"a nan result is a crashed run: keep it (shape (3,) with nan at index 1), don't drop it. got {out}"
    )


def test_stage1_refuses_inconsistent_results(nb):
    with pytest.raises(ValueError):
        nb.run_seeds(lambda s: {"acc": 1.0} if s == 0 else {"acc": 1.0, "loss": 2.0}, seeds=(0, 1))
    with pytest.raises(ValueError):
        nb.run_seeds(lambda s: {"acc": 1.0} if s == 0 else 1.0, seeds=(0, 1))
    with pytest.raises(ValueError):
        nb.run_seeds(lambda s: 1.0 if s == 0 else {"acc": 1.0}, seeds=(0, 1))
    with pytest.raises(ValueError):
        nb.run_seeds(lambda s: 1.0, seeds=())
    with pytest.raises(ValueError):
        nb.run_seeds(lambda s: np.ones(3), seeds=(0, 1))


# ---------------------------------------------------------------- stage 2: mean_ci


def _t_interval(v, alpha):
    v = np.asarray(v, dtype=float)
    half = sps.t.ppf(1 - alpha / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - half, v.mean() + half


def test_stage2_worked_example(nb):
    m, lo, hi = nb.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.isclose(m, 3.0), f"the mean of 1..5 is 3, got {m}"
    assert np.isclose(lo, 1.0368, atol=1e-4) and np.isclose(hi, 4.9632, atol=1e-4), (
        f"the 95% t interval for 1..5 is 3 ± 2.776 × 1.581 / √5 = [1.037, 4.963], got [{lo:.4f}, {hi:.4f}]. "
        "if you got [1.614, 4.386], you used 1.96 instead of the t quantile with n - 1 = 4 degrees of freedom; "
        "if [1.244, 4.756], the standard deviation with ddof=0."
    )
    assert all(isinstance(x, float) for x in (m, lo, hi)), "return plain floats"


def test_stage2_alpha_is_used(nb):
    rng = np.random.default_rng(0)
    v = rng.normal(size=7)
    for alpha in (0.01, 0.1, 0.32):
        got = nb.mean_ci(v, alpha=alpha)
        np.testing.assert_allclose(got, _t_interval(v, alpha), rtol=1e-9, atol=1e-12,
                                   err_msg=f"with alpha={alpha} the interval should use t.ppf(1 - alpha/2, n - 1)")


def test_stage2_t_interval_covers_at_five_seeds(nb):
    rng = np.random.default_rng(1)
    hits = 0
    n_sims = 2000
    for _ in range(n_sims):
        v = rng.normal(loc=0.7, scale=0.02, size=5)
        _, lo, hi = nb.mean_ci(v)
        hits += lo <= 0.7 <= hi
    cov = hits / n_sims
    assert 0.93 <= cov <= 0.97, (
        f"over {n_sims} simulated 5-seed experiments the 95% t interval should cover the true mean about 95% of "
        f"the time, got {cov:.3f}. a normal (1.96) interval covers about 88% at n = 5."
    )


def test_stage2_bootstrap_is_your_bootstrap_ci(nb):
    v = np.random.default_rng(2).exponential(size=12)
    for alpha in (0.05, 0.2):
        m, lo, hi = nb.mean_ci(v, alpha=alpha, method="bootstrap", rng=3)
        elo, ehi = bootstrap_ci(v, np.mean, alpha=alpha, rng=3)
        assert np.isclose(m, v.mean()), f"the point estimate is the sample mean {v.mean():.4f}, got {m}"
        assert np.isclose(lo, elo) and np.isclose(hi, ehi), (
            f"method='bootstrap' should return mlfs.stats.bootstrap_ci(values, np.mean, alpha=alpha, rng=rng): "
            f"expected [{elo:.4f}, {ehi:.4f}] at alpha={alpha}, got [{lo:.4f}, {hi:.4f}]. pass rng and alpha through."
        )


def test_stage2_refusals(nb):
    with pytest.raises(ValueError):
        nb.mean_ci([1.0])
    with pytest.raises(ValueError):
        nb.mean_ci([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        nb.mean_ci([1.0, np.inf, 2.0])
    with pytest.raises(ValueError):
        nb.mean_ci([1.0, 2.0], method="z")
    m, lo, hi = nb.mean_ci([0.5, 0.5, 0.5])
    assert m == lo == hi == 0.5, f"identical values give a zero-width interval, got ({m}, {lo}, {hi})"


# ---------------------------------------------------------------- stage 3: compare_runs


def _correlated_runs(seed=0, n=12, effect=0.01):
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 0.1, n)  # what the seed does to both methods
    a = 0.80 + effect + shared + rng.normal(0, 0.002, n)
    b = 0.80 + shared + rng.normal(0, 0.002, n)
    return a, b


def test_stage3_paired_matches_bootstrap_and_permutation_oracle(nb):
    a, b = _correlated_runs()
    got = nb.compare_runs(a, b, n_permutations=2000, rng=5)
    g = as_rng(5)
    ci = bootstrap_ci(a - b, np.mean, alpha=0.05, rng=g)
    _, p = permutation_test(a, b, n_permutations=2000, paired=True, rng=g)
    assert set(got) >= {"diff", "ci", "p_value", "n_a", "n_b", "paired"}, f"missing keys, got {sorted(got)}"
    assert np.isclose(got["diff"], np.mean(a - b)), f"paired diff is mean(a - b) = {np.mean(a - b):.5f}"
    assert np.allclose(got["ci"], ci), (
        f"paired ci should be bootstrap_ci(a - b, np.mean, alpha=alpha, rng=g) with g = as_rng(rng), drawn "
        f"first: expected {ci}, got {got['ci']}"
    )
    assert np.isclose(got["p_value"], p), (
        f"p_value should come from permutation_test(a, b, n_permutations=..., paired=True, rng=g), drawn from the "
        f"same generator after the interval: expected {p}, got {got['p_value']}"
    )
    assert got["n_a"] == got["n_b"] == len(a) and got["paired"] is True
    rev = nb.compare_runs(b, a, n_permutations=2000, rng=5)
    assert np.isclose(rev["diff"], -got["diff"]) and rev["ci"][1] < 0, (
        f"swapping a and b flips the sign of diff and of the interval: got diff {rev['diff']:.4f}, ci {rev['ci']}"
    )


def test_stage3_pairing_is_what_finds_the_effect(nb):
    a, b = _correlated_runs(seed=1)
    pa = nb.compare_runs(a, b, paired=True, n_permutations=2000, rng=0)
    un = nb.compare_runs(a, b, paired=False, n_permutations=2000, rng=0)
    assert pa["p_value"] < 0.01 and pa["ci"][0] > 0, (
        f"the seeds move both methods together; a paired comparison should find the +0.01 effect "
        f"(p < 0.01, interval above 0), got p = {pa['p_value']:.3f}, ci = {pa['ci']}"
    )
    w_pa, w_un = pa["ci"][1] - pa["ci"][0], un["ci"][1] - un["ci"][0]
    assert w_un > 4 * w_pa, (
        f"unpaired, the shared seed noise (sd 0.1) stays in the comparison, so its interval should be many times "
        f"wider than the paired one: widths {w_un:.4f} (unpaired) vs {w_pa:.4f} (paired). is paired=False ignored?"
    )
    assert un["paired"] is False


def test_stage3_unpaired_bootstrap_resamples_both_sides(nb):
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 100)
    b = rng.normal(0.0, 3.0, 150)
    got = nb.compare_runs(a, b, paired=False, n_permutations=200, rng=1)
    assert np.isclose(got["diff"], a.mean() - b.mean()), "unpaired diff is mean(a) - mean(b)"
    se = np.sqrt(a.var(ddof=1) / 100 + b.var(ddof=1) / 150)
    width = got["ci"][1] - got["ci"][0]
    assert 0.85 * 3.92 * se < width < 1.15 * 3.92 * se, (
        f"the unpaired interval should reflect the noise of BOTH means: width ≈ 3.92 × {se:.3f} = {3.92 * se:.3f}, "
        f"got {width:.3f}. if it's about {3.92 * a.std() / 10:.2f}, you only resampled a."
    )
    assert got["n_a"] == 100 and got["n_b"] == 150


def test_stage3_unpaired_interval_is_a_percentile_bootstrap(nb):
    a = np.array([0.0] * 7 + [10.0])  # one run in eight diverged upwards: a very skewed metric
    b = np.zeros(8)
    got = nb.compare_runs(a, b, paired=False, n_permutations=200, rng=3)
    lo, hi = got["ci"]
    assert np.isclose(got["diff"], 1.25)
    assert abs(lo) < 1e-12 and np.isclose(hi, 3.75), (
        f"the percentile bootstrap of mean(a*) - mean(b*) here takes values 0, 1.25, 2.5, ...: its 2.5% quantile is "
        f"0 and its 97.5% quantile 3.75, an interval far from symmetric around 1.25. got [{lo:.3f}, {hi:.3f}]. a "
        "normal or Welch interval (diff ± z·se) would be symmetric and dip below 0."
    )


def test_stage3_unpaired_p_value_is_a_permutation_test(nb):
    rng = np.random.default_rng(4)
    a, b = rng.normal(0.3, 1, 15), rng.normal(0.0, 1, 11)
    got = nb.compare_runs(a, b, paired=False, n_permutations=4000, rng=2)
    _, ref = permutation_test(a, b, n_permutations=20000, paired=False, rng=9)
    assert abs(got["p_value"] - ref) < 0.03, (
        f"unpaired p should be an (unpaired) permutation test of mean(a) - mean(b): about {ref:.3f}, "
        f"got {got['p_value']:.3f}"
    )


def test_stage3_alpha_rng_and_n_permutations_are_used(nb):
    a, b = _correlated_runs(seed=2, effect=0.004)
    wide = nb.compare_runs(a, b, alpha=0.05, n_permutations=500, rng=0)["ci"]
    narrow = nb.compare_runs(a, b, alpha=0.32, n_permutations=500, rng=0)["ci"]
    assert narrow[1] - narrow[0] < 0.75 * (wide[1] - wide[0]), (
        f"alpha=0.32 should give a much narrower interval than alpha=0.05, got {narrow} vs {wide}"
    )
    r1 = nb.compare_runs(a, b, n_permutations=500, rng=11)
    r2 = nb.compare_runs(a, b, n_permutations=500, rng=11)
    assert r1["ci"] == r2["ci"] and r1["p_value"] == r2["p_value"], "the same integer rng must give the same result"
    p = nb.compare_runs(a, b, n_permutations=9, rng=0)["p_value"]
    assert np.isclose(p * 10, round(p * 10)) and p >= 0.1, (
        f"with n_permutations=9 the permutation p-value is (1 + #extreme) / 10, a multiple of 0.1 that's at least "
        f"0.1; got {p}. pass n_permutations through to permutation_test."
    )


def test_stage3_refusals(nb):
    with pytest.raises(ValueError):
        nb.compare_runs([1.0, 2.0, 3.0], [1.0, 2.0], paired=True)
    with pytest.raises(ValueError):
        nb.compare_runs([1.0], [2.0], paired=False)
    with pytest.raises(ValueError):
        nb.compare_runs([1.0, np.nan], [2.0, 3.0])


# ---------------------------------------------------------------- stage 4: ablation_table


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _pm(v, alpha=0.05, fmt="{:.3f}"):
    m, lo, hi = _t_interval(v, alpha)
    return f"{fmt.format(m)} ± {fmt.format((hi - lo) / 2)}"


def test_stage4_worked_example(nb):
    results = {"full": [0.90, 0.92, 0.91], "no aug": [0.88, 0.90, 0.86]}
    got = nb.ablation_table(results, baseline="full")
    expected = "\n".join([
        "| config | mean ± 95% CI | n | Δ vs full ± 95% CI |",
        "|---|---|---|---|",
        "| **full** | **0.910 ± 0.025** | **3** | **(baseline)** |",
        "| no aug | 0.880 ± 0.050 | 3 | -0.030 ± 0.043 |",
    ])
    assert got.strip() == expected, (
        "the ablation table doesn't match the spec. expected:\n" + expected + "\n\ngot:\n" + got.strip() +
        "\n\n(t intervals with n - 1 dof; Δ is the PAIRED difference, row minus baseline, seed by seed; the best "
        "row's cells are each wrapped in **…**)"
    )


def test_stage4_no_baseline_and_lower_is_better(nb):
    results = {"a": [1.0, 1.2, 1.1], "b": [0.5, 0.7, 0.6], "c": [2.0, 2.0, 2.3]}
    got = nb.ablation_table(results, higher_is_better=False).strip().splitlines()
    assert got[0] == "| config | mean ± 95% CI | n |", f"without a baseline there's no Δ column, got {got[0]!r}"
    assert got[1] == "|---|---|---|"
    assert [line.split("|")[1].strip().strip("*") for line in got[2:]] == ["a", "b", "c"], (
        "rows must follow the order of results"
    )
    assert got[3] == _row([f"**{c}**" for c in ["b", _pm(results["b"]), "3"]]), (
        f"with higher_is_better=False the LOWEST mean (b) is the best row and is bold, got {got[3]!r}"
    )
    assert "**" not in got[2] and "**" not in got[4], "only the best row is bold"


def test_stage4_difference_is_paired_by_seed(nb):
    base = np.array([0.70, 0.80, 0.90, 0.75])
    results = {"base": base, "plus": base + 0.01}
    got = nb.ablation_table(results, baseline="base").strip().splitlines()
    assert got[3].endswith("| **+0.010 ± 0.000** |"), (
        f"every seed improves by exactly 0.010, so the paired difference is +0.010 ± 0.000 (and positive "
        f"differences carry a '+'), got {got[3]!r}. a difference of means with an unpaired interval would be "
        "± 0.1 or so."
    )


def test_stage4_alpha_fmt_and_ties(nb):
    results = {"x": [0.25, 0.75, 0.5, 0.5], "y": [0.5, 0.5, 0.25, 0.75]}
    got = nb.ablation_table(results, baseline="x", alpha=0.1, fmt="{:.1%}").strip().splitlines()
    assert got[0] == "| config | mean ± 90% CI | n | Δ vs x ± 90% CI |", (
        f"the header shows the confidence level 100(1 - alpha)%, got {got[0]!r}"
    )
    assert got[2] == _row([f"**{c}**" for c in ["x", _pm(results["x"], 0.1, "{:.1%}"), "4", "(baseline)"]]), (
        f"fmt formats every number and alpha sets the interval; tied best means are all bold. got {got[2]!r}"
    )
    d = np.array(results["y"]) - np.array(results["x"])
    half = sps.t.ppf(0.95, 3) * d.std(ddof=1) / 2
    assert got[3] == _row([f"**{c}**" for c in ["y", _pm(results["y"], 0.1, "{:.1%}"), "4",
                                                f"+0.0% ± {half:.1%}"]]), (
        f"y ties x for the best mean, so it's bold too, and its Δ interval also uses alpha=0.1 "
        f"(half-width {half:.1%}): got {got[3]!r}"
    )


def test_stage4_refusals(nb):
    with pytest.raises(ValueError):
        nb.ablation_table({"a": [1.0, 2.0]}, baseline="b")
    with pytest.raises(ValueError):
        nb.ablation_table({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]}, baseline="a")
    with pytest.raises(ValueError):
        nb.ablation_table({"a": [1.0]})


# ---------------------------------------------------------------- stage 5: expected_best


def test_stage5_worked_example(nb):
    got = nb.expected_best([1.0, 2.0, 3.0], 2)
    assert np.isclose(got, 22 / 9), (
        f"best of 2 draws with replacement from (1, 2, 3): 1·1/9 + 2·3/9 + 3·5/9 = 22/9 = 2.444, got {got}"
    )
    got = nb.expected_best([3.0, 1.0, 2.0], 3)
    assert np.isclose(got, 72 / 27), (
        f"best of 3 draws WITH replacement from (1, 2, 3) is 72/27 = 2.667, got {got}. 3.0 means you drew "
        "without replacement; the formula is order-free, so shuffled scores must give the same answer."
    )
    assert np.isclose(nb.expected_best([1.0, 2.0, 3.0], 2, higher_is_better=False), 14 / 9), (
        "with higher_is_better=False the best is the minimum: 3·1/9 + 2·3/9 + 1·5/9 = 14/9"
    )


def test_stage5_limits_and_monotonicity(nb):
    v = np.random.default_rng(0).normal(size=30)
    assert np.isclose(nb.expected_best(v, 1), v.mean()), "the best of 1 trial is on average the mean trial"
    ks = np.arange(1, 61)
    curve = np.asarray(nb.expected_best(v, ks))
    assert curve.shape == ks.shape, f"an array of k gives an array of the same shape, got {curve.shape}"
    assert np.all(np.diff(curve) >= -1e-12), "more trials can only raise the expected best"
    assert curve[-1] < v.max() and nb.expected_best(v, 10_000) > v.max() - 1e-6, (
        "it approaches the observed maximum only as k grows"
    )
    low = np.asarray(nb.expected_best(v, ks, higher_is_better=False))
    assert np.all(np.diff(low) <= 1e-12) and np.isclose(low[0], v.mean())


def test_stage5_matches_monte_carlo(nb):
    rng = np.random.default_rng(1)
    v = rng.gamma(2.0, size=12)
    draws = rng.choice(v, size=(200_000, 4), replace=True).max(1)
    got = nb.expected_best(v, 4)
    assert abs(got - draws.mean()) < 4 * draws.std() / np.sqrt(len(draws)), (
        f"Monte Carlo puts the best of 4 at {draws.mean():.4f}, got {got:.4f}"
    )


def test_stage5_refusals(nb):
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            nb.expected_best([1.0, 2.0], bad)
    with pytest.raises(ValueError):
        nb.expected_best([], 2)
