"""Checks for 30 · 04 finding questions.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. check_hypothesis rule by rule (one
broken field at a time, vague wording, no false alarms); paired_power against Monte Carlo t tests and its exact
values at zero effect, seeds_for_power on a hand example and its parameters, without hanging; run_hours and
largest_model on hand numbers and as inverses; budget_plan on a hand plan and its refusals; check_proposal
section by section, including how it combines the other four.
"""
import numpy as np
import pytest
from scipy import stats as sps

from mlfs.testing import bounded


def _hyp():
    return {
        "statement": "label smoothing of 0.1 raises the test accuracy of a 2-layer MLP on 5,000 MNIST images by at "
                     "least 0.3 points after 3 epochs",
        "metric": "test accuracy (points)", "method": "label smoothing 0.1", "baseline": "no label smoothing",
        "direction": "greater", "min_effect": 0.3, "setting": "MNIST, 5,000 training images, 784-128-10 MLP, SGD",
        "refuted_if": "the 95% interval of the paired difference over the planned seeds lies below +0.3 points",
        "n_seeds": 30,
    }


def _fields(problems):
    assert isinstance(problems, list) and all(isinstance(p, str) for p in problems), (
        f"return a list of 'field: message' strings, got {problems!r}"
    )
    return {p.split(":")[0] for p in problems}


# ---------------------------------------------------------------- stage 1: check_hypothesis


def test_stage1_good_hypothesis_passes(nb):
    h = _hyp()
    snapshot = repr(h)
    got = nb.check_hypothesis(h)
    assert got == [], f"a complete, falsifiable hypothesis has no problems, got {got}"
    assert repr(h) == snapshot, "check_hypothesis must not modify its argument"


def test_stage1_one_problem_per_broken_field(nb):
    cases = {
        "statement": [("statement", "")],
        "metric": [("metric", None)],
        "method": [("method", "  ")],
        "setting": [("setting", "")],
        "refuted_if": [("refuted_if", None)],
        "baseline": [("baseline", "Label Smoothing 0.1"), ("baseline", "")],
        "direction": [("direction", "better"), ("direction", None)],
        "min_effect": [("min_effect", 0), ("min_effect", -0.3), ("min_effect", "0.3"), ("min_effect", True),
                       ("min_effect", float("nan"))],
        "n_seeds": [("n_seeds", 1), ("n_seeds", 5.0), ("n_seeds", True), ("n_seeds", None)],
    }
    for field, changes in cases.items():
        for key, value in changes:
            h = _hyp()
            if value is None:
                h.pop(key)
            else:
                h[key] = value
            got = _fields(nb.check_hypothesis(h))
            assert got == {field}, (
                f"setting {key}={value!r} should give exactly one problem, prefixed '{field}: ...', got {sorted(got)}"
                + (" (a baseline equal to the method, ignoring case, isn't a baseline)" if field == "baseline" else "")
            )


def test_stage1_vague_wording(nb):
    for statement in ("we will explore whether label smoothing helps",
                      "label smoothing may improve accuracy by 0.3 points",
                      "this work will shed light on calibration",
                      "we investigate the effect of label smoothing",
                      "in some cases accuracy rises by 0.3 points"):
        h = _hyp()
        h["statement"] = statement
        got = nb.check_hypothesis(h)
        assert _fields(got) == {"statement"}, f"{statement!r} is vague (VAGUE matches it); flag it, got {got}"
    h = _hyp()
    h["statement"] = "We Will EXPLORE it, and it MIGHT work"
    (msg,) = nb.check_hypothesis(h)
    assert "explore" in msg and "might" in msg, f"name the vague words found (lowercased), got {msg!r}"
    for crisp in ("the mayor's dataset gets 0.3 points better accuracy", "studying time halves the error"):
        h = _hyp()
        h["statement"] = crisp
        assert nb.check_hypothesis(h) == [], f"{crisp!r} contains no vague WORD (only parts of words); don't flag it"


# ---------------------------------------------------------------- stage 2: paired_power, seeds_for_power


def test_stage2_power_at_zero_effect_is_alpha(nb):
    for alt in ("greater", "less", "two-sided"):
        for alpha in (0.05, 0.1):
            got = nb.paired_power(1.0, 0.0, 7, alpha=alpha, alternative=alt)
            assert np.isclose(got, alpha, atol=1e-9), (
                f"with no true effect, the power is the test's size alpha={alpha} ({alt}), got {got}"
            )


def test_stage2_power_matches_simulated_t_tests(nb):
    rng = np.random.default_rng(0)
    for sd, eff, n, alt in [(1.0, 0.5, 10, "greater"), (0.6, 0.3, 5, "greater"), (2.0, -1.0, 8, "less"),
                            (1.0, 1.0, 3, "greater"),
                            (1.0, 0.7, 12, "two-sided")]:
        d = rng.normal(eff, sd, size=(20000, n))
        p = sps.ttest_1samp(d, 0.0, axis=1, alternative=alt).pvalue
        sim = np.mean(p <= 0.05)
        got = nb.paired_power(sd, eff, n, alternative=alt)
        assert abs(got - sim) < 0.015, (
            f"paired_power(sd={sd}, effect={eff}, n={n}, {alt}) = {got:.3f}, but 20,000 simulated paired t tests "
            f"reject {sim:.3f} of the time. use the noncentral t: df = n - 1, nc = effect √n / sd."
        )
    assert np.isclose(nb.paired_power(1.0, -0.5, 10, alternative="less"), nb.paired_power(1.0, 0.5, 10))
    with pytest.raises(ValueError):
        nb.paired_power(1.0, 0.5, 10, alternative="bigger")


def _seeds(nb, *args, **kwargs):
    return bounded(nb.seeds_for_power, *args, what="seeds_for_power", hint="cap the search at max_seeds.", **kwargs)


def test_stage2_worked_example(nb):
    got = _seeds(nb, 1.0, 0.5)
    assert got == 27, (
        f"detecting an effect of half the sd with 80% power (one-sided, alpha 0.05) takes 27 pairs (power 0.812; "
        f"26 gives 0.798), got {got}"
    )
    assert _seeds(nb, 1.0, 0.5, alternative="two-sided") == 34, "two-sided needs 34"
    assert _seeds(nb, 0.6, 0.3) == 27, "only the ratio effect / sd matters"


def test_stage2_parameters_and_cap(nb):
    base = _seeds(nb, 1.0, 0.5)
    assert _seeds(nb, 1.0, 0.5, power=0.95) > base > _seeds(nb, 1.0, 0.5, power=0.5), "power must be passed through"
    assert _seeds(nb, 1.0, 0.5, alpha=0.01) > base, "alpha must be passed through"
    assert _seeds(nb, 1.0, 20.0) == 2, "never fewer than 2 pairs"
    with pytest.raises(ValueError):
        _seeds(nb, 1.0, 0.001, max_seeds=300)
    with pytest.raises(ValueError):
        _seeds(nb, 1.0, 0.5, max_seeds=26)
    assert _seeds(nb, 1.0, 0.5, max_seeds=27) == 27, "max_seeds is inclusive"


# ---------------------------------------------------------------- stage 3: run_hours, largest_model


def test_stage3_worked_example(nb):
    got = nb.run_hours(1e6, 2e7, flops_per_second=1e12, utilization=0.5)
    assert np.isclose(got, 1 / 15), (
        f"6 × 1e6 × 2e7 = 1.2e14 FLOPs at 0.5 × 1e12 FLOP/s is 240 s = 0.0667 h, got {got}"
    )
    assert np.isclose(nb.run_hours(1e6, 2e7, flops_per_second=1e12), 1 / 15 * 0.5 / 0.4), "default utilization 0.4"
    n, d = nb.largest_model(1.0, flops_per_second=1e12)
    assert np.isclose(n, 3.4641e6, rtol=1e-4) and np.isclose(d, 20 * n), (
        f"1 h at 0.4 × 1e12 FLOP/s is 1.44e15 FLOPs; Chinchilla (C = 6ND, D = 20N) gives N = √(1.44e15 / 120) = "
        f"3.46e6 parameters and 6.93e7 tokens, got ({n:.4g}, {d:.4g})"
    )
    assert isinstance(n, float) and isinstance(d, float), "return plain floats"


def test_stage3_round_trip_and_parameters(nb):
    for hours, flops, util, tpp in [(10.0, 5e12, 0.3, 20.0), (168.0, 2e13, 0.5, 40.0), (0.5, 1e11, 1.0, 5.0)]:
        n, d = nb.largest_model(hours, flops_per_second=flops, utilization=util, tokens_per_param=tpp)
        assert np.isclose(d / n, tpp), f"tokens_per_param={tpp} sets D / N, got {d / n}"
        back = nb.run_hours(n, d, flops_per_second=flops, utilization=util)
        assert np.isclose(back, hours), f"training the model largest_model returns should take {hours} h, got {back}"
    for bad in ({"utilization": 0.0}, {"utilization": 1.5}, {"flops_per_second": 0.0}):
        kwargs = {"flops_per_second": 1e12, **bad}
        with pytest.raises(ValueError):
            nb.run_hours(1e6, 1e7, **kwargs)
        with pytest.raises(ValueError):
            nb.largest_model(1.0, **kwargs)
    with pytest.raises(ValueError):
        nb.largest_model(0.0, flops_per_second=1e12)


# ---------------------------------------------------------------- stage 4: budget_plan

PLAN = [{"name": "pilot", "configs": 2, "seeds": 3, "hours_per_run": 0.5},
        {"name": "main", "configs": 4, "seeds": 10, "hours_per_run": 0.25}]


def test_stage4_worked_example(nb):
    got = nb.budget_plan(PLAN, 30.0)
    assert got["rows"] == [{"name": "pilot", "runs": 6, "hours": 3.0}, {"name": "main", "runs": 40, "hours": 10.0}], (
        f"one row per experiment, in order, with runs = configs × seeds and hours = runs × hours_per_run, got "
        f"{got['rows']}"
    )
    assert np.isclose(got["total"], 13.0) and np.isclose(got["with_safety"], 26.0) and got["budget"] == 30.0
    assert got["fits"] is True or got["fits"] == True, "26 h with the default safety factor 2 fits in 30 h"  # noqa: E712
    assert not nb.budget_plan(PLAN, 25.0)["fits"], "26 h doesn't fit in 25 h"
    loose = nb.budget_plan(PLAN, 25.0, safety=1.5)
    assert np.isclose(loose["with_safety"], 19.5) and loose["fits"], "safety=1.5 gives 19.5 h, which fits"
    assert nb.budget_plan(PLAN, 26.0)["fits"], "exactly at the budget fits"


def test_stage4_refusals(nb):
    bad_plans = [
        [PLAN[0], dict(PLAN[1], name="pilot")],
        [dict(PLAN[0], configs=0)],
        [dict(PLAN[0], seeds=2.5)],
        [dict(PLAN[0], hours_per_run=0)],
        [{"name": "x", "configs": 1, "seeds": 1}],
        [{"configs": 1, "seeds": 1, "hours_per_run": 1.0}],
        [dict(PLAN[0], name="")],
        [dict(PLAN[0], name=["pilot"])],
    ]
    for plan in bad_plans:
        with pytest.raises(ValueError):
            nb.budget_plan(plan, 10.0)
    for experiments in (["pilot"], None, PLAN[0]):
        with pytest.raises(ValueError):
            nb.budget_plan(experiments, 10.0)
    with pytest.raises(ValueError):
        nb.budget_plan(PLAN, "ten")
    with pytest.raises(ValueError):
        nb.budget_plan(PLAN, 0.0)
    with pytest.raises(ValueError):
        nb.budget_plan(PLAN, 10.0, safety=0.5)


# ---------------------------------------------------------------- stage 5: check_proposal


def _proposal_check(nb, *args, **kwargs):
    return bounded(nb.check_proposal, *args, what="check_proposal", hint="cap every search loop.", **kwargs)


def _proposal():
    return {
        "title": "does label smoothing help small MLPs?",
        "goal": "find out whether label smoothing raises the accuracy of small MLPs trained briefly on MNIST.",
        "state_of_the_art": "label smoothing is standard for large image and translation models; its effect on "
                            "small, briefly trained networks is rarely measured with enough seeds.",
        "approach": "paired runs over 30 seeds, a pilot to estimate the seed noise, a pre-registered margin.",
        "impact": "a measured answer for a default many course projects use.",
        "risks": ["the effect is smaller than the seed noise", "results depend on the learning rate"],
        "checkpoints": ["pilot done and sd measured (week 1)", "main runs done (week 3)"],
        "timeline_weeks": 4,
        "hypothesis": _hyp(),
        "pilot_sd": 0.5,
        "experiments": [{"name": "pilot", "configs": 2, "seeds": 5, "hours_per_run": 0.01},
                        {"name": "main", "configs": 2, "seeds": 30, "hours_per_run": 0.01}],
        "budget_hours": 2.0,
    }


def test_stage5_good_proposal_passes(nb):
    p = _proposal()
    snapshot = repr(p)
    got = _proposal_check(nb, p)
    assert got == [], f"this proposal is complete, powered (30 seeds; 19 needed) and within budget, got {got}"
    assert repr(p) == snapshot, "check_proposal must not modify the proposal"


def test_stage5_sections_and_length(nb):
    for field, value in [("title", ""), ("goal", None), ("state_of_the_art", " "), ("approach", None),
                         ("impact", ""), ("risks", []), ("checkpoints", ["week 1", ""]), ("timeline_weeks", 0),
                         ("timeline_weeks", True)]:
        p = _proposal()
        if value is None:
            p.pop(field)
        else:
            p[field] = value
        assert _fields(_proposal_check(nb, p)) == {field}, f"setting {field}={value!r} should be flagged as '{field}'"
    p = _proposal()
    n_sections = sum(len(p[f].split()) for f in ("title", "goal", "state_of_the_art", "approach", "impact"))
    assert _fields(_proposal_check(nb, p, max_words=n_sections + 2)) == {"length"}, (
        "the risks and checkpoints count towards the length too"
    )
    assert _fields(_proposal_check(nb, p, max_words=40)) == {"length"}, (
        "the sections, risks and checkpoints together are over 40 words: flag 'length'"
    )
    p["goal"] = " ".join(["word"] * 1000)
    assert _fields(_proposal_check(nb, p)) == {"length"}, "over the default 1,000 words"


def test_stage5_hypothesis_power_and_budget(nb):
    p = _proposal()
    p["hypothesis"]["min_effect"] = -1
    assert _fields(_proposal_check(nb, p)) == {"hypothesis.min_effect"}, (
        "hypothesis problems come prefixed 'hypothesis.' (and power isn't checked for a broken hypothesis)"
    )
    p = _proposal()
    p["hypothesis"]["statement"] = "we explore label smoothing"
    p["hypothesis"]["n_seeds"] = 10
    assert _fields(_proposal_check(nb, p)) == {"hypothesis.statement"}, (
        "fix the hypothesis first: power is only checked for a hypothesis without problems"
    )
    p = _proposal()
    p.pop("hypothesis")
    assert _fields(_proposal_check(nb, p)) == {"hypothesis"}
    p = _proposal()
    p["hypothesis"]["n_seeds"] = 10
    got = _proposal_check(nb, p)
    assert _fields(got) == {"power"} and any("19" in g for g in got), (
        f"detecting 0.3 with sd 0.5 at 80% power needs 19 seeds; a plan with 10 is underpowered, and the message "
        f"should say how many are needed. got {got}"
    )
    assert _proposal_check(nb, p, power=0.5) == [], "the power target is a parameter (10 seeds give 54%)"
    assert _fields(_proposal_check(nb, p, alpha=0.2)) == set(), "alpha is a parameter too"
    p = _proposal()
    p.pop("pilot_sd")
    assert _fields(_proposal_check(nb, p)) == {"pilot_sd"}
    p = _proposal()
    p["budget_hours"] = 1.0
    got = _proposal_check(nb, p)
    assert _fields(got) == {"budget"}, f"0.7 h × safety 2 = 1.4 h doesn't fit in 1 h, got {got}"
    assert _proposal_check(nb, p, safety=1.2) == [], "the safety factor is a parameter"
    for experiments, budget in ((None, 2.0), (["x"], 2.0), ([{"configs": 1, "seeds": 1, "hours_per_run": 1.0}], 2.0),
                                (_proposal()["experiments"], "two"),
                                (_proposal()["experiments"], None)):
        p = _proposal()
        p["experiments"], p["budget_hours"] = experiments, budget
        assert _fields(_proposal_check(nb, p)) == {"budget"}, (
            f"experiments={experiments!r}, budget_hours={budget!r}: a malformed plan is a 'budget' problem, not a crash"
        )
    p = _proposal()
    p["experiments"][0]["seeds"] = 0
    assert _fields(_proposal_check(nb, p)) == {"budget"}, "an invalid experiment plan is a budget problem, not a crash"
