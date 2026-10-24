"""Checks for 30 · 02 reproducing papers.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. equivalence_verdict on hand-computed
90% t intervals (each verdict, the alpha convention at a boundary) and its error rate at the margin's edge;
effect_verdict for all five verdicts in both directions (and the boundary between 'partly' and
'direction reproduced, size inconclusive'), pairing and Welch's interval; seeds_needed against the
formula, its invariances and its cap (without hanging); reproduction_report against an exact summary table plus
every section, option and refusal.
"""
import numpy as np
import pytest
from scipy import stats as sps

from mlfs.testing import bounded

RUNS = [0.70, 0.72, 0.71, 0.73, 0.69]  # mean 0.71, sd 0.0158; 90% t interval [0.6949, 0.7251]


# ---------------------------------------------------------------- stage 1: equivalence_verdict


def test_stage1_worked_example(nb):
    got = nb.equivalence_verdict(RUNS, 0.71, margin=0.02)
    assert set(got) >= {"mean", "ci", "n", "verdict"}, f"missing keys, got {sorted(got)}"
    assert np.isclose(got["mean"], 0.71) and got["n"] == 5
    assert np.allclose(got["ci"], (0.69493, 0.72507), atol=1e-5), (
        f"the interval is the (1 - 2 alpha) = 90% t interval: 0.71 ± t(0.95, 4) × 0.0158 / √5 = [0.6949, 0.7251], "
        f"got {got['ci']}. [0.6904, 0.7296] is the 95% interval: an equivalence test at level alpha uses 1 - 2 alpha."
    )
    assert got["verdict"] == "reproduced", f"[0.695, 0.725] lies inside 0.71 ± 0.02: reproduced, got {got['verdict']!r}"
    assert nb.equivalence_verdict(RUNS, 0.75, margin=0.02)["verdict"] == "not reproduced", (
        "against 0.75 ± 0.02 = [0.73, 0.77] the whole interval lies below the band: not reproduced"
    )
    assert nb.equivalence_verdict(RUNS, 0.735, margin=0.02)["verdict"] == "inconclusive", (
        "against [0.715, 0.755] the interval straddles the band's edge: inconclusive"
    )
    assert nb.equivalence_verdict(RUNS, 0.66, margin=0.02)["verdict"] == "not reproduced", (
        "above the band counts too: [0.695, 0.725] lies entirely above [0.64, 0.68]"
    )


def test_stage1_alpha_convention_at_the_boundary(nb):
    got = nb.equivalence_verdict(RUNS, 0.71, margin=0.017)
    assert got["verdict"] == "reproduced", (
        "the 90% interval [0.695, 0.725] fits inside 0.71 ± 0.017 but the 95% one [0.690, 0.730] doesn't: with "
        f"alpha=0.05 the verdict must be 'reproduced', got {got['verdict']!r}"
    )
    assert nb.equivalence_verdict(RUNS, 0.71, margin=0.017, alpha=0.025)["verdict"] == "inconclusive", (
        "with alpha=0.025 the interval is the 95% one, which pokes out of the band: inconclusive"
    )


def test_stage1_false_reproductions_are_rare_at_the_margin(nb):
    rng = np.random.default_rng(0)
    paper, margin, sd = 0.80, 0.01, 0.01
    hits = 0
    n_sims = 2000
    for _ in range(n_sims):
        runs = rng.normal(paper + margin, sd, size=5)  # the truth sits exactly at the edge of the band
        hits += nb.equivalence_verdict(runs, paper, margin=margin)["verdict"] == "reproduced"
    rate = hits / n_sims
    assert rate <= 0.065, (
        f"when the true value sits at the edge of the margin, 'reproduced' should come out at most about alpha = 5% "
        f"of the time, got {rate:.3f}. a normal (z) quantile, or no interval at all, makes false reproductions common."
    )


def test_stage1_refusals(nb):
    for bad in ([0.7], [0.7, np.nan, 0.71]):
        with pytest.raises(ValueError):
            nb.equivalence_verdict(bad, 0.7, margin=0.01)
    for margin in (0.0, -0.1):
        with pytest.raises(ValueError):
            nb.equivalence_verdict(RUNS, 0.7, margin=margin)
    with pytest.raises(ValueError):
        nb.equivalence_verdict(RUNS, 0.7, margin=0.01, alpha=0.5)


# ---------------------------------------------------------------- stage 2: effect_verdict

DIR = "direction reproduced, size inconclusive"
A = np.array([0.80, 0.82, 0.79, 0.81, 0.83])
B = np.array([0.78, 0.79, 0.78, 0.80, 0.80])  # paired differences (0.02, 0.03, 0.01, 0.01, 0.03): 90% CI [0.0105, 0.0295]


def test_stage2_worked_example(nb):
    got = nb.effect_verdict(A, B, 0.02, margin=0.012)
    assert set(got) >= {"diff", "ci", "n", "verdict"}, f"missing keys, got {sorted(got)}"
    assert np.isclose(got["diff"], 0.02) and got["n"] == 5
    assert np.allclose(got["ci"], (0.010466, 0.029534), atol=1e-5), (
        f"the paired differences are (0.02, 0.03, 0.01, 0.01, 0.03): 90% t interval [0.0105, 0.0295], got {got['ci']}"
    )
    assert got["verdict"] == "reproduced", "the interval lies inside 0.02 ± 0.012: reproduced"
    assert nb.effect_verdict(A, B, 0.05, margin=0.015)["verdict"] == "partly reproduced", (
        "the paper claims +0.05 ± 0.015; our interval [0.010, 0.030] is above 0 (the direction holds) but below the "
        "band (the size doesn't): partly reproduced"
    )
    assert nb.effect_verdict(A, B, -0.02, margin=0.01)["verdict"] == "not reproduced", (
        "the paper claims B beats A by 0.02; we find the opposite with an interval outside [-0.03, -0.01]: not "
        "reproduced"
    )


def test_stage2_inconclusive_and_null_effects(nb):
    noisy = B + np.array([0.05, -0.04, 0.03, -0.02, 0.01])  # interval [-0.029, 0.041]
    assert nb.effect_verdict(noisy, B, 0.02, margin=0.01)["verdict"] == "inconclusive", (
        "an interval of [-0.029, 0.041] overlaps the band [0.01, 0.03] and includes 0: inconclusive (run more seeds)"
    )
    assert nb.effect_verdict(B, noisy, -0.02, margin=0.01)["verdict"] == "inconclusive", (
        "for a negative claim the direction holds only if the interval is entirely below 0 (u < 0): [-0.041, 0.029] "
        "includes 0, so it's inconclusive"
    )
    flat = B + np.array([0.001, -0.001, 0.0, 0.002, -0.002])  # interval about ±0.0015
    assert nb.effect_verdict(flat, B, 0.02, margin=0.01)["verdict"] == "not reproduced", (
        "a tight interval around 0, entirely below the band [0.01, 0.03]: not reproduced"
    )
    assert nb.effect_verdict(flat, B, 0.0, margin=0.005)["verdict"] == "reproduced", (
        "the paper claims no difference (0 ± 0.005) and our interval fits inside: reproduced"
    )
    assert nb.effect_verdict(A, B, 0.0, margin=0.005)["verdict"] == "not reproduced", (
        "a claim of no difference has no direction, so it can't be 'partly' reproduced: [0.010, 0.030] lies above "
        "0 ± 0.005, not reproduced"
    )
    assert nb.effect_verdict(B, A, -0.05, margin=0.015)["verdict"] == "partly reproduced", (
        "negative effects work the same way: an interval entirely below 0 but outside -0.05 ± 0.015 is partly "
        "reproduced"
    )


def test_stage2_partly_versus_direction_only(nb):
    # the paired interval is [0.0105, 0.0295]
    cases = [
        (0.04, 0.0104, "partly reproduced",
         "the band [0.0296, 0.0504] starts just above the interval: the direction holds AND the size is shown to "
         "differ (smaller)"),
        (0.04, 0.0106, DIR,
         "the band [0.0294, 0.0506] overlaps the interval's top: the direction holds, but the size is neither "
         "confirmed nor ruled out"),
        (0.007, 0.0034, "partly reproduced",
         "the band [0.0036, 0.0104] ends just below the interval: the effect is shown to be LARGER than claimed"),
        (0.008, 0.0026, DIR, "the band [0.0054, 0.0106] overlaps the interval's bottom"),
    ]
    for eff, margin, want, why in cases:
        got = nb.effect_verdict(A, B, eff, margin=margin)["verdict"]
        assert got == want, f"paper effect {eff} ± {margin}: expected {want!r}, got {got!r}. {why}."
        got = nb.effect_verdict(B, A, -eff, margin=margin)["verdict"]
        assert got == want, f"the mirror image (paper effect {-eff} ± {margin}): expected {want!r}, got {got!r}"


def test_stage2_pairing_and_welch(nb):
    shared = np.array([-0.10, -0.05, 0.0, 0.05, 0.10, -0.08, 0.08, 0.02])  # what each seed does to both
    a = 0.80 + 0.02 + shared + np.array([0.001, -0.001, 0.002, -0.002, 0.0, 0.001, -0.001, 0.0])
    b = 0.80 + shared
    pa = nb.effect_verdict(a, b, 0.02, margin=0.01)
    un = nb.effect_verdict(a, b, 0.02, margin=0.01, paired=False)
    assert pa["verdict"] == "reproduced" and un["verdict"] == "inconclusive", (
        f"with shared seed noise, the paired interval is narrow (reproduced) and the unpaired one wide "
        f"(inconclusive): got {pa['verdict']!r} and {un['verdict']!r}. is paired ignored?"
    )
    rng = np.random.default_rng(1)
    x, y = rng.normal(1.0, 0.1, 6), rng.normal(0.8, 0.4, 9)
    got = nb.effect_verdict(x, y, 0.2, margin=0.15, paired=False)
    va, vb = x.var(ddof=1) / 6, y.var(ddof=1) / 9
    df = (va + vb) ** 2 / (va ** 2 / 5 + vb ** 2 / 8)
    half = sps.t.ppf(0.95, df) * np.sqrt(va + vb)
    d = x.mean() - y.mean()
    assert np.isclose(got["diff"], d) and np.allclose(got["ci"], (d - half, d + half)), (
        f"unpaired, use Welch's interval: diff of means ± t(0.95, Welch df = {df:.2f}) × sqrt(s_a²/n_a + s_b²/n_b) = "
        f"[{d - half:.4f}, {d + half:.4f}], got {got['ci']}"
    )
    assert got["n"] == 6, "for an unpaired comparison n is the smaller group's size"
    assert nb.effect_verdict(y, x, -0.2, margin=0.15, paired=False)["n"] == 6, (
        "the smaller group can be either one: with 9 runs against 6, n is still 6"
    )


def test_stage2_refusals(nb):
    with pytest.raises(ValueError):
        nb.effect_verdict(A, B, 0.02, margin=0.02)  # a margin as big as the effect can't test the claim
    with pytest.raises(ValueError):
        nb.effect_verdict(A, B[:4], 0.02, margin=0.01)
    with pytest.raises(ValueError):
        nb.effect_verdict(A, B, 0.02, margin=0.0)
    with pytest.raises(ValueError):
        nb.effect_verdict([0.8], [0.7], 0.1, margin=0.05, paired=False)
    with pytest.raises(ValueError):
        nb.effect_verdict(A, np.r_[B[:4], np.nan], 0.02, margin=0.01)


# ---------------------------------------------------------------- stage 3: seeds_needed


def _seeds(nb, *args, **kwargs):
    return bounded(nb.seeds_needed, *args, what="seeds_needed", hint="cap the search at max_seeds.", **kwargs)


def test_stage3_worked_example(nb):
    got = _seeds(nb, 1.0, 1.0)
    assert got == 10, (
        f"with sd = margin: (t(0.95, n-1) + z(0.9)) / sqrt(n) first drops below 1 at n = 10 (0.985; n = 9 gives "
        f"1.047), got {got}. check that you use t with n - 1 dof at 1 - alpha and z at (1 + power) / 2."
    )
    assert isinstance(got, (int, np.integer))


def test_stage3_matches_the_formula(nb):
    def ref(sd, margin, alpha=0.05, power=0.8):
        z = sps.norm.ppf((1 + power) / 2)
        n = 2
        while (sps.t.ppf(1 - alpha, n - 1) + z) * sd / np.sqrt(n) > margin:
            n += 1
        return n

    for sd, margin, alpha, power in [(0.01, 0.005, 0.05, 0.8), (0.3, 0.5, 0.05, 0.9), (1.0, 0.3, 0.01, 0.8),
                                     (2.0, 5.0, 0.05, 0.5), (0.007, 0.002, 0.1, 0.95)]:
        got = _seeds(nb, sd, margin, alpha=alpha, power=power)
        assert got == ref(sd, margin, alpha, power), (
            f"seeds_needed({sd}, {margin}, alpha={alpha}, power={power}) should be {ref(sd, margin, alpha, power)}, "
            f"got {got}"
        )
    assert _seeds(nb, 0.1, 10.0) == 2, "never fewer than 2 seeds"


def test_stage3_invariances(nb):
    assert _seeds(nb, 0.02, 0.01) == _seeds(nb, 2.0, 1.0), "only the ratio sd / margin matters"
    ns = [_seeds(nb, sd, 0.01) for sd in (0.002, 0.005, 0.01, 0.02)]
    assert ns == sorted(ns) and ns[0] < ns[-1], f"noisier runs need more seeds, got {ns}"
    assert _seeds(nb, 0.01, 0.01, power=0.95) > _seeds(nb, 0.01, 0.01, power=0.5), "more power needs more seeds"


def test_stage3_cap_and_refusals(nb):
    with pytest.raises(ValueError):
        _seeds(nb, 1.0, 0.001, max_seeds=500)  # would need about 6 million seeds
    with pytest.raises(ValueError):
        _seeds(nb, 1.0, 0.5, max_seeds=35)
    assert _seeds(nb, 1.0, 0.5, max_seeds=36) == 36, "max_seeds is the largest n allowed, and it's inclusive"
    for args in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            _seeds(nb, *args)
    with pytest.raises(ValueError):
        _seeds(nb, 1.0, 1.0, power=1.0)


# ---------------------------------------------------------------- stage 4: reproduction_report

CLAIMS = [
    {"id": "C1", "kind": "value", "statement": "the 20-layer ResNet reaches 71% test accuracy",
     "evidence": "table 6", "metric": "test accuracy", "paper": 0.71, "margin": 0.02},
    {"id": "C2", "kind": "effect", "statement": "residual beats plain at 20 layers",
     "evidence": "figure 6", "metric": "test accuracy", "paper": 0.02, "margin": 0.012},
]
RESULTS = {"C1": RUNS, "C2": (A, B)}
GAPS = [{"detail": "weight decay", "choice": "1e-4", "reason": "the value the paper uses for ImageNet"}]


def test_stage4_summary_table(nb):
    md, verdicts = nb.reproduction_report("ResNet on CIFAR-10", CLAIMS, RESULTS, gaps=GAPS,
                                          checklist={"seeds": "5 per configuration, seeds 0-4", "code": "  "})
    assert verdicts == {"C1": "reproduced", "C2": "reproduced"}, f"verdicts per claim id, got {verdicts}"
    lines = md.splitlines()
    assert lines[0] == "# ResNet on CIFAR-10", f"the report starts with '# title', got {lines[0]!r}"
    expected = [
        "| claim | metric | paper | ours (mean, 90% CI) | n | verdict |",
        "|---|---|---|---|---|---|",
        "| C1 | test accuracy | 0.71 ± 0.02 | 0.71 [0.695, 0.725] | 5 | **reproduced** |",
        "| C2 | test accuracy | Δ = 0.02 ± 0.012 | Δ = 0.02 [0.0105, 0.0295] | 5 | **reproduced** |",
    ]
    i = lines.index("## summary")
    assert lines[i + 2:i + 6] == expected, (
        "the summary table doesn't match the spec. expected:\n" + "\n".join(expected) + "\n\ngot:\n"
        + "\n".join(lines[i + 2:i + 6])
    )


def test_stage4_sections(nb):
    md, _ = nb.reproduction_report("ResNet on CIFAR-10", CLAIMS, RESULTS, gaps=GAPS,
                                   checklist={"seeds": "5 per configuration, seeds 0-4", "code": "  "})
    for heading in ("## summary", "## claims", "## gaps", "## reproducibility checklist"):
        assert heading in md.splitlines(), f"the report needs a '{heading}' line"
    assert md.index("## summary") < md.index("## claims") < md.index("## gaps") < md.index("## reproducibility")
    assert "### C2: residual beats plain at 20 layers" in md, "each claim gets '### id: statement'"
    assert "evidence in the paper: figure 6" in md, "each claim section names where the paper's evidence is"
    assert "| weight decay | 1e-4 | the value the paper uses for ImageNet |" in md, "each gap is a table row"
    assert "- [x] the number of seeds per configuration and how seeds were chosen are stated: 5 per configuration, " \
           "seeds 0-4" in md, "an answered checklist item is '- [x] item: answer'"
    assert "- [ ] the code, configurations and the command for each figure and table are available: **unanswered**" \
           in md, "a blank answer counts as unanswered"
    assert "1 of 10 items answered." in md, "the checklist section says how many of its 10 items are answered"
    assert sum(line.startswith("- [") for line in md.splitlines()) == 10, "every checklist item appears once"


def test_stage4_options(nb):
    md, verdicts = nb.reproduction_report("t", CLAIMS, RESULTS, alpha=0.1)
    assert "| claim | metric | paper | ours (mean, 80% CI) | n | verdict |" in md, (
        "with alpha=0.1 the intervals (and the header) are 1 - 2 alpha = 80%"
    )
    half = sps.t.ppf(0.9, 4) * np.std(RUNS, ddof=1) / np.sqrt(5)
    row = f"| C1 | test accuracy | 0.71 ± 0.02 | 0.71 [{0.71 - half:.3g}, {0.71 + half:.3g}] | 5 | **reproduced** |"
    assert row in md, f"alpha must reach the verdict functions too: expected the row\n{row}"
    assert "none listed." in md and "0 of 10 items answered." in md, "no gaps -> 'none listed.'; no checklist -> 0"
    claims = [dict(CLAIMS[1], paired=False)]
    _, v = nb.reproduction_report("t", claims, {"C2": (A, B)})
    paired_v = nb.reproduction_report("t", [CLAIMS[1]], {"C2": (A, B)})[1]
    assert v == {"C2": DIR} and paired_v == {"C2": "reproduced"}, (
        f"a claim's 'paired': False makes its effect unpaired (Welch): got {v} unpaired, {paired_v} paired"
    )


def test_stage4_refusals(nb):
    with pytest.raises(ValueError):
        nb.reproduction_report("t", [CLAIMS[0], CLAIMS[0]], RESULTS)
    with pytest.raises(ValueError):
        nb.reproduction_report("t", CLAIMS, {"C1": RUNS})
    with pytest.raises(ValueError):
        nb.reproduction_report("t", [dict(CLAIMS[0], kind="ratio")], RESULTS)
    with pytest.raises(ValueError):
        nb.reproduction_report("t", CLAIMS, RESULTS, gaps=[{"detail": "lr", "choice": "0.1", "reason": ""}])
    with pytest.raises(ValueError):
        nb.reproduction_report("t", CLAIMS, RESULTS, checklist={"seed": "5"})
