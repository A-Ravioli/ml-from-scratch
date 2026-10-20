import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections.abc import Mapping

    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks
    from mlfs.stats import bonferroni, bootstrap_ci, permutation_test  # yours, from ch 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 29 · capstone: the elective portfolio (reference solution)

    read [`brief.md`](brief.md) first. every elective block ends with a project that pits a
    method against a classical baseline: PC + IDA against a naive regression, replay against
    fine-tuning, a selective SSM against a transformer, K-FAC against Adam. this notebook
    builds one small toolkit that every one of those comparisons goes through, so that each
    report in your portfolio makes the same kind of claim with the same kind of evidence. then
    you write one portfolio entry per elective you completed.

    | stage | you build |
    |---|---|
    | 1 | `paired_differences`: line up method and baseline by seed, refuse anything unpaired |
    | 2 | `paired_ci`: a t or bootstrap interval for the mean paired difference |
    | 3 | `sign_flip_test`, `min_p_value`: an exact sign-flip test, and the smallest p-value $n$ seeds allow |
    | 4 | `holm`: Holm's step-down correction, for a portfolio that makes several claims |
    | 5 | `memory_cost`, `budget_table`: compare methods at a matched budget, with the unit made explicit |
    | 6 | `portfolio_entry`, `render_entry`: one structured, reproducible record per elective |

    when they're green, scroll down: two worked entries (causal inference and continual
    learning) show the format, and the write-up prompts are at the end.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 1: pair by seed

    a comparison is *paired* when method and baseline ran on the same seed, draw, split or
    task stream, so their scores share everything except the method. pairing removes the
    shared noise, but only if the pairs really line up: zipping two lists whose seeds come in
    different orders silently compares seed 3 with seed 0. so the toolkit takes scores keyed
    by seed and refuses anything it can't pair.
    """)
    return


@app.function
def paired_differences(method_scores, baseline_scores):
    """stage 1: (seeds, d): the seeds both runs share, sorted, and d = method - baseline per seed (float (n,))."""
    if isinstance(method_scores, Mapping) != isinstance(baseline_scores, Mapping):
        raise ValueError("pass both score sets as {seed: score} dicts, or both as sequences indexed by seed")
    if isinstance(method_scores, Mapping):
        if set(method_scores) != set(baseline_scores):
            only_m = sorted(set(method_scores) - set(baseline_scores))
            only_b = sorted(set(baseline_scores) - set(method_scores))
            raise ValueError(f"seeds don't match: only in method {only_m}, only in baseline {only_b}")
        seeds = sorted(method_scores)
        m = np.array([method_scores[s] for s in seeds], dtype=float)
        b = np.array([baseline_scores[s] for s in seeds], dtype=float)
    else:
        m = np.asarray(method_scores, dtype=float).ravel()
        b = np.asarray(baseline_scores, dtype=float).ravel()
        if len(m) != len(b):
            raise ValueError(f"unequal lengths: {len(m)} method scores, {len(b)} baseline scores")
        seeds = list(range(len(m)))
    if len(m) == 0:
        raise ValueError("no scores to compare")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(b))):
        raise ValueError("scores must be finite (a nan usually means a crashed run: report it, don't drop it)")
    return seeds, m - b


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", paired_differences=paired_differences)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 2: an interval for the mean difference

    with $n$ paired differences $d_i$, the Student $t$ interval is
    $\bar d \pm t_{1-\alpha/2,\,n-1}\, s_d/\sqrt n$, where $s_d$ uses $n - 1$. with 5 seeds the
    quantile is 2.776, not 1.96, and that matters: 1.96 would cover about 88% of the time
    instead of 95%. the percentile bootstrap (your `mlfs.stats.bootstrap_ci`) assumes less
    about the shape of the differences, but at 5 seeds it under-covers badly, because 5 values
    are a poor stand-in for their distribution. offer both; default to $t$ for few seeds.
    """)
    return


@app.function
def paired_ci(d, *, alpha=0.05, method="t", n_boot=10000, rng=None):
    """stage 2: (lo, hi) for mean(d): method 't' (Student t, n - 1 dof) or 'bootstrap' (mlfs.stats.bootstrap_ci)."""
    d = np.asarray(d, dtype=float).ravel()
    if len(d) < 2:
        raise ValueError(f"an interval needs at least 2 paired seeds, got {len(d)}")
    if method == "t":
        half = sps.t.ppf(1 - alpha / 2, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        return float(d.mean() - half), float(d.mean() + half)
    if method == "bootstrap":
        return bootstrap_ci(d, np.mean, n_boot=n_boot, alpha=alpha, rng=rng)
    raise ValueError(f"method must be 't' or 'bootstrap', got {method!r}")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", paired_ci=paired_ci)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 3: an exact sign-flip test

    under "the method makes no difference", each $d_i$ is as likely to be $+|d_i|$ as
    $-|d_i|$. with $n$ seeds there are only $2^n$ sign patterns, so for small $n$ you can
    enumerate all of them instead of sampling: the p-value is the fraction of patterns whose
    mean is at least as extreme as the observed one (the observed pattern included). that
    exactness exposes a fact Monte Carlo hides: with 5 seeds, the observed pattern and its
    mirror image are always the two most extreme, so a two-sided p-value can never go below
    $2/32 = 0.0625$. five seeds can't "reach significance" at 0.05, however large the effect.
    """)
    return


@app.function
def sign_flip_test(d, *, alternative="two-sided", n_permutations=10000, rng=None):
    """stage 3: p-value of mean(d) under random sign flips: exact over all 2**n patterns if 2**n <= n_permutations,
    else mlfs.stats.permutation_test(d, zeros, paired=True, ...)."""
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
    d = np.asarray(d, dtype=float).ravel()
    n = len(d)
    if 2**n > n_permutations:
        return float(permutation_test(d, np.zeros(n), n_permutations=n_permutations, alternative=alternative,
                                      paired=True, rng=rng)[1])
    signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
    null = signs @ d / n
    t = d.mean()
    tol = 1e-12 * max(1.0, abs(t))
    if alternative == "greater":
        extreme = null >= t - tol
    elif alternative == "less":
        extreme = null <= t + tol
    else:
        extreme = np.abs(null) >= abs(t) - tol
    return float(np.mean(extreme))


@app.function
def min_p_value(n, alternative="two-sided"):
    """stage 3: the smallest p-value an exact sign-flip test on n paired seeds can return."""
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}")
    return (2.0 if alternative == "two-sided" else 1.0) / 2**n


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", sign_flip_test=sign_flip_test, min_p_value=min_p_value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 4: several claims at once

    a portfolio of three electives makes at least three claims, often more (each project has
    several baselines). Bonferroni tests each of $m$ p-values at $\alpha/m$. **Holm**'s
    step-down version controls the same family-wise error rate and never rejects less: sort the
    p-values, compare the smallest with $\alpha/m$, the next with $\alpha/(m-1)$, and so on,
    stopping at the first failure. its adjusted p-values are
    $\tilde p_{(i)} = \max_{j \le i}\min\{1, (m - j + 1)\,p_{(j)}\}$, returned in the input's
    order.
    """)
    return


@app.function
def holm(p_values, alpha=0.05):
    """stage 4: (reject, adjusted): Holm step-down; bool (m,) reject mask and adjusted p-values (m,), input order."""
    p = np.asarray(p_values, dtype=float).ravel()
    m = len(p)
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted <= alpha, adjusted


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", holm=holm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 5: matched budgets, with units

    "my method is better" usually means "better at the same cost", and the cost needs a unit.
    the continual-learning project counted memory in *stored numbers* (its `n_floats()`): a
    Gaussian per class (15,680 float32s) beat 50 stored images (39,200 numbers). count the same
    memories in **bytes** and MNIST's pixels are uint8: 200 images fit in the bytes of 39,200
    float32s, and they edge out the generator. neither unit is wrong, but the winner depends on
    which you pick, so the table must say.

    a memory is a list of `(count, dtype)` components. `unit='floats'` counts stored numbers
    whatever their type; `unit='bytes'` multiplies each count by its dtype's itemsize.
    """)
    return


@app.function
def memory_cost(components, unit="bytes"):
    """stage 5: total size of [(count, dtype), ...]: unit 'floats' (number of stored values) or 'bytes'."""
    if unit == "floats":
        return int(sum(int(c) for c, _ in components))
    if unit == "bytes":
        return int(sum(int(c) * np.dtype(t).itemsize for c, t in components))
    raise ValueError(f"unit must be 'floats' or 'bytes', got {unit!r}")


@app.function
def budget_table(methods, budget, *, unit="bytes", higher_is_better=True):
    """stage 5: dict(rows, winner): one row per method (name, cost, fits, mean, sd, n), fitting rows first, best first;
    winner = best method with cost <= budget (None if none fits)."""
    rows = []
    for name, spec in methods.items():
        scores = np.asarray(spec["scores"], dtype=float).ravel()
        cost = memory_cost(spec["memory"], unit)
        rows.append({"name": name, "cost": cost, "fits": cost <= budget, "mean": float(scores.mean()),
                     "sd": float(scores.std(ddof=1)) if len(scores) > 1 else float("nan"), "n": len(scores)})
    sign = -1.0 if higher_is_better else 1.0
    rows.sort(key=lambda r: (not r["fits"], sign * r["mean"]))
    winner = rows[0]["name"] if rows and rows[0]["fits"] else None
    return {"rows": rows, "winner": winner, "budget": budget, "unit": unit}


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", memory_cost=memory_cost, budget_table=budget_table)
    return


@app.function
def render_budget_table(table):
    """markdown for a budget_table result (not a stage: a display helper)."""
    lines = [f"budget: **{table['budget']:,} {table['unit']}**", "",
             f"| method | cost ({table['unit']}) | fits | mean ± sd | seeds |", "|---|---|---|---|---|"]
    for r in table["rows"]:
        sd = "" if np.isnan(r["sd"]) else f" ± {r['sd']:.3f}"
        name = f"**{r['name']}**" if r["name"] == table["winner"] else r["name"]
        lines.append(f"| {name} | {r['cost']:,} | {'yes' if r['fits'] else 'no'} | {r['mean']:.3f}{sd} | {r['n']} |")
    return "\n".join(lines)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## stage 6: a portfolio entry

    one record per elective, built from per-seed scores so that every number in it can be
    recomputed. the verdict comes from the interval, not the p-value: **better** if the
    interval for (method − baseline) lies entirely on the good side of 0, **worse** if
    entirely on the bad side, **inconclusive** otherwise. the p-value and the smallest p-value
    your seed count allows sit next to it, so a reader can see when "not significant" only
    means "five seeds".
    """)
    return


@app.function
def portfolio_entry(block, question, method_name, method_scores, baseline_name, baseline_scores, *, metric,
                    higher_is_better=True, alpha=0.05, ci_method="t", n_permutations=10000, budget=None, notes="",
                    rng=None):
    """stage 6: a dict recording one elective's project: paired seeds, means, diff, ci, p, min_p and verdict."""
    rng = np.random.default_rng(rng)
    seeds, d = paired_differences(method_scores, baseline_scores)
    if isinstance(method_scores, Mapping):
        m = np.array([method_scores[s] for s in seeds], dtype=float)
        b = np.array([baseline_scores[s] for s in seeds], dtype=float)
    else:
        m = np.asarray(method_scores, dtype=float).ravel()
        b = np.asarray(baseline_scores, dtype=float).ravel()
    lo, hi = paired_ci(d, alpha=alpha, method=ci_method, rng=rng)
    p = sign_flip_test(d, n_permutations=n_permutations, rng=rng)
    good_lo, good_hi = (lo > 0, hi < 0) if higher_is_better else (hi < 0, lo > 0)
    verdict = "better" if good_lo else "worse" if good_hi else "inconclusive"
    return {"block": block, "question": question, "metric": metric, "higher_is_better": higher_is_better,
            "method": method_name, "baseline": baseline_name, "seeds": list(seeds), "n": len(d),
            "method_mean": float(m.mean()), "baseline_mean": float(b.mean()), "diff": float(d.mean()),
            "ci": (float(lo), float(hi)), "alpha": alpha, "ci_method": ci_method, "p": float(p),
            "min_p": min_p_value(len(d)), "verdict": verdict, "budget": budget, "notes": notes}


@app.function
def render_entry(entry):
    """stage 6: the entry as markdown, in the fixed format of brief.md."""
    lo, hi = entry["ci"]
    level = int(round(100 * (1 - entry["alpha"])))
    direction = "higher" if entry["higher_is_better"] else "lower"
    lines = [
        f"### {entry['block']}", "",
        f"**question.** {entry['question']}", "",
        f"| | {entry['metric']} (mean over n = {entry['n']} paired seeds) |", "|---|---|",
        f"| {entry['method']} | {entry['method_mean']:.3f} |",
        f"| {entry['baseline']} (baseline) | {entry['baseline_mean']:.3f} |",
        f"| difference (method − baseline) | {entry['diff']:+.3f}, {level}% CI [{lo:+.3f}, {hi:+.3f}] "
        f"({entry['ci_method']}) |",
        f"| sign-flip p (two-sided) | {entry['p']:.3g} (smallest possible with n = {entry['n']}: "
        f"{entry['min_p']:.3g}) |", "",
        f"**verdict:** {entry['verdict']} ({direction} is better).",
    ]
    if entry.get("budget"):
        lines += ["", f"**budget:** {entry['budget']}"]
    if entry.get("notes"):
        lines += ["", entry["notes"]]
    return "\n".join(lines)


@app.cell(hide_code=True)
def _():
    checks.run(6, _tests="test_capstone.py", portfolio_entry=portfolio_entry, render_entry=render_entry,
               paired_differences=paired_differences, paired_ci=paired_ci, sign_flip_test=sign_flip_test,
               min_p_value=min_p_value)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## why $t$ and not the bootstrap at five seeds

    a quick check of stage 2's claim: draw 5 differences from a normal distribution with mean
    0.3, build both intervals, and count how often each covers 0.3.
    """)
    return


@app.cell
def _():
    _rng = np.random.default_rng(0)
    _hits = {"t": 0, "bootstrap": 0}
    _R = 400
    for _ in range(_R):
        _d = 0.3 + _rng.normal(size=5)
        for _m in _hits:
            _lo, _hi = paired_ci(_d, method=_m, n_boot=1000, rng=_rng)
            _hits[_m] += _lo <= 0.3 <= _hi
    mo.md(f"coverage of a nominal 95% interval with 5 seeds, {_R} simulated experiments: "
          f"**t {_hits['t'] / _R:.2f}**, **percentile bootstrap {_hits['bootstrap'] / _R:.2f}** "
          f"(Monte Carlo standard error about 0.01–0.02).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## worked entry A: causal inference (topics 01–04)

    the block's project (topic 04's lab, experiment 3): a 7-variable linear world whose three
    root causes are real California-housing covariates (each column shuffled, 5,000 districts
    per draw), with a true effect of $T$ on $Y$ of $0.6 + 0.9 \times 0.4 = 0.96$. the method
    runs PC at $\alpha = 0.01$, reads $T$'s possible parent sets off the learned graph and
    averages the IDA estimates; the classical baseline is the regression of $Y$ on $T$ alone.
    the metric is the absolute error of the estimated effect, so **lower is better**, paired by
    draw (seeds 100–119). the numbers below were measured with the reference solution on the
    cached real covariates, exactly as in the lab.
    """)
    return


@app.cell
def _():
    # topic 04 lab, experiment 3, 20 fresh draws (seeds 100-119), independent roots; measured with the reference
    causal_ida = [0.975, 0.974, 0.953, 0.950, 0.927, 0.975, 0.974, 0.978, 0.963, 0.979,
                  0.930, 0.977, 0.946, 0.955, 0.955, 0.985, 0.942, 0.973, 0.960, 0.934]
    causal_naive = [1.130, 1.115, 1.119, 1.109, 1.108, 1.123, 1.140, 1.139, 1.105, 1.130,
                    1.119, 1.130, 1.107, 1.120, 1.110, 1.127, 1.114, 1.125, 1.123, 1.100]
    _truth = 0.96
    entry_causal = portfolio_entry(
        "causal inference (topics 01–04)",
        "can PC + IDA recover the effect of T on Y from semi-synthetic observational data better than a naive "
        "regression?",
        "PC (α = 0.01) + IDA", {100 + i: abs(v - _truth) for i, v in enumerate(causal_ida)},
        "regression of Y on T", {100 + i: abs(v - _truth) for i, v in enumerate(causal_naive)},
        metric="absolute error of the effect estimate (truth 0.96)", higher_is_better=False, rng=0,
        notes=(
            f"estimates: PC + IDA {np.mean(causal_ida):.3f} ± {np.std(causal_ida, ddof=1):.3f} (sd over draws), naive "
            f"{np.mean(causal_naive):.3f}, truth 0.96. PC returned the true CPDAG in 20 of 20 draws. the naive bias "
            "is income's confounding, 0.4 × 0.8 / 2 = 0.16, as measured. **caveat:** with the covariates' real "
            "dependence kept, a near-violation of faithfulness (ρ(inc, C | Y) = 0.037) makes PC orient a false "
            "v-structure at T in 10 of 20 draws, and IDA falls to 0.816 ± 0.147. the semi-synthetic world with "
            "independent roots is the setting PC was designed for; the real-dependence variant is the honest "
            "stress test."
        ),
    )
    mo.md(render_entry(entry_causal))
    return (entry_causal,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## worked entry B: continual learning (topics 12–14)

    the block's project (topic 14's lab): class-incremental split MNIST with the block's
    784–200–200–10 MLP, one epoch per task, and a memory budget of 39,200 stored numbers (50
    images). the method is a per-class diagonal Gaussian generator replaying 64 samples per
    step; the classical baseline is plain fine-tuning; seeds 0–4, paired (same network
    initialisation and data order). measured with the reference solution on the cached real
    MNIST.
    """)
    return


@app.cell
def _():
    # topic 14 lab, project, seeds 0-4 (reference solution, real MNIST); reservoir(200) measured the same way
    cont_ft = [0.196, 0.195, 0.196, 0.196, 0.196]
    cont_gauss = [0.791, 0.802, 0.765, 0.780, 0.760]
    cont_res50 = [0.561, 0.563, 0.583, 0.556, 0.564]
    cont_res200 = [0.794, 0.811, 0.829, 0.777, 0.808]
    entry_continual = portfolio_entry(
        "continual learning (topics 12–14)",
        "with 39,200 stored numbers of memory, can a replay strategy beat fine-tuning on class-incremental split "
        "MNIST?",
        "Gaussian generator, 64 samples per step", dict(enumerate(cont_gauss)),
        "fine-tuning", dict(enumerate(cont_ft)),
        metric="average accuracy after task 5 (ACC)", rng=0,
        budget="15,680 float32s (62,720 bytes) of a 39,200-number budget; see the budget tables below",
        notes=(
            f"a 50-image reservoir filling the budget reaches {np.mean(cont_res50):.3f} ± "
            f"{np.std(cont_res50, ddof=1):.3f}. the sign-flip p of 0.0625 is the floor for 5 seeds: the "
            "interval, not the p-value, carries this result. replay doubles the compute per step, which the "
            "memory budget doesn't count."
        ),
    )
    continual_methods = {
        "fine-tuning": {"memory": [], "scores": cont_ft},
        "Gaussian generator": {"memory": [(10 * 2 * 784, "float32")], "scores": cont_gauss},
        "reservoir, 50 images as float32": {"memory": [(50 * 784, "float32")], "scores": cont_res50},
        "reservoir, 200 images as uint8": {"memory": [(200 * 784, "uint8")], "scores": cont_res200},
    }
    _in_floats = budget_table(continual_methods, 39_200, unit="floats")
    _in_bytes = budget_table(continual_methods, 39_200 * 4, unit="bytes")
    mo.md(render_entry(entry_continual)
          + "\n\n**the same budget in two units.** stored numbers:\n\n" + render_budget_table(_in_floats)
          + "\n\nbytes (39,200 float32s = 156,800 bytes):\n\n" + render_budget_table(_in_bytes)
          + f"\n\nwinner in stored numbers: **{_in_floats['winner']}**; in bytes: **{_in_bytes['winner']}**. "
          "the bytes winner beats the generator by "
          f"{np.mean(cont_res200) - np.mean(cont_gauss):+.3f}, inside the seed-to-seed noise: "
          "say which unit your budget uses, and why.")
    return (entry_continual,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the portfolio's claims, corrected together

    each entry's primary comparison contributes one p-value; Holm keeps the chance of *any*
    false claim in the portfolio at $\alpha$.
    """)
    return


@app.cell
def _(entry_causal, entry_continual):
    _entries = [entry_causal, entry_continual]
    _reject, _adj = holm([e["p"] for e in _entries], alpha=0.05)
    mo.md("| elective | verdict (interval) | p | Holm-adjusted p | rejects at 0.05 |\n|---|---|---|---|---|\n"
          + "\n".join(f"| {e['block']} | {e['verdict']} | {e['p']:.3g} | {a:.3g} | {'yes' if r else 'no'} |"
                      for e, a, r in zip(_entries, _adj, _reject)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    one entry per elective you completed (at least two), each built with `portfolio_entry`
    from your own block project's per-seed scores, followed by a short report. for each
    elective, answer:

    1. **the question and the baseline.** what does the project ask, and why is your baseline
       the right classical comparison (the strongest simple method a skeptic would try first)?
    2. **the comparison.** is it fair: same seeds, same data, same budget (state the unit:
       floats or bytes, steps or wall-clock, parameters or FLOPs)? what does the budget not
       count?
    3. **the result.** the rendered entry, plus one figure from your project. is the verdict
       "better", "worse" or "inconclusive", and what would it take to change it (more seeds,
       a different unit, a harder setting)?
    4. **the setup's thumb on the scale.** in what way does your simulator or semi-synthetic
       world favour the method (or the baseline)? name one setting where you'd expect the
       ranking to flip.
    5. **across the portfolio.** apply Holm to the primary p-values. which claims survive, and
       which rest on the interval alone?

    see [`rubric.md`](rubric.md).
    """)
    return


if __name__ == "__main__":
    app.run()
