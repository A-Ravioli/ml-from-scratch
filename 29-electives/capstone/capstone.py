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
    # 29 · capstone: the elective portfolio

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

    when they're green, scroll down: you add one entry per elective you completed (the brief
    shows two worked entries, for causal inference and continual learning), and the write-up
    prompts are at the end.
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
    raise NotImplementedError("stage1: implement paired_differences")


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
    raise NotImplementedError("stage2: implement paired_ci")


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
    raise NotImplementedError("stage3: implement sign_flip_test")


@app.function
def min_p_value(n, alternative="two-sided"):
    """stage 3: the smallest p-value an exact sign-flip test on n paired seeds can return."""
    raise NotImplementedError("stage3: implement min_p_value")


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
    raise NotImplementedError("stage4: implement holm")


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
    raise NotImplementedError("stage5: implement memory_cost")


@app.function
def budget_table(methods, budget, *, unit="bytes", higher_is_better=True):
    """stage 5: dict(rows, winner): one row per method (name, cost, fits, mean, sd, n), fitting rows first, best first;
    winner = best method with cost <= budget (None if none fits)."""
    raise NotImplementedError("stage5: implement budget_table")


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
    raise NotImplementedError("stage6: implement portfolio_entry")


@app.function
def render_entry(entry):
    """stage 6: the entry as markdown, in the fixed format of brief.md."""
    raise NotImplementedError("stage6: implement render_entry")


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
    ## your portfolio entries

    one `portfolio_entry` per elective you completed, built from your own block project's
    per-seed scores (dicts `{seed: score}`, so the pairing is explicit). the brief shows two
    worked entries in the final format, for the causal-inference and continual-learning
    projects. for a budgeted comparison, build a `budget_table` too and say which unit it uses.
    """)
    return


@app.cell
def _():
    # append one entry per elective, for example:
    #
    # portfolio_entry(
    #     "continual learning (topics 12–14)",
    #     "with 39,200 stored numbers of memory, can a replay strategy beat fine-tuning?",
    #     "my strategy", {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},   # your per-seed scores
    #     "fine-tuning", {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
    #     metric="average accuracy after task 5 (ACC)", budget="...", notes="...", rng=0,
    # )
    my_entries = []
    mo.md("\n\n---\n\n".join(render_entry(_e) for _e in my_entries) if my_entries
          else "*no entries yet: add one per elective you completed.*")
    return (my_entries,)


@app.cell
def _(my_entries):
    if len(my_entries) >= 2:
        _reject, _adj = holm([_e["p"] for _e in my_entries], alpha=0.05)
        _out = mo.md(
            "| elective | verdict (interval) | p | Holm-adjusted p | rejects at 0.05 |\n|---|---|---|---|---|\n"
            + "\n".join(f"| {_e['block']} | {_e['verdict']} | {_e['p']:.3g} | {_a:.3g} | {'yes' if _r else 'no'} |"
                        for _e, _a, _r in zip(my_entries, _adj, _reject)))
    else:
        _out = mo.md("*with two or more entries, this cell applies Holm's correction to their primary p-values.*")
    _out
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
