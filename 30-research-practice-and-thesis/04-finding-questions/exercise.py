import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import re

    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.dltheory import compute_optimal  # yours, from ch 19 · topic 05

    # wording that makes a hypothesis hard to refute: activities instead of predictions, and hedges
    VAGUE = re.compile(
        r"\b(explore|investigate|study|understand(ing)?|shed light|look into|examine|analy[sz]e"
        r"|may|might|could|possibly|potentially|in some cases|interesting)\b",
        re.IGNORECASE,
    )
    # the proposal's sections, after the Heilmeier catechism
    SECTIONS = ("title", "goal", "state_of_the_art", "approach", "impact")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # finding questions · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make a proposal checker: is the hypothesis falsifiable,
    are there enough seeds to detect the effect you care about, does the compute fit, and is
    the proposal complete and short?

    | stage | you build |
    |---|---|
    | 1 | `check_hypothesis`: a linter for falsifiable hypotheses |
    | 2 | `paired_power`, `seeds_for_power`: the power of a paired $t$ test, and the seeds it needs |
    | 3 | `run_hours`, `largest_model`: compute in hours, and the biggest compute-optimal model you can afford |
    | 4 | `budget_plan`: does the whole experiment plan fit the budget, with a safety factor? |
    | 5 | `check_proposal`: the two-page proposal, checked end to end |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def check_hypothesis(h):
    """stage 1: list of problems ('field: message') with a hypothesis dict; [] means it's falsifiable as stated."""
    raise NotImplementedError("stage1: implement check_hypothesis")


@app.cell(hide_code=True)
def _():
    checks.run(1, check_hypothesis=check_hypothesis)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def paired_power(sd_diff, effect, n, *, alpha=0.05, alternative="greater"):
    """stage 2: power of a paired t test with n pairs to detect a true mean difference effect (noncentral t)."""
    raise NotImplementedError("stage2: implement paired_power")


@app.function
def seeds_for_power(sd_diff, min_effect, *, alpha=0.05, power=0.8, alternative="greater", max_seeds=1000):
    """stage 2: smallest n >= 2 with paired_power(sd_diff, min_effect, n) >= power; ValueError past max_seeds."""
    raise NotImplementedError("stage2: implement seeds_for_power")


@app.cell(hide_code=True)
def _():
    checks.run(2, paired_power=paired_power, seeds_for_power=seeds_for_power)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def run_hours(n_params, n_tokens, *, flops_per_second, utilization=0.4):
    """stage 3: hours to train n_params on n_tokens at 6 N D FLOPs, achieving utilization × flops_per_second."""
    raise NotImplementedError("stage3: implement run_hours")


@app.function
def largest_model(hours, *, flops_per_second, utilization=0.4, tokens_per_param=20.0):
    """stage 3: (n_params, n_tokens) compute-optimal for the FLOPs in `hours`, via mlfs.dltheory.compute_optimal."""
    raise NotImplementedError("stage3: implement largest_model")


@app.cell(hide_code=True)
def _():
    checks.run(3, run_hours=run_hours, largest_model=largest_model)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def budget_plan(experiments, budget_hours, *, safety=2.0):
    """stage 4: dict(rows, total, with_safety, budget, fits): runs and hours per experiment, total × safety vs budget."""
    raise NotImplementedError("stage4: implement budget_plan")


@app.cell(hide_code=True)
def _():
    checks.run(4, budget_plan=budget_plan)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def check_proposal(proposal, *, max_words=1000, alpha=0.05, power=0.8, safety=2.0):
    """stage 5: list of problems with a proposal: sections, length, hypothesis, power and budget; [] means ready."""
    raise NotImplementedError("stage5: implement check_proposal")


@app.cell(hide_code=True)
def _():
    checks.run(5, check_proposal=check_proposal)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then write your own
    two-page proposal as a dict (or load it from a file) and get `check_proposal` to return
    `[]` before you show it to anyone.
    """)
    return


if __name__ == "__main__":
    app.run()
