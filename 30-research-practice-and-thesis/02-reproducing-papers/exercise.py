import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson

    # the reproducibility checklist the report renders: (id, item). adapted from the ML reproducibility
    # checklist of Pineau et al. (2021); answer each one in your report.
    CHECKLIST = [
        ("claims", "every claim tested is stated with the paper's evidence and a success criterion fixed before the runs"),
        ("gaps", "every detail the paper leaves unspecified is listed with the choice made and why"),
        ("scale", "the scale-down is argued: what is preserved, what is shrunk, and why the claim should survive"),
        ("code", "the code, configurations and the command for each figure and table are available"),
        ("data", "the data, its version, preprocessing and splits are described or provided"),
        ("seeds", "the number of seeds per configuration and how seeds were chosen are stated"),
        ("stats", "every reported number has an interval, and the method used to compute it is stated"),
        ("compute", "hardware, wall-clock time per run and the total compute are reported"),
        ("versions", "library versions (and the authors' code version, if consulted) are recorded"),
        ("deviations", "every deviation from the paper, and every change made after reading the authors' code, is listed"),
    ]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # reproducing papers · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make the analysis half of the reproduction protocol: a
    verdict for a reproduced number, a verdict for a reproduced comparison, how many seeds you
    need for a verdict to be possible, and a report generator.

    | stage | you build |
    |---|---|
    | 1 | `equivalence_verdict`: does my number match the paper's within a margin? |
    | 2 | `effect_verdict`: does the paper's comparison (A beats B by Δ) hold? |
    | 3 | `seeds_needed`: how many seeds before "reproduced" is even possible? |
    | 4 | `reproduction_report`: the claims, verdicts, gaps and checklist, as Markdown |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def equivalence_verdict(runs, paper_value, *, margin, alpha=0.05):
    """stage 1: dict(mean, ci, n, verdict): the (1 - 2 alpha) t interval of runs against paper_value ± margin."""
    raise NotImplementedError("stage1: implement equivalence_verdict")


@app.cell(hide_code=True)
def _():
    checks.run(1, equivalence_verdict=equivalence_verdict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def effect_verdict(runs, baseline_runs, paper_effect, *, margin, alpha=0.05, paired=True):
    """stage 2: dict(diff, ci, n, verdict) for the comparison runs - baseline against paper_effect ± margin."""
    raise NotImplementedError("stage2: implement effect_verdict")


@app.cell(hide_code=True)
def _():
    checks.run(2, effect_verdict=effect_verdict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def seeds_needed(sd, margin, *, alpha=0.05, power=0.8, max_seeds=1000):
    """stage 3: smallest n >= 2 with (t_{1-alpha, n-1} + z_{(1+power)/2}) sd / sqrt(n) <= margin."""
    raise NotImplementedError("stage3: implement seeds_needed")


@app.cell(hide_code=True)
def _():
    checks.run(3, seeds_needed=seeds_needed)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def reproduction_report(title, claims, results, *, gaps=(), checklist=None, alpha=0.05):
    """stage 4: (markdown, verdicts): summary table, claims, gaps and checklist; verdicts maps claim id -> verdict."""
    raise NotImplementedError("stage4: implement reproduction_report")


@app.cell(hide_code=True)
def _():
    checks.run(4, reproduction_report=reproduction_report)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. for R1, R2 and R3, write
    your claims table as a list of claim dicts *before* the runs, and let
    `reproduction_report` render the summary once they're done.
    """)
    return


if __name__ == "__main__":
    app.run()
