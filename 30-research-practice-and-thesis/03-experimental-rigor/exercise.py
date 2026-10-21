import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections.abc import Mapping

    import marimo as mo
    import numpy as np
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.stats import bootstrap_ci, permutation_test  # yours, from ch 05
    from mlfs.utils import as_rng  # yours, from ch 00 · 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # experimental rigor · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. these five functions become `mlfs.research`, which every later topic
    in this chapter, the thesis capstone and your own projects use.

    | stage | you build |
    |---|---|
    | 1 | `run_seeds`: one experiment, many seeds, results collected per metric → `mlfs.research` |
    | 2 | `mean_ci`: a mean with a $t$ or bootstrap interval → `mlfs.research` |
    | 3 | `compare_runs`: a difference between two methods with an interval and a p-value → `mlfs.research` |
    | 4 | `ablation_table`: a Markdown ablation table with intervals and paired differences → `mlfs.research` |
    | 5 | `expected_best`: the expected best score of $k$ tuning trials (budget-aware baselines) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def run_seeds(fn, seeds=(0, 1, 2, 3, 4)):
    """stage 1: call fn(seed) once per seed, in order; scalars -> float ndarray (n_seeds,), dicts -> dict key -> ndarray."""
    raise NotImplementedError("stage1: implement run_seeds")


@app.cell(hide_code=True)
def _():
    checks.run(1, run_seeds=run_seeds)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def mean_ci(values, *, alpha=0.05, method="t", rng=None):
    """stage 2: (mean, lo, hi): a Student-t (n - 1 dof) or bootstrap (mlfs.stats.bootstrap_ci) interval for the mean."""
    raise NotImplementedError("stage2: implement mean_ci")


@app.cell(hide_code=True)
def _():
    checks.run(2, mean_ci=mean_ci)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def compare_runs(a, b, *, paired=True, alpha=0.05, n_permutations=10000, rng=None):
    """stage 3: dict(diff, ci, p_value, n_a, n_b, paired) for a vs b: bootstrap CI and permutation-test p."""
    raise NotImplementedError("stage3: implement compare_runs")


@app.cell(hide_code=True)
def _():
    checks.run(3, compare_runs=compare_runs)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ablation_table(results, *, baseline=None, alpha=0.05, higher_is_better=True, fmt="{:.3f}"):
    """stage 4: a Markdown table: mean ± t-CI per config, n, and (with a baseline) the paired difference ± CI; best row bold."""
    raise NotImplementedError("stage4: implement ablation_table")


@app.cell(hide_code=True)
def _():
    checks.run(4, ablation_table=ablation_table)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def expected_best(scores, k, *, higher_is_better=True):
    """stage 5: expected best of k trials drawn with replacement from the observed trial scores (Dodge et al., 2019)."""
    raise NotImplementedError("stage5: implement expected_best")


@app.cell(hide_code=True)
def _():
    checks.run(5, expected_best=expected_best)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.research import run_seeds, mean_ci, compare_runs, ablation_table` gives every
    experiment you run (your thesis included) seeds, intervals and paired comparisons in four
    lines.
    """)
    return


if __name__ == "__main__":
    app.run()
