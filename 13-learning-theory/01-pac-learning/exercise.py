import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PAC learning · build

    work through the stages in order. you'll build a sample-complexity calculator for finite
    hypothesis classes, then put it to the test on the simplest infinite class there is:
    thresholds on the line, $h_t(x) = \mathbb 1[x \ge t]$, with $x \sim \text{Uniform}[0, 1]$.
    you'll learn them by empirical risk minimization, compute the exact probability that the
    learner fails, and estimate the same probability by simulation.

    | stage | you build |
    |---|---|
    | 1 | `finite_class_sample_complexity` → `mlfs.theory`: how many examples a finite class needs |
    | 2 | `erm_threshold`: empirical risk minimization over all thresholds, in $O(n\log n)$ |
    | 3 | `threshold_risk`, `threshold_failure_probability`: the true risk, and the exact chance of an $\varepsilon$-bad hypothesis |
    | 4 | `estimate_failure_probability`: the same chance, by Monte Carlo, with or without label noise |
    | 5 | `empirical_sample_complexity`: the smallest sample size that works, measured |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def finite_class_sample_complexity(class_size, epsilon, delta, *, realizable=True):
    """stage 1: smallest int m with ln(|H|/delta)/epsilon <= m (realizable) or ln(2|H|/delta)/(2 epsilon^2) <= m (agnostic)."""
    raise NotImplementedError("stage1: implement finite_class_sample_complexity")


@app.cell(hide_code=True)
def _():
    checks.run(1, finite_class_sample_complexity=finite_class_sample_complexity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def erm_threshold(x, y):
    """stage 2: the threshold t among the distinct x values and +inf minimizing the training error of 1[x >= t] (ties: smallest t)."""
    raise NotImplementedError("stage2: implement erm_threshold")


@app.cell(hide_code=True)
def _():
    checks.run(2, erm_threshold=erm_threshold)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def threshold_risk(t, t_star, noise=0.0):
    """stage 3: true 0-1 risk of 1[x >= t] when x ~ U[0, 1] and the label 1[x >= t_star] is flipped with probability noise."""
    raise NotImplementedError("stage3: implement threshold_risk")


@app.function
def threshold_failure_probability(m, epsilon, t_star=0.5):
    """stage 3: exact P(threshold_risk(erm_threshold(sample)) > epsilon) for m noiseless examples, x ~ U[0, 1]."""
    raise NotImplementedError("stage3: implement threshold_failure_probability")


@app.cell(hide_code=True)
def _():
    checks.run(3, threshold_risk=threshold_risk, threshold_failure_probability=threshold_failure_probability,
               erm_threshold=erm_threshold)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def estimate_failure_probability(m, epsilon, *, t_star=0.5, noise=0.0, n_trials=1000, rng=None):
    """stage 4: fraction of n_trials simulated samples of size m on which the ERM threshold's excess risk exceeds epsilon."""
    raise NotImplementedError("stage4: implement estimate_failure_probability")


@app.cell(hide_code=True)
def _():
    checks.run(4, estimate_failure_probability=estimate_failure_probability, erm_threshold=erm_threshold,
               threshold_risk=threshold_risk, threshold_failure_probability=threshold_failure_probability)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def empirical_sample_complexity(epsilon, delta, m_grid, *, t_star=0.5, noise=0.0, n_trials=1000, rng=None):
    """stage 5: the first m in sorted(m_grid) whose estimated failure probability is <= delta, or None if none is."""
    raise NotImplementedError("stage5: implement empirical_sample_complexity")


@app.cell(hide_code=True)
def _():
    checks.run(5, empirical_sample_complexity=empirical_sample_complexity,
               estimate_failure_probability=estimate_failure_probability, erm_threshold=erm_threshold,
               threshold_risk=threshold_risk, threshold_failure_probability=threshold_failure_probability)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your calculator is part of `mlfs.theory` (try
    `from mlfs.theory import finite_class_sample_complexity` in a scratch cell). then open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
