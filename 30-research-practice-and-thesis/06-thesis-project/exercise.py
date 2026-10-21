import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from ch 00 · 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # thesis project · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; how you structure the
    code is up to you. together they make a planner for a project whose tasks you can't
    estimate exactly: the critical path of a plan, the probability of finishing it by a
    deadline, and a forecast calibrated on how wrong your own past estimates were.

    | stage | you build |
    |---|---|
    | 1 | `topo_order`, `critical_path`: earliest finishes, slack and the critical path of a task graph |
    | 2 | `simulate_makespan`, `schedule_risk`: Monte Carlo finish dates from three-point estimates |
    | 3 | `forecast`: a schedule forecast resampled from your own estimate-to-actual history |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def topo_order(tasks):
    """stage 1: task names, each after all the tasks it comes after (ties alphabetical); ValueError on cycles."""
    raise NotImplementedError("stage1: implement topo_order")


@app.function
def critical_path(tasks):
    """stage 1: dict(start, finish, slack, total, path) for tasks {name: {'weeks', 'after'}} started as early as possible."""
    raise NotImplementedError("stage1: implement critical_path")


@app.cell(hide_code=True)
def _():
    checks.run(1, topo_order=topo_order, critical_path=critical_path)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def simulate_makespan(tasks, durations):
    """stage 2: (makespan (n_sims,), critical {name: bool (n_sims,)}) for sampled durations {name: (n_sims,)}."""
    raise NotImplementedError("stage2: implement simulate_makespan")


@app.function
def schedule_risk(tasks, deadline, *, n_sims=10000, rng=None):
    """stage 2: dict(p_on_time, quantiles, criticality, likely_total) from triangular (o, m, p) estimates."""
    raise NotImplementedError("stage2: implement schedule_risk")


@app.cell(hide_code=True)
def _():
    checks.run(2, simulate_makespan=simulate_makespan, schedule_risk=schedule_risk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def forecast(tasks, history, deadline, *, n_sims=10000, rng=None):
    """stage 3: dict(p_on_time, quantiles, naive_total, median_ratio): each estimate × a ratio resampled from history."""
    raise NotImplementedError("stage3: implement forecast")


@app.cell(hide_code=True)
def _():
    checks.run(3, forecast=forecast)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. then write your thesis plan
    as a task graph with three-point estimates, and keep a log of (estimate, actual) pairs as
    you finish tasks: after a month, `forecast` tells you when you'll really be done.
    """)
    return


if __name__ == "__main__":
    app.run()
