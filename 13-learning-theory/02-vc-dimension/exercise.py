import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import math

    import marimo as mo
    import numpy as np
    from scipy.optimize import linprog

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # VC dimension · build

    work through the stages in order. a hypothesis class is represented here by a
    **realizability oracle** `realizable(X, y) -> bool`: can some member of the class produce
    exactly the labels `y` (0/1) on the points `X`? with one oracle per class you'll build a
    brute-force shattering checker, search for the VC dimension, and count the growth function
    against Sauer's lemma. every count is exponential in the number of points, so every
    function has a guard.

    | stage | you build |
    |---|---|
    | 1 | `interval_realizable`, `rectangle_realizable`: oracles for intervals on the line and axis-aligned boxes |
    | 2 | `halfspace_realizable`: the oracle for halfspaces, by linear programming |
    | 3 | `count_labelings`, `is_shattered`: enumerate all $2^n$ labelings |
    | 4 | `vc_dimension_search`: the largest shattered subset of a pool of points |
    | 5 | `sauer_bound`, `growth_lower_bound`: the growth function, measured and bounded |

    this topic adds nothing to `mlfs`: its lessons feed `vc_bound` in topic 04. you can also
    run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def interval_realizable(x, y):
    """stage 1: can 1[a <= x <= b] (or the empty interval) produce labels y on the points x (n,)?"""
    raise NotImplementedError("stage1: implement interval_realizable")


@app.function
def rectangle_realizable(X, y):
    """stage 1: can an axis-aligned box (or the empty box) produce labels y on the points X (n, d)?"""
    raise NotImplementedError("stage1: implement rectangle_realizable")


@app.cell(hide_code=True)
def _():
    checks.run(1, interval_realizable=interval_realizable, rectangle_realizable=rectangle_realizable)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def halfspace_realizable(X, y):
    """stage 2: can 1[w . x + b >= 0] produce labels y on X (n, d)? an LP feasibility problem: s_i (w . x_i + b) >= 1, s_i = +-1."""
    raise NotImplementedError("stage2: implement halfspace_realizable")


@app.cell(hide_code=True)
def _():
    checks.run(2, halfspace_realizable=halfspace_realizable)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def count_labelings(X, realizable, *, max_points=16):
    """stage 3: how many of the 2^n labelings of the n points X can realizable(X, y) produce? ValueError if n > max_points."""
    raise NotImplementedError("stage3: implement count_labelings")


@app.function
def is_shattered(X, realizable, *, max_points=16):
    """stage 3: True if realizable(X, y) holds for all 2^n labelings y (stop at the first failure). ValueError if n > max_points."""
    raise NotImplementedError("stage3: implement is_shattered")


@app.cell(hide_code=True)
def _():
    checks.run(3, count_labelings=count_labelings, is_shattered=is_shattered, interval_realizable=interval_realizable,
               rectangle_realizable=rectangle_realizable, halfspace_realizable=halfspace_realizable)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def vc_dimension_search(pool, realizable, *, max_d=5):
    """stage 4: (d, witness): the largest d <= max_d such that some d-subset of pool is shattered, and its indices (a tuple)."""
    raise NotImplementedError("stage4: implement vc_dimension_search")


@app.cell(hide_code=True)
def _():
    checks.run(4, vc_dimension_search=vc_dimension_search, is_shattered=is_shattered,
               interval_realizable=interval_realizable, rectangle_realizable=rectangle_realizable,
               halfspace_realizable=halfspace_realizable)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def sauer_bound(m, d):
    """stage 5: Sauer's bound on the growth function, sum_{i=0}^{d} C(m, i), as an exact int."""
    raise NotImplementedError("stage5: implement sauer_bound")


@app.function
def growth_lower_bound(pool, realizable, m, *, n_subsets=20, rng=None):
    """stage 5: max of count_labelings over n_subsets random m-point subsets of pool (a lower bound on the growth function)."""
    raise NotImplementedError("stage5: implement growth_lower_bound")


@app.cell(hide_code=True)
def _():
    checks.run(5, sauer_bound=sauer_bound, growth_lower_bound=growth_lower_bound, count_labelings=count_labelings,
               interval_realizable=interval_realizable, halfspace_realizable=halfspace_realizable)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
