import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.bayes import Factor  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # exact inference · build

    work through the stages in order. everything here operates on lists of your `Factor`s
    from topic 02: the model is their product, and inference means summing it without ever
    building the full joint table. `variable_elimination` and `sum_product` become part of
    `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `eliminate`: sum one variable out of a factor list |
    | 2 | `min_degree_order`, `elimination_cost`: choose an order, and predict what it costs |
    | 3 | `variable_elimination` → `mlfs.bayes`: any marginal or conditional, exactly |
    | 4 | `sum_product` → `mlfs.bayes`: every marginal of a tree at once, by message passing |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def eliminate(factors, var):
    """stage 1: multiply the factors that mention var, sum var out; returns the new list (the others first, in order)."""
    raise NotImplementedError("stage1: implement eliminate")


@app.cell(hide_code=True)
def _():
    checks.run(1, eliminate=eliminate, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def min_degree_order(factors, variables):
    """stage 2: a greedy elimination order for `variables`: always the one with fewest neighbours (ties: earliest in `variables`)."""
    raise NotImplementedError("stage2: implement min_degree_order")


@app.function
def elimination_cost(factors, order):
    """stage 2: the number of entries of the largest product factor formed when eliminating `order` (no arithmetic)."""
    raise NotImplementedError("stage2: implement elimination_cost")


@app.cell(hide_code=True)
def _():
    checks.run(2, min_degree_order=min_degree_order, elimination_cost=elimination_cost, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def variable_elimination(factors, query, evidence=None, order=None):
    """stage 3: P(query | evidence) as a normalized Factor over query (a name or tuple), eliminating the rest in `order`."""
    raise NotImplementedError("stage3: implement variable_elimination")


@app.cell(hide_code=True)
def _():
    checks.run(3, variable_elimination=variable_elimination, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def sum_product(factors, evidence=None):
    """stage 4: belief propagation on a tree-shaped factor graph: dict var -> normalized marginal (card,) given evidence."""
    raise NotImplementedError("stage4: implement sum_product")


@app.cell(hide_code=True)
def _():
    checks.run(4, sum_product=sum_product, Factor=Factor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `variable_elimination` and `sum_product` are part of
    `mlfs.bayes`. then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
