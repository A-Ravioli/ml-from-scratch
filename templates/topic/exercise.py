import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # __TITLE__ · build

    <!-- TEMPLATE: replace the mean/variance example with this topic's stages. -->

    work through the stages in order. each stage shows its part of the lesson, gives you a
    function to write, and checks your code live as you edit.

    | stage | you build |
    |---|---|
    | 1 | `mean`: the sample mean |
    | 2 | `variance`: the (biased) sample variance |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def mean(x: np.ndarray) -> float:
    """stage 1: the arithmetic mean of a 1-D array `x` (shape (n,), n >= 1)."""
    raise NotImplementedError("stage1: implement mean(x)")


@app.cell(hide_code=True)
def _():
    checks.run(1, mean=mean)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def variance(x: np.ndarray) -> float:
    """stage 2: the biased sample variance (divide by n) of a 1-D array `x`."""
    raise NotImplementedError("stage2: implement variance(x)")


@app.cell(hide_code=True)
def _():
    checks.run(2, variance=variance)
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
