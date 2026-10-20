import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # approximation theory · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. everything here is a plain `nn.Sequential` of `Linear` and `ReLU` layers whose
    weights you set by hand or train.

    | stage | you build |
    |---|---|
    | 1 | `relu_interpolant`: a one-hidden-layer ReLU network through any set of points |
    | 2 | `sawtooth_net`: a deep, width-2 network with $2^k$ linear pieces from $6k + 1$ weights |
    | 3 | `count_linear_pieces`: how many linear pieces a 1-D network really has |
    | 4 | `width_for_budget`: the width that gives an MLP of a given depth a given parameter count |
    | 5 | `fit_regressor`: the training loop for the depth-separation experiments |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def relu_interpolant(knots_x, knots_y):
    """stage 1: nn.Sequential(Linear(1, n-1), ReLU(), Linear(n-1, 1)) equal to the piecewise-linear interpolant of n >= 2 knots (x strictly increasing)."""
    raise NotImplementedError("stage1: implement relu_interpolant")


@app.cell(hide_code=True)
def _():
    checks.run(1, relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sawtooth_net(k):
    """stage 2: nn.Sequential with k hidden ReLU layers of width 2 computing the tent map applied k times on [0, 1]; 6k + 1 parameters."""
    raise NotImplementedError("stage2: implement sawtooth_net")


@app.cell(hide_code=True)
def _():
    checks.run(2, sawtooth_net=sawtooth_net)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def count_linear_pieces(f, lo=0.0, hi=1.0, *, n_grid=16385, tol=1e-4):
    """stage 3: 1 + number of maximal runs of grid points where |second difference| > tol * max |first difference|, for f on a float64 linspace(lo, hi, n_grid) (modules run on a float64 copy)."""
    raise NotImplementedError("stage3: implement count_linear_pieces")


@app.cell(hide_code=True)
def _():
    checks.run(3, count_linear_pieces=count_linear_pieces, sawtooth_net=sawtooth_net, relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def mlp_param_count(width, depth, *, in_dim=1, out_dim=1):
    """stage 4: parameters of `depth` hidden layers of `width` (Linear + ReLU) followed by a Linear output."""
    raise NotImplementedError("stage4: implement mlp_param_count")


@app.function
def width_for_budget(budget, depth, *, in_dim=1, out_dim=1):
    """stage 4: the width w >= 1 whose mlp_param_count is closest to budget (ties -> the smaller w)."""
    raise NotImplementedError("stage4: implement width_for_budget")


@app.cell(hide_code=True)
def _():
    checks.run(4, width_for_budget=width_for_budget, mlp_param_count=mlp_param_count)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def fit_regressor(model, x, y, *, steps=2000, lr=3e-3, batch_size=256, seed=0):
    """stage 5: Adam on minibatch MSE, lr cosine-decayed lr * (1 + cos(pi t / steps)) / 2 at step t = 0..steps-1, batches drawn with replacement from a generator seeded with seed; train mode, then eval; returns the steps losses."""
    raise NotImplementedError("stage5: implement fit_regressor")


@app.cell(hide_code=True)
def _():
    checks.run(5, fit_regressor=fit_regressor)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it pits deep networks
    against shallow ones at equal parameter counts, on a function depth makes easy to *represent*
    and on one it makes easy to *learn*.
    """)
    return


if __name__ == "__main__":
    app.run()
