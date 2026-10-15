import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.nn import Linear, ReLU, Sequential, fit, mse_loss, predict  # yours, from topic 04
    from mlfs.optim import Adam  # yours, from ch 06 · topic 04
    from mlfs.tensor import no_grad  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # universal approximation · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll prove the universal approximation
    theorem for ReLU networks in one dimension *by construction* (a network that hits any
    set of points exactly), then build a deep, narrow network that a shallow one needs
    exponentially many units to match, and measure both with a tool that counts linear
    pieces. everything is built from your `mlfs.nn` layers.

    | stage | you build |
    |---|---|
    | 1 | `set_linear` and `relu_interpolant`: a one-hidden-layer ReLU network through any points |
    | 2 | `sawtooth_net`: the tent map composed `depth` times, with 2 units per layer |
    | 3 | `count_linear_regions`: how many linear pieces a ReLU network has along a line |
    | 4 | `fit_function` and `max_error`: train a network to approximate a 1-D function |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def set_linear(layer, W, b):
    """stage 1: copy W (out, in) and b (out,) into a Linear layer's weight and bias IN PLACE; returns the layer."""
    raise NotImplementedError("stage1: implement set_linear")


@app.function
def relu_interpolant(knots, values):
    """stage 1: Sequential(Linear(1, k), ReLU(), Linear(k, 1)) through the k >= 2 points (knots, values), linear beyond them."""
    raise NotImplementedError("stage1: implement relu_interpolant")


@app.cell(hide_code=True)
def _():
    checks.run(1, set_linear=set_linear, relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sawtooth_net(depth):
    """stage 2: a width-2 ReLU network computing the tent map T(x) = 2 relu(x) - 4 relu(x - 1/2) composed depth times."""
    raise NotImplementedError("stage2: implement sawtooth_net")


@app.cell(hide_code=True)
def _():
    checks.run(2, sawtooth_net=sawtooth_net, set_linear=set_linear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def count_linear_regions(model, x):
    """stage 3: along the sorted 1-D points x, 1 + the number of times the on/off pattern of all ReLUs in the Sequential changes."""
    raise NotImplementedError("stage3: implement count_linear_regions")


@app.cell(hide_code=True)
def _():
    checks.run(3, count_linear_regions=count_linear_regions, sawtooth_net=sawtooth_net,
               relu_interpolant=relu_interpolant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fit_function(model, f, *, lo=0.0, hi=1.0, n_points=256, epochs=500, lr=0.01, batch_size=64, rng=None):
    """stage 4: train model on (x, f(x)) for n_points evenly spaced x in [lo, hi] with Adam and MSE; returns fit's history."""
    raise NotImplementedError("stage4: implement fit_function")


@app.function
def max_error(model, f, *, lo=0.0, hi=1.0, n_points=2001):
    """stage 4: max |model(x) - f(x)| over n_points evenly spaced x in [lo, hi] (the sup-norm error the theorem is about)."""
    raise NotImplementedError("stage4: implement max_error")


@app.cell(hide_code=True)
def _():
    checks.run(4, fit_function=fit_function, max_error=max_error, relu_interpolant=relu_interpolant)
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
