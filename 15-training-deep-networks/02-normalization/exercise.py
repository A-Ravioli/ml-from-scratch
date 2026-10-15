import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.nn import Module, Parameter  # yours, from ch 14 · topic 04
    from mlfs.tensor import Tensor  # yours, from ch 14 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # normalization · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. initialization sets a good scale at step
    0; normalization layers *keep* it, at every step, by measuring the activations and
    rescaling them. you'll derive their backward passes by hand (one formula serves all
    three) and plug them into your engine with `Tensor.from_op`, then the layers join
    **`mlfs.train`** with PyTorch's names, defaults and running-statistics rules.

    | stage | you build |
    |---|---|
    | 1 | `standardize` and `standardize_backward`: the shared core, forward and backward |
    | 2 | `BatchNorm1d`: batch statistics in training, running statistics in evaluation |
    | 3 | `LayerNorm`: the same normalization along the features of each example |
    | 4 | `RMSNorm`: no mean, no bias, just a scale |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def standardize(x, axis, eps=1e-5):
    """stage 1: (xhat, inv_std): xhat = (x - mean) * inv_std along `axis`, inv_std = 1/sqrt(biased var + eps) kept with keepdims."""
    raise NotImplementedError("stage1: implement standardize")


@app.function
def standardize_backward(g, xhat, inv_std, axis):
    """stage 1: dL/dx given g = dL/dxhat: inv_std * (g - mean(g) - xhat * mean(g * xhat)), means along `axis`."""
    raise NotImplementedError("stage1: implement standardize_backward")


@app.cell(hide_code=True)
def _():
    checks.run(1, standardize=standardize, standardize_backward=standardize_backward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class BatchNorm1d(Module):
    """normalizes each feature over the batch (N, C); learnable weight/bias; running statistics for evaluation."""

    def __init__(self, num_features, *, eps=1e-5, momentum=0.1):
        """stage 2: store num_features, eps, momentum; weight = ones, bias = zeros (Parameters); running_mean = 0s, running_var = 1s (ndarrays)."""
        raise NotImplementedError("stage2: implement BatchNorm1d.__init__")

    def forward(self, x):
        """stage 2: train: batch stats (and update running stats, unbiased var); eval: running stats. one Tensor.from_op node."""
        raise NotImplementedError("stage2: implement BatchNorm1d.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, BatchNorm1d=BatchNorm1d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class LayerNorm(Module):
    """normalizes the last dimension of each example; learnable weight/bias; same in training and evaluation."""

    def __init__(self, num_features, *, eps=1e-5):
        """stage 3: store num_features and eps; weight = ones, bias = zeros, both Parameters of shape (num_features,)."""
        raise NotImplementedError("stage3: implement LayerNorm.__init__")

    def forward(self, x):
        """stage 3: (..., C) -> (..., C) normalized along the last axis, then * weight + bias; one Tensor.from_op node."""
        raise NotImplementedError("stage3: implement LayerNorm.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, LayerNorm=LayerNorm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class RMSNorm(Module):
    """divides each example by the root mean square of its last dimension; learnable weight, no bias."""

    def __init__(self, num_features, *, eps=1e-6):
        """stage 4: store num_features and eps; weight = ones, a Parameter of shape (num_features,); no bias."""
        raise NotImplementedError("stage4: implement RMSNorm.__init__")

    def forward(self, x):
        """stage 4: x * r * weight with r = 1/sqrt(mean(x^2, last axis) + eps); one Tensor.from_op node."""
        raise NotImplementedError("stage4: implement RMSNorm.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, RMSNorm=RMSNorm)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.train import BatchNorm1d, LayerNorm, RMSNorm` gives you these layers, and
    `Sequential(Linear(784, 128), BatchNorm1d(128), ReLU(), ...)` just works.
    """)
    return


if __name__ == "__main__":
    app.run()
