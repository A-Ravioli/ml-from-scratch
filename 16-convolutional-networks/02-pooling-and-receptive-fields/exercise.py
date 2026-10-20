import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.cnn import as_pair, col2im, im2col  # yours, from topic 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pooling & receptive fields · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. a convolution sees a few pixels; a
    network needs to see a whole object. this topic is about how that happens: pooling
    layers (and their backward passes, built on your `im2col`/`col2im`), the arithmetic
    of how far each output can see, and a measurement of how far it *actually* looks.

    | stage | you build |
    |---|---|
    | 1 | `max_pool2d`, `avg_pool2d`: downsampling by taking the max or mean of each window (NumPy) |
    | 2 | `max_pool2d_backward`, `avg_pool2d_backward`: where pooling sends its gradients |
    | 3 | `receptive_field`, `rf_layers`: how many input pixels one output can depend on |
    | 4 | `effective_receptive_field`, `erf_radius`: how many it actually depends on (PyTorch autograd) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def max_pool2d(x, kernel_size, stride=None):
    """stage 1: the max of every kernel_size window, stride defaults to kernel_size, no padding: (N, C, H, W) -> (N, C, H_out, W_out)."""
    raise NotImplementedError("stage1: implement max_pool2d")


@app.function
def avg_pool2d(x, kernel_size, stride=None):
    """stage 1: the mean of every kernel_size window, stride defaults to kernel_size, no padding."""
    raise NotImplementedError("stage1: implement avg_pool2d")


@app.cell(hide_code=True)
def _():
    checks.run(1, max_pool2d=max_pool2d, avg_pool2d=avg_pool2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def max_pool2d_backward(dout, x, kernel_size, stride=None):
    """stage 2: dL/dx for max_pool2d: each window's gradient goes to its (first) max; overlapping windows add."""
    raise NotImplementedError("stage2: implement max_pool2d_backward")


@app.function
def avg_pool2d_backward(dout, x_shape, kernel_size, stride=None):
    """stage 2: dL/dx for avg_pool2d: each window's gradient is shared equally by its kh*kw pixels."""
    raise NotImplementedError("stage2: implement avg_pool2d_backward")


@app.cell(hide_code=True)
def _():
    checks.run(2, max_pool2d_backward=max_pool2d_backward, avg_pool2d_backward=avg_pool2d_backward,
               max_pool2d=max_pool2d, avg_pool2d=avg_pool2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def receptive_field(layers):
    """stage 3: layers = [(kernel_size, stride, dilation), ...] in order -> (rf, jump) of the last layer's outputs, in input pixels."""
    raise NotImplementedError("stage3: implement receptive_field")


@app.function
def rf_layers(model):
    """stage 3: the (kernel_size, stride, dilation) of every Conv2d, MaxPool2d and AvgPool2d in model.modules(), in order."""
    raise NotImplementedError("stage3: implement rf_layers")


@app.cell(hide_code=True)
def _():
    checks.run(3, receptive_field=receptive_field, rf_layers=rf_layers)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def effective_receptive_field(model, input_size, *, in_channels=1, n_samples=8, seed=0):
    """stage 4: mean |d(centre output, summed over channels) / d input| over n_samples Gaussian inputs, summed over input channels, normalized to sum 1 -> (H, W) ndarray."""
    raise NotImplementedError("stage4: implement effective_receptive_field")


@app.function
def erf_radius(erf, mass=0.9):
    """stage 4: the smallest r such that the (2r+1) x (2r+1) square centred at (H // 2, W // 2) holds at least `mass` of the erf."""
    raise NotImplementedError("stage4: implement erf_radius")


@app.cell(hide_code=True)
def _():
    checks.run(4, effective_receptive_field=effective_receptive_field, erf_radius=erf_radius)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on,
    `from mlfs.cnn import max_pool2d, receptive_field` gives you these tools, and you may use
    `torch.nn.MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` and their functional forms:
    you've built them.
    """)
    return


if __name__ == "__main__":
    app.run()
