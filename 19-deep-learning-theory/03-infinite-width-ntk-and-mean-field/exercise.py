import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # infinite width: NTK & mean field · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. stages 1–2 join **`mlfs.dltheory`**.

    | stage | you build |
    |---|---|
    | 1 | `empirical_ntk`: the neural tangent kernel of any scalar-output model → `mlfs.dltheory` |
    | 2 | `kernel_regression_predict`: (ridge) kernel regression from Gram matrices → `mlfs.dltheory` |
    | 3 | `NTKNet`, `relu_ntk`: a two-layer network in NTK or mean-field parameterization, and its infinite-width kernel |
    | 4 | `ntk_gd_predictions`: what gradient descent on the linearized network predicts, in closed form |
    | 5 | `gd_train`, `relative_change`: full-batch gradient descent, and how far the weights moved |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def empirical_ntk(model, x1, x2=None):
    """stage 1: (n1, n2) tensor K[i, j] = <d f(x1_i)/d theta, d f(x2_j)/d theta> over params with requires_grad; x2=None means x1."""
    raise NotImplementedError("stage1: implement empirical_ntk")


@app.cell(hide_code=True)
def _():
    checks.run(1, empirical_ntk=empirical_ntk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def kernel_regression_predict(K_train, y_train, K_test_train, *, ridge=0.0):
    """stage 2: K_test_train @ solve(K_train + ridge I, y_train); returns a tensor for tensor K_test_train, else an ndarray."""
    raise NotImplementedError("stage2: implement kernel_regression_predict")


@app.cell(hide_code=True)
def _():
    checks.run(2, kernel_regression_predict=kernel_regression_predict)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class NTKNet(nn.Module):
    """f(x) = scale * a . relu(W x): W (width, in_dim) and a (width,) ~ N(0, 1); scale 1/sqrt(width) ('ntk') or 1/width ('mf')."""

    def __init__(self, in_dim, width, *, parameterization="ntk"):
        """stage 3: parameters W (width, in_dim) and a (width,), standard normal; attributes width, parameterization."""
        raise NotImplementedError("stage3: implement NTKNet.__init__")

    def forward(self, x):
        """stage 3: x (n, in_dim) -> (n,)."""
        raise NotImplementedError("stage3: implement NTKNet.forward")


@app.function
def relu_ntk(X1, X2=None):
    """stage 3: the infinite-width NTK of NTKNet ('ntk'): |x||x'|/(2 pi) (sin t + (pi - t) cos t) + x.x' (pi - t)/(2 pi)."""
    raise NotImplementedError("stage3: implement relu_ntk")


@app.cell(hide_code=True)
def _():
    checks.run(3, NTKNet=NTKNet, relu_ntk=relu_ntk, empirical_ntk=empirical_ntk)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ntk_gd_predictions(K_train, K_test_train, y_train, f0_train, f0_test, *, lr, steps):
    """stage 4: (f_train, f_test) after `steps` GD steps of size lr on 1/2 sum (f - y)^2 for the model linearized at init."""
    raise NotImplementedError("stage4: implement ntk_gd_predictions")


@app.cell(hide_code=True)
def _():
    checks.run(4, ntk_gd_predictions=ntk_gd_predictions)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gd_train(model, x, y, *, lr, steps):
    """stage 5: `steps` plain full-batch GD steps on 1/2 sum (f(x) - y)^2, in place; returns the steps + 1 losses (before each step, then final)."""
    raise NotImplementedError("stage5: implement gd_train")


@app.function
def relative_change(model, initial_state):
    """stage 5: ||theta - theta_0|| / ||theta_0|| over model.named_parameters() (one global norm); initial_state maps names to tensors."""
    raise NotImplementedError("stage5: implement relative_change")


@app.cell(hide_code=True)
def _():
    checks.run(5, gd_train=gd_train, relative_change=relative_change, NTKNet=NTKNet)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.dltheory import empirical_ntk, kernel_regression_predict` gives you the NTK of any
    scalar-output PyTorch model, and the capstone's NTK option builds on all five stages.
    """)
    return


if __name__ == "__main__":
    app.run()
