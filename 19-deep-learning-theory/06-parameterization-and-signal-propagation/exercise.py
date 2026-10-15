import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # parameterization & signal propagation · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours.

    | stage | you build |
    |---|---|
    | 1 | `gaussian_expectation`, `variance_map`: how the size of a signal changes from one layer to the next |
    | 2 | `q_fixed_point`, `chi1`: where the signal settles, and whether nearby inputs are pulled together or apart |
    | 3 | `propagate`: the same quantities measured in a real random network |
    | 4 | `MuMLP`, `param_groups`: one MLP in standard and in maximal-update (μP) parameterization |
    | 5 | `activation_deltas`: the coordinate check, how much each layer's activations move in a training step |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def activation_fn(name):
    """phi and its derivative phi' for 'tanh' or 'relu' (helper, given)."""
    if name == "tanh":
        return np.tanh, lambda u: 1.0 - np.tanh(u) ** 2
    if name == "relu":
        return (lambda u: np.maximum(u, 0.0)), (lambda u: (u > 0).astype(float))
    raise ValueError(f"activation must be 'tanh' or 'relu', got {name!r}")


@app.function
def gaussian_expectation(f, q, *, n_points=100):
    """stage 1: E[f(sqrt(q) z)] for z ~ N(0, 1), by Gauss-Hermite quadrature with n_points nodes (np.polynomial.hermite_e.hermegauss)."""
    raise NotImplementedError("stage1: implement gaussian_expectation")


@app.function
def variance_map(q, sigma_w, sigma_b, activation):
    """stage 1: sigma_w^2 E[phi(sqrt(q) z)^2] + sigma_b^2: the next layer's pre-activation variance."""
    raise NotImplementedError("stage1: implement variance_map")


@app.cell(hide_code=True)
def _():
    checks.run(1, gaussian_expectation=gaussian_expectation, variance_map=variance_map)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def q_fixed_point(sigma_w, sigma_b, activation, *, q0=1.0, n_iter=500):
    """stage 2: iterate q <- variance_map(q) n_iter times from q0 and return the final q."""
    raise NotImplementedError("stage2: implement q_fixed_point")


@app.function
def chi1(q, sigma_w, activation):
    """stage 2: sigma_w^2 E[phi'(sqrt(q) z)^2]: the slope of the correlation map at c = 1 (ordered < 1 < chaotic)."""
    raise NotImplementedError("stage2: implement chi1")


@app.cell(hide_code=True)
def _():
    checks.run(2, q_fixed_point=q_fixed_point, chi1=chi1, variance_map=variance_map)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def propagate(x, depth, width, *, sigma_w, sigma_b, activation, generator=None):
    """stage 3: dict 'q' (depth, n) mean squared pre-activation of each input per layer, 'corr' (depth,) cosine of inputs 0 and 1's pre-activations."""
    raise NotImplementedError("stage3: implement propagate")


@app.cell(hide_code=True)
def _():
    checks.run(3, propagate=propagate, variance_map=variance_map, q_fixed_point=q_fixed_point, chi1=chi1)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class MuMLP(nn.Module):
    """ReLU MLP: `depth` hidden Linear layers of `width`, then a readout; 'sp' or 'mup' (readout input scaled by base_width / width, readout zero-initialized)."""

    def __init__(self, in_dim, width, out_dim, *, depth=2, parameterization="sp", base_width=64):
        """stage 4: hidden (ModuleList of depth nn.Linear), readout nn.Linear; weights N(0, 1/fan_in), biases 0; attrs width_mult, parameterization."""
        raise NotImplementedError("stage4: implement MuMLP.__init__")

    def activations(self, x):
        """stage 4: list of the depth hidden activations (after ReLU) and the output, for x (n, in_dim)."""
        raise NotImplementedError("stage4: implement MuMLP.activations")

    def forward(self, x):
        """stage 4: (n, out_dim) outputs."""
        raise NotImplementedError("stage4: implement MuMLP.forward")


@app.function
def param_groups(model, lr):
    """stage 4: optimizer groups: 'sp' one group at lr; 'mup' hidden[1:] weights at lr / width_mult, everything else at lr."""
    raise NotImplementedError("stage4: implement param_groups")


@app.cell(hide_code=True)
def _():
    checks.run(4, MuMLP=MuMLP, param_groups=param_groups)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def activation_deltas(model, x, y, *, lr, steps=1):
    """stage 5: train `steps` full-batch Adam steps (param_groups(model, lr), cross-entropy on (x, y)) in place; return the RMS change of each entry of model.activations(x)."""
    raise NotImplementedError("stage5: implement activation_deltas")


@app.cell(hide_code=True)
def _():
    checks.run(5, activation_deltas=activation_deltas, MuMLP=MuMLP, param_groups=param_groups)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. the capstone's μTransfer
    option is built from `MuMLP` and `param_groups`.
    """)
    return


if __name__ == "__main__":
    app.run()
