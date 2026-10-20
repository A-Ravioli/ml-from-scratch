import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # neural ODEs · build

    spec-driven (L4): the spec boxes in the lesson define every function, the checks enforce
    them (against analytic solutions, finite differences and memory measurements), and the
    internals are yours. a vector field is a callable `f(t, y)` returning $dy/dt$ with `y`'s
    shape, `t` a 0-d tensor; states are tensors of any shape.

    | stage | you build |
    |---|---|
    | 1 | `euler_step`, `midpoint_step`, `rk4_step`: one step of three fixed-step solvers |
    | 2 | `odeint`: integrate over a time grid, differentiable by backprop → `mlfs.ode` |
    | 3 | `odeint_adjoint`: the same solution, gradients from the adjoint ODE in O(1) memory → `mlfs.ode` |
    | 4 | `NeuralODE`: a network whose depth is continuous time → `mlfs.ode` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def euler_step(f, t, y, h):
    """stage 1: one explicit Euler step: y + h f(t, y)."""
    raise NotImplementedError("stage1: implement euler_step")


@app.function
def midpoint_step(f, t, y, h):
    """stage 1: one explicit midpoint step: y + h f(t + h/2, y + (h/2) f(t, y))."""
    raise NotImplementedError("stage1: implement midpoint_step")


@app.function
def rk4_step(f, t, y, h):
    """stage 1: one classical Runge-Kutta step (stages k1..k4, weights 1/6, 1/3, 1/3, 1/6)."""
    raise NotImplementedError("stage1: implement rk4_step")


@app.cell(hide_code=True)
def _():
    checks.run(1, euler_step=euler_step, midpoint_step=midpoint_step, rk4_step=rk4_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def steps_per_interval(t, step_size):
    """stage 2: equal substeps per interval of t: 1 each if step_size is None, else ceil(|dt| / step_size)."""
    raise NotImplementedError("stage2: implement steps_per_interval")


@app.function
def odeint(f, y0, t, *, method="rk4", step_size=None):
    """stage 2: solve dy/dt = f(t, y) from y(t[0]) = y0; returns (len(t), *y0.shape), differentiable by backprop."""
    raise NotImplementedError("stage2: implement odeint")


@app.cell(hide_code=True)
def _():
    checks.run(2, odeint=odeint, rk4_step=rk4_step, odeint_adjoint=odeint_adjoint)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def odeint_adjoint(f, y0, t, *, method="rk4", step_size=None):
    """stage 3: odeint's result; gradients w.r.t. y0 and f's parameters (f an nn.Module) from the adjoint ODE."""
    raise NotImplementedError("stage3: implement odeint_adjoint")


@app.cell(hide_code=True)
def _():
    checks.run(3, odeint_adjoint=odeint_adjoint, odeint=odeint)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class NeuralODE(nn.Module):
    """y(t1) for dy/dt = func(t, y), y(0) = y0: a continuous-depth block."""

    def __init__(self, func, *, t1=1.0, method="rk4", n_steps=10, adjoint=False):
        """stage 4: store func (an nn.Module, registered as a submodule) and the solver settings."""
        raise NotImplementedError("stage4: implement NeuralODE.__init__")

    def forward(self, y0):
        """stage 4: integrate from 0 to t1 in n_steps equal steps of `method`; adjoint=True uses odeint_adjoint."""
        raise NotImplementedError("stage4: implement NeuralODE.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, NeuralODE=NeuralODE, odeint=odeint)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.ode import odeint, odeint_adjoint, NeuralODE` gives you the solvers; chapter 22's
    probability-flow sampler integrates with your `odeint`.
    """)
    return


if __name__ == "__main__":
    app.run()
