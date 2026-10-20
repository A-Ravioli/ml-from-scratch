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
    # neural ODEs · build (reference solution)

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
    return y + h * f(t, y)


@app.function
def midpoint_step(f, t, y, h):
    """stage 1: one explicit midpoint step: y + h f(t + h/2, y + (h/2) f(t, y))."""
    return y + h * f(t + h / 2, y + (h / 2) * f(t, y))


@app.function
def rk4_step(f, t, y, h):
    """stage 1: one classical Runge-Kutta step (stages k1..k4, weights 1/6, 1/3, 1/3, 1/6)."""
    k1 = f(t, y)
    k2 = f(t + h / 2, y + (h / 2) * k1)
    k3 = f(t + h / 2, y + (h / 2) * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


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
    if step_size is None:
        return [1] * (len(t) - 1)
    return [max(1, math.ceil(abs(float(b - a)) / float(step_size) - 1e-9)) for a, b in zip(t[:-1], t[1:])]


@app.function
def odeint(f, y0, t, *, method="rk4", step_size=None):
    """stage 2: solve dy/dt = f(t, y) from y(t[0]) = y0; returns (len(t), *y0.shape), differentiable by backprop."""
    steppers = {"euler": euler_step, "midpoint": midpoint_step, "rk4": rk4_step}
    if method not in steppers:
        raise ValueError(f"method must be one of {sorted(steppers)}, got {method!r}")
    step = steppers[method]
    t = torch.as_tensor(t, dtype=y0.dtype if y0.is_floating_point() else torch.float64)
    ys, y = [y0], y0
    for k, n in enumerate(steps_per_interval(t, step_size)):
        h = (t[k + 1] - t[k]) / n
        for i in range(n):
            y = step(f, t[k] + i * h, y, h)
        ys.append(y)
    return torch.stack(ys)


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
    if not isinstance(f, nn.Module):
        raise TypeError("odeint_adjoint needs f to be an nn.Module, so it can find the parameters to differentiate")

    class AdjointSolve(torch.autograd.Function):
        """forward: solve without recording a graph; backward: solve the augmented adjoint ODE from t[-1] to t[0]."""

        @staticmethod
        def forward(ctx, y0, t, *params):
            ys = odeint(f, y0, t, method=method, step_size=step_size)  # Function.forward runs without grad
            ctx.save_for_backward(t, ys, *params)
            return ys

        @staticmethod
        def backward(ctx, grad_ys):
            t, ys, *params = ctx.saved_tensors
            shape, n = ys.shape[1:], ys[0].numel()
            sizes = [p.numel() for p in params]

            def augmented(s_t, s):
                y = s[:n].reshape(shape)
                a = s[n:2 * n].reshape(shape)
                with torch.enable_grad():
                    y_ = y.detach().requires_grad_(True)
                    fy = f(s_t, y_)
                    vjps = torch.autograd.grad(fy, (y_, *params), -a, allow_unused=True)
                vjps = [torch.zeros_like(x) if g is None else g for g, x in zip(vjps, (y_, *params))]
                return torch.cat([fy.detach().reshape(-1)] + [g.reshape(-1) for g in vjps])

            a = grad_ys[-1].reshape(-1)
            a_p = ys.new_zeros(sum(sizes))
            for k in range(len(t) - 1, 0, -1):
                s = torch.cat([ys[k].reshape(-1), a, a_p])
                s = odeint(augmented, s, t[k - 1:k + 1].flip(0), method=method, step_size=step_size)[-1]
                a = s[n:2 * n] + grad_ys[k - 1].reshape(-1)  # y is reset to the stored forward value at each output time
                a_p = s[2 * n:]
            grads = list(torch.split(a_p, sizes)) if sizes else []
            return (a.reshape(shape), None, *[g.reshape(p.shape) for g, p in zip(grads, params)])

    t = torch.as_tensor(t, dtype=y0.dtype)
    params = tuple(p for p in f.parameters() if p.requires_grad)
    return AdjointSolve.apply(y0, t, *params)


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
        super().__init__()
        self.func, self.t1, self.method, self.n_steps, self.adjoint = func, float(t1), method, int(n_steps), adjoint

    def forward(self, y0):
        """stage 4: integrate from 0 to t1 in n_steps equal steps of `method`; adjoint=True uses odeint_adjoint."""
        t = torch.tensor([0.0, self.t1], dtype=y0.dtype)
        solve = odeint_adjoint if self.adjoint else odeint
        return solve(self.func, y0, t, method=self.method, step_size=self.t1 / self.n_steps)[-1]


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
