import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MAML · build

    spec-driven (L4): the **spec** boxes in the lesson define every function, the checks
    enforce them, and the internals are yours. models here are *functional*: a model is a list
    of parameter tensors plus a `forward(params, x)` function, so that adapted parameters can be
    ordinary tensors computed from the originals, with the graph between them intact. a batch
    of $T$ tasks is a dict of tensors with a leading task dimension.

    | stage | you build |
    |---|---|
    | 1 | `init_mlp`, `mlp_forward`, `sample_sine_tasks`: a functional MLP that also runs with per-task weights, and Finn et al.'s sine-regression tasks |
    | 2 | `task_mse`, `adapt`: the inner loop, per-task gradient steps that keep (or cut) the graph |
    | 3 | `maml_loss`, `linear_meta_gradient`: the outer objective, and its gradient in closed form for a linear model |
    | 4 | `train_maml`: the outer loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def init_mlp(sizes, *, generator=None):
    """stage 1: [W1, b1, ..., WL, bL], float32 leaves with requires_grad; W_l (sizes[l], sizes[l+1]); entries ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    raise NotImplementedError("stage1: implement init_mlp")


@app.function
def mlp_forward(params, x):
    """stage 1: ReLU MLP x @ W + b per layer (no ReLU after the last); x (..., n, d_in), W (d_in, d_out) or per-task (T, d_in, d_out), b (d_out,) or (T, d_out)."""
    raise NotImplementedError("stage1: implement mlp_forward")


@app.function
def sample_sine_tasks(n_tasks, n_support, n_query, *, amp_range=(0.1, 5.0), phase_range=(0.0, math.pi),
                      x_range=(-5.0, 5.0), generator=None):
    """stage 1: dict xs (T, n_support, 1), ys, xq (T, n_query, 1), yq, amp (T,), phase (T,); y = amp * sin(x - phase); all uniform, independent."""
    raise NotImplementedError("stage1: implement sample_sine_tasks")


@app.cell(hide_code=True)
def _():
    checks.run(1, init_mlp=init_mlp, mlp_forward=mlp_forward, sample_sine_tasks=sample_sine_tasks)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def task_mse(pred, y):
    """stage 2: per-task mean squared error: (T, n, k) and (T, n, k) -> (T,)."""
    raise NotImplementedError("stage2: implement task_mse")


@app.function
def adapt(params, xs, ys, *, forward=None, loss_fn=None, inner_lr=0.01, n_steps=1, first_order=False):
    """stage 2: n_steps of per-task gradient descent from the shared params on (xs, ys); returns per-task params (T, *p.shape), graph kept unless first_order."""
    raise NotImplementedError("stage2: implement adapt")


@app.cell(hide_code=True)
def _():
    checks.run(2, init_mlp=init_mlp, mlp_forward=mlp_forward, task_mse=task_mse, adapt=adapt)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def maml_loss(params, batch, *, forward=None, loss_fn=None, inner_lr=0.01, n_steps=1, first_order=False):
    """stage 3: mean over tasks of loss_fn on the query set after adapt on the support set; differentiable w.r.t. params."""
    raise NotImplementedError("stage3: implement maml_loss")


@app.function
def linear_meta_gradient(theta, Xs, ys, Xq, yq, *, inner_lr, n_steps=1, first_order=False):
    """stage 3: one task, model y = X theta, mean-squared losses (NumPy float64): (I - a H_s)^k grad L_q(theta_k), or grad L_q(theta_k) if first_order."""
    raise NotImplementedError("stage3: implement linear_meta_gradient")


@app.cell(hide_code=True)
def _():
    checks.run(3, init_mlp=init_mlp, mlp_forward=mlp_forward, task_mse=task_mse, adapt=adapt,
               maml_loss=maml_loss, linear_meta_gradient=linear_meta_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_maml(params, sample_batch, *, n_iters, forward=None, loss_fn=None, meta_lr=1e-3, inner_lr=0.01,
               n_steps=1, first_order=False, optimizer="adam", seed=0):
    """stage 4: meta-train a copy of params on maml_loss over batches sample_batch(generator); returns (params, list of meta-losses)."""
    raise NotImplementedError("stage4: implement train_maml")


@app.cell(hide_code=True)
def _():
    checks.run(4, init_mlp=init_mlp, mlp_forward=mlp_forward, sample_sine_tasks=sample_sine_tasks,
               task_mse=task_mse, adapt=adapt, maml_loss=maml_loss, train_maml=train_maml)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. it meta-trains your MAML on
    sine waves and asks how many inner steps adaptation really needs.
    """)
    return


if __name__ == "__main__":
    app.run()
