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
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / math.sqrt(fan_in)
        W = (2 * torch.rand(fan_in, fan_out, generator=generator) - 1) * bound
        b = (2 * torch.rand(fan_out, generator=generator) - 1) * bound
        params += [W.requires_grad_(), b.requires_grad_()]
    return params


@app.function
def mlp_forward(params, x):
    """stage 1: ReLU MLP x @ W + b per layer (no ReLU after the last); x (..., n, d_in), W (d_in, d_out) or per-task (T, d_in, d_out), b (d_out,) or (T, d_out)."""
    n_layers = len(params) // 2
    h = x
    for layer in range(n_layers):
        W, b = params[2 * layer], params[2 * layer + 1]
        h = h @ W + b.unsqueeze(-2)
        if layer < n_layers - 1:
            h = torch.relu(h)
    return h


@app.function
def sample_sine_tasks(n_tasks, n_support, n_query, *, amp_range=(0.1, 5.0), phase_range=(0.0, math.pi),
                      x_range=(-5.0, 5.0), generator=None):
    """stage 1: dict xs (T, n_support, 1), ys, xq (T, n_query, 1), yq, amp (T,), phase (T,); y = amp * sin(x - phase); all uniform, independent."""
    def uniform(shape, lo_hi):
        lo, hi = lo_hi
        return lo + (hi - lo) * torch.rand(shape, generator=generator)

    amp = uniform((n_tasks,), amp_range)
    phase = uniform((n_tasks,), phase_range)
    xs = uniform((n_tasks, n_support, 1), x_range)
    xq = uniform((n_tasks, n_query, 1), x_range)

    def f(x):
        return amp[:, None, None] * torch.sin(x - phase[:, None, None])

    return {"xs": xs, "ys": f(xs), "xq": xq, "yq": f(xq), "amp": amp, "phase": phase}


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
    return ((pred - y) ** 2).flatten(1).mean(1)


@app.function
def adapt(params, xs, ys, *, forward=None, loss_fn=None, inner_lr=0.01, n_steps=1, first_order=False):
    """stage 2: n_steps of per-task gradient descent from the shared params on (xs, ys); returns per-task params (T, *p.shape), graph kept unless first_order."""
    forward = mlp_forward if forward is None else forward
    loss_fn = task_mse if loss_fn is None else loss_fn
    n_tasks = xs.shape[0]
    fast = []
    for p in params:
        if not p.requires_grad:  # evaluation with detached weights: still need inner gradients
            p = p.detach().requires_grad_()
        fast.append(p.expand(n_tasks, *p.shape))
    for _ in range(n_steps):
        # tasks are independent, so the gradient of the summed loss w.r.t. task t's copy is task t's gradient
        loss = loss_fn(forward(fast, xs), ys).sum()
        grads = torch.autograd.grad(loss, fast, create_graph=not first_order)
        fast = [w - inner_lr * g for w, g in zip(fast, grads)]
    return fast


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
    forward = mlp_forward if forward is None else forward
    loss_fn = task_mse if loss_fn is None else loss_fn
    fast = adapt(params, batch["xs"], batch["ys"], forward=forward, loss_fn=loss_fn, inner_lr=inner_lr,
                 n_steps=n_steps, first_order=first_order)
    return loss_fn(forward(fast, batch["xq"]), batch["yq"]).mean()


@app.function
def linear_meta_gradient(theta, Xs, ys, Xq, yq, *, inner_lr, n_steps=1, first_order=False):
    """stage 3: one task, model y = X theta, mean-squared losses (NumPy float64): (I - a H_s)^k grad L_q(theta_k), or grad L_q(theta_k) if first_order."""
    theta = np.asarray(theta, dtype=float)
    Xs, ys, Xq, yq = (np.asarray(a, dtype=float) for a in (Xs, ys, Xq, yq))
    H = 2.0 * Xs.T @ Xs / len(ys)
    th = theta.copy()
    for _ in range(n_steps):
        th = th - inner_lr * (2.0 * Xs.T @ (Xs @ th - ys) / len(ys))
    g = 2.0 * Xq.T @ (Xq @ th - yq) / len(yq)
    if first_order:
        return g
    # d theta_k / d theta = (I - a H)^k, symmetric, so the chain rule multiplies g by it
    J = np.linalg.matrix_power(np.eye(len(theta)) - inner_lr * H, n_steps)
    return J @ g


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
    if optimizer not in ("adam", "sgd"):
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")
    generator = torch.Generator().manual_seed(seed)
    theta = [p.detach().clone().requires_grad_() for p in params]
    opt = (torch.optim.Adam(theta, lr=meta_lr) if optimizer == "adam"
           else torch.optim.SGD(theta, lr=meta_lr))
    history = []
    for _ in range(n_iters):
        batch = sample_batch(generator)
        loss = maml_loss(theta, batch, forward=forward, loss_fn=loss_fn, inner_lr=inner_lr,
                         n_steps=n_steps, first_order=first_order)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return theta, history


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
