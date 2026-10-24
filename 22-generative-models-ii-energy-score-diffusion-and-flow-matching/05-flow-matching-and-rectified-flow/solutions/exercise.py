import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson
    from mlfs.ode import odeint  # yours, from ch 20 · topic 06


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # flow matching & rectified flow · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. flow matching learns a velocity
    field that carries noise to data along straight lines, by plain regression, without
    simulating anything during training. sampling is an ODE solve; reflow makes the paths
    straight enough that one Euler step nearly suffices.

    convention: `t = 0` is noise, `t = 1` is data, and the network predicts the velocity
    `x1 - x0`.

    | stage | you build |
    |---|---|
    | 1 | `flow_matching_loss`: conditional flow matching on the straight path → `mlfs.diffusion` |
    | 2 | `sample_flow`: Euler, midpoint or RK4 from noise to data → `mlfs.diffusion` |
    | 3 | `fit_flow_matching`: training, with independent noise or with given pairs |
    | 4 | `make_reflow_pairs`, `straightness`: rectified flow, and a number for how straight it is → `mlfs.diffusion` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def flow_matching_loss(model, x1, *, x0=None, y=None, generator=None):
    """stage 1: x0 = torch.randn(x1.shape, generator) unless given, then t = torch.rand(B, generator) (CPU draws moved to
    x1's device); x_t = (1 - t) x0 + t x1; MSE(model(x_t, t, y), x1 - x0), a mean over all elements."""
    if x0 is None:
        x0 = torch.randn(x1.shape, generator=generator).to(x1.device)
    t = torch.rand(x1.shape[0], generator=generator).to(x1.device)
    tt = t.view((-1,) + (1,) * (x1.dim() - 1))
    x_t = (1 - tt) * x0 + tt * x1
    return F.mse_loss(model(x_t, t, y), x1 - x0)


@app.cell(hide_code=True)
def _():
    checks.run(1, flow_matching_loss=flow_matching_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sample_flow(model, shape, *, n_steps=100, method="euler", x0=None, y=None, generator=None, device="cpu"):
    """stage 2: integrate dx/dt = model(x, t, y) from t = 0 (x0, or torch.randn(shape, generator) if None) to t = 1 with
    mlfs.ode.odeint in n_steps equal steps of `method` ('euler', 'midpoint', 'rk4'); t passed to the model as a (B,)
    tensor; no gradients; an nn.Module is evaluated in eval mode (restored). returns x1."""
    was = getattr(model, "training", None)
    if was is not None:
        model.eval()
    try:
        with torch.no_grad():
            x = (torch.randn(shape, generator=generator) if x0 is None else x0.detach().clone()).to(device)
            B = x.shape[0]

            def f(t, v):
                return model(v, torch.full((B,), float(t), device=device), y)

            # exactly n_steps solver steps (the slack keeps float32 rounding from adding one)
            x = odeint(f, x, torch.tensor([0.0, 1.0]), method=method, step_size=(1 + 1e-6) / n_steps)[-1]
    finally:
        if was is not None:
            model.train(was)
    return x


@app.cell(hide_code=True)
def _():
    checks.run(2, sample_flow=sample_flow)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def fit_flow_matching(model, X, *, X0=None, coupling=None, n_iters=2000, batch_size=256, lr=1e-3, seed=0):
    """stage 3: Adam(lr) on flow_matching_loss over n_iters minibatches, g = torch.Generator().manual_seed(seed):
    idx = randint(N, (batch_size,), g), x1 = X[idx], and x0 = X0[idx] when pairs X0 are given; with a coupling
    (a callable (x0_batch, x1_batch) -> x0_batch reordered, e.g. minibatch optimal transport), x0 =
    coupling(torch.randn(x1.shape, generator=g).to(x1.device), x1) for every batch; otherwise x0 is drawn inside the
    loss with g.
    ValueError if both X0 and coupling are given. model in train mode; returns the list of losses (floats)."""
    if X0 is not None and coupling is not None:
        raise ValueError("give pairs X0 or a coupling, not both")
    X = torch.as_tensor(X, dtype=torch.float32)
    X0 = None if X0 is None else torch.as_tensor(X0, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(n_iters):
        idx = torch.randint(len(X), (batch_size,), generator=g)
        x1 = X[idx]
        if X0 is not None:
            x0 = X0[idx]
        elif coupling is not None:
            x0 = coupling(torch.randn(x1.shape, generator=g).to(x1.device), x1)
        else:
            x0 = None
        loss = flow_matching_loss(model, x1, x0=x0, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, fit_flow_matching=fit_flow_matching)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def make_reflow_pairs(model, n, shape, *, n_steps=100, y=None, generator=None, device="cpu"):
    """stage 4: x0 = torch.randn((n, *shape), generator) (shape = one example's shape), x1 = sample_flow(model,
    (n, *shape), n_steps=n_steps, x0=x0, y=y, device=device) (Euler); returns (x0, x1), both on `device`."""
    shape = tuple(shape) if not isinstance(shape, int) else (shape,)
    x0 = torch.randn((n, *shape), generator=generator).to(device)
    x1 = sample_flow(model, (n, *shape), n_steps=n_steps, x0=x0, y=y, device=device)
    return x0, x1


@app.function
def straightness(model, x0, *, n_steps=100, y=None):
    """stage 4: Liu et al. (2023): Euler-integrate from x0 in n_steps steps (x_k at t_k = k / n_steps) and return the
    mean over the batch and the steps of ||(x_end - x0) - model(x_k, t_k, y)||^2 (summed over non-batch dims), a
    float; 0 means perfectly straight paths. eval mode, no gradients."""
    was = getattr(model, "training", None)
    if was is not None:
        model.eval()
    try:
        with torch.no_grad():
            B = x0.shape[0]
            x, vs = x0.clone(), []
            for k in range(n_steps):
                v = model(x, torch.full((B,), k / n_steps, device=x0.device), y)
                vs.append(v)
                x = x + v / n_steps
            disp = x - x0
            total = sum(((disp - v) ** 2).flatten(1).sum(1).mean() for v in vs) / n_steps
    finally:
        if was is not None:
            model.train(was)
    return float(total)


@app.cell(hide_code=True)
def _():
    checks.run(4, make_reflow_pairs=make_reflow_pairs, straightness=straightness)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import flow_matching_loss, sample_flow, make_reflow_pairs` gives
    you flow matching. topic 06 conditions it on labels; the optimal-transport elective
    replaces the independent pairing of noise and data with a minibatch OT coupling, through
    `fit_flow_matching(..., coupling=...)`.
    """)
    return


if __name__ == "__main__":
    app.run()
