import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # loss landscapes · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal structure
    is yours. stages 1–3 join **`mlfs.dltheory`**, and the chapter's capstone and later chapters
    use them to look at the landscapes of their own models.

    | stage | you build |
    |---|---|
    | 1 | `filter_normalized_direction`: a random direction scaled filter by filter to the weights → `mlfs.dltheory` |
    | 2 | `loss_surface_2d`: the loss on a plane through the weights → `mlfs.dltheory` |
    | 3 | `linear_interpolation`, `loss_barrier`: the loss along the straight line between two solutions → `mlfs.dltheory` |
    | 4 | `hessian_vector_product`, `top_hessian_eigenvalue`: sharpness by power iteration |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def filter_normalized_direction(model, *, generator=None):
    """stage 1: dict name -> random tensor like each parameter, each filter (row along dim 0) rescaled to that filter's norm; 1-D params zero."""
    direction = {}
    for name, p in model.named_parameters():
        d = torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device)
        if p.dim() <= 1:
            d.zero_()
        else:
            p_norm = p.detach().reshape(p.shape[0], -1).norm(dim=1)
            d_norm = d.reshape(d.shape[0], -1).norm(dim=1)
            scale = p_norm / (d_norm + 1e-10)
            d = d * scale.reshape(-1, *([1] * (p.dim() - 1)))
        direction[name] = d
    return direction


@app.cell(hide_code=True)
def _():
    checks.run(1, filter_normalized_direction=filter_normalized_direction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def loss_surface_2d(model, eval_loss, d1, d2, *, span=1.0, resolution=21):
    """stage 2: (alphas, betas, Z) with Z[i, j] = eval_loss at theta + alphas[i] d1 + betas[j] d2 on linspace(-span, span, resolution); weights restored."""
    alphas = np.linspace(-span, span, resolution)
    betas = np.linspace(-span, span, resolution)
    params = dict(model.named_parameters())
    theta = {k: p.detach().clone() for k, p in params.items()}
    Z = np.zeros((resolution, resolution))
    try:
        with torch.no_grad():
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    for k, p in params.items():
                        new = theta[k].clone()
                        if k in d1:
                            new += float(a) * d1[k].to(new)
                        if k in d2:
                            new += float(b) * d2[k].to(new)
                        p.copy_(new)
                    Z[i, j] = float(eval_loss(model))
    finally:
        with torch.no_grad():
            for k, p in params.items():
                p.copy_(theta[k])
    return alphas, betas, Z


@app.cell(hide_code=True)
def _():
    checks.run(2, loss_surface_2d=loss_surface_2d, filter_normalized_direction=filter_normalized_direction)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def linear_interpolation(model, state_a, state_b, eval_loss, *, n_points=21):
    """stage 3: (alphas, losses): eval_loss after loading (1 - a) state_a + a state_b (float entries; others from state_a); state restored."""
    alphas = np.linspace(0.0, 1.0, n_points)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # snapshot both ends: state_a may be model.state_dict() itself, whose tensors change as we load
    state_a = {k: v.detach().clone() for k, v in state_a.items()}
    state_b = {k: v.detach().clone() for k, v in state_b.items()}
    losses = np.zeros(n_points)
    try:
        for i, a in enumerate(alphas):
            mixed = {}
            for k, va in state_a.items():
                if torch.is_floating_point(va):
                    mixed[k] = (1.0 - float(a)) * va + float(a) * state_b[k].to(va)
                else:
                    mixed[k] = va.clone()
            model.load_state_dict(mixed)
            losses[i] = float(eval_loss(model))
    finally:
        model.load_state_dict(original)
    return alphas, losses


@app.function
def loss_barrier(losses):
    """stage 3: max over the path of loss minus the straight line between the two end losses (0 if the path never rises above it)."""
    losses = np.asarray(losses, dtype=float)
    t = np.linspace(0.0, 1.0, len(losses))
    line = (1 - t) * losses[0] + t * losses[-1]
    return float(max(0.0, np.max(losses - line)))


@app.cell(hide_code=True)
def _():
    checks.run(3, linear_interpolation=linear_interpolation, loss_barrier=loss_barrier)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def hessian_vector_product(loss_fn, params, vector):
    """stage 4: list of tensors H v, H the Hessian of loss_fn() with respect to params (list of tensors), v a list shaped like params; parameters the loss doesn't use get zeros."""
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    dot = sum((g * v).sum() for g, v in zip(grads, vector) if g is not None)
    if not torch.is_tensor(dot) or not dot.requires_grad:  # the loss is linear in (or ignores) every parameter
        return [torch.zeros_like(p) for p in params]
    hv = torch.autograd.grad(dot, params, allow_unused=True)
    return [torch.zeros_like(p) if h is None else h.detach() for h, p in zip(hv, params)]


@app.function
def top_hessian_eigenvalue(model, loss_fn, *, n_iters=20, generator=None):
    """stage 4: the Hessian eigenvalue of largest magnitude (signed), by n_iters steps of power iteration from a random start; no side effects on params or .grad."""
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn(p.shape, generator=generator, dtype=p.dtype).to(p.device) for p in params]
    norm = torch.sqrt(sum((x * x).sum() for x in v))
    v = [x / norm for x in v]
    eig = 0.0
    for _ in range(n_iters):
        hv = hessian_vector_product(loss_fn, params, v)
        eig = float(sum((h * x).sum() for h, x in zip(hv, v)))
        norm = torch.sqrt(sum((h * h).sum() for h in hv))
        if norm == 0:
            return 0.0
        v = [h / norm for h in hv]
    return eig


@app.cell(hide_code=True)
def _():
    checks.run(4, hessian_vector_product=hessian_vector_product, top_hessian_eigenvalue=top_hessian_eigenvalue)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.dltheory import filter_normalized_direction, loss_surface_2d, linear_interpolation`
    gives you the landscape tools for any PyTorch model.
    """)
    return


if __name__ == "__main__":
    app.run()
