import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.diffusion import alpha_bar_at, ddpm_loss, flow_matching_loss  # yours, from topics 03-05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # conditioning & guidance · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a class-conditional model is
    just a network that also sees a label. guidance is the trick that makes it *obey* the
    label: push the score (or velocity) further in the direction the label adds. you'll
    build the label dropout that trains a conditional and an unconditional model in one
    network, classifier-free guidance, classifier guidance, and a training loop for either
    objective.

    | stage | you build |
    |---|---|
    | 1 | `drop_labels`: replace labels by the null label at random → `mlfs.diffusion` |
    | 2 | `cfg_model`: classifier-free guidance as a model wrapper, one batched call → `mlfs.diffusion` |
    | 3 | `classifier_guided_eps`: classifier guidance for a noise-prediction model |
    | 4 | `fit_conditional`: class-conditional training with label dropout, for DDPM or flow matching |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def drop_labels(y, p_uncond, null_label, *, generator=None):
    """stage 1: a new tensor where label i becomes null_label when torch.rand(len(y), generator)[i] < p_uncond (drawn on
    the CPU); same dtype and device as y; y itself is not modified."""
    mask = (torch.rand(len(y), generator=generator) < p_uncond).to(y.device)
    return torch.where(mask, torch.full_like(y, null_label), y)


@app.cell(hide_code=True)
def _():
    checks.run(1, drop_labels=drop_labels)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class CFGModel(nn.Module):
    """g(x, t, y) = m(x, t, null) + w (m(x, t, y) - m(x, t, null)) in one batched call of m."""

    def __init__(self, model, guidance_scale, null_label):
        """stage 2: keep the model as a submodule (so .eval() / .train() reach it), w and the null label."""
        super().__init__()
        self.model, self.guidance_scale, self.null_label = model, guidance_scale, null_label

    def forward(self, x, t, y=None):
        """stage 2: one call model(cat([x, x]), cat([t, t]), cat([y, null])); y None -> the unconditional output."""
        if y is None:
            return self.model(x, t, torch.full((x.shape[0],), self.null_label, dtype=torch.long, device=x.device))
        null = torch.full_like(y, self.null_label)
        out = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))
        cond, uncond = out.chunk(2)
        return uncond + self.guidance_scale * (cond - uncond)


@app.function
def cfg_model(model, guidance_scale, null_label):
    """stage 2: returns CFGModel(model, guidance_scale, null_label): w = 1 is the plain conditional model, w = 0 the
    unconditional one; works for noise, score or velocity models."""
    return CFGModel(model, guidance_scale, null_label)


@app.cell(hide_code=True)
def _():
    checks.run(2, cfg_model=cfg_model, CFGModel=CFGModel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class ClassifierGuidedEps(nn.Module):
    """eps_guided(x, t, y) = eps(x, t) - w sqrt(1 - alpha_bar(t)) grad_x log p(y | x, t)."""

    def __init__(self, eps_model, classifier, schedule, guidance_scale):
        """stage 3: keep both networks as submodules, the schedule and w."""
        super().__init__()
        self.eps_model, self.classifier = eps_model, classifier
        self.schedule, self.guidance_scale = schedule, guidance_scale

    def forward(self, x, t, y=None):
        """stage 3: eps_model(x, t) (unconditional), classifier(x, t) -> logits (B, K); the gradient of
        sum_b log_softmax(logits)[b, y_b] w.r.t. x (with gradients on, even under no_grad); alpha_bar(t) from
        alpha_bar_at; y None -> the unguided eps."""
        eps = self.eps_model(x, t)
        if y is None:
            return eps
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            logp = torch.log_softmax(self.classifier(xg, t), dim=1)
            (grad,) = torch.autograd.grad(logp.gather(1, y.view(-1, 1)).sum(), xg)
        ab = alpha_bar_at(self.schedule, t).view((-1,) + (1,) * (x.dim() - 1))
        return eps - self.guidance_scale * (1 - ab).sqrt() * grad


@app.function
def classifier_guided_eps(eps_model, classifier, schedule, guidance_scale):
    """stage 3: returns ClassifierGuidedEps(eps_model, classifier, schedule, guidance_scale), usable by every
    noise-prediction sampler (ddpm_sample, ddim_sample, probability_flow_sample). eps_model is called WITHOUT labels,
    eps_model(x, t) (an unconditional model, or a conditional one that then uses its null label); the classifier is
    called as classifier(x, t) -> logits (B, K)."""
    return ClassifierGuidedEps(eps_model, classifier, schedule, guidance_scale)


@app.cell(hide_code=True)
def _():
    checks.run(3, classifier_guided_eps=classifier_guided_eps)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fit_conditional(model, X, y, n_classes, *, objective="flow", schedule=None, p_uncond=0.1, n_iters=2000,
                    batch_size=256, lr=1e-3, seed=0):
    """stage 4: Adam(lr), n_iters steps, g = torch.Generator().manual_seed(seed): idx = randint(N, (batch_size,), g),
    labels drop_labels(y[idx], p_uncond, n_classes, generator=g), then flow_matching_loss(model, X[idx], y=labels,
    generator=g) for objective 'flow' or ddpm_loss(model, schedule, X[idx], y=labels, generator=g) for 'ddpm'
    (ValueError otherwise, and for 'ddpm' without a schedule); train mode; returns the list of losses."""
    if objective not in ("flow", "ddpm"):
        raise ValueError(f"objective must be 'flow' or 'ddpm', got {objective!r}")
    if objective == "ddpm" and schedule is None:
        raise ValueError("objective='ddpm' needs a DiffusionSchedule (schedule=...)")
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(n_iters):
        idx = torch.randint(len(X), (batch_size,), generator=g)
        labels = drop_labels(y[idx], p_uncond, n_classes, generator=g)
        if objective == "flow":
            loss = flow_matching_loss(model, X[idx], y=labels, generator=g)
        else:
            loss = ddpm_loss(model, schedule, X[idx], y=labels, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(4, fit_conditional=fit_conditional)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import drop_labels, cfg_model` gives you classifier-free guidance
    for any of the chapter's samplers: `sample_flow(cfg_model(m, 3.0, 10), shape, y=labels)`.
    the capstone uses exactly that on CIFAR-10.
    """)
    return


if __name__ == "__main__":
    app.run()
