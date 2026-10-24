import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # score matching · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. the score $\nabla_x \log p(x)$
    is everything Langevin dynamics needs, and unlike the density it doesn't involve the
    normalizer. you'll learn it straight from data two ways (denoising and sliced score
    matching), condition it on a noise level, and sample with annealed Langevin dynamics,
    the sampler of the first score-based image models.

    | stage | you build |
    |---|---|
    | 1 | `dsm_loss`: denoising score matching, the regression that learns a score → `mlfs.diffusion` |
    | 2 | `sliced_score_matching_loss`: score matching without noise, through random projections → `mlfs.diffusion` |
    | 3 | `geometric_sigmas`, `NoiseConditionalScore`, `fit_dsm`: one network for every noise level |
    | 4 | `annealed_langevin`: Langevin from large noise down to small → `mlfs.diffusion` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def dsm_loss(score_fn, x, sigma, *, generator=None):
    """stage 1: denoising score matching. z = torch.randn(x.shape, generator) (CPU, moved to x's device),
    x_tilde = x + sigma z, target -(x_tilde - x) / sigma^2; score_fn(x_tilde, sigma_vec (B,)); loss = mean over the
    batch of 0.5 sigma^2 * sum over non-batch dims of (score - target)^2. sigma: float or tensor (B,)."""
    B = x.shape[0]
    sig = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    sig = sig.expand(B) if sig.dim() == 0 else sig
    z = torch.randn(x.shape, generator=generator).to(x.device)
    s_ = sig.view((B,) + (1,) * (x.dim() - 1))
    x_tilde = x + s_ * z
    target = -z / s_
    score = score_fn(x_tilde, sig)
    return (0.5 * sig ** 2 * ((score - target) ** 2).flatten(1).sum(1)).mean()


@app.cell(hide_code=True)
def _():
    checks.run(1, dsm_loss=dsm_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def sliced_score_matching_loss(score_fn, x, *, n_projections=1, generator=None):
    """stage 2: Song et al. (2019). v = torch.randn((n_projections, *x.shape), generator); per projection
    v^T grad_x (v^T s(x)) + 0.5 (v^T s(x))^2 per example; mean over projections and the batch. differentiable in the
    score's parameters (create_graph)."""
    x = x.detach().requires_grad_(True)
    s = score_fn(x)
    v = torch.randn((n_projections, *x.shape), generator=generator).to(x.device)
    total = x.new_zeros(())
    for k in range(n_projections):
        sv = (s * v[k]).flatten(1).sum(1)
        (gv,) = torch.autograd.grad(sv.sum(), x, create_graph=True)
        total = total + ((gv * v[k]).flatten(1).sum(1) + 0.5 * sv ** 2).mean()
    return total / n_projections


@app.cell(hide_code=True)
def _():
    checks.run(2, sliced_score_matching_loss=sliced_score_matching_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def geometric_sigmas(sigma_max, sigma_min, n):
    """stage 3: n noise levels from sigma_max down to sigma_min in a geometric sequence, float32 tensor (n,)."""
    return torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), n))


@app.class_definition
class NoiseConditionalScore(nn.Module):
    """s(x, sigma): an MLP on (x, log sigma) whose output is divided by sigma (the scale of the true score)."""

    def __init__(self, data_dim=2, *, hidden_dim=128, n_layers=3):
        """stage 3: an MLP (data_dim + 1) -> hidden (n_layers SiLU layers) -> data_dim."""
        super().__init__()
        layers, d = [], data_dim + 1
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma):
        """stage 3: x (B, data_dim), sigma float or (B,) -> score (B, data_dim) = net([x, log sigma]) / sigma."""
        sig = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        sig = sig.expand(x.shape[0]) if sig.dim() == 0 else sig
        h = torch.cat([x, sig.log()[:, None]], dim=1)
        return self.net(h) / sig[:, None]


@app.function
def fit_dsm(model, X, sigmas, *, n_iters=2000, batch_size=256, lr=1e-3, seed=0):
    """stage 3: Adam(lr) on dsm_loss over n_iters minibatches; per step (generator g seeded with `seed`):
    X[randint(N, (batch_size,), g)], one sigma per example sigmas[randint(len(sigmas), (batch_size,), g)], then
    dsm_loss(model, batch, sigma_vec, generator=g). returns the list of losses (floats)."""
    X = torch.as_tensor(X, dtype=torch.float32)
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32).reshape(-1)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(n_iters):
        xb = X[torch.randint(len(X), (batch_size,), generator=g)]
        sb = sigmas[torch.randint(len(sigmas), (batch_size,), generator=g)]
        loss = dsm_loss(model, xb, sb, generator=g)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, geometric_sigmas=geometric_sigmas, NoiseConditionalScore=NoiseConditionalScore, fit_dsm=fit_dsm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def annealed_langevin(score_fn, x0, sigmas, *, steps_per_sigma=100, eps=2e-05, generator=None):
    """stage 4: NCSN sampling (Song & Ermon, 2019). for each sigma_i in `sigmas` (in the given, decreasing order):
    alpha_i = eps * sigma_i^2 / sigma_L^2 (sigma_L = the last), then steps_per_sigma steps of
    x <- x + (alpha_i / 2) score_fn(x, sigma_vec) + sqrt(alpha_i) z with sigma_vec = full((B,), sigma_i) and
    z = torch.randn(x.shape, generator); no gradients; returns x."""
    sigmas = [float(s) for s in torch.as_tensor(sigmas).reshape(-1)]
    x = x0.detach().clone()
    B = x.shape[0]
    with torch.no_grad():
        for s in sigmas:
            alpha = eps * s ** 2 / sigmas[-1] ** 2
            sig = torch.full((B,), s, dtype=x.dtype, device=x.device)
            for _ in range(steps_per_sigma):
                z = torch.randn(x.shape, generator=generator).to(x.device)
                x = x + 0.5 * alpha * score_fn(x, sig) + math.sqrt(alpha) * z
    return x


@app.cell(hide_code=True)
def _():
    checks.run(4, annealed_langevin=annealed_langevin)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import dsm_loss, annealed_langevin` gives you score matching and its
    sampler. topic 03's diffusion models are this recipe with a continuum of noise levels
    and a better-designed sampler.
    """)
    return


if __name__ == "__main__":
    app.run()
