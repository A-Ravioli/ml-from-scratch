import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # stabilizing GANs · build (reference solution)

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. the original GAN's divergence
    gives the generator no direction when real and fake data don't overlap. the
    Wasserstein GAN replaces the discriminator with a **critic** that estimates an earth
    mover's distance, which needs the critic to be 1-Lipschitz. you'll build the losses
    and three ways to enforce the constraint: weight clipping, a gradient penalty, and
    spectral normalization, and then the WGAN-GP training loop.

    | stage | you build |
    |---|---|
    | 1 | `critic_loss`, `wgan_generator_loss`, `clip_weights_`: the Wasserstein objective → `mlfs.generative` |
    | 2 | `gradient_penalty`: WGAN-GP's soft Lipschitz constraint → `mlfs.generative` |
    | 3 | `spectral_normalize`, `SNLinear`: power iteration and spectral normalization → `mlfs.generative` |
    | 4 | `train_wgan`: the WGAN-GP loop with `n_critic` critic steps per generator step |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def critic_loss(real_scores, fake_scores):
    """stage 1: fake.mean() - real.mean() (the critic minimizes it, maximizing the Wasserstein estimate)."""
    return fake_scores.mean() - real_scores.mean()


@app.function
def wgan_generator_loss(fake_scores):
    """stage 1: -fake.mean()."""
    return -fake_scores.mean()


@app.function
def clip_weights_(module, c=0.01):
    """stage 1: clamp every parameter of module to [-c, c] in place (the original WGAN); returns module."""
    with torch.no_grad():
        for p in module.parameters():
            p.clamp_(-c, c)
    return module


@app.cell(hide_code=True)
def _():
    checks.run(1, critic_loss=critic_loss, wgan_generator_loss=wgan_generator_loss, clip_weights_=clip_weights_)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gradient_penalty(critic, real, fake, *, generator=None):
    """stage 2: E[(||grad critic(x_hat)||_2 - 1)^2] at x_hat = eps real + (1 - eps) fake, eps ~ U[0, 1] per example
    (torch.rand((B, 1, ..., 1), generator=generator)); norms over all non-batch dims; differentiable (create_graph)."""
    b = real.shape[0]
    eps = torch.rand((b,) + (1,) * (real.dim() - 1), generator=generator, dtype=real.dtype).to(real.device)
    x_hat = (eps * real + (1 - eps) * fake).detach().requires_grad_(True)
    scores = critic(x_hat)
    (grad,) = torch.autograd.grad(scores.sum(), x_hat, create_graph=True)
    norms = grad.reshape(b, -1).norm(dim=1)
    return ((norms - 1.0) ** 2).mean()


@app.cell(hide_code=True)
def _():
    checks.run(2, gradient_penalty=gradient_penalty)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def spectral_normalize(weight, u, n_power_iterations=1):
    """stage 3: W = weight.reshape(out, -1); without gradients, n times: v = normalize(W^T u), u = normalize(W v);
    sigma = u . (W v) with gradient through W only; returns (weight / sigma, u (detached), sigma)."""
    w = weight.reshape(weight.shape[0], -1)
    with torch.no_grad():
        v = F.normalize(w.T @ u, dim=0)
        for _ in range(n_power_iterations):
            v = F.normalize(w.T @ u, dim=0)
            u = F.normalize(w @ v, dim=0)
    sigma = torch.dot(u, w @ v)
    return weight / sigma, u, sigma


@app.class_definition
class SNLinear(nn.Linear):
    """a linear layer whose weight is divided by its spectral norm, estimated by power iteration (buffer `u`)."""

    def __init__(self, in_features, out_features, bias=True, *, n_power_iterations=1):
        """stage 3: nn.Linear plus a buffer u (out_features,), a random unit vector."""
        super().__init__(in_features, out_features, bias)
        self.n_power_iterations = n_power_iterations
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))

    def forward(self, x):
        """stage 3: F.linear with spectral_normalize(weight, u)[0]; the buffer u is updated in training mode only."""
        w, u, _ = spectral_normalize(self.weight, self.u, self.n_power_iterations)
        if self.training:
            self.u.copy_(u)
        return F.linear(x, w, self.bias)


@app.cell(hide_code=True)
def _():
    checks.run(3, spectral_normalize=spectral_normalize, SNLinear=SNLinear)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_wgan(G, critic, sample_real, *, n_steps, latent_dim, n_critic=5, gp_lambda=10.0, lr=1e-4,
               betas=(0.5, 0.9), generator=None, callback=None):
    """stage 4: WGAN-GP: per generator step, n_critic critic steps (each: real = sample_real(), z ~ randn(B, latent_dim),
    loss = critic_loss + gp_lambda * gradient_penalty), then one generator step on fresh z; Adam(lr, betas) for both.
    returns history dict of per-generator-step lists 'critic_loss', 'gp', 'wasserstein' (last critic step's values);
    callback(step, history), if given, runs after each generator step (step = 1, 2, ...)."""
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    hist = {"critic_loss": [], "gp": [], "wasserstein": []}
    for step in range(1, n_steps + 1):
        for _ in range(n_critic):
            real = sample_real()
            b = real.shape[0]
            z = torch.randn((b, latent_dim), generator=generator).to(real.device)
            fake = G(z).detach()
            real_s, fake_s = critic(real).reshape(b), critic(fake).reshape(b)
            gp = gradient_penalty(critic, real, fake, generator=generator)
            c_loss = critic_loss(real_s, fake_s)
            opt_c.zero_grad()
            (c_loss + gp_lambda * gp).backward()
            opt_c.step()
        z = torch.randn((b, latent_dim), generator=generator).to(real.device)
        g_loss = wgan_generator_loss(critic(G(z)).reshape(b))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        hist["critic_loss"].append(c_loss.item())
        hist["gp"].append(gp.item())
        hist["wasserstein"].append(-c_loss.item())
        if callback is not None:
            callback(step, hist)
    return hist


@app.cell(hide_code=True)
def _():
    checks.run(4, train_wgan=train_wgan, gradient_penalty=gradient_penalty, critic_loss=critic_loss)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import gradient_penalty, critic_loss, wgan_generator_loss,
    spectral_normalize` gives you these tools; the capstone can train its adversarial
    family as a WGAN-GP.
    """)
    return


if __name__ == "__main__":
    app.run()
