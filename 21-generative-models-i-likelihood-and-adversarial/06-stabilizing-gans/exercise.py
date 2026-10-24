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
    # stabilizing GANs · build

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
    raise NotImplementedError("stage1: implement critic_loss")


@app.function
def wgan_generator_loss(fake_scores):
    """stage 1: -fake.mean()."""
    raise NotImplementedError("stage1: implement wgan_generator_loss")


@app.function
def clip_weights_(module, c=0.01):
    """stage 1: clamp every parameter of module to [-c, c] in place (the original WGAN); returns module."""
    raise NotImplementedError("stage1: implement clip_weights_")


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
    raise NotImplementedError("stage2: implement gradient_penalty")


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
    raise NotImplementedError("stage3: implement spectral_normalize")


@app.class_definition
class SNLinear(nn.Linear):
    """a linear layer whose weight is divided by its spectral norm, estimated by power iteration (buffer `u`)."""

    def __init__(self, in_features, out_features, bias=True, *, n_power_iterations=1):
        """stage 3: nn.Linear plus a buffer u (out_features,), a random unit vector."""
        raise NotImplementedError("stage3: implement SNLinear.__init__")

    def forward(self, x):
        """stage 3: F.linear with spectral_normalize(weight, u)[0]; the buffer u is updated in training mode only."""
        raise NotImplementedError("stage3: implement SNLinear.forward")


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
    raise NotImplementedError("stage4: implement train_wgan")


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
