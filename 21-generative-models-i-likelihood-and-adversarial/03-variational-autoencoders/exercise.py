import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # variational autoencoders · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a VAE is a latent-variable
    model $p(x) = \int p(x \mid z)\,p(z)\,dz$ trained by maximizing a lower bound on its
    likelihood, with an encoder that guesses which $z$ explains each $x$. you'll build the
    two tricks that make it trainable, the model itself (MLP and convolutional), a tighter
    likelihood estimate, and the diagnostics for its best-known failure.

    | stage | you build |
    |---|---|
    | 1 | `reparameterize`, `kl_standard_normal`: differentiable sampling and the closed-form KL → `mlfs.generative` |
    | 2 | `reconstruction_nll`, `VAE`: the ELBO, encoder, decoder, sampler → `mlfs.generative` |
    | 3 | `iwae_nll`: an importance-weighted estimate of $-\log p(x)$, tighter than the ELBO |
    | 4 | `ConvVAE`: spatial latents for images (chapter 22's latent diffusion uses it) → `mlfs.generative` |
    | 5 | `active_units`, `kl_per_dim`: detecting posterior collapse |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def reparameterize(mu, logvar, *, generator=None):
    """stage 1: mu + exp(logvar / 2) * eps with eps ~ N(0, I) drawn by torch.randn(mu.shape, generator=generator)."""
    raise NotImplementedError("stage1: implement reparameterize")


@app.function
def kl_standard_normal(mu, logvar):
    """stage 1: KL(N(mu, exp(logvar)) || N(0, I)) summed over all non-batch dims -> (B,)."""
    raise NotImplementedError("stage1: implement kl_standard_normal")


@app.cell(hide_code=True)
def _():
    checks.run(1, reparameterize=reparameterize, kl_standard_normal=kl_standard_normal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def reconstruction_nll(x_out, x, likelihood):
    """stage 2: -log p(x | z) per example (B,), summed over non-batch dims: 'bernoulli' (x_out = logits) or
    'gaussian' (x_out = mean, unit variance, including the 0.5 log(2 pi) constant)."""
    raise NotImplementedError("stage2: implement reconstruction_nll")


@app.class_definition
class VAE(nn.Module):
    """MLP VAE: encoder x -> (mu, logvar), decoder z -> logits (bernoulli) or mean (gaussian)."""

    def __init__(self, in_dim, latent_dim, *, hidden_dims=(512, 256), likelihood="bernoulli"):
        """stage 2: encoder MLP in_dim -> hidden_dims -> 2 * latent_dim, decoder MLP latent_dim -> reversed -> in_dim."""
        raise NotImplementedError("stage2: implement VAE.__init__")

    def encode(self, x):
        """stage 2: x (B, in_dim) -> (mu, logvar), each (B, latent_dim)."""
        raise NotImplementedError("stage2: implement VAE.encode")

    def decode(self, z):
        """stage 2: z (B, latent_dim) -> logits (bernoulli) or means (gaussian), (B, in_dim)."""
        raise NotImplementedError("stage2: implement VAE.decode")

    def forward(self, x):
        """stage 2: (x_out, mu, logvar) with z = reparameterize(mu, logvar) (global RNG)."""
        raise NotImplementedError("stage2: implement VAE.forward")

    def loss(self, x, *, beta=1.0):
        """stage 2: one call to forward; (loss, recon, kl) as batch means of per-example sums, loss = recon + beta kl."""
        raise NotImplementedError("stage2: implement VAE.loss")

    @torch.no_grad()
    def sample(self, n, *, generator=None):
        """stage 2: decode z ~ N(0, I) (torch.randn((n, latent_dim), generator=generator)): probabilities or means."""
        raise NotImplementedError("stage2: implement VAE.sample")


@app.cell(hide_code=True)
def _():
    checks.run(2, reconstruction_nll=reconstruction_nll, VAE=VAE, kl_standard_normal=kl_standard_normal, reparameterize=reparameterize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def iwae_nll(vae, x, *, k=64, generator=None, chunk_size=None):
    """stage 3: -log (1/k) sum_j p(x | z_j) p(z_j) / q(z_j | x) with z_j ~ q(z | x), per example (B,); eps drawn as
    torch.randn((k, *mu.shape), generator=generator); decodes chunk_size samples per example at a time (None: all k);
    no gradients."""
    raise NotImplementedError("stage3: implement iwae_nll")


@app.cell(hide_code=True)
def _():
    checks.run(3, iwae_nll=iwae_nll, VAE=VAE, reconstruction_nll=reconstruction_nll, ConvVAE=ConvVAE, kl_standard_normal=kl_standard_normal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class ConvVAE(nn.Module):
    """convolutional VAE with spatial latents (B, latent_channels, H/4, W/4)."""

    def __init__(self, in_channels=3, *, latent_channels=4, base_channels=64, likelihood="gaussian"):
        """stage 4: encoder with two stride-2 convs -> 2 * latent_channels maps; decoder with two stride-2 transposed convs."""
        raise NotImplementedError("stage4: implement ConvVAE.__init__")

    def encode(self, x):
        """stage 4: x (B, C, H, W), H and W divisible by 4 (ValueError otherwise) -> (mu, logvar) (B, lc, H/4, W/4)."""
        raise NotImplementedError("stage4: implement ConvVAE.encode")

    def decode(self, z):
        """stage 4: z (B, lc, h, w) -> (B, C, 4h, 4w) means (gaussian) or logits (bernoulli)."""
        raise NotImplementedError("stage4: implement ConvVAE.decode")

    def forward(self, x):
        """stage 4: (x_out, mu, logvar) with z = reparameterize(mu, logvar)."""
        raise NotImplementedError("stage4: implement ConvVAE.forward")

    def loss(self, x, *, beta=1.0):
        """stage 4: one call to forward; (loss, recon, kl), batch means of per-example sums, loss = recon + beta kl."""
        raise NotImplementedError("stage4: implement ConvVAE.loss")

    @torch.no_grad()
    def sample(self, n, img_size, *, generator=None):
        """stage 4: decode z ~ N(0, I) of shape (n, lc, H/4, W/4); img_size int or (H, W); probabilities or means."""
        raise NotImplementedError("stage4: implement ConvVAE.sample")


@app.cell(hide_code=True)
def _():
    checks.run(4, ConvVAE=ConvVAE, reconstruction_nll=reconstruction_nll, kl_standard_normal=kl_standard_normal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def active_units(vae, x, *, threshold=0.01):
    """stage 5: (count, variances): variances (latent,) float64 ndarray = Var over the batch (ddof=0) of the encoder
    mean of each latent coordinate (flattened); count = number with variance > threshold."""
    raise NotImplementedError("stage5: implement active_units")


@app.function
def kl_per_dim(vae, x):
    """stage 5: batch mean of KL(q(z_i | x) || N(0, 1)) for each latent coordinate (flattened), float64 ndarray."""
    raise NotImplementedError("stage5: implement kl_per_dim")


@app.cell(hide_code=True)
def _():
    checks.run(5, active_units=active_units, kl_per_dim=kl_per_dim, VAE=VAE, kl_standard_normal=kl_standard_normal)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import VAE, ConvVAE` gives you these models: the capstone trains
    your VAE as the latent-variable family, and chapter 22 runs diffusion in the latent
    space of your `ConvVAE`.
    """)
    return


if __name__ == "__main__":
    app.run()
