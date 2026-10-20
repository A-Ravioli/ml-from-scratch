import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # GANs · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a GAN has no likelihood at
    all: a generator turns noise into samples, a discriminator tries to tell them from
    real data, and each trains against the other. you'll build the two losses, one step
    of the game, the DCGAN architecture, and the metric that catches the GAN's signature
    failure, mode collapse.

    | stage | you build |
    |---|---|
    | 1 | `discriminator_loss`, `generator_loss`: the minimax and non-saturating objectives → `mlfs.generative` |
    | 2 | `gan_step`: one discriminator update, then one generator update |
    | 3 | `DCGANGenerator`, `DCGANDiscriminator`: the convolutional architecture → `mlfs.generative` |
    | 4 | `mode_coverage`: how many modes of a known mixture a generator actually hits |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def discriminator_loss(real_logits, fake_logits):
    """stage 1: mean BCE-with-logits of real -> 1 plus mean BCE-with-logits of fake -> 0 (a scalar)."""
    raise NotImplementedError("stage1: implement discriminator_loss")


@app.function
def generator_loss(fake_logits, *, non_saturating=True):
    """stage 1: non_saturating: mean -log D(G(z)) = softplus(-l); else mean log(1 - D(G(z))) = -softplus(l)."""
    raise NotImplementedError("stage1: implement generator_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, discriminator_loss=discriminator_loss, generator_loss=generator_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gan_step(G, D, opt_g, opt_d, real, *, latent_dim, non_saturating=True, generator=None):
    """stage 2: one D update on (real, G(z1).detach()), then one G update on D(G(z2)); z1 and z2 drawn in that order by
    torch.randn((B, latent_dim), generator=generator); D outputs logits (B,) or (B, 1); returns (d_loss, g_loss) floats."""
    raise NotImplementedError("stage2: implement gan_step")


@app.cell(hide_code=True)
def _():
    checks.run(2, gan_step=gan_step, discriminator_loss=discriminator_loss, generator_loss=generator_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def dcgan_init(module):
    """DCGAN initialization: conv and linear weights N(0, 0.02), batch-norm weights N(1, 0.02), biases 0."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)


@app.class_definition
class DCGANGenerator(nn.Module):
    """z (B, latent_dim) -> images (B, C, img_size, img_size) in [-1, 1]: project, then two stride-2 upsamplings."""

    def __init__(self, latent_dim=100, out_channels=3, *, base_channels=64, img_size=32):
        """stage 3: img_size divisible by 4 (ValueError otherwise); linear to (4b, s/4, s/4), BN-ReLU transposed convs,
        tanh output; conv weights ~ N(0, 0.02), BN weights ~ N(1, 0.02), BN biases 0."""
        raise NotImplementedError("stage3: implement DCGANGenerator.__init__")

    def forward(self, z):
        """stage 3: (B, latent_dim) -> (B, C, img_size, img_size)."""
        raise NotImplementedError("stage3: implement DCGANGenerator.forward")


@app.class_definition
class DCGANDiscriminator(nn.Module):
    """images (B, C, s, s) -> logits (B,): two stride-2 convs with LeakyReLU(0.2), no batch norm on the first."""

    def __init__(self, in_channels=3, *, base_channels=64, img_size=32):
        """stage 3: img_size divisible by 4 (ValueError otherwise); conv weights ~ N(0, 0.02)."""
        raise NotImplementedError("stage3: implement DCGANDiscriminator.__init__")

    def forward(self, x):
        """stage 3: logits (B,)."""
        raise NotImplementedError("stage3: implement DCGANDiscriminator.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, DCGANGenerator=DCGANGenerator, DCGANDiscriminator=DCGANDiscriminator, gan_step=gan_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def mode_coverage(samples, centers, std, *, n_std=3.0, min_frac=0.01):
    """stage 4: (n_covered, frac_high_quality, counts): a sample is high quality if it lies within n_std * std of its
    nearest centre; counts (K,) int = high-quality samples per centre; a centre is covered if counts >= min_frac * n."""
    raise NotImplementedError("stage4: implement mode_coverage")


@app.cell(hide_code=True)
def _():
    checks.run(4, mode_coverage=mode_coverage)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import DCGANGenerator, DCGANDiscriminator, discriminator_loss,
    generator_loss` gives you these pieces; topic 06 stabilizes them.
    """)
    return


if __name__ == "__main__":
    app.run()
