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
    # discrete latents · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. a VQ-VAE replaces the VAE's
    Gaussian code with a choice from a learned **codebook**: every patch of the image
    becomes one of $K$ symbols. the choice (an argmin) has no gradient, so you'll build the
    straight-through estimator that trains through it anyway, the losses that keep
    encoder and codebook together, and the tools that keep the codebook alive.

    | stage | you build |
    |---|---|
    | 1 | `nearest_code`: vector quantization as a vectorized nearest-neighbour search |
    | 2 | `VectorQuantizer`: straight-through gradients, codebook and commitment losses → `mlfs.generative` |
    | 3 | `codebook_usage`, `restart_dead_codes`: measuring and fixing codebook collapse |
    | 4 | `VQVAE`: encoder, quantizer, decoder; images to indices and back → `mlfs.generative` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def nearest_code(z, codebook):
    """stage 1: z (N, D), codebook (K, D) -> int64 indices (N,) of the nearest code (squared Euclidean), ties to the
    lowest index; vectorized (an (N, K) distance matrix)."""
    raise NotImplementedError("stage1: implement nearest_code")


@app.cell(hide_code=True)
def _():
    checks.run(1, nearest_code=nearest_code)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class VectorQuantizer(nn.Module):
    """z_e (B, D, *spatial) -> (z_q straight-through, vq_loss = codebook + beta commitment, indices (B, *spatial))."""

    def __init__(self, n_codes, code_dim, *, beta=0.25):
        """stage 2: self.codebook = nn.Embedding(n_codes, code_dim), weights uniform in [-1/n_codes, 1/n_codes]."""
        raise NotImplementedError("stage2: implement VectorQuantizer.__init__")

    def forward(self, z_e):
        """stage 2: z_e (B, D, *spatial), e.g. images (B, D, H, W) or sequences (B, D, T); nearest codes per position;
        vq_loss = mse(z_q, sg(z_e)) + beta mse(z_e, sg(z_q)) (means over all elements); returns z_e + sg(z_q - z_e)
        so the decoder's gradient passes straight to the encoder, and int64 indices (B, *spatial)."""
        raise NotImplementedError("stage2: implement VectorQuantizer.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, VectorQuantizer=VectorQuantizer, nearest_code=nearest_code)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def codebook_usage(indices, n_codes):
    """stage 3: (perplexity, fraction_used): exp of the entropy (nats) of the empirical code distribution, and the
    fraction of the n_codes codes used at least once; two floats."""
    raise NotImplementedError("stage3: implement codebook_usage")


@app.function
def restart_dead_codes(vq, z_e, counts, *, threshold=1, generator=None):
    """stage 3: replace every code with counts < threshold by a random encoder output: z_e (B, D, *spatial) flattened
    to its N vectors (N, D) (or already (N, D)), rows drawn with torch.randint(N, (n_dead,), generator=generator),
    in place; returns n_dead."""
    raise NotImplementedError("stage3: implement restart_dead_codes")


@app.cell(hide_code=True)
def _():
    checks.run(3, codebook_usage=codebook_usage, restart_dead_codes=restart_dead_codes, VectorQuantizer=VectorQuantizer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class VQVAE(nn.Module):
    """conv encoder (downsample 4x) -> VectorQuantizer -> conv decoder (upsample 4x)."""

    def __init__(self, in_channels=3, *, n_codes=512, code_dim=64, base_channels=64):
        """stage 4: self.encoder, self.vq (VectorQuantizer(n_codes, code_dim)), self.decoder."""
        raise NotImplementedError("stage4: implement VQVAE.__init__")

    def forward(self, x):
        """stage 4: x (B, C, H, W), H, W divisible by 4 -> (x_recon (B, C, H, W), vq_loss, indices (B, H/4, W/4))."""
        raise NotImplementedError("stage4: implement VQVAE.forward")

    def encode(self, x):
        """stage 4: x -> int64 code indices (B, H/4, W/4)."""
        raise NotImplementedError("stage4: implement VQVAE.encode")

    def decode_indices(self, indices):
        """stage 4: indices (B, h, w) -> images (B, C, 4h, 4w) through the codebook and the decoder."""
        raise NotImplementedError("stage4: implement VQVAE.decode_indices")


@app.cell(hide_code=True)
def _():
    checks.run(4, VQVAE=VQVAE, VectorQuantizer=VectorQuantizer)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.generative import VectorQuantizer, VQVAE` gives you these modules (the
    audio elective stacks your `VectorQuantizer` into a residual quantizer for a neural
    codec).
    """)
    return


if __name__ == "__main__":
    app.run()
