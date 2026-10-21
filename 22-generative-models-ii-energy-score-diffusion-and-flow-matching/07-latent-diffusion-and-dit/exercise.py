import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.diffusion import TimeEmbedding  # yours, from topic 03
    from mlfs.transformer import FeedForward, MultiHeadAttention, PatchEmbedding, sinusoidal_positions  # ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # latent diffusion & DiT · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure is yours. two ideas behind most modern
    image generators: run the diffusion in the small latent space of a pretrained
    autoencoder instead of pixel space, and replace the U-Net with a transformer over
    latent patches whose every layer is steered by the time and class through adaptive
    layer norm.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `unpatchify`, `pos_embed_2d`: images as sequences of patches, and where each patch is |
    | 2 | `DiTBlock`: a transformer block modulated by adaLN-Zero |
    | 3 | `DiT`: the diffusion transformer → `mlfs.diffusion` |
    | 4 | `encode_latents`, `decode_latents`: from images to diffusable latents and back → `mlfs.diffusion` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(x, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, p*p*C) with N = (H/p)(W/p) patches in raster order and each patch vector ordered
    (row in patch, column in patch, channel); ValueError if p doesn't divide H and W."""
    raise NotImplementedError("stage1: implement patchify")


@app.function
def unpatchify(tokens, patch_size, channels):
    """stage 1: the inverse of patchify for square images: (B, N, p*p*C) -> (B, C, sqrt(N) p, sqrt(N) p)."""
    raise NotImplementedError("stage1: implement unpatchify")


@app.function
def pos_embed_2d(grid_size, d_model):
    """stage 1: fixed 2-D sin-cos positions (grid_size^2, d_model), raster order: row (r, c) is
    [sinusoidal_positions(grid_size, d_model // 2)[r], sinusoidal_positions(grid_size, d_model // 2)[c]]."""
    raise NotImplementedError("stage1: implement pos_embed_2d")


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, unpatchify=unpatchify, pos_embed_2d=pos_embed_2d)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class DiTBlock(nn.Module):
    """x + g1 * attn(modulate(LN(x), b1, s1)), then x + g2 * mlp(modulate(LN(x), b2, s2)); (b, s, g) from the condition."""

    def __init__(self, d_model, n_heads):
        """stage 2: two LayerNorms without affine parameters, mlfs.transformer MultiHeadAttention and FeedForward (4d),
        and adaLN: SiLU -> Linear(d_model, 6 d_model), zero-initialized (so the block starts as the identity)."""
        raise NotImplementedError("stage2: implement DiTBlock.__init__")

    def forward(self, x, c):
        """stage 2: x (B, N, d), c (B, d) -> (B, N, d); modulate(h, shift, scale) = h (1 + scale) + shift."""
        raise NotImplementedError("stage2: implement DiTBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, DiTBlock=DiTBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class DiT(nn.Module):
    """Peebles & Xie (2023): model(x (B, in_channels, input_size, input_size), t (B,), y=None) -> same shape."""

    def __init__(self, input_size=8, patch_size=2, in_channels=4, *, d_model=256, n_layers=6, n_heads=4, n_classes=None):
        """stage 3: mlfs.transformer PatchEmbedding + fixed pos_embed_2d (a registered buffer, not a parameter); condition
        c = TimeEmbedding(d_model)(t) (+ an (n_classes + 1)-row class table, last row = null); self.blocks, an
        nn.ModuleList of n_layers DiTBlocks; self.ada_out (SiLU -> Linear(d_model, 2 d_model): the final (shift, scale))
        and self.out (Linear(d_model, p*p*in_channels)), both zero-initialized; unpatchify."""
        raise NotImplementedError("stage3: implement DiT.__init__")

    def forward(self, x, t, y=None):
        """stage 3: y None -> null label for a conditional model; ValueError for labels without n_classes."""
        raise NotImplementedError("stage3: implement DiT.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, DiT=DiT, DiTBlock=DiTBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def encode_latents(vae, loader, *, device="cpu"):
    """stage 4: run vae.encode(x) -> (mu, logvar) over a loader of (x, y) batches (eval mode, restored; no gradients);
    returns (latents = the means mu concatenated on the CPU, (N, C, h, w), labels (N,) int64)."""
    raise NotImplementedError("stage4: implement encode_latents")


@app.function
def decode_latents(vae, z, *, scale=1.0, batch_size=256, device="cpu"):
    """stage 4: vae.decode(z * scale) in batches (eval mode, restored; no gradients), on the CPU; `scale` undoes the
    normalisation of the latents (pass the latents' std)."""
    raise NotImplementedError("stage4: implement decode_latents")


@app.cell(hide_code=True)
def _():
    checks.run(4, encode_latents=encode_latents, decode_latents=decode_latents)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.diffusion import DiT, encode_latents` gives you latent diffusion: train
    `flow_matching_loss` or `ddpm_loss` on `latents / latents.std()` with a `DiT`, sample,
    multiply by the std and decode. that's the recipe behind Stable Diffusion 3 and its
    relatives, at toy scale.
    """)
    return


if __name__ == "__main__":
    app.run()
