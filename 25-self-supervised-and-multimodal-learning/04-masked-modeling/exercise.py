import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # masked modeling · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `unpatchify`: images ↔ sequences of flattened patches, in MAE order → `mlfs.ssl` |
    | 2 | `random_masking`: per-sample shuffle, keep a subset, remember how to undo it → `mlfs.ssl` |
    | 3 | `MAE.__init__`, `MAE.encode`: the encoder, and features for evaluation → `mlfs.ssl` |
    | 4 | `MAE.forward`: encode the visible patches, decode all, loss on the masked ones → `mlfs.ssl` |
    | 5 | `train_mae`: the pretraining loop (AdamW) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(imgs, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, p*p*C), patches row-major, each flattened in (i, j, c) order; ValueError if H, W not divisible."""
    raise NotImplementedError("stage1: implement patchify")


@app.function
def unpatchify(patches, patch_size, channels=3):
    """stage 1: the inverse of patchify for square images: (B, N, p*p*C) -> (B, C, H, W)."""
    raise NotImplementedError("stage1: implement unpatchify")


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, unpatchify=unpatchify)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def random_masking(x, mask_ratio, *, generator=None):
    """stage 2: (x_visible (B, N_keep, D), mask (B, N) float 1=masked, ids_restore (B, N) int64); noise = torch.rand(B, N, generator) on the CPU."""
    raise NotImplementedError("stage2: implement random_masking")


@app.cell(hide_code=True)
def _():
    checks.run(2, random_masking=random_masking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MAE(nn.Module):
    """masked autoencoder: a ViT encoder on the visible patches, a small transformer decoder for all of them."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, *, encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=2, decoder_heads=4, mask_ratio=0.75, norm_pix_loss=False):
        """stage 3: patch_embed (Linear), fixed sin-cos positions, blocks + norm (encoder); decoder_embed, mask_token, decoder_blocks, decoder_norm, decoder_pred."""
        raise NotImplementedError("stage3: implement MAE.__init__")

    def encode(self, x):
        """stage 3: (B, C, H, W) -> (B, encoder_dim): every patch through the encoder (no masking), final norm, mean over tokens."""
        raise NotImplementedError("stage3: implement MAE.encode")

    def forward(self, x):
        """stage 4: (loss, pred (B, N, p*p*C), mask (B, N)): mask, encode the visible patches, decode all, MSE on the masked ones."""
        raise NotImplementedError("stage4: implement MAE.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, MAE=MAE, patchify=patchify)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, MAE=MAE, patchify=patchify, random_masking=random_masking)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_mae(model, images, *, epochs=1, batch_size=256, lr=1e-3, weight_decay=0.05, seed=0, device="cpu"):
    """stage 5: AdamW(lr, betas=(0.9, 0.95), weight_decay) on model(x)[0]; returns {'loss': per-step floats}."""
    raise NotImplementedError("stage5: implement train_mae")


@app.cell(hide_code=True)
def _():
    checks.run(5, train_mae=train_mae, MAE=MAE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `patchify`, `unpatchify`, `random_masking` and `MAE` are part
    of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
