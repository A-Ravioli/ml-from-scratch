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
    from mlfs.transformer import TransformerBlock  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # vision transformers · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `patchify`, `PatchEmbedding`: an image as a sequence of patch tokens → `mlfs.transformer` |
    | 2 | `ViT`: [CLS] token, learned positions, transformer blocks, a classification head → `mlfs.transformer` |
    | 3 | `attention_distance`: how far, in pixels, each head of a ViT looks |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def patchify(x, patch_size):
    """stage 1: (B, C, H, W) -> (B, N, C*p*p), patches in row-major order, each flattened in (c, i, j) order."""
    raise NotImplementedError("stage1: implement patchify")


@app.class_definition
class PatchEmbedding(nn.Module):
    """split an image into non-overlapping p x p patches and project each to d_model: Conv2d(kernel=stride=p)."""

    def __init__(self, img_size, patch_size, in_channels, d_model):
        """stage 1: proj = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size); n_patches; ValueError if not divisible."""
        raise NotImplementedError("stage1: implement PatchEmbedding.__init__")

    def forward(self, x):
        """stage 1: (B, C, H, W) -> (B, n_patches, d_model)."""
        raise NotImplementedError("stage1: implement PatchEmbedding.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, patchify=patchify, PatchEmbedding=PatchEmbedding)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class ViT(nn.Module):
    """vision transformer: patch tokens (+ [CLS]) + learned positions -> pre-LN blocks -> final norm -> head."""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10, *, d_model=192, n_layers=6, n_heads=3,
                 d_ff=None, dropout=0.0, pool="cls"):
        """stage 2: patch_embed, cls_token (if pool='cls'), pos_emb, blocks, norm, head (Identity if num_classes=0)."""
        raise NotImplementedError("stage2: implement ViT.__init__")

    def forward_features(self, x):
        """stage 2: (B, C, H, W) -> (B, N (+1), d_model) tokens after the final norm, [CLS] first."""
        raise NotImplementedError("stage2: implement ViT.forward_features")

    def forward(self, x):
        """stage 2: (B, C, H, W) -> logits (B, num_classes), or pooled features (B, d_model) if num_classes=0."""
        raise NotImplementedError("stage2: implement ViT.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, ViT=ViT, PatchEmbedding=PatchEmbedding)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
@torch.no_grad()
def attention_distance(model, x):
    """stage 3: (n_layers, n_heads) mean distance in pixels from each patch query to the patches it attends to."""
    raise NotImplementedError("stage3: implement attention_distance")


@app.cell(hide_code=True)
def _():
    checks.run(3, attention_distance=attention_distance, ViT=ViT)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import ViT, PatchEmbedding` gives you a vision transformer; chapter 22's
    diffusion transformer patchifies latents the same way, and chapter 25's MAE and CLIP use your
    ViT as their image encoder.
    """)
    return


if __name__ == "__main__":
    app.run()
