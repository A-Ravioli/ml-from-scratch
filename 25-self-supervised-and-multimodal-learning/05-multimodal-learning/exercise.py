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
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # multimodal learning · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `clip_loss`: the symmetric contrastive loss between images and captions → `mlfs.ssl` |
    | 2 | `TextTransformer`: a small text encoder, mean-pooled over real tokens |
    | 3 | `MiniCLIP`: two encoders, two projections, one learnable temperature → `mlfs.ssl` |
    | 4 | `zero_shot_predict`, `class_embeddings`: classifiers written in words → `mlfs.ssl` |
    | 5 | `train_clip`: the contrastive training loop |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def clip_loss(image_emb, text_emb, logit_scale):
    """stage 1: mean of cross-entropy over rows and over columns of logit_scale * cos-similarity matrix; pairs on the diagonal."""
    raise NotImplementedError("stage1: implement clip_loss")


@app.cell(hide_code=True)
def _():
    checks.run(1, clip_loss=clip_loss)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class TextTransformer(nn.Module):
    """token ids -> one vector per caption: embeddings + positions, transformer blocks, mean over real tokens."""

    def __init__(self, vocab_size, *, max_len=16, d_model=64, n_layers=2, n_heads=4, pad_id=0):
        """stage 2: tok_emb, fixed sinusoidal positions (buffer pos_emb), n_layers TransformerBlocks, final LayerNorm."""
        raise NotImplementedError("stage2: implement TextTransformer.__init__")

    def forward(self, tokens):
        """stage 2: tokens (B, T) int64 (T <= max_len, pad_id = padding) -> (B, d_model); pads are neither attended to nor pooled."""
        raise NotImplementedError("stage2: implement TextTransformer.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, TextTransformer=TextTransformer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MiniCLIP(nn.Module):
    """an image encoder and a text encoder, projected into one embedding space, with a learnable temperature."""

    def __init__(self, image_encoder, text_encoder, image_dim, text_dim, *, embed_dim=128, init_temperature=0.07):
        """stage 3: image_proj, text_proj (Linear, no bias) to embed_dim; log_scale = Parameter(log(1 / init_temperature))."""
        raise NotImplementedError("stage3: implement MiniCLIP.__init__")

    def encode_image(self, images):
        """stage 3: (B, embed_dim) unit vectors."""
        raise NotImplementedError("stage3: implement MiniCLIP.encode_image")

    def encode_text(self, texts):
        """stage 3: (B, embed_dim) unit vectors."""
        raise NotImplementedError("stage3: implement MiniCLIP.encode_text")

    def forward(self, images, texts):
        """stage 3: (image_emb, text_emb, logit_scale): normalised embeddings and exp(log_scale) clamped to at most 100."""
        raise NotImplementedError("stage3: implement MiniCLIP.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, MiniCLIP=MiniCLIP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def zero_shot_predict(image_emb, class_emb):
    """stage 4: (B, d) and (n_classes, d) -> (B,) int64: the class of highest cosine similarity."""
    raise NotImplementedError("stage4: implement zero_shot_predict")


@app.function
def class_embeddings(model, class_names, templates, tokenize):
    """stage 4: (n_classes, d) unit vectors: per class, the mean of the normalised text embeddings of every template, renormalised."""
    raise NotImplementedError("stage4: implement class_embeddings")


@app.cell(hide_code=True)
def _():
    checks.run(4, zero_shot_predict=zero_shot_predict, class_embeddings=class_embeddings, MiniCLIP=MiniCLIP)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def train_clip(model, images, texts, *, epochs=1, batch_size=256, lr=1e-3, seed=0, device="cpu"):
    """stage 5: Adam on clip_loss over shuffled (image, caption) batches; history {'loss', 'logit_scale'} per step."""
    raise NotImplementedError("stage5: implement train_clip")


@app.cell(hide_code=True)
def _():
    checks.run(5, train_clip=train_clip, MiniCLIP=MiniCLIP, zero_shot_predict=zero_shot_predict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `clip_loss`, `MiniCLIP`, `zero_shot_predict` and
    `class_embeddings` are part of `mlfs.ssl`. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
