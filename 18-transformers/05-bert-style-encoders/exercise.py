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
    from mlfs.transformer import TransformerBlock  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BERT-style encoders · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `mask_tokens`: BERT's 80/10/10 corruption for masked language modeling → `mlfs.transformer` |
    | 2 | `BERT`: a bidirectional encoder with a tied MLM head → `mlfs.transformer` |
    | 3 | `mlm_loss`, `pretrain_mlm`: the pretraining objective and loop |
    | 4 | `BERTClassifier`, `finetune`: a classification head on the pretrained encoder, and fine-tuning |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def mask_tokens(idx, *, mask_token_id, vocab_size, mask_prob=0.15, special_ids=(), generator=None):
    """stage 1: (inputs, labels): mask_prob of non-special positions selected; 80% [MASK], 10% random, 10% kept."""
    raise NotImplementedError("stage1: implement mask_tokens")


@app.cell(hide_code=True)
def _():
    checks.run(1, mask_tokens=mask_tokens)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class BERT(nn.Module):
    """a bidirectional transformer encoder: token + learned positions, pre-LN blocks, final norm, tied MLM head."""

    def __init__(self, vocab_size, *, max_len=128, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.1):
        """stage 2: tok_emb, pos_emb, drop, blocks, ln_f; MLM head: mlm_dense, mlm_norm, mlm_bias (decoder = tok_emb)."""
        raise NotImplementedError("stage2: implement BERT.__init__")

    def forward(self, idx, attention_mask=None):
        """stage 2: idx (B, T), attention_mask (B, T) bool True = real token -> hidden (B, T, d_model)."""
        raise NotImplementedError("stage2: implement BERT.forward")

    def mlm_logits(self, hidden):
        """stage 2: hidden (B, T, d) -> (B, T, V): dense -> GELU -> LayerNorm -> tied decoder + bias."""
        raise NotImplementedError("stage2: implement BERT.mlm_logits")


@app.cell(hide_code=True)
def _():
    checks.run(2, BERT=BERT)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def mlm_loss(model, inputs, labels, attention_mask=None):
    """stage 3: cross-entropy of model.mlm_logits(model(inputs, attention_mask)) against labels, -100 ignored."""
    raise NotImplementedError("stage3: implement mlm_loss")


@app.function
def pretrain_mlm(model, data, *, mask_token_id, special_ids=(), attention_mask=None, steps=500, batch_size=32,
                 lr=1e-3, mask_prob=0.15, seed=0):
    """stage 3: AdamW on mlm_loss over random rows of data (N, T), freshly masked every step; returns the list of losses."""
    raise NotImplementedError("stage3: implement pretrain_mlm")


@app.cell(hide_code=True)
def _():
    checks.run(3, mlm_loss=mlm_loss, pretrain_mlm=pretrain_mlm, BERT=BERT, mask_tokens=mask_tokens)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class BERTClassifier(nn.Module):
    """an encoder plus a classification head on the first ([CLS]) token or on the mean of the real tokens."""

    def __init__(self, encoder, num_classes, *, pool="cls", dropout=0.1):
        """stage 4: keeps `encoder`, a `head` nn.Linear(d_model, num_classes); pool 'cls' or 'mean' (ValueError otherwise)."""
        raise NotImplementedError("stage4: implement BERTClassifier.__init__")

    def forward(self, idx, attention_mask=None):
        """stage 4: idx (B, T), attention_mask (B, T) or None -> logits (B, num_classes)."""
        raise NotImplementedError("stage4: implement BERTClassifier.forward")


@app.function
def finetune(model, X, y, *, attention_mask=None, epochs=5, batch_size=32, lr=3e-4, seed=0):
    """stage 4: AdamW on cross-entropy over shuffled minibatches of (X, y) for `epochs`; returns per-epoch mean losses."""
    raise NotImplementedError("stage4: implement finetune")


@app.cell(hide_code=True)
def _():
    checks.run(4, BERTClassifier=BERTClassifier, finetune=finetune, BERT=BERT)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import BERT, mask_tokens` gives you a bidirectional encoder and its
    pretraining corruption. chapter 25 returns to masked modeling for images (MAE) and to
    contrastive pretraining, and chapter 27 to fine-tuning.
    """)
    return


if __name__ == "__main__":
    app.run()
