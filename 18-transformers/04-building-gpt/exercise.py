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
    from mlfs.train import CosineSchedule  # yours, from ch 15 · topic 04
    from mlfs.transformer import TransformerBlock, sinusoidal_positions  # yours, from topics 02-03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # building GPT · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. everything here joins **`mlfs.transformer`**, and it's the
    model of the chapter's capstone and of chapter 27.

    | stage | you build |
    |---|---|
    | 1 | `get_batch`: random windows of a token stream and their next-token targets → `mlfs.transformer` |
    | 2 | `GPT`: a decoder-only transformer language model (forward, loss, cache hook, hidden states) → `mlfs.transformer` |
    | 3 | `sample_next_token`: temperature, top-k and nucleus (top-p) sampling → `mlfs.transformer` |
    | 4 | `GPT.generate`: autoregressive generation |
    | 5 | `estimate_loss`, `train_lm`: evaluation and the training loop (AdamW, warmup + cosine, clipping) → `mlfs.transformer` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def get_batch(data, block_size, batch_size, *, generator=None, device="cpu"):
    """stage 1: (x, y) int64 (batch_size, block_size): x = data[s:s+T], y = data[s+1:s+T+1], s uniform in [0, len - T - 1]."""
    raise NotImplementedError("stage1: implement get_batch")


@app.cell(hide_code=True)
def _():
    checks.run(1, get_batch=get_batch)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GPT(nn.Module):
    """a decoder-only transformer LM: token + position embeddings, pre-LN causal blocks, final norm, (tied) head."""

    def __init__(self, vocab_size, *, block_size=256, n_layers=4, n_heads=4, d_model=128, d_ff=None, dropout=0.0,
                 pos_encoding="learned", tie_weights=True):
        """stage 2: tok_emb, pos_emb (learned only), drop, blocks, ln_f, lm_head; GPT-2 initialization."""
        raise NotImplementedError("stage2: implement GPT.__init__")

    def hidden_states(self, idx):
        """stage 2: idx (B, T) -> (B, T, d_model), the final-norm output that the head reads."""
        raise NotImplementedError("stage2: implement GPT.hidden_states")

    def forward(self, idx, targets=None, *, cache=None):
        """stage 2: (logits (B, T, V), loss or None); targets pre-shifted, -100 ignored; cache: block i gets cache[i]."""
        raise NotImplementedError("stage2: implement GPT.forward")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None):
        """stage 4: append max_new_tokens sampled tokens to idx (B, T); the context is cropped to block_size; no cache."""
        raise NotImplementedError("stage4: implement GPT.generate")


@app.cell(hide_code=True)
def _():
    checks.run(2, GPT=GPT)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def sample_next_token(logits, *, temperature=1.0, top_k=None, top_p=None, generator=None):
    """stage 3: logits (B, V) -> (B,) int64; temperature 0 is argmax; top_k, then top_p (nucleus) filtering."""
    raise NotImplementedError("stage3: implement sample_next_token")


@app.cell(hide_code=True)
def _():
    checks.run(3, sample_next_token=sample_next_token)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, GPT=GPT, sample_next_token=sample_next_token)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def estimate_loss(model, data, block_size, *, batch_size=32, n_batches=20, device="cpu"):
    """stage 5: mean loss over n_batches batches (fixed generator seed 0), eval mode, no_grad; restores the mode."""
    raise NotImplementedError("stage5: implement estimate_loss")


@app.function
def train_lm(model, train_data, block_size, *, val_data=None, batch_size=32, max_steps=1000, lr=3e-4,
             weight_decay=0.1, warmup_steps=100, grad_clip=1.0, eval_interval=100, eval_batches=20, device="cpu",
             seed=0):
    """stage 5: AdamW (betas 0.9, 0.95; decay on >= 2-D params), warmup from 0 + cosine to lr/10, clipping; history."""
    raise NotImplementedError("stage5: implement train_lm")


@app.cell(hide_code=True)
def _():
    checks.run(5, estimate_loss=estimate_loss, train_lm=train_lm, GPT=GPT, get_batch=get_batch)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import GPT, train_lm, estimate_loss, get_batch, sample_next_token`
    gives you a complete language-model toolkit. topic 06 makes generation fast with a KV cache,
    and the capstone trains three GPTs on Shakespeare.
    """)
    return


if __name__ == "__main__":
    app.run()
