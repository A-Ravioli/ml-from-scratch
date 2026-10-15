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
    from mlfs.transformer import MultiHeadAttention, sinusoidal_positions  # yours, from topics 01-02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the transformer block · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `FeedForward`: the position-wise MLP → `mlfs.transformer` |
    | 2 | `TransformerBlock`: attention + MLP on a residual stream, pre-LN or post-LN, optional cross-attention → `mlfs.transformer` |
    | 3 | `EncoderDecoder`: the original transformer, an encoder stack and a decoder stack that reads it |
    | 4 | `block_grad_norms`: how much gradient each block receives (for the lab) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class FeedForward(nn.Module):
    """the position-wise MLP: fc1 (d -> d_ff) -> activation -> fc2 (d_ff -> d) -> dropout."""

    def __init__(self, d_model, d_ff=None, *, dropout=0.0, activation="gelu"):
        """stage 1: fc1, fc2 (nn.Linear), d_ff defaults to 4 * d_model; activation 'gelu' or 'relu' (ValueError otherwise)."""
        raise NotImplementedError("stage1: implement FeedForward.__init__")

    def forward(self, x):
        """stage 1: (..., d_model) -> (..., d_model), applied to every position independently."""
        raise NotImplementedError("stage1: implement FeedForward.forward")


@app.cell(hide_code=True)
def _():
    checks.run(1, FeedForward=FeedForward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class TransformerBlock(nn.Module):
    """one transformer layer: self-attention, optional cross-attention and an MLP, each a residual branch."""

    def __init__(self, d_model, n_heads, *, d_ff=None, dropout=0.0, norm_first=True, cross_attention=False):
        """stage 2: ln1, attn, [ln_cross, cross_attn], ln2, mlp; dropout on each branch's output."""
        raise NotImplementedError("stage2: implement TransformerBlock.__init__")

    def forward(self, x, context=None, *, mask=None, is_causal=False, cache=None, context_mask=None):
        """stage 2: x (B, T, d) -> (B, T, d); mask/is_causal/cache go to self-attention, context/context_mask to cross."""
        raise NotImplementedError("stage2: implement TransformerBlock.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, TransformerBlock=TransformerBlock, FeedForward=FeedForward)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class EncoderDecoder(nn.Module):
    """the original transformer: an encoder over the source, a causal decoder with cross-attention over its output."""

    def __init__(self, src_vocab_size, tgt_vocab_size, *, d_model=128, n_heads=4, n_layers=2, d_ff=None, dropout=0.0,
                 max_len=256, norm_first=True):
        """stage 3: src_emb, tgt_emb, encoder and decoder (nn.ModuleList of TransformerBlock), enc_norm, dec_norm, head."""
        raise NotImplementedError("stage3: implement EncoderDecoder.__init__")

    def encode(self, src, src_mask=None):
        """stage 3: src (B, S) int64, src_mask (B, S) bool (True = real token) or None -> memory (B, S, d)."""
        raise NotImplementedError("stage3: implement EncoderDecoder.encode")

    def decode(self, tgt_in, memory, memory_mask=None):
        """stage 3: tgt_in (B, T), memory (B, S, d), memory_mask (B, S) bool or None -> logits (B, T, V), causal in tgt_in."""
        raise NotImplementedError("stage3: implement EncoderDecoder.decode")

    def forward(self, src, tgt_in, *, src_mask=None):
        """stage 3: teacher forcing: logits (B, T, tgt_vocab_size) for every position of tgt_in."""
        raise NotImplementedError("stage3: implement EncoderDecoder.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, EncoderDecoder=EncoderDecoder, TransformerBlock=TransformerBlock)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def block_grad_norms(blocks):
    """stage 4: list of floats, the 2-norm of the gradient over each block's parameters (after backward); 0.0 if none."""
    raise NotImplementedError("stage4: implement block_grad_norms")


@app.cell(hide_code=True)
def _():
    checks.run(4, block_grad_norms=block_grad_norms, TransformerBlock=TransformerBlock)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import TransformerBlock, FeedForward` gives you the block that GPT
    (topic 04), BERT (05) and the ViT (07) stack, and `nn.TransformerEncoderLayer`,
    `nn.TransformerDecoderLayer` and `nn.Transformer` are unlocked.
    """)
    return


if __name__ == "__main__":
    app.run()
