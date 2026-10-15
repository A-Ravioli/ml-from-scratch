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
    from mlfs.transformer import merge_heads, scaled_dot_product_attention, split_heads  # yours, from topic 01


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # positional information · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `sinusoidal_positions`: the original transformer's fixed position vectors → `mlfs.transformer` |
    | 2 | `RotaryEmbedding`: rotate queries and keys by an angle proportional to position (RoPE) → `mlfs.transformer` |
    | 3 | `alibi_bias`: a per-head linear penalty on distance, added to the scores → `mlfs.transformer` |
    | 4 | `PositionalLM`: a tiny causal language model that can use any of the five schemes, for the lab |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def sinusoidal_positions(n_positions, d_model):
    """stage 1: (n_positions, d_model) float32: PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(same angle)."""
    raise NotImplementedError("stage1: implement sinusoidal_positions")


@app.cell(hide_code=True)
def _():
    checks.run(1, sinusoidal_positions=sinusoidal_positions)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RotaryEmbedding(nn.Module):
    """rotary position embedding (RoPE), rotate-half convention: channel i is paired with channel i + head_dim/2."""

    def __init__(self, head_dim, *, base=10000.0, max_seq_len=2048):
        """stage 2: frequencies theta_i = base^(-2i/head_dim), i < head_dim/2; cos/sin tables for max_seq_len positions."""
        raise NotImplementedError("stage2: implement RotaryEmbedding.__init__")

    def forward(self, x, offset=0):
        """stage 2: x (..., T, head_dim) at positions offset .. offset + T - 1 -> rotated x, same shape and dtype."""
        raise NotImplementedError("stage2: implement RotaryEmbedding.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, RotaryEmbedding=RotaryEmbedding)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def alibi_slopes(n_heads):
    """stage 3: Press et al.'s head slopes: 2^(-8h/n) for h = 1..n when n is a power of 2, interleaved otherwise."""
    raise NotImplementedError("stage3: implement alibi_slopes")


@app.function
def alibi_bias(n_heads, T, *, device=None):
    """stage 3: (n_heads, T, T) float32: -slope_h * (i - j) for keys j <= i, -inf above the diagonal."""
    raise NotImplementedError("stage3: implement alibi_bias")


@app.cell(hide_code=True)
def _():
    checks.run(3, alibi_bias=alibi_bias, alibi_slopes=alibi_slopes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class PositionalLM(nn.Module):
    """a small causal LM (pre-LN attention + MLP layers) whose only difference between variants is `pos`."""

    def __init__(self, vocab_size, *, d_model=64, n_heads=4, n_layers=2, max_len=256, pos="rope"):
        """stage 4: pos in {'none', 'learned', 'sinusoidal', 'rope', 'alibi'}; ValueError for anything else."""
        raise NotImplementedError("stage4: implement PositionalLM.__init__")

    def forward(self, idx):
        """stage 4: idx (B, T) int64 -> logits (B, T, vocab_size), causal; 'learned' raises ValueError if T > max_len."""
        raise NotImplementedError("stage4: implement PositionalLM.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, PositionalLM=PositionalLM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import sinusoidal_positions, RotaryEmbedding, alibi_bias` gives you
    these schemes. GPT (topic 04) adds learned or sinusoidal positions to its input; chapter 27's
    Llama rotates its queries and keys with your `RotaryEmbedding`.
    """)
    return


if __name__ == "__main__":
    app.run()
