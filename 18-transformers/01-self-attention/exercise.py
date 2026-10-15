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
    # self-attention · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design. everything here becomes part of **`mlfs.transformer`**,
    which the rest of the chapter (and chapters 21, 22, 25, 27 and 28) builds on.

    | stage | you build |
    |---|---|
    | 1 | `causal_mask`, `scaled_dot_product_attention`: masked, scaled attention over any batch shape → `mlfs.transformer` |
    | 2 | `split_heads`, `merge_heads`: the reshapes that turn one wide attention into many narrow ones |
    | 3 | `MultiHeadAttention`: projections, heads, self- and cross-attention, a cache hook → `mlfs.transformer` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def causal_mask(T, *, device=None):
    """stage 1: (T, T) bool, True on and below the diagonal (query i may attend to keys j <= i)."""
    raise NotImplementedError("stage1: implement causal_mask")


@app.function
def scaled_dot_product_attention(q, k, v, *, mask=None, is_causal=False, dropout_p=0.0):
    """stage 1: softmax(q k^T / sqrt(d) + masking) v over the last two dims; returns (out, weights)."""
    raise NotImplementedError("stage1: implement scaled_dot_product_attention")


@app.cell(hide_code=True)
def _():
    checks.run(1, causal_mask=causal_mask, scaled_dot_product_attention=scaled_dot_product_attention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def split_heads(x, n_heads):
    """stage 2: (B, T, d) -> (B, n_heads, T, d // n_heads); head h holds channels h*dh .. (h+1)*dh - 1."""
    raise NotImplementedError("stage2: implement split_heads")


@app.function
def merge_heads(x):
    """stage 2: (B, H, T, dh) -> (B, T, H * dh), the inverse of split_heads."""
    raise NotImplementedError("stage2: implement merge_heads")


@app.cell(hide_code=True)
def _():
    checks.run(2, split_heads=split_heads, merge_heads=merge_heads)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MultiHeadAttention(nn.Module):
    """multi-head attention with q/k/v/out projections; self-attention, or cross-attention over `context`."""

    def __init__(self, d_model, n_heads, *, dropout=0.0, bias=True):
        """stage 3: q_proj, k_proj, v_proj, out_proj (nn.Linear d -> d); ValueError if n_heads doesn't divide d_model."""
        raise NotImplementedError("stage3: implement MultiHeadAttention.__init__")

    def forward(self, x, context=None, *, mask=None, is_causal=False, cache=None):
        """stage 3: x (B, T, d), context (B, S, d) or None -> (B, T, d); cache.update(k, v) runs before attending."""
        raise NotImplementedError("stage3: implement MultiHeadAttention.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, MultiHeadAttention=MultiHeadAttention)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import MultiHeadAttention, scaled_dot_product_attention, causal_mask`
    gives you these pieces, and `nn.MultiheadAttention` and `F.scaled_dot_product_attention`
    are unlocked. topic 02 gives the tokens positions; topic 03 wraps your attention in a block.
    """)
    return


if __name__ == "__main__":
    app.run()
