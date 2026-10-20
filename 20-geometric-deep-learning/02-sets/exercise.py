import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # sets · build

    spec-driven (L4): each stage's exact specification is in the lesson's **spec** box, the
    checks enforce it, and the internal structure (which layers, which attribute names) is
    yours. sets come in batches padded to a common size, with a boolean mask saying which
    rows are real; every function here must ignore the padding exactly.

    | stage | you build |
    |---|---|
    | 1 | `masked_pool`: sum, mean and max over the real elements of each set |
    | 2 | `DeepSets`: $\rho(\bigoplus_i \phi(x_i))$ → `mlfs.gnn` |
    | 3 | `PermEquivariant`: the equivariant linear layer $x_i\Lambda + \operatorname{pool}(X)\Gamma$ |
    | 4 | `masked_attention`, `AttentionPool`: pooling by attention, as in the set transformer |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def masked_pool(h, mask=None, aggregation="sum"):
    """stage 1: pool h (B, N, F) over the real elements (mask (B, N) bool, None = all real) -> (B, F); empty sets -> 0."""
    raise NotImplementedError("stage1: implement masked_pool")


@app.cell(hide_code=True)
def _():
    checks.run(1, masked_pool=masked_pool)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class DeepSets(nn.Module):
    """rho(pool_i phi(x_i)): a permutation-invariant network on padded batches of sets."""

    def __init__(self, in_dim, hidden_dim, out_dim, *, aggregation="sum"):
        """stage 2: phi: in_dim -> hidden_dim (an MLP), rho: hidden_dim -> out_dim (an MLP); store aggregation (validated)."""
        raise NotImplementedError("stage2: implement DeepSets.__init__")

    def forward(self, x, mask=None):
        """stage 2: x (B, N, in_dim), mask (B, N) bool or None -> (B, out_dim)."""
        raise NotImplementedError("stage2: implement DeepSets.forward")


@app.cell(hide_code=True)
def _():
    checks.run(2, DeepSets=DeepSets)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class PermEquivariant(nn.Module):
    """the permutation-equivariant linear layer y_i = x_i Lambda + pool(X) Gamma + b (Zaheer et al., 2017)."""

    def __init__(self, in_dim, out_dim, *, aggregation="mean"):
        """stage 3: parameters Lambda (in_dim, out_dim) with bias b, Gamma (in_dim, out_dim) without; store aggregation."""
        raise NotImplementedError("stage3: implement PermEquivariant.__init__")

    def forward(self, x, mask=None):
        """stage 3: x (B, N, in_dim) -> (B, N, out_dim); pool over real elements only; padded rows of the output are 0."""
        raise NotImplementedError("stage3: implement PermEquivariant.forward")


@app.cell(hide_code=True)
def _():
    checks.run(3, PermEquivariant=PermEquivariant)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def masked_attention(q, k, v, mask=None):
    """stage 4: softmax(q k^T / sqrt(d)) v over the real keys: q (B, M, d), k (B, N, d), v (B, N, dv), mask (B, N) -> (B, M, dv)."""
    raise NotImplementedError("stage4: implement masked_attention")


@app.class_definition
class AttentionPool(nn.Module):
    """pooling by multi-head attention (PMA): n_seeds learned queries attend over the set's elements."""

    def __init__(self, dim, *, heads=1, n_seeds=1):
        """stage 4: learned seeds (n_seeds, dim), query/key/value/output projections dim -> dim; dim % heads == 0."""
        raise NotImplementedError("stage4: implement AttentionPool.__init__")

    def forward(self, x, mask=None):
        """stage 4: x (B, N, dim) -> (B, n_seeds, dim); each head attends with masked_attention over real elements."""
        raise NotImplementedError("stage4: implement AttentionPool.forward")


@app.cell(hide_code=True)
def _():
    checks.run(4, masked_attention=masked_attention, AttentionPool=AttentionPool)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.gnn import DeepSets` gives you the invariant set network.
    """)
    return


if __name__ == "__main__":
    app.run()
