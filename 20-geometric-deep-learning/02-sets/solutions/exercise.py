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
    # sets · build (reference solution)

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
    if aggregation not in ("sum", "mean", "max"):
        raise ValueError(f"aggregation must be 'sum', 'mean' or 'max', got {aggregation!r}")
    if mask is None:
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
    m = mask.unsqueeze(-1)
    if aggregation == "max":
        out = h.masked_fill(~m, float("-inf")).amax(dim=1)
        return torch.where(m.any(dim=1), out, torch.zeros_like(out))
    total = (h * m).sum(dim=1)
    if aggregation == "sum":
        return total
    return total / m.sum(dim=1).clamp(min=1)


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
        super().__init__()
        if aggregation not in ("sum", "mean", "max"):
            raise ValueError(f"aggregation must be 'sum', 'mean' or 'max', got {aggregation!r}")
        self.aggregation = aggregation
        self.phi = MLP(in_dim, [hidden_dim], hidden_dim)
        self.rho = MLP(hidden_dim, [hidden_dim], out_dim)

    def forward(self, x, mask=None):
        """stage 2: x (B, N, in_dim), mask (B, N) bool or None -> (B, out_dim)."""
        return self.rho(masked_pool(self.phi(x), mask, self.aggregation))


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
        super().__init__()
        if aggregation not in ("sum", "mean", "max"):
            raise ValueError(f"aggregation must be 'sum', 'mean' or 'max', got {aggregation!r}")
        self.aggregation = aggregation
        self.lam = nn.Linear(in_dim, out_dim)
        self.gam = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, mask=None):
        """stage 3: x (B, N, in_dim) -> (B, N, out_dim); pool over real elements only; padded rows of the output are 0."""
        pooled = masked_pool(x, mask, self.aggregation)
        y = self.lam(x) + self.gam(pooled).unsqueeze(1)
        if mask is not None:
            y = y * mask.unsqueeze(-1)
        return y


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
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is None:
        return torch.softmax(scores, dim=-1) @ v
    keep = mask.unsqueeze(1)  # (B, 1, N)
    nonempty = mask.any(dim=-1)[:, None, None]  # (B, 1, 1)
    scores = scores.masked_fill(~keep, float("-inf"))
    scores = torch.where(nonempty, scores, torch.zeros_like(scores))  # no all--inf rows: no nan, even in gradients
    w = torch.softmax(scores, dim=-1) * nonempty  # a set with no real elements attends to nothing
    return w @ v


@app.class_definition
class AttentionPool(nn.Module):
    """pooling by multi-head attention (PMA): n_seeds learned queries attend over the set's elements."""

    def __init__(self, dim, *, heads=1, n_seeds=1):
        """stage 4: learned seeds (n_seeds, dim), query/key/value/output projections dim -> dim; dim % heads == 0."""
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.dim, self.heads, self.n_seeds = dim, heads, n_seeds
        self.seeds = nn.Parameter(torch.randn(n_seeds, dim) / math.sqrt(dim))
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, mask=None):
        """stage 4: x (B, N, dim) -> (B, n_seeds, dim); each head attends with masked_attention over real elements."""
        B, N, _ = x.shape
        h, d = self.heads, self.dim // self.heads

        def split(t):
            return t.reshape(B, -1, h, d).transpose(1, 2).reshape(B * h, -1, d)

        q = split(self.q(self.seeds).expand(B, -1, -1))
        k, v = split(self.k(x)), split(self.v(x))
        m = None if mask is None else mask.repeat_interleave(h, dim=0)
        out = masked_attention(q, k, v, m)
        out = out.reshape(B, h, self.n_seeds, d).transpose(1, 2).reshape(B, self.n_seeds, self.dim)
        return self.o(out)


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
