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
    # positional information · build (reference solution)

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
    pos = torch.arange(n_positions, dtype=torch.float64)[:, None]
    two_i = torch.arange(0, d_model, 2, dtype=torch.float64)
    angle = pos / 10000.0 ** (two_i / d_model)  # (T, ceil(d/2))
    pe = torch.zeros(n_positions, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)[:, : d_model // 2]
    return pe.float()


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
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim, self.base = head_dim, base
        # the tables are plain float32 attributes, not buffers, so model.to(torch.bfloat16) can't round them:
        # positions in the thousands need float32 angles. they follow the input's device on use.
        self._inv_freq64 = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
        self._build(max_seq_len, torch.device("cpu"))

    def _build(self, n, device):
        freqs = torch.outer(torch.arange(n, dtype=torch.float64), self._inv_freq64)  # (n, dh/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # angle of channel i and i + dh/2 is the same
        self.cos_cached = emb.cos().float().to(device)
        self.sin_cached = emb.sin().float().to(device)

    def forward(self, x, offset=0):
        """stage 2: x (..., T, head_dim) at positions offset .. offset + T - 1 -> rotated x, same shape and dtype."""
        T = x.shape[-2]
        if offset + T > self.cos_cached.shape[0]:
            self._build(2 * (offset + T), x.device)  # positions past max_seq_len: extend the tables
        if self.cos_cached.device != x.device:
            self.cos_cached, self.sin_cached = self.cos_cached.to(x.device), self.sin_cached.to(x.device)
        cos = self.cos_cached[offset:offset + T].to(x.dtype)
        sin = self.sin_cached[offset:offset + T].to(x.dtype)
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


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
    def power_of_two(n):
        start = 2.0 ** (-8.0 / n)
        return [start ** (h + 1) for h in range(n)]

    if math.log2(n_heads).is_integer():
        return torch.tensor(power_of_two(n_heads))
    m = 2 ** math.floor(math.log2(n_heads))
    return torch.tensor(power_of_two(m) + power_of_two(2 * m)[0::2][: n_heads - m])


@app.function
def alibi_bias(n_heads, T, *, device=None):
    """stage 3: (n_heads, T, T) float32: -slope_h * (i - j) for keys j <= i, -inf above the diagonal."""
    i = torch.arange(T, device=device)
    dist = (i[:, None] - i[None, :]).float()  # i - j
    bias = -alibi_slopes(n_heads).to(device)[:, None, None] * dist
    return bias.masked_fill(dist < 0, float("-inf"))


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
        super().__init__()
        if pos not in ("none", "learned", "sinusoidal", "rope", "alibi"):
            raise ValueError(f"unknown pos {pos!r}")
        self.pos, self.n_heads, self.max_len, self.d_model = pos, n_heads, max_len, d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        if pos == "learned":
            self.pos_emb = nn.Embedding(max_len, d_model)
        if pos == "rope":
            self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len=max_len)
        self.layers = nn.ModuleList(nn.ModuleDict({
            "ln1": nn.LayerNorm(d_model), "qkv": nn.Linear(d_model, 3 * d_model), "out": nn.Linear(d_model, d_model),
            "ln2": nn.LayerNorm(d_model),
            "mlp": nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)),
        }) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        """stage 4: idx (B, T) int64 -> logits (B, T, vocab_size), causal; 'learned' raises ValueError if T > max_len."""
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos == "learned":
            if T > self.max_len:
                raise ValueError(f"learned positions exist only up to max_len={self.max_len}, got T={T}")
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.pos == "sinusoidal":
            x = x + sinusoidal_positions(T, self.d_model).to(x.device)
        bias = alibi_bias(self.n_heads, T, device=idx.device) if self.pos == "alibi" else None
        for layer in self.layers:
            q, k, v = layer["qkv"](layer["ln1"](x)).chunk(3, dim=-1)
            q, k, v = (split_heads(t, self.n_heads) for t in (q, k, v))
            if self.pos == "rope":
                q, k = self.rope(q), self.rope(k)
            if bias is not None:  # alibi's bias already holds -inf above the diagonal
                out, _ = scaled_dot_product_attention(q, k, v, mask=bias)
            else:
                out, _ = scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + layer["out"](merge_heads(out))
            x = x + layer["mlp"](layer["ln2"](x))
        return self.head(self.ln_f(x))


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
