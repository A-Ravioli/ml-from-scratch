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
    # the transformer block · build (reference solution)

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
        super().__init__()
        if activation not in ("gelu", "relu"):
            raise ValueError(f"activation must be 'gelu' or 'relu', got {activation!r}")
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """stage 1: (..., d_model) -> (..., d_model), applied to every position independently."""
        return self.dropout(self.fc2(self.act(self.fc1(x))))


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
        super().__init__()
        self.norm_first, self.has_cross = norm_first, cross_attention
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        if cross_attention:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context=None, *, mask=None, is_causal=False, cache=None, context_mask=None):
        """stage 2: x (B, T, d) -> (B, T, d); mask/is_causal/cache go to self-attention, context/context_mask to cross."""
        if self.has_cross and context is None:
            raise ValueError("this block has cross-attention: pass a context")
        if not self.has_cross and (context is not None or context_mask is not None):
            raise ValueError("this block has no cross-attention: context and context_mask must be None")
        if self.norm_first:  # pre-LN: x + f(LN(x))
            x = x + self.drop(self.attn(self.ln1(x), mask=mask, is_causal=is_causal, cache=cache))
            if self.has_cross:
                x = x + self.drop(self.cross_attn(self.ln_cross(x), context, mask=context_mask))
            return x + self.mlp(self.ln2(x))
        # post-LN: LN(x + f(x))
        x = self.ln1(x + self.drop(self.attn(x, mask=mask, is_causal=is_causal, cache=cache)))
        if self.has_cross:
            x = self.ln_cross(x + self.drop(self.cross_attn(x, context, mask=context_mask)))
        return self.ln2(x + self.mlp(x))


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
        super().__init__()
        self.d_model, self.norm_first = d_model, norm_first
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.register_buffer("pos", sinusoidal_positions(max_len, d_model), persistent=False)
        kw = dict(d_ff=d_ff, dropout=dropout, norm_first=norm_first)
        self.encoder = nn.ModuleList(TransformerBlock(d_model, n_heads, **kw) for _ in range(n_layers))
        self.decoder = nn.ModuleList(TransformerBlock(d_model, n_heads, cross_attention=True, **kw)
                                     for _ in range(n_layers))
        # pre-LN stacks end with a final norm (the residual stream is never normalized otherwise)
        self.enc_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.dec_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.head = nn.Linear(d_model, tgt_vocab_size)
        self.drop = nn.Dropout(dropout)

    def _embed(self, emb, idx):
        return self.drop(emb(idx) + self.pos[: idx.shape[1]])

    def encode(self, src, src_mask=None):
        """stage 3: src (B, S) int64, src_mask (B, S) bool (True = real token) or None -> memory (B, S, d)."""
        mask = None if src_mask is None else src_mask[:, None, None, :]
        h = self._embed(self.src_emb, src)
        for block in self.encoder:
            h = block(h, mask=mask)
        return self.enc_norm(h)

    def decode(self, tgt_in, memory, memory_mask=None):
        """stage 3: tgt_in (B, T), memory (B, S, d), memory_mask (B, S) bool or None -> logits (B, T, V), causal in tgt_in."""
        cmask = None if memory_mask is None else memory_mask[:, None, None, :]
        h = self._embed(self.tgt_emb, tgt_in)
        for block in self.decoder:
            h = block(h, memory, is_causal=True, context_mask=cmask)
        return self.head(self.dec_norm(h))

    def forward(self, src, tgt_in, *, src_mask=None):
        """stage 3: teacher forcing: logits (B, T, tgt_vocab_size) for every position of tgt_in."""
        return self.decode(tgt_in, self.encode(src, src_mask), src_mask)


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
    norms = []
    for block in blocks:
        sq = [p.grad.detach().pow(2).sum() for p in block.parameters() if p.grad is not None]
        norms.append(math.sqrt(float(sum(sq))) if sq else 0.0)
    return norms


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
