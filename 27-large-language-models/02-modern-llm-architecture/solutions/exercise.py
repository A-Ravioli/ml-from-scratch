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
    from mlfs.torchnn import RMSNorm  # yours, from ch 15 · 07
    from mlfs.transformer import RotaryEmbedding, repeat_kv, sample_next_token  # yours, from ch 18


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # modern LLM architecture · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `SwiGLU`: the gated feed-forward layer of Llama, PaLM and most LLMs since → `mlfs.llm` |
    | 2 | `GroupedQueryAttention`: RoPE attention whose keys and values are shared across groups of heads → `mlfs.llm` |
    | 3 | `MoE`: a mixture-of-experts layer with top-k routing and a load-balancing loss → `mlfs.llm` |
    | 4 | `LlamaBlock`: pre-norm RMSNorm, attention, SwiGLU or MoE → `mlfs.llm` |
    | 5 | `Llama`: the whole model, compatible with your chapter 18 KV cache and `generate_cached` → `mlfs.llm` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class SwiGLU(nn.Module):
    """the gated MLP: w_down(silu(w_gate x) * w_up x), no biases."""

    def __init__(self, d_model, d_ff=None):
        """stage 1: w_gate, w_up (d_model -> d_ff) and w_down (d_ff -> d_model); d_ff default int(8 d / 3) rounded up to a multiple of 8."""
        super().__init__()
        if d_ff is None:
            d_ff = 8 * math.ceil(int(8 * d_model / 3) / 8)
        self.d_model, self.d_ff = d_model, d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """stage 1: (..., d_model) -> (..., d_model)."""
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


@app.cell(hide_code=True)
def _():
    checks.run(1, SwiGLU=SwiGLU)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GroupedQueryAttention(nn.Module):
    """causal self-attention with rotary positions and n_kv_heads key/value heads shared by groups of query heads."""

    def __init__(self, d_model, n_heads, n_kv_heads, *, rope_base=10000.0, max_seq_len=2048, dropout=0.0):
        """stage 2: q_proj, k_proj, v_proj, o_proj (no biases), rope = RotaryEmbedding(head_dim), save_weights = False; ValueError if heads don't divide."""
        super().__init__()
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError(f"need d_model % n_heads == 0 and n_heads % n_kv_heads == 0, got {d_model}, {n_heads}, "
                             f"{n_kv_heads}")
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base, max_seq_len=max_seq_len)
        self.save_weights = False  # set True to keep last_weights (B, H, Tq, Tk) for inspection (chapter 28)
        self.last_weights = None

    def forward(self, x, *, cache=None):
        """stage 2: x (B, T, d) -> (B, T, d); with a LayerKVCache, rotate at offset cache.seq_len and append the n_kv_heads k, v; save_weights stores last_weights."""
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        offset = cache.seq_len if cache is not None else 0
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, v = cache.update(k, v)
        n_rep = self.n_heads // self.n_kv_heads
        k, v = repeat_kv(k, n_rep), repeat_kv(v, n_rep)
        Tk = k.shape[-2]
        p = self.dropout if self.training else 0.0
        if self.save_weights:  # explicit softmax so the attention pattern can be stored
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            self.last_weights = weights.detach()
            out = F.dropout(weights, p=p, training=self.training) @ v
        elif Tk == T:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:  # queries are the last T positions: query i sees keys up to Tk - T + i
            mask = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim))


@app.cell(hide_code=True)
def _():
    checks.run(2, GroupedQueryAttention=GroupedQueryAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class MoE(nn.Module):
    """a sparse mixture of SwiGLU experts: each token goes to its top_k experts by router probability."""

    def __init__(self, d_model, d_ff, n_experts, *, top_k=2):
        """stage 3: router (d_model -> n_experts, no bias) and experts (ModuleList of SwiGLU(d_model, d_ff)); ValueError if top_k > n_experts."""
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must be in 1..n_experts={n_experts}, got {top_k}")
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(n_experts))
        self.last_counts = torch.zeros(n_experts, dtype=torch.long)

    def forward(self, x):
        """stage 3: x (B, T, d) -> (out (B, T, d), aux_loss); each expert runs only on its tokens; Switch load-balancing aux."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        probs = F.softmax(self.router(flat).float(), dim=-1)  # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)  # (N, k)
        gates = (top_p / top_p.sum(dim=-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, gates[rows, slot, None] * expert(flat[rows]))
        counts = torch.bincount(top_i.flatten(), minlength=self.n_experts)
        self.last_counts = counts.detach()
        frac = counts.float() / top_i.numel()  # f_e: share of the N * k assignments
        aux = self.n_experts * (frac * probs.mean(dim=0)).sum()
        return out.reshape(shape), aux


@app.cell(hide_code=True)
def _():
    checks.run(3, MoE=MoE, SwiGLU=SwiGLU)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class LlamaBlock(nn.Module):
    """pre-norm block: x + attn(norm1(x)), then h + mlp(norm2(h)), with RMSNorm and SwiGLU (or MoE)."""

    def __init__(self, d_model, n_heads, n_kv_heads, *, d_ff=None, n_experts=0, top_k=2, max_seq_len=2048):
        """stage 4: norm1, attn (GroupedQueryAttention), norm2, mlp (SwiGLU, or MoE when n_experts > 0); aux_loss = 0.0."""
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len=max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MoE(d_model, d_ff, n_experts, top_k=top_k) if n_experts > 0 else SwiGLU(d_model, d_ff)
        self.aux_loss = 0.0

    def forward(self, x, *, cache=None):
        """stage 4: (B, T, d) -> (B, T, d); stores the MoE aux loss in self.aux_loss (0.0 for a dense block)."""
        h = x + self.attn(self.norm1(x), cache=cache)
        m = self.mlp(self.norm2(h))
        if isinstance(m, tuple):
            m, self.aux_loss = m
        else:
            self.aux_loss = 0.0
        return h + m


@app.cell(hide_code=True)
def _():
    checks.run(4, LlamaBlock=LlamaBlock, MoE=MoE, SwiGLU=SwiGLU, GroupedQueryAttention=GroupedQueryAttention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class Llama(nn.Module):
    """a Llama-style decoder-only LM: token embedding, LlamaBlocks, RMSNorm, (tied) head; no position embedding (RoPE)."""

    def __init__(self, vocab_size, *, block_size=256, n_layers=4, n_heads=4, n_kv_heads=2, d_model=128, d_ff=None,
                 n_experts=0, top_k=2, aux_loss_coef=0.01, tie_weights=True):
        """stage 5: tok_emb, blocks, norm, lm_head (no bias); N(0, 0.02) init, residual outputs scaled by 1/sqrt(2 n_layers)."""
        super().__init__()
        self.vocab_size, self.block_size, self.n_layers = vocab_size, block_size, n_layers
        self.aux_loss_coef = aux_loss_coef
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            LlamaBlock(d_model, n_heads, n_kv_heads, d_ff=d_ff, n_experts=n_experts, top_k=top_k,
                       max_seq_len=block_size) for _ in range(n_layers))
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
        for name, p in self.named_parameters():
            if name.endswith("attn.o_proj.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layers))
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def _stream(self, idx, cache=None):
        T = idx.shape[1]
        offset = cache.seq_len if cache is not None else 0
        if offset + T > self.block_size:
            raise ValueError(f"sequence of {offset + T} tokens is longer than block_size={self.block_size}")
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cache=None if cache is None else cache[i])
        return self.norm(x)

    def hidden_states(self, idx):
        """stage 5: idx (B, T) -> (B, T, d_model), the final-norm output that lm_head reads."""
        return self._stream(idx)

    def forward(self, idx, targets=None, *, cache=None):
        """stage 5: (logits (B, T, V), loss or None); loss = CE (ignore -100) + aux_loss_coef * sum of blocks' aux_loss."""
        logits = self.lm_head(self._stream(idx, cache))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
            loss = loss + self.aux_loss_coef * sum(block.aux_loss for block in self.blocks)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None):
        """stage 5: append max_new_tokens tokens sampled with sample_next_token; context cropped to block_size; no cache."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.block_size:])
                nxt = sample_next_token(logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                                        generator=generator)
                idx = torch.cat([idx, nxt[:, None].to(idx.device)], dim=1)
        finally:
            self.train(was_training)
        return idx


@app.cell(hide_code=True)
def _():
    checks.run(5, Llama=Llama, LlamaBlock=LlamaBlock, MoE=MoE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.llm import Llama` is the model the rest of the chapter fine-tunes, aligns,
    decodes and evaluates, and chapter 28 opens up to look inside.
    """)
    return


if __name__ == "__main__":
    app.run()
