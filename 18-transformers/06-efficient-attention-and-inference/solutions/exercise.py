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
    from mlfs.transformer import sample_next_token  # yours, from topic 04


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # efficient attention & inference · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `LayerKVCache`, `KVCache`: storage for the keys and values of past tokens → `mlfs.transformer` |
    | 2 | `generate_cached`: generation that processes each token once → `mlfs.transformer` |
    | 3 | `online_softmax_attention`: exact attention over key blocks with a running max and normalizer (FlashAttention's forward) → `mlfs.transformer` |
    | 4 | `repeat_kv`, `kv_cache_bytes`: grouped-query attention's sharing, and what a cache costs |
    | 5 | `linear_attention`, `linear_attention_step`: attention with a kernel feature map, in parallel and recurrent form |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class LayerKVCache:
    """one layer's cache: keys and values (B, H, T, dh) of every token seen so far."""

    def __init__(self):
        """stage 1: an empty cache (no keys or values yet)."""
        self.k = None
        self.v = None

    @property
    def seq_len(self):
        """stage 1: the number of cached positions (0 when empty)."""
        return 0 if self.k is None else self.k.shape[-2]

    def update(self, k, v):
        """stage 1: append k, v (B, H, T_new, dh) along the time axis; return the full (k, v)."""
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=-2)
            self.v = torch.cat([self.v, v], dim=-2)
        return self.k, self.v


@app.class_definition
class KVCache:
    """a cache for a whole model: one LayerKVCache per layer."""

    def __init__(self, n_layers):
        """stage 1: n_layers empty LayerKVCache objects."""
        self.layers = [LayerKVCache() for _ in range(n_layers)]

    def __getitem__(self, i):
        """stage 1: layer i's cache."""
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    @property
    def seq_len(self):
        """stage 1: tokens cached so far (layer 0's count)."""
        return self.layers[0].seq_len

    def reset(self):
        """stage 1: empty every layer."""
        for layer in self.layers:
            layer.k = layer.v = None


@app.cell(hide_code=True)
def _():
    checks.run(1, LayerKVCache=LayerKVCache, KVCache=KVCache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, *, temperature=1.0, top_k=None, top_p=None, generator=None):
    """stage 2: prefill idx, then feed one token per step through model(token, cache=cache); returns (B, T + new)."""
    block_size = getattr(model, "block_size", None)
    if block_size is not None and idx.shape[1] + max_new_tokens > block_size:
        raise ValueError(f"{idx.shape[1]} + {max_new_tokens} tokens don't fit in block_size={block_size}; a cache "
                         "can't crop the context")
    was_training = model.training
    model.eval()
    try:
        cache = KVCache(len(model.blocks))
        logits, _ = model(idx, cache=cache)  # prefill: the whole prompt in one pass
        out = [idx]
        for step in range(max_new_tokens):
            nxt = sample_next_token(logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p,
                                    generator=generator)[:, None].to(idx.device)
            out.append(nxt)
            if step < max_new_tokens - 1:
                logits, _ = model(nxt, cache=cache)  # decode: one new token, attending to the cache
    finally:
        model.train(was_training)
    return torch.cat(out, dim=1)


@app.cell(hide_code=True)
def _():
    checks.run(2, generate_cached=generate_cached, KVCache=KVCache)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def online_softmax_attention(q, k, v, *, block_size=64, is_causal=False):
    """stage 3: softmax(q k^T / sqrt(d)) v computed over key blocks with a running max and normalizer; returns out."""
    Tq, Tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    m = torch.full((*q.shape[:-1], 1), float("-inf"), dtype=q.dtype, device=q.device)  # running max
    l = torch.zeros_like(m)  # running sum of exp(score - m)
    acc = torch.zeros((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)  # running numerator
    q_pos = torch.arange(Tq, device=q.device)[:, None] + (Tk - Tq)  # absolute positions, aligned to the end
    for start in range(0, Tk, block_size):
        kb, vb = k[..., start:start + block_size, :], v[..., start:start + block_size, :]
        s = q @ kb.transpose(-2, -1) / math.sqrt(d)
        if is_causal:
            k_pos = torch.arange(start, start + kb.shape[-2], device=q.device)[None, :]
            s = s.masked_fill(k_pos > q_pos, float("-inf"))
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)  # rows with nothing seen yet
        p = torch.exp(s - m_safe)
        scale = torch.exp(m - m_safe)
        l = l * scale + p.sum(dim=-1, keepdim=True)
        acc = acc * scale + p @ vb
        m = m_new
    return torch.where(l > 0, acc / l.clamp(min=torch.finfo(l.dtype).tiny), torch.zeros_like(acc))


@app.cell(hide_code=True)
def _():
    checks.run(3, online_softmax_attention=online_softmax_attention)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def repeat_kv(x, n_rep):
    """stage 4: (B, H_kv, T, dh) -> (B, H_kv * n_rep, T, dh), each kv head repeated n_rep times in a row."""
    if n_rep == 1:
        return x
    B, H, T, dh = x.shape
    return x[:, :, None].expand(B, H, n_rep, T, dh).reshape(B, H * n_rep, T, dh)


@app.function
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, *, batch_size=1, bytes_per_value=2):
    """stage 4: bytes to cache keys and values: 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes."""
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * bytes_per_value


@app.cell(hide_code=True)
def _():
    checks.run(4, repeat_kv=repeat_kv, kv_cache_bytes=kv_cache_bytes)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def linear_attention(q, k, v, *, is_causal=False, eps=1e-6):
    """stage 5: phi(q) (sum_j phi(k_j) v_j^T) / (phi(q) . sum_j phi(k_j)), phi = elu + 1; causal sums over j <= i."""
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    if not is_causal:
        kv = fk.transpose(-2, -1) @ v  # (..., d, d_v)
        z = fk.sum(dim=-2)  # (..., d)
        return (fq @ kv) / ((fq * z[..., None, :]).sum(-1, keepdim=True) + eps)
    kv = torch.cumsum(fk[..., :, :, None] * v[..., :, None, :], dim=-3)  # (..., T, d, d_v) prefix sums
    z = torch.cumsum(fk, dim=-2)  # (..., T, d)
    num = (fq[..., :, None, :] @ kv).squeeze(-2)
    return num / ((fq * z).sum(-1, keepdim=True) + eps)


@app.function
def linear_attention_step(q, k, v, state=None, *, eps=1e-6):
    """stage 5: one causal step: q, k (..., d), v (..., d_v); state (S (..., d, d_v), z (..., d)) -> (out, new state)."""
    fq, fk = F.elu(q) + 1, F.elu(k) + 1
    if state is None:
        S = torch.zeros((*k.shape, v.shape[-1]), dtype=k.dtype, device=k.device)
        z = torch.zeros_like(k)
    else:
        S, z = state
    S = S + fk[..., :, None] * v[..., None, :]
    z = z + fk
    out = (fq[..., None, :] @ S).squeeze(-2) / ((fq * z).sum(-1, keepdim=True) + eps)
    return out, (S, z)


@app.cell(hide_code=True)
def _():
    checks.run(5, linear_attention=linear_attention, linear_attention_step=linear_attention_step)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.transformer import KVCache, generate_cached, online_softmax_attention` gives you
    fast inference; chapter 26 profiles these ideas, and chapter 27's Llama with grouped-query
    attention uses your cache.
    """)
    return


if __name__ == "__main__":
    app.run()
