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
    # efficient attention & inference · build

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
        raise NotImplementedError("stage1: implement LayerKVCache.__init__")

    @property
    def seq_len(self):
        """stage 1: the number of cached positions (0 when empty)."""
        raise NotImplementedError("stage1: implement LayerKVCache.seq_len")

    def update(self, k, v):
        """stage 1: append k, v (B, H, T_new, dh) along the time axis; return the full (k, v)."""
        raise NotImplementedError("stage1: implement LayerKVCache.update")


@app.class_definition
class KVCache:
    """a cache for a whole model: one LayerKVCache per layer."""

    def __init__(self, n_layers):
        """stage 1: n_layers empty LayerKVCache objects."""
        raise NotImplementedError("stage1: implement KVCache.__init__")

    def __getitem__(self, i):
        """stage 1: layer i's cache."""
        raise NotImplementedError("stage1: implement KVCache.__getitem__")

    @property
    def seq_len(self):
        """stage 1: tokens cached so far (layer 0's count)."""
        raise NotImplementedError("stage1: implement KVCache.seq_len")

    def reset(self):
        """stage 1: empty every layer."""
        raise NotImplementedError("stage1: implement KVCache.reset")


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
    raise NotImplementedError("stage2: implement generate_cached")


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
    raise NotImplementedError("stage3: implement online_softmax_attention")


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
    raise NotImplementedError("stage4: implement repeat_kv")


@app.function
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, *, batch_size=1, bytes_per_value=2):
    """stage 4: bytes to cache keys and values: 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes."""
    raise NotImplementedError("stage4: implement kv_cache_bytes")


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
    raise NotImplementedError("stage5: implement linear_attention")


@app.function
def linear_attention_step(q, k, v, state=None, *, eps=1e-6):
    """stage 5: one causal step: q, k (..., d), v (..., d_v); state (S (..., d, d_v), z (..., d)) -> (out, new state)."""
    raise NotImplementedError("stage5: implement linear_attention_step")


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
