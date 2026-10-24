"""Checks for 18 · 06 efficient attention & inference.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the cache is checked for
the protocol GPT (and chapter 27's Llama) rely on; cached generation against GPT.generate (oracle) and
for feeding exactly one token per step; online-softmax attention against F.scaled_dot_product_attention,
for numerical safety, and (by watching every matrix product it computes) for never forming the full
score matrix; grouped-query sharing and cache sizes against hand computations; linear attention against
explicit loops, and its recurrent form against its parallel form.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

from mlfs.transformer import GPT, scaled_dot_product_attention  # noqa: E402  (reference, from topics 01 and 04)


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------- stage 1: the cache


def test_stage1_layer_cache(nb):
    c = nb.LayerKVCache()
    assert c.seq_len == 0, "an empty cache has seq_len 0"
    k1, v1 = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4, 5)
    k, v = c.update(k1, v1)
    assert torch.equal(k, k1) and torch.equal(v, v1) and c.seq_len == 4, "the first update stores and returns k, v"
    k2, v2 = torch.randn(2, 3, 1, 5), torch.randn(2, 3, 1, 5)
    k, v = c.update(k2, v2)
    assert tuple(k.shape) == (2, 3, 5, 5) and torch.equal(k[:, :, 4:], k2) and torch.equal(v[:, :, :4], v1), (
        "update appends along the time axis (dim -2) and returns everything cached so far")
    assert c.seq_len == 5, f"seq_len counts cached positions: 5; got {c.seq_len}"


def test_stage1_model_cache(nb):
    cache = nb.KVCache(3)
    assert cache.seq_len == 0
    layers = [cache[i] for i in range(3)]
    assert len({id(x) for x in layers}) == 3, "cache[i] gives three different LayerKVCache objects"
    assert cache[1] is layers[1], "cache[i] returns the same object every time (it holds that layer's state)"
    for i in range(3):
        cache[i].update(torch.randn(1, 2, 6, 4), torch.randn(1, 2, 6, 4))
    assert cache.seq_len == 6, "seq_len is the number of tokens cached so far"
    cache.reset()
    assert cache.seq_len == 0 and all(cache[i].seq_len == 0 for i in range(3)), "reset() empties every layer"


# ---------------------------------------------------------------- stage 2: cached generation


def test_stage2_matches_uncached_generation(nb):
    for pos in ("learned", "sinusoidal"):
        torch.manual_seed(0)
        m = GPT(13, block_size=32, n_layers=2, n_heads=2, d_model=16, pos_encoding=pos).train()
        idx = torch.randint(0, 13, (2, 5))
        out = nb.generate_cached(m, idx, 20, temperature=0.0)
        assert tuple(out.shape) == (2, 25) and torch.equal(out[:, :5], idx), "returns (B, T + max_new_tokens), prompt first"
        assert torch.equal(out, m.generate(idx, 20, temperature=0.0)), (
            f"pos_encoding={pos!r}: greedy cached generation must produce exactly GPT.generate's tokens. if they diverge "
            "after the first new token, the positions of cached steps are probably wrong")
        assert m.training, "generate_cached restores the model's mode"
        a = nb.generate_cached(m, idx, 15, top_k=5, generator=torch.Generator().manual_seed(3))
        b = m.generate(idx, 15, top_k=5, generator=torch.Generator().manual_seed(3))
        assert torch.equal(a, b), "with the same generator and options, sampling gives the same tokens as GPT.generate"
    with pytest.raises(ValueError):
        nb.generate_cached(m, idx, 28)  # 5 + 28 > block_size = 32


def test_stage2_sampling_options_and_eval_mode(nb):
    torch.manual_seed(5)
    m = GPT(13, block_size=40, n_layers=2, n_heads=2, d_model=16, dropout=0.3).train()
    idx = torch.randint(0, 13, (3, 4))
    for kw in (dict(temperature=0.7, top_p=0.9), dict(temperature=1.3, top_k=4), dict(temperature=0.5)):
        a = nb.generate_cached(m, idx, 25, generator=torch.Generator().manual_seed(9), **kw)
        b = m.generate(idx, 25, generator=torch.Generator().manual_seed(9), **kw)
        assert torch.equal(a, b), (
            f"with {kw} and the same generator, generate_cached must return exactly GPT.generate's tokens: pass "
            f"temperature, top_k and top_p to sample_next_token, and run the model in eval mode (this GPT has "
            f"dropout 0.3 and was left in training mode)")
    assert m.training, "and restore the training mode afterwards"


class _Recorder(nn.Module):
    """wraps a GPT; records the length of every input it's called with (no block_size attribute)."""

    def __init__(self, gpt):
        super().__init__()
        self.gpt, self.blocks, self.calls = gpt, gpt.blocks, []

    def forward(self, idx, targets=None, *, cache=None):
        self.calls.append((idx.shape[1], cache is not None))
        return self.gpt(idx, targets, cache=cache)


def test_stage2_one_token_per_step(nb):
    torch.manual_seed(1)
    rec = _Recorder(GPT(9, block_size=64, n_layers=2, n_heads=2, d_model=16))
    nb.generate_cached(rec, torch.randint(0, 9, (1, 6)), 10, temperature=0.0)
    lengths = [n for n, _ in rec.calls]
    assert all(c for _, c in rec.calls), "every call must pass the cache"
    assert lengths[0] == 6 and all(n == 1 for n in lengths[1:]), (
        f"prefill the whole prompt once, then feed ONE new token per step; the model was called with lengths {lengths}")
    assert len(lengths) <= 10, (
        f"10 new tokens need at most 10 model calls (the prefill gives the first); got {len(lengths)}")


# ---------------------------------------------------------------- stage 3: online softmax


class _MatmulSpy(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.bmm, torch.einsum,
                    torch.Tensor.__rmatmul__) and torch.is_tensor(out):
            self.shapes.append(tuple(out.shape))
        return out


def test_stage3_matches_sdpa(nb):
    q1, k1 = torch.ones(1, 1), torch.tensor([[1.0], [3.0], [2.0], [5.0]])
    v1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    got1 = nb.online_softmax_attention(q1, k1, v1, block_size=2)
    assert abs(got1.item() - 3.688) < 1e-3, (
        f"worked example: scores (1, 3, 2, 5), values (1, 2, 3, 4), blocks of 2 -> 3.688; got {got1.item():.4f}")
    g = torch.Generator().manual_seed(0)
    q, k, v = torch.randn(2, 3, 20, 8, generator=g), torch.randn(2, 3, 37, 8, generator=g), torch.randn(2, 3, 37, 5, generator=g)
    want = F.scaled_dot_product_attention(q, k, v)
    for bs in (1, 7, 16, 64):
        got = nb.online_softmax_attention(q, k, v, block_size=bs)
        assert tuple(got.shape) == (2, 3, 20, 5) and _close(got, want), (
            f"block_size={bs}: the blockwise result must equal ordinary attention exactly (up to rounding). rescale the "
            f"running sum AND the accumulated output by exp(m_old - m_new) whenever the max grows")
    qs = torch.randn(1, 2, 37, 8, generator=g)
    assert _close(nb.online_softmax_attention(qs, k[:1, :2], v[:1, :2], block_size=10, is_causal=True),
                  F.scaled_dot_product_attention(qs, k[:1, :2], v[:1, :2], is_causal=True)), "causal, Tq = Tk"
    want_end = scaled_dot_product_attention(q, k, v, is_causal=True)[0]
    assert _close(nb.online_softmax_attention(q, k, v, block_size=8, is_causal=True), want_end), (
        "causal with Tq < Tk aligns to the end, like topic 01: query i sees keys <= Tk - Tq + i")


def test_stage3_numerics_and_tiling(nb):
    g = torch.Generator().manual_seed(1)
    q = 60 * torch.randn(1, 6, 4, generator=g)
    k, v = torch.randn(1, 50, 4, generator=g), torch.randn(1, 50, 3, generator=g)
    got = nb.online_softmax_attention(q, k, v, block_size=8)
    assert torch.isfinite(got).all() and _close(got, F.scaled_dot_product_attention(q, k, v), atol=1e-4), (
        "scores in the hundreds must not overflow: subtract the running max before exponentiating")
    q2, k2, v2 = torch.randn(1, 12, 4, generator=g), torch.randn(1, 50, 4, generator=g), torch.randn(1, 50, 3, generator=g)
    spy = _MatmulSpy()
    with spy:
        nb.online_softmax_attention(q2, k2, v2, block_size=8)
    big = [s for s in spy.shapes if s[-1] > 8 or (len(s) >= 2 and s[-2] > 12)]
    assert spy.shapes and not big, (
        f"with block_size=8, no matrix product may produce more than 8 key columns at once: the point is never to "
        f"form the full (Tq, Tk) = (12, 50) score matrix. products seen: {sorted(set(spy.shapes))}")


# ---------------------------------------------------------------- stage 4: grouped-query attention


def test_stage4_repeat_kv(nb):
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    r = nb.repeat_kv(x, 2)
    assert tuple(r.shape) == (2, 6, 4, 5), f"(B, H_kv, T, dh) = (2, 3, 4, 5) with n_rep 2 -> (2, 6, 4, 5); got {tuple(r.shape)}"
    for h in range(6):
        assert torch.equal(r[:, h], x[:, h // 2]), (
            "query head h uses kv head h // n_rep: each kv head is repeated n_rep times IN A ROW (0, 0, 1, 1, 2, 2), "
            "not tiled (0, 1, 2, 0, 1, 2)")
    assert torch.equal(nb.repeat_kv(x, 1), x), "n_rep=1 is the identity (multi-head attention)"


def test_stage4_cache_bytes(nb):
    got = nb.kv_cache_bytes(32, 32, 128, 4096)
    assert got == 2 * 32 * 32 * 128 * 4096 * 2 == 2_147_483_648, (
        f"worked example: a Llama-2-7B-shaped cache (32 layers, 32 kv heads of 128, 4,096 tokens, 2-byte values) holds "
        f"keys AND values: 2 * 32 * 32 * 128 * 4096 * 2 = 2 GiB; got {got:,}")
    assert nb.kv_cache_bytes(32, 8, 128, 4096) == got // 4, "8 kv heads instead of 32 (GQA) is 4x smaller"
    assert nb.kv_cache_bytes(32, 32, 128, 4096, batch_size=3, bytes_per_value=4) == got * 6, "batch and dtype scale it"


# ---------------------------------------------------------------- stage 5: linear attention


def _phi(x):
    return F.elu(x) + 1


def test_stage5_parallel_forms(nb):
    g = torch.Generator().manual_seed(2)
    q, k = (torch.randn(2, 7, 4, generator=g, dtype=torch.float64) for _ in range(2))
    v = torch.randn(2, 7, 3, generator=g, dtype=torch.float64)
    got = nb.linear_attention(q, k, v)
    want = torch.zeros(2, 7, 3, dtype=torch.float64)
    want_c = torch.zeros_like(want)
    for b in range(2):
        for i in range(7):
            w = torch.stack([_phi(q[b, i]) @ _phi(k[b, j]) for j in range(7)])
            want[b, i] = (w[:, None] * v[b]).sum(0) / w.sum()
            want_c[b, i] = (w[:i + 1, None] * v[b, :i + 1]).sum(0) / w[:i + 1].sum()
    assert tuple(got.shape) == (2, 7, 3) and _close(got, want, atol=1e-6), (
        "non-causal linear attention: weights phi(q_i) . phi(k_j) with phi(x) = elu(x) + 1, normalized to sum to 1, "
        "times v. compute it as phi(q) (phi(k)^T v) / (phi(q) . sum_j phi(k_j)), never forming the (T, T) matrix")
    got_c = nb.linear_attention(q, k, v, is_causal=True)
    assert _close(got_c, want_c, atol=1e-6), "causal: the same, with sums over j <= i only (prefix sums)"
    const = nb.linear_attention(q, k, torch.ones(2, 7, 3, dtype=torch.float64) * 2.5, is_causal=True)
    assert _close(const, torch.full_like(const, 2.5), atol=1e-5), "the weights are normalized: constant values come out unchanged"


def test_stage5_recurrent_form(nb):
    g = torch.Generator().manual_seed(3)
    q, k, v = torch.randn(3, 9, 4, generator=g), torch.randn(3, 9, 4, generator=g), torch.randn(3, 9, 5, generator=g)
    want = nb.linear_attention(q, k, v, is_causal=True)
    state, outs = None, []
    for t in range(9):
        o, state = nb.linear_attention_step(q[:, t], k[:, t], v[:, t], state)
        outs.append(o)
    S, z = state
    assert tuple(S.shape) == (3, 4, 5) and tuple(z.shape) == (3, 4), (
        f"the state is (S (B, d, d_v), z (B, d)), whatever the sequence length; got {tuple(S.shape)}, {tuple(z.shape)}")
    assert _close(torch.stack(outs, 1), want, atol=1e-5), (
        "running linear_attention_step token by token must reproduce causal linear_attention: S accumulates "
        "phi(k_t) v_t^T and z accumulates phi(k_t), BEFORE computing the output of step t")
