"""Checks for 18 · 01 self-attention.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. attention is checked
against PyTorch's F.scaled_dot_product_attention and nn.MultiheadAttention (oracles), against the
lesson's worked example (by hand), and for the conventions the rest of the course relies on: True
means "may attend", a causal mask aligned to the END when there are fewer queries than keys (that
is what makes a KV cache work), rows with nothing to attend to give zeros, and a float mask is an
additive bias.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------- stage 1: scaled dot-product attention


def test_stage1_causal_mask(nb):
    m = nb.causal_mask(4)
    assert m.dtype == torch.bool and tuple(m.shape) == (4, 4), f"a (4, 4) bool tensor; got {m.dtype} {tuple(m.shape)}"
    want = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(m, want), (
        f"True = may attend, so row i (a query) is True for columns j <= i: lower-triangular including the "
        f"diagonal.\nexpected\n{want.int()}\ngot\n{m.int()}")


def test_stage1_worked_example(nb):
    q = torch.tensor([[1.0, 1, 1, 1], [1, 1, -1, -1], [0, 0, 0, 0]])
    k = torch.tensor([[1.0, 1, 1, 1], [1, 1, -1, -1], [0, 0, 0, 0]])
    v = torch.tensor([[1.0, 0], [0, 1], [1, 1]])
    out, w = nb.scaled_dot_product_attention(q, k, v)
    a, b = math.exp(2) / (math.exp(2) + 2), 1 / (math.exp(2) + 2)
    want_w = torch.tensor([[a, b, b], [b, a, b], [1 / 3, 1 / 3, 1 / 3]])
    assert tuple(w.shape) == (3, 3) and tuple(out.shape) == (3, 2), (
        f"weights (Tq, Tk) = (3, 3), out (Tq, d_v) = (3, 2); got {tuple(w.shape)}, {tuple(out.shape)}")
    if _close(w, torch.softmax(q @ k.T, -1), atol=1e-4):
        raise AssertionError("your weights are softmax(q k^T): you forgot to divide the scores by sqrt(d) = 2")
    assert _close(w, want_w, atol=1e-4), (
        f"worked example: the scores q k^T / sqrt(4) of query 1 are (2, 0, 0), so its weights are "
        f"(e^2, 1, 1) / (e^2 + 2) = ({a:.4f}, {b:.4f}, {b:.4f}).\nexpected\n{want_w}\ngot\n{w}")
    assert _close(out, want_w @ v, atol=1e-4), f"out = weights @ v = {want_w @ v}; got {out}"
    out_c, w_c = nb.scaled_dot_product_attention(q, k, v, is_causal=True)
    c = 1 / (1 + math.exp(2))
    want_c = torch.tensor([[1.0, 0, 0], [c, 1 - c, 0], [1 / 3, 1 / 3, 1 / 3]])
    assert _close(w_c, want_c, atol=1e-4), (
        f"with is_causal=True, query 1 sees only key 1 (weight 1), query 2 sees keys 1-2 with scores (0, 2).\n"
        f"expected\n{want_c}\ngot\n{w_c}")


def test_stage1_matches_pytorch(nb):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, 5, 8, generator=g)
    k = torch.randn(2, 3, 7, 8, generator=g)
    v = torch.randn(2, 3, 7, 6, generator=g)
    out, w = nb.scaled_dot_product_attention(q, k, v)
    assert tuple(out.shape) == (2, 3, 5, 6) and tuple(w.shape) == (2, 3, 5, 7), (
        f"batch dims are kept: out (2, 3, 5, 6), weights (2, 3, 5, 7); got {tuple(out.shape)}, {tuple(w.shape)}")
    assert _close(out, F.scaled_dot_product_attention(q, k, v)), (
        "out differs from F.scaled_dot_product_attention on random (B, H, T, d) inputs. use the last two dims only: "
        "q @ k.transpose(-2, -1), softmax over the LAST dim (the keys), scale by sqrt(q.shape[-1])")
    assert _close(w.sum(-1), torch.ones(2, 3, 5)), "each row of weights is a distribution over the keys: it sums to 1"
    mask = torch.rand(5, 7, generator=g) > 0.4
    mask[:, 0] = True
    out_m, w_m = nb.scaled_dot_product_attention(q, k, v, mask=mask)
    assert _close(out_m, F.scaled_dot_product_attention(q, k, v, attn_mask=mask)), (
        "with a bool mask (True = may attend), broadcast over the batch dims, out must match "
        "F.scaled_dot_product_attention(q, k, v, attn_mask=mask). if it looks inverted, check which value means "
        "'blocked': here it's False")
    assert (w_m[..., ~mask] == 0).all(), "weights at masked positions must be exactly 0 (fill the scores with -inf)"


def test_stage1_is_causal_aligned_to_the_end(nb):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(1, 6, 4, generator=g) for _ in range(3))
    out, _ = nb.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert _close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True)), (
        "with as many queries as keys, is_causal=True must equal the lower-triangular mask")
    q2 = torch.randn(1, 2, 4, generator=g)
    _, w = nb.scaled_dot_product_attention(q2, k, v, is_causal=True)
    want = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(w[0] > 0, want), (
        f"2 queries, 6 keys: the queries are the LAST two tokens of the sequence (positions 4 and 5), so query 0 sees "
        f"keys 0..4 and query 1 sees all 6 (tril with diagonal = Tk - Tq). this is what lets a KV cache feed one new "
        f"token at a time.\nexpected nonzero pattern\n{want.int()}\ngot\n{(w[0] > 0).int()}")
    m = torch.ones(6, 6, dtype=torch.bool)
    m[:, 1] = False
    _, w2 = nb.scaled_dot_product_attention(q, k, v, mask=m, is_causal=True)
    assert (w2[0, :, 1] == 0).all() and (w2[0].triu(1) == 0).all(), (
        "mask and is_causal together: a key is allowed only if both allow it")


def test_stage1_empty_rows_and_float_masks(nb):
    g = torch.Generator().manual_seed(2)
    q, k, v = torch.randn(3, 4, generator=g), torch.randn(5, 4, generator=g), torch.randn(5, 3, generator=g)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1] = False
    out, w = nb.scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.isfinite(out).all() and torch.isfinite(w).all(), (
        "a query whose whole row is masked must not produce NaN (softmax of all -inf is 0/0)")
    assert (out[1] == 0).all() and (w[1] == 0).all(), (
        "a query with no key to attend to gets all-zero weights and a zero output (as F.scaled_dot_product_attention "
        "does); find such rows and zero them")
    bias = torch.randn(3, 5, generator=g)
    out_b, _ = nb.scaled_dot_product_attention(q, k, v, mask=bias)
    assert _close(out_b, F.scaled_dot_product_attention(q, k, v, attn_mask=bias)), (
        "a float mask is an additive bias on the scores (after scaling), as in F.scaled_dot_product_attention. "
        "topic 02's ALiBi uses this")
    qc, kc, vc = torch.randn(2, 5, 4, generator=g), torch.randn(2, 5, 4, generator=g), torch.randn(2, 5, 3, generator=g)
    bias5 = torch.randn(5, 5, generator=g)
    combined = bias5 + torch.full((5, 5), float("-inf")).triu(1)
    out_c, w_c = nb.scaled_dot_product_attention(qc, kc, vc, mask=bias5, is_causal=True)
    assert _close(out_c, F.scaled_dot_product_attention(qc, kc, vc, attn_mask=combined)), (
        "a float mask together with is_causal=True: add the bias to the scores AND block the future (the bias "
        "doesn't replace the causal mask, and a float mask is not a bool mask)")
    assert (w_c.triu(1) == 0).all(), "with is_causal=True, future keys get weight 0 even when a float bias is given"


def test_stage1_gradients_and_dropout(nb):
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, 4, 3, dtype=torch.float64, generator=g, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda a, b, c: nb.scaled_dot_product_attention(a, b, c, is_causal=True)[0],
                                    (q, k, v)), "gradients don't match finite differences: use differentiable torch ops"
    q, k, v = (torch.randn(4, 3, generator=g) for _ in range(3))
    ref, w0 = nb.scaled_dot_product_attention(q, k, v)
    torch.manual_seed(0)
    outs = [nb.scaled_dot_product_attention(q, k, v, dropout_p=0.5) for _ in range(3000)]
    assert _close(outs[0][1], w0), "the returned weights are the attention probabilities before dropout"
    assert not _close(outs[0][0], ref), "with dropout_p=0.5 some attention weights are dropped, so out changes"
    mean = torch.stack([o for o, _ in outs]).mean(0)
    assert _close(mean, ref, atol=0.08), (
        "dropout on the weights must be inverted (survivors scaled by 1/(1-p)), so on average out is unchanged; "
        f"mean over 3000 draws {mean} vs {ref}")


# ---------------------------------------------------------------- stage 2: heads


def test_stage2_split_and_merge(nb):
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    h = nb.split_heads(x, 4)
    assert tuple(h.shape) == (2, 4, 3, 2), f"(B, T, d) = (2, 3, 8) with 4 heads -> (B, H, T, dh) = (2, 4, 3, 2); got {tuple(h.shape)}"
    assert torch.equal(h[1, 2, 0], x[1, 0, 4:6]), (
        "head h of token t holds channels h*dh .. (h+1)*dh - 1 of that token. view as (B, T, H, dh) THEN transpose to "
        "(B, H, T, dh); viewing directly as (B, H, T, dh) mixes tokens across heads")
    assert torch.equal(nb.merge_heads(h), x), "merge_heads(split_heads(x)) must give back x exactly"
    y = torch.randn(2, 4, 3, 2)
    assert torch.equal(nb.split_heads(nb.merge_heads(y), 4), y), "split_heads(merge_heads(y)) gives back y"


# ---------------------------------------------------------------- stage 3: multi-head attention


def _copy_to_torch(m, d, H, bias=True):
    ref = nn.MultiheadAttention(d, H, bias=bias, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(torch.cat([m.q_proj.weight, m.k_proj.weight, m.v_proj.weight]))
        ref.out_proj.weight.copy_(m.out_proj.weight)
        if bias:
            ref.in_proj_bias.copy_(torch.cat([m.q_proj.bias, m.k_proj.bias, m.v_proj.bias]))
            ref.out_proj.bias.copy_(m.out_proj.bias)
    return ref.eval()


def test_stage3_structure(nb):
    m = nb.MultiHeadAttention(16, 4)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        layer = getattr(m, name, None)
        assert isinstance(layer, nn.Linear) and tuple(layer.weight.shape) == (16, 16), (
            f"{name} must be an nn.Linear(d_model, d_model)")
    assert m.save_weights is False, "save_weights starts False"
    assert nb.MultiHeadAttention(16, 4, bias=False).q_proj.bias is None, "bias=False removes every projection bias"
    with pytest.raises(ValueError):
        nb.MultiHeadAttention(10, 4)


def test_stage3_matches_nn_multiheadattention(nb):
    torch.manual_seed(0)
    m = nb.MultiHeadAttention(16, 4).eval()
    ref = _copy_to_torch(m, 16, 4)
    x = torch.randn(2, 5, 16)
    assert tuple(m(x).shape) == (2, 5, 16), "output is (B, T, d_model)"
    assert _close(m(x), ref(x, x, x, need_weights=False)[0]), (
        "self-attention differs from nn.MultiheadAttention with the same weights. project, split into heads, attend "
        "in every head with sqrt(head_dim) scaling, merge, then out_proj")
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)  # torch's convention: True = blocked
    assert _close(m(x, is_causal=True), ref(x, x, x, attn_mask=causal, need_weights=False)[0]), (
        "causal self-attention differs from nn.MultiheadAttention with a causal mask")
    ctx = torch.randn(2, 7, 16)
    assert _close(m(x, ctx), ref(x, ctx, ctx, need_weights=False)[0]), (
        "cross-attention: queries from x, keys and values from context (B, S, d) with S != T")
    pad = torch.ones(2, 1, 1, 7, dtype=torch.bool)
    pad[1, ..., 5:] = False
    assert _close(m(x, ctx, mask=pad), ref(x, ctx, ctx, key_padding_mask=~pad[:, 0, 0], need_weights=False)[0]), (
        "a (B, 1, 1, S) bool mask (True = real key) must broadcast over heads and queries")
    m2 = nb.MultiHeadAttention(16, 2, bias=False).eval()
    assert _close(m2(x), _copy_to_torch(m2, 16, 2, bias=False)(x, x, x, need_weights=False)[0]), "bias=False version"


class _AppendCache:
    """the smallest object with the cache protocol: update(k, v) appends along the time axis and returns everything."""

    def __init__(self):
        self.k = self.v = None
        self.calls = 0

    def update(self, k, v):
        self.calls += 1
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v


def test_stage3_cache_gives_the_same_answer(nb):
    torch.manual_seed(1)
    m = nb.MultiHeadAttention(12, 3).eval()
    x = torch.randn(2, 6, 12)
    full = m(x, is_causal=True)
    cache = _AppendCache()
    pieces = [m(x[:, :3], is_causal=True, cache=cache)]
    for t in range(3, 6):
        pieces.append(m(x[:, t:t + 1], is_causal=True, cache=cache))
    assert cache.calls == 4, f"forward must call cache.update(k, v) once per call; it was called {cache.calls} times"
    assert tuple(cache.k.shape) == (2, 3, 6, 4), (
        f"the cache stores keys in the heads layout (B, H, T, head_dim) = (2, 3, 6, 4); got {tuple(cache.k.shape)}")
    assert _close(torch.cat(pieces, 1), full), (
        "running a prompt of 3 tokens and then one token at a time through the cache must reproduce causal attention "
        "over the whole sequence. update the cache before attending, and let is_causal align to the end")


def test_stage3_weights_dropout_equivariance(nb):
    torch.manual_seed(2)
    m = nb.MultiHeadAttention(8, 2, dropout=0.5)
    x = torch.randn(1, 4, 8)
    m.eval()
    assert _close(m(x), m(x)), "in eval mode attention dropout is off: the output is deterministic"
    m.train()
    assert not _close(m(x), m(x)), "in training mode the module applies attention dropout (pass dropout_p)"
    m.eval()
    m.save_weights = True
    m(x, is_causal=True)
    w = m.last_weights
    assert w is not None and tuple(w.shape) == (1, 2, 4, 4) and not w.requires_grad, (
        "with save_weights=True, forward stores last_weights (B, H, Tq, Tk), detached")
    perm = torch.tensor([2, 0, 3, 1])
    assert _close(m(x)[:, perm], m(x[:, perm])), (
        "without a mask, self-attention is permutation-equivariant: shuffling the tokens shuffles the outputs the "
        "same way. if this fails, something mixes positions other than attention")
