"""Checks for 27 · 02 modern LLM architecture.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. SwiGLU against its formula and
its d_ff rule; grouped-query attention against an attention written out here (rotary positions, causal
softmax, kv heads shared by consecutive query heads), for MHA (n_kv_heads == n_heads), GQA and MQA, with the
KV cache, dropout in train/eval mode and rope_base; the MoE layer against a dense loop over every token, for
its gates, top_k, the load-balancing loss and sparse dispatch; the block's pre-norm wiring; the model for its
loss (with aux_loss_coef), tied weights, exact parameter count, cached generation and learning.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.transformer import KVCache, LayerKVCache, RotaryEmbedding, generate_cached  # noqa: E402


# ---------------------------------------------------------------- stage 1: SwiGLU


def test_stage1_swiglu(nb):
    torch.manual_seed(0)
    m = nb.SwiGLU(12, 20)
    names = {n for n, _ in m.named_parameters()}
    assert names == {"w_gate.weight", "w_up.weight", "w_down.weight"}, (
        f"three bias-free linear layers named w_gate, w_up, w_down; got parameters {sorted(names)}")
    assert tuple(m.w_gate.weight.shape) == (20, 12) and tuple(m.w_down.weight.shape) == (12, 20)
    x = torch.randn(2, 5, 12)
    want = m.w_down(F.silu(m.w_gate(x)) * m.w_up(x))
    assert torch.allclose(m(x), want, atol=1e-6), (
        "SwiGLU(x) = w_down(silu(w_gate x) * w_up x): the gate goes through SiLU, the up projection doesn't")
    for d, dff in ((128, 344), (96, 256), (10, 32), (64, 176)):
        got = nb.SwiGLU(d).w_up.weight.shape[0]
        assert got == dff, (
            f"default d_ff for d_model={d}: int(8 * {d} / 3) = {int(8 * d / 3)}, rounded up to a multiple of 8 = {dff}; "
            f"got {got}")


# ---------------------------------------------------------------- stage 2: grouped-query attention


def _reference_attention(attn, x, rope_base=10000.0, offset=0):
    """attention written out: rotary q/k, kv head j serves query heads j*n_rep .. (j+1)*n_rep - 1, causal softmax."""
    B, T, d = x.shape
    H, Hkv = attn.n_heads, attn.n_kv_heads
    dh = d // H
    q = (x @ attn.q_proj.weight.T).view(B, T, H, dh).transpose(1, 2)
    k = (x @ attn.k_proj.weight.T).view(B, T, Hkv, dh).transpose(1, 2)
    v = (x @ attn.v_proj.weight.T).view(B, T, Hkv, dh).transpose(1, 2)
    rope = RotaryEmbedding(dh, base=rope_base)
    q, k = rope(q, offset), rope(k, offset)
    rep = H // Hkv
    k = torch.stack([k[:, h // rep] for h in range(H)], dim=1)
    v = torch.stack([v[:, h // rep] for h in range(H)], dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(dh)
    scores = scores.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), float("-inf"))
    out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d)
    return out @ attn.o_proj.weight.T


@pytest.mark.parametrize("n_heads, n_kv_heads", [(4, 4), (4, 2), (4, 1), (6, 3)])
def test_stage2_matches_written_out_attention(nb, n_heads, n_kv_heads):
    torch.manual_seed(0)
    d = 24
    attn = nb.GroupedQueryAttention(d, n_heads, n_kv_heads).eval()
    dh = d // n_heads
    assert tuple(attn.q_proj.weight.shape) == (d, d) and tuple(attn.k_proj.weight.shape) == (n_kv_heads * dh, d), (
        f"q_proj: d -> n_heads * head_dim; k_proj and v_proj: d -> n_kv_heads * head_dim = {n_kv_heads * dh}")
    assert all(getattr(attn, n).bias is None for n in ("q_proj", "k_proj", "v_proj", "o_proj")), "no biases (Llama)"
    x = torch.randn(2, 7, d)
    got = attn(x)
    want = _reference_attention(attn, x)
    what = {n_heads: "MHA", 1: "MQA"}.get(n_kv_heads, "GQA")
    assert torch.allclose(got, want, atol=1e-5), (
        f"{what} (n_heads={n_heads}, n_kv_heads={n_kv_heads}) differs from the written-out attention by "
        f"{(got - want).abs().max():.2e}. check: RoPE on q and k (not v), kv head j shared by query heads "
        "j*n_rep..(j+1)*n_rep-1 (repeat_kv order), causal mask, scaling by sqrt(head_dim)")


def test_stage2_errors_and_rope(nb):
    with pytest.raises(ValueError):
        nb.GroupedQueryAttention(24, 4, 3)
    with pytest.raises(ValueError):
        nb.GroupedQueryAttention(25, 4, 2)
    torch.manual_seed(1)
    attn = nb.GroupedQueryAttention(16, 4, 2, rope_base=100.0).eval()
    x = torch.randn(1, 9, 16)
    assert torch.allclose(attn(x), _reference_attention(attn, x, rope_base=100.0), atol=1e-5), (
        "rope_base sets the rotary frequencies: pass it to RotaryEmbedding(head_dim, base=rope_base, ...)")
    assert isinstance(attn.rope, nn.Module) and getattr(attn.rope, "base", 100.0) == 100.0
    big = nb.GroupedQueryAttention(16, 4, 2, max_seq_len=4096)
    cos = getattr(big.rope, "cos_cached", None)
    if cos is not None:
        assert cos.shape[0] >= 4096, "max_seq_len is passed to the RotaryEmbedding"


def test_stage2_causal_and_cache(nb):
    torch.manual_seed(2)
    attn = nb.GroupedQueryAttention(16, 4, 2).eval()
    x = torch.randn(2, 8, 16)
    full = attn(x)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(2, 3, 16)
    assert torch.allclose(attn(x2)[:, :5], full[:, :5], atol=1e-6), "causal: changing tokens 5..7 can't change outputs 0..4"
    cache = LayerKVCache()
    parts = [attn(x[:, :5], cache=cache)]
    k_cached = cache.k
    assert k_cached.shape == (2, 2, 5, 4), (
        f"the cache stores the n_kv_heads=2 rotated keys (B, n_kv_heads, T, head_dim) = (2, 2, 5, 4), before repeat_kv; "
        f"got {tuple(k_cached.shape)}. caching the repeated heads wastes the memory GQA is meant to save")
    for t in range(5, 8):
        parts.append(attn(x[:, t:t + 1], cache=cache))
    stitched = torch.cat(parts, dim=1)
    assert torch.allclose(stitched, full, atol=1e-5), (
        f"prefill 5 tokens, then 3 one at a time through the cache, must equal one pass over all 8 (max diff "
        f"{(stitched - full).abs().max():.2e}). new tokens are rotated at offset cache.seq_len (read before update) "
        "and attend to every cached key")
    cache2 = LayerKVCache()
    attn(x[:, :3], cache=cache2)
    two = attn(x[:, 3:5], cache=cache2)
    assert torch.allclose(two, full[:, 3:5], atol=1e-5), (
        "two new tokens at once with a cache: query i may see keys up to cache length + i, not the second new token")


def test_stage2_save_weights(nb):
    torch.manual_seed(4)
    attn = nb.GroupedQueryAttention(16, 4, 2).eval()
    assert attn.save_weights is False, "save_weights defaults to False"
    x = torch.randn(2, 6, 16)
    plain = attn(x)
    attn.save_weights = True
    out = attn(x)
    w = attn.last_weights
    assert torch.allclose(out, plain, atol=1e-5), "save_weights=True computes the same output (explicit softmax path)"
    assert w is not None and w.shape == (2, 4, 6, 6) and not w.requires_grad, (
        "last_weights: (B, n_heads, Tq, Tk), detached")
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-5), "each query's weights sum to 1"
    assert torch.all(w.triu(diagonal=1) == 0), "causal: no weight on future keys"
    cache = LayerKVCache()
    attn(x[:, :4], cache=cache)
    attn(x[:, 4:], cache=cache)
    w2 = attn.last_weights
    assert w2.shape == (2, 4, 2, 6) and torch.allclose(w2, w[:, :, 4:], atol=1e-5), (
        "with a cache, the 2 new queries attend to all 6 keys, the mask aligned to the end: the same rows as without it")
    drop = nb.GroupedQueryAttention(16, 4, 2, dropout=0.5).train()
    drop.save_weights = True
    drop(x)
    assert torch.allclose(drop.last_weights.sum(-1), torch.ones(2, 4, 6), atol=1e-5), (
        "last_weights are stored BEFORE attention dropout (as in chapter 18's MultiHeadAttention): rows sum to 1")


def test_stage2_dropout_modes(nb):
    torch.manual_seed(3)
    x = torch.randn(2, 6, 16)
    attn = nb.GroupedQueryAttention(16, 4, 2, dropout=0.5)
    attn.train()
    a, b = attn(x), attn(x)
    assert not torch.allclose(a, b), "dropout=0.5 in train mode: attention weights are dropped, two calls differ"
    attn.eval()
    assert torch.allclose(attn(x), attn(x)), "in eval mode no dropout: deterministic"
    assert torch.allclose(attn(x), _reference_attention(attn, x), atol=1e-5), "eval mode is plain attention"
    plain = nb.GroupedQueryAttention(16, 4, 2).train()
    assert torch.allclose(plain(x), plain(x)), "dropout defaults to 0"


# ---------------------------------------------------------------- stage 3: mixture of experts


def _dense_moe(moe, x):
    """every expert on every token, combined with renormalized top-k router probabilities."""
    flat = x.reshape(-1, x.shape[-1])
    probs = F.softmax(flat @ moe.router.weight.T, dim=-1)
    out = torch.zeros_like(flat)
    for n in range(flat.shape[0]):
        top_p, top_i = probs[n].topk(moe.top_k)
        gates = top_p / top_p.sum()
        for g, e in zip(gates, top_i):
            out[n] += g * moe.experts[int(e)](flat[n:n + 1])[0]
    return out.reshape(x.shape), probs


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_stage3_matches_dense_loop(nb, top_k):
    torch.manual_seed(0)
    moe = nb.MoE(8, 16, 4, top_k=top_k)
    assert isinstance(moe.router, nn.Linear) and moe.router.bias is None and len(moe.experts) == 4
    assert all(e.w_up.weight.shape == (16, 8) for e in moe.experts), "experts are SwiGLU(d_model, d_ff)"
    x = torch.randn(2, 5, 8)
    out, aux = moe(x)
    want, probs = _dense_moe(moe, x)
    assert out.shape == x.shape and torch.allclose(out, want, atol=1e-5), (
        f"top_k={top_k}: each token's output is the sum over its top-{top_k} experts (by router softmax) of "
        "gate * expert(token), gates = the chosen probabilities renormalized to sum to 1. max diff "
        f"{(out - want).abs().max():.2e}")
    assert aux.dim() == 0


def test_stage3_aux_loss(nb):
    torch.manual_seed(1)
    moe = nb.MoE(6, 8, 3, top_k=2)
    x = torch.randn(4, 5, 6)
    _, aux = moe(x)
    probs = F.softmax(x.reshape(-1, 6) @ moe.router.weight.T, dim=-1)
    top = probs.topk(2, dim=-1).indices
    f = torch.bincount(top.flatten(), minlength=3).float() / top.numel()
    want = 3 * (f * probs.mean(0)).sum()
    assert torch.allclose(aux, want, atol=1e-5), (
        f"aux = n_experts * sum_e f_e * P_e, f_e = share of the N*top_k (token, slot) assignments that go to e, P_e = "
        f"mean router probability of e; want {want.item():.4f}, got {float(aux):.4f}")
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[0, :] = 5.0  # every token prefers expert 0
    x = torch.rand(3, 4, 6) + 1
    _, aux = moe(x)
    assert aux.item() > 1.3, "a router that sends everything to one expert must be penalized (aux well above 1)"
    assert moe.last_counts.tolist()[0] == 12 and int(moe.last_counts.sum()) == 24, (
        f"last_counts: (token, slot) assignments per expert in the last forward; 12 tokens x top_k=2: expert 0 gets all "
        f"12 first slots; got {moe.last_counts.tolist()}")
    w = torch.randn(3, 6)
    moe.router.weight.data.copy_(w)
    xr = torch.randn(64, 6)
    xr.requires_grad_(True)
    out, aux = moe(xr[None])
    aux.backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0, (
        "the aux loss is differentiable in the router weights (through P_e)")
    moe.zero_grad()
    out, _ = moe(xr[None])
    out.sum().backward()
    assert moe.router.weight.grad.abs().sum() > 0, "the output depends on the router through the gates: gradients flow"


def test_stage3_sparse_dispatch(nb):
    torch.manual_seed(2)
    moe = nb.MoE(8, 16, 4, top_k=2)
    seen = {}
    for e, expert in enumerate(moe.experts):
        expert.register_forward_hook(lambda m, inp, out, e=e: seen.setdefault(e, []).append(inp[0].reshape(-1, 8).shape[0]))
    x = torch.randn(3, 10, 8)
    moe(x)
    rows = {e: sum(v) for e, v in seen.items()}
    assert sum(rows.values()) == 30 * 2, (
        f"30 tokens x top_k=2: the experts together process 60 rows, each expert only the tokens routed to it; "
        f"they processed {rows}. running every expert on every token is a dense MoE (top_k/n_experts of the work)")
    probs = F.softmax(x.reshape(-1, 8) @ moe.router.weight.T, dim=-1)
    want = torch.bincount(probs.topk(2, -1).indices.flatten(), minlength=4)
    assert [rows.get(e, 0) for e in range(4)] == want.tolist()
    with pytest.raises(ValueError):
        nb.MoE(8, 16, 4, top_k=5)


# ---------------------------------------------------------------- stage 4: block


def test_stage4_block_wiring(nb):
    torch.manual_seed(0)
    blk = nb.LlamaBlock(16, 4, 2, d_ff=40).eval()
    assert type(blk.norm1).__name__ == "RMSNorm" and type(blk.norm2).__name__ == "RMSNorm", (
        "norm1 and norm2 are RMSNorm (from mlfs.torchnn)")
    assert isinstance(blk.attn, nb.GroupedQueryAttention) and blk.attn.n_kv_heads == 2
    assert isinstance(blk.mlp, nb.SwiGLU) and blk.mlp.w_up.weight.shape[0] == 40, "d_ff is passed to the SwiGLU"
    with torch.no_grad():
        blk.norm1.weight.uniform_(0.5, 1.5)
        blk.norm2.weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 6, 16)
    h = x + blk.attn(blk.norm1(x))
    want = h + blk.mlp(blk.norm2(h))
    assert torch.allclose(blk(x), want, atol=1e-5), (
        "pre-norm: h = x + attn(norm1(x)); out = h + mlp(norm2(h)). (post-norm, or norms on the residual stream, "
        "give different outputs)")
    assert blk.aux_loss == 0.0, "a dense block's aux_loss is 0.0"
    cache = LayerKVCache()
    a = blk(x[:, :4], cache=cache)
    b = blk(x[:, 4:], cache=cache)
    assert torch.allclose(torch.cat([a, b], 1), blk(x), atol=1e-5), "cache is passed through to the attention"


def test_stage4_moe_block(nb):
    torch.manual_seed(1)
    blk = nb.LlamaBlock(16, 4, 4, d_ff=24, n_experts=3, top_k=1)
    assert isinstance(blk.mlp, nb.MoE) and len(blk.mlp.experts) == 3 and blk.mlp.top_k == 1, (
        "n_experts > 0 makes mlp an MoE with that many experts and top_k")
    x = torch.randn(2, 5, 16)
    y = blk(x)
    _, aux = blk.mlp(blk.norm2(x + blk.attn(blk.norm1(x))))
    assert torch.is_tensor(blk.aux_loss) and torch.allclose(blk.aux_loss, aux, atol=1e-6), (
        "after a forward, block.aux_loss holds the MoE's load-balancing loss (a tensor, so it can be backpropagated)")
    assert blk.aux_loss.requires_grad, "don't detach the aux loss: the model's loss backpropagates through it"
    assert y.shape == x.shape
    big = nb.LlamaBlock(16, 4, 2, max_seq_len=4096)
    cos = getattr(big.attn.rope, "cos_cached", None)
    if cos is not None:
        assert cos.shape[0] >= 4096, "max_seq_len is passed down to the attention's RotaryEmbedding"


# ---------------------------------------------------------------- stage 5: Llama


def _llama(nb, **kw):
    torch.manual_seed(0)
    cfg = dict(block_size=32, n_layers=2, n_heads=4, n_kv_heads=2, d_model=32)
    cfg.update(kw)
    return nb.Llama(17, **cfg)


def test_stage5_structure_and_params(nb):
    m = _llama(nb)
    assert isinstance(m.blocks, nn.ModuleList) and len(m.blocks) == 2 and type(m.norm).__name__ == "RMSNorm"
    assert m.lm_head.weight is m.tok_emb.weight, "tie_weights=True: lm_head.weight is tok_emb.weight"
    assert m.block_size == 32
    d, dff, V, L, kv = 32, 88, 17, 2, 2 * 8
    want = V * d + L * (d * d + 2 * d * kv + d * d + 3 * d * dff + 2 * d) + d
    got = sum(p.numel() for p in m.parameters())
    assert got == want, (
        f"parameter count {got}, want {want}: tied embedding V*d, per block q (d*d), k and v (d * n_kv_heads * "
        "head_dim each), o (d*d), SwiGLU 3*d*d_ff, two RMSNorm weights; final norm d. no biases, no position embedding")
    untied = _llama(nb, tie_weights=False)
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert sum(p.numel() for p in untied.parameters()) == want + V * d
    mha = _llama(nb, n_kv_heads=4)
    assert sum(p.numel() for p in mha.parameters()) == want + L * 2 * d * 16, "n_kv_heads is passed to every block"
    assert m.blocks[0].attn.n_kv_heads == 2 and mha.blocks[1].attn.n_kv_heads == 4
    std = m.tok_emb.weight.std().item()
    assert 0.015 < std < 0.025, f"embeddings and linear layers start at N(0, 0.02); tok_emb std {std:.4f}"
    o = m.blocks[0].attn.o_proj.weight.std().item()
    assert 0.007 < o < 0.013, f"residual output projections start at 0.02 / sqrt(2 n_layers) = 0.01; o_proj std {o:.4f}"
    wd = m.blocks[0].mlp.w_down.weight.std().item()
    assert 0.007 < wd < 0.013, f"w_down is a residual output too: std 0.01; got {wd:.4f}"
    moe = _llama(nb, n_experts=4)
    ew = torch.cat([e.w_down.weight.flatten() for e in moe.blocks[0].mlp.experts]).std().item()
    assert 0.007 < ew < 0.013, f"every expert's w_down writes to the residual stream too: std 0.01; got {ew:.4f}"


def test_stage5_forward_and_loss(nb):
    m = _llama(nb)
    idx = torch.randint(0, 17, (3, 10), generator=torch.Generator().manual_seed(0))
    y = torch.randint(0, 17, (3, 10), generator=torch.Generator().manual_seed(1))
    y[0, :4] = -100
    logits, loss = m(idx, y)
    assert logits.shape == (3, 10, 17)
    assert torch.allclose(loss, F.cross_entropy(logits.reshape(-1, 17), y.reshape(-1), ignore_index=-100), atol=1e-6), (
        "dense model: loss is plain cross-entropy over positions whose target isn't -100")
    assert m(idx)[1] is None
    hid = m.hidden_states(idx)
    assert torch.allclose(m.lm_head(hid), logits, atol=1e-5), "hidden_states is what lm_head reads"
    rms = hid.pow(2).mean(-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.02), (
        "hidden states pass through the final RMSNorm (weight 1 at init), so every position has mean square 1")
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))
    cache = KVCache(2)
    m(torch.zeros(1, 30, dtype=torch.long), cache=cache)
    with pytest.raises(ValueError):
        m(torch.zeros(1, 3, dtype=torch.long), cache=cache)  # 30 cached + 3 new > block_size 32
    moe = _llama(nb, n_experts=4, top_k=1, aux_loss_coef=0.5)
    assert all(isinstance(b.mlp, nb.MoE) and b.mlp.top_k == 1 and len(b.mlp.experts) == 4 for b in moe.blocks), (
        "n_experts and top_k reach every block")
    logits, loss = moe(idx, y)
    ce = F.cross_entropy(logits.reshape(-1, 17), y.reshape(-1), ignore_index=-100)
    aux = sum(b.aux_loss for b in moe.blocks)
    assert torch.allclose(loss, ce + 0.5 * aux, atol=1e-5), (
        f"MoE: loss = CE + aux_loss_coef * sum of the blocks' aux losses = {float(ce + 0.5 * aux):.4f}; got {float(loss):.4f}")
    loss.backward()
    assert moe.blocks[0].mlp.router.weight.grad is not None, "the aux loss trains the routers"


def test_stage5_generation(nb):
    m = _llama(nb).eval()
    prompt = torch.randint(0, 17, (2, 5), generator=torch.Generator().manual_seed(2))
    cached = generate_cached(m, prompt, 12, temperature=0.0)
    plain = m.generate(prompt, 12, temperature=0.0)
    assert torch.equal(cached, plain), (
        "your Llama must work with chapter 18's generate_cached: block i gets cache[i], positions start at "
        "cache.seq_len, and greedy decoding with the cache equals greedy decoding without it")
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    assert torch.equal(m.generate(prompt, 6, generator=g1), m.generate(prompt, 6, generator=g2))
    m.train()
    long = m.generate(torch.zeros(1, 30, dtype=torch.long), 10, temperature=0.0)
    assert long.shape == (1, 40), "generate crops the context to the last block_size tokens instead of failing"
    assert m.training, "generate restores train mode afterwards"
    hot = m.generate(prompt, 30, temperature=1.0, top_k=1)
    greedy = m.generate(prompt, 30, temperature=0.0)
    assert torch.equal(hot, greedy), "top_k=1 is passed to sample_next_token (= greedy)"
    g = torch.Generator().manual_seed(1)
    assert torch.equal(m.generate(prompt, 30, top_p=1e-6, generator=g), greedy), "top_p is passed (tiny top_p = greedy)"
    g = torch.Generator().manual_seed(1)
    assert torch.equal(m.generate(prompt, 30, temperature=1e-4, generator=g), greedy), (
        "temperature is passed (a tiny temperature is greedy)")


def test_stage5_learns(nb):
    torch.manual_seed(0)
    m = nb.Llama(8, block_size=16, n_layers=2, n_heads=2, n_kv_heads=1, d_model=32)
    seq = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7] * 3)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):
        starts = torch.randint(0, 8, (16,))
        x = torch.stack([seq[s:s + 16] for s in starts])
        y = torch.stack([seq[s + 1:s + 17] for s in starts])
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, (
        f"a cyclic sequence 0..7 is perfectly predictable; after 150 AdamW steps the loss should be < 0.1, got "
        f"{loss.item():.3f}. check the causal mask and that RoPE positions are right")


def test_stage2_save_weights_is_deterministic_in_eval(nb):
    torch.manual_seed(5)
    attn = nb.GroupedQueryAttention(16, 4, 2, dropout=0.5).eval()
    attn.save_weights = True
    x = torch.randn(2, 6, 16)
    a, b = attn(x), attn(x)
    assert torch.equal(a, b), "in eval mode the save_weights path must not apply dropout (two calls must agree)"
