"""Checks for 18 · 03 the transformer block.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the MLP and the block are
checked against PyTorch's nn.TransformerEncoderLayer / nn.TransformerDecoderLayer with copied weights
(oracles, pre-LN and post-LN), against the residual-stream equations, and with a stand-in KV cache;
the encoder-decoder for causality, padding and learning a small task; the gradient monitor against a
direct computation.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------- stage 1: the MLP


def test_stage1_feedforward(nb):
    torch.manual_seed(0)
    m = nb.FeedForward(8).eval()
    assert isinstance(m.fc1, nn.Linear) and tuple(m.fc1.weight.shape) == (32, 8), "fc1 is Linear(d, 4d) by default"
    assert isinstance(m.fc2, nn.Linear) and tuple(m.fc2.weight.shape) == (8, 32), "fc2 is Linear(4d, d)"
    assert tuple(nb.FeedForward(8, 20).fc1.weight.shape) == (20, 8), "d_ff sets the hidden width"
    x = torch.randn(2, 5, 8)
    assert _close(m(x), m.fc2(F.gelu(m.fc1(x)))), "fc1 -> GELU (the exact erf form, nn.GELU()) -> fc2"
    r = nb.FeedForward(8, activation="relu").eval()
    assert _close(r(x), r.fc2(F.relu(r.fc1(x)))), "activation='relu' uses ReLU"
    with pytest.raises(ValueError):
        nb.FeedForward(8, activation="swish")
    perm = torch.tensor([4, 2, 0, 1, 3])
    assert _close(m(x)[:, perm], m(x[:, perm])), "the MLP acts on every position separately"
    d = nb.FeedForward(8, dropout=0.5).train()
    assert not _close(d(x), d(x)), "dropout on the output in training mode"


# ---------------------------------------------------------------- stage 2: the block


def _randomize_norms(blk):
    """random LayerNorm gains and biases, so that a missing, extra or misplaced norm changes the output."""
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for m in blk.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.copy_(1 + 0.5 * torch.randn(m.weight.shape, generator=g))
                m.bias.copy_(0.5 * torch.randn(m.bias.shape, generator=g))
    return blk


def _encoder_oracle(blk, d, H, norm_first):
    ref = nn.TransformerEncoderLayer(d, H, dim_feedforward=4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                     norm_first=norm_first)
    with torch.no_grad():
        a = blk.attn
        ref.self_attn.in_proj_weight.copy_(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]))
        ref.self_attn.in_proj_bias.copy_(torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]))
        ref.self_attn.out_proj.load_state_dict(a.out_proj.state_dict())
        ref.linear1.load_state_dict(blk.mlp.fc1.state_dict())
        ref.linear2.load_state_dict(blk.mlp.fc2.state_dict())
        ref.norm1.load_state_dict(blk.ln1.state_dict())
        ref.norm2.load_state_dict(blk.ln2.state_dict())
    return ref.eval()


def test_stage2_structure_and_residual_stream(nb):
    torch.manual_seed(0)
    blk = nb.TransformerBlock(16, 4).eval()
    for name in ("ln1", "attn", "ln2", "mlp"):
        assert hasattr(blk, name), f"the block needs a submodule named {name!r}"
    assert isinstance(blk.ln1, nn.LayerNorm) and isinstance(blk.ln2, nn.LayerNorm), "ln1 and ln2 are nn.LayerNorm"
    assert not hasattr(blk, "cross_attn"), "no cross_attn unless cross_attention=True"
    dec = nb.TransformerBlock(16, 4, cross_attention=True)
    assert isinstance(dec.ln_cross, nn.LayerNorm) and hasattr(dec, "cross_attn"), (
        "cross_attention=True adds ln_cross and cross_attn")
    n = sum(p.numel() for p in nb.TransformerBlock(512, 8).parameters())
    n_cross = sum(p.numel() for p in nb.TransformerBlock(512, 8, cross_attention=True).parameters())
    assert n == 3_152_384 and n_cross == 4_204_032, (
        f"worked example: a block with d = 512 has 3,152,384 parameters (4,204,032 with cross-attention); got {n:,} "
        f"and {n_cross:,}. attention 4(d^2 + d), MLP 8d^2 + 5d, two LayerNorms 4d")
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        h = x + blk.attn(blk.ln1(x))
        want = h + blk.mlp(blk.ln2(h))
    assert _close(blk(x), want), "pre-LN (norm_first=True): x = x + attn(ln1(x)); x = x + mlp(ln2(x))"
    with torch.no_grad():
        blk.attn.out_proj.weight.zero_()
        blk.attn.out_proj.bias.zero_()
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()
    assert _close(blk(x), x), (
        "with both branches' output layers zeroed, a pre-LN block must be the identity: the residual stream passes "
        "through untouched, only ADDED to")


def test_stage2_matches_pytorch_encoder_layer(nb):
    for norm_first in (True, False):
        torch.manual_seed(1)
        blk = _randomize_norms(nb.TransformerBlock(16, 4, norm_first=norm_first).eval())
        ref = _encoder_oracle(blk, 16, 4, norm_first)
        x = torch.randn(3, 6, 16)
        kind = "pre-LN" if norm_first else "post-LN"
        assert _close(blk(x), ref(x)), (
            f"{kind} block differs from nn.TransformerEncoderLayer(norm_first={norm_first}) with the same weights. "
            + ("pre-LN: x + attn(ln1(x)), then x + mlp(ln2(x))" if norm_first else
               "post-LN: ln1(x + attn(x)), then ln2(x + mlp(x)): the norm comes AFTER the residual add"))
        causal = torch.ones(6, 6, dtype=torch.bool).triu(1)
        assert _close(blk(x, is_causal=True), ref(x, src_mask=causal)), f"{kind} with is_causal=True"
        pad = torch.ones(3, 1, 1, 6, dtype=torch.bool)
        pad[0, ..., 4:] = False
        assert _close(blk(x, mask=pad), ref(x, src_key_padding_mask=~pad[:, 0, 0])), f"{kind}: mask goes to attention"


def test_stage2_cross_attention_matches_decoder_layer(nb):
    for norm_first in (True, False):
        torch.manual_seed(2)
        blk = _randomize_norms(nb.TransformerBlock(16, 2, cross_attention=True, norm_first=norm_first).eval())
        ref = nn.TransformerDecoderLayer(16, 2, dim_feedforward=64, dropout=0.0, activation="gelu", batch_first=True,
                                         norm_first=norm_first)
        with torch.no_grad():
            for mine, theirs in ((blk.attn, ref.self_attn), (blk.cross_attn, ref.multihead_attn)):
                theirs.in_proj_weight.copy_(torch.cat([mine.q_proj.weight, mine.k_proj.weight, mine.v_proj.weight]))
                theirs.in_proj_bias.copy_(torch.cat([mine.q_proj.bias, mine.k_proj.bias, mine.v_proj.bias]))
                theirs.out_proj.load_state_dict(mine.out_proj.state_dict())
            ref.linear1.load_state_dict(blk.mlp.fc1.state_dict())
            ref.linear2.load_state_dict(blk.mlp.fc2.state_dict())
            ref.norm1.load_state_dict(blk.ln1.state_dict())
            ref.norm2.load_state_dict(blk.ln_cross.state_dict())
            ref.norm3.load_state_dict(blk.ln2.state_dict())
        ref.eval()
        x, mem = torch.randn(2, 5, 16), torch.randn(2, 7, 16)
        causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
        assert _close(blk(x, mem, is_causal=True), ref(x, mem, tgt_mask=causal)), (
            f"decoder block (norm_first={norm_first}) differs from nn.TransformerDecoderLayer: causal self-attention, "
            f"then cross-attention with queries from x and keys/values from the context, then the MLP, each a "
            f"residual branch with its own norm (ln1, ln_cross, ln2)")
        pad = torch.ones(2, 7, dtype=torch.bool)
        pad[0, 4:] = False
        got = blk(x, mem, is_causal=True, context_mask=pad[:, None, None, :])
        want = ref(x, mem, tgt_mask=causal, memory_key_padding_mask=~pad)
        assert _close(got, want), (
            f"context_mask (B, 1, 1, S), True = attend, must mask the cross-attention keys (norm_first={norm_first}): "
            f"pass it to cross_attn as its mask")


class _AppendCache:
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], -2)
        self.v = v if self.v is None else torch.cat([self.v, v], -2)
        return self.k, self.v


def test_stage2_context_errors_and_branch_dropout(nb):
    torch.manual_seed(5)
    enc = nb.TransformerBlock(8, 2)
    x, mem = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    with pytest.raises(ValueError):
        enc(x, mem)
    with pytest.raises(ValueError):
        enc(x, context_mask=torch.ones(1, 1, 1, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        nb.TransformerBlock(8, 2, cross_attention=True)(x)
    blk = nb.TransformerBlock(8, 2, dropout=0.5, cross_attention=True)
    with torch.no_grad():
        blk.mlp.fc2.weight.zero_()
        blk.mlp.fc2.bias.zero_()
        blk.cross_attn.out_proj.weight.zero_()
        blk.cross_attn.out_proj.bias.zero_()
    blk.attn.dropout = blk.cross_attn.dropout = 0.0  # attention-weight dropout off: only branch dropout is left
    blk.train()
    assert not _close(blk(x, mem), blk(x, mem)), (
        "dropout is applied to the self-attention branch's output before it's added to the stream")
    blk2 = nb.TransformerBlock(8, 2, dropout=0.5, cross_attention=True)
    with torch.no_grad():
        for lin in (blk2.mlp.fc2, blk2.attn.out_proj):
            lin.weight.zero_()
            lin.bias.zero_()
    blk2.attn.dropout = blk2.cross_attn.dropout = 0.0
    blk2.train()
    assert not _close(blk2(x, mem), blk2(x, mem)), "and to the cross-attention branch's output"


def test_stage2_cache_and_dropout(nb):
    torch.manual_seed(3)
    blk = nb.TransformerBlock(12, 3).eval()
    x = torch.randn(1, 5, 12)
    full = blk(x, is_causal=True)
    cache = _AppendCache()
    parts = [blk(x[:, :2], is_causal=True, cache=cache)] + [blk(x[:, t:t + 1], is_causal=True, cache=cache)
                                                            for t in range(2, 5)]
    assert _close(torch.cat(parts, 1), full), "the block must pass `cache` to its self-attention"
    d = nb.TransformerBlock(12, 3, dropout=0.5)
    d.train()
    assert not _close(d(x), d(x)), "dropout is active in training mode"
    d.eval()
    assert _close(d(x), d(x)), "and off in eval mode"


# ---------------------------------------------------------------- stage 3: encoder-decoder


def test_stage3_shapes_causality_padding(nb):
    torch.manual_seed(0)
    m = nb.EncoderDecoder(9, 11, d_model=16, n_heads=2, n_layers=2).eval()
    assert len(m.encoder) == 2 and len(m.decoder) == 2, "encoder and decoder are ModuleLists of n_layers blocks"
    assert all(hasattr(b, "cross_attn") for b in m.decoder) and not any(hasattr(b, "cross_attn") for b in m.encoder), (
        "decoder blocks have cross-attention, encoder blocks don't")
    src, tgt = torch.randint(0, 9, (2, 7)), torch.randint(0, 11, (2, 5))
    out = m(src, tgt)
    assert tuple(out.shape) == (2, 5, 11), f"logits (B, T, tgt_vocab) = (2, 5, 11); got {tuple(out.shape)}"
    tgt2 = tgt.clone()
    tgt2[:, 3:] = (tgt2[:, 3:] + 1) % 11
    assert _close(m(src, tgt2)[:, :3], out[:, :3]), "the decoder is causal: later target tokens can't change earlier logits"
    src2 = src.clone()
    src2[:, 6] = (src2[:, 6] + 1) % 9
    assert not _close(m(src2, tgt), out), "every target position may look at the whole source"
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, 5:] = False
    src3 = src.clone()
    src3[:, 5:] = (src3[:, 5:] + 3) % 9
    mem, mem3 = m.encode(src, mask), m.encode(src3, mask)
    assert _close(mem[:, :5], mem3[:, :5]), (
        "with src_mask (B, S) True = real token, the encoder must not let real tokens attend to padding: changing "
        "padded tokens must not change the memory at real positions")


def test_stage3_memory_padding_and_final_norms(nb):
    torch.manual_seed(6)
    m = nb.EncoderDecoder(9, 11, d_model=16, n_heads=2, n_layers=2).eval()
    src, tgt = torch.randint(0, 9, (2, 7)), torch.randint(0, 11, (2, 5))
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, 4:] = False
    src2 = src.clone()
    src2[:, 4:] = (src2[:, 4:] + 3) % 9
    a, b = m(src, tgt, src_mask=mask), m(src2, tgt, src_mask=mask)
    assert _close(a, b), (
        "with src_mask, the decoder's cross-attention must not attend to padded source positions either: forward "
        "passes src_mask to decode as memory_mask, and decode gives it to every decoder block as context_mask")
    mem = m.encode(src)
    assert isinstance(m.enc_norm, nn.LayerNorm) and isinstance(m.dec_norm, nn.LayerNorm), "pre-LN stacks end with a LayerNorm"
    assert _close(mem.mean(-1), torch.zeros(2, 7), atol=1e-4) and _close(mem.std(-1, unbiased=False), torch.ones(2, 7), atol=1e-2), (
        "encode returns the memory after enc_norm (at initialization: mean 0, std 1 per token)")
    with torch.no_grad():
        m.dec_norm.weight.zero_()
        m.dec_norm.bias.zero_()
    assert _close(m.decode(tgt, mem), m.head.bias.expand(2, 5, 11)), "decode applies dec_norm before the head"


def test_stage3_learns_to_reverse(nb):
    def run(seed):
        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        m = nb.EncoderDecoder(10, 10, d_model=32, n_heads=2, n_layers=1)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(150):
            src = torch.randint(2, 10, (32, 6), generator=g)
            tgt = torch.cat([torch.ones(32, 1, dtype=torch.long), src.flip(1)], 1)  # 1 = start symbol
            loss = F.cross_entropy(m(src, tgt[:, :-1]).reshape(-1, 10), tgt[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        src = torch.randint(2, 10, (200, 6), generator=g)
        tgt = torch.cat([torch.ones(200, 1, dtype=torch.long), src.flip(1)], 1)
        with torch.no_grad():
            return (m(src, tgt[:, :-1]).argmax(-1) == tgt[:, 1:]).float().mean().item()

    accs = [bounded(run, s, budget=80_000_000, what="training EncoderDecoder") for s in range(3)]
    assert sorted(accs)[1] > 0.9, (
        f"a 1-layer encoder-decoder should learn to reverse 6 tokens in 150 steps (teacher-forced accuracy > 0.9, "
        f"median of 3 seeds); got {[round(a, 2) for a in accs]}. check the cross-attention and the positions")


# ---------------------------------------------------------------- stage 4: gradient norms per block


def test_stage4_block_grad_norms(nb):
    torch.manual_seed(4)
    blocks = nn.ModuleList(nb.TransformerBlock(8, 2) for _ in range(3))
    x = torch.randn(2, 4, 8)
    h = x
    for b in blocks:
        h = b(h)
    h.pow(2).sum().backward()
    got = nb.block_grad_norms(blocks)
    want = [math.sqrt(sum(float(p.grad.pow(2).sum()) for p in b.parameters())) for b in blocks]
    assert isinstance(got, list) and len(got) == 3 and all(isinstance(v, float) for v in got), (
        f"a list of 3 Python floats, one per block; got {got!r}")
    assert all(math.isclose(a, b, rel_tol=1e-4) for a, b in zip(got, want)), (
        f"each entry is the 2-norm of the concatenated gradients of that block's parameters (sqrt of the sum of "
        f"squares over all of them, not the sum of per-parameter norms); expected {want}, got {got}")
    fresh = nn.ModuleList([nb.TransformerBlock(8, 2)])
    assert nb.block_grad_norms(fresh) == [0.0], "a block whose parameters have no gradient yet gives 0.0"
