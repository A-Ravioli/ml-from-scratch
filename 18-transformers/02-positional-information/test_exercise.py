"""Checks for 18 · 02 positional information.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. sinusoidal positions are
checked against the formula and the lesson's worked example; RoPE against an explicit 2x2-rotation
implementation, its relative-position property (q_m . k_n depends only on m - n), its offset (what a
KV cache relies on), positions beyond max_seq_len and gradients; ALiBi against Press et al.'s slopes
(including a head count that isn't a power of two); the positional LM for causality and for what each
scheme can and can't express.
"""
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


# ---------------------------------------------------------------- stage 1: sinusoidal


def test_stage1_formula_and_worked_example(nb):
    pe = nb.sinusoidal_positions(50, 16)
    assert isinstance(pe, torch.Tensor) and pe.dtype == torch.float32 and tuple(pe.shape) == (50, 16), (
        f"a float32 tensor (n_positions, d_model) = (50, 16); got {type(pe).__name__} {getattr(pe, 'dtype', '')} "
        f"{tuple(getattr(pe, 'shape', ()))}")
    p = torch.arange(50, dtype=torch.float64)[:, None]
    i = torch.arange(8, dtype=torch.float64)
    angle = p / 10000 ** (2 * i / 16)
    assert _close(pe[:, 0::2].double(), torch.sin(angle)) and _close(pe[:, 1::2].double(), torch.cos(angle)), (
        "PE[p, 2i] = sin(p / 10000^(2i/d)) on the even dims and PE[p, 2i+1] = cos of the SAME angle on the odd dims "
        "(not concatenated halves)")
    want = torch.tensor([math.sin(1), math.cos(1), math.sin(0.01), math.cos(0.01)])
    got = nb.sinusoidal_positions(3, 4)[1]
    assert _close(got.double(), want.double()), f"worked example: PE[1] for d = 4 is {want}; got {got}"
    assert tuple(nb.sinusoidal_positions(5, 7).shape) == (5, 7), "an odd d_model works too (the last sin has no cos)"


def test_stage1_relative_dot_products(nb):
    pe = nb.sinusoidal_positions(200, 32).double()
    dots = pe @ pe.T
    assert _close(dots[10, 15], dots[100, 105]) and _close(dots[3, 0], dots[120, 117]), (
        "PE[p] . PE[p + k] depends only on the offset k (sum of cos(k * w_i)): this is why sinusoidal positions let "
        "attention reason about distance")


# ---------------------------------------------------------------- stage 2: RoPE


def _rope_reference(x, positions, base):
    """explicit rotation of each channel pair (i, i + d/2) by angle position * base^(-2i/d)."""
    d = x.shape[-1]
    out = x.clone()
    for i in range(d // 2):
        theta = positions.double() * base ** (-2 * i / d)
        c, s = torch.cos(theta).to(x.dtype)[:, None], torch.sin(theta).to(x.dtype)[:, None]
        a, b = x[..., :, i:i + 1], x[..., :, i + d // 2:i + d // 2 + 1]
        out[..., :, i:i + 1] = a * c - b * s
        out[..., :, i + d // 2:i + d // 2 + 1] = a * s + b * c
    return out


def test_stage2_matches_explicit_rotation(nb):
    rope = nb.RotaryEmbedding(8, base=100.0, max_seq_len=64)
    x = torch.randn(2, 3, 10, 8, generator=torch.Generator().manual_seed(0))
    got = rope(x)
    assert tuple(got.shape) == x.shape and got.dtype == x.dtype, "RoPE keeps shape (B, H, T, head_dim) and dtype"
    want = _rope_reference(x, torch.arange(10), 100.0)
    assert _close(got, want), (
        "each channel pair (i, i + head_dim/2) must be rotated by the angle position * base^(-2i/head_dim) "
        "(the rotate-half convention: x * cos + rotate_half(x) * sin with rotate_half(x) = (-x2, x1)). if it is "
        "close but wrong, check you didn't pair neighbouring channels (2i, 2i+1), which is the other convention")
    ex = nb.RotaryEmbedding(4, base=100.0)(torch.tensor([[1.0, 2.0, 0.0, 0.0]]), offset=2)
    want_ex = torch.tensor([[math.cos(2), 2 * math.cos(0.2), math.sin(2), 2 * math.sin(0.2)]])
    assert _close(ex, want_ex), f"worked example: (1, 2, 0, 0) at position 2 with base 100 -> {want_ex}; got {ex}"
    with pytest.raises(ValueError):
        nb.RotaryEmbedding(7)


def test_stage2_relative_property_and_norms(nb):
    rope = nb.RotaryEmbedding(16)
    g = torch.Generator().manual_seed(1)
    q, k = torch.randn(16, dtype=torch.float64, generator=g), torch.randn(16, dtype=torch.float64, generator=g)
    Q = rope(q.expand(40, 16).clone())  # the same vector placed at positions 0..39
    K = rope(k.expand(40, 16).clone())
    S = Q @ K.T
    assert _close(S[5, 2], S[30, 27]) and _close(S[0, 9], S[20, 29]), (
        "after RoPE, q at position m dotted with k at position n must depend only on m - n")
    assert _close(Q.norm(dim=-1), q.norm().expand(40)), "a rotation preserves the norm of every vector"
    assert _close(Q[0], q), "position 0 is not rotated"


def test_stage2_offset_and_long_positions(nb):
    rope = nb.RotaryEmbedding(8, max_seq_len=16)
    x = torch.randn(1, 2, 24, 8, generator=torch.Generator().manual_seed(2))
    full = rope(x)
    assert tuple(full.shape) == (1, 2, 24, 8), "positions past max_seq_len must still work (extend your tables)"
    assert _close(rope(x[:, :, 10:13], offset=10), full[:, :, 10:13]), (
        "rope(x_chunk, offset=10) must rotate the chunk as positions 10, 11, 12: a KV cache feeds tokens with offset "
        "= the number already cached")
    assert _close(rope(x[:, :, 23:24], offset=23), full[:, :, 23:24]), "one token at offset 23 (past max_seq_len)"
    xd = torch.randn(1, 1, 5, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: rope(t, offset=3), (xd,)), "gradients must flow through RoPE"


def test_stage2_tables_survive_low_precision(nb):
    rope = nb.RotaryEmbedding(8, max_seq_len=4096)
    x = torch.randn(1, 1, 3, 8, generator=torch.Generator().manual_seed(4))
    want = rope(x, offset=3000)
    rope.to(torch.bfloat16)  # what a whole bf16 model does (chapter 27)
    got = rope(x, offset=3000)
    assert _close(got, want, atol=1e-4), (
        "after model.to(torch.bfloat16), RoPE must still rotate float32 inputs with float32 angles: at position 3000 a "
        "bfloat16 cos/sin table is off by whole radians. keep the tables in float32 (plain attributes rather than "
        "buffers, or recompute them) and cast at use")
    assert not any(k for k in rope.state_dict()), "the tables are derived from head_dim and base: keep them out of the state_dict"


# ---------------------------------------------------------------- stage 3: ALiBi


def test_stage3_slopes(nb):
    got8 = nb.alibi_slopes(8)
    assert _close(torch.as_tensor(got8, dtype=torch.float64), torch.tensor([2.0 ** -(h + 1) for h in range(8)]).double()), (
        f"8 heads: slopes 1/2, 1/4, ..., 1/256 (2^(-8h/n) for h = 1..n); got {got8}")
    want6 = torch.tensor([2 ** -2, 2 ** -4, 2 ** -6, 2 ** -8, 2 ** -1, 2 ** -3]).double()
    got6 = torch.as_tensor(nb.alibi_slopes(6), dtype=torch.float64)
    assert _close(got6, want6), (
        f"6 heads (not a power of 2): the 4-head slopes (1/4, 1/16, 1/64, 1/256), then every other 8-head slope "
        f"(1/2, 1/8, ...) until there are 6; expected {want6.tolist()}, got {got6.tolist()}")


def test_stage3_bias(nb):
    b = nb.alibi_bias(4, 5)
    assert tuple(b.shape) == (4, 5, 5) and b.dtype == torch.float32, f"(n_heads, T, T) float32; got {tuple(b.shape)} {b.dtype}"
    assert torch.isinf(b[:, 0, 1:]).all() and (b[:, 0, 1:] < 0).all() and torch.isinf(b[:, 2, 3:]).all(), (
        "above the diagonal (keys after the query) the bias is -inf: ALiBi includes the causal mask")
    assert (b.diagonal(dim1=1, dim2=2) == 0).all(), "a token attending to itself is distance 0: bias 0"
    assert _close(b[1, 4, 1], torch.tensor(-3 / 16)), (
        f"head 1 of 4 has slope 1/16; query 4, key 1 is distance 3: bias -3/16; got {b[1, 4, 1].item()}")
    assert _close(b[:, 3, 1], b[:, 4, 2]), "the bias depends only on the distance i - j"


# ---------------------------------------------------------------- stage 4: a positional LM


SCHEMES = ("none", "learned", "sinusoidal", "rope", "alibi")


def test_stage4_shapes_causality_and_lengths(nb):
    for pos in SCHEMES:
        torch.manual_seed(0)
        m = nb.PositionalLM(11, d_model=16, n_heads=2, n_layers=2, max_len=12, pos=pos).eval()
        idx = torch.randint(0, 11, (2, 10))
        out = m(idx)
        assert tuple(out.shape) == (2, 10, 11), f"pos={pos!r}: logits (B, T, V) = (2, 10, 11); got {tuple(out.shape)}"
        idx2 = idx.clone()
        idx2[:, 6:] = (idx2[:, 6:] + 1) % 11
        assert _close(m(idx2)[:, :6], out[:, :6]), (
            f"pos={pos!r}: changing tokens 6.. must not change the logits at positions 0..5. the model must be causal")
        if pos == "learned":
            with pytest.raises(ValueError):
                m(torch.randint(0, 11, (1, 13)))
        else:
            assert tuple(m(torch.randint(0, 11, (1, 30))).shape) == (1, 30, 11), (
                f"pos={pos!r} has no position table, so sequences longer than max_len must run")
    with pytest.raises(ValueError):
        nb.PositionalLM(11, pos="relative")
    n = {p: sum(t.numel() for t in nb.PositionalLM(11, d_model=16, n_heads=2, max_len=12, pos=p).parameters())
         for p in SCHEMES}
    assert n["learned"] - n["none"] == 12 * 16 and n["rope"] == n["alibi"] == n["sinusoidal"] == n["none"], (
        f"only 'learned' adds parameters (a (max_len, d_model) table); parameter counts {n}")


def test_stage4_what_each_scheme_can_express(nb):
    same = torch.full((1, 8), 3)
    for pos in ("none", "rope", "alibi"):
        torch.manual_seed(1)
        out = nb.PositionalLM(7, d_model=16, n_heads=2, pos=pos).eval()(same)
        assert _close(out, out[:, :1].expand_as(out), atol=1e-5), (
            f"pos={pos!r}: with the same token at every position, every value vector is the same, and any weighted "
            f"average of identical vectors is that vector, so the logits must be identical at every position. if "
            f"they differ, positional information is leaking in somewhere it shouldn't (e.g. RoPE applied to v, or "
            f"positions added to the input)")
    torch.manual_seed(1)
    out = nb.PositionalLM(7, d_model=16, n_heads=2, pos="learned").eval()(same)
    assert not _close(out, out[:, :1].expand_as(out)), "learned positions are added to the input: positions differ"


def test_stage4_learns_to_count(nb):
    """the same token everywhere; the target at position t is t. only absolute schemes can tell positions apart."""
    def train(pos):
        torch.manual_seed(0)
        m = nb.PositionalLM(12, d_model=32, n_heads=2, n_layers=1, max_len=12, pos=pos)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        x, y = torch.zeros(8, 12, dtype=torch.long), torch.arange(12).expand(8, 12)
        for _ in range(120):
            loss = nn.functional.cross_entropy(m(x).reshape(-1, 12), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return (m(x).argmax(-1) == y).float().mean().item()

    for pos in ("learned", "sinusoidal"):
        acc = bounded(train, pos, budget=50_000_000, what=f"training PositionalLM(pos={pos!r})")
        assert acc > 0.95, f"pos={pos!r} should learn to output its position on a constant input; accuracy {acc:.2f}"
    acc = bounded(train, "rope", budget=50_000_000, what="training PositionalLM(pos='rope')")
    assert acc < 0.2, f"pos='rope' cannot tell positions apart on a constant input; accuracy {acc:.2f}"
