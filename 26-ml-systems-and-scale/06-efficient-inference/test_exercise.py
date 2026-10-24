"""Checks for 26 · 06 efficient inference.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. quantization is checked
against hand-computed scales and levels (round half to even, clamping at +-127, zero channels), along
each axis, per tensor and per group; the int8 layer for its buffers, dtype handling and agreement with
the float layer; model conversion for which modules are replaced (and skipped) and for leaving the
original intact; pruning for exactly which weights are removed, globally versus per layer; the
distillation loss against its formula, its limits (alpha 0 and 1), the teacher being a constant, and
the T^2 factor; the decode model in both roofline regimes.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


# ---------------------------------------------------------------- stage 1: quantization


def test_stage1_hand_example(nb):
    w = torch.tensor([[0.5, -1.27, 0.0], [2.54, 1.0, -0.01]])
    q, s = nb.quantize_int8(w)
    assert q.dtype == torch.int8, f"q must be an int8 tensor; got {q.dtype}"
    assert s.shape == (2, 1) and torch.allclose(s, torch.tensor([[0.01], [0.02]])), (
        f"axis=0: one scale per row, max|w| / 127 = (0.01, 0.02), shaped (2, 1) to broadcast against w; got {s.flatten()}")
    assert q.tolist() == [[50, -127, 0], [127, 50, -0]], (
        f"q = round(w / scale): [[50, -127, 0], [127, 50, 0]]; got {q.tolist()}. -0.01 / 0.02 = -0.5 rounds to 0 "
        "(torch.round rounds halves to even)")
    assert torch.allclose(nb.dequantize_int8(q, s), torch.tensor([[0.5, -1.27, 0.0], [2.54, 1.0, 0.0]])), (
        "dequantize: q.float() * scale")


def test_stage1_axes_and_errors(nb):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(6, 9, generator=g)
    q1, s1 = nb.quantize_int8(w, axis=1)
    assert s1.shape == (1, 9) and torch.allclose(s1.flatten(), w.abs().amax(0) / 127), "axis=1: one scale per column"
    q0, s0 = nb.quantize_int8(w)
    err = (nb.dequantize_int8(q0, s0) - w).abs()
    assert (err <= s0 / 2 + 1e-7).all(), "rounding error is at most half a quantization step in every channel"
    assert q0.abs().max() == 127, "each channel's largest |w| maps to +-127"
    z = torch.zeros(3, 4)
    z[1] = torch.tensor([1.0, -2.0, 0.5, 0.0])
    qz, sz = nb.quantize_int8(z)
    assert torch.isfinite(sz).all() and (sz > 0).all() and (qz[0] == 0).all(), (
        "an all-zero channel must not produce a zero or nan scale: give it scale 1 (its q is all zeros)")
    w3 = torch.randn(2, 3, 4, generator=g)
    q3, s3 = nb.quantize_int8(w3, axis=1)
    assert s3.shape == (1, 3, 1) and torch.allclose(nb.dequantize_int8(q3, s3), w3, atol=float(s3.max())), (
        "for a 3-D tensor with axis=1 the scales have shape (1, 3, 1)")


def test_stage1_fake_quantize(nb):
    w = torch.tensor([[1.0, 0.3, -0.6, 0.15], [8.0, 0.5, 0.25, -4.0]])
    got = nb.fake_quantize(w, 3)  # 3 bits: levels -3..3, per row
    want = torch.tensor([[1.0, 1 / 3, -2 / 3, 0.0], [8.0, 0.0, 0.0, -16 / 3]])
    assert torch.allclose(got, want, atol=1e-6), (
        f"3 bits gives integer levels -3..3 (qmax = 2^(bits-1) - 1); row scales 1/3 and 8/3 (-4 / (8/3) = -1.5 rounds to even, -2). want {want.tolist()}, got "
        f"{got.tolist()}")
    cols = nb.fake_quantize(w, 3, axis=1)
    sc = w.abs().amax(0, keepdim=True) / 3
    assert torch.allclose(cols, torch.round(w / sc).clamp(-3, 3) * sc, atol=1e-6), (
        "axis=1: one scale per column (reduce over every other dimension), not per row")
    per_t = nb.fake_quantize(w, 3, axis=None)
    assert torch.allclose(per_t, torch.round(w / (8 / 3)).clamp(-3, 3) * (8 / 3)), (
        "axis=None: one scale for the whole tensor; row 0 then rounds to 0 entirely (the outlier 8.0 sets the scale)")
    grp = nb.fake_quantize(w, 3, group_size=2)
    want_g = torch.tensor([[1.0, 1 / 3, -0.6, 0.2], [8.0, 0.0, 0.0, -4.0]])
    assert torch.allclose(grp, want_g, atol=1e-6), (
        f"group_size=2: a scale for every 2 consecutive elements along the last dim; want {want_g.tolist()}, got "
        f"{grp.tolist()}")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(4, 64, generator=g)
    q, s = nb.quantize_int8(x)
    assert torch.allclose(nb.fake_quantize(x, 8), nb.dequantize_int8(q, s), atol=1e-6), "8-bit fake_quantize = int8 round trip"
    errs = [(nb.fake_quantize(x, b) - x).pow(2).mean().item() for b in (2, 4, 6, 8)]
    assert errs == sorted(errs, reverse=True), "error shrinks with more bits"


# ---------------------------------------------------------------- stage 2: int8 layers


def test_stage2_int8_linear(nb):
    torch.manual_seed(0)
    lin = nn.Linear(16, 8)
    ql = nb.Int8Linear.from_linear(lin)
    assert isinstance(ql, nn.Module)
    buffers = dict(ql.named_buffers())
    assert buffers["weight_q"].dtype == torch.int8 and tuple(buffers["weight_q"].shape) == (8, 16), (
        "Int8Linear stores weight_q as an int8 buffer of shape (out, in)")
    assert tuple(buffers["scale"].shape) == (8, 1), "one scale per output channel, shape (out, 1)"
    assert not list(ql.parameters()), "nothing to train: weights, scales and bias are buffers, not parameters"
    x = torch.randn(5, 16)
    want = F.linear(x, nb.dequantize_int8(*nb.quantize_int8(lin.weight)), lin.bias)
    assert torch.allclose(ql(x), want, atol=1e-6), "forward = F.linear(x, dequantized weight, bias)"
    assert (ql(x) - lin(x)).abs().max() < 0.05, "an int8 layer is close to the float one"
    assert ql(x.double()).dtype == torch.float64, "the dequantized weight is cast to x's dtype"
    nb_ = nb.Int8Linear.from_linear(nn.Linear(4, 3, bias=False))
    assert torch.allclose(nb_(torch.zeros(2, 4)), torch.zeros(2, 3)), "a layer without bias adds nothing"
    direct = nb.Int8Linear(torch.ones(2, 3, dtype=torch.int8), torch.tensor([[0.5], [2.0]]), torch.tensor([1.0, -1.0]))
    assert torch.allclose(direct(torch.ones(1, 3)), torch.tensor([[2.5, 5.0]])), "the constructor takes (weight_q, scale, bias)"


def test_stage2_quantize_model(nb):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Sequential(nn.Linear(20, 20), nn.ReLU()), nn.Linear(20, 4))
    x = torch.randn(7, 10)
    ref = model(x)
    qm = nb.quantize_linears(model)
    kinds = [type(m).__name__ for m in qm.modules()]
    assert kinds.count("Int8Linear") == 3 and "Linear" not in kinds, (
        f"every nn.Linear, including nested ones, is replaced by an Int8Linear; got {kinds}")
    assert all(isinstance(m, (nn.Linear, nn.ReLU, nn.Sequential)) for m in model.modules()), (
        "the original model must be left unchanged (work on a deep copy)")
    assert (qm(x) - ref).abs().max() < 0.05, "the quantized model's outputs stay close to the original's"
    partial = nb.quantize_linears(model, skip=("2.0",))
    assert isinstance(partial[2][0], nn.Linear) and isinstance(partial[0], nb.Int8Linear), (
        "skip takes qualified module names (as in named_modules()): '2.0' keeps the nested layer in float")
    fbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    qbytes = sum(b.numel() * b.element_size() for b in qm.buffers())
    assert qbytes < 0.4 * fbytes, f"int8 weights take about a quarter of float32's bytes; got {qbytes} vs {fbytes}"


# ---------------------------------------------------------------- stage 3: pruning


def _two_layers():
    m = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    with torch.no_grad():
        m[0].weight.copy_(torch.tensor([[0.1, -0.2], [0.3, -0.4]]))
        m[1].weight.copy_(torch.tensor([[5.0, -6.0], [7.0, -8.0]]))
    return m


def test_stage3_global_vs_layer(nb):
    m = _two_layers()
    masks = nb.magnitude_prune(m, 0.5)
    assert set(masks) == {"0", "1"}, f"one mask per nn.Linear, keyed by module name; got {set(masks)}"
    assert (m[0].weight == 0).all() and (m[1].weight != 0).all(), (
        "global pruning ranks all weights together: the 4 smallest |w| are all in layer 0, which is removed entirely")
    assert masks["0"].dtype == torch.bool and not masks["0"].any() and masks["1"].all()
    m = _two_layers()
    masks = nb.magnitude_prune(m, 0.5, scope="layer")
    assert torch.allclose(m[0].weight, torch.tensor([[0.0, 0.0], [0.3, -0.4]])) and torch.allclose(
            m[1].weight, torch.tensor([[0.0, 0.0], [7.0, -8.0]])), (
        "per-layer pruning removes the smallest half of each layer separately")
    with pytest.raises(ValueError):
        nb.magnitude_prune(_two_layers(), 0.5, scope="row")


def test_stage3_counts(nb):
    torch.manual_seed(2)
    m = nn.Sequential(nn.Linear(30, 40), nn.ReLU(), nn.Linear(40, 10))
    b0 = m[0].bias.detach().clone()
    weights = torch.cat([m[0].weight.detach().abs().flatten(), m[2].weight.detach().abs().flatten()])
    nb.magnitude_prune(m, 0.3)
    zeros = int((m[0].weight == 0).sum() + (m[2].weight == 0).sum())
    assert zeros == round(0.3 * 1600), f"30% of the 1,600 weights are zeroed: 480; got {zeros}"
    kept = torch.cat([m[0].weight.abs().flatten(), m[2].weight.abs().flatten()])
    assert kept[kept > 0].min() >= weights.sort().values[479], "the zeroed weights are exactly the smallest |w|"
    assert torch.equal(m[0].bias, b0), "biases aren't pruned"
    m2 = nn.Sequential(nn.Linear(4, 4))
    w = m2[0].weight.detach().clone()
    nb.magnitude_prune(m2, 0.0)
    assert torch.equal(m2[0].weight, w), "sparsity 0 changes nothing"


# ---------------------------------------------------------------- stage 4: distillation


def test_stage4_formula(nb):
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(6, 5, generator=g), torch.randn(6, 5, generator=g) * 3
    y = torch.randint(0, 5, (6,), generator=g)
    T = 3.0
    pt, ls = F.softmax(t / T, -1), F.log_softmax(s / T, -1)
    kl = (pt * (pt.log() - ls)).sum(-1).mean()
    ce = F.cross_entropy(s, y)
    got = nb.distillation_loss(s, t, y, temperature=T, alpha=0.3)
    want = 0.3 * T * T * kl + 0.7 * ce
    assert torch.allclose(got, want, atol=1e-5), (
        f"alpha T^2 KL(p_teacher || p_student) + (1 - alpha) CE, with the KL averaged over the batch (batchmean); want "
        f"{want.item():.5f}, got {got.item():.5f}. if you're off by a factor, check the T^2 and the KL direction")
    assert torch.allclose(nb.distillation_loss(s, t, y, alpha=0.0), ce), "alpha=0: pure cross-entropy"
    d = nb.distillation_loss(s, t, y)
    p2, l2 = F.softmax(t / 2, -1), F.log_softmax(s / 2, -1)
    assert torch.allclose(d, 0.5 * 4 * (p2 * (p2.log() - l2)).sum(-1).mean() + 0.5 * ce, atol=1e-5), (
        "defaults: temperature 2, alpha 0.5")


def test_stage4_gradients(nb):
    g = torch.Generator().manual_seed(1)
    s = torch.randn(4, 7, generator=g, requires_grad=True)
    t = torch.randn(4, 7, generator=g, requires_grad=True)
    y = torch.randint(0, 7, (4,), generator=g)
    nb.distillation_loss(s, t, y, alpha=1.0).backward()
    assert t.grad is None or torch.all(t.grad == 0), "the teacher's logits are a constant: detach them"
    same = nb.distillation_loss(s.detach(), s.detach(), y, alpha=1.0)
    assert abs(same.item()) < 1e-6, "a student that matches the teacher has zero distillation loss"
    grads = []
    for T in (1.0, 4.0, 16.0):
        s2 = s.detach().clone().requires_grad_(True)
        nb.distillation_loss(s2, t.detach() * 0.5, y, temperature=T, alpha=1.0).backward()
        grads.append(s2.grad.norm().item())
    assert max(grads) / min(grads) < 4, (
        f"thanks to the T^2 factor the soft-target gradients keep roughly the same size across temperatures; norms "
        f"{grads}. without it they'd shrink like 1 / T^2")


# ---------------------------------------------------------------- stage 5: batching


def test_stage5_decode_roofline(nb):
    N, bw, peak = 7e9, 1e11, 1e13
    t1 = nb.decode_tokens_per_second(N, 1, bandwidth=bw, peak_flops=peak)
    assert np.isclose(t1, 1 / (14e9 / 1e11)), (
        f"batch 1: each token reads all 14 GB of 16-bit weights at 100 GB/s: 0.14 s, about 7.1 tokens/s; got {t1:.2f}")
    t32 = nb.decode_tokens_per_second(N, 32, bandwidth=bw, peak_flops=peak)
    assert np.isclose(t32, 32 * t1), "batch 32 is still memory-bound: the same weight read serves 32 tokens"
    t_big = nb.decode_tokens_per_second(N, 1024, bandwidth=bw, peak_flops=peak)
    assert np.isclose(t_big, peak / (2 * N)), "large batches hit the compute roof: peak / (2 N) tokens/s"
    assert np.isclose(nb.decode_tokens_per_second(N, 1, bytes_per_param=1, bandwidth=bw, peak_flops=peak), 2 * t1), (
        "int8 weights (1 byte per parameter) double memory-bound decoding speed")
