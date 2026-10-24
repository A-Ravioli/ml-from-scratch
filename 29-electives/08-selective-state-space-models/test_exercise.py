"""Checks for exercise.py (selective state-space models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The parallel scan
is checked against the sequential one (values and gradients), for associativity of its combine
step, and for depth: on a long sequence it must finish within a budget of executed Python
lines that a loop over positions can't meet. The selective scan is checked against a NumPy
loop, for its two selection behaviours (a zero step ignores a token, a huge step forgets the
past), and for reducing to a time-invariant convolution when nothing depends on the input.
The layer and the block are checked against their own recurrent mode, for causality, and
against a manual composition of their parts.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)


def _np_selective(u, delta, A, B, C, D):
    batch, L, Dm = u.shape
    N = A.shape[1]
    y = np.zeros((batch, L, Dm))
    for b in range(batch):
        h = np.zeros((Dm, N))
        for t in range(L):
            h = np.exp(delta[b, t][:, None] * A) * h + (delta[b, t] * u[b, t])[:, None] * B[b, t][None, :]
            y[b, t] = h @ C[b, t] + (0 if D is None else D * u[b, t])
    return y


def _inputs(batch, L, Dm, N, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(batch, L, Dm, generator=g, dtype=torch.float64)
    delta = torch.rand(batch, L, Dm, generator=g, dtype=torch.float64) * 0.5 + 0.01
    A = -torch.rand(Dm, N, generator=g, dtype=torch.float64) * 2 - 0.1
    B = torch.randn(batch, L, N, generator=g, dtype=torch.float64)
    C = torch.randn(batch, L, N, generator=g, dtype=torch.float64)
    D = torch.randn(Dm, generator=g, dtype=torch.float64)
    return u, delta, A, B, C, D


# ---------------------------------------------------------------- stage 1: the sequential scan


def test_stage1_hand_example(nb):
    a = torch.tensor([[0.5, 0.5, 2.0]])
    b = torch.tensor([[1.0, 1.0, 1.0]])
    h = bounded(nb.sequential_scan, a, b, what="sequential_scan")
    assert h.shape == (1, 3) and torch.allclose(h, torch.tensor([[1.0, 1.5, 4.0]])), (
        f"h_t = a_t h_(t-1) + b_t from h_(-1) = 0: (1, 0.5 * 1 + 1, 2 * 1.5 + 1) = (1, 1.5, 4); got {h.tolist()}")
    h0 = nb.sequential_scan(a, b, torch.tensor([2.0]))
    assert torch.allclose(h0, torch.tensor([[2.0, 2.0, 5.0]])), (
        f"with h0 = 2: (0.5 * 2 + 1, 0.5 * 2 + 1, 2 * 2 + 1) = (2, 2, 5); got {h0.tolist()}")


def test_stage1_matches_numpy_on_extra_dims(nb):
    g = np.random.default_rng(0)
    a, b = g.uniform(-1, 1, (2, 9, 3, 4)), g.normal(size=(2, 9, 3, 4))
    h0 = g.normal(size=(2, 3, 4))
    want, h = np.zeros_like(b), h0.copy()
    for t in range(9):
        h = a[:, t] * h + b[:, t]
        want[:, t] = h
    got = nb.sequential_scan(torch.tensor(a), torch.tensor(b), torch.tensor(h0))
    assert got.shape == b.shape and np.allclose(got.numpy(), want), (
        "the scan runs along dim 1 and is elementwise over every other dim (batch first, then any trailing dims)")


# ---------------------------------------------------------------- stage 2: the parallel scan


def test_stage2_combine(nb):
    a, b = nb.combine((torch.tensor(2.0), torch.tensor(3.0)), (torch.tensor(5.0), torch.tensor(7.0)))
    assert float(a) == 10.0 and float(b) == 22.0, (
        f"applying h -> 2h + 3 and then h -> 5h + 7 gives h -> 10h + (5 * 3 + 7) = 10h + 22; got ({float(a)}, {float(b)})")
    g = torch.Generator().manual_seed(1)
    x, y, z = [(torch.randn(5, generator=g, dtype=torch.float64), torch.randn(5, generator=g, dtype=torch.float64))
               for _ in range(3)]
    left = nb.combine(nb.combine(x, y), z)
    right = nb.combine(x, nb.combine(y, z))
    assert all(torch.allclose(p, q) for p, q in zip(left, right)), "combine must be associative: that's what makes the scan parallel"
    swapped = nb.combine(y, x)
    assert not torch.allclose(nb.combine(x, y)[1], swapped[1]), "combine is not commutative: the order of the maps matters"


def test_stage2_parallel_equals_sequential(nb):
    g = torch.Generator().manual_seed(2)
    for L in (1, 2, 3, 7, 64, 100):
        a = torch.rand(3, L, 4, generator=g, dtype=torch.float64) * 1.8 - 0.9
        b = torch.randn(3, L, 4, generator=g, dtype=torch.float64)
        h0 = torch.randn(3, 4, generator=g, dtype=torch.float64)
        for init in (None, h0):
            want = nb.sequential_scan(a, b, init)
            a_before, b_before = a.clone(), b.clone()
            h0_before = None if init is None else init.clone()
            got = bounded(nb.parallel_scan, a, b, init, what="parallel_scan")
            assert torch.equal(a, a_before) and torch.equal(b, b_before) and (init is None or torch.equal(init, h0_before)), (
                "parallel_scan must not modify its inputs in place (fold h0 into a NEW tensor, e.g. with torch.cat)")
            assert got.shape == want.shape and torch.allclose(got, want), (
                f"L = {L}, h0 {'given' if init is not None else 'None'}: the parallel scan must give exactly the sequential "
                f"result; max difference {(got - want).abs().max().item():.2e}. non-power-of-two lengths need care")


def test_stage2_parallel_is_log_depth(nb):
    a = torch.rand(1, 4096, 2) * 0.5
    b = torch.randn(1, 4096, 2)
    out = bounded(nb.parallel_scan, a, b, budget=3000, what="parallel_scan",
                  hint="the parallel scan should loop over O(log L) levels with whole-tensor operations, not over "
                       "the L positions (a loop over 4,096 positions can't finish in 3,000 Python lines)")
    assert torch.allclose(out, nb.sequential_scan(a, b), atol=1e-5), "and it must still be right on a long sequence"


def test_stage2_gradients_match(nb):
    g = torch.Generator().manual_seed(3)
    a0 = torch.rand(2, 13, 3, generator=g, dtype=torch.float64)
    b0 = torch.randn(2, 13, 3, generator=g, dtype=torch.float64)
    w = torch.randn(2, 13, 3, generator=g, dtype=torch.float64)
    grads = []
    for scan in (nb.sequential_scan, nb.parallel_scan):
        a, b = a0.clone().requires_grad_(), b0.clone().requires_grad_()
        (scan(a, b) * w).sum().backward()
        grads.append((a.grad, b.grad))
    assert grads[1][0] is not None and torch.allclose(grads[0][0], grads[1][0]) and torch.allclose(grads[0][1], grads[1][1]), (
        "gradients through the parallel scan must equal those through the sequential scan (use differentiable torch ops)")


# ---------------------------------------------------------------- stage 3: the selective scan


def test_stage3_matches_numpy_loop(nb):
    u, delta, A, B, C, D = _inputs(2, 11, 3, 4, seed=4)
    want = _np_selective(*(t.numpy() for t in (u, delta, A, B, C, D)))
    for mode in ("parallel", "sequential"):
        y = nb.selective_scan(u, delta, A, B, C, D, mode=mode)
        assert y.shape == (2, 11, 3) and np.allclose(y.detach().numpy(), want), (
            f"mode={mode}: h_t = exp(delta_t A) * h_(t-1) + (delta_t u_t) B_t (Euler for B, as in Mamba), "
            "y_t = h_t C_t + D u_t, with delta and u per channel and B, C shared by the channels")
    y0 = nb.selective_scan(u, delta, A, B, C)
    assert np.allclose(y0.detach().numpy(), want - D.numpy() * u.numpy()), "D=None means no skip term"
    try:
        nb.selective_scan(u, delta, A, B, C, mode="fft")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown mode should raise ValueError")


def test_stage3_parallel_mode_uses_the_parallel_scan(nb):
    u, delta, A, B, C, D = _inputs(1, 2048, 2, 2, seed=8)
    y = bounded(nb.selective_scan, u, delta, A, B, C, D, budget=3000, what="selective_scan(mode='parallel')",
                hint="mode='parallel' (the default) should run your parallel_scan, not a loop over positions")
    assert torch.allclose(y, nb.selective_scan(u, delta, A, B, C, D, mode="sequential"), atol=1e-8), (
        "the two modes must agree on a long sequence")


def test_stage3_zero_step_ignores_a_token(nb):
    u, delta, A, B, C, D = _inputs(1, 12, 3, 4, seed=5)
    delta = delta.clone()
    delta[:, 5] = 0.0
    y1 = nb.selective_scan(u, delta, A, B, C)
    u2 = u.clone()
    u2[:, 5] += 10.0
    y2 = nb.selective_scan(u2, delta, A, B, C)
    assert torch.allclose(y1, y2), (
        "selection: with delta_t = 0, exp(0 * A) = 1 and delta_t B_t u_t = 0, so token t is skipped entirely and the state "
        "passes through unchanged; changing u_t must not change any output (D=None here)")


def test_stage3_huge_step_forgets_the_past(nb):
    u, delta, A, B, C, D = _inputs(1, 12, 3, 4, seed=6)
    A = A - 0.5  # every |A| >= 0.6, so exp(200 A) < 1e-50
    delta = delta.clone()
    delta[:, 6] = 200.0
    y1 = nb.selective_scan(u, delta, A, B, C, D)
    u2 = u.clone()
    u2[:, :6] = torch.randn(1, 6, 3, dtype=torch.float64) * 5
    y2 = nb.selective_scan(u2, delta, A, B, C, D)
    assert torch.allclose(y1[:, 6:], y2[:, 6:], atol=1e-8) and not torch.allclose(y1[:, :6], y2[:, :6]), (
        "selection: a huge delta_t makes exp(delta_t A) ~ 0, which resets the state at t; outputs from t on must not "
        "depend on anything before t")


def test_stage3_time_invariant_case_is_a_convolution(nb):
    u, delta, A, B, C, D = _inputs(2, 20, 3, 4, seed=7)
    d = torch.tensor([0.1, 0.3, 0.05], dtype=torch.float64)
    Bc = torch.randn(4, dtype=torch.float64)
    Cc = torch.randn(4, dtype=torch.float64)
    y = nb.selective_scan(u, d.expand(2, 20, 3), A, Bc.expand(2, 20, 4), Cc.expand(2, 20, 4))
    l = torch.arange(20, dtype=torch.float64)
    K = torch.einsum("n,hn,hnl->hl", Cc * Bc, d[:, None].expand(3, 4), torch.exp(d[:, None, None] * A[..., None] * l))
    want = torch.zeros_like(u)
    for k in range(20):
        want[:, k] = torch.einsum("bjh,hj->bh", u[:, :k + 1].flip(1), K[:, :k + 1])
    assert torch.allclose(y, want), (
        "with delta, B and C constant over time the selective scan is the time-invariant SSM of topic 07: a causal "
        "convolution with K_l = sum_n C_n delta B_n exp(delta A_n)^l")


# ---------------------------------------------------------------- stage 4: the selective layer


def test_stage4_parameters(nb):
    torch.manual_seed(0)
    layer = nb.SelectiveSSM(40, 6, dt_min=0.01, dt_max=0.2)
    assert layer.A_log.shape == (40, 6) and layer.D.shape == (40,), "A_log (Dm, N) and D (Dm,)"
    assert torch.allclose(-layer.A_log.exp(), -torch.arange(1.0, 7.0).repeat(40, 1)), (
        "A = -exp(A_log) starts at -(1, 2, ..., N) in every channel (S4D-Real)")
    assert torch.allclose(layer.D, torch.ones(40)), "D starts at 1"
    R = math.ceil(40 / 16)
    assert isinstance(layer.x_proj, nn.Linear) and layer.x_proj.weight.shape == (R + 12, 40) and layer.x_proj.bias is None, (
        f"x_proj: Linear(Dm, dt_rank + 2N) without bias, dt_rank = ceil(Dm / 16) = {R}")
    assert isinstance(layer.dt_proj, nn.Linear) and layer.dt_proj.weight.shape == (40, R) and layer.dt_proj.bias is not None, (
        "dt_proj: Linear(dt_rank, Dm) with bias")
    dt = F.softplus(layer.dt_proj.bias.detach())
    assert dt.min() >= 0.01 - 1e-5 and dt.max() <= 0.2 + 1e-5, (
        "softplus(dt_proj.bias) starts inside [dt_min, dt_max]: set the bias to the inverse softplus of the sampled dt")
    wide = nb.SelectiveSSM(3000, 1, dt_rank=1)
    z = (torch.log(F.softplus(wide.dt_proj.bias.detach())).numpy() - math.log(1e-3)) / (math.log(1e-1) - math.log(1e-3))
    ks = stats.kstest(z, "uniform").statistic
    assert ks < 0.045, f"the initial dt is log-uniform in [dt_min, dt_max] across channels (KS distance {ks:.3f})"
    assert nb.SelectiveSSM(8, 4, dt_rank=3).x_proj.weight.shape == (11, 8), "an explicit dt_rank overrides the default"
    w = nb.SelectiveSSM(2000, 1, dt_rank=4).dt_proj.weight.detach()
    bound = 4 ** -0.5
    assert w.abs().max() <= bound + 1e-6 and abs(w.std().item() - bound / math.sqrt(3)) < 0.01, (
        f"dt_proj.weight starts uniform in [-dt_rank^-1/2, dt_rank^-1/2] (here +-{bound}); got max |w| "
        f"{w.abs().max().item():.3f}, std {w.std().item():.3f} (uniform would give {bound / math.sqrt(3):.3f})")


def test_stage4_forward_is_the_selective_scan_of_its_projections(nb):
    torch.manual_seed(1)
    layer = nb.SelectiveSSM(8, 4)
    x = torch.randn(2, 10, 8)
    delta, B, C = layer.params(x)
    R = layer.dt_proj.in_features
    raw, B2, C2 = layer.x_proj(x).split([R, 4, 4], dim=-1)
    assert torch.allclose(delta, F.softplus(layer.dt_proj(raw))) and torch.allclose(B, B2) and torch.allclose(C, C2), (
        "params(x): x_proj(x) splits into (dt_raw: dt_rank, B: N, C: N) in that order; delta = softplus(dt_proj(dt_raw))")
    want = nb.selective_scan(x, delta, -torch.exp(layer.A_log), B, C, layer.D)
    assert torch.allclose(layer(x), want, atol=1e-6), "forward = selective_scan(x, delta, -exp(A_log), B, C, D)"


def test_stage4_recurrent_mode_and_causality(nb):
    torch.manual_seed(2)
    layer = nb.SelectiveSSM(6, 4)
    x = torch.randn(3, 15, 6)
    y = layer(x)
    h, ys = None, []
    for t in range(15):
        yt, h = layer.step(x[:, t], h)
        ys.append(yt)
    assert h.shape == (3, 6, 4), f"the recurrent state is (batch, Dm, N); got {tuple(h.shape)}"
    assert torch.allclose(y, torch.stack(ys, 1), atol=1e-5), "stepping through the sequence must reproduce forward"
    x2 = x.clone()
    x2[:, 9:] = torch.randn(3, 6, 6)
    assert torch.allclose(layer(x2)[:, :9], y[:, :9], atol=1e-6), "causal: outputs before t can't depend on inputs at t or later"


def test_stage4_gradients(nb):
    torch.manual_seed(3)
    layer = nb.SelectiveSSM(6, 4)
    layer(torch.randn(2, 8, 6)).pow(2).sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no gradient reached {name}"


# ---------------------------------------------------------------- stage 5: the Mamba block


def test_stage5_structure(nb):
    torch.manual_seed(4)
    blk = nb.MambaBlock(12, d_state=5, expand=3, d_conv=3)
    E = 36
    assert blk.in_proj.weight.shape == (2 * E, 12) and blk.in_proj.bias is None, "in_proj: Linear(d_model, 2 * expand * d_model), no bias"
    assert blk.out_proj.weight.shape == (12, E) and blk.out_proj.bias is None, "out_proj: Linear(expand * d_model, d_model), no bias"
    assert isinstance(blk.conv1d, nn.Conv1d) and blk.conv1d.groups == E and blk.conv1d.kernel_size == (3,) and \
        blk.conv1d.in_channels == E, "conv1d: depthwise Conv1d over the expanded channels (groups = channels), kernel d_conv"
    assert isinstance(blk.ssm, nb.SelectiveSSM) and blk.ssm.A_log.shape == (E, 5), "ssm: SelectiveSSM(expand * d_model, d_state)"
    assert hasattr(blk, "norm") and sum(p.numel() for p in blk.norm.parameters()) == 12, "norm: RMSNorm(d_model) (a weight, no bias)"
    assert nb.MambaBlock(8, dt_rank=3).ssm.dt_proj.in_features == 3, "MambaBlock passes dt_rank on to its SelectiveSSM"
    assert nb.MambaBlock(8).ssm.dt_proj.in_features == 1, "with dt_rank=None the SSM uses its default ceil(16 / 16) = 1"


def test_stage5_matches_manual_composition(nb):
    torch.manual_seed(5)
    blk = nb.MambaBlock(8, d_state=4)
    x = torch.randn(2, 11, 8)
    xi, z = blk.in_proj(blk.norm(x)).chunk(2, dim=-1)
    conv = F.conv1d(F.pad(xi.transpose(1, 2), (blk.conv1d.kernel_size[0] - 1, 0)), blk.conv1d.weight, blk.conv1d.bias,
                    groups=blk.conv1d.groups).transpose(1, 2)
    want = x + blk.out_proj(blk.ssm(F.silu(conv)) * F.silu(z))
    assert torch.allclose(blk(x), want, atol=1e-5), (
        "forward: (x_in, z) = in_proj(norm(x)).chunk(2); x_in -> causal depthwise conv -> SiLU -> ssm; gate with SiLU(z); "
        "out_proj; add the residual x")


def test_stage5_causal_residual_prenorm(nb):
    torch.manual_seed(6)
    blk = nb.MambaBlock(8, d_state=4, d_conv=4)
    x = torch.randn(2, 16, 8)
    y = blk(x)
    assert y.shape == x.shape, "the block maps (batch, L, d_model) to the same shape"
    x2 = x.clone()
    x2[:, 10:] = torch.randn(2, 6, 8)
    assert torch.allclose(blk(x2)[:, :10], y[:, :10], atol=1e-6), (
        "causal: outputs before position 10 changed when only positions >= 10 did. pad the convolution on the left only "
        "(or trim its output to the first L positions)")
    assert torch.allclose(blk(2 * x) - 2 * x, y - x, atol=1e-5), (
        "pre-norm residual: the branch sees RMSNorm(x), which is scale-invariant, so block(2x) - 2x = block(x) - x")
    with torch.no_grad():
        blk.out_proj.weight.zero_()
    assert torch.allclose(blk(x), x), "with out_proj = 0 the block is the identity (residual connection)"
