"""Checks for exercise.py (discrete latents).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Quantization is checked
against brute force, including ties and a vectorization check against a loop baseline. The vector
quantizer's straight-through estimator is checked exactly: the decoder's gradient reaches the encoder
unchanged, the codebook gets gradient only from the codebook loss, and beta scales only the commitment
gradient (each loss term is checked by its own gradient). Usage statistics are checked on hand
examples; dead-code restarts for which codes change and where the new codes come from. The VQ-VAE is
checked for shapes, for decode_indices(encode(x)) reproducing forward, and for training.
"""
import math
import time

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional


# ---------------------------------------------------------------- stage 1: nearest codes


def test_stage1_brute_force(nb):
    g = torch.Generator().manual_seed(0)
    z, cb = torch.randn(300, 5, generator=g), torch.randn(17, 5, generator=g)
    got = nb.nearest_code(z, cb)
    want = torch.cdist(z.double(), cb.double()).argmin(1)
    assert got.dtype == torch.int64 and got.shape == (300,), "int64 indices, one per row of z"
    assert torch.equal(got, want), f"{int((got != want).sum())} of 300 rows got the wrong code (squared Euclidean)"


def test_stage1_ties_and_hand_example(nb):
    cb = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    z = torch.tensor([[0.9, 0.0], [1.0, 0.0], [1.6, 0.1], [0.2, 1.5]])
    got = nb.nearest_code(z, cb).tolist()
    assert got == [0, 0, 1, 2], (
        f"expected [0, 0, 1, 2]; got {got}. (1, 0) is equally far from codes 0 and 1: ties go to the LOWEST index; "
        "code 3 duplicates code 1 and must never win")


def test_stage1_vectorized(nb):
    g = torch.Generator().manual_seed(1)
    z, cb = torch.randn(4000, 16, generator=g), torch.randn(256, 16, generator=g)

    def loop():
        return torch.stack([((cb - row) ** 2).sum(1).argmin() for row in z[:400]])

    t_loop = min(_timed(loop) for _ in range(2)) * 10  # 4000 rows
    t = min(_timed(lambda: nb.nearest_code(z, cb)) for _ in range(3))
    assert t < 0.5 * t_loop + 0.01, (
        f"nearest_code took {t * 1e3:.1f} ms for 4000 x 256, a Python loop over rows takes ~{t_loop * 1e3:.0f} ms: "
        "compute all distances at once, ||z||^2 - 2 z C^T + ||c||^2")


def _timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- stage 2: the vector quantizer


def test_stage2_forward_values(nb):
    torch.manual_seed(2)
    vq = nb.VectorQuantizer(10, 3, beta=0.25)
    assert isinstance(vq.codebook, torch.nn.Embedding) and vq.codebook.weight.shape == (10, 3), (
        "the codebook is an nn.Embedding(n_codes, code_dim)")
    w = vq.codebook.weight
    assert w.abs().max() <= 0.1 + 1e-6 and w.std() > 0.01, "initialize uniformly in [-1/n_codes, 1/n_codes]"
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.randn(10, 3))
    z_e = torch.randn(2, 3, 4, 5)
    z_q, loss, idx = vq(z_e)
    assert z_q.shape == z_e.shape and idx.shape == (2, 4, 5) and idx.dtype == torch.int64, (
        "z_q has z_e's shape (B, D, H, W); indices are int64 (B, H, W)")
    flat = z_e.permute(0, 2, 3, 1).reshape(-1, 3)
    want_idx = torch.cdist(flat, vq.codebook.weight).argmin(1).view(2, 4, 5)
    assert torch.equal(idx, want_idx), "index [b, h, w] is the nearest code to the vector z_e[b, :, h, w]"
    want_q = vq.codebook.weight[idx].permute(0, 3, 1, 2)
    assert torch.allclose(z_q, want_q, atol=1e-6), "the forward value of z_q is the selected code vectors, channels first"
    mse = F.mse_loss(want_q, z_e)
    assert torch.allclose(loss, 1.25 * mse, atol=1e-6), (
        f"vq_loss = mse(z_q, sg(z_e)) + beta * mse(z_e, sg(z_q)), means over all elements: its VALUE is "
        f"(1 + beta) * mse = {1.25 * mse.item():.5f}; got {loss.item():.5f}")


def test_stage2_straight_through(nb):
    torch.manual_seed(3)
    vq = nb.VectorQuantizer(8, 4, beta=0.25)
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.randn(8, 4))
    z_e = torch.randn(2, 4, 3, 3, requires_grad=True)
    z_q, _, _ = vq(z_e)
    w = torch.randn_like(z_q)
    (z_q * w).sum().backward()
    assert z_e.grad is not None and torch.allclose(z_e.grad, w), (
        "straight-through: the gradient of anything computed from z_q must reach z_e UNCHANGED (d z_q / d z_e = I); "
        "return z_e + (z_q - z_e).detach()")
    g = vq.codebook.weight.grad
    assert g is None or torch.all(g == 0), (
        "the decoder's gradient must not reach the codebook through z_q (only the codebook loss trains it)")


def test_stage2_loss_gradients(nb):
    for beta in (0.25, 1.0):
        _check_loss_gradients(nb, beta)


def _check_loss_gradients(nb, beta):
    torch.manual_seed(4)
    vq = nb.VectorQuantizer(6, 2, beta=beta)
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.randn(6, 2))
    z_e = torch.randn(3, 2, 2, 2, requires_grad=True)
    _, loss, idx = vq(z_e)
    loss.backward()
    e = vq.codebook.weight.detach()[idx].permute(0, 3, 1, 2)
    n = z_e.numel()
    assert torch.allclose(z_e.grad, beta * 2 * (z_e.detach() - e) / n, atol=1e-6), (
        f"beta = {beta}: the encoder gets gradient only from the commitment term, beta * 2 (z_e - e) / numel; check "
        "that beta multiplies mse(z_e, sg(z_q)) and that the codebook term detaches z_e")
    want_cb = torch.zeros(6, 2)
    flat_diff = (e - z_e.detach()).permute(0, 2, 3, 1).reshape(-1, 2)
    want_cb.index_add_(0, idx.flatten(), 2 * flat_diff / n)
    assert torch.allclose(vq.codebook.weight.grad, want_cb, atol=1e-6), (
        "the codebook gets gradient only from the codebook term, 2 (e - z_e) / numel per assignment, NOT scaled by beta")


# ---------------------------------------------------------------- stage 3: codebook health


def test_stage3_usage_hand_example(nb):
    idx = torch.tensor([[0, 0], [1, 1]])
    perp, frac = nb.codebook_usage(idx, 8)
    assert math.isclose(perp, 2.0, rel_tol=1e-9) and math.isclose(frac, 0.25), (
        f"two codes used equally: perplexity exp(ln 2) = 2, and 2 of 8 codes used = 0.25; got ({perp}, {frac})")
    perp, frac = nb.codebook_usage(torch.tensor([3, 3, 3, 5]), 6)
    h = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert math.isclose(perp, math.exp(h), rel_tol=1e-9) and math.isclose(frac, 2 / 6), (
        "perplexity = exp(entropy in nats) of the empirical distribution; unused codes contribute 0 (not nan)")
    perp, frac = nb.codebook_usage(torch.arange(16), 16)
    assert math.isclose(perp, 16.0, rel_tol=1e-9) and frac == 1.0, "uniform use of all K codes: perplexity K"
    assert isinstance(perp, float) and isinstance(frac, float)


def test_stage3_restart_dead_codes(nb):
    torch.manual_seed(5)
    vq = nb.VectorQuantizer(6, 3)
    before = vq.codebook.weight.detach().clone()
    z_e = torch.randn(2, 3, 4, 4)
    counts = torch.tensor([5, 0, 2, 0, 1, 0])
    n = nb.restart_dead_codes(vq, z_e, counts, generator=torch.Generator().manual_seed(0))
    after = vq.codebook.weight.detach()
    assert n == 3, f"codes 1, 3, 5 have count 0 < threshold 1: restart 3; got {n}"
    assert torch.equal(after[[0, 2, 4]], before[[0, 2, 4]]), "live codes must not change"
    flat = z_e.permute(0, 2, 3, 1).reshape(-1, 3)
    pick = torch.randint(len(flat), (3,), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(after[[1, 3, 5]], flat[pick]), (
        "each dead code becomes an encoder output vector z_e[b, :, h, w], drawn with "
        "torch.randint(N, (n_dead,), generator=generator) over the N = B*H*W flattened positions, in order")
    n2 = nb.restart_dead_codes(vq, z_e, torch.tensor([5, 1, 2, 1, 1, 1]), threshold=2)
    assert n2 == 4, "threshold=2: every code used fewer than 2 times is dead"
    assert nb.restart_dead_codes(vq, z_e, torch.ones(6)) == 0


def test_stage3_restarts_fix_collapse(nb):
    torch.manual_seed(6)
    vq = nb.VectorQuantizer(64, 2)
    g = torch.Generator().manual_seed(1)
    data = torch.randn(2000, 2, generator=g) * 3  # wide data, tiny initial codebook: most codes never win
    z_e = data.T.reshape(1, 2, 40, 50)
    idx = vq(z_e)[2]
    _, frac0 = nb.codebook_usage(idx, 64)
    counts = torch.bincount(idx.flatten(), minlength=64)
    nb.restart_dead_codes(vq, z_e, counts, generator=torch.Generator().manual_seed(2))
    _, frac1 = nb.codebook_usage(vq(z_e)[2], 64)
    assert frac0 < 0.4 and frac1 > 0.8, (
        f"with a codebook initialized near 0 and data of scale 3, few codes win ({frac0:.2f} used); after restarting "
        f"the dead ones at encoder outputs most should be in use (got {frac1:.2f})")


# ---------------------------------------------------------------- stage 4: the VQ-VAE


def test_stage4_shapes_and_roundtrip(nb):
    torch.manual_seed(7)
    m = nb.VQVAE(1, n_codes=16, code_dim=8, base_channels=16).eval()
    assert isinstance(m.vq, nb.VectorQuantizer) and m.vq.codebook.weight.shape == (16, 8), (
        "self.vq is a VectorQuantizer(n_codes, code_dim)")
    x = torch.rand(3, 1, 16, 12)
    recon, loss, idx = m(x)
    assert recon.shape == x.shape and idx.shape == (3, 4, 3) and loss.dim() == 0, (
        f"forward -> (x_recon (B, C, H, W), scalar vq_loss, indices (B, H/4, W/4)); got {tuple(recon.shape)}, "
        f"{tuple(idx.shape)}")
    assert torch.equal(m.encode(x), idx), "encode returns the same indices as forward"
    with torch.no_grad():
        again = m.decode_indices(idx)
    assert torch.allclose(again, recon, atol=1e-5), "decode_indices(encode(x)) must reproduce forward's reconstruction"
    assert m.decode_indices(torch.zeros(2, 5, 5, dtype=torch.int64)).shape == (2, 1, 20, 20)


def test_stage4_trains(nb):
    torch.manual_seed(8)
    g = torch.Generator().manual_seed(0)
    protos = torch.rand(4, 1, 8, 8, generator=g)
    x = protos[torch.randint(0, 4, (64,), generator=g)]
    m = nb.VQVAE(1, n_codes=16, code_dim=4, base_channels=8)
    recon, _, _ = m(x)
    F.mse_loss(recon, x).backward()  # the reconstruction loss alone, no vq_loss
    enc = [p for n, p in m.named_parameters() if not n.startswith("decoder") and "codebook" not in n]
    gsq = sum(float((p.grad ** 2).sum()) for p in enc if p.grad is not None)
    with_grad = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in enc)
    assert enc and gsq > 0 and with_grad >= 0.75 * len(enc), (
        "the encoder must get gradient from the reconstruction loss alone: the decoder's gradient flows through the "
        "quantizer by the straight-through estimator")
    m.zero_grad()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    def run():
        out = []
        for _ in range(60):
            opt.zero_grad()
            recon, vq_loss, _ = m(x)
            loss = F.mse_loss(recon, x) + vq_loss
            loss.backward()
            opt.step()
            out.append(F.mse_loss(recon, x).item())
        return out

    mse = bounded(run, budget=30_000_000, what="training a VQ-VAE")
    assert np.mean(mse[-5:]) < 0.65 * np.mean(mse[:5]), (
        f"reconstruction error should fall by a third or more in 60 steps ({np.mean(mse[:5]):.4f} -> {np.mean(mse[-5:]):.4f})")


# ---------------------------------------------------------------- added clauses


def test_stage2_default_beta_and_sequences(nb):
    torch.manual_seed(22)
    vq = nb.VectorQuantizer(5, 3)
    with torch.no_grad():
        vq.codebook.weight.copy_(torch.randn(5, 3))
    z = torch.randn(2, 3, 4, 4)
    _, loss, idx = vq(z)
    mse = F.mse_loss(vq.codebook.weight[idx].permute(0, 3, 1, 2), z)
    assert torch.allclose(loss, 1.25 * mse, atol=1e-6), "beta defaults to 0.25: vq_loss's value is 1.25 * mse"
    seq = torch.randn(2, 3, 7, requires_grad=True)  # audio-like (B, D, T)
    z_q, loss1, idx1 = vq(seq)
    assert z_q.shape == (2, 3, 7) and idx1.shape == (2, 7) and idx1.dtype == torch.int64, (
        "any number of trailing positions: (B, D, T) -> z_q (B, D, T), indices (B, T)")
    want = torch.cdist(seq.detach().transpose(1, 2).reshape(-1, 3), vq.codebook.weight.detach()).argmin(1).view(2, 7)
    assert torch.equal(idx1, want), "index [b, t] is the nearest code to seq[b, :, t]"
    assert torch.allclose(z_q, vq.codebook.weight[idx1].transpose(1, 2), atol=1e-6)
    w = torch.randn_like(z_q)
    (z_q * w).sum().backward()
    assert torch.allclose(seq.grad, w), "straight-through for sequences too"


def test_stage3_restart_sequences(nb):
    torch.manual_seed(23)
    vq = nb.VectorQuantizer(4, 2)
    z_e = torch.randn(3, 2, 5)
    n = nb.restart_dead_codes(vq, z_e, torch.tensor([0, 3, 0, 1]), generator=torch.Generator().manual_seed(0))
    flat = z_e.transpose(1, 2).reshape(-1, 2)
    pick = torch.randint(15, (2,), generator=torch.Generator().manual_seed(0))
    assert n == 2 and torch.allclose(vq.codebook.weight[[0, 2]], flat[pick]), (
        "z_e (B, D, T): the N = B*T vectors z_e[b, :, t] in that order")
    vq2 = nb.VectorQuantizer(4, 2)
    nb.restart_dead_codes(vq2, flat, torch.tensor([0, 3, 0, 1]), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(vq2.codebook.weight[[0, 2]], flat[pick]), "an already flattened (N, D) z_e works too"
