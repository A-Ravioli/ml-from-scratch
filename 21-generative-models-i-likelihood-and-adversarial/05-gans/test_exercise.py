"""Checks for exercise.py (GANs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The losses are checked
against their definitions in probabilities, for numerical stability at extreme logits, and for the
property that matters: the saturating generator loss has vanishing gradient where the discriminator
is confident, the non-saturating one doesn't. One GAN step is checked against an independent
re-implementation run on copies of the same networks with the same random draws (so the order of the
updates and the loss choice matter; the detach only saves compute, since zero_grad clears what it would leak),
and with counting wrappers. The DCGAN networks
are checked for shapes, ranges, initialization and validation. Mode coverage is checked on hand
examples and on samples from known mixtures.
"""
import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
nn = torch.nn


# ---------------------------------------------------------------- stage 1: the losses


def test_stage1_discriminator_loss(nb):
    real = torch.tensor([2.0, -1.0, 0.5])
    fake = torch.tensor([-3.0, 0.0])
    dr, df = torch.sigmoid(real), torch.sigmoid(fake)
    want = -torch.log(dr).mean() - torch.log(1 - df).mean()
    got = nb.discriminator_loss(real, fake)
    assert got.dim() == 0 and torch.allclose(got, want, atol=1e-6), (
        f"-mean log D(real) - mean log(1 - D(fake)) = {want.item():.5f}; got {got.item():.5f}. each term is a MEAN over "
        "its own batch (they can have different sizes)")
    assert math.isclose(nb.discriminator_loss(torch.zeros(4), torch.zeros(4)).item(), 2 * math.log(2), rel_tol=1e-6), (
        "a discriminator that outputs D = 1/2 everywhere costs 2 ln 2")


def test_stage1_generator_losses(nb):
    fake = torch.tensor([-2.0, 0.0, 1.5])
    d = torch.sigmoid(fake)
    ns = nb.generator_loss(fake)
    sat = nb.generator_loss(fake, non_saturating=False)
    assert torch.allclose(ns, -torch.log(d).mean(), atol=1e-6), "non-saturating (the default): mean -log D(G(z))"
    assert torch.allclose(sat, torch.log(1 - d).mean(), atol=1e-6), (
        "saturating (the original minimax): mean log(1 - D(G(z))), which the generator MINIMIZES")


def test_stage1_stable_at_extreme_logits(nb):
    big = torch.tensor([80.0, -80.0])
    for v in (nb.discriminator_loss(big, -big), nb.generator_loss(big), nb.generator_loss(-big),
              nb.generator_loss(big, non_saturating=False), nb.generator_loss(-big, non_saturating=False)):
        assert torch.isfinite(v), "losses must stay finite for logits of +-80: work with logits (softplus), not sigmoid"
    assert math.isclose(nb.generator_loss(torch.tensor([-80.0])).item(), 80.0, rel_tol=1e-4), (
        "-log sigmoid(-80) = 80 exactly (log(sigmoid) underflows to -inf; softplus(-l) doesn't)")


def test_stage1_saturation(nb):
    l = torch.tensor([-6.0], requires_grad=True)  # the discriminator confidently rejects the fake
    nb.generator_loss(l, non_saturating=False).backward()
    g_sat = l.grad.abs().item()
    l.grad = None
    nb.generator_loss(l).backward()
    g_ns = l.grad.abs().item()
    assert g_sat < 0.01 and g_ns > 0.9, (
        f"at logit -6 (D(G(z)) = 0.0025), d/dl log(1 - D) = -D is tiny ({g_sat:.4f}) but d/dl (-log D) = -(1 - D) is ~1 "
        f"({g_ns:.4f}): the non-saturating loss keeps the generator learning when it's losing")


# ---------------------------------------------------------------- stage 2: one step of the game


def _toy_nets(seed):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
    return G, D


def _manual_step(G, D, opt_g, opt_d, real, latent_dim, non_saturating, gen):
    b = real.shape[0]
    z1 = torch.randn((b, latent_dim), generator=gen)
    z2 = torch.randn((b, latent_dim), generator=gen)
    opt_d.zero_grad()
    rl, fl = D(real).view(b), D(G(z1).detach()).view(b)
    (F.softplus(-rl).mean() + F.softplus(fl).mean()).backward()
    opt_d.step()
    opt_g.zero_grad()
    fl = D(G(z2)).view(b)
    (F.softplus(-fl).mean() if non_saturating else -F.softplus(fl).mean()).backward()
    opt_g.step()


def test_stage2_matches_reference_step(nb):
    real = torch.randn(32, 2, generator=torch.Generator().manual_seed(0)) + 2
    for non_saturating in (True, False):
        G, D = _toy_nets(1)
        G2, D2 = copy.deepcopy(G), copy.deepcopy(D)
        og, od = torch.optim.SGD(G.parameters(), lr=0.1), torch.optim.SGD(D.parameters(), lr=0.1)
        og2, od2 = torch.optim.SGD(G2.parameters(), lr=0.1), torch.optim.SGD(D2.parameters(), lr=0.1)
        for k in range(3):
            out = nb.gan_step(G, D, og, od, real, latent_dim=3, non_saturating=non_saturating,
                              generator=torch.Generator().manual_seed(10 + k))
            _manual_step(G2, D2, og2, od2, real, 3, non_saturating, torch.Generator().manual_seed(10 + k))
        assert isinstance(out, tuple) and len(out) == 2 and all(isinstance(v, float) for v in out), (
            "return (d_loss, g_loss) as python floats")
        for (n, p), q in zip(list(D.named_parameters()), D2.parameters()):
            assert torch.allclose(p, q, atol=1e-5), (
                f"non_saturating={non_saturating}: discriminator parameter {n} differs from the reference after 3 steps. "
                "update D first, on D(real) and D(G(z1).detach()), with z1 then z2 drawn from `generator`")
        for (n, p), q in zip(list(G.named_parameters()), G2.parameters()):
            assert torch.allclose(p, q, atol=1e-5), (
                f"non_saturating={non_saturating}: generator parameter {n} differs from the reference. the generator "
                "step uses a FRESH z2, the just-updated D, and generator_loss(..., non_saturating=...)")


def test_stage2_d_step_does_not_touch_g(nb):
    G, D = _toy_nets(2)
    g_before = [p.detach().clone() for p in G.parameters()]
    og = torch.optim.SGD(G.parameters(), lr=0.0)  # a frozen generator
    od = torch.optim.SGD(D.parameters(), lr=0.1)
    calls = {"G": 0, "D": 0}
    G.register_forward_hook(lambda *a: calls.__setitem__("G", calls["G"] + 1))
    D.register_forward_hook(lambda *a: calls.__setitem__("D", calls["D"] + 1))
    d_loss, g_loss = nb.gan_step(G, D, og, od, torch.randn(8, 2), latent_dim=3, generator=torch.Generator().manual_seed(0))
    assert all(torch.equal(a, b) for a, b in zip(g_before, G.parameters())), "with lr 0 the generator can't change"
    assert calls["G"] == 2, f"two generator passes per step (one per update); counted {calls['G']}"
    assert calls["D"] in (2, 3), f"the D update needs D on real and fake, the G update D on fake; counted {calls['D']}"
    assert d_loss > 0 and np.isfinite(g_loss)


# ---------------------------------------------------------------- stage 3: DCGAN


def test_stage3_shapes_and_range(nb):
    torch.manual_seed(3)
    for size, ch in ((32, 3), (28, 1), (16, 1)):
        G = nb.DCGANGenerator(20, ch, base_channels=8, img_size=size)
        D = nb.DCGANDiscriminator(ch, base_channels=8, img_size=size)
        x = G(torch.randn(5, 20) * 10)
        assert x.shape == (5, ch, size, size), f"img_size {size}: G(z) is (B, C, {size}, {size}); got {tuple(x.shape)}"
        assert x.min() >= -1 and x.max() <= 1, "the generator ends in tanh: outputs in [-1, 1]"
        logits = D(x)
        assert logits.shape == (5,), f"the discriminator returns one logit per image, shape (B,); got {tuple(logits.shape)}"
        assert logits.abs().max() > 0
    with pytest.raises(ValueError):
        nb.DCGANGenerator(10, 1, img_size=30)
    with pytest.raises(ValueError):
        nb.DCGANDiscriminator(1, img_size=30)


def test_stage3_architecture_and_init(nb):
    torch.manual_seed(4)
    G = nb.DCGANGenerator(64, 3, base_channels=32, img_size=32)
    D = nb.DCGANDiscriminator(3, base_channels=32, img_size=32)
    convs = [m for m in G.modules() if isinstance(m, nn.ConvTranspose2d)]
    assert len(convs) >= 2 and all(m.stride == (2, 2) for m in convs[:2]), (
        "the generator upsamples with stride-2 transposed convolutions")
    assert any(isinstance(m, nn.BatchNorm2d) for m in G.modules()), "batch norm in the generator"
    assert isinstance(list(G.modules())[-1], nn.Tanh) or any(isinstance(m, nn.Tanh) for m in G.modules())
    dconvs = [m for m in D.modules() if isinstance(m, nn.Conv2d)]
    assert all(m.stride == (2, 2) for m in dconvs[:2]), "the discriminator downsamples with stride-2 convolutions"
    acts = [m for m in D.modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Tanh, nn.SiLU))]
    assert acts and all(isinstance(m, nn.LeakyReLU) and math.isclose(m.negative_slope, 0.2) for m in acts), (
        "every activation in the discriminator is a LeakyReLU(0.2): a plain ReLU zeroes the gradient for half the inputs")
    for net in (G, D):
        ws = torch.cat([m.weight.flatten() for m in net.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))])
        assert abs(ws.std().item() - 0.02) < 0.004 and abs(ws.mean().item()) < 0.002, (
            f"convolution weights are initialized N(0, 0.02) (DCGAN); got std {ws.std().item():.4f}")


def test_stage3_trains_one_step(nb):
    torch.manual_seed(5)
    G = nb.DCGANGenerator(16, 1, base_channels=8, img_size=16)
    D = nb.DCGANDiscriminator(1, base_channels=8, img_size=16)
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real = torch.rand(16, 1, 16, 16) * 2 - 1
    before = [p.detach().clone() for p in G.parameters()]
    nb.gan_step(G, D, og, od, real, latent_dim=16, generator=torch.Generator().manual_seed(0))
    assert any(not torch.equal(a, b) for a, b in zip(before, G.parameters())), "the DCGAN pair works with gan_step"


# ---------------------------------------------------------------- stage 4: mode coverage


def test_stage4_hand_example(nb):
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    samples = np.array([[0.1, 0.0], [0.2, 0.1], [9.9, 0.0], [5.0, 5.0], [0.0, 0.29], [0.0, 10.31]])
    n, frac, counts = nb.mode_coverage(samples, centers, 0.1, min_frac=0.3)
    # within 3 * 0.1 = 0.3 of the nearest centre: rows 0, 1, 2, 4 -> counts (3, 1, 0); covered if >= 0.3 * 6 = 1.8
    assert list(counts) == [3, 1, 0], f"high-quality samples per centre should be [3, 1, 0]; got {list(counts)}"
    assert math.isclose(frac, 4 / 6), f"4 of 6 samples lie within n_std * std of their nearest centre; got {frac}"
    assert n == 1, f"only centre 0 has at least min_frac * n = 1.8 high-quality samples; got {n}"
    assert nb.mode_coverage(samples, centers, 0.1)[0] == 2, "with the default min_frac = 0.01, any hit covers a mode"
    assert nb.mode_coverage(samples, centers, 0.1, min_frac=0.5)[0] == 1, (
        "covered means counts >= min_frac * n (inclusive): centre 0 has exactly 0.5 * 6 = 3")
    assert nb.mode_coverage(samples, centers, 0.1, min_frac=0.25)[0] == 1, (
        "the threshold is a fraction of ALL n samples (0.25 * 6 = 1.5), not of the high-quality ones")
    assert nb.mode_coverage(samples, centers, 0.1, n_std=1.0)[2].sum() == 2, "n_std sets the radius"


def test_stage4_mixtures(nb):
    rng = np.random.default_rng(0)
    ang = 2 * np.pi * np.arange(8) / 8
    centers = 2 * np.c_[np.cos(ang), np.sin(ang)]
    good = centers[rng.integers(0, 8, 4000)] + 0.05 * rng.normal(size=(4000, 2))
    n, frac, counts = nb.mode_coverage(good, centers, 0.05)
    assert n == 8 and frac > 0.97 and counts.sum() == int(round(frac * 4000)), (
        f"samples from the true mixture cover all 8 modes with ~99% high quality; got ({n}, {frac:.3f})")
    collapsed = centers[rng.integers(0, 2, 4000)] + 0.05 * rng.normal(size=(4000, 2))
    assert nb.mode_coverage(collapsed, centers, 0.05)[0] == 2, "a generator that only hits two modes covers 2"
    blurry = 2 * rng.normal(size=(4000, 2))
    n_b, frac_b, _ = nb.mode_coverage(blurry, centers, 0.05)
    assert frac_b < 0.1, f"a broad Gaussian puts few samples near any mode: low quality; got {frac_b:.3f}"
    assert isinstance(n, int) and isinstance(frac, float) and counts.shape == (8,)


def test_stage3_init_details(nb):
    torch.manual_seed(24)
    G = nb.DCGANGenerator(64, 1, base_channels=64, img_size=32)
    D = nb.DCGANDiscriminator(1, base_channels=64, img_size=32)
    for name, net in (("generator", G), ("discriminator", D)):
        bn = torch.cat([m.weight.detach().flatten() for m in net.modules()
                        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))])
        assert abs(bn.mean().item() - 1.0) < 0.01 and abs(bn.std().item() - 0.02) < 0.006, (
            f"{name}: batch-norm scales are initialized N(1, 0.02); got mean {bn.mean():.3f}, std {bn.std():.4f}")
        lin = torch.cat([m.weight.detach().flatten() for m in net.modules() if isinstance(m, nn.Linear)])
        assert abs(lin.std().item() - 0.02) < 0.004, f"{name}: linear weights are initialized N(0, 0.02) too"
    first_conv = next(m for m in D.modules() if isinstance(m, nn.Conv2d))
    mods = list(D.modules())
    after = mods[mods.index(first_conv) + 1:mods.index(first_conv) + 3]
    assert not any(isinstance(m, nn.BatchNorm2d) for m in after), "no batch norm after the discriminator's first layer"
    convs = [m for m in G.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    first_g = [m for m in G.modules() if isinstance(m, nn.Linear)]
    assert first_g and any(isinstance(m, nn.BatchNorm1d) for m in G.modules()), (
        "the generator's projection (Linear) is followed by batch norm, like every upsampling layer")
