"""Checks for exercise.py (non-contrastive methods).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The losses are checked
by hand and through their gradients (which must not reach the stop-gradient branch); the collapse
metrics on embeddings with known structure; SimSiam and BYOL against losses recomputed from their
own submodules, with the EMA update checked exactly; the training loop on tiny unlabelled data.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


def _cos(a, b):
    return F.cosine_similarity(a, b, dim=1)


# ---------------------------------------------------------------- stage 1: simsiam_loss, byol_loss

def test_stage1_worked_example(nb):
    p1 = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    p2 = torch.tensor([[1.0, 1.0], [3.0, 0.0]])
    z1 = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    z2 = torch.tensor([[1.0, 1.0], [0.0, -1.0]])
    got = nb.simsiam_loss(p1, p2, z1, z2)
    # cos(p1, z2) = (1/sqrt2, -1); cos(p2, z1) = (1/sqrt2, 1/sqrt2)
    want = -((1 / math.sqrt(2) - 1) / 2 + (1 / math.sqrt(2) + 1 / math.sqrt(2)) / 2) / 2
    assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor"
    assert abs(got.item() - want) < 1e-6, (
        f"lesson worked example: SimSiam loss {got.item():.4f}, expected {want:.4f} = -(mean cos(p1, z2) + "
        "mean cos(p2, z1)) / 2. note the pairing: each view's prediction is compared with the OTHER view's projection")
    b = nb.byol_loss(p1, z2)
    want_b = ((2 - 2 / math.sqrt(2)) + (2 + 2)) / 2
    assert abs(b.item() - want_b) < 1e-6, (
        f"lesson worked example: BYOL loss {b.item():.4f}, expected {want_b:.4f} = mean(2 - 2 cos(p, z)), which "
        "equals the mean squared distance between the L2-normalised p and z")


def test_stage1_matches_normalised_mse(nb):
    g = torch.Generator().manual_seed(0)
    p, z = torch.randn(7, 5, generator=g), torch.randn(7, 5, generator=g) * 4
    got = nb.byol_loss(p, z).item()
    want = (F.normalize(p, dim=1) - F.normalize(z, dim=1)).pow(2).sum(1).mean().item()
    assert abs(got - want) < 1e-5, f"byol_loss {got:.5f}, expected {want:.5f} (squared distance of unit vectors)"
    p1, p2, z1, z2 = (torch.randn(6, 4, generator=g) for _ in range(4))
    a = nb.simsiam_loss(p1, p2, z1, z2).item()
    assert abs(a - nb.simsiam_loss(p2, p1, z2, z1).item()) < 1e-6, "the SimSiam loss is symmetric in the two views"
    assert abs(a - nb.simsiam_loss(3 * p1, p2, z1, 0.1 * z2).item()) < 1e-5, "use cosines: scale must not matter"
    assert -1 - 1e-6 <= a <= 1 + 1e-6


def test_stage1_stop_gradient(nb):
    g = torch.Generator().manual_seed(1)
    p1, p2, z1, z2 = (torch.randn(5, 3, generator=g, requires_grad=True) for _ in range(4))
    nb.simsiam_loss(p1, p2, z1, z2).backward()
    assert z1.grad is None and z2.grad is None, (
        "no gradient may reach z1 or z2 through simsiam_loss: they are targets, use .detach() (the stop-gradient)")
    assert p1.grad is not None and p1.grad.abs().sum() > 0 and p2.grad.abs().sum() > 0, "p1 and p2 must get gradients"
    # the gradient on p1 is that of -mean cos(p1, z2) / 2
    q = p1.detach().clone().requires_grad_(True)
    (-_cos(q, z2.detach()).mean() / 2).backward()
    torch.testing.assert_close(p1.grad, q.grad, msg="p1's gradient should be that of -mean cos(p1, z2) / 2")
    p, z = torch.randn(4, 3, generator=g, requires_grad=True), torch.randn(4, 3, generator=g, requires_grad=True)
    nb.byol_loss(p, z).backward()
    assert z.grad is None, "byol_loss must not send gradients into its target z (detach it)"
    assert p.grad is not None and p.grad.abs().sum() > 0


# ---------------------------------------------------------------- stage 2: collapse_metrics

def test_stage2_collapsed_and_isotropic(nb):
    g = torch.Generator().manual_seed(2)
    same = torch.randn(1, 16, generator=g).repeat(200, 1)
    m = nb.collapse_metrics(same)
    assert set(m) >= {"std", "effective_rank"}, "return a dict with keys 'std' and 'effective_rank'"
    assert abs(m["std"]) < 1e-6, f"identical embeddings (complete collapse) must have std 0, got {m['std']}"
    assert abs(m["effective_rank"] - 1) < 1e-3, f"identical embeddings have effective rank 1, got {m['effective_rank']}"
    iso = torch.randn(20_000, 16, generator=g)
    m = nb.collapse_metrics(iso)
    assert abs(m["std"] - 1 / math.sqrt(16)) < 0.005, (
        f"isotropic Gaussian embeddings in d=16: the per-dimension std of the L2-normalised rows should be about "
        f"1/sqrt(16) = 0.25, got {m['std']:.4f}")
    assert 15.5 < m["effective_rank"] <= 16 + 1e-6, f"isotropic embeddings in d=16 have effective rank ~16, got {m['effective_rank']:.2f}"
    assert isinstance(m["std"], float) and isinstance(m["effective_rank"], float), "return Python floats"


def test_stage2_hand_examples(nb):
    # rows (1, 0) and (0, 1), half each: normalised already; per-dim population std = 0.5
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(50, 1)
    m = nb.collapse_metrics(z * 3.0)
    assert abs(m["std"] - 0.5) < 1e-6, (
        f"std of (1,0)/(0,1) rows: each dimension is 0 or 1 half the time, population std 0.5; got {m['std']:.4f}. "
        "normalise the rows first, and use the population std (divide by n)")
    assert abs(m["effective_rank"] - 2) < 1e-4, f"two equal singular values: effective rank 2, got {m['effective_rank']}"
    # singular values 3 and 1: p = (0.75, 0.25), erank = exp(H(p))
    z = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    want = math.exp(-(0.75 * math.log(0.75) + 0.25 * math.log(0.25)))
    got = nb.collapse_metrics(z)["effective_rank"]
    assert abs(got - want) < 1e-4, (
        f"singular values (3, 1): p = (0.75, 0.25) and the effective rank is exp(H(p)) = {want:.4f}, got {got:.4f}. "
        "normalise the singular values by their SUM (not their squares) and don't centre or normalise the rows")
    low = torch.randn(500, 3) @ torch.randn(3, 32)  # rank 3 in 32 dimensions
    assert nb.collapse_metrics(low)["effective_rank"] < 3 + 1e-3, "rank-3 embeddings have effective rank at most 3"
    z = torch.randn(64, 8, requires_grad=True)
    nb.collapse_metrics(z)  # must not need .detach() from the caller


# ---------------------------------------------------------------- stage 3: SimSiam

def _encoder(seed, d_in=12, d=10):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU())


def test_stage3_structure(nb):
    m = nb.SimSiam(_encoder(0), 10, proj_dim=6, pred_dim=3)
    assert hasattr(m, "encoder") and hasattr(m, "projector") and hasattr(m, "predictor"), (
        "SimSiam needs attributes encoder, projector and predictor")
    kinds = [type(x).__name__ for x in m.projector.modules() if not list(x.children())]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Linear", "BatchNorm1d"], f"projector layers {kinds}"
    kinds = [type(x).__name__ for x in m.predictor.modules() if not list(x.children())]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Linear"], f"predictor layers {kinds}"
    lin = [x for x in m.predictor.modules() if isinstance(x, nn.Linear)]
    assert (lin[0].in_features, lin[0].out_features, lin[1].out_features) == (6, 3, 6), (
        "the predictor is a bottleneck: proj_dim -> pred_dim -> proj_dim")
    lin = [x for x in m.projector.modules() if isinstance(x, nn.Linear)]
    assert (lin[0].in_features, lin[0].out_features, lin[1].out_features) == (10, 6, 6), (
        "the projector maps feature_dim -> proj_dim -> proj_dim")
    torch.manual_seed(1)
    loss, z1 = m(torch.randn(8, 12), torch.randn(8, 12))
    assert loss.dim() == 0 and z1.shape == (8, 6), "forward returns (scalar loss, z1 of shape (B, proj_dim))"


def test_stage3_loss_and_stop_gradient(nb):
    torch.manual_seed(2)
    x1, x2 = torch.randn(16, 12), torch.randn(16, 12)
    for sg in (True, False):
        torch.manual_seed(3)
        m = nb.SimSiam(_encoder(4), 10, proj_dim=6, pred_dim=3, stop_gradient=sg)
        m.train()
        loss, z1 = m(x1, x2)
        z1r, z2r = m.projector(m.encoder(x1)), m.projector(m.encoder(x2))
        p1, p2 = m.predictor(z1r), m.predictor(z2r)
        want = -(_cos(p1, z2r).mean() + _cos(p2, z1r).mean()) / 2
        assert abs(loss.item() - want.item()) < 1e-5, (
            f"stop_gradient={sg}: loss {loss.item():.5f}, expected {want.item():.5f} = -(cos(p1, z2) + cos(p2, z1)) / 2 "
            "with z = projector(encoder(x)) and p = predictor(z)")
        torch.testing.assert_close(z1, z1r, msg="the second output must be view 1's projection z1")
        grads = torch.autograd.grad(loss, list(m.encoder.parameters()))
        if sg:
            ref = simsiam_ref = -(_cos(p1, z2r.detach()).mean() + _cos(p2, z1r.detach()).mean()) / 2
        else:
            ref = want
        want_g = torch.autograd.grad(ref, list(m.encoder.parameters()))
        for a, b in zip(grads, want_g):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4, msg=(
                f"stop_gradient={sg}: the encoder's gradient is wrong. with the stop-gradient, the z's are constants "
                "(use simsiam_loss); without it, gradients flow through both branches"))
    m = nb.SimSiam(_encoder(0), 10)
    assert m.stop_gradient is True and m.projector[0].out_features == 128, "defaults: proj_dim=128, stop_gradient=True"


# ---------------------------------------------------------------- stage 4: BYOL

def test_stage4_target_is_a_frozen_copy(nb):
    enc = _encoder(5)
    m = nb.BYOL(enc, 10, proj_dim=6, hidden_dim=8, momentum=0.9)
    assert m.encoder is enc, "keep the encoder you were given as the online encoder"
    assert m.target_encoder is not enc and m.target_projector is not m.projector, "the target must be a deep copy"
    for a, b in zip(list(m.encoder.parameters()) + list(m.projector.parameters()),
                    list(m.target_encoder.parameters()) + list(m.target_projector.parameters())):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr(), "the target starts equal to the online network"
        assert not b.requires_grad, "target parameters must have requires_grad=False"
    kinds = [type(x).__name__ for x in m.predictor.modules() if not list(x.children())]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Linear"], f"predictor layers {kinds}"
    lin = [x for x in m.projector.modules() if isinstance(x, nn.Linear)]
    assert (lin[0].in_features, lin[0].out_features, lin[1].out_features) == (10, 8, 6), (
        "projector: feature_dim -> hidden_dim -> proj_dim")


def test_stage4_loss(nb):
    torch.manual_seed(6)
    m = nb.BYOL(_encoder(6), 10, proj_dim=6, hidden_dim=8)
    with torch.no_grad():  # make the target differ from the online network
        for p in m.target_projector.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x1, x2 = torch.randn(16, 12), torch.randn(16, 12)
    m.train()
    loss, z1 = m(x1, x2)
    z1r, z2r = m.projector(m.encoder(x1)), m.projector(m.encoder(x2))
    p1, p2 = m.predictor(z1r), m.predictor(z2r)
    t1, t2 = m.target_projector(m.target_encoder(x1)), m.target_projector(m.target_encoder(x2))
    want = (2 - 2 * _cos(p1, t2)).mean() + (2 - 2 * _cos(p2, t1)).mean()
    assert abs(loss.item() - want.item()) < 1e-5, (
        f"loss {loss.item():.5f}, expected {want.item():.5f} = byol_loss(p1, t2) + byol_loss(p2, t1), the targets "
        "t computed by the TARGET encoder and projector")
    torch.testing.assert_close(z1, z1r)
    loss.backward()
    assert all(p.grad is None for p in m.target_encoder.parameters()), "no gradient may reach the target network"
    assert m.predictor[0].weight.grad is not None and m.encoder[0].weight.grad is not None


def test_stage4_update_target_is_an_ema(nb):
    for mom in (0.9, 0.0, 1.0):
        m = nb.BYOL(_encoder(7), 10, proj_dim=6, hidden_dim=8, momentum=mom)
        with torch.no_grad():
            for p in list(m.encoder.parameters()) + list(m.projector.parameters()):
                p.add_(torch.randn_like(p))
            m.encoder[1].running_mean.fill_(2.0)
        old = [p.detach().clone() for p in list(m.target_encoder.parameters()) + list(m.target_projector.parameters())]
        new = [p.detach().clone() for p in list(m.encoder.parameters()) + list(m.projector.parameters())]
        m.update_target()
        got = list(m.target_encoder.parameters()) + list(m.target_projector.parameters())
        for g, o, n in zip(got, old, new):
            torch.testing.assert_close(g.detach(), mom * o + (1 - mom) * n, msg=(
                f"momentum={mom}: target <- momentum * target + (1 - momentum) * online, for the encoder AND projector"))
        assert torch.equal(m.target_encoder[1].running_mean, m.encoder[1].running_mean), (
            "BatchNorm buffers are copied from the online network (ema_update does this)")


# ---------------------------------------------------------------- stage 5: train_ssl

def _images(n, seed, size=8):
    rng = np.random.default_rng(seed)
    X = 0.1 * rng.random((n, 3, size, size))
    for i in range(n):
        r = rng.integers(1, size - 1)
        colour = rng.random(3)[:, None]
        if rng.random() < 0.5:
            X[i, :, r, :] = colour
        else:
            X[i, :, :, r] = colour
    return X.astype(np.float32)


def _conv_encoder(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(192, 32), nn.BatchNorm1d(32), nn.ReLU())


def test_stage5_simsiam_learns(nb):
    drops = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = nb.SimSiam(_conv_encoder(seed), 32, proj_dim=16, pred_dim=8)
        h = nb.train_ssl(m, _images(256, seed), epochs=5, batch_size=64, lr=3e-3, seed=seed)
        assert len(h["loss"]) == 20 and len(h["z_std"]) == 20, (
            f"5 epochs of 256 images in batches of 64 = 20 steps; history lengths {len(h['loss'])}, {len(h['z_std'])}")
        drops.append(np.mean(h["loss"][:3]) - np.mean(h["loss"][-3:]))
        assert all(0 <= s <= 1 for s in h["z_std"]), "z_std entries are collapse_metrics(z1)['std'] per step"
    assert np.median(drops) > 0.05, f"the SimSiam loss fell by {np.round(drops, 3)} over 3 seeds; expected > 0.05"


def test_stage5_byol_target_updates_each_step(nb):
    calls = []
    torch.manual_seed(0)
    m = nb.BYOL(_conv_encoder(0), 32, proj_dim=8, hidden_dim=16, momentum=0.5)
    orig = m.update_target
    m.update_target = lambda: (calls.append(1), orig())[1]
    t0 = m.target_encoder[1].weight.detach().clone()
    h = nb.train_ssl(m, _images(96, 1), epochs=2, batch_size=32, lr=1e-2, seed=0)
    assert len(calls) == 6 == len(h["loss"]), (
        f"call model.update_target() once after every optimizer step when the model has one ({len(calls)} calls, "
        f"{len(h['loss'])} steps)")
    assert not torch.equal(m.target_encoder[1].weight, t0), "the target should have moved towards the online network"

    torch.manual_seed(0)
    m = nb.BYOL(_conv_encoder(0), 32, proj_dim=8, hidden_dim=16, momentum=0.0)
    nb.train_ssl(m, _images(64, 1), epochs=1, batch_size=32, lr=1e-2, seed=0)
    assert torch.equal(m.target_encoder[1].weight, m.encoder[1].weight), (
        "with momentum 0 the target must equal the online network after training: update the target AFTER "
        "optimizer.step(), so it follows the weights the step produced")


def test_stage5_any_model_two_views(nb):
    seen, zs = [], []

    class Spy(nn.Module):
        """the train_ssl protocol: model(x1, x2) -> (loss, z1)."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(192, 4)

        def forward(self, x1, x2):
            seen.append((x1.detach().clone(), x2.detach().clone()))
            z1, z2 = self.lin(x1.flatten(1)), self.lin(x2.flatten(1))
            zs.append(z1.detach().clone())
            return (z1 - z2).pow(2).mean(), z1

    X = _images(40, 4)
    spy = Spy()
    h = nb.train_ssl(spy, X, epochs=1, batch_size=16, seed=0)
    assert len(seen) == 2 and len(h["loss"]) == 2, (
        f"40 images in batches of 16: N // bs = 2 FULL batches (the 8 left over are skipped); took {len(h['loss'])} steps")
    x1, x2 = seen[0]
    assert x1.shape == (16, 3, 8, 8) and (x1 - x2).abs().max() > 1e-3, (
        "pass two independent augmentations of the batch: model(augment(xb), augment(xb))")
    want = F.normalize(zs[-1], dim=1).std(dim=0, correction=0).mean().item()
    assert abs(h["z_std"][-1] - want) < 1e-5, (
        "z_std must be collapse_metrics(z1)['std'] of the z1 the model returned for that step")


def test_stage5_seed_and_options(nb):
    X = _images(64, 2)

    def run(**kw):
        torch.manual_seed(0)
        m = nb.SimSiam(_conv_encoder(0), 32, proj_dim=8, pred_dim=4)
        args = dict(epochs=2, batch_size=32, lr=1e-2, seed=5)
        args.update(kw)
        return m, nb.train_ssl(m, X, **args)

    a, b, c = run()[1]["loss"], run()[1]["loss"], run(seed=6)[1]["loss"]
    assert np.allclose(a, b), "the same seed must give the same run"
    assert not np.allclose(a, c), "a different seed should give a different run"
    s = run(strength=1.0)[1]["loss"]
    assert abs(s[0] - a[0]) > 1e-6, "strength is ignored: pass it to SimCLRAugment"
    m, _ = run(lr=0.0)
    torch.manual_seed(0)
    fresh = nb.SimSiam(_conv_encoder(0), 32, proj_dim=8, pred_dim=4)
    assert torch.equal(m.encoder[1].weight, fresh.encoder[1].weight), "lr=0 must leave the weights unchanged"
    m, _ = run(lr=1e-2)
    assert not torch.equal(m.encoder[1].weight, fresh.encoder[1].weight), "training should change the encoder"
    assert not torch.equal(m.predictor[0].weight, fresh.predictor[0].weight), "the predictor must be trained too"
    m.eval()
    run_after = nb.train_ssl(m, X, epochs=1, batch_size=32, lr=0.0, seed=0)
    assert m.training, "train_ssl must put the model in train mode"
    assert len(run_after["loss"]) == 2


def test_stage5_device(nb):
    class Spy(nn.Module):
        """records where batches and the model are sent; computes on the CPU whatever the inputs' device."""

        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 4)
            self.moved_to, self.seen = [], []

        def to(self, *args, **kwargs):
            self.moved_to.append(str(args[0] if args else kwargs.get("device")))
            return self

        def forward(self, x1, x2):
            self.seen += [x1.device.type, x2.device.type]
            z = self.lin(torch.ones(x1.shape[0], 3))
            return z.pow(2).mean(), z

    spy = Spy()
    nb.train_ssl(spy, _images(32, 3), epochs=1, batch_size=16, device="meta")
    assert spy.moved_to == ["meta"], f"move the model to `device` with model.to(device); saw {spy.moved_to}"
    assert spy.seen == ["meta"] * 4, "move every batch to `device` (augment it there)"
