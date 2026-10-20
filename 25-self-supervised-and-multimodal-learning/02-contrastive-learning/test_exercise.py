"""Checks for exercise.py (contrastive learning).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The loss is checked by
hand, against a slow independent implementation and with gradcheck; the augmentations are decoded
from images whose pixels encode their own coordinates (so the crop box and flip can be read back)
and from constant-colour images (so brightness and the other jitter factors can be read back), over
thousands of draws; training is checked on tiny unlabelled data over several seeds.
"""
import math

import numpy as np
import pytest
import torch
from torch import nn

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


# ---------------------------------------------------------------- stage 1: info_nce_loss

def _slow_nt_xent(z1, z2, t):
    z = torch.cat([z1, z2]).double()
    z = z / z.norm(dim=1, keepdim=True)
    n, b = len(z), len(z1)
    total = 0.0
    for i in range(n):
        pos = (i + b) % n
        num = torch.exp(z[i] @ z[pos] / t)
        den = sum(torch.exp(z[i] @ z[j] / t) for j in range(n) if j != i)
        total += -torch.log(num / den)
    return total / n


def test_stage1_worked_example(nb):
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    got = nb.info_nce_loss(z1, z2, temperature=1.0)
    want = math.log(1 + 2 / math.e)
    assert torch.is_tensor(got) and got.dim() == 0, "return a scalar tensor (so you can call .backward() on it)"
    assert abs(got.item() - want) < 1e-6, (
        f"lesson worked example: loss {got.item():.4f}, expected log(1 + 2/e) = {want:.4f}. each of the 4 views has "
        "1 positive (similarity 1) and 2 negatives (similarity 0) once its own similarity is excluded. "
        f"if you got {math.log(2 + 2 / math.e):.4f}, you left each view's similarity with itself in the "
        "denominator; if you got "
        f"{math.log(1 + 1 / math.e):.4f}, you only used the other side's views as negatives")


def test_stage1_matches_slow_reference(nb):
    g = torch.Generator().manual_seed(0)
    for b, d, t in ((3, 4, 0.1), (5, 8, 0.5), (4, 3, 1.0)):
        z1, z2 = torch.randn(b, d, generator=g), torch.randn(b, d, generator=g)
        got = nb.info_nce_loss(z1, z2, temperature=t)
        want = _slow_nt_xent(z1, z2, t)
        assert abs(got.item() - want.item()) < 1e-4, (
            f"B={b}, d={d}, temperature={t}: loss {got.item():.5f}, expected {want.item():.5f}. the loss is the mean "
            "over all 2B views of -log(exp(sim(i, pos)/T) / sum_{j != i} exp(sim(i, j)/T)), with cosine similarity")


def test_stage1_invariances(nb):
    g = torch.Generator().manual_seed(1)
    z1, z2 = torch.randn(6, 5, generator=g), torch.randn(6, 5, generator=g)
    base = nb.info_nce_loss(z1, z2, temperature=0.2).item()
    scale = torch.rand(6, 1, generator=g) * 10 + 0.1
    assert abs(nb.info_nce_loss(z1 * scale, z2 * 3.0, temperature=0.2).item() - base) < 1e-4, (
        "rescaling embeddings changed the loss: similarities must be cosines (L2-normalise first)")
    assert abs(nb.info_nce_loss(z2, z1, temperature=0.2).item() - base) < 1e-5, (
        "swapping z1 and z2 must not change the loss: every view of both sides is an anchor")
    perm = torch.randperm(6, generator=g)
    assert abs(nb.info_nce_loss(z1[perm], z2[perm], temperature=0.2).item() - base) < 1e-5
    other = nb.info_nce_loss(z1, z2, temperature=1.0).item()
    assert abs(other - base) > 1e-3, "the loss didn't change with the temperature: divide the similarities by it"
    one = nb.info_nce_loss(z1[:1], z2[:1], temperature=0.2).item()
    assert abs(one) < 1e-6, f"with B=1 there are no negatives, so the loss is 0; got {one}"


def test_stage1_gradient(nb):
    g = torch.Generator().manual_seed(2)
    z1 = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    z2 = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda a, b: nb.info_nce_loss(a, b, temperature=0.3), (z1, z2), eps=1e-6, atol=1e-5)
    assert ok, "the gradient doesn't match finite differences: avoid in-place edits of tensors autograd needs"
    loss = nb.info_nce_loss(z1, z2, temperature=0.3)
    loss.backward()
    assert torch.isfinite(z1.grad).all(), "masking the self-similarities with -inf must not produce nan gradients"


# ---------------------------------------------------------------- stage 2: SimCLRAugment

def _coord_image(n, size=32):
    """pixels encode their own position: R = x, G = y (pixel centres in (0, 1)), B = 0.5."""
    c = (torch.arange(size) + 0.5) / size
    img = torch.stack([c.view(1, -1).expand(size, size), c.view(-1, 1).expand(size, size), torch.full((size, size), 0.5)])
    return img.unsqueeze(0).repeat(n, 1, 1, 1)


def _decode_boxes(out):
    """crop width/height (fractions of the image), centre and flip sign, from coordinate-coded outputs."""
    s = out.shape[-1]
    mid = s // 2
    R, G = out[:, 0], out[:, 1]
    dx = (R[:, mid, s - 2] - R[:, mid, 1]) * s / (s - 3)  # interior pixels: away from the border clamp
    dy = (G[:, s - 2, mid] - G[:, 1, mid]) * s / (s - 3)
    cx = R[:, mid - 1:mid + 1, mid - 1:mid + 1].mean(dim=(1, 2))
    return dx.abs(), dy.abs(), torch.sign(dx), cx


def _not_gray(out):
    return (out[:, 0] - out[:, 2]).abs().amax(dim=(1, 2)) > 1e-4


def test_stage2_shapes_and_range(nb):
    torch.manual_seed(0)
    x = torch.rand(8, 3, 32, 32)
    out = nb.SimCLRAugment(24, strength=0.5)(x)
    assert out.shape == (8, 3, 24, 24), f"output should be (B, C, size, size) = (8, 3, 24, 24), got {tuple(out.shape)}"
    assert out.dtype == x.dtype
    assert out.min() >= 0 and out.max() <= 1, "outputs must stay in [0, 1]: clamp after each colour operation"
    gray = nb.SimCLRAugment(28, strength=1.0)(torch.rand(4, 1, 28, 28))
    assert gray.shape == (4, 1, 28, 28), "single-channel (MNIST) images must work too"


def test_stage2_each_image_gets_its_own_draw(nb):
    torch.manual_seed(1)
    x = torch.rand(1, 3, 32, 32).repeat(6, 1, 1, 1)
    aug = nb.SimCLRAugment(32, strength=0.5)
    a, b = aug(x), aug(x)
    diffs = [(a[i] - a[j]).abs().max().item() for i in range(6) for j in range(i + 1, 6)]
    assert min(diffs) > 1e-3, "identical images in one batch came out identical: draw random parameters per image"
    assert (a - b).abs().max() > 1e-3, "two calls must give two different views"


def test_stage2_crop_and_flip(nb):
    torch.manual_seed(2)
    out = nb.SimCLRAugment(32, strength=0.0)(_coord_image(3000))
    keep = _not_gray(out)
    w, h, sgn, cx = _decode_boxes(out[keep])
    area = w * h
    assert area.min() > 0.17 and area.min() < 0.3, (
        f"smallest crop area {area.min():.3f} of the image: crops should cover between 20% and 100% of the area")
    assert area.max() > 0.9, f"largest crop area {area.max():.3f}: the area fraction should reach (almost) 1"
    assert 0.5 < area.mean() < 0.64, f"mean crop area {area.mean():.3f}: expected about 0.57 (uniform in [0.2, 1], clipped)"
    ratio = w / h
    small = area < 0.5  # unclipped crops: aspect ratio in [3/4, 4/3]
    assert ratio[small].min() > 0.7 and ratio[small].max() < 1.42, (
        f"aspect ratios (w/h) range over [{ratio[small].min():.2f}, {ratio[small].max():.2f}], expected within [3/4, 4/3]")
    assert ratio[small].min() < 0.8 and ratio[small].max() > 1.25, "aspect ratios should vary over [3/4, 4/3]"
    unclipped = area < 0.7  # sqrt(area * 4/3) < 1: neither side was clipped
    mean_log = torch.log(ratio[unclipped]).mean().item()
    assert abs(mean_log) < 0.016, (
        f"mean log aspect ratio {mean_log:.3f} over {int(unclipped.sum())} crops: draw log(ratio) uniformly in "
        "[log 3/4, log 4/3] (log-uniform), so wide and tall crops are equally likely and log(ratio) averages 0. "
        "a ratio uniform in [3/4, 4/3] averages log(ratio) of about +0.03")
    frac = (sgn < 0).float().mean().item()
    assert abs(frac - 0.5) < 0.04, f"{frac:.3f} of the images were flipped horizontally, expected 0.5"
    assert cx.std() > 0.04, "crop positions don't vary: the crop's centre must be uniform over valid positions"
    assert (out[keep][:, 2] - 0.5).abs().max() < 1e-4, "with strength=0 there must be no colour change"


def test_stage2_grayscale_and_jitter_probabilities(nb):
    torch.manual_seed(3)
    c = torch.tensor([0.3, 0.4, 0.5]).view(1, 3, 1, 1)
    x = c.expand(4000, 3, 16, 16).clone()
    out = nb.SimCLRAugment(16, strength=0.5)(x)
    colours = out.mean(dim=(2, 3))
    assert (out - out.mean(dim=(2, 3), keepdim=True)).abs().max() < 1e-5, (
        "a constant image must stay constant: jitter factors are per image, not per pixel")
    gray = (colours[:, 0] - colours[:, 2]).abs() < 1e-5
    assert abs(gray.float().mean().item() - 0.2) < 0.03, (
        f"{gray.float().mean():.3f} of the images became grayscale, expected 0.2")
    changed = (colours[~gray] - c.view(1, 3)).abs().amax(dim=1) > 1e-5
    assert abs(changed.float().mean().item() - 0.8) < 0.03, (
        f"colour jitter changed {changed.float().mean():.3f} of the (non-gray) images, expected 0.8")


def test_stage2_jitter_factors(nb):
    luma = torch.tensor([0.299, 0.587, 0.114])
    c = torch.tensor([0.3, 0.4, 0.5])
    for strength in (0.25, 0.5):
        torch.manual_seed(4)
        x = c.view(1, 3, 1, 1).expand(4000, 3, 8, 8).clone()
        out = nb.SimCLRAugment(8, strength=strength)(x).mean(dim=(2, 3))
        gray = (out[:, 0] - out[:, 2]).abs() < 1e-5
        changed = ~gray & ((out - c).abs().amax(dim=1) > 1e-5)
        o = out[changed]
        b = (o @ luma) / (c @ luma)  # contrast and saturation keep the luma: what's left is brightness
        s = 0.8 * strength
        assert b.min() > 1 - s - 1e-3 and b.max() < 1 + s + 1e-3, (
            f"strength {strength}: brightness factors span [{b.min():.3f}, {b.max():.3f}], expected within "
            f"[{1 - s:.2f}, {1 + s:.2f}] (0.8 * strength either side of 1)")
        assert b.min() < 1 - 0.8 * s and b.max() > 1 + 0.8 * s, f"strength {strength}: brightness barely varies"
        chroma = (o - (o @ luma)[:, None]).norm(dim=1) / (b * (c - c @ luma).norm())
        assert chroma.min() > (1 - s) ** 2 - 1e-3 and chroma.max() < (1 + s) ** 2 + 1e-3, (
            "the colour's distance from gray should scale by contrast x saturation, each in [1 - 0.8s, 1 + 0.8s]")
        assert chroma.max() > (1 + s) ** 1.6, (
            f"strength {strength}: the colour's distance from gray never grows beyond {chroma.max():.3f}: apply BOTH "
            "contrast (blend with the image's mean gray) and saturation (blend with each pixel's gray)")


# ---------------------------------------------------------------- stage 3: ProjectionHead

def test_stage3_structure(nb):
    torch.manual_seed(0)
    head = nb.ProjectionHead(10, hidden_dim=16, out_dim=6)
    assert isinstance(head, nn.Module)
    out = head(torch.randn(5, 10))
    assert out.shape == (5, 6), f"output should be (B, out_dim) = (5, 6), got {tuple(out.shape)}"
    kinds = [type(m).__name__ for m in head.modules() if not list(m.children())]
    assert kinds == ["Linear", "BatchNorm1d", "ReLU", "Linear"], (
        f"layers {kinds}: expected Linear -> BatchNorm1d -> ReLU -> Linear")
    n = sum(p.numel() for p in head.parameters())
    assert n == 10 * 16 + 16 + 2 * 16 + 16 * 6 + 6, f"{n} parameters: check hidden_dim and out_dim are used"
    d = nb.ProjectionHead(7)
    assert d(torch.randn(3, 7)).shape == (3, 128), "defaults: hidden_dim=512, out_dim=128"
    assert sum(p.numel() for p in d.parameters()) == 7 * 512 + 512 + 1024 + 512 * 128 + 128


# ---------------------------------------------------------------- stage 4: train_simclr

def _images(n, seed, size=8):
    """unlabelled toy images: a bright bar (horizontal or vertical) at a random place, random colour."""
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


def _encoder(seed, size=8):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 16))


def test_stage4_loss_decreases(nb):
    drops = []
    for seed in range(3):
        _, hist = nb.train_simclr(_encoder(seed), 16, _images(256, seed), epochs=6, batch_size=64, temperature=0.2,
                                  lr=3e-3, hidden_dim=32, proj_dim=16, seed=seed)
        losses = hist["loss"]
        assert len(losses) == 24, f"6 epochs of 256 images in batches of 64 should take 24 steps, took {len(losses)}"
        drops.append(np.mean(losses[:3]) - np.mean(losses[-3:]))
    assert np.median(drops) > 0.3, (
        f"the loss fell by {np.round(drops, 3)} over 3 seeds; expected a clear drop (> 0.3). are both views passed "
        "through encoder and head, and is the optimizer stepping both?")


def test_stage4_matches_replay(nb):
    X = _images(40, 5)
    enc = _encoder(2)
    head, h = nb.train_simclr(enc, 16, X, epochs=2, batch_size=16, temperature=0.3, lr=2e-3, strength=0.4,
                              hidden_dim=12, proj_dim=6, seed=7)
    assert len(h["loss"]) == 4, (
        f"40 images in batches of 16: N // bs = 2 FULL batches per epoch (the 8 left over are skipped), so 2 epochs "
        f"are 4 steps; got {len(h['loss'])}")
    ref = _encoder(2)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    rhead = nb.ProjectionHead(16, 12, 6)
    aug = nb.SimCLRAugment(8, strength=0.4)
    opt = torch.optim.Adam(list(ref.parameters()) + list(rhead.parameters()), lr=2e-3)
    Xt = torch.as_tensor(X)
    losses = []
    for _ in range(2):
        ref.train()
        rhead.train()
        perm = rng.permutation(40)
        for s in (0, 16):
            xb = Xt[perm[s:s + 16]]
            loss = nb.info_nce_loss(rhead(ref(aug(xb))), rhead(ref(aug(xb))), temperature=0.3)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    assert np.allclose(h["loss"], losses, rtol=1e-4, atol=1e-5), (
        f"losses {np.round(h['loss'], 4)} differ from the specified loop {np.round(losses, 4)}: torch.manual_seed(seed) "
        "and np.random.default_rng(seed) first, then ProjectionHead(feature_dim, hidden_dim, proj_dim) and "
        "SimCLRAugment(H, strength); a fresh rng.permutation(N) per epoch; view 1 then view 2")


def test_stage4_returns_head_and_trains_encoder(nb):
    enc = _encoder(0)
    enc.eval()  # must be switched to train mode
    w0 = enc[1].weight.detach().clone()
    rm0 = enc[2].running_mean.clone()
    head, hist = nb.train_simclr(enc, 16, _images(40, 0), epochs=1, batch_size=64, hidden_dim=24, proj_dim=8, seed=0)
    assert isinstance(head, nb.ProjectionHead), "return the ProjectionHead you trained (as the first element)"
    assert head(torch.randn(4, 16)).shape == (4, 8), "the head maps feature_dim -> proj_dim"
    assert sum(p.numel() for p in head.parameters()) == 16 * 24 + 24 + 48 + 24 * 8 + 8, "use hidden_dim for the head"
    assert len(hist["loss"]) == 1, "a batch_size larger than the dataset is capped at n: one step per epoch here"
    assert not torch.equal(enc[1].weight, w0), "the encoder must be trained in place"
    assert not torch.equal(enc[2].running_mean, rm0), "train in train mode (BatchNorm statistics should update)"


def test_stage4_two_different_views(nb):
    seen = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _encoder(0)

        def forward(self, x):
            seen.append(x.detach().clone())
            return self.inner(x)

    X = _images(32, 1)
    nb.train_simclr(Spy(), 16, X, epochs=1, batch_size=32, hidden_dim=16, proj_dim=8, seed=0)
    assert len(seen) == 2, f"one step should call the encoder twice (once per view), called it {len(seen)} times"
    assert seen[0].shape == (32, 3, 8, 8)
    assert (seen[0] - seen[1]).abs().max() > 1e-3, "the two views must be two independent augmentations"
    raw = torch.as_tensor(X).flatten(1)
    for v, view in enumerate(seen):
        closest = torch.cdist(view.flatten(1), raw).min(dim=1).values
        assert closest.min() > 1e-4, f"view {v + 1} contains un-augmented images: augment BOTH views"


def test_stage4_epochs_shuffle_the_data(nb):
    rng = np.random.default_rng(7)
    colours = rng.uniform(0.2, 0.8, size=(64, 3)).astype(np.float32)
    X = np.broadcast_to(colours[:, :, None, None], (64, 3, 4, 4)).copy()  # constant images: identifiable after cropping
    luma = colours @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    order = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Sequential(nn.Flatten(), nn.Linear(48, 8))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls % 2 == 1:  # first view of each step
                c = x.mean(dim=(2, 3)).numpy()
                d = np.minimum(np.abs(c[:, None, :] - colours[None]).max(-1), np.abs(c[:, None, :1] - luma[None, :, None]).max(-1))
                order.extend(d.argmin(1).tolist())
            return self.inner(x)

    nb.train_simclr(Spy(), 8, X, epochs=2, batch_size=16, hidden_dim=8, proj_dim=4, strength=0.0, seed=0)
    assert len(order) == 128, f"2 epochs of 64 images in batches of 16 should show 128 images, saw {len(order)}"
    first, second = order[:64], order[64:]
    assert sorted(first) == list(range(64)) and sorted(second) == list(range(64)), (
        "each epoch must visit every image exactly once (a permutation of the dataset)")
    assert first != list(range(64)), "shuffle the images every epoch (a random permutation), don't take them in order"
    assert first != second, "draw a NEW permutation each epoch"


def test_stage4_seed_and_options(nb):
    X = _images(64, 2)
    kw = dict(epochs=2, batch_size=32, hidden_dim=16, proj_dim=8)
    a = nb.train_simclr(_encoder(0), 16, X, seed=5, **kw)[1]["loss"]
    b = nb.train_simclr(_encoder(0), 16, X, seed=5, **kw)[1]["loss"]
    c = nb.train_simclr(_encoder(0), 16, X, seed=6, **kw)[1]["loss"]
    assert np.allclose(a, b), "the same seed must give the same run: seed torch and the shuffling from `seed`"
    assert not np.allclose(a, c), "a different seed should give a different run"
    t = nb.train_simclr(_encoder(0), 16, X, seed=5, temperature=1.0, **kw)[1]["loss"]
    assert abs(t[0] - a[0]) > 1e-3, "temperature is ignored: pass it to info_nce_loss"
    s = nb.train_simclr(_encoder(0), 16, X, seed=5, strength=1.0, **kw)[1]["loss"]
    assert abs(s[0] - a[0]) > 1e-6, "strength is ignored: pass it to SimCLRAugment"
    enc = _encoder(0)
    w0 = enc[1].weight.detach().clone()
    nb.train_simclr(enc, 16, X, seed=5, lr=0.0, **kw)
    assert torch.equal(enc[1].weight, w0), "lr=0 must leave the encoder's weights unchanged: pass lr to Adam"
    torch.manual_seed(5)
    fresh = nb.ProjectionHead(16, 16, 8)  # the head train_simclr builds right after torch.manual_seed(seed)
    head, _ = nb.train_simclr(_encoder(0), 16, X, seed=5, **kw)
    assert not torch.equal(head.net[0].weight if hasattr(head, "net") else next(head.parameters()),
                           fresh.net[0].weight if hasattr(fresh, "net") else next(fresh.parameters())), (
        "the projection head didn't change: optimise the head's parameters together with the encoder's")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device (MPS) to observe `device`")
def test_stage4_device(nb):
    enc = _encoder(0)
    head, _ = nb.train_simclr(enc, 16, _images(32, 3), epochs=1, batch_size=32, hidden_dim=16, proj_dim=8,
                              device="mps")
    assert next(enc.parameters()).device.type == "mps" and next(head.parameters()).device.type == "mps", (
        "move the encoder, the head and every batch to `device`")
