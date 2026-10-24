"""Checks for capstone.py (a class-conditional generator for CIFAR-10).

Each stage's checks pin down one piece on small synthetic images (8 x 8 RGB, two classes whose top or bottom half is
bright), so nothing is downloaded and everything runs on a CPU in seconds. the data conversion against its
definition; the model and schedule against their configs; training for both objectives under `mlfs.testing.bounded`,
for every config option (iterations, batch size, label dropout, learning-rate warmup, EMA, flips, seed); sampling for
classifier-free guidance in one batched call per step, the same number of network calls for both objectives, and
for samples that obey their labels; the evaluation against its definition.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")

TINY = dict(base_channels=8, channel_mults=(1, 2), dropout=0.0, n_iters=200, batch_size=32, lr=3e-3, warmup=10,
            ema_decay=0.9, p_uncond=0.1, hflip=False, n_train_steps=100, img_size=8, clip=True)


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _warm_up_adam():
    """torch's first Adam step runs millions of lines of one-time setup; do it outside `bounded`'s line budget."""
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w])
    w.sum().backward()
    opt.step()


def _toy(n, seed=0):
    """class 0: bright top half; class 1: bright bottom half; values in [-1, 1]."""
    g = _gen(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    x = -0.8 + 0.1 * torch.randn(n, 3, 8, 8, generator=g)
    x[y == 0, :, :4] += 1.4
    x[y == 1, :, 4:] += 1.4
    return x.clamp(-1, 1), y


def _cfg(**kw):
    return {**TINY, **kw}


_TRAINED = {}


def _trained(nb, obj):
    """train the tiny model once per objective (shared by the stage 3 and stage 4 behaviour checks)."""
    key = (nb.train_generator, nb.build_model, obj)
    if key not in _TRAINED:
        X, y = _toy(600, seed=4)
        _TRAINED[key] = nb.train_generator(obj, nb.build_model(TINY, seed=0), X, y, _cfg(n_iters=160), seed=0)
    return _TRAINED[key]


class _Rec(torch.nn.Module):
    """a recording stand-in for the U-Net: model(x, t, y) -> 0.1 x (+ a parameter so Adam has work)."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.1))
        self.calls = []

    def forward(self, x, t, y=None):
        self.calls.append((x.shape[0], t.detach().clone(), None if y is None else y.clone(), self.training))
        assert len(self.calls) <= 20_000, "the network was called more than 20,000 times: a loop never ends"
        return self.w * x


# ---------------------------------------------------------------- stage 1: data


def test_stage1_prepare_and_back(nb):
    X = np.random.default_rng(0).integers(0, 256, (5, 32, 32, 3)).astype(np.uint8)
    y = np.array([3, 1, 4, 1, 5])
    x, yt = nb.prepare_cifar(X, y)
    assert x.shape == (5, 3, 32, 32) and x.dtype == torch.float32, "NHWC uint8 -> NCHW float32"
    assert math.isclose(x[2, 1, 7, 9].item(), X[2, 7, 9, 1] / 127.5 - 1, abs_tol=1e-6), "x = X / 127.5 - 1, channel-first"
    assert yt.dtype == torch.int64 and yt.tolist() == [3, 1, 4, 1, 5], "labels as an int64 tensor"
    assert x.min() >= -1 and x.max() <= 1
    u = nb.to_unit(x)
    assert torch.allclose(u, torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2) / 255, atol=1e-6), "to_unit: (x + 1)/2"
    assert torch.all(nb.to_unit(torch.tensor([-3.0, 3.0])) == torch.tensor([0.0, 1.0])), "to_unit clamps to [0, 1]"


def test_stage1_random_hflip(nb):
    x = torch.arange(6 * 2 * 3 * 4, dtype=torch.float32).reshape(6, 2, 3, 4)
    got = nb.random_hflip(x, generator=_gen(1))
    mask = torch.rand(6, generator=_gen(1)) < 0.5
    want = torch.where(mask.view(-1, 1, 1, 1), x.flip(-1), x)
    assert torch.equal(got, want), "flip image i left-right where torch.rand(B, generator)[i] < 0.5"
    assert torch.equal(x, torch.arange(6 * 2 * 3 * 4, dtype=torch.float32).reshape(6, 2, 3, 4)), "return a new tensor"
    many = nb.random_hflip(torch.arange(4.0).view(1, 1, 1, 4).repeat(2000, 1, 1, 1), generator=_gen(2))
    assert 0.45 < (many[:, 0, 0, 0] == 3).float().mean() < 0.55, "each image flipped with probability 1/2"


# ---------------------------------------------------------------- stage 2: model and schedule


def test_stage2_model_and_schedule(nb):
    from mlfs.diffusion import DiffusionSchedule, TimeUNet

    m = nb.build_model(TINY, seed=3)
    assert isinstance(m, TimeUNet), "the network is your TimeUNet"
    tables = [mod for mod in m.modules() if isinstance(mod, torch.nn.Embedding)]
    assert len(tables) == 1 and tables[0].num_embeddings == 11, "class-conditional with 10 classes (+ the null label)"
    m2 = nb.build_model(TINY, seed=3)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters())), "the seed fixes the initialization"
    m3 = nb.build_model(TINY, seed=4)
    assert not all(torch.equal(a, b) for a, b in zip(m.parameters(), m3.parameters())), "a different seed, another init"
    wide = nb.build_model(_cfg(base_channels=16))
    assert sum(p.numel() for p in wide.parameters()) > sum(p.numel() for p in m.parameters()), "use base_channels"
    deep = nb.build_model(_cfg(channel_mults=(1, 2, 2)))
    with pytest.raises(ValueError):
        deep(torch.zeros(1, 3, 6, 6), torch.zeros(1))  # three levels need H, W divisible by 4: channel_mults is used
    drop = nb.build_model(_cfg(dropout=0.5)).train()
    with torch.no_grad():
        for p in drop.parameters():
            p.normal_(0, 0.1, generator=_gen(5))
    xx, tt = torch.randn(2, 3, 8, 8), torch.rand(2)
    assert not torch.allclose(drop(xx, tt), drop(xx, tt)), "dropout from the config: train-mode outputs are random"
    s = nb.make_schedule(_cfg(n_train_steps=200))
    assert isinstance(s, DiffusionSchedule) and s.n_steps == 200
    assert math.isclose(s.betas[0].item(), 5e-4, rel_tol=1e-4) and math.isclose(s.betas[-1].item(), 0.1, rel_tol=1e-4), (
        "betas rescaled by 1000 / n_train_steps: from 1e-4 * 1000/n to 0.02 * 1000/n")


# ---------------------------------------------------------------- stage 3: training


def test_stage3_training_options(nb):
    X, y = _toy(200)
    for obj in ("ddpm", "flow"):
        m = _Rec().eval()
        out = nb.train_generator(obj, m, X, y, _cfg(n_iters=6, batch_size=10, warmup=4, p_uncond=0.5), seed=0)
        assert len(out["losses"]) == 6 and all(isinstance(v, float) for v in out["losses"]), "one float loss per step"
        assert [c[0] for c in m.calls] == [10] * 6 and all(c[3] for c in m.calls), "batch_size rows per step, train mode"
        labels = torch.cat([c[2] for c in m.calls])
        assert 0.25 < (labels == 10).float().mean() < 0.75, "p_uncond = 0.5: about half the labels become the null 10"
        assert np.allclose(out["lrs"], [3e-3 * k / 4 for k in (1, 2, 3, 4)] + [3e-3, 3e-3]), (
            f"linear warmup: lr * (step + 1) / warmup for the first `warmup` steps, then lr; got {out['lrs']}")
        assert isinstance(out["seconds"], float) and out["seconds"] >= 0
    ts = torch.cat([c[1] for c in m.calls]) * 100
    assert not torch.allclose(ts, ts.round(), atol=1e-3), "'flow' samples continuous times"
    m = _Rec()
    nb.train_generator("ddpm", m, X, y, _cfg(n_iters=3, batch_size=8), seed=0)
    ts = torch.cat([c[1] for c in m.calls]) * 100
    assert torch.allclose(ts, ts.round(), atol=1e-4), "'ddpm' uses ddpm_loss on make_schedule(config): t = step / n_steps"
    with pytest.raises(ValueError):
        nb.train_generator("gan", _Rec(), X, y, TINY)


def test_stage3_ema_seed_and_flips(nb):
    X, y = _toy(100)
    a, b = _Rec(), _Rec()
    ra = nb.train_generator("flow", a, X, y, _cfg(n_iters=5, batch_size=8), seed=7)
    rb = nb.train_generator("flow", b, X, y, _cfg(n_iters=5, batch_size=8), seed=7)
    assert ra["losses"] == rb["losses"], "all randomness from a generator seeded with `seed`: reproducible"
    rc = nb.train_generator("flow", _Rec(), X, y, _cfg(n_iters=5, batch_size=8), seed=8)
    assert rc["losses"] != ra["losses"], "a different seed, a different run"
    m0 = _Rec()
    r0 = nb.train_generator("flow", m0, X, y, _cfg(n_iters=5, batch_size=8, ema_decay=0.0), seed=0)
    assert torch.allclose(r0["ema"].w, m0.w), "ema_decay = 0: the EMA copy equals the trained model"
    assert r0["ema"] is not m0 and not r0["ema"].training, "a separate EMA copy, returned in eval mode"
    m1 = _Rec()
    r1 = nb.train_generator("flow", m1, X, y, _cfg(n_iters=1, batch_size=8, ema_decay=0.99), seed=0)
    assert torch.allclose(r1["ema"].w, 0.1 * 0.1 + 0.9 * m1.w, atol=1e-6), (
        "after the first step the warmed-up decay is min(ema_decay, 1/10) = 0.1: ema = 0.1 * initial + 0.9 * trained")
    m2 = _Rec()
    r2 = nb.train_generator("flow", m2, X, y, _cfg(n_iters=1, batch_size=8, ema_decay=0.05), seed=0)
    assert torch.allclose(r2["ema"].w, 0.05 * 0.1 + 0.95 * m2.w, atol=1e-6), "use config['ema_decay'] when it is smaller"
    rf = nb.train_generator("flow", _Rec(), X, y, _cfg(n_iters=5, batch_size=8, hflip=True), seed=0)
    assert rf["losses"] != r0["losses"], "hflip=True must change training (random_hflip on each batch)"


def test_stage3_both_objectives_learn(nb):
    X, y = _toy(400)
    _warm_up_adam()
    bounded(nb.train_generator, "flow", _Rec(), X, y, _cfg(n_iters=2, batch_size=4), budget=1_000_000,
            what="train_generator", hint="loop over n_iters steps")
    for obj, floor in (("ddpm", 0.3), ("flow", 1.0)):
        out = _trained(nb, obj)
        first, last = np.mean(out["losses"][:10]), np.mean(out["losses"][-20:])
        assert last < 0.7 * first, f"{obj}: the loss should fall by 30% in 80 steps (from {first:.3f} to {last:.3f})"


# ---------------------------------------------------------------- stage 4: sampling


def test_stage4_guidance_calls_and_equal_compute(nb):
    labels = torch.tensor([0, 1, 1, 0, 1])
    for obj, increasing in (("ddpm", False), ("flow", True)):
        m = _Rec().eval()
        img = nb.generate(obj, m, labels, TINY, guidance_scale=2.0, n_steps=6, generator=_gen(0), batch_size=3)
        assert img.shape == (5, 3, 8, 8) and img.min() >= 0 and img.max() <= 1 and img.device.type == "cpu", (
            "images (N, 3, img_size, img_size) in [0, 1] on the CPU")
        sizes = [c[0] for c in m.calls]
        assert sizes == [6] * 6 + [4] * 6, (
            f"{obj}: n_steps = 6 network calls per chunk of batch_size = 3 labels, each on 2 x chunk rows (classifier-free "
            f"guidance in one batched call); got call sizes {sizes}")
        assert torch.equal(m.calls[0][2], torch.tensor([0, 1, 1, 10, 10, 10])), "labels [y, null] with null = 10"
        t0 = [c[1][0].item() for c in m.calls[:6]]
        assert (np.diff(t0) > 0).all() == increasing, (
            f"{obj}: {'flow matching integrates t from 0 up' if increasing else 'DDIM visits t from high to low'}; "
            f"times {np.round(t0, 3).tolist()}")
    with pytest.raises(ValueError):
        nb.generate("gan", _Rec(), labels, TINY)


def test_stage4_guidance_scale_and_reproducibility(nb):
    torch.manual_seed(0)
    m = nb.build_model(TINY, seed=1).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2, generator=_gen(2))
    labels = torch.tensor([0, 1, 2, 3])
    noclip = _cfg(clip=False)
    a = nb.generate("flow", m, labels, noclip, guidance_scale=1.0, n_steps=4, generator=_gen(3))
    b = nb.generate("flow", m, labels, noclip, guidance_scale=1.0, n_steps=4, generator=_gen(3))
    c = nb.generate("flow", m, labels, noclip, guidance_scale=4.0, n_steps=4, generator=_gen(3))
    assert torch.equal(a, b), "the same generator gives the same samples"
    assert not torch.allclose(a, c, atol=1e-4), "guidance_scale must change the samples"
    from mlfs.diffusion import sample_flow

    plain = ((sample_flow(m, (4, 3, 8, 8), n_steps=4, y=labels, generator=_gen(3)) + 1) / 2).clamp(0, 1)
    assert torch.allclose(a, plain, atol=1e-4), (
        "w = 1 without clipping is the plain conditional model: same as sample_flow(model, y=labels)")
    from mlfs.diffusion import ddim_sample

    e = nb.generate("ddpm", m, labels, noclip, guidance_scale=1.0, n_steps=4, generator=_gen(3))
    ref = ((ddim_sample(m, nb.make_schedule(noclip), (4, 3, 8, 8), n_steps=4, y=labels, generator=_gen(3)) + 1) / 2)
    assert torch.allclose(e, ref.clamp(0, 1), atol=1e-4), (
        "'ddpm' samples with deterministic DDIM (eta = 0) on make_schedule(config): at w = 1 without clipping it must "
        "equal ddim_sample(model, schedule, shape, n_steps=n_steps, y=labels)")
    d = nb.generate("flow", m, labels, TINY, guidance_scale=1.0, n_steps=4, generator=_gen(3))
    assert not torch.allclose(a, d, atol=1e-4), "config['clip'] must wrap the guided model in ClipPrediction"


def test_stage4_clip_prediction_by_hand(nb):
    s = nb.make_schedule(TINY)
    toy = lambda x, t, y=None: 3.0 * x + t.view(-1, 1, 1, 1)
    x = torch.randn(3, 1, 2, 2, generator=_gen(8)) * 2
    t = torch.tensor([0.0, 0.5, 0.555])
    w = nb.ClipPrediction(toy, "ddpm", s)
    from mlfs.diffusion import alpha_bar_at

    ab = alpha_bar_at(s, t).view(-1, 1, 1, 1)  # continuous time: 0.555 lies between grid points
    eps = toy(x, t)
    x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
    assert torch.allclose(w(x, t), (x - ab.sqrt() * x0) / (1 - ab).sqrt(), atol=1e-4), (
        "ddpm: clip x0 = (x - sqrt(1 - ab) eps)/sqrt(ab) to [-1, 1] and return the eps that implies it (ab from "
        "alpha_bar_at(schedule, t))")
    f = nb.ClipPrediction(toy, "flow")
    v = toy(x, t)
    x1 = (x + (1 - t.view(-1, 1, 1, 1)) * v).clamp(-1, 1)
    assert torch.allclose(f(x, t), (x1 - x) / (1 - t.view(-1, 1, 1, 1)), atol=1e-5), (
        "flow: clip x1 = x + (1 - t) v to [-1, 1] and return the velocity (x1 - x)/(1 - t) that implies it")
    m = _Rec()
    nb.ClipPrediction(m, "flow").eval()
    assert not m.training, "keep the model as a submodule (eval mode must reach it)"


def test_stage4_samples_obey_labels(nb):
    cfg = TINY
    for obj, floor in (("ddpm", 0.3), ("flow", 1.0)):
        out = _trained(nb, obj)
        labels = torch.tensor([0] * 8 + [1] * 8)
        img = nb.generate(obj, out["ema"], labels, cfg, guidance_scale=2.0, n_steps=10, generator=_gen(5))
        top = img[:, :, :4].mean((1, 2, 3)) - img[:, :, 4:].mean((1, 2, 3))
        # reference over 10 trainings x 20 sample seeds: separation >= 0.35 (ddpm), 1.24 (flow); a sampler that
        # ignores the labels beats the floor 4/200 (ddpm) and 0/200 (flow, max 0.94) times
        sep = top[:8].mean() - top[8:].mean()
        assert sep > floor and top[:8].mean() > 0 and top[8:].mean() < 0, (
            f"{obj}: class 0 samples should be bright on top and class 1 at the bottom (top-minus-bottom brightness "
            f"{top[:8].mean():.2f} and {top[8:].mean():.2f}, a difference of {sep:.2f}, expected above {floor}); train on the "
            "labels and pass them to the guided model")


# ---------------------------------------------------------------- stage 5: evaluation


class _FakeFeat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 4)
        torch.nn.init.eye_(self.lin.weight[:3])
        self.modes = []

    def forward(self, x):
        self.modes.append(self.training)
        return torch.cat([x.mean((2, 3)), x.mean((1, 2, 3))[:, None] * 0], 1) * torch.tensor([1.0, -1.0, 0.5, 0.0])

    def embed(self, x):
        return x.mean((2, 3)).double().numpy()


def test_stage5_evaluate(nb):
    from mlfs.generative import fid_from_features

    g = _gen(6)
    imgs = torch.rand(40, 3, 4, 4, generator=g)
    labels = torch.randint(0, 3, (40,), generator=g)
    real = np.random.default_rng(7).random((50, 3))
    f = _FakeFeat().train()
    got = nb.evaluate(imgs, labels, f, real)
    assert math.isclose(got["fid"], fid_from_features(real, imgs.mean((2, 3)).double().numpy()), rel_tol=1e-6), (
        "fid = fid_from_features(real_feats, featnet.embed(images))")
    preds = f(imgs).argmax(1)
    assert math.isclose(got["accuracy"], (preds == labels).float().mean().item()), (
        "accuracy = mean(featnet(images).argmax(1) == labels)")
    assert f.training and f.modes[0] is False, "classify in eval mode, then restore the feature network's mode"
