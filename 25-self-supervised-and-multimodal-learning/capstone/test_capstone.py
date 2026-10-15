"""Checks for capstone.py (label efficiency).

Label subsets are checked against the exact specified draws; pretraining and supervised training
against exact replays of the specified loops (optimizer, schedule, shuffling, RNG order) on tiny
synthetic data, plus learning thresholds over several seeds; the summary by hand.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.ssl import ProjectionHead, SimCLRAugment, info_nce_loss
from mlfs.testing import bounded
from mlfs.train import CosineSchedule

torch.set_num_threads(1)  # tiny networks: one thread is fastest, and avoids oversubscribing a busy machine


# ---------------------------------------------------------------- stage 1: label_subset

def test_stage1_exact_and_balanced(nb):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=503) * 2 + 1  # classes 1, 3, 5, 7, 9
    for frac in (0.01, 0.1, 0.37, 1.0):
        got = nb.label_subset(labels, frac, seed=3)
        r = np.random.default_rng(3)
        want = []
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            want.append(r.permutation(idx)[:max(1, math.floor(frac * len(idx) + 0.5))])
        want = np.sort(np.concatenate(want))
        assert isinstance(got, np.ndarray) and got.dtype == np.int64, "return an int64 ndarray of indices"
        assert np.array_equal(got, want), (
            f"fraction {frac}: wrong indices. for each class in sorted order, take the first max(1, round(fraction * "
            "n_c)) entries of rng.permutation(that class's indices), with ONE rng = np.random.default_rng(seed); "
            "return them sorted")
    counts = np.bincount(labels[nb.label_subset(labels, 0.01, seed=0)])[1::2]
    assert (counts == 1).all(), f"1% of ~100 per class rounds to 1 each, got {counts}"
    rare = np.array([0] * 20 + [1] * 300)
    counts = np.bincount(rare[nb.label_subset(rare, 0.01, seed=0)])
    assert counts.tolist() == [1, 3], (
        f"1% of a class of 20 rounds to 0: keep at least one example per class (got {counts.tolist()}, want [1, 3])")
    assert np.array_equal(nb.label_subset(labels, 1.0), np.arange(503)), "fraction 1.0 selects everything"


def test_stage1_nested_and_seeded(nb):
    labels = np.repeat(np.arange(10), 100)
    small, mid, big = (set(nb.label_subset(labels, f, seed=7).tolist()) for f in (0.01, 0.1, 0.5))
    assert small <= mid <= big, "subsets for the same seed must be nested (1% inside 10% inside 50%)"
    assert len(small) == 10 and len(mid) == 100 and len(big) == 500
    a, b = nb.label_subset(labels, 0.1, seed=0), nb.label_subset(labels, 0.1, seed=1)
    assert not np.array_equal(a, b), "different seeds must give different subsets"
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            nb.label_subset(labels, bad)


# ---------------------------------------------------------------- stage 2: pretrain_simclr

def _images(n, seed, size=8):
    """uint8 NCHW toy images: two classes of bars (horizontal / vertical) at random places, random colours."""
    rng = np.random.default_rng(seed)
    X = (25 * rng.random((n, 3, size, size))).astype(np.uint8)
    y = rng.integers(0, 2, size=n)
    for i in range(n):
        r = rng.integers(1, size - 1)
        colour = (100 + 155 * rng.random(3)).astype(np.uint8)[:, None]
        if y[i]:
            X[i, :, r, :] = colour
        else:
            X[i, :, :, r] = colour
    return X, y


def _encoder(seed, size=8, d=16):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * size * size, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, d))


def test_stage2_matches_replay(nb):
    X, _ = _images(40, 0)
    enc = _encoder(1)
    enc.eval()  # must be switched to train mode
    head, h = nb.pretrain_simclr(enc, 16, X, epochs=3, batch_size=16, lr=3e-3, temperature=0.3, strength=0.4,
                                 warmup_epochs=1, weight_decay=0.01, seed=5)
    assert len(h["loss"]) == 6 and len(h["lr"]) == 6, "3 epochs x (40 // 16) full batches = 6 steps"
    ref = _encoder(1)
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    rhead = ProjectionHead(16)
    aug = SimCLRAugment(8, strength=0.4)
    opt = torch.optim.AdamW(list(ref.parameters()) + list(rhead.parameters()), lr=3e-3, weight_decay=0.01)
    sched = CosineSchedule(opt, 6, warmup_steps=2)
    Xt = torch.as_tensor(X)
    losses, lrs = [], []
    for _ in range(3):
        ref.train()
        rhead.train()
        perm = rng.permutation(40)
        for s in (0, 16):
            xb = Xt[perm[s:s + 16]].float() / 255
            loss = info_nce_loss(rhead(ref(aug(xb))), rhead(ref(aug(xb))), temperature=0.3)
            lrs.append(opt.param_groups[0]["lr"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
    assert np.allclose(h["lr"], lrs), (
        f"learning rates {np.round(h['lr'], 5)}, expected {np.round(lrs, 5)}: CosineSchedule(optimizer, epochs * "
        "steps_per_epoch, warmup_steps=warmup_epochs * steps_per_epoch), stepped after every optimizer step; record "
        "the lr each step USED")
    assert np.allclose(h["loss"], losses, rtol=1e-4, atol=1e-5), (
        f"losses {np.round(h['loss'], 4)} differ from the specified loop {np.round(losses, 4)}. check the order: "
        "torch.manual_seed(seed), np.random.default_rng(seed), ProjectionHead(feature_dim), SimCLRAugment(H, "
        "strength), AdamW(lr, weight_decay); uint8 images divided by 255; view 1 then view 2")
    for a, b in zip(enc.parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4, msg="the encoder must be trained in place")
    assert isinstance(head, ProjectionHead)


def test_stage2_float_input_and_learning(nb):
    X, _ = _images(32, 1)
    a = nb.pretrain_simclr(_encoder(0), 16, X, epochs=1, batch_size=16, seed=0)[1]["loss"]
    b = nb.pretrain_simclr(_encoder(0), 16, X.astype(np.float32) / 255, epochs=1, batch_size=16, seed=0)[1]["loss"]
    assert np.allclose(a, b, rtol=1e-5), "uint8 images (0..255) and the same images as floats in [0, 1] must train identically"
    drops = []
    for seed in range(3):
        X, _ = _images(256, seed)
        h = nb.pretrain_simclr(_encoder(seed), 16, X, epochs=6, batch_size=64, lr=3e-3, seed=seed)[1]
        drops.append(np.mean(h["loss"][:3]) - np.mean(h["loss"][-3:]))
    assert np.median(drops) > 0.3, f"the SimCLR loss fell by {np.round(drops, 3)} (3 seeds); expected > 0.3"


# ---------------------------------------------------------------- stage 3: train_classifier, accuracy

def test_stage3_matches_replay(nb):
    X, y = _images(36, 2)
    enc = _encoder(3)
    model = nb.train_classifier(enc, 16, 2, X, y, steps=5, batch_size=16, lr=0.1, weight_decay=1e-3, augment=False,
                                seed=4)
    assert isinstance(model, nn.Sequential) and model[0] is enc and isinstance(model[1], nn.Linear), (
        "return nn.Sequential(encoder, nn.Linear(feature_dim, n_classes))")
    assert not model.training, "return the model in eval mode"
    ref_enc = _encoder(3)
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    ref = nn.Sequential(ref_enc, nn.Linear(16, 2))
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-3)
    sched = CosineSchedule(opt, 5)
    ref.train()
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    done = 0
    while done < 5:
        perm = rng.permutation(36)
        for s in (0, 16):
            if done == 5:
                break
            idx = perm[s:s + 16]
            loss = F.cross_entropy(ref(Xt[idx].float() / 255), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            done += 1
    for a, b in zip(model.parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4, msg=(
            "the trained weights differ from the specified loop: torch.manual_seed(seed) then the head; SGD(lr, "
            "momentum=0.9, weight_decay); CosineSchedule(optimizer, steps); a fresh permutation from "
            "np.random.default_rng(seed) whenever an epoch (n // bs full batches) runs out; stop after exactly `steps`"))


def test_stage3_steps_and_augmentation(nb):
    seen = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _encoder(0)

        def forward(self, x):
            seen.append(x.detach().clone())
            return self.inner(x)

    X, y = _images(10, 3)
    bounded(nb.train_classifier, Spy(), 16, 2, X, y, steps=7, batch_size=64, augment=True, seed=0,
            what="train_classifier", budget=3_000_000,
            hint="with batch_size larger than the data, use bs = min(batch_size, n), or no batch ever runs.")
    assert len(seen) == 7, f"exactly `steps` = 7 training steps, the encoder was called {len(seen)} times"
    assert all(s.shape == (10, 3, 8, 8) for s in seen), "batch_size larger than the data: each batch is all n images"
    raw = torch.as_tensor(X).float() / 255
    assert all((s.sort(dim=0).values - raw.sort(dim=0).values).abs().amax() > 1e-6 for s in seen), (
        "with augment=True the images must be flipped/cropped (use flip_crop)")
    seen.clear()
    X2, y2 = _images(36, 4)
    nb.train_classifier(Spy(), 16, 2, X2, y2, steps=5, batch_size=16, seed=0)
    assert len(seen) == 5, (
        f"36 images in batches of 16 (2 full batches per epoch), steps=5: exactly 5 steps, took {len(seen)}. stop in the "
        "middle of an epoch when the step budget runs out")
    seen.clear()
    nb.train_classifier(Spy(), 16, 2, X, y, steps=1, batch_size=64, augment=False, seed=0)
    assert torch.allclose(seen[0].sort(dim=0).values, raw.sort(dim=0).values), "augment=False must use the raw images"


def test_stage3_learns(nb):
    accs = []
    for seed in range(3):
        X, y = _images(300, 10 + seed)
        Xte, yte = _images(200, 20 + seed)
        model = nb.train_classifier(_encoder(seed), 16, 2, X, y, steps=60, batch_size=32, lr=0.05, augment=False,
                                    seed=seed)
        accs.append(nb.accuracy(model, Xte, yte))
    assert np.median(accs) > 0.9, f"test accuracies over 3 seeds {np.round(accs, 3)}; expected > 0.9"


def test_stage3_accuracy(nb):
    class Fixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(2)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            assert not self.training and not torch.is_grad_enabled(), "evaluate in eval mode under torch.no_grad()"
            score = x.flatten(1).mean(1)
            return torch.stack([score, 0.5 * torch.ones_like(score)], dim=1)  # class 0 iff mean > 0.5

    X = np.zeros((7, 1, 2, 2), dtype=np.uint8)
    X[:3] = 255
    y = np.array([0, 0, 1, 1, 1, 1, 0])
    m = Fixed()
    m.train()
    got = nb.accuracy(m, X, y, batch_size=3)
    assert abs(got - 5 / 7) < 1e-9, f"accuracy {got}, expected 5/7 (uint8 images are divided by 255)"
    assert m.calls == 3, f"batch_size=3 over 7 images is 3 batches, got {m.calls}"
    assert m.training, "restore the model's mode afterwards"
    assert abs(nb.accuracy(Fixed(), X.astype(np.float32) / 255, y) - 5 / 7) < 1e-9, "float images in [0, 1] work too"


# ---------------------------------------------------------------- stage 4: summarize

def test_stage4_summary(nb):
    recs = [{"method": "probe", "fraction": 0.1, "accuracy": a} for a in (0.5, 0.6, 0.7)]
    recs += [{"method": "probe", "fraction": 1.0, "accuracy": 0.8},
             {"method": "scratch", "fraction": 0.1, "accuracy": 0.4}, {"method": "scratch", "fraction": 0.1, "accuracy": 0.44}]
    s = nb.summarize(recs)
    assert set(s) == {("probe", 0.1), ("probe", 1.0), ("scratch", 0.1)}, "one entry per (method, fraction)"
    assert abs(s[("probe", 0.1)]["mean"] - 0.6) < 1e-12 and s[("probe", 0.1)]["n"] == 3
    assert abs(s[("probe", 0.1)]["std"] - 0.1) < 1e-12, "std is the sample standard deviation (ddof=1)"
    assert abs(s[("scratch", 0.1)]["std"] - np.std([0.4, 0.44], ddof=1)) < 1e-12
    assert s[("probe", 1.0)]["n"] == 1 and math.isnan(s[("probe", 1.0)]["std"]), "a single run has std nan"
