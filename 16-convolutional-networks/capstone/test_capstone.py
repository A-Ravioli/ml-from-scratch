"""Checks for capstone.py (CIFAR-10 with your ResNet).

Each stage's checks pin down one part of the pipeline: normalization against NumPy; augmentation by
exhaustive search over the crops it could have produced and the binomial spread of its flips; the
loader's permutations and ordering; the optimizer's parameter groups; the training run on small
synthetic colour images (under `mlfs.testing.bounded`, CPU, seconds), including its learning-rate
schedule; and the ablation table's intervals against their definitions. Data are synthetic and
generated here; nothing is downloaded.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs.testing import bounded  # noqa: E402

TINY = dict(block_counts=(1, 1), base_width=8, residual=True, epochs=4, batch_size=32, lr=0.05, momentum=0.9,
            nesterov=True, weight_decay=5e-4, label_smoothing=0.1, warmup_epochs=1, schedule="cosine", augment=True)


def _images(n, seed, size=16, k=4):
    """uint8 NHWC colour images: k smooth class templates plus noise (the templates don't depend on seed)."""
    t = np.random.default_rng(1234).normal(size=(k, size, size, 3))
    for axis in (1, 2):
        for _ in range(3):
            t = (t + np.roll(t, 1, axis=axis) + np.roll(t, -1, axis=axis)) / 3
    t /= t.std()
    r = np.random.default_rng(seed)
    y = r.integers(0, k, n)
    x = np.clip(127.5 + 50 * (t[y] + 0.6 * r.normal(size=(n, size, size, 3))), 0, 255).astype(np.uint8)
    return x, y.astype(np.int64)


def _data(nb, n_train=384, n_test=200, seed=0):
    Xtr, ytr = _images(n_train, seed)
    Xte, yte = _images(n_test, seed + 100)
    mean, std = nb.channel_stats(Xtr)
    return {"X_train": nb.to_tensor(Xtr, mean, std), "y_train": ytr, "X_test": nb.to_tensor(Xte, mean, std),
            "y_test": yte}


def _bounded(fn, *args, **kwargs):
    return bounded(lambda: fn(*args, **kwargs), budget=30_000_000, what=getattr(fn, "__name__", "the pipeline"),
                   hint="every loop should run a fixed number of times (epochs, batches).")


# ---------------------------------------------------------------- stage 1: normalization


def test_stage1_channel_stats(nb):
    X, _ = _images(50, 1)
    mean, std = nb.channel_stats(X)
    want_m = X.reshape(-1, 3).mean(axis=0) / 255
    want_s = (X.reshape(-1, 3) / 255).std(axis=0)
    assert np.shape(mean) == (3,) and np.shape(std) == (3,), "one mean and one std per colour channel"
    assert np.allclose(mean, want_m, atol=1e-5) and np.allclose(std, want_s, atol=1e-5), (
        f"per-channel mean/std of the pixels on the [0, 1] scale, over all images and positions: expected "
        f"{np.round(want_m, 4)}, {np.round(want_s, 4)}; got {np.round(mean, 4)}, {np.round(std, 4)}")
    assert np.asarray(mean).dtype == np.float32 and np.asarray(std).dtype == np.float32


def test_stage1_to_tensor(nb):
    X, _ = _images(20, 2)
    mean, std = nb.channel_stats(X)
    t = nb.to_tensor(X, mean, std)
    assert isinstance(t, torch.Tensor) and t.dtype == torch.float32, "a float32 torch tensor"
    assert tuple(t.shape) == (20, 3, 16, 16), f"NHWC (20, 16, 16, 3) -> NCHW (20, 3, 16, 16); got {tuple(t.shape)}"
    want = (X[3, 5, 7, 2] / 255 - mean[2]) / std[2]
    assert math.isclose(t[3, 2, 5, 7].item(), float(want), rel_tol=1e-5, abs_tol=1e-5), (
        "pixel (n=3, row 5, col 7, channel 2) must land at t[3, 2, 5, 7], scaled to [0, 1] and standardized with "
        "channel 2's mean and std")
    assert torch.allclose(t.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4), "standardized with its own stats: mean 0"
    assert torch.allclose(t.std(dim=(0, 2, 3), unbiased=False), torch.ones(3), atol=1e-3)


# ---------------------------------------------------------------- stage 2: augmentation


def _crop_offsets(orig, out, p):
    """all (dy, dx) such that out equals the zero-padded orig cropped at (dy, dx)."""
    padded = torch.nn.functional.pad(orig, (p, p, p, p))
    h, w = orig.shape[-2:]
    return [(dy, dx) for dy in range(2 * p + 1) for dx in range(2 * p + 1)
            if torch.equal(padded[..., dy:dy + h, dx:dx + w], out)]


def test_stage2_identity_and_shapes(nb):
    x = torch.randn(6, 3, 8, 8)
    out = nb.augment_batch(x, padding=0, flip=False, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out, x), "padding=0 and flip=False: nothing changes"
    out = nb.augment_batch(x, generator=torch.Generator().manual_seed(0))
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device


def test_stage2_crops_are_per_image_translations(nb):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 3, 8, 8, generator=torch.Generator().manual_seed(5))
    out = nb.augment_batch(x, padding=2, flip=False, generator=g)
    offsets = []
    for i in range(len(x)):
        found = _crop_offsets(x[i], out[i], 2)
        assert found, (
            f"image {i} is not a crop of the zero-padded image at any offset in [0, 2p]: pad with zeros by `padding` "
            f"on every side, then cut an H x W window")
        offsets.append(found[0])
    assert len(set(offsets)) >= 10, (
        f"each image gets its own random offset (40 images, 25 possible offsets); only {len(set(offsets))} distinct "
        f"offsets appeared. draw one (dy, dx) per image, not one per batch")
    assert {o[0] for o in offsets} >= {0, 4} and {o[1] for o in offsets} >= {0, 4}, (
        "offsets range over the whole [0, 2 * padding] in both directions (shifts of -p to +p)")


def test_stage2_flips(nb):
    g = torch.Generator().manual_seed(2)
    x = torch.randn(400, 1, 4, 4, generator=torch.Generator().manual_seed(6))
    out = nb.augment_batch(x, padding=0, flip=True, generator=g)
    same = torch.all((out == x).flatten(1), dim=1)
    flipped = torch.all((out == x.flip(3)).flatten(1), dim=1)
    assert bool(torch.all(same | flipped)), "flip=True: every image is either unchanged or mirrored left-right (dim 3)"
    k = int(flipped.sum())
    assert 160 <= k <= 240, f"each image is flipped with probability 0.5: 400 images should give 160-240 flips; got {k}"


def test_stage2_seeded(nb):
    x = torch.randn(16, 3, 8, 8)
    a = nb.augment_batch(x, generator=torch.Generator().manual_seed(3))
    b = nb.augment_batch(x, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "draw every random number from `generator`, so a seeded generator reproduces the batch"


# ---------------------------------------------------------------- stage 3: loader, model, optimizer


def test_stage3_loader(nb):
    X = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1).expand(10, 1, 4, 4).contiguous()
    y = np.arange(10)
    ev = nb.BatchLoader(X, y, batch_size=4, train=False)
    assert len(ev) == 3, f"ceil(10 / 4) = 3 batches; got {len(ev)}"
    got = [yb.tolist() for _, yb in ev]
    assert got == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]], f"train=False: the original order, no augmentation; got {got}"
    assert all(xb.dtype == torch.float32 and yb.dtype == torch.int64 for xb, yb in ev)
    assert torch.equal(torch.cat([xb for xb, _ in ev]), X), (
        "train=False never augments (even with augment=True): evaluation sees the images as they are")
    tr = nb.BatchLoader(X, y, batch_size=4, train=True, augment=False, seed=0)
    e1 = torch.cat([yb for _, yb in tr]).tolist()
    e2 = torch.cat([yb for _, yb in tr]).tolist()
    assert sorted(e1) == list(range(10)) and sorted(e2) == list(range(10)), "every example exactly once per epoch"
    assert e1 != list(range(10)) and e1 != e2, "train=True: a fresh random permutation every epoch"
    assert all(torch.equal(xb[:, 0, 0, 0].long(), yb) for xb, yb in tr), "images and labels stay paired"
    tr2 = nb.BatchLoader(X, y, batch_size=4, train=True, augment=False, seed=0)
    assert torch.cat([yb for _, yb in tr2]).tolist() == e1, "the same seed gives the same order"


def test_stage3_model_and_optimizer(nb):
    m = nb.build_model(dict(TINY, block_counts=(1, 2), base_width=8, residual=False))
    assert type(m).__name__ == "ResNet" and m.feature_dim == 16, "your ResNet with the config's block counts and width"
    assert len(m.layers) == 2 and len(m.layers[1]) == 2
    assert not any(getattr(b, "residual", False) for b in m.modules() if type(b).__name__ == "BasicBlock")
    assert m(torch.zeros(2, 3, 16, 16)).shape == (2, 10)
    convs = [x for x in m.stem.modules() if isinstance(x, torch.nn.Conv2d)]
    assert convs[0].kernel_size == (3, 3) and convs[0].stride == (1, 1), "the CIFAR stem (small_input=True)"
    opt = nb.make_optimizer(m, dict(TINY, lr=0.07, weight_decay=1e-3))
    assert isinstance(opt, torch.optim.SGD), "SGD"
    g = opt.param_groups
    assert all(x["momentum"] == 0.9 and x["nesterov"] and x["lr"] == 0.07 for x in g), "momentum, nesterov and lr from config"
    params = [p for x in g for p in x["params"]]
    assert len(params) == len(list(m.parameters())) and len({id(p) for p in params}) == len(params), (
        "every parameter in exactly one group")
    for x in g:
        dims = {p.ndim for p in x["params"]}
        if x["weight_decay"] > 0:
            assert x["weight_decay"] == 1e-3 and dims and min(dims) > 1, "decay only conv and linear weights (ndim > 1)"
        else:
            assert max(dims) <= 1, "biases and batch-norm parameters (ndim <= 1) get no weight decay"


# ---------------------------------------------------------------- stage 4: training and evaluation


def test_stage4_predict(nb):
    torch.manual_seed(0)
    m = nb.build_model(TINY)
    m.train()
    X = torch.randn(30, 3, 16, 16)
    tracked = m.stem[1].num_batches_tracked.item()
    pred = nb.predict(m, X, batch_size=7)
    assert isinstance(pred, np.ndarray) and pred.dtype == np.int64 and pred.shape == (30,)
    assert m.stem[1].num_batches_tracked.item() == tracked, (
        "predict must run in eval mode: in train mode every batch updates batch norm's running statistics")
    assert m.training, "restore the model's mode"
    m.eval()
    with torch.no_grad():
        want = m(X).argmax(1).numpy()
    assert np.array_equal(pred, want), "predict in eval mode (batch norm uses running statistics), in batches"


def test_stage4_train_and_evaluate(nb):
    data = _data(nb)
    res = _bounded(nb.train_and_evaluate, dict(TINY, epochs=6), data, device="cpu", seed=0)
    assert {"acc", "correct", "history", "seconds", "n_params"} <= set(res), f"result keys; got {sorted(res)}"
    c = np.asarray(res["correct"])
    assert c.shape == (200,) and c.dtype == bool and math.isclose(res["acc"], c.mean()), (
        "correct: one boolean per test image; acc = its mean")
    assert res["acc"] >= 0.7, (
        f"6 epochs of the recipe on 4 easy synthetic classes should reach 70% test accuracy (typically 100%); got {res['acc']:.3f}")
    lrs = res["history"]["lr"]
    assert len(lrs) == 6 and lrs[-1] < 1e-3 and max(lrs) > 0.04, (
        f"the cosine schedule (per optimizer step, with warmup) decays the lr to ~0 by the end; per-epoch lrs {lrs}")
    assert res["n_params"] == sum(p.numel() for p in nb.build_model(TINY).parameters())


def test_stage4_uses_label_smoothing_and_warmup(nb):
    data = _data(nb, n_train=96, n_test=40)  # 96 / 32 = 3 batches per epoch: warmup_steps = 3
    seen_ls, step_lrs = [], []
    original = torch.nn.functional.cross_entropy

    def spy(*args, **kwargs):
        seen_ls.append(kwargs.get("label_smoothing", 0.0))
        return original(*args, **kwargs)

    from torch.optim.optimizer import register_optimizer_step_pre_hook

    handle = register_optimizer_step_pre_hook(
        lambda opt, args, kwargs: step_lrs.append(opt.param_groups[0]["lr"]))
    torch.nn.functional.cross_entropy = spy
    try:
        _bounded(nb.train_and_evaluate, dict(TINY, epochs=2, label_smoothing=0.23), data, seed=3)
    finally:
        torch.nn.functional.cross_entropy = original
        handle.remove()
    assert seen_ls and all(math.isclose(v, 0.23) for v in seen_ls), (
        f"the loss must use config['label_smoothing'] (F.cross_entropy(..., label_smoothing=...)); the training loss "
        f"was called with label_smoothing values {sorted(set(seen_ls))[:5]}")
    assert len(step_lrs) == 6, f"2 epochs x 3 batches = 6 optimizer steps; saw {len(step_lrs)}"
    assert math.isclose(step_lrs[1], 0.05 / 3, rel_tol=1e-6), (
        f"a linear warmup over warmup_epochs * batches_per_epoch = 3 steps (CosineSchedule(..., warmup_steps=3)) puts "
        f"the lr at 0.05 / 3 for the second step; got {step_lrs[1]:.5f} (lrs per step: "
        f"{[round(v, 4) for v in step_lrs]})")


def test_stage4_augment_flag(nb):
    data = _data(nb, n_train=96, n_test=40)
    cfg = dict(TINY, epochs=1)
    on = _bounded(nb.train_and_evaluate, dict(cfg, augment=True), data, seed=2)
    off = _bounded(nb.train_and_evaluate, dict(cfg, augment=False), data, seed=2)
    assert on["history"]["train_loss"] != off["history"]["train_loss"], (
        "config['augment'] must reach the training loader: with the same seed, augment=True and augment=False "
        "should train on different images")


def test_stage4_constant_schedule(nb):
    data = _data(nb, n_train=96, n_test=40)
    res = _bounded(nb.train_and_evaluate, dict(TINY, epochs=2, schedule="constant", augment=False), data, seed=1)
    assert all(math.isclose(v, 0.05) for v in res["history"]["lr"]), (
        f"schedule='constant': no scheduler, the lr stays at config['lr']; got {res['history']['lr']}")


# ---------------------------------------------------------------- stage 5: the ablation table


def test_stage5_ablation_table(nb):
    rng = np.random.default_rng(0)
    base = rng.random(2000) < 0.9
    better = base | (rng.random(2000) < 0.3)  # paired: mostly the same images right
    unrelated = rng.random(2000) < 0.9
    results = {"full recipe": {"correct": base}, "variant": {"correct": better}, "other": {"correct": unrelated}}
    rows = nb.ablation_table(results, n_boot=1000, seed=0)
    assert [r["name"] for r in rows] == ["full recipe", "variant", "other"], "one row per configuration, in order"
    r0, r1, r2 = rows
    assert math.isclose(r0["acc"], base.mean()) and r0["delta"] == 0.0 and r0["delta_lo"] == r0["delta_hi"] == 0.0
    se = math.sqrt(base.mean() * (1 - base.mean()) / 2000)
    assert r0["lo"] < r0["acc"] < r0["hi"] and abs((r0["hi"] - r0["lo"]) - 3.92 * se) < 0.4 * 3.92 * se, (
        f"a 95% bootstrap interval for the accuracy is about +-1.96 standard errors ({1.96 * se:.4f}); got "
        f"[{r0['lo']:.4f}, {r0['hi']:.4f}]")
    assert math.isclose(r1["delta"], better.mean() - base.mean())
    assert r1["delta_lo"] > 0, (
        f"'variant' gets right every image the baseline does, plus more: the PAIRED interval of the difference (resample "
        f"the same test images for both) excludes 0; got [{r1['delta_lo']:.4f}, {r1['delta_hi']:.4f}]")
    paired_width = r1["delta_hi"] - r1["delta_lo"]
    unpaired_se = math.sqrt(2) * se
    assert paired_width < 0.6 * 3.92 * unpaired_se, (
        "pairing removes the shared per-image variation: the difference's interval must be much narrower than an "
        "unpaired one built from two independent accuracies")
    assert r2["delta_lo"] < r2["delta"] < r2["delta_hi"]
