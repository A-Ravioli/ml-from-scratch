"""Checks for 19 · capstone: reproduce a theory figure (model-wise double descent).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. data, training, the sweep,
the threshold finder and the intervals are checked on small synthetic problems: exact noise counts and
sampling without replacement, interpolation of random labels, both losses, eval-mode handling, agreement
of the sweep with your own pieces, hand-made curves, and scipy's t quantiles. everything runs on the CPU
in a few seconds; the real experiment is the notebook's full config.
"""
import math

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

def _run(fn, *args, **kwargs):
    """every call that loops runs under a line budget, so a learner's endless loop fails instead of hanging."""
    return bounded(fn, *args, budget=20_000_000, what=getattr(fn, "__name__", None), **kwargs)



def _warm_up():
    """torch imports parts of itself on first use; do that here, outside any line budget."""
    m = nn.Linear(2, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    (F.cross_entropy(m(torch.zeros(3, 2)), torch.zeros(3, dtype=torch.long))
     + ((m(torch.zeros(3, 2)) - F.one_hot(torch.zeros(3, dtype=torch.long), 2).float()) ** 2).sum()).backward()
    opt.step()


_warm_up()


def _images(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, 6, 6)).astype(np.uint8), rng.integers(0, 10, size=n)


# ---------------------------------------------------------------- stage 1: the data


def test_stage1_noisy_subset(nb):
    X, y = _images()
    out = nb.noisy_subset(X, y, 50, noise=0.3, rng=0)
    assert set(out) >= {"X", "y_clean", "y_noisy"}, "return a dict with X, y_clean, y_noisy"
    Xs, yc, yn = out["X"], out["y_clean"], out["y_noisy"]
    assert torch.is_tensor(Xs) and Xs.dtype == torch.float32 and tuple(Xs.shape) == (50, 36), (
        f"X is a float32 tensor (n, d) = (50, 36), images flattened; got {getattr(Xs, 'dtype', None)} {tuple(Xs.shape)}")
    assert float(Xs.min()) >= 0 and float(Xs.max()) <= 1 and float(Xs.max()) > 0.9, "pixels scaled to [0, 1] (divide by 255)"
    assert yc.dtype == torch.int64 and yn.dtype == torch.int64, "labels are int64 tensors"
    rows = {tuple((Xs[i] * 255).round().to(torch.int64).tolist()) for i in range(50)}
    assert len(rows) == 50, "the n images are drawn WITHOUT replacement (no image twice)"
    src = {tuple(X[i].reshape(-1).tolist()): int(y[i]) for i in range(len(X))}
    for i in range(50):
        key = tuple((Xs[i] * 255).round().to(torch.int64).tolist())
        assert src[key] == int(yc[i]), "y_clean holds each chosen image's true label"
    assert int((yc != yn).sum()) == 15, f"exactly round(0.3 * 50) = 15 labels change, got {int((yc != yn).sum())}"
    a = nb.noisy_subset(X, y, 50, noise=0.3, rng=0)
    b = nb.noisy_subset(X, y, 50, noise=0.3, rng=1)
    assert torch.equal(a["X"], Xs) and not torch.equal(b["X"], Xs), "rng fixes the subset: same seed same draw"
    assert int((nb.noisy_subset(X, y, 37, noise=0.7, rng=2)["y_noisy"] != nb.noisy_subset(X, y, 37, noise=0.7, rng=2)["y_clean"]).sum()) == 26, (
        "round(0.7 * 37) = 26 labels change")
    with pytest.raises(ValueError):
        nb.noisy_subset(X, y, 500, rng=0)


def test_stage1_wrong_class_uniform(nb):
    X = np.zeros((6000, 2, 2), dtype=np.uint8)
    y = np.full(6000, 4)
    out = nb.noisy_subset(X, y, 6000, noise=1.0, rng=0)
    counts = np.bincount(out["y_noisy"].numpy(), minlength=10)
    assert counts[4] == 0, "a corrupted label is always a different class"
    others = np.delete(counts, 4)
    chi2 = float(((others - others.mean()) ** 2 / others.mean()).sum())
    assert chi2 < stats.chi2.ppf(0.999, 8), f"the wrong class is uniform over the other 9: counts {others.tolist()}"


# ---------------------------------------------------------------- stage 2: training and evaluation


def test_stage2_interpolates_random_labels(nb):
    torch.manual_seed(0)
    X = torch.randn(40, 10)
    y = torch.randint(0, 3, (40,))
    m = nn.Sequential(nn.Linear(10, 200), nn.ReLU(), nn.Linear(200, 3))
    res = bounded(nb.train_interpolator, m, X, y, steps=400, lr=1e-2, n_classes=3, what="train_interpolator",
                  budget=5_000_000)
    assert len(res["loss"]) == 400, f"history['loss'] has one full-batch loss per step (400), got {len(res['loss'])}"
    assert res["train_error"] == 0.0, f"a 200-unit network must fit 40 random labels exactly; train error {res['train_error']}"
    with torch.no_grad():
        out = m(X)
    mse = float(((out - F.one_hot(y, 3).float()) ** 2).sum(1).mean())
    assert mse < 0.05, (
        f"with loss='mse' (the default) the outputs approach the one-hot targets: squared error {mse:.3f}. the loss is "
        "the squared error to one-hot, summed over classes, averaged over examples")
    assert res["loss"][-1] < res["loss"][0] / 10, "the loss falls during training"


def test_stage2_loss_kinds_lr_and_modes(nb):
    torch.manual_seed(1)
    X = torch.randn(30, 5)
    y = torch.randint(0, 2, (30,))
    lin = nn.Linear(5, 2)
    with torch.no_grad():
        target = ((lin(X) - F.one_hot(y, 2).float()) ** 2).sum(1).mean()
        ce0 = F.cross_entropy(lin(X), y)
    w0 = lin.weight.detach().clone()
    b0 = lin.bias.detach().clone()
    r = _run(nb.train_interpolator, lin, X, y, steps=3, lr=0.0, n_classes=2)
    assert torch.equal(lin.weight, w0), "lr = 0 must leave the weights unchanged"
    r = _run(nb.train_interpolator, lin, X, y, steps=3, lr=0.1, n_classes=2)
    with torch.no_grad():
        lin.weight.copy_(w0)
        lin.bias.copy_(b0)
    assert math.isclose(r["loss"][0], float(target), rel_tol=1e-5), (
        f"the first recorded loss is the squared error to one-hot of the untrained model ({float(target):.5f}), got "
        f"{r['loss'][0]:.5f}: sum over classes, mean over examples, recorded before the step")
    r = _run(nb.train_interpolator, lin, X, y, steps=2, lr=0.0, loss="ce", n_classes=2)
    assert math.isclose(r["loss"][0], float(ce0), rel_tol=1e-5), "loss='ce' trains on cross-entropy"
    m = nn.Sequential(nn.Linear(5, 8), nn.Dropout(0.5), nn.Linear(8, 2))
    m.eval()
    seen = []
    m.register_forward_pre_hook(lambda mod, inp: seen.append(mod.training))
    _run(nb.train_interpolator, m, X, y, steps=2, lr=1e-3, n_classes=2)
    assert seen[0] is True, "train in train mode (call model.train())"


def test_stage2_evaluate(nb):
    lin = nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    X = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 3.0], [0.5, 0.0]])
    y = torch.tensor([0, 0, 1, 2])
    m = nn.Sequential(lin, nn.Dropout(0.9))
    m.train()
    ev = nb.evaluate(m, X, y, n_classes=3)
    out = X @ lin.weight.T
    want_mse = float(((out - F.one_hot(y, 3).float()) ** 2).sum(1).mean())
    assert math.isclose(ev["error"], 0.5), f"argmax predictions 0, 1, 1, 0 against labels 0, 0, 1, 2: error 0.5, got {ev['error']}"
    assert math.isclose(ev["loss"], want_mse, rel_tol=1e-6), (
        f"loss is the squared error to one-hot summed over classes and averaged ({want_mse:.4f}); got {ev['loss']:.4f}. "
        "evaluate in eval mode (dropout off)")
    assert m.training, "evaluate restores the model's mode"
    ce = nb.evaluate(m, X, y, loss="ce", n_classes=3)
    assert math.isclose(ce["loss"], float(F.cross_entropy(out, y)), rel_tol=1e-6), "loss='ce' reports cross-entropy"


# ---------------------------------------------------------------- stage 3: the sweep


def _toy(n=60, m=200, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n + m, d, generator=g)
    w = torch.randn(d, 3, generator=g)
    y = (X @ w).argmax(1)
    return X[:n], y[:n], X[n:], y[n:]


def test_stage3_curve_matches_your_pieces(nb):
    Xtr, _, Xte, yte = _toy()
    ytr = torch.randint(0, 3, (60,), generator=torch.Generator().manual_seed(5))  # random labels: 2 units can't fit them
    out = bounded(nb.model_wise_curve, Xtr, ytr, Xte, yte, [2, 64], seeds=(0, 1), steps=300, lr=1e-2, n_classes=3,
                  what="model_wise_curve", budget=5_000_000)
    for k in ("train_error", "test_error", "test_loss"):
        assert np.asarray(out[k]).shape == (2, 2), f"{k} is (len(widths), len(seeds)) = (2, 2)"
    assert list(out["n_params"]) == [12 * 2 + 2 + 2 * 3 + 3, 12 * 64 + 64 + 64 * 3 + 3], (
        f"n_params of the two-layer model d·w + w + w·K + K: expected [35, 1027], got {list(out['n_params'])}")
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 3))
    res = _run(nb.train_interpolator, m, Xtr, ytr, steps=300, lr=1e-2, n_classes=3)
    ev = nb.evaluate(m, Xte, yte, n_classes=3)
    assert math.isclose(out["train_error"][1, 1], res["train_error"]) and math.isclose(out["test_loss"][1, 1], ev["loss"], rel_tol=1e-5), (
        "width 64, seed 1 must equal: torch.manual_seed(1), two_layer(d, 64, K), train_interpolator(steps, lr, loss), "
        "evaluate on the test set; pass steps, lr and loss through")
    again = _run(nb.model_wise_curve, Xtr, ytr, Xte, yte, [64], seeds=(0,), steps=300, lr=1e-2, n_classes=3)
    assert np.allclose(again["test_loss"][0, 0], out["test_loss"][1, 0]), "the same seed must reproduce the same result"
    assert out["train_error"][1].max() <= 1 / 60 and out["train_error"][0].mean() > out["train_error"][1].mean(), (
        "64 units (almost) interpolate 60 points; 2 units can't")


def test_stage3_loss_passthrough(nb):
    Xtr, ytr, Xte, yte = _toy(seed=1)
    a = _run(nb.model_wise_curve, Xtr, ytr, Xte, yte, [16], seeds=(0,), steps=30, lr=1e-2, n_classes=3, loss="ce")
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(12, 16), nn.ReLU(), nn.Linear(16, 3))
    _run(nb.train_interpolator, m, Xtr, ytr, steps=30, lr=1e-2, n_classes=3, loss="ce")
    assert math.isclose(a["test_loss"][0, 0], nb.evaluate(m, Xte, yte, n_classes=3, loss="ce")["loss"], rel_tol=1e-5), (
        "loss='ce' must reach both training and evaluation")


# ---------------------------------------------------------------- stage 4: reading the curve


def test_stage4_locate_threshold(nb):
    widths = [4, 8, 16, 32, 64, 128]
    train = np.array([[0.5, 0.4], [0.2, 0.3], [0.05, 0.002], [0.012, 0.006], [0.0, 0.0], [0.0, 0.0]])
    metric = np.array([[2.0, 2.2], [0.8, 0.7], [0.9, 1.1], [1.6, 1.8], [1.2, 1.0], [0.6, 0.5]])
    loc = nb.locate_threshold(widths, train, metric)
    assert loc["interpolation_width"] == 32, (
        f"the first width whose SEED-MEAN train error is <= 0.01 is 32 (mean 0.009; at width 16 one seed is at 0.002 "
        f"but the mean is 0.026); got {loc['interpolation_width']}")
    assert loc["peak_width"] == 32, (
        f"the peak is the largest seed-mean test metric at or beyond the interpolation width (32 here), not the "
        f"underfit width 4 whose metric is larger still; got {loc['peak_width']}")
    assert loc["first_min_width"] == 8, f"below the peak, the seed-mean metric is lowest at width 8; got {loc['first_min_width']}"
    loc = nb.locate_threshold(widths, train, metric, tol=0.1)
    assert loc["interpolation_width"] == 16 and loc["peak_width"] == 32, "tol sets the interpolation criterion"
    exact = nb.locate_threshold([1, 2, 3], np.array([[0.5], [0.25], [0.0]]), np.array([[1.0], [3.0], [2.0]]), tol=0.25)
    assert exact["interpolation_width"] == 2, "a seed-mean train error exactly equal to tol counts (<= tol)"
    none = nb.locate_threshold([1, 2], np.ones((2, 1)), np.array([[3.0], [1.0]]))
    assert none["interpolation_width"] is None and none["first_min_width"] is None and none["peak_width"] == 1, (
        "with no interpolating width the peak is searched over all widths; None when the peak is the first width")


# ---------------------------------------------------------------- stage 5: intervals over seeds


def test_stage5_mean_ci(nb):
    v = np.array([[0.30, 0.34, 0.31], [0.10, 0.10, 0.13]])
    m, lo, hi = nb.mean_ci(v)
    half = stats.t.ppf(0.975, 2) * v.std(axis=1, ddof=1) / math.sqrt(3)
    assert np.allclose(m, v.mean(1)) and np.allclose(lo, v.mean(1) - half) and np.allclose(hi, v.mean(1) + half), (
        "mean +- t(0.975, n - 1) * sample standard deviation (ddof=1) / sqrt(n), over the seeds (last axis)")
    _, lo90, hi90 = nb.mean_ci(v, alpha=0.1)
    assert np.all(hi90 - lo90 < hi - lo), "alpha sets the confidence level (90% intervals are narrower)"
    m1, lo1, hi1 = nb.mean_ci(np.array([[0.2], [0.4]]))
    assert np.allclose(m1, [0.2, 0.4]) and np.all(np.isnan(lo1)) and np.all(np.isnan(hi1)), "one seed: no interval (nan)"
