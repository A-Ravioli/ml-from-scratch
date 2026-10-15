"""Checks for exercise.py (regularization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The L2 penalty is
checked by value and gradient, and against SGD's built-in weight decay; Dropout by its statistics,
gradient masking and train/eval behaviour; label smoothing by hand values, gradcheck and (when
installed) torch's F.cross_entropy(label_smoothing=...); augmentations by exact draws and by
invariants; EarlyStopping on hand-written metric sequences.
"""
import numpy as np
import pytest

from mlfs import nn
from mlfs.optim import SGD, Adam, AdamW
from mlfs.tensor import Tensor, gradcheck


# ---------------------------------------------------------------- stage 1: l2_penalty


def test_stage1_value_and_gradient(nb):
    a = nn.Parameter(np.array([1.0, -2.0]))
    b = nn.Parameter(np.array([[3.0]]))
    pen = nb.l2_penalty([a, b], 0.1)
    assert isinstance(pen, Tensor) and pen.size == 1, "l2_penalty returns a one-element Tensor you can add to a loss"
    assert np.isclose(pen.item(), 0.05 * (1 + 4 + 9)), (
        f"(weight_decay / 2) * sum of squares = 0.05 * 14 = 0.7; got {pen.item()}. (the 1/2 makes the gradient "
        "exactly weight_decay * w)")
    pen.backward()
    assert np.allclose(a.grad, [0.1, -0.2]) and np.allclose(b.grad, [[0.3]]), "its gradient is weight_decay * w"
    assert np.isclose(nb.l2_penalty([], 0.1).item(), 0.0), "no parameters, no penalty"


def test_stage1_equals_sgd_weight_decay_but_not_adamw(nb):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(20, 3)), rng.normal(size=(20, 1))
    runs = {}
    for kind in ["penalty", "builtin"]:
        model = nn.Linear(3, 1, rng=1)
        opt = SGD(model.parameters(), lr=0.1, weight_decay=0.0 if kind == "penalty" else 0.05)
        for _ in range(5):
            opt.zero_grad()
            loss = nn.mse_loss(model(X), y)
            if kind == "penalty":
                loss = loss + nb.l2_penalty(model.parameters(), 0.05)
            loss.backward()
            opt.step()
        runs[kind] = model.weight.data.copy()
    assert np.allclose(runs["penalty"], runs["builtin"]), (
        "for SGD, adding l2_penalty(params, wd) to the loss is exactly SGD(weight_decay=wd): both add wd * w to the "
        "gradient")
    model_a, model_w = nn.Linear(3, 1, rng=1), nn.Linear(3, 1, rng=1)
    opt_a, opt_w = Adam(model_a.parameters(), lr=0.1), AdamW(model_w.parameters(), lr=0.1, weight_decay=0.05)
    for _ in range(5):
        for model, opt, pen in [(model_a, opt_a, True), (model_w, opt_w, False)]:
            opt.zero_grad()
            loss = nn.mse_loss(model(X), y)
            loss = loss + nb.l2_penalty(model.parameters(), 0.05) if pen else loss
            loss.backward()
            opt.step()
    assert not np.allclose(model_a.weight.data, model_w.weight.data, atol=1e-4), (
        "with Adam, an L2 penalty and decoupled weight decay (AdamW) are different algorithms: if they agree, your "
        "penalty isn't reaching the gradient")


# ---------------------------------------------------------------- stage 2: Dropout


def test_stage2_statistics_and_scaling(nb):
    d = nb.Dropout(0.3, rng=0)
    assert d.p == 0.3, "store the probability as self.p (topic 07's to_torch reads it)"
    x = np.ones((400, 500))
    out = d(Tensor(x)).data
    frac = np.mean(out == 0)
    assert abs(frac - 0.3) < 0.01, f"about a fraction p = 0.3 of the entries are zeroed; got {frac:.3f}"
    kept = out[out != 0]
    assert np.allclose(kept, 1 / 0.7), (
        f"inverted dropout scales the kept entries by 1 / (1 - p) = {1 / 0.7:.4f} so the expected output equals the "
        f"input; got kept values like {kept[:3]}")
    assert abs(out.mean() - 1.0) < 0.01, "E[dropout(x)] = x"
    assert not np.array_equal(d(Tensor(x)).data, out), "a new mask on every call (the generator moves on)"


def test_stage2_gradient_uses_the_same_mask(nb):
    d = nb.Dropout(0.5, rng=1)
    x = Tensor(np.random.default_rng(2).normal(size=(6, 7)), requires_grad=True)
    out = d(x)
    out.backward(np.ones(out.shape))
    mask = out.data != 0
    assert np.allclose(x.grad, mask * 2.0), (
        "the gradient flows only through the kept entries, scaled by 1 / (1 - p): dropout is x * (mask / (1 - p))")


def test_stage2_eval_and_edge_cases(nb):
    x = np.random.default_rng(3).normal(size=(4, 5))
    d = nb.Dropout(0.5, rng=0)
    d.eval()
    assert np.array_equal(d(x).data, x), "in eval mode dropout is the identity (no mask, no scaling)"
    d.train()
    assert np.array_equal(nb.Dropout(0.0, rng=0)(x).data, x), "p = 0 is the identity"
    assert np.all(nb.Dropout(1.0, rng=0)(x).data == 0), "p = 1 zeroes everything (and doesn't divide by zero)"
    assert nn.Sequential(nn.Linear(5, 3, rng=0), nb.Dropout(0.5)).parameters()[-1].shape == (3,), (
        "Dropout has no parameters")
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            nb.Dropout(bad)
    a = nb.Dropout(0.5, rng=7)(np.ones((3, 3))).data
    b = nb.Dropout(0.5, rng=7)(np.ones((3, 3))).data
    assert np.array_equal(a, b), "the same seed gives the same mask: store self.rng = as_rng(rng) once"


def test_stage2_switches_with_the_model(nb):
    net = nn.Sequential(nn.Linear(10, 10, rng=0), nb.Dropout(0.5, rng=0))
    x = np.ones((2, 10))
    net.eval()
    assert np.array_equal(net(x).data, net(x).data), "model.eval() turns the Dropout inside a Sequential off"
    net.train()
    assert np.mean(net(np.ones((50, 10))).data == 0) > 0.3, "model.train() turns it back on"


# ---------------------------------------------------------------- stage 3: label smoothing


def test_stage3_hand_example(nb):
    logits = Tensor(np.array([[2.0, 0.0, 0.0]]))
    got = nb.label_smoothing_cross_entropy(logits, np.array([0]), smoothing=0.3).item()
    logp = np.array([2.0, 0, 0]) - np.log(np.exp(2) + 2)
    target = 0.7 * np.array([1, 0, 0]) + 0.1
    want = -(target * logp).sum()
    assert np.isclose(got, want), (
        f"target distribution (1 - 0.3) * one_hot + 0.3 / 3 = (0.8, 0.1, 0.1); loss = -sum target * log_softmax = "
        f"{want:.5f}; got {got:.5f}. (smoothing spreads s over ALL k classes, the true one included)")
    plain = nb.label_smoothing_cross_entropy(logits, np.array([0]), smoothing=0.0).item()
    assert np.isclose(plain, -logp[0]), "smoothing = 0 is ordinary cross-entropy"


def test_stage3_gradcheck_and_stability(nb):
    rng = np.random.default_rng(4)
    z = Tensor(rng.normal(size=(5, 4)), requires_grad=True)
    t = rng.integers(0, 4, 5)
    assert gradcheck(lambda v: nb.label_smoothing_cross_entropy(v, t, smoothing=0.2), [z]), (
        "label_smoothing_cross_entropy's gradient disagrees with finite differences (build it from your Tensor ops)")
    big = Tensor(np.array([[1000.0, 0.0, -1000.0]]))
    val = nb.label_smoothing_cross_entropy(big, np.array([0]), smoothing=0.1).item()
    assert np.isfinite(val) and np.isclose(val, 0.1 / 3 * (1000 + 2000)), (
        "huge logits must give a finite loss: use logsumexp, never log(softmax). here: (s/k) * (0 + 1000 + 2000) = 100; "
        f"got {val}")


def test_stage3_minimum_is_not_infinite_confidence(nb):
    s, k = 0.1, 4
    gaps = np.linspace(0, 10, 201)
    losses = [nb.label_smoothing_cross_entropy(Tensor(np.array([[g, 0, 0, 0]])), np.array([0]), smoothing=s).item()
              for g in gaps]
    best = gaps[int(np.argmin(losses))]
    want = np.log((1 - s + s / k) / (s / k))
    assert abs(best - want) < 0.1, (
        f"with smoothing the loss is minimized at a FINITE logit gap log((1 - s + s/k) / (s/k)) = {want:.3f}, not at "
        f"infinity; your loss is minimized at {best:.2f}")


def test_stage3_matches_torch(nb):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    z, t = rng.normal(size=(6, 5)), rng.integers(0, 5, 6)
    want = torch.nn.functional.cross_entropy(torch.tensor(z), torch.tensor(t), label_smoothing=0.15).item()
    got = nb.label_smoothing_cross_entropy(Tensor(z), t, smoothing=0.15).item()
    assert np.isclose(got, want), f"torch's F.cross_entropy(label_smoothing=0.15) gives {want:.6f}, you {got:.6f}"


# ---------------------------------------------------------------- stage 4: augmentation


def test_stage4_random_crop_exact(nb):
    imgs = np.arange(2 * 1 * 4 * 4, dtype=float).reshape(2, 1, 4, 4)
    out = nb.random_crop(imgs, 3, padding=1, rng=0)
    assert out.shape == (2, 1, 3, 3), f"size x size crops, one per image: (2, 1, 3, 3); got {out.shape}"
    r = np.random.default_rng(0)
    tops, lefts = r.integers(0, 4, size=2), r.integers(0, 4, size=2)
    padded = np.pad(imgs, ((0, 0), (0, 0), (1, 1), (1, 1)))
    want = np.stack([padded[i, :, tops[i]:tops[i] + 3, lefts[i]:lefts[i] + 3] for i in range(2)])
    assert np.array_equal(out, want), (
        "draw order: rng = as_rng(rng); tops = rng.integers(0, H + 2*padding - size + 1, size=n), THEN lefts the same "
        "way; image i is padded[i, :, tops[i]:tops[i]+size, lefts[i]:lefts[i]+size] (zero padding)")


def test_stage4_random_crop_properties(nb):
    rng = np.random.default_rng(6)
    imgs = rng.normal(size=(50, 3, 8, 8))
    same = nb.random_crop(imgs, 8, padding=0, rng=1)
    assert np.array_equal(same, imgs), "a full-size crop with no padding returns the images unchanged"
    out = nb.random_crop(imgs, 8, padding=2, rng=1)
    assert out.shape == imgs.shape, "padding 2 then cropping back to 8 x 8 keeps the shape (a random shift)"
    shifts = set()
    for i in range(50):
        padded = np.pad(imgs[i], ((0, 0), (2, 2), (2, 2)))
        found = [(a, b) for a in range(5) for b in range(5) if np.array_equal(padded[:, a:a + 8, b:b + 8], out[i])]
        assert found, f"crop {i} isn't a window of the zero-padded image (channels must be cropped together)"
        shifts.add(found[0])
    assert len(shifts) > 5, "each image gets its own random offset"
    with pytest.raises(ValueError):
        nb.random_crop(imgs, 13, padding=2, rng=0)


def test_stage4_horizontal_flip(nb):
    imgs = np.arange(4 * 2 * 2 * 3, dtype=float).reshape(4, 2, 2, 3)
    before = imgs.copy()
    out = nb.random_horizontal_flip(imgs, p=0.5, rng=3)
    flip = np.random.default_rng(3).random(4) < 0.5
    want = np.where(flip[:, None, None, None], imgs[..., ::-1], imgs)
    assert np.array_equal(out, want), (
        "flip = rng.random(n) < p decides per image; a flip mirrors the LAST axis (width), for every channel")
    assert np.array_equal(imgs, before), "return a new array: don't modify the caller's images"
    assert np.array_equal(nb.random_horizontal_flip(imgs, p=1.0, rng=0), imgs[..., ::-1])
    assert np.array_equal(nb.random_horizontal_flip(imgs, p=0.0, rng=0), imgs)


# ---------------------------------------------------------------- stage 5: EarlyStopping


def test_stage5_min_mode_patience(nb):
    es = nb.EarlyStopping(patience=2)
    seq = [1.0, 0.8, 0.85, 0.79, 0.9, 0.95]
    got = [es.step(v) for v in seq]
    assert got == [False, False, False, False, False, True], (
        f"patience=2: stop after 2 steps IN A ROW without improvement. for {seq} the bad steps are 0.85 (reset by "
        f"0.79), then 0.9 and 0.95, so step() returns True only on the last; got {got}")
    assert es.best == 0.79 and es.best_step == 3, f"best = 0.79 at step index 3 (0-based); got {es.best}, {es.best_step}"
    assert es.should_stop is True and es.step(0.1) is True, "once stopped, it stays stopped"
    small = nb.EarlyStopping(patience=1, min_delta=0.1)
    small.step(1.0)
    assert small.step(0.95) is True and small.best == 1.0, (
        "in mode 'min', 0.95 doesn't beat 1.0 by more than min_delta = 0.1, so it isn't an improvement")


def test_stage5_max_mode_and_min_delta(nb):
    es = nb.EarlyStopping(patience=3, mode="max", min_delta=0.01)
    for v in [0.5, 0.505, 0.509, 0.52]:
        es.step(v)
    assert es.best == 0.52 and es.best_step == 3, (
        "mode='max' looks for increases; with min_delta=0.01, 0.505 and 0.509 don't count as improvements over 0.5, "
        f"but 0.52 does; got best={es.best} at {es.best_step}")
    assert [es.step(v) for v in [0.525, 0.51, 0.529]] == [False, False, True], (
        "3 steps that fail to beat 0.52 by more than 0.01 exhaust patience=3")
    with pytest.raises(ValueError):
        nb.EarlyStopping(mode="median")
    with pytest.raises(ValueError):
        nb.EarlyStopping(patience=0)  # would stop on the very first step, even an improvement


def test_stage5_framework_agnostic(nb):
    es = nb.EarlyStopping(patience=1)
    assert es.best is None and es.best_step == -1 and es.should_stop is False
    es.step(np.float32(2.0))
    es.step(np.array(1.5))
    assert isinstance(es.best, float) and es.best == 1.5, "values are converted with float(): any scalar works"
    torch = pytest.importorskip("torch")
    es2 = nb.EarlyStopping(patience=1)
    es2.step(torch.tensor(1.0))
    assert es2.step(torch.tensor(1.2)) is True and es2.best == 1.0, "torch scalars work too (float(value))"


def test_stage5_plugs_into_fit(nb):
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(40, 2)), rng.integers(0, 2, 40)
    model = nn.Sequential(nn.Linear(2, 2, rng=0))
    es = nb.EarlyStopping(patience=2)
    h = nn.fit(model, nn.cross_entropy, SGD(model.parameters(), lr=0.0), X, y, epochs=50, X_val=X[:10], y_val=y[:10],
               rng=0, callback=lambda epoch, hist: es.step(hist["val_loss"][-1]))
    assert len(h["train_loss"]) == 3, (
        "with lr = 0 the validation loss never changes, so patience=2 stops fit after 3 epochs "
        f"(1 best + 2 bad); got {len(h['train_loss'])}")
