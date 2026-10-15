"""Checks for capstone.py (fix the training run).

Each check targets one planted bug, and is written so that it fails only because of that bug (the
others are neutralized: e.g. the early-stopping check uses a model without dropout, so a broken
`evaluate` can't mask it). failure messages describe the symptom and the intended behaviour, not
the line to change. data are small synthetic 'digits' with a blank border, like MNIST's; training
runs under `mlfs.testing.bounded` and seeds torch explicitly.
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from mlfs.testing import bounded  # noqa: E402
from mlfs.torchnn import make_loader  # noqa: E402


def _digits(n, seed=0, noise=0.8):
    """synthetic 28 x 28 'digits' in [0, 1], flattened: 10 smooth templates + noise, with a blank 3-pixel border."""
    rng = np.random.default_rng(1234)  # the templates are fixed; only the samples depend on seed
    t = rng.normal(size=(10, 28, 28))
    for axis in (1, 2):
        for _ in range(3):
            t = (t + np.roll(t, 1, axis=axis) + np.roll(t, -1, axis=axis)) / 3
    t /= t.std()
    r = np.random.default_rng(seed)
    y = r.integers(0, 10, n)
    x = np.clip(0.5 + 0.2 * (t[y] + noise * r.normal(size=(n, 28, 28))), 0, 1)
    x[:, :3], x[:, -3:], x[:, :, :3], x[:, :, -3:] = 0.0, 0.0, 0.0, 0.0
    return x.reshape(n, 784), y.astype(np.int64)


def _small_net(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(784, 32), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(32, 10))


def _bounded(fn, *args, **kwargs):
    return bounded(lambda: fn(*args, **kwargs), budget=20_000_000, what=getattr(fn, "__name__", "the pipeline"),
                   hint="every loop should run a fixed number of times (epochs, batches).")


CONFIG = dict(n_train=1500, n_val=300, hidden=(64,), epochs=6, batch_size=64, lr=0.05, lr_step_epochs=3, patience=10,
              n_boot=200)


# ---------------------------------------------------------------- stage 1: data


class _Probe(torch.nn.Module):
    """identity layer recording the grad mode and training flag it runs under."""

    def __init__(self):
        super().__init__()
        self.grad_modes, self.train_flags = [], []

    def forward(self, x):
        self.grad_modes.append(torch.is_grad_enabled())
        self.train_flags.append(self.training)
        return x


def test_stage1_split_keeps_images_and_labels_together(nb):
    X = np.repeat(np.arange(50, dtype=float)[:, None], 4, axis=1)  # row i is filled with the number i
    y = np.arange(50)
    Xt, yt, Xv, yv = nb.split_data(X, y, 10, rng=0)
    assert len(Xv) == 10 and len(Xt) == 40
    assert np.array_equal(Xt[:, 0], yt) and np.array_equal(Xv[:, 0], yv), (
        "after the split, some images carry another image's label: whatever reorders the images must reorder the "
        "labels the same way. (in a real run this caps accuracy at chance, and nothing crashes.)")
    perm = np.random.default_rng(0).permutation(50)
    assert np.array_equal(yv, perm[:10]), "validation = the first n_val indices of one permutation"


def test_stage1_normalized_data_is_finite(nb):
    X, _ = _digits(200)
    a, b = nb.normalize(X[:150], X[150:])
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (
        f"normalize produced {np.sum(~np.isfinite(a)) + np.sum(~np.isfinite(b))} nan/inf values. these images have a "
        "blank border, like MNIST's: what is the standard deviation of a pixel that is 0 in every image? the spec asks "
        "for the training set's OVERALL pixel mean and std (two scalars)")
    assert abs(a.mean()) < 0.1 and abs(a.std() - 1) < 0.1, (
        f"the normalized training set should have mean ~0 and std ~1 OVERALL (all pixels together); got mean "
        f"{a.mean():.3f}, std {a.std():.3f}. standardizing each pixel separately leaves the always-blank ones at 0")


def test_stage1_normalization_uses_only_the_training_set(nb):
    X, _ = _digits(300, seed=1)
    tr, other = X[:200], X[200:]
    a1, b1 = nb.normalize(tr, other)
    changed = other.copy()
    changed[50:] = np.clip(changed[50:] * 3.0, 0, 3)  # change half of the evaluation set drastically
    a2, b2 = nb.normalize(tr, changed)
    assert np.allclose(a1, a2, equal_nan=True), (
        "the normalized TRAINING data changes when you change the evaluation data: the statistics leak information "
        "from the validation/test set into training. compute them from X_train alone")
    assert np.allclose(b1[:50], b2[:50], equal_nan=True), (
        "an evaluation image's normalized values depend on the OTHER evaluation images: the statistics must come from "
        "the training set only")


# ---------------------------------------------------------------- stage 2: the model


def test_stage2_outputs_are_logits(nb):
    model = nb.build_model(hidden=(32,), seed=0)
    kinds = [type(m).__name__ for m in model.modules()]
    out = model(torch.zeros(4, 784) + 0.1)
    assert out.shape == (4, 10)
    assert not any(k in ("Softmax", "LogSoftmax", "Sigmoid") for k in kinds) and not torch.allclose(
        out.sum(1), torch.ones(4)), (
        "the model's outputs are already probabilities (rows sum to 1), but F.cross_entropy applies its own softmax: "
        "the loss becomes a softmax of probabilities, which can't get confident and trains very slowly. "
        f"the model should end with a plain Linear layer that outputs logits; got layers {kinds}")


def test_stage2_initialization_scale(nb):
    model = nb.build_model(hidden=(256, 128), seed=0)
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    w0, w1 = linears[0].weight, linears[1].weight
    for w, fan_in in [(w0, 784), (w1, 256)]:
        want = np.sqrt(2.0 / fan_in)
        got = w.std().item()
        assert abs(got - want) < 0.1 * want, (
            f"a hidden layer with fan_in {fan_in} has weight std {got:.3f}; kaiming-normal for ReLU (topic 01) gives "
            f"sqrt(2/{fan_in}) = {want:.4f}. with the current scale the first layer's pre-activations have a std of "
            "about sqrt(fan_in) times the inputs'")
    assert all(torch.all(m.bias == 0) for m in linears[:-1]), "hidden biases start at zero"
    x = torch.randn(512, 784)
    with torch.no_grad():
        h = model[1](model[0](x))
    assert 0.3 < h.pow(2).mean().item() < 3, "with kaiming init the first ReLU's outputs keep a second moment near 1"


def test_stage2_structure(nb):
    model = nb.build_model(hidden=(32, 16), dropout=0.3, seed=1)
    kinds = [type(m).__name__ for m in model if not isinstance(m, torch.nn.Softmax)]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Dropout", "Linear"], kinds
    again = nb.build_model(hidden=(32, 16), dropout=0.3, seed=1)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), again.parameters())), "same seed, same model"


# ---------------------------------------------------------------- stage 3: the training step


def test_stage3_each_step_uses_only_its_own_gradient(nb):
    X, y = _digits(64, seed=2)
    model = _small_net(0)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: the model never changes, so the batches' gradients are fixed
    loader = make_loader(X, y, batch_size=32, shuffle=False)
    _bounded(nb.train_one_epoch, model, loader, opt)
    ref = copy.deepcopy(model)
    ref.zero_grad()
    F.cross_entropy(ref(torch.tensor(X[32:], dtype=torch.float32)), torch.tensor(y[32:])).backward()
    got, want = model[0].weight.grad, ref[0].weight.grad
    assert got is not None and torch.allclose(got, want, atol=1e-6), (
        "after an epoch of 2 batches, .grad should hold the LAST batch's gradient only. it holds something else "
        f"(|grad| = {0 if got is None else got.norm().item():.4g}, last batch alone: {want.norm().item():.4g}): "
        "gradients from earlier steps are piling up, because backward() ADDS to .grad")


def test_stage3_clipping_bounds_the_step(nb):
    model = _small_net(1)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(30.0)  # huge weights: huge logits, huge gradients
    X, y = _digits(32, seed=3)
    before = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    _bounded(nb.train_one_epoch, model, make_loader(X, y, batch_size=32, shuffle=False), opt, grad_clip=1.0)
    step = torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before))).item()
    assert step <= 0.1 * 1.0 + 1e-5, (
        f"with grad_clip=1.0 and SGD at lr 0.1, one step can move the parameters by at most 0.1; it moved {step:.3g}. "
        "the clipping isn't acting on the gradient the optimizer uses: when is .grad filled in?")


def test_stage3_returns_the_mean_loss(nb):
    X, y = _digits(50, seed=4)
    model = _small_net(2)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    got = _bounded(nb.train_one_epoch, model, make_loader(X, y, batch_size=16, shuffle=False), opt)
    with torch.no_grad():
        want = F.cross_entropy(model(torch.tensor(X, dtype=torch.float32)), torch.tensor(y)).item()
    assert abs(got - want) < 1e-5, f"the epoch's mean loss per example should be {want:.5f}; got {got:.5f}"
    probe = _Probe()
    net = torch.nn.Sequential(_small_net(3), probe)
    net.eval()  # e.g. left in eval mode by the previous epoch's evaluation
    _bounded(nb.train_one_epoch, net, make_loader(X, y, batch_size=25, shuffle=False), torch.optim.SGD(net.parameters(), lr=0.0))
    assert probe.train_flags and all(probe.train_flags), (
        "the epoch ran with the model in eval mode (dropout off): train_one_epoch must switch it to train mode first")


# ---------------------------------------------------------------- stage 4: evaluation and the fit loop


def test_stage4_evaluate_is_deterministic(nb):
    X, y = _digits(300, seed=5)
    model = _small_net(3, dropout=0.5)
    model.train()
    a = nb.evaluate(model, X, y)
    b = nb.evaluate(model, X, y)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
    want = (F.cross_entropy(out, torch.tensor(y)).item(), (out.argmax(1).numpy() == y).mean())
    assert np.allclose(a, b) and np.allclose(a, want, atol=1e-6), (
        f"evaluating the same model twice gave {a} and {b}; with dropout off it should give {want}. evaluation must "
        "run the model in eval mode (and under no_grad)")
    model.train()
    nb.evaluate(model, X, y)
    assert model.training, "evaluate restores the model's previous mode (fit keeps training afterwards)"
    probe = _Probe()
    nb.evaluate(torch.nn.Sequential(_small_net(4), probe), X[:50], y[:50])
    assert probe.grad_modes and not any(probe.grad_modes), (
        "evaluation should run under torch.no_grad(): no graph is needed, and building one wastes memory")


def test_stage4_learning_rate_schedule_counts_epochs(nb):
    X, y = _digits(400, seed=6)
    model = _small_net(4)
    h = _bounded(nb.fit, model, X[:320], y[:320], X[320:], y[320:], epochs=5, batch_size=32, lr=0.08, lr_step_epochs=2,
                 patience=100, seed=0)
    assert h["lr"] == pytest.approx([0.08, 0.08, 0.04, 0.04, 0.02]), (
        f"the learning rate at the start of each epoch should halve every lr_step_epochs = 2 EPOCHS: "
        f"[0.08, 0.08, 0.04, 0.04, 0.02]; got {[round(v, 6) for v in h['lr']]}. a scheduler defined in epochs is "
        "being stepped at some other frequency")
    X2, y2 = X[:96], y[:96]
    one = _small_net(7)
    _bounded(nb.fit, one, X2, y2, X[320:], y[320:], epochs=1, batch_size=32, lr=0.08, lr_step_epochs=1, patience=100,
             seed=0)
    ref = _small_net(7)
    _bounded(nb.train_one_epoch, ref, make_loader(X2, y2, batch_size=32, seed=0),
             torch.optim.SGD(ref.parameters(), lr=0.08, momentum=0.9), grad_clip=5.0)
    assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(one.parameters(), ref.parameters())), (
        "the first epoch should train at the full learning rate: the epoch-level scheduler steps AFTER each epoch's "
        "training, not before it")


def test_stage4_early_stopping_keeps_the_best_accuracy(nb):
    X, y = _digits(700, seed=7, noise=1.5)
    model = _small_net(5)  # no dropout, so evaluation is deterministic whatever evaluate does
    h = _bounded(nb.fit, model, X[:500], y[:500], X[500:], y[500:], epochs=8, batch_size=32, lr=0.02, lr_step_epochs=100,
                 patience=2, seed=0)
    acc = h["val_acc"]
    assert len(acc) >= 4 or max(acc) == acc[0], (
        f"training stopped after {len(acc)} epochs while the validation accuracy was still improving: {acc}")
    final = nb.evaluate(model, X[500:], y[500:])[1]
    assert abs(final - max(acc)) < 1e-9, (
        f"after fit, the model should hold the weights of its best validation-accuracy epoch ({max(acc):.4f}); it "
        f"scores {final:.4f} (accuracies by epoch: {[round(a, 4) for a in acc]}). which way does an accuracy improve, "
        "and which way does the early stopper think it does?")


def test_stage4_restores_the_best_epoch_weights(nb):
    X, y = _digits(500, seed=8)
    y_val = np.random.default_rng(9).integers(0, 10, 200)  # random validation labels: accuracy wanders, never trends
    found_mid_best = False
    for seed in range(3):
        model = _small_net(6 + seed)
        h = _bounded(nb.fit, model, X[:300], y[:300], X[300:], y_val, epochs=10, batch_size=32, lr=0.05,
                     lr_step_epochs=100, patience=2, seed=seed)
        best = int(np.argmax(h["val_acc"]))  # the first epoch with the best accuracy
        found_mid_best |= best < len(h["val_acc"]) - 1
        final_loss = nb.evaluate(model, X[300:], y_val)[0]
        assert abs(final_loss - h["val_loss"][best]) < 1e-6, (
            f"after fit the model should hold the weights saved at its best epoch ({best + 1}), whose validation loss "
            f"was {h['val_loss'][best]:.6f}; it has {final_loss:.6f} (losses by epoch: "
            f"{[round(v, 4) for v in h['val_loss']]}). were the best weights saved as a real copy, and restored?")
    assert found_mid_best, "(test setup) at least one run should have its best epoch before the last"


# ---------------------------------------------------------------- stage 5: the whole pipeline


def test_stage5_the_fixed_pipeline_learns(nb):
    accs = []
    for seed in range(3):
        X, y = _digits(CONFIG["n_train"] + CONFIG["n_val"], seed=10 + seed)
        X_test, y_test = _digits(500, seed=50 + seed)
        model, history, X_test_n = _bounded(nb.run_pipeline, X, y, X_test, config=CONFIG, seed=seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.as_tensor(X_test_n, dtype=torch.float32)).argmax(1).numpy()
        accs.append(float(np.mean(pred == y_test)))
    assert np.median(accs) >= 0.9, (
        f"the whole pipeline should reach >= 90% test accuracy on synthetic digits (median of 3 seeds); got {accs}. "
        "if stages 1-4 pass, check how the pieces are wired together")
