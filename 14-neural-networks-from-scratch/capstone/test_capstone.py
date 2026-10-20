"""Checks for capstone.py (MNIST with your own framework).

Small synthetic digit-like data only (no downloads): blurry class templates plus noise, as uint8
28 x 28 images. Preprocessing and splitting are checked exactly; the network's structure against
the spec; accuracy and its bootstrap interval on models whose answers are known; training by a
validation-accuracy threshold aggregated over seeds and by the best-weights rule. Training runs
under `mlfs.testing.bounded`.
"""
import numpy as np

from mlfs import nn
from mlfs.testing import bounded


def _digits(n, seed=0, noise=0.8):
    """synthetic 'digits': 10 smooth random templates (28 x 28) plus noise, as uint8 images, and labels."""
    rng = np.random.default_rng(1234)  # the templates are fixed; only the samples depend on seed
    t = rng.normal(size=(10, 28, 28))
    for axis in (1, 2):
        for _ in range(3):
            t = (t + np.roll(t, 1, axis=axis) + np.roll(t, -1, axis=axis)) / 3
    t /= t.std()
    r = np.random.default_rng(seed)
    y = r.integers(0, 10, n)
    x = t[y] + noise * r.normal(size=(n, 28, 28))
    return np.clip(127.5 + 50 * x, 0, 255).astype(np.uint8), y.astype(np.int64)


def _train(nb, *args, **kwargs):
    return bounded(nb.train_classifier, *args, budget=40_000_000, what="train_classifier",
                   hint="call fit once with epochs=epochs; don't loop until an accuracy is reached.", **kwargs)


# ---------------------------------------------------------------- stage 1: data


def test_stage1_preprocess(nb):
    imgs, _ = _digits(5)
    X = nb.preprocess(imgs)
    assert isinstance(X, np.ndarray) and X.shape == (5, 784) and X.dtype == np.float64, (
        f"(n, 28, 28) uint8 images become a float64 (n, 784) matrix; got {getattr(X, 'shape', None)}, "
        f"{getattr(X, 'dtype', None)}")
    assert np.allclose(X, imgs.reshape(5, -1) / 255.0), "pixel values are divided by 255 (so they lie in [0, 1])"
    assert np.allclose(nb.preprocess(imgs.reshape(5, -1)), X), "already-flat (n, 784) images work too"
    assert nb.preprocess(np.array([[[255] * 28] * 28], dtype=np.uint8)).max() == 1.0, (
        "uint8 arithmetic overflows: convert to float before dividing")


def test_stage1_split_validation(nb):
    X = np.arange(100, dtype=float)[:, None] * np.ones((1, 3))
    y = np.arange(100)
    Xt, yt, Xv, yv = nb.split_validation(X, y, 20, rng=0)
    assert len(Xv) == len(yv) == 20 and len(Xt) == len(yt) == 80, "n_val rows for validation, the rest for training"
    assert np.array_equal(Xt[:, 0], yt) and np.array_equal(Xv[:, 0], yv), "rows and labels must stay aligned"
    assert sorted(np.concatenate([yt, yv]).tolist()) == list(range(100)), "every row lands in exactly one part"
    assert not np.array_equal(yv, np.arange(20)), "the split is random (a permutation), not the first n_val rows"
    again = nb.split_validation(X, y, 20, rng=0)
    assert np.array_equal(again[3], yv), "the same rng seed gives the same split"
    perm = np.random.default_rng(0).permutation(100)
    assert np.array_equal(yv, perm[:20]), "validation = the first n_val indices of rng.permutation(n)"


# ---------------------------------------------------------------- stage 2: the network


def test_stage2_build_mlp(nb):
    net = nb.build_mlp((784, 64, 32, 10), rng=0)
    kinds = [type(net[i]).__name__ for i in range(len(net))]
    assert kinds == ["Linear", "ReLU", "Linear", "ReLU", "Linear"], (
        f"784 -> 64 -> 32 -> 10 is Linear, ReLU, Linear, ReLU, Linear (no ReLU after the output); got {kinds}")
    assert [net[i].weight.shape for i in (0, 2, 4)] == [(64, 784), (32, 64), (10, 32)], "torch layout: (out, in)"
    n_params = sum(p.size for p in net.parameters())
    assert n_params == 785 * 64 + 65 * 32 + 33 * 10, f"parameter count should be {785 * 64 + 65 * 32 + 33 * 10}, got {n_params}"
    again = nb.build_mlp((784, 64, 32, 10), rng=0)
    assert all(np.array_equal(a.data, b.data) for a, b in zip(net.parameters(), again.parameters())), (
        "the same seed gives the same network")
    u1, u2 = net[0].weight.data[0, :10] * np.sqrt(784), net[2].weight.data[0, :10] * np.sqrt(64)  # undo the bounds
    assert not np.allclose(u1, u2), (
        "layers draw from ONE generator in turn: pass the same Generator to every Linear, not the same int seed")
    lin = nb.build_mlp((784, 10), rng=1)
    assert len(lin) == 1 and type(lin[0]).__name__ == "Linear", "sizes of length 2: a single Linear layer (softmax regression)"


# ---------------------------------------------------------------- stage 3: evaluation


def _oracle(n_classes=10):
    """a Linear model that returns its input as logits: predictions = argmax of the input row."""
    net = nn.Sequential(nn.Linear(n_classes, n_classes, rng=0))
    net[0].weight.data[...] = np.eye(n_classes)
    net[0].bias.data[...] = 0.0
    return net


def test_stage3_accuracy(nb):
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    X = np.eye(10)[[0, 1, 2, 3, 4, 5, 9, 9]]  # the last two predictions (class 9) are wrong
    got = nb.accuracy(_oracle(), X, y)
    assert isinstance(got, float) and np.isclose(got, 6 / 8), f"6 of 8 correct: accuracy 0.75, got {got}"


def test_stage3_accuracy_ci(nb):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 400)
    pred = np.where(np.arange(400) % 10 == 3, (y + 1) % 10, y)  # exactly 1 in 10 wrong, in every block of 100 too
    X = np.eye(10)[pred]
    acc, lo, hi = bounded(nb.accuracy_ci, _oracle(), X, y, n_boot=1000, rng=0, what="accuracy_ci")
    want = np.mean(pred == y)
    assert np.isclose(acc, want), f"the point estimate is the plain accuracy {want}, got {acc}"
    se = np.sqrt(want * (1 - want) / 400)
    assert lo < acc < hi and abs((hi - lo) - 2 * 1.96 * se) < 0.02, (
        f"a 95% interval for an accuracy near {want:.2f} on 400 examples should be about +-1.96 * {se:.3f} wide; "
        f"got [{lo:.3f}, {hi:.3f}]")
    acc2, lo2, hi2 = nb.accuracy_ci(_oracle(), X[:100], y[:100], n_boot=1000, rng=0)
    assert np.isclose(acc2, 0.9) and (hi2 - lo2) > 1.5 * (hi - lo), (
        "with 4 times fewer examples (same accuracy) the interval should be about twice as wide")


# ---------------------------------------------------------------- stage 4: training


def test_stage4_learns_synthetic_digits(nb):
    imgs, y = _digits(700, seed=1)
    X = nb.preprocess(imgs)
    accs = []
    for seed in range(3):
        Xt, yt, Xv, yv = nb.split_validation(X, y, 200, rng=seed)
        net = nb.build_mlp((784, 32, 10), rng=seed)
        hist = _train(nb, net, Xt, yt, Xv, yv, epochs=4, batch_size=32, lr=1e-3, rng=seed)
        assert len(hist["train_loss"]) == 4 and len(hist["val_acc"]) == 4 and len(hist["val_loss"]) == 4, (
            "history has train_loss and val_loss from fit, plus one val_acc per epoch")
        accs.append(nb.accuracy(net, Xv, yv))
    assert np.median(accs) >= 0.9, (
        f"a 784-32-10 MLP should reach >= 90% validation accuracy on the synthetic digits in 4 epochs (median of 3 "
        f"seeds); got {np.round(accs, 3)}")


def test_stage4_keeps_the_best_weights(nb):
    # a validation set whose labels are shifted by one class: the better the model learns the real task, the WORSE
    # its validation accuracy gets, so the best epoch is an early one and keeping the last weights is wrong
    imgs, y = _digits(400, seed=3, noise=1.5)
    X = nb.preprocess(imgs)
    Xt, yt, Xv, yv = X[:250], y[:250], X[250:], (y[250:] + 1) % 10
    net = nb.build_mlp((784, 32, 10), rng=0)
    hist = _train(nb, net, Xt, yt, Xv, yv, epochs=5, batch_size=25, lr=3e-4, rng=0)
    final = nb.accuracy(net, Xv, yv)
    assert np.isclose(final, max(hist["val_acc"])), (
        f"after training, the model must hold the weights of its best epoch by validation accuracy "
        f"({max(hist['val_acc']):.3f}, epoch {int(np.argmax(hist['val_acc'])) + 1} of {hist['val_acc']}); it scores "
        f"{final:.3f}. snapshot state_dict() in the callback whenever val_acc improves, and load_state_dict() it at "
        "the end")
