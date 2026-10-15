"""Checks for exercise.py (from linear models to neurons).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Layers and networks are
checked against hand examples and NumPy; the hand-wired XOR network on the four corners; the
hand-derived gradients against central differences (your `mlfs.numgrad`, the reference under
pytest); training against an exact first step and a learning check over several seeds. Training
runs under `mlfs.testing.bounded`, so a loop that never ends fails instead of hanging.
"""
import warnings

import numpy as np

from mlfs.numgrad import numerical_gradient
from mlfs.testing import bounded

X_XOR = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_XOR = np.array([0.0, 1.0, 1.0, 0.0])


def _train(nb, *args, **kwargs):
    return bounded(nb.train_two_layer, *args, budget=3_000_000, what="train_two_layer",
                   hint="run exactly n_steps gradient steps.", **kwargs)


def _random_params(rng, d=3, hidden=4):
    return [(rng.normal(size=(hidden, d)), rng.normal(size=hidden)), (rng.normal(size=(1, hidden)), rng.normal(size=1))]


# ---------------------------------------------------------------- stage 1: layers and activations


def test_stage1_linear_layer(nb):
    X = np.array([[1.0, 2.0], [3.0, 4.0], [0.0, -1.0]])
    W = np.array([[1.0, 0.0], [2.0, -1.0], [0.5, 0.5]])  # (out=3, in=2)
    b = np.array([0.0, 1.0, -1.0])
    out = nb.linear_layer(X, W, b)
    want = np.array([[1.0, 1.0, 0.5], [3.0, 3.0, 2.5], [0.0, 2.0, -1.5]])
    assert np.shape(out) == (3, 3), (
        f"X (3, 2) with W (out=3, in=2) gives (3, 3); got {np.shape(out)}. the layout is torch's: W is (out, in), so "
        "compute X @ W.T + b")
    assert np.allclose(out, want), f"row i is W x_i + b: want {want.tolist()}, got {np.round(out, 3).tolist()}"


def test_stage1_activations(nb):
    z = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert np.allclose(nb.relu(z), [0, 0, 0, 0.5, 3]), "relu(z) = max(z, 0)"
    s = nb.sigmoid(z)
    assert np.allclose(s, 1 / (1 + np.exp(-z))) and np.isclose(s[2], 0.5)
    assert np.allclose(nb.sigmoid(-z), 1 - s), "sigmoid(-z) = 1 - sigmoid(z)"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            big = nb.sigmoid(np.array([-1000.0, 1000.0]))
        except RuntimeWarning as w:
            raise AssertionError(f"sigmoid overflowed ({w}): use np.logaddexp, or split on the sign of z") from None
    assert np.allclose(big, [0.0, 1.0]), f"sigmoid(-1000) = 0 and sigmoid(1000) = 1, got {big}"


# ---------------------------------------------------------------- stage 2: MLPs and collapse


def test_stage2_mlp_forward(nb):
    X = np.array([[1.0, -1.0], [2.0, 0.0]])
    params = [(np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 0.0]]), np.array([0.0, 0.0, -1.0])),
              (np.array([[1.0, -1.0, 2.0]]), np.array([-5.0]))]
    # hidden pre-activations: [[0, 2, -2], [2, 2, -3]] -> relu: [[0, 2, 0], [2, 2, 0]]; output: [[-7], [-5]]
    out = nb.mlp_forward(X, params)
    assert np.shape(out) == (2, 1) and np.allclose(out, [[-7.0], [-5.0]]), (
        f"hidden = relu(X W1^T + b1) = [[0, 2, 0], [2, 2, 0]], output = hidden W2^T + b2 = [[-7], [-5]]; got {out}. "
        "no activation after the last layer (the output can be negative)")
    out_tanh = nb.mlp_forward(X, params, np.tanh)
    h = np.tanh(np.array([[0.0, 2.0, -2.0], [2.0, 2.0, -3.0]]))
    assert np.allclose(out_tanh, h @ params[1][0].T + params[1][1]), "the activation argument replaces relu"
    one = nb.mlp_forward(X, params[:1])
    assert np.allclose(one, [[0.0, 2.0, -2.0], [2.0, 2.0, -3.0]]), "a single layer is just linear_layer (no activation)"


def test_stage2_collapse_linear(nb):
    rng = np.random.default_rng(0)
    for widths in [(3, 5, 2), (4, 8, 8, 6, 1), (2, 2), (3, 3, 3, 3)]:
        params = [(rng.normal(size=(o, i)), rng.normal(size=o)) for i, o in zip(widths[:-1], widths[1:])]
        W, b = nb.collapse_linear(params)
        assert np.shape(W) == (widths[-1], widths[0]) and np.shape(b) == (widths[-1],), (
            f"layers {widths} collapse to one layer W ({widths[-1]}, {widths[0]}), b ({widths[-1]},); "
            f"got {np.shape(W)}, {np.shape(b)}")
        X = rng.normal(size=(7, widths[0]))
        deep = nb.mlp_forward(X, params, lambda z: z)
        assert np.allclose(X @ W.T + b, deep), (
            "the collapsed layer must compute exactly what the stack does with no activation: W = WL ... W1 and "
            "b = WL(... (W2 b1 + b2) ...) + bL")


# ---------------------------------------------------------------- stage 3: XOR by hand


def test_stage3_xor_params(nb):
    params = nb.xor_params()
    (W1, b1), (W2, b2) = params
    assert np.shape(W1) == (2, 2) and np.shape(b1) == (2,) and np.shape(W2) == (1, 2) and np.shape(b2) == (1,), (
        "a 2-2-1 network: W1 (2, 2), b1 (2,), W2 (1, 2), b2 (1,)")
    out = np.asarray(nb.mlp_forward(X_XOR, params))
    assert out.shape == (4, 1) and np.allclose(out[:, 0], Y_XOR), (
        f"on the corners (0,0), (0,1), (1,0), (1,1) the network must output exactly 0, 1, 1, 0 (with relu); got "
        f"{np.round(out.ravel(), 3)}")
    W, b = nb.collapse_linear(params)
    lin = X_XOR @ W.T + b
    assert not np.allclose(lin[:, 0], Y_XOR), "without the relu, the same weights can't compute XOR"


# ---------------------------------------------------------------- stage 4: hand-derived gradients


def test_stage4_hand_example(nb):
    X = np.array([[1.0, 2.0], [0.0, -1.0], [3.0, 1.0], [2.0, 2.0]])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    zeros = [(np.zeros((3, 2)), np.zeros(3)), (np.zeros((1, 3)), np.zeros(1))]
    loss, grads = nb.two_layer_loss_and_grads(X, y, zeros)
    assert np.isclose(loss, np.log(2)), f"all-zero weights predict 1/2 everywhere: loss = log 2 = 0.693, got {loss}"
    (dW1, db1), (dW2, db2) = grads
    assert np.allclose(db2, [0.0]), f"db2 = mean(sigmoid(0) - y) = mean(0.5 - y) = 0 here, got {db2}"
    assert np.allclose(dW1, 0) and np.allclose(db1, 0) and np.allclose(dW2, 0), (
        "with W2 = 0 no gradient reaches the first layer, and with h = tanh(0) = 0, dW2 = 0 too: all-zero init is a "
        "saddle point")


def test_stage4_gradients_match_finite_differences(nb):
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(10, 3)), rng.integers(0, 2, 10).astype(float)
    for activation in ["tanh", "identity"]:
        params = _random_params(rng)
        loss, grads = nb.two_layer_loss_and_grads(X, y, params, activation)
        flat = [a for pair in params for a in pair]
        gflat = [g for pair in grads for g in pair]
        names = ["dW1", "db1", "dW2", "db2"]
        for k in range(4):
            def f(arr, k=k):
                p = [a.copy() for a in flat]
                p[k] = arr
                return nb.two_layer_loss_and_grads(X, y, [(p[0], p[1]), (p[2], p[3])], activation)[0]

            g_fd = numerical_gradient(f, flat[k])
            assert np.shape(gflat[k]) == np.shape(flat[k]), f"{names[k]} must have its parameter's shape {np.shape(flat[k])}"
            assert np.allclose(gflat[k], g_fd, atol=1e-7), (
                f"activation={activation}: {names[k]} disagrees with finite differences (largest gap "
                f"{np.max(np.abs(gflat[k] - g_fd)):.2e}). is the loss a MEAN over the n examples?")


def test_stage4_stable_for_large_logits(nb):
    X = np.array([[1.0], [-1.0]])
    y = np.array([0.0, 1.0])
    params = [(np.array([[1.0]]), np.array([0.0])), (np.array([[800.0]]), np.array([0.0]))]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            loss, grads = nb.two_layer_loss_and_grads(X, y, params)
        except FloatingPointError as e:
            raise AssertionError(f"overflow ({e}): compute the cross-entropy from the logit, log(1 + e^z) - y z, "
                                 "with np.logaddexp(0, z)") from None
    assert np.isfinite(loss) and loss > 100, f"two confidently wrong predictions: a huge but finite loss, got {loss}"


# ---------------------------------------------------------------- stage 5: training


def test_stage5_init_and_first_step(nb):
    X, y = X_XOR, Y_XOR
    params, losses = _train(nb, X, y, 3, lr=0.5, n_steps=1, rng=7)
    rng = np.random.default_rng(7)
    init = [(rng.normal(size=(3, 2)), rng.normal(size=3)), (rng.normal(size=(1, 3)), rng.normal(size=1))]
    loss0, grads = nb.two_layer_loss_and_grads(X, y, init)
    assert len(losses) == 1 and np.isclose(losses[0], loss0), (
        "init: draw W1 (hidden, d), b1, W2 (1, hidden), b2 from rng.normal in that order; losses[0] is the loss "
        "at the initial parameters, before the first step")
    for (W, b), (W0, b0), (dW, db) in zip(params, init, grads):
        assert np.allclose(W, W0 - 0.5 * dW) and np.allclose(b, b0 - 0.5 * db), "one step: param -= lr * grad"


def test_stage5_learns_xor(nb):
    solved, finals = 0, []
    for seed in range(5):
        params, losses = _train(nb, X_XOR, Y_XOR, 8, lr=1.0, n_steps=1500, rng=seed)
        assert len(losses) == 1500
        pred = nb.sigmoid(nb.mlp_forward(X_XOR, params, np.tanh))[:, 0]
        solved += np.all((pred > 0.5) == Y_XOR)
        finals.append(losses[-1])
    assert solved >= 4, (
        f"with 8 tanh hidden units, gradient descent should solve XOR from almost any init: solved {solved} of 5 "
        f"seeds (final losses {np.round(finals, 3)})")
    params, losses = _train(nb, X_XOR, Y_XOR, 8, lr=1.0, n_steps=1500, activation="identity", rng=0)
    pred = nb.sigmoid(nb.mlp_forward(X_XOR, params, lambda z: z))[:, 0]
    assert np.sum((pred > 0.5) == Y_XOR) <= 3 and losses[-1] > 0.4, (
        "with an identity activation the network is linear and can't fit XOR (at most 3 of 4 corners)")
