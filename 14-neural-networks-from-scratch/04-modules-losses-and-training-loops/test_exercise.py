"""Checks for exercise.py (modules, losses & training loops).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Modules are checked
against hand-built examples and PyTorch's naming conventions; losses against hand computations,
finite differences and (when installed) torch.nn.functional; the training loop against an exact
full-batch gradient step and a small learning task aggregated over seeds. Training runs under
`mlfs.testing.bounded`, so a loop that never ends fails instead of hanging.
"""
import numpy as np
import pytest

from mlfs.numgrad import numerical_gradient
from mlfs.optim import SGD, Adam
from mlfs.tensor import Tensor, is_grad_enabled
from mlfs.testing import bounded


def _fit(nb, *args, budget=4_000_000, **kwargs):
    return bounded(nb.fit, *args, budget=budget, what="fit",
                   hint="loop over range(epochs), and over one epoch of minibatch_indices per epoch.", **kwargs)


def _two_layer(nb, rng=0):
    return nb.Sequential(nb.Linear(3, 4, rng=rng), nb.ReLU(), nb.Linear(4, 2, rng=rng + 1))


# ---------------------------------------------------------------- stage 1: Parameter and Module


def test_stage1_parameter(nb):
    a = np.array([1, 2, 3])
    p = nb.Parameter(a)
    assert isinstance(p, Tensor), "Parameter is a Tensor subclass, so all your ops work on it"
    assert p.requires_grad is True and p.grad is None and p.data.dtype == np.float64
    a[0] = 9
    assert p.data[0] == 1.0, "Parameter keeps its own copy of the data"


def _toy_module(nb):
    class Block(nb.Module):
        def __init__(self):
            super().__init__()
            self.w = nb.Parameter(np.ones(2))

        def forward(self, x):
            return x * self.w

    class Net(nb.Module):
        def __init__(self):
            super().__init__()
            self.scale = nb.Parameter(np.array(2.0))
            self.note = np.zeros(3)  # a plain array: not a parameter
            self.first = Block()
            self.blocks = [Block(), Block()]
            self.pair = (Block(),)
            self.tied = self.first.w  # the same Parameter again: listed once
            self.count = 7

        def forward(self, x, *, k=1.0):
            for b in [self.first, *self.blocks]:
                x = b(x)
            return x * self.scale * k

    return Net()


def test_stage1_parameter_discovery(nb):
    net = _toy_module(nb)
    names = [n for n, _ in net.named_parameters()]
    want = ["scale", "first.w", "blocks.0.w", "blocks.1.w", "pair.0.w"]
    assert names == want, (
        f"named_parameters should find Parameters, sub-Modules and lists/tuples of Modules, in assignment order, "
        f"with dotted names, each Parameter once (self.tied is first.w again): want {want}, got {names}")
    params = net.parameters()
    assert isinstance(params, list) and len(params) == 5 and params[1] is net.first.w, (
        "parameters() is the list of the Parameter objects themselves (not copies)")
    out = net(Tensor(np.array([1.0, 2.0])), k=0.5)
    assert np.allclose(out.data, [1.0, 2.0]), "__call__ passes positional and keyword arguments to forward"


def test_stage1_train_eval_zero_grad(nb):
    net = _toy_module(nb)
    assert net.training is True and net.blocks[1].training is True, "modules start in training mode"
    r = net.eval()
    assert r is net, "train()/eval() return the module itself (so you can write model.eval()(x))"
    assert net.training is False and net.first.training is False and net.blocks[1].training is False \
        and net.pair[0].training is False, "eval() switches every sub-module, including those in lists and tuples"
    net.train()
    assert net.blocks[0].training is True
    for p in net.parameters():
        p.grad = np.ones_like(p.data)
    net.zero_grad()
    assert all(p.grad is None for p in net.parameters()), "zero_grad sets every .grad to None"


def test_stage1_state_dict(nb):
    net = _toy_module(nb)
    sd = net.state_dict()
    assert list(sd) == [n for n, _ in net.named_parameters()], "state_dict keys are the dotted parameter names"
    sd["scale"][...] = 100.0
    assert net.scale.data == 2.0, "state_dict returns copies: editing them doesn't touch the model"
    other = _toy_module(nb)
    for p in other.parameters():
        p.data[...] = 5.0
    before = other.first.w.data
    other.load_state_dict(net.state_dict())
    assert other.first.w.data is before, (
        "load_state_dict copies into the existing arrays IN PLACE (p.data[...] = value), so anything holding "
        "the parameter keeps seeing the same array")
    assert np.allclose(other.first.w.data, 1.0) and np.allclose(other.scale.data, 2.0)
    bad = net.state_dict()
    bad["first.w"] = np.ones(1)  # would broadcast silently into shape (2,)
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    missing = net.state_dict()
    del missing["scale"]
    with pytest.raises(KeyError):
        other.load_state_dict(missing)
    extra = net.state_dict()
    extra["nonexistent.weight"] = np.zeros(2)
    with pytest.raises(KeyError):
        other.load_state_dict(extra)  # an unexpected name usually means the wrong checkpoint: refuse it


# ---------------------------------------------------------------- stage 2: layers


def test_stage2_linear_layout_and_init(nb):
    layer = nb.Linear(400, 300, rng=0)
    W, b = layer.weight, layer.bias
    assert isinstance(W, nb.Parameter) and W.shape == (300, 400) and b.shape == (300,), (
        f"torch layout: weight (out_features, in_features) = (300, 400), bias (300,); got {W.shape}, "
        f"{None if b is None else b.shape}")
    assert [n for n, _ in layer.named_parameters()] == ["weight", "bias"]
    bound = 1 / np.sqrt(400)
    assert np.abs(W.data).max() <= bound and np.abs(b.data).max() <= bound, "init is U(-1/sqrt(in), 1/sqrt(in))"
    assert abs(W.data.std() - bound / np.sqrt(3)) < 0.05 * bound and abs(W.data.mean()) < 0.02 * bound, (
        f"a uniform U(-k, k) has std k/sqrt(3) = {bound / np.sqrt(3):.4f}; got {W.data.std():.4f}")
    again = nb.Linear(400, 300, rng=0)
    assert np.array_equal(again.weight.data, W.data), "the same rng seed gives the same weights"
    small = nb.Linear(3, 2, rng=5)
    r = np.random.default_rng(5)
    k = 1 / np.sqrt(3)
    W_want, b_want = r.uniform(-k, k, size=(2, 3)), r.uniform(-k, k, size=2)
    assert np.allclose(small.weight.data, W_want) and np.allclose(small.bias.data, b_want), (
        "draw order: rng = as_rng(rng); weight = rng.uniform(-k, k, (out, in)) FIRST, then bias = rng.uniform(-k, k, "
        "out), with k = 1/sqrt(in) (the bias is random too, not zero)")
    nb_bias = nb.Linear(4, 2, bias=False, rng=0)
    assert nb_bias.bias is None and len(nb_bias.parameters()) == 1


def test_stage2_linear_forward_and_grad(nb):
    rng = np.random.default_rng(0)
    layer = nb.Linear(3, 2, rng=1)
    X = rng.normal(size=(5, 3))
    out = layer(X)
    assert isinstance(out, Tensor) and out.shape == (5, 2), "Linear accepts an ndarray and returns a Tensor (5, 2)"
    want = X @ layer.weight.data.T + layer.bias.data
    assert np.allclose(out.data, want), "Linear computes x @ W.T + b"
    G = rng.normal(size=(5, 2))
    out.backward(G)
    assert np.allclose(layer.weight.grad, G.T @ X) and np.allclose(layer.bias.grad, G.sum(0)), (
        "dL/dW = G^T X (shape (out, in)) and dL/db = the column sums of G")
    x = Tensor(X, requires_grad=True)
    layer(x).backward(G)
    assert np.allclose(x.grad, G @ layer.weight.data), "a Tensor input gets its gradient G W"


def test_stage2_activations(nb):
    x = np.array([[-2.0, 0.0, 3.0]])
    for cls, f in [(nb.ReLU, lambda v: np.maximum(v, 0)), (nb.Tanh, np.tanh), (nb.Sigmoid, lambda v: 1 / (1 + np.exp(-v)))]:
        m = cls()
        out = m(Tensor(x))
        assert isinstance(m, nb.Module) and m.parameters() == [], f"{cls.__name__} is a Module with no parameters"
        assert np.allclose(out.data, f(x)), f"{cls.__name__} values"
        assert np.allclose(m(x).data, f(x)), f"{cls.__name__} accepts an ndarray too"


def test_stage2_sequential(nb):
    net = _two_layer(nb)
    assert len(net) == 3 and isinstance(net[0], nb.Linear) and isinstance(net[-1], nb.Linear)
    names = [n for n, _ in net.named_parameters()]
    assert names == ["0.weight", "0.bias", "2.weight", "2.bias"], (
        f"store layer i as attribute str(i) so names match torch's Sequential state_dict keys "
        f"(0.weight, 0.bias, 2.weight, 2.bias); got {names}")
    X = np.random.default_rng(0).normal(size=(4, 3))
    h = np.maximum(X @ net[0].weight.data.T + net[0].bias.data, 0)
    want = h @ net[2].weight.data.T + net[2].bias.data
    assert np.allclose(net(X).data, want), "Sequential applies its layers in order"
    head = net[:2]
    assert isinstance(head, nb.Sequential) and len(head) == 2 and head[0] is net[0], (
        "slicing gives a new Sequential sharing the same layer objects")
    assert np.allclose(head(X).data, h)
    net.eval()
    assert net[0].training is False, "eval() reaches the layers inside a Sequential"


def test_stage2_matches_torch_linear(nb):
    torch = pytest.importorskip("torch")
    net = _two_layer(nb)
    tnet = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).double()
    assert list(net.state_dict()) == list(tnet.state_dict()), "same parameter names as torch.nn.Sequential"
    tnet.load_state_dict({k: torch.tensor(v) for k, v in net.state_dict().items()})
    X = np.random.default_rng(1).normal(size=(6, 3))
    assert np.allclose(net(X).data, tnet(torch.tensor(X)).detach().numpy()), "same outputs as torch with the same weights"


# ---------------------------------------------------------------- stage 3: losses


def test_stage3_mse_loss(nb):
    pred = Tensor(np.array([[1.0], [2.0], [4.0]]), requires_grad=True)
    loss = nb.mse_loss(pred, np.array([[0.0], [2.0], [1.0]]))
    assert np.isclose(loss.item(), (1 + 0 + 9) / 3), f"mean of (1, 0, 9) is 10/3, got {loss.item()}"
    loss.backward()
    assert np.allclose(pred.grad, [[2 / 3], [0.0], [2.0]]), f"d/dpred = 2 (pred - target) / n, got {pred.grad.ravel()}"
    assert np.isclose(nb.mse_loss(pred, Tensor([[0.0], [2.0], [1.0]])).item(), 10 / 3), "target may be a Tensor"
    assert np.isclose(nb.MSELoss()(pred, np.array([[0.0], [2.0], [1.0]])).item(), 10 / 3), "MSELoss is the module form"
    with pytest.raises(ValueError):
        nb.mse_loss(pred, np.array([0.0, 2.0, 1.0]))  # (3, 1) vs (3,): would broadcast to (3, 3)
    wide = nb.mse_loss(Tensor(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 4.0]])), np.zeros((2, 3)))
    assert np.isclose(wide.item(), 30 / 6), f"mean over ALL 6 entries of a (2, 3) prediction: 30 / 6 = 5, got {wide.item()}"


def test_stage3_cross_entropy(nb):
    logits = Tensor(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]), requires_grad=True)
    y = np.array([0, 1, 1])
    loss = nb.cross_entropy(logits, y)
    want = np.mean([np.log(2), np.log(1 + np.exp(2)), np.log(1 + np.exp(-3))])
    assert np.isclose(loss.item(), want), f"cross-entropy = mean of logsumexp - logit of the true class = {want:.5f}, got {loss.item()}"
    loss.backward()
    p = np.exp(logits.data) / np.exp(logits.data).sum(1, keepdims=True)
    assert np.allclose(logits.grad, (p - np.eye(2)[y]) / 3), "the gradient is (softmax - one-hot) / n"
    assert np.isclose(nb.CrossEntropyLoss()(logits, y).item(), want)
    huge = Tensor(np.array([[1000.0, 0.0, -1000.0]]), requires_grad=True)
    with np.errstate(over="raise", invalid="raise"):
        try:
            big = nb.cross_entropy(huge, np.array([1]))
        except FloatingPointError as e:
            raise AssertionError(f"cross_entropy overflowed ({e}): use logsumexp, never exp(logits) directly") from None
    assert np.isclose(big.item(), 1000.0), f"logits (1000, 0, -1000), true class 1: loss = 1000, got {big.item()}"


def test_stage3_binary_cross_entropy_with_logits(nb):
    z = Tensor(np.array([0.0, 2.0, -1.0, 1000.0, -1000.0]), requires_grad=True)
    y = np.array([1, 0, 0, 1, 1])
    with np.errstate(over="raise", invalid="raise"):
        try:
            loss = nb.binary_cross_entropy_with_logits(z, y)
        except FloatingPointError as e:
            raise AssertionError(f"overflow ({e}): softplus(z) = log(1 + e^z) must be computed stably") from None
    terms = [np.log(2), np.log(1 + np.exp(2)), np.log(1 + np.exp(-1)), 0.0, 1000.0]
    assert np.isclose(loss.item(), np.mean(terms)), f"mean of softplus(z) - y z = {np.mean(terms):.5f}, got {loss.item()}"
    loss.backward()
    s = 1 / (1 + np.exp(-np.clip(z.data, -500, 500)))
    assert np.allclose(z.grad, (s - y) / 5), f"the gradient is (sigmoid(z) - y) / n, got {z.grad}"


def test_stage3_losses_gradcheck(nb):
    rng = np.random.default_rng(3)
    Z, y = rng.normal(size=(6, 4)), rng.integers(0, 4, 6)

    def f(arr):
        return nb.cross_entropy(Tensor(arr), y).item()

    t = Tensor(Z, requires_grad=True)
    nb.cross_entropy(t, y).backward()
    assert np.allclose(t.grad, numerical_gradient(f, Z), atol=1e-7), "cross_entropy's gradient disagrees with finite differences"
    z1, y1 = rng.normal(size=7) * 3, rng.integers(0, 2, 7)
    t1 = Tensor(z1, requires_grad=True)
    nb.binary_cross_entropy_with_logits(t1, y1).backward()
    g_fd = numerical_gradient(lambda a: nb.binary_cross_entropy_with_logits(Tensor(a), y1).item(), z1)
    assert np.allclose(t1.grad, g_fd, atol=1e-7), "binary_cross_entropy_with_logits' gradient disagrees with finite differences"


def test_stage3_matches_torch(nb):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(4)
    Z, y = rng.normal(size=(8, 5)) * 4, rng.integers(0, 5, 8)
    assert np.isclose(nb.cross_entropy(Tensor(Z), y).item(), F.cross_entropy(torch.tensor(Z), torch.tensor(y)).item())
    z, t = rng.normal(size=8) * 4, rng.integers(0, 2, 8)
    want = F.binary_cross_entropy_with_logits(torch.tensor(z), torch.tensor(t, dtype=torch.float64)).item()
    assert np.isclose(nb.binary_cross_entropy_with_logits(Tensor(z), t).item(), want)


# ---------------------------------------------------------------- stage 4: the training loop


def test_stage4_full_batch_step_is_exact(nb):
    rng = np.random.default_rng(5)
    X, y = rng.normal(size=(10, 3)), rng.normal(size=(10, 1))
    model = nb.Linear(3, 1, rng=0)
    W0, b0 = model.weight.data.copy(), model.bias.data.copy()
    opt = SGD(model.parameters(), lr=0.1)
    hist = _fit(nb, model, nb.mse_loss, opt, X, y, epochs=2, batch_size=10, shuffle=False)
    W, b = W0, b0
    losses = []
    for _ in range(2):
        r = X @ W.T + b - y
        losses.append(np.mean(r ** 2))
        W, b = W - 0.1 * 2 * r.T @ X / 10, b - 0.1 * 2 * r.sum(0) / 10
    assert len(hist["train_loss"]) == 2 and hist["val_loss"] == [], (
        "history has one train loss per epoch, and an empty val_loss list without validation data")
    assert np.allclose(hist["train_loss"], losses), f"train losses should be {losses}, got {hist['train_loss']}"
    assert np.allclose(model.weight.data, W) and np.allclose(model.bias.data, b), (
        "with batch_size = n, each epoch is exactly one gradient step. if your weights are off, check that you "
        "call optimizer.zero_grad() before every backward (gradients accumulate otherwise)")


def test_stage4_epoch_means_and_validation(nb):
    rng = np.random.default_rng(6)
    X, y = rng.normal(size=(25, 2)), rng.normal(size=(25, 1))
    Xv, yv = rng.normal(size=(9, 2)), rng.normal(size=(9, 1))
    model = nb.Linear(2, 1, rng=1)
    seen = []

    def spy(pred, target):
        loss = nb.mse_loss(pred, target)
        seen.append((len(target), loss.item(), is_grad_enabled(), model.training))
        return loss

    opt = SGD(model.parameters(), lr=0.0)  # lr 0: the model never changes, so losses are easy to check
    model.eval()  # fit must switch it back to training mode itself
    hist = _fit(nb, model, spy, opt, X, y, epochs=1, batch_size=10, X_val=Xv, y_val=yv, rng=0)
    train = [s for s in seen if s[2]]
    sizes = sorted(s[0] for s in train)
    assert sizes == [5, 10, 10], f"25 examples in batches of 10: batch sizes 10, 10, 5; got {sizes}"
    full = np.mean((X @ model.weight.data.T + model.bias.data - y) ** 2)
    assert np.isclose(hist["train_loss"][0], full), (
        f"the epoch's train loss is the mean over all 25 examples ({full:.5f}): weight each batch's mean loss by "
        f"its size. got {hist['train_loss'][0]:.5f}")
    assert all(s[3] for s in train), "fit calls model.train() at the start of every epoch (validation switched it to eval)"
    val = [s for s in seen if not s[2]]
    assert val and all(not s[3] for s in val), "validation runs under no_grad and in eval mode"
    want_val = np.mean((Xv @ model.weight.data.T + model.bias.data - yv) ** 2)
    assert np.isclose(hist["val_loss"][0], want_val) and np.isclose(nb.evaluate(model, nb.mse_loss, Xv, yv), want_val)


def test_stage4_callback_stops_early(nb):
    rng = np.random.default_rng(7)
    X, y = rng.normal(size=(20, 2)), rng.normal(size=(20, 1))
    model = nb.Linear(2, 1, rng=0)
    calls = []

    def cb(epoch, history):
        calls.append((epoch, len(history["train_loss"]), len(history["val_loss"])))
        return epoch == 2

    hist = _fit(nb, model, nb.mse_loss, SGD(model.parameters(), lr=0.01), X, y, epochs=50, batch_size=8, rng=0,
                X_val=X[:5], y_val=y[:5], callback=cb)
    assert calls == [(0, 1, 1), (1, 2, 2), (2, 3, 3)] and len(hist["train_loss"]) == 3, (
        f"callback(epoch, history) runs after each epoch, once history has that epoch's train AND validation loss, "
        f"and True stops training; calls were (epoch, len(train_loss), len(val_loss)) = {calls}")


def test_stage4_minibatch_order(nb):
    X = np.arange(12.0)[:, None]
    y = np.arange(12.0)[:, None]
    model = nb.Linear(1, 1, rng=0)
    batches = []

    def spy(pred, target):
        if is_grad_enabled():
            batches.append(np.asarray(getattr(target, "data", target)).ravel().astype(int).tolist())
        return nb.mse_loss(pred, target)

    _fit(nb, model, spy, SGD(model.parameters(), lr=0.0), X, y, epochs=2, batch_size=4, shuffle=False)
    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]] * 2, (
        f"shuffle=False: batches in order, every epoch; got {batches}")
    batches.clear()
    _fit(nb, model, spy, SGD(model.parameters(), lr=0.0), X, y, epochs=2, batch_size=4, rng=3)
    r = np.random.default_rng(3)
    want = [p[i:i + 4].tolist() for p in (r.permutation(12), r.permutation(12)) for i in range(0, 12, 4)]
    assert batches == want, (
        "shuffle=True: turn rng into ONE generator (as_rng) before the epoch loop and pass that generator to "
        "minibatch_indices every epoch, so each epoch gets a new permutation from the same stream "
        f"(passing the int seed again would repeat the first permutation). want {want}, got {batches}")


def test_stage4_predict_and_evaluate(nb):
    rng = np.random.default_rng(8)
    model = _two_layer(nb)
    X = rng.normal(size=(70, 3))
    modes = []

    class Probe(nb.Module):
        def forward(self, x):
            modes.append(self.training)
            return x

    probed = nb.Sequential(Probe(), *[model[i] for i in range(len(model))])
    out = nb.predict(probed, X, batch_size=32)
    assert modes and not any(modes), "predict runs the model in eval mode (every layer's .training is False)"
    assert isinstance(out, np.ndarray) and out.shape == (70, 2), "predict returns one ndarray (n, outputs)"
    assert np.allclose(out, model(X).data), "predict gives the same outputs as calling the model"
    assert probed.training is True and model[0].training is True, (
        "predict switches to eval mode while it runs, then restores the previous mode (on every layer)")
    y = rng.integers(0, 2, 70)
    want = nb.cross_entropy(model(X), y).item()
    assert np.isclose(nb.evaluate(model, nb.cross_entropy, X, y, batch_size=16), want), (
        "evaluate is the mean loss over all examples (batch means weighted by batch size)")
    assert all(p.grad is None for p in model.parameters()), "evaluate and predict never touch gradients"


def test_stage4_learns_a_nonlinear_task(nb):
    accs = []
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        X = rng.uniform(-1, 1, size=(200, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(np.int64)  # XOR of the signs: not linearly separable
        model = nb.Sequential(nb.Linear(2, 16, rng=seed), nb.Tanh(), nb.Linear(16, 2, rng=seed + 50))
        hist = _fit(nb, model, nb.CrossEntropyLoss(), Adam(model.parameters(), lr=0.03), X, y, epochs=60,
                    batch_size=32, rng=seed, budget=20_000_000)
        accs.append(np.mean(nb.predict(model, X).argmax(1) == y))
    assert hist["train_loss"][-1] < hist["train_loss"][0], "training should lower the loss"
    assert np.median(accs) > 0.9, (
        f"a 2-16-2 tanh network trained with Adam should fit the sign-XOR task (median accuracy over 5 seeds > 0.9); "
        f"got {np.round(accs, 3)}")
