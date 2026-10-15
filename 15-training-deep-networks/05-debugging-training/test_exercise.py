"""Checks for exercise.py (debugging training).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The helpers are
checked on hand examples; the logger against values recomputed directly from the model after real
optimizer steps; `summary` both on hand-crafted histories (one rule at a time, so each failure
message points at one rule) and on real healthy and broken training runs; `plot` by its structure.
"""
import math

import matplotlib
import numpy as np
import pytest

from mlfs import nn
from mlfs.optim import SGD, Adam, Param
from mlfs.tensor import Tensor

matplotlib.use("Agg")


def _mlp(seed=0, sizes=(4, 8, 8, 3)):
    rng = np.random.default_rng(seed)
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b, rng=rng))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _data(seed=0, n=64):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int)
    return X, y


def _train(nb, model, opt, X, y, steps, *, logger=None, zero_grad=True):
    logger = logger or nb.DiagnosticsLogger(model)
    for t in range(steps):
        if zero_grad:
            opt.zero_grad()
        loss = nn.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        logger.log(t, loss.item(), lr=opt.lr)
    return logger


# ---------------------------------------------------------------- stage 1: grad_norm, named_modules


def test_stage1_grad_norm(nb):
    a, b, c = Param(np.zeros(2)), Param(np.zeros((2, 2))), Param(np.zeros(1))
    a.grad, b.grad = np.array([1.0, 2.0]), np.array([[2.0, 0.0], [0.0, 4.0]])
    got = nb.grad_norm([a, b, c])
    assert isinstance(got, float) and math.isclose(got, 5.0), (
        f"global norm = sqrt(1 + 4 + 4 + 16) = 5 over every gradient entry, skipping None grads; got {got}")
    assert nb.grad_norm([c]) == 0.0, "no gradients at all: the norm is 0"
    assert math.isclose(nb.grad_norm(p for p in [a, b]), 5.0), "params may be a generator"


def test_stage1_named_modules(nb):
    net = _mlp()
    names = [n for n, _ in nb.named_modules(net)]
    assert names == ["", "0", "1", "2", "3", "4"], (
        f"the model itself first (name ''), then every sub-module with its dotted name, in assignment order; got {names}")
    assert nb.named_modules(net)[2][1] is net[1]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = nn.ReLU()
            self.lin = nn.Linear(2, 2, rng=0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = Block()
            self.blocks = [Block(), Block()]

    names = [n for n, _ in nb.named_modules(Net())]
    assert names == ["", "first", "first.act", "first.lin", "blocks.0", "blocks.0.act", "blocks.0.lin", "blocks.1",
                     "blocks.1.act", "blocks.1.lin"], (
        f"recurse depth first into sub-modules and lists of modules (name.i), like Module.named_parameters; got {names}")


# ---------------------------------------------------------------- stage 2: update_ratio, dead_fraction


def test_stage2_update_ratio(nb):
    got = nb.update_ratio(np.array([3.0, 4.0]), np.array([3.0, 4.5]))
    want = 0.5 / math.hypot(3.0, 4.5)
    assert math.isclose(got, want), (
        f"update_ratio = ||after - before|| / ||after|| = 0.5 / {math.hypot(3, 4.5):.4f} = {want:.5f}; got {got}")
    assert nb.update_ratio(np.zeros(3), np.zeros(3)) == 0.0, "no weights and no change: 0.0"
    assert nb.update_ratio(np.ones((2, 2)), np.ones((2, 2))) == 0.0
    assert math.isclose(nb.update_ratio(np.ones(4), 2 * np.ones(4)), 0.5), "a weight that doubles changed by half its new size"


def test_stage2_dead_fraction(nb):
    a = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 2.0, 3.0, 0.0]])
    assert math.isclose(nb.dead_fraction(a), 0.5), (
        "units 0 and 3 are zero for EVERY example (dead), unit 2 is zero for one example only (alive): 2 of 4 dead")
    b = np.zeros((3, 2, 2))
    b[0, 1, 1] = 1.0
    assert math.isclose(nb.dead_fraction(b), 0.75), "extra axes count as units: (batch, 2, 2) has 4 units per example"
    assert nb.dead_fraction(np.ones((5, 3))) == 0.0


# ---------------------------------------------------------------- stage 3: DiagnosticsLogger.log


def test_stage3_logs_what_the_model_did(nb):
    model, (X, y) = _mlp(), _data()
    opt = SGD(model.parameters(), lr=0.1)
    logger = nb.DiagnosticsLogger(model)
    start = {n: p.data.copy() for n, p in model.named_parameters()}
    opt.zero_grad()
    loss = nn.cross_entropy(model(X), y)
    loss.backward()
    opt.step()
    logger.log(0, loss.item(), lr=0.1)
    h = logger.history
    assert h["step"] == [0] and math.isclose(h["loss"][0], loss.item()) and h["lr"] == [0.1], (
        "history['step'], ['loss'], ['lr'] are lists with one entry per log() call")
    assert math.isclose(h["grad_norm"][0], nb.grad_norm(model.parameters())), "history['grad_norm'] is the global norm"
    for name, p in model.named_parameters():
        assert math.isclose(h[f"grad/{name}"][0], np.linalg.norm(p.grad)), f"history['grad/{name}'] = ||grad||"
        assert math.isclose(h[f"weight/{name}"][0], np.linalg.norm(p.data)), f"history['weight/{name}'] = ||weight||"
        want = np.linalg.norm(p.data - start[name]) / np.linalg.norm(p.data)
        assert math.isclose(h[f"update/{name}"][0], want, rel_tol=1e-9), (
            f"the first update ratio compares with the weights as they were when the logger was created; "
            f"history['update/{name}']: want {want:.4g}, got {h[f'update/{name}'][0]:.4g}")
    snap = {n: p.data.copy() for n, p in model.named_parameters()}
    opt.zero_grad()
    loss = nn.cross_entropy(model(X), y)
    loss.backward()
    opt.step()
    logger.log(1, loss)
    w = model[0].weight.data
    assert math.isclose(h["update/0.weight"][1], np.linalg.norm(w - snap["0.weight"]) / np.linalg.norm(w), rel_tol=1e-9), (
        "each update ratio compares with the weights at the PREVIOUS log() call (keep a copy, not a reference)")
    assert math.isnan(h["lr"][1]), "lr=None is recorded as nan"
    assert isinstance(h["loss"][1], float), "a loss Tensor is recorded as a float (use .item())"


def test_stage3_watches_relus_in_training_mode_only(nb):
    model = _mlp(sizes=(4, 6, 3))
    model[0].bias.data[:3] = -100.0  # units 0-2 of the hidden layer can never fire
    X, y = _data()
    logger = nb.DiagnosticsLogger(model)
    out = model(X)
    assert out.shape == (64, 3), "watching a ReLU must not change what the model computes"
    want = nn.Sequential(model[0], model[1])(X).data
    assert np.allclose(model[1](model[0](X)).data, want)
    loss = nn.cross_entropy(out, y)
    loss.backward()
    logger.log(0, loss.item())
    assert "dead/1" in logger.history and math.isclose(logger.history["dead/1"][0], 0.5), (
        "the ReLU at index 1 of the Sequential: 3 of its 6 units are dead, so history['dead/1'] = [0.5]. wrap each "
        "ReLU's forward in __init__ to record dead_fraction of its output")
    model.eval()
    model[0].bias.data[:] = -100.0
    model(X)  # an evaluation pass: not recorded
    model.train()
    logger.log(1, loss.item())
    assert math.isclose(logger.history["dead/1"][1], 0.5), (
        "forward passes in eval mode (validation, predict) aren't recorded: only training batches count")
    plain = nn.Sequential(nn.Linear(2, 2, rng=0))
    lg = nb.DiagnosticsLogger(plain)
    lg.log(0, 1.0)
    assert not any(k.startswith("dead/") for k in lg.history), "no ReLUs, no dead/ keys"


def test_stage3_no_grad_parameters(nb):
    model = _mlp()
    logger = nb.DiagnosticsLogger(model)
    logger.log(0, 2.0)  # before any backward: every grad is None
    assert logger.history["grad_norm"] == [0.0] and math.isnan(logger.history["grad/0.weight"][0]), (
        "a parameter without a gradient is recorded as nan (and doesn't count towards the global norm)")


# ---------------------------------------------------------------- stage 4: summary and plot


def _crafted(nb, n=40, **override):
    """a logger whose history is replaced by a hand-made healthy one, with some series overridden."""
    model = _mlp(sizes=(4, 8, 3))
    lg = nb.DiagnosticsLogger(model)
    names = [nm for nm, _ in model.named_parameters()]
    h = {"step": list(range(n)), "loss": list(np.linspace(2.0, 0.5, n)), "lr": [0.1] * n, "grad_norm": [1.0] * n}
    for nm in names:
        h[f"grad/{nm}"] = [0.5] * n
        h[f"weight/{nm}"] = [3.0] * n
        h[f"update/{nm}"] = [1e-3] * n
    h["dead/1"] = [0.1] * n
    h.update(override)
    lg.history = h
    return lg


def _tags(warnings):
    """the tag of each warning: the known tag it starts with (else the text before the first colon)."""
    known = ["non-finite", "exploding", "vanishing", "update ratio", "dead relu", "plateau"]
    return [next((k for k in known if w.startswith(k)), w.split(":")[0]) for w in warnings]


def test_stage4_summary_rules_one_at_a_time(nb):
    healthy = _crafted(nb).summary()
    assert healthy == [], f"a healthy history (falling loss, steady gradients, ratios ~1e-3) gives no warnings; got {healthy}"
    cases = {
        "non-finite": dict(loss=[2.0] * 20 + [math.nan] * 20),
        "exploding": dict(grad_norm=[1.0] * 39 + [500.0]),
        "vanishing": {"grad/0.weight": [1e-5] * 40},
        "update ratio": {"update/2.weight": [0.5] * 40},
        "dead relu": {"dead/1": [0.1] * 39 + [0.8]},
        "plateau": dict(loss=[2.3] * 40),
    }
    for tag, override in cases.items():
        got = _tags(_crafted(nb, **override).summary())
        assert tag in got, (
            f"this history has the '{tag}' problem ({list(override)} overridden), so summary() should include a warning "
            f"starting with '{tag}'; got tags {got}. see the lesson's table of rules and thresholds")
    spike = _tags(_crafted(nb, grad_norm=[1.0] + [0.001] * 38 + [50.0]).summary())
    assert "exploding" in spike, (
        "the gradient norm jumps to 50 after 38 steps at 0.001: compare each norm with the MEDIAN of the norms before "
        f"it (0.001), not with the first one (1.0); got tags {spike}")
    quiet_bias = _tags(_crafted(nb, **{"grad/0.bias": [1e-6] * 40, "grad/2.bias": [1e-6] * 40}).summary())
    assert "vanishing" not in quiet_bias, (
        "biases' gradients are naturally small: the vanishing rule looks at weight matrices (2+ dimensions) only")
    small = _crafted(nb, **{"update/0.weight": [1e-7] * 40}).summary()
    assert any(w.startswith("update ratio") and "small" in w for w in small), (
        "a median update ratio below 1e-5 means the learning rate is far too small: 'update ratio too small: ...'")
    assert nb.DiagnosticsLogger(_mlp()).summary() == [], "nothing logged yet: no warnings"


def test_stage4_summary_on_real_runs(nb):
    X, y = _data(1, n=128)
    healthy = _train(nb, m := _mlp(1, sizes=(4, 32, 32, 3)), Adam(m.parameters(), lr=1e-2), X, y, 60).summary()
    assert healthy == [], f"a healthy Adam run on an easy task should give no warnings; got {healthy}"
    dead = _mlp(2)
    dead[0].bias.data[...] = -50.0
    dead[2].bias.data[...] = -50.0
    tags = _tags(_train(nb, dead, SGD(dead.parameters(), lr=0.1), X, y, 30).summary())
    assert "dead relu" in tags, f"hidden biases of -50 kill every ReLU: expected a 'dead relu' warning, got {tags}"
    slow = _mlp(3)
    tags = _train(nb, slow, SGD(slow.parameters(), lr=1e-7), X, y, 30).summary()
    assert any(w.startswith("update ratio") for w in tags) and "plateau" in _tags(tags), (
        f"lr = 1e-7 barely moves anything: expected 'update ratio too small' and 'plateau'; got {tags}")


def test_stage4_plot(nb):
    X, y = _data()
    lg = _train(nb, m := _mlp(), SGD(m.parameters(), lr=0.1), X, y, 5)
    fig = lg.plot()
    assert isinstance(fig, matplotlib.figure.Figure), "plot() returns a matplotlib Figure"
    assert len(fig.axes) >= 4, f"four panels: loss, gradient norms, update ratios, dead fractions; got {len(fig.axes)}"
    matplotlib.pyplot.close(fig)
