"""Checks for exercise.py (initialization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Fans and gains are
checked against hand values (and torch.nn.init when installed); initializers against the exact
draws of a seeded generator, their statistics, and in-place semantics; deep_mlp against its spec
and the variance-propagation law; activation_stats against a hand-built network.
"""
import numpy as np
import pytest

from mlfs import nn
from mlfs.tensor import Tensor, is_grad_enabled


# ---------------------------------------------------------------- stage 1: fans and gains


def test_stage1_fans_linear_and_conv(nb):
    assert tuple(nb.calculate_fan((256, 784))) == (784, 256), (
        "a Linear weight is (out, in) in torch layout, so (256, 784) has fan_in 784 and fan_out 256; "
        f"got {nb.calculate_fan((256, 784))}")
    assert tuple(nb.calculate_fan((64, 3, 3, 3))) == (27, 576), (
        "a conv weight (out=64, in=3, 3, 3): each output sees in * 3 * 3 = 27 inputs (fan_in), each input feeds "
        f"out * 3 * 3 = 576 outputs (fan_out); got {nb.calculate_fan((64, 3, 3, 3))}")
    assert tuple(nb.calculate_fan((5, 7, 2))) == (14, 10)
    fi, fo = nb.calculate_fan(np.zeros((3, 4)).shape)
    assert isinstance(fi, (int, np.integer)) and isinstance(fo, (int, np.integer)), "fans are integers"
    with pytest.raises(ValueError):
        nb.calculate_fan((10,))  # a bias has no fan: refuse it


def test_stage1_gains(nb):
    want = {"linear": 1.0, "sigmoid": 1.0, "tanh": 5 / 3, "relu": np.sqrt(2.0)}
    for name, g in want.items():
        got = nb.calculate_gain(name)
        assert np.isclose(got, g), f"calculate_gain({name!r}) should be {g:.4f}, got {got}"
    assert np.isclose(nb.calculate_gain("leaky_relu"), np.sqrt(2 / (1 + 0.01 ** 2))), (
        "leaky_relu's default negative slope is 0.01 (like torch): gain sqrt(2 / (1 + 0.01^2))")
    assert np.isclose(nb.calculate_gain("leaky_relu", 0.2), np.sqrt(2 / 1.04))
    with pytest.raises(ValueError):
        nb.calculate_gain("softmax")


def test_stage1_matches_torch(nb):
    torch = pytest.importorskip("torch")
    for shape in [(10, 20), (8, 4, 3, 3), (6, 2, 5)]:
        w = torch.empty(shape)
        assert tuple(nb.calculate_fan(shape)) == torch.nn.init._calculate_fan_in_and_fan_out(w), shape
    for name in ["linear", "sigmoid", "tanh", "relu", "leaky_relu"]:
        assert np.isclose(nb.calculate_gain(name), torch.nn.init.calculate_gain(name)), name


# ---------------------------------------------------------------- stage 2: xavier


def test_stage2_worked_example_bounds(nb):
    p = nn.Parameter(np.zeros((256, 784)))
    out = nb.xavier_uniform_(p, rng=0)
    a = np.sqrt(6 / (784 + 256))
    assert out is p, "initializers return the parameter they filled (like torch.nn.init's trailing-underscore functions)"
    assert np.abs(p.data).max() <= a and np.abs(p.data).max() > 0.99 * a, (
        f"xavier_uniform_ on (256, 784) draws from U(-a, a) with a = sqrt(6 / 1040) = {a:.5f}; "
        f"the largest |w| was {np.abs(p.data).max():.5f}")
    q = nn.Parameter(np.zeros((256, 784)))
    nb.xavier_normal_(q, rng=0)
    std = np.sqrt(2 / 1040)
    assert abs(q.data.std() - std) < 0.01 * std, f"xavier_normal_ std should be sqrt(2/1040) = {std:.5f}, got {q.data.std():.5f}"
    assert abs(p.data.std() - std) < 0.01 * std, "both xavier variants have the SAME variance 2/(fan_in + fan_out)"


def test_stage2_exact_draws_and_gain(nb):
    p = nn.Parameter(np.zeros((3, 5)))
    nb.xavier_uniform_(p, gain=2.0, rng=7)
    a = 2.0 * np.sqrt(6 / 8)
    want = np.random.default_rng(7).uniform(-a, a, size=(3, 5))
    assert np.allclose(p.data, want), (
        "draw the whole weight with ONE call: as_rng(rng).uniform(-a, a, size=shape), a = gain * sqrt(6 / (fan_in + fan_out))")
    q = nn.Parameter(np.zeros((4, 2)))
    nb.xavier_normal_(q, gain=0.5, rng=3)
    want = np.random.default_rng(3).normal(0.0, 0.5 * np.sqrt(2 / 6), size=(4, 2))
    assert np.allclose(q.data, want), "xavier_normal_: as_rng(rng).normal(0, gain * sqrt(2 / (fan_in + fan_out)), size=shape)"


def test_stage2_in_place(nb):
    layer = nn.Linear(6, 4, rng=0)
    before = layer.weight.data
    nb.xavier_normal_(layer.weight, rng=1)
    assert layer.weight.data is before, (
        "fill the existing array in place (arr[...] = draws), don't replace param.data: an optimizer may already "
        "hold that array")
    arr = np.zeros((4, 6))
    nb.xavier_uniform_(arr, rng=1)
    assert arr.std() > 0, "a plain ndarray works too, and is filled in place"
    g = np.random.default_rng(5)
    nb.xavier_uniform_(nn.Parameter(np.zeros((2, 2))), rng=g)
    x1 = nn.Parameter(np.zeros((2, 2)))
    nb.xavier_uniform_(x1, rng=g)
    ref = np.random.default_rng(5)
    ref.uniform(size=(2, 2))
    assert np.allclose(x1.data, ref.uniform(-np.sqrt(6 / 4), np.sqrt(6 / 4), size=(2, 2))), (
        "a Generator passed as rng is used as is (as_rng), so two calls continue the same stream")


# ---------------------------------------------------------------- stage 3: kaiming


def test_stage3_worked_example(nb):
    p = nn.Parameter(np.zeros((256, 784)))
    nb.kaiming_normal_(p, rng=0)
    std = np.sqrt(2 / 784)
    assert abs(p.data.std() - std) < 0.01 * std, (
        f"kaiming_normal_ for relu, fan_in mode: std = sqrt(2) / sqrt(784) = {std:.5f}; got {p.data.std():.5f}. "
        "(if you got sqrt(2/1040), you used xavier's fan_in + fan_out)")
    q = nn.Parameter(np.zeros((256, 784)))
    nb.kaiming_uniform_(q, rng=0)
    b = np.sqrt(6 / 784)
    assert np.abs(q.data).max() <= b and np.abs(q.data).max() > 0.99 * b, (
        f"kaiming_uniform_: bound sqrt(3) * sqrt(2/784) = sqrt(6/784) = {b:.5f}; largest |w| {np.abs(q.data).max():.5f}")


def test_stage3_exact_draws_modes_and_nonlinearities(nb):
    p = nn.Parameter(np.zeros((3, 12)))
    nb.kaiming_normal_(p, rng=4)
    assert np.allclose(p.data, np.random.default_rng(4).normal(0, np.sqrt(2) / np.sqrt(12), size=(3, 12))), (
        "kaiming_normal_: ONE call as_rng(rng).normal(0, gain / sqrt(fan), size=shape)")
    nb.kaiming_normal_(p, mode="fan_out", rng=4)
    assert np.allclose(p.data, np.random.default_rng(4).normal(0, np.sqrt(2) / np.sqrt(3), size=(3, 12))), (
        "mode='fan_out' divides by sqrt(fan_out) = sqrt(3) here (it preserves the variance of the backward pass)")
    nb.kaiming_uniform_(p, nonlinearity="tanh", rng=2)
    b = np.sqrt(3) * (5 / 3) / np.sqrt(12)
    assert np.allclose(p.data, np.random.default_rng(2).uniform(-b, b, size=(3, 12))), (
        "kaiming_uniform_: bound = sqrt(3) * gain / sqrt(fan), with gain = calculate_gain(nonlinearity)")
    nb.kaiming_normal_(p, nonlinearity="linear", rng=1)
    assert np.allclose(p.data, np.random.default_rng(1).normal(0, 1 / np.sqrt(12), size=(3, 12))), (
        "nonlinearity='linear' has gain 1 (LeCun init)")
    with pytest.raises(ValueError):
        nb.kaiming_normal_(p, mode="fan_avg", rng=0)
    assert nb.kaiming_uniform_(p, rng=0) is p


def test_stage3_conv_shape_matches_torch(nb):
    torch = pytest.importorskip("torch")
    w = nn.Parameter(np.zeros((32, 16, 3, 3)))
    nb.kaiming_normal_(w, rng=0)
    t = torch.empty(32, 16, 3, 3, dtype=torch.float64)
    torch.manual_seed(0)
    torch.nn.init.kaiming_normal_(t, nonlinearity="relu")
    want = np.sqrt(2 / 144)
    assert abs(t.std().item() - want) < 0.05 * want  # torch itself follows the formula
    assert abs(w.data.std() - want) < 0.05 * want, (
        f"for a conv weight fan_in = in * kh * kw = 144, so the std is sqrt(2/144) = {want:.4f} (torch: "
        f"{t.std().item():.4f}); yours is {w.data.std():.4f}")


# ---------------------------------------------------------------- stage 4: deep_mlp


def test_stage4_structure_and_biases(nb):
    net = nb.deep_mlp((20, 16, 16, 3), activation="tanh", init="xavier", rng=0)
    kinds = [type(net[i]).__name__ for i in range(len(net))]
    assert kinds == ["Linear", "Tanh", "Linear", "Tanh", "Linear"], (
        f"Linear layers with the activation between them, none after the last; got {kinds}")
    assert all(np.all(net[i].bias.data == 0) for i in (0, 2, 4)), "every init except 'default' zeroes the biases"
    assert [net[i].weight.shape for i in (0, 2, 4)] == [(16, 20), (16, 16), (3, 16)]
    lin = nb.deep_mlp((5, 4, 2), activation="linear", init="kaiming", rng=0)
    assert len(lin) == 2, "activation='linear' means no activation modules at all"
    d = nb.deep_mlp((5, 4, 2), init="default", rng=0)
    assert np.any(d[0].bias.data != 0), "init='default' keeps Linear's own init (its random bias too)"


def test_stage4_init_scales(nb):
    n = 400
    for init, act, std in [("kaiming", "relu", np.sqrt(2 / n)), ("kaiming", "tanh", (5 / 3) / np.sqrt(n)),
                           ("xavier", "relu", np.sqrt(2 / (2 * n))), (0.05, "relu", 0.05),
                           ("default", "relu", 1 / np.sqrt(3 * n))]:
        net = nb.deep_mlp((n, n, n), activation=act, init=init, rng=1)
        got = net[2].weight.data.std()
        assert abs(got - std) < 0.03 * std, (
            f"init={init!r} with activation={act!r} on a {n}x{n} layer should give weight std {std:.4f}, got {got:.4f}. "
            "xavier is xavier_normal_ with gain 1; kaiming is kaiming_normal_ with nonlinearity=activation; "
            "a float is N(0, std^2)")
    a = nb.deep_mlp((10, 10, 10), init="kaiming", rng=3)
    b = nb.deep_mlp((10, 10, 10), init="kaiming", rng=3)
    assert all(np.array_equal(p.data, q.data) for p, q in zip(a.parameters(), b.parameters())), (
        "the same seed gives the same network: draw everything from ONE generator, as_rng(rng)")
    assert not np.allclose(a[0].weight.data, a[2].weight.data), "layers draw in turn from the generator (not the same seed)"
    d = nb.deep_mlp((10, 10, 10), init="default", rng=3)
    assert not np.allclose(d[0].weight.data, d[2].weight.data), (
        "with init='default' too, every Linear must get the shared generator (rng=rng), not the same int seed")


def test_stage4_variance_is_preserved(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 256))
    ratios = {"kaiming": [], "xavier": []}
    for init in ratios:
        for seed in (1, 2, 3):  # finite width makes single networks fluctuate: look at the median of three
            net = nb.deep_mlp([256] * 21, activation="relu", init=init, rng=seed)
            h = Tensor(X)
            for i in range(len(net) - 1):  # through the 20th ReLU (just before the last Linear)
                h = net[i](h)
            ratios[init].append(float(np.mean(h.data ** 2)))
    k, x = np.median(ratios["kaiming"]), max(ratios["xavier"])
    assert 0.3 < k < 3, (
        f"with kaiming init a 20-layer ReLU net keeps the second moment of its activations near the input's "
        f"(E[x^2] = 1 in, about 1 out); median of 3 networks: {k:.3g}")
    assert x < 1e-4, (
        f"with xavier init each ReLU layer halves the second moment: after 20 layers it should be ~2^-20; got {x:.3g}")


# ---------------------------------------------------------------- stage 5: activation_stats


def test_stage5_hand_example(nb):
    net = nn.Sequential(nn.Linear(2, 2, rng=0), nn.ReLU())
    net[0].weight.data[...] = np.array([[1.0, 0.0], [0.0, -1.0]])
    net[0].bias.data[...] = 0.0
    X = np.array([[1.0, 2.0], [3.0, -4.0]])
    stats = nb.activation_stats(net, X)
    assert isinstance(stats, list) and len(stats) == 2, "one dict per layer of the Sequential (Linear and ReLU both count)"
    lin, relu = stats
    assert set(lin) >= {"name", "mean", "std", "frac_zero"}, f"each dict has name, mean, std, frac_zero; got {sorted(lin)}"
    # Linear output [[1, -2], [3, 4]], ReLU output [[1, 0], [3, 4]]
    assert np.isclose(lin["mean"], 1.5) and np.isclose(lin["std"], np.std([1, -2, 3, 4])), (
        f"Linear output is [[1, -2], [3, 4]]: mean 1.5, std {np.std([1, -2, 3, 4]):.4f} (over ALL entries, ddof=0); got {lin}")
    assert np.isclose(relu["frac_zero"], 0.25) and np.isclose(relu["mean"], 2.0), (
        f"ReLU output is [[1, 0], [3, 4]]: one entry in four is exactly zero; got {relu}")
    assert lin["name"].startswith("0") and "Linear" in lin["name"] and "ReLU" in relu["name"], (
        f"names are '<index>.<layer class>', e.g. '0.Linear', '1.ReLU'; got {lin['name']!r}, {relu['name']!r}")
    assert all(isinstance(v, float) for v in (lin["mean"], lin["std"], relu["frac_zero"])), "values are Python floats"


def test_stage5_no_graph_and_mode_untouched(nb):
    net = nn.Sequential(nn.Linear(3, 3, rng=0), nn.Tanh(), nn.Linear(3, 1, rng=1))
    net.train()
    for p in net.parameters():
        p.grad = None
    stats = nb.activation_stats(net, np.ones((4, 3)))
    assert len(stats) == 3 and net.training is True, "activation_stats doesn't change the model's mode"
    assert all(p.grad is None for p in net.parameters())
    assert np.isclose(stats[1]["frac_zero"], 0.0)

    class Probe(nn.Module):  # an identity layer that records whether a graph is being recorded
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            self.seen.append((is_grad_enabled(), getattr(x, "requires_grad", False)))
            return x

    probe = Probe()
    nb.activation_stats(nn.Sequential(nn.Linear(3, 3, rng=0), probe), np.ones((2, 3)))
    assert probe.seen and probe.seen[0] == (False, False), (
        "activation_stats is a measurement: run the layers inside `with no_grad():`, so no graph is recorded "
        f"(the layer after a Linear saw grad mode / requires_grad = {probe.seen[0] if probe.seen else None})")
    tiny = nb.activation_stats(nn.Sequential(nn.Tanh()), np.full((2, 3), 1e-5))
    assert tiny[0]["frac_zero"] == 0.0, "frac_zero counts EXACT zeros: tanh(1e-5) is small but not zero"


def test_stage5_vanishing_signal(nb):
    X = np.random.default_rng(0).normal(size=(500, 128))
    small = nb.activation_stats(nb.deep_mlp([128] * 11, activation="tanh", init=0.01, rng=0), X)
    good = nb.activation_stats(nb.deep_mlp([128] * 11, activation="relu", init="kaiming", rng=0), X)
    assert small[-2]["std"] < 1e-6, (
        f"tanh with N(0, 0.01^2) weights multiplies the std by about 0.01 * sqrt(128) = 0.11 per layer: after 10 "
        f"layers it's ~1e-10; got {small[-2]['std']:.3g}")
    relus = [s for s in good if "ReLU" in s["name"]]
    assert all(0.3 < s["frac_zero"] < 0.7 for s in relus), "about half of every ReLU layer's outputs are zero"
