"""Checks for exercise.py (universal approximation).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The constructions are
checked exactly (interpolation through the given points, the tent map composed d times, 2^d linear
pieces); the region counter against nets whose pieces are known; training with fixed seeds and
loose thresholds aggregated over seeds. Training runs under `mlfs.testing.bounded`.
"""
import numpy as np

from mlfs import nn
from mlfs.testing import bounded

GRID = np.linspace(0.0, 1.0, 10_000)[1:-1]  # avoids every dyadic rational, where the kinks of the sawtooth are


def _out(model, x):
    return nn.predict(model, np.asarray(x, dtype=float)[:, None])[:, 0]


def _tent(x):
    return np.where(x < 0.5, 2 * x, 2 - 2 * x)


# ---------------------------------------------------------------- stage 1: interpolation by construction


def test_stage1_set_linear(nb):
    layer = nn.Linear(3, 2, rng=0)
    W_before, b_before = layer.weight.data, layer.bias.data
    out = nb.set_linear(layer, [[1, 2, 3], [4, 5, 6]], [7, 8])
    assert out is layer, "set_linear returns the layer"
    assert layer.weight.data is W_before and layer.bias.data is b_before, "copy into the existing arrays (in place)"
    assert np.allclose(layer.weight.data, [[1, 2, 3], [4, 5, 6]]) and np.allclose(layer.bias.data, [7, 8])


def test_stage1_interpolates_exactly(nb):
    rng = np.random.default_rng(0)
    knots = rng.uniform(-2, 2, 8)
    values = rng.normal(size=8)
    net = nb.relu_interpolant(knots, values)
    assert isinstance(net, nn.Sequential) and len(net) == 3, "Sequential(Linear(1, k), ReLU(), Linear(k, 1))"
    assert net[0].weight.shape == (8, 1) and net[2].weight.shape == (1, 8), (
        f"k = 8 points need exactly k = 8 hidden units; got hidden width {net[0].weight.shape[0]}")
    assert np.allclose(_out(net, knots), values, atol=1e-9), "the network must pass through every given point exactly"
    order = np.argsort(knots)
    xs = np.linspace(knots.min(), knots.max(), 500)
    assert np.allclose(_out(net, xs), np.interp(xs, knots[order], values[order]), atol=1e-9), (
        "between the knots the network is the straight line joining neighbouring points (np.interp)")


def test_stage1_linear_beyond_the_knots(nb):
    net = nb.relu_interpolant([0.0, 1.0, 3.0], [1.0, 3.0, 2.0])
    left, right = _out(net, [-2.0, -1.0]), _out(net, [4.0, 5.0])
    assert np.allclose(left, [-3.0, -1.0]), (
        f"left of the first knot the first segment's line continues (slope 2 through (0, 1)): want (-3, -1), got {left}")
    assert np.allclose(right, [1.5, 1.0]), (
        f"right of the last knot the last segment continues (slope -1/2 through (3, 2)): want (1.5, 1), got {right}")
    two = nb.relu_interpolant([2.0, 0.0], [0.0, 4.0])  # unsorted, k = 2: a single line
    assert np.allclose(_out(two, [-1.0, 1.0, 3.0]), [6.0, 2.0, -2.0]), "two points give the line through them (sort the knots)"


# ---------------------------------------------------------------- stage 2: the sawtooth


def test_stage2_tent_and_compositions(nb):
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 300)
    for depth in [1, 2, 3, 5]:
        net = nb.sawtooth_net(depth)
        want = x.copy()
        for _ in range(depth):
            want = _tent(want)
        assert np.allclose(_out(net, x), want, atol=1e-9), (
            f"depth {depth}: the network must compute the tent map T(x) = 2x (x < 1/2), 2 - 2x (x >= 1/2) composed "
            f"{depth} times on [0, 1]")


def test_stage2_width_two(nb):
    for depth in [1, 4]:
        net = nb.sawtooth_net(depth)
        linears = [net[i] for i in range(len(net)) if isinstance(net[i], nn.Linear)]
        relus = [net[i] for i in range(len(net)) if isinstance(net[i], nn.ReLU)]
        assert len(relus) == depth and len(linears) == depth + 1, (
            f"depth {depth}: one ReLU layer per tent ({depth}), and {depth + 1} Linear layers (merge the output of one "
            f"tent with the input of the next into a single (2, 2) layer); got {len(relus)} ReLUs, {len(linears)} Linears")
        assert all(lin.weight.shape[0] <= 2 for lin in linears), "every layer has at most 2 units"
        n_params = sum(p.size for p in net.parameters())
        assert n_params == 4 + 6 * (depth - 1) + 3, f"depth {depth}: 4 + 6 (depth - 1) + 3 parameters, got {n_params}"


# ---------------------------------------------------------------- stage 3: counting pieces


def test_stage3_counts_sawtooth_pieces(nb):
    for depth in [1, 2, 3, 4, 6]:
        got = bounded(nb.count_linear_regions, nb.sawtooth_net(depth), GRID, what="count_linear_regions")
        assert got == 2 ** depth, (
            f"the tent map composed {depth} times has 2^{depth} = {2 ** depth} linear pieces on [0, 1]; counted {got}. "
            "record the on/off pattern (output > 0) of every ReLU layer at every point, and count where it changes")


def test_stage3_counts_interpolant_pieces(nb):
    net = nb.relu_interpolant([0.1, 0.3, 0.45, 0.7, 0.9], [0.0, 1.0, -1.0, 2.0, 0.5])
    xs = np.linspace(-1.0, 2.0, 7001)
    assert nb.count_linear_regions(net, xs) == 5, "5 knots: the units switch at the first 4 knots, giving 5 regions"
    shuffled = np.random.default_rng(0).permutation(xs)
    assert nb.count_linear_regions(net, shuffled) == 5, "the points may come in any order: sort them first"
    assert nb.count_linear_regions(nn.Sequential(nn.Linear(1, 1, rng=0)), xs) == 1, "no ReLU: one region"


def test_stage3_matches_brute_force(nb):
    rng = np.random.default_rng(2)
    net = nn.Sequential(nn.Linear(1, 5, rng=3), nn.ReLU(), nn.Linear(5, 4, rng=4), nn.ReLU(), nn.Linear(4, 1, rng=5))
    xs = np.sort(rng.uniform(-3, 3, 4000))
    h1 = xs[:, None] @ net[0].weight.data.T + net[0].bias.data
    a1 = np.maximum(h1, 0)
    h2 = a1 @ net[2].weight.data.T + net[2].bias.data
    P = np.concatenate([h1 > 0, h2 > 0], axis=1)
    want = 1 + int(np.any(P[1:] != P[:-1], axis=1).sum())
    assert nb.count_linear_regions(net, xs) == want, "patterns from every ReLU layer count, not just the first"


# ---------------------------------------------------------------- stage 4: training


def test_stage4_max_error(nb):
    v_net = nb.relu_interpolant([0.0, 0.5, 1.0], [0.5, 0.0, 0.5])
    assert nb.max_error(v_net, lambda x: np.abs(x - 0.5)) < 1e-9, "a network that equals f has max error 0"
    line = nb.relu_interpolant([0.0, 1.0], [0.0, 1.0])
    got = nb.max_error(line, lambda x: x ** 2)
    assert np.isclose(got, 0.25, atol=1e-6), f"max |x - x^2| on [0, 1] is 1/4 (at x = 1/2), got {got}"
    got = nb.max_error(line, lambda x: x ** 2, lo=1.0, hi=2.0)
    assert np.isclose(got, 2.0, atol=1e-6), f"on [1, 2], max |x - x^2| is 2 (at x = 2), got {got}: use lo and hi"


def test_stage4_fit_function(nb):
    f = lambda t: np.sin(2 * np.pi * t)  # noqa: E731
    errs, finals = [], []
    for seed in range(3):
        model = nn.Sequential(nn.Linear(1, 32, rng=seed), nn.Tanh(), nn.Linear(32, 1, rng=seed + 10))
        hist = bounded(nb.fit_function, model, f, epochs=300, n_points=128, batch_size=32, lr=0.01, rng=seed,
                       budget=60_000_000, what="fit_function")
        assert len(hist["train_loss"]) == 300, "fit_function returns fit's history (one train loss per epoch)"
        finals.append(hist["train_loss"][-1])
        errs.append(nb.max_error(model, f))
    assert np.median(finals) < 0.02 and np.median(errs) < 0.45, (
        f"a 32-unit tanh network trained for 300 epochs should approximate sin(2 pi x) on [0, 1] (median MSE < 0.02, "
        f"median max error < 0.45 over 3 seeds); got MSEs {np.round(finals, 4)}, max errors {np.round(errs, 3)}")
