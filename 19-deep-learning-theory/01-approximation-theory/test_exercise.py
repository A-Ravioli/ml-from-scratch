"""Checks for 19 · 01 approximation theory.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the interpolant is checked
at and between its knots, outside them and for its architecture; the sawtooth against the tent map
composed by hand; the piece counter against functions whose pieces are known (kinks on and off the
grid, noise, sub-intervals, coarse grids); the width search against brute force; the training loop for
its schedule (through Adam's constant-gradient steps), batch sizes, seeding, modes and learning.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

def _run(fn, *args, **kwargs):
    """every call that loops runs under a line budget, so a learner's endless loop fails instead of hanging."""
    return bounded(fn, *args, budget=20_000_000, what=getattr(fn, "__name__", None), **kwargs)



def _eval(net, xs):
    with torch.no_grad():
        return net(torch.tensor(xs, dtype=torch.float32).reshape(-1, 1)).reshape(-1).numpy()


def _tent_k(x, k):
    x = np.asarray(x, dtype=float)
    for _ in range(k):
        x = 1 - np.abs(2 * x - 1)
    return x


# ---------------------------------------------------------------- stage 1: shallow interpolation


def test_stage1_interpolant_values_and_shape(nb):
    kx, ky = [0.0, 1.0, 3.0, 4.0], [0.0, 2.0, 0.0, 4.0]
    net = nb.relu_interpolant(kx, ky)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    assert isinstance(net, nn.Sequential) and len(lin) == 2 and lin[0].in_features == 1 and lin[0].out_features == 3 \
        and lin[1].out_features == 1 and any(isinstance(m, nn.ReLU) for m in net), (
            "return nn.Sequential(Linear(1, n - 1), ReLU(), Linear(n - 1, 1)): n knots need only n - 1 hidden units")
    got = _eval(net, [0.0, 1.0, 3.0, 4.0, 0.5, 2.0, 3.5, -1.0, 5.0])
    want = [0, 2, 0, 4, 1, 1, 2, 0, 8]
    assert np.allclose(got, want, atol=1e-5), (
        f"at x = 0, 1, 3, 4 (knots), 0.5, 2, 3.5 (between), -1 and 5 (outside) expected {want}, got {np.round(got, 4).tolist()}. "
        "left of the first knot the network is flat at y0; right of the last it continues the last segment")
    rng = np.random.default_rng(0)
    kx = np.linspace(-3, 3, 9) + rng.uniform(-0.25, 0.25, 9)  # irregular but not nearly coincident knots
    ky = rng.normal(size=9)
    xs = np.linspace(kx[0], kx[-1], 101)
    assert np.allclose(_eval(nb.relu_interpolant(kx, ky), xs), np.interp(xs, kx, ky), atol=1e-4), (
        "on [x_0, x_{n-1}] the network must equal the piecewise-linear interpolant (np.interp)")


def test_stage1_rejects_bad_knots(nb):
    for kx, ky, why in (([0.0, 2.0, 1.0], [0, 1, 2], "x not increasing"), ([0.0, 1.0, 1.0], [0, 1, 2], "repeated x"),
                        ([0.0], [1.0], "a single knot"), ([0.0, 1.0, 2.0], [0.0, 1.0], "more x than y values")):
        with pytest.raises(ValueError):
            nb.relu_interpolant(kx, ky)
            raise AssertionError(f"relu_interpolant must raise ValueError for {why}")


# ---------------------------------------------------------------- stage 2: the deep sawtooth


@pytest.mark.parametrize("k", [1, 2, 3, 6])
def test_stage2_sawtooth_matches_tent_map(nb, k):
    net = nb.sawtooth_net(k)
    lin = [m for m in net if isinstance(m, nn.Linear)]
    relus = [m for m in net if isinstance(m, nn.ReLU)]
    assert len(lin) == k + 1 and len(relus) == k and all(m.out_features == 2 for m in lin[:-1]), (
        f"sawtooth_net({k}) has {k} hidden ReLU layers of width 2 and a Linear output")
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params == 6 * k + 1, f"{k} hidden layers of width 2 have 6k + 1 = {6 * k + 1} parameters, got {n_params}"
    xs = np.linspace(0, 1, 1025)
    got = _eval(net, xs)
    assert np.allclose(got, _tent_k(xs, k), atol=1e-5), (
        f"sawtooth_net({k}) must compute the tent map t(x) = 1 - |2x - 1| applied {k} times on [0, 1]. "
        "t(x) = 2 relu(x) - 4 relu(x - 1/2) there; feed 2 h1 - 4 h2 into the next layer's two units")
    with pytest.raises(ValueError):
        nb.sawtooth_net(0)


# ---------------------------------------------------------------- stage 3: counting pieces


def test_stage3_counts_known_functions(nb):
    for k in (1, 3, 6):
        got = bounded(nb.count_linear_pieces, nb_saw(k), what="count_linear_pieces")
        assert got == 2 ** k, f"the tent map applied {k} times has {2 ** k} linear pieces on [0, 1], counted {got}"
    rng = np.random.default_rng(1)
    kx = np.sort(rng.uniform(0.02, 0.98, 7))  # kinks between grid points: each bends two second differences
    kx[1:] = np.maximum(kx[1:], kx[:-1] + 0.05).clip(max=0.99)
    net = _interp_net(kx, rng.normal(size=7) * 3)
    sub_interval = _run(nb.count_linear_pieces, lambda x: torch.relu(x - 0.5), 0.6, 1.0)
    assert sub_interval == 1, (
        f"relu(x - 0.5) is a single straight piece on [0.6, 1.0], counted {sub_interval}: the grid must run from lo "
        "to hi, not over [0, 1]")
    off_grid = _run(nb.count_linear_pieces, lambda x: torch.relu(x - 0.50003))
    assert off_grid == 2, (
        f"relu(x - 0.50003) has 2 pieces, counted {off_grid}. its kink falls between two grid points, so TWO "
        "consecutive second differences are large: count runs of marked points, not marked points")
    got = _run(nb.count_linear_pieces, net, float(kx[0]), float(kx[-1]))
    if 6 < got <= 12:
        why = ("a kink between two grid points makes TWO consecutive second differences large: count runs of marked "
               "points, not marked points")
    else:
        why = ("check that the grid runs from lo to hi (here the first to the last knot) and that the threshold is "
               "tol times the largest |first difference|, not an absolute number")
    assert got == 6, (
        f"a ReLU interpolant through 7 knots has 6 pieces between its first and last knot, counted {got}. {why}")
    assert _run(nb.count_linear_pieces, lambda x: 3 * x + 1) == 1 and _run(nb.count_linear_pieces, lambda x: 0 * x + 2) == 1, (
        "a straight line (or a constant) has one piece")


def nb_saw(k):
    def f(x):
        assert tuple(x.shape)[1:] == (1,), f"f is called with an (n_grid, 1) tensor, got shape {tuple(x.shape)}"
        assert x.dtype == torch.float64, f"the grid is float64 (rounding in float32 creates fake kinks), got {x.dtype}"
        return torch.as_tensor(_tent_k(x.numpy(), k))
    return f


def _interp_net(kx, ky):
    slopes = np.diff(ky) / np.diff(kx)
    coef = np.concatenate([[slopes[0]], np.diff(slopes)])

    def f(x):
        x = x.double().reshape(-1)
        return ky[0] + sum(c * torch.relu(x - k) for c, k in zip(coef, kx[:-1]))
    return f


def test_stage3_interval_grid_and_tolerance(nb):
    got = _run(nb.count_linear_pieces, nb_saw(3), 0.0, 0.5)
    assert got == 4, f"on [0, 0.5] the 8-piece sawtooth has 4 pieces; counted {got} (lo and hi set the grid's range)"
    got = _run(nb.count_linear_pieces, nb_saw(4), n_grid=9)
    assert got < 16, f"with only 9 grid points a 16-piece sawtooth can't be resolved; counted {got} (use n_grid)"
    def noisy(x):
        noise = torch.tensor(np.random.default_rng(0).normal(0, 1e-13, x.shape[0]), dtype=torch.float64)
        return 2 * x.double().reshape(-1) + noise

    assert _run(nb.count_linear_pieces, noisy) == 1, (
        "rounding-level noise (1e-13) on a straight line must not count as kinks: compare |second difference| with "
        "tol times the largest |first difference|")
    for scale in (1e5, 1e-5):
        got = _run(nb.count_linear_pieces, lambda x, s=scale: s * nb_saw(3)(x))
        assert got == 8, (
            f"multiplying the function by {scale:g} doesn't change its pieces (counted {got}, expected 8): the "
            "threshold must be relative to the function's own first differences, not an absolute number")

    def gentle(x):
        return x + 0.001 * torch.relu(x - 0.5)

    assert _run(nb.count_linear_pieces, gentle) == 2 and _run(nb.count_linear_pieces, gentle, tol=0.01) == 1, (
        "a kink whose slope change is 0.1% of the largest slope counts at the default tol = 1e-4 but not at tol = 0.01")
    flat = nn.Linear(1, 1)
    with torch.no_grad():
        flat.weight.fill_(1e-4)
        flat.bias.fill_(1.0)
    got = _run(nb.count_linear_pieces, flat)
    assert got == 1, (
        f"nn.Linear with slope 1e-4 around 1.0 is one straight piece, counted {got}: in float32 its outputs are a "
        "staircase. evaluate modules on a float64 copy (copy.deepcopy(f).double())")
    assert flat.weight.dtype == torch.float32, "counting must not change the module you pass in (work on a copy)"


# ---------------------------------------------------------------- stage 4: equal-budget widths


def _count(width, depth, in_dim, out_dim):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, out_dim))
    return sum(p.numel() for p in nn.Sequential(*layers).parameters())


def _formula(w, depth, i, o):
    return (i + 1) * w + (depth - 1) * (w + 1) * w + (w + 1) * o


def test_stage4_param_count_and_width(nb):
    for w, depth, i, o in ((5, 1, 1, 1), (7, 3, 4, 2), (1, 4, 3, 1), (40, 2, 784, 10)):
        assert nb.mlp_param_count(w, depth, in_dim=i, out_dim=o) == _count(w, depth, i, o), (
            f"width {w}, depth {depth}, in {i}, out {o}: an MLP with {depth} hidden Linear+ReLU layers has "
            f"{_count(w, depth, i, o)} parameters (weights and biases)")
    for budget, depth, i in ((500, 1, 1), (500, 4, 1), (3000, 3, 4), (3000, 1, 4), (100000, 2, 196), (2, 3, 1)):
        best = min(range(1, budget + 2), key=lambda w: (abs(_formula(w, depth, i, 1) - budget), w))
        got = bounded(nb.width_for_budget, budget, depth, in_dim=i, what="width_for_budget")
        assert got == best, (
            f"budget {budget}, depth {depth}, in_dim {i}: the closest parameter count comes from width {best} "
            f"({_formula(best, depth, i, 1)} parameters); got width {got}")
    assert _run(nb.width_for_budget, 11, 1, in_dim=2) == 2, (
        "with in_dim=2, depth 1, widths 2 and 3 give 9 and 13 parameters: budget 11 is a tie, broken towards the smaller width")


# ---------------------------------------------------------------- stage 5: the training loop


class Recorder(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.shapes, self.modes = [], []

    def forward(self, x):
        self.shapes.append(x.shape[0])
        self.modes.append(self.training)
        return self.inner(x)


def test_stage5_schedule_through_adam(nb):
    lin = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        lin.weight.zero_()
    x = torch.ones(8, 1)
    y = torch.full((8,), 1e4)
    _run(nb.fit_regressor, lin, x, y, steps=4, lr=0.01, batch_size=8, seed=0)
    w = float(lin.weight.detach())
    want = 0.01 * sum(0.5 * (1 + math.cos(math.pi * t / 4)) for t in range(4))
    assert math.isclose(w, want, rel_tol=1e-3), (
        f"with a constant-sign gradient each Adam step moves the weight by its learning rate, so 4 steps of the cosine "
        f"schedule lr (1 + cos(pi t / 4)) / 2, t = 0..3, move it by {want:.5f}; it moved {w:.5f}. "
        "(0.04 means no schedule; 0.015 means the schedule started at t = 1)")


def test_stage5_batches_seeds_modes(nb):
    torch.manual_seed(0)
    rec = Recorder(nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1)))
    rec.eval()  # as if a previous evaluation left it in eval mode
    x = torch.linspace(-1, 1, 50).reshape(-1, 1)
    y = x.abs().reshape(-1)
    losses = _run(nb.fit_regressor, rec, x, y, steps=30, lr=1e-2, batch_size=7, seed=3)
    assert len(losses) == 30 and all(isinstance(v, float) for v in losses), "return one float loss per step"
    assert set(rec.shapes) == {7}, f"every step uses a minibatch of batch_size = 7 rows; saw sizes {sorted(set(rec.shapes))}"
    assert all(rec.modes), "the model must be in train mode while it trains"
    assert not rec.training, "fit_regressor leaves the model in eval mode"
    rec.shapes.clear()
    _run(nb.fit_regressor, rec, x, y, steps=3, batch_size=500, seed=0)
    assert set(rec.shapes) == {50}, "a batch_size larger than the data uses all n rows"
    runs = []
    for seed in (0, 0, 1):
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
        runs.append(nb.fit_regressor(m, x, y, steps=10, lr=1e-2, batch_size=5, seed=seed))
    assert runs[0] == runs[1] and runs[0] != runs[2], "the same seed gives the same batches; another seed differs"
    const = nn.Linear(1, 1)
    with torch.no_grad():
        const.weight.zero_()
        const.bias.zero_()
    first = _run(nb.fit_regressor, const, x, torch.full((50, 1), 3.0), steps=1, lr=0.0, batch_size=10)[0]
    assert math.isclose(first, 9.0, rel_tol=1e-6), f"the loss is the MEAN squared error (9 for predicting 0 against 3), got {first}"
    assert float(const.bias.detach()) == 0.0, "lr = 0 leaves the weights unchanged"


class RowRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)
        self.batches = []

    def forward(self, x):
        self.batches.append(x.detach().reshape(-1).tolist())
        return self.lin(x)


def test_stage5_samples_with_replacement(nb):
    rec = RowRecorder()
    x = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
    _run(nb.fit_regressor, rec, x, torch.zeros(10), steps=20, lr=1e-3, batch_size=10, seed=0)
    dup = sum(len(set(b)) < len(b) for b in rec.batches)
    assert len(rec.batches) == 20 and dup > 0, (
        "minibatches are drawn WITH replacement (torch.randint): with batch_size = n = 10, a batch of 10 independent "
        "draws repeats some row almost surely, but none of your 20 batches did. a permutation (sampling without "
        "replacement) is a different sampler")


def test_stage5_learns(nb):
    x = torch.linspace(-1, 1, 200).reshape(-1, 1)
    y = (x.abs() - 0.5).reshape(-1)
    finals = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
        _run(nb.fit_regressor, m, x, y, steps=400, lr=1e-2, batch_size=64, seed=seed)
        with torch.no_grad():
            finals.append(float(((m(x).reshape(-1) - y) ** 2).mean()))
    assert np.median(finals) < 0.005, (
        f"fitting |x| - 1/2 with 16 ReLUs for 400 steps should reach a mean squared error below 0.005 (median of 3 "
        f"seeds); got {np.round(finals, 4).tolist()}")
