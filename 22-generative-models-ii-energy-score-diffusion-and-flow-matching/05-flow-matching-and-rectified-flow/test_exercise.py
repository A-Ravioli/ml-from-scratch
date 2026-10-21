"""Checks for exercise.py (flow matching and rectified flow).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Gaussian data N(mu, s^2) are used
because the marginal velocity of the straight-line path is known exactly,
v(x, t) = mu + (t s^2 - (1 - t)) (x - t mu) / ((1 - t)^2 + t^2 s^2), and the ODE it defines maps noise x0 to
mu + s x0. The loss is checked against a hand computation with the same random draws and for its irreducible
value at the exact field, the integral of s^2 / ((1 - t)^2 + t^2 s^2) (it never reaches zero). The sampler is
checked against the exact map, for its solver orders and call counts, and in eval mode without gradients.
Training is checked against the exact field and, with given pairs, on a coupling whose loss can reach zero. The
reflow pairs and the straightness measure are checked by hand.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _warm_up_adam():
    """torch's first Adam step runs millions of lines of one-time setup; do it outside `bounded`'s line budget."""
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w])
    w.sum().backward()
    opt.step()


class _Exact(torch.nn.Module):
    """the exact marginal velocity for data N(mu, s^2) (elementwise, any shape) and noise N(0, 1)."""

    def __init__(self, mu, s):
        super().__init__()
        self.mu, self.s = mu, s
        self.calls, self.modes, self.ts, self.ys = 0, [], [], []

    def forward(self, x, t, y=None):
        self.calls += 1
        assert self.calls <= 20_000, "the network was called more than 20,000 times: a loop never ends"
        self.modes.append(self.training)
        self.ts.append(t.clone())
        self.ys.append(y)
        t = t.view((-1,) + (1,) * (x.dim() - 1))
        return self.mu + (t * self.s ** 2 - (1 - t)) * (x - t * self.mu) / ((1 - t) ** 2 + t ** 2 * self.s ** 2)


def _cap_module(module, limit=20_000):
    """fail (instead of hanging) if a loop calls the network more than `limit` times."""
    count = [0]

    def hook(mod, inp):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"the network was called more than {limit:,} times: a loop never ends")

    module.register_forward_pre_hook(hook)
    return module


def _capped(fn, limit=20_000):
    """wrap a velocity function so a loop that never stops fails instead of hanging."""
    count = [0]

    def wrapped(*args, **kwargs):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"the network was called more than {limit:,} times: a loop never ends")
        return fn(*args, **kwargs)

    return wrapped


def _floor(s, d=1):
    t = np.linspace(0, 1, 200001)
    return d * np.trapezoid(s ** 2 / ((1 - t) ** 2 + t ** 2 * s ** 2), t)


# ---------------------------------------------------------------- stage 1: the loss


def test_stage1_loss_by_hand(nb):
    seen = []

    def model(xt, t, y=None):
        seen.append((xt.clone(), t.clone(), y))
        return 0.3 * xt

    x1 = torch.randn(5, 2, generator=_gen(0))
    y = torch.arange(5)
    got = nb.flow_matching_loss(model, x1, y=y, generator=_gen(1))
    g = _gen(1)
    x0 = torch.randn(x1.shape, generator=g)
    t = torch.rand(5, generator=g)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    assert torch.allclose(seen[0][1], t), "draw x0 = torch.randn(x1.shape, generator) first, then t = torch.rand(B, generator)"
    assert torch.allclose(seen[0][0], xt, atol=1e-6), "x_t = (1 - t) x0 + t x1"
    assert seen[0][2] is y, "pass y to the model"
    want = ((0.3 * xt - (x1 - x0)) ** 2).mean()
    assert torch.allclose(got, want, atol=1e-6), f"MSE against the velocity x1 - x0: {want.item():.5f}, got {float(got):.5f}"
    x0_given = torch.randn(5, 2, generator=_gen(2))
    seen.clear()
    nb.flow_matching_loss(model, x1, x0=x0_given, generator=_gen(3))
    t2 = torch.rand(5, generator=_gen(3))
    assert torch.allclose(seen[0][1], t2), "with x0 given, t is the first (and only) draw"
    assert torch.allclose(seen[0][0], (1 - t2[:, None]) * x0_given + t2[:, None] * x1, atol=1e-6), "use the given x0"


def test_stage1_loss_floor_and_images(nb):
    torch.manual_seed(4)
    mu, s = 1.0, 0.5
    x1 = mu + s * torch.randn(100000, 1)
    got = nb.flow_matching_loss(_Exact(mu, s), x1, generator=_gen(5)).item()
    floor = _floor(s)
    assert abs(got - floor) < 0.03, (
        f"at the exact marginal velocity the loss is the irreducible integral of s^2 / ((1-t)^2 + t^2 s^2) = "
        f"{floor:.3f}, not 0; got {got:.3f}")
    imgs = torch.randn(3, 1, 4, 4, generator=_gen(6))
    val = nb.flow_matching_loss(lambda xt, t, y=None: torch.zeros_like(xt), imgs, generator=_gen(7))
    assert val.dim() == 0, "images (B, C, H, W): t broadcast over (C, H, W)"
    w = torch.nn.Parameter(torch.tensor(0.5))
    nb.flow_matching_loss(lambda xt, t, y=None: w * xt, imgs, generator=_gen(8)).backward()
    assert w.grad is not None, "differentiable in the model's parameters"


# ---------------------------------------------------------------- stage 2: sampling


def test_stage2_exact_map(nb):
    mu, s = 1.0, 0.5
    m = _Exact(mu, s)
    x1 = bounded(nb.sample_flow, m, (2000, 1), n_steps=20, method="rk4", generator=_gen(9), budget=2_000_000,
                 what="sample_flow", hint="exactly n_steps solver steps from t = 0 to 1")
    x0 = torch.randn((2000, 1), generator=_gen(9))
    assert torch.allclose(x1, mu + s * x0, atol=1e-3), (
        f"the exact Gaussian velocity carries x0 to mu + s x0; max error {(x1 - mu - s * x0).abs().max().item():.4f}. "
        "x0 = torch.randn(shape, generator); integrate from t = 0 to 1")
    assert m.calls == 80, f"rk4 with 20 steps: 80 calls, got {m.calls}"
    assert not any(m.modes) and m.training, "eval mode while sampling, then the model's mode restored"
    errs = {}
    for method, calls in (("euler", 20), ("midpoint", 40), ("rk4", 80)):
        mm = _Exact(mu, s)
        out = nb.sample_flow(mm, (2000, 1), n_steps=20, method=method, generator=_gen(9))
        errs[method] = (out - mu - s * x0).abs().max().item()
        assert mm.calls == calls, f"{method} with 20 steps: {calls} calls, got {mm.calls}"
    assert errs["euler"] > errs["midpoint"] > errs["rk4"], f"first, second and fourth order: errors {errs}"
    assert nb.sample_flow(_Exact(mu, s), (3, 1), generator=_gen(1)).shape == (3, 1)


def test_stage2_options(nb):
    m = _Exact(0.0, 2.0)
    x0 = torch.tensor([[1.0], [-0.5]])
    out = nb.sample_flow(m, (2, 1), n_steps=4, x0=x0)
    ref, want = _Exact(0.0, 2.0), x0.clone()
    for k in range(4):
        want = want + ref(want, torch.full((2,), k / 4)) / 4
    assert torch.allclose(out, want, atol=1e-5), "use the given x0; the default method is 'euler'"
    assert [round(t[0].item(), 3) for t in m.ts] == [0.0, 0.25, 0.5, 0.75], (
        "euler with 4 steps: the model sees t = 0, 0.25, 0.5, 0.75 as (B,) tensors")
    assert torch.equal(x0, torch.tensor([[1.0], [-0.5]])), "don't modify x0"
    seen = []
    w = torch.nn.Parameter(torch.tensor(1.0))
    y = torch.tensor([3, 1])
    def velocity(x, t, yy=None):
        seen.append(yy)
        assert len(seen) <= 1000, "sample_flow never stops calling the model"
        return w * x

    res = nb.sample_flow(velocity, (2, 2), n_steps=2, y=y, generator=_gen(2))
    assert all(v is y for v in seen) and not res.requires_grad, "pass y; no gradients"


# ---------------------------------------------------------------- stage 3: training


def test_stage3_fit_matches_exact_velocity(nb):
    from mlfs.diffusion import TimeMLP

    rng = np.random.default_rng(10)
    X = 1.0 + 0.5 * rng.normal(size=(4000, 1))
    torch.manual_seed(11)
    _warm_up_adam()
    bounded(nb.fit_flow_matching, TimeMLP(1, hidden_dim=8, n_layers=1, time_dim=8), X, n_iters=3, batch_size=8,
            budget=3_000_000, what="fit_flow_matching", hint="loop over n_iters minibatches")
    m = TimeMLP(1, hidden_dim=32, n_layers=2, time_dim=32)
    losses = nb.fit_flow_matching(_cap_module(m), X, n_iters=300, batch_size=256, lr=5e-3, seed=0)
    assert len(losses) == 300 and all(isinstance(v, float) for v in losses), "300 losses as floats"
    g = _gen(100)
    x0 = torch.randn(2000, 1, generator=g)
    x1 = 1.0 + 0.5 * torch.randn(2000, 1, generator=g)
    t = 0.05 + 0.9 * torch.rand(2000, generator=g)
    x = (1 - t[:, None]) * x0 + t[:, None] * x1  # points where the path actually goes
    with torch.no_grad():
        err = (m(x, t) - _Exact(1.0, 0.5)(x, t)).abs().mean().item()
    assert err < 0.2, (
        f"the trained velocity should be close to the exact marginal velocity of N(1, 0.5^2) (mean error {err:.3f}, "
        "typical |v| about 1); "
        "check the loss's target x1 - x0 and the interpolation (1 - t) x0 + t x1")


def test_stage3_fit_with_pairs(nb):
    from mlfs.diffusion import TimeMLP

    rng = np.random.default_rng(12)
    X0 = rng.normal(size=(2000, 2))
    X1 = X0 + np.array([3.0, -1.0])  # each pair differs by a constant: a straight, learnable coupling
    torch.manual_seed(13)
    m = TimeMLP(2, hidden_dim=32, n_layers=2, time_dim=16)
    losses = nb.fit_flow_matching(_cap_module(m), X1, X0=X0, n_iters=300, batch_size=128, lr=5e-3, seed=0)
    assert np.mean(losses[-30:]) < 0.05, (
        f"with pairs x0 = X0[idx], x1 = X1[idx] the velocity is the constant (3, -1) and the loss can reach 0; final "
        f"loss {np.mean(losses[-30:]):.3f}. use the SAME indices for X0 and X")
    seen, modes = [], []

    class Rec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x, t, y=None):
            seen.append(x.shape[0])
            assert len(seen) <= 1000, "fit_flow_matching never stops"
            modes.append(self.training)
            return self.lin(x)

    r = Rec().eval()
    nb.fit_flow_matching(r, X1, n_iters=3, batch_size=17, seed=0)
    assert seen == [17, 17, 17] and all(modes), "n_iters minibatches of batch_size, in train mode"
    a, b = Rec(), Rec()
    b.load_state_dict(a.state_dict())
    assert nb.fit_flow_matching(a, X1, n_iters=3, seed=5) == nb.fit_flow_matching(b, X1, n_iters=3, seed=5), "seeded"
    before = [p.clone() for p in a.parameters()]
    nb.fit_flow_matching(a, X1, n_iters=2, lr=0.0)
    assert all(torch.equal(p, q) for p, q in zip(before, a.parameters())), "use the lr argument"


def test_stage3_fit_with_a_coupling(nb):
    X1 = np.random.default_rng(14).normal(size=(500, 2)) + 2.0
    seen = []

    class Rec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x, t, y=None):
            seen.append(x.detach().clone())
            assert len(seen) <= 1000, "fit_flow_matching never stops"
            return self.lin(x)

    a, b, c = Rec(), Rec(), Rec()
    b.load_state_dict(a.state_dict())
    c.load_state_dict(a.state_dict())
    plain = nb.fit_flow_matching(a, X1, n_iters=4, batch_size=16, seed=3)
    calls = []

    def identity(x0, x1):
        calls.append((x0.shape, x1.shape))
        return x0

    same = nb.fit_flow_matching(b, X1, coupling=identity, n_iters=4, batch_size=16, seed=3)
    assert same == plain and len(calls) == 4 and calls[0] == ((16, 2), (16, 2)), (
        "coupling(x0, x1) is called once per batch on fresh noise x0 = torch.randn(x1.shape, generator=g) and the "
        "minibatch x1; an identity coupling must reproduce the default run exactly")
    flipped = nb.fit_flow_matching(c, X1, coupling=lambda x0, x1: x0.flip(0), n_iters=4, batch_size=16, seed=3)
    assert flipped != plain, "use the coupling's output as x0"
    with pytest.raises(ValueError):
        nb.fit_flow_matching(Rec(), X1, X0=X1, coupling=identity, n_iters=1)


# ---------------------------------------------------------------- stage 4: reflow and straightness


def test_stage4_reflow_pairs(nb):
    m = _Exact(1.0, 0.5)
    y = torch.tensor([0, 1, 2, 3])
    x0, x1 = nb.make_reflow_pairs(m, 4, (3,), n_steps=10, y=y, generator=_gen(14))
    want0 = torch.randn((4, 3), generator=_gen(14))
    assert x0.shape == (4, 3) and torch.allclose(x0, want0), "x0 = torch.randn((n, *shape), generator)"
    ref = _Exact(1.0, 0.5)
    want1 = x0.clone()
    for k in range(10):
        want1 = want1 + ref(want1, torch.full((4,), k / 10)) / 10
    assert torch.allclose(x1, want1, atol=1e-5), "x1 = sample_flow(model, (n, *shape), n_steps=n_steps, x0=x0) (Euler)"
    assert m.calls == 10, "n_steps network calls"
    assert all(v is y for v in m.ys), "pass y to sample_flow (and so to the model)"
    x0b, _ = nb.make_reflow_pairs(m, 2, 3, n_steps=2, generator=_gen(15))
    assert x0b.shape == (2, 3), "an int shape means one dimension"


def test_stage4_straightness(nb):
    const = _capped(lambda x, t, y=None: torch.ones_like(x) * torch.tensor([2.0, -1.0]))
    x0 = torch.randn(10, 2, generator=_gen(16))
    assert nb.straightness(const, x0, n_steps=5) < 1e-10, "a constant velocity field has perfectly straight paths: 0"
    lin = _capped(lambda x, t, y=None: 0.5 * x)
    xs, vs = [x0.clone()], []
    for k in range(4):
        vs.append(0.5 * xs[-1])
        xs.append(xs[-1] + vs[-1] / 4)
    disp = xs[-1] - x0
    want = np.mean([((disp - v) ** 2).sum(1).mean().item() for v in vs])
    got = nb.straightness(lin, x0, n_steps=4)
    assert isinstance(got, float) and math.isclose(got, want, rel_tol=1e-4), (
        f"mean over steps and batch of ||(x_end - x0) - v(x_k, t_k)||^2 along the Euler path: {want:.6f}, got {got}")
    m = _Exact(1.0, 0.5)
    y = torch.arange(10)
    bounded(nb.straightness, m, x0[:, :1], n_steps=3, y=y, budget=300_000, what="straightness",
            hint="exactly n_steps Euler steps")
    assert [round(t[0].item(), 3) for t in m.ts] == [0.0, 0.333, 0.667] and not any(m.modes), (
        "evaluate at t_k = k / n_steps, in eval mode")
    assert m.training, "restore the model's mode afterwards"
    assert all(v is y for v in m.ys), "pass y to the model"
