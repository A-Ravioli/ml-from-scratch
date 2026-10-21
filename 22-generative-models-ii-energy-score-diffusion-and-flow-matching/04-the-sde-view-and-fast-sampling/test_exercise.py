"""Checks for exercise.py (the SDE view and fast sampling).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The continuous-time schedule is
checked on and between grid points and against finite differences. Every sampler is checked step by step against a
hand computation with the same random draws (tiny schedules, toy networks), for its number of network calls, for
eval mode and no gradients, and in closed form on Gaussian data N(0, s^2), where the exact noise predictor is known
at every continuous time: the probability-flow ODE is then linear and maps x_T to x_T * sqrt(v_0 / v_T) with
v_t = ab(t) s^2 + 1 - ab(t), so your sampler's output must match that number for number; DDIM (eta = 0) is a
product of known per-step factors; the reverse SDE must reproduce the variance.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _schedule(n, b0, b1):
    from mlfs.diffusion import DiffusionSchedule

    return DiffusionSchedule(n, beta_start=b0, beta_end=b1)


def _ab_at(schedule, t):
    """the spec's continuous alpha_bar: log-linear between grid points k / n, clamped."""
    n = schedule.n_steps
    la = schedule.alpha_bars.double().log()
    u = (t.double() * n).clamp(0, n - 1)
    k = u.floor().long().clamp(max=n - 2)
    w = u - k
    return torch.exp((1 - w) * la[k] + w * la[k + 1])


class _Oracle(torch.nn.Module):
    """exact noise predictor for data N(0, s^2) at continuous times."""

    def __init__(self, schedule, s):
        super().__init__()
        self.schedule, self.s, self.calls, self.modes = schedule, s, 0, []

    def forward(self, x, t, y=None):
        self.calls += 1
        assert self.calls <= 20_000, "the sampler called the network more than 20,000 times: a loop never ends"
        self.modes.append(self.training)
        a = _ab_at(self.schedule, t).float().view((-1,) + (1,) * (x.dim() - 1))
        return (1 - a).sqrt() * x / (a * self.s ** 2 + 1 - a)


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ts, self.ys, self.modes = [], [], []

    def forward(self, x, t, y=None):
        self.ts.append(t.clone())
        assert len(self.ts) <= 20_000, "the sampler called the network more than 20,000 times: a loop never ends"
        self.ys.append(y)
        self.modes.append(self.training)
        return 0.2 * x + t.view((-1,) + (1,) * (x.dim() - 1))


# ---------------------------------------------------------------- stage 1: continuous time


def test_stage1_alpha_bar_at(nb):
    s = _schedule(10, 0.02, 0.3)
    k = torch.arange(10)
    assert torch.allclose(nb.alpha_bar_at(s, k / 10), s.alpha_bars, atol=1e-6), "on the grid t = k / n, alpha_bars[k]"
    mid = nb.alpha_bar_at(s, torch.tensor([0.35]))
    assert torch.allclose(mid, (s.alpha_bars[3] * s.alpha_bars[4]).sqrt(), atol=1e-6), (
        "halfway between grid points the LOG alpha_bar is interpolated linearly: the geometric mean")
    edge = nb.alpha_bar_at(s, torch.tensor([-0.5, 0.95, 1.0]))
    assert torch.allclose(edge, s.alpha_bars[[0, 9, 9]], atol=1e-6), "clamp t to [0, (n - 1) / n]"
    assert nb.alpha_bar_at(s, torch.full((2, 3), 0.5)).shape == (2, 3), "any shape of t"


def test_stage1_beta_at(nb):
    s = _schedule(10, 0.02, 0.3)
    la = s.alpha_bars.double().log()
    t = torch.tensor([0.0, 0.33, 0.85, 0.9])
    want = torch.stack([10 * (la[0] - la[1]), 10 * (la[3] - la[4]), 10 * (la[8] - la[9]), 10 * (la[8] - la[9])])
    assert torch.allclose(nb.beta_at(s, t).double(), want, atol=1e-4), (
        "beta(t) = n (log ab_k - log ab_{k+1}) on segment k; at t = (n - 1)/n use the last segment")
    h = 1e-3
    tt = torch.tensor([0.12, 0.47], dtype=torch.float64)
    fd = -(nb.alpha_bar_at(s, tt + h).double().log() - nb.alpha_bar_at(s, tt - h).double().log()) / (2 * h)
    assert torch.allclose(nb.beta_at(s, tt).double(), fd, rtol=1e-3), "beta(t) = -d log alpha_bar(t) / dt"


def test_stage1_eps_to_score_and_drift(nb):
    eps = torch.randn(2, 1, 2, 2, generator=_gen(0))
    got = nb.eps_to_score(eps, torch.tensor([0.36, 0.64]))
    assert torch.allclose(got, -eps / torch.tensor([0.8, 0.6]).view(2, 1, 1, 1), atol=1e-6), (
        "score = -eps / sqrt(1 - alpha_bar), alpha_bar per example broadcast over (C, H, W)")
    assert torch.allclose(nb.eps_to_score(eps, 0.75), -2 * eps, atol=1e-6), "a float alpha_bar works too"
    s = _schedule(10, 0.02, 0.3)
    x = torch.randn(3, 2, generator=_gen(1))
    t = torch.tensor([0.1, 0.45, 0.9])
    model = _Toy()
    y = torch.tensor([0, 1, 2])
    got = nb.pf_ode_drift(model, s, x, t, y)
    a = _ab_at(s, t).float()[:, None]
    la = s.alpha_bars.double().log()
    k = (t.double() * 10).clamp(0, 9).floor().long().clamp(max=8)
    beta = (10 * (la[k] - la[k + 1])).float()[:, None]
    want = -0.5 * beta * (x - (0.2 * x + t[:, None]) / (1 - a).sqrt())
    assert torch.allclose(got, want, atol=1e-5), "dx/dt = -beta(t)/2 (x + score), score = -model(x, t, y)/sqrt(1 - ab(t))"
    assert model.ys[0] is y and torch.equal(model.ts[0], t), "call model(x, t, y) with the given t and y"


# ---------------------------------------------------------------- stage 2: DDIM


def test_stage2_timesteps(nb):
    got = nb.ddim_timesteps(1000, 5)
    assert got.dtype == torch.int64 and got.tolist() == torch.linspace(0, 999, 5).round().long().tolist(), (
        f"round(linspace(0, n_train - 1, n_steps)); got {got.tolist()}")
    assert nb.ddim_timesteps(10, 10).tolist() == list(range(10)), "all steps when n_steps = n_train"
    assert nb.ddim_timesteps(10, 1).tolist() == [9], "a single step starts from the noisiest step n_train - 1"


def _ddim_by_hand(s, model, shape, n_steps, eta, gen, y=None):
    ab = s.alpha_bars.double()
    taus = torch.linspace(0, s.n_steps - 1, n_steps).round().long().tolist()
    x = torch.randn(shape, generator=gen)
    for k in reversed(range(len(taus))):
        a = ab[taus[k]].item()
        ap = ab[taus[k - 1]].item() if k > 0 else 1.0
        e = model(x, torch.full((shape[0],), taus[k] / s.n_steps), y)
        x0 = (x - math.sqrt(1 - a) * e) / math.sqrt(a)
        sig = eta * math.sqrt((1 - ap) / (1 - a)) * math.sqrt(1 - a / ap)
        x = math.sqrt(ap) * x0 + math.sqrt(1 - ap - sig ** 2) * e
        if sig > 0:
            x = x + sig * torch.randn(shape, generator=gen)
    return x


def test_stage2_ddim_by_hand(nb):
    s = _schedule(10, 0.05, 0.4)
    for eta in (0.0, 0.7):
        model = _Toy()
        y = torch.tensor([3, 4])
        got = bounded(nb.ddim_sample, model, s, (2, 3), n_steps=3, eta=eta, y=y, generator=_gen(2), budget=400_000,
                      what="ddim_sample", hint="one update per tau")
        want = _ddim_by_hand(s, lambda x, t, yy: 0.2 * x + t[:, None], (2, 3), 3, eta, _gen(2))
        assert [round(t[0].item(), 3) for t in model.ts] == [0.9, 0.4, 0.0], (
            "visit taus = round(linspace(0, n - 1, n_steps)) from the last to the first, at t = tau / n")
        assert all(v is y for v in model.ys), "pass y to the model"
        assert torch.allclose(got, want, atol=1e-5), (
            f"eta={eta}: expected {want.tolist()}, got {got.tolist()}: x0 = (x - sqrt(1-ab) eps)/sqrt(ab), sigma = eta "
            "sqrt((1-ab_prev)/(1-ab)) sqrt(1 - ab/ab_prev), x = sqrt(ab_prev) x0 + sqrt(1 - ab_prev - sigma^2) eps + "
            "sigma z, ab_prev = 1 after the first tau, z drawn only when sigma > 0")
        assert not any(model.modes) and model.training, "eval mode while sampling, then restore the model's mode"


def test_stage2_ddim_draws_noise_only_when_needed(nb):
    s = _schedule(10, 0.05, 0.4)
    for eta, n_draws in ((0.0, 1), (0.7, 3)):
        g = _gen(20)
        nb.ddim_sample(_Toy(), s, (2, 3), n_steps=3, eta=eta, generator=g)
        ref = _gen(20)
        for _ in range(n_draws):
            torch.randn((2, 3), generator=ref)
        assert torch.equal(torch.rand(4, generator=g), torch.rand(4, generator=ref)), (
            f"eta={eta}: draw x_T, then z only at steps where sigma > 0 ({n_draws} draws of `shape` in all; sigma is 0 "
            "for eta = 0 and at the last jump to x0)")


def test_stage2_ddim_gaussian_closed_form(nb):
    s = _schedule(1000, 1e-4, 0.02)
    sd = 0.5
    oracle = _Oracle(s, sd)
    got = nb.ddim_sample(oracle, s, (2000, 1), n_steps=10, generator=_gen(3))
    assert oracle.calls == 10, f"n_steps network calls: 10, got {oracle.calls}"
    ab = s.alpha_bars.double()
    taus = torch.linspace(0, 999, 10).round().long().tolist()
    factor = 1.0
    for k in reversed(range(10)):
        a = ab[taus[k]].item()
        ap = ab[taus[k - 1]].item() if k > 0 else 1.0
        v = a * sd ** 2 + 1 - a
        factor *= (math.sqrt(ap * a) * sd ** 2 + math.sqrt((1 - ap) * (1 - a))) / v
    want = torch.randn((2000, 1), generator=_gen(3)) * factor
    assert torch.allclose(got, want, atol=1e-4), (
        f"on N(0, {sd}^2) with the exact noise predictor, deterministic DDIM multiplies x_T by a known factor "
        f"({factor:.4f} for 10 steps); got a ratio of {(got / want).median().item():.4f}")
    assert got.std().item() < 0.42, "10 DDIM steps shrink this Gaussian (std about 0.36, not 0.5): see on paper 3"


# ---------------------------------------------------------------- stage 3: the probability-flow ODE


def test_stage3_pf_ode_closed_form(nb):
    s = _schedule(1000, 1e-4, 0.02)
    sd = 0.5
    oracle = _Oracle(s, sd)
    got = nb.probability_flow_sample(oracle, s, (2000, 1), n_steps=50, method="rk4", generator=_gen(4))
    a0 = s.alpha_bars[0].item()
    aT = _ab_at(s, torch.tensor([0.999])).item()
    v = lambda a: a * sd ** 2 + 1 - a
    xT = torch.randn((2000, 1), generator=_gen(4))
    want = xT * math.sqrt(v(a0) / v(aT)) * math.sqrt(a0) * sd ** 2 / v(a0)
    assert torch.allclose(got, want, atol=5e-3), (
        f"on N(0, {sd}^2) the PF-ODE from t = (n-1)/n to 0 is linear: x_0 = x_T sqrt(v_0 / v_T), v = ab s^2 + 1 - ab, "
        "followed by the denoising step x0 = (x - sqrt(1 - ab_0) eps)/sqrt(ab_0); max error "
        f"{(got - want).abs().max().item():.4f}. check the drift's sign, beta(t), and the time direction")
    assert oracle.calls == 4 * 50 + 1, f"rk4 with 50 steps: 200 network calls plus one denoising call; got {oracle.calls}"
    assert not any(oracle.modes) and oracle.training, "eval mode while sampling, then the model's mode restored"
    errs = {}
    for method, calls in (("euler", 51), ("midpoint", 101)):
        o = _Oracle(s, sd)
        out = nb.probability_flow_sample(o, s, (2000, 1), n_steps=50, method=method, generator=_gen(4))
        assert o.calls == calls, f"{method} with 50 steps: {calls} network calls; got {o.calls}"
        errs[method] = (out - want).abs().max().item()
    errs["rk4"] = (got - want).abs().max().item()
    assert errs["euler"] > 2 * max(errs["midpoint"], errs["rk4"]), f"pass `method` to odeint; errors {errs}"
    assert errs["midpoint"] != errs["rk4"], "midpoint and rk4 are different solvers"


def test_stage3_pf_ode_by_hand(nb):
    s = _schedule(10, 0.05, 0.4)
    model = _Toy()
    y = torch.tensor([1, 2, 3])
    w = torch.nn.Parameter(torch.tensor(1.0))
    out = bounded(nb.probability_flow_sample, lambda x, t, yy=None: w * model(x, t, yy), s, (3, 2), n_steps=2,
                  budget=400_000, what="probability_flow_sample", method="euler",
                                     y=y, generator=_gen(5))
    assert not out.requires_grad, "no gradients"
    assert all(v is y for v in model.ys), "pass y to every model call"
    ts = [round(t[0].item(), 4) for t in model.ts]
    assert ts == [0.9, 0.45, 0.0], (
        f"euler with 2 steps: drift evaluations at t = 0.9 and 0.45 (from (n-1)/n down to 0), then the denoising call at "
        f"t = 0; times seen {ts}")
    la = s.alpha_bars.double().log()
    x = torch.randn((3, 2), generator=_gen(5))
    for t in (0.9, 0.45):
        k = min(int(math.floor(t * 10 + 1e-9)), 8)
        beta = 10 * (la[k] - la[k + 1]).item()
        a = _ab_at(s, torch.tensor([t])).item()
        drift = -0.5 * beta * (x - (0.2 * x + t) / math.sqrt(1 - a))
        x = x - 0.45 * drift
    a0 = s.alpha_bars[0].item()
    x = (x - math.sqrt(1 - a0) * 0.2 * x) / math.sqrt(a0)
    assert torch.allclose(out.detach(), x, atol=1e-5), (
        f"expected {x.tolist()}, got {out.tolist()}: x_T = torch.randn(shape, generator), n_steps solver steps of the "
        "drift backwards in time, then x0 = (x - sqrt(1 - ab_0) model(x, 0, y)) / sqrt(ab_0)")


# ---------------------------------------------------------------- stage 4: the reverse SDE


def test_stage4_reverse_sde_by_hand(nb):
    s = _schedule(5, 0.1, 0.4)
    model = _Toy()
    got = bounded(nb.reverse_sde_sample, model, s, (2, 2), n_steps=2, generator=_gen(6), budget=400_000,
                  what="reverse_sde_sample", hint="n_steps Euler-Maruyama steps, then the denoising step")
    g = _gen(6)
    x = torch.randn((2, 2), generator=g)
    la = s.alpha_bars.double().log()
    h = 0.8 / 2
    for j in range(2):
        t = 0.8 - j * h
        k = min(int(math.floor(t * 5 + 1e-9)), 3)
        beta = 5 * (la[k] - la[k + 1]).item()
        a = _ab_at(s, torch.tensor([t])).item()
        score = -(0.2 * x + t) / math.sqrt(1 - a)
        x = x + h * beta * (0.5 * x + score) + math.sqrt(beta * h) * torch.randn((2, 2), generator=g)
    a0 = s.alpha_bars[0].item()
    x = (x - math.sqrt(1 - a0) * (0.2 * x)) / math.sqrt(a0)
    assert torch.allclose(got, x, atol=1e-5), (
        f"expected {x.tolist()}, got {got.tolist()}: x_T first, then per step x + h beta (x/2 + score) + sqrt(beta h) z "
        "from t = (n-1)/n down in n_steps steps of h = t_max / n_steps, then the denoising step at t = 0")
    assert len(model.ts) == 3 and not any(model.modes), "n_steps + 1 network calls, in eval mode"
    assert model.training, "restore the model's mode after sampling"


def test_stage4_reverse_sde_distribution(nb):
    s = _schedule(1000, 1e-4, 0.02)
    x = nb.reverse_sde_sample(_Oracle(s, 0.5), s, (20000, 1), n_steps=200, generator=_gen(7))
    assert abs(x.mean().item()) < 0.02 and abs(x.std().item() - 0.5) < 0.02, (
        f"with the exact score the reverse SDE must recover N(0, 0.5^2); got mean {x.mean().item():.3f}, std "
        f"{x.std().item():.3f}")
