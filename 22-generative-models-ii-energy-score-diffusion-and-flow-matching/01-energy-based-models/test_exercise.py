"""Checks for exercise.py (energy-based models).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Langevin dynamics is checked
step by step against a hand computation with the same random draws, and in distribution against the closed
form for its *biased* stationary variance on a Gaussian (s^2 / (1 - h / 4s^2)), which catches a missing 1/2 or
a wrong noise scale. MALA is checked for the exact Gaussian variance at a step size where plain Langevin is
off by a factor of 2, on a non-Gaussian double well against numerical integration, and for its first step
against a hand computation. Contrastive divergence is checked against its formula, and the training loop on a
Gaussian energy with two learnable moments (persistent chains must recover the data's mean and spread), on a
fixed energy whose persistent chains must reach Langevin's stationary distribution, and for every option.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _warm_up_adam():
    """torch's first Adam step runs millions of lines of one-time setup; do it outside `bounded`'s line budget."""
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w])
    w.sum().backward()
    opt.step()


def _capped(fn, limit=20_000, what="the sampler"):
    """wrap a score/energy so a sampler that never stops fails (after `limit` calls) instead of hanging."""
    count = [0]

    def wrapped(*args, **kwargs):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"{what} called the score or energy more than {limit:,} times: a loop never ends")
        return fn(*args, **kwargs)

    return wrapped


def _gen(seed):
    return torch.Generator().manual_seed(seed)


class _Quadratic(torch.nn.Module):
    """E(x) = 0.5 * sum(((x - mu) / s)^2) with learnable mu and log s: p_theta = N(mu, s^2)."""

    def __init__(self, dim=2):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(dim))
        self.log_s = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        self.n = getattr(self, "n", 0) + 1
        assert self.n <= 200_000, "train_ebm called the energy more than 200,000 times: a loop never ends"
        return 0.5 * (((x - self.mu) / self.log_s.exp()) ** 2).sum(1)


class _Pull(torch.nn.Module):
    """a fixed energy 0.5 * k * ||x - c||^2 (a dummy parameter so Adam has something to hold)."""

    def __init__(self, c, k=1.0):
        super().__init__()
        self.c, self.k = torch.as_tensor(c, dtype=torch.float32), k
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, x):
        self.calls.append(x.shape[0])
        assert len(self.calls) <= 200_000, "train_ebm called the energy more than 200,000 times: a loop never ends"
        return 0.5 * self.k * ((x - self.c) ** 2).sum(1) + 0.0 * self.dummy


# ---------------------------------------------------------------- stage 1: Langevin dynamics


def test_stage1_score_from_energy(nb):
    mu, s = torch.tensor([1.0, -2.0]), torch.tensor([0.5, 2.0])
    score = nb.score_from_energy(lambda x: 0.5 * (((x - mu) / s) ** 2).sum(1))
    x = torch.randn(4, 2, generator=_gen(0))
    want = -(x - mu) / s ** 2
    got = score(x)
    assert got.shape == x.shape and torch.allclose(got, want, atol=1e-6), (
        "score_fn(x) = -grad_x E(x): for E = 0.5 ((x - mu)/s)^2 that's -(x - mu)/s^2")
    with torch.no_grad():
        got2 = score(x)
    assert torch.allclose(got2, want, atol=1e-6), (
        "score_fn must work inside torch.no_grad() (samplers call it there): turn gradients on with "
        "torch.enable_grad() around the autograd call")
    assert not got.requires_grad, "return a detached tensor: samplers must not build a graph across steps"
    imgs = torch.randn(2, 1, 3, 3, generator=_gen(1))
    assert torch.allclose(nb.score_from_energy(lambda v: (v ** 2).flatten(1).sum(1))(imgs), -2 * imgs, atol=1e-6), (
        "inputs of any shape: images (B, C, H, W) with an energy per image")


def test_stage1_langevin_step_by_hand(nb):
    x0 = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    score = lambda x: -x  # standard normal
    got = bounded(nb.langevin_dynamics, score, x0, step_size=0.1, n_steps=2, generator=_gen(3), budget=200_000,
                  what="langevin_dynamics", hint="loop exactly n_steps times")
    g = _gen(3)
    x = x0.clone()
    for _ in range(2):
        z = torch.randn(x.shape, generator=g)
        x = x + 0.05 * (-x) + math.sqrt(0.1) * z
    assert torch.allclose(got, x, atol=1e-6), (
        f"expected {x.tolist()}, got {got.tolist()}: each step is x + (step_size/2) score(x) + sqrt(step_size) z "
        "with z = torch.randn(x.shape, generator=generator), one draw per step")
    assert torch.equal(x0, torch.tensor([[1.0, -1.0], [0.5, 2.0]])), "don't modify x0 in place"


def test_stage1_langevin_options(nb):
    calls = []

    def score(x):
        calls.append(1)
        assert len(calls) <= 1000, "langevin_dynamics never stops calling score_fn"
        return -x

    nb.langevin_dynamics(score, torch.zeros(3, 2), n_steps=7, generator=_gen(0))
    assert len(calls) == 7, f"n_steps=7 should call score_fn 7 times, got {len(calls)}"
    a = nb.langevin_dynamics(_capped(lambda x: -x), torch.zeros(5, 2), step_size=0.3, n_steps=3, generator=_gen(1))
    b = nb.langevin_dynamics(_capped(lambda x: -x), torch.zeros(5, 2), step_size=0.3, n_steps=3, generator=_gen(1))
    assert torch.equal(a, b), "the same generator seed must give the same chain"
    c = nb.langevin_dynamics(_capped(lambda x: -x), torch.zeros(5, 2), n_steps=3, generator=_gen(1))
    assert not torch.allclose(a, c), "step_size must be used"
    c2 = nb.langevin_dynamics(_capped(lambda x: -x), torch.zeros(5, 2), step_size=0.01, n_steps=3, generator=_gen(1))
    assert torch.allclose(c, c2), "step_size defaults to 0.01"
    w = torch.nn.Parameter(torch.tensor(1.0))
    out = nb.langevin_dynamics(_capped(lambda x: -w * x), torch.zeros(2, 2), n_steps=2, generator=_gen(2))
    assert not out.requires_grad, "return a detached tensor even if score_fn depends on parameters"
    d = nb.langevin_dynamics(_capped(lambda x: -x), torch.ones(2, 2), n_steps=0)
    assert torch.equal(d, torch.ones(2, 2)), "n_steps=0 returns (a copy of) x0"


def test_stage1_langevin_stationary_variance(nb):
    # on N(0, s^2) with step h, Langevin's chain is AR(1): x' = (1 - h/(2 s^2)) x + sqrt(h) z, whose stationary
    # variance is s^2 / (1 - h / (4 s^2)): biased upward, more so for larger steps.
    s2 = 1.0
    for h in (0.5, 1.0):
        x = nb.langevin_dynamics(_capped(lambda v: -v / s2), torch.zeros(20000, 1), step_size=h, n_steps=60,
                                 generator=_gen(10))
        want = s2 / (1 - h / (4 * s2))
        got = x.var().item()
        assert abs(got - want) < 0.05 * want, (
            f"step {h}: stationary variance {got:.3f}, expected {want:.3f} (= s^2 / (1 - h/(4 s^2)), the known bias of "
            "unadjusted Langevin). 0.667 or 1.0 at h=0.5 means a missing 1/2 on the score term; about twice the "
            "expected value means noise sqrt(2h) instead of sqrt(h)")


# ---------------------------------------------------------------- stage 2: MALA


def test_stage2_mala_exact_gaussian(nb):
    energy = _capped(lambda x: 0.5 * (x ** 2).sum(1))
    rates = []
    for h in (0.5, 2.0):
        x, rate = bounded(nb.mala, energy, torch.zeros(20000, 1), step_size=h, n_steps=80, generator=_gen(4),
                          budget=5_000_000, what="mala", hint="loop exactly n_steps times")
        assert isinstance(rate, float) and 0.0 < rate < 1.0, "return (x, acceptance rate as a python float in (0, 1))"
        assert abs(x.var().item() - 1.0) < 0.05, (
            f"step {h}: MALA's samples must have the target's variance 1.0 (got {x.var().item():.3f}); unadjusted "
            f"Langevin would give {1 / (1 - h / 4):.2f}. check the proposal densities in the acceptance ratio")
        rates.append(rate)
    assert rates[1] < rates[0], f"bigger steps are rejected more often; rates {rates}"


def test_stage2_mala_double_well(nb):
    energy = _capped(lambda x: (x[:, 0] ** 2 - 1) ** 2 * 2.0)  # two wells at +-1
    x0 = torch.empty(20000, 1).uniform_(-1.5, 1.5, generator=_gen(5))
    x, _ = nb.mala(energy, x0, step_size=0.05, n_steps=300, generator=_gen(6))
    grid = np.linspace(-3, 3, 20001)
    p = np.exp(-2.0 * (grid ** 2 - 1) ** 2)
    m2 = (grid ** 2 * p).sum() / p.sum()
    m4 = (grid ** 4 * p).sum() / p.sum()
    got2, got4 = (x ** 2).mean().item(), (x ** 4).mean().item()
    assert abs(got2 - m2) < 0.015 and abs(got4 - m4) < 0.04, (
        f"E[x^2] = {got2:.3f} and E[x^4] = {got4:.3f} should match the double well's {m2:.3f} and {m4:.3f} (unadjusted "
        "Langevin at this step gives E[x^2] = 0.82: the Metropolis correction is what removes that bias)")


def test_stage2_mala_steps_by_hand(nb):
    energy = _capped(lambda x: 0.5 * (x ** 2).sum(1))
    x0 = torch.tensor([[0.3], [-1.2], [2.0], [0.0], [1.1], [-0.4], [0.8], [-2.2]])
    h = 2.5  # large enough that some moves are rejected
    got, rate = bounded(nb.mala, energy, x0, step_size=h, n_steps=4, generator=_gen(7), budget=2_000_000,
                        what="mala", hint="loop exactly n_steps times")
    g = _gen(7)
    lq = lambda to, frm: -((to - frm + 0.5 * h * frm) ** 2).sum(1) / (2 * h)
    x, rates = x0.clone(), []
    for _ in range(4):
        z = torch.randn(x0.shape, generator=g)
        prop = x - 0.5 * h * x + math.sqrt(h) * z
        log_a = 0.5 * (x ** 2).sum(1) - 0.5 * (prop ** 2).sum(1) + lq(x, prop) - lq(prop, x)
        u = torch.rand(8, generator=g)
        acc = torch.log(u) < log_a
        x = torch.where(acc[:, None], prop, x)
        rates.append(acc.float().mean().item())
    assert torch.allclose(got, x, atol=1e-5), (
        f"expected {x.flatten().tolist()}, got {got.flatten().tolist()}: per step draw z = randn(x.shape), propose "
        "x - (h/2) grad E + sqrt(h) z, then u = rand(B), accept where log u < E(x) - E(x') + log q(x|x') - log q(x'|x)")
    assert math.isclose(rate, float(np.mean(rates)), abs_tol=1e-9), (
        f"the acceptance rate is the fraction of accepted moves over all rows AND all steps: {np.mean(rates):.4f}, got "
        f"{rate:.4f}")
    stuck, r = nb.mala(energy, x0, step_size=200.0, n_steps=5, generator=_gen(8))
    assert r < 0.3 and (stuck == x0).all(1).float().mean() >= 0.5, (
        "rejected rows keep their current state: with a huge step almost everything is rejected")


# ---------------------------------------------------------------- stage 3: energy network and CD loss


def test_stage3_energy_mlp(nb):
    torch.manual_seed(0)
    e = nb.EnergyMLP(3, hidden_dim=16, n_layers=2)
    x = torch.randn(5, 3)
    assert e(x).shape == (5,), "energies (B,): one number per input"
    n = sum(p.numel() for p in e.parameters())
    assert n == 3 * 16 + 16 + 16 * 16 + 16 + 16 + 1, f"an MLP 3 -> 16 -> 16 -> 1 has 353 parameters; got {n}"
    n3 = sum(p.numel() for p in nb.EnergyMLP(2, hidden_dim=8, n_layers=3).parameters())
    assert n3 == 2 * 8 + 8 + 2 * (8 * 8 + 8) + 8 + 1, "n_layers hidden layers of width hidden_dim"
    xv = torch.randn(2, 3).requires_grad_(True)
    (g,) = torch.autograd.grad(e(xv).sum(), xv, create_graph=True)
    (h,) = torch.autograd.grad(g[:, 0].sum(), xv)
    assert h.abs().sum() > 1e-6, (
        "the energy's Hessian is exactly zero: ReLU energies are piecewise linear, so their score is piecewise "
        "constant. use a smooth activation (SiLU, softplus, tanh)")


def test_stage3_ebm_loss(nb):
    er = torch.tensor([1.0, -2.0, 0.5])
    ef = torch.tensor([3.0, 1.0])
    want = er.mean() - ef.mean() + 0.3 * ((er ** 2).mean() + (ef ** 2).mean())
    got = nb.ebm_loss(er, ef, alpha=0.3)
    assert torch.allclose(got, want), f"expected {want.item():.4f}, got {float(got):.4f}"
    want_default = er.mean() - ef.mean() + 0.1 * ((er ** 2).mean() + (ef ** 2).mean())
    assert torch.allclose(nb.ebm_loss(er, ef), want_default), "alpha defaults to 0.1"
    er2 = torch.zeros(4, requires_grad=True)
    ef2 = torch.zeros(4, requires_grad=True)
    nb.ebm_loss(er2, ef2, alpha=0.0).backward()
    assert (er2.grad > 0).all() and (ef2.grad < 0).all(), (
        "minimizing the loss must push real energies DOWN and fake energies UP (signs)")


# ---------------------------------------------------------------- stage 4: training


def test_stage4_recovers_gaussian_moments(nb):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2)) * np.array([0.5, 1.0]) + np.array([1.0, -1.0])
    _warm_up_adam()
    bounded(nb.train_ebm, _Quadratic(), X, n_iters=3, batch_size=16, langevin_steps=3, buffer_size=32, seed=0,
            budget=3_000_000, what="train_ebm", hint="loop over n_iters iterations and langevin_steps steps; no while loops")
    torch.manual_seed(0)
    q = _Quadratic()
    hist = nb.train_ebm(q, X, n_iters=300, batch_size=256, lr=3e-2, langevin_steps=20, step_size=0.05,
                        buffer_size=1000, alpha=0.0, seed=0)
    mu, s = q.mu.detach().numpy(), q.log_s.exp().detach().numpy()
    assert np.abs(mu - [1.0, -1.0]).max() < 0.2 and np.abs(s - [0.5, 1.0]).max() < 0.16, (
        f"a Gaussian energy trained by persistent CD should match the data's mean (1, -1) and std (0.5, 1); got mean "
        f"{mu.round(3)}, std {s.round(3)}. a std far below 1 in the second coordinate means the chains aren't "
        "persistent: write the Langevin samples back into the buffer")
    for k in ("loss", "e_real", "e_fake"):
        assert len(hist[k]) == 300 and all(isinstance(v, float) for v in hist[k]), f"hist[{k!r}]: 300 floats"


def test_stage4_alpha_and_lr(nb):
    class Const(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Parameter(torch.tensor(3.0))

        def forward(self, x):
            self.n = getattr(self, "n", 0) + 1
            assert self.n <= 10_000, "train_ebm never stops"
            return self.c.expand(x.shape[0]) + 0.0 * x.sum(1)

    X = np.random.default_rng(1).normal(size=(50, 2))
    e = Const()
    hist = nb.train_ebm(e, X, n_iters=2, batch_size=8, lr=0.0, langevin_steps=1, buffer_size=20, alpha=0.5, seed=0)
    assert np.allclose(hist["loss"], [9.0, 9.0], atol=1e-5), (
        f"a constant energy c=3 gives loss = c - c + alpha (c^2 + c^2) = 9 with alpha=0.5; got {hist['loss']}")
    assert e.c.item() == 3.0, "lr=0 must leave the energy unchanged: use the lr argument"
    assert np.allclose(hist["e_real"], 3.0) and np.allclose(hist["e_fake"], 3.0), "e_real/e_fake: mean energies"
    corners = np.array([[-1.0, -1.0], [1.0, 1.0]] * 50)  # real points at the corners of the box [-1, 1]^2
    h2 = nb.train_ebm(_Pull([0.0, 0.0]), corners, n_iters=3, batch_size=500, lr=0.0, langevin_steps=0,
                      buffer_size=2000, reinit_prob=0.0, seed=0)
    assert np.allclose(h2["e_real"], 1.0) and np.allclose(h2["e_fake"], 1 / 3, atol=0.05), (
        f"with 0.5 ||x||^2 the real corners have energy 1 and uniform box noise 1/3 on average; got e_real "
        f"{np.round(h2['e_real'], 3)}, e_fake {np.round(h2['e_fake'], 3)}: record the real and the fake batch's mean "
        "energies")


def test_stage4_persistent_chains_reach_stationarity(nb):
    # a fixed energy 0.5 ||x - c||^2 with lr = 0: the buffer is a set of persistent Langevin chains, which must reach
    # N(c, var) with Langevin's stationary variance 1 / (1 - h/4) (stage 1), far from the data box [-1, 1]^2.
    X = np.random.default_rng(2).uniform(-1, 1, size=(500, 2))
    _warm_up_adam()
    for h in (0.2, 1.0):
        e = _Pull([3.0, 3.0])
        hist = bounded(nb.train_ebm, e, X, n_iters=200, batch_size=200, lr=0.0, langevin_steps=5, step_size=h,
                       buffer_size=600, reinit_prob=0.0, seed=1, budget=40_000_000, what="train_ebm")
        buf = hist["buffer"]
        assert tuple(buf.shape) == (600, 2), f"the buffer holds buffer_size points (600, 2); got {tuple(buf.shape)}"
        want = 1 / (1 - h / 4)
        assert torch.allclose(buf.mean(0), torch.tensor([3.0, 3.0]), atol=0.3) and abs(buf.var(0).mean().item() - want) < 0.15 * want, (
            f"step {h}: persistent chains on 0.5||x - (3, 3)||^2 should reach mean (3, 3) and variance {want:.2f}; got "
            f"mean {buf.mean(0).numpy().round(2)}, variance {buf.var(0).numpy().round(2)}. write the samples back, "
            "use step_size, and initialize the buffer on the data's bounding box")
    fresh = _Pull([3.0, 3.0])
    hist = nb.train_ebm(fresh, X, n_iters=150, batch_size=100, lr=0.0, langevin_steps=5, step_size=0.2,
                        buffer_size=200, reinit_prob=1.0, seed=1)
    pulled = nb.train_ebm(_Pull([3.0, 3.0]), X, n_iters=3, batch_size=100, lr=0.0, langevin_steps=5, step_size=1.0,
                          buffer_size=200, reinit_prob=1.0, seed=2)
    e_box = 0.5 * ((torch.rand(20000, 2, generator=_gen(3)) * 2 - 1 - 3.0) ** 2).sum(1).mean().item()  # about 9.3
    assert max(pulled["e_fake"]) < 0.5 * e_box, (
        f"e_fake must be the energy of the Langevin samples (near the minimum at (3, 3), about 1.3), not of their "
        f"box-noise starting points (about {e_box:.1f}); got {np.round(pulled['e_fake'], 2)}")
    far = np.random.default_rng(4).uniform(5, 6, size=(300, 2))
    boxed = nb.train_ebm(_Pull([0.0, 0.0]), far, n_iters=20, batch_size=100, lr=0.0, langevin_steps=0, buffer_size=50,
                         reinit_prob=1.0, seed=3)["buffer"]
    assert ((boxed >= 5 - 1e-6) & (boxed <= 6 + 1e-6)).all(), (
        "re-initialized chains restart from fresh noise on the data's bounding box (here [5, 6]^2)")
    d = (hist["buffer"] - torch.tensor([3.0, 3.0])).norm(dim=1).mean().item()
    assert d > 2.0, (
        f"with reinit_prob=1 every chain restarts from box noise, so 5 small steps can't get near (3, 3) (mean distance "
        f"{d:.2f}): re-initialize each selected entry with probability reinit_prob")


def test_stage4_options_and_seed(nb):
    X = np.random.default_rng(3).uniform(-1, 1, size=(300, 2))
    e = _Pull([0.0, 0.0])
    nb.train_ebm(e, X, n_iters=3, batch_size=17, lr=0.0, langevin_steps=4, buffer_size=50, seed=0)
    assert len(e.calls) == 3 * (4 + 2) and set(e.calls) == {17}, (
        f"each iteration: langevin_steps energy evaluations on batch_size buffer points, then one on the real batch and "
        f"one on the fakes (3 x 6 calls of 17 rows); got {len(e.calls)} calls with sizes {sorted(set(e.calls))}")
    e0 = _Pull([0.0, 0.0])
    h0 = nb.train_ebm(e0, X, n_iters=4, batch_size=16, lr=0.0, langevin_steps=0, buffer_size=40, reinit_prob=0.0, seed=2)
    b = h0["buffer"]
    lo, hi = torch.tensor(X.min(0), dtype=torch.float32), torch.tensor(X.max(0), dtype=torch.float32)
    assert ((b >= lo - 1e-6) & (b <= hi + 1e-6)).all() and b.std(0).min() > 0.3, (
        "the buffer starts as uniform noise on the data's bounding box (per-coordinate min to max)")
    torch.manual_seed(0)
    q1 = _Quadratic()
    q2 = _Quadratic()
    q2.load_state_dict(q1.state_dict())
    a = nb.train_ebm(q1, X, n_iters=5, batch_size=32, buffer_size=64, seed=7)
    c = nb.train_ebm(q2, X, n_iters=5, batch_size=32, buffer_size=64, seed=7)
    assert a["loss"] == c["loss"], "everything random comes from a torch.Generator seeded with `seed`: reproducible"
    d = nb.train_ebm(_Quadratic(), X, n_iters=5, batch_size=32, buffer_size=64, seed=8)
    assert d["loss"] != a["loss"], "a different seed gives a different run"
