"""Checks for exercise.py (conditioning and guidance).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Label dropout is checked against
the same random draws and for its rate. Classifier-free guidance is checked against its formula, for one batched
call with labels [y, null], and for passing eval mode through to the wrapped network. Classifier guidance is
checked in closed form: for a 1-D two-class Gaussian mixture the noisy class posteriors and every noise predictor
are known exactly, and by Bayes' rule the unconditional predictor guided by the exact classifier with w = 1 IS
the class-conditional predictor. Training is checked for its label dropout, its two objectives, and by sampling
each class of a trained model.
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


def _cap_module(module, limit=20_000):
    """fail (instead of hanging) if a loop calls the network more than `limit` times."""
    count = [0]

    def hook(mod, inp):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"the network was called more than {limit:,} times: a loop never ends")

    module.register_forward_pre_hook(hook)
    return module


class _Rec(torch.nn.Module):
    """model(x, t, y) = x * (1 + t) + y (as floats), recording every call."""

    def __init__(self):
        super().__init__()
        self.calls = []
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x, t, y=None):
        self.calls.append((x.shape[0], None if y is None else y.clone(), self.training))
        assert len(self.calls) <= 20_000, "the network was called more than 20,000 times: a loop never ends"
        yy = torch.zeros(x.shape[0]) if y is None else y.float()
        shape = (-1,) + (1,) * (x.dim() - 1)
        return x * (1 + t.view(shape)) + yy.view(shape) + 0.0 * self.lin.weight.sum()


# ---------------------------------------------------------------- stage 1: label dropout


def test_stage1_drop_labels(nb):
    y = torch.arange(10)
    got = nb.drop_labels(y, 0.4, 99, generator=_gen(0))
    mask = torch.rand(10, generator=_gen(0)) < 0.4
    assert torch.equal(got, torch.where(mask, torch.tensor(99), y)), (
        "label i becomes null_label where torch.rand(len(y), generator)[i] < p_uncond")
    assert torch.equal(y, torch.arange(10)), "return a new tensor; don't modify y in place"
    assert got.dtype == torch.int64, "same dtype as y"
    assert torch.equal(nb.drop_labels(y, 0.0, 99), y) and torch.all(nb.drop_labels(y, 1.0, 99) == 99), (
        "p_uncond = 0 keeps every label; 1 drops all")
    big = nb.drop_labels(torch.zeros(100000, dtype=torch.long), 0.1, 7, generator=_gen(1))
    frac = (big == 7).float().mean().item()
    assert abs(frac - 0.1) < 0.004, f"about 10% dropped at p_uncond = 0.1; got {100 * frac:.2f}%"


# ---------------------------------------------------------------- stage 2: classifier-free guidance


def test_stage2_cfg_formula_and_batching(nb):
    m = _Rec()
    g = nb.cfg_model(m, 3.0, 5)
    x = torch.randn(4, 1, generator=_gen(2))
    t = torch.tensor([0.1, 0.2, 0.3, 0.4])
    y = torch.tensor([0, 1, 2, 3])
    out = g(x, t, y)
    assert len(m.calls) == 1 and m.calls[0][0] == 8, (
        "one batched call on 2B rows: model(cat([x, x]), cat([t, t]), cat([y, null]))")
    assert torch.equal(m.calls[0][1], torch.tensor([0, 1, 2, 3, 5, 5, 5, 5])), "labels [y, null] in that order"
    cond = x * (1 + t[:, None]) + y[:, None].float()
    unc = x * (1 + t[:, None]) + 5.0
    assert torch.allclose(out, unc + 3.0 * (cond - unc), atol=1e-6), "g = m(null) + w (m(y) - m(null))"
    assert torch.allclose(nb.cfg_model(_Rec(), 1.0, 5)(x, t, y), cond, atol=1e-6), "w = 1: the conditional model"
    assert torch.allclose(nb.cfg_model(_Rec(), 0.0, 5)(x, t, y), unc, atol=1e-6), "w = 0: the unconditional model"
    imgs = torch.randn(2, 3, 4, 4, generator=_gen(3))
    assert nb.cfg_model(_Rec(), 2.0, 9)(imgs, torch.rand(2), torch.tensor([1, 2])).shape == imgs.shape, "any data shape"


def test_stage2_cfg_is_a_module(nb):
    m = _Rec()
    g = nb.cfg_model(m, 2.0, 5)
    assert isinstance(g, torch.nn.Module), (
        "return an nn.Module wrapping the model, so samplers can switch it (and the model inside) to eval mode")
    g.eval()
    assert not m.training, "g.eval() must reach the wrapped model (keep it as a submodule)"
    g(torch.zeros(2, 1), torch.zeros(2), None)
    assert m.calls[-1][1] is not None and torch.all(m.calls[-1][1] == 5), "y=None: the unconditional output (null label)"


# ---------------------------------------------------------------- stage 3: classifier guidance

MUS = torch.tensor([-1.0, 1.0])
SD = 0.5


def _ab(schedule, t):
    n = schedule.n_steps
    la = schedule.alpha_bars.double().log()
    u = (t.double() * n).clamp(0, n - 1)
    k = u.floor().long().clamp(max=n - 2)
    w = u - k
    return torch.exp((1 - w) * la[k] + w * la[k + 1]).float()


def _parts(schedule, x, t):
    """per-class noise predictors (B, 2) and posteriors (B, 2) for the two-class mixture, at continuous t."""
    a = _ab(schedule, t)[:, None]
    v = a * SD ** 2 + 1 - a
    eps_k = (1 - a).sqrt() * (x - a.sqrt() * MUS) / v
    logp = -0.5 * (x - a.sqrt() * MUS) ** 2 / v
    return eps_k, torch.softmax(logp, dim=1)


def test_stage3_classifier_guidance_closed_form(nb):
    from mlfs.diffusion import DiffusionSchedule

    s = DiffusionSchedule(100, beta_start=1e-3, beta_end=0.2)

    class Uncond(torch.nn.Module):
        def forward(self, x, t, y=None):
            e, r = _parts(s, x, t)
            return (r * e).sum(1, keepdim=True)

    class Clf(torch.nn.Module):
        def forward(self, x, t):  # unnormalized logits: log posteriors plus an x-dependent shift shared by all classes
            return torch.log(_parts(s, x, t)[1]) + 3 * x ** 2

    x = torch.linspace(-2.5, 2.5, 11)[:, None]
    t = torch.full((11,), 0.437)
    y = torch.tensor([0, 1] * 5 + [0])
    guided = nb.classifier_guided_eps(Uncond(), Clf(), s, 1.0)
    with torch.no_grad():
        got = guided(x, t, y)
    eps_k, _ = _parts(s, x, t)
    want = eps_k.gather(1, y[:, None])
    assert torch.allclose(got, want, atol=1e-5), (
        "with the exact classifier and w = 1, classifier guidance turns the unconditional predictor into the exact "
        "class-conditional one: eps - sqrt(1 - ab(t)) grad log p(y | x_t). (it must work under torch.no_grad: samplers "
        "call it there.) use alpha_bar_at for ab(t) at continuous t")
    g3 = nb.classifier_guided_eps(Uncond(), Clf(), s, 3.0)
    unc = Uncond()(x, t)
    assert torch.allclose(g3(x, t, y), unc + 3.0 * (want - unc), atol=1e-4), "linear in the guidance scale"
    assert torch.allclose(guided(x, t, None), unc, atol=1e-6), "y=None: no guidance"
    assert isinstance(guided, torch.nn.Module), "return an nn.Module holding both networks"


# ---------------------------------------------------------------- stage 4: conditional training


def test_stage4_label_dropout_and_objectives(nb):
    from mlfs.diffusion import DiffusionSchedule

    X = np.random.default_rng(4).normal(size=(500, 1))
    y = np.arange(500) % 3
    m = _Rec().eval()
    nb.fit_conditional(m, X, y, 3, p_uncond=0.5, n_iters=20, batch_size=100, seed=0)
    labels = torch.cat([c[1] for c in m.calls])
    frac = (labels == 3).float().mean().item()
    assert 0.4 < frac < 0.6 and len(m.calls) == 20 and m.calls[0][0] == 100, (
        f"p_uncond = 0.5 should replace about half the labels by the null label n_classes = 3 (got {frac:.2f}); "
        "n_iters calls on batch_size rows")
    assert all(c[2] for c in m.calls), "fit_conditional must put the model in train mode"
    m0 = _Rec()
    nb.fit_conditional(m0, X, y, 3, p_uncond=0.0, n_iters=5, batch_size=50, seed=0)
    assert not (torch.cat([c[1] for c in m0.calls]) == 3).any(), "p_uncond = 0: no null labels"
    s = DiffusionSchedule(10, beta_start=0.05, beta_end=0.3)
    seen_t = []

    class T(_Rec):
        def forward(self, x, t, y=None):
            seen_t.append(t.clone())
            return super().forward(x, t, y)

    nb.fit_conditional(T(), X, y, 3, objective="ddpm", schedule=s, n_iters=3, batch_size=64, seed=1)
    tt = torch.cat(seen_t) * 10
    assert torch.allclose(tt, tt.round(), atol=1e-5), "objective='ddpm': ddpm_loss, so t = step / n_steps"
    seen_t.clear()
    nb.fit_conditional(T(), X, y, 3, objective="flow", n_iters=3, batch_size=64, seed=1)
    tt = torch.cat(seen_t) * 10
    assert not torch.allclose(tt, tt.round(), atol=1e-3), "objective='flow': flow_matching_loss, t ~ U[0, 1]"
    with pytest.raises(ValueError):
        nb.fit_conditional(_Rec(), X, y, 3, objective="gan", n_iters=1)
    with pytest.raises(ValueError):
        nb.fit_conditional(_Rec(), X, y, 3, objective="ddpm", schedule=None, n_iters=1)
    a, b = _Rec(), _Rec()
    b.load_state_dict(a.state_dict())
    la = nb.fit_conditional(a, X, y, 3, n_iters=3, batch_size=16, seed=4)
    assert la == nb.fit_conditional(b, X, y, 3, n_iters=3, batch_size=16, seed=4), (
        "all randomness from a generator seeded with `seed`: two identical models train identically")
    assert la != nb.fit_conditional(_Rec(), X, y, 3, n_iters=3, batch_size=16, seed=5), "a different seed, another run"
    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)

        def forward(self, x, t, y=None):
            return self.lin(x)

    frozen = Lin()
    before = frozen.lin.weight.clone()
    nb.fit_conditional(frozen, X, y, 3, n_iters=2, batch_size=16, lr=0.0)
    assert torch.equal(frozen.lin.weight, before), "use the lr argument (lr = 0: no change)"


def test_stage4_trained_model_obeys_labels(nb):
    from mlfs.diffusion import TimeMLP, sample_flow

    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 4000)
    X = (MUS.numpy()[y] + SD * rng.normal(size=4000))[:, None]
    torch.manual_seed(6)
    m = TimeMLP(1, hidden_dim=32, n_layers=2, time_dim=16, n_classes=2)
    _warm_up_adam()
    bounded(nb.fit_conditional, TimeMLP(1, hidden_dim=8, n_layers=1, time_dim=8, n_classes=2), X, y, 2, n_iters=3,
            batch_size=8, budget=3_000_000, what="fit_conditional", hint="loop over n_iters minibatches")
    losses = nb.fit_conditional(_cap_module(m), X, y, 2, p_uncond=0.2, n_iters=400, batch_size=256, lr=5e-3, seed=0)
    assert len(losses) == 400 and all(isinstance(v, float) for v in losses)
    out = {}
    for lab in (0, 1, 2):
        out[lab] = sample_flow(m, (2000, 1), n_steps=20, method="midpoint", y=torch.full((2000,), lab),
                               generator=_gen(7)).squeeze(1)
    assert abs(out[0].mean() + 1) < 0.35 and abs(out[1].mean() - 1) < 0.35, (
        f"class means should be about -1 and +1; got {out[0].mean():.2f} and {out[1].mean():.2f}")
    frac_pos = (out[2] > 0).float().mean().item()
    assert 0.35 < frac_pos < 0.65 and out[2].std() > 0.85, (
        f"the null label must give the unconditional mixture (half the samples on each side, std about 1.1); got "
        f"{100 * frac_pos:.0f}% positive, std {out[2].std():.2f}. the null label is n_classes")
