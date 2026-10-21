"""Checks for exercise.py (diffusion, DDPM).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The schedules are checked against
their formulas; `q_sample` against its closed form. The networks are checked for shapes, for really using t
and y (after randomizing their weights, so a zero-initialized output layer can't hide anything), for the null
label, for dropout only in train mode, and the U-Net for a receptive field that spans the image. The loss and
the sampler are checked against hand computations with the same random draws, and against the closed form for
Gaussian data, whose optimal noise predictor is known exactly: eps*(x_t) = sqrt(1 - ab)(x_t - sqrt(ab) mu) /
(ab s^2 + 1 - ab). Plugging it into your sampler must give back N(mu, s^2); its loss must equal the
irreducible error E_t[ab s^2 / (ab s^2 + 1 - ab)].
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _capped(fn, limit=20_000):
    """wrap a model function so a loop that never stops fails (after `limit` calls) instead of hanging."""
    count = [0]

    def wrapped(*args, **kwargs):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"the network was called more than {limit:,} times: a loop never ends")
        return fn(*args, **kwargs)

    return wrapped


def _cap_module(module, limit=20_000):
    """the same for an nn.Module, through a forward pre-hook."""
    count = [0]

    def hook(mod, inp):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"the network was called more than {limit:,} times: a loop never ends")

    module.register_forward_pre_hook(hook)
    return module


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _warm_up_adam():
    """torch's first Adam step runs millions of lines of one-time setup; do it outside `bounded`'s line budget."""
    w = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([w])
    w.sum().backward()
    opt.step()


def _randomize(module, scale=0.3, seed=0):
    g = _gen(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return module


class _Oracle(torch.nn.Module):
    """the exact noise predictor for data N(mu, s^2) (any shape) under a schedule with n steps."""

    def __init__(self, schedule, mu, s):
        super().__init__()
        self.ab, self.n, self.mu, self.s = schedule.alpha_bars.double().cpu(), schedule.n_steps, mu, s
        self.calls = []

    def forward(self, x, t, y=None):
        self.calls.append((t.clone(), None if y is None else y.clone(), self.training))
        assert len(self.calls) <= 20_000, "the sampler called the network more than 20,000 times: a loop never ends"
        i = torch.round(t.double().cpu() * self.n).long()
        a = self.ab[i].float().view((-1,) + (1,) * (x.dim() - 1))
        return (1 - a).sqrt() * (x - a.sqrt() * self.mu) / (a * self.s ** 2 + 1 - a)


# ---------------------------------------------------------------- stage 1: the schedule


def test_stage1_linear_schedule(nb):
    s = nb.DiffusionSchedule(10, beta_start=0.01, beta_end=0.2)
    want = torch.linspace(0.01, 0.2, 10)
    assert s.n_steps == 10 and s.betas.shape == (10,) and s.betas.dtype == torch.float32, (
        "attrs n_steps and float32 tensors betas, alphas, alpha_bars of shape (n_steps,)")
    assert torch.allclose(s.betas, want, atol=1e-6), "linear: betas = linspace(beta_start, beta_end, n_steps)"
    assert torch.allclose(s.alphas, 1 - want, atol=1e-6), "alphas = 1 - betas"
    assert torch.allclose(s.alpha_bars, torch.cumprod(1 - want, 0), atol=1e-6), (
        "alpha_bars[t] = prod_{s <= t} alphas[s] (so alpha_bars[0] = alphas[0], not 1)")
    d = nb.DiffusionSchedule()
    assert d.n_steps == 1000 and math.isclose(d.betas[0].item(), 1e-4, rel_tol=1e-5) and math.isclose(
        d.betas[-1].item(), 0.02, rel_tol=1e-5), "defaults: 1000 steps from 1e-4 to 0.02 (Ho et al., 2020)"
    assert 3e-5 < d.alpha_bars[-1].item() < 5e-5, "the default linear schedule ends at alpha_bar about 4e-5"
    with pytest.raises(ValueError):
        nb.DiffusionSchedule(10, schedule="quadratic")


def test_stage1_cosine_schedule(nb):
    s = nb.DiffusionSchedule(50, schedule="cosine")
    u = torch.arange(51, dtype=torch.float64) / 50
    f = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
    ab = f[1:] / f[0]
    betas = (1 - ab / torch.cat([f[:1] / f[0], ab[:-1]])).clamp(max=0.999)
    assert torch.allclose(s.betas.double(), betas, atol=1e-5), "cosine: betas = 1 - ab_t / ab_{t-1}, clipped at 0.999"
    assert torch.allclose(s.alpha_bars.double(), torch.cumprod(1 - betas, 0), atol=1e-5) and math.isclose(
            s.alpha_bars[-1].item(), torch.cumprod(1 - betas, 0)[-1].item(), rel_tol=1e-3), (
        "cosine alpha_bars = cumprod(1 - betas), which is f(t + 1)/f(0) up to the clipping at the last step")
    assert torch.allclose(s.alphas, 1 - s.betas), "alphas = 1 - betas"


def test_stage1_q_sample(nb):
    s = nb.DiffusionSchedule(20, beta_start=0.01, beta_end=0.3)
    x0 = torch.randn(4, 1, 3, 3, generator=_gen(0))
    t = torch.tensor([0, 5, 19, 5])
    noise = torch.randn(x0.shape, generator=_gen(1))
    got = s.q_sample(x0, t, noise)
    ab = s.alpha_bars[t].view(4, 1, 1, 1)
    assert torch.allclose(got, ab.sqrt() * x0 + (1 - ab).sqrt() * noise, atol=1e-6), (
        "x_t = sqrt(ab_t) x0 + sqrt(1 - ab_t) noise, with ab_t picked per example and broadcast over (C, H, W)")
    torch.manual_seed(2)
    x = s.q_sample(torch.full((20000, 1), 2.0), torch.full((20000,), 10))
    a = s.alpha_bars[10].item()
    assert abs(x.mean().item() - 2 * math.sqrt(a)) < 0.03 and abs(x.var().item() - (1 - a)) < 0.03, (
        "noise=None draws standard normal noise")
    assert s.to("cpu") is s, "to(device) returns the schedule itself"
    m = nb.DiffusionSchedule(5).to("meta")
    assert all(v.device.type == "meta" for v in (m.betas, m.alphas, m.alpha_bars)), (
        "to(device) moves betas, alphas and alpha_bars")


# ---------------------------------------------------------------- stage 2: time-conditioned networks


def test_stage2_sinusoidal_features(nb):
    t = torch.tensor([0.0, 0.25, 1.0])
    got = nb.sinusoidal_features(t, 6)
    k = torch.arange(3, dtype=torch.float32)
    f = 10000.0 ** (-k / 3)
    args = 1000 * t[:, None] * f[None]
    want = torch.cat([torch.cos(args), torch.sin(args)], 1)
    assert got.shape == (3, 6) and torch.allclose(got, want, atol=1e-4), (
        "[cos(1000 t f_k), sin(1000 t f_k)], f_k = max_period^(-k / (dim // 2)), cosines first")
    odd = nb.sinusoidal_features(t, 7)
    assert odd.shape == (3, 7) and torch.allclose(odd[:, :6], want, atol=1e-4) and torch.all(odd[:, 6] == 0), (
        "odd dim: append a zero column")
    g = nb.sinusoidal_features(t, 6, max_period=100.0)
    assert torch.allclose(g[:, 1], torch.cos(1000 * t * 100.0 ** (-1 / 3)), atol=1e-4), "use max_period"


def test_stage2_time_embedding(nb):
    torch.manual_seed(3)
    e = nb.TimeEmbedding(8)
    out = e(torch.tensor([0.1, 0.5, 0.9]))
    assert out.shape == (3, 8), "(B,) -> (B, dim)"
    assert not torch.allclose(out[0], out[1]), "different times, different embeddings"
    n = sum(p.numel() for p in e.parameters())
    assert n == 2 * (8 * 8 + 8), f"a 2-layer MLP dim -> dim -> dim on the sinusoidal features: 144 parameters, got {n}"
    e2 = nb.TimeEmbedding(8, max_period=50.0)
    e2.load_state_dict(e.state_dict())
    assert not torch.allclose(e2(torch.tensor([0.5])), e(torch.tensor([0.5]))), "max_period must be used"


def test_stage2_time_mlp(nb):
    torch.manual_seed(4)
    m = _randomize(nb.TimeMLP(3, hidden_dim=32, n_layers=2, time_dim=16))
    x = torch.randn(5, 3)
    t = torch.full((5,), 0.3)
    out = m(x, t)
    assert out.shape == (5, 3), "model(x (B, data_dim), t (B,)) -> (B, data_dim)"
    assert not torch.allclose(out, m(x, torch.full((5,), 0.7)), atol=1e-4), "the output must depend on t"
    with pytest.raises(ValueError):
        m(x, t, torch.zeros(5, dtype=torch.long))
    c = _randomize(nb.TimeMLP(2, hidden_dim=32, n_layers=2, time_dim=16, n_classes=4), seed=5)
    tables = [mod for mod in c.modules() if isinstance(mod, torch.nn.Embedding)]
    assert len(tables) == 1 and tables[0].num_embeddings == 5, "one class table with n_classes + 1 = 5 rows (last = null)"
    x2 = torch.randn(4, 2)
    t2 = torch.full((4,), 0.5)
    y = torch.tensor([0, 1, 2, 3])
    assert not torch.allclose(c(x2, t2, y), c(x2, t2, torch.zeros(4, dtype=torch.long)), atol=1e-4), (
        "the output must depend on y")
    assert torch.allclose(c(x2, t2), c(x2, t2, torch.full((4,), 4)), atol=1e-6), "y=None means the null label n_classes"
    small = sum(p.numel() for p in nb.TimeMLP(2, hidden_dim=16, n_layers=1).parameters())
    deeper = sum(p.numel() for p in nb.TimeMLP(2, hidden_dim=16, n_layers=3).parameters())
    wider = sum(p.numel() for p in nb.TimeMLP(2, hidden_dim=64, n_layers=1).parameters())
    assert small < deeper and small < wider, "hidden_dim and n_layers must change the network"


# ---------------------------------------------------------------- stage 3: the loss


def test_stage3_ddpm_loss_by_hand(nb):
    s = nb.DiffusionSchedule(10, beta_start=0.01, beta_end=0.2)
    seen = []

    def model(xt, t, y=None):
        seen.append((xt.clone(), t.clone(), y))
        return 0.5 * xt

    x0 = torch.randn(6, 2, generator=_gen(6))
    y = torch.arange(6)
    got = nb.ddpm_loss(model, s, x0, y=y, generator=_gen(7))
    g = _gen(7)
    t = torch.randint(10, (6,), generator=g)
    noise = torch.randn(x0.shape, generator=g)
    xt = s.q_sample(x0, t, noise)
    assert torch.allclose(seen[0][1], t.float() / 10), "the model sees t = step / n_steps as floats"
    assert seen[0][2] is y, "pass y through to the model"
    assert torch.allclose(seen[0][0], xt, atol=1e-6), (
        "draw t = torch.randint(n_steps, (B,), generator) first, then noise = torch.randn(x0.shape, generator)")
    want = ((0.5 * xt - noise) ** 2).mean()
    assert torch.allclose(got, want, atol=1e-6), f"MSE between prediction and noise: {want.item():.5f}, got {float(got):.5f}"


def test_stage3_oracle_loss_is_the_floor(nb):
    s = nb.DiffusionSchedule(100, beta_start=1e-3, beta_end=0.2)
    mu, sd = 1.5, 0.5
    torch.manual_seed(8)
    x0 = mu + sd * torch.randn(50000, 1)
    got = nb.ddpm_loss(_Oracle(s, mu, sd), s, x0, generator=_gen(9)).item()
    ab = s.alpha_bars.double()
    floor = (ab * sd ** 2 / (ab * sd ** 2 + 1 - ab)).mean().item()
    assert abs(got - floor) < 0.01, (
        f"with the exact noise predictor the loss is the irreducible E_t[ab s^2 / (ab s^2 + 1 - ab)] = {floor:.4f}; got "
        f"{got:.4f}")


def test_stage3_fit(nb):
    s = nb.DiffusionSchedule(100, beta_start=1e-3, beta_end=0.2)
    rng = np.random.default_rng(10)
    X = 1.5 + 0.5 * rng.normal(size=(4000, 1))
    torch.manual_seed(11)
    m = nb.TimeMLP(1, hidden_dim=32, n_layers=2, time_dim=32)
    _warm_up_adam()
    bounded(nb.fit_ddpm, nb.TimeMLP(1, hidden_dim=8, n_layers=1, time_dim=8), s, X, n_iters=3, batch_size=8,
            budget=3_000_000, what="fit_ddpm", hint="loop over n_iters minibatches")
    losses = nb.fit_ddpm(_cap_module(m), s, X, n_iters=300, batch_size=256, lr=5e-3, seed=0)
    assert len(losses) == 300 and all(isinstance(v, float) for v in losses), "300 losses as floats"
    ab = s.alpha_bars.double()
    floor = (ab * 0.25 / (ab * 0.25 + 1 - ab)).mean().item()
    with torch.no_grad():
        x0 = torch.tensor(X[:4000], dtype=torch.float32)
        got = np.mean([nb.ddpm_loss(m, s, x0, generator=_gen(k)).item() for k in range(5)])
    assert got < floor + 0.03, (
        f"after 300 steps the loss should be near the irreducible floor {floor:.3f}; got {got:.3f}")


def test_stage3_fit_options(nb):
    s = nb.DiffusionSchedule(10, beta_start=0.01, beta_end=0.2)
    X = np.random.default_rng(12).normal(size=(100, 2))
    m = nb.TimeMLP(2, hidden_dim=8, n_layers=1, time_dim=8)
    modes, sizes = [], []
    def hook(mod, inp):
        modes.append(mod.training)
        sizes.append(inp[0].shape[0])

    m.register_forward_pre_hook(hook)
    m.eval()
    nb.fit_ddpm(_cap_module(m), s, X, n_iters=3, batch_size=13, seed=0)
    assert all(modes) and len(modes) == 3, "fit_ddpm must put the model in train mode (dropout, if any, on)"
    assert set(sizes) == {13}, "minibatches of batch_size"
    a, b = nb.TimeMLP(2, hidden_dim=8, n_layers=1, time_dim=8), nb.TimeMLP(2, hidden_dim=8, n_layers=1, time_dim=8)
    b.load_state_dict(a.state_dict())
    _cap_module(a), _cap_module(b)
    assert nb.fit_ddpm(a, s, X, n_iters=3, seed=4) == nb.fit_ddpm(b, s, X, n_iters=3, seed=4), "seeded"
    before = [p.clone() for p in a.parameters()]
    nb.fit_ddpm(a, s, X, n_iters=2, lr=0.0)
    assert all(torch.equal(p, q) for p, q in zip(before, a.parameters())), "use the lr argument"


# ---------------------------------------------------------------- stage 4: ancestral sampling


def test_stage4_sample_by_hand(nb):
    s = nb.DiffusionSchedule(3, beta_start=0.1, beta_end=0.3)
    seen = []

    class Toy(torch.nn.Module):
        def forward(self, x, t, y=None):
            seen.append((t.clone(), y, self.training))
            assert len(seen) <= 1000, "ddpm_sample never stops calling the model"
            return 0.1 * x + t[:, None]

    model = Toy()
    y = torch.tensor([1, 2])
    got = bounded(nb.ddpm_sample, model, s, (2, 2), y=y, generator=_gen(13), budget=300_000, what="ddpm_sample",
                  hint="one step per i = n_steps-1 .. 0")
    g = _gen(13)
    x = torch.randn((2, 2), generator=g)
    for i in (2, 1, 0):
        b, a, ab = s.betas[i].item(), s.alphas[i].item(), s.alpha_bars[i].item()
        eps = 0.1 * x + i / 3
        x = (x - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
        if i > 0:
            x = x + math.sqrt(b) * torch.randn((2, 2), generator=g)
    assert [round(c[0][0].item(), 4) for c in seen] == [0.6667, 0.3333, 0.0], (
        "call the model at t = i / n_steps for i = n_steps-1 down to 0")
    assert all(c[1] is y for c in seen), "pass y to the model"
    assert torch.allclose(got, x, atol=1e-5), (
        f"expected {x.tolist()}, got {got.tolist()}: x_T = torch.randn(shape, generator), then mean = (x - beta/sqrt(1 - "
        "ab) eps)/sqrt(alpha), plus sqrt(beta) z (z = torch.randn(shape, generator)) except at the last step")
    assert not any(c[2] for c in seen) and model.training, (
        "sample in eval mode (dropout off) and restore the model's previous mode afterwards")


def test_stage4_oracle_recovers_gaussian(nb):
    s = nb.DiffusionSchedule(100, beta_start=1e-3, beta_end=0.2)
    oracle = _Oracle(s, 1.5, 0.5)
    x = nb.ddpm_sample(oracle, s, (40000, 1), generator=_gen(14))
    assert len(oracle.calls) == 100, f"one network call per step: 100, got {len(oracle.calls)}"
    assert abs(x.mean().item() - 1.5) < 0.02 and abs(x.std().item() - 0.5) < 0.02, (
        f"with the exact noise predictor, samples must follow the data N(1.5, 0.5^2); got mean {x.mean().item():.3f}, "
        f"std {x.std().item():.3f}")
    assert not x.requires_grad
    w = torch.nn.Parameter(torch.tensor(0.5))
    out = nb.ddpm_sample(_capped(lambda v, t, y=None: w * v), s, (3, 1), generator=_gen(15))
    assert not out.requires_grad, "no gradients during sampling"


# ---------------------------------------------------------------- stage 5: the U-Net


def test_stage5_unet_shapes_and_divisibility(nb):
    torch.manual_seed(16)
    u = nb.TimeUNet(3, base_channels=8, channel_mults=(1, 2, 2))
    x = torch.randn(2, 3, 8, 8)
    assert u(x, torch.rand(2)).shape == x.shape, "the output has the input's shape"
    with pytest.raises(ValueError):
        u(torch.randn(1, 3, 6, 6), torch.rand(1))
    u2 = nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2))
    assert u2(torch.randn(2, 1, 6, 6), torch.rand(2)).shape == (2, 1, 6, 6), (
        "channel_mults=(1, 2) downsamples once, so 6 x 6 works; (1, 2, 2) needs H, W divisible by 4")
    small = sum(p.numel() for p in nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2)).parameters())
    wide = sum(p.numel() for p in nb.TimeUNet(1, base_channels=16, channel_mults=(1, 2)).parameters())
    assert wide > small, "base_channels must set the widths"


def test_stage5_unet_conditioning(nb):
    torch.manual_seed(17)
    u = _randomize(nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2), n_classes=3), scale=0.1).eval()
    x = torch.randn(3, 1, 8, 8)
    t = torch.full((3,), 0.4)
    y = torch.tensor([0, 1, 2])
    base = u(x, t, y)
    assert not torch.allclose(base, u(x, torch.full((3,), 0.8), y), atol=1e-4), "the output must depend on t"
    assert not torch.allclose(base, u(x, t, torch.tensor([2, 0, 1])), atol=1e-4), "the output must depend on y"
    assert torch.allclose(u(x, t), u(x, t, torch.full((3,), 3)), atol=1e-5), "y=None means the null label n_classes"
    tables = [m for m in u.modules() if isinstance(m, torch.nn.Embedding)]
    assert len(tables) == 1 and tables[0].num_embeddings == 4, "one class table with n_classes + 1 rows"
    unc = nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2))
    with pytest.raises(ValueError):
        unc(x, t, y)


def test_stage5_unet_dropout_and_receptive_field(nb):
    torch.manual_seed(18)
    u = _randomize(nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2, 2), dropout=0.5), scale=0.1)
    x = torch.randn(2, 1, 16, 16)
    t = torch.full((2,), 0.5)
    u.train()
    assert not torch.allclose(u(x, t), u(x, t)), "dropout=0.5 must make train-mode outputs random"
    u.eval()
    assert torch.allclose(u(x, t), u(x, t)), "eval mode is deterministic"
    d = _randomize(nb.TimeUNet(1, base_channels=8, channel_mults=(1, 2, 2), dropout=0.0), scale=0.1).train()
    assert torch.allclose(d(x, t), d(x, t)), "dropout=0.0: no randomness even in train mode"
    xg = x[:1].clone().requires_grad_(True)
    d.eval()(xg, t[:1])[0, 0, 0, 0].backward()
    assert xg.grad[0, 0, -1, -1].abs() > 0, (
        "the output's top-left pixel must depend on the input's bottom-right one: a denoiser needs global context "
        "(downsampling levels and/or attention at the lowest resolution)")
