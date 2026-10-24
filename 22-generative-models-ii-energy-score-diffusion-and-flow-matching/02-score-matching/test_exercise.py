"""Checks for exercise.py (score matching).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Gaussian data are used
throughout because their noisy scores are known exactly: -(x - mu) / (s^2 + sigma^2). The denoising loss is
checked against a hand computation with the same random draws, for its sigma^2 weighting, and for its expected
value at the exact score (0.5 D s^2 / (s^2 + sigma^2), the irreducible part). Sliced score matching is checked
against the exact Hyvarinen objective of a linear score and for its gradient (which needs create_graph). The
noise-conditional network is trained on Gaussian data and compared with the exact noisy score at two noise
levels. Annealed Langevin is checked step by step against a hand computation and in distribution.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")


def _capped(fn, limit=20_000, what="the sampler"):
    """wrap a score so a sampler that never stops fails (after `limit` calls) instead of hanging."""
    count = [0]

    def wrapped(*args, **kwargs):
        count[0] += 1
        if count[0] > limit:
            raise AssertionError(f"{what} called the score more than {limit:,} times: a loop never ends")
        return fn(*args, **kwargs)

    return wrapped


def _cap_module(module, limit=20_000):
    """the same for a network: fail after `limit` forward calls."""
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


MU = torch.tensor([1.0, -1.0])
S = torch.tensor([0.5, 1.0])


def _exact(x, sig):
    """the exact score of N(MU, diag(S^2)) blurred by N(0, sigma^2 I)."""
    return -(x - MU) / (S ** 2 + sig[:, None] ** 2)


# ---------------------------------------------------------------- stage 1: denoising score matching


def test_stage1_dsm_by_hand(nb):
    x = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])
    seen = []

    def score(xt, sig):
        seen.append((xt.clone(), sig.clone()))
        return -xt

    got = nb.dsm_loss(score, x, 0.5, generator=_gen(0))
    z = torch.randn(x.shape, generator=_gen(0))
    xt = x + 0.5 * z
    want = (0.5 * 0.25 * ((-xt + z / 0.5) ** 2).sum(1)).mean()
    assert torch.allclose(seen[0][0], xt, atol=1e-6), (
        "x_tilde = x + sigma z with z = torch.randn(x.shape, generator=generator)")
    assert seen[0][1].shape == (3,) and torch.allclose(seen[0][1], torch.full((3,), 0.5)), (
        "call score_fn(x_tilde, sigma_vec) with sigma_vec a (B,) tensor, even when sigma is a float")
    assert torch.allclose(got, want, atol=1e-5), (
        f"expected {want.item():.5f}, got {float(got):.5f}: mean over the batch of 0.5 sigma^2 sum_d (s - target)^2 with "
        "target -(x_tilde - x)/sigma^2")


def test_stage1_dsm_weighting(nb):
    x = torch.zeros(4000, 3)
    zero = lambda xt, sig: torch.zeros_like(xt)
    for sigma in (0.1, 2.0):
        got = nb.dsm_loss(zero, x, sigma, generator=_gen(1)).item()
        assert abs(got - 1.5) < 0.1, (
            f"sigma={sigma}: a zero score has loss 0.5 sigma^2 E||z/sigma||^2 = 0.5 D = 1.5; got {got:.3f}. "
            "the sigma^2 weight makes every noise level contribute on the same scale")
    sig = torch.tensor([0.1, 1.0, 10.0, 1.0])
    seen = []
    nb.dsm_loss(lambda xt, s: seen.append(s) or torch.zeros_like(xt), torch.zeros(4, 2), sig, generator=_gen(2))
    assert torch.equal(seen[0], sig), "a (B,) sigma gives every example its own noise level"
    xt_scale = []
    nb.dsm_loss(lambda xt, s: xt_scale.append(xt.abs().mean(1)) or torch.zeros_like(xt),
                torch.zeros(4, 500), sig, generator=_gen(3))
    ratio = xt_scale[0] / sig
    assert torch.allclose(ratio, torch.full((4,), math.sqrt(2 / math.pi)), rtol=0.15), (
        "example b is perturbed with its own sigma_b")


def test_stage1_dsm_optimum(nb):
    torch.manual_seed(0)
    x = torch.randn(40000, 2) * S + MU
    sigma = 0.5
    best = nb.dsm_loss(_exact, x, sigma, generator=_gen(4)).item()
    floor = (0.5 * S ** 2 / (S ** 2 + sigma ** 2)).sum().item()
    assert abs(best - floor) < 0.02, (
        f"at the exact noisy score the loss is the irreducible 0.5 sum_d s_d^2 / (s_d^2 + sigma^2) = {floor:.3f}; "
        f"got {best:.3f}")
    for c in (0.8, 1.2):
        worse = nb.dsm_loss(lambda xt, sg: c * _exact(xt, sg), x, sigma, generator=_gen(4)).item()
        assert worse > best, "scaling the exact score away from optimum must increase the loss"
    w = torch.nn.Parameter(torch.tensor(1.0))
    nb.dsm_loss(lambda xt, sg: w * _exact(xt, sg), x[:100], sigma, generator=_gen(5)).backward()
    assert w.grad is not None and w.grad.abs() > 0, "the loss must be differentiable in the score's parameters"
    imgs = torch.randn(3, 1, 4, 4, generator=_gen(6))
    v = nb.dsm_loss(lambda xt, sg: -xt, imgs, 0.3, generator=_gen(7))
    assert v.dim() == 0, "images (B, C, H, W) work too: sum over every non-batch dim, mean over the batch"


# ---------------------------------------------------------------- stage 2: sliced score matching


def test_stage2_ssm_linear_score(nb):
    torch.manual_seed(1)
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    x = torch.randn(300, 2)
    score = lambda v: -v @ A
    got = nb.sliced_score_matching_loss(score, x, n_projections=2000, generator=_gen(8)).item()
    exact = (0.5 * ((x @ A) ** 2).sum(1)).mean().item() - torch.trace(A).item()
    assert abs(got - exact) < 0.05 * abs(exact) + 0.05, (
        f"for s(x) = -A x the expected SSM loss is E[0.5 ||s||^2] + tr(grad s) = {exact:.3f}; got {got:.3f}. "
        "the trace comes from v^T grad_x(v^T s(x)) averaged over Gaussian v")


def test_stage2_ssm_by_hand_and_projections(nb):
    torch.manual_seed(2)
    W = torch.randn(3, 3)
    score = lambda v: torch.tanh(v @ W)
    x = torch.randn(5, 3)
    got = nb.sliced_score_matching_loss(score, x, n_projections=3, generator=_gen(9))
    vs = torch.randn((3, 5, 3), generator=_gen(9))
    xg = x.clone().requires_grad_(True)
    s = score(xg)
    total = 0.0
    for v in vs:
        sv = (s * v).sum(1)
        (gv,) = torch.autograd.grad(sv.sum(), xg, retain_graph=True)
        total = total + ((gv * v).sum(1) + 0.5 * sv ** 2).mean()
    want = total / 3
    assert torch.allclose(got, want, atol=1e-5), (
        f"expected {float(want):.5f}, got {float(got):.5f}: v = torch.randn((n_projections, *x.shape), generator), "
        "then the mean over projections and the batch of v^T grad(v^T s) + 0.5 (v^T s)^2")
    assert x.grad is None and not x.requires_grad, "don't modify the caller's x (detach a copy before requires_grad_)"


def test_stage2_ssm_gradient(nb):
    torch.manual_seed(3)
    A = torch.nn.Parameter(torch.tensor([[1.5, 0.0], [0.0, 0.5]]))
    x = torch.randn(400, 2)
    loss = nb.sliced_score_matching_loss(lambda v: -v @ A, x, n_projections=500, generator=_gen(10))
    loss.backward()
    want = (x.T @ x) / len(x) @ A.detach() - torch.eye(2)
    assert A.grad is not None and torch.allclose(A.grad, want, atol=0.08), (
        f"d loss / dA should be about E[x x^T] A - I = {want.numpy().round(2).tolist()}; got "
        f"{None if A.grad is None else A.grad.numpy().round(2).tolist()}. the -I part comes through the trace term: "
        "compute grad_x(v^T s) with create_graph=True so it can be differentiated again")


# ---------------------------------------------------------------- stage 3: noise-conditional score network


def test_stage3_geometric_sigmas(nb):
    s = nb.geometric_sigmas(10.0, 0.01, 4)
    assert isinstance(s, torch.Tensor) and s.shape == (4,), "a (n,) tensor"
    assert torch.allclose(s, torch.tensor([10.0, 1.0, 0.1, 0.01]), rtol=1e-4), (
        f"geometric from sigma_max to sigma_min: [10, 1, 0.1, 0.01]; got {s.tolist()}")


def test_stage3_network(nb):
    torch.manual_seed(4)
    m = nb.NoiseConditionalScore(3, hidden_dim=16, n_layers=2)
    x = torch.randn(5, 3)
    out = m(x, torch.full((5,), 0.5))
    assert out.shape == (5, 3), "score (B, data_dim)"
    assert not torch.allclose(out, m(x, torch.full((5,), 2.0))), "the output must depend on sigma"
    assert torch.allclose(m(x, 0.5), out, atol=1e-6), "a float sigma means the same sigma for every row"
    n = sum(p.numel() for p in m.parameters())
    assert n == 4 * 16 + 16 + 16 * 16 + 16 + 16 * 3 + 3, f"an MLP (3 + 1) -> 16 -> 16 -> 3 has 403 parameters; got {n}"
    raw = [m(x, s) * s for s in (0.1, 1.0, 10.0)]
    assert not torch.allclose(raw[0], raw[1], atol=1e-4) and not torch.allclose(raw[1], raw[2], atol=1e-4), (
        "the network must see log sigma as an input (not only through the 1/sigma scaling)")
    lin = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        lin[-1].bias.fill_(1.0)
    assert torch.allclose(m(x, 0.25), torch.full((5, 3), 4.0)), (
        "the output is the MLP's output divided by sigma: with only the last bias set to 1, the score is 1/sigma")
    one = nb.NoiseConditionalScore(2, hidden_dim=4, n_layers=1)  # Linear(3 -> 4), SiLU, Linear(4 -> 2)
    l1, l2 = [mod for mod in one.modules() if isinstance(mod, torch.nn.Linear)]
    with torch.no_grad():
        for p in one.parameters():
            p.zero_()
        l1.weight[:, 2] = 1.0  # read only the last input, the noise-level feature
        l2.weight.fill_(0.25)
    e = math.e
    want = torch.nn.functional.silu(torch.tensor(1.0)) / e  # the feature is log(e) = 1, the output divided by e
    assert torch.allclose(one(torch.zeros(3, 2), e), torch.full((3, 2), want.item()), atol=1e-6), (
        "the network's input is the concatenation [x, log sigma] (log sigma last); with weights that read only that "
        "feature, the score at sigma = e must be silu(log e) / e")


def test_stage3_fit_matches_exact_score(nb):
    torch.manual_seed(5)
    X = (torch.randn(4000, 2) * S + MU).numpy()
    m = nb.NoiseConditionalScore(2, hidden_dim=64, n_layers=2)
    _warm_up_adam()
    bounded(nb.fit_dsm, nb.NoiseConditionalScore(2, hidden_dim=8, n_layers=1), X, [0.3], n_iters=3, batch_size=8,
            budget=2_000_000, what="fit_dsm", hint="loop over n_iters minibatches")
    losses = nb.fit_dsm(_cap_module(m), X, [0.3, 1.0], n_iters=600, batch_size=256, lr=3e-3, seed=0)
    assert len(losses) == 600 and all(isinstance(v, float) for v in losses), "600 losses as floats"
    xs = torch.tensor(X[:500])
    for sg in (0.3, 1.0):
        sv = torch.full((500,), sg)
        with torch.no_grad():
            err = ((m(xs, sv) - _exact(xs, sv)).norm(dim=1).mean() / _exact(xs, sv).norm(dim=1).mean()).item()
        assert err < 0.3, (
            f"sigma={sg}: the trained score is {100 * err:.0f}% off the exact noisy score -(x - mu)/(s^2 + sigma^2) on "
            "the data; check dsm_loss's target and weighting, and that sigmas are drawn per example")


def test_stage3_fit_options(nb):
    X = np.random.default_rng(6).normal(size=(200, 2))
    seen = []

    class Rec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)

        def forward(self, x, sig):
            seen.append((x.shape[0], sig.clone()))
            return self.lin(x)

    nb.fit_dsm(_cap_module(Rec()), X, [0.1, 0.5, 2.0], n_iters=4, batch_size=48, seed=0)
    assert len(seen) == 4 and {b for b, _ in seen} == {48}, "n_iters calls on batch_size examples"
    assert all(len(set(s.tolist())) > 1 for _, s in seen), "one sigma drawn PER EXAMPLE, not one per batch"
    vals = sorted({round(v, 4) for v in torch.cat([s for _, s in seen]).tolist()})
    assert vals == [0.1, 0.5, 2.0], f"every example gets a sigma drawn from `sigmas`; saw {vals}"
    r1, r2 = _cap_module(Rec()), _cap_module(Rec())
    r2.load_state_dict(r1.state_dict())
    assert nb.fit_dsm(r1, X, [0.5], n_iters=3, seed=1) == nb.fit_dsm(r2, X, [0.5], n_iters=3, seed=1), (
        "seeded: identical models give identical losses")
    before = [p.clone() for p in r1.parameters()]
    nb.fit_dsm(r1, X, [0.5], n_iters=2, lr=0.0)
    assert all(torch.equal(a, b) for a, b in zip(before, r1.parameters())), "use the lr argument"


# ---------------------------------------------------------------- stage 4: annealed Langevin


def test_stage4_annealed_by_hand(nb):
    sig = torch.tensor([1.0, 0.5])
    x0 = torch.tensor([[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]])
    seen = []

    def score(x, s):
        seen.append(s.clone())
        return -x / (1 + s[:, None] ** 2)

    got = bounded(nb.annealed_langevin, score, x0, sig, steps_per_sigma=2, eps=0.1, generator=_gen(11),
                  budget=200_000, what="annealed_langevin", hint="steps_per_sigma steps for each sigma, then stop")
    g = _gen(11)
    x = x0.clone()
    for s in (1.0, 0.5):
        a = 0.1 * s ** 2 / 0.25
        for _ in range(2):
            z = torch.randn(x.shape, generator=g)
            x = x + 0.5 * a * (-x / (1 + s ** 2)) + math.sqrt(a) * z
    assert torch.allclose(got, x, atol=1e-5), (
        f"expected {x.tolist()}, got {got.tolist()}: step alpha_i = eps sigma_i^2 / sigma_L^2 (sigma_L the last sigma), "
        "x + (alpha/2) score + sqrt(alpha) z, z = torch.randn(x.shape, generator), steps_per_sigma steps per level")
    assert len(seen) == 4 and all(s.shape == (3,) for s in seen), "score_fn(x, sigma_vec (B,)) once per step"
    assert [round(s[0].item(), 3) for s in seen] == [1.0, 1.0, 0.5, 0.5], "levels in the given (decreasing) order"
    assert torch.equal(x0, torch.tensor([[0.0, 2.0], [1.0, -1.0], [3.0, 0.0]])), "don't modify x0"


def test_stage4_annealed_distribution(nb):
    sig = torch.exp(torch.linspace(math.log(3.0), math.log(0.05), 8))
    x = nb.annealed_langevin(_capped(_exact), torch.zeros(20000, 2), sig, steps_per_sigma=50, eps=1e-3, generator=_gen(12))
    assert torch.allclose(x.mean(0), MU, atol=0.05), f"samples should be centred on {MU.tolist()}; got {x.mean(0).tolist()}"
    var = x.var(0)
    assert ((var > S ** 2 * 0.95) & (var < S ** 2 + 0.2)).all(), (
        f"with the exact scores the samples' variances should be just above the data's {(S ** 2).tolist()} (the last "
        f"levels barely move them); got {var.tolist()}")
    w = torch.nn.Parameter(torch.tensor(1.0))
    out = nb.annealed_langevin(_capped(lambda v, s: -w * v), torch.zeros(2, 2), sig[:2], steps_per_sigma=1, generator=_gen(13))
    assert not out.requires_grad, "sampling builds no graph: run under torch.no_grad()"
