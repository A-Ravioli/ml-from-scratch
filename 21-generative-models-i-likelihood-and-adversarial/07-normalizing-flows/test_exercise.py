"""Checks for exercise.py (normalizing flows).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Parameters are randomized
before checking, so a flow that starts as the identity can't pass by accident. The coupling layer and the
full RealNVP are checked for exact invertibility, for log-determinants against log|det J| computed with
autograd, for the masked coordinates passing through unchanged, and (in 2-D) for a density that
integrates to 1 on a grid. Sampling is checked for reproducibility and for being the inverse map of
base samples. Training is checked on a toy density under `mlfs.testing.bounded`. The logit transform's
log-determinant is checked against autograd, and the bits/dim pipeline against a hand recomputation with
the same random draws.
"""
import math

import numpy as np
import pytest

from mlfs.testing import bounded

torch = pytest.importorskip("torch")
F = torch.nn.functional


def _randomize(module, scale=0.4, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(scale * torch.randn(p.shape, generator=g))
    return module


def _logdet_autograd(fn, x):
    """log |det d fn(x_b) / d x_b| for each row of x, by building the Jacobian row by row."""
    out = []
    for xb in x:
        xb = xb.clone().requires_grad_(True)
        y = fn(xb[None])[0]
        rows = [torch.autograd.grad(y[i], xb, retain_graph=True)[0] for i in range(len(y))]
        out.append(torch.linalg.slogdet(torch.stack(rows))[1])
    return torch.stack(out)


# ---------------------------------------------------------------- stage 1: the coupling layer


def test_stage1_standard_normal_log_prob(nb):
    z = torch.randn(5, 3, generator=torch.Generator().manual_seed(0))
    want = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum(1)
    assert torch.allclose(nb.standard_normal_log_prob(z), want, atol=1e-5), "sum of log N(z_i; 0, 1) over dims, (B,)"
    z4 = torch.randn(2, 1, 2, 2)
    assert nb.standard_normal_log_prob(z4).shape == (2,), "sum over every non-batch dim"


def test_stage1_coupling_invertible_and_masked(nb):
    torch.manual_seed(1)
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    layer = _randomize(nb.AffineCoupling(4, mask, hidden_dim=16))
    x = torch.randn(6, 4)
    y, ld = layer(x)
    assert y.shape == x.shape and ld.shape == (6,), "forward -> (y (B, dim), log_det (B,))"
    assert torch.allclose(y[:, mask == 1], x[:, mask == 1]), "coordinates with mask 1 pass through unchanged"
    assert not torch.allclose(y[:, mask == 0], x[:, mask == 0]), "the other coordinates are transformed"
    x_back, ld_inv = layer.inverse(y)
    assert torch.allclose(x_back, x, atol=1e-5), "inverse(forward(x)) must give back x exactly"
    assert torch.allclose(ld_inv, -ld, atol=1e-5), "the inverse's log-det is minus the forward's"


def test_stage1_coupling_logdet(nb):
    torch.manual_seed(2)
    for mask in (torch.tensor([0.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])):
        layer = _randomize(nb.AffineCoupling(3, mask, hidden_dim=16), seed=3)
        x = torch.randn(4, 3)
        _, ld = layer(x)
        want = _logdet_autograd(lambda v: layer(v)[0], x)
        assert torch.allclose(ld, want, atol=1e-4), (
            f"mask {mask.tolist()}: log_det {ld.detach().numpy().round(4)} must equal log|det J| = "
            f"{want.numpy().round(4)}: the sum of s(x_m) over the TRANSFORMED (mask 0) coordinates only")


def test_stage1_coupling_conditioning(nb):
    torch.manual_seed(4)
    mask = torch.tensor([1.0, 0.0])
    layer = _randomize(nb.AffineCoupling(2, mask, hidden_dim=8), seed=5)
    x = torch.tensor([[0.3, -1.0], [0.3, 2.0], [0.3, 5.0]])
    y, ld = layer(x)
    assert torch.allclose(ld, ld[0].expand(3), atol=1e-6), (
        "s and t may depend only on the masked coordinates x_m: changing x[1] must not change the log-det")
    slope = (y[2, 1] - y[1, 1]) / 3.0
    assert torch.allclose(y[1, 1] - y[0, 1], 3.0 * slope, atol=1e-4), (
        "for fixed x_m, the transformed coordinate is affine in its own input: y = x exp(s) + t")


# ---------------------------------------------------------------- stage 2: RealNVP


def test_stage2_structure_and_inverse(nb):
    torch.manual_seed(6)
    flow = nb.RealNVP(3, n_layers=4, hidden_dim=16)
    assert len(flow.layers) == 4 and all(isinstance(l, nb.AffineCoupling) for l in flow.layers), (
        "self.layers: n_layers AffineCoupling layers")
    masks = [l.mask.tolist() for l in flow.layers]
    assert masks == [[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]], f"masks alternate (i + layer) % 2; got {masks}"
    _randomize(flow, 0.3, 7)
    x = torch.randn(5, 3)
    z, ld = flow(x)
    assert z.shape == (5, 3) and ld.shape == (5,)
    assert torch.allclose(flow.inverse(z), x, atol=1e-4), "inverse(forward(x)) = x through the whole stack"
    want = _logdet_autograd(lambda v: flow(v)[0], x)
    assert torch.allclose(ld, want, atol=1e-3), "the flow's log-det is the sum of its layers' log-dets"
    assert torch.allclose(flow.log_prob(x), nb.standard_normal_log_prob(z) + ld, atol=1e-5), (
        "log_prob(x) = log N(f(x); 0, I) + log|det df/dx|")


def test_stage2_density_integrates_to_one(nb):
    torch.manual_seed(8)
    flow = _randomize(nb.RealNVP(2, n_layers=4, hidden_dim=16), 0.3, 9)
    g = torch.linspace(-8, 8, 401)
    h = (g[1] - g[0]).item()
    grid = torch.cartesian_prod(g, g)
    with torch.no_grad():
        mass = torch.exp(flow.log_prob(grid)).sum().item() * h * h
    assert abs(mass - 1.0) < 0.02, (
        f"an exact density must integrate to 1; numerical integral over [-8, 8]^2 = {mass:.4f}. a wrong log-det "
        "(missing, doubled, or with the wrong sign) breaks this")


def test_stage2_sample(nb):
    torch.manual_seed(10)
    flow = _randomize(nb.RealNVP(2, n_layers=2, hidden_dim=8), 0.3, 11)
    s = flow.sample(50, generator=torch.Generator().manual_seed(0))
    assert s.shape == (50, 2) and not s.requires_grad
    z = torch.randn((50, 2), generator=torch.Generator().manual_seed(0))
    assert torch.allclose(s, flow.inverse(z).detach(), atol=1e-5), (
        "sample = inverse(torch.randn((n, dim), generator=generator))")


# ---------------------------------------------------------------- stage 3: training


def test_stage3_fit(nb):
    torch.manual_seed(12)
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=1000), rng.normal(size=1000) * 0.3]
    X[:, 1] += 0.5 * X[:, 0] ** 2 - 0.5  # a banana: not Gaussian, but an easy flow target
    flow = nb.RealNVP(2, n_layers=4, hidden_dim=32)
    losses = bounded(nb.fit_flow, flow, X, n_steps=300, batch_size=128, lr=3e-3, seed=0,
                     budget=30_000_000, what="fit_flow")
    assert isinstance(losses, list) and len(losses) == 300 and all(isinstance(v, float) for v in losses), (
        "fit_flow returns the 300 per-step losses as floats")
    with torch.no_grad():
        nll = -flow.log_prob(torch.tensor(X, dtype=torch.float32)).mean().item()
    gauss_nll = 0.5 * np.linalg.slogdet(2 * np.pi * np.e * np.cov(X.T))[1]
    assert nll < gauss_nll - 0.3, (
        f"after 300 steps the flow's NLL ({nll:.3f}) should beat the best Gaussian fit ({gauss_nll:.3f}) by 0.3 nats")
    flow2 = nb.RealNVP(2, n_layers=4, hidden_dim=32)
    flow3 = nb.RealNVP(2, n_layers=4, hidden_dim=32)
    flow3.load_state_dict(flow2.state_dict())
    a = nb.fit_flow(flow2, X, n_steps=5, batch_size=64, seed=1)
    b = nb.fit_flow(flow3, X, n_steps=5, batch_size=64, seed=1)
    assert a == b, "minibatches come from a torch.Generator seeded with `seed`: two identical flows train identically"
    before = [q.detach().clone() for q in flow2.parameters()]
    nb.fit_flow(flow2, X, n_steps=3, lr=0.0, seed=2)
    assert all(torch.equal(q0, q) for q0, q in zip(before, flow2.parameters())), "use the `lr` argument (lr=0: no change)"


# ---------------------------------------------------------------- stage 4: images


def test_stage4_logit_transform(nb):
    x = torch.rand(3, 5, generator=torch.Generator().manual_seed(13))
    y, ld = nb.logit_transform(x, alpha=0.05)
    p = 0.05 + 0.9 * x
    assert torch.allclose(y, torch.log(p / (1 - p)), atol=1e-5), "y = logit(alpha + (1 - 2 alpha) x)"
    want = _logdet_autograd(lambda v: nb.logit_transform(v, alpha=0.05)[0], x)
    assert ld.shape == (3,) and torch.allclose(ld, want, atol=1e-4), (
        "log_det = sum over dims of log((1 - 2 alpha) / (p (1 - p))), the log-derivative of the map")
    y0, _ = nb.logit_transform(torch.zeros(1, 2), alpha=0.1)
    assert torch.allclose(y0, torch.full((1, 2), math.log(0.1 / 0.9)), atol=1e-6), "alpha keeps 0 away from -inf"
    assert nb.logit_transform(torch.rand(2, 1, 3, 3))[1].shape == (2,), "images: sum over C, H, W"


class _GaussianFlow:
    def __init__(self):
        self.shapes = []

    def log_prob(self, y):
        self.shapes.append(tuple(y.shape))
        return (-0.5 * y ** 2 - 0.5 * math.log(2 * math.pi)).flatten(1).sum(1)


def test_stage4_bits_per_dim(nb):
    x = torch.randint(0, 256, (4, 1, 3, 3), generator=torch.Generator().manual_seed(14))
    flow = _GaussianFlow()
    got = nb.flow_bits_per_dim(flow, x, generator=torch.Generator().manual_seed(15))
    assert flow.shapes == [(4, 1, 3, 3)], (
        f"pass y to flow.log_prob in the data's own shape (an image flow needs (B, C, H, W)); got {flow.shapes}")
    u = (x.float() + torch.rand(x.shape, generator=torch.Generator().manual_seed(15))) / 256
    p = 0.05 + 0.9 * u
    y = torch.log(p / (1 - p)).flatten(1)
    log_det = (math.log(0.9) - torch.log(p) - torch.log(1 - p)).flatten(1).sum(1)
    log_p = (-0.5 * y ** 2 - 0.5 * math.log(2 * math.pi)).sum(1) + log_det
    want = (-log_p + 9 * math.log(256)) / (9 * math.log(2))
    assert got.shape == (4,) and torch.allclose(got, want, atol=1e-4), (
        f"expected {want.numpy().round(4)}, got {got.numpy().round(4)}: dequantize with `generator`, logit-transform "
        "(adding its log-det to the flow's log_prob), then discrete_nll and bits_per_dim over D = C*H*W dims")
    xb = torch.randint(0, 2, (3, 4), generator=torch.Generator().manual_seed(16))
    b2 = nb.flow_bits_per_dim(_GaussianFlow(), xb, n_levels=2, alpha=0.1, generator=torch.Generator().manual_seed(0))
    u = (xb.float() + torch.rand(xb.shape, generator=torch.Generator().manual_seed(0))) / 2
    p = 0.1 + 0.8 * u
    y = torch.log(p / (1 - p))
    log_p = (-0.5 * y ** 2 - 0.5 * math.log(2 * math.pi)).sum(1) + (math.log(0.8) - torch.log(p) - torch.log(1 - p)).sum(1)
    want2 = (-log_p + 4 * math.log(2)) / (4 * math.log(2))
    assert torch.allclose(b2, want2, atol=1e-4), (
        "binary data (n_levels=2) with alpha=0.1: pass n_levels to dequantize and discrete_nll, and alpha to "
        "logit_transform")


def test_stage3_fit_batch_size(nb):
    X = np.random.default_rng(0).normal(size=(200, 2))
    f1, f2 = nb.RealNVP(2, n_layers=2, hidden_dim=8), nb.RealNVP(2, n_layers=2, hidden_dim=8)
    f2.load_state_dict(f1.state_dict())
    calls = []
    f2.register_forward_pre_hook(lambda mod, inp: calls.append(inp[0].shape[0]))
    nb.fit_flow(f2, X, n_steps=3, batch_size=17, seed=0)
    assert calls and all(c == 17 for c in calls), f"each step's minibatch has batch_size rows; got {calls}"


def test_stage4_logit_alpha_default(nb):
    y, _ = nb.logit_transform(torch.zeros(1, 1))
    assert math.isclose(y.item(), math.log(0.05 / 0.95), rel_tol=1e-5), "alpha defaults to 0.05"
