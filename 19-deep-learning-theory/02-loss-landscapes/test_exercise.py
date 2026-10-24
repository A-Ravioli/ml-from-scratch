"""Checks for 19 · 02 loss landscapes.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. directions are checked
filter by filter on layers whose filters have very different norms; surfaces and interpolation paths
against losses known in closed form on hand-built models (asymmetric, so transposed grids fail);
sharpness against exact Hessians from `torch.autograd.functional.hessian`, including one whose
largest-magnitude eigenvalue is negative.
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



class Vec(nn.Module):
    """a model that is just a parameter vector (and an optional frozen one)."""

    def __init__(self, w, frozen=None):
        super().__init__()
        self.w = nn.Parameter(torch.as_tensor(w, dtype=torch.float64).clone())
        if frozen is not None:
            self.f = nn.Parameter(torch.as_tensor(frozen, dtype=torch.float64).clone(), requires_grad=False)


def _net(seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(6, 5), nn.LayerNorm(5), nn.Tanh(), nn.Conv1d(1, 4, 3), nn.Flatten(), nn.Linear(12, 3))
    with torch.no_grad():  # give the filters very different norms so per-layer scaling is wrong
        m[0].weight.mul_(torch.tensor([0.1, 1.0, 3.0, 0.5, 7.0])[:, None])
        m[3].weight.mul_(torch.tensor([0.2, 1.0, 4.0, 9.0])[:, None, None])
        m[5].weight.mul_(torch.tensor([5.0, 0.3, 1.0])[:, None])
    return m


class Net(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.body = _net(seed)

    def forward(self, x):
        h = self.body[2](self.body[1](self.body[0](x)))
        return self.body[5](self.body[4](self.body[3](h[:, None, :])))


# ---------------------------------------------------------------- stage 1: filter-normalized directions


def test_stage1_keys_shapes_and_filter_norms(nb):
    m = Net()
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    d = nb.filter_normalized_direction(m, generator=torch.Generator().manual_seed(0))
    names = [n for n, _ in m.named_parameters()]
    assert isinstance(d, dict) and sorted(d) == sorted(names), (
        f"return a dict keyed by every name in model.named_parameters(); expected {names}, got {sorted(d)}")
    for n, p in m.named_parameters():
        assert tuple(d[n].shape) == tuple(p.shape), f"direction for {n} has shape {tuple(d[n].shape)}, the parameter {tuple(p.shape)}"
        if p.dim() <= 1:
            assert torch.all(d[n] == 0), (
                f"{n} is 1-D (a bias or a norm's scale): Li et al. set its direction to zero, got nonzero entries")
            continue
        dn = d[n].detach().reshape(p.shape[0], -1).norm(dim=1)
        pn = p.detach().reshape(p.shape[0], -1).norm(dim=1)
        assert torch.allclose(dn, pn, rtol=1e-4), (
            f"{n}: each filter (slice along dim 0: a row of a Linear weight, an output channel of a conv) must have "
            f"the same norm as that filter of the weights.\n  weight filter norms {pn.tolist()}\n  direction filter "
            f"norms {dn.tolist()}\nif they're all scaled by one number you normalized the whole layer; if the shapes "
            "work but the numbers don't, you may have normalized along dim 1")
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), f"filter_normalized_direction changed the model's {k}; it must only read it"


def test_stage1_random_not_the_weights_and_generator(nb):
    torch.manual_seed(0)
    m = nn.Linear(400, 50)
    d1 = nb.filter_normalized_direction(m, generator=torch.Generator().manual_seed(1))
    d1b = nb.filter_normalized_direction(m, generator=torch.Generator().manual_seed(1))
    d2 = nb.filter_normalized_direction(m, generator=torch.Generator().manual_seed(2))
    assert torch.equal(d1["weight"], d1b["weight"]), "the same generator seed must give the same direction (draw with generator=)"
    assert not torch.allclose(d1["weight"], d2["weight"]), "different generator seeds must give different directions"
    cos = torch.nn.functional.cosine_similarity(d1["weight"], m.weight.detach(), dim=1)
    assert cos.abs().mean() < 0.15, (
        f"the direction should be a random Gaussian direction, unrelated to the weights (|cos| ~ 0.04 per 400-d row); "
        f"mean |cos| with the weights is {cos.abs().mean():.2f}. did you start from the weights instead of torch.randn?")
    w = d1["weight"].detach()
    assert abs((w > 0).float().mean().item() - 0.5) < 0.03, "a Gaussian direction has entries of both signs equally often"


def test_stage1_dtype_follows_parameters(nb):
    m = nn.Linear(5, 3).double()
    d = nb.filter_normalized_direction(m, generator=torch.Generator().manual_seed(0))
    assert d["weight"].dtype == torch.float64 and d["bias"].dtype == torch.float64, (
        f"each direction has its parameter's dtype (float64 here), got weight {d['weight'].dtype} and bias "
        f"{d['bias'].dtype}: draw with torch.randn(p.shape, generator=generator, dtype=p.dtype)")


# ---------------------------------------------------------------- stage 2: 2-D surfaces


def _quad_loss(m):
    w = m.w.detach()
    return float((w[0] - 1.0) ** 2 + 3.0 * w[1] ** 2 + 0.5 * w[0] * w[1] + w[2])


def test_stage2_surface_matches_closed_form(nb):
    m = Vec([0.2, -0.4, 0.7])
    d1 = {"w": torch.tensor([1.0, 0.0, 0.5], dtype=torch.float64)}
    d2 = {"w": torch.tensor([0.0, 1.0, -0.25], dtype=torch.float64)}
    calls = []

    def ev(model):
        calls.append(1)
        return _quad_loss(model)

    al, be, Z = bounded(nb.loss_surface_2d, m, ev, d1, d2, span=2.0, resolution=7, what="loss_surface_2d")
    assert np.allclose(al, np.linspace(-2, 2, 7)) and np.allclose(be, np.linspace(-2, 2, 7)), (
        f"alphas and betas are np.linspace(-span, span, resolution); got {np.round(al, 3)} and {np.round(be, 3)}")
    Z = np.asarray(Z)
    assert Z.shape == (7, 7), f"Z has shape (resolution, resolution) = (7, 7), got {Z.shape}"
    expect = np.zeros((7, 7))
    for i, a in enumerate(al):
        for j, b in enumerate(be):
            expect[i, j] = _quad_loss(Vec(np.array([0.2, -0.4, 0.7]) + a * d1["w"].numpy() + b * d2["w"].numpy()))
    if np.allclose(Z, expect.T, atol=1e-6) and not np.allclose(Z, expect, atol=1e-6):
        raise AssertionError("Z is transposed: Z[i, j] is the loss at alphas[i] along d1 and betas[j] along d2")
    assert np.allclose(Z, expect, atol=1e-6), (
        f"Z[i, j] must be eval_loss at theta + alphas[i] d1 + betas[j] d2, from the ORIGINAL theta each time.\n"
        f"expected row 0 {np.round(expect[0], 3)}\ngot      row 0 {np.round(Z[0], 3)}\n"
        "if the values drift, you're adding offsets to already-moved weights")
    assert len(calls) == 49, f"eval_loss should be called once per grid point (49), it was called {len(calls)} times"
    assert torch.allclose(m.w, torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)), "the weights must be restored afterwards"


def test_stage2_missing_keys_and_restore_on_error(nb):
    m = Vec([1.0, 2.0, 3.0], frozen=[5.0])
    d1 = {"w": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)}  # no entry for "f": it doesn't move
    d2 = {"w": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), "f": torch.tensor([1.0], dtype=torch.float64)}
    seen = []

    def ev(model):
        seen.append((model.w.detach().clone(), model.f.detach().clone()))
        return 0.0

    _run(nb.loss_surface_2d, m, ev, d1, d2, span=1.0, resolution=3)
    fs = sorted({round(float(f[0]), 6) for _, f in seen})
    assert fs == [4.0, 5.0, 6.0], (
        f"a parameter moves by beta along d2['f'] even though d1 has no 'f' (missing keys mean zero); the frozen "
        f"parameter took the values {fs}, expected [4, 5, 6]")
    calls = []

    def boom(model):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("eval failed")
        return 0.0

    with pytest.raises(RuntimeError):
        _run(nb.loss_surface_2d, m, boom, d1, d2, span=1.0, resolution=3)
    assert torch.allclose(m.w, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)) and float(m.f) == 5.0, (
        "if eval_loss raises, the weights must still be restored (use try/finally)")


# ---------------------------------------------------------------- stage 3: interpolation and barriers


class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)


def test_stage3_interpolation_path(nb):
    m = BNNet()
    sa = {k: v.clone() for k, v in m.state_dict().items()}
    sb = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        sa["lin.weight"].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        sb["lin.weight"].copy_(torch.tensor([[-1.0, 2.0], [0.0, 3.0]]))
        sa["bn.running_mean"].fill_(0.0)
        sb["bn.running_mean"].fill_(4.0)
        sa["bn.num_batches_tracked"].fill_(5)
        sb["bn.num_batches_tracked"].fill_(9)
    sa0 = {k: v.clone() for k, v in sa.items()}
    sb0 = {k: v.clone() for k, v in sb.items()}
    with torch.no_grad():
        m.lin.weight.fill_(7.0)
    orig = {k: v.clone() for k, v in m.state_dict().items()}
    tracked = []

    def ev(model):
        tracked.append((int(model.bn.num_batches_tracked), float(model.bn.running_mean[0])))
        return float(model.lin.weight[0, 0].detach())

    al, losses = bounded(nb.linear_interpolation, m, sa, sb, ev, n_points=5, what="linear_interpolation")
    assert np.allclose(al, np.linspace(0, 1, 5)), f"alphas are np.linspace(0, 1, n_points), got {al}"
    expect = 1 - 2 * np.linspace(0, 1, 5)
    assert np.allclose(losses, expect, atol=1e-6), (
        f"weight[0,0] goes from 1 (state_a) to -1 (state_b), so the 'losses' (weight[0,0] itself) are 1 - 2a = "
        f"{expect}; got {np.round(losses, 4)}. if they run backwards you swapped the ends: "
        "load (1 - a) * state_a + a * state_b (a = 0 is state_a)")
    assert [t[0] for t in tracked] == [5] * 5, (
        f"integer entries (num_batches_tracked) can't be interpolated: take them from state_a; saw {[t[0] for t in tracked]}")
    assert np.allclose([t[1] for t in tracked], 4 * np.linspace(0, 1, 5)), (
        "float buffers (BatchNorm running statistics) are interpolated like parameters")
    for k in orig:
        assert torch.equal(m.state_dict()[k], orig[k]), f"the model's state must be restored afterwards ({k} changed)"
        assert torch.equal(sa[k], sa0[k]) and torch.equal(sb[k], sb0[k]), f"state_a / state_b must not be modified ({k})"


def test_stage3_live_state_dict(nb):
    torch.manual_seed(0)
    m = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    sb = {"weight": torch.tensor([[3.0, 2.0, 1.0]])}
    seen = []
    _, losses = _run(nb.linear_interpolation, m, m.state_dict(), sb, lambda model: seen.append(model.weight[0, 0].item()) or 0.0,
                                        n_points=5)
    assert np.allclose(seen, [1.0, 1.5, 2.0, 2.5, 3.0]), (
        f"passing model.state_dict() as state_a is the common case, and its tensors ARE the model's weights: they "
        f"change every time you load a mixture. weight[0,0] along the path should be 1, 1.5, 2, 2.5, 3; got {seen}. "
        "copy both states (detach().clone()) before the loop")
    assert torch.allclose(m.weight, torch.tensor([[1.0, 2.0, 3.0]])), "the weights must be restored afterwards"


def test_stage3_loss_barrier(nb):
    cases = [([1.0, 3.0, 1.0], 2.0), ([1.0, 4.0, 3.0, 3.0, 3.0], 2.5), ([1.0, 0.5, 2.0], 0.0), ([2.0, 2.0], 0.0),
             ([0.0, 1.0, 4.0], 0.0)]
    for losses, want in cases:
        got = nb.loss_barrier(np.array(losses))
        assert math.isclose(got, want, abs_tol=1e-9), (
            f"loss_barrier({losses}) should be {want}, got {got}: the largest amount by which the path rises above "
            "the straight line between its two end losses (not above the average or the max of the ends), and 0 "
            "if it never does")


# ---------------------------------------------------------------- stage 4: sharpness


def _small_net():
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1)).double()
    x = torch.randn(10, 3, dtype=torch.float64)
    y = torch.randn(10, 1, dtype=torch.float64)
    return m, (lambda: ((m(x) - y) ** 2).mean()), x, y


def _exact_hessian(m, x, y):
    shapes = [p.shape for p in m.parameters()]
    sizes = [p.numel() for p in m.parameters()]
    flat0 = torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    def f(flat):
        ps = torch.split(flat, sizes)
        h = torch.tanh(x @ ps[0].reshape(shapes[0]).T + ps[1])
        out = h @ ps[2].reshape(shapes[2]).T + ps[3]
        return ((out - y) ** 2).mean()

    return torch.autograd.functional.hessian(f, flat0)


def test_stage4_hvp_matches_exact_hessian(nb):
    m, loss_fn, x, y = _small_net()
    H = _exact_hessian(m, x, y)
    params = list(m.parameters())
    g = torch.Generator().manual_seed(0)
    v = [torch.randn(p.shape, generator=g, dtype=torch.float64) for p in params]
    try:
        hv = nb.hessian_vector_product(loss_fn, params, v)
    except RuntimeError as e:
        raise AssertionError(
            f"hessian_vector_product raised RuntimeError: {str(e).splitlines()[0]}. compute the gradient with "
            "torch.autograd.grad(loss, params, create_graph=True), not loss.backward(), then differentiate its dot "
            "product with v a second time") from None
    assert isinstance(hv, (list, tuple)) and len(hv) == len(params), "return one tensor per parameter, in order"
    got = torch.cat([h.reshape(-1) for h in hv])
    want = H @ torch.cat([t.reshape(-1) for t in v])
    assert torch.allclose(got, want, atol=1e-8), (
        f"H v differs from the exact Hessian times v (max error {(got - want).abs().max():.2e}). differentiate "
        "the gradient (create_graph=True) dotted with v a second time")


def _eig(nb, *args, **kwargs):
    try:
        return _run(nb.top_hessian_eigenvalue, *args, **kwargs)
    except RuntimeError as e:
        raise AssertionError(
            f"top_hessian_eigenvalue raised RuntimeError: {str(e).splitlines()[0]}. likely causes: differentiating with "
            "respect to a parameter with requires_grad=False (use only the trainable ones), calling loss.backward() "
            "instead of torch.autograd.grad, or reusing a graph that was already freed (call loss_fn() for every "
            "Hessian-vector product)") from None


def test_stage4_top_eigenvalue_signed_and_exact(nb):
    A = torch.diag(torch.tensor([1.0, 2.0, -5.0], dtype=torch.float64))
    m = Vec([0.3, -0.2, 0.1])

    def quad():
        return 0.5 * m.w @ A @ m.w

    lam = bounded(nb.top_hessian_eigenvalue, m, quad, n_iters=60, generator=torch.Generator().manual_seed(0),
                  what="top_hessian_eigenvalue")
    assert math.isclose(lam, -5.0, rel_tol=1e-4), (
        f"the Hessian is diag(1, 2, -5): power iteration converges to the eigenvalue of largest MAGNITUDE, -5, and "
        f"the Rayleigh quotient v^T H v keeps its sign; got {lam}. (the norm of H v would give +5)")
    m2, loss_fn, x, y = _small_net()
    ev = torch.linalg.eigvalsh(_exact_hessian(m2, x, y))
    top = float(ev[ev.abs().argmax()])
    lam2 = _eig(nb, m2, loss_fn, n_iters=200, generator=torch.Generator().manual_seed(0))
    assert math.isclose(lam2, top, rel_tol=1e-3), f"top eigenvalue of a small MLP's Hessian is {top:.5f}, got {lam2:.5f}"


def test_stage4_iterations_generator_and_no_side_effects(nb):
    A = torch.diag(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=torch.float64))
    m = Vec(0.1 * np.arange(8), frozen=np.ones(3))
    m.f.requires_grad_(False)

    def quad():
        return 0.5 * m.w @ A @ m.w + 100.0 * (m.f ** 2).sum() * m.w.sum() ** 2

    sentinel = torch.full((8,), 0.123, dtype=torch.float64)
    m.w.grad = sentinel.clone()
    one = [_eig(nb, m, quad, n_iters=1, generator=torch.Generator().manual_seed(s)) for s in range(5)]
    many = _eig(nb, m, quad, n_iters=200, generator=torch.Generator().manual_seed(0))
    assert len({round(o, 9) for o in one}) > 1, (
        "five different generator seeds gave the same one-step estimate: the start vector must be RANDOM, drawn "
        "with torch.randn(..., generator=generator)")
    assert torch.equal(m.w.grad, sentinel), (
        "top_hessian_eigenvalue must not touch .grad (use torch.autograd.grad, not loss.backward())")
    assert torch.allclose(m.w, 0.1 * torch.arange(8, dtype=torch.float64)), "the parameters must not change"
    eigs = torch.linalg.eigvalsh(A + 600.0 * torch.ones(8, 8, dtype=torch.float64))
    assert math.isclose(many, float(eigs.max()), rel_tol=1e-4), (
        f"the trainable-parameter Hessian's top eigenvalue is {float(eigs.max()):.3f}; got {many}. only parameters with "
        f"requires_grad=True count: the frozen f is a constant")
    assert max(abs(o - many) for o in one) > 1.0, (
        "with n_iters=1 the estimate should be far from converged; did you ignore n_iters?")
    a = _eig(nb, m, quad, n_iters=1, generator=torch.Generator().manual_seed(7))
    b = _eig(nb, m, quad, n_iters=1, generator=torch.Generator().manual_seed(7))
    assert a == b, "the same generator seed must give the same estimate (draw the start with generator=)"
    for k in (1, 3, 7):
        calls = []

        def counted():
            calls.append(1)
            return quad()

        _eig(nb, m, counted, n_iters=k, generator=torch.Generator().manual_seed(1))
        assert len(calls) == k, (
            f"n_iters={k} means {k} Hessian-vector products, each evaluating loss_fn() once; loss_fn was called "
            f"{len(calls)} times")


def test_stage4_unused_parameter(nb):
    class TwoParams(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
            self.unused = nn.Parameter(torch.zeros(3, dtype=torch.float64))

    m = TwoParams()

    def quad():
        return 3.0 * (m.w ** 2).sum()

    lam = _eig(nb, m, quad, n_iters=20, generator=torch.Generator().manual_seed(0))
    assert math.isclose(lam, 6.0, rel_tol=1e-6), (
        f"a trainable parameter the loss doesn't use has zero curvature (its Hessian rows are zero), so the top "
        f"eigenvalue of 3 |w|^2 is 6; got {lam}. pass allow_unused=True to torch.autograd.grad and treat None as zeros")
