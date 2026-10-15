"""Checks for 19 · 03 infinite width: NTK & mean field.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. the empirical NTK is
checked against kernels known exactly (a linear model, an explicit Jacobian from `torch.func`), with
frozen parameters and both output shapes; kernel regression against a direct solve in both array
types; the analytic ReLU kernel against hand values and a 20,000-unit network; the closed-form
linearized dynamics against gradient descent on a linear model with a singular kernel; the trainer
against a hand-computed step.
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



def _mlp(out_dim=1, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, out_dim)).double()


def _jac(model, x):
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v for k, v in model.named_parameters() if not v.requires_grad}

    def f(p):
        return torch.func.functional_call(model, {**p, **buffers}, (x,)).reshape(-1)

    J = torch.func.jacrev(f)(params)
    return torch.cat([J[k].reshape(x.shape[0], -1) for k in params], dim=1)


# ---------------------------------------------------------------- stage 1: the empirical NTK


def test_stage1_linear_model_kernel(nb):
    torch.manual_seed(0)
    lin = nn.Linear(4, 1).double()
    x1 = torch.randn(5, 4, dtype=torch.float64)
    x2 = torch.randn(3, 4, dtype=torch.float64)
    K = bounded(nb.empirical_ntk, lin, x1, x2, what="empirical_ntk")
    assert torch.is_tensor(K) and tuple(K.shape) == (5, 3), f"K is an (n1, n2) = (5, 3) tensor, got {getattr(K, 'shape', K)}"
    want = x1 @ x2.T + 1.0
    assert torch.allclose(K, want, atol=1e-10), (
        "for f(x) = w.x + b the gradient is (x, 1), so K(x, x') = x.x' + 1.\n"
        f"expected row 0 {want[0].tolist()}\ngot {K[0].tolist()}\n"
        "if every row is the same, you differentiated the SUM of the outputs instead of each output separately")
    for p in lin.parameters():
        assert p.grad is None, "empirical_ntk must not leave .grad on the parameters (use torch.autograd.grad)"


def test_stage1_matches_explicit_jacobian(nb):
    torch.manual_seed(1)
    x1 = torch.randn(6, 3, dtype=torch.float64)
    x2 = torch.randn(4, 3, dtype=torch.float64)
    for out_dim, squeeze in ((1, False), (1, True)):
        m = _mlp(out_dim)
        model = (lambda m: (lambda x: m(x).squeeze(-1)))(m) if squeeze else m
        J1, J2 = _jac(m, x1), _jac(m, x2)

        class Wrap(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = m

            def forward(self, x):
                return model(x)

        K = _run(nb.empirical_ntk, Wrap(), x1, x2)
        assert torch.allclose(K, J1 @ J2.T, atol=1e-10), (
            f"K must equal J(x1) J(x2)^T for outputs of shape {'(n,)' if squeeze else '(n, 1)'}; max error "
            f"{(K - J1 @ J2.T).abs().max():.2e}")
        Ks = _run(nb.empirical_ntk, Wrap(), x1)
        assert tuple(Ks.shape) == (6, 6) and torch.allclose(Ks, J1 @ J1.T, atol=1e-10), "x2=None means x2 = x1"


def test_stage1_frozen_parameters_excluded(nb):
    m = _mlp(1, seed=2)
    m[0].weight.requires_grad_(False)
    m[0].bias.requires_grad_(False)
    torch.manual_seed(3)
    x = torch.randn(5, 3, dtype=torch.float64)
    J = _jac(m, x)  # trainable only
    K = _run(nb.empirical_ntk, m, x)
    assert torch.allclose(K, J @ J.T, atol=1e-10), (
        "only parameters with requires_grad=True are part of the tangent kernel (freezing the first layer leaves "
        "the second layer's features as the kernel)")


# ---------------------------------------------------------------- stage 2: kernel regression


def test_stage2_interpolates_and_matches_solve(nb):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 9))
    K = A @ A.T
    y = rng.normal(size=6)
    B = rng.normal(size=(4, 9))
    Kx = B @ A.T
    got = nb.kernel_regression_predict(K, y, K)
    assert isinstance(got, np.ndarray) and np.allclose(got, y, atol=1e-8), (
        "with ridge=0, predicting at the training points reproduces y exactly (kernel regression interpolates)")
    got = nb.kernel_regression_predict(K, y, Kx, ridge=0.7)
    want = Kx @ np.linalg.solve(K + 0.7 * np.eye(6), y)
    assert got.shape == (4,) and np.allclose(got, want, atol=1e-10), (
        f"expected K_test_train @ solve(K_train + ridge I, y) = {np.round(want, 4)}, got {np.round(got, 4)}. "
        "the ridge is added to the diagonal once (not times n)")
    Y = rng.normal(size=(6, 2))
    got2 = nb.kernel_regression_predict(K, Y, Kx, ridge=0.1)
    assert got2.shape == (4, 2) and np.allclose(got2, Kx @ np.linalg.solve(K + 0.1 * np.eye(6), Y)), (
        "y_train may be (n, k): solve for every column at once")


def test_stage2_tensors_in_tensors_out(nb):
    torch.manual_seed(0)
    A = torch.randn(5, 7, dtype=torch.float64)
    K = A @ A.T
    y = torch.randn(5, dtype=torch.float64)
    Kx = torch.randn(3, 7, dtype=torch.float64) @ A.T
    got = nb.kernel_regression_predict(K, y, Kx, ridge=0.3)
    assert torch.is_tensor(got), f"tensor inputs give a tensor, got {type(got).__name__}"
    assert torch.allclose(got, Kx @ torch.linalg.solve(K + 0.3 * torch.eye(5, dtype=torch.float64), y)), (
        "tensor result differs from K_test_train @ solve(K + ridge I, y)")


# ---------------------------------------------------------------- stage 3: the two-layer network and its kernel


def test_stage3_ntknet_definition(nb):
    torch.manual_seed(0)
    net = nb.NTKNet(3, 4000)
    assert tuple(net.W.shape) == (4000, 3) and tuple(net.a.shape) == (4000,), (
        f"W is (width, in_dim) and a is (width,); got {tuple(net.W.shape)} and {tuple(net.a.shape)}")
    assert {n for n, _ in net.named_parameters()} == {"W", "a"}, "the only parameters are W and a (no biases)"
    for name, p in (("W", net.W), ("a", net.a)):
        assert abs(p.detach().std().item() - 1.0) < 0.05 and abs(p.detach().mean().item()) < 0.05, (
            f"{name} starts standard normal (the scale lives in the forward pass, not the initialization); "
            f"std {p.detach().std():.3f}")
    x = torch.randn(7, 3)
    want = torch.relu(x @ net.W.T) @ net.a / math.sqrt(4000)
    out = net(x)
    assert tuple(out.shape) == (7,), f"forward returns (n,), got {tuple(out.shape)}"
    assert torch.allclose(out, want, atol=1e-4), "NTK parameterization: f(x) = a . relu(W x) / sqrt(width)"
    mf = nb.NTKNet(3, 4000, parameterization="mf")
    with torch.no_grad():
        mf.W.copy_(net.W)
        mf.a.copy_(net.a)
    assert torch.allclose(mf(x), want / math.sqrt(4000), atol=1e-6), "mean-field: f(x) = a . relu(W x) / width"
    with pytest.raises(ValueError):
        nb.NTKNet(3, 10, parameterization="standard")


def test_stage3_relu_ntk_hand_values(nb):
    x = torch.tensor([[3.0, 4.0], [0.0, 2.0], [-3.0, -4.0], [3.0, 4.0]], dtype=torch.float64)
    K = nb.relu_ntk(x)
    assert tuple(K.shape) == (4, 4) and not torch.isnan(K).any(), (
        "K is (4, 4) with no NaN (identical inputs can give a cosine of 1.0000000002: clamp it to [-1, 1])")
    assert math.isclose(float(K[0, 0]), 25.0, rel_tol=1e-9), (
        f"K(x, x) = |x|^2 (theta = 0: |x|^2/2 from the output layer plus |x|^2/2 from the hidden layer); "
        f"got {float(K[0, 0])} for |x| = 5")
    assert math.isclose(float(K[0, 2]), 0.0, abs_tol=1e-9), f"opposite inputs (theta = pi) give 0, got {float(K[0, 2])}"
    t = math.acos(4.0 / 5.0)  # angle between (3, 4) and (0, 2)
    want = 10 / (2 * math.pi) * (math.sin(t) + (math.pi - t) * 0.8) + 8.0 * (math.pi - t) / (2 * math.pi)
    assert math.isclose(float(K[0, 1]), want, rel_tol=1e-9), (
        f"K((3,4), (0,2)) should be {want:.6f}, got {float(K[0, 1]):.6f}")
    assert math.isclose(float(K[0, 3]), 25.0, rel_tol=1e-9) and torch.allclose(K, K.T), "K is symmetric"
    K2 = nb.relu_ntk(x[:2], 2.0 * x[2:])
    assert torch.allclose(K2, 2.0 * nb.relu_ntk(x[:2], x[2:])), "the kernel is homogeneous: K(x, c x') = c K(x, x')"


def test_stage3_wide_network_matches_formula(nb):
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    x = x / x.norm(dim=1, keepdim=True)
    net = nb.NTKNet(3, 20000)
    Ke = _run(nb.empirical_ntk, net, x).double()
    Ka = nb.relu_ntk(x.double())
    err = ((Ke - Ka).norm() / Ka.norm()).item()
    assert err < 0.05, (
        f"at width 20,000 the empirical NTK should be within a few percent of relu_ntk (relative Frobenius error "
        f"{err:.3f}). check both terms: the output layer's relu(w.x) relu(w.x') and the hidden layer's x.x' 1[w.x>0] 1[w.x'>0]")
    mf = nb.NTKNet(3, 200, parameterization="mf")
    nt = nb.NTKNet(3, 200)
    with torch.no_grad():
        mf.W.copy_(nt.W)
        mf.a.copy_(nt.a)
    assert torch.allclose(nb.empirical_ntk(mf, x) * 200, _run(nb.empirical_ntk, nt, x), rtol=1e-4, atol=1e-6), (
        "with the same weights, the mean-field network's NTK is the NTK-parameterized one divided by width")


# ---------------------------------------------------------------- stage 4: linearized dynamics in closed form


def test_stage4_matches_gd_on_a_linear_model(nb):
    rng = np.random.default_rng(0)
    Phi = rng.normal(size=(8, 5))  # 8 points, 5 features: K = Phi Phi^T has rank 5 (three zero eigenvalues)
    Phx = rng.normal(size=(3, 5))
    y = rng.normal(size=8)
    w0 = rng.normal(size=5)
    lr, steps = 0.02, 37
    w = w0.copy()
    for _ in range(steps):
        w -= lr * Phi.T @ (Phi @ w - y)
    K, Kx = Phi @ Phi.T, Phx @ Phi.T
    f_tr, f_te = bounded(nb.ntk_gd_predictions, K, Kx, y, Phi @ w0, Phx @ w0, lr=lr, steps=steps,
                         what="ntk_gd_predictions")
    f_tr, f_te = np.asarray(f_tr), np.asarray(f_te)
    assert np.allclose(f_tr, Phi @ w, atol=1e-8), (
        f"train predictions after {steps} GD steps on the linear model are {np.round(Phi @ w, 4)}; got {np.round(f_tr, 4)}. "
        "f_t = y - (I - lr K)^t (y - f_0)")
    assert np.allclose(f_te, Phx @ w, atol=1e-8), (
        f"test predictions should be {np.round(Phx @ w, 4)}, got {np.round(f_te, 4)}. this K is singular "
        "(8 points, 5 features): K^-1 doesn't exist but K^-1 (I - (I - lr K)^t) does; compute it in K's eigenbasis")
    f_tr0, f_te0 = _run(nb.ntk_gd_predictions, K, Kx, y, Phi @ w0, Phx @ w0, lr=lr, steps=0)
    assert np.allclose(np.asarray(f_tr0), Phi @ w0) and np.allclose(np.asarray(f_te0), Phx @ w0), "steps=0 returns f_0"
    w = w0.copy()
    for _ in range(5):
        w -= 0.05 * Phi.T @ (Phi @ w - y)
    f_tr5, _ = _run(nb.ntk_gd_predictions, torch.tensor(K), torch.tensor(Kx), torch.tensor(y), torch.tensor(Phi @ w0),
                                     torch.tensor(Phx @ w0), lr=0.05, steps=5)
    assert np.allclose(np.asarray(f_tr5), Phi @ w, atol=1e-8), "the learning rate must enter as lr (tensor inputs work too)"


def test_stage4_exactly_singular_kernel(nb):
    Phi = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0]])  # a zero feature row: K has an exactly zero eigenvalue
    Phx = np.array([[0.5, 1.0], [-1.0, 2.0]])
    y = np.array([1.0, -2.0, 0.5])
    w0 = np.array([0.3, -0.1])
    w = w0.copy()
    for _ in range(25):
        w -= 0.03 * Phi.T @ (Phi @ w - y)
    f_tr, f_te = _run(nb.ntk_gd_predictions, Phi @ Phi.T, Phx @ Phi.T, y, Phi @ w0, Phx @ w0, lr=0.03, steps=25)
    f_tr, f_te = np.asarray(f_tr), np.asarray(f_te)
    assert np.all(np.isfinite(f_tr)) and np.all(np.isfinite(f_te)), (
        "with an exactly zero eigenvalue, (1 - (1 - lr lam)^t) / lam is 0 / 0 = nan: use its limit lr * t there")
    assert np.allclose(f_tr, Phi @ w, atol=1e-8) and np.allclose(f_te, Phx @ w, atol=1e-8), (
        "a point whose features are all zero never moves (its prediction stays f_0), and the others follow GD")


# ---------------------------------------------------------------- stage 5: training and weight movement


def test_stage5_gd_step_and_losses(nb):
    torch.manual_seed(0)
    net = nb.NTKNet(2, 50).double()
    x = torch.randn(6, 2, dtype=torch.float64)
    y = torch.randn(6, dtype=torch.float64)
    W0, a0 = net.W.detach().clone(), net.a.detach().clone()
    h = torch.relu(x @ W0.T)
    r = h @ a0 / math.sqrt(50) - y
    gA = h.T @ r / math.sqrt(50)
    gW = ((r[:, None] * (x @ W0.T > 0) * a0[None, :]) / math.sqrt(50)).T @ x
    losses = bounded(nb.gd_train, net, x, y, lr=0.1, steps=1, what="gd_train")
    assert len(losses) == 2, f"steps=1 returns 2 losses (before the step, after it), got {len(losses)}"
    assert math.isclose(losses[0], 0.5 * float((r ** 2).sum()), rel_tol=1e-9), (
        f"the loss is 1/2 * SUM of squared errors ({0.5 * float((r ** 2).sum()):.6f}), got {losses[0]:.6f}")
    assert torch.allclose(net.a, a0 - 0.1 * gA, atol=1e-10) and torch.allclose(net.W, W0 - 0.1 * gW, atol=1e-10), (
        "one step is theta <- theta - lr * grad of 1/2 sum (f - y)^2, in place, with no momentum")
    state = {k: v.clone() for k, v in net.state_dict().items()}
    l2 = _run(nb.gd_train, net, x, y[:, None], lr=0.1, steps=20)
    net.load_state_dict(state)
    l2_flat = _run(nb.gd_train, net, x, y, lr=0.1, steps=20)
    assert len(l2) == 21 and l2[-1] < l2[0], "20 steps give 21 losses and the loss goes down"
    assert np.allclose(l2, l2_flat, rtol=1e-10), (
        "y of shape (n, 1) must give exactly the same run as y of shape (n,): flatten both the outputs and y to (n,). "
        "subtracting an (n, 1) target from (n,) outputs broadcasts to (n, n) and sums over every pair")
    l3 = _run(nb.gd_train, net, x, y, lr=0.0, steps=3)
    assert np.allclose(l3, l3[0]), "lr=0 leaves the loss unchanged"
    lin = nn.Linear(2, 1, bias=False).double()
    w = lin.weight.detach().clone().reshape(-1)
    for _ in range(4):
        w = w - 0.05 * x.T @ (x @ w - y)
    _run(nb.gd_train, lin, x, y, lr=0.05, steps=4)
    assert torch.allclose(lin.weight.reshape(-1), w, atol=1e-10), (
        "four plain GD steps on a linear model don't match w <- w - lr X^T (X w - y): no momentum, no Adam, "
        "no averaging over examples")


class WithBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([3.0, 0.0]))
        self.q = nn.Parameter(torch.tensor([0.0, 4.0]))
        self.register_buffer("count", torch.tensor([100.0]))


def test_stage5_relative_change(nb):
    m = WithBuffer()
    init = {k: v.detach().clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        m.p.add_(torch.tensor([0.0, 1.0]))
        m.q.add_(torch.tensor([1.0, 0.0]))
        m.count.add_(50.0)
    got = nb.relative_change(m, init)
    assert math.isclose(got, math.sqrt(2) / 5, rel_tol=1e-9), (
        f"||theta - theta_0|| / ||theta_0|| over all parameters together: sqrt(1 + 1) / sqrt(9 + 16) = "
        f"{math.sqrt(2) / 5:.6f}; got {got:.6f}. one global norm, not an average of per-layer ratios, and buffers "
        "(like the 'count' here) don't count")
