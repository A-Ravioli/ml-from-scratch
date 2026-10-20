"""Checks for exercise.py (neural ODEs).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. solvers are checked
on hand-computed steps, on analytic solutions (exponentials, the harmonic oscillator, a
time-dependent field), for their order of accuracy (error ratios when the step halves), and for the
exact number of vector-field evaluations the step grid implies. gradients are checked against
analytic derivatives, backprop through the solver, and finite differences; the adjoint's memory is
measured with saved-tensor hooks (constant in the number of steps). solver loops run under
`mlfs.testing.bounded`, so a time loop that never reaches the end fails instead of hanging.
"""
import math

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402

D = torch.float64


def _solve(nb, *args, adjoint=False, **kwargs):
    fn = nb.odeint_adjoint if adjoint else nb.odeint
    return bounded(fn, *args, budget=3_000_000, what=fn.__name__,
                   hint="step through each interval a fixed, integer number of times; don't loop on `while t < t1`.",
                   **kwargs)


class Counter:
    """wraps f(t, y) and counts calls."""

    def __init__(self, f):
        self.f, self.calls = f, 0

    def __call__(self, t, y):
        self.calls += 1
        return self.f(t, y)


class MLPField(nn.Module):
    def __init__(self, dim=2, hidden=16, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(dim + 1, hidden), nn.Tanh(), nn.Linear(hidden, dim)).double()

    def forward(self, t, y):
        tt = t.reshape(1, 1).expand(y.shape[0], 1).to(y.dtype)
        return self.net(torch.cat([y, tt], dim=1))


# ---------------------------------------------------------------- stage 1: single steps


def test_stage1_hand_example(nb):
    f = lambda t, y: y  # noqa: E731
    y, t, h = torch.tensor([1.0], dtype=D), torch.tensor(0.0, dtype=D), 0.1
    want = {"euler_step": 1.1, "midpoint_step": 1.105, "rk4_step": 1.1051708333333333}
    for name, w in want.items():
        got = float(getattr(nb, name)(f, t, y, h)[0])
        assert abs(got - w) < 1e-12, (
            f"{name} on dy/dt = y from y = 1 with h = 0.1 should give {w!r}, got {got!r}. "
            "(exact: e^0.1 = 1.105170918...)")


def test_stage1_time_dependence(nb):
    f = lambda t, y: t * torch.ones_like(y)  # noqa: E731  # y(h) = h^2 / 2 from y(0) = 0
    y, t, h = torch.zeros(3, dtype=D), torch.tensor(0.0, dtype=D), 0.4
    assert torch.allclose(nb.euler_step(f, t, y, h), torch.zeros(3, dtype=D)), "Euler uses f at the start only: 0"
    for name in ("midpoint_step", "rk4_step"):
        got = getattr(nb, name)(f, t, y, h)
        assert torch.allclose(got, torch.full((3,), 0.08, dtype=D), atol=1e-14), (
            f"{name} integrates dy/dt = t exactly (y(0.4) = 0.08); got {got.tolist()}. "
            "the intermediate stages evaluate f at t + h/2 (and t + h), not at t")


def test_stage1_order_of_accuracy(nb):
    f = lambda t, y: y * torch.cos(t)  # noqa: E731  # exact y = exp(sin t)
    t0 = torch.tensor(0.3, dtype=D)
    y0 = torch.exp(torch.sin(t0)).reshape(1)
    for name, order in (("euler_step", 1), ("midpoint_step", 2), ("rk4_step", 4)):
        errs = []
        for h in (0.1, 0.05):
            exact = math.exp(math.sin(0.3 + h))
            errs.append(abs(float(getattr(nb, name)(f, t0, y0, h)[0]) - exact))
        ratio = errs[0] / errs[1]
        want = 2 ** (order + 1)
        assert 0.7 * want < ratio < 1.4 * want, (
            f"{name} is order {order}: its one-step error should shrink by about 2^{order + 1} = {want} when h halves; "
            f"got a ratio of {ratio:.1f}. check the stage coefficients")


def test_stage1_any_shape(nb):
    A = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=D)
    f = lambda t, y: y @ A.T  # noqa: E731
    y = torch.randn(3, 2, dtype=D, generator=torch.Generator().manual_seed(0))
    out = nb.rk4_step(f, torch.tensor(0.0, dtype=D), y, 0.1)
    assert tuple(out.shape) == (3, 2) and torch.allclose(out, y @ torch.linalg.matrix_exp(0.1 * A).T, atol=1e-6)


# ---------------------------------------------------------------- stage 2: odeint


def test_stage2_exponential_and_shapes(nb):
    y0 = torch.tensor([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]], dtype=D)
    t = torch.linspace(0, 2, 5, dtype=D)
    ys = _solve(nb, lambda t, y: -y, y0, t, step_size=0.01)
    assert tuple(ys.shape) == (5, 2, 3), f"odeint returns (len(t), *y0.shape) = (5, 2, 3); got {tuple(ys.shape)}"
    assert torch.equal(ys[0], y0), "the first output is y0 itself"
    want = torch.exp(-t)[:, None, None] * y0
    assert torch.allclose(ys, want, atol=1e-9), (
        f"dy/dt = -y has y(t) = y0 e^-t; max error {float((ys - want).abs().max()):.1e} with RK4 and step 0.01")


def test_stage2_oscillator_backwards_and_methods(nb):
    f = lambda t, y: torch.stack([y[1], -y[0]])  # noqa: E731
    y0 = torch.tensor([1.0, 0.0], dtype=D)
    ys = _solve(nb, f, y0, torch.tensor([0.0, math.pi, 2 * math.pi], dtype=D), step_size=0.01)
    assert torch.allclose(ys[1], torch.tensor([-1.0, 0.0], dtype=D), atol=1e-8) and torch.allclose(ys[2], y0, atol=1e-8), (
        "the harmonic oscillator returns to its start after 2 pi")
    back = _solve(nb, lambda t, y: y, torch.tensor([math.e], dtype=D), torch.tensor([1.0, 0.0], dtype=D), step_size=0.01)
    assert abs(float(back[-1, 0]) - 1.0) < 1e-9, "t may decrease: integrating dy/dt = y from t = 1 back to 0 divides by e"
    for method, order in (("euler", 1), ("midpoint", 2), ("rk4", 4)):
        errs = [abs(float(_solve(nb, lambda t, y: y * torch.cos(t), torch.ones(1, dtype=D),
                                 torch.tensor([0.0, 2.0], dtype=D), method=method, step_size=h)[-1, 0])
                    - math.exp(math.sin(2.0))) for h in (0.1, 0.05)]
        ratio = errs[0] / errs[1]
        assert 0.7 * 2 ** order < ratio < 1.4 * 2 ** order, (
            f"method={method!r}: the global error should shrink by about 2^{order} when step_size halves; ratio {ratio:.2f}")
    with pytest.raises(ValueError):
        nb.odeint(f, y0, torch.tensor([0.0, 1.0], dtype=D), method="rk45")


def test_stage2_step_grid(nb):
    f = Counter(lambda t, y: -y)
    y0 = torch.ones(2, dtype=D)
    _solve(nb, f, y0, torch.tensor([0.0, 1.0], dtype=D), step_size=0.3)
    assert f.calls == 16, (
        f"t = [0, 1] with step_size 0.3 needs ceil(1 / 0.3) = 4 equal RK4 steps of 0.25 = 16 evaluations of f; got "
        f"{f.calls}. each interval of t is split into ceil(|dt| / step_size) equal substeps")
    f.calls = 0
    _solve(nb, f, y0, torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4], dtype=D))
    assert f.calls == 16, f"step_size=None takes one step per interval of t: 4 RK4 steps = 16 evaluations; got {f.calls}"
    f.calls = 0
    _solve(nb, f, y0, torch.tensor([0.0, 0.3], dtype=D), method="euler", step_size=0.1)
    assert f.calls == 3, (
        f"0.3 / 0.1 is 2.9999999999999996 in floating point: that's still 3 steps, not 4; got {f.calls} evaluations")
    f.calls = 0
    _solve(nb, f, y0, torch.tensor([0.0, 0.1 + 0.2], dtype=D), method="euler", step_size=0.1)
    assert f.calls == 3, (
        f"(0.1 + 0.2) / 0.1 is 3.0000000000000004 in floating point: rounding error mustn't add a fourth step (allow a "
        f"tiny tolerance before the ceil); got {f.calls} evaluations")


def test_stage2_differentiable(nb):
    lam = torch.tensor(-0.7, dtype=D, requires_grad=True)
    y0 = torch.tensor([1.5], dtype=D, requires_grad=True)
    yT = _solve(nb, lambda t, y: lam * y, y0, torch.tensor([0.0, 2.0], dtype=D), step_size=0.01)[-1, 0]
    yT.backward()
    assert abs(float(lam.grad) - 2.0 * math.exp(-1.4) * 1.5) < 1e-8, (
        "backprop through the solver: d y(T) / d lambda = T e^(lambda T) y0 for dy/dt = lambda y")
    assert abs(float(y0.grad) - math.exp(-1.4)) < 1e-8, "d y(T) / d y0 = e^(lambda T)"


# ---------------------------------------------------------------- stage 3: the adjoint


def _loss(ys, seed=3):
    w = torch.randn(ys.shape, dtype=D, generator=torch.Generator().manual_seed(seed))
    return (w * ys).sum() + (ys[-1] ** 2).sum()


def test_stage3_same_solution_and_type_check(nb):
    field = MLPField()
    y0 = torch.randn(4, 2, dtype=D, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0.0, 0.5, 1.0], dtype=D)
    a = _solve(nb, field, y0, t, adjoint=True, step_size=0.1)
    b = _solve(nb, field, y0, t, step_size=0.1)
    assert tuple(a.shape) == (3, 4, 2) and torch.allclose(a, b, atol=1e-12), "odeint_adjoint returns odeint's solution"
    with pytest.raises(TypeError):
        nb.odeint_adjoint(lambda t, y: -y, y0, t)


def test_stage3_gradients_match_backprop(nb):
    field = MLPField(seed=2)
    y0 = torch.randn(4, 2, dtype=D, generator=torch.Generator().manual_seed(2), requires_grad=True)
    t = torch.tensor([0.0, 0.4, 1.0], dtype=D)
    _loss(_solve(nb, field, y0, t, step_size=0.02)).backward()
    want = [y0.grad.clone()] + [p.grad.clone() for p in field.parameters()]
    y0.grad = None
    field.zero_grad()
    _loss(_solve(nb, field, y0, t, adjoint=True, step_size=0.02)).backward()
    got = [y0.grad] + [p.grad for p in field.parameters()]
    assert all(g is not None for g in got), "odeint_adjoint must produce gradients for y0 and every parameter of f"
    for name, g, w in zip(["y0"] + [n for n, _ in field.named_parameters()], got, want):
        err = float((g - w).abs().max() / (w.abs().max() + 1e-12))
        assert err < 1e-6, (
            f"the adjoint gradient for {name} should match backprop through the solver to ~1e-6 (relative) with RK4 and "
            f"step 0.02; relative error {err:.1e}. the loss uses every output time: add dL/dy(t_k) to the adjoint state "
            "when the backward integration reaches t_k")


def test_stage3_finite_differences(nb):
    field = nn.Linear(2, 2, bias=False).double()
    with torch.no_grad():
        field.weight.copy_(torch.tensor([[-0.5, 1.0], [-1.0, -0.2]], dtype=D))

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = field

        def forward(self, t, y):
            return self.lin(y)

    f = Lin()
    y0 = torch.tensor([[1.0, -0.5]], dtype=D, requires_grad=True)
    t = torch.tensor([0.0, 1.5], dtype=D)
    c = torch.tensor([[0.3, 2.0]], dtype=D)
    ((_solve(nb, f, y0, t, adjoint=True, step_size=0.01)[-1] * c).sum()).backward()
    A = field.weight.detach()
    want_y0 = c @ torch.linalg.matrix_exp(A * 1.5)
    assert torch.allclose(y0.grad, want_y0, atol=1e-8), (
        f"for dy/dt = A y, dL/dy0 = c^T exp(A T); got {y0.grad.tolist()}, want {want_y0.tolist()}")
    fd = torch.zeros(2, 2, dtype=D)
    for i in range(2):
        for j in range(2):
            vals = []
            for s in (1e-6, -1e-6):
                Ap = A.clone()
                Ap[i, j] += s
                vals.append(float((y0.detach() @ torch.linalg.matrix_exp(Ap * 1.5).T * c).sum()))
            fd[i, j] = (vals[0] - vals[1]) / 2e-6
    assert torch.allclose(field.weight.grad, fd, atol=1e-6), (
        f"dL/dA from the adjoint should match finite differences of the exact solution; got\n{field.weight.grad}\n"
        f"want\n{fd}")


def test_stage3_constant_memory(nb):
    field = MLPField(hidden=32, seed=4)
    y0 = torch.randn(16, 2, dtype=D, generator=torch.Generator().manual_seed(4), requires_grad=True)
    t = torch.tensor([0.0, 1.0], dtype=D)

    def saved_bytes(adjoint, step):
        total = [0]

        def pack(x):
            total[0] += x.numel() * x.element_size()
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            _solve(nb, field, y0, t, adjoint=adjoint, step_size=step)
        return total[0]

    a10, a100 = saved_bytes(True, 0.1), saved_bytes(True, 0.01)
    b10, b100 = saved_bytes(False, 0.1), saved_bytes(False, 0.01)
    assert b100 > 5 * b10, "(sanity: backprop through the solver saves activations for every step)"
    assert a100 <= 1.1 * a10 + 1024, (
        f"odeint_adjoint must not store anything per step: autograd saved {a10:,} bytes with 10 steps and {a100:,} "
        "with 100. run the forward solve without recording a graph (torch.no_grad) and keep only the outputs")

    calls_with_grad = [0]

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = field

        def forward(self, t, y):
            calls_with_grad[0] += int(torch.is_grad_enabled() and y.requires_grad)
            return self.inner(t, y)

    out = _solve(nb, Spy(), y0, t, adjoint=True, step_size=0.1)
    assert calls_with_grad[0] == 0, "during the forward solve, f must run without building a graph"
    out.sum().backward()
    assert calls_with_grad[0] > 0, "(the backward pass evaluates vector-Jacobian products of f with grad enabled)"


# ---------------------------------------------------------------- stage 4: NeuralODE


def test_stage4_forward_and_parameters(nb):
    field = MLPField(seed=5)
    node = nb.NeuralODE(field, t1=2.0, n_steps=8)
    assert any(p is field.net[0].weight for p in node.parameters()), "func is a submodule: its parameters are the model's"
    y0 = torch.randn(3, 2, dtype=D, generator=torch.Generator().manual_seed(5))
    want = nb.odeint(field, y0, torch.tensor([0.0, 2.0], dtype=D), step_size=0.25)[-1]
    out = node(y0)
    assert tuple(out.shape) == (3, 2) and torch.allclose(out, want, atol=1e-12), (
        "NeuralODE(func, t1=2, n_steps=8) returns y(t1) from 8 equal RK4 steps of 0.25")
    count = Counter(field)

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = field

        def forward(self, t, y):
            return count(t, y)

    nb.NeuralODE(Wrap(), method="midpoint", n_steps=5)(y0)
    assert count.calls == 10, f"midpoint with n_steps=5 evaluates func 10 times; got {count.calls}"


def test_stage4_adjoint_flag(nb):
    field = MLPField(seed=6)
    y0 = torch.randn(3, 2, dtype=D, generator=torch.Generator().manual_seed(6))
    plain, adj = nb.NeuralODE(field, n_steps=20), nb.NeuralODE(field, n_steps=20, adjoint=True)
    assert torch.allclose(plain(y0), adj(y0), atol=1e-12)
    (plain(y0) ** 2).sum().backward()
    g1 = field.net[0].weight.grad.clone()
    field.zero_grad()
    (adj(y0) ** 2).sum().backward()
    assert torch.allclose(field.net[0].weight.grad, g1, rtol=1e-5, atol=1e-8), "adjoint=True: same gradients, from the adjoint"

    def saved(n_steps):
        total = [0]

        def pack(x):
            total[0] += x.numel() * x.element_size()
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            nb.NeuralODE(field, n_steps=n_steps, adjoint=True)(y0)
        return total[0]

    assert saved(100) <= 1.1 * saved(10) + 1024, (
        "NeuralODE(..., adjoint=True) must use odeint_adjoint: its saved memory can't grow with n_steps")


def test_stage4_learns_a_flow(nb):
    target = torch.tensor([[0.0, 1.0], [-1.0, -0.3]])

    def run(seed):
        torch.manual_seed(seed)
        lin = nn.Linear(2, 2, bias=False)

        class Field(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = lin

            def forward(self, t, y):
                return self.lin(y)

        node = nb.NeuralODE(Field(), t1=1.0, n_steps=10)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(128, 2, generator=g)
        y = x @ torch.linalg.matrix_exp(target).T
        opt = torch.optim.Adam(node.parameters(), lr=0.05)
        first = None
        for _ in range(150):
            loss = ((node(x) - y) ** 2).mean()
            first = float(loss) if first is None else first
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss) / first

    ratios = sorted(run(s) for s in range(5))
    assert ratios[2] < 1e-2, (
        f"a NeuralODE with a linear field should learn the flow y -> exp(A) y, cutting the loss 100-fold in 150 Adam "
        f"steps; median ratio over 5 seeds {ratios[2]:.2e}")
