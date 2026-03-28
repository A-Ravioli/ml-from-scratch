import torch

from exercise import EulerSolver, NeuralODE, NeuralODEFunction, RungeKutta4Solver


def _decay(t, y):
    return -y


def test_euler_solver_shape():
    solver = EulerSolver(_decay)
    y0 = torch.tensor([[1.0], [2.0]])
    t = torch.linspace(0.0, 1.0, 6)
    sol = solver.integrate(y0, t)
    assert sol.shape == (6, 2, 1)


def test_rk4_solver_accuracy():
    solver = RungeKutta4Solver(_decay)
    y0 = torch.tensor([[1.0]])
    t = torch.linspace(0.0, 1.0, 11)
    sol = solver.integrate(y0, t)
    expected = torch.exp(-t).view(-1, 1, 1)
    assert torch.max(torch.abs(sol - expected)) < 0.05


def test_neural_ode_forward():
    func = NeuralODEFunction(dim=3, hidden_dim=8)
    model = NeuralODE(func, solver="rk4")
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
