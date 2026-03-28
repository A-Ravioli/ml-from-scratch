import torch
import torch.nn as nn


class EulerSolver:
    def __init__(self, func):
        self.func = func

    def integrate(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        states = [y0]
        y = y0
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            y = y + dt * self.func(t[i - 1], y)
            states.append(y)
        return torch.stack(states, dim=0)


class RungeKutta4Solver(EulerSolver):
    def integrate(self, y0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        states = [y0]
        y = y0
        for i in range(1, len(t)):
            t_prev = t[i - 1]
            dt = t[i] - t_prev
            k1 = self.func(t_prev, y)
            k2 = self.func(t_prev + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.func(t_prev + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.func(t_prev + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            states.append(y)
        return torch.stack(states, dim=0)


class NeuralODEFunction(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        layers = [nn.Linear(dim + 1, hidden_dim), nn.Tanh()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        time_feature = torch.full((y.shape[0], 1), float(t), dtype=y.dtype, device=y.device)
        return self.net(torch.cat([y, time_feature], dim=-1))


class NeuralODE(nn.Module):
    def __init__(self, func: NeuralODEFunction, solver: str = "rk4"):
        super().__init__()
        self.func = func
        self.solver_name = solver

    def _make_solver(self):
        if self.solver_name == "euler":
            return EulerSolver(self.func)
        return RungeKutta4Solver(self.func)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if t is None:
            t = torch.tensor([0.0, 1.0], dtype=x.dtype, device=x.device)
        solver = self._make_solver()
        solution = solver.integrate(x, t)
        return solution[-1]
