import torch
import torch.nn as nn


class PlanarFlow(nn.Module):
    def __init__(self, dim: int, activation: str = "tanh"):
        super().__init__()
        self.dim = dim
        self.activation = activation
        self.u = nn.Parameter(torch.randn(dim) * 0.1)
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(1))

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def _activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 - torch.tanh(x) ** 2

    def forward(self, z: torch.Tensor):
        linear = z @ self.w + self.b
        transformed = z + self.u * self._activation(linear).unsqueeze(-1)
        psi = self._activation_derivative(linear).unsqueeze(-1) * self.w
        log_det = torch.log(torch.abs(1.0 + psi @ self.u.unsqueeze(-1)).squeeze(-1).squeeze(-1) + 1e-6)
        return transformed, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, dim: int, n_flows: int, flow_type: str = "planar"):
        super().__init__()
        del flow_type
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])

    def forward(self, z: torch.Tensor):
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, log_det = flow(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det
