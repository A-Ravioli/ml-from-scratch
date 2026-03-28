import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def _compute_eigendecomposition(self, laplacian: torch.Tensor):
        return torch.linalg.eigh(laplacian)

    def _spectral_filter(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        return torch.exp(-eigenvalues)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = self._compute_eigendecomposition(laplacian)
        filt = self._spectral_filter(eigenvalues)
        filtered = eigenvectors @ torch.diag(filt) @ eigenvectors.T
        return filtered @ x @ self.weight


class ChebyshevGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(torch.randn(K, in_features, out_features) * 0.1)

    def _chebyshev_polynomial(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if k == 0:
            return torch.ones_like(x)
        if k == 1:
            return x
        t0 = torch.ones_like(x)
        t1 = x
        for _ in range(2, k + 1):
            t0, t1 = t1, 2 * x * t1 - t0
        return t1

    def _normalize_laplacian(self, laplacian: torch.Tensor) -> torch.Tensor:
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        max_eig = torch.clamp(eigenvalues.max(), min=1e-6)
        identity = torch.eye(laplacian.shape[0], device=laplacian.device, dtype=laplacian.dtype)
        return (2.0 / max_eig) * laplacian - identity

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_laplacian(laplacian)
        t0 = x
        outputs = t0 @ self.weight[0]
        if self.K == 1:
            return outputs
        t1 = normalized @ x
        outputs = outputs + t1 @ self.weight[1]
        for k in range(2, self.K):
            t2 = 2 * normalized @ t1 - t0
            outputs = outputs + t2 @ self.weight[k]
            t0, t1 = t1, t2
        return outputs


class SpatialGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, add_self_loops: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_self_loops = add_self_loops
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)

    def _normalize_adjacency(self, adj: torch.Tensor) -> torch.Tensor:
        if self.add_self_loops:
            adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        degree = adj.sum(dim=1)
        inv_sqrt = torch.pow(torch.clamp(degree, min=1.0), -0.5)
        D = torch.diag(inv_sqrt)
        return D @ adj @ D

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        norm_adj = self._normalize_adjacency(adj)
        return norm_adj @ x @ self.weight


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * max(1, num_layers - 1) + [output_dim]
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(SpatialGraphConv(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x, adj)
            if index < len(self.layers) - 1:
                x = F.relu(x)
        return x
