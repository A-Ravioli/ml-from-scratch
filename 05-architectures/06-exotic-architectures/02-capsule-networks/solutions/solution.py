import torch
import torch.nn as nn
import torch.nn.functional as F


class SquashFunction(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        squared_norm = torch.sum(s * s, dim=self.dim, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm)
        return scale * s / torch.sqrt(squared_norm + 1e-8)


class PrimaryCapsuleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        capsule_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.squash = SquashFunction(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        batch, channels, height, width = features.shape
        features = features.view(batch, self.out_channels, self.capsule_dim, height * width)
        features = features.permute(0, 1, 3, 2).reshape(batch, -1, self.capsule_dim)
        return self.squash(features)


class DynamicRouting(nn.Module):
    def __init__(
        self,
        num_input_capsules: int,
        num_output_capsules: int,
        input_capsule_dim: int,
        output_capsule_dim: int,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_input_capsules = num_input_capsules
        self.num_output_capsules = num_output_capsules
        self.num_iterations = num_iterations
        self.weight = nn.Parameter(
            torch.randn(num_input_capsules, num_output_capsules, input_capsule_dim, output_capsule_dim) * 0.1
        )
        self.squash = SquashFunction(dim=-1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_hat = torch.einsum("bid,iodh->bioh", u, self.weight)
        logits = torch.zeros(u_hat.shape[:3], device=u.device, dtype=u.dtype)
        for _ in range(self.num_iterations):
            coupling = F.softmax(logits, dim=-1)
            s = torch.einsum("bio,bioh->boh", coupling, u_hat)
            v = self.squash(s)
            logits = logits + torch.einsum("boh,bioh->bio", v, u_hat)
        return v
