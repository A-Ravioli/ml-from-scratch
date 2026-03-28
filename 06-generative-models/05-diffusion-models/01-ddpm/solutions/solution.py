from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseScheduler:
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, schedule_type: str = "linear"):
        self.num_timesteps = num_timesteps
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor):
        noise = torch.randn_like(x)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1).to(x.device)
        noisy = torch.sqrt(alpha_bar) * x + torch.sqrt(1.0 - alpha_bar) * noise
        return noisy, noise


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_emb_dim: int = 128, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        del hidden_dims
        self.time_mlp = nn.Sequential(nn.Linear(1, time_emb_dim), nn.ReLU(), nn.Linear(time_emb_dim, in_channels))
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t.float().view(-1, 1)).view(-1, x.shape[1], 1, 1)
        h = F.relu(self.conv1(x + time_emb))
        return self.conv2(h)


class DDPM(nn.Module):
    def __init__(self, model: UNet, noise_scheduler: NoiseScheduler, device: str = "cpu"):
        super().__init__()
        self.model = model.to(device)
        self.noise_scheduler = noise_scheduler
        self.device = device

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        t = torch.randint(0, self.noise_scheduler.num_timesteps, (x.shape[0],), device=x.device)
        noisy, noise = self.noise_scheduler.add_noise(x, t.cpu())
        noisy = noisy.to(self.device)
        noise = noise.to(self.device)
        pred = self.model(noisy, t)
        return F.mse_loss(pred, noise)


class DDPMSampler:
    def __init__(self, ddpm: DDPM):
        self.ddpm = ddpm

    def sample(self, shape: Tuple[int, ...], steps: int = 10) -> torch.Tensor:
        x = torch.randn(*shape, device=self.ddpm.device)
        for step in reversed(range(steps)):
            t = torch.full((shape[0],), step, dtype=torch.long, device=self.ddpm.device)
            pred_noise = self.ddpm.model(x, t)
            x = x - 0.1 * pred_noise
        return x
