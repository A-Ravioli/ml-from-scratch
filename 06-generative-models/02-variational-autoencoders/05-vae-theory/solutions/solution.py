from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAETheory:
    @staticmethod
    def compute_elbo(reconstruction_loss: torch.Tensor, kl_divergence: torch.Tensor) -> torch.Tensor:
        return -(reconstruction_loss + kl_divergence)


class VariationalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, output_distribution: str = "bernoulli"):
        super().__init__()
        self.output_distribution = output_distribution
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        output = self.net(z)
        if self.output_distribution == "bernoulli":
            return torch.sigmoid(output)
        return output


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, output_distribution: str = "bernoulli"):
        super().__init__()
        self.encoder = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VariationalDecoder(latent_dim, hidden_dim, input_dim, output_distribution)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def loss_function(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        recon_loss = F.mse_loss(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {"loss": recon_loss + kl, "reconstruction": recon_loss, "kl": kl}


class VAEAnalyzer:
    @staticmethod
    def posterior_collapse_detection(vae: VariationalAutoencoder, dataloader, threshold: float = 0.1) -> Dict[str, float]:
        if isinstance(dataloader, torch.Tensor):
            data = dataloader
        else:
            batches = [batch[0] if isinstance(batch, (tuple, list)) else batch for batch in dataloader]
            data = torch.cat(batches, dim=0)
        with torch.no_grad():
            mu, logvar = vae.encoder(data)
        variances = torch.exp(logvar).mean(dim=0)
        collapsed = int((variances < threshold).sum().item())
        return {"collapsed_dims": collapsed, "mean_variance": float(variances.mean().item())}


def create_synthetic_vae_data(n_samples: int = 1000, data_type: str = "gaussian_mixture") -> torch.Tensor:
    if data_type == "gaussian_mixture":
        centers = torch.tensor([[1.5, 0.0], [-1.5, 0.0], [0.0, 1.5]])
        assignments = torch.randint(0, len(centers), (n_samples,))
        noise = 0.2 * torch.randn(n_samples, 2)
        return centers[assignments] + noise
    return torch.randn(n_samples, 2)
