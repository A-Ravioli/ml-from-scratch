import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 100, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VanillaGAN:
    def __init__(self, generator: Generator, discriminator: Discriminator, noise_dim: int = 100, device: str = "cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.noise_dim = noise_dim
        self.device = device

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=self.device)

    def compute_losses(self, real_data: torch.Tensor):
        real_data = real_data.to(self.device)
        fake = self.generator(self.sample_noise(real_data.shape[0]))
        real_logits = self.discriminator(real_data)
        fake_logits = self.discriminator(fake.detach())
        generator_logits = self.discriminator(fake)
        d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        d_loss = d_loss + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        g_loss = F.binary_cross_entropy_with_logits(generator_logits, torch.ones_like(generator_logits))
        return {"generator": g_loss, "discriminator": d_loss}
