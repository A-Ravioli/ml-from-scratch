import torch
import torch.nn as nn


class WassersteinCritic(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WassersteinGenerator(nn.Module):
    def __init__(self, noise_dim: int = 100, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(noise_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class WassersteinGAN:
    def __init__(self, generator: WassersteinGenerator, critic: WassersteinCritic, noise_dim: int = 100, device: str = "cpu"):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.noise_dim = noise_dim
        self.device = device

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=self.device)

    def compute_losses(self, real_data: torch.Tensor):
        real_data = real_data.to(self.device)
        fake = self.generator(self.sample_noise(real_data.shape[0]))
        real_score = self.critic(real_data)
        fake_score = self.critic(fake.detach())
        critic_loss = fake_score.mean() - real_score.mean()
        generator_loss = -self.critic(fake).mean()
        return {"generator": generator_loss, "critic": critic_loss}


class WassersteinGANGP(WassersteinGAN):
    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
        alpha = torch.rand(real_data.shape[0], 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        score = self.critic(interpolated)
        grads = torch.autograd.grad(score.sum(), interpolated, create_graph=True)[0]
        penalty = ((grads.norm(dim=1) - 1.0) ** 2).mean()
        return lambda_gp * penalty
