import torch

from exercise import WassersteinCritic, WassersteinGAN, WassersteinGANGP, WassersteinGenerator


def test_wgan_module_shapes():
    gen = WassersteinGenerator(noise_dim=4, hidden_dim=8, output_dim=6)
    critic = WassersteinCritic(input_dim=6, hidden_dim=8)
    z = torch.randn(5, 4)
    fake = gen(z)
    scores = critic(fake)
    assert fake.shape == (5, 6)
    assert scores.shape == (5, 1)


def test_wgan_losses_and_gradient_penalty():
    gen = WassersteinGenerator(noise_dim=4, hidden_dim=8, output_dim=6)
    critic = WassersteinCritic(input_dim=6, hidden_dim=8)
    gan = WassersteinGAN(gen, critic, noise_dim=4)
    real = torch.randn(5, 6)
    losses = gan.compute_losses(real)
    assert "generator" in losses and "critic" in losses
    gp_model = WassersteinGANGP(gen, critic, noise_dim=4)
    fake = gen(torch.randn(5, 4))
    penalty = gp_model.gradient_penalty(real, fake)
    assert penalty.item() >= 0.0
