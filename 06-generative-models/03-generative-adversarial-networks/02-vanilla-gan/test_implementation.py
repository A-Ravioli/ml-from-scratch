import torch

from exercise import Discriminator, Generator, VanillaGAN


def test_generator_and_discriminator_shapes():
    gen = Generator(noise_dim=4, hidden_dim=8, output_dim=6)
    disc = Discriminator(input_dim=6, hidden_dim=8)
    z = torch.randn(5, 4)
    fake = gen(z)
    logits = disc(fake)
    assert fake.shape == (5, 6)
    assert logits.shape == (5, 1)


def test_vanilla_gan_losses():
    gen = Generator(noise_dim=4, hidden_dim=8, output_dim=6)
    disc = Discriminator(input_dim=6, hidden_dim=8)
    gan = VanillaGAN(gen, disc, noise_dim=4)
    real = torch.randn(5, 6)
    losses = gan.compute_losses(real)
    assert losses["generator"].item() > 0.0
    assert losses["discriminator"].item() > 0.0
