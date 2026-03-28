import torch

from exercise import DDPM, DDPMSampler, NoiseScheduler, UNet


def test_noise_scheduler_add_noise():
    scheduler = NoiseScheduler(num_timesteps=10, beta_start=1e-4, beta_end=0.02)
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([1, 5], dtype=torch.long)
    noisy, noise = scheduler.add_noise(x, t)
    assert noisy.shape == x.shape
    assert noise.shape == x.shape


def test_unet_forward_shape():
    model = UNet(in_channels=1, out_channels=1, time_emb_dim=16, hidden_dims=[8, 16])
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([1, 2], dtype=torch.long)
    out = model(x, t)
    assert out.shape == x.shape


def test_ddpm_loss_and_sampling():
    scheduler = NoiseScheduler(num_timesteps=10, beta_start=1e-4, beta_end=0.02)
    model = UNet(in_channels=1, out_channels=1, time_emb_dim=16, hidden_dims=[8, 16])
    ddpm = DDPM(model, scheduler)
    x = torch.randn(2, 1, 8, 8)
    loss = ddpm.loss(x)
    sampler = DDPMSampler(ddpm)
    samples = sampler.sample((2, 1, 8, 8), steps=4)
    assert loss.item() >= 0.0
    assert samples.shape == (2, 1, 8, 8)
