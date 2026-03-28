"""
Compact tests for hierarchical VAE reference implementations.
"""

import torch

from exercise import HierarchicalLoss, LadderBlock, LadderVAE, ResidualBlock


def test_ladder_block_shapes():
    block = LadderBlock(input_dim=16, latent_dim=4, hidden_dim=8)
    out = block(torch.randn(2, 16), None)
    assert out["z_sample"].shape == (2, 4)
    assert out["td_features"].shape == (2, 8)


def test_ladder_vae_initialization():
    model = LadderVAE(input_dim=32, latent_dims=[8, 4], hidden_dims=[16, 8])
    assert model.num_levels == 2
    assert len(model.ladder_blocks) == 2


def test_residual_block_preserves_shape():
    block = ResidualBlock(channels=8)
    x = torch.randn(2, 8, 4, 4)
    out = block(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_hierarchical_loss_is_constructible():
    loss_fn = HierarchicalLoss(num_levels=2, beta_schedule="constant")
    assert loss_fn.num_levels == 2
