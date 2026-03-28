"""
Compact tests for VQ-VAE reference implementations.
"""

import torch

from exercise import VQLoss, VQVAE, VectorQuantizer


def test_vector_quantizer_output_shapes():
    quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8, use_ema=False)
    inputs = torch.randn(2, 4, 4, 8)
    out = quantizer(inputs)
    assert out["quantized"].shape == inputs.shape
    assert out["encoding_indices"].shape == (2, 4, 4)


def test_vqvae_forward_shapes():
    model = VQVAE(in_channels=3, hidden_channels=32, num_embeddings=32, embedding_dim=8, use_ema=False)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out["reconstruction"].shape == x.shape
    assert out["vq_loss"].dim() == 0
    assert out["commit_loss"].dim() == 0


def test_vqvae_loss_keys():
    model = VQVAE(in_channels=3, hidden_channels=32, num_embeddings=32, embedding_dim=8, use_ema=False)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    loss_fn = VQLoss()
    loss = loss_fn(out, x)
    assert set(loss.keys()) == {"total_loss", "reconstruction_loss", "vq_loss", "commitment_loss"}
