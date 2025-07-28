"""
Test Suite for Vanilla VAE Implementation

Tests all components of the VAE implementation to ensure correctness
and adherence to mathematical principles.

Author: ML-from-Scratch Course
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from exercise import (
    VAEEncoder, VAEDecoder, VanillaVAE, VAELoss,
    ConditionalVAE, BetaVAE, VAETrainer, VAEAnalysis
)


class TestVAEEncoder:
    """Test VAE encoder implementation."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        input_dim = 784
        hidden_dims = [512, 256]
        latent_dim = 20
        
        encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        
        assert encoder.input_dim == input_dim
        assert encoder.hidden_dims == hidden_dims
        assert encoder.latent_dim == latent_dim
        
        # Check if encoder has the required components
        assert hasattr(encoder, 'forward')
        
    def test_encoder_forward_shape(self):
        """Test encoder forward pass output shapes."""
        input_dim = 784
        hidden_dims = [512, 256]
        latent_dim = 20
        batch_size = 32
        
        encoder = VAEEncoder(input_dim, hidden_dims, latent_dim)
        x = torch.randn(batch_size, input_dim)
        
        mu, logvar = encoder(x)
        
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)
        
    def test_encoder_parameters(self):
        """Test encoder has learnable parameters."""
        encoder = VAEEncoder(784, [512, 256], 20)
        params = list(encoder.parameters())
        assert len(params) > 0
        
        # Test parameters are trainable
        for param in params:
            assert param.requires_grad


class TestVAEDecoder:
    """Test VAE decoder implementation."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        latent_dim = 20
        hidden_dims = [256, 512]
        output_dim = 784
        
        decoder = VAEDecoder(latent_dim, hidden_dims, output_dim)
        
        assert decoder.latent_dim == latent_dim
        assert decoder.hidden_dims == hidden_dims
        assert decoder.output_dim == output_dim
        
    def test_decoder_forward_shape(self):
        """Test decoder forward pass output shapes."""
        latent_dim = 20
        hidden_dims = [256, 512]
        output_dim = 784
        batch_size = 32
        
        decoder = VAEDecoder(latent_dim, hidden_dims, output_dim)
        z = torch.randn(batch_size, latent_dim)
        
        recon = decoder(z)
        
        assert recon.shape == (batch_size, output_dim)
        
    def test_decoder_output_activation(self):
        """Test decoder output activation."""
        decoder_sigmoid = VAEDecoder(20, [256], 784, output_activation='sigmoid')
        decoder_tanh = VAEDecoder(20, [256], 784, output_activation='tanh')
        
        z = torch.randn(10, 20)
        
        out_sigmoid = decoder_sigmoid(z)
        out_tanh = decoder_tanh(z)
        
        # Sigmoid output should be in [0, 1]
        assert torch.all(out_sigmoid >= 0) and torch.all(out_sigmoid <= 1)
        
        # Tanh output should be in [-1, 1]
        assert torch.all(out_tanh >= -1) and torch.all(out_tanh <= 1)


class TestVanillaVAE:
    """Test complete VAE model."""
    
    def test_vae_initialization(self):
        """Test VAE initialization."""
        vae = VanillaVAE(784, [512, 256], 20)
        
        assert hasattr(vae, 'encoder')
        assert hasattr(vae, 'decoder')
        assert vae.input_dim == 784
        assert vae.latent_dim == 20
        
    def test_vae_forward_pass(self):
        """Test VAE forward pass."""
        batch_size = 16
        input_dim = 784
        latent_dim = 20
        
        vae = VanillaVAE(input_dim, [512, 256], latent_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = vae(x)
        
        # Check output dictionary structure
        required_keys = ['reconstruction', 'mu', 'logvar', 'z']
        for key in required_keys:
            assert key in output
            
        # Check output shapes
        assert output['reconstruction'].shape == (batch_size, input_dim)
        assert output['mu'].shape == (batch_size, latent_dim)
        assert output['logvar'].shape == (batch_size, latent_dim)
        assert output['z'].shape == (batch_size, latent_dim)
        
    def test_reparameterization_trick(self):
        """Test reparameterization trick implementation."""
        vae = VanillaVAE(784, [512, 256], 20)
        
        mu = torch.zeros(10, 20)
        logvar = torch.zeros(10, 20)  # std = 1
        
        # Test multiple samples to check stochasticity
        z1 = vae.reparameterize(mu, logvar)
        z2 = vae.reparameterize(mu, logvar)
        
        # Should be different (stochastic)
        assert not torch.allclose(z1, z2, atol=1e-6)
        
        # Should have correct moments approximately
        z_samples = torch.stack([vae.reparameterize(mu, logvar) for _ in range(1000)])
        sample_mean = z_samples.mean(dim=0)
        sample_std = z_samples.std(dim=0)
        
        assert torch.allclose(sample_mean, mu, atol=0.1)
        assert torch.allclose(sample_std, torch.ones_like(sample_std), atol=0.1)
        
    def test_vae_sampling(self):
        """Test VAE sampling from prior."""
        vae = VanillaVAE(784, [512, 256], 20)
        device = torch.device('cpu')
        
        samples = vae.sample(10, device)
        
        assert samples.shape == (10, 784)
        assert samples.device == device
        
    def test_vae_gradients(self):
        """Test that gradients flow through VAE."""
        vae = VanillaVAE(784, [512, 256], 20)
        x = torch.randn(4, 784)
        
        output = vae(x)
        loss = F.mse_loss(output['reconstruction'], x)
        loss.backward()
        
        # Check that parameters have gradients
        for param in vae.parameters():
            assert param.grad is not None


class TestVAELoss:
    """Test VAE loss function."""
    
    def test_loss_initialization(self):
        """Test loss function initialization."""
        loss_fn = VAELoss()
        assert loss_fn.reconstruction_loss == 'mse'
        assert loss_fn.beta == 1.0
        
        loss_fn_bce = VAELoss(reconstruction_loss='bce', beta=2.0)
        assert loss_fn_bce.reconstruction_loss == 'bce'
        assert loss_fn_bce.beta == 2.0
        
    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        loss_fn = VAELoss()
        
        # Test case: standard normal should have KL = 0
        mu = torch.zeros(10, 20)
        logvar = torch.zeros(10, 20)  # variance = 1
        
        kl = loss_fn.kl_divergence(mu, logvar)
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)
        
        # Test case: non-zero mean should increase KL
        mu_nonzero = torch.ones(10, 20)
        kl_nonzero = loss_fn.kl_divergence(mu_nonzero, logvar)
        assert kl_nonzero > 0
        
        # Test case: high variance should decrease KL
        logvar_high = torch.ones(10, 20) * 2  # variance = e^2
        kl_high_var = loss_fn.kl_divergence(mu, logvar_high)
        assert kl_high_var < 0  # Negative KL when variance > 1
        
    def test_reconstruction_loss(self):
        """Test reconstruction loss computation."""
        loss_fn_mse = VAELoss(reconstruction_loss='mse')
        loss_fn_bce = VAELoss(reconstruction_loss='bce')
        
        x_target = torch.randn(10, 784)
        x_recon = torch.randn(10, 784)
        
        # MSE loss
        mse_loss = loss_fn_mse.reconstruction_loss_fn(x_recon, x_target)
        expected_mse = F.mse_loss(x_recon, x_target)
        assert torch.isclose(mse_loss, expected_mse)
        
        # BCE loss (need to sigmoid the reconstruction for BCE)
        x_target_binary = torch.sigmoid(x_target)
        x_recon_binary = torch.sigmoid(x_recon)
        bce_loss = loss_fn_bce.reconstruction_loss_fn(x_recon_binary, x_target_binary)
        expected_bce = F.binary_cross_entropy(x_recon_binary, x_target_binary)
        assert torch.isclose(bce_loss, expected_bce)
        
    def test_total_loss_computation(self):
        """Test total loss computation."""
        loss_fn = VAELoss(beta=2.0)
        
        # Create mock model output
        model_output = {
            'reconstruction': torch.randn(10, 784),
            'mu': torch.randn(10, 20),
            'logvar': torch.randn(10, 20)
        }
        target = torch.randn(10, 784)
        
        loss_dict = loss_fn(model_output, target)
        
        # Check loss dictionary structure
        required_keys = ['total_loss', 'reconstruction_loss', 'kl_loss']
        for key in required_keys:
            assert key in loss_dict
            
        # Check that total loss is sum of components
        expected_total = loss_dict['reconstruction_loss'] + 2.0 * loss_dict['kl_loss']
        assert torch.isclose(loss_dict['total_loss'], expected_total)


class TestConditionalVAE:
    """Test Conditional VAE implementation."""
    
    def test_cvae_initialization(self):
        """Test Conditional VAE initialization."""
        cvae = ConditionalVAE(784, 10, [512, 256], 20)
        
        assert cvae.input_dim == 784
        assert cvae.condition_dim == 10
        assert cvae.latent_dim == 20
        
    def test_cvae_forward_pass(self):
        """Test Conditional VAE forward pass."""
        batch_size = 16
        input_dim = 784
        condition_dim = 10
        latent_dim = 20
        
        cvae = ConditionalVAE(input_dim, condition_dim, [512, 256], latent_dim)
        x = torch.randn(batch_size, input_dim)
        condition = torch.randn(batch_size, condition_dim)
        
        output = cvae(x, condition)
        
        # Similar to VAE but with conditioning
        required_keys = ['reconstruction', 'mu', 'logvar', 'z']
        for key in required_keys:
            assert key in output
            
        assert output['reconstruction'].shape == (batch_size, input_dim)


class TestBetaVAE:
    """Test β-VAE implementation."""
    
    def test_beta_vae_initialization(self):
        """Test β-VAE initialization."""
        beta_vae = BetaVAE(784, [512, 256], 20, beta=4.0)
        
        assert beta_vae.beta == 4.0
        assert isinstance(beta_vae, VanillaVAE)  # Should inherit from VAE
        
    def test_beta_vae_forward_pass(self):
        """Test β-VAE forward pass."""
        beta_vae = BetaVAE(784, [512, 256], 20, beta=4.0)
        x = torch.randn(16, 784)
        
        output = beta_vae(x)
        
        # Should have same output structure as regular VAE
        required_keys = ['reconstruction', 'mu', 'logvar', 'z']
        for key in required_keys:
            assert key in output


class TestVAETrainer:
    """Test VAE training utilities."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        vae = VanillaVAE(784, [512, 256], 20)
        device = torch.device('cpu')
        trainer = VAETrainer(vae, device)
        
        assert trainer.model == vae
        assert trainer.device == device
        
    def test_training_step(self):
        """Test single training step."""
        vae = VanillaVAE(784, [512, 256], 20)
        device = torch.device('cpu')
        trainer = VAETrainer(vae, device)
        
        batch = torch.randn(16, 784)
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        loss_fn = VAELoss()
        
        # Get initial parameters
        initial_params = [p.clone() for p in vae.parameters()]
        
        # Training step
        loss_dict = trainer.train_step(batch, optimizer, loss_fn)
        
        # Check loss dictionary
        assert isinstance(loss_dict, dict)
        assert 'total_loss' in loss_dict
        
        # Check parameters were updated
        for initial, current in zip(initial_params, vae.parameters()):
            assert not torch.equal(initial, current)


class TestVAEAnalysis:
    """Test VAE analysis utilities."""
    
    def test_latent_interpolation(self):
        """Test latent space interpolation."""
        vae = VanillaVAE(784, [512, 256], 20)
        
        x1 = torch.randn(784)
        x2 = torch.randn(784)
        
        interpolations = VAEAnalysis.latent_space_interpolation(vae, x1, x2, num_steps=5)
        
        assert interpolations.shape == (5, 784)
        
        # First and last should be close to original reconstructions
        with torch.no_grad():
            recon1 = vae(x1.unsqueeze(0))['reconstruction'][0]
            recon2 = vae(x2.unsqueeze(0))['reconstruction'][0]
            
        # Check endpoints are close to original reconstructions
        assert torch.allclose(interpolations[0], recon1, atol=1e-5)
        assert torch.allclose(interpolations[-1], recon2, atol=1e-5)


class TestMathematicalProperties:
    """Test mathematical properties of VAE implementation."""
    
    def test_elbo_properties(self):
        """Test ELBO mathematical properties."""
        vae = VanillaVAE(784, [512, 256], 20)
        loss_fn = VAELoss()
        
        x = torch.randn(16, 784)
        output = vae(x)
        loss_dict = loss_fn(output, x)
        
        # ELBO should be finite
        assert torch.isfinite(loss_dict['total_loss'])
        assert torch.isfinite(loss_dict['reconstruction_loss'])
        assert torch.isfinite(loss_dict['kl_loss'])
        
        # All loss components should be non-negative
        assert loss_dict['reconstruction_loss'] >= 0
        # Note: KL can be negative if posterior variance > 1
        
    def test_kl_divergence_properties(self):
        """Test KL divergence mathematical properties."""
        loss_fn = VAELoss()
        
        # Property 1: KL(p||p) = 0
        mu = torch.randn(10, 20)
        logvar = torch.randn(10, 20)
        kl_self = loss_fn.kl_divergence(mu, logvar)
        
        # For same distribution, we're computing KL(q||p) where p=N(0,I)
        # This is not necessarily 0, but should be well-defined
        assert torch.isfinite(kl_self)
        
        # Property 2: KL divergence for standard normal
        mu_zero = torch.zeros(10, 20)
        logvar_zero = torch.zeros(10, 20)  # std = 1
        kl_standard = loss_fn.kl_divergence(mu_zero, logvar_zero)
        
        # KL(N(0,1) || N(0,1)) = 0
        assert torch.isclose(kl_standard, torch.tensor(0.0), atol=1e-6)
        
    def test_reparameterization_gradient_flow(self):
        """Test that reparameterization allows gradient flow."""
        vae = VanillaVAE(784, [256], 20)
        
        x = torch.randn(4, 784, requires_grad=True)
        output = vae(x)
        
        # Compute loss and backpropagate
        loss = output['z'].sum()  # Simple loss on latent variables
        loss.backward()
        
        # Check that gradients flow back to input
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestNumericalStability:
    """Test numerical stability of implementation."""
    
    def test_numerical_stability_extreme_values(self):
        """Test stability with extreme input values."""
        vae = VanillaVAE(784, [256], 20)
        
        # Test with very large values
        x_large = torch.ones(4, 784) * 100
        output_large = vae(x_large)
        assert torch.all(torch.isfinite(output_large['reconstruction']))
        
        # Test with very small values
        x_small = torch.ones(4, 784) * 1e-6
        output_small = vae(x_small)
        assert torch.all(torch.isfinite(output_small['reconstruction']))
        
    def test_logvar_numerical_stability(self):
        """Test numerical stability of log variance."""
        loss_fn = VAELoss()
        
        # Test with very negative logvar (very small variance)
        mu = torch.zeros(10, 20)
        logvar_small = torch.ones(10, 20) * -10  # Very small variance
        
        kl = loss_fn.kl_divergence(mu, logvar_small)
        assert torch.isfinite(kl)
        
        # Test with very positive logvar (very large variance)
        logvar_large = torch.ones(10, 20) * 10  # Very large variance
        kl_large = loss_fn.kl_divergence(mu, logvar_large)
        assert torch.isfinite(kl_large)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])