"""
Test Suite for VQ-VAE Implementation

This test suite verifies correctness of VQ-VAE implementations.
Run with: python test_implementation.py

Author: ML-from-Scratch Course
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import math
from typing import List, Tuple, Dict

from exercise import (
    VectorQuantizer, VQVAE, VQVAEHierarchical, VQGAN, VQGANDiscriminator,
    ResidualVectorQuantizer, FiniteScalarQuantizer, VQVAELoss, VQVAETrainer,
    VQVAEEvaluation, visualize_codebook, interpolate_in_codebook_space
)


class TestVectorQuantizer:
    """Test core vector quantization functionality."""
    
    def test_quantizer_forward_shapes(self):
        """Test that quantizer produces correct output shapes."""
        num_embeddings, embedding_dim = 512, 64
        batch_size, height, width = 4, 8, 8
        
        quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Test input
        inputs = torch.randn(batch_size, height, width, embedding_dim)
        outputs = quantizer(inputs)
        
        # Check output shapes
        expected_keys = ['quantized', 'vq_loss', 'commitment_loss', 'encodings', 'encoding_indices']
        for key in expected_keys:
            assert key in outputs, f"Missing output key: {key}"
            
        # Quantized output should match input shape
        assert outputs['quantized'].shape == inputs.shape, \
            f"Quantized shape {outputs['quantized'].shape} doesn't match input {inputs.shape}"
            
        # Encoding indices should be integers
        indices = outputs['encoding_indices']
        assert indices.dtype in [torch.long, torch.int64], \
            f"Encoding indices should be integers, got {indices.dtype}"
            
        # Indices should be in valid range
        assert indices.min() >= 0 and indices.max() < num_embeddings, \
            f"Indices out of range: [{indices.min()}, {indices.max()}], expected [0, {num_embeddings-1}]"
            
    def test_straight_through_gradient(self):
        """Test straight-through estimator for gradient flow."""
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=32)
        
        # Create input that requires gradients
        inputs = torch.randn(2, 4, 4, 32, requires_grad=True)
        outputs = quantizer(inputs)
        
        # Create dummy loss and backpropagate
        loss = outputs['quantized'].sum()
        loss.backward()
        
        # Input should receive gradients (straight-through)
        assert inputs.grad is not None, "Input should receive gradients through quantizer"
        assert inputs.grad.shape == inputs.shape, "Gradient shape should match input shape"
        
        # Gradients should not be zero (would indicate blocked flow)
        assert not torch.allclose(inputs.grad, torch.zeros_like(inputs.grad)), \
            "Gradients should flow through quantizer"
            
    def test_quantization_consistency(self):
        """Test that quantization is deterministic and consistent."""
        quantizer = VectorQuantizer(num_embeddings=16, embedding_dim=8)
        
        inputs = torch.randn(2, 4, 4, 8)
        
        # Run quantization twice
        outputs1 = quantizer(inputs)
        outputs2 = quantizer(inputs)
        
        # Should get identical results
        assert torch.allclose(outputs1['quantized'], outputs2['quantized']), \
            "Quantization should be deterministic"
        assert torch.equal(outputs1['encoding_indices'], outputs2['encoding_indices']), \
            "Encoding indices should be identical"
            
    def test_codebook_updates(self):
        """Test EMA codebook updates."""
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16, use_ema=True)
        
        # Get initial codebook
        initial_embeddings = quantizer.embedding.weight.clone()
        
        # Run several forward passes
        for _ in range(10):
            inputs = torch.randn(4, 8, 8, 16)
            outputs = quantizer(inputs)
            
        # Codebook should have changed (if EMA is working)
        final_embeddings = quantizer.embedding.weight
        assert not torch.allclose(initial_embeddings, final_embeddings, atol=1e-6), \
            "Codebook should update during training"
            
    def test_codebook_utilization(self):
        """Test codebook utilization tracking."""
        quantizer = VectorQuantizer(num_embeddings=64, embedding_dim=32)
        
        # Run forward pass
        inputs = torch.randn(8, 16, 16, 32)
        outputs = quantizer(inputs)
        
        # Get usage statistics
        usage_stats = quantizer.get_codebook_usage()
        
        expected_keys = ['perplexity', 'num_active_codes', 'usage_distribution']
        for key in expected_keys:
            assert key in usage_stats, f"Missing usage statistic: {key}"
            
        # Perplexity should be positive and <= num_embeddings
        perplexity = usage_stats['perplexity']
        assert 1 <= perplexity <= 64, f"Perplexity {perplexity} out of reasonable range"


class TestVQVAE:
    """Test complete VQ-VAE model."""
    
    def test_vqvae_forward_shapes(self):
        """Test VQ-VAE forward pass shapes."""
        model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=512)
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 64, 64)
        
        outputs = model(x)
        
        # Check required outputs
        required_keys = ['reconstruction', 'vq_loss', 'commitment_loss']
        for key in required_keys:
            assert key in outputs, f"Missing output: {key}"
            
        # Reconstruction should match input shape
        recon = outputs['reconstruction']
        assert recon.shape == x.shape, \
            f"Reconstruction shape {recon.shape} doesn't match input {x.shape}"
            
        # Losses should be scalars
        assert outputs['vq_loss'].dim() == 0, "VQ loss should be scalar"
        assert outputs['commitment_loss'].dim() == 0, "Commitment loss should be scalar"
        
    def test_vqvae_encode_decode_consistency(self):
        """Test encode/decode consistency."""
        model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=256)
        
        x = torch.randn(2, 3, 32, 32)
        
        # Encode to discrete codes
        codes = model.encode(x)
        
        # Decode back
        reconstruction = model.decode_codes(codes)
        
        # Should match original input shape
        assert reconstruction.shape == x.shape, \
            f"Decode shape {reconstruction.shape} doesn't match input {x.shape}"
            
        # Codes should be integers
        assert codes.dtype in [torch.long, torch.int64], \
            f"Codes should be integers, got {codes.dtype}"
            
    def test_vqvae_reconstruction_quality(self):
        """Test that VQ-VAE can reconstruct simple patterns."""
        model = VQVAE(in_channels=1, embedding_dim=32, num_embeddings=64)
        
        # Create simple pattern that should be easy to encode
        batch_size = 4
        x = torch.zeros(batch_size, 1, 16, 16)
        x[:, :, :8, :8] = 1.0  # Top-left quadrant
        
        outputs = model(x)
        reconstruction = outputs['reconstruction']
        
        # Reconstruction should be reasonable (not all zeros or ones)
        assert 0.1 < reconstruction.mean() < 0.9, \
            "Reconstruction should be in reasonable range"
        assert reconstruction.std() > 0.01, \
            "Reconstruction should have reasonable variance"
            
    def test_vqvae_loss_computation(self):
        """Test that losses are computed correctly."""
        model = VQVAE(in_channels=3, embedding_dim=64, num_embeddings=128)
        
        x = torch.randn(2, 3, 32, 32)
        outputs = model(x)
        
        # All losses should be non-negative
        assert outputs['vq_loss'] >= 0, "VQ loss should be non-negative"
        assert outputs['commitment_loss'] >= 0, "Commitment loss should be non-negative"
        
        # Losses should not be NaN or infinite
        assert not torch.isnan(outputs['vq_loss']), "VQ loss should not be NaN"
        assert not torch.isinf(outputs['vq_loss']), "VQ loss should not be infinite"


class TestVQVAEHierarchical:
    """Test hierarchical VQ-VAE (VQ-VAE-2 style)."""
    
    def test_hierarchical_forward_shapes(self):
        """Test hierarchical VQ-VAE forward pass."""
        model = VQVAEHierarchical(in_channels=3, embedding_dim=64)
        
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x)
        
        # Should have losses for both levels
        expected_keys = ['reconstruction', 'vq_loss_top', 'vq_loss_bottom', 
                        'commitment_loss_top', 'commitment_loss_bottom']
        for key in expected_keys:
            assert key in outputs, f"Missing hierarchical output: {key}"
            
        # Reconstruction should match input
        assert outputs['reconstruction'].shape == x.shape
        
    def test_hierarchical_encode_decode(self):
        """Test hierarchical encoding and decoding."""
        model = VQVAEHierarchical(in_channels=3, embedding_dim=64)
        
        x = torch.randn(2, 3, 32, 32)
        
        # Encode hierarchically
        codes_top, codes_bottom = model.encode_hierarchical(x)
        
        # Decode from hierarchical codes
        reconstruction = model.decode_hierarchical(codes_top, codes_bottom)
        
        assert reconstruction.shape == x.shape, \
            "Hierarchical reconstruction should match input shape"
            
        # Both code levels should be integers
        assert codes_top.dtype in [torch.long, torch.int64]
        assert codes_bottom.dtype in [torch.long, torch.int64]
        
        # Top level should have lower spatial resolution than bottom level
        assert codes_top.numel() < codes_bottom.numel(), \
            "Top level should have fewer codes than bottom level"


class TestVQGAN:
    """Test VQ-GAN with adversarial training."""
    
    def test_vqgan_generator_mode(self):
        """Test VQ-GAN in generator mode."""
        model = VQGAN(in_channels=3, embedding_dim=256, num_embeddings=1024)
        
        x = torch.randn(2, 3, 64, 64)
        outputs = model(x, mode='generator')
        
        # Should have VQ-VAE outputs plus discriminator predictions
        expected_keys = ['reconstruction', 'vq_loss', 'commitment_loss']
        for key in expected_keys:
            assert key in outputs, f"Missing generator output: {key}"
            
    def test_vqgan_discriminator_mode(self):
        """Test VQ-GAN discriminator."""
        model = VQGAN(in_channels=3, embedding_dim=256, num_embeddings=1024)
        
        real = torch.randn(2, 3, 64, 64)
        fake = torch.randn(2, 3, 64, 64)
        
        # Test discriminator on real and fake
        real_pred = model.discriminator(real)
        fake_pred = model.discriminator(fake)
        
        # Discriminator should output predictions
        assert real_pred.dim() >= 1, "Discriminator should output predictions"
        assert fake_pred.shape == real_pred.shape, \
            "Real and fake predictions should have same shape"
            
    def test_vqgan_gan_loss_computation(self):
        """Test GAN loss computation."""
        model = VQGAN(in_channels=3, embedding_dim=128, num_embeddings=512)
        
        real = torch.randn(2, 3, 32, 32)
        fake = torch.randn(2, 3, 32, 32)
        
        gan_losses = model.compute_gan_loss(real, fake)
        
        expected_keys = ['d_loss', 'g_loss']
        for key in expected_keys:
            assert key in gan_losses, f"Missing GAN loss: {key}"
            
        # Losses should be scalars
        assert gan_losses['d_loss'].dim() == 0, "Discriminator loss should be scalar"
        assert gan_losses['g_loss'].dim() == 0, "Generator loss should be scalar"


class TestResidualVectorQuantizer:
    """Test Residual Vector Quantization."""
    
    def test_residual_vq_forward(self):
        """Test residual VQ forward pass."""
        num_stages = 4
        quantizer = ResidualVectorQuantizer(num_stages=num_stages, 
                                          num_embeddings=256, embedding_dim=64)
        
        inputs = torch.randn(4, 8, 8, 64)
        outputs = quantizer(inputs)
        
        # Should have outputs for all stages
        expected_keys = ['quantized', 'vq_loss', 'commitment_loss', 'stage_codes']
        for key in expected_keys:
            assert key in outputs, f"Missing RVQ output: {key}"
            
        # Should have codes for each stage
        stage_codes = outputs['stage_codes']
        assert len(stage_codes) == num_stages, \
            f"Expected {num_stages} stage codes, got {len(stage_codes)}"
            
    def test_residual_encoding(self):
        """Test residual encoding to multiple codes."""
        quantizer = ResidualVectorQuantizer(num_stages=3, num_embeddings=128)
        
        inputs = torch.randn(2, 16, 16, 64)
        codes_list = quantizer.encode_residual(inputs)
        
        assert len(codes_list) == 3, "Should get codes for each residual stage"
        
        for codes in codes_list:
            assert codes.dtype in [torch.long, torch.int64], \
                "Residual codes should be integers"


class TestFiniteScalarQuantizer:
    """Test Finite Scalar Quantization."""
    
    def test_fsq_forward(self):
        """Test FSQ forward pass."""
        levels = [8, 5, 5, 5]  # Different levels per dimension
        quantizer = FiniteScalarQuantizer(levels)
        
        inputs = torch.randn(4, 8, 8, 4)  # Last dim matches len(levels)
        outputs = quantizer(inputs)
        
        # Check outputs
        assert 'quantized' in outputs, "Missing quantized output"
        assert outputs['quantized'].shape == inputs.shape, \
            "FSQ output should match input shape"
            
    def test_fsq_no_learnable_parameters(self):
        """Test that FSQ has no learnable parameters."""
        levels = [7, 7, 7, 7]
        quantizer = FiniteScalarQuantizer(levels)
        
        # Should have no parameters to optimize
        num_params = sum(p.numel() for p in quantizer.parameters())
        assert num_params == 0, f"FSQ should have no parameters, found {num_params}"
        
    def test_fsq_quantization_levels(self):
        """Test that FSQ respects quantization levels."""
        levels = [3, 5, 3]  # Simple levels for testing
        quantizer = FiniteScalarQuantizer(levels)
        
        # Create test input
        inputs = torch.randn(2, 4, 4, 3)
        outputs = quantizer(inputs)
        
        quantized = outputs['quantized']
        
        # Each dimension should be quantized to appropriate number of levels
        # (This test depends on implementation details)
        assert not torch.allclose(inputs, quantized), \
            "FSQ should actually quantize the input"


class TestVQVAELoss:
    """Test VQ-VAE loss computation."""
    
    def test_loss_components(self):
        """Test individual loss components."""
        loss_fn = VQVAELoss(reconstruction_weight=1.0, vq_weight=1.0, 
                           commitment_weight=0.25)
        
        # Mock model outputs
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)
        model_outputs = {
            'reconstruction': x + torch.randn_like(x) * 0.1,
            'vq_loss': torch.tensor(0.5),
            'commitment_loss': torch.tensor(0.1)
        }
        
        losses = loss_fn(model_outputs, x)
        
        # Check loss components
        expected_keys = ['total_loss', 'reconstruction_loss', 'vq_loss', 'commitment_loss']
        for key in expected_keys:
            assert key in losses, f"Missing loss component: {key}"
            
        # All losses should be non-negative
        for key, value in losses.items():
            assert value >= 0, f"{key} should be non-negative, got {value}"
            
    def test_perceptual_loss(self):
        """Test perceptual loss computation."""
        loss_fn = VQVAELoss(perceptual_weight=1.0)
        
        x_recon = torch.randn(2, 3, 64, 64)
        x_target = torch.randn(2, 3, 64, 64)
        
        perceptual_loss = loss_fn.perceptual_loss(x_recon, x_target)
        
        assert perceptual_loss >= 0, "Perceptual loss should be non-negative"
        assert not torch.isnan(perceptual_loss), "Perceptual loss should not be NaN"


class TestVQVAEEvaluation:
    """Test VQ-VAE evaluation metrics."""
    
    def test_codebook_utilization_metrics(self):
        """Test codebook utilization analysis."""
        model = VQVAE(in_channels=3, num_embeddings=256)
        
        # Create simple data loader
        data = [torch.randn(8, 3, 32, 32) for _ in range(4)]
        
        utilization = VQVAEEvaluation.codebook_utilization(model, data)
        
        expected_keys = ['perplexity', 'active_codes', 'entropy']
        for key in expected_keys:
            assert key in utilization, f"Missing utilization metric: {key}"
            
        # Perplexity should be in reasonable range
        perplexity = utilization['perplexity']
        assert 1 <= perplexity <= 256, f"Perplexity {perplexity} out of range"
        
    def test_reconstruction_quality_metrics(self):
        """Test reconstruction quality evaluation."""
        model = VQVAE(in_channels=3, num_embeddings=128)
        
        # Mock data
        data = [torch.randn(4, 3, 32, 32) for _ in range(3)]
        
        quality_metrics = VQVAEEvaluation.reconstruction_quality(model, data)
        
        expected_keys = ['mse', 'psnr', 'ssim']
        for key in expected_keys:
            assert key in quality_metrics, f"Missing quality metric: {key}"
            
        # PSNR should be reasonable (> 0)
        assert quality_metrics['psnr'] > 0, "PSNR should be positive"
        
    def test_latent_space_analysis(self):
        """Test latent space analysis."""
        model = VQVAE(in_channels=3, num_embeddings=64)
        
        data = [torch.randn(4, 3, 16, 16) for _ in range(2)]
        
        latent_analysis = VQVAEEvaluation.latent_space_analysis(model, data)
        
        expected_keys = ['code_frequencies', 'nearest_neighbors']
        for key in expected_keys:
            assert key in latent_analysis, f"Missing latent analysis: {key}"


class TestQuantizationProperties:
    """Test mathematical properties of quantization."""
    
    def test_quantization_idempotency(self):
        """Test that quantizing already quantized data doesn't change it."""
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16)
        
        inputs = torch.randn(4, 8, 8, 16)
        
        # First quantization
        outputs1 = quantizer(inputs)
        quantized1 = outputs1['quantized']
        
        # Second quantization of already quantized data
        outputs2 = quantizer(quantized1)
        quantized2 = outputs2['quantized']
        
        # Should be identical
        assert torch.allclose(quantized1, quantized2, atol=1e-6), \
            "Quantizing quantized data should be idempotent"
            
    def test_quantization_nearest_neighbor(self):
        """Test that quantization selects nearest codebook entry."""
        num_embeddings, embedding_dim = 16, 8
        quantizer = VectorQuantizer(num_embeddings, embedding_dim, use_ema=False)
        
        # Get codebook embeddings
        codebook = quantizer.embedding.weight.data
        
        # Create input that is very close to one specific embedding
        target_embedding = codebook[5:6]  # Select 6th embedding
        noise = torch.randn_like(target_embedding) * 0.01  # Small noise
        inputs = (target_embedding + noise).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 8]
        
        outputs = quantizer(inputs)
        indices = outputs['encoding_indices']
        
        # Should select the target embedding
        assert indices.item() == 5, \
            f"Should select nearest embedding (5), got {indices.item()}"
            
    def test_commitment_loss_behavior(self):
        """Test commitment loss encourages encoder to stay close to codebook."""
        quantizer = VectorQuantizer(num_embeddings=32, embedding_dim=16, 
                                  commitment_cost=1.0, use_ema=False)
        
        # Input far from any codebook entry
        inputs = torch.ones(2, 4, 4, 16) * 10.0  # Large values
        outputs = quantizer(inputs)
        
        # Commitment loss should be large when inputs are far from codebook
        commitment_loss = outputs['commitment_loss']
        assert commitment_loss > 1.0, \
            f"Commitment loss should be large for distant inputs, got {commitment_loss}"


def run_vqvae_tests():
    """Run all VQ-VAE tests."""
    print("Running VQ-VAE Test Suite...")
    print("=" * 50)
    
    test_classes = [
        TestVectorQuantizer,
        TestVQVAE,
        TestVQVAEHierarchical,
        TestVQGAN,
        TestResidualVectorQuantizer,
        TestFiniteScalarQuantizer,
        TestVQVAELoss,
        TestVQVAEEvaluation,
        TestQuantizationProperties
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                instance = test_class()
                if hasattr(instance, 'setup_method'):
                    instance.setup_method()
                    
                test_method = getattr(instance, method_name)
                test_method()
                
                print(f"  ✓ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {total_tests - passed_tests} tests failed")
        
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_vqvae_tests()
    exit(0 if success else 1)