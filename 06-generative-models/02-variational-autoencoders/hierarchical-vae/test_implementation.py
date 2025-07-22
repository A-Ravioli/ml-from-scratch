"""
Test Suite for Hierarchical VAE Implementation

This test suite verifies correctness of hierarchical VAE implementations.
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
    HierarchicalVAE, LadderBlock, LadderVAE, ResidualBlock, 
    VeryDeepVAELevel, VeryDeepVAE, MultiScaleImageVAE,
    HierarchicalLoss, HierarchicalTrainer, HierarchicalEvaluation,
    visualize_hierarchical_representations, compare_hierarchical_architectures
)


class TestHierarchicalStructure:
    """Test basic hierarchical VAE structure and constraints."""
    
    def test_hierarchy_levels_consistency(self):
        """Test that hierarchy levels are consistent across architecture."""
        # Test parameters
        input_dim = 784
        latent_dims = [64, 32, 16]  # 3 levels
        hidden_dims = [256, 128, 64]
        
        model = LadderVAE(input_dim, latent_dims, hidden_dims)
        
        # Check num_levels consistency
        assert model.num_levels == len(latent_dims), \
            f"Expected {len(latent_dims)} levels, got {model.num_levels}"
            
        # Check that model has appropriate number of components
        # This would test that ladder blocks match the hierarchy depth
        
    def test_hierarchical_forward_pass_shapes(self):
        """Test that forward pass produces correct shapes for all levels."""
        batch_size = 4
        input_dim = 784
        latent_dims = [64, 32, 16]
        hidden_dims = [256, 128, 64]
        
        model = LadderVAE(input_dim, latent_dims, hidden_dims)
        x = torch.randn(batch_size, input_dim)
        
        outputs = model(x)
        
        # Check that we get parameters for all levels
        expected_keys = ['reconstruction', 'z_samples', 'z_mu', 'z_logvar']
        for key in expected_keys:
            assert key in outputs, f"Missing output key: {key}"
            
        # Check shapes for hierarchical components
        if 'z_samples' in outputs:
            z_samples = outputs['z_samples']
            assert len(z_samples) == len(latent_dims), \
                f"Expected {len(latent_dims)} latent levels, got {len(z_samples)}"
                
            for i, z in enumerate(z_samples):
                expected_shape = (batch_size, latent_dims[i])
                assert z.shape == expected_shape, \
                    f"Level {i}: expected shape {expected_shape}, got {z.shape}"


class TestLadderVAE:
    """Test Ladder VAE specific functionality."""
    
    def test_ladder_block_bidirectional_flow(self):
        """Test that ladder blocks properly handle bidirectional information flow."""
        input_dim, latent_dim, hidden_dim = 128, 32, 64
        batch_size = 4
        
        ladder_block = LadderBlock(input_dim, latent_dim, hidden_dim)
        
        # Test bottom-up only (top level)
        bottom_up = torch.randn(batch_size, input_dim)
        output_top = ladder_block(bottom_up, None)
        
        assert 'z_sample' in output_top, "Missing z_sample in output"
        assert 'td_features' in output_top, "Missing top-down features"
        
        # Test with both bottom-up and top-down
        top_down = torch.randn(batch_size, hidden_dim)
        output_mid = ladder_block(bottom_up, top_down)
        
        # Top-down information should influence the latent distribution
        assert not torch.allclose(output_top['z_mu'], output_mid['z_mu']), \
            "Top-down information should affect latent distribution"
            
    def test_ladder_skip_connections(self):
        """Test that skip connections work properly in Ladder VAE."""
        # This test would verify that information flows properly
        # between encoder and decoder at each level
        
        input_dim = 784
        latent_dims = [64, 32, 16]
        hidden_dims = [256, 128, 64]
        
        model = LadderVAE(input_dim, latent_dims, hidden_dims)
        
        batch_size = 4
        x = torch.randn(batch_size, input_dim)
        
        # Get encoding
        latent_params = model.encode(x)
        assert len(latent_params) == len(latent_dims), \
            "Encoding should produce parameters for all levels"
            
        # Test reconstruction from latent samples
        z_samples = []
        for params in latent_params:
            if 'mu' in params and 'logvar' in params:
                z = params['mu'] + torch.exp(0.5 * params['logvar']) * torch.randn_like(params['mu'])
                z_samples.append(z)
        
        reconstruction = model.decode(z_samples)
        assert reconstruction.shape == x.shape, \
            f"Reconstruction shape {reconstruction.shape} doesn't match input {x.shape}"


class TestVeryDeepVAE:
    """Test Very Deep VAE architecture."""
    
    def test_residual_blocks(self):
        """Test residual block functionality."""
        channels = 64
        batch_size, height, width = 4, 32, 32
        
        residual_block = ResidualBlock(channels)
        
        x = torch.randn(batch_size, channels, height, width)
        output = residual_block(x)
        
        # Check shape preservation
        assert output.shape == x.shape, \
            f"Residual block should preserve shape, got {output.shape} vs {x.shape}"
            
        # Check that it's not just identity (should learn something)
        assert not torch.allclose(output, x, atol=1e-3), \
            "Residual block should modify input (not pure identity)"
            
    def test_vdvae_depth_scaling(self):
        """Test that VDVAE can handle many levels."""
        input_shape = (3, 64, 64)
        num_levels = 20  # Deep hierarchy
        latent_dims = [64] * num_levels
        
        model = VeryDeepVAE(input_shape, latent_dims)
        
        batch_size = 2  # Small batch for memory
        x = torch.randn(batch_size, *input_shape)
        
        # Test that forward pass works without memory issues
        try:
            outputs = model(x)
            assert 'reconstruction' in outputs, "Missing reconstruction output"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Not enough GPU memory for very deep test")
            else:
                raise e
                
    def test_vdvae_gradient_flow(self):
        """Test that gradients flow through all levels in VDVAE."""
        input_shape = (3, 32, 32)  # Smaller for testing
        latent_dims = [32, 32, 32]  # 3 levels
        
        model = VeryDeepVAE(input_shape, latent_dims)
        
        batch_size = 2
        x = torch.randn(batch_size, *input_shape)
        
        # Forward pass
        outputs = model(x)
        
        # Create dummy loss
        loss = outputs['reconstruction'].sum()
        loss.backward()
        
        # Check that gradients exist for parameters at all levels
        has_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grads.append(name)
                
        assert len(has_grads) > 0, "No parameters received gradients"
        
        # Should have gradients for multiple levels
        level_grads = [name for name in has_grads if 'level' in name.lower()]
        assert len(level_grads) > 1, "Gradients should flow to multiple levels"


class TestMultiScaleVAE:
    """Test Multi-Scale VAE functionality."""
    
    def test_multiscale_encoding(self):
        """Test that multi-scale encoding produces representations at different scales."""
        image_channels = 3
        base_resolution = 8
        
        model = MultiScaleImageVAE(image_channels, base_resolution)
        
        # Test with image that has multiple scales of structure
        batch_size = 2
        image_size = 32  # Should work with different resolutions
        x = torch.randn(batch_size, image_channels, image_size, image_size)
        
        latent_params = model.encode(x)
        
        # Should get parameters for multiple scales
        assert len(latent_params) > 1, "Multi-scale VAE should have multiple levels"
        
        # Different levels should have different dimensionalities
        # (representing different scales of information)
        latent_dims = []
        for params in latent_params:
            if 'mu' in params:
                latent_dims.append(params['mu'].shape[-1])
                
        # Check that we have varying latent dimensions
        assert len(set(latent_dims)) > 1, "Different scales should have different latent dims"
        
    def test_multiscale_generation(self):
        """Test multi-scale generation produces coherent results."""
        model = MultiScaleImageVAE(image_channels=3, base_resolution=8)
        
        batch_size = 2
        image_size = 32
        x = torch.randn(batch_size, 3, image_size, image_size)
        
        outputs = model(x)
        reconstruction = outputs['reconstruction']
        
        # Check reconstruction shape matches input
        assert reconstruction.shape == x.shape, \
            f"Reconstruction shape {reconstruction.shape} doesn't match input {x.shape}"
            
        # Check that reconstruction is reasonable (not all zeros or constant)
        assert reconstruction.std() > 1e-3, "Reconstruction should have reasonable variance"
        assert not torch.isnan(reconstruction).any(), "Reconstruction shouldn't contain NaN"


class TestHierarchicalLoss:
    """Test hierarchical loss computation."""
    
    def test_level_specific_beta_scheduling(self):
        """Test that β parameters can be scheduled per level."""
        num_levels = 3
        loss_fn = HierarchicalLoss(num_levels, beta_schedule='linear')
        
        # Test β scheduling
        initial_betas = [getattr(loss_fn, f'beta_{i}', None) for i in range(num_levels)]
        
        loss_fn.update_beta_schedule(epoch=50, warmup_epochs=100)
        
        updated_betas = [getattr(loss_fn, f'beta_{i}', None) for i in range(num_levels)]
        
        # At least some β values should change during warmup
        # (Implementation dependent, but at least verify the method exists)
        
    def test_free_bits_constraint(self):
        """Test free bits constraint in KL loss."""
        num_levels = 2
        free_bits = 1.0
        loss_fn = HierarchicalLoss(num_levels, free_bits=free_bits)
        
        # Create test latent parameters
        batch_size, latent_dim = 4, 32
        z_mu = torch.randn(batch_size, latent_dim)
        z_logvar = torch.randn(batch_size, latent_dim)
        
        kl_loss = loss_fn.compute_kl_loss(z_mu, z_logvar, level=0)
        
        # KL loss should respect free bits constraint
        assert kl_loss >= 0, "KL loss should be non-negative"
        
        # With free bits, very small KL should be clamped
        very_small_logvar = torch.full_like(z_logvar, -10)  # Very small variance
        very_small_mu = torch.zeros_like(z_mu)  # Zero mean
        
        small_kl = loss_fn.compute_kl_loss(very_small_mu, very_small_logvar, level=0)
        
        if free_bits > 0:
            # Should be at least free_bits per dimension
            expected_min_kl = free_bits * latent_dim
            assert small_kl >= expected_min_kl * 0.9, \
                f"KL {small_kl} should be at least {expected_min_kl} with free bits"
                
    def test_hierarchy_penalty(self):
        """Test hierarchy penalty computation."""
        num_levels = 3
        hierarchy_penalty = 0.1
        loss_fn = HierarchicalLoss(num_levels, hierarchy_penalty=hierarchy_penalty)
        
        batch_size, latent_dim = 4, 32
        
        # Create latent parameters for multiple levels
        latent_params = []
        for level in range(num_levels):
            params = {
                'mu': torch.randn(batch_size, latent_dim),
                'logvar': torch.randn(batch_size, latent_dim)
            }
            latent_params.append(params)
            
        penalty = loss_fn.compute_hierarchy_penalty(latent_params)
        
        assert penalty >= 0, "Hierarchy penalty should be non-negative"
        
        # With identical latent parameters, penalty should be higher
        identical_params = [latent_params[0]] * num_levels
        identical_penalty = loss_fn.compute_hierarchy_penalty(identical_params)
        
        assert identical_penalty >= penalty, \
            "Identical parameters should have higher penalty"


class TestHierarchicalEvaluation:
    """Test hierarchical evaluation metrics."""
    
    def test_hierarchy_interpolation(self):
        """Test hierarchical interpolation functionality."""
        input_dim = 784
        latent_dims = [64, 32, 16]
        hidden_dims = [256, 128, 64]
        
        model = LadderVAE(input_dim, latent_dims, hidden_dims)
        
        # Create two test samples
        x1 = torch.randn(1, input_dim)
        x2 = torch.randn(1, input_dim)
        
        interpolation = HierarchicalEvaluation.visualize_hierarchy_interpolation(
            model, x1, x2, num_steps=5
        )
        
        # Should get interpolation results for each level
        assert interpolation.shape[0] == len(latent_dims), \
            f"Expected {len(latent_dims)} interpolation levels"
            
        # Should have requested number of steps
        assert interpolation.shape[1] == 5, "Should have 5 interpolation steps"
        
    def test_information_flow_analysis(self):
        """Test information flow analysis between levels."""
        # Create simple dataset for testing
        batch_size = 16
        input_dim = 100
        
        model = LadderVAE(input_dim, [32, 16], [64, 32])
        
        # Create test data loader (simplified)
        test_data = torch.randn(batch_size, input_dim)
        data_loader = [(test_data,)]  # Simplified data loader
        
        info_flow = HierarchicalEvaluation.analyze_information_flow(model, data_loader)
        
        # Should return analysis of information between levels
        assert isinstance(info_flow, dict), "Information flow should return dict"
        
        # Should contain metrics about level interactions
        expected_keys = ['mutual_information', 'level_utilization']
        # Note: Actual keys depend on implementation
        
    def test_disentanglement_per_level(self):
        """Test disentanglement computation per hierarchy level."""
        # This would require a dataset with known ground truth factors
        # For now, test that the method exists and returns proper structure
        
        model = LadderVAE(784, [64, 32, 16], [256, 128, 64])
        
        # Mock data with ground truth factors
        batch_size = 32
        data_loader = [(torch.randn(batch_size, 784),)]
        ground_truth = torch.randint(0, 10, (batch_size, 3))  # 3 factors
        
        disentanglement = HierarchicalEvaluation.compute_disentanglement_per_level(
            model, data_loader, ground_truth
        )
        
        assert isinstance(disentanglement, dict), \
            "Disentanglement should return dict with per-level scores"
            
        # Should have scores for each level
        for level in range(model.num_levels):
            assert level in disentanglement, f"Missing disentanglement for level {level}"


class TestTrainingStability:
    """Test training stability of hierarchical VAEs."""
    
    def test_gradient_magnitudes(self):
        """Test that gradients have reasonable magnitudes across levels."""
        model = LadderVAE(784, [64, 32, 16], [256, 128, 64])
        
        batch_size = 4
        x = torch.randn(batch_size, 784)
        
        outputs = model(x)
        
        # Create loss (simplified)
        recon_loss = F.mse_loss(outputs['reconstruction'], x)
        kl_losses = []
        
        if 'z_mu' in outputs and 'z_logvar' in outputs:
            for mu, logvar in zip(outputs['z_mu'], outputs['z_logvar']):
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_losses.append(kl)
        
        total_loss = recon_loss + sum(kl_losses)
        total_loss.backward()
        
        # Check gradient magnitudes
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        assert len(grad_norms) > 0, "Should have gradients"
        
        # Check that gradients aren't too large or too small
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        
        assert max_grad < 100, f"Gradients too large: {max_grad}"
        assert min_grad > 1e-8, f"Gradients too small: {min_grad}"
        
    def test_posterior_collapse_prevention(self):
        """Test mechanisms to prevent posterior collapse."""
        model = LadderVAE(784, [64, 32, 16], [256, 128, 64])
        loss_fn = HierarchicalLoss(3, free_bits=1.0)
        
        batch_size = 8
        x = torch.randn(batch_size, 784)
        
        outputs = model(x)
        
        # Compute losses
        loss_dict = loss_fn(outputs, x)
        
        # Check that KL losses are not zero (indicating active latents)
        if 'kl_losses' in loss_dict:
            for i, kl_loss in enumerate(loss_dict['kl_losses']):
                assert kl_loss.item() > 0.1, f"Level {i} KL loss too small: {kl_loss.item()}"


def run_hierarchical_tests():
    """Run all hierarchical VAE tests."""
    print("Running Hierarchical VAE Test Suite...")
    print("=" * 50)
    
    test_classes = [
        TestHierarchicalStructure,
        TestLadderVAE,
        TestVeryDeepVAE,
        TestMultiScaleVAE,
        TestHierarchicalLoss,
        TestHierarchicalEvaluation,
        TestTrainingStability
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
    success = run_hierarchical_tests()
    exit(0 if success else 1)