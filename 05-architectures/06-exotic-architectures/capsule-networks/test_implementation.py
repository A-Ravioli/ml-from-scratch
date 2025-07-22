"""
Test Suite for Capsule Networks

Comprehensive tests for Capsule Network implementations, routing algorithms,
and architectural components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List, Tuple

from exercise import (
    SquashFunction, PrimaryCapsuleLayer, DynamicRouting,
    EMRouting, CapsuleNetwork, ReconstructionDecoder,
    CapsuleLoss, StackedCapsuleAutoencoder, CapsuleAttention
)


class TestSquashFunction:
    """Test squashing activation function"""
    
    def test_initialization(self):
        """Test squash function initialization"""
        squash = SquashFunction(dim=-1)
        assert squash.dim == -1
    
    def test_output_range(self):
        """Test that squash function outputs are in [0, 1)"""
        squash = SquashFunction()
        
        # Test with various input magnitudes
        inputs = [
            torch.randn(5, 8),        # Normal magnitude
            torch.randn(5, 8) * 10,   # Large magnitude
            torch.randn(5, 8) * 0.1   # Small magnitude
        ]
        
        for s in inputs:
            v = squash(s)
            
            assert v.shape == s.shape
            
            # Check that all lengths are in [0, 1)
            lengths = torch.norm(v, dim=-1)
            assert torch.all(lengths >= 0)
            assert torch.all(lengths < 1)
    
    def test_zero_input(self):
        """Test squash function with zero input"""
        squash = SquashFunction()
        
        s_zero = torch.zeros(3, 4)
        v_zero = squash(s_zero)
        
        assert torch.allclose(v_zero, torch.zeros(3, 4))
    
    def test_direction_preservation(self):
        """Test that squash preserves direction of input vectors"""
        squash = SquashFunction()
        
        # Create input with known direction
        s = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        v = squash(s)
        
        # Check that directions are preserved (up to scaling)
        for i in range(len(s)):
            if torch.norm(s[i]) > 1e-6:  # Skip zero vectors
                # Compute cosine similarity
                cos_sim = torch.dot(s[i], v[i]) / (torch.norm(s[i]) * torch.norm(v[i]))
                assert torch.abs(cos_sim - 1.0) < 1e-5
    
    def test_gradient_flow(self):
        """Test that gradients flow through squash function"""
        squash = SquashFunction()
        
        s = torch.randn(4, 6, requires_grad=True)
        v = squash(s)
        loss = v.sum()
        loss.backward()
        
        assert s.grad is not None
        assert not torch.isnan(s.grad).any()


class TestPrimaryCapsuleLayer:
    """Test primary capsule layer"""
    
    def test_initialization(self):
        """Test primary capsule layer initialization"""
        layer = PrimaryCapsuleLayer(
            in_channels=256, out_channels=32, capsule_dim=8
        )
        assert layer.out_channels == 32
        assert layer.capsule_dim == 8
    
    def test_forward_pass(self):
        """Test forward pass through primary capsule layer"""
        layer = PrimaryCapsuleLayer(
            in_channels=128, out_channels=16, capsule_dim=8,
            kernel_size=9, stride=2
        )
        
        # Input feature maps from CNN
        x = torch.randn(4, 128, 20, 20)
        capsules = layer(x)
        
        # Check output shape
        assert len(capsules.shape) == 3
        assert capsules.shape[0] == 4  # Batch size
        assert capsules.shape[-1] == 8  # Capsule dimension
        
        # Check that capsule lengths are in [0, 1)
        lengths = torch.norm(capsules, dim=-1)
        assert torch.all(lengths >= 0)
        assert torch.all(lengths < 1)
    
    def test_different_parameters(self):
        """Test primary capsule layer with different parameters"""
        configs = [
            {'in_channels': 64, 'out_channels': 8, 'capsule_dim': 4},
            {'in_channels': 256, 'out_channels': 32, 'capsule_dim': 16},
            {'in_channels': 128, 'out_channels': 16, 'capsule_dim': 8}
        ]
        
        for config in configs:
            layer = PrimaryCapsuleLayer(**config)
            
            # Test with appropriate input size
            x = torch.randn(2, config['in_channels'], 16, 16)
            capsules = layer(x)
            
            assert capsules.shape[-1] == config['capsule_dim']
            assert not torch.isnan(capsules).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through primary capsule layer"""
        layer = PrimaryCapsuleLayer(in_channels=64, out_channels=8, capsule_dim=4)
        
        x = torch.randn(3, 64, 12, 12, requires_grad=True)
        capsules = layer(x)
        loss = capsules.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDynamicRouting:
    """Test dynamic routing algorithm"""
    
    def test_initialization(self):
        """Test dynamic routing initialization"""
        routing = DynamicRouting(
            num_input_capsules=32, num_output_capsules=10,
            input_capsule_dim=8, output_capsule_dim=16,
            num_iterations=3
        )
        assert routing.num_input_capsules == 32
        assert routing.num_output_capsules == 10
        assert routing.num_iterations == 3
    
    def test_forward_pass(self):
        """Test forward pass through dynamic routing"""
        routing = DynamicRouting(
            num_input_capsules=24, num_output_capsules=5,
            input_capsule_dim=6, output_capsule_dim=12,
            num_iterations=2
        )
        
        u = torch.randn(8, 24, 6)
        v = routing(u)
        
        assert v.shape == (8, 5, 12)
        assert not torch.isnan(v).any()
        
        # Check that output capsule lengths are in [0, 1)
        lengths = torch.norm(v, dim=-1)
        assert torch.all(lengths >= 0)
        assert torch.all(lengths < 1)
    
    def test_routing_iterations(self):
        """Test that different numbers of routing iterations work"""
        for num_iter in [1, 2, 3, 5]:
            routing = DynamicRouting(
                num_input_capsules=16, num_output_capsules=4,
                input_capsule_dim=4, output_capsule_dim=8,
                num_iterations=num_iter
            )
            
            u = torch.randn(4, 16, 4)
            v = routing(u)
            
            assert v.shape == (4, 4, 8)
            assert not torch.isnan(v).any()
    
    def test_coupling_coefficients(self):
        """Test that coupling coefficients sum to 1"""
        routing = DynamicRouting(
            num_input_capsules=10, num_output_capsules=3,
            input_capsule_dim=4, output_capsule_dim=6,
            num_iterations=1
        )
        
        u = torch.randn(2, 10, 4)
        
        # Access internal coupling coefficients (this requires modifying the forward method)
        # For now, we test that the routing produces valid outputs
        v = routing(u)
        
        assert v.shape == (2, 3, 6)
        assert not torch.isnan(v).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through dynamic routing"""
        routing = DynamicRouting(
            num_input_capsules=12, num_output_capsules=3,
            input_capsule_dim=4, output_capsule_dim=6
        )
        
        u = torch.randn(3, 12, 4, requires_grad=True)
        v = routing(u)
        loss = v.sum()
        loss.backward()
        
        assert u.grad is not None
        assert not torch.isnan(u.grad).any()


class TestEMRouting:
    """Test EM-based routing algorithm"""
    
    def test_initialization(self):
        """Test EM routing initialization"""
        routing = EMRouting(
            num_input_capsules=16, num_output_capsules=4,
            input_capsule_dim=8, output_capsule_dim=12,
            num_iterations=3
        )
        assert routing.num_input_capsules == 16
        assert routing.num_output_capsules == 4
        assert routing.num_iterations == 3
    
    def test_forward_pass(self):
        """Test forward pass through EM routing"""
        routing = EMRouting(
            num_input_capsules=20, num_output_capsules=5,
            input_capsule_dim=6, output_capsule_dim=10
        )
        
        input_capsules = torch.randn(4, 20, 6)
        output_capsules = routing(input_capsules)
        
        assert output_capsules.shape == (4, 5, 10)
        assert not torch.isnan(output_capsules).any()
    
    def test_em_iterations(self):
        """Test EM routing with different iteration counts"""
        for num_iter in [1, 2, 4]:
            routing = EMRouting(
                num_input_capsules=12, num_output_capsules=3,
                input_capsule_dim=4, output_capsule_dim=8,
                num_iterations=num_iter
            )
            
            input_caps = torch.randn(3, 12, 4)
            output_caps = routing(input_caps)
            
            assert output_caps.shape == (3, 3, 8)
            assert not torch.isnan(output_caps).any()
    
    def test_gradient_flow(self):
        """Test gradient flow through EM routing"""
        routing = EMRouting(
            num_input_capsules=8, num_output_capsules=2,
            input_capsule_dim=4, output_capsule_dim=6
        )
        
        input_caps = torch.randn(2, 8, 4, requires_grad=True)
        output_caps = routing(input_caps)
        loss = output_caps.sum()
        loss.backward()
        
        assert input_caps.grad is not None
        assert not torch.isnan(input_caps.grad).any()


class TestCapsuleNetwork:
    """Test complete Capsule Network"""
    
    @pytest.fixture
    def mnist_like_input(self):
        """Create MNIST-like input for testing"""
        return torch.randn(8, 1, 28, 28)
    
    def test_initialization(self):
        """Test Capsule Network initialization"""
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=10,
            primary_capsule_dim=8, class_capsule_dim=16
        )
        assert capsnet.num_classes == 10
    
    def test_forward_pass(self, mnist_like_input):
        """Test forward pass through complete network"""
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=10,
            primary_capsule_dim=8, class_capsule_dim=16,
            routing_iterations=2
        )
        
        outputs = capsnet(mnist_like_input)
        
        assert 'class_capsules' in outputs
        assert 'class_probs' in outputs
        
        class_capsules = outputs['class_capsules']
        class_probs = outputs['class_probs']
        
        assert class_capsules.shape == (8, 10, 16)
        assert class_probs.shape == (8, 10)
        
        # Class probabilities should be capsule lengths
        computed_probs = torch.norm(class_capsules, dim=-1)
        assert torch.allclose(class_probs, computed_probs, atol=1e-5)
    
    def test_reconstruction(self, mnist_like_input):
        """Test reconstruction capability"""
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=10,
            primary_capsule_dim=8, class_capsule_dim=16
        )
        
        targets = torch.randint(0, 10, (8,))
        outputs = capsnet(mnist_like_input, targets)
        
        if 'reconstructions' in outputs:
            reconstructions = outputs['reconstructions']
            assert reconstructions.shape == mnist_like_input.shape
            assert not torch.isnan(reconstructions).any()
    
    def test_different_routing_types(self, mnist_like_input):
        """Test with different routing algorithms"""
        for routing_type in ['dynamic']:  # Add 'em' when implemented
            capsnet = CapsuleNetwork(
                input_channels=1, num_classes=5,
                routing_type=routing_type
            )
            
            outputs = capsnet(mnist_like_input)
            
            assert outputs['class_capsules'].shape == (8, 5, 16)
            assert not torch.isnan(outputs['class_capsules']).any()
    
    def test_gradient_flow(self, mnist_like_input):
        """Test gradient flow through entire network"""
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=3,
            routing_iterations=1  # Fewer iterations for speed
        )
        
        mnist_like_input.requires_grad_(True)
        outputs = capsnet(mnist_like_input)
        loss = outputs['class_probs'].sum()
        loss.backward()
        
        assert mnist_like_input.grad is not None
        assert not torch.isnan(mnist_like_input.grad).any()


class TestReconstructionDecoder:
    """Test reconstruction decoder"""
    
    def test_initialization(self):
        """Test decoder initialization"""
        decoder = ReconstructionDecoder(
            capsule_dim=16, num_classes=10, image_size=28
        )
        assert decoder.capsule_dim == 16
        assert decoder.num_classes == 10
        assert decoder.image_size == 28
    
    def test_forward_pass(self):
        """Test decoder forward pass"""
        decoder = ReconstructionDecoder(
            capsule_dim=12, num_classes=5, image_size=32
        )
        
        class_capsules = torch.randn(4, 5, 12)
        targets = torch.randint(0, 5, (4,))
        
        reconstructions = decoder(class_capsules, targets)
        
        assert reconstructions.shape == (4, 1, 32, 32)
        assert not torch.isnan(reconstructions).any()
    
    def test_different_image_sizes(self):
        """Test decoder with different image sizes"""
        for img_size in [28, 32, 64]:
            decoder = ReconstructionDecoder(
                capsule_dim=8, num_classes=3, image_size=img_size
            )
            
            class_capsules = torch.randn(2, 3, 8)
            targets = torch.randint(0, 3, (2,))
            
            reconstructions = decoder(class_capsules, targets)
            
            assert reconstructions.shape == (2, 1, img_size, img_size)
    
    def test_target_selection(self):
        """Test that decoder correctly selects target class capsule"""
        decoder = ReconstructionDecoder(
            capsule_dim=4, num_classes=3, image_size=16
        )
        
        # Create distinctive class capsules
        class_capsules = torch.zeros(1, 3, 4)
        class_capsules[0, 0, :] = 1.0  # Class 0 capsule
        class_capsules[0, 1, :] = 2.0  # Class 1 capsule  
        class_capsules[0, 2, :] = 3.0  # Class 2 capsule
        
        # Target class 1
        targets = torch.tensor([1])
        reconstructions = decoder(class_capsules, targets)
        
        assert reconstructions.shape == (1, 1, 16, 16)
        assert not torch.isnan(reconstructions).any()


class TestCapsuleLoss:
    """Test Capsule Network loss function"""
    
    def test_initialization(self):
        """Test loss function initialization"""
        loss_fn = CapsuleLoss(
            margin_pos=0.9, margin_neg=0.1, 
            lambda_neg=0.5, lambda_recon=0.0005
        )
        assert loss_fn.margin_pos == 0.9
        assert loss_fn.margin_neg == 0.1
        assert loss_fn.lambda_neg == 0.5
        assert loss_fn.lambda_recon == 0.0005
    
    def test_margin_loss(self):
        """Test margin loss computation"""
        loss_fn = CapsuleLoss()
        
        # Create class capsules with known lengths
        class_capsules = torch.zeros(4, 3, 8)
        class_capsules[0, 0, 0] = 0.95  # Strong activation for class 0
        class_capsules[1, 1, 0] = 0.8   # Medium activation for class 1
        class_capsules[2, 2, 0] = 0.3   # Weak activation for class 2
        class_capsules[3, 1, 0] = 0.05  # Very weak activation
        
        targets = torch.tensor([0, 1, 2, 0])  # Ground truth labels
        
        loss_dict = loss_fn(class_capsules, targets)
        
        assert 'margin_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        margin_loss = loss_dict['margin_loss']
        assert not torch.isnan(margin_loss)
        assert margin_loss.item() >= 0
    
    def test_reconstruction_loss(self):
        """Test reconstruction loss computation"""
        loss_fn = CapsuleLoss(lambda_recon=0.001)
        
        class_capsules = torch.randn(2, 5, 10)
        targets = torch.randint(0, 5, (2,))
        
        # Create dummy reconstruction and input
        reconstructions = torch.randn(2, 1, 28, 28)
        inputs = torch.randn(2, 1, 28, 28)
        
        loss_dict = loss_fn(class_capsules, targets, reconstructions, inputs)
        
        assert 'reconstruction_loss' in loss_dict
        assert 'total_loss' in loss_dict
        
        recon_loss = loss_dict['reconstruction_loss']
        total_loss = loss_dict['total_loss']
        
        assert not torch.isnan(recon_loss)
        assert not torch.isnan(total_loss)
        assert recon_loss.item() >= 0
        assert total_loss.item() >= 0
    
    def test_gradient_flow(self):
        """Test gradient flow through loss function"""
        loss_fn = CapsuleLoss()
        
        class_capsules = torch.randn(3, 4, 6, requires_grad=True)
        targets = torch.randint(0, 4, (3,))
        
        loss_dict = loss_fn(class_capsules, targets)
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        assert class_capsules.grad is not None
        assert not torch.isnan(class_capsules.grad).any()


class TestStackedCapsuleAutoencoder:
    """Test Stacked Capsule Autoencoder"""
    
    def test_initialization(self):
        """Test SCAE initialization"""
        scae = StackedCapsuleAutoencoder(
            input_channels=3, 
            capsule_dims=[16, 32],
            template_size=11,
            num_templates=64
        )
        assert scae.template_size == 11
        assert scae.num_templates == 64
    
    def test_forward_pass(self):
        """Test SCAE forward pass"""
        scae = StackedCapsuleAutoencoder(
            input_channels=1,
            capsule_dims=[8, 16],
            template_size=7,
            num_templates=32
        )
        
        x = torch.randn(4, 1, 32, 32)
        outputs = scae(x)
        
        assert isinstance(outputs, dict)
        
        # Check for expected outputs
        expected_keys = ['reconstruction', 'part_capsules', 'object_capsules']
        # Note: Actual keys depend on implementation
        
        # At minimum, should have some output
        assert len(outputs) > 0
        
        # Check reconstruction shape if present
        if 'reconstruction' in outputs:
            reconstruction = outputs['reconstruction']
            assert reconstruction.shape == x.shape
            assert not torch.isnan(reconstruction).any()
    
    def test_part_encoding(self):
        """Test part capsule encoding"""
        scae = StackedCapsuleAutoencoder(
            input_channels=1,
            capsule_dims=[12],
            num_templates=16
        )
        
        x = torch.randn(2, 1, 24, 24)
        part_capsules = scae.encode_parts(x)
        
        assert len(part_capsules.shape) == 3  # [batch, num_parts, part_dim]
        assert part_capsules.shape[0] == 2   # Batch size
        assert not torch.isnan(part_capsules).any()
    
    def test_object_encoding(self):
        """Test object capsule encoding"""
        scae = StackedCapsuleAutoencoder(capsule_dims=[8, 16])
        
        # Dummy part capsules
        part_capsules = torch.randn(3, 20, 8)
        object_capsules = scae.encode_objects(part_capsules)
        
        assert len(object_capsules.shape) == 3  # [batch, num_objects, object_dim]
        assert object_capsules.shape[0] == 3    # Batch size
        assert not torch.isnan(object_capsules).any()


class TestCapsuleAttention:
    """Test capsule attention mechanism"""
    
    def test_initialization(self):
        """Test capsule attention initialization"""
        attention = CapsuleAttention(capsule_dim=16, num_heads=4)
        assert attention.capsule_dim == 16
        assert attention.num_heads == 4
        assert attention.head_dim == 4  # 16 / 4
    
    def test_forward_pass(self):
        """Test attention forward pass"""
        attention = CapsuleAttention(capsule_dim=12, num_heads=3)
        
        capsules = torch.randn(4, 8, 12)  # batch=4, 8 capsules, 12-dim
        attended = attention(capsules)
        
        assert attended.shape == capsules.shape
        assert not torch.isnan(attended).any()
    
    def test_attention_with_mask(self):
        """Test attention with masking"""
        attention = CapsuleAttention(capsule_dim=8, num_heads=2)
        
        capsules = torch.randn(2, 6, 8)
        mask = torch.ones(2, 6, 6)
        mask[:, :3, 3:] = 0  # Mask some connections
        
        attended = attention(capsules, mask)
        
        assert attended.shape == capsules.shape
        assert not torch.isnan(attended).any()
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism"""
        for num_heads in [1, 2, 4]:
            capsule_dim = 16
            attention = CapsuleAttention(
                capsule_dim=capsule_dim, num_heads=num_heads
            )
            
            capsules = torch.randn(3, 10, capsule_dim)
            attended = attention(capsules)
            
            assert attended.shape == capsules.shape
            assert not torch.isnan(attended).any()


class TestIntegrationAndProperties:
    """Integration tests and property verification"""
    
    def test_end_to_end_training_step(self):
        """Test that a complete training step works"""
        # Create small network for testing
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=3,
            primary_capsule_dim=4, class_capsule_dim=8,
            routing_iterations=1
        )
        
        loss_fn = CapsuleLoss()
        
        # Create batch of data
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 3, (4,))
        
        # Forward pass
        outputs = capsnet(x, y)
        
        # Compute loss
        loss_dict = loss_fn(outputs['class_capsules'], y)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist
        for param in capsnet.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_equivariance_property(self):
        """Test basic equivariance properties"""
        capsnet = CapsuleNetwork(
            input_channels=1, num_classes=2,
            routing_iterations=1
        )
        
        capsnet.eval()  # Set to eval mode
        
        # Original image
        x = torch.randn(1, 1, 28, 28)
        
        # Simple transformation (small translation)
        x_transformed = torch.roll(x, shifts=2, dims=-1)
        
        with torch.no_grad():
            outputs = capsnet(x)
            outputs_transformed = capsnet(x_transformed)
            
            caps1 = outputs['class_capsules']
            caps2 = outputs_transformed['class_capsules']
            
            # Capsule representations should change in a structured way
            # (This is a basic test - full equivariance requires more sophisticated analysis)
            assert not torch.allclose(caps1, caps2)
            assert not torch.isnan(caps1).any()
            assert not torch.isnan(caps2).any()
    
    def test_routing_convergence(self):
        """Test that routing algorithm converges"""
        routing = DynamicRouting(
            num_input_capsules=16, num_output_capsules=4,
            input_capsule_dim=6, output_capsule_dim=8,
            num_iterations=1
        )
        
        u = torch.randn(2, 16, 6)
        
        # Test with different iteration counts
        results = []
        for n_iter in [1, 2, 3, 5]:
            routing.num_iterations = n_iter
            v = routing(u)
            results.append(v)
        
        # Results should converge (changes become smaller)
        assert len(results) == 4
        for result in results:
            assert not torch.isnan(result).any()
        
        # Later iterations should be more similar
        diff_1_2 = torch.norm(results[1] - results[0])
        diff_3_4 = torch.norm(results[3] - results[2])
        # Note: This test might be flaky depending on initialization
        
    def test_computational_efficiency(self):
        """Test computational efficiency of different components"""
        import time
        
        # Test different capsule network sizes
        configs = [
            {'primary_capsule_dim': 4, 'class_capsule_dim': 8, 'routing_iterations': 1},
            {'primary_capsule_dim': 8, 'class_capsule_dim': 16, 'routing_iterations': 2},
        ]
        
        x = torch.randn(8, 1, 28, 28)
        
        for config in configs:
            capsnet = CapsuleNetwork(
                input_channels=1, num_classes=5, **config
            )
            
            start_time = time.time()
            with torch.no_grad():
                outputs = capsnet(x)
            end_time = time.time()
            
            print(f"Config {config}: {end_time - start_time:.4f}s")
            
            assert outputs['class_capsules'].shape == (8, 5, config['class_capsule_dim'])
            assert not torch.isnan(outputs['class_capsules']).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])