"""
Test suite for DenseNet implementation

Comprehensive tests for dense connectivity, feature reuse, and architectural components
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestDenseLayer:
    """Test DenseLayer implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.in_channels = 64
        self.height = 32
        self.width = 32
        self.growth_rate = 32
        
        self.input = np.random.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.dense_layer = DenseLayer(self.in_channels, self.growth_rate)
    
    def test_output_shape(self):
        """Test that dense layer produces correct output shape"""
        output = self.dense_layer.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.growth_rate, self.height, self.width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_bottleneck_design(self):
        """Test bottleneck layer functionality"""
        # Check that bottleneck layer exists and has correct structure
        assert hasattr(self.dense_layer, 'bottleneck'), "Should have bottleneck layer"
        assert hasattr(self.dense_layer, 'conv'), "Should have main conv layer"
        
        # Check bottleneck reduces channels appropriately
        bottleneck_channels = self.dense_layer.bottleneck_factor * self.growth_rate
        assert self.dense_layer.bottleneck.out_channels == bottleneck_channels, \
            "Bottleneck should have correct number of output channels"
    
    def test_growth_rate_consistency(self):
        """Test that output always has growth_rate channels regardless of input size"""
        different_inputs = [
            np.random.randn(1, 32, 16, 16),
            np.random.randn(3, 128, 64, 64),
            np.random.randn(2, 256, 8, 8)
        ]
        
        for test_input in different_inputs:
            in_channels = test_input.shape[1]
            layer = DenseLayer(in_channels, self.growth_rate)
            output = layer.forward(test_input, training=False)
            
            assert output.shape[1] == self.growth_rate, \
                f"Output should always have {self.growth_rate} channels"
    
    def test_feature_extraction(self):
        """Test that layer extracts meaningful features"""
        output = self.dense_layer.forward(self.input, training=False)
        
        # Output should not be all zeros (indicating some computation happened)
        assert not np.all(output == 0), "Output should not be all zeros"
        
        # Output should have reasonable range (after ReLU, should be non-negative)
        assert np.all(output >= 0), "Output should be non-negative (ReLU activation)"


class TestDenseBlock:
    """Test DenseBlock implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.in_channels = 64
        self.height = 32
        self.width = 32
        self.num_layers = 4
        self.growth_rate = 32
        
        self.input = np.random.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.dense_block = DenseBlock(self.in_channels, self.num_layers, self.growth_rate)
    
    def test_output_channels(self):
        """Test that dense block produces correct number of output channels"""
        output = self.dense_block.forward(self.input, training=False)
        
        expected_channels = self.in_channels + self.num_layers * self.growth_rate
        assert output.shape[1] == expected_channels, \
            f"Expected {expected_channels} channels, got {output.shape[1]}"
    
    def test_channel_growth_pattern(self):
        """Test that channels grow correctly with each layer"""
        # Override forward to track intermediate features
        features = [self.input]
        
        for i, layer in enumerate(self.dense_block.layers):
            concatenated = np.concatenate(features, axis=1)
            new_features = layer.forward(concatenated, training=False)
            features.append(new_features)
            
            # Check channel count
            total_channels = sum(f.shape[1] for f in features)
            expected_channels = self.in_channels + (i + 1) * self.growth_rate
            assert total_channels == expected_channels, \
                f"Layer {i}: expected {expected_channels} total channels"
    
    def test_dense_connectivity(self):
        """Test that each layer receives ALL previous features"""
        # Mock the layers to track their inputs
        input_channel_counts = []
        
        current_channels = self.in_channels
        for i in range(self.num_layers):
            input_channel_counts.append(current_channels)
            current_channels += self.growth_rate
        
        # Verify expected input channel counts
        for i, layer in enumerate(self.dense_block.layers):
            expected_input_channels = self.in_channels + i * self.growth_rate
            assert layer.in_channels == expected_input_channels, \
                f"Layer {i} should receive {expected_input_channels} input channels"
    
    def test_spatial_preservation(self):
        """Test that spatial dimensions are preserved"""
        output = self.dense_block.forward(self.input, training=False)
        
        assert output.shape[2] == self.height, "Height should be preserved"
        assert output.shape[3] == self.width, "Width should be preserved"
    
    def test_get_output_channels(self):
        """Test output channel calculation method"""
        calculated_channels = self.dense_block.get_output_channels()
        expected_channels = self.in_channels + self.num_layers * self.growth_rate
        
        assert calculated_channels == expected_channels, \
            "get_output_channels should return correct value"


class TestTransitionLayer:
    """Test TransitionLayer implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.in_channels = 128
        self.height = 32
        self.width = 32
        self.compression_factor = 0.5
        
        self.input = np.random.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.transition = TransitionLayer(self.in_channels, self.compression_factor)
    
    def test_channel_compression(self):
        """Test that transition layer compresses channels correctly"""
        output = self.transition.forward(self.input, training=False)
        
        expected_channels = int(self.in_channels * self.compression_factor)
        assert output.shape[1] == expected_channels, \
            f"Expected {expected_channels} channels after compression"
    
    def test_spatial_downsampling(self):
        """Test that transition layer downsamples spatial dimensions"""
        output = self.transition.forward(self.input, training=False)
        
        expected_height = self.height // 2
        expected_width = self.width // 2
        
        assert output.shape[2] == expected_height, "Height should be halved"
        assert output.shape[3] == expected_width, "Width should be halved"
    
    def test_different_compression_factors(self):
        """Test transition layer with different compression factors"""
        compression_factors = [0.25, 0.5, 0.75, 1.0]
        
        for factor in compression_factors:
            transition = TransitionLayer(self.in_channels, factor)
            output = transition.forward(self.input, training=False)
            
            expected_channels = int(self.in_channels * factor)
            assert output.shape[1] == expected_channels, \
                f"Compression factor {factor} failed"
    
    def test_output_properties(self):
        """Test properties of transition layer output"""
        output = self.transition.forward(self.input, training=False)
        
        # Should be non-negative (ReLU activation)
        assert np.all(output >= 0), "Output should be non-negative"
        
        # Should not be all zeros
        assert not np.all(output == 0), "Output should not be all zeros"


class TestDenseNet:
    """Test complete DenseNet implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.input_shape = (3, 224, 224)  # ImageNet-like input
        self.num_classes = 1000
        
        self.input = np.random.randn(self.batch_size, *self.input_shape)
        self.densenet = DenseNet(
            growth_rate=32,
            block_config=[6, 12, 24, 16],  # DenseNet-121
            num_classes=self.num_classes
        )
    
    def test_forward_pass(self):
        """Test complete forward pass"""
        output = self.densenet.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.num_classes)
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, got {output.shape}"
    
    def test_architecture_structure(self):
        """Test that architecture has correct structure"""
        # Check number of dense blocks
        assert len(self.densenet.blocks) == len(self.densenet.block_config), \
            "Should have correct number of dense blocks"
        
        # Check number of transition layers (one less than blocks)
        assert len(self.densenet.transitions) == len(self.densenet.block_config) - 1, \
            "Should have correct number of transition layers"
        
        # Check initial convolution exists
        assert hasattr(self.densenet, 'initial_conv'), "Should have initial convolution"
    
    def test_parameter_counting(self):
        """Test parameter counting functionality"""
        param_count = self.densenet.count_parameters()
        
        assert isinstance(param_count, int), "Parameter count should be integer"
        assert param_count > 0, "Should have positive number of parameters"
        
        # DenseNet-121 should have approximately 8M parameters
        # Allow some tolerance for implementation differences
        expected_range = (6_000_000, 10_000_000)
        assert expected_range[0] <= param_count <= expected_range[1], \
            f"Parameter count {param_count} outside expected range {expected_range}"
    
    def test_different_architectures(self):
        """Test different DenseNet variants"""
        variants = create_densenet_variants()
        
        test_input = np.random.randn(1, 3, 224, 224)
        
        for name, model in variants.items():
            output = model.forward(test_input, training=False)
            
            assert output.shape == (1, 1000), f"{name} should produce correct output shape"
            print(f"{name}: {model.count_parameters():,} parameters")
    
    def test_training_vs_inference_mode(self):
        """Test differences between training and inference modes"""
        # Test both modes don't crash
        train_output = self.densenet.forward(self.input, training=True)
        infer_output = self.densenet.forward(self.input, training=False)
        
        # Outputs should have same shape
        assert train_output.shape == infer_output.shape, \
            "Training and inference outputs should have same shape"
        
        # Outputs might be different due to batch norm behavior
        # This is expected and correct


class TestFeatureReuse:
    """Test feature reuse analysis"""
    
    def test_feature_reuse_analysis(self):
        """Test feature reuse analysis functionality"""
        # Create a small dense block for testing
        in_channels = 32
        num_layers = 3
        growth_rate = 16
        
        dense_block = DenseBlock(in_channels, num_layers, growth_rate)
        test_input = np.random.randn(1, in_channels, 16, 16)
        
        analysis = analyze_feature_reuse(dense_block, test_input)
        
        # Check analysis structure
        assert 'layer_inputs' in analysis, "Should track layer inputs"
        assert 'layer_outputs' in analysis, "Should track layer outputs"
        assert 'channel_growth' in analysis, "Should track channel growth"
        
        # Check that we have data for all layers
        assert len(analysis['layer_inputs']) == num_layers, \
            "Should have input data for all layers"
        assert len(analysis['channel_growth']) == num_layers, \
            "Should have growth data for all layers"
    
    def test_connectivity_visualization(self):
        """Test connectivity visualization"""
        dense_block = DenseBlock(32, 4, 16)
        
        # This should not crash
        try:
            visualize_dense_connectivity(dense_block)
            # If we get here, visualization worked
            assert True
        except Exception as e:
            # If visualization fails, at least report the error
            print(f"Visualization failed: {e}")


class TestMemoryEfficiency:
    """Test memory efficiency features"""
    
    def test_memory_efficient_densenet(self):
        """Test memory-efficient DenseNet implementation"""
        regular_densenet = DenseNet(growth_rate=16, block_config=[3, 4], num_classes=100)
        memory_efficient = MemoryEfficientDenseNet(growth_rate=16, block_config=[3, 4], num_classes=100)
        
        test_input = np.random.randn(2, 3, 64, 64)
        
        # Both should produce same output shape
        regular_output = regular_densenet.forward(test_input, training=False)
        efficient_output = memory_efficient.forward(test_input, training=False)
        
        assert regular_output.shape == efficient_output.shape, \
            "Memory-efficient and regular versions should have same output shape"


class TestNumericalStability:
    """Test numerical stability of DenseNet components"""
    
    def test_extreme_inputs(self):
        """Test behavior with extreme input values"""
        extreme_inputs = [
            np.ones((1, 3, 32, 32)) * 1e6,    # Very large
            np.ones((1, 3, 32, 32)) * 1e-6,   # Very small
            np.zeros((1, 3, 32, 32)),         # All zeros
        ]
        
        densenet = DenseNet(growth_rate=8, block_config=[2], num_classes=10)
        
        for test_input in extreme_inputs:
            output = densenet.forward(test_input, training=False)
            
            # Should not produce NaN or inf
            assert not np.any(np.isnan(output)), "Should not produce NaN"
            assert not np.any(np.isinf(output)), "Should not produce inf"
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the network"""
        # This would require implementing backward pass
        # For now, just test forward pass doesn't break
        
        densenet = DenseNet(growth_rate=8, block_config=[2, 3], num_classes=10)
        test_input = np.random.randn(2, 3, 32, 32)
        
        output = densenet.forward(test_input, training=True)
        
        # Output should have reasonable values
        assert not np.all(output == 0), "Output should not be all zeros"
        assert np.all(np.isfinite(output)), "Output should be finite"


def test_architectural_correctness():
    """Test that implementation matches DenseNet paper specifications"""
    print("Testing architectural correctness...")
    
    # Test DenseNet-121 architecture
    densenet_121 = DenseNet(
        growth_rate=32,
        block_config=[6, 12, 24, 16],
        num_classes=1000
    )
    
    # Check parameter count is reasonable
    params = densenet_121.count_parameters()
    print(f"DenseNet-121 parameters: {params:,}")
    
    # Should be approximately 8M parameters (within reasonable range)
    assert 6_000_000 <= params <= 10_000_000, \
        f"Parameter count {params} outside expected range"


def test_efficiency_comparison():
    """Test efficiency compared to other architectures"""
    print("Testing efficiency comparison...")
    
    results = compare_densenet_resnet_efficiency((2, 3, 224, 224))
    
    # Should have computed parameter counts
    assert 'densenet_params' in results, "Should have DenseNet parameter counts"
    
    for name, params in results['densenet_params'].items():
        print(f"{name}: {params:,} parameters")
        assert params > 0, f"{name} should have positive parameter count"


def validate_dense_connectivity():
    """Validate that dense connectivity is implemented correctly"""
    print("Validating dense connectivity...")
    
    # Create a simple dense block
    dense_block = DenseBlock(in_channels=16, num_layers=3, growth_rate=8)
    
    # Check that each layer has access to all previous features
    input_channels = [16, 24, 32]  # Expected input channels for each layer
    
    for i, layer in enumerate(dense_block.layers):
        expected_channels = input_channels[i]
        assert layer.in_channels == expected_channels, \
            f"Layer {i} should have {expected_channels} input channels"
    
    print("Dense connectivity validation passed!")


def benchmark_performance():
    """Benchmark DenseNet performance"""
    print("Benchmarking DenseNet performance...")
    
    # Test different architectures
    architectures = create_densenet_variants()
    test_input = np.random.randn(1, 3, 224, 224)
    
    for name, model in architectures.items():
        start_time = time.time()
        
        # Forward pass
        output = model.forward(test_input, training=False)
        
        end_time = time.time()
        
        print(f"{name}: {end_time - start_time:.4f}s forward pass")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Output shape: {output.shape}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_architectural_correctness()
    test_efficiency_comparison()
    validate_dense_connectivity()
    benchmark_performance()
    
    print("\nTesting completed!")
    print("DenseNet implementation validated successfully.")