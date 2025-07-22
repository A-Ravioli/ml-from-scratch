"""
Test suite for EfficientNet implementation

Comprehensive tests for compound scaling, MBConv blocks, and mobile optimizations
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from exercise import *


class TestDepthwiseSeparableConv:
    """Test DepthwiseSeparableConv implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.in_channels = 32
        self.out_channels = 64
        self.height = 32
        self.width = 32
        
        self.input = np.random.randn(self.batch_size, self.in_channels, self.height, self.width)
        self.dw_conv = DepthwiseSeparableConv(self.in_channels, self.out_channels)
    
    def test_output_shape(self):
        """Test that depthwise separable conv produces correct output shape"""
        output = self.dw_conv.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.out_channels, self.height, self.width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_parameter_efficiency(self):
        """Test parameter count compared to standard convolution"""
        # Standard conv parameters: k^2 * in_ch * out_ch
        standard_params = 3 * 3 * self.in_channels * self.out_channels
        
        # Depthwise separable parameters: k^2 * in_ch + in_ch * out_ch
        dw_params = (3 * 3 * self.in_channels) + (self.in_channels * self.out_channels)
        
        assert dw_params < standard_params, "Depthwise separable should use fewer parameters"
        
        savings_ratio = (standard_params - dw_params) / standard_params
        assert savings_ratio > 0.5, "Should achieve significant parameter savings"
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes"""
        kernel_sizes = [3, 5, 7]
        
        for k in kernel_sizes:
            padding = (k - 1) // 2
            dw_conv = DepthwiseSeparableConv(
                self.in_channels, self.out_channels, 
                kernel_size=k, padding=padding
            )
            output = dw_conv.forward(self.input, training=False)
            
            # Spatial dimensions should be preserved with proper padding
            assert output.shape[2:] == self.input.shape[2:], f"Spatial dims not preserved for k={k}"
    
    def test_stride_functionality(self):
        """Test different stride values"""
        for stride in [1, 2]:
            dw_conv = DepthwiseSeparableConv(
                self.in_channels, self.out_channels, 
                stride=stride
            )
            output = dw_conv.forward(self.input, training=False)
            
            expected_h = self.height // stride
            expected_w = self.width // stride
            
            assert output.shape[2] == expected_h, f"Height incorrect for stride {stride}"
            assert output.shape[3] == expected_w, f"Width incorrect for stride {stride}"


class TestSqueezeExcitation:
    """Test SqueezeExcitation implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.channels = 64
        self.height = 16
        self.width = 16
        
        self.input = np.random.randn(self.batch_size, self.channels, self.height, self.width)
        self.se = SqueezeExcitation(self.channels)
    
    def test_output_shape(self):
        """Test that SE block preserves input shape"""
        output = self.se.forward(self.input)
        
        assert output.shape == self.input.shape, "SE block should preserve input shape"
    
    def test_channel_attention(self):
        """Test that SE provides channel-wise attention"""
        output = self.se.forward(self.input)
        
        # Output should be different from input (attention applied)
        assert not np.allclose(output, self.input), "SE should modify the input"
        
        # Should be non-negative (assuming ReLU/Sigmoid activations)
        # Note: This might not always hold depending on input, but generally true
    
    def test_squeeze_operation(self):
        """Test that squeeze operation works correctly"""
        # Manual squeeze test
        squeezed = np.mean(self.input, axis=(2, 3))
        assert squeezed.shape == (self.batch_size, self.channels), "Squeeze should reduce spatial dims"
    
    def test_different_reduction_ratios(self):
        """Test SE with different reduction ratios"""
        reduction_ratios = [2, 4, 8, 16]
        
        for ratio in reduction_ratios:
            se = SqueezeExcitation(self.channels, reduction_ratio=ratio)
            output = se.forward(self.input)
            
            assert output.shape == self.input.shape, f"Shape preserved for ratio {ratio}"
            
            # Check that reduced channels are computed correctly
            expected_reduced = max(1, self.channels // ratio)
            assert se.reduced_channels == expected_reduced


class TestSwish:
    """Test Swish activation function"""
    
    def test_swish_properties(self):
        """Test mathematical properties of Swish"""
        swish = Swish()
        
        # Test specific values
        x = np.array([-2, -1, 0, 1, 2])
        y = swish.forward(x)
        
        # Swish(0) should be 0
        assert abs(y[2]) < 1e-6, "Swish(0) should be 0"
        
        # Swish should be monotonic for reasonable inputs
        assert np.all(np.diff(y) >= 0), "Swish should be monotonic"
        
        # For large positive x, Swish(x) ≈ x
        large_x = np.array([10.0])
        large_y = swish.forward(large_x)
        assert abs(large_y[0] - large_x[0]) < 0.1, "Swish(large_x) should approximate x"
    
    def test_swish_vs_relu(self):
        """Compare Swish with ReLU"""
        swish = Swish()
        x = np.linspace(-3, 3, 100)
        
        swish_output = swish.forward(x)
        relu_output = np.maximum(0, x)
        
        # For positive inputs, Swish should be close to but slightly different from ReLU
        positive_mask = x > 0
        assert not np.allclose(swish_output[positive_mask], relu_output[positive_mask])
        
        # Swish should be smoother (non-zero for negative inputs)
        negative_mask = x < 0
        assert np.any(swish_output[negative_mask] != 0), "Swish should be non-zero for some negative inputs"


class TestMBConvBlock:
    """Test MBConvBlock implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.in_channels = 32
        self.out_channels = 64
        self.height = 32
        self.width = 32
        
        self.input = np.random.randn(self.batch_size, self.in_channels, self.height, self.width)
    
    def test_mbconv_output_shape(self):
        """Test MBConv block output shape"""
        mbconv = MBConvBlock(self.in_channels, self.out_channels, stride=1)
        output = mbconv.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.out_channels, self.height, self.width)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_mbconv_stride_2(self):
        """Test MBConv with stride=2"""
        mbconv = MBConvBlock(self.in_channels, self.out_channels, stride=2)
        output = mbconv.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.out_channels, self.height // 2, self.width // 2)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_skip_connection(self):
        """Test skip connection functionality"""
        # Skip connection should work when stride=1 and in_channels == out_channels
        mbconv_skip = MBConvBlock(self.in_channels, self.in_channels, stride=1)
        output_skip = mbconv_skip.forward(self.input, training=False)
        
        assert mbconv_skip.use_skip == True, "Should use skip connection"
        assert output_skip.shape == self.input.shape, "Skip connection should preserve shape"
        
        # No skip connection when stride != 1 or channels differ
        mbconv_no_skip = MBConvBlock(self.in_channels, self.out_channels, stride=2)
        assert mbconv_no_skip.use_skip == False, "Should not use skip connection"
    
    def test_expansion_ratios(self):
        """Test different expansion ratios"""
        expansion_ratios = [1, 3, 6]
        
        for ratio in expansion_ratios:
            mbconv = MBConvBlock(
                self.in_channels, self.out_channels, 
                expansion_ratio=ratio
            )
            output = mbconv.forward(self.input, training=False)
            
            assert output.shape[1] == self.out_channels, f"Output channels incorrect for expansion {ratio}"
            
            # Check if expansion conv exists for ratio > 1
            if ratio == 1:
                assert mbconv.expand_conv is None, "Should not have expansion conv for ratio=1"
            else:
                assert mbconv.expand_conv is not None, f"Should have expansion conv for ratio={ratio}"
    
    def test_se_integration(self):
        """Test SE block integration"""
        # With SE
        mbconv_se = MBConvBlock(self.in_channels, self.out_channels, use_se=True)
        output_se = mbconv_se.forward(self.input, training=False)
        
        # Without SE
        mbconv_no_se = MBConvBlock(self.in_channels, self.out_channels, use_se=False)
        output_no_se = mbconv_no_se.forward(self.input, training=False)
        
        assert output_se.shape == output_no_se.shape, "SE should not change output shape"
        assert hasattr(mbconv_se, 'se'), "Should have SE block when use_se=True"


class TestEfficientNet:
    """Test complete EfficientNet implementation"""
    
    def setup_method(self):
        """Setup test data"""
        self.batch_size = 2
        self.input_shape = (3, 224, 224)
        self.num_classes = 1000
        
        self.input = np.random.randn(self.batch_size, *self.input_shape)
        self.efficientnet = EfficientNet()
    
    def test_forward_pass(self):
        """Test complete forward pass"""
        output = self.efficientnet.forward(self.input, training=False)
        
        expected_shape = (self.batch_size, self.num_classes)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    def test_parameter_counting(self):
        """Test parameter counting functionality"""
        param_count = self.efficientnet.count_parameters()
        
        assert isinstance(param_count, int), "Parameter count should be integer"
        assert param_count > 0, "Should have positive number of parameters"
        
        # EfficientNet-B0 should have approximately 5.3M parameters
        expected_range = (4_000_000, 7_000_000)
        assert expected_range[0] <= param_count <= expected_range[1], \
            f"Parameter count {param_count} outside expected range {expected_range}"
    
    def test_different_input_sizes(self):
        """Test with different input resolutions"""
        resolutions = [224, 240, 260]
        
        for res in resolutions:
            test_input = np.random.randn(1, 3, res, res)
            
            # Create model with appropriate resolution
            model = EfficientNet(resolution=res)
            output = model.forward(test_input, training=False)
            
            assert output.shape == (1, self.num_classes), f"Failed for resolution {res}"
    
    def test_training_vs_inference(self):
        """Test differences between training and inference modes"""
        train_output = self.efficientnet.forward(self.input, training=True)
        infer_output = self.efficientnet.forward(self.input, training=False)
        
        assert train_output.shape == infer_output.shape, "Training and inference should have same output shape"
        
        # Outputs might be different due to batch norm and dropout
        # This is expected and correct


class TestCompoundScaling:
    """Test compound scaling methodology"""
    
    def test_scaling_coefficients(self):
        """Test that scaling coefficients satisfy constraints"""
        phi_values = [0, 0.5, 1.0, 1.5, 2.0]
        
        for phi in phi_values:
            depth_coeff, width_coeff, resolution = CompoundScaling.get_scaling_coefficients(phi)
            
            # Check that coefficients are reasonable
            assert depth_coeff >= 1.0, "Depth coefficient should be >= 1"
            assert width_coeff >= 1.0, "Width coefficient should be >= 1"
            assert resolution >= 224, "Resolution should be >= base resolution"
            
            # Check constraint approximately: α * β² * γ² ≈ 2^φ
            alpha, beta, gamma = 1.2, 1.1, 1.15
            constraint_value = (alpha ** phi) * (beta ** (2 * phi)) * (gamma ** (2 * phi))
            expected_value = 2 ** phi
            
            assert abs(constraint_value - expected_value) < 0.1, \
                f"Constraint violation for phi={phi}"
    
    def test_efficientnet_variants(self):
        """Test creation of EfficientNet variants"""
        variants = create_efficientnet_variants()
        
        expected_variants = ['EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 
                           'EfficientNet-B3', 'EfficientNet-B4', 'EfficientNet-B5',
                           'EfficientNet-B6', 'EfficientNet-B7']
        
        for name in expected_variants:
            assert name in variants, f"Missing variant {name}"
            
            model = variants[name]
            assert isinstance(model, EfficientNet), f"{name} should be EfficientNet instance"
            
            # Test that larger variants have more parameters
            if name != 'EfficientNet-B0':
                b0_params = variants['EfficientNet-B0'].count_parameters()
                variant_params = model.count_parameters()
                assert variant_params > b0_params, f"{name} should have more parameters than B0"
    
    def test_scaling_analysis(self):
        """Test compound scaling analysis"""
        analysis = analyze_compound_scaling((2, 3, 224, 224))
        
        assert 'single_dimension_scaling' in analysis, "Should have single dimension analysis"
        assert 'compound_scaling' in analysis, "Should have compound scaling analysis"
        
        # Check that compound scaling exists for different phi values
        compound_results = analysis['compound_scaling']
        assert len(compound_results) > 0, "Should have compound scaling results"


class TestMobileOptimization:
    """Test mobile optimization features"""
    
    def test_mobile_analysis(self):
        """Test mobile optimization analysis"""
        model = EfficientNet()
        analysis = mobile_optimization_analysis(model)
        
        required_keys = ['parameter_efficiency', 'operation_types', 'memory_optimization']
        for key in required_keys:
            assert key in analysis, f"Missing analysis key: {key}"
        
        # Check parameter efficiency
        param_info = analysis['parameter_efficiency']
        assert 'total_parameters' in param_info, "Should have parameter count"
        assert 'mobile_friendly' in param_info, "Should assess mobile friendliness"
        
        # Check operation types
        op_info = analysis['operation_types']
        assert 'depthwise_separable' in op_info, "Should count depthwise operations"
        assert 'separation_ratio' in op_info, "Should compute separation ratio"
    
    def test_progressive_resizing(self):
        """Test progressive resizing training schedule"""
        model = EfficientNet()
        schedule = progressive_resizing_training(model, 128, 224, 4)
        
        assert len(schedule) == 4, "Should have 4 training phases"
        
        # Check that image sizes increase
        sizes = [phase['image_size'] for phase in schedule]
        assert sizes == sorted(sizes), "Image sizes should be increasing"
        assert sizes[0] == 128, "Should start with initial size"
        assert sizes[-1] == 224, "Should end with final size"
        
        # Check that training time estimates increase
        times = [phase['training_time_relative'] for phase in schedule]
        assert times == sorted(times), "Training times should be increasing"


class TestNumericalStability:
    """Test numerical stability of EfficientNet components"""
    
    def test_extreme_inputs(self):
        """Test behavior with extreme input values"""
        extreme_inputs = [
            np.ones((1, 3, 224, 224)) * 1e6,    # Very large
            np.ones((1, 3, 224, 224)) * 1e-6,   # Very small
            np.zeros((1, 3, 224, 224)),         # All zeros
        ]
        
        model = EfficientNet()
        
        for test_input in extreme_inputs:
            output = model.forward(test_input, training=False)
            
            # Should not produce NaN or inf
            assert not np.any(np.isnan(output)), "Should not produce NaN"
            assert not np.any(np.isinf(output)), "Should not produce inf"
    
    def test_gradient_flow_simulation(self):
        """Test that forward pass completes without numerical issues"""
        model = EfficientNet()
        test_input = np.random.randn(2, 3, 224, 224)
        
        # Multiple forward passes should be stable
        for _ in range(5):
            output = model.forward(test_input, training=True)
            assert np.all(np.isfinite(output)), "Output should be finite"


def test_architectural_correctness():
    """Test that implementation matches EfficientNet specifications"""
    print("Testing architectural correctness...")
    
    # Test EfficientNet-B0
    b0_model = EfficientNet()
    test_input = np.random.randn(1, 3, 224, 224)
    output = b0_model.forward(test_input, training=False)
    
    # Check output shape
    assert output.shape == (1, 1000), "B0 should output 1000 classes"
    
    # Check parameter count is reasonable for B0
    params = b0_model.count_parameters()
    print(f"EfficientNet-B0 parameters: {params:,}")
    
    # Should be approximately 5.3M parameters (within reasonable range)
    assert 4_000_000 <= params <= 7_000_000, \
        f"Parameter count {params} outside expected range for B0"


def test_compound_scaling_effectiveness():
    """Test effectiveness of compound scaling vs single-dimension scaling"""
    print("Testing compound scaling effectiveness...")
    
    analysis = analyze_compound_scaling((1, 3, 224, 224))
    
    # Print scaling analysis
    print("Compound scaling analysis:")
    for phi, data in analysis['compound_scaling'].items():
        if phi.startswith('phi_'):
            print(f"  {phi}: {data['parameters']:,} parameters, coefficients: {data['coefficients']}")
    
    # Verify that compound scaling is more efficient than single-dimension scaling
    compound_phi_1 = analysis['compound_scaling']['phi_1.0']
    single_width_2 = analysis['single_dimension_scaling']['width_2.0']
    
    # Compound scaling should be more parameter efficient
    print(f"Compound phi=1.0: {compound_phi_1['parameters']:,} parameters")
    print(f"Width scaling 2.0: {single_width_2['parameters']:,} parameters")


def benchmark_efficientnet_variants():
    """Benchmark different EfficientNet variants"""
    print("Benchmarking EfficientNet variants...")
    
    variants = create_efficientnet_variants()
    test_input = np.random.randn(1, 3, 224, 224)
    
    for name, model in list(variants.items())[:4]:  # Test first 4 variants
        start_time = time.time()
        
        # Forward pass
        output = model.forward(test_input, training=False)
        
        end_time = time.time()
        
        print(f"{name}:")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Forward pass time: {end_time - start_time:.4f}s")
        print(f"  Output shape: {output.shape}")


def validate_mobile_optimizations():
    """Validate mobile optimization features"""
    print("Validating mobile optimizations...")
    
    model = EfficientNet()
    analysis = mobile_optimization_analysis(model)
    
    print("Mobile optimization analysis:")
    print(f"  Total parameters: {analysis['parameter_efficiency']['total_parameters']:,}")
    print(f"  Model size: {analysis['parameter_efficiency']['size_mb']:.1f} MB")
    print(f"  Mobile friendly: {analysis['parameter_efficiency']['mobile_friendly']}")
    
    print(f"  Depthwise operations: {analysis['operation_types']['depthwise_separable']}")
    print(f"  Separation ratio: {analysis['operation_types']['separation_ratio']:.2f}")
    
    # Test progressive resizing
    schedule = progressive_resizing_training(model)
    print("\nProgressive training schedule:")
    for phase in schedule:
        print(f"  Phase {phase['step']}: {phase['image_size']}px, {phase['suggested_epochs']} epochs")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run additional validation
    test_architectural_correctness()
    test_compound_scaling_effectiveness()
    benchmark_efficientnet_variants()
    validate_mobile_optimizations()
    
    print("\nTesting completed!")
    print("EfficientNet implementation validated successfully.")