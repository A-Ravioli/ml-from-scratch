"""
DenseNet Implementation Exercise

Implement DenseNet from scratch with focus on dense connectivity patterns.
Study feature reuse, memory efficiency, and architectural innovations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time


class ConvLayer:
    """Basic convolutional layer with batch norm and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, use_bias: bool = False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        
        # Initialize weights (He initialization)
        fan_in = in_channels * kernel_size * kernel_size
        self.weight = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        
        if use_bias:
            self.bias = np.zeros(out_channels)
        
        # Batch normalization parameters
        self.bn_gamma = np.ones(out_channels)
        self.bn_beta = np.zeros(out_channels)
        self.running_mean = np.zeros(out_channels)
        self.running_var = np.ones(out_channels)
        self.momentum = 0.9
        self.eps = 1e-5
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: Conv → BatchNorm → ReLU"""
        # TODO: Implement convolution operation
        # This is a simplified version - full implementation would use optimized conv
        
        # Convolution (simplified - assumes specific input/output shapes)
        batch_size, in_c, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Simplified convolution implementation
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # TODO: Implement efficient convolution
        # For this exercise, we'll use a simplified approach
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Extract patch
                        i_start = i * self.stride - self.padding
                        j_start = j * self.stride - self.padding
                        i_end = i_start + self.kernel_size
                        j_end = j_start + self.kernel_size
                        
                        # Handle padding
                        patch = np.zeros((in_c, self.kernel_size, self.kernel_size))
                        for ic in range(in_c):
                            for ki in range(self.kernel_size):
                                for kj in range(self.kernel_size):
                                    ii = i_start + ki
                                    jj = j_start + kj
                                    if 0 <= ii < in_h and 0 <= jj < in_w:
                                        patch[ic, ki, kj] = x[b, ic, ii, jj]
                        
                        # Convolution
                        output[b, out_c, i, j] = np.sum(patch * self.weight[out_c])
        
        if self.use_bias:
            output += self.bias.reshape(1, -1, 1, 1)
        
        # Batch normalization
        if training:
            # Compute batch statistics
            batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)
            
            # Normalize
            output_norm = (output - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.squeeze()
        else:
            # Use running statistics
            output_norm = (output - self.running_mean.reshape(1, -1, 1, 1)) / np.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        output_norm = self.bn_gamma.reshape(1, -1, 1, 1) * output_norm + self.bn_beta.reshape(1, -1, 1, 1)
        
        # ReLU activation
        output_final = np.maximum(0, output_norm)
        
        return output_final


class DenseLayer:
    """
    Single dense layer: BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)
    
    Implements the bottleneck design for computational efficiency
    """
    
    def __init__(self, in_channels: int, growth_rate: int, bottleneck_factor: int = 4):
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.bottleneck_factor = bottleneck_factor
        
        # Bottleneck layer (1x1 conv)
        bottleneck_channels = bottleneck_factor * growth_rate
        self.bottleneck = ConvLayer(in_channels, bottleneck_channels, kernel_size=1, padding=0)
        
        # Main layer (3x3 conv)
        self.conv = ConvLayer(bottleneck_channels, growth_rate, kernel_size=3, padding=1)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through dense layer
        
        Args:
            x: Concatenated features from all previous layers in the block
            
        Returns:
            New feature maps of size growth_rate
        """
        # TODO: Implement dense layer forward pass
        # 1. Apply bottleneck layer (1x1 conv)
        # 2. Apply main layer (3x3 conv)
        # 3. Return new features (to be concatenated with input)
        
        # Bottleneck
        bottleneck_out = self.bottleneck.forward(x, training)
        
        # Main convolution
        output = self.conv.forward(bottleneck_out, training)
        
        return output


class DenseBlock:
    """
    Dense block containing multiple dense layers with dense connectivity
    
    Each layer receives concatenated features from ALL previous layers
    """
    
    def __init__(self, in_channels: int, num_layers: int, growth_rate: int,
                 bottleneck_factor: int = 4):
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # Create dense layers
        self.layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            layer = DenseLayer(current_channels, growth_rate, bottleneck_factor)
            self.layers.append(layer)
            current_channels += growth_rate  # Channels grow with each layer
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through dense block
        
        Implements dense connectivity: each layer receives ALL previous features
        """
        # TODO: Implement dense block forward pass
        # 1. Start with input features
        # 2. For each layer:
        #    - Pass concatenated features through layer
        #    - Concatenate new features with existing features
        # 3. Return final concatenated features
        
        features = [x]  # List to store all feature maps
        
        for layer in self.layers:
            # Concatenate all previous features
            concatenated_features = np.concatenate(features, axis=1)
            
            # Pass through current layer
            new_features = layer.forward(concatenated_features, training)
            
            # Add to feature list
            features.append(new_features)
        
        # Return concatenation of all features
        return np.concatenate(features, axis=1)
    
    def get_output_channels(self) -> int:
        """Calculate number of output channels"""
        return self.in_channels + self.num_layers * self.growth_rate


class TransitionLayer:
    """
    Transition layer between dense blocks
    
    Performs: BN → ReLU → Conv(1×1) → AvgPool(2×2)
    Optionally compresses features by factor theta
    """
    
    def __init__(self, in_channels: int, compression_factor: float = 0.5):
        self.in_channels = in_channels
        self.compression_factor = compression_factor
        self.out_channels = int(in_channels * compression_factor)
        
        # 1x1 convolution for channel reduction
        self.conv = ConvLayer(in_channels, self.out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through transition layer"""
        # TODO: Implement transition layer
        # 1. Apply 1x1 convolution (with BN and ReLU)
        # 2. Apply average pooling (2x2)
        
        # Convolution (includes BN and ReLU)
        conv_out = self.conv.forward(x, training)
        
        # Average pooling (2x2)
        batch_size, channels, height, width = conv_out.shape
        pooled_height = height // 2
        pooled_width = width // 2
        
        pooled = np.zeros((batch_size, channels, pooled_height, pooled_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        # Average over 2x2 region
                        region = conv_out[b, c, i*2:(i+1)*2, j*2:(j+1)*2]
                        pooled[b, c, i, j] = np.mean(region)
        
        return pooled


class DenseNet:
    """
    Complete DenseNet architecture
    
    Structure: Initial Conv → Dense Block → Transition → ... → Global Pool → FC
    """
    
    def __init__(self, growth_rate: int = 32, block_config: List[int] = [6, 12, 24, 16],
                 num_classes: int = 1000, compression_factor: float = 0.5,
                 initial_channels: int = 64):
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_classes = num_classes
        self.compression_factor = compression_factor
        
        # Initial convolution
        self.initial_conv = ConvLayer(3, initial_channels, kernel_size=7, stride=2, padding=3)
        
        # Build dense blocks and transitions
        self.blocks = []
        self.transitions = []
        
        current_channels = initial_channels
        
        for i, num_layers in enumerate(block_config):
            # Dense block
            block = DenseBlock(current_channels, num_layers, growth_rate)
            self.blocks.append(block)
            current_channels = block.get_output_channels()
            
            # Transition layer (except after last block)
            if i < len(block_config) - 1:
                transition = TransitionLayer(current_channels, compression_factor)
                self.transitions.append(transition)
                current_channels = transition.out_channels
        
        # Final layers
        self.final_channels = current_channels
        
        # Global average pooling will be applied in forward pass
        # Final fully connected layer
        self.fc_weight = np.random.randn(num_classes, current_channels) * np.sqrt(2.0 / current_channels)
        self.fc_bias = np.zeros(num_classes)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through complete DenseNet"""
        # TODO: Implement complete DenseNet forward pass
        # 1. Initial convolution
        # 2. For each dense block:
        #    - Apply dense block
        #    - Apply transition (if not last block)
        # 3. Global average pooling
        # 4. Fully connected layer
        
        # Initial convolution
        features = self.initial_conv.forward(x, training)
        
        # Apply max pooling after initial conv (common in DenseNet)
        batch_size, channels, height, width = features.shape
        pooled_height = height // 2
        pooled_width = width // 2
        pooled_features = np.zeros((batch_size, channels, pooled_height, pooled_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        region = features[b, c, i*2:(i+1)*2, j*2:(j+1)*2]
                        pooled_features[b, c, i, j] = np.max(region)
        
        features = pooled_features
        
        # Dense blocks and transitions
        for i, block in enumerate(self.blocks):
            features = block.forward(features, training)
            
            if i < len(self.transitions):
                features = self.transitions[i].forward(features, training)
        
        # Global average pooling
        global_pool = np.mean(features, axis=(2, 3))  # Average over spatial dimensions
        
        # Fully connected layer
        output = global_pool @ self.fc_weight.T + self.fc_bias
        
        return output
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        # TODO: Implement parameter counting
        # This would sum up all weight and bias parameters in the network
        total_params = 0
        
        # Initial conv
        total_params += np.prod(self.initial_conv.weight.shape)
        if hasattr(self.initial_conv, 'bias'):
            total_params += np.prod(self.initial_conv.bias.shape)
        
        # Dense blocks and transitions
        for block in self.blocks:
            for layer in block.layers:
                total_params += np.prod(layer.bottleneck.weight.shape)
                total_params += np.prod(layer.conv.weight.shape)
        
        for transition in self.transitions:
            total_params += np.prod(transition.conv.weight.shape)
        
        # Final FC
        total_params += np.prod(self.fc_weight.shape)
        total_params += np.prod(self.fc_bias.shape)
        
        return total_params


class MemoryEfficientDenseNet(DenseNet):
    """
    Memory-efficient implementation of DenseNet
    
    Uses gradient checkpointing and memory optimization techniques
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpointing = True
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Memory-efficient forward pass"""
        # TODO: Implement memory-efficient forward pass
        # 1. Use gradient checkpointing
        # 2. Recompute intermediate activations during backward pass
        # 3. Manage memory allocation efficiently
        
        # For this simplified implementation, we'll just call the standard forward
        # In practice, this would involve more sophisticated memory management
        return super().forward(x, training)


def create_densenet_variants() -> Dict[str, DenseNet]:
    """Create standard DenseNet variants"""
    variants = {
        'DenseNet-121': DenseNet(
            growth_rate=32,
            block_config=[6, 12, 24, 16],
            num_classes=1000
        ),
        'DenseNet-169': DenseNet(
            growth_rate=32,
            block_config=[6, 12, 32, 32],
            num_classes=1000
        ),
        'DenseNet-201': DenseNet(
            growth_rate=32,
            block_config=[6, 12, 48, 32],
            num_classes=1000
        ),
        'DenseNet-264': DenseNet(
            growth_rate=32,
            block_config=[6, 12, 64, 48],
            num_classes=1000
        )
    }
    
    return variants


def analyze_feature_reuse(dense_block: DenseBlock, input_features: np.ndarray) -> Dict:
    """
    Analyze feature reuse patterns in a dense block
    """
    analysis = {
        'layer_inputs': [],
        'layer_outputs': [],
        'channel_growth': [],
        'feature_reuse_matrix': None
    }
    
    # TODO: Implement feature reuse analysis
    # 1. Track input/output features for each layer
    # 2. Measure feature reuse patterns
    # 3. Analyze channel growth and connectivity
    
    features = [input_features]
    
    for i, layer in enumerate(dense_block.layers):
        # Concatenate all previous features
        concatenated = np.concatenate(features, axis=1)
        analysis['layer_inputs'].append(concatenated.shape)
        
        # Forward through layer
        new_features = layer.forward(concatenated, training=False)
        features.append(new_features)
        analysis['layer_outputs'].append(new_features.shape)
        
        # Track channel growth
        total_channels = sum(f.shape[1] for f in features)
        analysis['channel_growth'].append(total_channels)
    
    # TODO: Compute feature reuse matrix
    # This would analyze how features from different layers are used
    
    return analysis


def compare_densenet_resnet_efficiency(input_shape: Tuple[int, int, int, int]) -> Dict:
    """
    Compare parameter efficiency between DenseNet and ResNet
    """
    results = {
        'densenet_params': {},
        'resnet_params': {},  # Would need ResNet implementation
        'efficiency_ratios': {}
    }
    
    # Create DenseNet variants
    densenet_variants = create_densenet_variants()
    
    for name, model in densenet_variants.items():
        params = model.count_parameters()
        results['densenet_params'][name] = params
        print(f"{name}: {params:,} parameters")
    
    # TODO: Compare with equivalent ResNet architectures
    # This would require ResNet implementation for fair comparison
    
    return results


def visualize_dense_connectivity(dense_block: DenseBlock, save_path: str = None):
    """
    Visualize the dense connectivity pattern
    """
    num_layers = len(dense_block.layers)
    
    # Create connectivity matrix
    connectivity_matrix = np.zeros((num_layers + 1, num_layers + 1))
    
    # Fill connectivity matrix
    for i in range(num_layers):
        # Each layer connects to ALL previous layers
        for j in range(i + 1):
            connectivity_matrix[i + 1, j] = 1
    
    # Plot connectivity matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(connectivity_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Dense Connectivity Pattern')
    plt.xlabel('Source Layer')
    plt.ylabel('Target Layer')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(label='Connection')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_dense_layer_implementation():
    """
    Exercise 1: Implement and test dense layers
    
    Tasks:
    1. Complete DenseLayer implementation
    2. Test dense connectivity within a layer
    3. Verify feature concatenation works correctly
    4. Analyze computational complexity
    """
    
    print("=== Exercise 1: Dense Layer Implementation ===")
    
    # TODO: Test dense layer implementation
    # Create simple test case and verify functionality
    
    # Test input
    batch_size, in_channels, height, width = 2, 64, 32, 32
    test_input = np.random.randn(batch_size, in_channels, height, width)
    
    # Create dense layer
    growth_rate = 32
    dense_layer = DenseLayer(in_channels, growth_rate)
    
    # Forward pass
    output = dense_layer.forward(test_input, training=False)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {growth_rate}")
    
    assert output.shape[1] == growth_rate, "Output should have growth_rate channels"
    
    pass


def exercise_2_dense_block_analysis():
    """
    Exercise 2: Implement and analyze dense blocks
    
    Tasks:
    1. Complete DenseBlock implementation
    2. Study feature growth patterns
    3. Analyze memory usage
    4. Visualize connectivity patterns
    """
    
    print("=== Exercise 2: Dense Block Analysis ===")
    
    # TODO: Analyze dense block properties
    
    pass


def exercise_3_transition_layers():
    """
    Exercise 3: Implement transition layers
    
    Tasks:
    1. Complete TransitionLayer implementation
    2. Study compression effects
    3. Analyze spatial downsampling
    4. Test different compression factors
    """
    
    print("=== Exercise 3: Transition Layers ===")
    
    # TODO: Test transition layer functionality
    
    pass


def exercise_4_complete_densenet():
    """
    Exercise 4: Build complete DenseNet architecture
    
    Tasks:
    1. Complete DenseNet implementation
    2. Test on sample inputs
    3. Verify architecture matches specifications
    4. Compare with standard implementations
    """
    
    print("=== Exercise 4: Complete DenseNet ===")
    
    # TODO: Test complete DenseNet
    
    pass


def exercise_5_efficiency_analysis():
    """
    Exercise 5: Analyze DenseNet efficiency
    
    Tasks:
    1. Compare parameter counts with ResNet
    2. Measure memory usage during training
    3. Analyze computational complexity
    4. Study accuracy vs efficiency trade-offs
    """
    
    print("=== Exercise 5: Efficiency Analysis ===")
    
    # TODO: Comprehensive efficiency analysis
    
    pass


def exercise_6_memory_optimization():
    """
    Exercise 6: Implement memory optimizations
    
    Tasks:
    1. Implement memory-efficient DenseNet
    2. Use gradient checkpointing
    3. Optimize memory allocation patterns
    4. Compare memory usage vs standard implementation
    """
    
    print("=== Exercise 6: Memory Optimization ===")
    
    # TODO: Implement memory optimizations
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_dense_layer_implementation()
    exercise_2_dense_block_analysis()
    exercise_3_transition_layers()
    exercise_4_complete_densenet()
    exercise_5_efficiency_analysis()
    exercise_6_memory_optimization()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Dense connectivity pattern and feature reuse")
    print("2. Memory vs performance trade-offs")
    print("3. Parameter efficiency compared to other architectures")
    print("4. Implementation challenges and optimization techniques")