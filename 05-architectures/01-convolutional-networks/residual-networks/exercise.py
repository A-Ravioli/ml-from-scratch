"""
Residual Networks Implementation

Implementation of ResNet architectures with residual connections and skip connections
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class ConvLayer:
    """2D Convolutional layer with batch normalization support"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization for ReLU networks
        fan_out = out_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / fan_out)
        
        self.weight = np.random.normal(0, std, 
                                     (out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through convolution"""
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Add padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant')
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        receptive_field = x[b, :, h_start:h_end, w_start:w_end]
                        output[b, oc, oh, ow] = np.sum(receptive_field * self.weight[oc]) + self.bias[oc]
        
        return output


class BatchNorm2d:
    """2D Batch Normalization"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = np.ones(num_features)  # gamma
        self.bias = np.zeros(num_features)   # beta
        
        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through batch normalization"""
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        
        return weight * x_norm + bias


class BasicBlock:
    """Basic ResNet block for ResNet-18/34"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv layer
        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        
        # Second conv layer
        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, 1, stride)
            self.shortcut_bn = BatchNorm2d(out_channels)
        else:
            self.shortcut = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through basic block"""
        identity = x
        
        # First conv
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu(out)
        
        # Second conv
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        
        # Shortcut
        if self.shortcut is not None:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, training)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class Bottleneck:
    """Bottleneck block for ResNet-50/101/152"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Bottleneck design: 1x1 -> 3x3 -> 1x1
        self.conv1 = ConvLayer(in_channels, out_channels // 4, 1)
        self.bn1 = BatchNorm2d(out_channels // 4)
        
        self.conv2 = ConvLayer(out_channels // 4, out_channels // 4, 3, stride, padding=1)
        self.bn2 = BatchNorm2d(out_channels // 4)
        
        self.conv3 = ConvLayer(out_channels // 4, out_channels, 1)
        self.bn3 = BatchNorm2d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, 1, stride)
            self.shortcut_bn = BatchNorm2d(out_channels)
        else:
            self.shortcut = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through bottleneck block"""
        identity = x
        
        # 1x1 conv
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu(out)
        
        # 3x3 conv
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        out = self.relu(out)
        
        # 1x1 conv
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training)
        
        # Shortcut
        if self.shortcut is not None:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, training)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class ResNet:
    """ResNet architecture"""
    
    def __init__(self, block_type: str, layers: List[int], num_classes: int = 1000):
        self.block_type = block_type
        self.num_classes = num_classes
        self.in_channels = 64
        
        # Initial conv layer
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        
        # Max pooling
        self.maxpool = MaxPool2d(3, stride=2, padding=1)
        
        # Residual layers
        if block_type == 'basic':
            self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
            final_channels = 512
        else:  # bottleneck
            self.layer1 = self._make_layer(Bottleneck, 256, layers[0], stride=1)
            self.layer2 = self._make_layer(Bottleneck, 512, layers[1], stride=2)
            self.layer3 = self._make_layer(Bottleneck, 1024, layers[2], stride=2)
            self.layer4 = self._make_layer(Bottleneck, 2048, layers[3], stride=2)
            final_channels = 2048
        
        # Global average pooling and final FC
        self.avgpool = GlobalAvgPool2d()
        self.fc = LinearLayer(final_channels, num_classes)
    
    def _make_layer(self, block_class, out_channels: int, num_blocks: int, stride: int):
        """Create a layer with multiple blocks"""
        layers = []
        
        # First block (may have stride > 1)
        layers.append(block_class(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block_class(self.in_channels, out_channels, stride=1))
        
        return layers
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through ResNet"""
        # Initial conv
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training)
        x = self.relu(x)
        x = self.maxpool.forward(x)
        
        # Residual layers
        for block in self.layer1:
            x = block.forward(x, training)
        
        for block in self.layer2:
            x = block.forward(x, training)
            
        for block in self.layer3:
            x = block.forward(x, training)
            
        for block in self.layer4:
            x = block.forward(x, training)
        
        # Global average pooling
        x = self.avgpool.forward(x)
        
        # Flatten and final FC
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc.forward(x)
        
        return x
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class MaxPool2d:
    """Max pooling layer"""
    
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through max pooling"""
        batch_size, channels, in_height, in_width = x.shape
        
        # Add padding
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                          (self.padding, self.padding)), mode='constant', constant_values=-np.inf)
        
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = ow * self.stride
                        w_end = w_start + self.kernel_size
                        
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.max(pool_region)
        
        return output


class GlobalAvgPool2d:
    """Global average pooling"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through global average pooling"""
        return np.mean(x, axis=(2, 3), keepdims=True)


class LinearLayer:
    """Fully connected layer"""
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        std = np.sqrt(2.0 / (in_features + out_features))
        self.weight = np.random.normal(0, std, (out_features, in_features))
        self.bias = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through linear layer"""
        return x @ self.weight.T + self.bias


def resnet18(num_classes: int = 1000) -> ResNet:
    """ResNet-18 model"""
    return ResNet('basic', [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 1000) -> ResNet:
    """ResNet-34 model"""
    return ResNet('basic', [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    """ResNet-50 model"""
    return ResNet('bottleneck', [3, 4, 6, 3], num_classes)


# Exercises
def exercise_1_basic_block():
    """Exercise 1: Test basic ResNet block"""
    print("=== Exercise 1: Basic Block ===")
    
    block = BasicBlock(64, 64, stride=1)
    
    batch_size = 2
    x = np.random.randn(batch_size, 64, 32, 32)
    
    output = block.forward(x, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == x.shape
    print("✓ Basic block working correctly")

def exercise_2_bottleneck_block():
    """Exercise 2: Test bottleneck ResNet block"""
    print("=== Exercise 2: Bottleneck Block ===")
    
    block = Bottleneck(256, 1024, stride=2)
    
    batch_size = 2
    x = np.random.randn(batch_size, 256, 32, 32)
    
    output = block.forward(x, training=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1024, 16, 16)
    print("✓ Bottleneck block working correctly")

def exercise_3_resnet18():
    """Exercise 3: Test ResNet-18"""
    print("=== Exercise 3: ResNet-18 ===")
    
    model = resnet18(num_classes=1000)
    
    batch_size = 1
    x = np.random.randn(batch_size, 3, 224, 224)
    
    output = model.forward(x, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1000)
    print("✓ ResNet-18 working correctly")


if __name__ == "__main__":
    exercise_1_basic_block()
    exercise_2_bottleneck_block()
    exercise_3_resnet18()
    print("\nResNet implementations completed!")