"""
Classical CNN Architectures Implementation

Implementation of classical CNN architectures: LeNet, AlexNet, VGG, and analysis of their design principles
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
import time

class ConvLayer:
    """2D Convolutional layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier initialization
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size
        std = np.sqrt(2.0 / (fan_in + fan_out))
        
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

class PoolingLayer:
    """Max pooling layer"""
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through max pooling"""
        batch_size, channels, in_height, in_width = x.shape
        
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        h_end = h_start + self.pool_size
                        w_start = ow * self.stride
                        w_end = w_start + self.pool_size
                        
                        pool_region = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.max(pool_region)
        
        return output

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

class LeNet5:
    """LeNet-5 architecture (LeCun et al., 1998)"""
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = ConvLayer(1, 6, 5)  # 32x32 -> 28x28
        self.pool1 = PoolingLayer(2, 2)  # 28x28 -> 14x14
        self.conv2 = ConvLayer(6, 16, 5)  # 14x14 -> 10x10
        self.pool2 = PoolingLayer(2, 2)  # 10x10 -> 5x5
        self.fc1 = LinearLayer(16 * 5 * 5, 120)
        self.fc2 = LinearLayer(120, 84)
        self.fc3 = LinearLayer(84, num_classes)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through LeNet-5"""
        x = self.conv1.forward(x)
        x = self.tanh(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.tanh(x)
        x = self.pool2.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        x = self.fc1.forward(x)
        x = self.tanh(x)
        x = self.fc2.forward(x)
        x = self.tanh(x)
        x = self.fc3.forward(x)
        
        return x
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

class AlexNet:
    """AlexNet architecture (Krizhevsky et al., 2012)"""
    
    def __init__(self, num_classes: int = 1000):
        # Feature extractor
        self.conv1 = ConvLayer(3, 96, 11, stride=4, padding=2)  # 224x224 -> 55x55
        self.pool1 = PoolingLayer(3, 2)  # 55x55 -> 27x27
        
        self.conv2 = ConvLayer(96, 256, 5, padding=2)  # 27x27 -> 27x27
        self.pool2 = PoolingLayer(3, 2)  # 27x27 -> 13x13
        
        self.conv3 = ConvLayer(256, 384, 3, padding=1)  # 13x13 -> 13x13
        self.conv4 = ConvLayer(384, 384, 3, padding=1)  # 13x13 -> 13x13
        self.conv5 = ConvLayer(384, 256, 3, padding=1)  # 13x13 -> 13x13
        self.pool3 = PoolingLayer(3, 2)  # 13x13 -> 6x6
        
        # Classifier
        self.fc1 = LinearLayer(256 * 6 * 6, 4096)
        self.fc2 = LinearLayer(4096, 4096)
        self.fc3 = LinearLayer(4096, num_classes)
        
        self.dropout_rate = 0.5
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through AlexNet"""
        # Features
        x = self.relu(self.conv1.forward(x))
        x = self.pool1.forward(x)
        
        x = self.relu(self.conv2.forward(x))
        x = self.pool2.forward(x)
        
        x = self.relu(self.conv3.forward(x))
        x = self.relu(self.conv4.forward(x))
        x = self.relu(self.conv5.forward(x))
        x = self.pool3.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Classifier
        x = self.relu(self.fc1.forward(x))
        if training:
            x = self.dropout(x, self.dropout_rate)
        
        x = self.relu(self.fc2.forward(x))
        if training:
            x = self.dropout(x, self.dropout_rate)
        
        x = self.fc3.forward(x)
        
        return x
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Simple dropout implementation"""
        if rate == 0:
            return x
        mask = np.random.random(x.shape) > rate
        return x * mask / (1 - rate)

class VGG16:
    """VGG-16 architecture (Simonyan & Zisserman, 2014)"""
    
    def __init__(self, num_classes: int = 1000):
        # Block 1
        self.conv1_1 = ConvLayer(3, 64, 3, padding=1)
        self.conv1_2 = ConvLayer(64, 64, 3, padding=1)
        self.pool1 = PoolingLayer(2, 2)
        
        # Block 2
        self.conv2_1 = ConvLayer(64, 128, 3, padding=1)
        self.conv2_2 = ConvLayer(128, 128, 3, padding=1)
        self.pool2 = PoolingLayer(2, 2)
        
        # Block 3
        self.conv3_1 = ConvLayer(128, 256, 3, padding=1)
        self.conv3_2 = ConvLayer(256, 256, 3, padding=1)
        self.conv3_3 = ConvLayer(256, 256, 3, padding=1)
        self.pool3 = PoolingLayer(2, 2)
        
        # Block 4
        self.conv4_1 = ConvLayer(256, 512, 3, padding=1)
        self.conv4_2 = ConvLayer(512, 512, 3, padding=1)
        self.conv4_3 = ConvLayer(512, 512, 3, padding=1)
        self.pool4 = PoolingLayer(2, 2)
        
        # Block 5
        self.conv5_1 = ConvLayer(512, 512, 3, padding=1)
        self.conv5_2 = ConvLayer(512, 512, 3, padding=1)
        self.conv5_3 = ConvLayer(512, 512, 3, padding=1)
        self.pool5 = PoolingLayer(2, 2)
        
        # Classifier
        self.fc1 = LinearLayer(512 * 7 * 7, 4096)
        self.fc2 = LinearLayer(4096, 4096)
        self.fc3 = LinearLayer(4096, num_classes)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through VGG-16"""
        # Block 1
        x = self.relu(self.conv1_1.forward(x))
        x = self.relu(self.conv1_2.forward(x))
        x = self.pool1.forward(x)
        
        # Block 2
        x = self.relu(self.conv2_1.forward(x))
        x = self.relu(self.conv2_2.forward(x))
        x = self.pool2.forward(x)
        
        # Block 3
        x = self.relu(self.conv3_1.forward(x))
        x = self.relu(self.conv3_2.forward(x))
        x = self.relu(self.conv3_3.forward(x))
        x = self.pool3.forward(x)
        
        # Block 4
        x = self.relu(self.conv4_1.forward(x))
        x = self.relu(self.conv4_2.forward(x))
        x = self.relu(self.conv4_3.forward(x))
        x = self.pool4.forward(x)
        
        # Block 5
        x = self.relu(self.conv5_1.forward(x))
        x = self.relu(self.conv5_2.forward(x))
        x = self.relu(self.conv5_3.forward(x))
        x = self.pool5.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Classifier
        x = self.relu(self.fc1.forward(x))
        if training:
            x = self.dropout(x, 0.5)
        
        x = self.relu(self.fc2.forward(x))
        if training:
            x = self.dropout(x, 0.5)
        
        x = self.fc3.forward(x)
        
        return x
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        if rate == 0:
            return x
        mask = np.random.random(x.shape) > rate
        return x * mask / (1 - rate)

# Exercises
def exercise_1_lenet():
    """Exercise 1: Test LeNet-5 architecture"""
    print("=== Exercise 1: LeNet-5 ===")
    
    model = LeNet5(num_classes=10)
    
    # Test forward pass
    batch_size = 4
    x = np.random.randn(batch_size, 1, 32, 32)  # MNIST-style input
    
    output = model.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 10)
    print("✓ LeNet-5 working correctly")

def exercise_2_alexnet():
    """Exercise 2: Test AlexNet architecture"""
    print("=== Exercise 2: AlexNet ===")
    
    model = AlexNet(num_classes=1000)
    
    # Test forward pass
    batch_size = 2
    x = np.random.randn(batch_size, 3, 224, 224)  # ImageNet-style input
    
    output = model.forward(x, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1000)
    print("✓ AlexNet working correctly")

def exercise_3_vgg16():
    """Exercise 3: Test VGG-16 architecture"""
    print("=== Exercise 3: VGG-16 ===")
    
    model = VGG16(num_classes=1000)
    
    # Test forward pass (smaller batch due to memory)
    batch_size = 1
    x = np.random.randn(batch_size, 3, 224, 224)
    
    output = model.forward(x, training=False)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 1000)
    print("✓ VGG-16 working correctly")

if __name__ == "__main__":
    exercise_1_lenet()
    exercise_2_alexnet() 
    exercise_3_vgg16()
    print("\nClassical CNN implementations completed!")