"""
EfficientNet Implementation Exercise

Implement EfficientNet from scratch focusing on compound scaling methodology
and mobile-optimized building blocks (MBConv, SE blocks).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time
import math


class DepthwiseSeparableConv:
    """Depthwise separable convolution: Depthwise + Pointwise"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution weights (one filter per input channel)
        fan_in = kernel_size * kernel_size
        self.depthwise_weight = np.random.randn(in_channels, 1, kernel_size, kernel_size) * np.sqrt(2.0 / fan_in)
        
        # Pointwise convolution weights (1x1 conv)
        self.pointwise_weight = np.random.randn(out_channels, in_channels, 1, 1) * np.sqrt(2.0 / in_channels)
        self.pointwise_bias = np.zeros(out_channels)
        
        # Batch normalization parameters for both convolutions
        self.dw_bn_gamma = np.ones(in_channels)
        self.dw_bn_beta = np.zeros(in_channels)
        self.dw_running_mean = np.zeros(in_channels)
        self.dw_running_var = np.ones(in_channels)
        
        self.pw_bn_gamma = np.ones(out_channels)
        self.pw_bn_beta = np.zeros(out_channels)
        self.pw_running_mean = np.zeros(out_channels)
        self.pw_running_var = np.ones(out_channels)
        
        self.momentum = 0.9
        self.eps = 1e-5
    
    def _batch_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                    running_mean: np.ndarray, running_var: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply batch normalization"""
        if training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Update running statistics
            running_mean[:] = self.momentum * running_mean + (1 - self.momentum) * batch_mean.squeeze()
            running_var[:] = self.momentum * running_var + (1 - self.momentum) * batch_var.squeeze()
        else:
            x_norm = (x - running_mean.reshape(1, -1, 1, 1)) / np.sqrt(running_var.reshape(1, -1, 1, 1) + self.eps)
        
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)
    
    def _simple_conv2d(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None,
                       stride: int = 1, padding: int = 0) -> np.ndarray:
        """Simplified 2D convolution implementation"""
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weight.shape
        
        # Apply padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        out_h = (x_padded.shape[2] - k_h) // stride + 1
        out_w = (x_padded.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        output[b, oc, i, j] = np.sum(
                            x_padded[b, :, h_start:h_end, w_start:w_end] * weight[oc]
                        )
        
        if bias is not None:
            output += bias.reshape(1, -1, 1, 1)
        
        return output
    
    def _depthwise_conv2d(self, x: np.ndarray, weight: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
        """Depthwise convolution - apply one filter per input channel"""
        batch_size, in_channels, in_h, in_w = x.shape
        _, _, k_h, k_w = weight.shape
        
        # Apply padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        out_h = (x_padded.shape[2] - k_h) // stride + 1
        out_w = (x_padded.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, in_channels, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        output[b, c, i, j] = np.sum(
                            x_padded[b, c, h_start:h_end, w_start:w_end] * weight[c, 0]
                        )
        
        return output
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass: Depthwise Conv → BN → ReLU → Pointwise Conv → BN"""
        # TODO: Implement depthwise separable convolution
        # 1. Depthwise convolution
        # 2. Batch normalization + ReLU
        # 3. Pointwise convolution (1x1)
        # 4. Batch normalization
        
        # Depthwise convolution
        dw_out = self._depthwise_conv2d(x, self.depthwise_weight, self.stride, self.padding)
        
        # Batch norm + ReLU
        dw_bn = self._batch_norm(dw_out, self.dw_bn_gamma, self.dw_bn_beta,
                                self.dw_running_mean, self.dw_running_var, training)
        dw_relu = np.maximum(0, dw_bn)
        
        # Pointwise convolution (1x1)
        pw_out = self._simple_conv2d(dw_relu, self.pointwise_weight, self.pointwise_bias)
        
        # Batch norm
        output = self._batch_norm(pw_out, self.pw_bn_gamma, self.pw_bn_beta,
                                 self.pw_running_mean, self.pw_running_var, training)
        
        return output


class SqueezeExcitation:
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction_ratio: int = 4):
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(1, channels // reduction_ratio)
        
        # Squeeze: Global average pooling (implemented in forward)
        # Excitation: Two FC layers
        self.fc1_weight = np.random.randn(self.reduced_channels, channels) * np.sqrt(2.0 / channels)
        self.fc1_bias = np.zeros(self.reduced_channels)
        
        self.fc2_weight = np.random.randn(channels, self.reduced_channels) * np.sqrt(2.0 / self.reduced_channels)
        self.fc2_bias = np.zeros(channels)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: Squeeze → Excitation → Scale"""
        # TODO: Implement Squeeze-and-Excitation
        # 1. Global average pooling (squeeze)
        # 2. Two FC layers with ReLU and Sigmoid (excitation)
        # 3. Scale input features
        
        batch_size, channels, height, width = x.shape
        
        # Squeeze: Global average pooling
        squeezed = np.mean(x, axis=(2, 3))  # Shape: (batch_size, channels)
        
        # Excitation: FC → ReLU → FC → Sigmoid
        excited = squeezed @ self.fc1_weight.T + self.fc1_bias
        excited = np.maximum(0, excited)  # ReLU
        excited = excited @ self.fc2_weight.T + self.fc2_bias
        excited = 1 / (1 + np.exp(-excited))  # Sigmoid
        
        # Scale: Broadcast and multiply
        scale = excited.reshape(batch_size, channels, 1, 1)
        output = x * scale
        
        return output


class Swish:
    """Swish activation function: x * sigmoid(x)"""
    
    def __init__(self, beta: float = 1.0):
        self.beta = beta
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Swish activation: x * sigmoid(beta * x)"""
        return x * (1 / (1 + np.exp(-self.beta * x)))


class MBConvBlock:
    """
    Mobile Inverted Bottleneck Convolution Block
    
    Structure: Expansion → Depthwise → SE → Projection → Skip
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, expansion_ratio: int = 6, se_ratio: float = 0.25,
                 use_se: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion_ratio = expansion_ratio
        self.se_ratio = se_ratio
        self.use_se = use_se
        self.use_skip = (stride == 1 and in_channels == out_channels)
        
        # Expansion phase (1x1 conv)
        expanded_channels = in_channels * expansion_ratio
        self.expand_conv = None
        if expansion_ratio != 1:
            self.expand_conv = self._conv1x1_bn_swish(in_channels, expanded_channels)
            self.expansion_channels = expanded_channels
        else:
            self.expansion_channels = in_channels
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = DepthwiseSeparableConv(
            self.expansion_channels, self.expansion_channels, 
            kernel_size, stride, padding
        )
        
        # Squeeze-and-Excitation
        if use_se:
            self.se = SqueezeExcitation(self.expansion_channels, int(1 / se_ratio))
        
        # Projection phase (1x1 conv without activation)
        self.project_conv = self._conv1x1_bn(self.expansion_channels, out_channels)
        
        # Swish activation
        self.swish = Swish()
    
    def _conv1x1_bn_swish(self, in_ch: int, out_ch: int):
        """1x1 convolution with batch norm and Swish activation"""
        return {
            'weight': np.random.randn(out_ch, in_ch, 1, 1) * np.sqrt(2.0 / in_ch),
            'bias': np.zeros(out_ch),
            'bn_gamma': np.ones(out_ch),
            'bn_beta': np.zeros(out_ch),
            'running_mean': np.zeros(out_ch),
            'running_var': np.ones(out_ch)
        }
    
    def _conv1x1_bn(self, in_ch: int, out_ch: int):
        """1x1 convolution with batch norm (no activation)"""
        return {
            'weight': np.random.randn(out_ch, in_ch, 1, 1) * np.sqrt(2.0 / in_ch),
            'bias': np.zeros(out_ch),
            'bn_gamma': np.ones(out_ch),
            'bn_beta': np.zeros(out_ch),
            'running_mean': np.zeros(out_ch),
            'running_var': np.ones(out_ch)
        }
    
    def _apply_conv1x1_bn_swish(self, x: np.ndarray, conv_params: dict, training: bool = True) -> np.ndarray:
        """Apply 1x1 conv + BN + Swish"""
        # Convolution
        batch_size, in_channels, height, width = x.shape
        out_channels = conv_params['weight'].shape[0]
        
        output = np.zeros((batch_size, out_channels, height, width))
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(height):
                    for j in range(width):
                        output[b, oc, i, j] = np.sum(x[b, :, i, j] * conv_params['weight'][oc, :, 0, 0]) + conv_params['bias'][oc]
        
        # Batch normalization
        if training:
            batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)
            output_norm = (output - batch_mean) / np.sqrt(batch_var + 1e-5)
            conv_params['running_mean'][:] = 0.9 * conv_params['running_mean'] + 0.1 * batch_mean.squeeze()
            conv_params['running_var'][:] = 0.9 * conv_params['running_var'] + 0.1 * batch_var.squeeze()
        else:
            output_norm = (output - conv_params['running_mean'].reshape(1, -1, 1, 1)) / np.sqrt(conv_params['running_var'].reshape(1, -1, 1, 1) + 1e-5)
        
        output_norm = conv_params['bn_gamma'].reshape(1, -1, 1, 1) * output_norm + conv_params['bn_beta'].reshape(1, -1, 1, 1)
        
        # Swish activation
        return self.swish.forward(output_norm)
    
    def _apply_conv1x1_bn(self, x: np.ndarray, conv_params: dict, training: bool = True) -> np.ndarray:
        """Apply 1x1 conv + BN (no activation)"""
        # Convolution
        batch_size, in_channels, height, width = x.shape
        out_channels = conv_params['weight'].shape[0]
        
        output = np.zeros((batch_size, out_channels, height, width))
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(height):
                    for j in range(width):
                        output[b, oc, i, j] = np.sum(x[b, :, i, j] * conv_params['weight'][oc, :, 0, 0]) + conv_params['bias'][oc]
        
        # Batch normalization
        if training:
            batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)
            output_norm = (output - batch_mean) / np.sqrt(batch_var + 1e-5)
            conv_params['running_mean'][:] = 0.9 * conv_params['running_mean'] + 0.1 * batch_mean.squeeze()
            conv_params['running_var'][:] = 0.9 * conv_params['running_var'] + 0.1 * batch_var.squeeze()
        else:
            output_norm = (output - conv_params['running_mean'].reshape(1, -1, 1, 1)) / np.sqrt(conv_params['running_var'].reshape(1, -1, 1, 1) + 1e-5)
        
        return conv_params['bn_gamma'].reshape(1, -1, 1, 1) * output_norm + conv_params['bn_beta'].reshape(1, -1, 1, 1)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through MBConv block"""
        # TODO: Implement MBConv forward pass
        # 1. Expansion (if expansion_ratio > 1)
        # 2. Depthwise convolution with Swish
        # 3. Squeeze-and-Excitation (if enabled)
        # 4. Projection
        # 5. Skip connection (if applicable)
        
        residual = x
        
        # Expansion phase
        if self.expand_conv is not None:
            x = self._apply_conv1x1_bn_swish(x, self.expand_conv, training)
        
        # Depthwise convolution
        x = self.depthwise.forward(x, training)
        x = self.swish.forward(x)
        
        # Squeeze-and-Excitation
        if self.use_se:
            x = self.se.forward(x)
        
        # Projection
        x = self._apply_conv1x1_bn(x, self.project_conv, training)
        
        # Skip connection
        if self.use_skip:
            x = x + residual
        
        return x


class EfficientNet:
    """
    EfficientNet architecture with compound scaling
    
    Implements the base EfficientNet-B0 and scaling methodology
    """
    
    def __init__(self, width_coefficient: float = 1.0, depth_coefficient: float = 1.0,
                 resolution: int = 224, dropout_rate: float = 0.2, num_classes: int = 1000):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.resolution = resolution
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        
        # EfficientNet-B0 base configuration
        self.base_config = [
            # (expansion_ratio, channels, num_layers, stride, kernel_size)
            (1, 16, 1, 1, 3),    # Stage 1
            (6, 24, 2, 2, 3),    # Stage 2
            (6, 40, 2, 2, 5),    # Stage 3
            (6, 80, 3, 2, 3),    # Stage 4
            (6, 112, 3, 1, 5),   # Stage 5
            (6, 192, 4, 2, 5),   # Stage 6
            (6, 320, 1, 1, 3),   # Stage 7
        ]
        
        # Build the network
        self._build_network()
    
    def _round_filters(self, filters: int) -> int:
        """Round filter count based on width coefficient"""
        filters *= self.width_coefficient
        new_filters = max(8, int(filters + 4) // 8 * 8)  # Ensure divisible by 8
        if new_filters < 0.9 * filters:  # Prevent too much rounding
            new_filters += 8
        return int(new_filters)
    
    def _round_repeats(self, repeats: int) -> int:
        """Round repeat count based on depth coefficient"""
        return int(math.ceil(self.depth_coefficient * repeats))
    
    def _build_network(self):
        """Build the EfficientNet architecture"""
        # Initial convolution
        initial_filters = self._round_filters(32)
        self.initial_conv = self._conv_bn_swish(3, initial_filters, kernel_size=3, stride=2)
        
        # Build MBConv blocks
        self.blocks = []
        in_channels = initial_filters
        
        for stage_idx, (expansion_ratio, base_channels, num_layers, stride, kernel_size) in enumerate(self.base_config):
            out_channels = self._round_filters(base_channels)
            num_layers = self._round_repeats(num_layers)
            
            # First block in stage (handles stride and channel change)
            block = MBConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expansion_ratio=expansion_ratio,
                se_ratio=0.25,
                use_se=True
            )
            self.blocks.append(block)
            in_channels = out_channels
            
            # Remaining blocks in stage (stride=1, same channels)
            for _ in range(num_layers - 1):
                block = MBConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    expansion_ratio=expansion_ratio,
                    se_ratio=0.25,
                    use_se=True
                )
                self.blocks.append(block)
        
        # Final convolution
        final_filters = self._round_filters(1280)
        self.final_conv = self._conv_bn_swish(in_channels, final_filters, kernel_size=1)
        
        # Classification head
        self.fc_weight = np.random.randn(num_classes, final_filters) * np.sqrt(2.0 / final_filters)
        self.fc_bias = np.zeros(num_classes)
        
        self.final_channels = final_filters
    
    def _conv_bn_swish(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        """Create conv + BN + Swish layer"""
        padding = (kernel_size - 1) // 2
        return {
            'weight': np.random.randn(out_ch, in_ch, kernel_size, kernel_size) * np.sqrt(2.0 / (in_ch * kernel_size * kernel_size)),
            'bias': np.zeros(out_ch),
            'bn_gamma': np.ones(out_ch),
            'bn_beta': np.zeros(out_ch),
            'running_mean': np.zeros(out_ch),
            'running_var': np.ones(out_ch),
            'stride': stride,
            'padding': padding
        }
    
    def _apply_conv_bn_swish(self, x: np.ndarray, layer_params: dict, training: bool = True) -> np.ndarray:
        """Apply conv + BN + Swish"""
        # Simplified convolution
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = layer_params['weight'].shape
        stride = layer_params['stride']
        padding = layer_params['padding']
        
        # Apply padding
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
        else:
            x_padded = x
        
        out_h = (x_padded.shape[2] - k_h) // stride + 1
        out_w = (x_padded.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        h_end = h_start + k_h
                        w_end = w_start + k_w
                        
                        output[b, oc, i, j] = np.sum(
                            x_padded[b, :, h_start:h_end, w_start:w_end] * layer_params['weight'][oc]
                        ) + layer_params['bias'][oc]
        
        # Batch normalization
        if training:
            batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)
            output_norm = (output - batch_mean) / np.sqrt(batch_var + 1e-5)
            layer_params['running_mean'][:] = 0.9 * layer_params['running_mean'] + 0.1 * batch_mean.squeeze()
            layer_params['running_var'][:] = 0.9 * layer_params['running_var'] + 0.1 * batch_var.squeeze()
        else:
            output_norm = (output - layer_params['running_mean'].reshape(1, -1, 1, 1)) / np.sqrt(layer_params['running_var'].reshape(1, -1, 1, 1) + 1e-5)
        
        output_norm = layer_params['bn_gamma'].reshape(1, -1, 1, 1) * output_norm + layer_params['bn_beta'].reshape(1, -1, 1, 1)
        
        # Swish activation
        swish = Swish()
        return swish.forward(output_norm)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through EfficientNet"""
        # TODO: Implement complete EfficientNet forward pass
        # 1. Initial convolution
        # 2. MBConv blocks
        # 3. Final convolution
        # 4. Global average pooling
        # 5. Dropout (if training)
        # 6. Fully connected layer
        
        # Initial convolution
        x = self._apply_conv_bn_swish(x, self.initial_conv, training)
        
        # MBConv blocks
        for block in self.blocks:
            x = block.forward(x, training)
        
        # Final convolution
        x = self._apply_conv_bn_swish(x, self.final_conv, training)
        
        # Global average pooling
        x = np.mean(x, axis=(2, 3))  # Shape: (batch_size, final_channels)
        
        # Dropout (simplified - just for training flag demonstration)
        if training and self.dropout_rate > 0:
            dropout_mask = np.random.random(x.shape) > self.dropout_rate
            x = x * dropout_mask / (1 - self.dropout_rate)
        
        # Fully connected layer
        output = x @ self.fc_weight.T + self.fc_bias
        
        return output
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        total_params = 0
        
        # Initial conv
        total_params += np.prod(self.initial_conv['weight'].shape)
        total_params += np.prod(self.initial_conv['bias'].shape)
        
        # MBConv blocks (simplified counting)
        for block in self.blocks:
            if block.expand_conv is not None:
                total_params += np.prod(block.expand_conv['weight'].shape)
                total_params += np.prod(block.expand_conv['bias'].shape)
            
            total_params += np.prod(block.depthwise.depthwise_weight.shape)
            total_params += np.prod(block.depthwise.pointwise_weight.shape)
            total_params += np.prod(block.depthwise.pointwise_bias.shape)
            
            if block.use_se:
                total_params += np.prod(block.se.fc1_weight.shape)
                total_params += np.prod(block.se.fc1_bias.shape)
                total_params += np.prod(block.se.fc2_weight.shape)
                total_params += np.prod(block.se.fc2_bias.shape)
            
            total_params += np.prod(block.project_conv['weight'].shape)
            total_params += np.prod(block.project_conv['bias'].shape)
        
        # Final conv
        total_params += np.prod(self.final_conv['weight'].shape)
        total_params += np.prod(self.final_conv['bias'].shape)
        
        # FC layer
        total_params += np.prod(self.fc_weight.shape)
        total_params += np.prod(self.fc_bias.shape)
        
        return total_params


class CompoundScaling:
    """Compound scaling methodology for EfficientNet variants"""
    
    @staticmethod
    def get_scaling_coefficients(phi: float) -> Tuple[float, float, int]:
        """Get scaling coefficients for given phi"""
        # EfficientNet scaling coefficients
        alpha = 1.2  # depth
        beta = 1.1   # width  
        gamma = 1.15 # resolution
        
        # Ensure constraint: alpha * beta^2 * gamma^2 ≈ 2
        depth_coefficient = alpha ** phi
        width_coefficient = beta ** phi
        resolution = int(224 * (gamma ** phi))
        
        return depth_coefficient, width_coefficient, resolution
    
    @staticmethod
    def create_efficientnet_variant(phi: float, num_classes: int = 1000) -> EfficientNet:
        """Create EfficientNet variant with compound scaling"""
        depth_coeff, width_coeff, resolution = CompoundScaling.get_scaling_coefficients(phi)
        
        return EfficientNet(
            width_coefficient=width_coeff,
            depth_coefficient=depth_coeff,
            resolution=resolution,
            num_classes=num_classes
        )


def create_efficientnet_variants() -> Dict[str, EfficientNet]:
    """Create standard EfficientNet variants (B0-B7)"""
    variants = {}
    
    scaling_configs = [
        ('B0', 0),
        ('B1', 0.5),
        ('B2', 1),
        ('B3', 2),
        ('B4', 3),
        ('B5', 4),
        ('B6', 5),
        ('B7', 6)
    ]
    
    for name, phi in scaling_configs:
        variants[f'EfficientNet-{name}'] = CompoundScaling.create_efficientnet_variant(phi)
    
    return variants


def analyze_compound_scaling(input_shape: Tuple[int, int, int, int]) -> Dict:
    """Analyze the effects of compound scaling"""
    results = {
        'single_dimension_scaling': {},
        'compound_scaling': {},
        'efficiency_comparison': {}
    }
    
    base_model = EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, resolution=224)
    base_params = base_model.count_parameters()
    
    # Test single dimension scaling
    scaling_factors = [1.0, 1.2, 1.5, 2.0]
    
    for factor in scaling_factors:
        # Width scaling only
        width_model = EfficientNet(width_coefficient=factor, depth_coefficient=1.0, resolution=224)
        results['single_dimension_scaling'][f'width_{factor}'] = {
            'parameters': width_model.count_parameters(),
            'efficiency': width_model.count_parameters() / base_params
        }
        
        # Depth scaling only
        depth_model = EfficientNet(width_coefficient=1.0, depth_coefficient=factor, resolution=224)
        results['single_dimension_scaling'][f'depth_{factor}'] = {
            'parameters': depth_model.count_parameters(),
            'efficiency': depth_model.count_parameters() / base_params
        }
    
    # Test compound scaling
    phi_values = [0, 0.5, 1.0, 1.5, 2.0]
    for phi in phi_values:
        compound_model = CompoundScaling.create_efficientnet_variant(phi)
        results['compound_scaling'][f'phi_{phi}'] = {
            'parameters': compound_model.count_parameters(),
            'efficiency': compound_model.count_parameters() / base_params,
            'coefficients': CompoundScaling.get_scaling_coefficients(phi)
        }
    
    return results


def progressive_resizing_training(model: EfficientNet, initial_size: int = 128, 
                                final_size: int = 224, steps: int = 4) -> List[Dict]:
    """Simulate progressive resizing training strategy"""
    training_schedule = []
    
    sizes = np.linspace(initial_size, final_size, steps).astype(int)
    
    for step, size in enumerate(sizes):
        # Simulate training phase
        phase_info = {
            'step': step + 1,
            'image_size': size,
            'estimated_memory': size * size * 3 * 4,  # Simplified memory estimate
            'training_time_relative': (size / initial_size) ** 2,  # Quadratic with image size
            'suggested_epochs': max(1, 10 - step * 2)  # Fewer epochs for larger images
        }
        training_schedule.append(phase_info)
    
    return training_schedule


def mobile_optimization_analysis(model: EfficientNet) -> Dict:
    """Analyze mobile optimization aspects of EfficientNet"""
    analysis = {
        'parameter_efficiency': {},
        'operation_types': {},
        'memory_optimization': {},
        'quantization_readiness': {}
    }
    
    # Parameter efficiency
    total_params = model.count_parameters()
    analysis['parameter_efficiency'] = {
        'total_parameters': total_params,
        'size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'mobile_friendly': total_params < 10_000_000  # < 10M parameters
    }
    
    # Operation analysis
    depthwise_ops = 0
    pointwise_ops = 0
    standard_conv_ops = 0
    
    for block in model.blocks:
        depthwise_ops += 1  # Each MBConv has depthwise
        pointwise_ops += 2  # Expansion + projection
    
    standard_conv_ops = 2  # Initial + final conv
    
    analysis['operation_types'] = {
        'depthwise_separable': depthwise_ops,
        'pointwise': pointwise_ops,
        'standard_conv': standard_conv_ops,
        'separation_ratio': (depthwise_ops + pointwise_ops) / standard_conv_ops
    }
    
    return analysis


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_depthwise_separable_convolution():
    """
    Exercise 1: Implement and analyze depthwise separable convolution
    
    Tasks:
    1. Complete DepthwiseSeparableConv implementation
    2. Compare with standard convolution
    3. Analyze computational savings
    4. Test on sample inputs
    """
    
    print("=== Exercise 1: Depthwise Separable Convolution ===")
    
    # TODO: Test depthwise separable convolution
    
    # Test input
    batch_size, in_channels, height, width = 2, 64, 32, 32
    test_input = np.random.randn(batch_size, in_channels, height, width)
    
    # Create depthwise separable conv
    out_channels = 128
    dw_conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3)
    
    # Forward pass
    output = dw_conv.forward(test_input, training=False)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate parameter savings
    # Standard conv: k^2 * in_ch * out_ch
    # Depthwise sep: k^2 * in_ch + in_ch * out_ch
    standard_params = 3*3 * in_channels * out_channels
    dw_params = 3*3 * in_channels + in_channels * out_channels
    savings = (standard_params - dw_params) / standard_params
    
    print(f"Standard conv parameters: {standard_params:,}")
    print(f"Depthwise separable parameters: {dw_params:,}")
    print(f"Parameter savings: {savings:.1%}")
    
    assert output.shape == (batch_size, out_channels, height, width), "Incorrect output shape"
    
    pass


def exercise_2_squeeze_excitation():
    """
    Exercise 2: Implement Squeeze-and-Excitation blocks
    
    Tasks:
    1. Complete SqueezeExcitation implementation
    2. Study channel attention patterns
    3. Analyze impact on feature representations
    4. Test with different reduction ratios
    """
    
    print("=== Exercise 2: Squeeze-and-Excitation ===")
    
    # TODO: Test SE block functionality
    
    pass


def exercise_3_mbconv_blocks():
    """
    Exercise 3: Implement MBConv blocks
    
    Tasks:
    1. Complete MBConvBlock implementation
    2. Test different expansion ratios
    3. Analyze skip connections
    4. Study SE integration effects
    """
    
    print("=== Exercise 3: MBConv Blocks ===")
    
    # TODO: Test MBConv functionality
    
    pass


def exercise_4_compound_scaling():
    """
    Exercise 4: Implement compound scaling methodology
    
    Tasks:
    1. Complete CompoundScaling implementation
    2. Create EfficientNet-B0 through B7
    3. Analyze scaling laws
    4. Compare single vs compound scaling
    """
    
    print("=== Exercise 4: Compound Scaling ===")
    
    # TODO: Test compound scaling
    
    pass


def exercise_5_efficientnet_architecture():
    """
    Exercise 5: Build complete EfficientNet
    
    Tasks:
    1. Complete EfficientNet implementation
    2. Test on sample inputs
    3. Verify architecture specifications
    4. Compare with standard implementations
    """
    
    print("=== Exercise 5: Complete EfficientNet ===")
    
    # TODO: Test complete EfficientNet
    
    pass


def exercise_6_mobile_optimization():
    """
    Exercise 6: Analyze mobile optimization features
    
    Tasks:
    1. Study parameter efficiency
    2. Analyze operation types
    3. Implement quantization considerations
    4. Compare with other mobile architectures
    """
    
    print("=== Exercise 6: Mobile Optimization ===")
    
    # TODO: Mobile optimization analysis
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_depthwise_separable_convolution()
    exercise_2_squeeze_excitation()
    exercise_3_mbconv_blocks()
    exercise_4_compound_scaling()
    exercise_5_efficientnet_architecture()
    exercise_6_mobile_optimization()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Compound scaling methodology and efficiency gains")
    print("2. Mobile-optimized building blocks (MBConv, SE)")
    print("3. Parameter efficiency vs standard architectures")
    print("4. Progressive training and optimization strategies")