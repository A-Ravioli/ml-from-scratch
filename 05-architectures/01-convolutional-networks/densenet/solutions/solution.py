"""
DenseNet reference solution.

This is a completed clone of `exercise.py`'s public API, with deterministic behavior and
without placeholder markers.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class ConvLayer:
    """Basic convolutional layer with batch norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_bias: bool = False,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.use_bias = bool(use_bias)

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        self.weight = np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size) * np.sqrt(
            2.0 / max(1, fan_in)
        )
        if self.use_bias:
            self.bias = np.zeros(self.out_channels, dtype=float)

        self.bn_gamma = np.ones(self.out_channels, dtype=float)
        self.bn_beta = np.zeros(self.out_channels, dtype=float)
        self.running_mean = np.zeros(self.out_channels, dtype=float)
        self.running_var = np.ones(self.out_channels, dtype=float)
        self.momentum = 0.9
        self.eps = 1e-5

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        batch_size, in_c, in_h, in_w = x.shape
        out_h = (in_h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (in_w + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=float)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start = i * self.stride - self.padding
                        j_start = j * self.stride - self.padding
                        patch = np.zeros((in_c, self.kernel_size, self.kernel_size), dtype=float)
                        for ic in range(in_c):
                            for ki in range(self.kernel_size):
                                for kj in range(self.kernel_size):
                                    ii = i_start + ki
                                    jj = j_start + kj
                                    if 0 <= ii < in_h and 0 <= jj < in_w:
                                        patch[ic, ki, kj] = x[b, ic, ii, jj]
                        output[b, out_c, i, j] = float(np.sum(patch * self.weight[out_c]))

        if self.use_bias:
            output = output + self.bias.reshape(1, -1, 1, 1)

        if training:
            batch_mean = np.mean(output, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(output, axis=(0, 2, 3), keepdims=True)
            output_norm = (output - batch_mean) / np.sqrt(batch_var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean.squeeze()
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * batch_var.squeeze()
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            output_norm = (output - mean) / np.sqrt(var + self.eps)

        output_norm = self.bn_gamma.reshape(1, -1, 1, 1) * output_norm + self.bn_beta.reshape(1, -1, 1, 1)
        return np.maximum(0.0, output_norm)


class DenseLayer:
    """
    Single dense layer: Conv(1×1) then Conv(3×3), both implemented via ConvLayer (includes BN+ReLU).
    """

    def __init__(self, in_channels: int, growth_rate: int, bottleneck_factor: int = 4):
        self.in_channels = int(in_channels)
        self.growth_rate = int(growth_rate)
        self.bottleneck_factor = int(bottleneck_factor)

        bottleneck_channels = self.bottleneck_factor * self.growth_rate
        self.bottleneck = ConvLayer(self.in_channels, bottleneck_channels, kernel_size=1, padding=0)
        self.conv = ConvLayer(bottleneck_channels, self.growth_rate, kernel_size=3, padding=1)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        bottleneck_out = self.bottleneck.forward(x, training)
        return self.conv.forward(bottleneck_out, training)


class DenseBlock:
    """Dense block containing multiple dense layers with dense connectivity."""

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int, bottleneck_factor: int = 4):
        self.in_channels = int(in_channels)
        self.num_layers = int(num_layers)
        self.growth_rate = int(growth_rate)
        self.layers: List[DenseLayer] = []

        current_channels = self.in_channels
        for _ in range(self.num_layers):
            layer = DenseLayer(current_channels, self.growth_rate, bottleneck_factor)
            self.layers.append(layer)
            current_channels += self.growth_rate

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        features = [x]
        for layer in self.layers:
            concatenated = np.concatenate(features, axis=1)
            new_features = layer.forward(concatenated, training)
            features.append(new_features)
        return np.concatenate(features, axis=1)

    def get_output_channels(self) -> int:
        return int(self.in_channels + self.num_layers * self.growth_rate)


class TransitionLayer:
    """Transition layer: 1x1 conv then 2x2 average pooling, with optional compression."""

    def __init__(self, in_channels: int, compression_factor: float = 0.5):
        self.in_channels = int(in_channels)
        self.compression_factor = float(compression_factor)
        self.out_channels = int(self.in_channels * self.compression_factor)
        self.conv = ConvLayer(self.in_channels, self.out_channels, kernel_size=1, padding=0)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        conv_out = self.conv.forward(x, training)
        batch_size, channels, height, width = conv_out.shape
        pooled_height = height // 2
        pooled_width = width // 2
        pooled = np.zeros((batch_size, channels, pooled_height, pooled_width), dtype=float)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        region = conv_out[b, c, i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
                        pooled[b, c, i, j] = float(np.mean(region))
        return pooled


class DenseNet:
    """Complete DenseNet architecture: Initial Conv → Dense Blocks (+ Transitions) → Global Pool → FC."""

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: List[int] = [6, 12, 24, 16],
        num_classes: int = 1000,
        compression_factor: float = 0.5,
        initial_channels: int = 64,
    ):
        self.growth_rate = int(growth_rate)
        self.block_config = list(block_config)
        self.num_classes = int(num_classes)
        self.compression_factor = float(compression_factor)

        self.initial_conv = ConvLayer(3, int(initial_channels), kernel_size=7, stride=2, padding=3)

        self.blocks: List[DenseBlock] = []
        self.transitions: List[TransitionLayer] = []
        current_channels = int(initial_channels)

        for i, num_layers in enumerate(self.block_config):
            block = DenseBlock(current_channels, int(num_layers), self.growth_rate)
            self.blocks.append(block)
            current_channels = block.get_output_channels()
            if i < len(self.block_config) - 1:
                transition = TransitionLayer(current_channels, self.compression_factor)
                self.transitions.append(transition)
                current_channels = transition.out_channels

        self.final_channels = int(current_channels)
        self.fc_weight = np.random.randn(self.num_classes, self.final_channels) * np.sqrt(2.0 / max(1, self.final_channels))
        self.fc_bias = np.zeros(self.num_classes, dtype=float)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        features = self.initial_conv.forward(x, training)

        # Max pooling 2x2 (stride 2) after initial conv.
        batch_size, channels, height, width = features.shape
        pooled_height = height // 2
        pooled_width = width // 2
        pooled = np.zeros((batch_size, channels, pooled_height, pooled_width), dtype=float)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        region = features[b, c, i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
                        pooled[b, c, i, j] = float(np.max(region))
        features = pooled

        for i, block in enumerate(self.blocks):
            features = block.forward(features, training)
            if i < len(self.transitions):
                features = self.transitions[i].forward(features, training)

        global_pool = np.mean(features, axis=(2, 3))
        return global_pool @ self.fc_weight.T + self.fc_bias

    def count_parameters(self) -> int:
        total_params = 0
        total_params += int(np.prod(self.initial_conv.weight.shape))
        if hasattr(self.initial_conv, "bias"):
            total_params += int(np.prod(self.initial_conv.bias.shape))
        for block in self.blocks:
            for layer in block.layers:
                total_params += int(np.prod(layer.bottleneck.weight.shape))
                total_params += int(np.prod(layer.conv.weight.shape))
        for transition in self.transitions:
            total_params += int(np.prod(transition.conv.weight.shape))
        total_params += int(np.prod(self.fc_weight.shape))
        total_params += int(np.prod(self.fc_bias.shape))
        return int(total_params)


class MemoryEfficientDenseNet(DenseNet):
    """Memory-efficient DenseNet variant (toy wrapper in this NumPy implementation)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpointing = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return super().forward(x, training)


def create_densenet_variants() -> Dict[str, DenseNet]:
    return {
        "DenseNet-121": DenseNet(growth_rate=32, block_config=[6, 12, 24, 16], num_classes=1000),
        "DenseNet-169": DenseNet(growth_rate=32, block_config=[6, 12, 32, 32], num_classes=1000),
        "DenseNet-201": DenseNet(growth_rate=32, block_config=[6, 12, 48, 32], num_classes=1000),
        "DenseNet-264": DenseNet(growth_rate=32, block_config=[6, 12, 64, 48], num_classes=1000),
    }


def analyze_feature_reuse(dense_block: DenseBlock, input_features: np.ndarray) -> Dict:
    analysis = {"layer_inputs": [], "layer_outputs": [], "feature_similarity": [], "reuse_patterns": []}
    features = [input_features]
    for layer in dense_block.layers:
        concatenated = np.concatenate(features, axis=1)
        analysis["layer_inputs"].append(concatenated.shape[1])
        new_features = layer.forward(concatenated, training=False)
        analysis["layer_outputs"].append(new_features.shape[1])
        features.append(new_features)
    return analysis


def exercise_1_dense_layer_implementation():
    batch_size, in_channels, height, width = 2, 64, 32, 32
    test_input = np.random.randn(batch_size, in_channels, height, width)
    growth_rate = 32
    dense_layer = DenseLayer(in_channels, growth_rate)
    output = dense_layer.forward(test_input, training=False)
    assert output.shape[1] == growth_rate
    return output


def exercise_2_dense_block_analysis():
    x = np.random.randn(1, 16, 8, 8)
    block = DenseBlock(16, num_layers=3, growth_rate=4)
    y = block.forward(x, training=False)
    analysis = analyze_feature_reuse(block, x)
    return {"output_shape": y.shape, "analysis": analysis}


def exercise_3_transition_layers():
    x = np.random.randn(2, 20, 9, 9)
    layer = TransitionLayer(20, compression_factor=0.5)
    y = layer.forward(x, training=False)
    return y


def exercise_4_complete_densenet():
    model = DenseNet(growth_rate=8, block_config=[2, 2], num_classes=10, initial_channels=8)
    x = np.random.randn(1, 3, 32, 32)
    y = model.forward(x, training=False)
    return y


def exercise_5_efficiency_analysis():
    model = DenseNet(growth_rate=8, block_config=[2, 2], num_classes=10, initial_channels=8)
    return {"parameters": model.count_parameters()}


def exercise_6_memory_optimization():
    model = MemoryEfficientDenseNet(growth_rate=8, block_config=[2, 2], num_classes=10, initial_channels=8)
    x = np.random.randn(1, 3, 32, 32)
    y = model.forward(x, training=False)
    return y


if __name__ == "__main__":
    exercise_1_dense_layer_implementation()
    exercise_4_complete_densenet()
    print("\nAll exercises completed!")

