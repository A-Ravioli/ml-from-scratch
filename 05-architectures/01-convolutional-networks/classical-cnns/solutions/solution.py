"""
Classical CNN architectures reference solution.

This is a completed clone of `exercise.py`'s public API.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class ConvLayer:
    """2D Convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.weight = np.random.normal(0.0, std, (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = np.zeros(self.out_channels, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, in_channels, in_height, in_width = x.shape
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )

        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype=float)

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
    """Max pooling layer."""

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = int(pool_size)
        self.stride = int(stride)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=float)
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
    """Fully connected layer."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.weight = np.random.normal(0.0, std, (self.out_features, self.in_features))
        self.bias = np.zeros(self.out_features, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T + self.bias


class LeNet5:
    """LeNet-5 architecture."""

    def __init__(self, num_classes: int = 10):
        self.conv1 = ConvLayer(1, 6, 5)
        self.pool1 = PoolingLayer(2, 2)
        self.conv2 = ConvLayer(6, 16, 5)
        self.pool2 = PoolingLayer(2, 2)
        self.fc1 = LinearLayer(16 * 5 * 5, 120)
        self.fc2 = LinearLayer(120, 84)
        self.fc3 = LinearLayer(84, int(num_classes))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.tanh(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.tanh(x)
        x = self.pool2.forward(x)
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
    """AlexNet architecture."""

    def __init__(self, num_classes: int = 1000):
        self.conv1 = ConvLayer(3, 96, 11, stride=4, padding=2)
        self.pool1 = PoolingLayer(3, 2)
        self.conv2 = ConvLayer(96, 256, 5, padding=2)
        self.pool2 = PoolingLayer(3, 2)
        self.conv3 = ConvLayer(256, 384, 3, padding=1)
        self.conv4 = ConvLayer(384, 384, 3, padding=1)
        self.conv5 = ConvLayer(384, 256, 3, padding=1)
        self.pool3 = PoolingLayer(3, 2)
        self.fc1 = LinearLayer(256 * 6 * 6, 4096)
        self.fc2 = LinearLayer(4096, 4096)
        self.fc3 = LinearLayer(4096, int(num_classes))
        self.dropout_rate = 0.5

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.relu(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = self.relu(self.conv2.forward(x))
        x = self.pool2.forward(x)
        x = self.relu(self.conv3.forward(x))
        x = self.relu(self.conv4.forward(x))
        x = self.relu(self.conv5.forward(x))
        x = self.pool3.forward(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1.forward(x))
        if training:
            x = self.dropout(x, self.dropout_rate)
        x = self.relu(self.fc2.forward(x))
        if training:
            x = self.dropout(x, self.dropout_rate)
        x = self.fc3.forward(x)
        return x

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        rate = float(rate)
        if rate == 0.0:
            return x
        mask = np.random.random(x.shape) > rate
        return x * mask / (1.0 - rate)


class VGG16:
    """VGG-16 architecture."""

    def __init__(self, num_classes: int = 1000):
        self.conv1_1 = ConvLayer(3, 64, 3, padding=1)
        self.conv1_2 = ConvLayer(64, 64, 3, padding=1)
        self.pool1 = PoolingLayer(2, 2)
        self.conv2_1 = ConvLayer(64, 128, 3, padding=1)
        self.conv2_2 = ConvLayer(128, 128, 3, padding=1)
        self.pool2 = PoolingLayer(2, 2)
        self.conv3_1 = ConvLayer(128, 256, 3, padding=1)
        self.conv3_2 = ConvLayer(256, 256, 3, padding=1)
        self.conv3_3 = ConvLayer(256, 256, 3, padding=1)
        self.pool3 = PoolingLayer(2, 2)
        self.conv4_1 = ConvLayer(256, 512, 3, padding=1)
        self.conv4_2 = ConvLayer(512, 512, 3, padding=1)
        self.conv4_3 = ConvLayer(512, 512, 3, padding=1)
        self.pool4 = PoolingLayer(2, 2)
        self.conv5_1 = ConvLayer(512, 512, 3, padding=1)
        self.conv5_2 = ConvLayer(512, 512, 3, padding=1)
        self.conv5_3 = ConvLayer(512, 512, 3, padding=1)
        self.pool5 = PoolingLayer(2, 2)
        self.fc1 = LinearLayer(512 * 7 * 7, 4096)
        self.fc2 = LinearLayer(4096, 4096)
        self.fc3 = LinearLayer(4096, int(num_classes))

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.relu(self.conv1_1.forward(x))
        x = self.relu(self.conv1_2.forward(x))
        x = self.pool1.forward(x)
        x = self.relu(self.conv2_1.forward(x))
        x = self.relu(self.conv2_2.forward(x))
        x = self.pool2.forward(x)
        x = self.relu(self.conv3_1.forward(x))
        x = self.relu(self.conv3_2.forward(x))
        x = self.relu(self.conv3_3.forward(x))
        x = self.pool3.forward(x)
        x = self.relu(self.conv4_1.forward(x))
        x = self.relu(self.conv4_2.forward(x))
        x = self.relu(self.conv4_3.forward(x))
        x = self.pool4.forward(x)
        x = self.relu(self.conv5_1.forward(x))
        x = self.relu(self.conv5_2.forward(x))
        x = self.relu(self.conv5_3.forward(x))
        x = self.pool5.forward(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1.forward(x))
        if training:
            x = self.dropout(x, 0.5)
        x = self.relu(self.fc2.forward(x))
        if training:
            x = self.dropout(x, 0.5)
        x = self.fc3.forward(x)
        return x

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def dropout(self, x: np.ndarray, rate: float) -> np.ndarray:
        rate = float(rate)
        if rate == 0.0:
            return x
        mask = np.random.random(x.shape) > rate
        return x * mask / (1.0 - rate)


def exercise_1_lenet():
    print("=== Exercise 1: LeNet-5 ===")
    model = LeNet5(num_classes=10)
    x = np.random.randn(4, 1, 32, 32)
    output = model.forward(x)
    assert output.shape == (4, 10)
    return output


def exercise_2_alexnet():
    print("=== Exercise 2: AlexNet ===")
    model = AlexNet(num_classes=1000)
    x = np.random.randn(2, 3, 224, 224)
    output = model.forward(x, training=False)
    assert output.shape == (2, 1000)
    return output


def exercise_3_vgg16():
    print("=== Exercise 3: VGG-16 ===")
    model = VGG16(num_classes=1000)
    x = np.random.randn(1, 3, 224, 224)
    output = model.forward(x, training=False)
    assert output.shape == (1, 1000)
    return output


if __name__ == "__main__":
    exercise_1_lenet()
    print("\nClassical CNN implementations completed!")

