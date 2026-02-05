"""
Residual Networks reference solution.

Completed clone of `exercise.py`'s public API. Uses a naive NumPy convolution for clarity.
"""

from __future__ import annotations

from typing import List

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class ConvLayer:
    """2D Convolutional layer with batch normalization support."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        fan_out = self.out_channels * self.kernel_size * self.kernel_size
        std = np.sqrt(2.0 / fan_out)
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


class BatchNorm2d:
    """2D Batch Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = np.ones(self.num_features, dtype=float)
        self.bias = np.zeros(self.num_features, dtype=float)
        self.running_mean = np.zeros(self.num_features, dtype=float)
        self.running_var = np.ones(self.num_features, dtype=float)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
            self.running_var = (1.0 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_norm = (x - mean) / np.sqrt(var + self.eps)

        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        return weight * x_norm + bias


class BasicBlock:
    """Basic ResNet block for ResNet-18/34."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, 1, stride)
            self.shortcut_bn = BatchNorm2d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        if self.shortcut is not None:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, training)
        out = out + identity
        out = self.relu(out)
        return out

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)


class Bottleneck:
    """Bottleneck block for ResNet-50/101/152."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.stride = int(stride)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        mid = out_channels // 4
        self.conv1 = ConvLayer(in_channels, mid, 1)
        self.bn1 = BatchNorm2d(mid)
        self.conv2 = ConvLayer(mid, mid, 3, stride, padding=1)
        self.bn2 = BatchNorm2d(mid)
        self.conv3 = ConvLayer(mid, out_channels, 1)
        self.bn3 = BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvLayer(in_channels, out_channels, 1, stride)
            self.shortcut_bn = BatchNorm2d(out_channels)
        else:
            self.shortcut = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        identity = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out, training)
        out = self.relu(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training)
        out = self.relu(out)
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training)
        if self.shortcut is not None:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, training)
        out = out + identity
        out = self.relu(out)
        return out

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)


class MaxPool2d:
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)

    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, in_height, in_width = x.shape
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
                constant_values=-np.inf,
            )
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=float)
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
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=(2, 3), keepdims=True)


class LinearLayer:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        self.weight = np.random.normal(0.0, std, (self.out_features, self.in_features))
        self.bias = np.zeros(self.out_features, dtype=float)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight.T + self.bias


class ResNet:
    def __init__(self, block_type: str, layers: List[int], num_classes: int = 1000):
        self.block_type = str(block_type)
        self.num_classes = int(num_classes)
        self.in_channels = 64
        self.conv1 = ConvLayer(3, 64, 7, stride=2, padding=3)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = MaxPool2d(3, stride=2, padding=1)

        if self.block_type == "basic":
            self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
            final_channels = 512
        else:
            self.layer1 = self._make_layer(Bottleneck, 256, layers[0], stride=1)
            self.layer2 = self._make_layer(Bottleneck, 512, layers[1], stride=2)
            self.layer3 = self._make_layer(Bottleneck, 1024, layers[2], stride=2)
            self.layer4 = self._make_layer(Bottleneck, 2048, layers[3], stride=2)
            final_channels = 2048

        self.avgpool = GlobalAvgPool2d()
        self.fc = LinearLayer(final_channels, self.num_classes)

    def _make_layer(self, block_class, out_channels: int, num_blocks: int, stride: int):
        layers: List = []
        layers.append(block_class(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, int(num_blocks)):
            layers.append(block_class(self.in_channels, out_channels, stride=1))
        return layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = self.conv1.forward(x)
        x = self.bn1.forward(x, training)
        x = self.relu(x)
        x = self.maxpool.forward(x)
        for block in self.layer1:
            x = block.forward(x, training)
        for block in self.layer2:
            x = block.forward(x, training)
        for block in self.layer3:
            x = block.forward(x, training)
        for block in self.layer4:
            x = block.forward(x, training)
        x = self.avgpool.forward(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.fc.forward(x)
        return x

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet("basic", [2, 2, 2, 2], num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet("basic", [3, 4, 6, 3], num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet("bottleneck", [3, 4, 6, 3], num_classes)


def exercise_1_basic_block():
    block = BasicBlock(64, 64, stride=1)
    x = np.random.randn(2, 64, 32, 32)
    output = block.forward(x, training=True)
    assert output.shape == x.shape
    return output


def exercise_2_bottleneck_block():
    block = Bottleneck(256, 1024, stride=2)
    x = np.random.randn(2, 256, 32, 32)
    output = block.forward(x, training=True)
    assert output.shape == (2, 1024, 16, 16)
    return output


def exercise_3_resnet18():
    model = resnet18(num_classes=1000)
    x = np.random.randn(1, 3, 224, 224)
    output = model.forward(x, training=False)
    assert output.shape == (1, 1000)
    return output


if __name__ == "__main__":
    exercise_1_basic_block()
    exercise_2_bottleneck_block()
    print("\nResNet implementations completed!")

