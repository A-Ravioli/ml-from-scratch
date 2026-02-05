"""
EfficientNet reference solution.

Completed clone of `exercise.py`'s public API, implemented in NumPy with small, deterministic
components suitable for fast unit tests.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


class DepthwiseSeparableConv:
    """Depthwise separable convolution: depthwise then pointwise, each followed by batch norm + activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        fan_in = self.kernel_size * self.kernel_size
        self.depthwise_weight = np.random.randn(self.in_channels, 1, self.kernel_size, self.kernel_size) * np.sqrt(
            2.0 / max(1, fan_in)
        )

        self.pointwise_weight = np.random.randn(self.out_channels, self.in_channels, 1, 1) * np.sqrt(2.0 / max(1, self.in_channels))
        self.pointwise_bias = np.zeros(self.out_channels, dtype=float)

        self.dw_bn_gamma = np.ones(self.in_channels, dtype=float)
        self.dw_bn_beta = np.zeros(self.in_channels, dtype=float)
        self.dw_running_mean = np.zeros(self.in_channels, dtype=float)
        self.dw_running_var = np.ones(self.in_channels, dtype=float)

        self.pw_bn_gamma = np.ones(self.out_channels, dtype=float)
        self.pw_bn_beta = np.zeros(self.out_channels, dtype=float)
        self.pw_running_mean = np.zeros(self.out_channels, dtype=float)
        self.pw_running_var = np.ones(self.out_channels, dtype=float)

        self.momentum = 0.9
        self.eps = 1e-5
        self.activation = Swish()

    def _batch_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        training: bool = True,
    ) -> np.ndarray:
        if training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            running_mean[:] = self.momentum * running_mean + (1.0 - self.momentum) * batch_mean.squeeze()
            running_var[:] = self.momentum * running_var + (1.0 - self.momentum) * batch_var.squeeze()
        else:
            x_norm = (x - running_mean.reshape(1, -1, 1, 1)) / np.sqrt(running_var.reshape(1, -1, 1, 1) + self.eps)
        return gamma.reshape(1, -1, 1, 1) * x_norm + beta.reshape(1, -1, 1, 1)

    def _simple_conv2d(self, x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None, stride: int = 1, padding: int = 0) -> np.ndarray:
        batch_size, in_channels, in_h, in_w = x.shape
        out_channels, _, k_h, k_w = weight.shape
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        else:
            x_padded = x
        out_h = (x_padded.shape[2] - k_h) // stride + 1
        out_w = (x_padded.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=float)
        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        output[b, oc, i, j] = float(np.sum(x_padded[b, :, h_start : h_start + k_h, w_start : w_start + k_w] * weight[oc]))
        if bias is not None:
            output = output + bias.reshape(1, -1, 1, 1)
        return output

    def _depthwise_conv2d(self, x: np.ndarray, weight: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
        batch_size, in_channels, in_h, in_w = x.shape
        _, _, k_h, k_w = weight.shape
        if padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        else:
            x_padded = x
        out_h = (x_padded.shape[2] - k_h) // stride + 1
        out_w = (x_padded.shape[3] - k_w) // stride + 1
        output = np.zeros((batch_size, in_channels, out_h, out_w), dtype=float)
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride
                        w_start = j * stride
                        patch = x_padded[b, c, h_start : h_start + k_h, w_start : w_start + k_w]
                        output[b, c, i, j] = float(np.sum(patch * weight[c, 0]))
        return output

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        dw = self._depthwise_conv2d(x, self.depthwise_weight, stride=self.stride, padding=self.padding)
        dw = self._batch_norm(dw, self.dw_bn_gamma, self.dw_bn_beta, self.dw_running_mean, self.dw_running_var, training)
        dw = self.activation.forward(dw)

        pw = self._simple_conv2d(dw, self.pointwise_weight, bias=self.pointwise_bias, stride=1, padding=0)
        pw = self._batch_norm(pw, self.pw_bn_gamma, self.pw_bn_beta, self.pw_running_mean, self.pw_running_var, training)
        pw = self.activation.forward(pw)
        return pw


class SqueezeExcitation:
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction_ratio: int = 4):
        self.channels = int(channels)
        self.reduction_ratio = int(reduction_ratio)
        self.reduced_channels = max(1, self.channels // self.reduction_ratio)
        self.fc1_weight = np.random.randn(self.reduced_channels, self.channels) * np.sqrt(2.0 / max(1, self.channels))
        self.fc1_bias = np.zeros(self.reduced_channels, dtype=float)
        self.fc2_weight = np.random.randn(self.channels, self.reduced_channels) * np.sqrt(2.0 / max(1, self.reduced_channels))
        self.fc2_bias = np.zeros(self.channels, dtype=float)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        s = np.mean(x, axis=(2, 3))  # (B, C)
        z = np.maximum(0.0, s @ self.fc1_weight.T + self.fc1_bias)
        a = self._sigmoid(z @ self.fc2_weight.T + self.fc2_bias)  # (B, C)
        return x * a.reshape(a.shape[0], a.shape[1], 1, 1)


class Swish:
    """Swish activation: x * sigmoid(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x / (1.0 + np.exp(-x))


class MBConvBlock:
    """
    Mobile Inverted Bottleneck Convolution block: expand -> depthwise -> SE -> project (+ optional skip).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion_ratio: int = 6,
        se_ratio: float = 0.25,
        kernel_size: int = 3,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.expansion_ratio = int(expansion_ratio)
        self.se_ratio = float(se_ratio)
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size - 1) // 2

        expanded_channels = self.in_channels * self.expansion_ratio
        self.use_expansion = self.expansion_ratio != 1
        if self.use_expansion:
            self.expand = DepthwiseSeparableConv(self.in_channels, expanded_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.expand = None

        self.depthwise = DepthwiseSeparableConv(expanded_channels if self.use_expansion else self.in_channels, expanded_channels if self.use_expansion else self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        se_channels = expanded_channels if self.use_expansion else self.in_channels
        self.use_se = self.se_ratio > 0.0
        self.se = SqueezeExcitation(se_channels, reduction_ratio=max(1, int(round(1.0 / self.se_ratio)))) if self.use_se else None

        self.project = DepthwiseSeparableConv(se_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.use_skip = self.stride == 1 and self.in_channels == self.out_channels

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        identity = x
        out = x
        if self.expand is not None:
            out = self.expand.forward(out, training=training)
        out = self.depthwise.forward(out, training=training)
        if self.se is not None:
            out = self.se.forward(out)
        out = self.project.forward(out, training=training)
        if self.use_skip:
            out = out + identity
        return out


@dataclass
class CompoundScaling:
    width_coefficient: float = 1.0
    depth_coefficient: float = 1.0
    resolution: int = 224
    dropout_rate: float = 0.2

    def scale_channels(self, channels: int) -> int:
        return max(1, int(round(channels * self.width_coefficient)))

    def scale_repeats(self, repeats: int) -> int:
        return max(1, int(math.ceil(repeats * self.depth_coefficient)))


class EfficientNet:
    """
    EfficientNet (toy) with a small, configurable set of MBConv blocks.
    """

    def __init__(
        self,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        resolution: int = 224,
        dropout_rate: float = 0.2,
        num_classes: int = 1000,
        include_top: bool = True,
    ):
        self.scaling = CompoundScaling(width_coefficient, depth_coefficient, int(resolution), float(dropout_rate))
        self.num_classes = int(num_classes)
        self.include_top = bool(include_top)

        # Stem maps RGB -> base channel width.
        stem_channels = self.scaling.scale_channels(16)
        self.stem = DepthwiseSeparableConv(3, stem_channels, kernel_size=3, stride=1, padding=1)

        # A tiny stage configuration: (repeats, in_ch, out_ch, stride, expansion)
        base_stages = [
            (1, 16, 16, 1, 1),
            (1, 16, 24, 2, 6),
        ]

        self.stages: List[List[MBConvBlock]] = []
        final_channels = stem_channels
        for repeats, in_ch, out_ch, stride, exp in base_stages:
            r = self.scaling.scale_repeats(repeats)
            in_scaled = self.scaling.scale_channels(in_ch)
            out_scaled = self.scaling.scale_channels(out_ch)
            stage: List[MBConvBlock] = []
            for j in range(r):
                stage_stride = stride if j == 0 else 1
                stage_in = in_scaled if j == 0 else out_scaled
                stage.append(MBConvBlock(stage_in, out_scaled, stride=stage_stride, expansion_ratio=exp))
            self.stages.append(stage)
            final_channels = out_scaled

        self.head_weight = np.random.randn(self.num_classes, final_channels) * np.sqrt(2.0 / max(1, final_channels))
        self.head_bias = np.zeros(self.num_classes, dtype=float)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.asarray(x, dtype=float)
        out = self.stem.forward(out, training=training)
        for stage in self.stages:
            for block in stage:
                out = block.forward(out, training=training)
        if not self.include_top:
            return out
        pooled = np.mean(out, axis=(2, 3))
        return pooled @ self.head_weight.T + self.head_bias


def create_efficientnet_variants() -> Dict[str, EfficientNet]:
    return {
        "EfficientNet-B0": EfficientNet(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2),
        "EfficientNet-B1": EfficientNet(width_coefficient=1.0, depth_coefficient=1.1, resolution=240, dropout_rate=0.2),
        "EfficientNet-B2": EfficientNet(width_coefficient=1.1, depth_coefficient=1.2, resolution=260, dropout_rate=0.3),
    }


def analyze_compound_scaling(base_model: EfficientNet, scaling_factors: List[Tuple[float, float, int]]) -> Dict:
    analysis: Dict[str, Dict] = {}
    for w, d, r in scaling_factors:
        model = EfficientNet(width_coefficient=w, depth_coefficient=d, resolution=r, dropout_rate=base_model.scaling.dropout_rate, num_classes=base_model.num_classes)
        analysis[f"w={w}_d={d}_r={r}"] = {"num_classes": model.num_classes, "include_top": model.include_top}
    return analysis


def progressive_resizing_training(model: EfficientNet, resolutions: List[int], x: np.ndarray) -> Dict:
    # Toy stub: return the sequence for documentation; no training loop.
    return {"resolutions": list(resolutions), "input_shape": tuple(np.asarray(x).shape)}


def mobile_optimization_analysis(model: EfficientNet) -> Dict:
    # Toy analysis: return a rough parameter estimate from head weights only.
    return {"head_params": int(np.prod(model.head_weight.shape) + np.prod(model.head_bias.shape))}


def exercise_1_depthwise_separable_convolution():
    x = np.random.randn(1, 3, 8, 8)
    layer = DepthwiseSeparableConv(3, 4)
    y = layer.forward(x, training=False)
    return y


def exercise_2_squeeze_excitation():
    x = np.random.randn(2, 4, 5, 5)
    se = SqueezeExcitation(4, reduction_ratio=2)
    return se.forward(x)


def exercise_3_mbconv_blocks():
    x = np.random.randn(1, 3, 8, 8)
    block = MBConvBlock(3, 3, stride=1, expansion_ratio=1)
    return block.forward(x, training=False)


def exercise_4_compound_scaling():
    base = EfficientNet()
    return analyze_compound_scaling(base, scaling_factors=[(1.0, 1.0, 224), (0.5, 0.5, 128)])


def exercise_5_efficientnet_architecture():
    model = EfficientNet(width_coefficient=0.5, depth_coefficient=0.5, resolution=32, dropout_rate=0.0, num_classes=10)
    x = np.random.randn(1, 3, 32, 32)
    return model.forward(x, training=False)


def exercise_6_mobile_optimization():
    model = EfficientNet(width_coefficient=0.5, depth_coefficient=0.5, resolution=32, dropout_rate=0.0, num_classes=10)
    return mobile_optimization_analysis(model)


if __name__ == "__main__":
    start = time.time()
    _ = exercise_5_efficientnet_architecture()
    end = time.time()
    print(f"Completed demo in {end - start:.3f}s")
