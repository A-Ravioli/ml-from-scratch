"""
Deterministic test suite for the Neural Network Theory chapter.

These tests are designed to be fast and to validate core numerical correctness:
- activation function forward/derivative properties
- forward propagation shapes and determinism
- backpropagation gradients (finite-difference check)
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import (
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Swish,
    NeuralNetwork,
)


def _finite_diff(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


class TestActivations:
    def test_sigmoid_properties_and_derivative(self):
        act = Sigmoid()
        x = np.linspace(-10.0, 10.0, 2001)
        y = act.forward(x)

        assert np.all(y > 0.0) and np.all(y < 1.0)
        assert np.all(np.diff(y) >= -1e-12)

        # symmetry: σ(-x) = 1 - σ(x)
        y_neg = act.forward(-x)
        np.testing.assert_allclose(y_neg, 1.0 - y, rtol=0.0, atol=5e-7)

        # derivative check (avoid extreme saturation)
        x_mid = np.linspace(-3.0, 3.0, 257)
        dy_num = _finite_diff(act.forward, x_mid)
        dy_ana = act.derivative(x_mid)
        np.testing.assert_allclose(dy_ana, dy_num, rtol=3e-4, atol=3e-4)

    def test_tanh_properties_and_derivative(self):
        act = Tanh()
        x = np.linspace(-10.0, 10.0, 2001)
        y = act.forward(x)

        assert np.all(y > -1.0) and np.all(y < 1.0)
        assert np.all(np.diff(y) >= -1e-12)

        y_neg = act.forward(-x)
        np.testing.assert_allclose(y_neg, -y, rtol=0.0, atol=1e-12)

        x_mid = np.linspace(-3.0, 3.0, 257)
        dy_num = _finite_diff(act.forward, x_mid)
        dy_ana = act.derivative(x_mid)
        np.testing.assert_allclose(dy_ana, dy_num, rtol=3e-4, atol=3e-4)

    def test_relu_properties_and_derivative(self):
        act = ReLU()
        x = np.array([-3.0, -1.0, -0.1, 0.1, 1.0, 3.0])
        y = act.forward(x)
        assert np.all(y >= 0.0)
        np.testing.assert_allclose(y[:3], 0.0)
        np.testing.assert_allclose(y[3:], x[3:])

        dy = act.derivative(x)
        # derivative convention: 0 for x<0, 1 for x>0 (ignore x==0)
        np.testing.assert_allclose(dy[:3], 0.0)
        np.testing.assert_allclose(dy[3:], 1.0)

    def test_leaky_relu_derivative_finite_difference(self):
        act = LeakyReLU(alpha=0.05)
        x = np.linspace(-2.0, 2.0, 257)
        x = x[np.abs(x) > 1e-3]  # avoid kink
        dy_num = _finite_diff(act.forward, x)
        dy_ana = act.derivative(x)
        np.testing.assert_allclose(dy_ana, dy_num, rtol=5e-4, atol=5e-4)

    def test_swish_derivative_finite_difference(self):
        act = Swish()
        x = np.linspace(-3.0, 3.0, 257)
        dy_num = _finite_diff(act.forward, x)
        dy_ana = act.derivative(x)
        np.testing.assert_allclose(dy_ana, dy_num, rtol=5e-4, atol=5e-4)


class TestNeuralNetwork:
    def test_forward_shapes_and_determinism(self):
        np.random.seed(0)
        net = NeuralNetwork(layer_sizes=[2, 3, 1], activations=[Sigmoid(), Sigmoid()])

        X = np.random.randn(5, 2)
        y1 = net.forward(X)
        y2 = net.forward(X)

        assert y1.shape == (5, 1)
        np.testing.assert_allclose(y1, y2, rtol=0.0, atol=0.0)

    def test_backward_matches_finite_difference(self):
        np.random.seed(0)
        net = NeuralNetwork(layer_sizes=[2, 3, 1], activations=[Tanh(), Sigmoid()])

        X = np.random.randn(4, 2)
        y = (np.random.rand(4, 1) > 0.5).astype(float)

        def loss_value() -> float:
            out = net.forward(X)
            return float(0.5 * np.mean((out - y) ** 2))

        out = net.forward(X)
        w_grads, b_grads = net.backward(X, y, out)

        # Check a few parameters across layers
        checks = [
            ("w0", (0, 0), 0),
            ("w0", (1, 2), 0),
            ("b0", (0,), 0),
            ("w1", (2, 0), 1),
            ("b1", (0,), 1),
        ]

        eps = 1e-6
        base_loss = loss_value()
        assert np.isfinite(base_loss)

        for kind, idx, layer in checks:
            if kind.startswith("w"):
                arr = net.weights[layer]
                grad = w_grads[layer][idx]
            else:
                arr = net.biases[layer]
                grad = b_grads[layer][idx]

            orig = float(arr[idx])

            arr[idx] = orig + eps
            lp = loss_value()
            arr[idx] = orig - eps
            lm = loss_value()
            arr[idx] = orig

            g_num = (lp - lm) / (2.0 * eps)

            assert np.isfinite(g_num)
            assert np.isfinite(grad)
            np.testing.assert_allclose(grad, g_num, rtol=2e-3, atol=2e-3)
