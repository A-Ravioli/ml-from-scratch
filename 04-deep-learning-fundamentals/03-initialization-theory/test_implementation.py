"""
Deterministic tests for initialization theory implementations.

Tests are designed to be fast and avoid long training loops.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import (
    ZeroInitializer,
    RandomNormalInitializer,
    XavierUniformInitializer,
    XavierNormalInitializer,
    HeUniformInitializer,
    HeNormalInitializer,
    LeCunInitializer,
    OrthogonalInitializer,
    VarianceScalingInitializer,
    InitializationAnalyzer,
)


class TestInitializers:
    def test_shapes_and_biases(self):
        np.random.seed(0)
        init = RandomNormalInitializer(std=0.2)
        W = init.initialize_weights(10, 5)
        b = init.initialize_biases(5)
        assert W.shape == (5, 10)
        assert b.shape == (5,)
        assert np.allclose(b, 0.0)

    def test_zero_initializer(self):
        init = ZeroInitializer()
        W = init.initialize_weights(7, 3)
        b = init.initialize_biases(3)
        assert np.all(W == 0.0)
        assert np.all(b == 0.0)

    def test_xavier_uniform_bounds(self):
        np.random.seed(0)
        fan_in, fan_out = 100, 50
        init = XavierUniformInitializer()
        W = init.initialize_weights(fan_in, fan_out)
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        assert np.max(np.abs(W)) <= bound + 1e-12

    def test_xavier_normal_variance(self):
        np.random.seed(0)
        fan_in, fan_out = 200, 100
        init = XavierNormalInitializer()
        W = init.initialize_weights(fan_in, fan_out)
        expected_var = 2.0 / (fan_in + fan_out)
        emp_var = float(np.var(W))
        assert abs(emp_var - expected_var) < 0.25 * expected_var

    def test_he_initialization_stats(self):
        np.random.seed(0)
        fan_in, fan_out = 128, 64

        init_u = HeUniformInitializer()
        Wu = init_u.initialize_weights(fan_in, fan_out)
        bound = np.sqrt(6.0 / fan_in)
        assert np.max(np.abs(Wu)) <= bound + 1e-12

        init_n = HeNormalInitializer()
        Wn = init_n.initialize_weights(fan_in, fan_out)
        expected_var = 2.0 / fan_in
        emp_var = float(np.var(Wn))
        assert abs(emp_var - expected_var) < 0.25 * expected_var

    def test_lecun_variance(self):
        np.random.seed(0)
        fan_in, fan_out = 200, 50
        init = LeCunInitializer()
        W = init.initialize_weights(fan_in, fan_out)
        expected_var = 1.0 / fan_in
        emp_var = float(np.var(W))
        assert abs(emp_var - expected_var) < 0.25 * expected_var

    def test_orthogonal_initializer_square(self):
        np.random.seed(0)
        init = OrthogonalInitializer(gain=1.0)
        W = init.initialize_weights(32, 32)
        prod = W @ W.T
        np.testing.assert_allclose(prod, np.eye(32), rtol=0.0, atol=1e-6)

    def test_orthogonal_initializer_rectangular(self):
        np.random.seed(0)
        init = OrthogonalInitializer(gain=0.5)

        # fan_out < fan_in: rows orthonormal => W W^T = gain^2 I
        W1 = init.initialize_weights(64, 16)
        np.testing.assert_allclose(W1 @ W1.T, (0.5**2) * np.eye(16), rtol=0.0, atol=1e-6)

        # fan_out > fan_in: cols orthonormal => W^T W = gain^2 I
        W2 = init.initialize_weights(16, 64)
        np.testing.assert_allclose(W2.T @ W2, (0.5**2) * np.eye(16), rtol=0.0, atol=1e-6)

    def test_variance_scaling_special_cases(self):
        np.random.seed(0)
        fan_in, fan_out = 120, 30

        # Xavier uniform variance: 2/(fan_in+fan_out) == 1/fan_avg
        vs_xu = VarianceScalingInitializer(scale=1.0, mode="fan_avg", distribution="uniform")
        W = vs_xu.initialize_weights(fan_in, fan_out)
        expected_var = 2.0 / (fan_in + fan_out)
        assert abs(float(np.var(W)) - expected_var) < 0.3 * expected_var

        # He normal variance: 2/fan_in
        vs_hn = VarianceScalingInitializer(scale=2.0, mode="fan_in", distribution="normal")
        W = vs_hn.initialize_weights(fan_in, fan_out)
        expected_var = 2.0 / fan_in
        assert abs(float(np.var(W)) - expected_var) < 0.3 * expected_var

        # LeCun normal variance: 1/fan_in
        vs_ln = VarianceScalingInitializer(scale=1.0, mode="fan_in", distribution="normal")
        W = vs_ln.initialize_weights(fan_in, fan_out)
        expected_var = 1.0 / fan_in
        assert abs(float(np.var(W)) - expected_var) < 0.3 * expected_var


class TestAnalyzer:
    def test_activation_statistics_structure(self):
        np.random.seed(0)
        analyzer = InitializationAnalyzer()
        layer_sizes = [8, 6, 4]
        results = analyzer.analyze_activation_statistics(layer_sizes, np.tanh, n_samples=128)
        assert isinstance(results, dict)
        assert "xavier_uniform" in results

        layer_stats = results["xavier_uniform"]
        assert len(layer_stats) == len(layer_sizes) - 1
        s0 = layer_stats[0]
        assert "pre_activation" in s0 and "post_activation" in s0
        assert "mean" in s0["pre_activation"]
        assert "saturation_rate" in s0["post_activation"]

    def test_gradient_flow_returns_per_layer_norms(self):
        np.random.seed(0)
        analyzer = InitializationAnalyzer()
        layer_sizes = [8, 6, 4]
        tanh = np.tanh
        dtanh = lambda z: 1.0 - np.tanh(z) ** 2
        results = analyzer.analyze_gradient_flow(layer_sizes, tanh, dtanh, n_samples=32)
        assert isinstance(results, dict)
        for name, norms in results.items():
            assert len(norms) == len(layer_sizes) - 1
            assert all(np.isfinite(n) and n >= 0 for n in norms)

