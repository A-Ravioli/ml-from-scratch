"""
Deterministic tests for Neural Tangent Kernel implementations.

Focuses on kernel matrix invariants (symmetry/PSD) and basic finite-width Jacobian correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import NeuralTangentKernel, FiniteWidthNTK, NTKAnalyzer


class TestInfiniteWidthNTK:
    def test_kernel_symmetry_and_psd_relu(self):
        np.random.seed(0)
        X = np.linspace(-1.0, 1.0, 7).reshape(-1, 1)
        ntk = NeuralTangentKernel("relu", depth=2)
        K = ntk.compute_ntk_matrix(X)
        assert K.shape == (7, 7)
        np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1e-10)
        evals = np.linalg.eigvalsh((K + K.T) / 2.0)
        assert np.min(evals) >= -1e-8

    def test_kernel_symmetry_and_psd_tanh(self):
        np.random.seed(0)
        X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ntk = NeuralTangentKernel("tanh", depth=1)
        K = ntk.compute_ntk_matrix(X)
        assert K.shape == (4, 4)
        np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1e-10)
        evals = np.linalg.eigvalsh((K + K.T) / 2.0)
        assert np.min(evals) >= -1e-6

    def test_cross_matrix_transpose(self):
        X = np.linspace(-1.0, 1.0, 6).reshape(-1, 1)
        X1 = X[:2]
        X2 = X[2:]
        ntk = NeuralTangentKernel("relu", depth=1)
        K12 = ntk.compute_ntk_matrix(X1, X2)
        K21 = ntk.compute_ntk_matrix(X2, X1)
        np.testing.assert_allclose(K12, K21.T, rtol=0.0, atol=1e-10)


class TestFiniteWidthNTK:
    def test_finite_ntk_symmetry(self):
        np.random.seed(0)
        X = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 1.0]])
        finite = FiniteWidthNTK(width=32, activation_fn="relu", depth=1)
        K = finite.compute_finite_ntk(X)
        assert K.shape == (3, 3)
        np.testing.assert_allclose(K, K.T, rtol=0.0, atol=1e-8)

    def test_parameter_gradient_matches_finite_difference(self):
        np.random.seed(0)
        X = np.array([[0.2, -0.1]])
        net = FiniteWidthNTK(width=16, activation_fn="tanh", depth=1)
        # initialize weights deterministically
        _ = net.forward(X[0])

        # pick a single weight element in first layer
        W0 = net.weights[0]
        i, j = 3, 1

        def f_with_delta(delta: float) -> float:
            orig = float(W0[i, j])
            W0[i, j] = orig + delta
            y = net.forward(X[0])
            W0[i, j] = orig
            return float(y)

        eps = 1e-6
        g_num = (f_with_delta(eps) - f_with_delta(-eps)) / (2.0 * eps)

        grad = net._compute_parameter_gradient(X[0])
        # flattening order is per-layer, row-major
        offset0 = 0
        idx = offset0 + i * W0.shape[1] + j
        g_ana = float(grad[idx])
        np.testing.assert_allclose(g_ana, g_num, rtol=2e-3, atol=2e-3)


class TestAnalyzer:
    def test_compare_infinite_finite_returns_shapes(self):
        np.random.seed(0)
        X = np.linspace(-1.0, 1.0, 6).reshape(-1, 1)
        res = NTKAnalyzer().compare_infinite_finite_ntk(X, widths=[8, 16], activation_fn="relu", depth=1)
        assert "infinite_ntk" in res
        assert len(res["ntk_matrices"]) == 2
        assert res["infinite_ntk"].shape == (6, 6)

