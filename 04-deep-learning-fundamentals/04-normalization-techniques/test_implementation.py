"""
Deterministic unit tests for normalization layers.

Focuses on correctness of forward normalization invariants and basic backward gradients.
"""

from __future__ import annotations

import numpy as np
import pytest

from exercise import (
    BatchNormalization,
    LayerNormalization,
    GroupNormalization,
    InstanceNormalization,
    RMSNorm,
)


def _finite_diff_scalar(f, eps: float = 1e-6) -> float:
    return (f(+eps) - f(-eps)) / (2.0 * eps)


class TestBatchNorm:
    def test_forward_training_normalizes(self):
        np.random.seed(0)
        x = np.random.randn(32, 10)
        bn = BatchNormalization(10)
        y = bn.forward(x, training=True)
        assert y.shape == x.shape
        # mean ~ beta (0), std ~ gamma (1)
        np.testing.assert_allclose(np.mean(y, axis=0), bn.beta, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(np.std(y, axis=0), bn.gamma, atol=5e-5, rtol=0.0)

    def test_running_stats_update(self):
        np.random.seed(0)
        x = np.random.randn(16, 5)
        bn = BatchNormalization(5, momentum=0.5)
        m0 = bn.running_mean.copy()
        v0 = bn.running_var.copy()
        _ = bn.forward(x, training=True)
        assert not np.allclose(bn.running_mean, m0)
        assert not np.allclose(bn.running_var, v0)

    def test_backward_numeric_single_element(self):
        np.random.seed(0)
        x = np.random.randn(8, 4)
        bn = BatchNormalization(4)
        y = bn.forward(x, training=True)
        dout = np.random.randn(*y.shape)
        dx, grads = bn.backward(dout)
        assert dx.shape == x.shape
        assert set(grads.keys()) == {"gamma", "beta"}

        # finite-diff check one x element for scalar loss = sum(y*dout)
        i, j = 2, 1

        def loss_with_delta(delta: float) -> float:
            x2 = x.copy()
            x2[i, j] += delta
            y2 = bn.forward(x2, training=True)
            return float(np.sum(y2 * dout))

        g_num = _finite_diff_scalar(loss_with_delta)
        np.testing.assert_allclose(dx[i, j], g_num, rtol=2e-3, atol=2e-3)


class TestLayerNorm:
    def test_forward_per_sample_normalizes(self):
        np.random.seed(0)
        x = np.random.randn(32, 10)
        ln = LayerNormalization(10)
        y = ln.forward(x, training=True)
        assert y.shape == x.shape
        # each sample mean ~ 0 and std ~ 1 initially
        np.testing.assert_allclose(np.mean(y, axis=1), 0.0, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(np.std(y, axis=1), 1.0, atol=5e-5, rtol=0.0)

    def test_backward_numeric_single_element(self):
        np.random.seed(0)
        x = np.random.randn(6, 5)
        ln = LayerNormalization(5)
        y = ln.forward(x, training=True)
        dout = np.random.randn(*y.shape)
        dx, grads = ln.backward(dout)
        assert dx.shape == x.shape
        assert set(grads.keys()) == {"gamma", "beta"}

        i, j = 3, 2

        def loss_with_delta(delta: float) -> float:
            x2 = x.copy()
            x2[i, j] += delta
            y2 = ln.forward(x2, training=True)
            return float(np.sum(y2 * dout))

        g_num = _finite_diff_scalar(loss_with_delta)
        np.testing.assert_allclose(dx[i, j], g_num, rtol=2e-3, atol=2e-3)


class TestGroupNorm:
    def test_forward_group_normalizes(self):
        np.random.seed(0)
        x = np.random.randn(16, 12)
        gn = GroupNormalization(12, 3)  # group_size=4
        y = gn.forward(x, training=True)
        assert y.shape == x.shape

        # check each group mean ~0, std~1 (gamma=1,beta=0)
        yg = y.reshape(16, 3, 4)
        np.testing.assert_allclose(np.mean(yg, axis=2), 0.0, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(np.std(yg, axis=2), 1.0, atol=5e-5, rtol=0.0)

    def test_backward_shapes(self):
        np.random.seed(0)
        x = np.random.randn(8, 12)
        gn = GroupNormalization(12, 3)
        y = gn.forward(x, training=True)
        dout = np.random.randn(*y.shape)
        dx, grads = gn.backward(dout)
        assert dx.shape == x.shape
        assert set(grads.keys()) == {"gamma", "beta"}


class TestInstanceNorm:
    def test_forward_normalizes_per_sample(self):
        np.random.seed(0)
        x = np.random.randn(10, 7)
        inn = InstanceNormalization(7)
        y = inn.forward(x, training=True)
        np.testing.assert_allclose(np.mean(y, axis=1), 0.0, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(np.std(y, axis=1), 1.0, atol=5e-5, rtol=0.0)


class TestRMSNorm:
    def test_forward_rms_is_one(self):
        np.random.seed(0)
        x = np.random.randn(9, 11)
        rn = RMSNorm(11)
        y = rn.forward(x, training=True)
        assert y.shape == x.shape
        # rms of (y/gamma) should be ~1
        y0 = y / rn.gamma
        rms = np.sqrt(np.mean(y0 * y0, axis=1))
        np.testing.assert_allclose(rms, 1.0, atol=1e-6, rtol=0.0)

