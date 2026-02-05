"""
Tests for Distributed Optimization (synchronous data-parallel SGD).
"""

import numpy as np

from exercise import (
    PartitionedDataset,
    Worker,
    ParameterServer,
    mean_aggregate,
    weighted_mean_aggregate,
    topk_sparsify,
    synchronous_data_parallel_sgd,
)


def test_mean_aggregate():
    g1 = np.array([1.0, 2.0])
    g2 = np.array([3.0, 4.0])
    np.testing.assert_allclose(mean_aggregate([g1, g2]), np.array([2.0, 3.0]))


def test_weighted_mean_aggregate():
    g1 = np.array([1.0, 2.0])
    g2 = np.array([3.0, 4.0])
    out = weighted_mean_aggregate([g1, g2], [0.25, 0.75])
    np.testing.assert_allclose(out, 0.25 * g1 + 0.75 * g2)


def test_topk_sparsify():
    g = np.array([1.0, -3.0, 2.0, 0.5])
    s = topk_sparsify(g, k=2)
    # Keep -3 and 2
    np.testing.assert_allclose(s, np.array([0.0, -3.0, 2.0, 0.0]))


def test_worker_full_gradient_matches_full_data_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y = rng.normal(size=(20,))
    x = rng.normal(size=(3,))

    w = Worker(X, y)
    g = w.full_gradient(x)

    r = X @ x - y
    g_expected = (2.0 / X.shape[0]) * (X.T @ r)
    np.testing.assert_allclose(g, g_expected)


def test_synchronous_matches_single_worker_mean_gradient():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = rng.normal(size=(40,))

    X1, X2 = X[:20], X[20:]
    y1, y2 = y[:20], y[20:]
    dataset = PartitionedDataset([X1, X2], [y1, y2])

    x0 = np.array([0.0, 0.0])
    lr = 0.1
    traj = synchronous_data_parallel_sgd(dataset, x0, lr=lr, n_steps=1)

    # Equivalent to applying mean of per-worker gradients at x0
    g1 = Worker(X1, y1).full_gradient(x0)
    g2 = Worker(X2, y2).full_gradient(x0)
    g_mean = 0.5 * (g1 + g2)
    np.testing.assert_allclose(traj[1], x0 - lr * g_mean)

