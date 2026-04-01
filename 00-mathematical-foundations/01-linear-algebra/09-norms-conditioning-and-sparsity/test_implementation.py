"""
Tests for Module 9: Norms, Conditioning, and Sparse Matrices.
"""

import numpy as np

from exercise import (
    build_sparse_formats,
    construct_matrix_with_condition_number,
    gradient_descent_quadratic,
    graph_laplacian,
    matrix_norms,
    perturbation_sensitivity,
    sparse_dense_matmul_benchmark,
    sparsity_pattern,
    unit_ball_points,
    vector_norms,
)


def test_vector_and_matrix_norms():
    v = np.array([3.0, -4.0])
    norms = vector_norms(v)
    assert norms["l1"] == 7.0
    assert np.isclose(norms["l2"], 5.0)
    assert norms["linf"] == 4.0

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    m_norms = matrix_norms(A)
    assert np.isclose(m_norms["frobenius"], np.sqrt(30.0))
    assert m_norms["spectral"] >= 0.0


def test_unit_ball_points():
    assert unit_ball_points("l2").shape[1] == 2
    assert unit_ball_points("l1").shape == (5, 2)
    assert unit_ball_points("linf").shape == (5, 2)


def test_constructed_condition_number():
    A = construct_matrix_with_condition_number(4, 100.0, seed=1)
    assert np.isclose(np.linalg.cond(A), 100.0, rtol=1e-5)


def test_perturbation_amplification_is_larger_for_ill_conditioned_matrix():
    b = np.array([1.0, -1.0, 0.5])
    delta_b = np.array([1e-3, -1e-3, 5e-4])
    A_good = construct_matrix_with_condition_number(3, 2.0, seed=2)
    A_bad = construct_matrix_with_condition_number(3, 1e4, seed=2)

    good = perturbation_sensitivity(A_good, b, delta_b)
    bad = perturbation_sensitivity(A_bad, b, delta_b)
    assert bad["amplification"] > good["amplification"]


def test_gradient_descent_is_slower_on_ill_conditioned_quadratic():
    Q_good = np.diag([1.0, 2.0])
    Q_bad = np.diag([1.0, 100.0])
    x0 = np.array([1.0, 1.0])
    losses_good = gradient_descent_quadratic(Q_good, x0, lr=0.5, steps=50)
    losses_bad = gradient_descent_quadratic(Q_bad, x0, lr=0.01, steps=50)
    assert losses_bad[-1] > losses_good[-1]


def test_sparse_formats_and_benchmark():
    A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    formats = build_sparse_formats(A)
    for matrix in formats.values():
        assert np.allclose(matrix.toarray(), A)

    stats = sparse_dense_matmul_benchmark(A, np.array([1.0, 2.0, 3.0]), repeats=2)
    assert stats["sparse_seconds"] > 0.0
    assert stats["dense_seconds"] > 0.0


def test_sparsity_pattern_and_graph_laplacian():
    A = np.array([[1.0, 0.0], [0.0, 2.0]])
    pattern = sparsity_pattern(A)
    assert pattern.shape == (2, 2)

    L = graph_laplacian(3, [(0, 1), (1, 2)])
    assert np.allclose(L.toarray().sum(axis=1), 0.0)
