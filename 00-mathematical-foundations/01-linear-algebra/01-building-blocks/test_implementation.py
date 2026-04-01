"""
Tests for Module 1: The Building Blocks.
"""

import numpy as np

from exercise import (
    basis_vectors,
    benchmark_matmul,
    dot_product,
    generate_special_matrices,
    matmul_triple_loop,
    plot_2d_vectors,
    plot_column_space_2d,
    scalar_multiply,
    unit_vector,
    vector_add,
    verify_special_matrix_properties,
)


def test_basic_vector_operations():
    u = np.array([1.0, -2.0, 3.0])
    v = np.array([4.0, 0.5, -1.0])

    assert np.allclose(vector_add(u, v), np.array([5.0, -1.5, 2.0]))
    assert np.allclose(scalar_multiply(-2.0, u), np.array([-2.0, 4.0, -6.0]))
    assert dot_product(u, v) == np.sum(u * v)


def test_basis_vectors_and_unit_vector():
    basis = basis_vectors(3)
    assert np.allclose(basis, np.eye(3))

    v = np.array([3.0, 4.0])
    u = unit_vector(v)
    assert np.allclose(np.linalg.norm(u), 1.0)
    assert np.allclose(u, np.array([0.6, 0.8]))


def test_triple_loop_matches_numpy():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(3, 5))
    assert np.allclose(matmul_triple_loop(A, B), A @ B)


def test_benchmark_returns_expected_keys():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 8))
    B = rng.normal(size=(8, 8))
    results = benchmark_matmul(A, B, repeats=2)

    assert set(results) == {"loop_seconds", "numpy_seconds", "speedup"}
    assert results["loop_seconds"] > 0.0
    assert results["numpy_seconds"] > 0.0


def test_column_space_visualization_returns_points():
    A = np.array([[1.0, 2.0], [0.0, 1.0]])
    fig, ax, points = plot_column_space_2d(A, samples=5)
    assert points.shape == (25, 2)
    assert fig is not None
    assert ax is not None


def test_special_matrix_properties_hold():
    matrices = generate_special_matrices(4, seed=5)
    checks = verify_special_matrix_properties(matrices)
    assert all(checks.values())


def test_vector_plot_returns_figure():
    fig, ax = plot_2d_vectors([np.array([1.0, 0.0]), np.array([0.0, 1.0])], labels=["e1", "e2"])
    assert fig is not None
    assert ax is not None
