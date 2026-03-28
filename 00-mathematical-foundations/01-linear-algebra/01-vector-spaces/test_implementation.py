import numpy as np

from exercise import (
    check_linear_independence,
    gram_schmidt,
    low_rank_approximation,
    matrix_condition_number,
    power_method,
    projection_matrix,
)


def test_linear_independence_and_gram_schmidt():
    vectors = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0])]
    assert check_linear_independence(vectors)
    ortho = gram_schmidt(vectors)
    assert len(ortho) == 3
    assert np.allclose(np.column_stack(ortho).T @ np.column_stack(ortho), np.eye(3), atol=1e-6)


def test_projection_matrix_properties():
    basis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    proj = projection_matrix(basis)
    assert np.allclose(proj @ proj, proj, atol=1e-6)
    assert np.allclose(proj, proj.T, atol=1e-6)


def test_power_method_and_condition_number():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    eigenvalue, eigenvector = power_method(A)
    assert np.allclose(A @ eigenvector, eigenvalue * eigenvector, atol=1e-5)
    assert matrix_condition_number(np.eye(2)) == 1.0


def test_low_rank_approximation_shape():
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    approx = low_rank_approximation(A, rank=1)
    assert approx.shape == A.shape
    assert np.linalg.matrix_rank(approx) == 1
