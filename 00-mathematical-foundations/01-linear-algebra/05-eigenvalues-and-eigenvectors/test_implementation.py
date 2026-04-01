"""
Tests for Module 5: Eigenvalues and Eigenvectors.
"""

import numpy as np

from exercise import (
    diagonalize_symmetric,
    eigendecompose,
    is_positive_semidefinite,
    matrix_power_via_eig,
    random_symmetric_matrix,
    transform_vectors,
    verify_spectral_theorem,
)


def test_eigenpairs_satisfy_definition():
    A = np.array([[3.0, 1.0], [0.0, 2.0]])
    eigenvalues, eigenvectors = eigendecompose(A)
    for idx in range(len(eigenvalues)):
        v = eigenvectors[:, idx]
        assert np.allclose(A @ v, eigenvalues[idx] * v)


def test_transform_vectors():
    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    vectors = [np.array([1.0, 1.0]), np.array([2.0, -1.0])]
    transformed = transform_vectors(A, vectors)
    expected = np.column_stack([A @ v for v in vectors])
    assert np.allclose(transformed, expected)


def test_matrix_power_via_eig_matches_numpy():
    A = np.array([[2.0, 1.0], [0.0, 3.0]])
    assert np.allclose(matrix_power_via_eig(A, 4), np.linalg.matrix_power(A, 4))


def test_diagonalize_symmetric_reconstructs_matrix():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    Q, Lambda = diagonalize_symmetric(A)
    assert np.allclose(Q @ Lambda @ Q.T, A)
    assert np.allclose(Q.T @ Q, np.eye(2))


def test_spectral_theorem_checks_pass():
    A = random_symmetric_matrix(4, seed=3)
    checks = verify_spectral_theorem(A)
    assert all(checks.values())


def test_positive_semidefinite_detection():
    A = np.array([[2.0, -1.0], [-1.0, 2.0]])
    assert is_positive_semidefinite(A)

    B = np.array([[1.0, 2.0], [2.0, -1.0]])
    assert not is_positive_semidefinite(B)
