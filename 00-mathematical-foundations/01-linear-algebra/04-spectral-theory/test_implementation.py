import numpy as np

from exercise import gershgorin_disks, inverse_iteration, power_iteration, symmetric_eigendecomposition


def test_symmetric_eigendecomposition_reconstructs():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigenvalues, eigenvectors = symmetric_eigendecomposition(A)
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    assert np.allclose(reconstructed, A, atol=1e-6)


def test_power_and_inverse_iteration():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam1, v1 = power_iteration(A)
    lam2, v2 = inverse_iteration(A, mu=1.5)
    assert np.allclose(A @ v1, lam1 * v1, atol=1e-6)
    assert np.allclose(A @ v2, lam2 * v2, atol=1e-6)


def test_gershgorin_disks_count():
    A = np.array([[3.0, -1.0], [2.0, 4.0]])
    disks = gershgorin_disks(A)
    assert len(disks) == 2
