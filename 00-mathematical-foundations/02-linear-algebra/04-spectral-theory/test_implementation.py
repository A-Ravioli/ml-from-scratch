"""
Tests for Spectral Theory utilities.
"""

import numpy as np
import pytest

from exercise import (
    symmetric_eigendecomposition, diagonalize,
    power_iteration, inverse_iteration, rayleigh_quotient, rayleigh_quotient_iteration,
    gershgorin_disks, matrix_function_via_eigen, graph_laplacian_spectrum
)


class TestSpectralDecompositions:
    def test_symmetric_decomposition(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        vals, vecs = symmetric_eigendecomposition(A)
        # Reconstruct
        A_rec = vecs @ np.diag(vals) @ vecs.T
        assert np.allclose(A, A_rec, atol=1e-8)
        # Orthonormality
        assert np.allclose(vecs.T @ vecs, np.eye(2), atol=1e-8)

    def test_diagonalize(self):
        A = np.array([[1.0, 1.0], [0.0, 2.0]])
        V, D = diagonalize(A)
        A_rec = V @ D @ np.linalg.inv(V)
        assert np.allclose(A, A_rec, atol=1e-8)


class TestIterativeMethods:
    def test_power_and_rayleigh(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        lam, v = power_iteration(A)
        assert np.allclose(A @ v, lam * v, atol=1e-6)
        rq = rayleigh_quotient(A, v)
        assert abs(rq - lam) < 1e-6

    def test_inverse_and_rqi(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        lam, v = inverse_iteration(A, mu=1.5)
        assert np.allclose(A @ v, lam * v, atol=1e-6)
        mu, x = rayleigh_quotient_iteration(A, np.array([1.0, 0.0]))
        assert np.allclose(A @ x, mu * x, atol=1e-6)


class TestGershgorinAndFunctions:
    def test_gershgorin_contains_eigs(self):
        A = np.array([[3.0, -1.0, 0.0], [2.0, 4.0, 1.0], [0.0, -2.0, 1.0]])
        disks = gershgorin_disks(A)
        eigs = np.linalg.eigvals(A)
        # Each eigenvalue should lie in at least one disk
        ok = True
        for lam in eigs:
            if not any(abs(lam - c) <= r + 1e-8 for c, r in disks):
                ok = False
                break
        assert ok

    def test_matrix_function(self):
        A = np.array([[1.0, 0.0], [0.0, 2.0]])
        expA = matrix_function_via_eigen(A, np.exp)
        assert np.allclose(expA, np.diag([np.e, np.e**2]), atol=1e-12)


class TestGraphLaplacian:
    def test_laplacian_spectrum(self):
        # Simple path graph on 3 nodes
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        vals, vecs = graph_laplacian_spectrum(adj)
        # Smallest eigenvalue ~ 0 for connected graph
        assert abs(vals[0]) < 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


