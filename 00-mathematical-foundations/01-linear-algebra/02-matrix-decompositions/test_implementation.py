"""
Tests for Matrix Decompositions.
"""

import numpy as np
import pytest

from exercise import (
    lu_decomposition, solve_lu, determinant_via_lu,
    qr_decomposition, least_squares_qr,
    cholesky_decomposition, solve_cholesky
)


class TestLU:
    def test_lu_reconstruction(self):
        A = np.array([[2.0, 1.0, 1.0],
                      [4.0, -6.0, 0.0],
                      [-2.0, 7.0, 2.0]])
        P, L, U = lu_decomposition(A)
        assert np.allclose(P @ A, L @ U, atol=1e-8)

    def test_solve_lu(self):
        A = np.array([[3.0, 2.0], [1.0, 2.0]])
        b = np.array([5.0, 5.0])
        x = solve_lu(A, b)
        assert np.allclose(A @ x, b, atol=1e-8)

    def test_det_via_lu(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        det = determinant_via_lu(A)
        assert abs(det - np.linalg.det(A)) < 1e-8


class TestQR:
    def test_qr_reconstruction(self):
        A = np.random.randn(6, 3)
        Q, R = qr_decomposition(A, method="householder")
        assert np.allclose(Q @ R, A, atol=1e-8)
        assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)

    def test_least_squares(self):
        m, n = 50, 3
        A = np.random.randn(m, n)
        x_true = np.array([1.0, -2.0, 0.5])
        b = A @ x_true + 0.01 * np.random.randn(m)
        x_hat = least_squares_qr(A, b)
        assert np.linalg.norm(A @ x_hat - b) <= np.linalg.norm(A @ x_true - b) + 1e-3


class TestCholesky:
    def test_cholesky_and_solve(self):
        M = np.random.randn(5, 5)
        A = M @ M.T + 1e-3 * np.eye(5)  # SPD
        L = cholesky_decomposition(A)
        assert np.allclose(A, L @ L.T, atol=1e-8)
        b = np.random.randn(5)
        x = solve_cholesky(A, b)
        assert np.allclose(A @ x, b, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


