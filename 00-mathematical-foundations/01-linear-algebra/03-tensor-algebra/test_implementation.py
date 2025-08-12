"""
Tests for Tensor Algebra utilities.
"""

import numpy as np
import pytest

from exercise import (
    kronecker_product, khatri_rao_product, tensor_contract,
    matricize_mode_n, mode_n_product, rank1_approximation
)


class TestProducts:
    def test_kronecker_and_khatri_rao(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[0, 5], [6, 7]])
        K = kronecker_product(A, B)
        assert np.allclose(K, np.kron(A, B))

        A2 = np.array([[1, 2, 3], [4, 5, 6]])
        B2 = np.array([[7, 8, 9], [10, 11, 12]])
        KR = khatri_rao_product(A2, B2)
        # Column-wise kronecker
        KR_ref = np.stack([np.kron(A2[:, i], B2[:, i]) for i in range(A2.shape[1])], axis=1)
        assert np.allclose(KR, KR_ref)


class TestContractions:
    def test_tensor_contract(self):
        A = np.random.randn(3, 4, 5)
        B = np.random.randn(5, 4, 2)
        C = tensor_contract(A, B, axes=([1, 2], [1, 0]))  # sum over dims 4 and 5
        C_ref = np.tensordot(A, B, axes=([1, 2], [1, 0]))
        assert np.allclose(C, C_ref)


class TestModeN:
    def test_matricize_and_mode_product(self):
        X = np.random.randn(3, 4, 5)
        X1 = matricize_mode_n(X, mode=1)
        assert X1.shape == (4, 3*5)

        U = np.random.randn(6, 4)
        Y = mode_n_product(X, U, mode=1)
        assert Y.shape == (3, 6, 5)


class TestRank1:
    def test_rank1_recovery(self):
        a = np.array([1.0, -2.0, 0.5])
        b = np.array([2.0, 1.0])
        c = np.array([0.5, -1.0, 3.0])
        X = np.einsum('i,j,k->ijk', a, b, c)
        f1, f2, f3 = rank1_approximation(X)
        X_hat = np.einsum('i,j,k->ijk', f1, f2, f3)
        # Allow scale/sign ambiguities: compare normalized tensors
        Xn = X / np.linalg.norm(X)
        Xhn = X_hat / np.linalg.norm(X_hat)
        assert np.allclose(Xn, Xhn, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


