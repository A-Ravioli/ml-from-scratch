"""
Test suite for Vector Spaces and Linear Transformations implementations.
"""

import numpy as np
import pytest
from exercise import (
    VectorSpace, check_linear_independence, find_basis, gram_schmidt,
    projection_matrix, LinearTransformation, eigendecomposition, power_method,
    svd_from_scratch, matrix_condition_number, low_rank_approximation, PCA,
    visualize_linear_transformation, demonstrate_eigenspace_invariance,
    analyze_gradient_descent_convergence
)


class TestVectorSpaceOperations:
    """Test basic vector space operations."""
    
    def test_linear_independence(self):
        """Test linear independence checking."""
        # Linearly independent vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])
        assert check_linear_independence([v1, v2, v3]) == True
        
        # Linearly dependent vectors
        v1 = np.array([1, 0])
        v2 = np.array([2, 0])  # 2*v1
        assert check_linear_independence([v1, v2]) == False
        
        # More complex case
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])
        v3 = np.array([7, 8, 9])  # v3 = -v1 + 2*v2
        assert check_linear_independence([v1, v2, v3]) == False
    
    def test_find_basis(self):
        """Test basis finding."""
        # Redundant set of vectors
        vectors = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0]),  # Linear combination of first two
            np.array([0, 0, 1])
        ]
        basis = find_basis(vectors)
        
        # Should have 3 basis vectors
        assert len(basis) == 3
        
        # Basis should be linearly independent
        assert check_linear_independence(basis) == True
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization."""
        # Simple case
        vectors = [
            np.array([1.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0]),
            np.array([0.0, 1.0, 1.0])
        ]
        
        ortho_vectors = gram_schmidt(vectors, normalize=True)
        
        # Check orthonormality
        for i in range(len(ortho_vectors)):
            for j in range(len(ortho_vectors)):
                dot_product = np.dot(ortho_vectors[i], ortho_vectors[j])
                if i == j:
                    assert abs(dot_product - 1.0) < 1e-10  # Unit norm
                else:
                    assert abs(dot_product) < 1e-10  # Orthogonal
    
    def test_projection_matrix(self):
        """Test projection matrix computation."""
        # Project onto x-y plane in 3D
        basis = [
            np.array([1, 0, 0]),
            np.array([0, 1, 0])
        ]
        
        P = projection_matrix(basis)
        
        # Test projection properties
        # P^2 = P (idempotent)
        assert np.allclose(P @ P, P)
        
        # P is symmetric
        assert np.allclose(P, P.T)
        
        # Test specific projection
        v = np.array([1, 2, 3])
        projected = P @ v
        assert np.allclose(projected, np.array([1, 2, 0]))


class TestLinearTransformations:
    """Test linear transformation operations."""
    
    def test_kernel_finding(self):
        """Test finding kernel of linear transformation."""
        # Transformation that projects onto x-axis
        A = np.array([[1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]])
        
        T = LinearTransformation(matrix=A)
        kernel_basis = T.find_kernel()
        
        # Kernel should be y-z plane
        assert len(kernel_basis) == 2
        
        # Check that kernel vectors are mapped to zero
        for v in kernel_basis:
            assert np.allclose(T.apply(v), np.zeros_like(v))
    
    def test_image_finding(self):
        """Test finding image of linear transformation."""
        # Rank-2 matrix
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        T = LinearTransformation(matrix=A)
        image_basis = T.find_image()
        
        # Image should be 2-dimensional (rank 2)
        assert len(image_basis) == 2
    
    def test_rank_nullity_theorem(self):
        """Test verification of rank-nullity theorem."""
        # 3x3 matrix with rank 2
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        T = LinearTransformation(matrix=A)
        result = T.verify_rank_nullity()
        
        assert result['domain_dim'] == 3
        assert result['kernel_dim'] + result['image_dim'] == result['domain_dim']


class TestEigendecomposition:
    """Test eigendecomposition algorithms."""
    
    def test_power_method(self):
        """Test power method for dominant eigenvalue."""
        # Symmetric matrix with known eigenvalues
        A = np.array([[4, 1], [1, 3]])
        
        eigenval, eigenvec = power_method(A)
        
        # Check eigenvalue equation
        assert np.allclose(A @ eigenvec, eigenval * eigenvec, atol=1e-6)
        
        # Known largest eigenvalue is approximately 4.414
        assert abs(eigenval - 4.414) < 0.01
    
    def test_full_eigendecomposition(self):
        """Test full eigendecomposition."""
        # Symmetric matrix
        A = np.array([[4, -2], [-2, 1]])
        
        eigenvals, eigenvecs = eigendecomposition(A)
        
        # Check each eigenvalue equation
        for i in range(len(eigenvals)):
            v = eigenvecs[:, i]
            assert np.allclose(A @ v, eigenvals[i] * v, atol=1e-6)
        
        # Check orthogonality of eigenvectors (for symmetric matrix)
        assert np.allclose(eigenvecs.T @ eigenvecs, np.eye(2), atol=1e-6)


class TestSVD:
    """Test Singular Value Decomposition."""
    
    def test_svd_basic(self):
        """Test basic SVD functionality."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        
        U, s, Vt = svd_from_scratch(A)
        
        # Reconstruct matrix
        S = np.zeros((U.shape[0], Vt.shape[0]))
        np.fill_diagonal(S, s)
        A_reconstructed = U @ S @ Vt
        
        assert np.allclose(A, A_reconstructed, atol=1e-6)
        
        # Check orthogonality
        assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)
        assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-6)
    
    def test_condition_number(self):
        """Test condition number computation."""
        # Well-conditioned matrix
        A = np.array([[1, 0], [0, 1]])
        assert abs(matrix_condition_number(A) - 1.0) < 1e-10
        
        # Ill-conditioned matrix
        A = np.array([[1, 1], [1, 1.0001]])
        cond = matrix_condition_number(A)
        assert cond > 1000  # Should be large
    
    def test_low_rank_approximation(self):
        """Test low-rank approximation."""
        # Create a rank-2 matrix
        A = np.outer([1, 2, 3], [4, 5]) + np.outer([6, 7, 8], [9, 10])
        
        # Rank-1 approximation
        A1 = low_rank_approximation(A, rank=1)
        assert np.linalg.matrix_rank(A1) == 1
        
        # Rank-2 approximation should be exact
        A2 = low_rank_approximation(A, rank=2)
        assert np.allclose(A, A2, atol=1e-6)


class TestPCA:
    """Test Principal Component Analysis."""
    
    def test_pca_basic(self):
        """Test basic PCA functionality."""
        # Generate correlated data
        np.random.seed(42)
        n_samples = 1000
        mean = [0, 0]
        cov = [[1, 0.8], [0.8, 1]]
        X = np.random.multivariate_normal(mean, cov, n_samples)
        
        # Fit PCA
        pca = PCA(n_components=2)
        pca.fit(X)
        
        # Transform and inverse transform
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        
        # Should be perfect reconstruction with all components
        assert np.allclose(X, X_reconstructed, atol=1e-6)
        
        # Check that principal components are orthogonal
        assert np.allclose(pca.components @ pca.components.T, np.eye(2), atol=1e-6)
    
    def test_pca_variance_explained(self):
        """Test that PCA captures maximum variance."""
        # Generate data with clear principal components
        np.random.seed(42)
        n_samples = 1000
        
        # Data varies more in one direction
        X = np.random.randn(n_samples, 2)
        X[:, 0] *= 3  # More variance in first dimension
        
        # Rotate data
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        X = X @ R.T
        
        pca = PCA(n_components=1)
        pca.fit(X)
        
        # First PC should explain most variance
        total_var = np.var(X, axis=0).sum()
        explained_ratio = pca.explained_variance[0] / total_var
        assert explained_ratio > 0.8


def test_visualizations():
    """Test that visualization functions run without errors."""
    # Test linear transformation visualization
    A = np.array([[2, 1], [0, 1]])
    T = LinearTransformation(matrix=A)
    visualize_linear_transformation(T)
    
    # Test eigenspace invariance demonstration
    A = np.array([[3, 1], [0, 2]])
    demonstrate_eigenspace_invariance(A)
    
    # Test gradient descent convergence analysis
    A = np.array([[10, 0], [0, 1]])  # Ill-conditioned
    b = np.array([1, 1])
    learning_rates = [0.01, 0.1, 0.15]
    analyze_gradient_descent_convergence(A, b, learning_rates)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])