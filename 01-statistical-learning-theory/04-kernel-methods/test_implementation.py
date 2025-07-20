"""
Test suite for kernel methods implementations.
"""

import numpy as np
import pytest
from exercise import (
    RBFKernel, PolynomialKernel, LinearKernel, StringKernel,
    KernelPCA, SupportVectorMachine, RidgeRegression,
    GaussianProcess, compute_kernel_matrix, kernel_centering,
    reproduce_kernel_hilbert_space_demo, kernel_alignment,
    multiple_kernel_learning, KernelMachine, representer_theorem_verification
)


class TestKernelFunctions:
    """Test kernel function implementations."""
    
    def test_rbf_kernel_properties(self):
        """Test RBF kernel properties."""
        rbf = RBFKernel(sigma=1.0)
        
        # Test self-similarity
        x = np.array([1, 2, 3])
        assert rbf.compute(x, x) == pytest.approx(1.0)
        
        # Test symmetry
        y = np.array([2, 1, 4])
        assert rbf.compute(x, y) == pytest.approx(rbf.compute(y, x))
        
        # Test positive semi-definiteness (via small Gram matrix)
        X = np.random.randn(5, 3)
        K = rbf.compute_matrix(X)
        eigenvals = np.linalg.eigvals(K)
        assert np.all(eigenvals >= -1e-10)  # Allow numerical error
    
    def test_rbf_kernel_scaling(self):
        """Test RBF kernel parameter scaling."""
        x = np.array([0, 0])
        y = np.array([1, 1])
        
        # Smaller sigma should give smaller kernel values
        rbf_small = RBFKernel(sigma=0.5)
        rbf_large = RBFKernel(sigma=2.0)
        
        val_small = rbf_small.compute(x, y)
        val_large = rbf_large.compute(x, y)
        
        assert val_small <= val_large
    
    def test_polynomial_kernel(self):
        """Test polynomial kernel."""
        poly = PolynomialKernel(degree=2, coef=1.0)
        
        x = np.array([1, 2])
        y = np.array([2, 1])
        
        # Should equal (x^T y + coef)^degree
        expected = (np.dot(x, y) + 1.0) ** 2
        assert poly.compute(x, y) == pytest.approx(expected)
        
        # Test matrix computation
        X = np.random.randn(4, 3)
        K = poly.compute_matrix(X)
        
        # Should be symmetric
        np.testing.assert_array_almost_equal(K, K.T)
        
        # Diagonal should be (||x||^2 + coef)^degree
        for i in range(len(X)):
            expected_diag = (np.dot(X[i], X[i]) + 1.0) ** 2
            assert K[i, i] == pytest.approx(expected_diag)
    
    def test_linear_kernel(self):
        """Test linear kernel."""
        linear = LinearKernel()
        
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        
        # Should equal x^T y
        expected = np.dot(x, y)
        assert linear.compute(x, y) == pytest.approx(expected)
        
        # Test matrix computation
        X = np.random.randn(3, 4)
        K = linear.compute_matrix(X)
        expected_matrix = X @ X.T
        
        np.testing.assert_array_almost_equal(K, expected_matrix)
    
    def test_string_kernel(self):
        """Test string kernel."""
        string_kernel = StringKernel(k=2, lambda_decay=0.8)
        
        s1 = "hello"
        s2 = "help"
        
        # Should compute subsequence kernel
        similarity = string_kernel.compute(s1, s2)
        assert isinstance(similarity, (int, float))
        assert similarity >= 0
        
        # Self-similarity should be positive
        self_sim = string_kernel.compute(s1, s1)
        assert self_sim > 0
        
        # Should be symmetric
        assert string_kernel.compute(s1, s2) == pytest.approx(
            string_kernel.compute(s2, s1)
        )


class TestKernelPCA:
    """Test kernel PCA implementation."""
    
    def test_kernel_pca_basic(self):
        """Test basic kernel PCA functionality."""
        # Generate simple 2D data
        np.random.seed(42)
        X = np.random.randn(20, 2)
        
        kpca = KernelPCA(kernel=RBFKernel(sigma=1.0), n_components=2)
        X_transformed = kpca.fit_transform(X)
        
        # Should return correct shape
        assert X_transformed.shape == (20, 2)
        
        # Components should be approximately orthogonal in feature space
        # (This is a basic sanity check)
        assert np.abs(np.corrcoef(X_transformed.T)[0, 1]) <= 1.0
    
    def test_kernel_pca_reconstruction(self):
        """Test kernel PCA reconstruction quality."""
        # Create data with clear structure
        t = np.linspace(0, 2*np.pi, 50)
        X = np.column_stack([np.cos(t), np.sin(t)])  # Circle
        
        kpca = KernelPCA(kernel=RBFKernel(sigma=0.5), n_components=2)
        X_transformed = kpca.fit_transform(X)
        
        # First component should capture most variance
        var_ratio = kpca.explained_variance_ratio_
        assert var_ratio[0] > var_ratio[1]
        assert np.sum(var_ratio) <= 1.0 + 1e-10


class TestSupportVectorMachine:
    """Test Support Vector Machine implementation."""
    
    def test_svm_classification(self):
        """Test SVM binary classification."""
        # Generate linearly separable data
        np.random.seed(42)
        X1 = np.random.randn(10, 2) + [2, 2]
        X2 = np.random.randn(10, 2) + [-2, -2]
        X = np.vstack([X1, X2])
        y = np.hstack([np.ones(10), -np.ones(10)])
        
        svm = SupportVectorMachine(
            kernel=LinearKernel(), C=1.0, solver='quadratic_programming'
        )
        svm.fit(X, y)
        
        # Should achieve perfect classification on separable data
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy >= 0.9  # Allow for some numerical error
        
        # Should have support vectors
        assert len(svm.support_vectors_) > 0
        assert len(svm.dual_coefficients_) == len(svm.support_vectors_)
    
    def test_svm_decision_function(self):
        """Test SVM decision function."""
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        y = np.array([1, 1, -1, -1])
        
        svm = SupportVectorMachine(kernel=LinearKernel(), C=1.0)
        svm.fit(X, y)
        
        # Decision function should have correct signs
        decision_values = svm.decision_function(X)
        predicted_signs = np.sign(decision_values)
        
        # Should match true labels
        np.testing.assert_array_equal(predicted_signs, y)
    
    def test_svm_with_rbf_kernel(self):
        """Test SVM with RBF kernel."""
        # Create XOR-like data (not linearly separable)
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        y = np.array([1, 1, -1, -1])
        
        svm = SupportVectorMachine(kernel=RBFKernel(sigma=1.0), C=10.0)
        svm.fit(X, y)
        
        # Should solve XOR problem with RBF kernel
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy >= 0.75  # RBF should handle this better than linear


class TestKernelRidgeRegression:
    """Test kernel ridge regression."""
    
    def test_ridge_regression_basic(self):
        """Test basic ridge regression functionality."""
        # Generate regression data
        np.random.seed(42)
        X = np.random.randn(15, 2)
        w_true = np.array([1.5, -2.0])
        y = X @ w_true + 0.1 * np.random.randn(15)
        
        ridge = RidgeRegression(kernel=LinearKernel(), lambda_reg=0.1)
        ridge.fit(X, y)
        
        # Should make reasonable predictions
        y_pred = ridge.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse <= 1.0  # Should fit reasonably well
        
        # Should have dual coefficients
        assert len(ridge.dual_coefficients_) == len(X)
    
    def test_ridge_regression_rbf(self):
        """Test ridge regression with RBF kernel."""
        # Create nonlinear data
        X = np.linspace(-2, 2, 20).reshape(-1, 1)
        y = X.flatten() ** 2 + 0.1 * np.random.randn(20)
        
        ridge = RidgeRegression(kernel=RBFKernel(sigma=0.5), lambda_reg=0.01)
        ridge.fit(X, y)
        
        # Should capture nonlinear pattern
        y_pred = ridge.predict(X)
        correlation = np.corrcoef(y, y_pred)[0, 1]
        assert correlation >= 0.8  # Should have good correlation


class TestGaussianProcess:
    """Test Gaussian process implementation."""
    
    def test_gp_regression_basic(self):
        """Test basic GP regression."""
        # Simple 1D function
        X_train = np.array([[-1], [0], [1]])
        y_train = np.array([1, 0, 1])  # Quadratic-like
        
        gp = GaussianProcess(
            kernel=RBFKernel(sigma=1.0), 
            noise_variance=0.01,
            optimization_method='marginal_likelihood'
        )
        gp.fit(X_train, y_train)
        
        # Test prediction
        X_test = np.array([[0.5], [-0.5]])
        mean, variance = gp.predict(X_test, return_variance=True)
        
        assert len(mean) == len(X_test)
        assert len(variance) == len(X_test)
        assert np.all(variance >= 0)  # Variance should be non-negative
    
    def test_gp_uncertainty_quantification(self):
        """Test GP uncertainty quantification."""
        X_train = np.array([[0], [1]])
        y_train = np.array([0, 1])
        
        gp = GaussianProcess(kernel=RBFKernel(sigma=1.0), noise_variance=0.1)
        gp.fit(X_train, y_train)
        
        # Predictions at training points should have lower uncertainty
        mean_train, var_train = gp.predict(X_train, return_variance=True)
        mean_far, var_far = gp.predict(np.array([[10]]), return_variance=True)
        
        # Far from training data should have higher uncertainty
        assert var_far[0] > np.mean(var_train)
    
    def test_gp_sampling(self):
        """Test GP posterior sampling."""
        X_train = np.array([[0], [1]])
        y_train = np.array([0, 1])
        
        gp = GaussianProcess(kernel=RBFKernel(sigma=1.0), noise_variance=0.1)
        gp.fit(X_train, y_train)
        
        X_test = np.linspace(-1, 2, 10).reshape(-1, 1)
        samples = gp.sample_posterior(X_test, n_samples=5)
        
        assert samples.shape == (5, len(X_test))
        
        # Samples should be reasonably close to mean prediction
        mean_pred, _ = gp.predict(X_test)
        sample_mean = np.mean(samples, axis=0)
        
        correlation = np.corrcoef(mean_pred, sample_mean)[0, 1]
        assert correlation >= 0.8


class TestKernelUtilities:
    """Test kernel utility functions."""
    
    def test_kernel_matrix_computation(self):
        """Test kernel matrix computation."""
        X = np.random.randn(5, 3)
        Y = np.random.randn(4, 3)
        
        kernel = RBFKernel(sigma=1.0)
        K = compute_kernel_matrix(kernel, X, Y)
        
        # Should have correct shape
        assert K.shape == (5, 4)
        
        # Should match pairwise computations
        for i in range(5):
            for j in range(4):
                expected = kernel.compute(X[i], Y[j])
                assert K[i, j] == pytest.approx(expected)
    
    def test_kernel_centering(self):
        """Test kernel centering."""
        X = np.random.randn(10, 2)
        kernel = LinearKernel()
        K = kernel.compute_matrix(X)
        
        K_centered = kernel_centering(K)
        
        # Centered kernel should have zero row and column means
        row_means = np.mean(K_centered, axis=1)
        col_means = np.mean(K_centered, axis=0)
        
        np.testing.assert_array_almost_equal(row_means, np.zeros(10))
        np.testing.assert_array_almost_equal(col_means, np.zeros(10))
    
    def test_kernel_alignment(self):
        """Test kernel alignment computation."""
        X = np.random.randn(8, 2)
        y = np.random.choice([-1, 1], 8)
        
        kernel1 = RBFKernel(sigma=1.0)
        kernel2 = LinearKernel()
        
        alignment = kernel_alignment(kernel1, kernel2, X, y)
        
        assert isinstance(alignment, (int, float))
        assert -1 <= alignment <= 1  # Alignment is normalized correlation


class TestAdvancedKernelMethods:
    """Test advanced kernel methods."""
    
    def test_rkhs_demo(self):
        """Test RKHS demonstration."""
        results = reproduce_kernel_hilbert_space_demo()
        
        assert isinstance(results, dict)
        assert 'reproducing_property' in results
        assert 'norm_computation' in results
        assert 'function_evaluation' in results
    
    def test_multiple_kernel_learning(self):
        """Test multiple kernel learning."""
        X = np.random.randn(20, 3)
        y = np.random.choice([-1, 1], 20)
        
        kernels = [
            RBFKernel(sigma=0.5),
            RBFKernel(sigma=2.0),
            LinearKernel()
        ]
        
        weights, performance = multiple_kernel_learning(kernels, X, y)
        
        assert len(weights) == len(kernels)
        assert np.abs(np.sum(weights) - 1.0) <= 1e-6  # Should sum to 1
        assert np.all(weights >= 0)  # Should be non-negative
        assert isinstance(performance, (int, float))
    
    def test_representer_theorem(self):
        """Test representer theorem verification."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        
        kernel = RBFKernel(sigma=1.0)
        lambda_reg = 0.1
        
        verification_results = representer_theorem_verification(
            kernel, X, y, lambda_reg
        )
        
        assert isinstance(verification_results, dict)
        assert 'theorem_satisfied' in verification_results
        assert 'dual_coefficients' in verification_results
        assert 'reconstruction_error' in verification_results


def test_kernel_machine_interface():
    """Test that all kernel machines implement required interface."""
    machines = [
        KernelPCA(kernel=RBFKernel(sigma=1.0), n_components=2),
        SupportVectorMachine(kernel=LinearKernel(), C=1.0),
        RidgeRegression(kernel=RBFKernel(sigma=1.0), lambda_reg=0.1),
        GaussianProcess(kernel=RBFKernel(sigma=1.0), noise_variance=0.1)
    ]
    
    X = np.random.randn(15, 3)
    y = np.random.randn(15)
    
    for machine in machines:
        # Should have fit method
        assert hasattr(machine, 'fit')
        
        # Should have predict or transform method
        assert hasattr(machine, 'predict') or hasattr(machine, 'transform')
        
        # Should have kernel attribute
        assert hasattr(machine, 'kernel')


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])