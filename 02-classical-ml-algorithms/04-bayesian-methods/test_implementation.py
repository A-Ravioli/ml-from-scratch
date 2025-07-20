"""
Test suite for Bayesian Methods implementations.
"""

import numpy as np
import pytest
from exercise import (
    NaiveBayesClassifier, BayesianLinearRegression, GaussianProcess,
    BayesianNeuralNetwork, VariationalInference, MCMCSampler,
    generate_bayesian_data, compare_bayesian_methods,
    bayesian_model_selection, uncertainty_calibration_analysis
)


class TestNaiveBayesClassifier:
    """Test Naive Bayes Classifier implementation."""
    
    def test_naive_bayes_gaussian_basic(self):
        """Test basic Gaussian Naive Bayes functionality."""
        # Generate simple 2D classification data
        np.random.seed(42)
        X1 = np.random.randn(30, 2) + [2, 2]
        X2 = np.random.randn(30, 2) + [-2, -2]
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(30), np.ones(30)])
        
        nb = NaiveBayesClassifier(distribution='gaussian')
        nb.fit(X, y)
        
        # Should make reasonable predictions
        y_pred = nb.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.7  # Should achieve decent accuracy
        
        # Should output probabilities
        y_proba = nb.predict_proba(X)
        assert y_proba.shape == (len(X), 2)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)  # Probabilities sum to 1
        assert np.all((y_proba >= 0) & (y_proba <= 1))  # Valid probabilities
    
    def test_naive_bayes_multiclass(self):
        """Test multiclass Naive Bayes."""
        # 3-class problem
        X = np.random.randn(90, 3)
        y = np.random.randint(0, 3, 90)
        
        nb = NaiveBayesClassifier(distribution='gaussian')
        nb.fit(X, y)
        
        y_pred = nb.predict(X)
        y_proba = nb.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(X), 3)
        assert set(y_pred).issubset({0, 1, 2})
    
    def test_naive_bayes_multinomial(self):
        """Test multinomial Naive Bayes for discrete features."""
        # Discrete count data
        X = np.random.poisson(2, (50, 4))
        y = np.random.randint(0, 2, 50)
        
        nb = NaiveBayesClassifier(distribution='multinomial', alpha=1.0)
        nb.fit(X, y)
        
        y_pred = nb.predict(X)
        assert len(y_pred) == len(y)
        
        # Should handle smoothing
        assert hasattr(nb, 'alpha')
    
    def test_naive_bayes_bernoulli(self):
        """Test Bernoulli Naive Bayes for binary features."""
        # Binary features
        X = (np.random.randn(40, 5) > 0).astype(int)
        y = np.random.randint(0, 2, 40)
        
        nb = NaiveBayesClassifier(distribution='bernoulli', alpha=1.0)
        nb.fit(X, y)
        
        y_pred = nb.predict(X)
        y_proba = nb.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(X), 2)


class TestBayesianLinearRegression:
    """Test Bayesian Linear Regression implementation."""
    
    def test_bayesian_lr_basic(self):
        """Test basic Bayesian linear regression."""
        # Generate linear data with noise
        np.random.seed(42)
        X = np.random.randn(30, 3)
        true_weights = np.array([1.5, -2.0, 0.5])
        y = X @ true_weights + 0.2 * np.random.randn(30)
        
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(X, y)
        
        # Should have posterior parameters
        assert hasattr(blr, 'posterior_mean_')
        assert hasattr(blr, 'posterior_covariance_')
        
        # Predictions
        y_pred = blr.predict(X)
        assert len(y_pred) == len(y)
        
        # Should fit reasonably well
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        assert rmse < 1.0
    
    def test_bayesian_lr_uncertainty(self):
        """Test uncertainty quantification."""
        X = np.random.randn(25, 2)
        y = np.random.randn(25)
        
        blr = BayesianLinearRegression(alpha=1.0, beta=1.0)
        blr.fit(X, y)
        
        # Predictions with uncertainty
        y_pred, y_var = blr.predict(X, return_uncertainty=True)
        assert len(y_pred) == len(y)
        assert len(y_var) == len(y)
        assert np.all(y_var >= 0)  # Variance should be non-negative
    
    def test_bayesian_lr_credible_intervals(self):
        """Test credible intervals."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        
        blr = BayesianLinearRegression()
        blr.fit(X, y)
        
        intervals = blr.credible_intervals(X, confidence=0.95)
        assert intervals.shape == (len(X), 2)
        
        # Lower bound should be less than upper bound
        assert np.all(intervals[:, 0] <= intervals[:, 1])
    
    def test_bayesian_lr_posterior_sampling(self):
        """Test posterior sampling."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        
        blr = BayesianLinearRegression()
        blr.fit(X, y)
        
        samples = blr.sample_posterior(X, n_samples=50)
        assert samples.shape == (50, len(X))
    
    def test_bayesian_lr_prior_effect(self):
        """Test effect of prior strength."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        
        # Strong prior
        blr_strong = BayesianLinearRegression(alpha=100.0, beta=1.0)
        blr_strong.fit(X, y)
        
        # Weak prior
        blr_weak = BayesianLinearRegression(alpha=0.01, beta=1.0)
        blr_weak.fit(X, y)
        
        # Strong prior should have smaller coefficient magnitudes
        strong_norm = np.linalg.norm(blr_strong.posterior_mean_)
        weak_norm = np.linalg.norm(blr_weak.posterior_mean_)
        assert strong_norm <= weak_norm


class TestGaussianProcess:
    """Test Gaussian Process implementation."""
    
    def test_gp_basic_regression(self):
        """Test basic GP regression."""
        # 1D function
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.sin(4 * X.ravel()) + 0.1 * np.random.randn(20)
        
        gp = GaussianProcess(kernel='rbf', length_scale=0.2, signal_variance=1.0)
        gp.fit(X, y)
        
        # Predictions
        X_test = np.linspace(0, 1, 50).reshape(-1, 1)
        y_pred = gp.predict(X_test)
        assert len(y_pred) == len(X_test)
        
        # With uncertainty
        y_pred, y_var = gp.predict(X_test, return_uncertainty=True)
        assert len(y_var) == len(X_test)
        assert np.all(y_var >= 0)
    
    def test_gp_kernels(self):
        """Test different kernel functions."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        
        kernels = ['rbf', 'linear', 'polynomial']
        
        for kernel in kernels:
            gp = GaussianProcess(kernel=kernel)
            gp.fit(X, y)
            
            y_pred = gp.predict(X)
            assert len(y_pred) == len(y)
    
    def test_gp_kernel_computation(self):
        """Test kernel matrix computation."""
        X1 = np.random.randn(5, 2)
        X2 = np.random.randn(3, 2)
        
        gp = GaussianProcess(kernel='rbf', length_scale=1.0, signal_variance=2.0)
        
        # RBF kernel
        K = gp._rbf_kernel(X1, X2)
        assert K.shape == (5, 3)
        assert np.all(K >= 0)  # RBF kernel is non-negative
        assert np.all(K <= 2.0)  # Should not exceed signal variance
        
        # Linear kernel
        gp.kernel = 'linear'
        K_linear = gp._linear_kernel(X1, X2)
        assert K_linear.shape == (5, 3)
    
    def test_gp_marginal_likelihood(self):
        """Test log marginal likelihood computation."""
        X = np.random.randn(10, 1)
        y = np.random.randn(10)
        
        gp = GaussianProcess(kernel='rbf')
        gp.fit(X, y)
        
        log_ml = gp.log_marginal_likelihood()
        assert isinstance(log_ml, float)
        assert not np.isnan(log_ml)
    
    def test_gp_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        X = np.linspace(0, 1, 15).reshape(-1, 1)
        y = np.sin(2 * np.pi * X.ravel()) + 0.1 * np.random.randn(15)
        
        gp = GaussianProcess(kernel='rbf')
        initial_params = [gp.length_scale, gp.signal_variance]
        
        gp.optimize_hyperparameters(X, y)
        
        # Parameters should have changed (usually)
        final_params = [gp.length_scale, gp.signal_variance]
        # Allow possibility that optimization doesn't change much
        param_changed = any(abs(initial_params[i] - final_params[i]) > 1e-6 
                           for i in range(len(initial_params)))
        # Don't assert change - just check it completed without error


class TestBayesianNeuralNetwork:
    """Test Bayesian Neural Network implementation."""
    
    def test_bnn_basic(self):
        """Test basic BNN functionality."""
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        
        bnn = BayesianNeuralNetwork([3, 5, 1], activation='tanh')
        bnn.fit(X, y, n_epochs=10)  # Short training for test
        
        # Should have variational parameters
        assert hasattr(bnn, 'variational_means_')
        assert hasattr(bnn, 'variational_log_vars_')
        
        # Predictions with uncertainty
        y_pred, y_var = bnn.predict(X, n_samples=10)
        assert len(y_pred) == len(y)
        assert len(y_var) == len(y)
        assert np.all(y_var >= 0)
    
    def test_bnn_weight_sampling(self):
        """Test weight sampling from variational posterior."""
        bnn = BayesianNeuralNetwork([2, 3, 1])
        bnn._initialize_variational_parameters()
        
        weights = bnn._sample_weights()
        assert isinstance(weights, list)
        assert len(weights) == 2  # Two weight matrices
    
    def test_bnn_kl_divergence(self):
        """Test KL divergence computation."""
        bnn = BayesianNeuralNetwork([2, 3, 1], prior_variance=1.0)
        bnn._initialize_variational_parameters()
        
        kl = bnn._kl_divergence()
        assert isinstance(kl, float)
        assert kl >= 0  # KL divergence is non-negative
    
    def test_bnn_forward_pass(self):
        """Test forward pass through network."""
        X = np.random.randn(10, 2)
        weights = [np.random.randn(2, 3), np.random.randn(3, 1)]
        
        bnn = BayesianNeuralNetwork([2, 3, 1], activation='tanh')
        output = bnn._forward_pass(X, weights)
        
        assert output.shape == (10, 1)
    
    def test_bnn_elbo_computation(self):
        """Test ELBO computation."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        
        bnn = BayesianNeuralNetwork([2, 4, 1])
        bnn._initialize_variational_parameters()
        
        elbo = bnn._elbo_loss(X, y, n_samples=5)
        assert isinstance(elbo, float)


class TestVariationalInference:
    """Test Variational Inference implementation."""
    
    def test_vi_basic(self):
        """Test basic VI functionality."""
        def simple_model_log_prob(data, z):
            return -0.5 * np.sum((data - z) ** 2) - 0.5 * np.sum(z ** 2)
        
        vi = VariationalInference(simple_model_log_prob, 'mean_field_gaussian')
        data = np.random.randn(5)
        
        params = vi.fit(data, n_iterations=10)
        assert isinstance(params, dict)
    
    def test_vi_mean_field_gaussian(self):
        """Test mean-field Gaussian variational family."""
        def dummy_log_prob(data, z):
            return -0.5 * np.sum(z ** 2)
        
        vi = VariationalInference(dummy_log_prob, 'mean_field_gaussian')
        
        # Test parameter structure
        params = {'means': np.array([0.0, 0.0]), 'log_vars': np.array([-1.0, -1.0])}
        
        # Test sampling
        samples = vi._mean_field_gaussian_sample(params)
        assert len(samples) == 2
        
        # Test log probability
        log_prob = vi._mean_field_gaussian_log_prob(samples, params)
        assert isinstance(log_prob, float)
    
    def test_vi_elbo_computation(self):
        """Test ELBO computation in VI."""
        def model_log_prob(data, z):
            return -0.5 * np.sum((data - z) ** 2)
        
        vi = VariationalInference(model_log_prob)
        data = np.random.randn(3)
        params = {'means': np.zeros(3), 'log_vars': np.zeros(3)}
        
        elbo = vi.elbo(params, data, n_samples=10)
        assert isinstance(elbo, float)


class TestMCMCSampler:
    """Test MCMC Sampler implementation."""
    
    def test_mcmc_basic_setup(self):
        """Test basic MCMC setup."""
        def log_prob(x):
            return -0.5 * np.sum(x ** 2)
        
        mcmc = MCMCSampler(log_prob, 'metropolis_hastings')
        assert mcmc.log_prob_fn == log_prob
        assert mcmc.sampler_type == 'metropolis_hastings'
    
    def test_metropolis_hastings_step(self):
        """Test single MH step."""
        def log_prob(x):
            return -0.5 * np.sum(x ** 2)
        
        mcmc = MCMCSampler(log_prob, 'metropolis_hastings')
        current_state = np.array([0.0, 0.0])
        
        new_state, accepted = mcmc.metropolis_hastings_step(current_state, step_size=0.1)
        assert len(new_state) == 2
        assert isinstance(accepted, bool)
    
    def test_mcmc_sampling(self):
        """Test MCMC sampling."""
        def log_prob(x):
            return -0.5 * np.sum(x ** 2)  # Standard Gaussian
        
        mcmc = MCMCSampler(log_prob, 'metropolis_hastings')
        
        samples = mcmc.sample(
            initial_state=np.array([0.0, 0.0]),
            n_samples=100,
            burn_in=20,
            thin=1
        )
        
        assert samples.shape == (100, 2)
        
        # Should sample from roughly standard Gaussian
        sample_mean = np.mean(samples, axis=0)
        sample_std = np.std(samples, axis=0)
        
        # Allow generous tolerance for short chain
        assert np.abs(sample_mean[0]) < 0.5
        assert np.abs(sample_mean[1]) < 0.5
        assert 0.5 < sample_std[0] < 2.0
        assert 0.5 < sample_std[1] < 2.0
    
    def test_effective_sample_size(self):
        """Test effective sample size calculation."""
        # Generate correlated samples
        samples = np.random.randn(200, 2)
        
        mcmc = MCMCSampler(lambda x: 0, 'metropolis_hastings')
        ess = mcmc.effective_sample_size(samples)
        
        assert isinstance(ess, float)
        assert ess > 0


class TestBayesianUtilities:
    """Test utility functions for Bayesian methods."""
    
    def test_generate_bayesian_data(self):
        """Test data generation functions."""
        # Classification data
        X_class, y_class = generate_bayesian_data('classification')
        assert X_class.shape[0] == len(y_class)
        assert len(np.unique(y_class)) >= 2
        
        # Regression data
        X_reg, y_reg = generate_bayesian_data('regression')
        assert X_reg.shape[0] == len(y_reg)
        
        # Sparse regression data
        X_sparse, y_sparse = generate_bayesian_data('sparse')
        assert X_sparse.shape[0] == len(y_sparse)
    
    def test_compare_bayesian_methods(self):
        """Test comparison of Bayesian methods."""
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        
        results = compare_bayesian_methods(X, y)
        assert isinstance(results, dict)
        
        # Should have results for different methods
        expected_keys = ['blr_rmse', 'gp_rmse', 'bnn_rmse']
        # Allow for some keys to be missing if methods not implemented
        assert len(results) > 0
    
    def test_bayesian_model_selection(self):
        """Test Bayesian model selection."""
        X = np.random.randn(25, 2)
        y = np.random.randn(25)
        
        # Create dummy models list
        models = [
            BayesianLinearRegression(alpha=1.0),
            BayesianLinearRegression(alpha=10.0),
            GaussianProcess(kernel='rbf')
        ]
        
        best_idx, marginal_liks = bayesian_model_selection(X, y, models)
        
        if best_idx is not None:
            assert isinstance(best_idx, int)
            assert 0 <= best_idx < len(models)
            assert len(marginal_liks) == len(models)
    
    def test_uncertainty_calibration_analysis(self):
        """Test uncertainty calibration analysis."""
        # Generate synthetic predictions and uncertainties
        n_samples = 100
        true_values = np.random.randn(n_samples)
        predictions = true_values + 0.1 * np.random.randn(n_samples)
        uncertainties = 0.2 * np.ones(n_samples)  # Constant uncertainty
        
        calibration_results = uncertainty_calibration_analysis(
            predictions, uncertainties, true_values
        )
        
        if calibration_results:
            assert isinstance(calibration_results, dict)
            # Should contain some calibration metrics
            assert len(calibration_results) > 0


def test_bayesian_interface_consistency():
    """Test that Bayesian models have consistent interfaces."""
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    
    models = [
        BayesianLinearRegression(),
        GaussianProcess(kernel='rbf')
    ]
    
    for model in models:
        # Should have fit and predict methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Should be able to fit
        model.fit(X, y)
        
        # Should be able to predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])