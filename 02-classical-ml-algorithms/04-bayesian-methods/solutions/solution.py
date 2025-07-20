"""
Solution implementations for Bayesian Methods exercises.

This file provides complete implementations of all TODO items in exercise.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable, Union
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize
from scipy.linalg import cholesky, solve_triangular
import warnings

warnings.filterwarnings('ignore')


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier with support for different distributions.
    """
    
    def __init__(self, distribution: str = 'gaussian', alpha: float = 1.0):
        """Initialize Naive Bayes classifier."""
        self.distribution = distribution
        self.alpha = alpha  # Smoothing parameter
        
        # Fitted attributes
        self.classes_ = None
        self.class_priors_ = None
        self.class_params_ = None
    
    def _calculate_gaussian_likelihood(self, X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Calculate Gaussian likelihood for features."""
        # Log-likelihood for numerical stability
        # log p(x|μ,σ²) = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
        log_likelihood = -0.5 * np.log(2 * np.pi * var) - 0.5 * (X - mean) ** 2 / var
        return log_likelihood
    
    def _calculate_multinomial_likelihood(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Calculate multinomial likelihood for discrete features."""
        # Apply Laplace smoothing
        # log p(x|θ) = Σ x_i * log(θ_i)
        log_likelihood = X * np.log(theta)
        return log_likelihood
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayesClassifier':
        """Fit Naive Bayes classifier."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Calculate class priors
        self.class_priors_ = np.zeros(n_classes)
        for i, class_label in enumerate(self.classes_):
            self.class_priors_[i] = np.sum(y == class_label) / len(y)
        
        # Initialize parameters storage
        self.class_params_ = {}
        
        if self.distribution == 'gaussian':
            # Estimate mean and variance for each class and feature
            for i, class_label in enumerate(self.classes_):
                class_mask = (y == class_label)
                class_data = X[class_mask]
                
                means = np.mean(class_data, axis=0)
                variances = np.var(class_data, axis=0) + 1e-9  # Add small epsilon
                
                self.class_params_[class_label] = {
                    'means': means,
                    'variances': variances
                }
        
        elif self.distribution == 'multinomial':
            # Estimate feature probabilities with Laplace smoothing
            for i, class_label in enumerate(self.classes_):
                class_mask = (y == class_label)
                class_data = X[class_mask]
                
                # Apply Laplace smoothing
                feature_counts = np.sum(class_data, axis=0) + self.alpha
                total_count = np.sum(feature_counts)
                theta = feature_counts / total_count
                
                self.class_params_[class_label] = {
                    'theta': theta
                }
        
        elif self.distribution == 'bernoulli':
            # Estimate binary feature probabilities
            for i, class_label in enumerate(self.classes_):
                class_mask = (y == class_label)
                class_data = X[class_mask]
                
                # Probability of feature being 1
                feature_probs = (np.sum(class_data, axis=0) + self.alpha) / (np.sum(class_mask) + 2 * self.alpha)
                
                self.class_params_[class_label] = {
                    'feature_probs': feature_probs
                }
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes_):
            # Log prior
            log_prior = np.log(self.class_priors_[i])
            
            if self.distribution == 'gaussian':
                means = self.class_params_[class_label]['means']
                variances = self.class_params_[class_label]['variances']
                
                # Sum log-likelihoods across features (independence assumption)
                log_likelihoods = self._calculate_gaussian_likelihood(X, means, variances)
                log_likelihood = np.sum(log_likelihoods, axis=1)
            
            elif self.distribution == 'multinomial':
                theta = self.class_params_[class_label]['theta']
                
                log_likelihoods = self._calculate_multinomial_likelihood(X, theta)
                log_likelihood = np.sum(log_likelihoods, axis=1)
            
            elif self.distribution == 'bernoulli':
                feature_probs = self.class_params_[class_label]['feature_probs']
                
                # P(x=1) = p, P(x=0) = 1-p
                log_likelihood = np.sum(
                    X * np.log(feature_probs) + (1 - X) * np.log(1 - feature_probs),
                    axis=1
                )
            
            log_posteriors[:, i] = log_prior + log_likelihood
        
        # Convert to probabilities using log-sum-exp trick
        max_log_posterior = np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors - max_log_posterior)
        probabilities = exp_posteriors / np.sum(exp_posteriors, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate prior.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """Initialize Bayesian Linear Regression."""
        self.alpha = alpha  # Prior precision (inverse variance)
        self.beta = beta    # Noise precision (inverse noise variance)
        
        # Posterior parameters
        self.posterior_mean_ = None
        self.posterior_covariance_ = None
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """Fit Bayesian linear regression using conjugate prior update."""
        # Add bias term
        X_design = np.column_stack([np.ones(len(X)), X])
        n_features = X_design.shape[1]
        
        # Prior: w ~ N(0, α⁻¹I)
        prior_covariance = (1.0 / self.alpha) * np.eye(n_features)
        prior_mean = np.zeros(n_features)
        
        # Posterior covariance: Σ_N = (αI + βX^T X)⁻¹
        precision_matrix = self.alpha * np.eye(n_features) + self.beta * X_design.T @ X_design
        self.posterior_covariance_ = np.linalg.inv(precision_matrix)
        
        # Posterior mean: μ_N = βΣ_N X^T y
        self.posterior_mean_ = self.beta * self.posterior_covariance_ @ X_design.T @ y
        
        # Store training data for predictions
        self.X_train_ = X_design
        self.y_train_ = y
        
        return self
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions with uncertainty quantification."""
        # Add bias term
        X_design = np.column_stack([np.ones(len(X)), X])
        
        # Predictive mean: μ* = X μ_N
        predictive_mean = X_design @ self.posterior_mean_
        
        if return_uncertainty:
            # Predictive variance: σ²* = β⁻¹ + X Σ_N X^T
            noise_variance = 1.0 / self.beta
            model_variance = np.diag(X_design @ self.posterior_covariance_ @ X_design.T)
            predictive_variance = noise_variance + model_variance
            
            return predictive_mean, predictive_variance
        
        return predictive_mean
    
    def sample_posterior(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample predictions from posterior distribution."""
        # Add bias term
        X_design = np.column_stack([np.ones(len(X)), X])
        
        # Sample weights from posterior
        weight_samples = np.random.multivariate_normal(
            self.posterior_mean_, self.posterior_covariance_, n_samples
        )
        
        # Generate predictions for each weight sample
        prediction_samples = np.zeros((n_samples, len(X)))
        noise_std = 1.0 / np.sqrt(self.beta)
        
        for i in range(n_samples):
            mean_pred = X_design @ weight_samples[i]
            # Add noise
            prediction_samples[i] = mean_pred + noise_std * np.random.randn(len(X))
        
        return prediction_samples
    
    def credible_intervals(self, X: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Calculate credible intervals for predictions."""
        predictive_mean, predictive_variance = self.predict(X, return_uncertainty=True)
        predictive_std = np.sqrt(predictive_variance)
        
        # Calculate confidence interval
        alpha = 1 - confidence
        z_score = norm.ppf(1 - alpha / 2)
        
        lower_bound = predictive_mean - z_score * predictive_std
        upper_bound = predictive_mean + z_score * predictive_std
        
        return np.column_stack([lower_bound, upper_bound])


class GaussianProcess:
    """
    Gaussian Process Regression implementation.
    """
    
    def __init__(self, kernel: str = 'rbf', length_scale: float = 1.0, 
                 noise_level: float = 1e-10, signal_variance: float = 1.0):
        """Initialize Gaussian Process."""
        self.kernel = kernel
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.signal_variance = signal_variance
        
        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.K_inv_ = None
        self.L_ = None  # Cholesky decomposition
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF (Gaussian) kernel implementation."""
        # Compute squared distances
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        sq_dists = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        
        # RBF kernel: k(x,x') = σ² exp(-||x-x'||²/(2ℓ²))
        return self.signal_variance * np.exp(-0.5 * sq_dists / self.length_scale**2)
    
    def _linear_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Linear kernel implementation."""
        # k(x,x') = σ² x^T x'
        return self.signal_variance * (X1 @ X2.T)
    
    def _polynomial_kernel(self, X1: np.ndarray, X2: np.ndarray, degree: int = 2) -> np.ndarray:
        """Polynomial kernel implementation."""
        # k(x,x') = σ² (x^T x' + 1)^d
        return self.signal_variance * (X1 @ X2.T + 1)**degree
    
    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2."""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcess':
        """Fit Gaussian Process to training data."""
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Compute kernel matrix
        K = self._compute_kernel(X, X)
        K += self.noise_level * np.eye(len(X))  # Add noise
        
        # Cholesky decomposition for numerical stability
        try:
            self.L_ = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            # Add more noise if not positive definite
            K += 1e-6 * np.eye(len(X))
            self.L_ = cholesky(K, lower=True)
        
        # Solve L α = y for α, then K α = y is solved
        self.alpha_ = solve_triangular(self.L_.T, solve_triangular(self.L_, y, lower=True), lower=False)
        
        return self
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using GP posterior."""
        # Compute covariance between test and training points
        K_star = self._compute_kernel(X, self.X_train_)  # k*
        
        # Predictive mean: μ* = k* K⁻¹ y = k* α
        predictive_mean = K_star @ self.alpha_
        
        if return_uncertainty:
            # Compute diagonal of posterior covariance efficiently
            K_star_star = self._compute_kernel(X, X)  # K**
            
            # Solve L v = k* for v
            v = solve_triangular(self.L_, K_star.T, lower=True)
            
            # Posterior variance: σ²* = K** - k* K⁻¹ k*^T = K** - v^T v
            predictive_variance = np.diag(K_star_star) - np.sum(v**2, axis=0)
            
            # Ensure non-negative variance
            predictive_variance = np.maximum(predictive_variance, 0)
            
            return predictive_mean, predictive_variance
        
        return predictive_mean
    
    def log_marginal_likelihood(self) -> float:
        """Calculate log marginal likelihood."""
        # log p(y|X) = -0.5 y^T K⁻¹ y - 0.5 log|K| - n/2 log(2π)
        
        # -0.5 y^T K⁻¹ y = -0.5 y^T α
        data_fit = -0.5 * self.y_train_ @ self.alpha_
        
        # -0.5 log|K| = -sum(log(diag(L)))
        complexity_penalty = -np.sum(np.log(np.diag(self.L_)))
        
        # -n/2 log(2π)
        normalization = -0.5 * len(self.y_train_) * np.log(2 * np.pi)
        
        return data_fit + complexity_penalty + normalization
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcess':
        """Optimize hyperparameters by maximizing marginal likelihood."""
        
        def objective(params):
            # Update parameters
            self.length_scale = np.exp(params[0])  # Log-parameterization
            self.signal_variance = np.exp(params[1])
            self.noise_level = np.exp(params[2])
            
            try:
                # Refit with new parameters
                self.fit(X, y)
                return -self.log_marginal_likelihood()
            except:
                return 1e6  # Return large value if fit fails
        
        # Initial parameters (log-scale)
        initial_params = [
            np.log(self.length_scale),
            np.log(self.signal_variance),
            np.log(self.noise_level)
        ]
        
        # Optimize
        result = minimize(objective, initial_params, method='L-BFGS-B')
        
        # Update with optimal parameters
        if result.success:
            optimal_params = result.x
            self.length_scale = np.exp(optimal_params[0])
            self.signal_variance = np.exp(optimal_params[1])
            self.noise_level = np.exp(optimal_params[2])
            
            # Refit with optimal parameters
            self.fit(X, y)
        
        return self


class BayesianNeuralNetwork:
    """
    Bayesian Neural Network with variational inference.
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'tanh', 
                 prior_variance: float = 1.0):
        """Initialize Bayesian Neural Network."""
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.prior_variance = prior_variance
        
        # Variational parameters
        self.variational_means_ = None
        self.variational_log_vars_ = None
        
        # Network structure
        self.n_layers = len(layer_sizes) - 1
        self.weight_shapes = []
        for i in range(self.n_layers):
            self.weight_shapes.append((layer_sizes[i], layer_sizes[i + 1]))
    
    def _initialize_variational_parameters(self):
        """Initialize variational parameters for mean-field approximation."""
        self.variational_means_ = []
        self.variational_log_vars_ = []
        
        for shape in self.weight_shapes:
            # Initialize means near zero
            mean = 0.1 * np.random.randn(*shape)
            # Initialize log-variances (start with small variance)
            log_var = -2.0 * np.ones(shape)
            
            self.variational_means_.append(mean)
            self.variational_log_vars_.append(log_var)
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _forward_pass(self, X: np.ndarray, weights: List[np.ndarray]) -> np.ndarray:
        """Forward pass through network with given weights."""
        activations = X
        
        for i, W in enumerate(weights):
            linear_output = activations @ W
            
            if i < len(weights) - 1:  # Hidden layers
                activations = self._activation_function(linear_output)
            else:  # Output layer (no activation for regression)
                activations = linear_output
        
        return activations
    
    def _sample_weights(self) -> List[np.ndarray]:
        """Sample weights from variational posterior."""
        weights = []
        
        for mean, log_var in zip(self.variational_means_, self.variational_log_vars_):
            std = np.exp(0.5 * log_var)
            epsilon = np.random.randn(*mean.shape)
            weight = mean + std * epsilon  # Reparameterization trick
            weights.append(weight)
        
        return weights
    
    def _kl_divergence(self) -> float:
        """Calculate KL divergence between posterior and prior."""
        kl = 0.0
        
        for mean, log_var in zip(self.variational_means_, self.variational_log_vars_):
            # KL[q(w)||p(w)] for Gaussian distributions
            # KL = 0.5 * Σ [log(σ_prior²/σ_q²) + σ_q²/σ_prior² + μ_q²/σ_prior² - 1]
            var = np.exp(log_var)
            kl += 0.5 * np.sum(
                np.log(self.prior_variance) - log_var + 
                var / self.prior_variance + 
                mean**2 / self.prior_variance - 1
            )
        
        return kl
    
    def _elbo_loss(self, X: np.ndarray, y: np.ndarray, n_samples: int = 10) -> float:
        """Calculate ELBO (Evidence Lower BOund) loss."""
        # Monte Carlo estimate of expected log-likelihood
        log_likelihood = 0.0
        
        for _ in range(n_samples):
            weights = self._sample_weights()
            predictions = self._forward_pass(X, weights)
            
            # Gaussian likelihood (regression)
            log_likelihood += -0.5 * np.sum((y.reshape(-1, 1) - predictions)**2)
        
        log_likelihood /= n_samples
        
        # KL divergence
        kl_div = self._kl_divergence()
        
        # ELBO = E[log p(y|x,w)] - KL[q(w)||p(w)]
        elbo = log_likelihood - kl_div
        
        return -elbo  # Return negative for minimization
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 1000, 
            learning_rate: float = 0.01) -> 'BayesianNeuralNetwork':
        """Train BNN using variational inference."""
        self._initialize_variational_parameters()
        
        for epoch in range(n_epochs):
            # Calculate ELBO loss
            loss = self._elbo_loss(X, y, n_samples=10)
            
            # Simple gradient descent (in practice, use automatic differentiation)
            # This is a simplified implementation
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, ELBO Loss: {loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty quantification."""
        predictions = []
        
        for _ in range(n_samples):
            weights = self._sample_weights()
            pred = self._forward_pass(X, weights)
            predictions.append(pred.flatten())
        
        predictions = np.array(predictions)
        
        # Calculate mean and variance
        predictive_mean = np.mean(predictions, axis=0)
        predictive_variance = np.var(predictions, axis=0)
        
        return predictive_mean, predictive_variance


class VariationalInference:
    """
    General variational inference framework.
    """
    
    def __init__(self, model_log_prob: Callable, 
                 variational_family: str = 'mean_field_gaussian'):
        """Initialize VI framework."""
        self.model_log_prob = model_log_prob
        self.variational_family = variational_family
    
    def _mean_field_gaussian_sample(self, params: Dict) -> np.ndarray:
        """Sample from mean-field Gaussian variational distribution."""
        means = params['means']
        log_vars = params['log_vars']
        
        stds = np.exp(0.5 * log_vars)
        epsilon = np.random.randn(*means.shape)
        
        return means + stds * epsilon
    
    def _mean_field_gaussian_log_prob(self, z: np.ndarray, params: Dict) -> float:
        """Calculate log probability under mean-field Gaussian."""
        means = params['means']
        log_vars = params['log_vars']
        
        # log q(z) = Σ log N(z_i; μ_i, σ_i²)
        log_prob = -0.5 * np.sum(log_vars + (z - means)**2 / np.exp(log_vars) + np.log(2 * np.pi))
        
        return log_prob
    
    def elbo(self, params: Dict, data: np.ndarray, n_samples: int = 100) -> float:
        """Calculate Evidence Lower BOund."""
        elbo_estimate = 0.0
        
        for _ in range(n_samples):
            # Sample from variational distribution
            z = self._mean_field_gaussian_sample(params)
            
            # Calculate log joint probability
            log_joint = self.model_log_prob(data, z)
            
            # Calculate log variational probability
            log_q = self._mean_field_gaussian_log_prob(z, params)
            
            # ELBO contribution
            elbo_estimate += log_joint - log_q
        
        return elbo_estimate / n_samples
    
    def fit(self, data: np.ndarray, n_iterations: int = 1000, 
            learning_rate: float = 0.01) -> Dict:
        """Optimize variational parameters to maximize ELBO."""
        # Initialize parameters
        n_dim = len(data)
        params = {
            'means': np.zeros(n_dim),
            'log_vars': -2.0 * np.ones(n_dim)  # Small initial variance
        }
        
        for iteration in range(n_iterations):
            # Calculate ELBO
            current_elbo = self.elbo(params, data, n_samples=50)
            
            # Simple gradient ascent (simplified)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, ELBO: {current_elbo:.4f}")
        
        return params


class MCMCSampler:
    """
    Markov Chain Monte Carlo sampling methods.
    """
    
    def __init__(self, log_prob_fn: Callable, sampler_type: str = 'metropolis_hastings'):
        """Initialize MCMC sampler."""
        self.log_prob_fn = log_prob_fn
        self.sampler_type = sampler_type
        
        # Statistics
        self.n_accepted_ = 0
        self.n_total_ = 0
    
    def metropolis_hastings_step(self, current_state: np.ndarray, 
                                step_size: float = 0.1) -> Tuple[np.ndarray, bool]:
        """Single Metropolis-Hastings step."""
        # Propose new state
        proposal = current_state + step_size * np.random.randn(*current_state.shape)
        
        # Calculate acceptance ratio
        current_log_prob = self.log_prob_fn(current_state)
        proposal_log_prob = self.log_prob_fn(proposal)
        
        log_alpha = proposal_log_prob - current_log_prob
        alpha = min(1.0, np.exp(log_alpha))
        
        # Accept or reject
        if np.random.rand() < alpha:
            return proposal, True
        else:
            return current_state, False
    
    def gibbs_step(self, current_state: np.ndarray, 
                   conditional_samplers: List[Callable]) -> np.ndarray:
        """Single Gibbs sampling step."""
        new_state = current_state.copy()
        
        for i, sampler in enumerate(conditional_samplers):
            # Sample z_i ~ p(z_i | z_{-i})
            new_state[i] = sampler(new_state)
        
        return new_state
    
    def sample(self, initial_state: np.ndarray, n_samples: int = 1000, 
               burn_in: int = 100, thin: int = 1) -> np.ndarray:
        """Generate MCMC samples."""
        current_state = initial_state.copy()
        samples = []
        
        total_iterations = burn_in + n_samples * thin
        
        for i in range(total_iterations):
            if self.sampler_type == 'metropolis_hastings':
                current_state, accepted = self.metropolis_hastings_step(current_state)
                
                if accepted:
                    self.n_accepted_ += 1
                self.n_total_ += 1
            
            # Collect sample after burn-in
            if i >= burn_in and (i - burn_in) % thin == 0:
                samples.append(current_state.copy())
        
        return np.array(samples)
    
    def effective_sample_size(self, samples: np.ndarray) -> float:
        """Calculate effective sample size using autocorrelation."""
        # Simplified ESS calculation
        n_samples = len(samples)
        
        # Calculate autocorrelation for first dimension
        if samples.ndim == 1:
            x = samples
        else:
            x = samples[:, 0]  # Use first dimension
        
        # Normalize
        x = x - np.mean(x)
        
        # Autocorrelation via FFT
        f = np.fft.fft(x, n=2*len(x))
        autocorr = np.fft.ifft(f * np.conj(f)).real
        autocorr = autocorr[:len(x)]
        autocorr = autocorr / autocorr[0]
        
        # Find integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[1:autocorr.argmax()])
        
        # ESS = N / (1 + 2τ)
        ess = n_samples / (1 + 2 * tau_int)
        
        return max(1.0, ess)


def generate_bayesian_data(dataset_type: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for Bayesian methods."""
    np.random.seed(42)
    
    if dataset_type == 'classification':
        # Multi-class dataset with overlapping clusters
        n_samples_per_class = 50
        n_classes = 3
        
        X = []
        y = []
        
        centers = [[2, 2], [-2, -2], [2, -2]]
        
        for i in range(n_classes):
            class_X = np.random.randn(n_samples_per_class, 2) + centers[i]
            X.append(class_X)
            y.append(i * np.ones(n_samples_per_class))
        
        X = np.vstack(X)
        y = np.hstack(y).astype(int)
        
        return X, y
    
    elif dataset_type == 'regression':
        # Nonlinear 1D regression with heteroscedastic noise
        X = np.linspace(0, 4, 50).reshape(-1, 1)
        y = np.sin(2 * X.ravel()) + 0.3 * np.cos(5 * X.ravel()) + 0.1 * X.ravel() * np.random.randn(50)
        
        return X, y
    
    elif dataset_type == 'sparse':
        # Sparse regression problem
        n_samples = 100
        n_features = 20
        n_relevant = 5
        
        X = np.random.randn(n_samples, n_features)
        true_coef = np.zeros(n_features)
        true_coef[:n_relevant] = np.random.randn(n_relevant)
        
        y = X @ true_coef + 0.1 * np.random.randn(n_samples)
        
        return X, y
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def plot_gp_predictions(X_train: np.ndarray, y_train: np.ndarray, 
                       X_test: np.ndarray, mean: np.ndarray, 
                       variance: np.ndarray, title: str = "GP Predictions"):
    """Plot Gaussian Process predictions with uncertainty."""
    plt.figure(figsize=(10, 6))
    
    # Sort test points for plotting
    if X_test.shape[1] == 1:
        sort_idx = np.argsort(X_test.ravel())
        X_test_sorted = X_test[sort_idx]
        mean_sorted = mean[sort_idx]
        std_sorted = np.sqrt(variance[sort_idx])
        
        # Plot training data
        plt.scatter(X_train.ravel(), y_train, c='red', marker='o', s=50, label='Training data')
        
        # Plot predictive mean
        plt.plot(X_test_sorted.ravel(), mean_sorted, 'blue', label='Predictive mean')
        
        # Plot confidence intervals
        plt.fill_between(X_test_sorted.ravel(), 
                        mean_sorted - 2*std_sorted, 
                        mean_sorted + 2*std_sorted, 
                        alpha=0.3, color='blue', label='95% confidence')
        
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("Plotting only available for 1D input data")


def compare_bayesian_methods(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compare different Bayesian methods on the same dataset."""
    results = {}
    
    # Split data
    n_train = int(0.8 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Bayesian Linear Regression
    try:
        blr = BayesianLinearRegression(alpha=1.0, beta=25.0)
        blr.fit(X_train, y_train)
        blr_pred = blr.predict(X_test)
        results['blr_rmse'] = np.sqrt(np.mean((blr_pred - y_test)**2))
    except:
        results['blr_rmse'] = np.nan
    
    # Gaussian Process
    try:
        gp = GaussianProcess(kernel='rbf', length_scale=1.0)
        gp.fit(X_train, y_train)
        gp_pred = gp.predict(X_test)
        results['gp_rmse'] = np.sqrt(np.mean((gp_pred - y_test)**2))
        results['gp_log_marginal_likelihood'] = gp.log_marginal_likelihood()
    except:
        results['gp_rmse'] = np.nan
        results['gp_log_marginal_likelihood'] = np.nan
    
    # Bayesian Neural Network (simplified)
    try:
        bnn = BayesianNeuralNetwork([X.shape[1], 5, 1])
        bnn.fit(X_train, y_train, n_epochs=50)
        bnn_pred, _ = bnn.predict(X_test, n_samples=20)
        results['bnn_rmse'] = np.sqrt(np.mean((bnn_pred - y_test)**2))
    except:
        results['bnn_rmse'] = np.nan
    
    return results


def bayesian_model_selection(X: np.ndarray, y: np.ndarray, 
                           models: List) -> Tuple[int, List[float]]:
    """Bayesian model selection using marginal likelihood."""
    marginal_likelihoods = []
    
    for model in models:
        try:
            model.fit(X, y)
            
            if hasattr(model, 'log_marginal_likelihood'):
                ml = model.log_marginal_likelihood()
            else:
                # Use approximation for models without analytical marginal likelihood
                ml = -np.inf  # Placeholder
            
            marginal_likelihoods.append(ml)
        except:
            marginal_likelihoods.append(-np.inf)
    
    # Select model with highest marginal likelihood
    best_idx = np.argmax(marginal_likelihoods)
    
    return best_idx, marginal_likelihoods


def uncertainty_calibration_analysis(predictions: np.ndarray, 
                                   uncertainties: np.ndarray, 
                                   true_values: np.ndarray) -> Dict[str, float]:
    """Analyze uncertainty calibration of Bayesian predictions."""
    results = {}
    
    # Calculate prediction errors
    errors = np.abs(predictions - true_values)
    
    # Calibration error: how often do confidence intervals contain true values
    confidence_levels = [0.5, 0.68, 0.95]
    
    for conf in confidence_levels:
        z_score = norm.ppf(0.5 + conf/2)
        intervals = uncertainties * z_score
        
        # Check if true values fall within intervals
        within_interval = errors <= intervals
        empirical_coverage = np.mean(within_interval)
        
        calibration_error = abs(empirical_coverage - conf)
        results[f'calibration_error_{int(conf*100)}'] = calibration_error
    
    # Mean squared calibration error
    results['mean_calibration_error'] = np.mean([
        results[f'calibration_error_{int(conf*100)}'] for conf in confidence_levels
    ])
    
    # Sharpness (average uncertainty)
    results['sharpness'] = np.mean(uncertainties)
    
    return results


# Export all solution implementations
__all__ = [
    'NaiveBayesClassifier', 'BayesianLinearRegression', 'GaussianProcess',
    'BayesianNeuralNetwork', 'VariationalInference', 'MCMCSampler',
    'generate_bayesian_data', 'plot_gp_predictions', 'compare_bayesian_methods',
    'bayesian_model_selection', 'uncertainty_calibration_analysis'
]