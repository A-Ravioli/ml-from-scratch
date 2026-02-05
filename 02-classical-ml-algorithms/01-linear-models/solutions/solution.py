"""
Solution implementations for Linear Models exercises.

This file provides complete implementations of all exercise items in exercise.py.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Union
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from scipy import linalg, stats
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings


# Base Classes

class LinearModel(ABC):
    """Base class for linear models."""
    
    def __init__(self, name: str):
        self.name = name
        self.coefficients_ = None
        self.intercept_ = None
        self.fitted_ = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        ...
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to design matrix."""
        return np.column_stack([np.ones(len(X)), X])


# Linear Regression

class LinearRegression(LinearModel):
    """Linear Regression with multiple solvers."""
    
    def __init__(self, solver: str = 'normal_equation', 
                 regularization: Optional[str] = None,
                 lambda_reg: float = 0.0, learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6):
        super().__init__("Linear Regression")
        self.solver = solver
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        
        # Fitted attributes
        self.r_squared_ = None
        self.mse_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit linear regression model.
        
        Supports multiple solvers:
        - normal_equation: (X^T X)^(-1) X^T y
        - qr_decomposition: QR decomposition
        - svd: Singular Value Decomposition
        - gradient_descent: Iterative optimization
        """
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        if self.solver == 'normal_equation':
            self._fit_normal_equation(X_with_intercept, y)
        elif self.solver == 'qr_decomposition':
            self._fit_qr_decomposition(X_with_intercept, y)
        elif self.solver == 'svd':
            self._fit_svd(X_with_intercept, y)
        elif self.solver == 'gradient_descent':
            self._fit_gradient_descent(X_with_intercept, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Extract intercept and coefficients
        self.intercept_ = self.coefficients_[0]
        self.coefficients_ = self.coefficients_[1:]
        
        # Compute metrics
        self._compute_metrics(X, y)
        self.fitted_ = True
        
        return self
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """Solve using normal equation."""
        if self.regularization == 'ridge':
            # Ridge: (X^T X + λI)^(-1) X^T y
            XTX = X.T @ X
            XTX[0, 0] -= self.lambda_reg  # Don't regularize intercept
            XTX[1:, 1:] += self.lambda_reg * np.eye(X.shape[1] - 1)
            self.coefficients_ = linalg.solve(XTX, X.T @ y)
        else:
            # OLS: (X^T X)^(-1) X^T y
            self.coefficients_ = linalg.solve(X.T @ X, X.T @ y)
    
    def _fit_qr_decomposition(self, X: np.ndarray, y: np.ndarray):
        """Solve using QR decomposition."""
        Q, R = linalg.qr(X)
        self.coefficients_ = linalg.solve(R, Q.T @ y)
    
    def _fit_svd(self, X: np.ndarray, y: np.ndarray):
        """Solve using SVD (most numerically stable)."""
        U, s, Vt = linalg.svd(X, full_matrices=False)
        
        # Handle singular values close to zero
        s_inv = np.where(s > 1e-10, 1/s, 0)
        
        self.coefficients_ = Vt.T @ (s_inv[:, np.newaxis] * (U.T @ y))
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Solve using gradient descent."""
        n, p = X.shape
        self.coefficients_ = np.zeros(p)
        
        for iteration in range(self.max_iter):
            # Predictions and residuals
            y_pred = X @ self.coefficients_
            residuals = y_pred - y
            
            # Gradient
            gradient = (2/n) * X.T @ residuals
            
            # Add regularization gradient
            if self.regularization == 'ridge':
                gradient[1:] += 2 * self.lambda_reg * self.coefficients_[1:]
            
            # Update
            new_coefficients = self.coefficients_ - self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(new_coefficients - self.coefficients_) < self.tol:
                break
            
            self.coefficients_ = new_coefficients
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        return X @ self.coefficients_ + self.intercept_
    
    def _compute_metrics(self, X: np.ndarray, y: np.ndarray):
        """Compute model metrics."""
        y_pred = self.predict(X)
        
        # Mean Squared Error
        self.mse_ = np.mean((y - y_pred) ** 2)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        self.r_squared_ = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


# Ridge Regression

class RidgeRegression(LinearModel):
    """Ridge Regression with L2 regularization."""
    
    def __init__(self, lambda_reg: float = 1.0, cv_folds: Optional[int] = None):
        super().__init__("Ridge Regression")
        self.lambda_reg = lambda_reg
        self.cv_folds = cv_folds
        self.cv_score_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Ridge regression."""
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        # Ridge regression: (X^T X + λI)^(-1) X^T y
        XTX = X_with_intercept.T @ X_with_intercept
        
        # Don't regularize intercept
        regularization_matrix = np.eye(p + 1)
        regularization_matrix[0, 0] = 0
        
        XTX_reg = XTX + self.lambda_reg * regularization_matrix
        full_coefficients = linalg.solve(XTX_reg, X_with_intercept.T @ y)
        
        self.intercept_ = full_coefficients[0]
        self.coefficients_ = full_coefficients[1:]
        
        # Cross-validation if requested
        if self.cv_folds:
            self.cv_score_ = self._cross_validate(X, y)
        
        self.fitted_ = True
        return self
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Perform cross-validation."""
        n = len(X)
        fold_size = n // self.cv_folds
        scores = []
        
        for fold in range(self.cv_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < self.cv_folds - 1 else n
            
            mask = np.ones(n, dtype=bool)
            mask[start_idx:end_idx] = False
            
            X_train, X_val = X[mask], X[~mask]
            y_train, y_val = y[mask], y[~mask]
            
            # Fit on training fold
            ridge_fold = RidgeRegression(self.lambda_reg)
            ridge_fold.fit(X_train, y_train)
            
            # Evaluate on validation fold
            y_pred = ridge_fold.predict(X_val)
            score = np.mean((y_val - y_pred) ** 2)
            scores.append(score)
        
        return np.mean(scores)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        return X @ self.coefficients_ + self.intercept_


# Lasso Regression

class LassoRegression(LinearModel):
    """Lasso Regression with L1 regularization."""
    
    def __init__(self, lambda_reg: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-6, solver: str = 'coordinate_descent'):
        super().__init__("Lasso Regression")
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.converged_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Lasso regression using coordinate descent."""
        n, p = X.shape
        
        # Standardize features for coordinate descent
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_centered = y - np.mean(y)
        
        if self.solver == 'coordinate_descent':
            self.coefficients_ = self._coordinate_descent(X_std, y_centered)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Compute intercept
        self.intercept_ = np.mean(y) - np.mean(X, axis=0) @ self.coefficients_
        
        self.fitted_ = True
        return self
    
    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Coordinate descent algorithm for Lasso."""
        n, p = X.shape
        coefficients = np.zeros(p)
        
        # Precompute X^T X diagonal
        XTX_diag = np.sum(X ** 2, axis=0)
        
        for iteration in range(self.max_iter):
            coefficients_old = coefficients.copy()
            
            for j in range(p):
                # Compute partial residual
                residual = y - X @ coefficients + coefficients[j] * X[:, j]
                
                # Coordinate update with soft thresholding
                rho = X[:, j] @ residual
                
                if XTX_diag[j] == 0:
                    coefficients[j] = 0
                else:
                    coefficients[j] = self._soft_threshold(rho, self.lambda_reg) / XTX_diag[j]
            
            # Check convergence
            if np.linalg.norm(coefficients - coefficients_old) < self.tol:
                self.converged_ = True
                break
        
        return coefficients
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        return X @ self.coefficients_ + self.intercept_


# Elastic Net Regression

class ElasticNetRegression(LinearModel):
    """Elastic Net Regression combining L1 and L2 regularization."""
    
    def __init__(self, lambda_reg: float = 1.0, alpha: float = 0.5,
                 max_iter: int = 1000, tol: float = 1e-6):
        super().__init__("Elastic Net Regression")
        self.lambda_reg = lambda_reg
        self.alpha = alpha  # L1 ratio: 0=Ridge, 1=Lasso
        self.max_iter = max_iter
        self.tol = tol
        self.converged_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Elastic Net using coordinate descent."""
        n, p = X.shape
        
        # Standardize
        X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        y_centered = y - np.mean(y)
        
        self.coefficients_ = self._coordinate_descent_elastic_net(X_std, y_centered)
        
        # Compute intercept
        self.intercept_ = np.mean(y) - np.mean(X, axis=0) @ self.coefficients_
        
        self.fitted_ = True
        return self
    
    def _coordinate_descent_elastic_net(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Coordinate descent for Elastic Net."""
        n, p = X.shape
        coefficients = np.zeros(p)
        
        # Precompute
        XTX_diag = np.sum(X ** 2, axis=0)
        
        for iteration in range(self.max_iter):
            coefficients_old = coefficients.copy()
            
            for j in range(p):
                # Partial residual
                residual = y - X @ coefficients + coefficients[j] * X[:, j]
                rho = X[:, j] @ residual
                
                if XTX_diag[j] == 0:
                    coefficients[j] = 0
                else:
                    # Elastic Net update
                    l1_penalty = self.alpha * self.lambda_reg
                    l2_penalty = (1 - self.alpha) * self.lambda_reg
                    
                    denominator = XTX_diag[j] + l2_penalty
                    coefficients[j] = self._soft_threshold(rho, l1_penalty) / denominator
            
            # Check convergence
            if np.linalg.norm(coefficients - coefficients_old) < self.tol:
                self.converged_ = True
                break
        
        return coefficients
    
    def _soft_threshold(self, x: float, threshold: float) -> float:
        """Soft thresholding operator."""
        if x > threshold:
            return x - threshold
        elif x < -threshold:
            return x + threshold
        else:
            return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        return X @ self.coefficients_ + self.intercept_


# Logistic Regression

class LogisticRegression(LinearModel):
    """Logistic Regression for classification."""
    
    def __init__(self, regularization: Optional[str] = None, lambda_reg: float = 0.0,
                 solver: str = 'gradient_descent', learning_rate: float = 0.01,
                 max_iter: int = 1000, tol: float = 1e-6,
                 multi_class: str = 'ovr'):
        super().__init__("Logistic Regression")
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.multi_class = multi_class
        
        # Fitted attributes
        self.classes_ = None
        self.n_classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit logistic regression."""
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.n_classes_ == 2:
            # Binary classification
            self._fit_binary(X, y)
        else:
            # Multiclass classification
            if self.multi_class == 'ovr':
                self._fit_ovr(X, y)
            elif self.multi_class == 'multinomial':
                self._fit_multinomial(X, y)
            else:
                raise ValueError(f"Unknown multi_class strategy: {self.multi_class}")
        
        self.fitted_ = True
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """Fit binary logistic regression."""
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        # Convert labels to 0/1
        y_binary = (y == self.classes_[1]).astype(int)
        
        if self.solver == 'gradient_descent':
            self.coefficients_ = self._gradient_descent_binary(X_with_intercept, y_binary)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        self.intercept_ = self.coefficients_[0]
        self.coefficients_ = self.coefficients_[1:]
    
    def _gradient_descent_binary(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient descent for binary logistic regression."""
        n, p = X.shape
        coefficients = np.zeros(p)
        
        for iteration in range(self.max_iter):
            # Predictions
            z = X @ coefficients
            probabilities = self._sigmoid(z)
            
            # Gradient of log-likelihood
            gradient = X.T @ (probabilities - y) / n
            
            # Add regularization gradient
            if self.regularization == 'l2':
                gradient[1:] += self.lambda_reg * coefficients[1:]  # Don't regularize intercept
            elif self.regularization == 'l1':
                gradient[1:] += self.lambda_reg * np.sign(coefficients[1:])
            
            # Update
            new_coefficients = coefficients - self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(new_coefficients - coefficients) < self.tol:
                break
            
            coefficients = new_coefficients
        
        return coefficients
    
    def _fit_ovr(self, X: np.ndarray, y: np.ndarray):
        """Fit one-vs-rest multiclass."""
        self.coefficients_ = np.zeros((self.n_classes_, X.shape[1]))
        self.intercept_ = np.zeros(self.n_classes_)
        
        for i, cls in enumerate(self.classes_):
            # Create binary labels
            y_binary = (y == cls).astype(int)
            
            # Fit binary classifier
            binary_lr = LogisticRegression(
                regularization=self.regularization,
                lambda_reg=self.lambda_reg,
                solver=self.solver,
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                tol=self.tol
            )
            binary_lr.fit(X, y_binary)
            
            self.coefficients_[i] = binary_lr.coefficients_
            self.intercept_[i] = binary_lr.intercept_
    
    def _fit_multinomial(self, X: np.ndarray, y: np.ndarray):
        """Fit multinomial logistic regression."""
        # Simplified multinomial implementation
        # In practice, would use more sophisticated optimization
        self._fit_ovr(X, y)  # Fallback to OvR
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if self.n_classes_ == 2:
            # Binary case
            z = X @ self.coefficients_ + self.intercept_
            prob_1 = self._sigmoid(z)
            return np.column_stack([1 - prob_1, prob_1])
        else:
            # Multiclass case
            scores = X @ self.coefficients_.T + self.intercept_
            
            if self.multi_class == 'ovr':
                # One-vs-rest: normalize scores
                probabilities = self._sigmoid(scores)
                probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
            else:
                # Multinomial: softmax
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            return probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes_[predicted_indices]


# Poisson Regression

class PoissonRegression(LinearModel):
    """Poisson Regression for count data."""
    
    def __init__(self, max_iter: int = 1000, tol: float = 1e-6,
                 learning_rate: float = 0.01):
        super().__init__("Poisson Regression")
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Poisson regression using IRLS or gradient descent."""
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        # Initialize coefficients
        coefficients = np.zeros(p + 1)
        
        for iteration in range(self.max_iter):
            # Linear predictor
            eta = X_with_intercept @ coefficients
            
            # Mean (inverse link)
            mu = self._inverse_link(eta)
            
            # Gradient of log-likelihood
            gradient = X_with_intercept.T @ (y - mu) / n
            
            # Update
            new_coefficients = coefficients + self.learning_rate * gradient
            
            # Check convergence
            if np.linalg.norm(new_coefficients - coefficients) < self.tol:
                break
            
            coefficients = new_coefficients
        
        self.intercept_ = coefficients[0]
        self.coefficients_ = coefficients[1:]
        self.fitted_ = True
        
        return self
    
    def _inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link function (exponential for Poisson)."""
        # Clip to prevent overflow
        eta = np.clip(eta, -700, 700)
        return np.exp(eta)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expected counts."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        eta = X @ self.coefficients_ + self.intercept_
        return self._inverse_link(eta)


# Bayesian Linear Regression

class BayesianLinearRegression(LinearModel):
    """Bayesian Linear Regression with conjugate priors."""
    
    def __init__(self, prior_precision: float = 1.0, noise_precision: float = 1.0):
        super().__init__("Bayesian Linear Regression")
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision
        
        # Posterior parameters
        self.posterior_mean_ = None
        self.posterior_covariance_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Bayesian linear regression."""
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        # Prior parameters
        prior_mean = np.zeros(p + 1)
        prior_cov_inv = self.prior_precision * np.eye(p + 1)
        prior_cov_inv[0, 0] = 1e-6  # Weak prior on intercept
        
        # Posterior parameters (conjugate update)
        self.posterior_covariance_inv_ = (prior_cov_inv + 
                                        self.noise_precision * X_with_intercept.T @ X_with_intercept)
        self.posterior_covariance_ = linalg.inv(self.posterior_covariance_inv_)
        
        self.posterior_mean_ = self.posterior_covariance_ @ (
            prior_cov_inv @ prior_mean + self.noise_precision * X_with_intercept.T @ y
        )
        
        # Point estimates
        self.intercept_ = self.posterior_mean_[0]
        self.coefficients_ = self.posterior_mean_[1:]
        
        self.fitted_ = True
        return self
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = False):
        """Predict with uncertainty quantification."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        X_with_intercept = self._add_intercept(X)
        
        # Predictive mean
        mean = X_with_intercept @ self.posterior_mean_
        
        if return_uncertainty:
            # Predictive variance
            variance = np.diag(X_with_intercept @ self.posterior_covariance_ @ X_with_intercept.T)
            variance += 1 / self.noise_precision  # Add noise variance
            return mean, variance
        
        return mean
    
    def credible_intervals(self, X: np.ndarray, confidence: float = 0.95) -> np.ndarray:
        """Compute credible intervals."""
        mean, variance = self.predict(X, return_uncertainty=True)
        std = np.sqrt(variance)
        
        # Critical value for normal distribution
        alpha = 1 - confidence
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower = mean - z_score * std
        upper = mean + z_score * std
        
        return np.column_stack([lower, upper])
    
    def sample_posterior(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Sample from posterior predictive distribution."""
        X_with_intercept = self._add_intercept(X)
        
        # Sample coefficient vectors from posterior
        coeff_samples = np.random.multivariate_normal(
            self.posterior_mean_, self.posterior_covariance_, n_samples
        )
        
        # Compute predictions for each sample
        predictions = X_with_intercept @ coeff_samples.T
        
        # Add noise
        noise_std = 1 / np.sqrt(self.noise_precision)
        predictions += np.random.normal(0, noise_std, predictions.shape)
        
        return predictions.T


# Generalized Linear Model

class GeneralizedLinearModel(LinearModel):
    """Generalized Linear Model framework."""
    
    def __init__(self, family: str = 'gaussian', link: str = 'identity',
                 max_iter: int = 100, tol: float = 1e-6):
        super().__init__("Generalized Linear Model")
        self.family = family
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        
        # Set up family and link functions
        self._setup_family_link()
    
    def _setup_family_link(self):
        """Set up family and link functions."""
        if self.family == 'gaussian':
            if self.link == 'identity':
                self.link_func = lambda x: x
                self.inverse_link_func = lambda x: x
                self.variance_func = lambda mu: np.ones_like(mu)
            else:
                raise ValueError(f"Unsupported link {self.link} for family {self.family}")
        
        elif self.family == 'binomial':
            if self.link == 'logit':
                self.link_func = lambda p: np.log(p / (1 - p))
                self.inverse_link_func = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))
                self.variance_func = lambda mu: mu * (1 - mu)
            else:
                raise ValueError(f"Unsupported link {self.link} for family {self.family}")
        
        elif self.family == 'poisson':
            if self.link == 'log':
                self.link_func = lambda mu: np.log(mu)
                self.inverse_link_func = lambda x: np.exp(np.clip(x, -700, 700))
                self.variance_func = lambda mu: mu
            else:
                raise ValueError(f"Unsupported link {self.link} for family {self.family}")
        
        else:
            raise ValueError(f"Unsupported family: {self.family}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GLM using IRLS (Iteratively Reweighted Least Squares)."""
        n, p = X.shape
        X_with_intercept = self._add_intercept(X)
        
        # Initialize
        coefficients = np.zeros(p + 1)
        
        for iteration in range(self.max_iter):
            # Linear predictor
            eta = X_with_intercept @ coefficients
            
            # Mean
            mu = self.inverse_link_func(eta)
            mu = np.clip(mu, 1e-10, 1 - 1e-10)  # Avoid boundary issues
            
            # Variance
            var = self.variance_func(mu)
            var = np.maximum(var, 1e-10)  # Avoid division by zero
            
            # Derivative of inverse link
            dmu_deta = self._derivative_inverse_link(eta)
            
            # Working response
            z = eta + (y - mu) / dmu_deta
            
            # Weights
            w = dmu_deta**2 / var
            w = np.maximum(w, 1e-10)
            
            # Weighted least squares
            W = np.diag(w)
            XTW = X_with_intercept.T @ W
            
            try:
                new_coefficients = linalg.solve(XTW @ X_with_intercept, XTW @ z)
            except linalg.LinAlgError:
                # Add regularization if singular
                reg_matrix = 1e-6 * np.eye(p + 1)
                new_coefficients = linalg.solve(XTW @ X_with_intercept + reg_matrix, XTW @ z)
            
            # Check convergence
            if np.linalg.norm(new_coefficients - coefficients) < self.tol:
                break
            
            coefficients = new_coefficients
        
        self.intercept_ = coefficients[0]
        self.coefficients_ = coefficients[1:]
        self.fitted_ = True
        
        return self
    
    def _derivative_inverse_link(self, eta: np.ndarray) -> np.ndarray:
        """Derivative of inverse link function."""
        if self.family == 'gaussian' and self.link == 'identity':
            return np.ones_like(eta)
        elif self.family == 'binomial' and self.link == 'logit':
            p = self.inverse_link_func(eta)
            return p * (1 - p)
        elif self.family == 'poisson' and self.link == 'log':
            return self.inverse_link_func(eta)
        else:
            raise ValueError("Derivative not implemented for this family/link")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean response."""
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        eta = X @ self.coefficients_ + self.intercept_
        return self.inverse_link_func(eta)


# Feature Engineering

class PolynomialFeatures:
    """Generate polynomial features."""
    
    def __init__(self, degree: int = 2, include_bias: bool = True,
                 interaction_only: bool = False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        
        self.n_features_in_ = None
        self.n_features_out_ = None
        self.powers_ = None
    
    def fit(self, X: np.ndarray):
        """Fit polynomial feature generator."""
        self.n_features_in_ = X.shape[1]
        self.powers_ = self._generate_powers()
        self.n_features_out_ = len(self.powers_)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to polynomial features."""
        if self.powers_ is None:
            raise ValueError("Must fit before transform")
        
        n_samples = X.shape[0]
        X_poly = np.zeros((n_samples, self.n_features_out_))
        
        for i, powers in enumerate(self.powers_):
            feature = np.ones(n_samples)
            for j, power in enumerate(powers):
                if power > 0:
                    feature *= X[:, j] ** power
            X_poly[:, i] = feature
        
        return X_poly
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def _generate_powers(self) -> List[List[int]]:
        """Generate all power combinations."""
        powers = []
        
        # Generate all combinations up to degree
        for total_degree in range(self.degree + 1):
            if total_degree == 0 and not self.include_bias:
                continue
            
            for power_combo in self._combinations_with_replacement(
                self.n_features_in_, total_degree):
                
                if self.interaction_only and total_degree > 0:
                    # Only include if no feature has power > 1
                    if max(power_combo) <= 1:
                        powers.append(power_combo)
                else:
                    powers.append(power_combo)
        
        return powers
    
    def _combinations_with_replacement(self, n_features: int, degree: int) -> List[List[int]]:
        """Generate all combinations with replacement for given degree."""
        if degree == 0:
            return [[0] * n_features]
        
        combinations = []
        
        def backtrack(current_combo, remaining_degree, start_feature):
            if remaining_degree == 0:
                combinations.append(current_combo[:])
                return
            
            for feature in range(start_feature, n_features):
                current_combo[feature] += 1
                backtrack(current_combo, remaining_degree - 1, feature)
                current_combo[feature] -= 1
        
        backtrack([0] * n_features, degree, 0)
        return combinations


# Regularization Path

class RegularizationPath:
    """Compute regularization path for Lasso/Ridge."""
    
    def __init__(self, model_type: str = 'lasso'):
        self.model_type = model_type
    
    def compute_path(self, X: np.ndarray, y: np.ndarray, 
                    lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute regularization path.
        
        Returns:
            coefficients: (n_lambdas, n_features) array
            scores: (n_lambdas,) array of cross-validation scores
        """
        n_lambdas = len(lambdas)
        n_features = X.shape[1]
        
        coefficients = np.zeros((n_lambdas, n_features))
        scores = np.zeros(n_lambdas)
        
        for i, lambda_val in enumerate(lambdas):
            if self.model_type == 'lasso':
                model = LassoRegression(lambda_reg=lambda_val)
            elif self.model_type == 'ridge':
                model = RidgeRegression(lambda_reg=lambda_val)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            model.fit(X, y)
            coefficients[i] = model.coefficients_
            
            # Compute cross-validation score (simplified)
            y_pred = model.predict(X)
            scores[i] = np.mean((y - y_pred) ** 2)
        
        return coefficients, scores


# Cross Validation

class CrossValidation:
    """Cross validation implementation."""
    
    def __init__(self, cv_type: str = 'kfold', n_folds: int = 5,
                 scoring: str = 'mse', random_state: Optional[int] = None):
        self.cv_type = cv_type
        self.n_folds = n_folds
        self.scoring = scoring
        self.random_state = random_state
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perform cross validation."""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n = len(X)
        
        if self.cv_type == 'kfold':
            indices = self._kfold_split(n)
        elif self.cv_type == 'stratified':
            indices = self._stratified_split(y)
        else:
            raise ValueError(f"Unknown CV type: {self.cv_type}")
        
        scores = []
        
        for train_idx, val_idx in indices:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            # Predict and score
            y_pred = model_copy.predict(X_val)
            score = self._compute_score(y_val, y_pred)
            scores.append(score)
        
        return np.array(scores)
    
    def _kfold_split(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate K-fold splits."""
        indices = np.random.permutation(n)
        fold_size = n // self.n_folds
        
        splits = []
        for fold in range(self.n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < self.n_folds - 1 else n
            
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    def _stratified_split(self, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified splits."""
        classes = np.unique(y)
        n = len(y)
        
        # Create fold assignments for each class
        fold_assignments = np.zeros(n, dtype=int)
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            cls_folds = np.tile(np.arange(self.n_folds), len(cls_indices) // self.n_folds + 1)
            cls_folds = cls_folds[:len(cls_indices)]
            np.random.shuffle(cls_folds)
            
            fold_assignments[cls_indices] = cls_folds
        
        # Generate splits
        splits = []
        for fold in range(self.n_folds):
            val_idx = np.where(fold_assignments == fold)[0]
            train_idx = np.where(fold_assignments != fold)[0]
            splits.append((train_idx, val_idx))
        
        return splits
    
    def _clone_model(self, model):
        """Create a copy of the model."""
        # Simple cloning by creating new instance with same parameters
        model_class = type(model)
        
        # Get model parameters (simplified)
        if hasattr(model, '__dict__'):
            params = {k: v for k, v in model.__dict__.items() 
                     if not k.endswith('_')}  # Exclude fitted attributes
            return model_class(**params)
        else:
            return model_class()
    
    def _compute_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute score based on scoring method."""
        if self.scoring == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        elif self.scoring == 'mae':
            return np.mean(np.abs(y_true - y_pred))
        elif self.scoring == 'r2':
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        elif self.scoring == 'accuracy':
            return np.mean(y_true == y_pred)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring}")


# Feature Selection

class FeatureSelection:
    """Feature selection methods."""
    
    def __init__(self, method: str = 'univariate', k: int = 10,
                 threshold: Optional[float] = None):
        self.method = method
        self.k = k
        self.threshold = threshold
        
        self.selected_features_ = None
        self.scores_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit feature selector."""
        if self.method == 'univariate':
            self._fit_univariate(X, y)
        elif self.method == 'rfe':
            self._fit_rfe(X, y)
        elif self.method == 'l1_based':
            self._fit_l1_based(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def _fit_univariate(self, X: np.ndarray, y: np.ndarray):
        """Univariate feature selection using correlation."""
        n_features = X.shape[1]
        scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Compute correlation (for regression) or chi-square (for classification)
            if len(np.unique(y)) > 10:  # Assume regression
                scores[i] = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            else:  # Assume classification
                # Simplified: use absolute mean difference
                classes = np.unique(y)
                if len(classes) == 2:
                    mean_0 = np.mean(X[y == classes[0], i])
                    mean_1 = np.mean(X[y == classes[1], i])
                    scores[i] = np.abs(mean_0 - mean_1)
                else:
                    scores[i] = np.var([np.mean(X[y == cls, i]) for cls in classes])
        
        # Handle NaN scores
        scores = np.nan_to_num(scores)
        
        # Select top k features
        self.scores_ = scores
        self.selected_features_ = np.argsort(scores)[-self.k:]
    
    def _fit_rfe(self, X: np.ndarray, y: np.ndarray):
        """Recursive feature elimination."""
        n_features = X.shape[1]
        remaining_features = list(range(n_features))
        
        while len(remaining_features) > self.k:
            # Fit model on remaining features
            X_subset = X[:, remaining_features]
            
            if len(np.unique(y)) > 10:  # Regression
                model = LinearRegression()
            else:  # Classification
                model = LogisticRegression()
            
            model.fit(X_subset, y)
            
            # Rank features by coefficient magnitude
            feature_importance = np.abs(model.coefficients_)
            worst_feature_idx = np.argmin(feature_importance)
            
            # Remove worst feature
            removed_feature = remaining_features.pop(worst_feature_idx)
        
        self.selected_features_ = np.array(remaining_features)
    
    def _fit_l1_based(self, X: np.ndarray, y: np.ndarray):
        """L1-based feature selection."""
        if len(np.unique(y)) > 10:  # Regression
            model = LassoRegression(lambda_reg=0.1)
        else:  # Classification
            model = LogisticRegression(regularization='l1', lambda_reg=0.1)
        
        model.fit(X, y)
        
        # Select features with non-zero coefficients
        if self.threshold is not None:
            selected_mask = np.abs(model.coefficients_) > self.threshold
        else:
            # Select top k by coefficient magnitude
            feature_importance = np.abs(model.coefficients_)
            selected_mask = feature_importance >= np.sort(feature_importance)[-self.k]
        
        self.selected_features_ = np.where(selected_mask)[0]
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using selected features."""
        if self.selected_features_ is None:
            raise ValueError("Must fit before transform")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


# Export all solution implementations
__all__ = [
    'LinearModel', 'LinearRegression', 'RidgeRegression', 'LassoRegression',
    'ElasticNetRegression', 'LogisticRegression', 'PoissonRegression',
    'BayesianLinearRegression', 'GeneralizedLinearModel', 'PolynomialFeatures',
    'RegularizationPath', 'CrossValidation', 'FeatureSelection'
]
