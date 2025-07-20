"""
Test suite for Linear Models implementations.
"""

import numpy as np
import pytest
from exercise import (
    LinearRegression, RidgeRegression, LassoRegression, ElasticNetRegression,
    LogisticRegression, PoissonRegression, BayesianLinearRegression,
    GeneralizedLinearModel, PolynomialFeatures, RegularizationPath,
    CrossValidation, FeatureSelection, LinearModel
)


class TestLinearRegression:
    """Test Linear Regression implementation."""
    
    def test_linear_regression_basic(self):
        """Test basic linear regression functionality."""
        # Generate simple linear data
        np.random.seed(42)
        X = np.random.randn(50, 3)
        true_weights = np.array([2.0, -1.5, 0.8])
        y = X @ true_weights + 0.1 * np.random.randn(50)
        
        lr = LinearRegression(solver='normal_equation')
        lr.fit(X, y)
        
        # Predictions should be close to true values
        y_pred = lr.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 1.0  # Should fit well
        
        # Coefficients should be close to true weights
        coeff_error = np.linalg.norm(lr.coefficients_ - true_weights)
        assert coeff_error < 0.5
        
        # Should have computed R-squared
        assert hasattr(lr, 'r_squared_')
        assert 0 <= lr.r_squared_ <= 1
    
    def test_linear_regression_solvers(self):
        """Test different solvers give similar results."""
        X = np.random.randn(30, 2)
        y = X @ [1, -1] + 0.1 * np.random.randn(30)
        
        lr_normal = LinearRegression(solver='normal_equation')
        lr_qr = LinearRegression(solver='qr_decomposition')
        lr_svd = LinearRegression(solver='svd')
        lr_gradient = LinearRegression(solver='gradient_descent', learning_rate=0.01, max_iter=1000)
        
        lr_normal.fit(X, y)
        lr_qr.fit(X, y)
        lr_svd.fit(X, y)
        lr_gradient.fit(X, y)
        
        # All should give similar results
        coeff_diff_qr = np.linalg.norm(lr_normal.coefficients_ - lr_qr.coefficients_)
        coeff_diff_svd = np.linalg.norm(lr_normal.coefficients_ - lr_svd.coefficients_)
        coeff_diff_grad = np.linalg.norm(lr_normal.coefficients_ - lr_gradient.coefficients_)
        
        assert coeff_diff_qr < 1e-10
        assert coeff_diff_svd < 1e-10
        assert coeff_diff_grad < 0.1  # Gradient descent less precise
    
    def test_linear_regression_regularization_scaling(self):
        """Test coefficient scaling with regularization."""
        X = np.random.randn(40, 3)
        y = np.random.randn(40)
        
        lr = LinearRegression(regularization='ridge', lambda_reg=1.0)
        lr.fit(X, y)
        
        # Should have regularization effects
        assert hasattr(lr, 'coefficients_')
        assert len(lr.coefficients_) == 3


class TestRidgeRegression:
    """Test Ridge Regression implementation."""
    
    def test_ridge_regression_basic(self):
        """Test basic ridge regression."""
        X = np.random.randn(25, 4)
        y = np.random.randn(25)
        
        ridge = RidgeRegression(lambda_reg=0.1)
        ridge.fit(X, y)
        
        y_pred = ridge.predict(X)
        assert len(y_pred) == len(y)
        
        # Should have smaller coefficients than unregularized
        lr = LinearRegression()
        lr.fit(X, y)
        
        ridge_norm = np.linalg.norm(ridge.coefficients_)
        lr_norm = np.linalg.norm(lr.coefficients_)
        assert ridge_norm <= lr_norm  # Ridge should shrink coefficients
    
    def test_ridge_lambda_effect(self):
        """Test effect of regularization parameter."""
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        
        ridge_small = RidgeRegression(lambda_reg=0.01)
        ridge_large = RidgeRegression(lambda_reg=10.0)
        
        ridge_small.fit(X, y)
        ridge_large.fit(X, y)
        
        # Larger lambda should give smaller coefficients
        small_norm = np.linalg.norm(ridge_small.coefficients_)
        large_norm = np.linalg.norm(ridge_large.coefficients_)
        assert large_norm < small_norm
    
    def test_ridge_cross_validation(self):
        """Test ridge regression with cross-validation."""
        X = np.random.randn(30, 4)
        y = np.random.randn(30)
        
        ridge = RidgeRegression(lambda_reg=1.0, cv_folds=3)
        ridge.fit(X, y)
        
        # Should have computed CV score
        assert hasattr(ridge, 'cv_score_')
        assert isinstance(ridge.cv_score_, float)


class TestLassoRegression:
    """Test Lasso Regression implementation."""
    
    def test_lasso_sparsity(self):
        """Test that Lasso produces sparse solutions."""
        # Create data with some irrelevant features
        np.random.seed(42)
        X = np.random.randn(40, 10)
        true_weights = np.array([2, 0, 0, 1.5, 0, 0, 0, -1, 0, 0])
        y = X @ true_weights + 0.1 * np.random.randn(40)
        
        lasso = LassoRegression(lambda_reg=0.1, max_iter=1000)
        lasso.fit(X, y)
        
        # Should have some zero coefficients
        zero_coeffs = np.sum(np.abs(lasso.coefficients_) < 1e-3)
        assert zero_coeffs > 0  # Should be sparse
    
    def test_lasso_coordinate_descent(self):
        """Test coordinate descent solver."""
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        
        lasso = LassoRegression(lambda_reg=0.5, solver='coordinate_descent')
        lasso.fit(X, y)
        
        # Should converge
        assert hasattr(lasso, 'coefficients_')
        assert hasattr(lasso, 'converged_')
    
    def test_lasso_vs_ridge(self):
        """Test Lasso vs Ridge differences."""
        X = np.random.randn(25, 6)
        y = np.random.randn(25)
        
        lasso = LassoRegression(lambda_reg=0.2)
        ridge = RidgeRegression(lambda_reg=0.2)
        
        lasso.fit(X, y)
        ridge.fit(X, y)
        
        # Lasso should be more sparse
        lasso_zeros = np.sum(np.abs(lasso.coefficients_) < 1e-3)
        ridge_zeros = np.sum(np.abs(ridge.coefficients_) < 1e-3)
        assert lasso_zeros >= ridge_zeros


class TestElasticNetRegression:
    """Test Elastic Net Regression implementation."""
    
    def test_elastic_net_basic(self):
        """Test basic Elastic Net functionality."""
        X = np.random.randn(35, 4)
        y = np.random.randn(35)
        
        enet = ElasticNetRegression(lambda_reg=0.1, alpha=0.5)
        enet.fit(X, y)
        
        y_pred = enet.predict(X)
        assert len(y_pred) == len(y)
        assert hasattr(enet, 'coefficients_')
    
    def test_elastic_net_alpha_extremes(self):
        """Test Elastic Net at alpha extremes (Ridge and Lasso)."""
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        
        # alpha=0 should be like Ridge
        enet_ridge = ElasticNetRegression(lambda_reg=0.1, alpha=0.0)
        ridge = RidgeRegression(lambda_reg=0.1)
        
        enet_ridge.fit(X, y)
        ridge.fit(X, y)
        
        # Should give similar results (allowing some numerical difference)
        coeff_diff = np.linalg.norm(enet_ridge.coefficients_ - ridge.coefficients_)
        assert coeff_diff < 0.1
        
        # alpha=1 should be like Lasso
        enet_lasso = ElasticNetRegression(lambda_reg=0.1, alpha=1.0)
        lasso = LassoRegression(lambda_reg=0.1)
        
        enet_lasso.fit(X, y)
        lasso.fit(X, y)
        
        coeff_diff_lasso = np.linalg.norm(enet_lasso.coefficients_ - lasso.coefficients_)
        assert coeff_diff_lasso < 0.1


class TestLogisticRegression:
    """Test Logistic Regression implementation."""
    
    def test_logistic_regression_binary(self):
        """Test binary logistic regression."""
        # Generate separable binary data
        np.random.seed(42)
        X1 = np.random.randn(20, 2) + [2, 2]
        X2 = np.random.randn(20, 2) + [-2, -2]
        X = np.vstack([X1, X2])
        y = np.hstack([np.ones(20), np.zeros(20)])
        
        logistic = LogisticRegression(solver='gradient_descent', learning_rate=0.1, max_iter=1000)
        logistic.fit(X, y)
        
        # Should achieve good accuracy
        y_pred = logistic.predict(X)
        accuracy = np.mean(y_pred == y)
        assert accuracy >= 0.8
        
        # Probabilities should be reasonable
        y_prob = logistic.predict_proba(X)
        assert np.all((y_prob >= 0) & (y_prob <= 1))
    
    def test_logistic_regression_multiclass(self):
        """Test multiclass logistic regression."""
        # 3-class problem
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 3, 60)
        
        logistic = LogisticRegression(multi_class='ovr')
        logistic.fit(X, y)
        
        y_pred = logistic.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset(set([0, 1, 2]))
        
        # Test multinomial approach
        logistic_multi = LogisticRegression(multi_class='multinomial')
        logistic_multi.fit(X, y)
        
        y_pred_multi = logistic_multi.predict(X)
        assert len(y_pred_multi) == len(y)
    
    def test_logistic_regularization(self):
        """Test logistic regression with regularization."""
        X = np.random.randn(40, 5)
        y = np.random.randint(0, 2, 40)
        
        logistic_l1 = LogisticRegression(regularization='l1', lambda_reg=0.1)
        logistic_l2 = LogisticRegression(regularization='l2', lambda_reg=0.1)
        
        logistic_l1.fit(X, y)
        logistic_l2.fit(X, y)
        
        # L1 should produce sparser coefficients
        l1_zeros = np.sum(np.abs(logistic_l1.coefficients_) < 1e-3)
        l2_zeros = np.sum(np.abs(logistic_l2.coefficients_) < 1e-3)
        assert l1_zeros >= l2_zeros


class TestPoissonRegression:
    """Test Poisson Regression implementation."""
    
    def test_poisson_regression_basic(self):
        """Test basic Poisson regression."""
        # Generate Poisson count data
        X = np.random.randn(30, 2)
        linear_pred = X @ [0.5, -0.3] + 1.0
        y = np.random.poisson(np.exp(linear_pred))
        
        poisson = PoissonRegression()
        poisson.fit(X, y)
        
        y_pred = poisson.predict(X)
        assert len(y_pred) == len(y)
        assert np.all(y_pred >= 0)  # Poisson predictions should be non-negative
    
    def test_poisson_link_function(self):
        """Test Poisson link function (log)."""
        poisson = PoissonRegression()
        
        # Test log link
        linear_pred = np.array([0, 1, 2])
        mean_pred = poisson._inverse_link(linear_pred)
        expected = np.exp(linear_pred)
        np.testing.assert_array_almost_equal(mean_pred, expected)


class TestBayesianLinearRegression:
    """Test Bayesian Linear Regression implementation."""
    
    def test_bayesian_regression_basic(self):
        """Test basic Bayesian regression."""
        X = np.random.randn(25, 3)
        y = np.random.randn(25)
        
        bayes_lr = BayesianLinearRegression(prior_precision=1.0, noise_precision=1.0)
        bayes_lr.fit(X, y)
        
        # Should have posterior parameters
        assert hasattr(bayes_lr, 'posterior_mean_')
        assert hasattr(bayes_lr, 'posterior_covariance_')
        
        # Predictions with uncertainty
        y_pred, y_var = bayes_lr.predict(X, return_uncertainty=True)
        assert len(y_pred) == len(y)
        assert len(y_var) == len(y)
        assert np.all(y_var >= 0)  # Variance should be non-negative
    
    def test_bayesian_credible_intervals(self):
        """Test credible intervals."""
        X = np.random.randn(20, 2)
        y = np.random.randn(20)
        
        bayes_lr = BayesianLinearRegression()
        bayes_lr.fit(X, y)
        
        intervals = bayes_lr.credible_intervals(X, confidence=0.95)
        assert intervals.shape == (len(X), 2)  # Lower and upper bounds
        
        # Lower bound should be less than upper bound
        assert np.all(intervals[:, 0] <= intervals[:, 1])
    
    def test_bayesian_sampling(self):
        """Test posterior sampling."""
        X = np.random.randn(15, 2)
        y = np.random.randn(15)
        
        bayes_lr = BayesianLinearRegression()
        bayes_lr.fit(X, y)
        
        samples = bayes_lr.sample_posterior(X, n_samples=10)
        assert samples.shape == (10, len(X))


class TestGeneralizedLinearModel:
    """Test Generalized Linear Model implementation."""
    
    def test_glm_gaussian(self):
        """Test GLM with Gaussian family (should match linear regression)."""
        X = np.random.randn(30, 3)
        y = np.random.randn(30)
        
        glm = GeneralizedLinearModel(family='gaussian', link='identity')
        glm.fit(X, y)
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        # Should give similar results
        pred_diff = np.linalg.norm(glm.predict(X) - lr.predict(X))
        assert pred_diff < 0.1
    
    def test_glm_families(self):
        """Test different GLM families."""
        X = np.random.randn(25, 2)
        
        # Binomial family
        y_binomial = np.random.randint(0, 2, 25)
        glm_binomial = GeneralizedLinearModel(family='binomial', link='logit')
        glm_binomial.fit(X, y_binomial)
        pred_binomial = glm_binomial.predict(X)
        assert np.all((pred_binomial >= 0) & (pred_binomial <= 1))
        
        # Poisson family
        y_poisson = np.random.poisson(2, 25)
        glm_poisson = GeneralizedLinearModel(family='poisson', link='log')
        glm_poisson.fit(X, y_poisson)
        pred_poisson = glm_poisson.predict(X)
        assert np.all(pred_poisson >= 0)


class TestPolynomialFeatures:
    """Test Polynomial Features implementation."""
    
    def test_polynomial_features_basic(self):
        """Test basic polynomial feature generation."""
        X = np.array([[1, 2], [3, 4]])
        
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_poly = poly.fit_transform(X)
        
        # Should have correct number of features
        # For degree 2 with 2 variables: 1 + 2 + 3 = 6 features
        # [1, x1, x2, x1^2, x1*x2, x2^2]
        assert X_poly.shape[1] == 6
        
        # Test first row: [1, 1, 2, 1, 2, 4]
        expected_first_row = np.array([1, 1, 2, 1, 2, 4])
        np.testing.assert_array_almost_equal(X_poly[0], expected_first_row)
    
    def test_polynomial_degrees(self):
        """Test different polynomial degrees."""
        X = np.random.randn(10, 2)
        
        for degree in [1, 2, 3]:
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            
            # Should have correct number of features
            from math import comb
            expected_features = sum(comb(2 + d - 1, d) for d in range(degree + 1))
            assert X_poly.shape[1] == expected_features
    
    def test_polynomial_interaction_only(self):
        """Test interaction-only features."""
        X = np.array([[1, 2, 3]])
        
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Should include: [x1, x2, x3, x1*x2, x1*x3, x2*x3]
        # No squared terms: x1^2, x2^2, x3^2
        assert X_poly.shape[1] == 6


class TestRegularizationPath:
    """Test Regularization Path computation."""
    
    def test_lasso_path(self):
        """Test Lasso regularization path."""
        X = np.random.randn(30, 5)
        y = np.random.randn(30)
        
        lambdas = np.logspace(-3, 1, 10)
        path = RegularizationPath(model_type='lasso')
        coefficients, scores = path.compute_path(X, y, lambdas)
        
        assert coefficients.shape == (len(lambdas), X.shape[1])
        assert len(scores) == len(lambdas)
        
        # Coefficients should generally decrease in magnitude
        coeff_norms = np.linalg.norm(coefficients, axis=1)
        # Allow some non-monotonicity due to optimization
        assert coeff_norms[0] >= coeff_norms[-1]
    
    def test_ridge_path(self):
        """Test Ridge regularization path."""
        X = np.random.randn(25, 4)
        y = np.random.randn(25)
        
        lambdas = np.logspace(-2, 2, 8)
        path = RegularizationPath(model_type='ridge')
        coefficients, scores = path.compute_path(X, y, lambdas)
        
        assert coefficients.shape == (len(lambdas), X.shape[1])
        
        # Ridge coefficients should shrink monotonically
        coeff_norms = np.linalg.norm(coefficients, axis=1)
        # Generally decreasing (allow some tolerance)
        decreasing_count = np.sum(np.diff(coeff_norms) <= 0.1)
        assert decreasing_count >= len(lambdas) // 2


class TestCrossValidation:
    """Test Cross Validation implementation."""
    
    def test_kfold_cv(self):
        """Test K-fold cross validation."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        cv = CrossValidation(cv_type='kfold', n_folds=5, scoring='mse')
        
        model = LinearRegression()
        scores = cv.cross_validate(model, X, y)
        
        assert len(scores) == 5  # 5-fold CV
        assert all(isinstance(score, (int, float)) for score in scores)
    
    def test_cv_scoring_methods(self):
        """Test different scoring methods."""
        X = np.random.randn(40, 2)
        y = np.random.randn(40)
        
        cv = CrossValidation(cv_type='kfold', n_folds=3)
        model = LinearRegression()
        
        # Test different scoring methods
        for scoring in ['mse', 'mae', 'r2']:
            cv.scoring = scoring
            scores = cv.cross_validate(model, X, y)
            assert len(scores) == 3
    
    def test_stratified_cv(self):
        """Test stratified cross validation."""
        X = np.random.randn(60, 2)
        y = np.random.randint(0, 3, 60)  # 3-class classification
        
        cv = CrossValidation(cv_type='stratified', n_folds=3, scoring='accuracy')
        model = LogisticRegression(multi_class='ovr')
        
        scores = cv.cross_validate(model, X, y)
        assert len(scores) == 3


class TestFeatureSelection:
    """Test Feature Selection implementation."""
    
    def test_univariate_selection(self):
        """Test univariate feature selection."""
        # Create data with some relevant features
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0] + 2 * X[:, 2] + 0.1 * np.random.randn(50)  # Only features 0, 2 relevant
        
        selector = FeatureSelection(method='univariate', k=3)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 3  # Selected 3 features
        assert hasattr(selector, 'selected_features_')
        
        # Should select some of the relevant features
        relevant_selected = len(set(selector.selected_features_) & {0, 2})
        assert relevant_selected >= 1
    
    def test_recursive_feature_elimination(self):
        """Test recursive feature elimination."""
        X = np.random.randn(40, 8)
        y = np.random.randn(40)
        
        selector = FeatureSelection(method='rfe', k=4)
        X_selected = selector.fit_transform(X, y)
        
        assert X_selected.shape[1] == 4
        assert len(selector.selected_features_) == 4
    
    def test_l1_based_selection(self):
        """Test L1-based feature selection."""
        X = np.random.randn(35, 6)
        y = np.random.randn(35)
        
        selector = FeatureSelection(method='l1_based', threshold=0.01)
        X_selected = selector.fit_transform(X, y)
        
        # Should select some features (exact number depends on L1 sparsity)
        assert X_selected.shape[1] <= X.shape[1]
        assert X_selected.shape[1] > 0


def test_linear_model_interface():
    """Test that all models implement the LinearModel interface."""
    models = [
        LinearRegression(),
        RidgeRegression(lambda_reg=0.1),
        LassoRegression(lambda_reg=0.1),
        ElasticNetRegression(lambda_reg=0.1, alpha=0.5),
        LogisticRegression(),
        PoissonRegression(),
        BayesianLinearRegression()
    ]
    
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    
    for model in models:
        # Should have fit and predict methods
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Should be able to fit and predict
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert hasattr(model, 'coefficients_')


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])