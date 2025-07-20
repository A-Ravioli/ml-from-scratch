"""
Test suite for PAC Learning implementations.
"""

import numpy as np
import pytest
from exercise import (
    LinearClassifiers, AxisAlignedRectangles, IntervalClassifiers,
    estimate_vc_dimension_empirically, test_all_labelings_shatterable,
    ERM, pac_learning_experiment, growth_function_computation,
    verify_sauer_shelah_lemma, HypothesisClass
)


class TestHypothesisClasses:
    """Test hypothesis class implementations."""
    
    def test_linear_classifiers_vc_dimension(self):
        """Test VC dimension of linear classifiers."""
        # 1D linear classifiers
        linear_1d = LinearClassifiers(dimension=1)
        assert linear_1d.compute_vc_dimension() == 2
        
        # 2D linear classifiers  
        linear_2d = LinearClassifiers(dimension=2)
        assert linear_2d.compute_vc_dimension() == 3
        
        # High dimensional
        linear_10d = LinearClassifiers(dimension=10)
        assert linear_10d.compute_vc_dimension() == 11
    
    def test_linear_classifier_predictions(self):
        """Test linear classifier predictions."""
        linear_2d = LinearClassifiers(dimension=2)
        
        # Simple test case
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        params = np.array([1, 1, 0])  # w=[1,1], b=0
        
        predictions = linear_2d.predict(params, X)
        expected = np.array([1, -1, 1, 1])  # sign(x1 + x2)
        
        assert np.array_equal(predictions, expected)
    
    def test_linear_shattering(self):
        """Test shattering capability of linear classifiers."""
        linear_2d = LinearClassifiers(dimension=2)
        
        # 3 points in general position should be shatterable
        points = np.array([[0, 0], [1, 0], [0, 1]])
        assert linear_2d.can_shatter(points) == True
        
        # 4 points in 2D should not be shatterable
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        assert linear_2d.can_shatter(points) == False
    
    def test_rectangles_vc_dimension(self):
        """Test VC dimension of rectangles."""
        rectangles = AxisAlignedRectangles()
        assert rectangles.compute_vc_dimension() == 4
    
    def test_rectangle_predictions(self):
        """Test rectangle predictions."""
        rectangles = AxisAlignedRectangles()
        
        X = np.array([[0.5, 0.5], [1.5, 1.5], [0.5, 1.5], [1.5, 0.5]])
        params = np.array([0, 1, 0, 1])  # Rectangle [0,1] × [0,1]
        
        predictions = rectangles.predict(params, X)
        expected = np.array([1, 0, 1, 1])
        
        assert np.array_equal(predictions, expected)
    
    def test_intervals_vc_dimension(self):
        """Test VC dimension of intervals."""
        intervals = IntervalClassifiers()
        assert intervals.compute_vc_dimension() == 2
    
    def test_interval_predictions(self):
        """Test interval predictions."""
        intervals = IntervalClassifiers()
        
        X = np.array([-1, 0, 0.5, 1, 2])
        params = np.array([0, 1])  # Interval [0, 1]
        
        predictions = intervals.predict(params, X)
        expected = np.array([0, 1, 1, 1, 0])
        
        assert np.array_equal(predictions, expected)


class TestSampleComplexity:
    """Test sample complexity calculations."""
    
    def test_linear_sample_complexity(self):
        """Test sample complexity bounds for linear classifiers."""
        linear_2d = LinearClassifiers(dimension=2)
        
        epsilon, delta = 0.1, 0.1
        bound = linear_2d.sample_complexity_bound(epsilon, delta, realizable=True)
        
        # Should be O((d + log(1/δ))/ε²)
        assert bound > 0
        assert isinstance(bound, int)
        
        # Smaller epsilon should require more samples
        bound_small = linear_2d.sample_complexity_bound(0.05, delta, realizable=True)
        assert bound_small > bound
    
    def test_sample_complexity_scaling(self):
        """Test sample complexity scaling with parameters."""
        linear_2d = LinearClassifiers(dimension=2)
        
        # Test epsilon scaling
        bounds_eps = []
        for eps in [0.1, 0.05, 0.02]:
            bound = linear_2d.sample_complexity_bound(eps, 0.1)
            bounds_eps.append(bound)
        
        # Should increase as epsilon decreases
        assert bounds_eps[0] < bounds_eps[1] < bounds_eps[2]
        
        # Test delta scaling  
        bounds_delta = []
        for delta in [0.1, 0.05, 0.01]:
            bound = linear_2d.sample_complexity_bound(0.1, delta)
            bounds_delta.append(bound)
        
        # Should increase as delta decreases
        assert bounds_delta[0] < bounds_delta[1] < bounds_delta[2]


class TestEmpiricalVCDimension:
    """Test empirical VC dimension estimation."""
    
    def test_linear_vc_estimation(self):
        """Test empirical VC dimension estimation for linear classifiers."""
        linear_2d = LinearClassifiers(dimension=2)
        
        def random_2d_points(n):
            return np.random.randn(n, 2)
        
        estimated_vc = estimate_vc_dimension_empirically(
            linear_2d, random_2d_points, max_dimension=5, n_trials=20
        )
        
        # Should be close to theoretical value of 3
        assert estimated_vc >= 3
        assert estimated_vc <= 5  # Allow some empirical error
    
    def test_interval_vc_estimation(self):
        """Test empirical VC dimension estimation for intervals."""
        intervals = IntervalClassifiers()
        
        def random_1d_points(n):
            return np.random.randn(n, 1)
        
        estimated_vc = estimate_vc_dimension_empirically(
            intervals, random_1d_points, max_dimension=5, n_trials=50
        )
        
        # Should be close to theoretical value of 2
        assert estimated_vc >= 2
        assert estimated_vc <= 3


class TestGrowthFunction:
    """Test growth function computation."""
    
    def test_interval_growth_function(self):
        """Test growth function for intervals."""
        intervals = IntervalClassifiers()
        
        def random_1d_points(n):
            return np.random.uniform(-2, 2, (n, 1))
        
        growth_values = growth_function_computation(
            intervals, random_1d_points, max_size=6
        )
        
        # For intervals: Π(1) = 2, Π(2) = 4, Π(m) ≤ m²+1 for m ≥ 3
        assert growth_values[0] == 2  # Π(1)
        assert growth_values[1] == 4  # Π(2)
        
        # Should satisfy Sauer-Shelah lemma
        assert verify_sauer_shelah_lemma(2, growth_values)
    
    def test_sauer_shelah_verification(self):
        """Test Sauer-Shelah lemma verification."""
        # Manual test case
        vc_dim = 2
        growth = [2, 4, 7, 8, 9, 10]  # Example values
        
        # Compute Sauer-Shelah bound
        def sauer_bound(m, d):
            from math import comb
            return sum(comb(m, i) for i in range(min(d+1, m+1)))
        
        # Check each value
        for m, pi_m in enumerate(growth, 1):
            bound = sauer_bound(m, vc_dim)
            assert pi_m <= bound


class TestERM:
    """Test Empirical Risk Minimization."""
    
    def test_erm_linear_separable(self):
        """Test ERM on linearly separable data."""
        linear_2d = LinearClassifiers(dimension=2)
        erm = ERM(linear_2d)
        
        # Generate linearly separable data
        np.random.seed(42)
        X = np.random.randn(50, 2)
        w_true = np.array([1, -1])
        y = np.sign(X @ w_true)
        
        def param_sampler():
            return np.random.randn(3)  # [w1, w2, b]
        
        best_params = erm.fit(X, y, param_sampler, n_candidates=1000)
        predictions = erm.predict(X, best_params)
        
        # Should achieve zero training error
        training_error = np.mean(predictions != y)
        assert training_error <= 0.1  # Allow some numerical error
    
    def test_erm_intervals(self):
        """Test ERM on interval learning."""
        intervals = IntervalClassifiers()
        erm = ERM(intervals)
        
        # Generate data with interval pattern
        X = np.array([-2, -1, 0, 0.5, 1, 1.5, 2]).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0, 0])  # Interval [0, 1]
        
        def param_sampler():
            a = np.random.uniform(-3, 2)
            b = np.random.uniform(a, 3)
            return np.array([a, b])
        
        best_params = erm.fit(X, y, param_sampler, n_candidates=1000)
        predictions = erm.predict(X, best_params)
        
        # Should achieve low training error
        training_error = np.mean(predictions != y)
        assert training_error <= 0.2


class TestPACExperiment:
    """Test PAC learning experiments."""
    
    def test_pac_experiment_structure(self):
        """Test that PAC experiment returns proper structure."""
        linear_1d = LinearClassifiers(dimension=1)
        
        def target_function(X):
            return np.sign(X[:, 0] - 0.5)
        
        def data_distribution(n):
            return np.random.uniform(-1, 2, (n, 1))
        
        results = pac_learning_experiment(
            linear_1d, target_function, data_distribution,
            sample_sizes=[20, 50], n_trials=5
        )
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'sample_sizes' in results
        assert 'true_risks' in results
        assert 'empirical_risks' in results
        
        # Check dimensions
        assert len(results['sample_sizes']) == 2
        assert len(results['true_risks']) == 2
        assert len(results['empirical_risks']) == 2


def test_hypothesis_class_interface():
    """Test that all hypothesis classes implement required interface."""
    classes = [
        LinearClassifiers(dimension=2),
        AxisAlignedRectangles(),
        IntervalClassifiers()
    ]
    
    for cls in classes:
        # Should have VC dimension
        vc_dim = cls.compute_vc_dimension()
        assert isinstance(vc_dim, int)
        assert vc_dim > 0
        
        # Should compute sample complexity
        bound = cls.sample_complexity_bound(0.1, 0.1)
        assert isinstance(bound, int)
        assert bound > 0


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])