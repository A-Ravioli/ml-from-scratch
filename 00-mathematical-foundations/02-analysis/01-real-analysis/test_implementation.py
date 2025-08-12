"""
Test suite for Real Analysis implementations.

Run this file to verify your implementations are correct.
"""

import numpy as np
import pytest
from exercise import (
    MetricSpace, euclidean_distance, manhattan_distance, chebyshev_distance,
    Sequence, ContinuousFunction, FixedPointIterator, GradientDescent,
    visualize_metric_balls, demonstrate_continuity_breakdown
)


class TestMetrics:
    """Test metric implementations."""
    
    def test_euclidean_distance(self):
        """Test Euclidean distance computation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 6, 8])
        
        expected = np.sqrt((4-1)**2 + (6-2)**2 + (8-3)**2)
        assert abs(euclidean_distance(x, y) - expected) < 1e-10
        
        # Test zero distance
        assert euclidean_distance(x, x) == 0
    
    def test_manhattan_distance(self):
        """Test Manhattan distance computation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 6, 8])
        
        expected = abs(4-1) + abs(6-2) + abs(8-3)
        assert abs(manhattan_distance(x, y) - expected) < 1e-10
    
    def test_chebyshev_distance(self):
        """Test Chebyshev distance computation."""
        x = np.array([1, 2, 3])
        y = np.array([4, 6, 8])
        
        expected = max(abs(4-1), abs(6-2), abs(8-3))
        assert abs(chebyshev_distance(x, y) - expected) < 1e-10
    
    def test_metric_properties(self):
        """Test that distances satisfy metric properties."""
        points = [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1])
        ]
        
        for dist_func in [euclidean_distance, manhattan_distance, chebyshev_distance]:
            space = MetricSpace(dist_func)
            assert space.verify_metric_properties(points)


class TestSequences:
    """Test sequence convergence analysis."""
    
    def test_convergent_sequence(self):
        """Test detection of convergent sequence."""
        # Sequence x_n = 1/n converges to 0 in ℝ
        def terms(n):
            return np.array([1/n])
        
        space = MetricSpace(euclidean_distance)
        seq = Sequence(terms, space)
        
        limit = np.array([0])
        converges, N = seq.check_convergence(limit, epsilon=1e-3)
        
        assert converges
        assert N is not None
        assert terms(N)[0] < 1e-3
    
    def test_cauchy_sequence(self):
        """Test Cauchy sequence detection."""
        # Sequence x_n = 1/n is Cauchy in ℝ
        def terms(n):
            return np.array([1/n])
        
        space = MetricSpace(euclidean_distance)
        seq = Sequence(terms, space)
        
        is_cauchy, N = seq.is_cauchy(epsilon=1e-3)
        
        assert is_cauchy
        assert N is not None
    
    def test_divergent_sequence(self):
        """Test detection of divergent sequence."""
        # Sequence x_n = n diverges
        def terms(n):
            return np.array([n])
        
        space = MetricSpace(euclidean_distance)
        seq = Sequence(terms, space)
        
        # Should not converge to any finite limit
        limit = np.array([0])
        converges, _ = seq.check_convergence(limit, max_n=100)
        assert not converges


class TestContinuity:
    """Test continuity checking."""
    
    def test_continuous_function(self):
        """Test continuity of f(x) = 2x."""
        def f(x):
            return 2 * x
        
        space = MetricSpace(euclidean_distance)
        func = ContinuousFunction(f, space, space)
        
        x0 = np.array([1.0])
        is_cont, delta = func.check_continuity_at_point(x0, epsilon=0.1)
        
        assert is_cont
        assert delta is not None
        assert delta > 0
    
    def test_discontinuous_function(self):
        """Test discontinuity detection."""
        # Step function
        def f(x):
            return np.array([1.0 if x[0] >= 0 else -1.0])
        
        space = MetricSpace(euclidean_distance)
        func = ContinuousFunction(f, space, space)
        
        x0 = np.array([0.0])
        is_cont, _ = func.check_continuity_at_point(x0, epsilon=0.5)
        
        assert not is_cont
    
    def test_uniform_continuity(self):
        """Test uniform continuity on bounded domain."""
        # f(x) = x^2 is uniformly continuous on [0, 1]
        def f(x):
            return x**2
        
        space = MetricSpace(euclidean_distance)
        func = ContinuousFunction(f, space, space)
        
        # Sample points from [0, 1]
        domain_points = [np.array([i/10]) for i in range(11)]
        
        is_uniform, delta = func.check_uniform_continuity(domain_points, epsilon=0.1)
        
        assert is_uniform
        assert delta is not None


class TestFixedPoint:
    """Test fixed point iteration."""
    
    def test_contraction_mapping(self):
        """Test fixed point of contraction mapping."""
        # f(x) = 0.5x + 1 has fixed point at x = 2
        def f(x):
            return 0.5 * x + 1
        
        space = MetricSpace(euclidean_distance)
        iterator = FixedPointIterator(f, space)
        
        # Estimate Lipschitz constant
        sample_points = [np.array([i]) for i in range(-5, 6)]
        L = iterator.estimate_lipschitz_constant(sample_points)
        
        assert L < 1  # Contraction
        assert abs(L - 0.5) < 0.1  # Should be close to 0.5
        
        # Find fixed point
        x0 = np.array([0.0])
        fixed_point, distances, converged = iterator.iterate(x0)
        
        assert converged
        assert abs(fixed_point[0] - 2.0) < 1e-6
        
        # Verify it's actually a fixed point
        assert abs(f(fixed_point)[0] - fixed_point[0]) < 1e-6
        
        # Check convergence is geometric
        ratios = [distances[i+1] / distances[i] for i in range(len(distances)-1) if distances[i] > 1e-10]
        if ratios:
            assert all(r < 0.6 for r in ratios)  # Should be approximately 0.5
    
    def test_banach_verification(self):
        """Test Banach fixed point theorem verification."""
        def f(x):
            return 0.5 * x + 1
        
        space = MetricSpace(euclidean_distance)
        iterator = FixedPointIterator(f, space)
        
        x0 = np.array([0.0])
        sample_points = [np.array([i]) for i in range(-5, 6)]
        
        results = iterator.verify_banach_theorem(x0, sample_points)
        
        assert results['is_contraction']
        assert results['converged']
        assert results['unique_fixed_point']


class TestGradientDescent:
    """Test gradient descent implementation."""
    
    def test_quadratic_optimization(self):
        """Test GD on quadratic function."""
        # f(x) = 0.5 * ||x||^2
        def loss(x):
            return 0.5 * np.sum(x**2)
        
        def grad(x):
            return x
        
        optimizer = GradientDescent(loss, grad)
        
        x0 = np.array([1.0, 2.0])
        results = optimizer.optimize(x0, learning_rate=0.1, max_iters=100)
        
        final_x = results['trajectory'][-1]
        assert np.linalg.norm(final_x) < 1e-5  # Should converge to origin
        
        # Check that loss decreases
        losses = results['losses']
        assert all(losses[i+1] <= losses[i] for i in range(len(losses)-1))
    
    def test_convergence_rate_analysis(self):
        """Test convergence rate analysis."""
        def loss(x):
            return 0.5 * np.sum(x**2)
        
        def grad(x):
            return x
        
        optimizer = GradientDescent(loss, grad)
        
        x0 = np.array([10.0])
        results = optimizer.optimize(x0, learning_rate=0.5, max_iters=20)
        
        analysis = optimizer.analyze_convergence_rate(results)
        
        # For this problem with lr=0.5, convergence should be linear with rate 0.5
        assert analysis['convergence_type'] == 'linear'
        assert abs(analysis['rate'] - 0.5) < 0.1


def test_visualizations():
    """Test that visualization functions run without errors."""
    metrics = [
        ("Euclidean", MetricSpace(euclidean_distance)),
        ("Manhattan", MetricSpace(manhattan_distance)),
        ("Chebyshev", MetricSpace(chebyshev_distance))
    ]
    
    # Should create visualization without errors
    visualize_metric_balls(metrics)
    
    # Should demonstrate continuity breakdown
    demonstrate_continuity_breakdown()


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])