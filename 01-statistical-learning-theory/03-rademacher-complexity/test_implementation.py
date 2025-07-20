"""
Test suite for Rademacher complexity implementations.
"""

import numpy as np
import pytest
from exercise import (
    LinearFunctions, FiniteFunctionClass, empirical_rademacher_complexity,
    rademacher_generalization_bound, RadmacherComplexityAnalyzer,
    symmetrization_lemma_verification, GaussianComplexity,
    local_rademacher_complexity, chaining_bound_verification,
    StabilityRadmacherConnection, kernel_rademacher_complexity,
    neural_network_rademacher_analysis, FunctionClass
)


class TestFunctionClasses:
    """Test function class implementations."""
    
    def test_linear_functions_basic(self):
        """Test basic linear function class functionality."""
        linear_class = LinearFunctions(dimension=3, bound=2.0)
        
        # Test function sampling
        params = linear_class.sample_function()
        assert len(params) == 3
        assert np.linalg.norm(params) <= 2.0 + 1e-10  # Allow numerical error
        
        # Test evaluation
        X = np.random.randn(5, 3)
        values = linear_class.evaluate(params, X)
        
        # Should equal X @ params
        expected = X @ params
        np.testing.assert_array_almost_equal(values, expected)
    
    def test_linear_functions_theoretical_bound(self):
        """Test theoretical Rademacher complexity for linear functions."""
        linear_class = LinearFunctions(dimension=4, bound=1.5)
        
        X = np.random.randn(20, 4)
        theoretical_rad = linear_class.theoretical_rademacher_complexity(X)
        
        # Should be reasonable value
        assert isinstance(theoretical_rad, (int, float))
        assert theoretical_rad >= 0
    
    def test_finite_function_class(self):
        """Test finite function class."""
        # Create simple finite class
        functions = [
            np.array([1, -1, 1, -1]),
            np.array([-1, 1, -1, 1]),
            np.array([1, 1, -1, -1])
        ]
        finite_class = FiniteFunctionClass(functions)
        
        # Test sampling
        params = finite_class.sample_function()
        assert len(params) == 1
        assert 0 <= params[0] < len(functions)
        
        # Test evaluation
        X = np.zeros((4, 1))  # Dummy X, not used for finite classes
        values = finite_class.evaluate(params, X)
        
        # Should return one of the predefined functions
        idx = int(params[0])
        np.testing.assert_array_equal(values, functions[idx])
    
    def test_finite_class_theoretical_bound(self):
        """Test theoretical bound for finite function class."""
        functions = [np.random.randn(10) for _ in range(5)]
        finite_class = FiniteFunctionClass(functions)
        
        X = np.zeros((10, 1))  # Not used
        theoretical_rad = finite_class.theoretical_rademacher_complexity(X)
        
        # Should use Massart's lemma
        assert isinstance(theoretical_rad, (int, float))
        assert theoretical_rad >= 0


class TestEmpiricalComplexity:
    """Test empirical Rademacher complexity computation."""
    
    def test_empirical_complexity_linear(self):
        """Test empirical complexity for linear functions."""
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        X = np.random.randn(15, 2)
        
        emp_rad, emp_std = empirical_rademacher_complexity(
            linear_class, X, n_samples=100
        )
        
        # Should return reasonable values
        assert isinstance(emp_rad, (int, float))
        assert isinstance(emp_std, (int, float))
        assert emp_rad >= 0
        assert emp_std >= 0
    
    def test_empirical_complexity_finite(self):
        """Test empirical complexity for finite class."""
        functions = [np.random.randn(8) for _ in range(4)]
        finite_class = FiniteFunctionClass(functions)
        X = np.zeros((8, 1))
        
        emp_rad, emp_std = empirical_rademacher_complexity(
            finite_class, X, n_samples=200
        )
        
        assert emp_rad >= 0
        assert emp_std >= 0
    
    def test_complexity_scaling(self):
        """Test that complexity scales appropriately."""
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        
        # Larger sample size should give smaller complexity
        X_small = np.random.randn(10, 2)
        X_large = np.random.randn(50, 2)
        
        rad_small, _ = empirical_rademacher_complexity(linear_class, X_small)
        rad_large, _ = empirical_rademacher_complexity(linear_class, X_large)
        
        # Generally expect rad_small >= rad_large (though not guaranteed)
        assert rad_small >= 0 and rad_large >= 0


class TestGeneralizationBounds:
    """Test generalization bound computation."""
    
    def test_rademacher_bound(self):
        """Test Rademacher generalization bound."""
        empirical_risk = 0.15
        rademacher_complexity = 0.05
        confidence = 0.1
        sample_size = 100
        
        bound = rademacher_generalization_bound(
            empirical_risk, rademacher_complexity, confidence, sample_size
        )
        
        # Should be reasonable upper bound
        assert isinstance(bound, (int, float))
        assert bound >= empirical_risk  # Should be upper bound
    
    def test_bound_properties(self):
        """Test properties of generalization bounds."""
        base_params = {
            'empirical_risk': 0.1,
            'rademacher_complexity': 0.03,
            'sample_size': 100
        }
        
        # Lower confidence should give tighter bound
        bound_high_conf = rademacher_generalization_bound(
            confidence=0.01, **base_params
        )
        bound_low_conf = rademacher_generalization_bound(
            confidence=0.1, **base_params
        )
        
        assert bound_high_conf >= bound_low_conf


class TestComplexityAnalyzer:
    """Test Rademacher complexity analyzer."""
    
    def test_scaling_analysis(self):
        """Test scaling analysis with sample size."""
        analyzer = RadmacherComplexityAnalyzer()
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        
        def data_generator(n):
            return np.random.randn(n, 2)
        
        results = analyzer.analyze_scaling_with_sample_size(
            linear_class, data_generator, 
            sample_sizes=[10, 20, 30], n_trials=5
        )
        
        # Should return structured results
        assert isinstance(results, dict)
        assert 'sample_sizes' in results
        assert 'complexities' in results
    
    def test_theoretical_empirical_comparison(self):
        """Test comparison of theoretical and empirical complexity."""
        analyzer = RadmacherComplexityAnalyzer()
        linear_class = LinearFunctions(dimension=3, bound=1.0)
        X = np.random.randn(25, 3)
        
        comparison = analyzer.compare_theoretical_empirical(
            linear_class, X, n_monte_carlo=50
        )
        
        assert isinstance(comparison, dict)
        assert 'empirical' in comparison
        assert 'theoretical' in comparison
    
    def test_composition_properties(self):
        """Test function composition properties study."""
        analyzer = RadmacherComplexityAnalyzer()
        
        # Create base classes
        linear1 = LinearFunctions(dimension=2, bound=1.0)
        linear2 = LinearFunctions(dimension=2, bound=0.5)
        base_classes = [linear1, linear2]
        
        X = np.random.randn(20, 2)
        
        results = analyzer.study_composition_properties(base_classes, X)
        
        assert isinstance(results, dict)
        assert 'scaling_property' in results
        assert 'sum_property' in results


class TestSymmetrizationLemma:
    """Test symmetrization lemma verification."""
    
    def test_symmetrization_verification(self):
        """Test empirical verification of symmetrization lemma."""
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        
        def distribution():
            return np.random.randn(2)
        
        results = symmetrization_lemma_verification(
            linear_class, distribution, sample_size=20, n_trials=30
        )
        
        # Should verify the lemma relationship
        assert isinstance(results, dict)
        assert 'lemma_satisfied' in results
        assert 'generalization_gap' in results
        assert 'rademacher_bound' in results


class TestGaussianComplexity:
    """Test Gaussian complexity computation."""
    
    def test_gaussian_complexity_computation(self):
        """Test Gaussian complexity computation."""
        gauss_analyzer = GaussianComplexity()
        linear_class = LinearFunctions(dimension=3, bound=1.0)
        X = np.random.randn(15, 3)
        
        gauss_complexity = gauss_analyzer.compute_gaussian_complexity(
            linear_class, X, n_samples=100
        )
        
        assert isinstance(gauss_complexity, (int, float))
        assert gauss_complexity >= 0
    
    def test_rademacher_gaussian_comparison(self):
        """Test comparison between Rademacher and Gaussian complexity."""
        gauss_analyzer = GaussianComplexity()
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        X = np.random.randn(20, 2)
        
        comparison = gauss_analyzer.compare_rademacher_gaussian(linear_class, X)
        
        assert isinstance(comparison, dict)
        assert 'rademacher' in comparison
        assert 'gaussian' in comparison
        assert 'ratio' in comparison
        
        # Theory: Rademacher ≤ √(π/2) * Gaussian
        ratio = comparison['ratio']
        assert ratio <= np.sqrt(np.pi/2) + 0.1  # Allow some empirical error


class TestLocalComplexity:
    """Test local Rademacher complexity."""
    
    def test_local_complexity(self):
        """Test local Rademacher complexity computation."""
        linear_class = LinearFunctions(dimension=2, bound=1.0)
        X = np.random.randn(10, 2)
        center_function = np.array([0.5, -0.3])
        radius = 0.2
        
        local_rad = local_rademacher_complexity(
            linear_class, X, center_function, radius, n_samples=50
        )
        
        assert isinstance(local_rad, (int, float))
        assert local_rad >= 0
        
        # Local complexity should be smaller than global
        global_rad, _ = empirical_rademacher_complexity(linear_class, X, 50)
        assert local_rad <= global_rad + 0.1  # Allow some empirical variation


class TestChainingBound:
    """Test chaining bound verification."""
    
    def test_chaining_verification(self):
        """Test Dudley's chaining bound verification."""
        results = chaining_bound_verification()
        
        # Should return verification results
        assert isinstance(results, dict)
        assert 'chaining_bound' in results
        assert 'empirical_complexity' in results
        assert 'bound_satisfied' in results


class TestStabilityConnection:
    """Test stability-Rademacher complexity connection."""
    
    def test_stability_analysis(self):
        """Test stable algorithm analysis."""
        stability_analyzer = StabilityRadmacherConnection()
        
        def simple_algorithm(X, y):
            # Simple stable algorithm (return mean)
            return np.mean(X, axis=0)
        
        def data_generator():
            return np.random.randn(20, 2), np.random.choice([-1, 1], 20)
        
        results = stability_analyzer.analyze_stable_algorithm(
            simple_algorithm, data_generator, 
            stability_parameter=0.1, sample_sizes=[10, 20]
        )
        
        assert isinstance(results, dict)
        assert 'complexity_bounds' in results
        assert 'stability_verification' in results


class TestKernelComplexity:
    """Test kernel Rademacher complexity."""
    
    def test_kernel_complexity_bound(self):
        """Test kernel complexity bound computation."""
        # Create simple kernel matrix
        X = np.random.randn(15, 3)
        K = X @ X.T  # Linear kernel
        radius = 1.0
        
        kernel_rad = kernel_rademacher_complexity(K, radius)
        
        assert isinstance(kernel_rad, (int, float))
        assert kernel_rad >= 0
        
        # Should scale with trace of kernel matrix
        large_radius_rad = kernel_rademacher_complexity(K, radius * 2)
        assert large_radius_rad >= kernel_rad
    
    def test_kernel_properties(self):
        """Test kernel complexity properties."""
        # Diagonal kernel matrix
        n = 10
        K_diag = np.eye(n)
        
        rad_diag = kernel_rademacher_complexity(K_diag, 1.0)
        
        # Full kernel matrix  
        K_full = np.ones((n, n))
        rad_full = kernel_rademacher_complexity(K_full, 1.0)
        
        # Full matrix should have higher complexity
        assert rad_full >= rad_diag


class TestNeuralNetworkComplexity:
    """Test neural network Rademacher complexity analysis."""
    
    def test_neural_network_bound(self):
        """Test neural network complexity bound."""
        layer_widths = [5, 10, 8, 1]
        spectral_norms = [1.0, 1.5, 2.0]  # One per layer (excluding input)
        input_bound = 1.0
        sample_size = 50
        
        nn_rad = neural_network_rademacher_analysis(
            layer_widths, spectral_norms, input_bound, sample_size
        )
        
        assert isinstance(nn_rad, (int, float))
        assert nn_rad >= 0
    
    def test_network_scaling(self):
        """Test network complexity scaling."""
        base_widths = [3, 5, 1]
        base_norms = [1.0, 1.0]
        
        # Larger spectral norms should increase complexity
        small_rad = neural_network_rademacher_analysis(
            base_widths, base_norms, 1.0, 30
        )
        large_rad = neural_network_rademacher_analysis(
            base_widths, [2.0, 2.0], 1.0, 30
        )
        
        assert large_rad >= small_rad


def test_function_class_interface():
    """Test that all function classes implement required interface."""
    classes = [
        LinearFunctions(dimension=3, bound=1.0),
        FiniteFunctionClass([np.random.randn(5) for _ in range(3)])
    ]
    
    for cls in classes:
        # Should have evaluate method
        X = np.random.randn(4, 3)
        params = cls.sample_function()
        values = cls.evaluate(params, X)
        
        assert isinstance(values, np.ndarray)
        assert len(values) == len(X)
        
        # Should sample functions
        params2 = cls.sample_function()
        assert isinstance(params2, np.ndarray)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])