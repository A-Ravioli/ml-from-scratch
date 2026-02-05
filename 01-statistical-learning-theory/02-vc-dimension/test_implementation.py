"""
Test suite for VC dimension and complexity measures implementations.
"""

import numpy as np
import pytest
from exercise import (
    PolynomialClassifiers, UnionOfIntervals, DecisionStumps,
    sauer_shelah_bound, verify_sauer_shelah_empirically,
    EmpiricalVCEstimator, growth_function_analysis,
    vc_dimension_composition_rules, compare_complexity_measures,
    compute_fat_shattering_dimension, annealed_vc_entropy
)


class TestAdvancedHypothesisClasses:
    """Test advanced hypothesis class implementations."""
    
    def test_polynomial_classifiers_theoretical_vc(self):
        """Test theoretical VC dimension computation for polynomials."""
        # 1D linear (degree 1): VC = 2
        poly_1d_deg1 = PolynomialClassifiers(dimension=1, degree=1)
        assert poly_1d_deg1.compute_vc_dimension_theoretical() == 2
        
        # 2D quadratic (degree 2): VC = C(2+2,2) = C(4,2) = 6
        poly_2d_deg2 = PolynomialClassifiers(dimension=2, degree=2)
        assert poly_2d_deg2.compute_vc_dimension_theoretical() == 6
        
        # 3D cubic (degree 3): VC = C(3+3,3) = C(6,3) = 20
        poly_3d_deg3 = PolynomialClassifiers(dimension=3, degree=3)
        assert poly_3d_deg3.compute_vc_dimension_theoretical() == 20
    
    def test_polynomial_feature_generation(self):
        """Test polynomial feature generation."""
        poly_2d_deg2 = PolynomialClassifiers(dimension=2, degree=2)
        
        # Test simple case
        X = np.array([[1, 2], [0, 1]])
        features = poly_2d_deg2._polynomial_features(X)
        
        # Should have columns: [1, x1, x2, x1^2, x1*x2, x2^2]
        assert features.shape[1] == 6
        assert features.shape[0] == 2
        
        # Check first row: [1, 1, 2, 1, 2, 4]
        expected_row1 = np.array([1, 1, 2, 1, 2, 4])
        np.testing.assert_array_almost_equal(features[0], expected_row1)
    
    def test_union_of_intervals_vc(self):
        """Test VC dimension of union of intervals."""
        # 1 interval: VC = 2
        union_1 = UnionOfIntervals(k=1)
        assert union_1.compute_vc_dimension_theoretical() == 2
        
        # 3 intervals: VC = 6
        union_3 = UnionOfIntervals(k=3)
        assert union_3.compute_vc_dimension_theoretical() == 6
        
        # 10 intervals: VC = 20
        union_10 = UnionOfIntervals(k=10)
        assert union_10.compute_vc_dimension_theoretical() == 20
    
    def test_union_intervals_predictions(self):
        """Test predictions for union of intervals."""
        union_2 = UnionOfIntervals(k=2)
        
        # Two intervals: [0,1] and [3,4]
        params = np.array([0, 1, 3, 4])
        X = np.array([-1, 0.5, 2, 3.5, 5])
        
        predictions = union_2.predict(params, X)
        expected = np.array([0, 1, 0, 1, 0])
        
        np.testing.assert_array_equal(predictions, expected)
    
    def test_decision_stumps_vc(self):
        """Test VC dimension of decision stumps."""
        # 1D stumps: VC = 1
        stumps_1d = DecisionStumps(dimension=1)
        assert stumps_1d.compute_vc_dimension_theoretical() == 1
        
        # 5D stumps: VC = 5
        stumps_5d = DecisionStumps(dimension=5)
        assert stumps_5d.compute_vc_dimension_theoretical() == 5
    
    def test_decision_stump_predictions(self):
        """Test decision stump predictions."""
        stumps_2d = DecisionStumps(dimension=2)
        
        X = np.array([[1, 2], [3, 1], [0, 4], [2, 3]])
        # Use coordinate 0, threshold 1.5, positive sign
        params = np.array([0, 1.5, 1])
        
        predictions = stumps_2d.predict(params, X)
        # x[0] - 1.5 for each point: [-0.5, 1.5, -1.5, 0.5]
        # Sign: [-1, 1, -1, 1]
        expected = np.array([-1, 1, -1, 1])
        
        np.testing.assert_array_equal(predictions, expected)


class TestSauerShelahBound:
    """Test Sauer-Shelah bound computation and verification."""
    
    def test_sauer_shelah_computation(self):
        """Test Sauer-Shelah bound computation."""
        # For VC dimension 2
        assert sauer_shelah_bound(1, 2) == 2  # C(1,0) + C(1,1) = 1 + 1
        assert sauer_shelah_bound(2, 2) == 4  # C(2,0) + C(2,1) + C(2,2) = 1 + 2 + 1
        assert sauer_shelah_bound(3, 2) == 7  # C(3,0) + C(3,1) + C(3,2) = 1 + 3 + 3
        assert sauer_shelah_bound(4, 2) == 11 # C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6
    
    def test_sauer_shelah_properties(self):
        """Test properties of Sauer-Shelah bound."""
        # Should be increasing in m
        d = 3
        bounds = [sauer_shelah_bound(m, d) for m in range(1, 8)]
        assert all(bounds[i] <= bounds[i+1] for i in range(len(bounds)-1))
        
        # Should equal 2^m for m <= d
        for m in range(1, 4):
            assert sauer_shelah_bound(m, 3) == 2**m
    
    def test_empirical_verification(self):
        """Test empirical verification of Sauer-Shelah lemma."""
        union_2 = UnionOfIntervals(k=2)  # VC dimension = 4
        
        # Should satisfy lemma
        satisfies = verify_sauer_shelah_empirically(
            union_2, vc_dimension=4, max_points=8
        )
        assert satisfies == True


class TestEmpiricalVCEstimation:
    """Test empirical VC dimension estimation."""
    
    def test_vc_estimator_intervals(self):
        """Test VC estimation for intervals."""
        intervals = UnionOfIntervals(k=1)  # True VC = 2
        estimator = EmpiricalVCEstimator(intervals)
        
        def random_1d_data(n):
            return np.random.uniform(-2, 2, (n, 1))
        
        estimated_vc, confidence = estimator.estimate_vc_dimension(
            random_1d_data, max_dimension=5, n_trials_per_dim=20
        )
        
        # Should be close to true value
        assert estimated_vc >= 2
        assert estimated_vc <= 4  # Allow some empirical error
        assert 0 <= confidence <= 1
    
    def test_vc_validation(self):
        """Test VC dimension validation."""
        stumps_3d = DecisionStumps(dimension=3)  # True VC = 3
        estimator = EmpiricalVCEstimator(stumps_3d)
        
        def random_3d_data(n):
            return np.random.randn(n, 3)
        
        # Should validate correctly for true VC dimension
        validation_score = estimator.validate_vc_estimate(
            3, random_3d_data, n_validation_trials=50
        )
        assert validation_score > 0.5  # Should succeed reasonably often
        
        # Should fail for too high estimate
        validation_score_high = estimator.validate_vc_estimate(
            10, random_3d_data, n_validation_trials=50
        )
        assert validation_score_high < validation_score


class TestGrowthFunction:
    """Test growth function analysis."""
    
    def test_growth_function_properties(self):
        """Test growth function analysis properties."""
        intervals = UnionOfIntervals(k=1)  # VC dimension = 2
        stumps_2d = DecisionStumps(dimension=2)  # VC dimension = 2
        
        classes = [intervals, stumps_2d]
        results = growth_function_analysis(classes, max_points=6)
        
        # Should return structured results
        assert isinstance(results, dict)
        assert len(results) == len(classes)
        
        # Each class should have growth function values
        for class_name, data in results.items():
            assert 'growth_function' in data
            assert 'theoretical_bounds' in data
            assert len(data['growth_function']) <= 6


class TestComplexityMeasures:
    """Test different complexity measures comparison."""
    
    def test_complexity_comparison(self):
        """Test comparison of different complexity measures."""
        poly_2d = PolynomialClassifiers(dimension=2, degree=1)  # Linear
        union_2 = UnionOfIntervals(k=2)
        
        classes = [poly_2d, union_2]
        sample_sizes = np.array([10, 20, 30])
        
        results = compare_complexity_measures(classes, sample_sizes)
        
        # Should return comparison data
        assert isinstance(results, dict)
        assert len(results) == len(classes)
        
        # Each class should have complexity measures
        for class_name, data in results.items():
            assert 'vc_dimension' in data
            assert 'growth_function' in data
    
    def test_fat_shattering_dimension(self):
        """Test fat-shattering dimension computation."""
        # Create simple function class
        def simple_function_class(X, params):
            return X @ params  # Linear functions
        
        def data_generator(n):
            return np.random.randn(n, 2)
        
        fat_dim = compute_fat_shattering_dimension(
            simple_function_class, gamma=0.1, 
            data_generator=data_generator, max_dimension=5
        )
        
        # Should return reasonable value
        assert isinstance(fat_dim, int)
        assert fat_dim >= 0


class TestCompositionRules:
    """Test VC dimension composition rules."""
    
    def test_composition_rules(self):
        """Test VC dimension composition rules."""
        results = vc_dimension_composition_rules()
        
        # Should demonstrate various composition properties
        assert isinstance(results, dict)
        assert 'union_bound' in results
        assert 'intersection_bound' in results
        assert 'product_bound' in results


class TestAnnealedVCEntropy:
    """Test annealed VC entropy computation."""
    
    def test_annealed_entropy(self):
        """Test annealed VC entropy computation."""
        stumps_2d = DecisionStumps(dimension=2)
        
        # Generate distribution samples
        distribution_samples = [np.random.randn(10, 2) for _ in range(5)]
        temperature_schedule = np.array([1.0, 0.5, 0.1])
        
        entropy_values = annealed_vc_entropy(
            stumps_2d, distribution_samples, temperature_schedule
        )
        
        # Should return entropy values for each temperature
        assert len(entropy_values) == len(temperature_schedule)
        assert all(isinstance(val, (int, float)) for val in entropy_values)


def test_hypothesis_class_interface():
    """Test that all hypothesis classes implement required interface."""
    classes = [
        PolynomialClassifiers(dimension=2, degree=2),
        UnionOfIntervals(k=3),
        DecisionStumps(dimension=5)
    ]
    
    rng = np.random.default_rng(0)

    for cls in classes:
        # Should have name
        assert hasattr(cls, 'name')
        assert isinstance(cls.name, str)
        
        # Should compute restrictions
        if hasattr(cls, "dimension"):
            points = rng.normal(size=(3, int(cls.dimension)))
        else:
            points = rng.normal(size=(3, 1))

        restrictions = cls.enumerate_restrictions(points)
        assert isinstance(restrictions, set)
        for labeling in restrictions:
            assert isinstance(labeling, tuple)
            assert len(labeling) == len(points)
        
        # Should compute growth function
        np.random.seed(0)
        growth_val = cls.compute_growth_function(5, n_trials=10)
        assert isinstance(growth_val, int)
        assert 1 <= growth_val <= 2**5


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
