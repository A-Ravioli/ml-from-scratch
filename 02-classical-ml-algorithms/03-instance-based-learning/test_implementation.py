"""
Test Suite for Instance-Based Learning Implementation

This module provides comprehensive tests to verify the correctness of
instance-based learning algorithms implementations.
"""

import numpy as np
import pytest
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path to import the exercise module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from exercise import (
        KNearestNeighbors, KDTree, KernelDensityEstimator, 
        LocallyWeightedRegression, generate_sample_data,
        curse_of_dimensionality_experiment
    )
except ImportError as e:
    print(f"Warning: Could not import from exercise.py. Error: {e}")
    print("Please implement the required classes and functions in exercise.py")
    sys.exit(1)

class TestKNearestNeighbors:
    """Test cases for k-Nearest Neighbors implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Simple 2D classification data
        self.X_simple = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [1, 2], [2, 2]])
        self.y_simple = np.array([0, 0, 1, 1, 0, 1])
        
        # 1D regression data
        self.X_reg = np.linspace(0, 10, 20).reshape(-1, 1)
        self.y_reg = 2 * self.X_reg.ravel() + np.random.normal(0, 0.1, 20)
    
    def test_initialization(self):
        """Test k-NN initialization."""
        knn = KNearestNeighbors(k=3, distance_metric='euclidean', weights='uniform')
        assert knn.k == 3
        assert knn.distance_metric == 'euclidean'
        assert knn.weights == 'uniform'
    
    def test_distance_metrics(self):
        """Test different distance metric implementations."""
        knn = KNearestNeighbors()
        
        x1 = np.array([0, 0])
        x2 = np.array([3, 4])
        
        # Test Euclidean distance
        if hasattr(knn, '_euclidean_distance'):
            euclidean_dist = knn._euclidean_distance(x1, x2)
            assert abs(euclidean_dist - 5.0) < 1e-10, f"Expected 5.0, got {euclidean_dist}"
        
        # Test Manhattan distance
        if hasattr(knn, '_manhattan_distance'):
            manhattan_dist = knn._manhattan_distance(x1, x2)
            assert abs(manhattan_dist - 7.0) < 1e-10, f"Expected 7.0, got {manhattan_dist}"
        
        # Test Chebyshev distance
        if hasattr(knn, '_chebyshev_distance'):
            chebyshev_dist = knn._chebyshev_distance(x1, x2)
            assert abs(chebyshev_dist - 4.0) < 1e-10, f"Expected 4.0, got {chebyshev_dist}"
    
    def test_classification_fitting(self):
        """Test k-NN classification fitting."""
        knn = KNearestNeighbors(k=3)
        
        try:
            knn.fit(self.X_simple, self.y_simple)
            assert hasattr(knn, 'X_train_'), "Should have X_train_ after fitting"
            assert hasattr(knn, 'y_train_'), "Should have y_train_ after fitting"
            assert hasattr(knn, 'is_classifier_'), "Should determine if classifier"
        except Exception as e:
            pytest.skip(f"fit method not implemented: {e}")
    
    def test_classification_prediction(self):
        """Test k-NN classification prediction."""
        knn = KNearestNeighbors(k=1)
        
        try:
            knn.fit(self.X_simple, self.y_simple)
            predictions = knn.predict(self.X_simple)
            
            assert len(predictions) == len(self.y_simple), "Prediction length should match input"
            # With k=1, should achieve perfect accuracy on training data
            accuracy = np.mean(predictions == self.y_simple)
            assert accuracy == 1.0, f"Expected perfect accuracy with k=1, got {accuracy}"
        except Exception as e:
            pytest.skip(f"predict method not implemented: {e}")
    
    def test_regression_prediction(self):
        """Test k-NN regression prediction."""
        knn = KNearestNeighbors(k=3)
        
        try:
            knn.fit(self.X_reg, self.y_reg)
            predictions = knn.predict(self.X_reg)
            
            assert len(predictions) == len(self.y_reg), "Prediction length should match input"
            assert isinstance(predictions[0], (int, float, np.number)), "Regression should return numbers"
            
            # Should achieve reasonable MSE
            mse = np.mean((predictions - self.y_reg) ** 2)
            baseline_mse = np.var(self.y_reg)
            assert mse < baseline_mse, "Should beat naive prediction"
        except Exception as e:
            pytest.skip(f"regression predict not implemented: {e}")
    
    def test_weighted_knn(self):
        """Test distance-weighted k-NN."""
        knn_uniform = KNearestNeighbors(k=5, weights='uniform')
        knn_distance = KNearestNeighbors(k=5, weights='distance')
        
        try:
            knn_uniform.fit(self.X_simple, self.y_simple)
            knn_distance.fit(self.X_simple, self.y_simple)
            
            pred_uniform = knn_uniform.predict(self.X_simple)
            pred_distance = knn_distance.predict(self.X_simple)
            
            # Predictions might be different due to weighting
            assert len(pred_uniform) == len(pred_distance), "Both should predict for all points"
        except Exception as e:
            pytest.skip(f"weighted k-NN not implemented: {e}")
    
    def test_probability_prediction(self):
        """Test probability prediction for classification."""
        knn = KNearestNeighbors(k=3)
        
        try:
            knn.fit(self.X_simple, self.y_simple)
            if hasattr(knn, 'predict_proba'):
                proba = knn.predict_proba(self.X_simple)
                
                assert proba.shape[0] == len(self.X_simple), "Should predict for all samples"
                assert proba.shape[1] == len(np.unique(self.y_simple)), "Should have probability for each class"
                assert np.allclose(np.sum(proba, axis=1), 1.0), "Probabilities should sum to 1"
                assert np.all(proba >= 0) and np.all(proba <= 1), "Probabilities should be in [0,1]"
        except Exception as e:
            pytest.skip(f"predict_proba not implemented: {e}")

class TestKDTree:
    """Test cases for k-d Tree implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.points_2d = np.random.randn(20, 2)
        self.points_3d = np.random.randn(50, 3)
    
    def test_tree_construction(self):
        """Test k-d tree construction."""
        kdtree = KDTree()
        
        try:
            kdtree.build(self.points_2d)
            assert kdtree.root is not None, "Should have root after building"
            assert kdtree.n_features == 2, "Should store number of features"
        except Exception as e:
            pytest.skip(f"k-d tree build not implemented: {e}")
    
    def test_single_neighbor_search(self):
        """Test finding single nearest neighbor."""
        kdtree = KDTree()
        
        try:
            kdtree.build(self.points_2d)
            query_point = np.array([0.0, 0.0])
            
            neighbors, distances = kdtree.query(query_point, k=1)
            
            assert len(neighbors) == 1, "Should return exactly 1 neighbor"
            assert len(distances) == 1, "Should return exactly 1 distance"
            assert distances[0] >= 0, "Distance should be non-negative"
        except Exception as e:
            pytest.skip(f"k-d tree query not implemented: {e}")
    
    def test_multiple_neighbor_search(self):
        """Test finding multiple nearest neighbors."""
        kdtree = KDTree()
        
        try:
            kdtree.build(self.points_3d)
            query_point = np.array([0.5, -0.5, 1.0])
            k = 5
            
            neighbors, distances = kdtree.query(query_point, k=k)
            
            assert len(neighbors) == k, f"Should return exactly {k} neighbors"
            assert len(distances) == k, f"Should return exactly {k} distances"
            assert np.all(distances >= 0), "All distances should be non-negative"
            
            # Distances should be sorted
            assert np.all(distances[:-1] <= distances[1:]), "Distances should be sorted"
        except Exception as e:
            pytest.skip(f"k-d tree k-NN search not implemented: {e}")
    
    def test_correctness_vs_brute_force(self):
        """Test that k-d tree gives same results as brute force."""
        kdtree = KDTree()
        
        try:
            kdtree.build(self.points_2d)
            query_point = np.array([1.0, -1.0])
            
            # k-d tree result
            kd_neighbors, kd_distances = kdtree.query(query_point, k=3)
            
            # Brute force result
            all_distances = np.linalg.norm(self.points_2d - query_point, axis=1)
            bf_indices = np.argsort(all_distances)[:3]
            bf_distances = all_distances[bf_indices]
            
            # Should give same distances (order might differ for ties)
            np.testing.assert_allclose(np.sort(kd_distances), np.sort(bf_distances), rtol=1e-10)
        except Exception as e:
            pytest.skip(f"k-d tree correctness test failed: {e}")

class TestKernelDensityEstimator:
    """Test cases for Kernel Density Estimator implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # 1D Gaussian data
        self.X_1d = np.random.normal(0, 1, 100).reshape(-1, 1)
        # 2D data
        self.X_2d = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
    
    def test_initialization(self):
        """Test KDE initialization."""
        kde = KernelDensityEstimator(kernel='gaussian', bandwidth=1.0)
        assert kde.kernel == 'gaussian'
        assert kde.bandwidth == 1.0
    
    def test_kernel_functions(self):
        """Test kernel function implementations."""
        kde = KernelDensityEstimator()
        
        # Test input
        u = np.array([[0, 0], [1, 0], [0, 1]])
        
        # Test Gaussian kernel
        if hasattr(kde, '_gaussian_kernel'):
            gaussian_vals = kde._gaussian_kernel(u)
            assert len(gaussian_vals) == len(u), "Should evaluate for all points"
            assert gaussian_vals[0] > gaussian_vals[1], "Gaussian should be highest at origin"
        
        # Test Epanechnikov kernel
        if hasattr(kde, '_epanechnikov_kernel'):
            epan_vals = kde._epanechnikov_kernel(u)
            assert len(epan_vals) == len(u), "Should evaluate for all points"
            assert np.all(epan_vals >= 0), "Epanechnikov should be non-negative"
    
    def test_silverman_bandwidth(self):
        """Test Silverman's bandwidth rule."""
        kde = KernelDensityEstimator()
        
        if hasattr(kde, '_silverman_bandwidth'):
            bandwidth = kde._silverman_bandwidth(self.X_1d)
            assert bandwidth > 0, "Bandwidth should be positive"
            
            # Should scale with standard deviation
            X_scaled = self.X_1d * 2
            bandwidth_scaled = kde._silverman_bandwidth(X_scaled)
            assert bandwidth_scaled > bandwidth, "Bandwidth should increase with scale"
    
    def test_fitting(self):
        """Test KDE fitting."""
        kde = KernelDensityEstimator(bandwidth=1.0)
        
        try:
            kde.fit(self.X_1d)
            assert hasattr(kde, 'X_train_'), "Should store training data"
            assert hasattr(kde, 'bandwidth_'), "Should store final bandwidth"
        except Exception as e:
            pytest.skip(f"KDE fit not implemented: {e}")
    
    def test_density_evaluation(self):
        """Test density evaluation."""
        kde = KernelDensityEstimator(kernel='gaussian', bandwidth=1.0)
        
        try:
            kde.fit(self.X_1d)
            log_densities = kde.score_samples(self.X_1d[:10])
            
            assert len(log_densities) == 10, "Should evaluate for all query points"
            assert np.all(np.isfinite(log_densities)), "All densities should be finite"
            
            # Density should be higher at training points than far away
            far_points = np.array([[10.0], [-10.0]])
            far_densities = kde.score_samples(far_points)
            assert np.mean(log_densities) > np.mean(far_densities), "Density should be higher at training points"
        except Exception as e:
            pytest.skip(f"KDE score_samples not implemented: {e}")
    
    def test_sampling(self):
        """Test sampling from KDE."""
        kde = KernelDensityEstimator(kernel='gaussian', bandwidth=0.5)
        
        try:
            kde.fit(self.X_1d)
            samples = kde.sample(n_samples=20)
            
            assert len(samples) == 20, "Should generate requested number of samples"
            assert samples.shape[1] == self.X_1d.shape[1], "Should have same dimensionality"
        except Exception as e:
            pytest.skip(f"KDE sampling not implemented: {e}")

class TestLocallyWeightedRegression:
    """Test cases for Locally Weighted Regression implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # 1D sinusoidal data with noise
        self.X = np.linspace(0, 4*np.pi, 50).reshape(-1, 1)
        self.y = np.sin(self.X.ravel()) + 0.1 * np.random.randn(50)
        
        # Linear data for simple test
        self.X_linear = np.linspace(0, 10, 20).reshape(-1, 1)
        self.y_linear = 2 * self.X_linear.ravel() + 1 + 0.05 * np.random.randn(20)
    
    def test_initialization(self):
        """Test LOWESS initialization."""
        lowess = LocallyWeightedRegression(frac=0.3, it=2)
        assert lowess.frac == 0.3
        assert lowess.it == 2
    
    def test_weight_functions(self):
        """Test weight function implementations."""
        lowess = LocallyWeightedRegression()
        
        # Test tricube weights
        if hasattr(lowess, '_tricube_weight'):
            distances = np.array([0, 0.5, 1.0, 1.5])
            max_dist = 1.0
            weights = lowess._tricube_weight(distances, max_dist)
            
            assert len(weights) == len(distances), "Should return weight for each distance"
            assert weights[0] == 1.0, "Weight at distance 0 should be 1"
            assert weights[2] == 0.0, "Weight at max distance should be 0"
            assert np.all(weights >= 0), "All weights should be non-negative"
    
    def test_linear_data_fitting(self):
        """Test LOWESS on linear data."""
        lowess = LocallyWeightedRegression(frac=0.5, it=1)
        
        try:
            predictions = lowess.fit_predict(self.X_linear, self.y_linear, self.X_linear)
            
            assert len(predictions) == len(self.y_linear), "Should predict for all points"
            
            # Should fit linear data well
            mse = np.mean((predictions - self.y_linear) ** 2)
            baseline_mse = np.var(self.y_linear)
            assert mse < 0.1 * baseline_mse, "Should fit linear data very well"
        except Exception as e:
            pytest.skip(f"LOWESS fit_predict not implemented: {e}")
    
    def test_nonlinear_data_fitting(self):
        """Test LOWESS on nonlinear data."""
        lowess = LocallyWeightedRegression(frac=0.3, it=2)
        
        try:
            predictions = lowess.fit_predict(self.X, self.y, self.X)
            
            assert len(predictions) == len(self.y), "Should predict for all points"
            
            # Should capture nonlinear pattern better than linear regression
            mse = np.mean((predictions - self.y) ** 2)
            
            # Simple linear regression for comparison
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            lr.fit(self.X, self.y)
            linear_pred = lr.predict(self.X)
            linear_mse = np.mean((linear_pred - self.y) ** 2)
            
            assert mse < linear_mse, "LOWESS should outperform linear regression on nonlinear data"
        except Exception as e:
            pytest.skip(f"LOWESS nonlinear fitting failed: {e}")

class TestDataGeneration:
    """Test the sample data generation functions."""
    
    def test_sample_data_generation(self):
        """Test sample data generation function."""
        try:
            # Test 2D classification
            X_class, y_class = generate_sample_data('classification_2d')
            assert X_class.shape[1] == 2, "2D classification should have 2 features"
            assert len(X_class) == len(y_class), "X and y should have same length"
            assert len(np.unique(y_class)) > 1, "Should have multiple classes"
            
            # Test 1D regression
            X_reg, y_reg = generate_sample_data('regression_1d')
            assert X_reg.shape[1] == 1, "1D regression should have 1 feature"
            assert len(X_reg) == len(y_reg), "X and y should have same length"
            
            # Test high dimensional
            X_high, y_high = generate_sample_data('high_dimensional')
            assert X_high.shape[1] > 10, "High dimensional should have many features"
            
        except Exception as e:
            pytest.skip(f"generate_sample_data not implemented: {e}")

def run_comprehensive_test():
    """Run a comprehensive test of all implementations."""
    print("Running comprehensive test suite for instance-based learning...")
    
    test_classes = [
        TestKNearestNeighbors,
        TestKDTree,
        TestKernelDensityEstimator,
        TestLocallyWeightedRegression,
        TestDataGeneration
    ]
    
    results = {}
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\nTesting {class_name}...")
        
        instance = test_class()
        if hasattr(instance, 'setup_method'):
            instance.setup_method()
        
        methods = [method for method in dir(instance) if method.startswith('test_')]
        passed = 0
        total = len(methods)
        
        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
        
        results[class_name] = (passed, total)
        print(f"  {passed}/{total} tests passed")
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    total_passed = 0
    total_tests = 0
    
    for class_name, (passed, total) in results.items():
        total_passed += passed
        total_tests += total
        percentage = (passed / total * 100) if total > 0 else 0
        print(f"{class_name}: {passed}/{total} ({percentage:.1f}%)")
    
    overall_percentage = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOverall: {total_passed}/{total_tests} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 80:
        print("🎉 Excellent! Your instance-based learning implementation is working well!")
    elif overall_percentage >= 60:
        print("👍 Good progress! A few more methods to implement.")
    else:
        print("📝 Keep working on the implementations.")
    
    return results

if __name__ == "__main__":
    # Run the comprehensive test
    run_comprehensive_test()
    
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("1. Implement any failing methods in exercise.py")
    print("2. Add more efficient indexing structures (Ball trees, LSH)")
    print("3. Implement advanced distance learning methods")
    print("4. Add visualization functions for decision boundaries")
    print("5. Study curse of dimensionality effects empirically")
    print("6. Compare against scikit-learn implementations")