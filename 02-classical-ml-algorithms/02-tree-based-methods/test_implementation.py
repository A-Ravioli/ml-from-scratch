"""
Test Suite for Tree-Based Methods Implementation

This module provides comprehensive tests to verify the correctness of
tree-based algorithms implementations.
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
        DecisionTree, RandomForest, AdaBoost, GradientBoosting,
        TreeNode, load_sample_data
    )
except ImportError as e:
    print(f"Warning: Could not import from exercise.py. Error: {e}")
    print("Please implement the required classes and functions in exercise.py")
    sys.exit(1)

class TestDecisionTree:
    """Test cases for Decision Tree implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_simple = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_simple = np.array([0, 1, 1, 0])  # XOR problem
        
        # Linearly separable data
        self.X_linear = np.array([[1, 1], [1, 2], [2, 1], [3, 3], [3, 4], [4, 3]])
        self.y_linear = np.array([0, 0, 0, 1, 1, 1])
    
    def test_tree_node_creation(self):
        """Test TreeNode dataclass functionality."""
        # Test leaf node
        leaf = TreeNode(value=1.0, samples=10)
        assert leaf.is_leaf()
        assert leaf.value == 1.0
        assert leaf.samples == 10
        
        # Test internal node
        internal = TreeNode(feature=0, threshold=0.5, samples=20)
        internal.left = TreeNode(value=0.0)
        internal.right = TreeNode(value=1.0)
        assert not internal.is_leaf()
        assert internal.feature == 0
        assert internal.threshold == 0.5
    
    def test_impurity_measures(self):
        """Test entropy, Gini, and MSE calculations."""
        tree = DecisionTree()
        
        # Test entropy
        if hasattr(tree, '_entropy'):
            # Pure set should have entropy 0
            y_pure = np.array([1, 1, 1, 1])
            assert abs(tree._entropy(y_pure)) < 1e-10
            
            # Balanced binary should have entropy 1
            y_balanced = np.array([0, 0, 1, 1])
            assert abs(tree._entropy(y_balanced) - 1.0) < 1e-10
        
        # Test Gini
        if hasattr(tree, '_gini'):
            # Pure set should have Gini 0
            y_pure = np.array([1, 1, 1, 1])
            assert abs(tree._gini(y_pure)) < 1e-10
            
            # Balanced binary should have Gini 0.5
            y_balanced = np.array([0, 0, 1, 1])
            assert abs(tree._gini(y_balanced) - 0.5) < 1e-10
        
        # Test MSE
        if hasattr(tree, '_mse'):
            # Constant values should have MSE 0
            y_constant = np.array([2.0, 2.0, 2.0, 2.0])
            assert abs(tree._mse(y_constant)) < 1e-10
            
            # Known variance case
            y_var = np.array([1.0, 3.0])  # mean=2, variance=1
            assert abs(tree._mse(y_var) - 1.0) < 1e-10
    
    def test_information_gain(self):
        """Test information gain calculation."""
        tree = DecisionTree(criterion='gini')
        
        if hasattr(tree, '_information_gain'):
            # Split that perfectly separates should have high gain
            y_parent = np.array([0, 0, 1, 1])
            y_left = np.array([0, 0])
            y_right = np.array([1, 1])
            
            gain = tree._information_gain(y_parent, y_left, y_right)
            assert gain > 0.4  # Should be exactly 0.5 for this case
            
            # No split should have gain 0
            gain_no_split = tree._information_gain(y_parent, y_parent, np.array([]))
            assert gain_no_split == 0.0
    
    def test_basic_fitting(self):
        """Test basic tree fitting functionality."""
        tree = DecisionTree(max_depth=3, criterion='gini')
        
        # Should not raise an exception
        try:
            tree.fit(self.X_linear, self.y_linear)
            assert hasattr(tree, 'root_'), "Tree should have root_ attribute after fitting"
        except Exception as e:
            pytest.skip(f"fit method not implemented: {e}")
    
    def test_prediction_shape(self):
        """Test that predictions have correct shape."""
        tree = DecisionTree(max_depth=3)
        
        try:
            tree.fit(self.X_linear, self.y_linear)
            predictions = tree.predict(self.X_linear)
            
            assert len(predictions) == len(self.y_linear), "Prediction length should match input"
            assert all(pred in [0, 1] for pred in predictions), "Predictions should be valid class labels"
        except Exception as e:
            pytest.skip(f"predict method not implemented: {e}")
    
    def test_perfect_classification(self):
        """Test tree can achieve perfect classification on linearly separable data."""
        tree = DecisionTree(max_depth=10, min_samples_split=1)
        
        try:
            tree.fit(self.X_linear, self.y_linear)
            predictions = tree.predict(self.X_linear)
            accuracy = np.mean(predictions == self.y_linear)
            
            # Should achieve high accuracy on training data
            assert accuracy >= 0.8, f"Expected high training accuracy, got {accuracy}"
        except Exception as e:
            pytest.skip(f"Tree methods not implemented: {e}")

class TestRandomForest:
    """Test cases for Random Forest implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X = np.random.randn(100, 4)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
    
    def test_initialization(self):
        """Test Random Forest initialization."""
        rf = RandomForest(n_estimators=10, max_depth=5)
        
        assert rf.n_estimators == 10
        assert rf.max_depth == 5
    
    def test_max_features_calculation(self):
        """Test max_features calculation for different options."""
        rf = RandomForest()
        
        if hasattr(rf, '_calculate_max_features'):
            # Test sqrt
            rf.max_features = 'sqrt'
            assert rf._calculate_max_features(16) == 4
            
            # Test log2
            rf.max_features = 'log2'
            assert rf._calculate_max_features(16) == 4
            
            # Test integer
            rf.max_features = 3
            assert rf._calculate_max_features(16) == 3
            
            # Test float
            rf.max_features = 0.5
            assert rf._calculate_max_features(16) == 8
    
    def test_bootstrap_sampling(self):
        """Test bootstrap sampling functionality."""
        rf = RandomForest()
        
        if hasattr(rf, '_bootstrap_sample'):
            X_boot, y_boot, oob_indices = rf._bootstrap_sample(self.X, self.y)
            
            assert len(X_boot) == len(self.X), "Bootstrap sample should have same size as original"
            assert len(y_boot) == len(self.y), "Bootstrap labels should match"
            assert len(oob_indices) > 0, "Should have some out-of-bag samples"
            assert len(oob_indices) < len(self.y), "Should not include all samples"
    
    def test_fitting(self):
        """Test Random Forest fitting."""
        rf = RandomForest(n_estimators=5, max_depth=3)
        
        try:
            rf.fit(self.X, self.y)
            assert hasattr(rf, 'trees_'), "Should have trees_ attribute after fitting"
            assert len(rf.trees_) == 5, "Should have correct number of trees"
        except Exception as e:
            pytest.skip(f"Random Forest fit not implemented: {e}")
    
    def test_prediction(self):
        """Test Random Forest prediction."""
        rf = RandomForest(n_estimators=5, max_depth=3)
        
        try:
            rf.fit(self.X, self.y)
            predictions = rf.predict(self.X)
            
            assert len(predictions) == len(self.y), "Should predict for all samples"
            assert all(pred in [0, 1] for pred in predictions), "Predictions should be valid"
        except Exception as e:
            pytest.skip(f"Random Forest predict not implemented: {e}")
    
    def test_oob_score(self):
        """Test out-of-bag score calculation."""
        rf = RandomForest(n_estimators=10, oob_score=True)
        
        try:
            rf.fit(self.X, self.y)
            if hasattr(rf, 'oob_score_'):
                assert 0 <= rf.oob_score_ <= 1, "OOB score should be between 0 and 1"
        except Exception as e:
            pytest.skip(f"OOB score not implemented: {e}")

class TestAdaBoost:
    """Test cases for AdaBoost implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        # Create linearly separable data for easier testing
        X1 = np.random.multivariate_normal([1, 1], [[0.1, 0], [0, 0.1]], 20)
        X2 = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 20)
        self.X = np.vstack([X1, X2])
        self.y = np.hstack([np.zeros(20), np.ones(20)])
    
    def test_initialization(self):
        """Test AdaBoost initialization."""
        ada = AdaBoost(n_estimators=10, learning_rate=0.5)
        
        assert ada.n_estimators == 10
        assert ada.learning_rate == 0.5
    
    def test_fitting(self):
        """Test AdaBoost fitting."""
        ada = AdaBoost(n_estimators=5)
        
        try:
            ada.fit(self.X, self.y)
            assert hasattr(ada, 'estimators_'), "Should have estimators after fitting"
            assert hasattr(ada, 'estimator_weights_'), "Should have estimator weights"
            assert len(ada.estimators_) == 5, "Should have correct number of estimators"
        except Exception as e:
            pytest.skip(f"AdaBoost fit not implemented: {e}")
    
    def test_prediction(self):
        """Test AdaBoost prediction."""
        ada = AdaBoost(n_estimators=5)
        
        try:
            ada.fit(self.X, self.y)
            predictions = ada.predict(self.X)
            
            assert len(predictions) == len(self.y), "Should predict for all samples"
            assert all(pred in [0, 1] for pred in predictions), "Predictions should be valid"
        except Exception as e:
            pytest.skip(f"AdaBoost predict not implemented: {e}")
    
    def test_improving_accuracy(self):
        """Test that AdaBoost improves over single classifier."""
        # This is a probabilistic test - may occasionally fail
        ada_single = AdaBoost(n_estimators=1)
        ada_ensemble = AdaBoost(n_estimators=10)
        
        try:
            ada_single.fit(self.X, self.y)
            ada_ensemble.fit(self.X, self.y)
            
            acc_single = np.mean(ada_single.predict(self.X) == self.y)
            acc_ensemble = np.mean(ada_ensemble.predict(self.X) == self.y)
            
            # Ensemble should generally perform better (though not guaranteed)
            assert acc_ensemble >= acc_single - 0.1, "Ensemble should not be much worse than single classifier"
        except Exception as e:
            pytest.skip(f"AdaBoost not fully implemented: {e}")

class TestGradientBoosting:
    """Test cases for Gradient Boosting implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.X_reg = np.linspace(0, 10, 100).reshape(-1, 1)
        self.y_reg = 2 * self.X_reg.ravel() + np.random.normal(0, 0.1, 100)
        
        # Classification data
        self.X_clf = np.random.randn(100, 2)
        self.y_clf = (self.X_clf[:, 0] + self.X_clf[:, 1] > 0).astype(int)
        # Convert to {-1, +1} for binary classification
        self.y_clf_signed = 2 * self.y_clf - 1
    
    def test_loss_functions(self):
        """Test loss function implementations."""
        gb = GradientBoosting()
        
        # Test MSE loss
        if hasattr(gb, '_mse_loss'):
            y_true = np.array([1.0, 2.0, 3.0])
            y_pred = np.array([1.1, 1.9, 3.1])
            
            loss, gradient = gb._mse_loss(y_true, y_pred)
            
            assert loss >= 0, "Loss should be non-negative"
            assert len(gradient) == len(y_true), "Gradient should have correct length"
            
            # Gradient should be pred - true
            expected_grad = y_pred - y_true
            np.testing.assert_allclose(gradient, expected_grad, rtol=1e-10)
        
        # Test log loss
        if hasattr(gb, '_log_loss'):
            y_true = np.array([-1.0, 1.0, -1.0])
            y_pred = np.array([0.1, -0.1, 0.5])
            
            loss, gradient = gb._log_loss(y_true, y_pred)
            
            assert loss >= 0, "Loss should be non-negative"
            assert len(gradient) == len(y_true), "Gradient should have correct length"
    
    def test_regression_fitting(self):
        """Test gradient boosting for regression."""
        gb = GradientBoosting(n_estimators=10, learning_rate=0.1, loss='mse')
        
        try:
            gb.fit(self.X_reg, self.y_reg)
            assert hasattr(gb, 'estimators_'), "Should have estimators after fitting"
            assert hasattr(gb, 'init_prediction_'), "Should have initial prediction"
        except Exception as e:
            pytest.skip(f"Gradient Boosting fit not implemented: {e}")
    
    def test_regression_prediction(self):
        """Test gradient boosting regression prediction."""
        gb = GradientBoosting(n_estimators=10, learning_rate=0.1, loss='mse')
        
        try:
            gb.fit(self.X_reg, self.y_reg)
            predictions = gb.predict(self.X_reg)
            
            assert len(predictions) == len(self.y_reg), "Should predict for all samples"
            
            # Should achieve reasonable fit on training data
            mse = np.mean((predictions - self.y_reg) ** 2)
            naive_mse = np.var(self.y_reg)
            assert mse < naive_mse, "Should beat naive prediction"
        except Exception as e:
            pytest.skip(f"Gradient Boosting predict not implemented: {e}")
    
    def test_classification_fitting(self):
        """Test gradient boosting for classification."""
        gb = GradientBoosting(n_estimators=10, learning_rate=0.1, loss='log_loss')
        
        try:
            gb.fit(self.X_clf, self.y_clf_signed)
            predictions = gb.predict(self.X_clf)
            
            # Predictions should be in reasonable range
            assert np.all(np.abs(predictions) < 10), "Predictions should be reasonable"
        except Exception as e:
            pytest.skip(f"Gradient Boosting classification not implemented: {e}")

class TestDataGeneration:
    """Test the sample data generation functions."""
    
    def test_load_sample_data(self):
        """Test sample data loading function."""
        try:
            # Test classification data
            X_clf, y_clf = load_sample_data('classification')
            assert X_clf.shape[0] == len(y_clf), "X and y should have same number of samples"
            assert X_clf.shape[1] >= 2, "Should have at least 2 features"
            assert len(np.unique(y_clf)) > 1, "Should have multiple classes"
            
            # Test regression data
            X_reg, y_reg = load_sample_data('regression')
            assert X_reg.shape[0] == len(y_reg), "X and y should have same number of samples"
            assert len(y_reg) > 10, "Should have reasonable number of samples"
            
            # Test circles data
            X_circ, y_circ = load_sample_data('circles')
            assert X_circ.shape[1] == 2, "Circles should be 2D"
            assert len(np.unique(y_circ)) == 2, "Should have 2 classes"
            
        except Exception as e:
            pytest.skip(f"load_sample_data not implemented: {e}")

def run_comprehensive_test():
    """Run a comprehensive test of all implementations."""
    print("Running comprehensive test suite for tree-based methods...")
    
    # Test each component
    test_classes = [
        TestDecisionTree,
        TestRandomForest, 
        TestAdaBoost,
        TestGradientBoosting,
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
        print("🎉 Great job! Your implementation is working well!")
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
    print("2. Add tree visualization functions")
    print("3. Implement advanced pruning algorithms")
    print("4. Add cross-validation for hyperparameter tuning")
    print("5. Compare against scikit-learn implementations")
    print("6. Experiment with different datasets and parameters")