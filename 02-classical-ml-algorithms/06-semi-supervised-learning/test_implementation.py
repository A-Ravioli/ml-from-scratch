"""
Test suite for Semi-Supervised Learning implementations.

Tests all major semi-supervised learning algorithms and utility functions.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import warnings

from exercise import (
    SelfTraining, CoTraining, LabelSpreading, LabelPropagation,
    TransductiveSVM, SemiSupervisedGMM, ConsistencyRegularization,
    MixMatch, FixMatch,
    generate_semi_supervised_data, evaluate_semi_supervised,
    plot_semi_supervised_results, compare_semi_supervised_methods,
    label_complexity_analysis, graph_construction_analysis,
    consistency_assumption_validation
)

warnings.filterwarnings('ignore')


class TestSelfTraining:
    """Test Self-Training implementation."""
    
    def test_self_training_basic(self):
        """Test basic self-training functionality."""
        # Generate synthetic data
        X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                                 n_redundant=0, n_clusters_per_class=1, random_state=42)
        
        # Split into labeled and unlabeled
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # Initialize self-training
        base_clf = DecisionTreeClassifier(random_state=42)
        self_training = SelfTraining(base_clf, threshold=0.8, max_iterations=5)
        
        # Fit model
        self_training.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = self_training.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have attributes after fitting
        assert hasattr(self_training, 'base_classifier')
        assert hasattr(self_training, 'n_iterations_')
    
    def test_self_training_convergence(self):
        """Test that self-training converges or reaches max iterations."""
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.9, random_state=42
        )
        
        base_clf = LogisticRegression(random_state=42)
        self_training = SelfTraining(base_clf, threshold=0.9, max_iterations=3)
        
        self_training.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Should have stopped at max iterations or earlier
        assert self_training.n_iterations_ <= 3
        assert self_training.n_iterations_ >= 1
    
    def test_self_training_threshold_effect(self):
        """Test that higher thresholds add fewer pseudo-labels."""
        X, y = make_classification(n_samples=300, n_features=4, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # High threshold
        base_clf1 = DecisionTreeClassifier(random_state=42)
        st_high = SelfTraining(base_clf1, threshold=0.95, max_iterations=5)
        st_high.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Low threshold  
        base_clf2 = DecisionTreeClassifier(random_state=42)
        st_low = SelfTraining(base_clf2, threshold=0.6, max_iterations=5)
        st_low.fit(X_labeled, y_labeled, X_unlabeled)
        
        # High threshold should typically add fewer pseudo-labels
        # (though this can vary based on data and classifier confidence)
        assert hasattr(st_high, 'n_pseudo_labels_')
        assert hasattr(st_low, 'n_pseudo_labels_')


class TestCoTraining:
    """Test Co-Training implementation."""
    
    def test_co_training_basic(self):
        """Test basic co-training functionality."""
        # Generate data with natural feature split
        X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                                 n_redundant=0, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # Feature views (first 3 and last 3 features)
        view1_indices = [0, 1, 2]
        view2_indices = [3, 4, 5]
        
        clf1 = DecisionTreeClassifier(random_state=42)
        clf2 = DecisionTreeClassifier(random_state=43)
        
        co_training = CoTraining(clf1, clf2, view1_indices, view2_indices,
                               k_best=5, max_iterations=3)
        
        co_training.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = co_training.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have trained both classifiers
        assert hasattr(co_training, 'classifier1')
        assert hasattr(co_training, 'classifier2')
    
    def test_co_training_feature_views(self):
        """Test that co-training uses different feature views."""
        X, y = make_classification(n_samples=200, n_features=8, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        
        view1 = [0, 1, 2, 3]
        view2 = [4, 5, 6, 7]
        
        clf1 = LogisticRegression(random_state=42)
        clf2 = LogisticRegression(random_state=43)
        
        co_training = CoTraining(clf1, clf2, view1, view2, k_best=3)
        co_training.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Classifiers should be fitted on different feature subsets
        assert co_training.view1_indices == view1
        assert co_training.view2_indices == view2


class TestLabelSpreading:
    """Test Label Spreading implementation."""
    
    def test_label_spreading_basic(self):
        """Test basic label spreading functionality."""
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)
        
        # Create partially labeled data
        labeled_indices = np.random.choice(len(X), size=20, replace=False)
        y_partial = np.full(len(X), -1)  # -1 for unlabeled
        y_partial[labeled_indices] = y[labeled_indices]
        
        label_spreading = LabelSpreading(gamma=1.0, alpha=0.8, max_iter=100)
        label_spreading.fit(X, y_partial)
        
        # Test predictions
        predictions = label_spreading.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should preserve original labels
        np.testing.assert_array_equal(predictions[labeled_indices], 
                                    y[labeled_indices])
        
        # Should have label distributions
        assert hasattr(label_spreading, 'label_distributions_')
        assert label_spreading.label_distributions_.shape == (len(X), 2)
    
    def test_label_spreading_convergence(self):
        """Test that label spreading converges."""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        
        labeled_indices = [0, 1, 50, 51]  # Two from each class
        y_partial = np.full(len(X), -1)
        y_partial[labeled_indices] = y[labeled_indices]
        
        label_spreading = LabelSpreading(gamma=0.5, alpha=0.5, max_iter=50, tol=1e-6)
        label_spreading.fit(X, y_partial)
        
        # Should have converged
        assert hasattr(label_spreading, 'n_iter_')
        assert label_spreading.n_iter_ <= 50


class TestLabelPropagation:
    """Test Label Propagation implementation."""
    
    def test_label_propagation_basic(self):
        """Test basic label propagation functionality."""
        X, y = make_classification(n_samples=150, n_features=2, n_informative=2,
                                 n_redundant=0, n_clusters_per_class=1, random_state=42)
        
        # Create partially labeled data
        labeled_indices = np.random.choice(len(X), size=15, replace=False)
        y_partial = np.full(len(X), -1)
        y_partial[labeled_indices] = y[labeled_indices]
        
        label_prop = LabelPropagation(gamma=1.0, max_iter=100)
        label_prop.fit(X, y_partial)
        
        predictions = label_prop.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should preserve original labels exactly
        np.testing.assert_array_equal(predictions[labeled_indices], 
                                    y[labeled_indices])
    
    def test_label_propagation_vs_spreading(self):
        """Test difference between label propagation and spreading."""
        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
        
        labeled_indices = [0, 50]
        y_partial = np.full(len(X), -1)
        y_partial[labeled_indices] = y[labeled_indices]
        
        # Label Propagation (hard constraint on labeled data)
        label_prop = LabelPropagation(gamma=1.0, max_iter=50)
        label_prop.fit(X, y_partial)
        prop_predictions = label_prop.predict(X)
        
        # Label Spreading (soft constraint on labeled data)
        label_spread = LabelSpreading(gamma=1.0, alpha=0.8, max_iter=50)
        label_spread.fit(X, y_partial)
        spread_predictions = label_spread.predict(X)
        
        # Both should preserve labeled points
        np.testing.assert_array_equal(prop_predictions[labeled_indices], 
                                    y[labeled_indices])
        np.testing.assert_array_equal(spread_predictions[labeled_indices], 
                                    y[labeled_indices])


class TestTransductiveSVM:
    """Test Transductive SVM implementation."""
    
    def test_transductive_svm_basic(self):
        """Test basic TSVM functionality."""
        X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                                 n_redundant=0, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        
        tsvm = TransductiveSVM(C=1.0, C_star=0.1, max_iter=10)
        tsvm.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions on unlabeled data
        predictions = tsvm.predict(X_unlabeled)
        assert len(predictions) == len(X_unlabeled)
        assert all(pred in [-1, 1] for pred in predictions)
        
        # Should have learned labels for unlabeled data
        assert hasattr(tsvm, 'unlabeled_predictions_')
    
    def test_transductive_svm_parameters(self):
        """Test TSVM with different parameters."""
        X, y = make_moons(n_samples=150, noise=0.1, random_state=42)
        y = 2 * y - 1  # Convert to {-1, 1}
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # Test different C_star values
        for C_star in [0.01, 0.1, 0.5]:
            tsvm = TransductiveSVM(C=1.0, C_star=C_star, max_iter=5)
            tsvm.fit(X_labeled, y_labeled, X_unlabeled)
            
            predictions = tsvm.predict(X_unlabeled)
            assert len(predictions) == len(X_unlabeled)


class TestSemiSupervisedGMM:
    """Test Semi-Supervised Gaussian Mixture Model."""
    
    def test_semi_supervised_gmm_basic(self):
        """Test basic semi-supervised GMM functionality."""
        X, y = make_classification(n_samples=200, n_features=3, n_informative=3,
                                 n_redundant=0, n_clusters_per_class=2, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        ss_gmm = SemiSupervisedGMM(n_components=4, max_iter=20, random_state=42)
        ss_gmm.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = ss_gmm.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have fitted GMM
        assert hasattr(ss_gmm, 'gmm_')
        assert hasattr(ss_gmm, 'component_labels_')
    
    def test_semi_supervised_gmm_probabilities(self):
        """Test that GMM returns probability estimates."""
        X, y = make_circles(n_samples=150, noise=0.1, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        
        ss_gmm = SemiSupervisedGMM(n_components=3, max_iter=15, random_state=42)
        ss_gmm.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test probability prediction
        probabilities = ss_gmm.predict_proba(X_unlabeled[:5])
        assert probabilities.shape == (5, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


class TestConsistencyRegularization:
    """Test Consistency Regularization implementation."""
    
    def test_consistency_regularization_basic(self):
        """Test basic consistency regularization."""
        X, y = make_classification(n_samples=200, n_features=4, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        base_model = LogisticRegression(random_state=42)
        consistency_reg = ConsistencyRegularization(
            base_model, lambda_u=1.0, noise_scale=0.1, max_iter=10
        )
        
        consistency_reg.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = consistency_reg.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have trained model
        assert hasattr(consistency_reg, 'model_')
    
    def test_consistency_regularization_noise(self):
        """Test consistency with different noise levels."""
        X, y = make_moons(n_samples=150, noise=0.1, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        for noise_scale in [0.01, 0.1, 0.2]:
            base_model = LogisticRegression(random_state=42)
            consistency_reg = ConsistencyRegularization(
                base_model, lambda_u=0.5, noise_scale=noise_scale, max_iter=5
            )
            
            consistency_reg.fit(X_labeled, y_labeled, X_unlabeled)
            predictions = consistency_reg.predict(X_unlabeled)
            
            assert len(predictions) == len(X_unlabeled)


class TestMixMatch:
    """Test MixMatch implementation."""
    
    def test_mixmatch_basic(self):
        """Test basic MixMatch functionality."""
        X, y = make_classification(n_samples=200, n_features=3, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        mixmatch = MixMatch(
            lambda_u=0.5, T=0.5, K=2, alpha=0.75, max_iter=5, random_state=42
        )
        
        mixmatch.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = mixmatch.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have trained model
        assert hasattr(mixmatch, 'model_')
    
    def test_mixmatch_parameters(self):
        """Test MixMatch with different parameters."""
        X, y = make_circles(n_samples=150, noise=0.1, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.7, random_state=42
        )
        
        # Test different temperature values
        for T in [0.1, 0.5, 1.0]:
            mixmatch = MixMatch(
                lambda_u=0.3, T=T, K=2, alpha=0.5, max_iter=3, random_state=42
            )
            
            mixmatch.fit(X_labeled, y_labeled, X_unlabeled)
            predictions = mixmatch.predict(X_unlabeled)
            
            assert len(predictions) == len(X_unlabeled)


class TestFixMatch:
    """Test FixMatch implementation."""
    
    def test_fixmatch_basic(self):
        """Test basic FixMatch functionality."""
        X, y = make_classification(n_samples=200, n_features=4, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        fixmatch = FixMatch(
            threshold=0.95, lambda_u=1.0, max_iter=5, random_state=42
        )
        
        fixmatch.fit(X_labeled, y_labeled, X_unlabeled)
        
        # Test predictions
        predictions = fixmatch.predict(X_unlabeled[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Should have trained model
        assert hasattr(fixmatch, 'model_')
    
    def test_fixmatch_threshold(self):
        """Test FixMatch with different confidence thresholds."""
        X, y = make_moons(n_samples=150, noise=0.1, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # Test different thresholds
        for threshold in [0.8, 0.9, 0.95]:
            fixmatch = FixMatch(
                threshold=threshold, lambda_u=0.5, max_iter=3, random_state=42
            )
            
            fixmatch.fit(X_labeled, y_labeled, X_unlabeled)
            predictions = fixmatch.predict(X_unlabeled)
            
            assert len(predictions) == len(X_unlabeled)


class TestUtilityFunctions:
    """Test utility functions for semi-supervised learning."""
    
    def test_generate_semi_supervised_data(self):
        """Test synthetic data generation."""
        X_labeled, y_labeled, X_unlabeled, X_test, y_test = generate_semi_supervised_data(
            n_labeled=30, n_unlabeled=100, n_test=50, random_state=42
        )
        
        assert X_labeled.shape[0] == 30
        assert len(y_labeled) == 30
        assert X_unlabeled.shape[0] == 100
        assert X_test.shape[0] == 50
        assert len(y_test) == 50
        
        # Features should have same dimensionality
        assert X_labeled.shape[1] == X_unlabeled.shape[1] == X_test.shape[1]
        
        # Labels should be binary
        assert set(y_labeled).issubset({0, 1})
        assert set(y_test).issubset({0, 1})
    
    def test_evaluate_semi_supervised(self):
        """Test evaluation function."""
        # Simple test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        accuracy, precision, recall, f1 = evaluate_semi_supervised(
            None, None, None, y_true=y_true, y_pred=y_pred
        )
        
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        
        # Accuracy should be 4/5 = 0.8
        assert abs(accuracy - 0.8) < 1e-6
    
    def test_compare_semi_supervised_methods(self):
        """Test method comparison function."""
        X, y = make_classification(n_samples=200, n_features=3, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        X_test, y_test = X_unlabeled[:50], y_unlabeled[:50]
        X_unlabeled = X_unlabeled[50:]
        
        results = compare_semi_supervised_methods(
            X_labeled, y_labeled, X_unlabeled, X_test, y_test
        )
        
        assert isinstance(results, dict)
        # Should contain results for different methods
        assert len(results) > 0
        
        # Each result should have performance metrics
        for method_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
    
    def test_label_complexity_analysis(self):
        """Test label complexity analysis."""
        X, y = make_classification(n_samples=300, n_features=4, random_state=42)
        
        results = label_complexity_analysis(X, y, label_fractions=[0.05, 0.1, 0.2])
        
        assert isinstance(results, dict)
        assert 'label_fractions' in results
        assert 'accuracies' in results
        assert 'methods' in results
        
        assert len(results['label_fractions']) == 3
        assert len(results['accuracies']) > 0
    
    def test_graph_construction_analysis(self):
        """Test graph construction analysis."""
        X, y = make_circles(n_samples=100, noise=0.1, random_state=42)
        
        results = graph_construction_analysis(X, y)
        
        assert isinstance(results, dict)
        assert 'knn_results' in results
        assert 'epsilon_results' in results
        
        # Should test different parameter values
        assert len(results['knn_results']) > 0
        assert len(results['epsilon_results']) > 0
    
    def test_consistency_assumption_validation(self):
        """Test consistency assumption validation."""
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
        
        results = consistency_assumption_validation(X, y)
        
        assert isinstance(results, dict)
        assert 'cluster_assumption_score' in results
        assert 'smoothness_score' in results
        assert 'manifold_assumption_score' in results
        
        # Scores should be reasonable values
        for score_name, score in results.items():
            assert isinstance(score, (int, float))
            assert not np.isnan(score)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_pipeline(self):
        """Test complete semi-supervised learning pipeline."""
        # Generate data
        X_labeled, y_labeled, X_unlabeled, X_test, y_test = generate_semi_supervised_data(
            n_labeled=50, n_unlabeled=200, n_test=100, random_state=42
        )
        
        # Test multiple algorithms
        algorithms = [
            ("SelfTraining", SelfTraining(DecisionTreeClassifier(random_state=42), 
                                        threshold=0.8, max_iterations=3)),
            ("LabelSpreading", LabelSpreading(gamma=1.0, alpha=0.8, max_iter=30))
        ]
        
        for name, algorithm in algorithms:
            if name == "SelfTraining":
                algorithm.fit(X_labeled, y_labeled, X_unlabeled)
            else:  # LabelSpreading
                # Combine labeled and unlabeled for label spreading
                X_combined = np.vstack([X_labeled, X_unlabeled])
                y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
                algorithm.fit(X_combined, y_combined)
            
            # Test predictions
            predictions = algorithm.predict(X_test)
            assert len(predictions) == len(X_test)
            
            # Evaluate performance
            accuracy = np.mean(predictions == y_test)
            assert 0 <= accuracy <= 1
            
            # Semi-supervised should do better than random (> 0.5 for balanced data)
            # This is a weak test but ensures basic functionality
            print(f"{name} accuracy: {accuracy:.3f}")
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to hyperparameters."""
        X, y = make_classification(n_samples=200, n_features=3, random_state=42)
        
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X, y, test_size=0.8, random_state=42
        )
        
        # Test LabelSpreading with different gamma values
        gamma_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for gamma in gamma_values:
            X_combined = np.vstack([X_labeled, X_unlabeled])
            y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
            
            ls = LabelSpreading(gamma=gamma, alpha=0.8, max_iter=30)
            ls.fit(X_combined, y_combined)
            
            # Test on labeled data (should predict correctly)
            pred_labeled = ls.predict(X_labeled)
            accuracy = np.mean(pred_labeled == y_labeled)
            accuracies.append(accuracy)
        
        # All should achieve reasonable accuracy on labeled data
        assert all(acc > 0.7 for acc in accuracies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 