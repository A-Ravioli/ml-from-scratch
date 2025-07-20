"""
Test suite for Ensemble Methods implementations.
"""

import numpy as np
import pytest
from exercise import (
    BaggingEnsemble, RandomForestAdvanced, AdaBoostAdvanced,
    GradientBoostingAdvanced, VotingEnsemble, StackingEnsemble,
    MultiLevelStacking, DynamicEnsembleSelection, EnsemblePruning,
    create_diverse_base_learners, calculate_ensemble_diversity,
    bias_variance_decomposition, ensemble_learning_curves,
    optimal_ensemble_size_analysis, ensemble_interpretability_analysis
)


class DummyClassifier:
    """Simple dummy classifier for testing."""
    
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.classes_ = None
        self.class_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if self.strategy == 'most_frequent':
            from collections import Counter
            self.class_ = Counter(y).most_common(1)[0][0]
        return self
    
    def predict(self, X):
        return np.full(len(X), self.class_)
    
    def predict_proba(self, X):
        n_classes = len(self.classes_)
        proba = np.zeros((len(X), n_classes))
        class_idx = np.where(self.classes_ == self.class_)[0][0]
        proba[:, class_idx] = 1.0
        return proba


class DummyRegressor:
    """Simple dummy regressor for testing."""
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.constant_ = None
    
    def fit(self, X, y):
        if self.strategy == 'mean':
            self.constant_ = np.mean(y)
        elif self.strategy == 'median':
            self.constant_ = np.median(y)
        return self
    
    def predict(self, X):
        return np.full(len(X), self.constant_)


class TestBaggingEnsemble:
    """Test Bagging Ensemble implementation."""
    
    def test_bagging_basic_classification(self):
        """Test basic bagging functionality for classification."""
        # Generate simple classification data
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        bagging = BaggingEnsemble(
            base_estimator=DummyClassifier(),
            n_estimators=10,
            bootstrap=True
        )
        bagging.fit(X, y)
        
        # Should make predictions
        y_pred = bagging.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1})
        
        # Should have fitted estimators
        assert hasattr(bagging, 'estimators_')
        assert len(bagging.estimators_) == 10
    
    def test_bagging_regression(self):
        """Test bagging for regression."""
        X = np.random.randn(50, 3)
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(50)
        
        bagging = BaggingEnsemble(
            base_estimator=DummyRegressor(),
            n_estimators=5,
            bootstrap=True
        )
        bagging.fit(X, y)
        
        y_pred = bagging.predict(X)
        assert len(y_pred) == len(y)
        assert isinstance(y_pred[0], (int, float, np.number))
    
    def test_bagging_bootstrap_sampling(self):
        """Test bootstrap sampling functionality."""
        X = np.random.randn(30, 2)
        y = np.random.randint(0, 2, 30)
        
        bagging = BaggingEnsemble(
            base_estimator=DummyClassifier(),
            n_estimators=3,
            max_samples=0.8,
            bootstrap=True
        )
        
        # Test bootstrap sample creation
        X_sample, y_sample, oob_indices = bagging._bootstrap_sample(X, y)
        
        # Should respect max_samples
        expected_size = int(0.8 * len(X))
        assert len(X_sample) == expected_size
        assert len(y_sample) == expected_size
        
        # OOB indices should be valid
        assert len(oob_indices) <= len(X)
        assert all(0 <= idx < len(X) for idx in oob_indices)
    
    def test_bagging_oob_score(self):
        """Test out-of-bag score computation."""
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 3, 60)
        
        bagging = BaggingEnsemble(
            base_estimator=DummyClassifier(),
            n_estimators=8,
            oob_score=True,
            bootstrap=True
        )
        bagging.fit(X, y)
        
        # Should compute OOB score
        assert hasattr(bagging, 'oob_score_')
        assert isinstance(bagging.oob_score_, float)
        assert 0 <= bagging.oob_score_ <= 1
    
    def test_bagging_predict_proba(self):
        """Test probability prediction."""
        X = np.random.randn(40, 2)
        y = np.random.randint(0, 2, 40)
        
        bagging = BaggingEnsemble(
            base_estimator=DummyClassifier(),
            n_estimators=5
        )
        bagging.fit(X, y)
        
        y_proba = bagging.predict_proba(X)
        assert y_proba.shape == (len(X), 2)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
        assert np.all((y_proba >= 0) & (y_proba <= 1))


class TestRandomForestAdvanced:
    """Test Advanced Random Forest implementation."""
    
    def test_random_forest_basic(self):
        """Test basic Random Forest functionality."""
        X = np.random.randn(80, 5)
        y = np.random.randint(0, 3, 80)
        
        rf = RandomForestAdvanced(
            n_estimators=10,
            max_depth=5,
            max_features='sqrt'
        )
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1, 2})
    
    def test_random_forest_feature_importance(self):
        """Test feature importance calculation."""
        X = np.random.randn(60, 4)
        y = X[:, 0] + 2 * X[:, 1] + np.random.randn(60) * 0.1  # Features 0,1 important
        y = (y > np.median(y)).astype(int)
        
        rf = RandomForestAdvanced(
            n_estimators=10,
            feature_importances_method='impurity'
        )
        rf.fit(X, y)
        
        # Test impurity-based importance
        importances = rf.feature_importances()
        assert len(importances) == X.shape[1]
        assert np.allclose(np.sum(importances), 1.0)
        assert np.all(importances >= 0)
        
        # Features 0 and 1 should be more important
        assert importances[0] > importances[2]
        assert importances[1] > importances[3]
    
    def test_random_forest_permutation_importance(self):
        """Test permutation-based feature importance."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        rf = RandomForestAdvanced(
            n_estimators=5,
            feature_importances_method='permutation'
        )
        rf.fit(X, y)
        
        importances = rf.feature_importances(X, y)
        assert len(importances) == X.shape[1]
        assert all(isinstance(imp, (int, float, np.number)) for imp in importances)
    
    def test_random_forest_class_weights(self):
        """Test class weight handling."""
        # Create imbalanced dataset
        X = np.random.randn(100, 3)
        y = np.random.choice([0, 1], 100, p=[0.9, 0.1])  # Imbalanced
        
        rf = RandomForestAdvanced(
            n_estimators=5,
            class_weight='balanced'
        )
        rf.fit(X, y)
        
        y_pred = rf.predict(X)
        assert len(y_pred) == len(y)


class TestAdaBoostAdvanced:
    """Test Advanced AdaBoost implementation."""
    
    def test_adaboost_samme_basic(self):
        """Test SAMME algorithm."""
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 2, 60)
        
        ada = AdaBoostAdvanced(
            base_estimator=DummyClassifier(),
            n_estimators=5,
            algorithm='SAMME'
        )
        ada.fit(X, y)
        
        y_pred = ada.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1})
        
        # Should have estimator weights
        assert hasattr(ada, 'estimator_weights_')
        assert len(ada.estimator_weights_) <= 5  # May stop early
    
    def test_adaboost_samme_r(self):
        """Test SAMME.R algorithm."""
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        
        ada = AdaBoostAdvanced(
            base_estimator=DummyClassifier(),
            n_estimators=3,
            algorithm='SAMME.R'
        )
        ada.fit(X, y)
        
        y_pred = ada.predict(X)
        assert len(y_pred) == len(y)
    
    def test_adaboost_multiclass(self):
        """Test AdaBoost with multi-class data."""
        X = np.random.randn(70, 4)
        y = np.random.randint(0, 3, 70)
        
        ada = AdaBoostAdvanced(
            base_estimator=DummyClassifier(),
            n_estimators=5,
            algorithm='SAMME'
        )
        ada.fit(X, y)
        
        y_pred = ada.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1, 2})
    
    def test_adaboost_staged_predict(self):
        """Test staged predictions."""
        X = np.random.randn(40, 2)
        y = np.random.randint(0, 2, 40)
        
        ada = AdaBoostAdvanced(
            base_estimator=DummyClassifier(),
            n_estimators=4
        )
        ada.fit(X, y)
        
        staged_preds = ada.staged_predict(X)
        assert len(staged_preds) <= 4  # May stop early
        
        for pred in staged_preds:
            assert len(pred) == len(y)


class TestGradientBoostingAdvanced:
    """Test Advanced Gradient Boosting implementation."""
    
    def test_gradient_boosting_regression(self):
        """Test GBM for regression."""
        X = np.random.randn(60, 3)
        y = np.sum(X, axis=1) + 0.2 * np.random.randn(60)
        
        gbm = GradientBoostingAdvanced(
            n_estimators=10,
            learning_rate=0.1,
            loss='mse'
        )
        gbm.fit(X, y)
        
        y_pred = gbm.predict(X)
        assert len(y_pred) == len(y)
        
        # Should improve over iterations
        assert hasattr(gbm, 'train_scores_')
        assert len(gbm.train_scores_) <= 10
    
    def test_gradient_boosting_loss_functions(self):
        """Test different loss functions."""
        X = np.random.randn(40, 2)
        y = np.random.randn(40)
        
        loss_functions = ['mse', 'mae', 'huber', 'quantile']
        
        for loss in loss_functions:
            gbm = GradientBoostingAdvanced(
                n_estimators=3,
                loss=loss,
                alpha=0.7 if loss in ['huber', 'quantile'] else 0.9
            )
            gbm.fit(X, y)
            
            y_pred = gbm.predict(X)
            assert len(y_pred) == len(y)
    
    def test_gradient_boosting_huber_loss(self):
        """Test Huber loss implementation."""
        gbm = GradientBoostingAdvanced(loss='huber')
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.5, 3.5])
        
        loss, gradient = gbm._huber_loss(y_true, y_pred, alpha=0.5)
        
        assert isinstance(loss, float)
        assert len(gradient) == len(y_true)
    
    def test_gradient_boosting_quantile_loss(self):
        """Test quantile loss implementation."""
        gbm = GradientBoostingAdvanced(loss='quantile')
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.2, 1.8, 3.2])
        
        loss, gradient = gbm._quantile_loss(y_true, y_pred, alpha=0.5)
        
        assert isinstance(loss, float)
        assert len(gradient) == len(y_true)
    
    def test_gradient_boosting_regularization(self):
        """Test regularization parameters."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        gbm = GradientBoostingAdvanced(
            n_estimators=5,
            l1_regularization=0.1,
            l2_regularization=0.1,
            subsample=0.8
        )
        gbm.fit(X, y)
        
        y_pred = gbm.predict(X)
        assert len(y_pred) == len(y)
    
    def test_gradient_boosting_early_stopping(self):
        """Test early stopping functionality."""
        X = np.random.randn(80, 3)
        y = np.random.randn(80)
        
        gbm = GradientBoostingAdvanced(
            n_estimators=20,
            early_stopping_rounds=3,
            validation_fraction=0.2
        )
        gbm.fit(X, y)
        
        # Should stop before max iterations (potentially)
        assert hasattr(gbm, 'estimators_')


class TestVotingEnsemble:
    """Test Voting Ensemble implementation."""
    
    def test_voting_hard(self):
        """Test hard voting."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier()),
            ('dummy3', DummyClassifier())
        ]
        
        voting = VotingEnsemble(estimators, voting='hard')
        voting.fit(X, y)
        
        y_pred = voting.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1})
    
    def test_voting_soft(self):
        """Test soft voting."""
        X = np.random.randn(40, 2)
        y = np.random.randint(0, 2, 40)
        
        estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        
        voting = VotingEnsemble(estimators, voting='soft')
        voting.fit(X, y)
        
        y_pred = voting.predict(X)
        y_proba = voting.predict_proba(X)
        
        assert len(y_pred) == len(y)
        assert y_proba.shape == (len(X), 2)
        assert np.allclose(np.sum(y_proba, axis=1), 1.0)
    
    def test_voting_weights(self):
        """Test weighted voting."""
        X = np.random.randn(30, 2)
        y = np.random.randint(0, 2, 30)
        
        estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        weights = [0.7, 0.3]
        
        voting = VotingEnsemble(estimators, voting='soft', weights=weights)
        voting.fit(X, y)
        
        y_proba = voting.predict_proba(X)
        assert y_proba.shape == (len(X), 2)


class TestStackingEnsemble:
    """Test Stacking Ensemble implementation."""
    
    def test_stacking_basic(self):
        """Test basic stacking functionality."""
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 2, 60)
        
        base_estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        meta_estimator = DummyClassifier()
        
        stacking = StackingEnsemble(
            base_estimators=base_estimators,
            meta_estimator=meta_estimator,
            cv_folds=3
        )
        stacking.fit(X, y)
        
        y_pred = stacking.predict(X)
        assert len(y_pred) == len(y)
        assert set(y_pred).issubset({0, 1})
    
    def test_stacking_meta_features(self):
        """Test meta-feature creation."""
        X = np.random.randn(40, 2)
        y = np.random.randint(0, 2, 40)
        
        base_estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        
        stacking = StackingEnsemble(
            base_estimators=base_estimators,
            cv_folds=3
        )
        
        meta_features = stacking._create_meta_features(X, y)
        
        assert meta_features.shape[0] == len(X)
        assert meta_features.shape[1] == len(base_estimators)
    
    def test_stacking_probabilities(self):
        """Test stacking with probabilities."""
        X = np.random.randn(30, 2)
        y = np.random.randint(0, 3, 30)
        
        base_estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        
        stacking = StackingEnsemble(
            base_estimators=base_estimators,
            use_probas=True,
            cv_folds=3
        )
        stacking.fit(X, y)
        
        y_pred = stacking.predict(X)
        assert len(y_pred) == len(y)


class TestMultiLevelStacking:
    """Test Multi-Level Stacking implementation."""
    
    def test_multilevel_stacking_basic(self):
        """Test basic multi-level stacking."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        levels = [
            [('dummy1', DummyClassifier()), ('dummy2', DummyClassifier())],
            [('dummy3', DummyClassifier())]
        ]
        final_estimator = DummyClassifier()
        
        mlstacking = MultiLevelStacking(
            levels=levels,
            final_estimator=final_estimator,
            cv_folds=3
        )
        mlstacking.fit(X, y)
        
        y_pred = mlstacking.predict(X)
        assert len(y_pred) == len(y)


class TestDynamicEnsembleSelection:
    """Test Dynamic Ensemble Selection implementation."""
    
    def test_des_basic(self):
        """Test basic DES functionality."""
        X = np.random.randn(60, 3)
        y = np.random.randint(0, 2, 60)
        
        # Split for validation
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        
        estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier()),
            ('dummy3', DummyClassifier())
        ]
        
        des = DynamicEnsembleSelection(
            estimators=estimators,
            selection_method='local_accuracy',
            k_neighbors=3
        )
        des.fit(X_train, y_train, X_val, y_val)
        
        y_pred = des.predict(X_val)
        assert len(y_pred) == len(y_val)
    
    def test_des_local_accuracy(self):
        """Test local accuracy selection."""
        X_val = np.random.randn(20, 2)
        y_val = np.random.randint(0, 2, 20)
        
        estimators = [
            ('dummy1', DummyClassifier()),
            ('dummy2', DummyClassifier())
        ]
        
        des = DynamicEnsembleSelection(
            estimators=estimators,
            k_neighbors=5
        )
        
        # Mock fit to set up validation data
        des.X_val_ = X_val
        des.y_val_ = y_val
        des.estimators_ = [est[1] for est in estimators]
        
        # Fit estimators on validation data for testing
        for est in des.estimators_:
            est.fit(X_val, y_val)
        
        x_test = X_val[0]
        selected = des._local_accuracy_selection(x_test)
        
        assert isinstance(selected, list)
        assert all(isinstance(idx, int) for idx in selected)
    
    def test_des_diversity_measure(self):
        """Test diversity measure calculation."""
        predictions = np.array([
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ])
        
        des = DynamicEnsembleSelection([])
        diversity = des._diversity_measure(predictions)
        
        assert isinstance(diversity, float)
        assert diversity >= 0


class TestEnsemblePruning:
    """Test Ensemble Pruning implementation."""
    
    def test_pruning_ranking(self):
        """Test ranking-based pruning."""
        X_val = np.random.randn(30, 2)
        y_val = np.random.randint(0, 2, 30)
        
        estimators = [DummyClassifier() for _ in range(5)]
        for est in estimators:
            est.fit(X_val, y_val)
        
        pruner = EnsemblePruning(
            pruning_method='ranking',
            target_size=3
        )
        
        selected_indices = pruner._ranking_pruning(estimators, X_val, y_val)
        
        assert isinstance(selected_indices, list)
        assert len(selected_indices) <= 3
        assert all(isinstance(idx, int) for idx in selected_indices)
    
    def test_pruning_clustering(self):
        """Test clustering-based pruning."""
        X_val = np.random.randn(25, 2)
        y_val = np.random.randint(0, 2, 25)
        
        estimators = [DummyClassifier() for _ in range(4)]
        for est in estimators:
            est.fit(X_val, y_val)
        
        pruner = EnsemblePruning(pruning_method='clustering')
        selected_indices = pruner._clustering_pruning(estimators, X_val, y_val)
        
        assert isinstance(selected_indices, list)
        assert len(selected_indices) <= len(estimators)


class TestEnsembleUtilities:
    """Test utility functions."""
    
    def test_create_diverse_base_learners(self):
        """Test diverse base learner creation."""
        base_learners = create_diverse_base_learners()
        
        if base_learners:
            assert isinstance(base_learners, list)
            assert all(isinstance(item, tuple) and len(item) == 2 for item in base_learners)
            assert all(isinstance(item[0], str) for item in base_learners)
    
    def test_calculate_ensemble_diversity(self):
        """Test diversity calculation."""
        # Create diverse predictions
        predictions = np.array([
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1]
        ])
        
        diversity_metrics = calculate_ensemble_diversity(predictions)
        
        if diversity_metrics:
            assert isinstance(diversity_metrics, dict)
            assert all(isinstance(v, (int, float, np.number)) for v in diversity_metrics.values())
    
    def test_bias_variance_decomposition(self):
        """Test bias-variance decomposition."""
        X = np.random.randn(40, 2)
        y = np.random.randint(0, 2, 40)
        
        # Mock ensemble for testing
        ensemble = BaggingEnsemble(base_estimator=DummyClassifier(), n_estimators=3)
        
        bv_results = bias_variance_decomposition(ensemble, X, y, n_bootstrap=5)
        
        if bv_results:
            assert isinstance(bv_results, dict)
            expected_keys = ['bias_squared', 'variance', 'noise']
            available_keys = set(bv_results.keys())
            assert len(available_keys.intersection(expected_keys)) > 0
    
    def test_ensemble_learning_curves(self):
        """Test learning curve generation."""
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        
        ensembles = [
            BaggingEnsemble(base_estimator=DummyClassifier(), n_estimators=3),
            VotingEnsemble([('dummy', DummyClassifier())], voting='hard')
        ]
        
        learning_curves = ensemble_learning_curves(ensembles, X, y)
        
        if learning_curves:
            assert isinstance(learning_curves, dict)
            assert all(isinstance(v, list) for v in learning_curves.values())
    
    def test_optimal_ensemble_size_analysis(self):
        """Test optimal size analysis."""
        X = np.random.randn(60, 2)
        y = np.random.randint(0, 2, 60)
        
        optimal_size, performance_curve = optimal_ensemble_size_analysis(
            BaggingEnsemble, X, y, max_estimators=10
        )
        
        if optimal_size is not None:
            assert isinstance(optimal_size, int)
            assert optimal_size > 0
            assert isinstance(performance_curve, list)
    
    def test_ensemble_interpretability_analysis(self):
        """Test interpretability analysis."""
        X = np.random.randn(40, 3)
        y = np.random.randint(0, 2, 40)
        
        ensemble = BaggingEnsemble(base_estimator=DummyClassifier(), n_estimators=3)
        ensemble.fit(X, y)
        
        feature_names = ['feature1', 'feature2', 'feature3']
        
        interpretability = ensemble_interpretability_analysis(ensemble, X, feature_names)
        
        if interpretability:
            assert isinstance(interpretability, dict)


def test_ensemble_interface_consistency():
    """Test that ensemble methods have consistent interfaces."""
    X = np.random.randn(30, 3)
    y = np.random.randint(0, 2, 30)
    
    ensembles = [
        BaggingEnsemble(base_estimator=DummyClassifier(), n_estimators=3),
        RandomForestAdvanced(n_estimators=3),
        AdaBoostAdvanced(base_estimator=DummyClassifier(), n_estimators=3),
        VotingEnsemble([('dummy', DummyClassifier())], voting='hard')
    ]
    
    for ensemble in ensembles:
        # Should have fit and predict methods
        assert hasattr(ensemble, 'fit')
        assert hasattr(ensemble, 'predict')
        
        # Should be able to fit and predict
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(y)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])