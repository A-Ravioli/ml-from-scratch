"""
Solution implementations for Ensemble Methods exercises.

This file provides complete implementations of all TODO items in exercise.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
from collections import Counter
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


class BaseEnsemble(ABC):
    """Base class for ensemble methods."""
    
    def __init__(self, base_estimator=None, n_estimators: int = 10, random_state: int = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the ensemble."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass


class BaggingEnsemble(BaseEnsemble):
    """
    Bootstrap Aggregating (Bagging) implementation.
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 10, 
                 max_samples: float = 1.0, max_features: float = 1.0,
                 bootstrap: bool = True, bootstrap_features: bool = False,
                 oob_score: bool = False, random_state: int = None):
        """Initialize Bagging ensemble."""
        super().__init__(base_estimator, n_estimators, random_state)
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        
        # Fitted attributes
        self.estimators_ = []
        self.estimators_features_ = []
        self.oob_score_ = None
        self.oob_indices_ = []
        
        # Default base estimator
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create bootstrap sample of data."""
        n_samples, n_features = X.shape
        
        # Sample size
        n_sample_size = int(self.max_samples * n_samples)
        n_feature_size = int(self.max_features * n_features)
        
        # Bootstrap samples
        if self.bootstrap:
            sample_indices = np.random.choice(n_samples, n_sample_size, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), sample_indices)
        else:
            sample_indices = np.random.choice(n_samples, n_sample_size, replace=False)
            oob_indices = np.setdiff1d(np.arange(n_samples), sample_indices)
        
        # Bootstrap features
        if self.bootstrap_features:
            feature_indices = np.random.choice(n_features, n_feature_size, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        X_sample = X[np.ix_(sample_indices, feature_indices)]
        y_sample = y[sample_indices]
        
        return X_sample, y_sample, oob_indices, feature_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaggingEnsemble':
        """Fit bagging ensemble."""
        # Determine if classification or regression
        self.is_classifier_ = len(np.unique(y)) <= 20 and np.all(y == y.astype(int))
        
        # Initialize OOB arrays
        if self.oob_score:
            oob_predictions = np.zeros((len(X), self.n_estimators))
            oob_mask = np.zeros((len(X), self.n_estimators), dtype=bool)
        
        # Train estimators
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample, oob_indices, feature_indices = self._bootstrap_sample(X, y)
            
            # Clone base estimator
            estimator = self._clone_estimator(self.base_estimator)
            
            # Fit estimator
            estimator.fit(X_sample, y_sample)
            
            # Store estimator and features
            self.estimators_.append(estimator)
            self.estimators_features_.append(feature_indices)
            self.oob_indices_.append(oob_indices)
            
            # Store OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                X_oob = X[np.ix_(oob_indices, feature_indices)]
                oob_pred = estimator.predict(X_oob)
                oob_predictions[oob_indices, i] = oob_pred
                oob_mask[oob_indices, i] = True
        
        # Calculate OOB score
        if self.oob_score:
            self._calculate_oob_score(y, oob_predictions, oob_mask)
        
        return self
    
    def _clone_estimator(self, estimator):
        """Clone estimator."""
        from sklearn.base import clone
        try:
            return clone(estimator)
        except:
            # Fallback for custom estimators
            return type(estimator)()
    
    def _calculate_oob_score(self, y: np.ndarray, oob_predictions: np.ndarray, oob_mask: np.ndarray):
        """Calculate out-of-bag score."""
        oob_pred_final = np.zeros(len(y))
        valid_oob = np.sum(oob_mask, axis=1) > 0
        
        if np.sum(valid_oob) == 0:
            self.oob_score_ = 0.0
            return
        
        for i in range(len(y)):
            if valid_oob[i]:
                if self.is_classifier_:
                    # Majority vote
                    votes = oob_predictions[i, oob_mask[i, :]]
                    oob_pred_final[i] = Counter(votes).most_common(1)[0][0]
                else:
                    # Average
                    oob_pred_final[i] = np.mean(oob_predictions[i, oob_mask[i, :]])
        
        # Calculate score
        if self.is_classifier_:
            self.oob_score_ = np.mean(oob_pred_final[valid_oob] == y[valid_oob])
        else:
            self.oob_score_ = -np.mean((oob_pred_final[valid_oob] - y[valid_oob]) ** 2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote (classification) or averaging (regression)."""
        predictions = np.zeros((len(X), self.n_estimators))
        
        for i, (estimator, feature_indices) in enumerate(zip(self.estimators_, self.estimators_features_)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = estimator.predict(X_subset)
        
        if self.is_classifier_:
            # Majority vote
            final_predictions = np.zeros(len(X))
            for i in range(len(X)):
                votes = predictions[i, :]
                final_predictions[i] = Counter(votes).most_common(1)[0][0]
            return final_predictions.astype(int)
        else:
            # Average
            return np.mean(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (for classification)."""
        if not self.is_classifier_:
            raise ValueError("predict_proba only available for classification")
        
        # Get all unique classes
        all_classes = set()
        for estimator in self.estimators_:
            if hasattr(estimator, 'classes_'):
                all_classes.update(estimator.classes_)
        all_classes = sorted(list(all_classes))
        
        avg_probabilities = np.zeros((len(X), len(all_classes)))
        
        for estimator, feature_indices in zip(self.estimators_, self.estimators_features_):
            X_subset = X[:, feature_indices]
            
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_subset)
                estimator_classes = estimator.classes_
                
                # Map to all_classes
                for j, cls in enumerate(estimator_classes):
                    cls_idx = all_classes.index(cls)
                    avg_probabilities[:, cls_idx] += proba[:, j]
            else:
                # Convert predictions to probabilities
                pred = estimator.predict(X_subset)
                for j, cls in enumerate(all_classes):
                    mask = (pred == cls)
                    avg_probabilities[mask, j] += 1
        
        # Average and normalize
        avg_probabilities /= self.n_estimators
        return avg_probabilities


class RandomForestAdvanced(BaseEnsemble):
    """
    Advanced Random Forest with additional features.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True, oob_score: bool = False,
                 feature_importances_method: str = 'impurity',
                 class_weight: Union[str, Dict] = None,
                 random_state: int = None):
        """Initialize Advanced Random Forest."""
        super().__init__(None, n_estimators, random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.feature_importances_method = feature_importances_method
        self.class_weight = class_weight
        
        # Fitted attributes
        self.trees_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.is_classifier_ = None
    
    def _calculate_max_features(self, n_features: int) -> int:
        """Calculate number of features to consider for each split."""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features
    
    def _calculate_feature_importance_impurity(self) -> np.ndarray:
        """Calculate feature importance based on impurity decrease."""
        n_features = len(self.trees_[0].feature_importances_)
        importances = np.zeros(n_features)
        
        for tree in self.trees_:
            importances += tree.feature_importances_
        
        # Average and normalize
        importances /= len(self.trees_)
        importances /= np.sum(importances) if np.sum(importances) > 0 else 1
        
        return importances
    
    def _calculate_feature_importance_permutation(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate feature importance using permutation method."""
        baseline_score = self._score(X, y)
        importances = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            # Permute feature i
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate performance drop
            permuted_score = self._score(X_permuted, y)
            importances[i] = baseline_score - permuted_score
        
        # Normalize
        importances = np.maximum(importances, 0)
        importances /= np.sum(importances) if np.sum(importances) > 0 else 1
        
        return importances
    
    def _score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate score for permutation importance."""
        predictions = self.predict(X)
        if self.is_classifier_:
            return accuracy_score(y, predictions)
        else:
            return -mean_squared_error(y, predictions)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestAdvanced':
        """Fit advanced random forest."""
        # Determine if classification or regression
        unique_y = np.unique(y)
        self.is_classifier_ = len(unique_y) <= 20 and np.all(y == y.astype(int))
        
        n_features = X.shape[1]
        max_features = self._calculate_max_features(n_features)
        
        # Initialize OOB arrays
        if self.oob_score:
            oob_predictions = np.zeros((len(X), self.n_estimators))
            oob_mask = np.zeros((len(X), self.n_estimators), dtype=bool)
        
        # Train trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(len(X), len(X), replace=True)
                oob_indices = np.setdiff1d(np.arange(len(X)), indices)
            else:
                indices = np.arange(len(X))
                oob_indices = np.array([])
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create tree
            if self.is_classifier_:
                tree = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_features,
                    class_weight=self.class_weight,
                    random_state=self.random_state + i if self.random_state else None
                )
            else:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state + i if self.random_state else None
                )
            
            # Fit tree
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
            
            # Store OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices, i] = oob_pred
                oob_mask[oob_indices, i] = True
        
        # Calculate OOB score
        if self.oob_score:
            self._calculate_oob_score(y, oob_predictions, oob_mask)
        
        return self
    
    def _calculate_oob_score(self, y: np.ndarray, oob_predictions: np.ndarray, oob_mask: np.ndarray):
        """Calculate out-of-bag score."""
        oob_pred_final = np.zeros(len(y))
        valid_oob = np.sum(oob_mask, axis=1) > 0
        
        if np.sum(valid_oob) == 0:
            self.oob_score_ = 0.0
            return
        
        for i in range(len(y)):
            if valid_oob[i]:
                if self.is_classifier_:
                    # Majority vote
                    votes = oob_predictions[i, oob_mask[i, :]]
                    oob_pred_final[i] = Counter(votes).most_common(1)[0][0]
                else:
                    # Average
                    oob_pred_final[i] = np.mean(oob_predictions[i, oob_mask[i, :]])
        
        # Calculate score
        if self.is_classifier_:
            self.oob_score_ = np.mean(oob_pred_final[valid_oob] == y[valid_oob])
        else:
            self.oob_score_ = -np.mean((oob_pred_final[valid_oob] - y[valid_oob]) ** 2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        if self.is_classifier_:
            # Majority vote
            final_predictions = np.zeros(len(X))
            for i in range(len(X)):
                votes = predictions[:, i]
                final_predictions[i] = Counter(votes).most_common(1)[0][0]
            return final_predictions.astype(int)
        else:
            # Average
            return np.mean(predictions, axis=0)
    
    def feature_importances(self, X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """Calculate feature importances."""
        if self.feature_importances_method == 'impurity':
            return self._calculate_feature_importance_impurity()
        elif self.feature_importances_method == 'permutation':
            if X is None or y is None:
                raise ValueError("X and y required for permutation importance")
            return self._calculate_feature_importance_permutation(X, y)
        else:
            raise ValueError(f"Unknown method: {self.feature_importances_method}")


class AdaBoostAdvanced(BaseEnsemble):
    """
    Advanced AdaBoost with different algorithms and base learners.
    """
    
    def __init__(self, base_estimator=None, n_estimators: int = 50,
                 learning_rate: float = 1.0, algorithm: str = 'SAMME',
                 random_state: int = None):
        """Initialize Advanced AdaBoost."""
        super().__init__(base_estimator, n_estimators, random_state)
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        
        # Fitted attributes
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.classes_ = None
        
        # Default base estimator
        if base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=1)
    
    def _samme_algorithm(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostAdvanced':
        """SAMME (Stagewise Additive Modeling using Multi-class Exponential loss)."""
        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes")
        
        # Initialize uniform weights
        sample_weights = np.ones(len(X)) / len(X)
        
        for iboost in range(self.n_estimators):
            # Clone base estimator
            from sklearn.base import clone
            estimator = clone(self.base_estimator)
            
            # Fit weighted estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Get predictions
            y_pred = estimator.predict(X)
            
            # Calculate weighted error
            incorrect = y_pred != y
            error = np.average(incorrect, weights=sample_weights)
            
            # Stop if perfect or too poor
            if error >= 1.0 - (1.0 / n_classes):
                if len(self.estimators_) == 0:
                    raise ValueError("First weak learner has error >= 1 - 1/K")
                break
            
            if error <= 0:
                self.estimators_.append(estimator)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(error)
                break
            
            # Calculate alpha
            alpha = self.learning_rate * np.log((1.0 - error) / error) + np.log(n_classes - 1.0)
            
            # Store estimator
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
            # Update weights
            sample_weights *= np.exp(alpha * incorrect)
            sample_weights /= np.sum(sample_weights)
        
        return self
    
    def _samme_r_algorithm(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostAdvanced':
        """SAMME.R (Real AdaBoost) algorithm."""
        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes < 2:
            raise ValueError("Need at least 2 classes")
        
        # Initialize uniform weights
        sample_weights = np.ones(len(X)) / len(X)
        
        for iboost in range(self.n_estimators):
            # Clone base estimator
            from sklearn.base import clone
            estimator = clone(self.base_estimator)
            
            # Fit weighted estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Get class probability estimates
            if hasattr(estimator, 'predict_proba'):
                y_proba = estimator.predict_proba(X)
            else:
                # Convert predictions to probabilities
                y_pred = estimator.predict(X)
                y_proba = np.zeros((len(X), n_classes))
                for i, cls in enumerate(self.classes_):
                    y_proba[y_pred == cls, i] = 1.0
            
            # Clip probabilities
            y_proba = np.clip(y_proba, np.finfo(y_proba.dtype).eps, None)
            y_proba /= np.sum(y_proba, axis=1, keepdims=True)
            
            # Calculate class indicator matrix
            y_indicator = np.zeros((len(X), n_classes))
            for i, cls in enumerate(self.classes_):
                y_indicator[y == cls, i] = 1.0
            
            # Update weights using SAMME.R formula
            # w_i *= exp(-((K-1)/K) * sum_k(y_ik * log(p_ik)))
            log_proba = np.log(y_proba)
            weight_update = np.sum(y_indicator * log_proba, axis=1)
            sample_weights *= np.exp(-((n_classes - 1) / n_classes) * weight_update)
            sample_weights /= np.sum(sample_weights)
            
            # Store estimator (weight is 1 for SAMME.R)
            self.estimators_.append(estimator)
            self.estimator_weights_.append(1.0)
            self.estimator_errors_.append(0.0)  # Not used in SAMME.R
        
        return self
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostAdvanced':
        """Fit AdaBoost using specified algorithm."""
        if self.algorithm == 'SAMME':
            return self._samme_algorithm(X, y)
        elif self.algorithm == 'SAMME.R':
            return self._samme_r_algorithm(X, y)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using weighted voting."""
        if self.algorithm == 'SAMME':
            # Weighted sum of predictions
            decision = np.zeros((len(X), len(self.classes_)))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                pred = estimator.predict(X)
                for i, cls in enumerate(self.classes_):
                    decision[pred == cls, i] += weight
            
            return self.classes_[np.argmax(decision, axis=1)]
        
        elif self.algorithm == 'SAMME.R':
            # Sum of log probabilities
            decision = np.zeros((len(X), len(self.classes_)))
            
            for estimator in self.estimators_:
                if hasattr(estimator, 'predict_proba'):
                    proba = estimator.predict_proba(X)
                else:
                    # Convert predictions to probabilities
                    pred = estimator.predict(X)
                    proba = np.zeros((len(X), len(self.classes_)))
                    for i, cls in enumerate(self.classes_):
                        proba[pred == cls, i] = 1.0
                
                # Clip and normalize
                proba = np.clip(proba, np.finfo(proba.dtype).eps, None)
                proba /= np.sum(proba, axis=1, keepdims=True)
                
                decision += np.log(proba)
            
            return self.classes_[np.argmax(decision, axis=1)]
    
    def staged_predict(self, X: np.ndarray) -> List[np.ndarray]:
        """Return staged predictions (after each boosting iteration)."""
        staged_predictions = []
        
        if self.algorithm == 'SAMME':
            decision = np.zeros((len(X), len(self.classes_)))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                pred = estimator.predict(X)
                for i, cls in enumerate(self.classes_):
                    decision[pred == cls, i] += weight
                
                staged_pred = self.classes_[np.argmax(decision, axis=1)]
                staged_predictions.append(staged_pred)
        
        elif self.algorithm == 'SAMME.R':
            decision = np.zeros((len(X), len(self.classes_)))
            
            for estimator in self.estimators_:
                if hasattr(estimator, 'predict_proba'):
                    proba = estimator.predict_proba(X)
                else:
                    pred = estimator.predict(X)
                    proba = np.zeros((len(X), len(self.classes_)))
                    for i, cls in enumerate(self.classes_):
                        proba[pred == cls, i] = 1.0
                
                proba = np.clip(proba, np.finfo(proba.dtype).eps, None)
                proba /= np.sum(proba, axis=1, keepdims=True)
                decision += np.log(proba)
                
                staged_pred = self.classes_[np.argmax(decision, axis=1)]
                staged_predictions.append(staged_pred)
        
        return staged_predictions


class GradientBoostingAdvanced(BaseEnsemble):
    """
    Advanced Gradient Boosting with various loss functions and regularization.
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, subsample: float = 1.0,
                 max_features: Union[str, int, float] = None,
                 loss: str = 'mse', alpha: float = 0.9,
                 l1_regularization: float = 0.0, l2_regularization: float = 0.0,
                 early_stopping_rounds: int = None, validation_fraction: float = 0.1,
                 random_state: int = None):
        """Initialize Advanced Gradient Boosting."""
        super().__init__(None, n_estimators, random_state)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.loss = loss
        self.alpha = alpha
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        
        # Fitted attributes
        self.estimators_ = []
        self.train_scores_ = []
        self.validation_scores_ = []
        self.init_prediction_ = None
    
    def _huber_loss(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.9) -> Tuple[float, np.ndarray]:
        """Huber loss function and its gradient."""
        residual = y_true - y_pred
        abs_residual = np.abs(residual)
        
        # Loss
        quadratic_mask = abs_residual <= alpha
        loss = np.where(
            quadratic_mask,
            0.5 * residual ** 2,
            alpha * abs_residual - 0.5 * alpha ** 2
        )
        
        # Gradient
        gradient = np.where(
            quadratic_mask,
            -residual,
            -alpha * np.sign(residual)
        )
        
        return np.mean(loss), gradient
    
    def _quantile_loss(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> Tuple[float, np.ndarray]:
        """Quantile loss function and its gradient."""
        residual = y_true - y_pred
        
        # Loss
        loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
        
        # Gradient (subgradient)
        gradient = np.where(residual >= 0, -alpha, -(alpha - 1))
        
        return np.mean(loss), gradient
    
    def _compute_loss_and_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient based on chosen loss function."""
        if self.loss == 'mse':
            loss = 0.5 * np.mean((y_true - y_pred) ** 2)
            gradient = y_pred - y_true
        elif self.loss == 'mae':
            loss = np.mean(np.abs(y_true - y_pred))
            gradient = np.sign(y_pred - y_true)
        elif self.loss == 'huber':
            loss, gradient = self._huber_loss(y_true, y_pred, self.alpha)
        elif self.loss == 'quantile':
            loss, gradient = self._quantile_loss(y_true, y_pred, self.alpha)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        
        return loss, gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingAdvanced':
        """Fit advanced gradient boosting."""
        n_samples = len(X)
        
        # Split data for early stopping
        if self.early_stopping_rounds:
            n_val = int(self.validation_fraction * n_samples)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Initialize prediction
        if self.loss in ['mse', 'mae', 'huber']:
            self.init_prediction_ = np.mean(y_train)
        elif self.loss == 'quantile':
            self.init_prediction_ = np.percentile(y_train, self.alpha * 100)
        
        # Initialize predictions
        predictions = np.full(len(y_train), self.init_prediction_)
        if X_val is not None:
            val_predictions = np.full(len(y_val), self.init_prediction_)
        
        # Early stopping variables
        best_val_score = np.inf
        rounds_without_improvement = 0
        
        # Training loop
        for i in range(self.n_estimators):
            # Compute gradients
            loss, gradients = self._compute_loss_and_gradient(y_train, predictions)
            self.train_scores_.append(loss)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_size = int(self.subsample * len(X_train))
                sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_sample = X_train[sample_indices]
                gradients_sample = gradients[sample_indices]
            else:
                X_sample = X_train
                gradients_sample = gradients
            
            # Fit tree to negative gradients
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_sample, -gradients_sample)
            
            # Make predictions and update
            tree_predictions = tree.predict(X_train)
            
            # Apply regularization
            if self.l1_regularization > 0:
                tree_predictions = np.sign(tree_predictions) * np.maximum(
                    0, np.abs(tree_predictions) - self.l1_regularization
                )
            
            predictions += self.learning_rate * tree_predictions
            
            # Store estimator
            self.estimators_.append(tree)
            
            # Validation score for early stopping
            if X_val is not None:
                val_tree_pred = tree.predict(X_val)
                if self.l1_regularization > 0:
                    val_tree_pred = np.sign(val_tree_pred) * np.maximum(
                        0, np.abs(val_tree_pred) - self.l1_regularization
                    )
                val_predictions += self.learning_rate * val_tree_pred
                
                val_loss, _ = self._compute_loss_and_gradient(y_val, val_predictions)
                self.validation_scores_.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_score:
                    best_val_score = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Start with initial prediction
        predictions = np.full(len(X), self.init_prediction_)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            tree_pred = tree.predict(X)
            
            # Apply regularization
            if self.l1_regularization > 0:
                tree_pred = np.sign(tree_pred) * np.maximum(
                    0, np.abs(tree_pred) - self.l1_regularization
                )
            
            predictions += self.learning_rate * tree_pred
        
        return predictions
    
    def staged_predict(self, X: np.ndarray) -> List[np.ndarray]:
        """Return staged predictions."""
        staged_predictions = []
        predictions = np.full(len(X), self.init_prediction_)
        
        for tree in self.estimators_:
            tree_pred = tree.predict(X)
            
            if self.l1_regularization > 0:
                tree_pred = np.sign(tree_pred) * np.maximum(
                    0, np.abs(tree_pred) - self.l1_regularization
                )
            
            predictions += self.learning_rate * tree_pred
            staged_predictions.append(predictions.copy())
        
        return staged_predictions


class VotingEnsemble(BaseEnsemble):
    """
    Voting ensemble combining different algorithms.
    """
    
    def __init__(self, estimators: List[Tuple[str, Any]], 
                 voting: str = 'hard', weights: List[float] = None):
        """Initialize Voting ensemble."""
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        
        # Fitted attributes
        self.estimators_ = []
        self.classes_ = None
        self.is_classifier_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'VotingEnsemble':
        """Fit all estimators."""
        # Determine if classification or regression
        unique_y = np.unique(y)
        self.is_classifier_ = len(unique_y) <= 20 and np.all(y == y.astype(int))
        
        if self.is_classifier_:
            self.classes_ = unique_y
        
        # Fit each estimator
        for name, estimator in self.estimators:
            fitted_estimator = estimator.fit(X, y)
            self.estimators_.append((name, fitted_estimator))
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using voting."""
        if self.voting == 'hard' or not self.is_classifier_:
            return self._predict_hard(X)
        elif self.voting == 'soft':
            return self._predict_soft(X)
        else:
            raise ValueError(f"Unknown voting method: {self.voting}")
    
    def _predict_hard(self, X: np.ndarray) -> np.ndarray:
        """Hard voting: majority vote of class predictions."""
        predictions = []
        weights = self.weights if self.weights else [1] * len(self.estimators_)
        
        for (name, estimator), weight in zip(self.estimators_, weights):
            pred = estimator.predict(X)
            predictions.append((pred, weight))
        
        if self.is_classifier_:
            # Weighted majority vote
            final_predictions = np.zeros(len(X))
            for i in range(len(X)):
                vote_counts = {}
                for pred, weight in predictions:
                    label = pred[i]
                    vote_counts[label] = vote_counts.get(label, 0) + weight
                final_predictions[i] = max(vote_counts, key=vote_counts.get)
            return final_predictions.astype(int)
        else:
            # Weighted average
            weighted_sum = np.zeros(len(X))
            total_weight = 0
            for pred, weight in predictions:
                weighted_sum += weight * pred
                total_weight += weight
            return weighted_sum / total_weight
    
    def _predict_soft(self, X: np.ndarray) -> np.ndarray:
        """Soft voting: average of class probabilities."""
        if not self.is_classifier_:
            raise ValueError("Soft voting only available for classification")
        
        avg_probabilities = None
        weights = self.weights if self.weights else [1] * len(self.estimators_)
        total_weight = sum(weights)
        
        for (name, estimator), weight in zip(self.estimators_, weights):
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                # Convert predictions to probabilities
                pred = estimator.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for j, cls in enumerate(self.classes_):
                    proba[pred == cls, j] = 1.0
            
            if avg_probabilities is None:
                avg_probabilities = np.zeros_like(proba)
            
            avg_probabilities += (weight / total_weight) * proba
        
        return self.classes_[np.argmax(avg_probabilities, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using soft voting."""
        if not self.is_classifier_:
            raise ValueError("predict_proba only available for classification")
        
        avg_probabilities = None
        weights = self.weights if self.weights else [1] * len(self.estimators_)
        total_weight = sum(weights)
        
        for (name, estimator), weight in zip(self.estimators_, weights):
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                pred = estimator.predict(X)
                proba = np.zeros((len(X), len(self.classes_)))
                for j, cls in enumerate(self.classes_):
                    proba[pred == cls, j] = 1.0
            
            if avg_probabilities is None:
                avg_probabilities = np.zeros_like(proba)
            
            avg_probabilities += (weight / total_weight) * proba
        
        return avg_probabilities


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner.
    """
    
    def __init__(self, base_estimators: List[Tuple[str, Any]], 
                 meta_estimator=None, cv_folds: int = 5,
                 use_probas: bool = False, average_probas: bool = True):
        """Initialize Stacking ensemble."""
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv_folds = cv_folds
        self.use_probas = use_probas
        self.average_probas = average_probas
        
        # Fitted attributes
        self.fitted_base_estimators_ = []
        self.meta_estimator_ = None
        self.classes_ = None
        self.is_classifier_ = None
        
        # Default meta estimator
        if meta_estimator is None:
            self.meta_estimator = LogisticRegression()
    
    def _create_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create meta-features using cross-validation."""
        # Determine if classification
        unique_y = np.unique(y)
        self.is_classifier_ = len(unique_y) <= 20 and np.all(y == y.astype(int))
        
        if self.is_classifier_:
            self.classes_ = unique_y
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Initialize meta-features array
        if self.use_probas and self.is_classifier_:
            n_meta_features = len(self.base_estimators) * len(self.classes_)
        else:
            n_meta_features = len(self.base_estimators)
        
        meta_features = np.zeros((len(X), n_meta_features))
        
        # Cross-validation to create meta-features
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            feature_idx = 0
            for name, estimator in self.base_estimators:
                # Clone and fit estimator
                from sklearn.base import clone
                estimator_fold = clone(estimator)
                estimator_fold.fit(X_train_fold, y_train_fold)
                
                if self.use_probas and self.is_classifier_ and hasattr(estimator_fold, 'predict_proba'):
                    # Use probabilities
                    proba = estimator_fold.predict_proba(X_val_fold)
                    n_classes = proba.shape[1]
                    meta_features[val_idx, feature_idx:feature_idx + n_classes] = proba
                    feature_idx += n_classes
                else:
                    # Use predictions
                    pred = estimator_fold.predict(X_val_fold)
                    meta_features[val_idx, feature_idx] = pred
                    feature_idx += 1
        
        return meta_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """Fit stacking ensemble."""
        # Create meta-features
        meta_features = self._create_meta_features(X, y)
        
        # Fit all base estimators on full data
        for name, estimator in self.base_estimators:
            fitted_estimator = estimator.fit(X, y)
            self.fitted_base_estimators_.append((name, fitted_estimator))
        
        # Fit meta-estimator
        self.meta_estimator_ = self.meta_estimator.fit(meta_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking."""
        # Get meta-features from base estimators
        meta_features = self._get_meta_features(X)
        
        # Predict with meta-estimator
        return self.meta_estimator_.predict(meta_features)
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from fitted base estimators."""
        if self.use_probas and self.is_classifier_:
            n_meta_features = len(self.fitted_base_estimators_) * len(self.classes_)
        else:
            n_meta_features = len(self.fitted_base_estimators_)
        
        meta_features = np.zeros((len(X), n_meta_features))
        feature_idx = 0
        
        for name, estimator in self.fitted_base_estimators_:
            if self.use_probas and self.is_classifier_ and hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
                n_classes = proba.shape[1]
                meta_features[:, feature_idx:feature_idx + n_classes] = proba
                feature_idx += n_classes
            else:
                pred = estimator.predict(X)
                meta_features[:, feature_idx] = pred
                feature_idx += 1
        
        return meta_features


class MultiLevelStacking:
    """
    Multi-level stacking with multiple meta-learning levels.
    """
    
    def __init__(self, levels: List[List[Tuple[str, Any]]], 
                 final_estimator=None, cv_folds: int = 5):
        """Initialize multi-level stacking."""
        self.levels = levels
        self.final_estimator = final_estimator
        self.cv_folds = cv_folds
        
        # Fitted attributes
        self.fitted_levels_ = []
        self.final_estimator_ = None
        
        # Default final estimator
        if final_estimator is None:
            self.final_estimator = LogisticRegression()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiLevelStacking':
        """Fit multi-level stacking ensemble."""
        current_features = X
        
        # Fit each level
        for level_estimators in self.levels:
            # Create stacking ensemble for this level
            stacking = StackingEnsemble(
                base_estimators=level_estimators,
                cv_folds=self.cv_folds
            )
            
            # Get meta-features for this level
            meta_features = stacking._create_meta_features(current_features, y)
            
            # Fit base estimators
            fitted_level = []
            for name, estimator in level_estimators:
                fitted_estimator = estimator.fit(current_features, y)
                fitted_level.append((name, fitted_estimator))
            
            self.fitted_levels_.append(fitted_level)
            
            # Meta-features become input for next level
            current_features = meta_features
        
        # Fit final estimator
        self.final_estimator_ = self.final_estimator.fit(current_features, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using multi-level stacking."""
        current_features = X
        
        # Pass through each level
        for fitted_level in self.fitted_levels_:
            meta_features = np.zeros((len(X), len(fitted_level)))
            
            for i, (name, estimator) in enumerate(fitted_level):
                meta_features[:, i] = estimator.predict(current_features)
            
            current_features = meta_features
        
        # Final prediction
        return self.final_estimator_.predict(current_features)


class DynamicEnsembleSelection:
    """
    Dynamic Ensemble Selection - select best models for each test instance.
    """
    
    def __init__(self, estimators: List[Tuple[str, Any]], 
                 selection_method: str = 'local_accuracy',
                 k_neighbors: int = 5, diversity_weight: float = 0.0):
        """Initialize Dynamic Ensemble Selection."""
        self.estimators = estimators
        self.selection_method = selection_method
        self.k_neighbors = k_neighbors
        self.diversity_weight = diversity_weight
        
        # Fitted attributes
        self.estimators_ = []
        self.X_val_ = None
        self.y_val_ = None
        self.val_predictions_ = None
    
    def _local_accuracy_selection(self, x: np.ndarray) -> List[int]:
        """Select estimators based on local accuracy."""
        # Find k nearest neighbors
        distances = np.linalg.norm(self.X_val_ - x, axis=1)
        neighbor_indices = np.argsort(distances)[:self.k_neighbors]
        
        # Calculate accuracy of each estimator on neighbors
        accuracies = []
        for i, estimator in enumerate(self.estimators_):
            neighbor_preds = self.val_predictions_[i][neighbor_indices]
            neighbor_targets = self.y_val_[neighbor_indices]
            accuracy = np.mean(neighbor_preds == neighbor_targets)
            accuracies.append(accuracy)
        
        # Select best performers
        best_indices = np.argsort(accuracies)[-max(1, len(self.estimators_) // 2):]
        return best_indices.tolist()
    
    def _diversity_measure(self, predictions: np.ndarray) -> float:
        """Calculate diversity among predictions."""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise disagreement
        n_estimators = len(predictions)
        disagreements = 0
        total_pairs = 0
        
        for i in range(n_estimators):
            for j in range(i + 1, n_estimators):
                disagreements += np.mean(predictions[i] != predictions[j])
                total_pairs += 1
        
        return disagreements / total_pairs if total_pairs > 0 else 0.0
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> 'DynamicEnsembleSelection':
        """Fit DES system."""
        # Store validation set
        self.X_val_ = X_val
        self.y_val_ = y_val
        
        # Train all estimators
        for name, estimator in self.estimators:
            fitted_estimator = estimator.fit(X, y)
            self.estimators_.append(fitted_estimator)
        
        # Get validation predictions
        self.val_predictions_ = []
        for estimator in self.estimators_:
            val_pred = estimator.predict(X_val)
            self.val_predictions_.append(val_pred)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using dynamic selection."""
        predictions = np.zeros(len(X))
        
        for i, x in enumerate(X):
            if self.selection_method == 'local_accuracy':
                selected_indices = self._local_accuracy_selection(x)
            elif self.selection_method == 'overall_accuracy':
                # Select best overall performers
                accuracies = []
                for j, estimator in enumerate(self.estimators_):
                    accuracy = np.mean(self.val_predictions_[j] == self.y_val_)
                    accuracies.append(accuracy)
                selected_indices = [np.argmax(accuracies)]
            else:
                selected_indices = list(range(len(self.estimators_)))
            
            # Get predictions from selected estimators
            selected_preds = []
            for idx in selected_indices:
                pred = self.estimators_[idx].predict(x.reshape(1, -1))[0]
                selected_preds.append(pred)
            
            # Combine predictions (majority vote)
            predictions[i] = Counter(selected_preds).most_common(1)[0][0]
        
        return predictions


class EnsemblePruning:
    """
    Ensemble pruning to select optimal subset of models.
    """
    
    def __init__(self, pruning_method: str = 'ranking', 
                 target_size: int = None, diversity_threshold: float = 0.1):
        """Initialize ensemble pruning."""
        self.pruning_method = pruning_method
        self.target_size = target_size
        self.diversity_threshold = diversity_threshold
    
    def _ranking_pruning(self, estimators: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[int]:
        """Prune ensemble using accuracy ranking."""
        accuracies = []
        
        for estimator in estimators:
            pred = estimator.predict(X_val)
            accuracy = np.mean(pred == y_val)
            accuracies.append(accuracy)
        
        # Select top performers
        n_select = self.target_size if self.target_size else len(estimators) // 2
        selected_indices = np.argsort(accuracies)[-n_select:]
        
        return selected_indices.tolist()
    
    def _clustering_pruning(self, estimators: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[int]:
        """Prune ensemble using clustering of predictions."""
        # Get predictions from all estimators
        predictions = np.array([est.predict(X_val) for est in estimators])
        
        # Cluster estimators based on prediction similarity
        n_clusters = self.target_size if self.target_size else max(2, len(estimators) // 3)
        
        # Use correlation as similarity measure
        correlation_matrix = np.corrcoef(predictions)
        
        # Simple clustering based on correlation
        selected_indices = []
        remaining_indices = list(range(len(estimators)))
        
        for _ in range(n_clusters):
            if not remaining_indices:
                break
            
            # Select estimator with highest average correlation to remaining
            if len(selected_indices) == 0:
                # First selection: highest individual accuracy
                accuracies = [np.mean(predictions[i] == y_val) for i in remaining_indices]
                best_idx = remaining_indices[np.argmax(accuracies)]
            else:
                # Subsequent selections: balance accuracy and diversity
                scores = []
                for idx in remaining_indices:
                    accuracy = np.mean(predictions[idx] == y_val)
                    # Diversity: negative correlation with selected
                    diversity = -np.mean([correlation_matrix[idx, sel_idx] 
                                        for sel_idx in selected_indices])
                    score = accuracy + self.diversity_threshold * diversity
                    scores.append(score)
                best_idx = remaining_indices[np.argmax(scores)]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return selected_indices
    
    def _genetic_pruning(self, estimators: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[int]:
        """Prune ensemble using genetic algorithm."""
        # Simplified genetic algorithm
        population_size = 20
        n_generations = 10
        n_estimators = len(estimators)
        target_size = self.target_size if self.target_size else n_estimators // 2
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = np.random.choice(n_estimators, target_size, replace=False)
            population.append(individual)
        
        # Evolution
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                # Combine predictions from selected estimators
                selected_preds = np.array([estimators[i].predict(X_val) for i in individual])
                ensemble_pred = np.array([Counter(selected_preds[:, j]).most_common(1)[0][0] 
                                        for j in range(len(X_val))])
                accuracy = np.mean(ensemble_pred == y_val)
                fitness_scores.append(accuracy)
            
            # Selection (top 50%)
            top_indices = np.argsort(fitness_scores)[-population_size // 2:]
            new_population = [population[i] for i in top_indices]
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(top_indices, 2, replace=False)
                
                # Crossover
                mask = np.random.rand(target_size) < 0.5
                child = np.where(mask, population[parent1], population[parent2])
                
                # Mutation
                if np.random.rand() < 0.1:
                    mut_idx = np.random.randint(target_size)
                    child[mut_idx] = np.random.randint(n_estimators)
                
                # Ensure unique elements
                child = np.unique(child)
                if len(child) < target_size:
                    remaining = np.setdiff1d(np.arange(n_estimators), child)
                    additional = np.random.choice(remaining, target_size - len(child), replace=False)
                    child = np.concatenate([child, additional])
                
                new_population.append(child[:target_size])
            
            population = new_population
        
        # Return best individual
        final_fitness = []
        for individual in population:
            selected_preds = np.array([estimators[i].predict(X_val) for i in individual])
            ensemble_pred = np.array([Counter(selected_preds[:, j]).most_common(1)[0][0] 
                                    for j in range(len(X_val))])
            accuracy = np.mean(ensemble_pred == y_val)
            final_fitness.append(accuracy)
        
        best_individual = population[np.argmax(final_fitness)]
        return best_individual.tolist()
    
    def prune(self, estimators: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[int]:
        """Prune ensemble to optimal subset."""
        if self.pruning_method == 'ranking':
            return self._ranking_pruning(estimators, X_val, y_val)
        elif self.pruning_method == 'clustering':
            return self._clustering_pruning(estimators, X_val, y_val)
        elif self.pruning_method == 'genetic':
            return self._genetic_pruning(estimators, X_val, y_val)
        else:
            raise ValueError(f"Unknown pruning method: {self.pruning_method}")


def create_diverse_base_learners() -> List[Tuple[str, Any]]:
    """Create a diverse set of base learners for ensembles."""
    base_learners = [
        ('decision_tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('decision_tree_deep', DecisionTreeClassifier(max_depth=10, random_state=43)),
        ('logistic_regression', LogisticRegression(random_state=42, max_iter=1000)),
        ('naive_bayes', GaussianNB()),
        ('svm_linear', SVC(kernel='linear', probability=True, random_state=42)),
        ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42)),
        ('random_forest', RandomForestAdvanced(n_estimators=10, random_state=42))
    ]
    
    return base_learners


def calculate_ensemble_diversity(predictions: np.ndarray) -> Dict[str, float]:
    """Calculate various diversity measures for ensemble."""
    n_estimators, n_samples = predictions.shape
    diversity_measures = {}
    
    # Q-statistic (average pairwise Q)
    q_statistics = []
    for i in range(n_estimators):
        for j in range(i + 1, n_estimators):
            # Confusion matrix elements
            n11 = np.sum((predictions[i] == 1) & (predictions[j] == 1))
            n01 = np.sum((predictions[i] == 0) & (predictions[j] == 1))
            n10 = np.sum((predictions[i] == 1) & (predictions[j] == 0))
            n00 = np.sum((predictions[i] == 0) & (predictions[j] == 0))
            
            if (n11 * n00 + n01 * n10) != 0:
                q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                q_statistics.append(q)
    
    diversity_measures['q_statistic'] = np.mean(q_statistics) if q_statistics else 0.0
    
    # Disagreement measure
    disagreements = []
    for i in range(n_estimators):
        for j in range(i + 1, n_estimators):
            disagreement = np.mean(predictions[i] != predictions[j])
            disagreements.append(disagreement)
    
    diversity_measures['disagreement'] = np.mean(disagreements) if disagreements else 0.0
    
    # Double-fault measure
    double_faults = []
    for i in range(n_estimators):
        for j in range(i + 1, n_estimators):
            # Assume true labels are majority vote
            majority_vote = np.array([Counter(predictions[:, k]).most_common(1)[0][0] 
                                    for k in range(n_samples)])
            
            both_wrong = (predictions[i] != majority_vote) & (predictions[j] != majority_vote)
            double_fault = np.mean(both_wrong)
            double_faults.append(double_fault)
    
    diversity_measures['double_fault'] = np.mean(double_faults) if double_faults else 0.0
    
    # Entropy measure
    entropies = []
    for i in range(n_samples):
        sample_predictions = predictions[:, i]
        unique, counts = np.unique(sample_predictions, return_counts=True)
        probabilities = counts / n_estimators
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))
        entropies.append(entropy)
    
    diversity_measures['entropy'] = np.mean(entropies)
    
    return diversity_measures


def bias_variance_decomposition(ensemble, X: np.ndarray, y: np.ndarray, 
                               n_bootstrap: int = 100) -> Dict[str, float]:
    """Empirical bias-variance decomposition for ensemble."""
    n_samples = len(X)
    n_test = min(50, n_samples // 4)  # Use subset for testing
    
    # Split data
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.setdiff1d(np.arange(n_samples), test_indices)
    
    X_train_pool = X[train_indices]
    y_train_pool = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    # Bootstrap experiments
    predictions = np.zeros((n_bootstrap, n_test))
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_indices = np.random.choice(len(X_train_pool), len(X_train_pool), replace=True)
        X_boot = X_train_pool[boot_indices]
        y_boot = y_train_pool[boot_indices]
        
        # Train ensemble
        ensemble_copy = type(ensemble)(**ensemble.__dict__)
        ensemble_copy.fit(X_boot, y_boot)
        
        # Predict on test set
        pred = ensemble_copy.predict(X_test)
        predictions[i] = pred
    
    # Calculate bias and variance
    mean_predictions = np.mean(predictions, axis=0)
    
    # Bias squared
    bias_squared = np.mean((mean_predictions - y_test) ** 2)
    
    # Variance
    variance = np.mean(np.var(predictions, axis=0))
    
    # Noise (irreducible error)
    noise = 0.0  # Assume no noise for synthetic data
    
    # Total error
    total_error = bias_squared + variance + noise
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': noise,
        'total_error': total_error
    }


def ensemble_learning_curves(ensembles: List[Any], X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
    """Generate learning curves comparing different ensemble methods."""
    train_sizes = np.linspace(0.1, 1.0, 10)
    n_samples = len(X)
    
    results = {}
    
    for i, ensemble in enumerate(ensembles):
        ensemble_name = f"ensemble_{i}"
        train_scores = []
        val_scores = []
        
        for train_size in train_sizes:
            # Create train/validation split
            n_train = int(train_size * n_samples * 0.8)  # 80% for training
            n_val = int(train_size * n_samples * 0.2)    # 20% for validation
            
            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                continue
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_val = X[val_indices]
            y_val = y[val_indices]
            
            # Train ensemble
            ensemble_copy = type(ensemble)(**ensemble.__dict__)
            ensemble_copy.fit(X_train, y_train)
            
            # Evaluate
            train_pred = ensemble_copy.predict(X_train)
            val_pred = ensemble_copy.predict(X_val)
            
            # Determine if classification or regression
            if len(np.unique(y)) <= 20 and np.all(y == y.astype(int)):
                train_score = np.mean(train_pred == y_train)
                val_score = np.mean(val_pred == y_val)
            else:
                train_score = -np.mean((train_pred - y_train) ** 2)
                val_score = -np.mean((val_pred - y_val) ** 2)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        results[f"{ensemble_name}_train"] = train_scores
        results[f"{ensemble_name}_val"] = val_scores
    
    return results


def optimal_ensemble_size_analysis(ensemble_class, X: np.ndarray, y: np.ndarray, 
                                 max_estimators: int = 200) -> Tuple[int, List[float]]:
    """Find optimal ensemble size using validation curve."""
    estimator_range = range(1, max_estimators + 1, max(1, max_estimators // 20))
    performance_scores = []
    
    # Split data
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]
    
    for n_estimators in estimator_range:
        # Create ensemble
        ensemble = ensemble_class(n_estimators=n_estimators)
        ensemble.fit(X_train, y_train)
        
        # Evaluate
        pred = ensemble.predict(X_val)
        
        # Determine scoring
        if len(np.unique(y)) <= 20 and np.all(y == y.astype(int)):
            score = np.mean(pred == y_val)
        else:
            score = -np.mean((pred - y_val) ** 2)
        
        performance_scores.append(score)
    
    # Find optimal size (point of diminishing returns)
    # Look for where improvement becomes minimal
    improvements = np.diff(performance_scores)
    
    # Find where improvement drops below threshold
    threshold = np.max(improvements) * 0.1
    optimal_idx = 0
    for i, improvement in enumerate(improvements):
        if improvement < threshold:
            optimal_idx = i
            break
    
    optimal_size = list(estimator_range)[optimal_idx]
    
    return optimal_size, performance_scores


def ensemble_interpretability_analysis(ensemble, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
    """Analyze ensemble for interpretability."""
    results = {}
    
    # Feature importance
    if hasattr(ensemble, 'feature_importances'):
        try:
            importances = ensemble.feature_importances(X)
            results['feature_importances'] = importances
            
            if feature_names:
                results['feature_importance_ranking'] = [
                    feature_names[i] for i in np.argsort(importances)[::-1]
                ]
        except:
            pass
    
    # Prediction confidence (for classification)
    if hasattr(ensemble, 'predict_proba'):
        try:
            probabilities = ensemble.predict_proba(X)
            max_probabilities = np.max(probabilities, axis=1)
            results['prediction_confidence'] = {
                'mean_confidence': np.mean(max_probabilities),
                'confidence_std': np.std(max_probabilities),
                'low_confidence_samples': np.sum(max_probabilities < 0.6)
            }
        except:
            pass
    
    # Base learner analysis
    if hasattr(ensemble, 'estimators_'):
        n_estimators = len(ensemble.estimators_)
        results['n_base_learners'] = n_estimators
        
        # Estimator weights (if available)
        if hasattr(ensemble, 'estimator_weights_'):
            weights = np.array(ensemble.estimator_weights_)
            results['estimator_weights'] = {
                'weights': weights,
                'weight_entropy': -np.sum((weights / np.sum(weights)) * 
                                        np.log(weights / np.sum(weights) + 1e-8))
            }
    
    return results


# Export all solution implementations
__all__ = [
    'BaggingEnsemble', 'RandomForestAdvanced', 'AdaBoostAdvanced',
    'GradientBoostingAdvanced', 'VotingEnsemble', 'StackingEnsemble',
    'MultiLevelStacking', 'DynamicEnsembleSelection', 'EnsemblePruning',
    'create_diverse_base_learners', 'calculate_ensemble_diversity',
    'bias_variance_decomposition', 'ensemble_learning_curves',
    'optimal_ensemble_size_analysis', 'ensemble_interpretability_analysis'
]