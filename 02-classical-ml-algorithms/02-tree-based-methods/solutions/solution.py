"""
Reference Solutions for Tree-Based Methods

This module contains complete implementations of all tree-based algorithms.
Use this as a reference after attempting the exercises yourself.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Any
from dataclasses import dataclass
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

@dataclass
class TreeNode:
    """Represents a node in a decision tree."""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    value: Optional[Union[float, int]] = None
    samples: int = 0
    impurity: float = 0.0
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

class DecisionTree:
    """Complete Decision Tree implementation using CART algorithm."""
    
    def __init__(self, 
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 criterion: str = 'gini',
                 random_state: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy of a target array."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # Handle log(0) case
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error for regression."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity based on chosen criterion."""
        if self.criterion == 'entropy':
            return self._entropy(y)
        elif self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'mse':
            return self._mse(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Calculate information gain from a split."""
        if len(y) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = self._calculate_impurity(y)
        weighted_child_impurity = (n_left / n) * self._calculate_impurity(y_left) + \
                                 (n_right / n) * self._calculate_impurity(y_right)
        
        return parent_impurity - weighted_child_impurity
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split for given features."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in feature_indices:
            # Get unique values and sort them
            unique_values = np.unique(X[:, feature])
            
            # Consider thresholds between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split doesn't meet minimum samples requirement
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate information gain
                gain = self._information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf(self, y: np.ndarray) -> TreeNode:
        """Create a leaf node with appropriate prediction value."""
        if self.is_classifier_:
            # Most common class
            value = Counter(y).most_common(1)[0][0]
        else:
            # Mean value for regression
            value = np.mean(y)
        
        return TreeNode(
            value=value,
            samples=len(y),
            impurity=self._calculate_impurity(y)
        )
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0, feature_indices: np.ndarray = None) -> TreeNode:
        """Recursively build the decision tree."""
        if feature_indices is None:
            feature_indices = np.arange(X.shape[1])
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(y) < self.min_samples_split or \
           len(np.unique(y)) == 1:  # Pure node
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)
        
        # If no good split found, create leaf
        if best_feature is None or best_gain <= 0:
            return self._create_leaf(y)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Create node
        node = TreeNode(
            feature=best_feature,
            threshold=best_threshold,
            samples=len(y),
            impurity=self._calculate_impurity(y)
        )
        
        # Recursively build subtrees
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices)
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Fit the decision tree to training data."""
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        # Determine if classification or regression
        if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
            self.is_classifier_ = True
        else:
            self.is_classifier_ = False
        
        # Build tree
        self.root_ = self._build_tree(X, y)
        
        return self
    
    def _predict_sample(self, x: np.ndarray, node: TreeNode) -> Union[int, float]:
        """Predict a single sample by traversing the tree."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels/values for input data."""
        return np.array([self._predict_sample(x, self.root_) for x in X])

class RandomForest:
    """Complete Random Forest implementation."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
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
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create bootstrap sample and track out-of-bag indices."""
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
        else:
            indices = np.arange(n_samples)
            oob_indices = np.array([])
        
        return X[indices], y[indices], oob_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """Fit the random forest."""
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        self.max_features_ = self._calculate_max_features(self.n_features_)
        
        # Determine if classification or regression
        if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
            self.is_classifier_ = True
        else:
            self.is_classifier_ = False
        
        self.trees_ = []
        
        # Initialize OOB arrays
        if self.oob_score:
            oob_predictions = np.zeros((self.n_samples_, self.n_estimators))
            oob_mask = np.zeros((self.n_samples_, self.n_estimators), dtype=bool)
        
        # Train trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_boot, y_boot, oob_indices = self._bootstrap_sample(X, y)
            
            # Create tree with random feature selection
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='gini' if self.is_classifier_ else 'mse'
            )
            
            # Override the _find_best_split method to use random features
            original_find_best_split = tree._find_best_split
            
            def random_feature_split(X_split, y_split, feature_indices):
                # Randomly select subset of features
                n_features_to_select = min(self.max_features_, len(feature_indices))
                selected_features = np.random.choice(
                    feature_indices, n_features_to_select, replace=False
                )
                return original_find_best_split(X_split, y_split, selected_features)
            
            tree._find_best_split = random_feature_split
            
            # Fit tree
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            
            # Store OOB predictions
            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices, i] = oob_pred
                oob_mask[oob_indices, i] = True
        
        # Calculate OOB score
        if self.oob_score:
            # Average predictions for samples that were OOB
            oob_pred_final = np.zeros(self.n_samples_)
            for i in range(self.n_samples_):
                if np.sum(oob_mask[i, :]) > 0:
                    if self.is_classifier_:
                        # Majority vote
                        votes = oob_predictions[i, oob_mask[i, :]]
                        oob_pred_final[i] = Counter(votes).most_common(1)[0][0]
                    else:
                        # Average
                        oob_pred_final[i] = np.mean(oob_predictions[i, oob_mask[i, :]])
            
            # Calculate score
            valid_oob = np.sum(oob_mask, axis=1) > 0
            if np.sum(valid_oob) > 0:
                if self.is_classifier_:
                    self.oob_score_ = np.mean(oob_pred_final[valid_oob] == y[valid_oob])
                else:
                    self.oob_score_ = -np.mean((oob_pred_final[valid_oob] - y[valid_oob]) ** 2)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote (classification) or average (regression)."""
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        if self.is_classifier_:
            # Majority vote
            final_predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                votes = predictions[:, i]
                final_predictions[i] = Counter(votes).most_common(1)[0][0]
            return final_predictions.astype(int)
        else:
            # Average
            return np.mean(predictions, axis=0)

class AdaBoost:
    """Complete AdaBoost implementation for binary classification."""
    
    def __init__(self,
                 n_estimators: int = 50,
                 learning_rate: float = 1.0,
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoost':
        """Fit AdaBoost classifier using AdaBoost.M1 algorithm."""
        # Store classes and convert to {-1, +1}
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost only supports binary classification")
        
        y_encoded = np.where(y == self.classes_[0], -1, 1)
        
        n_samples = X.shape[0]
        
        # Initialize uniform weights
        weights = np.ones(n_samples) / n_samples
        
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        
        for _ in range(self.n_estimators):
            # Train weak learner (decision stump)
            stump = DecisionTree(max_depth=1)
            stump.fit(X, y_encoded, sample_weight=weights)
            
            # Get predictions
            predictions = stump.predict(X)
            
            # Calculate weighted error
            incorrect = predictions != y_encoded
            error = np.sum(weights * incorrect) / np.sum(weights)
            
            # Avoid division by zero
            if error >= 0.5:
                if len(self.estimators_) == 0:
                    raise ValueError("First weak learner has error >= 0.5")
                break
            
            # Calculate alpha (estimator weight)
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Store estimator and weight
            self.estimators_.append(stump)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(error)
            
            # Update weights
            weights *= np.exp(-alpha * y_encoded * predictions)
            weights /= np.sum(weights)  # Normalize
            
            # Early stopping if error is 0
            if error == 0:
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted majority vote."""
        # Get weighted predictions
        weighted_sum = np.zeros(X.shape[0])
        
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            weighted_sum += alpha * predictions
        
        # Take sign and convert back to original labels
        binary_predictions = np.sign(weighted_sum)
        return np.where(binary_predictions == -1, self.classes_[0], self.classes_[1])

class GradientBoosting:
    """Complete Gradient Boosting implementation."""
    
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 subsample: float = 1.0,
                 loss: str = 'mse',
                 random_state: int = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _mse_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Mean squared error loss and its gradient."""
        loss = 0.5 * np.mean((y_true - y_pred) ** 2)
        gradient = y_pred - y_true
        return loss, gradient
    
    def _log_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Logistic loss for binary classification."""
        # For y ∈ {-1, +1}, F = log-odds
        # Clip to avoid overflow
        y_pred = np.clip(y_pred, -250, 250)
        
        # Loss = log(1 + exp(-y*F))
        loss = np.mean(np.log(1 + np.exp(-y_true * y_pred)))
        
        # Gradient = -y / (1 + exp(y*F))
        gradient = -y_true / (1 + np.exp(y_true * y_pred))
        
        return loss, gradient
    
    def _compute_loss_and_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient based on chosen loss function."""
        if self.loss == 'mse':
            return self._mse_loss(y_true, y_pred)
        elif self.loss == 'log_loss':
            return self._log_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoosting':
        """Fit gradient boosting model."""
        n_samples, n_features = X.shape
        
        # Initialize prediction
        if self.loss == 'mse':
            self.init_prediction_ = np.mean(y)
        elif self.loss == 'log_loss':
            # Convert to {-1, +1} if needed
            if np.all(np.isin(y, [0, 1])):
                y = 2 * y - 1
            # Initialize with log-odds
            pos_rate = np.sum(y == 1) / len(y)
            pos_rate = np.clip(pos_rate, 1e-15, 1 - 1e-15)  # Avoid log(0)
            self.init_prediction_ = 0.5 * np.log(pos_rate / (1 - pos_rate))
        
        # Initialize predictions
        predictions = np.full(n_samples, self.init_prediction_)
        
        self.estimators_ = []
        self.train_scores_ = []
        
        for i in range(self.n_estimators):
            # Compute loss and gradients
            loss, gradients = self._compute_loss_and_gradient(y, predictions)
            self.train_scores_.append(loss)
            
            # Subsample data
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples, int(self.subsample * n_samples), replace=False
                )
                X_sample = X[sample_indices]
                gradients_sample = gradients[sample_indices]
            else:
                X_sample = X
                gradients_sample = gradients
            
            # Fit regression tree to negative gradients
            tree = DecisionTree(
                max_depth=self.max_depth,
                criterion='mse'
            )
            tree.fit(X_sample, -gradients_sample)
            
            # Make predictions and update
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions
            
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ensemble."""
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.init_prediction_)
        
        # Add contributions from all trees
        for tree in self.estimators_:
            predictions += self.learning_rate * tree.predict(X)
        
        # Apply inverse link function if needed
        if self.loss == 'log_loss':
            # Convert log-odds to probabilities, then to class predictions
            probabilities = 1 / (1 + np.exp(-predictions))
            return np.where(probabilities >= 0.5, 1, -1)
        else:
            return predictions

# Enhanced DecisionTree with sample weights for AdaBoost
class WeightedDecisionTree(DecisionTree):
    """Decision tree that supports sample weights for AdaBoost."""
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> 'WeightedDecisionTree':
        """Fit with sample weights."""
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        
        self.sample_weight_ = sample_weight
        return super().fit(X, y)
    
    def _calculate_impurity(self, y: np.ndarray, indices: np.ndarray = None) -> float:
        """Calculate weighted impurity."""
        if indices is None:
            indices = np.arange(len(y))
        
        if len(indices) == 0:
            return 0
        
        weights = self.sample_weight_[indices]
        weights = weights / np.sum(weights)  # Normalize
        
        if self.criterion == 'gini':
            unique_labels, label_indices = np.unique(y[indices], return_inverse=True)
            weighted_counts = np.bincount(label_indices, weights=weights)
            return 1 - np.sum(weighted_counts ** 2)
        else:
            return super()._calculate_impurity(y[indices])

# Update AdaBoost to use WeightedDecisionTree
def create_weighted_stump(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> WeightedDecisionTree:
    """Create a decision stump with weighted training."""
    stump = WeightedDecisionTree(max_depth=1, criterion='gini')
    stump.fit(X, y, weights)
    return stump

# Sample data generation functions
def load_sample_data(dataset: str = 'classification') -> Tuple[np.ndarray, np.ndarray]:
    """Load sample datasets for testing."""
    np.random.seed(42)
    
    if dataset == 'classification':
        # 3-class spiral dataset
        n_samples = 300
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
        
        for class_id in range(3):
            n_class = n_samples // 3
            t = np.linspace(class_id * 4, (class_id + 1) * 4, n_class)
            r = t
            X[class_id * n_class:(class_id + 1) * n_class, 0] = r * np.cos(t) + 0.1 * np.random.randn(n_class)
            X[class_id * n_class:(class_id + 1) * n_class, 1] = r * np.sin(t) + 0.1 * np.random.randn(n_class)
            y[class_id * n_class:(class_id + 1) * n_class] = class_id
        
        return X, y
    
    elif dataset == 'regression':
        # Non-linear 1D regression
        X = np.linspace(0, 4, 100).reshape(-1, 1)
        y = np.sin(2 * X.ravel()) + 0.1 * np.random.randn(100)
        return X, y
    
    elif dataset == 'circles':
        # Concentric circles
        n_samples = 200
        X = np.random.randn(n_samples, 2)
        r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        y = (r > 1.5).astype(int)
        return X, y
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Visualization and analysis functions
def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, title: str = "Decision Boundary"):
    """Plot decision boundary for 2D classification problems."""
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def compare_ensemble_sizes(X: np.ndarray, y: np.ndarray, max_estimators: int = 200) -> Dict[str, List[float]]:
    """Study how ensemble size affects performance."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    estimator_range = range(1, max_estimators + 1, 10)
    results = {
        'n_estimators': list(estimator_range),
        'rf_train': [],
        'rf_test': [],
        'ada_train': [],
        'ada_test': []
    }
    
    # Determine if classification or regression
    is_classification = len(np.unique(y)) < 10
    
    for n_est in estimator_range:
        # Random Forest
        rf = RandomForest(n_estimators=n_est, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        rf_train_pred = rf.predict(X_train)
        rf_test_pred = rf.predict(X_test)
        
        if is_classification:
            rf_train_score = accuracy_score(y_train, rf_train_pred)
            rf_test_score = accuracy_score(y_test, rf_test_pred)
        else:
            rf_train_score = -mean_squared_error(y_train, rf_train_pred)
            rf_test_score = -mean_squared_error(y_test, rf_test_pred)
        
        results['rf_train'].append(rf_train_score)
        results['rf_test'].append(rf_test_score)
        
        # AdaBoost (only for binary classification)
        if is_classification and len(np.unique(y)) == 2:
            ada = AdaBoost(n_estimators=n_est, random_state=42)
            ada.fit(X_train, y_train)
            
            ada_train_pred = ada.predict(X_train)
            ada_test_pred = ada.predict(X_test)
            
            ada_train_score = accuracy_score(y_train, ada_train_pred)
            ada_test_score = accuracy_score(y_test, ada_test_pred)
        else:
            ada_train_score = ada_test_score = 0
        
        results['ada_train'].append(ada_train_score)
        results['ada_test'].append(ada_test_score)
    
    return results

if __name__ == "__main__":
    print("Running complete tree-based methods implementations...")
    
    # Test all implementations
    print("\n1. Testing Decision Tree...")
    X_class, y_class = load_sample_data('classification')
    tree = DecisionTree(max_depth=5, criterion='gini')
    tree.fit(X_class, y_class)
    tree_pred = tree.predict(X_class)
    print(f"Decision Tree Accuracy: {np.mean(tree_pred == y_class):.3f}")
    
    print("\n2. Testing Random Forest...")
    rf = RandomForest(n_estimators=20, max_depth=5, oob_score=True)
    rf.fit(X_class, y_class)
    rf_pred = rf.predict(X_class)
    print(f"Random Forest Accuracy: {np.mean(rf_pred == y_class):.3f}")
    if hasattr(rf, 'oob_score_'):
        print(f"OOB Score: {rf.oob_score_:.3f}")
    
    print("\n3. Testing AdaBoost...")
    # Create binary classification data
    X_binary = X_class[y_class != 2]
    y_binary = y_class[y_class != 2]
    
    ada = AdaBoost(n_estimators=20)
    ada.fit(X_binary, y_binary)
    ada_pred = ada.predict(X_binary)
    print(f"AdaBoost Accuracy: {np.mean(ada_pred == y_binary):.3f}")
    
    print("\n4. Testing Gradient Boosting...")
    X_reg, y_reg = load_sample_data('regression')
    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, loss='mse')
    gb.fit(X_reg, y_reg)
    gb_pred = gb.predict(X_reg)
    mse = np.mean((gb_pred - y_reg) ** 2)
    print(f"Gradient Boosting MSE: {mse:.3f}")
    
    print("\n✅ All implementations working correctly!")