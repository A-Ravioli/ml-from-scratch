"""
Solution implementations for Instance-Based Learning exercises.

This file provides complete implementations of all TODO items in exercise.py.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Union
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cdist
import warnings


# Base Classes

class DistanceMetric:
    """Base class for distance metrics."""
    
    def __init__(self, name: str):
        self.name = name
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two points."""
        raise NotImplementedError
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute distance matrix between sets of points."""
        if X2 is None:
            X2 = X1
        
        n1, n2 = len(X1), len(X2)
        distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                distances[i, j] = self.compute(X1[i], X2[j])
        
        return distances


class EuclideanDistance(DistanceMetric):
    """Euclidean (L2) distance metric."""
    
    def __init__(self):
        super().__init__("Euclidean")
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Efficient matrix computation using broadcasting."""
        if X2 is None:
            X2 = X1
        
        # Use scipy for efficiency
        return cdist(X1, X2, metric='euclidean')


class ManhattanDistance(DistanceMetric):
    """Manhattan (L1) distance metric."""
    
    def __init__(self):
        super().__init__("Manhattan")
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Manhattan distance."""
        return np.sum(np.abs(x1 - x2))
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Efficient matrix computation."""
        if X2 is None:
            X2 = X1
        return cdist(X1, X2, metric='manhattan')


class MinkowskiDistance(DistanceMetric):
    """Minkowski distance metric (generalizes Euclidean and Manhattan)."""
    
    def __init__(self, p: float = 2.0):
        super().__init__(f"Minkowski-{p}")
        self.p = p
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Minkowski distance."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1.0 / self.p)
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Efficient matrix computation."""
        if X2 is None:
            X2 = X1
        return cdist(X1, X2, metric='minkowski', p=self.p)


class CosineDistance(DistanceMetric):
    """Cosine distance metric."""
    
    def __init__(self):
        super().__init__("Cosine")
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute cosine distance."""
        # Cosine distance = 1 - cosine similarity
        dot_product = np.dot(x1, x2)
        norm_product = np.linalg.norm(x1) * np.linalg.norm(x2)
        
        if norm_product == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_sim = dot_product / norm_product
        return 1.0 - cosine_sim
    
    def compute_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Efficient matrix computation."""
        if X2 is None:
            X2 = X1
        return cdist(X1, X2, metric='cosine')


class MahalanobisDistance(DistanceMetric):
    """Mahalanobis distance metric."""
    
    def __init__(self, cov_matrix: Optional[np.ndarray] = None):
        super().__init__("Mahalanobis")
        self.cov_matrix = cov_matrix
        self.inv_cov_matrix = None
    
    def fit(self, X: np.ndarray):
        """Fit the covariance matrix from training data."""
        if self.cov_matrix is None:
            self.cov_matrix = np.cov(X.T)
        
        # Compute inverse with regularization for numerical stability
        try:
            self.inv_cov_matrix = np.linalg.inv(self.cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization
            reg_cov = self.cov_matrix + 1e-6 * np.eye(self.cov_matrix.shape[0])
            self.inv_cov_matrix = np.linalg.inv(reg_cov)
    
    def compute(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Mahalanobis distance."""
        if self.inv_cov_matrix is None:
            raise ValueError("Must fit covariance matrix before computing distance")
        
        diff = x1 - x2
        return np.sqrt(diff @ self.inv_cov_matrix @ diff)


# K-Nearest Neighbors

class KNearestNeighbors:
    """K-Nearest Neighbors for classification and regression."""
    
    def __init__(self, k: int = 3, distance_metric: DistanceMetric = None,
                 algorithm: str = 'brute', weights: str = 'uniform',
                 leaf_size: int = 30):
        self.k = k
        self.distance_metric = distance_metric or EuclideanDistance()
        self.algorithm = algorithm
        self.weights = weights
        self.leaf_size = leaf_size
        
        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.is_classifier_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the KNN model."""
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Determine if classification or regression
        unique_y = np.unique(y)
        self.is_classifier_ = (len(unique_y) <= 20 and 
                             np.all(y == y.astype(int)))
        
        if self.is_classifier_:
            self.classes_ = unique_y
        
        # Fit distance metric if needed
        if hasattr(self.distance_metric, 'fit'):
            self.distance_metric.fit(X)
        
        return self
    
    def _find_neighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each query point."""
        if self.algorithm == 'brute':
            return self._brute_force_neighbors(X)
        elif self.algorithm == 'kd_tree':
            return self._kd_tree_neighbors(X)
        elif self.algorithm == 'ball_tree':
            return self._ball_tree_neighbors(X)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _brute_force_neighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors using brute force search."""
        distances = self.distance_metric.compute_matrix(X, self.X_train_)
        
        # Get k nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, :self.k]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        
        return neighbor_distances, neighbor_indices
    
    def _kd_tree_neighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors using KD-tree (simplified implementation)."""
        # For simplicity, fall back to brute force
        # In practice, would implement proper KD-tree
        return self._brute_force_neighbors(X)
    
    def _ball_tree_neighbors(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors using Ball tree (simplified implementation)."""
        # For simplicity, fall back to brute force
        # In practice, would implement proper Ball tree
        return self._brute_force_neighbors(X)
    
    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        """Compute weights for neighbors."""
        if self.weights == 'uniform':
            return np.ones_like(distances)
        elif self.weights == 'distance':
            # Inverse distance weighting, avoid division by zero
            weights = 1 / (distances + 1e-8)
            return weights
        else:
            raise ValueError(f"Unknown weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for query points."""
        if self.X_train_ is None:
            raise ValueError("Must fit before prediction")
        
        neighbor_distances, neighbor_indices = self._find_neighbors(X)
        weights = self._compute_weights(neighbor_distances)
        
        if self.is_classifier_:
            return self._predict_classification(neighbor_indices, weights)
        else:
            return self._predict_regression(neighbor_indices, weights)
    
    def _predict_classification(self, neighbor_indices: np.ndarray, 
                              weights: np.ndarray) -> np.ndarray:
        """Predict class labels using weighted voting."""
        n_queries = neighbor_indices.shape[0]
        predictions = np.zeros(n_queries)
        
        for i in range(n_queries):
            neighbor_labels = self.y_train_[neighbor_indices[i]]
            neighbor_weights = weights[i]
            
            # Weighted voting
            vote_counts = {}
            for label, weight in zip(neighbor_labels, neighbor_weights):
                vote_counts[label] = vote_counts.get(label, 0) + weight
            
            # Get label with maximum weighted votes
            predictions[i] = max(vote_counts, key=vote_counts.get)
        
        return predictions.astype(int)
    
    def _predict_regression(self, neighbor_indices: np.ndarray,
                          weights: np.ndarray) -> np.ndarray:
        """Predict continuous values using weighted average."""
        n_queries = neighbor_indices.shape[0]
        predictions = np.zeros(n_queries)
        
        for i in range(n_queries):
            neighbor_values = self.y_train_[neighbor_indices[i]]
            neighbor_weights = weights[i]
            
            # Weighted average
            predictions[i] = np.average(neighbor_values, weights=neighbor_weights)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for classification."""
        if not self.is_classifier_:
            raise ValueError("predict_proba only available for classification")
        
        neighbor_distances, neighbor_indices = self._find_neighbors(X)
        weights = self._compute_weights(neighbor_distances)
        
        n_queries = neighbor_indices.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_queries, n_classes))
        
        for i in range(n_queries):
            neighbor_labels = self.y_train_[neighbor_indices[i]]
            neighbor_weights = weights[i]
            
            # Compute weighted class probabilities
            for j, class_label in enumerate(self.classes_):
                class_mask = (neighbor_labels == class_label)
                class_weight = np.sum(neighbor_weights[class_mask])
                probabilities[i, j] = class_weight
            
            # Normalize to probabilities
            total_weight = np.sum(probabilities[i])
            if total_weight > 0:
                probabilities[i] /= total_weight
            else:
                probabilities[i] = 1.0 / n_classes  # Uniform if no neighbors
        
        return probabilities


# Radius-based Neighbors

class RadiusNeighbors:
    """Radius-based neighbors for classification and regression."""
    
    def __init__(self, radius: float = 1.0, distance_metric: DistanceMetric = None,
                 weights: str = 'uniform', outlier_label: Optional[int] = None):
        self.radius = radius
        self.distance_metric = distance_metric or EuclideanDistance()
        self.weights = weights
        self.outlier_label = outlier_label
        
        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = None
        self.is_classifier_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the radius neighbors model."""
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Determine if classification or regression
        unique_y = np.unique(y)
        self.is_classifier_ = (len(unique_y) <= 20 and 
                             np.all(y == y.astype(int)))
        
        if self.is_classifier_:
            self.classes_ = unique_y
        
        # Fit distance metric if needed
        if hasattr(self.distance_metric, 'fit'):
            self.distance_metric.fit(X)
        
        return self
    
    def _find_radius_neighbors(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Find neighbors within radius for each query point."""
        distances = self.distance_metric.compute_matrix(X, self.X_train_)
        
        neighbors_list = []
        for i in range(len(X)):
            within_radius = distances[i] <= self.radius
            neighbor_indices = np.where(within_radius)[0]
            neighbor_distances = distances[i][within_radius]
            neighbors_list.append((neighbor_distances, neighbor_indices))
        
        return neighbors_list
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for query points."""
        if self.X_train_ is None:
            raise ValueError("Must fit before prediction")
        
        neighbors_list = self._find_radius_neighbors(X)
        
        if self.is_classifier_:
            return self._predict_classification(neighbors_list)
        else:
            return self._predict_regression(neighbors_list)
    
    def _predict_classification(self, neighbors_list: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Predict class labels using neighbors within radius."""
        predictions = []
        
        for neighbor_distances, neighbor_indices in neighbors_list:
            if len(neighbor_indices) == 0:
                # No neighbors within radius
                if self.outlier_label is not None:
                    predictions.append(self.outlier_label)
                else:
                    # Use most common class as default
                    predictions.append(Counter(self.y_train_).most_common(1)[0][0])
                continue
            
            neighbor_labels = self.y_train_[neighbor_indices]
            
            if self.weights == 'uniform':
                # Simple majority vote
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            elif self.weights == 'distance':
                # Distance-weighted voting
                weights = 1 / (neighbor_distances + 1e-8)
                vote_counts = {}
                for label, weight in zip(neighbor_labels, weights):
                    vote_counts[label] = vote_counts.get(label, 0) + weight
                prediction = max(vote_counts, key=vote_counts.get)
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def _predict_regression(self, neighbors_list: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Predict continuous values using neighbors within radius."""
        predictions = []
        
        for neighbor_distances, neighbor_indices in neighbors_list:
            if len(neighbor_indices) == 0:
                # No neighbors within radius - use global mean
                predictions.append(np.mean(self.y_train_))
                continue
            
            neighbor_values = self.y_train_[neighbor_indices]
            
            if self.weights == 'uniform':
                prediction = np.mean(neighbor_values)
            elif self.weights == 'distance':
                weights = 1 / (neighbor_distances + 1e-8)
                prediction = np.average(neighbor_values, weights=weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)


# Locally Weighted Learning

class LocallyWeightedRegression:
    """Locally Weighted Regression (LOWESS/LOESS)."""
    
    def __init__(self, kernel: str = 'gaussian', bandwidth: float = 1.0,
                 distance_metric: DistanceMetric = None):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.distance_metric = distance_metric or EuclideanDistance()
        
        # Fitted attributes
        self.X_train_ = None
        self.y_train_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the locally weighted regression model."""
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Fit distance metric if needed
        if hasattr(self.distance_metric, 'fit'):
            self.distance_metric.fit(X)
        
        return self
    
    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        """Compute kernel weights based on distances."""
        if self.kernel == 'gaussian':
            return np.exp(-0.5 * (distances / self.bandwidth) ** 2)
        elif self.kernel == 'epanechnikov':
            normalized_dist = distances / self.bandwidth
            weights = np.maximum(0, 1 - normalized_dist ** 2)
            return 0.75 * weights
        elif self.kernel == 'tricube':
            normalized_dist = np.minimum(distances / self.bandwidth, 1.0)
            weights = (1 - normalized_dist ** 3) ** 3
            return weights
        elif self.kernel == 'uniform':
            return (distances <= self.bandwidth).astype(float)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using locally weighted regression."""
        if self.X_train_ is None:
            raise ValueError("Must fit before prediction")
        
        predictions = np.zeros(len(X))
        
        for i, query_point in enumerate(X):
            # Compute distances to all training points
            distances = np.array([
                self.distance_metric.compute(query_point, train_point)
                for train_point in self.X_train_
            ])
            
            # Compute kernel weights
            weights = self._kernel_function(distances)
            
            # Handle case where all weights are zero
            if np.sum(weights) == 0:
                predictions[i] = np.mean(self.y_train_)
                continue
            
            # Fit weighted linear regression
            predictions[i] = self._weighted_linear_regression(
                query_point, weights
            )
        
        return predictions
    
    def _weighted_linear_regression(self, query_point: np.ndarray,
                                  weights: np.ndarray) -> float:
        """Fit weighted linear regression for a single query point."""
        # Add bias term
        X_design = np.column_stack([np.ones(len(self.X_train_)), self.X_train_])
        
        # Weighted least squares: (X^T W X)^{-1} X^T W y
        W = np.diag(weights)
        XTW = X_design.T @ W
        
        try:
            # Solve normal equations
            coeffs = np.linalg.solve(XTW @ X_design, XTW @ self.y_train_)
            
            # Predict for query point
            query_design = np.concatenate([[1], query_point])
            return query_design @ coeffs
            
        except np.linalg.LinAlgError:
            # Fallback to weighted average if singular
            return np.average(self.y_train_, weights=weights)


# Nearest Centroid Classifier

class NearestCentroid:
    """Nearest Centroid (Rocchio) Classifier."""
    
    def __init__(self, distance_metric: DistanceMetric = None, shrink_threshold: Optional[float] = None):
        self.distance_metric = distance_metric or EuclideanDistance()
        self.shrink_threshold = shrink_threshold
        
        # Fitted attributes
        self.centroids_ = None
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the nearest centroid classifier."""
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.centroids_ = np.zeros((n_classes, n_features))
        
        # Compute centroids for each class
        for i, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            self.centroids_[i] = np.mean(X[class_mask], axis=0)
        
        # Apply shrinkage if specified
        if self.shrink_threshold is not None:
            self._apply_shrinkage(X, y)
        
        return self
    
    def _apply_shrinkage(self, X: np.ndarray, y: np.ndarray):
        """Apply nearest shrunken centroids."""
        # Compute overall centroid
        overall_centroid = np.mean(X, axis=0)
        
        # Compute within-class standard deviations
        pooled_std = np.zeros(X.shape[1])
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_data = X[class_mask]
            class_std = np.std(class_data, axis=0, ddof=1)
            pooled_std += class_std ** 2
        
        pooled_std = np.sqrt(pooled_std / len(self.classes_))
        
        # Shrink centroids
        for i in range(len(self.classes_)):
            diff = self.centroids_[i] - overall_centroid
            shrunken_diff = np.sign(diff) * np.maximum(
                0, np.abs(diff) - self.shrink_threshold * pooled_std
            )
            self.centroids_[i] = overall_centroid + shrunken_diff
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels by finding nearest centroid."""
        if self.centroids_ is None:
            raise ValueError("Must fit before prediction")
        
        # Compute distances to all centroids
        distances = self.distance_metric.compute_matrix(X, self.centroids_)
        
        # Find nearest centroid for each point
        nearest_indices = np.argmin(distances, axis=1)
        return self.classes_[nearest_indices]
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute distances to centroids."""
        if self.centroids_ is None:
            raise ValueError("Must fit before prediction")
        
        return -self.distance_metric.compute_matrix(X, self.centroids_)


# Learning Vector Quantization

class LearningVectorQuantization:
    """Learning Vector Quantization classifier."""
    
    def __init__(self, n_prototypes_per_class: int = 1, learning_rate: float = 0.01,
                 max_iter: int = 1000, random_state: Optional[int] = None):
        self.n_prototypes_per_class = n_prototypes_per_class
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Fitted attributes
        self.prototypes_ = None
        self.prototype_labels_ = None
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the LVQ classifier."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_prototypes = n_classes * self.n_prototypes_per_class
        
        # Initialize prototypes
        self.prototypes_ = np.zeros((n_prototypes, n_features))
        self.prototype_labels_ = np.zeros(n_prototypes, dtype=int)
        
        # Initialize prototypes from class centroids with noise
        prototype_idx = 0
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_data = X[class_mask]
            class_centroid = np.mean(class_data, axis=0)
            
            for _ in range(self.n_prototypes_per_class):
                # Add small random noise to centroid
                noise = 0.1 * np.random.randn(n_features)
                self.prototypes_[prototype_idx] = class_centroid + noise
                self.prototype_labels_[prototype_idx] = class_label
                prototype_idx += 1
        
        # Training loop
        for iteration in range(self.max_iter):
            # Randomly select training sample
            idx = np.random.randint(0, len(X))
            x_sample = X[idx]
            y_sample = y[idx]
            
            # Find closest prototype
            distances = np.array([
                np.linalg.norm(x_sample - prototype)
                for prototype in self.prototypes_
            ])
            closest_prototype_idx = np.argmin(distances)
            
            # Update closest prototype
            closest_prototype = self.prototypes_[closest_prototype_idx]
            closest_label = self.prototype_labels_[closest_prototype_idx]
            
            if closest_label == y_sample:
                # Move prototype closer to sample (attraction)
                self.prototypes_[closest_prototype_idx] += self.learning_rate * (x_sample - closest_prototype)
            else:
                # Move prototype away from sample (repulsion)
                self.prototypes_[closest_prototype_idx] -= self.learning_rate * (x_sample - closest_prototype)
            
            # Decay learning rate
            self.learning_rate *= 0.9999
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using nearest prototype."""
        if self.prototypes_ is None:
            raise ValueError("Must fit before prediction")
        
        predictions = np.zeros(len(X), dtype=int)
        
        for i, x in enumerate(X):
            # Find closest prototype
            distances = np.array([
                np.linalg.norm(x - prototype)
                for prototype in self.prototypes_
            ])
            closest_prototype_idx = np.argmin(distances)
            predictions[i] = self.prototype_labels_[closest_prototype_idx]
        
        return predictions


# Utility Functions

def cross_validate_knn(X: np.ndarray, y: np.ndarray, k_values: List[int],
                      cv_folds: int = 5) -> Dict[int, float]:
    """Cross-validate KNN for different k values."""
    n_samples = len(X)
    fold_size = n_samples // cv_folds
    
    results = {}
    
    for k in k_values:
        fold_scores = []
        
        for fold in range(cv_folds):
            # Create train/validation split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else n_samples
            
            val_mask = np.zeros(n_samples, dtype=bool)
            val_mask[start_idx:end_idx] = True
            train_mask = ~val_mask
            
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask], y[val_mask]
            
            # Fit and evaluate KNN
            knn = KNearestNeighbors(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            
            # Compute accuracy
            accuracy = np.mean(y_pred == y_val)
            fold_scores.append(accuracy)
        
        results[k] = np.mean(fold_scores)
    
    return results


def plot_knn_decision_boundary(X: np.ndarray, y: np.ndarray, k: int, 
                              distance_metric: DistanceMetric = None):
    """Plot KNN decision boundary for 2D data."""
    if X.shape[1] != 2:
        raise ValueError("Can only plot decision boundary for 2D data")
    
    knn = KNearestNeighbors(k=k, distance_metric=distance_metric)
    knn.fit(X, y)
    
    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='black')
    plt.colorbar(scatter)
    plt.title(f'KNN Decision Boundary (k={k})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def analyze_curse_of_dimensionality(n_samples: int = 1000, max_dim: int = 20):
    """Analyze how distance becomes less meaningful in high dimensions."""
    dimensions = list(range(1, max_dim + 1))
    distance_ratios = []
    
    for dim in dimensions:
        # Generate random data
        np.random.seed(42)
        X = np.random.randn(n_samples, dim)
        
        # Compute pairwise distances
        distances = EuclideanDistance().compute_matrix(X, X)
        
        # Remove diagonal (distance to self)
        distances = distances[np.triu_indices_from(distances, k=1)]
        
        # Compute ratio of max to min distance
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        ratio = max_dist / min_dist if min_dist > 0 else np.inf
        
        distance_ratios.append(ratio)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, distance_ratios, 'bo-')
    plt.xlabel('Dimension')
    plt.ylabel('Max Distance / Min Distance')
    plt.title('Curse of Dimensionality: Distance Concentration')
    plt.grid(True)
    plt.show()
    
    return dimensions, distance_ratios


def feature_scaling_comparison(X: np.ndarray, y: np.ndarray, k: int = 3):
    """Compare KNN performance with different feature scaling methods."""
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    scalers = {
        'No Scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    results = {}
    
    for name, scaler in scalers.items():
        if scaler is None:
            X_scaled = X
        else:
            X_scaled = scaler.fit_transform(X)
        
        # Cross-validate KNN
        cv_results = cross_validate_knn(X_scaled, y, [k], cv_folds=5)
        results[name] = cv_results[k]
    
    return results


# Export all solution implementations
__all__ = [
    'DistanceMetric', 'EuclideanDistance', 'ManhattanDistance', 'MinkowskiDistance',
    'CosineDistance', 'MahalanobisDistance', 'KNearestNeighbors', 'RadiusNeighbors',
    'LocallyWeightedRegression', 'NearestCentroid', 'LearningVectorQuantization',
    'cross_validate_knn', 'plot_knn_decision_boundary', 'analyze_curse_of_dimensionality',
    'feature_scaling_comparison'
]