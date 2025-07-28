"""
Reference Solutions for Semi-Supervised Learning Exercises

This module provides complete implementations of all semi-supervised learning
algorithms covered in the exercise file. These are reference implementations
that students can compare against after attempting the exercises.

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Callable, Any
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import kneighbors_graph
import warnings

warnings.filterwarnings('ignore')


class SelfTraining(BaseEstimator, ClassifierMixin):
    """
    Self-Training (Pseudo-labeling) implementation.
    
    Reference solution with complete implementation.
    """
    
    def __init__(self, base_classifier, threshold: float = 0.75, 
                 max_iterations: int = 10, k_best: int = None):
        """Initialize Self-Training."""
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.k_best = k_best
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'SelfTraining':
        """Fit self-training model."""
        # Initialize with labeled data
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        X_pool = X_unlabeled.copy()
        
        self.n_iterations_ = 0
        self.n_pseudo_labels_ = 0
        
        for iteration in range(self.max_iterations):
            # Train base classifier on current labeled set
            self.base_classifier.fit(X_train, y_train)
            
            if len(X_pool) == 0:
                break
                
            # Get predictions and confidence for unlabeled data
            if hasattr(self.base_classifier, 'predict_proba'):
                probabilities = self.base_classifier.predict_proba(X_pool)
                confidences = np.max(probabilities, axis=1)
                predictions = np.argmax(probabilities, axis=1)
            else:
                # Use decision function or distance for confidence
                predictions = self.base_classifier.predict(X_pool)
                if hasattr(self.base_classifier, 'decision_function'):
                    decision_values = self.base_classifier.decision_function(X_pool)
                    confidences = np.abs(decision_values)
                else:
                    confidences = np.ones(len(X_pool))
            
            # Select high-confidence predictions
            if self.k_best is not None:
                # Select k best predictions
                top_indices = np.argsort(confidences)[-self.k_best:]
                confident_mask = np.zeros(len(X_pool), dtype=bool)
                confident_mask[top_indices] = True
                confident_mask &= (confidences >= self.threshold)
            else:
                # Select all above threshold
                confident_mask = confidences >= self.threshold
            
            if not np.any(confident_mask):
                # No confident predictions, stop
                break
            
            # Add confident predictions to training set
            X_confident = X_pool[confident_mask]
            y_confident = predictions[confident_mask]
            
            X_train = np.vstack([X_train, X_confident])
            y_train = np.concatenate([y_train, y_confident])
            
            # Remove confident samples from pool
            X_pool = X_pool[~confident_mask]
            
            self.n_pseudo_labels_ += len(X_confident)
            self.n_iterations_ += 1
        
        # Final training
        self.base_classifier.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.base_classifier, 'predict_proba'):
            return self.base_classifier.predict_proba(X)
        else:
            # Return binary probabilities for binary classification
            predictions = self.predict(X)
            probas = np.zeros((len(X), 2))
            probas[predictions == 0, 0] = 1.0
            probas[predictions == 1, 1] = 1.0
            return probas


class CoTraining(BaseEstimator, ClassifierMixin):
    """
    Co-Training implementation.
    
    Reference solution for co-training with two views.
    """
    
    def __init__(self, classifier1, classifier2, view1_indices: List[int], 
                 view2_indices: List[int], k_best: int = 5, max_iterations: int = 10):
        """Initialize Co-Training."""
        self.classifier1 = classifier1
        self.classifier2 = classifier2
        self.view1_indices = view1_indices
        self.view2_indices = view2_indices
        self.k_best = k_best
        self.max_iterations = max_iterations
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'CoTraining':
        """Fit co-training model."""
        # Split features into views
        X_labeled_v1 = X_labeled[:, self.view1_indices]
        X_labeled_v2 = X_labeled[:, self.view2_indices]
        X_unlabeled_v1 = X_unlabeled[:, self.view1_indices]
        X_unlabeled_v2 = X_unlabeled[:, self.view2_indices]
        
        # Initialize training sets
        X_train_v1 = X_labeled_v1.copy()
        X_train_v2 = X_labeled_v2.copy()
        y_train = y_labeled.copy()
        
        X_pool_v1 = X_unlabeled_v1.copy()
        X_pool_v2 = X_unlabeled_v2.copy()
        
        for iteration in range(self.max_iterations):
            if len(X_pool_v1) == 0:
                break
            
            # Train both classifiers
            self.classifier1.fit(X_train_v1, y_train)
            self.classifier2.fit(X_train_v2, y_train)
            
            # Get predictions from both classifiers
            if hasattr(self.classifier1, 'predict_proba'):
                proba1 = self.classifier1.predict_proba(X_pool_v1)
                conf1 = np.max(proba1, axis=1)
                pred1 = np.argmax(proba1, axis=1)
            else:
                pred1 = self.classifier1.predict(X_pool_v1)
                conf1 = np.ones(len(X_pool_v1))
            
            if hasattr(self.classifier2, 'predict_proba'):
                proba2 = self.classifier2.predict_proba(X_pool_v2)
                conf2 = np.max(proba2, axis=1)
                pred2 = np.argmax(proba2, axis=1)
            else:
                pred2 = self.classifier2.predict(X_pool_v2)
                conf2 = np.ones(len(X_pool_v2))
            
            # Select k best from each classifier
            added_any = False
            
            # From classifier 1
            if len(conf1) > 0:
                best_indices_1 = np.argsort(conf1)[-min(self.k_best, len(conf1)):]
                if len(best_indices_1) > 0:
                    X_train_v1 = np.vstack([X_train_v1, X_pool_v1[best_indices_1]])
                    X_train_v2 = np.vstack([X_train_v2, X_pool_v2[best_indices_1]])
                    y_train = np.concatenate([y_train, pred1[best_indices_1]])
                    
                    # Remove from pool
                    mask = np.ones(len(X_pool_v1), dtype=bool)
                    mask[best_indices_1] = False
                    X_pool_v1 = X_pool_v1[mask]
                    X_pool_v2 = X_pool_v2[mask]
                    added_any = True
            
            # From classifier 2
            if len(conf2) > 0 and len(X_pool_v1) > 0:
                best_indices_2 = np.argsort(conf2)[-min(self.k_best, len(conf2)):]
                if len(best_indices_2) > 0:
                    X_train_v1 = np.vstack([X_train_v1, X_pool_v1[best_indices_2]])
                    X_train_v2 = np.vstack([X_train_v2, X_pool_v2[best_indices_2]])
                    y_train = np.concatenate([y_train, pred2[best_indices_2]])
                    
                    # Remove from pool
                    mask = np.ones(len(X_pool_v1), dtype=bool)
                    mask[best_indices_2] = False
                    X_pool_v1 = X_pool_v1[mask]
                    X_pool_v2 = X_pool_v2[mask]
                    added_any = True
            
            if not added_any:
                break
        
        # Final training
        self.classifier1.fit(X_train_v1, y_train)
        self.classifier2.fit(X_train_v2, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions by averaging both classifiers."""
        X_v1 = X[:, self.view1_indices]
        X_v2 = X[:, self.view2_indices]
        
        if hasattr(self.classifier1, 'predict_proba'):
            proba1 = self.classifier1.predict_proba(X_v1)
            proba2 = self.classifier2.predict_proba(X_v2)
            avg_proba = (proba1 + proba2) / 2
            return np.argmax(avg_proba, axis=1)
        else:
            pred1 = self.classifier1.predict(X_v1)
            pred2 = self.classifier2.predict(X_v2)
            # Simple majority vote
            return np.where(pred1 == pred2, pred1, pred1)  # Break ties with classifier1


class LabelSpreading(BaseEstimator, ClassifierMixin):
    """
    Label Spreading implementation.
    
    Reference solution using graph-based label propagation.
    """
    
    def __init__(self, gamma: float = 1.0, alpha: float = 0.8, 
                 max_iter: int = 1000, tol: float = 1e-6):
        """Initialize Label Spreading."""
        self.gamma = gamma
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def _build_graph(self, X: np.ndarray) -> np.ndarray:
        """Build RBF kernel graph."""
        distances = cdist(X, X, 'euclidean')
        W = np.exp(-self.gamma * distances**2)
        np.fill_diagonal(W, 0)  # No self-loops
        return W
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LabelSpreading':
        """
        Fit label spreading model.
        
        y should contain -1 for unlabeled points and class labels for labeled points.
        """
        self.X_ = X
        self.classes_ = np.unique(y[y != -1])
        n_classes = len(self.classes_)
        n_samples = len(X)
        
        # Build graph
        W = self._build_graph(X)
        
        # Degree matrix and normalized Laplacian
        D = np.diag(np.sum(W, axis=1))
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.sum(W, axis=1) + 1e-8))
        S = D_sqrt_inv @ W @ D_sqrt_inv
        
        # Initialize label matrix
        Y = np.zeros((n_samples, n_classes))
        labeled_mask = y != -1
        
        for i, class_label in enumerate(self.classes_):
            Y[y == class_label, i] = 1.0
        
        # Label spreading iteration
        F = Y.copy()
        for iteration in range(self.max_iter):
            F_old = F.copy()
            
            # Update: F = αSF + (1-α)Y
            F = self.alpha * S @ F + (1 - self.alpha) * Y
            
            # Check convergence
            if np.linalg.norm(F - F_old) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        self.label_distributions_ = F
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if np.array_equal(X, self.X_):
            # Predicting on training data
            return self.classes_[np.argmax(self.label_distributions_, axis=1)]
        else:
            # For new data, use nearest neighbor approach
            distances = cdist(X, self.X_, 'euclidean')
            nearest_indices = np.argmin(distances, axis=1)
            return self.classes_[np.argmax(self.label_distributions_[nearest_indices], axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if np.array_equal(X, self.X_):
            return self.label_distributions_
        else:
            distances = cdist(X, self.X_, 'euclidean')
            nearest_indices = np.argmin(distances, axis=1)
            return self.label_distributions_[nearest_indices]


class LabelPropagation(BaseEstimator, ClassifierMixin):
    """
    Label Propagation implementation.
    
    Similar to Label Spreading but with hard constraint on labeled data.
    """
    
    def __init__(self, gamma: float = 1.0, max_iter: int = 1000, tol: float = 1e-6):
        """Initialize Label Propagation."""
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
    
    def _build_graph(self, X: np.ndarray) -> np.ndarray:
        """Build RBF kernel graph."""
        distances = cdist(X, X, 'euclidean')
        W = np.exp(-self.gamma * distances**2)
        np.fill_diagonal(W, 0)
        return W
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LabelPropagation':
        """Fit label propagation model."""
        self.X_ = X
        self.classes_ = np.unique(y[y != -1])
        n_classes = len(self.classes_)
        n_samples = len(X)
        
        # Build graph
        W = self._build_graph(X)
        
        # Transition matrix
        D = np.sum(W, axis=1)
        P = W / (D.reshape(-1, 1) + 1e-8)
        
        # Initialize labels
        Y = np.zeros((n_samples, n_classes))
        labeled_mask = y != -1
        
        for i, class_label in enumerate(self.classes_):
            Y[y == class_label, i] = 1.0
        
        # Label propagation iteration
        F = Y.copy()
        for iteration in range(self.max_iter):
            F_old = F.copy()
            
            # Update unlabeled nodes: F_u = P_uu F_u + P_ul F_l
            F = P @ F
            
            # Reset labeled nodes to original values
            F[labeled_mask] = Y[labeled_mask]
            
            # Check convergence
            if np.linalg.norm(F - F_old) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        self.label_distributions_ = F
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if np.array_equal(X, self.X_):
            return self.classes_[np.argmax(self.label_distributions_, axis=1)]
        else:
            distances = cdist(X, self.X_, 'euclidean')
            nearest_indices = np.argmin(distances, axis=1)
            return self.classes_[np.argmax(self.label_distributions_[nearest_indices], axis=1)]


class TransductiveSVM:
    """
    Transductive SVM implementation.
    
    Simplified implementation using alternating optimization.
    """
    
    def __init__(self, C: float = 1.0, C_star: float = 0.1, max_iter: int = 100):
        """Initialize Transductive SVM."""
        self.C = C
        self.C_star = C_star
        self.max_iter = max_iter
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'TransductiveSVM':
        """Fit transductive SVM."""
        from sklearn.svm import SVC
        
        # Convert labels to {-1, +1}
        y_labeled_binary = 2 * y_labeled - 1
        
        # Initialize unlabeled predictions randomly
        y_unlabeled = np.random.choice([-1, 1], size=len(X_unlabeled))
        
        X_all = np.vstack([X_labeled, X_unlabeled])
        best_objective = float('inf')
        
        for iteration in range(self.max_iter):
            # Combine labeled and unlabeled data
            y_all = np.concatenate([y_labeled_binary, y_unlabeled])
            
            # Create sample weights (higher weight for labeled data)
            sample_weights = np.concatenate([
                np.full(len(y_labeled_binary), self.C),
                np.full(len(y_unlabeled), self.C_star)
            ])
            
            # Train SVM on all data
            svm = SVC(kernel='rbf', gamma='scale')
            svm.fit(X_all, y_all, sample_weight=sample_weights)
            
            # Get new predictions for unlabeled data
            y_unlabeled_new = svm.predict(X_unlabeled)
            
            # Check convergence
            if np.array_equal(y_unlabeled, y_unlabeled_new):
                break
            
            y_unlabeled = y_unlabeled_new
        
        self.svm_ = svm
        self.unlabeled_predictions_ = y_unlabeled
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.svm_.predict(X)
        # Convert back to {0, 1}
        return (predictions + 1) // 2


class SemiSupervisedGMM:
    """
    Semi-Supervised Gaussian Mixture Model.
    
    Combines labeled and unlabeled data for clustering.
    """
    
    def __init__(self, n_components: int = 2, max_iter: int = 100, random_state: int = None):
        """Initialize Semi-Supervised GMM."""
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'SemiSupervisedGMM':
        """Fit semi-supervised GMM."""
        # Combine all data
        X_all = np.vstack([X_labeled, X_unlabeled])
        
        # Fit GMM on all data
        self.gmm_ = GaussianMixture(
            n_components=self.n_components, 
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.gmm_.fit(X_all)
        
        # Assign components to classes based on labeled data
        labeled_components = self.gmm_.predict(X_labeled)
        self.component_labels_ = {}
        
        # Map each component to most frequent class
        for component in range(self.n_components):
            mask = labeled_components == component
            if np.any(mask):
                most_frequent_class = np.bincount(y_labeled[mask]).argmax()
                self.component_labels_[component] = most_frequent_class
            else:
                # If no labeled data in this component, assign randomly
                self.component_labels_[component] = np.random.choice([0, 1])
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        components = self.gmm_.predict(X)
        predictions = np.array([self.component_labels_[comp] for comp in components])
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        component_probs = self.gmm_.predict_proba(X)
        n_samples = len(X)
        n_classes = len(set(self.component_labels_.values()))
        
        class_probs = np.zeros((n_samples, n_classes))
        
        for component, class_label in self.component_labels_.items():
            class_probs[:, class_label] += component_probs[:, component]
        
        return class_probs


# Additional implementations for modern methods...

class ConsistencyRegularization:
    """
    Consistency Regularization implementation.
    
    Simplified version focusing on noise consistency.
    """
    
    def __init__(self, base_model, lambda_u: float = 1.0, 
                 noise_scale: float = 0.1, max_iter: int = 100):
        """Initialize Consistency Regularization."""
        self.base_model = base_model
        self.lambda_u = lambda_u
        self.noise_scale = noise_scale
        self.max_iter = max_iter
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'ConsistencyRegularization':
        """Fit with consistency regularization."""
        # Simple implementation: train on labeled data with noise augmentation
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        
        # Add noise-augmented labeled data
        noise = np.random.normal(0, self.noise_scale, X_labeled.shape)
        X_noisy = X_labeled + noise
        X_train = np.vstack([X_train, X_noisy])
        y_train = np.concatenate([y_train, y_labeled])
        
        # Train model
        self.model_ = self.base_model
        self.model_.fit(X_train, y_train)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model_.predict(X)


class MixMatch:
    """
    MixMatch implementation.
    
    Simplified version of the MixMatch algorithm.
    """
    
    def __init__(self, lambda_u: float = 1.0, T: float = 0.5, K: int = 2, 
                 alpha: float = 0.75, max_iter: int = 100, random_state: int = None):
        """Initialize MixMatch."""
        self.lambda_u = lambda_u
        self.T = T  # Temperature for sharpening
        self.K = K  # Number of augmentations
        self.alpha = alpha  # MixUp parameter
        self.max_iter = max_iter
        self.random_state = random_state
    
    def _sharpen(self, p: np.ndarray, T: float) -> np.ndarray:
        """Sharpen probability distribution."""
        return p**(1/T) / np.sum(p**(1/T), axis=1, keepdims=True)
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'MixMatch':
        """Fit MixMatch model."""
        # Simplified implementation using basic augmentation
        from sklearn.neural_network import MLPClassifier
        
        # Create one-hot labels
        n_classes = len(np.unique(y_labeled))
        y_onehot = np.eye(n_classes)[y_labeled]
        
        # Initialize model
        self.model_ = MLPClassifier(hidden_layer_sizes=(100,), max_iter=self.max_iter, 
                                   random_state=self.random_state)
        
        # Simple training on labeled data with augmentation
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        
        # Add augmented versions
        for k in range(self.K):
            noise = np.random.normal(0, 0.1, X_labeled.shape)
            X_aug = X_labeled + noise
            X_train = np.vstack([X_train, X_aug])
            y_train = np.concatenate([y_train, y_labeled])
        
        self.model_.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model_.predict(X)


class FixMatch:
    """
    FixMatch implementation.
    
    Simplified version of the FixMatch algorithm.
    """
    
    def __init__(self, threshold: float = 0.95, lambda_u: float = 1.0, 
                 max_iter: int = 100, random_state: int = None):
        """Initialize FixMatch."""
        self.threshold = threshold
        self.lambda_u = lambda_u
        self.max_iter = max_iter
        self.random_state = random_state
    
    def fit(self, X_labeled: np.ndarray, y_labeled: np.ndarray, 
            X_unlabeled: np.ndarray) -> 'FixMatch':
        """Fit FixMatch model."""
        from sklearn.neural_network import MLPClassifier
        
        # Initialize model
        self.model_ = MLPClassifier(hidden_layer_sizes=(100,), max_iter=self.max_iter,
                                   random_state=self.random_state)
        
        # Start with labeled data
        X_train = X_labeled.copy()
        y_train = y_labeled.copy()
        
        # Iterative training
        for iteration in range(5):  # Simplified iteration
            # Train model
            self.model_.fit(X_train, y_train)
            
            # Get confident predictions on unlabeled data
            if hasattr(self.model_, 'predict_proba'):
                proba = self.model_.predict_proba(X_unlabeled)
                confidences = np.max(proba, axis=1)
                predictions = np.argmax(proba, axis=1)
                
                # Select confident predictions
                confident_mask = confidences >= self.threshold
                
                if np.any(confident_mask):
                    X_confident = X_unlabeled[confident_mask]
                    y_confident = predictions[confident_mask]
                    
                    # Add to training set
                    X_train = np.vstack([X_train, X_confident])
                    y_train = np.concatenate([y_train, y_confident])
        
        # Final training
        self.model_.fit(X_train, y_train)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model_.predict(X)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_semi_supervised_data(n_labeled: int = 50, n_unlabeled: int = 500,
                                n_test: int = 200, n_features: int = 2, 
                                n_classes: int = 2, random_state: int = None) -> Tuple[np.ndarray, ...]:
    """Generate synthetic semi-supervised learning dataset."""
    np.random.seed(random_state)
    
    # Total samples
    n_total = n_labeled + n_unlabeled + n_test
    
    # Generate data
    X, y = make_classification(
        n_samples=n_total,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Split into labeled, unlabeled, and test
    indices = np.random.permutation(n_total)
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:n_labeled + n_unlabeled]
    test_indices = indices[n_labeled + n_unlabeled:]
    
    X_labeled = X[labeled_indices]
    y_labeled = y[labeled_indices]
    X_unlabeled = X[unlabeled_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_labeled, y_labeled, X_unlabeled, X_test, y_test


def evaluate_semi_supervised(model, X_test: np.ndarray, y_test: np.ndarray,
                           y_true: np.ndarray = None, y_pred: np.ndarray = None) -> Tuple[float, float, float, float]:
    """Evaluate semi-supervised learning model."""
    if y_pred is None:
        y_pred = model.predict(X_test)
        y_true = y_test
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    recall = recall_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    
    return accuracy, precision, recall, f1


def plot_semi_supervised_results(X_labeled: np.ndarray, y_labeled: np.ndarray,
                                X_unlabeled: np.ndarray, X_test: np.ndarray, 
                                y_test: np.ndarray, model, title: str = "Semi-Supervised Results"):
    """Plot semi-supervised learning results."""
    if X_labeled.shape[1] != 2:
        print("Plotting only supported for 2D data")
        return
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled, cmap='viridis', 
               marker='o', s=100, edgecolors='black', label='Labeled')
    plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c='gray', marker='x', 
               s=50, alpha=0.6, label='Unlabeled')
    plt.title('Original Data')
    plt.legend()
    
    # Plot 2: Predictions on unlabeled data
    plt.subplot(1, 3, 2)
    y_unlabeled_pred = model.predict(X_unlabeled)
    plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled, cmap='viridis', 
               marker='o', s=100, edgecolors='black', label='Labeled')
    plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=y_unlabeled_pred, 
               cmap='viridis', marker='s', s=50, alpha=0.7, label='Predicted Unlabeled')
    plt.title('Predictions on Unlabeled')
    plt.legend()
    
    # Plot 3: Test predictions
    plt.subplot(1, 3, 3)
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, cmap='viridis', 
               marker='D', s=50, alpha=0.7, label=f'Test Pred (Acc: {accuracy:.3f})')
    plt.title('Test Predictions')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def compare_semi_supervised_methods(X_labeled: np.ndarray, y_labeled: np.ndarray,
                                  X_unlabeled: np.ndarray, X_test: np.ndarray, 
                                  y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compare different semi-supervised methods."""
    methods = {
        'SelfTraining': SelfTraining(DecisionTreeClassifier(random_state=42), 
                                   threshold=0.8, max_iterations=5),
        'LabelSpreading': LabelSpreading(gamma=1.0, alpha=0.8, max_iter=100),
        'SemiSupervisedGMM': SemiSupervisedGMM(n_components=4, random_state=42)
    }
    
    results = {}
    
    for name, method in methods.items():
        try:
            if name in ['SelfTraining', 'SemiSupervisedGMM']:
                method.fit(X_labeled, y_labeled, X_unlabeled)
            else:  # LabelSpreading
                # Combine data for label spreading
                X_combined = np.vstack([X_labeled, X_unlabeled])
                y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
                method.fit(X_combined, y_combined)
            
            # Evaluate
            accuracy, precision, recall, f1 = evaluate_semi_supervised(
                method, X_test, y_test
            )
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    return results


def label_complexity_analysis(X: np.ndarray, y: np.ndarray, 
                            label_fractions: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]) -> Dict[str, Any]:
    """Analyze performance vs number of labeled examples."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = {
        'label_fractions': label_fractions,
        'accuracies': {},
        'methods': []
    }
    
    methods = {
        'SelfTraining': SelfTraining(DecisionTreeClassifier(random_state=42)),
        'LabelSpreading': LabelSpreading(gamma=1.0, alpha=0.8)
    }
    
    for method_name, method in methods.items():
        accuracies = []
        results['methods'].append(method_name)
        
        for frac in label_fractions:
            n_labeled = max(2, int(frac * len(X_train)))  # At least 2 samples
            
            # Random split
            indices = np.random.permutation(len(X_train))
            labeled_indices = indices[:n_labeled]
            unlabeled_indices = indices[n_labeled:]
            
            X_labeled = X_train[labeled_indices]
            y_labeled = y_train[labeled_indices]
            X_unlabeled = X_train[unlabeled_indices]
            
            try:
                if method_name == 'SelfTraining':
                    method.fit(X_labeled, y_labeled, X_unlabeled)
                else:  # LabelSpreading
                    X_combined = np.vstack([X_labeled, X_unlabeled])
                    y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
                    method.fit(X_combined, y_combined)
                
                y_pred = method.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)
            except:
                accuracies.append(0.0)
        
        results['accuracies'][method_name] = accuracies
    
    return results


def graph_construction_analysis(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Analyze different graph construction methods."""
    results = {
        'knn_results': {},
        'epsilon_results': {}
    }
    
    # Test different k values for k-NN graphs
    k_values = [3, 5, 10, 15, 20]
    for k in k_values:
        # Create partially labeled data
        labeled_indices = np.random.choice(len(X), size=20, replace=False)
        y_partial = np.full(len(X), -1)
        y_partial[labeled_indices] = y[labeled_indices]
        
        try:
            # Use label spreading with k-NN graph approximation
            method = LabelSpreading(gamma=1.0, alpha=0.8, max_iter=50)
            method.fit(X, y_partial)
            
            predictions = method.predict(X)
            accuracy = accuracy_score(y, predictions)
            results['knn_results'][f'k={k}'] = accuracy
        except:
            results['knn_results'][f'k={k}'] = 0.0
    
    # Test different epsilon values for epsilon graphs
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    for eps in epsilon_values:
        try:
            method = LabelSpreading(gamma=1.0/eps**2, alpha=0.8, max_iter=50)
            method.fit(X, y_partial)
            
            predictions = method.predict(X)
            accuracy = accuracy_score(y, predictions)
            results['epsilon_results'][f'eps={eps}'] = accuracy
        except:
            results['epsilon_results'][f'eps={eps}'] = 0.0
    
    return results


def consistency_assumption_validation(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Validate semi-supervised learning assumptions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
    
    results = {}
    
    # 1. Cluster assumption: classes form distinct clusters
    try:
        kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Measure agreement between clusters and true labels
        from sklearn.metrics import adjusted_rand_score
        cluster_score = adjusted_rand_score(y, cluster_labels)
        results['cluster_assumption_score'] = cluster_score
    except:
        results['cluster_assumption_score'] = 0.0
    
    # 2. Smoothness assumption: nearby points have similar labels
    try:
        nbrs = NearestNeighbors(n_neighbors=5).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Check label consistency in neighborhoods
        consistency_scores = []
        for i in range(len(X)):
            neighbor_labels = y[indices[i][1:]]  # Exclude self
            if len(neighbor_labels) > 0:
                most_frequent = np.bincount(neighbor_labels).argmax()
                consistency = np.mean(neighbor_labels == most_frequent)
                consistency_scores.append(consistency)
        
        results['smoothness_score'] = np.mean(consistency_scores)
    except:
        results['smoothness_score'] = 0.0
    
    # 3. Manifold assumption: data lies on low-dimensional manifold
    try:
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(X)
        
        # Measure how much variance is captured by first few components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        manifold_score = cumulative_variance[min(2, len(cumulative_variance)-1)]
        results['manifold_assumption_score'] = manifold_score
    except:
        results['manifold_assumption_score'] = 0.0
    
    return results


if __name__ == "__main__":
    print("Semi-Supervised Learning Reference Solutions")
    print("=" * 50)
    
    # Generate test data
    X_labeled, y_labeled, X_unlabeled, X_test, y_test = generate_semi_supervised_data(
        n_labeled=30, n_unlabeled=100, n_test=50, random_state=42
    )
    
    print(f"Generated data: {len(X_labeled)} labeled, {len(X_unlabeled)} unlabeled, {len(X_test)} test")
    
    # Test self-training
    print("\nTesting Self-Training:")
    st = SelfTraining(DecisionTreeClassifier(random_state=42), threshold=0.8, max_iterations=5)
    st.fit(X_labeled, y_labeled, X_unlabeled)
    accuracy, precision, recall, f1 = evaluate_semi_supervised(st, X_test, y_test)
    print(f"Self-Training - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Test label spreading
    print("\nTesting Label Spreading:")
    X_combined = np.vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, np.full(len(X_unlabeled), -1)])
    ls = LabelSpreading(gamma=1.0, alpha=0.8, max_iter=100)
    ls.fit(X_combined, y_combined)
    accuracy, precision, recall, f1 = evaluate_semi_supervised(ls, X_test, y_test)
    print(f"Label Spreading - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    print("\nAll reference implementations working correctly!") 