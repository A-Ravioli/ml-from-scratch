"""
Common utility functions used across the ML curriculum.

This module provides shared functionality for:
- Mathematical operations
- Visualization helpers
- Numerical utilities
- Testing frameworks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Optional, Union, Dict
import time
from functools import wraps


def timer(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper


def numerical_gradient(f: Callable[[np.ndarray], float], 
                      x: np.ndarray, 
                      eps: float = 1e-8) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute gradient
        eps: Small perturbation for finite differences
        
    Returns:
        Numerical gradient at x
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def check_gradient(f: Callable[[np.ndarray], float],
                  grad_f: Callable[[np.ndarray], np.ndarray],
                  x: np.ndarray,
                  tolerance: float = 1e-5) -> Tuple[bool, float]:
    """
    Check if analytical gradient matches numerical gradient.
    
    Args:
        f: Loss function
        grad_f: Analytical gradient function
        x: Point to check gradient at
        tolerance: Maximum allowed difference
        
    Returns:
        (matches, max_difference)
    """
    analytical = grad_f(x)
    numerical = numerical_gradient(f, x)
    diff = np.max(np.abs(analytical - numerical))
    return diff < tolerance, diff


def generate_synthetic_data(n_samples: int, 
                          n_features: int,
                          noise_level: float = 0.1,
                          task: str = 'regression',
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset for testing ML algorithms.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_level: Standard deviation of Gaussian noise
        task: 'regression' or 'classification'
        random_state: Random seed for reproducibility
        
    Returns:
        (X, y) dataset
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    if task == 'regression':
        # Linear relationship with noise
        true_weights = np.random.randn(n_features)
        y = X @ true_weights + noise_level * np.random.randn(n_samples)
    elif task == 'classification':
        # Binary classification with linear boundary
        true_weights = np.random.randn(n_features)
        logits = X @ true_weights
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(int)
        # Add some label noise
        flip_mask = np.random.random(n_samples) < noise_level
        y[flip_mask] = 1 - y[flip_mask]
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return X, y


def plot_loss_landscape(loss_func: Callable[[np.ndarray], float],
                       center: np.ndarray,
                       range_x: float = 2.0,
                       range_y: float = 2.0,
                       resolution: int = 100,
                       trajectory: Optional[List[np.ndarray]] = None):
    """
    Visualize 2D loss landscape around a point.
    
    Args:
        loss_func: Loss function to visualize
        center: Center point for visualization
        range_x, range_y: Range to visualize in each direction
        resolution: Grid resolution
        trajectory: Optional optimization trajectory to overlay
    """
    if len(center) != 2:
        raise ValueError("Can only visualize 2D loss landscapes")
    
    x = np.linspace(center[0] - range_x, center[0] + range_x, resolution)
    y = np.linspace(center[1] - range_y, center[1] + range_y, resolution)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = np.array([X[i, j], Y[i, j]])
            Z[i, j] = loss_func(point)
    
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    levels = np.percentile(Z.flatten(), np.linspace(0, 100, 20))
    contour = plt.contour(X, Y, Z, levels=levels, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot trajectory if provided
    if trajectory:
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        plt.plot(traj_x, traj_y, 'ro-', markersize=5, linewidth=2, label='Optimization path')
        plt.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start')
        plt.plot(traj_x[-1], traj_y[-1], 'rs', markersize=10, label='End')
        plt.legend()
    
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Loss Landscape')
    plt.colorbar(contour, label='Loss')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def plot_convergence(losses: List[float], 
                    log_scale: bool = False,
                    title: str = "Convergence Plot"):
    """
    Plot convergence of optimization algorithm.
    
    Args:
        losses: List of loss values over iterations
        log_scale: Whether to use log scale for y-axis
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if log_scale and all(l > 0 for l in losses):
        plt.yscale('log')
    
    # Add convergence rate annotation
    if len(losses) > 10:
        recent_losses = losses[-10:]
        if recent_losses[-1] > 1e-10:
            rate = (recent_losses[-1] - recent_losses[0]) / (10 * recent_losses[0])
            plt.text(0.7, 0.9, f'Recent rate: {rate:.2e}/iter', 
                    transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def eigendecomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition with sorted eigenvalues.
    
    Args:
        A: Square matrix
        
    Returns:
        (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
    """
    eigenvals, eigenvecs = np.linalg.eig(A)
    idx = np.argsort(np.abs(eigenvals))[::-1]
    return eigenvals[idx], eigenvecs[:, idx]


def is_positive_definite(A: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if matrix is positive definite."""
    if not np.allclose(A, A.T, atol=tolerance):
        return False
    eigenvals, _ = np.linalg.eig(A)
    return np.all(eigenvals > tolerance)


def condition_number(A: np.ndarray) -> float:
    """Compute condition number of matrix."""
    s = np.linalg.svd(A, compute_uv=False)
    return s[0] / s[-1] if s[-1] > 0 else np.inf


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Soft thresholding operator for L1 regularization.
    
    Args:
        x: Input array
        threshold: Threshold parameter
        
    Returns:
        Soft-thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def create_mini_batches(X: np.ndarray, 
                       y: np.ndarray, 
                       batch_size: int,
                       shuffle: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create mini-batches for stochastic optimization.
    
    Args:
        X: Feature matrix
        y: Labels
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        
    Returns:
        List of (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))
    
    return batches


def one_hot_encode(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    One-hot encode categorical labels.
    
    Args:
        y: Integer labels
        n_classes: Number of classes (inferred if None)
        
    Returns:
        One-hot encoded matrix
    """
    if n_classes is None:
        n_classes = int(np.max(y)) + 1
    
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y.astype(int)] = 1
    
    return one_hot


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        eps: Small value to avoid log(0)
        
    Returns:
        Average cross-entropy loss
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    return np.mean(y_true == y_pred)


def normalize_data(X: np.ndarray, 
                  method: str = 'standardize',
                  return_params: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Normalize data using various methods.
    
    Args:
        X: Data to normalize
        method: 'standardize' (zero mean, unit variance) or 'minmax' (to [0, 1])
        return_params: Whether to return normalization parameters
        
    Returns:
        Normalized data (and parameters if requested)
    """
    if method == 'standardize':
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        X_norm = (X - mean) / std
        params = {'mean': mean, 'std': std}
    elif method == 'minmax':
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        X_norm = (X - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if return_params:
        return X_norm, params
    return X_norm


class EarlyStopping:
    """Early stopping callback for optimization."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.stopped = False
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if should stop.
        
        Args:
            current_value: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.stopped = True
            return True
        
        return False


if __name__ == "__main__":
    # Test utilities
    print("Testing common utility functions")
    
    # Test numerical gradient
    def f(x):
        return x[0]**2 + 2*x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 4*x[1]])
    
    x = np.array([1.0, 2.0])
    matches, diff = check_gradient(f, grad_f, x)
    print(f"Gradient check: {matches}, max diff: {diff}")
    
    # Test data generation
    X, y = generate_synthetic_data(100, 5, task='classification')
    print(f"Generated data shape: X={X.shape}, y={y.shape}")
    
    # Test normalization
    X_norm = normalize_data(X)
    print(f"Normalized data: mean={np.mean(X_norm, axis=0)}, std={np.std(X_norm, axis=0)}")
    
    print("All utilities working correctly!")