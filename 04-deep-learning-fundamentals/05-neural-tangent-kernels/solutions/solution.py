"""
Reference Solutions for Neural Tangent Kernels Exercise

Implementation of Neural Tangent Kernel theory and infinite-width limits.

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple


class NeuralTangentKernel:
    """Neural Tangent Kernel computation for infinite-width networks"""
    
    @staticmethod
    def ntk_relu_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute NTK for ReLU activation at infinite width"""
        # Normalize inputs
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Dot product
        dot_product = np.dot(x1, x2)
        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1, 1)
        
        theta = np.arccos(cos_theta)
        
        # NTK formula for ReLU
        ntk_value = (norm1 * norm2 / (2 * np.pi)) * (np.sin(theta) + (np.pi - theta) * cos_theta)
        
        return ntk_value
    
    @staticmethod
    def ntk_matrix(X: np.ndarray) -> np.ndarray:
        """Compute NTK matrix for dataset"""
        n = X.shape[0]
        K = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                K[i, j] = NeuralTangentKernel.ntk_relu_kernel(X[i], X[j])
        
        return K
    
    @staticmethod
    def finite_width_approximation(X: np.ndarray, width: int, n_samples: int = 1000) -> np.ndarray:
        """Approximate NTK with finite-width networks"""
        n = X.shape[0]
        d = X.shape[1]
        
        K_empirical = np.zeros((n, n))
        
        for sample in range(n_samples):
            # Random weights
            W = np.random.normal(0, 1, (d, width)) / np.sqrt(d)
            
            # Features
            features = np.maximum(0, X @ W)  # ReLU activation
            
            # Contribution to NTK
            K_empirical += (features @ features.T) / width
        
        return K_empirical / n_samples


def demonstrate_ntk():
    """Demonstrate Neural Tangent Kernel properties"""
    print("Neural Tangent Kernel Demonstration")
    print("=" * 50)
    
    # Generate simple 2D data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    
    # 1. Compute theoretical NTK
    print("\n1. Computing theoretical NTK...")
    K_theory = NeuralTangentKernel.ntk_matrix(X)
    print(f"Theoretical NTK matrix shape: {K_theory.shape}")
    print(f"NTK eigenvalues (first 5): {np.linalg.eigvals(K_theory)[:5]}")
    
    # 2. Finite-width approximations
    print("\n2. Finite-width approximations...")
    widths = [10, 100, 1000]
    
    for width in widths:
        K_finite = NeuralTangentKernel.finite_width_approximation(X, width, n_samples=100)
        
        # Compare with theory
        error = np.linalg.norm(K_finite - K_theory, 'fro') / np.linalg.norm(K_theory, 'fro')
        print(f"Width {width}: Relative error = {error:.4f}")
    
    # 3. Visualization
    print("\n3. Visualizing convergence...")
    
    plt.figure(figsize=(12, 4))
    
    # Plot theoretical NTK
    plt.subplot(1, 3, 1)
    plt.imshow(K_theory, cmap='viridis')
    plt.title('Theoretical NTK')
    plt.colorbar()
    
    # Plot finite-width approximation
    K_finite_large = NeuralTangentKernel.finite_width_approximation(X, 1000, n_samples=500)
    plt.subplot(1, 3, 2)
    plt.imshow(K_finite_large, cmap='viridis')
    plt.title('Finite Width (1000)')
    plt.colorbar()
    
    # Plot difference
    plt.subplot(1, 3, 3)
    plt.imshow(K_theory - K_finite_large, cmap='RdBu')
    plt.title('Difference')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstrate_ntk() 