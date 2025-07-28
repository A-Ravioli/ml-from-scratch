"""
Reference Solutions for Initialization Theory Exercise

Complete implementation of various initialization schemes with theoretical analysis.

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


def xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot uniform initialization"""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def xavier_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier/Glorot normal initialization"""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, (fan_in, fan_out))


def he_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """He uniform initialization (for ReLU)"""
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def he_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """He normal initialization (for ReLU)"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, (fan_in, fan_out))


def lecun_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    """LeCun uniform initialization"""
    limit = np.sqrt(3.0 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def lecun_normal(fan_in: int, fan_out: int) -> np.ndarray:
    """LeCun normal initialization"""
    std = np.sqrt(1.0 / fan_in)
    return np.random.normal(0, std, (fan_in, fan_out))


class InitializationAnalyzer:
    """Analyze effects of different initialization schemes"""
    
    @staticmethod
    def analyze_variance_preservation(init_fn, activation, n_layers: int = 10, 
                                    layer_size: int = 100) -> Dict:
        """Analyze how initialization affects variance through layers"""
        np.random.seed(42)
        
        # Input
        x = np.random.randn(1000, layer_size)
        current_output = x
        
        variances = [np.var(current_output)]
        
        for layer in range(n_layers):
            # Initialize weights
            weights = init_fn(layer_size, layer_size)
            
            # Forward pass
            current_output = current_output @ weights
            
            # Apply activation
            if activation == 'sigmoid':
                current_output = 1.0 / (1.0 + np.exp(-np.clip(current_output, -500, 500)))
            elif activation == 'tanh':
                current_output = np.tanh(current_output)
            elif activation == 'relu':
                current_output = np.maximum(0, current_output)
            
            variances.append(np.var(current_output))
        
        return {
            'variances': variances,
            'mean_variance': np.mean(variances[1:]),
            'variance_decay': variances[-1] / variances[0]
        }
    
    @staticmethod
    def gradient_flow_analysis(init_fn, n_layers: int = 5) -> Dict:
        """Analyze gradient flow with different initializations"""
        np.random.seed(42)
        
        # Simple network setup
        layer_size = 50
        batch_size = 32
        
        # Forward pass data
        x = np.random.randn(batch_size, layer_size)
        y_true = np.random.randn(batch_size, layer_size)
        
        # Initialize network
        weights = []
        activations = []
        
        current = x
        for layer in range(n_layers):
            w = init_fn(layer_size, layer_size)
            weights.append(w)
            
            # Forward pass
            z = current @ w
            a = np.tanh(z)  # Use tanh activation
            
            activations.append((z, a))
            current = a
        
        # Compute loss (MSE)
        loss = 0.5 * np.mean((current - y_true)**2)
        
        # Backward pass
        grad_output = (current - y_true) / batch_size
        gradient_norms = []
        
        for layer in reversed(range(n_layers)):
            z, a = activations[layer]
            
            # Gradient through activation
            grad_z = grad_output * (1 - np.tanh(z)**2)
            
            # Gradient w.r.t weights
            if layer == 0:
                grad_w = x.T @ grad_z
            else:
                grad_w = activations[layer-1][1].T @ grad_z
            
            gradient_norms.append(np.linalg.norm(grad_w))
            
            # Gradient w.r.t input for next layer
            grad_output = grad_z @ weights[layer].T
        
        return {
            'gradient_norms': list(reversed(gradient_norms)),
            'total_gradient_norm': np.sum(gradient_norms)
        }


def compare_initializations():
    """Compare different initialization schemes"""
    print("Initialization Theory Analysis")
    print("=" * 50)
    
    # Define initialization schemes
    init_schemes = {
        'Xavier Uniform': xavier_uniform,
        'Xavier Normal': xavier_normal,
        'He Uniform': he_uniform,
        'He Normal': he_normal,
        'LeCun Normal': lecun_normal
    }
    
    activations = ['sigmoid', 'tanh', 'relu']
    
    # 1. Variance preservation analysis
    print("\n1. Variance preservation analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, activation in enumerate(activations):
        ax = axes[i]
        
        for name, init_fn in init_schemes.items():
            if activation == 'relu' and 'Xavier' in name:
                continue  # Skip Xavier for ReLU
            if activation != 'relu' and 'He' in name:
                continue  # Skip He for non-ReLU
            
            result = InitializationAnalyzer.analyze_variance_preservation(
                init_fn, activation, n_layers=10
            )
            
            ax.plot(result['variances'], label=name, marker='o', markersize=4)
        
        ax.set_title(f'Variance Flow - {activation.upper()}')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Gradient flow analysis
    print("\n2. Gradient flow analysis...")
    
    plt.figure(figsize=(12, 8))
    
    for i, (name, init_fn) in enumerate(init_schemes.items()):
        result = InitializationAnalyzer.gradient_flow_analysis(init_fn, n_layers=5)
        
        plt.subplot(2, 3, i+1)
        plt.plot(result['gradient_norms'], 'o-')
        plt.title(f'{name}\nTotal Norm: {result["total_gradient_norm"]:.3f}')
        plt.xlabel('Layer (output to input)')
        plt.ylabel('Gradient Norm')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Practical recommendations
    print("\n3. Initialization recommendations:")
    print("- Sigmoid/Tanh: Use Xavier initialization")
    print("- ReLU/Leaky ReLU: Use He initialization")  
    print("- Linear layers: Use LeCun initialization")
    print("- Xavier preserves variance for symmetric activations")
    print("- He accounts for ReLU's non-linearity")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    compare_initializations() 