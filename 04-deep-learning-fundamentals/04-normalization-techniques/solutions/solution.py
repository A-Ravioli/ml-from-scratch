"""
Reference Solutions for Normalization Techniques Exercise

Complete implementation of batch normalization, layer normalization, and others.

Author: ML-from-Scratch Course
"""

import numpy as np
from typing import Tuple, Optional


class BatchNormalization:
    """Batch Normalization implementation"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Training mode
        self.training = True
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass with caching for backprop"""
        if self.training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.eps)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Cache for backward pass
            cache = (x, x_normalized, batch_mean, batch_var)
        else:
            # Use running statistics
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            cache = None
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        return out, cache
    
    def backward(self, dout: np.ndarray, cache: Tuple) -> np.ndarray:
        """Backward pass"""
        x, x_normalized, batch_mean, batch_var = cache
        N = x.shape[0]
        
        # Gradients w.r.t. gamma and beta
        dgamma = np.sum(dout * x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient w.r.t. normalized input
        dx_normalized = dout * self.gamma
        
        # Gradient w.r.t. variance
        dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (batch_var + self.eps)**(-1.5), axis=0)
        
        # Gradient w.r.t. mean
        dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + self.eps), axis=0) + \
                dvar * np.sum(-2 * (x - batch_mean), axis=0) / N
        
        # Gradient w.r.t. input
        dx = dx_normalized / np.sqrt(batch_var + self.eps) + \
             dvar * 2 * (x - batch_mean) / N + \
             dmean / N
        
        # Store gradients
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class LayerNormalization:
    """Layer Normalization implementation"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """Forward pass"""
        # Compute statistics along feature dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        # Cache for backward pass
        cache = (x, x_normalized, mean, var)
        return out, cache
    
    def backward(self, dout: np.ndarray, cache: Tuple) -> np.ndarray:
        """Backward pass"""
        x, x_normalized, mean, var = cache
        
        # Gradients w.r.t. gamma and beta
        dgamma = np.sum(dout * x_normalized, axis=0)
        dbeta = np.sum(dout, axis=0)
        
        # Gradient w.r.t. input
        N = x.shape[-1]
        dx_normalized = dout * self.gamma
        
        dvar = np.sum(dx_normalized * (x - mean) * -0.5 * (var + self.eps)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True) + \
                dvar * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / N
        
        dx = dx_normalized / np.sqrt(var + self.eps) + \
             dvar * 2 * (x - mean) / N + \
             dmean / N
        
        # Store gradients
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


def demonstrate_normalization():
    """Demonstrate normalization techniques"""
    print("Normalization Techniques Demonstration")
    print("=" * 50)
    
    # Test data
    np.random.seed(42)
    x = np.random.randn(32, 10) * 5 + 2  # Batch of 32, features of 10
    
    # 1. Batch Normalization
    print("\n1. Batch Normalization")
    bn = BatchNormalization(10)
    bn.training = True
    
    out_bn, cache_bn = bn.forward(x)
    print(f"Input mean: {np.mean(x):.3f}, std: {np.std(x):.3f}")
    print(f"BN output mean: {np.mean(out_bn):.3f}, std: {np.std(out_bn):.3f}")
    
    # 2. Layer Normalization
    print("\n2. Layer Normalization")
    ln = LayerNormalization(10)
    
    out_ln, cache_ln = ln.forward(x)
    print(f"LN output mean: {np.mean(out_ln):.3f}, std: {np.std(out_ln):.3f}")
    
    # 3. Compare statistics
    print("\n3. Comparing normalizations...")
    print("Batch Norm - per-feature normalization across batch")
    print("Layer Norm - per-sample normalization across features")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstrate_normalization() 