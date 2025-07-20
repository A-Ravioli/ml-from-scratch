"""
Backpropagation Calculus Implementation Exercises

This module implements automatic differentiation and backpropagation from scratch:
- Basic automatic differentiation framework
- Forward-mode and reverse-mode AD
- Neural network backpropagation
- Gradient checking utilities
- Computational graph visualization

Each implementation focuses on mathematical understanding over efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

class Tensor:
    """
    Basic tensor class with automatic differentiation support.
    
    This is a simplified version of tensors in modern frameworks like PyTorch.
    """
    
    def __init__(self, 
                 data: np.ndarray, 
                 requires_grad: bool = False,
                 grad_fn: Optional['Function'] = None):
        """
        Initialize tensor.
        
        TODO: 
        1. Store data as numpy array
        2. Initialize gradient to None
        3. Store gradient function for backpropagation
        4. Set requires_grad flag
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: Optional[np.ndarray] = None) -> None:
        """
        Perform backward pass to compute gradients.
        
        TODO:
        1. If grad_output is None, initialize to ones (for scalar output)
        2. If grad_fn exists, call it to propagate gradients
        3. Accumulate gradient in self.grad
        """
        # YOUR CODE HERE
        pass
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.grad = None
    
    # Arithmetic operations with gradient tracking
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Addition with gradient tracking."""
        # YOUR CODE HERE - implement addition and set grad_fn
        pass
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Element-wise multiplication with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def sum(self, axis: Optional[int] = None) -> 'Tensor':
        """Sum with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def mean(self, axis: Optional[int] = None) -> 'Tensor':
        """Mean with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def exp(self) -> 'Tensor':
        """Exponential with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def log(self) -> 'Tensor':
        """Natural logarithm with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def relu(self) -> 'Tensor':
        """ReLU activation with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def sigmoid(self) -> 'Tensor':
        """Sigmoid activation with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    def tanh(self) -> 'Tensor':
        """Tanh activation with gradient tracking."""
        # YOUR CODE HERE
        pass
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape."""
        return self.data.shape
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data

class Function(ABC):
    """Abstract base class for differentiable functions."""
    
    @abstractmethod
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """Forward pass computation."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Backward pass gradient computation."""
        pass

class AddFunction(Function):
    """Addition function for automatic differentiation."""
    
    def __init__(self, input1: Tensor, input2: Tensor):
        """
        TODO: Store input tensors for backward pass
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """
        Forward pass for addition.
        
        TODO: Implement a + b
        """
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Backward pass for addition.
        
        TODO: 
        1. Gradient of addition is identity for both inputs
        2. Handle broadcasting if shapes don't match
        3. Return gradients for both inputs
        """
        # YOUR CODE HERE
        pass

class MulFunction(Function):
    """Element-wise multiplication function."""
    
    def __init__(self, input1: Tensor, input2: Tensor):
        """TODO: Store inputs for backward pass"""
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """TODO: Implement element-wise multiplication"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        TODO: Backward pass for element-wise multiplication
        - Gradient w.r.t. input1: grad_output * input2
        - Gradient w.r.t. input2: grad_output * input1
        """
        # YOUR CODE HERE
        pass

class MatMulFunction(Function):
    """Matrix multiplication function."""
    
    def __init__(self, input1: Tensor, input2: Tensor):
        """TODO: Store inputs"""
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """TODO: Implement matrix multiplication"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        TODO: Backward pass for matrix multiplication
        For C = AB:
        - dL/dA = dL/dC @ B^T
        - dL/dB = A^T @ dL/dC
        """
        # YOUR CODE HERE
        pass

class SumFunction(Function):
    """Sum reduction function."""
    
    def __init__(self, input_tensor: Tensor, axis: Optional[int] = None):
        """TODO: Store input and axis"""
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """TODO: Implement sum reduction"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        TODO: Backward pass for sum
        - Sum gradient is broadcasted to input shape
        - Use np.broadcast_to to expand gradient
        """
        # YOUR CODE HERE
        pass

class ReLUFunction(Function):
    """ReLU activation function."""
    
    def __init__(self, input_tensor: Tensor):
        """TODO: Store input for backward pass"""
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """TODO: Implement ReLU: max(0, x)"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        TODO: Backward pass for ReLU
        - Gradient is 1 where input > 0, 0 otherwise
        - Use self.input.data to access stored input
        """
        # YOUR CODE HERE
        pass

class SigmoidFunction(Function):
    """Sigmoid activation function."""
    
    def __init__(self, input_tensor: Tensor):
        """TODO: Store input and computed output"""
        # YOUR CODE HERE
        pass
    
    def forward(self, *inputs: Tensor) -> np.ndarray:
        """TODO: Implement sigmoid: 1 / (1 + exp(-x))"""
        # YOUR CODE HERE
        pass
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        TODO: Backward pass for sigmoid
        - Gradient: sigmoid(x) * (1 - sigmoid(x))
        - Use stored output to avoid recomputation
        """
        # YOUR CODE HERE
        pass

class LinearLayer:
    """
    Linear/Dense layer for neural networks.
    
    Implements: y = xW + b
    """
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize linear layer.
        
        TODO:
        1. Initialize weights with proper scaling
        2. Initialize biases to zero
        3. Set requires_grad=True for parameters
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through linear layer.
        
        TODO: Implement y = xW + b using tensor operations
        """
        # YOUR CODE HERE
        pass
    
    def parameters(self) -> List[Tensor]:
        """Return list of parameters for optimization."""
        return [self.weight, self.bias]

class SimpleNeuralNetwork:
    """
    Simple feedforward neural network for demonstrating backpropagation.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize network with one hidden layer.
        
        TODO: Create linear layers and activation function
        """
        # YOUR CODE HERE
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through network.
        
        TODO:
        1. First linear layer
        2. ReLU activation
        3. Second linear layer
        4. Return logits (no final activation)
        """
        # YOUR CODE HERE
        pass
    
    def parameters(self) -> List[Tensor]:
        """Return all network parameters."""
        params = []
        params.extend(self.layer1.parameters())
        params.extend(self.layer2.parameters())
        return params

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean squared error loss.
    
    TODO: Implement MSE = mean((predictions - targets)^2)
    """
    # YOUR CODE HERE
    pass

def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross-entropy loss for classification.
    
    TODO: 
    1. Compute softmax probabilities
    2. Compute cross-entropy: -sum(targets * log(probs))
    3. Handle numerical stability
    """
    # YOUR CODE HERE
    pass

def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax activation function.
    
    TODO:
    1. Subtract max for numerical stability
    2. Compute exp(x)
    3. Normalize by sum
    """
    # YOUR CODE HERE
    pass

class GradientChecker:
    """Utility class for checking gradient correctness."""
    
    @staticmethod
    def check_gradients(func: Callable[[Tensor], Tensor], 
                       inputs: Tensor,
                       eps: float = 1e-5,
                       tolerance: float = 1e-3) -> bool:
        """
        Check gradients using finite differences.
        
        TODO:
        1. Compute analytical gradients using backpropagation
        2. Compute numerical gradients using finite differences
        3. Compare and return whether they match within tolerance
        
        Finite difference: (f(x + eps) - f(x - eps)) / (2 * eps)
        """
        # YOUR CODE HERE
        pass

class ComputationalGraph:
    """
    Computational graph for visualization and analysis.
    """
    
    def __init__(self):
        """TODO: Initialize graph storage"""
        # YOUR CODE HERE
        pass
    
    def add_node(self, node_id: str, operation: str, inputs: List[str]) -> None:
        """TODO: Add node to graph"""
        # YOUR CODE HERE
        pass
    
    def visualize(self) -> None:
        """
        Visualize computational graph.
        
        TODO: Create a simple text-based or matplotlib visualization
        """
        # YOUR CODE HERE
        pass

def train_simple_network(X: np.ndarray, 
                        y: np.ndarray, 
                        epochs: int = 100,
                        learning_rate: float = 0.01) -> Dict[str, List[float]]:
    """
    Train a simple neural network and track metrics.
    
    TODO:
    1. Create network and convert data to tensors
    2. Training loop:
       - Forward pass
       - Compute loss
       - Backward pass
       - Update parameters
       - Track loss
    3. Return training history
    """
    # YOUR CODE HERE
    pass

def compare_forward_reverse_ad(func: Callable, 
                              input_dims: List[int], 
                              output_dims: List[int]) -> Dict[str, float]:
    """
    Compare efficiency of forward-mode vs reverse-mode AD.
    
    TODO:
    1. Implement simple forward-mode AD
    2. Time both forward and reverse mode
    3. Compare memory usage
    4. Return comparison results
    """
    # YOUR CODE HERE
    pass

def analyze_gradient_flow(network: SimpleNeuralNetwork, 
                         sample_input: Tensor) -> Dict[str, np.ndarray]:
    """
    Analyze gradient magnitudes throughout network.
    
    TODO:
    1. Perform forward and backward pass
    2. Extract gradient magnitudes for each layer
    3. Return analysis results
    """
    # YOUR CODE HERE
    pass

def demonstrate_vanishing_gradients(depth: int = 10) -> Dict[str, Any]:
    """
    Demonstrate vanishing gradient problem in deep networks.
    
    TODO:
    1. Create deep network with specified depth
    2. Initialize with different schemes
    3. Analyze gradient magnitudes across layers
    4. Return demonstration results
    """
    # YOUR CODE HERE
    pass

# Helper functions for creating sample data
def generate_classification_data(n_samples: int = 1000, 
                               n_features: int = 20, 
                               n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    W_true = np.random.randn(n_features, n_classes)
    logits = X @ W_true
    y = np.argmax(logits, axis=1)
    return X, y

def generate_regression_data(n_samples: int = 1000, 
                           n_features: int = 20,
                           noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + noise * np.random.randn(n_samples)
    return X, y

if __name__ == "__main__":
    print("Testing Backpropagation Implementation...")
    
    # Test basic tensor operations
    print("\n1. Testing Basic Tensor Operations...")
    try:
        a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
        c = a + b
        c.backward()
        print(f"Addition test passed: a.grad = {a.grad}, b.grad = {b.grad}")
    except Exception as e:
        print(f"Addition test failed: {e}")
    
    # Test neural network
    print("\n2. Testing Neural Network...")
    try:
        X, y = generate_classification_data(n_samples=100, n_features=10, n_classes=3)
        network = SimpleNeuralNetwork(input_size=10, hidden_size=20, output_size=3)
        
        x_tensor = Tensor(X[:5], requires_grad=False)
        output = network.forward(x_tensor)
        print(f"Network forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"Neural network test failed: {e}")
    
    # Test gradient checking
    print("\n3. Testing Gradient Checking...")
    try:
        def simple_func(x):
            return (x * x).sum()
        
        x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
        checker = GradientChecker()
        is_correct = checker.check_gradients(simple_func, x)
        print(f"Gradient checking result: {'PASSED' if is_correct else 'FAILED'}")
    except Exception as e:
        print(f"Gradient checking test failed: {e}")
    
    # Test training
    print("\n4. Testing Training Loop...")
    try:
        X, y = generate_regression_data(n_samples=100, n_features=5)
        history = train_simple_network(X, y, epochs=10, learning_rate=0.01)
        if history:
            print(f"Training completed, final loss: {history['loss'][-1]:.4f}")
    except Exception as e:
        print(f"Training test failed: {e}")
    
    print("\nBackpropagation implementation testing completed! 🧮")
    print("\nNext steps:")
    print("1. Implement all TODOs marked in the code")
    print("2. Add more sophisticated activation functions")
    print("3. Implement batch processing")
    print("4. Add support for different optimizers")
    print("5. Create comprehensive gradient checking utilities")
    print("6. Implement memory-efficient backpropagation")
    print("7. Add computational graph visualization")