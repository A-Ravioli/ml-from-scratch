"""
Reference Solutions for Backpropagation Calculus Exercise

Complete implementation of backpropagation algorithm with computational graph,
automatic differentiation, and gradient checking.

Author: ML-from-Scratch Course
"""

import numpy as np
from typing import Callable, List, Tuple, Dict, Any
import matplotlib.pyplot as plt


class ComputationNode:
    """Node in computational graph for automatic differentiation"""
    
    def __init__(self, value: np.ndarray, grad_fn: Callable = None, children: List = None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.grad_fn = grad_fn  # Function to compute gradient
        self.children = children or []
    
    def backward(self, grad_output: np.ndarray = None):
        """Compute gradients via backpropagation"""
        if grad_output is None:
            grad_output = np.ones_like(self.value)
        
        self.grad += grad_output
        
        if self.grad_fn:
            child_grads = self.grad_fn(grad_output)
            if not isinstance(child_grads, (list, tuple)):
                child_grads = [child_grads]
            
            for child, child_grad in zip(self.children, child_grads):
                if child is not None:
                    child.backward(child_grad)


def add(a: ComputationNode, b: ComputationNode) -> ComputationNode:
    """Addition operation with gradient computation"""
    result_value = a.value + b.value
    
    def grad_fn(grad_output):
        return [grad_output, grad_output]
    
    return ComputationNode(result_value, grad_fn, [a, b])


def multiply(a: ComputationNode, b: ComputationNode) -> ComputationNode:
    """Multiplication operation with gradient computation"""
    result_value = a.value * b.value
    
    def grad_fn(grad_output):
        return [grad_output * b.value, grad_output * a.value]
    
    return ComputationNode(result_value, grad_fn, [a, b])


def matmul(a: ComputationNode, b: ComputationNode) -> ComputationNode:
    """Matrix multiplication with gradient computation"""
    result_value = a.value @ b.value
    
    def grad_fn(grad_output):
        grad_a = grad_output @ b.value.T
        grad_b = a.value.T @ grad_output
        return [grad_a, grad_b]
    
    return ComputationNode(result_value, grad_fn, [a, b])


def sigmoid(x: ComputationNode) -> ComputationNode:
    """Sigmoid activation with gradient computation"""
    sig_value = 1.0 / (1.0 + np.exp(-np.clip(x.value, -500, 500)))
    
    def grad_fn(grad_output):
        return [grad_output * sig_value * (1 - sig_value)]
    
    return ComputationNode(sig_value, grad_fn, [x])


def relu(x: ComputationNode) -> ComputationNode:
    """ReLU activation with gradient computation"""
    result_value = np.maximum(0, x.value)
    
    def grad_fn(grad_output):
        return [grad_output * (x.value > 0)]
    
    return ComputationNode(result_value, grad_fn, [x])


def mse_loss(predictions: ComputationNode, targets: ComputationNode) -> ComputationNode:
    """Mean squared error loss with gradient computation"""
    diff = add(predictions, ComputationNode(-targets.value))
    squared = multiply(diff, diff)
    result_value = np.mean(squared.value)
    
    def grad_fn(grad_output):
        n = len(predictions.value)
        return [grad_output * 2 * diff.value / n]
    
    return ComputationNode(np.array(result_value), grad_fn, [diff])


class AutogradLayer:
    """Neural network layer with automatic differentiation"""
    
    def __init__(self, input_size: int, output_size: int):
        self.weights = ComputationNode(
            np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        )
        self.bias = ComputationNode(np.zeros(output_size))
    
    def forward(self, x: ComputationNode) -> ComputationNode:
        """Forward pass"""
        linear = add(matmul(x, self.weights), self.bias)
        return sigmoid(linear)
    
    def get_parameters(self) -> List[ComputationNode]:
        """Get trainable parameters"""
        return [self.weights, self.bias]


class AutogradNetwork:
    """Neural network with automatic differentiation"""
    
    def __init__(self, layer_sizes: List[int]):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = AutogradLayer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
    
    def forward(self, x: ComputationNode) -> ComputationNode:
        """Forward pass through network"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def get_parameters(self) -> List[ComputationNode]:
        """Get all network parameters"""
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params


def numerical_gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient using finite differences"""
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        
        x_plus.flat[i] += h
        x_minus.flat[i] -= h
        
        grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    
    return grad


def gradient_check(analytical_grad: np.ndarray, numerical_grad: np.ndarray, 
                  tolerance: float = 1e-5) -> bool:
    """Check if analytical and numerical gradients match"""
    diff = np.linalg.norm(analytical_grad - numerical_grad)
    norm = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    
    if norm == 0:
        return diff < tolerance
    
    relative_error = diff / norm
    return relative_error < tolerance


def demonstrate_backpropagation():
    """Demonstrate backpropagation with various examples"""
    print("Backpropagation Calculus Demonstrations")
    print("=" * 50)
    
    # 1. Simple computation graph
    print("\n1. Simple computation graph...")
    a = ComputationNode(np.array([2.0]))
    b = ComputationNode(np.array([3.0]))
    c = multiply(a, b)  # c = a * b = 6
    d = add(c, a)       # d = c + a = 8
    
    d.backward()
    print(f"a.value = {a.value}, a.grad = {a.grad}")  # Should be [5.0]
    print(f"b.value = {b.value}, b.grad = {b.grad}")  # Should be [2.0]
    
    # 2. Neural network example
    print("\n2. Neural network backpropagation...")
    np.random.seed(42)
    
    # Create simple network
    network = AutogradNetwork([2, 3, 1])
    
    # Forward pass
    x = ComputationNode(np.array([[1.0, 2.0]]))
    y_true = ComputationNode(np.array([[1.0]]))
    
    y_pred = network.forward(x)
    loss = mse_loss(y_pred, y_true)
    
    print(f"Prediction: {y_pred.value}")
    print(f"Loss: {loss.value}")
    
    # Backward pass
    loss.backward()
    
    # Print parameter gradients
    params = network.get_parameters()
    for i, param in enumerate(params):
        print(f"Parameter {i} gradient norm: {np.linalg.norm(param.grad):.6f}")
    
    # 3. Gradient checking
    print("\n3. Gradient checking...")
    
    def simple_function(x):
        return np.sum(x**2)
    
    def simple_gradient(x):
        return 2 * x
    
    x_test = np.random.randn(5)
    analytical = simple_gradient(x_test)
    numerical = numerical_gradient(simple_function, x_test)
    
    is_correct = gradient_check(analytical, numerical)
    print(f"Gradient check passed: {is_correct}")
    print(f"Analytical gradient: {analytical}")
    print(f"Numerical gradient: {numerical}")
    
    print("\nDemonstrations complete!")


if __name__ == "__main__":
    demonstrate_backpropagation() 