"""
Reference Solutions for Neural Network Theory Exercise

Complete implementations of all neural network components with theoretical analysis.
Focus on understanding capacity, expressivity, and universal approximation.

Author: ML-from-Scratch Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time


class ActivationFunction(ABC):
    """Base class for activation functions"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute activation function"""
        pass
    
    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function"""
        pass


class Sigmoid(ActivationFunction):
    """Sigmoid activation: σ(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement sigmoid with numerical stability"""
        # Use the identity: σ(x) = exp(x) / (1 + exp(x)) for x >= 0
        #                        = 1 / (1 + exp(-x)) for x < 0
        x = np.array(x, dtype=np.float64)
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        
        result = np.zeros_like(x)
        
        # For positive values: σ(x) = exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x))
        result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
        
        # For negative values: σ(x) = exp(x) / (1 + exp(x))
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1.0 + exp_x)
        
        return result
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))"""
        sig_x = self.forward(x)
        return sig_x * (1.0 - sig_x)


class Tanh(ActivationFunction):
    """Hyperbolic tangent: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement tanh using numpy's stable implementation"""
        return np.tanh(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement tanh derivative: tanh'(x) = 1 - tanh²(x)"""
        tanh_x = self.forward(x)
        return 1.0 - tanh_x**2


class ReLU(ActivationFunction):
    """Rectified Linear Unit: ReLU(x) = max(0, x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement ReLU"""
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement ReLU derivative (step function)"""
        # Use convention: derivative at x=0 is 0
        return (x > 0).astype(np.float64)


class LeakyReLU(ActivationFunction):
    """Leaky ReLU: LeakyReLU(x) = max(αx, x) where α < 1"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement Leaky ReLU"""
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement Leaky ReLU derivative"""
        return np.where(x > 0, 1.0, self.alpha)


class ELU(ActivationFunction):
    """Exponential Linear Unit: ELU(x) = x if x > 0, α(exp(x) - 1) if x ≤ 0"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement ELU"""
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement ELU derivative"""
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))


class Swish(ActivationFunction):
    """Swish activation: Swish(x) = x * σ(x)"""
    
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Implement Swish"""
        return x * self.sigmoid.forward(x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Implement Swish derivative"""
        sig_x = self.sigmoid.forward(x)
        return sig_x + x * sig_x * (1 - sig_x)


class Layer:
    """Single layer of neural network"""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction, use_bias: bool = True):
        """Initialize layer"""
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Xavier/Glorot initialization
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.normal(0, scale, (input_size, output_size))
        self.bias = np.zeros(output_size) if use_bias else None
        
        # Cache for backpropagation
        self.last_input = None
        self.last_linear_output = None
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.last_input = x
        
        # Linear transformation
        self.last_linear_output = x @ self.weights
        if self.use_bias:
            self.last_linear_output += self.bias
        
        # Activation
        self.last_output = self.activation.forward(self.last_linear_output)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass"""
        # Gradient through activation
        grad_activation = self.activation.derivative(self.last_linear_output)
        grad_linear = grad_output * grad_activation
        
        # Gradient with respect to weights
        grad_weights = self.last_input.T @ grad_linear
        
        # Gradient with respect to bias
        grad_bias = np.sum(grad_linear, axis=0) if self.use_bias else None
        
        # Gradient with respect to input
        grad_input = grad_linear @ self.weights.T
        
        return grad_input, grad_weights, grad_bias


class NeuralNetwork:
    """Multi-layer neural network"""
    
    def __init__(self, layer_sizes: List[int], activations: List[ActivationFunction],
                 use_bias: bool = True):
        """Initialize neural network"""
        self.layer_sizes = layer_sizes
        self.layers = []
        
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i], use_bias)
            self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, x: np.ndarray, y: np.ndarray, 
                loss_derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> List[Tuple]:
        """Backward pass through network"""
        # Forward pass to compute outputs
        self.forward(x)
        
        # Compute loss gradient
        output = self.layers[-1].last_output
        grad_output = loss_derivative(output, y)
        
        # Backward pass through layers
        gradients = []
        for layer in reversed(self.layers):
            grad_input, grad_weights, grad_bias = layer.backward(grad_output)
            gradients.append((grad_weights, grad_bias))
            grad_output = grad_input
        
        return list(reversed(gradients))
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(x)
    
    def get_parameter_count(self) -> int:
        """Count total number of parameters"""
        count = 0
        for layer in self.layers:
            count += layer.weights.size
            if layer.use_bias:
                count += layer.bias.size
        return count


class LossFunction(ABC):
    """Base class for loss functions"""
    
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute loss"""
        pass
    
    @abstractmethod
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute gradient of loss"""
        pass


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss"""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss"""
        diff = predictions - targets
        return 0.5 * np.mean(diff**2)
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute MSE gradient"""
        return (predictions - targets) / len(predictions)


class CrossEntropy(LossFunction):
    """Cross-entropy loss for classification"""
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss"""
        # Add small epsilon for numerical stability
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        if targets.ndim == 1:
            # Binary classification
            return -np.mean(targets * np.log(predictions) + 
                          (1 - targets) * np.log(1 - predictions))
        else:
            # Multi-class classification
            return -np.mean(np.sum(targets * np.log(predictions), axis=1))
    
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Compute cross-entropy gradient"""
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        if targets.ndim == 1:
            # Binary classification
            return -(targets / predictions - (1 - targets) / (1 - predictions)) / len(predictions)
        else:
            # Multi-class classification
            return -(targets / predictions) / len(predictions)


class SGDOptimizer:
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def update(self, network: NeuralNetwork, gradients: List[Tuple]):
        """Update network parameters"""
        if self.velocity is None:
            self.velocity = [(np.zeros_like(grad_w), 
                            np.zeros_like(grad_b) if grad_b is not None else None) 
                           for grad_w, grad_b in gradients]
        
        for i, (layer, (grad_w, grad_b)) in enumerate(zip(network.layers, gradients)):
            # Update velocity
            self.velocity[i] = (
                self.momentum * self.velocity[i][0] + self.learning_rate * grad_w,
                self.momentum * self.velocity[i][1] + self.learning_rate * grad_b 
                if grad_b is not None else None
            )
            
            # Update parameters
            layer.weights -= self.velocity[i][0]
            if layer.use_bias:
                layer.bias -= self.velocity[i][1]


class UniversalApproximationAnalyzer:
    """Analyze universal approximation properties"""
    
    @staticmethod
    def approximate_function(target_func: Callable[[np.ndarray], np.ndarray],
                           domain: Tuple[float, float], n_hidden: int,
                           n_samples: int = 1000, epochs: int = 1000) -> Tuple[NeuralNetwork, float]:
        """
        Demonstrate universal approximation by approximating a target function.
        """
        # Generate training data
        x_min, x_max = domain
        x = np.linspace(x_min, x_max, n_samples).reshape(-1, 1)
        y = target_func(x.ravel()).reshape(-1, 1)
        
        # Create network
        network = NeuralNetwork(
            layer_sizes=[1, n_hidden, 1],
            activations=[Sigmoid(), Sigmoid()]
        )
        
        # Training setup
        loss_fn = MeanSquaredError()
        optimizer = SGDOptimizer(learning_rate=0.1)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            # Forward and backward pass
            gradients = network.backward(x, y, loss_fn.derivative)
            
            # Update parameters
            optimizer.update(network, gradients)
            
            # Compute loss
            predictions = network.forward(x)
            loss = loss_fn.forward(predictions, y)
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return network, losses[-1]
    
    @staticmethod
    def capacity_analysis(max_hidden: int = 100, step: int = 10) -> Dict[str, List]:
        """Analyze relationship between network capacity and approximation error"""
        target_func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
        
        hidden_sizes = list(range(step, max_hidden + 1, step))
        final_losses = []
        param_counts = []
        
        for n_hidden in hidden_sizes:
            print(f"Testing {n_hidden} hidden units...")
            network, final_loss = UniversalApproximationAnalyzer.approximate_function(
                target_func, domain=(-1, 1), n_hidden=n_hidden, epochs=500
            )
            
            final_losses.append(final_loss)
            param_counts.append(network.get_parameter_count())
        
        return {
            'hidden_sizes': hidden_sizes,
            'final_losses': final_losses,
            'param_counts': param_counts
        }


class ExpressivityAnalyzer:
    """Analyze expressivity of different network architectures"""
    
    @staticmethod
    def compute_decision_boundary_complexity(network: NeuralNetwork, 
                                           resolution: int = 100) -> float:
        """Measure complexity of decision boundary"""
        # Generate 2D grid
        x = np.linspace(-2, 2, resolution)
        y = np.linspace(-2, 2, resolution)
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Get predictions
        predictions = network.forward(grid)
        pred_grid = predictions.reshape(xx.shape)
        
        # Compute total variation (measure of complexity)
        dx = np.gradient(pred_grid, axis=1)
        dy = np.gradient(pred_grid, axis=0)
        total_variation = np.sum(np.sqrt(dx**2 + dy**2))
        
        return total_variation
    
    @staticmethod
    def analyze_depth_vs_width(max_depth: int = 5, max_width: int = 50) -> Dict:
        """Compare expressivity of deep vs wide networks"""
        results = {
            'depths': [],
            'widths': [],
            'complexities': [],
            'param_counts': []
        }
        
        # Fixed parameter budget
        param_budget = 1000
        
        for depth in range(2, max_depth + 1):
            for width in range(5, max_width + 1, 5):
                # Create network architecture
                layer_sizes = [2] + [width] * depth + [1]
                activations = [ReLU()] * (depth + 1)
                
                network = NeuralNetwork(layer_sizes, activations)
                
                # Skip if too many parameters
                if network.get_parameter_count() > param_budget:
                    continue
                
                # Initialize with random weights
                complexity = ExpressivityAnalyzer.compute_decision_boundary_complexity(network)
                
                results['depths'].append(depth)
                results['widths'].append(width)
                results['complexities'].append(complexity)
                results['param_counts'].append(network.get_parameter_count())
        
        return results


class ActivationAnalyzer:
    """Analyze properties of different activation functions"""
    
    @staticmethod
    def plot_activation_functions():
        """Plot various activation functions and their derivatives"""
        x = np.linspace(-5, 5, 1000)
        
        activations = {
            'Sigmoid': Sigmoid(),
            'Tanh': Tanh(),
            'ReLU': ReLU(),
            'LeakyReLU': LeakyReLU(0.1),
            'ELU': ELU(),
            'Swish': Swish()
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name, activation) in enumerate(activations.items()):
            ax = axes[i]
            
            # Plot function and derivative
            y = activation.forward(x)
            dy = activation.derivative(x)
            
            ax.plot(x, y, label=f'{name}', linewidth=2)
            ax.plot(x, dy, label=f'{name} derivative', linestyle='--', alpha=0.7)
            
            ax.set_title(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def gradient_flow_analysis(network_depth: int = 10) -> Dict:
        """Analyze gradient flow through deep networks"""
        # Create deep network
        layer_sizes = [10] + [50] * network_depth + [10]
        
        results = {}
        
        for activation_name, activation in [('Sigmoid', Sigmoid()), 
                                          ('Tanh', Tanh()), 
                                          ('ReLU', ReLU())]:
            
            activations = [activation] * (network_depth + 1)
            network = NeuralNetwork(layer_sizes, activations)
            
            # Forward pass with random input
            x = np.random.randn(32, 10)
            y = np.random.randn(32, 10)
            
            # Backward pass
            gradients = network.backward(x, y, MeanSquaredError().derivative)
            
            # Compute gradient norms for each layer
            grad_norms = []
            for grad_w, grad_b in gradients:
                grad_norms.append(np.linalg.norm(grad_w))
            
            results[activation_name] = grad_norms
        
        return results


def demonstration():
    """Demonstrate neural network capabilities"""
    print("Neural Network Theory Demonstrations")
    print("=" * 50)
    
    # 1. Activation function visualization
    print("\n1. Plotting activation functions...")
    ActivationAnalyzer.plot_activation_functions()
    
    # 2. Universal approximation
    print("\n2. Universal approximation demonstration...")
    target_func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)
    
    for n_hidden in [10, 20, 50]:
        print(f"\nApproximating with {n_hidden} hidden units...")
        network, final_loss = UniversalApproximationAnalyzer.approximate_function(
            target_func, domain=(-1, 1), n_hidden=n_hidden, epochs=300
        )
        print(f"Final approximation error: {final_loss:.6f}")
    
    # 3. Capacity analysis
    print("\n3. Network capacity analysis...")
    capacity_results = UniversalApproximationAnalyzer.capacity_analysis(max_hidden=50, step=10)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(capacity_results['hidden_sizes'], capacity_results['final_losses'], 'o-')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Final Approximation Error')
    plt.title('Capacity vs Approximation Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(capacity_results['param_counts'], capacity_results['final_losses'], 'o-')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Final Approximation Error')
    plt.title('Parameters vs Approximation Error')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Gradient flow analysis
    print("\n4. Gradient flow analysis...")
    gradient_results = ActivationAnalyzer.gradient_flow_analysis(network_depth=5)
    
    plt.figure(figsize=(10, 6))
    for activation_name, grad_norms in gradient_results.items():
        plt.plot(range(len(grad_norms)), grad_norms, 'o-', label=activation_name)
    
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Through Deep Networks')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    demonstration() 