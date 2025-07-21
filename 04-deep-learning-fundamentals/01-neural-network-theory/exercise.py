"""
Neural Network Theory Implementation Exercise

Implement neural networks from scratch with focus on theoretical understanding.
Explore universal approximation, expressivity, and capacity of neural networks.
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
        # TODO: Implement sigmoid with numerical stability
        # Use the identity: σ(x) = exp(x) / (1 + exp(x)) for x >= 0
        #                        = 1 / (1 + exp(-x)) for x < 0
        pass
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement sigmoid derivative: σ'(x) = σ(x)(1 - σ(x))
        pass


class Tanh(ActivationFunction):
    """Hyperbolic tangent: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement tanh
        pass
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement tanh derivative: tanh'(x) = 1 - tanh²(x)
        pass


class ReLU(ActivationFunction):
    """Rectified Linear Unit: ReLU(x) = max(0, x)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement ReLU
        pass
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement ReLU derivative (step function)
        # Note: derivative at x=0 is undefined, use convention of 0 or 1
        pass


class LeakyReLU(ActivationFunction):
    """Leaky ReLU: LeakyReLU(x) = max(αx, x) where α < 1"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement Leaky ReLU
        pass
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement Leaky ReLU derivative
        pass


class Swish(ActivationFunction):
    """Swish activation: Swish(x) = x * σ(x)"""
    
    def __init__(self):
        self.sigmoid = Sigmoid()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement Swish
        pass
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement Swish derivative using product rule
        # d/dx[x * σ(x)] = σ(x) + x * σ'(x)
        pass


class NeuralNetwork:
    """
    Feedforward neural network from scratch
    
    Focus on theoretical understanding rather than efficiency
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[ActivationFunction],
                 weight_init: str = 'xavier', bias_init: str = 'zeros'):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.n_layers = len(layer_sizes) - 1
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters(weight_init, bias_init)
        
        # Store forward pass information for backpropagation
        self.z_values = []  # Pre-activations
        self.a_values = []  # Activations
    
    def _initialize_parameters(self, weight_init: str, bias_init: str):
        """Initialize network parameters"""
        # TODO: Implement different initialization schemes
        # 1. Xavier/Glorot initialization
        # 2. He initialization
        # 3. Random normal
        # 4. Random uniform
        
        for i in range(self.n_layers):
            input_dim = self.layer_sizes[i]
            output_dim = self.layer_sizes[i + 1]
            
            # TODO: Initialize weights based on scheme
            if weight_init == 'xavier':
                # Xavier: W ~ N(0, 1/n_in)
                pass
            elif weight_init == 'he':
                # He: W ~ N(0, 2/n_in)
                pass
            elif weight_init == 'random_normal':
                # Standard normal
                pass
            elif weight_init == 'random_uniform':
                # Uniform [-1, 1]
                pass
            
            # TODO: Initialize biases
            if bias_init == 'zeros':
                pass
            elif bias_init == 'random':
                pass
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network
        
        Args:
            X: Input data (batch_size, input_dim)
        
        Returns:
            Output of the network (batch_size, output_dim)
        """
        # TODO: Implement forward propagation
        # 1. Store input as first activation
        # 2. For each layer: compute z = Wa + b, then a = activation(z)
        # 3. Store intermediate values for backpropagation
        # 4. Return final output
        
        self.z_values = []
        self.a_values = [X]  # First activation is input
        
        current_input = X
        
        for i in range(self.n_layers):
            # TODO: Compute pre-activation and activation
            pass
        
        return current_input
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation to compute gradients
        
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        # TODO: Implement backpropagation algorithm
        # 1. Compute output layer error
        # 2. Backpropagate errors through layers
        # 3. Compute gradients w.r.t. weights and biases
        
        batch_size = X.shape[0]
        
        # Initialize gradient storage
        weight_grads = [np.zeros_like(w) for w in self.weights]
        bias_grads = [np.zeros_like(b) for b in self.biases]
        
        # TODO: Output layer error (depends on loss function)
        # For MSE: δ_L = (output - y) * σ'(z_L)
        
        # TODO: Backpropagate errors
        for i in range(self.n_layers - 1, -1, -1):
            # Compute gradients for layer i
            pass
        
        return weight_grads, bias_grads
    
    def train_step(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """Single training step"""
        # TODO: Implement training step
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass
        # 4. Update parameters
        
        # Forward pass
        output = self.forward(X)
        
        # Compute loss (MSE for regression)
        loss = 0.5 * np.mean((output - y) ** 2)
        
        # Backward pass
        weight_grads, bias_grads = self.backward(X, y, output)
        
        # Update parameters
        for i in range(self.n_layers):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (forward pass without storing gradients)"""
        return self.forward(X)


class UniversalApproximationExperiment:
    """
    Experiments to verify universal approximation theorem
    """
    
    def __init__(self):
        self.target_functions = {
            'polynomial': lambda x: x**3 - 2*x**2 + x,
            'trigonometric': lambda x: np.sin(3*x) + 0.5*np.cos(7*x),
            'step_function': lambda x: np.where(x < 0, -1, np.where(x < 0.5, 0, 1)),
            'discontinuous': lambda x: np.where(np.abs(x) < 0.3, 1, 0),
            'high_frequency': lambda x: np.sin(20*x) * np.exp(-x**2)
        }
    
    def generate_data(self, func_name: str, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data for target function"""
        # TODO: Generate training data
        # 1. Sample x values uniformly
        # 2. Compute y = f(x) + noise
        # 3. Return (X, y) suitable for training
        pass
    
    def test_approximation_capacity(self, func_name: str, hidden_sizes: List[int],
                                  n_epochs: int = 1000) -> Dict:
        """Test network's ability to approximate target function"""
        
        results = {}
        X_train, y_train = self.generate_data(func_name, 1000)
        X_test, y_test = self.generate_data(func_name, 200)
        
        for hidden_size in hidden_sizes:
            print(f"Testing {func_name} with {hidden_size} hidden units...")
            
            # TODO: Create and train network
            # 1. Build network architecture
            # 2. Train for specified epochs
            # 3. Evaluate approximation quality
            # 4. Store results
            
            pass
        
        return results
    
    def analyze_depth_vs_width(self, func_name: str) -> Dict:
        """Compare deep narrow vs shallow wide networks"""
        
        # TODO: Compare different architectures:
        # 1. Shallow wide: [1, 100, 1]
        # 2. Medium: [1, 50, 50, 1]  
        # 3. Deep narrow: [1, 25, 25, 25, 25, 1]
        # Test approximation quality and training dynamics
        
        pass


class ExpressionCapacityAnalysis:
    """
    Analyze the expressivity and representational capacity of neural networks
    """
    
    def count_linear_regions(self, network: NeuralNetwork, input_bounds: Tuple[float, float],
                           resolution: int = 1000) -> int:
        """
        Count number of linear regions for ReLU networks
        
        ReLU networks are piecewise linear functions
        """
        # TODO: Implement linear region counting
        # 1. Sample inputs densely
        # 2. Track sign patterns of pre-activations
        # 3. Count distinct sign patterns (= linear regions)
        pass
    
    def measure_function_diversity(self, layer_sizes: List[int], n_samples: int = 1000) -> Dict:
        """Measure diversity of functions representable by architecture"""
        
        # TODO: Analyze function diversity
        # 1. Sample random weight configurations
        # 2. Evaluate resulting functions on test points  
        # 3. Measure diversity using various metrics
        # 4. Compare across architectures
        
        pass
    
    def lottery_ticket_experiment(self, network: NeuralNetwork, sparsity_levels: List[float]) -> Dict:
        """
        Test lottery ticket hypothesis: sparse subnetworks can match full performance
        """
        
        # TODO: Implement lottery ticket experiment
        # 1. Train full network to completion
        # 2. Prune smallest weights at different sparsity levels
        # 3. Retrain pruned networks from original initialization
        # 4. Compare performance vs sparsity
        
        pass


class ActivationFunctionAnalysis:
    """
    Systematic analysis of different activation functions
    """
    
    def __init__(self):
        self.activations = {
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'relu': ReLU(),
            'leaky_relu': LeakyReLU(),
            'swish': Swish()
        }
    
    def gradient_flow_analysis(self, activation_name: str, depth: int = 10) -> Dict:
        """Analyze gradient flow through deep networks"""
        
        # TODO: Analyze gradient flow
        # 1. Create deep network with specified activation
        # 2. Compute gradients at different depths
        # 3. Measure gradient magnitudes
        # 4. Identify vanishing/exploding gradient issues
        
        pass
    
    def saturation_analysis(self, activation_name: str) -> Dict:
        """Analyze activation saturation patterns"""
        
        # TODO: Study activation saturation
        # 1. Train network and monitor activations
        # 2. Measure fraction of saturated neurons
        # 3. Study effect on learning dynamics
        # 4. Compare across activation functions
        
        pass
    
    def expressivity_comparison(self) -> Dict:
        """Compare expressivity of different activation functions"""
        
        results = {}
        
        for name, activation in self.activations.items():
            # TODO: Test expressivity
            # 1. Train networks with each activation on same task
            # 2. Measure approximation quality
            # 3. Analyze convergence speed
            # 4. Compare final performance
            
            pass
        
        return results


def visualize_universal_approximation(target_func: Callable, approximation: Callable,
                                    x_range: Tuple[float, float] = (-2, 2)):
    """Visualize how well network approximates target function"""
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    y_target = target_func(x)
    y_approx = approximation(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_target, 'b-', label='Target Function', linewidth=2)
    plt.plot(x, y_approx, 'r--', label='Neural Network', linewidth=2)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Universal Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Compute approximation error
    mse = np.mean((y_target - y_approx) ** 2)
    print(f"Mean Squared Error: {mse:.6f}")


def plot_activation_functions():
    """Plot various activation functions and their derivatives"""
    
    activations = {
        'Sigmoid': Sigmoid(),
        'Tanh': Tanh(),
        'ReLU': ReLU(),
        'Leaky ReLU': LeakyReLU(),
        'Swish': Swish()
    }
    
    x = np.linspace(-3, 3, 1000)
    
    fig, axes = plt.subplots(2, len(activations), figsize=(15, 8))
    
    for i, (name, activation) in enumerate(activations.items()):
        # Plot activation function
        y = activation.forward(x)
        axes[0, i].plot(x, y)
        axes[0, i].set_title(f'{name}')
        axes[0, i].grid(True)
        
        # Plot derivative
        dy = activation.derivative(x)
        axes[1, i].plot(x, dy)
        axes[1, i].set_title(f'{name} Derivative')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_activation_functions():
    """
    Exercise 1: Implement and analyze activation functions
    
    Tasks:
    1. Complete all activation function implementations
    2. Plot activation functions and derivatives
    3. Analyze properties (range, monotonicity, saturation)
    4. Compare gradient flow characteristics
    """
    
    print("=== Exercise 1: Activation Functions ===")
    
    # TODO: Test all activation function implementations
    # Plot and analyze their properties
    
    pass


def exercise_2_network_implementation():
    """
    Exercise 2: Build neural network from scratch
    
    Tasks:
    1. Complete NeuralNetwork class implementation
    2. Test forward and backward propagation
    3. Verify gradients using finite differences
    4. Train on simple datasets (XOR, regression)
    """
    
    print("=== Exercise 2: Neural Network Implementation ===")
    
    # TODO: Test complete neural network implementation
    
    pass


def exercise_3_universal_approximation():
    """
    Exercise 3: Verify universal approximation theorem
    
    Tasks:
    1. Test approximation of various target functions
    2. Study effect of network width on approximation quality
    3. Compare shallow vs deep networks
    4. Analyze sample complexity
    """
    
    print("=== Exercise 3: Universal Approximation ===")
    
    # TODO: Comprehensive universal approximation experiments
    
    pass


def exercise_4_expressivity_analysis():
    """
    Exercise 4: Analyze network expressivity
    
    Tasks:
    1. Count linear regions in ReLU networks
    2. Measure function diversity across architectures
    3. Study effect of depth vs width
    4. Implement lottery ticket experiments
    """
    
    print("=== Exercise 4: Expressivity Analysis ===")
    
    # TODO: Deep analysis of network expressivity
    
    pass


def exercise_5_gradient_flow():
    """
    Exercise 5: Study gradient flow in deep networks
    
    Tasks:
    1. Analyze vanishing/exploding gradients
    2. Compare different activation functions
    3. Study effect of network depth
    4. Implement gradient clipping strategies
    """
    
    print("=== Exercise 5: Gradient Flow Analysis ===")
    
    # TODO: Comprehensive gradient flow analysis
    
    pass


def exercise_6_theoretical_properties():
    """
    Exercise 6: Explore theoretical properties
    
    Tasks:
    1. Study approximation rates vs network size
    2. Analyze bias-variance tradeoffs
    3. Investigate memorization vs generalization
    4. Connect theory to practical observations
    """
    
    print("=== Exercise 6: Theoretical Properties ===")
    
    # TODO: Deep theoretical analysis
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_activation_functions()
    exercise_2_network_implementation()
    exercise_3_universal_approximation()
    exercise_4_expressivity_analysis()
    exercise_5_gradient_flow()
    exercise_6_theoretical_properties()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Universal approximation capabilities of neural networks")
    print("2. Role of activation functions in expressivity")
    print("3. Trade-offs between depth and width")
    print("4. Gradient flow and training dynamics")
    print("5. Theoretical foundations of deep learning")