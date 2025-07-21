"""
Neural Tangent Kernels Implementation Exercise

Implement and analyze Neural Tangent Kernels (NTK) theory.
Study the infinite-width limit of neural networks and their connection to kernel methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Union
from abc import ABC, abstractmethod
import time


class ActivationFunction:
    """Activation functions with their NTK-relevant properties"""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    @staticmethod
    def erf(x: np.ndarray) -> np.ndarray:
        """Error function activation (smooth approximation to ReLU)"""
        from scipy.special import erf
        return erf(x / np.sqrt(2))
    
    @staticmethod
    def erf_derivative(x: np.ndarray) -> np.ndarray:
        """Error function derivative"""
        return np.sqrt(2/np.pi) * np.exp(-x**2 / 2)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh derivative"""
        return 1 - np.tanh(x)**2


class NeuralTangentKernel:
    """
    Neural Tangent Kernel computation for fully connected networks
    
    Computes the infinite-width limit kernel for deep neural networks
    """
    
    def __init__(self, activation_fn: str = 'relu', depth: int = 1):
        self.activation_fn = activation_fn
        self.depth = depth
        
        # Set activation function and its derivative
        if activation_fn == 'relu':
            self.sigma = ActivationFunction.relu
            self.sigma_prime = ActivationFunction.relu_derivative
        elif activation_fn == 'erf':
            self.sigma = ActivationFunction.erf
            self.sigma_prime = ActivationFunction.erf_derivative
        elif activation_fn == 'tanh':
            self.sigma = ActivationFunction.tanh
            self.sigma_prime = ActivationFunction.tanh_derivative
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")
    
    def compute_ntk_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Compute the Neural Tangent Kernel matrix
        
        Args:
            X1: First set of inputs (n1, d)
            X2: Second set of inputs (n2, d), if None use X1
        
        Returns:
            NTK matrix (n1, n2)
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = X1.shape[0], X2.shape[0]
        ntk_matrix = np.zeros((n1, n2))
        
        # TODO: Compute NTK matrix
        # Use recursive formulation for deep networks
        
        for i in range(n1):
            for j in range(n2):
                ntk_matrix[i, j] = self._compute_ntk_entry(X1[i], X2[j])
        
        return ntk_matrix
    
    def _compute_ntk_entry(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute single entry of NTK matrix
        
        Args:
            x1, x2: Input vectors
        
        Returns:
            NTK(x1, x2)
        """
        # TODO: Implement NTK computation for single pair
        # Use recursive formulation based on network depth
        
        # Initialize with input kernel
        Sigma_0 = np.dot(x1, x2)  # Linear kernel on inputs
        
        if self.depth == 1:
            # Single layer case
            return self._single_layer_ntk(x1, x2, Sigma_0)
        else:
            # Multi-layer case - use recursion
            return self._deep_ntk_recursive(x1, x2, Sigma_0, self.depth)
    
    def _single_layer_ntk(self, x1: np.ndarray, x2: np.ndarray, Sigma_0: float) -> float:
        """Compute NTK for single hidden layer network"""
        # TODO: Implement single layer NTK
        # Formula depends on activation function
        
        if self.activation_fn == 'relu':
            return self._relu_ntk_single_layer(x1, x2, Sigma_0)
        elif self.activation_fn == 'erf':
            return self._erf_ntk_single_layer(x1, x2, Sigma_0)
        else:
            raise NotImplementedError(f"Single layer NTK not implemented for {self.activation_fn}")
    
    def _relu_ntk_single_layer(self, x1: np.ndarray, x2: np.ndarray, Sigma_0: float) -> float:
        """ReLU NTK for single layer"""
        # TODO: Implement ReLU NTK formula
        # For ReLU: involves arc-cosine kernels
        
        # Normalize inputs
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        # Cosine of angle between inputs
        cos_theta = Sigma_0 / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
        
        # Arc-cosine kernel of order 1
        theta = np.arccos(cos_theta)
        arc_cos_k1 = (1/np.pi) * (norm1 * norm2) * (np.sin(theta) + (np.pi - theta) * cos_theta)
        
        # NTK combines neural network Gaussian process kernel with its derivative
        # TODO: Complete ReLU NTK formula
        ntk_value = arc_cos_k1  # Simplified - complete the full formula
        
        return ntk_value
    
    def _erf_ntk_single_layer(self, x1: np.ndarray, x2: np.ndarray, Sigma_0: float) -> float:
        """Error function NTK for single layer"""
        # TODO: Implement error function NTK
        # Has closed form involving Gaussian integrals
        pass
    
    def _deep_ntk_recursive(self, x1: np.ndarray, x2: np.ndarray, 
                           Sigma_0: float, depth: int) -> float:
        """Compute NTK for deep networks using recursion"""
        # TODO: Implement recursive NTK computation
        # Follow the recursive formulation from NTK theory
        
        if depth == 1:
            return self._single_layer_ntk(x1, x2, Sigma_0)
        
        # Recursive case
        # TODO: Implement depth recursion
        # Each layer transforms the kernel according to activation function
        
        pass


class FiniteWidthNTK:
    """
    Finite-width approximation to NTK
    
    Studies the approach to infinite-width limit
    """
    
    def __init__(self, width: int, activation_fn: str = 'relu', depth: int = 1):
        self.width = width
        self.activation_fn = activation_fn
        self.depth = depth
        
        # Initialize network weights
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize network weights with NTK parameterization"""
        # TODO: Initialize weights for NTK parameterization
        # Weights should be scaled appropriately for infinite-width limit
        
        weights = []
        layer_sizes = self._get_layer_sizes()
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # NTK parameterization: scale by 1/sqrt(width)
            W = np.random.randn(fan_out, fan_in) / np.sqrt(fan_in)
            weights.append(W)
        
        return weights
    
    def _get_layer_sizes(self) -> List[int]:
        """Get layer sizes for the network"""
        # TODO: Define network architecture
        # Input dimension should be flexible
        input_dim = 1  # Will be set based on data
        hidden_dims = [self.width] * self.depth
        output_dim = 1
        
        return [input_dim] + hidden_dims + [output_dim]
    
    def compute_finite_ntk(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Compute finite-width NTK matrix
        
        This involves computing the Jacobian of the network w.r.t. parameters
        """
        if X2 is None:
            X2 = X1
        
        # TODO: Compute finite-width NTK
        # 1. Compute network Jacobians w.r.t. parameters
        # 2. Compute J(X1) @ J(X2)^T where J is Jacobian
        
        n1, n2 = X1.shape[0], X2.shape[0]
        
        # Compute Jacobians
        J1 = self._compute_jacobian(X1)  # (n1, n_params)
        J2 = self._compute_jacobian(X2)  # (n2, n_params)
        
        # NTK matrix is J1 @ J2^T
        ntk_matrix = J1 @ J2.T
        
        return ntk_matrix
    
    def _compute_jacobian(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian of network output w.r.t. parameters
        
        Args:
            X: Input data (n, d)
        
        Returns:
            Jacobian matrix (n, n_params)
        """
        # TODO: Implement Jacobian computation
        # Use automatic differentiation or manual computation
        
        n_samples = X.shape[0]
        n_params = sum(w.size for w in self.weights)
        jacobian = np.zeros((n_samples, n_params))
        
        for i, x in enumerate(X):
            # Compute gradient of f(x) w.r.t. all parameters
            grad = self._compute_parameter_gradient(x)
            jacobian[i] = grad
        
        return jacobian
    
    def _compute_parameter_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient of network output w.r.t. parameters for single input"""
        # TODO: Implement parameter gradient computation
        # This requires backpropagation through the network
        pass
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass through finite-width network"""
        # TODO: Implement forward pass
        activation = x
        
        for i, W in enumerate(self.weights[:-1]):
            # Linear transformation
            activation = W @ activation
            
            # Apply activation function (except last layer)
            if self.activation_fn == 'relu':
                activation = np.maximum(0, activation)
            elif self.activation_fn == 'tanh':
                activation = np.tanh(activation)
        
        # Final layer (linear)
        output = self.weights[-1] @ activation
        return output[0]  # Scalar output


class NTKAnalyzer:
    """
    Analyze properties of Neural Tangent Kernels
    """
    
    def __init__(self):
        pass
    
    def compare_infinite_finite_ntk(self, X: np.ndarray, widths: List[int],
                                  activation_fn: str = 'relu', depth: int = 1) -> Dict:
        """
        Compare infinite-width NTK with finite-width approximations
        """
        results = {
            'widths': widths,
            'ntk_matrices': [],
            'spectral_norms': [],
            'frobenius_norms': []
        }
        
        # Compute infinite-width NTK
        infinite_ntk = NeuralTangentKernel(activation_fn, depth)
        infinite_matrix = infinite_ntk.compute_ntk_matrix(X)
        
        print("Computing finite-width approximations...")
        for width in widths:
            print(f"Width: {width}")
            
            # TODO: Compute finite-width NTK
            finite_ntk = FiniteWidthNTK(width, activation_fn, depth)
            finite_matrix = finite_ntk.compute_finite_ntk(X)
            
            # Compare with infinite-width
            diff_matrix = finite_matrix - infinite_matrix
            
            results['ntk_matrices'].append(finite_matrix)
            results['spectral_norms'].append(np.linalg.norm(diff_matrix, ord=2))
            results['frobenius_norms'].append(np.linalg.norm(diff_matrix, ord='fro'))
        
        results['infinite_ntk'] = infinite_matrix
        return results
    
    def study_depth_effects(self, X: np.ndarray, depths: List[int],
                          activation_fn: str = 'relu') -> Dict:
        """
        Study how network depth affects the NTK
        """
        results = {
            'depths': depths,
            'ntk_matrices': [],
            'eigenvalues': [],
            'condition_numbers': []
        }
        
        for depth in depths:
            print(f"Analyzing depth: {depth}")
            
            ntk = NeuralTangentKernel(activation_fn, depth)
            ntk_matrix = ntk.compute_ntk_matrix(X)
            
            # Analyze eigenspectrum
            eigenvals = np.linalg.eigvals(ntk_matrix)
            eigenvals = np.real(eigenvals[eigenvals > 1e-12])  # Remove numerical zeros
            eigenvals = np.sort(eigenvals)[::-1]  # Sort descending
            
            condition_number = eigenvals[0] / eigenvals[-1] if len(eigenvals) > 1 else 1
            
            results['ntk_matrices'].append(ntk_matrix)
            results['eigenvalues'].append(eigenvals)
            results['condition_numbers'].append(condition_number)
        
        return results
    
    def compare_activation_functions(self, X: np.ndarray, 
                                   activation_fns: List[str]) -> Dict:
        """
        Compare NTK for different activation functions
        """
        results = {
            'activation_fns': activation_fns,
            'ntk_matrices': {},
            'kernel_similarities': {}
        }
        
        ntk_matrices = {}
        
        for activation_fn in activation_fns:
            print(f"Computing NTK for {activation_fn}")
            
            ntk = NeuralTangentKernel(activation_fn, depth=1)
            ntk_matrix = ntk.compute_ntk_matrix(X)
            ntk_matrices[activation_fn] = ntk_matrix
        
        # Compute pairwise similarities
        for i, act1 in enumerate(activation_fns):
            for j, act2 in enumerate(activation_fns):
                if i <= j:
                    # Normalized similarity (cosine similarity for matrices)
                    K1 = ntk_matrices[act1]
                    K2 = ntk_matrices[act2]
                    
                    similarity = np.trace(K1 @ K2) / (np.linalg.norm(K1, 'fro') * np.linalg.norm(K2, 'fro'))
                    results['kernel_similarities'][(act1, act2)] = similarity
        
        results['ntk_matrices'] = ntk_matrices
        return results
    
    def analyze_learning_dynamics(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Analyze learning dynamics using NTK theory
        
        NTK theory predicts that infinite-width networks learn via kernel regression
        """
        # TODO: Implement learning dynamics analysis
        # 1. Compute NTK matrix
        # 2. Solve kernel regression problem
        # 3. Compare with actual network training
        
        results = {
            'ntk_predictions': None,
            'kernel_regression_solution': None,
            'learning_curves': None
        }
        
        # Compute NTK
        ntk = NeuralTangentKernel('relu', depth=2)
        K_train = ntk.compute_ntk_matrix(X_train)
        K_test_train = ntk.compute_ntk_matrix(X_test, X_train)
        
        # Solve kernel regression
        # y_pred = K_test_train @ (K_train + λI)^{-1} @ y_train
        regularization = 1e-6
        K_reg = K_train + regularization * np.eye(K_train.shape[0])
        
        try:
            alpha = np.linalg.solve(K_reg, y_train)
            y_pred = K_test_train @ alpha
            
            results['ntk_predictions'] = y_pred
            results['kernel_regression_solution'] = alpha
        except np.linalg.LinAlgError:
            print("Warning: Kernel matrix is singular")
        
        return results


def plot_ntk_comparison(results: Dict):
    """Plot comparison between infinite and finite-width NTK"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot convergence to infinite-width limit
    axes[0].semilogx(results['widths'], results['spectral_norms'], 'o-')
    axes[0].set_xlabel('Network Width')
    axes[0].set_ylabel('Spectral Norm Difference')
    axes[0].set_title('Convergence to Infinite-Width NTK')
    axes[0].grid(True)
    
    # Plot Frobenius norm difference
    axes[1].semilogx(results['widths'], results['frobenius_norms'], 'o-')
    axes[1].set_xlabel('Network Width')
    axes[1].set_ylabel('Frobenius Norm Difference')
    axes[1].set_title('Matrix Difference (Frobenius Norm)')
    axes[1].grid(True)
    
    # Plot infinite-width NTK matrix
    im = axes[2].imshow(results['infinite_ntk'], cmap='viridis')
    axes[2].set_title('Infinite-Width NTK Matrix')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.show()


def plot_depth_analysis(results: Dict):
    """Plot analysis of depth effects on NTK"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot condition numbers vs depth
    axes[0].plot(results['depths'], results['condition_numbers'], 'o-')
    axes[0].set_xlabel('Network Depth')
    axes[0].set_ylabel('Condition Number')
    axes[0].set_title('NTK Condition Number vs Depth')
    axes[0].grid(True)
    
    # Plot eigenvalue spectra
    for i, (depth, eigenvals) in enumerate(zip(results['depths'], results['eigenvalues'])):
        axes[1].semilogy(eigenvals, label=f'Depth {depth}')
    
    axes[1].set_xlabel('Eigenvalue Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('NTK Eigenvalue Spectra')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_ntk_implementation():
    """
    Exercise 1: Implement basic NTK computation
    
    Tasks:
    1. Complete NeuralTangentKernel class
    2. Implement ReLU and other activation NTKs
    3. Verify against known analytical results
    4. Test on simple datasets
    """
    
    print("=== Exercise 1: NTK Implementation ===")
    
    # TODO: Test NTK implementation
    # Create simple 1D dataset and compute NTK
    
    X = np.linspace(-1, 1, 10).reshape(-1, 1)
    
    # Test single layer ReLU NTK
    ntk = NeuralTangentKernel('relu', depth=1)
    K = ntk.compute_ntk_matrix(X)
    
    print(f"NTK matrix shape: {K.shape}")
    print(f"NTK matrix positive definite: {np.all(np.linalg.eigvals(K) > 0)}")
    
    pass


def exercise_2_finite_width_convergence():
    """
    Exercise 2: Study finite-width convergence to infinite-width limit
    
    Tasks:
    1. Implement finite-width NTK computation
    2. Study convergence as width increases
    3. Analyze convergence rates
    4. Compare different architectures
    """
    
    print("=== Exercise 2: Finite-Width Convergence ===")
    
    # TODO: Study finite-width convergence
    
    pass


def exercise_3_depth_and_activation_analysis():
    """
    Exercise 3: Analyze effect of depth and activation functions
    
    Tasks:
    1. Study how depth affects NTK properties
    2. Compare different activation functions
    3. Analyze eigenspectra and condition numbers
    4. Study expressivity implications
    """
    
    print("=== Exercise 3: Depth and Activation Analysis ===")
    
    # TODO: Comprehensive analysis of depth and activation effects
    
    pass


def exercise_4_learning_dynamics():
    """
    Exercise 4: Connect NTK to learning dynamics
    
    Tasks:
    1. Implement kernel regression with NTK
    2. Compare with actual neural network training
    3. Study when NTK theory is accurate
    4. Analyze optimization trajectories
    """
    
    print("=== Exercise 4: Learning Dynamics ===")
    
    # TODO: Study learning dynamics via NTK
    
    pass


def exercise_5_practical_implications():
    """
    Exercise 5: Practical implications of NTK theory
    
    Tasks:
    1. Study when networks behave like kernel methods
    2. Analyze feature learning vs kernel regime
    3. Study generalization through NTK lens
    4. Connect to practical network design
    """
    
    print("=== Exercise 5: Practical Implications ===")
    
    # TODO: Study practical implications
    
    pass


def exercise_6_advanced_ntk():
    """
    Exercise 6: Advanced NTK topics
    
    Tasks:
    1. Implement NTK for convolutional networks
    2. Study NTK for other architectures (ResNet, etc.)
    3. Analyze NTK evolution during training
    4. Connect to recent theoretical developments
    """
    
    print("=== Exercise 6: Advanced NTK Topics ===")
    
    # TODO: Advanced NTK analysis
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_ntk_implementation()
    exercise_2_finite_width_convergence()
    exercise_3_depth_and_activation_analysis()
    exercise_4_learning_dynamics()
    exercise_5_practical_implications()
    exercise_6_advanced_ntk()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Infinite-width limit of neural networks")
    print("2. Connection between neural networks and kernel methods")
    print("3. Role of architecture in determining kernel properties")
    print("4. When and why NTK theory applies to finite networks")