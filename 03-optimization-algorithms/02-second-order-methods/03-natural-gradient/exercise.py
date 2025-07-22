"""
Natural Gradient Methods Implementation Exercise

Implement natural gradient descent using Fisher Information Matrix.
Focus on understanding geometry-aware optimization and its applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict
from abc import ABC, abstractmethod
import scipy.linalg
import time


class ProbabilisticModel(ABC):
    """Base class for probabilistic models with natural gradient computation"""
    
    def __init__(self, n_params: int):
        self.n_params = n_params
    
    @abstractmethod
    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """Compute log-likelihood of data given parameters"""
        pass
    
    @abstractmethod
    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute score function (gradient of log-likelihood)"""
        pass
    
    @abstractmethod
    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute Fisher Information Matrix"""
        pass
    
    @abstractmethod
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample from the model given parameters"""
        pass


class GaussianModel(ProbabilisticModel):
    """
    Multivariate Gaussian model with mean μ and covariance Σ
    Parameters: θ = [μ, vec(L)] where Σ = L L^T (Cholesky decomposition)
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        # Parameters: μ (dim,) + lower triangular L (dim*(dim+1)/2,)
        n_params = dim + dim * (dim + 1) // 2
        super().__init__(n_params)
    
    def _unpack_params(self, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpack parameters into mean and Cholesky factor"""
        # TODO: Split params into mean μ and Cholesky factor L
        # Return μ and L where Σ = L L^T
        pass
    
    def _pack_params(self, mu: np.ndarray, L: np.ndarray) -> np.ndarray:
        """Pack mean and Cholesky factor into parameter vector"""
        # TODO: Combine μ and vec(L) into single parameter vector
        pass
    
    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """Compute log-likelihood of multivariate Gaussian"""
        # TODO: Implement log-likelihood
        # log p(x|μ,Σ) = -1/2 * [(x-μ)^T Σ^{-1} (x-μ) + log|2πΣ|]
        # Use Cholesky decomposition for efficient computation
        pass
    
    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute score function ∇_θ log p(x|θ)"""
        # TODO: Implement score function
        # For Gaussian: ∇_μ log p(x) = Σ^{-1} (x - μ)
        #              ∇_Σ log p(x) = 1/2 [Σ^{-1} - Σ^{-1}(x-μ)(x-μ)^T Σ^{-1}]
        # Convert to derivatives w.r.t. Cholesky factor L
        pass
    
    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray = None) -> np.ndarray:
        """Compute Fisher Information Matrix"""
        # TODO: Implement Fisher Information Matrix for Gaussian
        # F_θ = E[∇_θ log p(x|θ) ∇_θ log p(x|θ)^T]
        # For Gaussian, this has a known analytical form
        pass
    
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample from multivariate Gaussian"""
        mu, L = self._unpack_params(params)
        # TODO: Sample using μ and L: x = μ + L * z where z ~ N(0,I)
        pass


class CategoricalModel(ProbabilisticModel):
    """
    Categorical distribution with softmax parameterization
    Parameters: θ ∈ R^{k-1} (k-th probability is 1 - sum of others)
    """
    
    def __init__(self, n_categories: int):
        self.n_categories = n_categories
        # Use k-1 parameters due to sum-to-one constraint
        super().__init__(n_categories - 1)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax computation"""
        # TODO: Implement numerically stable softmax
        # Add zero for last category to handle constraint
        pass
    
    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """Compute log-likelihood for categorical data"""
        # TODO: Implement categorical log-likelihood
        # data should be one-hot encoded or category indices
        pass
    
    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Score function for categorical distribution"""
        # TODO: Implement score function
        # ∇_θ log p(x|θ) = x - p(θ) where p is softmax probabilities
        pass
    
    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray = None) -> np.ndarray:
        """Fisher Information Matrix for categorical distribution"""
        # TODO: Implement Fisher matrix
        # F = diag(p) - p p^T where p are the probabilities
        pass
    
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """Sample from categorical distribution"""
        probs = self._softmax(params)
        # TODO: Sample categories according to probabilities
        pass


class NeuralNetworkModel(ProbabilisticModel):
    """
    Neural network with probabilistic output (e.g., classification)
    Demonstrates natural gradient on more complex models
    """
    
    def __init__(self, layer_sizes: List[int], activation: str = 'tanh'):
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Count parameters
        n_params = 0
        for i in range(len(layer_sizes) - 1):
            n_params += layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]  # weights + biases
        
        super().__init__(n_params)
    
    def _unpack_params(self, params: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Unpack parameter vector into weights and biases"""
        # TODO: Split params into list of (W_i, b_i) for each layer
        pass
    
    def _forward(self, params: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through network"""
        # TODO: Implement forward pass
        # Return final output and list of activations for backprop
        pass
    
    def log_likelihood(self, params: np.ndarray, data: np.ndarray) -> float:
        """Log-likelihood for classification (cross-entropy)"""
        # TODO: Implement log-likelihood
        # Assumes data is (features, labels) tuple
        pass
    
    def score_function(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Score function using backpropagation"""
        # TODO: Implement gradient computation via backprop
        pass
    
    def fisher_information_matrix(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Approximate Fisher matrix using empirical Fisher"""
        # TODO: Implement empirical Fisher Information Matrix
        # F ≈ (1/n) sum_i ∇_θ log p(y_i|x_i,θ) ∇_θ log p(y_i|x_i,θ)^T
        pass
    
    def sample(self, params: np.ndarray, n_samples: int) -> np.ndarray:
        """Not applicable for discriminative models"""
        raise NotImplementedError("Sampling not applicable for discriminative models")


class NaturalGradientOptimizer:
    """
    Natural Gradient Descent: θ_{t+1} = θ_t + η F^{-1} ∇_θ L
    where F is the Fisher Information Matrix
    """
    
    def __init__(self, learning_rate: float = 0.01, regularization: float = 1e-6,
                 fisher_estimation: str = 'empirical', max_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.fisher_estimation = fisher_estimation
        self.max_iterations = max_iterations
        
        self.history = {
            'log_likelihood': [],
            'gradient_norm': [],
            'natural_gradient_norm': [],
            'fisher_condition': []
        }
    
    def optimize(self, model: ProbabilisticModel, data: np.ndarray, 
                initial_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run natural gradient descent"""
        
        params = initial_params.copy()
        
        for iteration in range(self.max_iterations):
            # TODO: Implement natural gradient iteration
            # 1. Compute log-likelihood
            # 2. Compute score function (gradient)
            # 3. Compute or estimate Fisher Information Matrix
            # 4. Solve F * natural_grad = gradient for natural gradient
            # 5. Update parameters: params += learning_rate * natural_grad
            # 6. Record metrics
            
            pass
        
        return params, self.history
    
    def _compute_natural_gradient(self, gradient: np.ndarray, fisher: np.ndarray) -> np.ndarray:
        """Compute natural gradient by solving Fisher system"""
        # TODO: Solve F * ng = g for natural gradient ng
        # Add regularization for numerical stability: (F + λI) * ng = g
        # Use appropriate linear solver (Cholesky, LU, CG)
        pass
    
    def _estimate_fisher_matrix(self, model: ProbabilisticModel, params: np.ndarray, 
                               data: np.ndarray) -> np.ndarray:
        """Estimate Fisher Information Matrix"""
        if self.fisher_estimation == 'exact':
            return model.fisher_information_matrix(params, data)
        elif self.fisher_estimation == 'empirical':
            # TODO: Compute empirical Fisher matrix
            # F_emp = (1/n) sum_i s_i s_i^T where s_i is score for sample i
            pass
        elif self.fisher_estimation == 'diagonal':
            # TODO: Use diagonal approximation for efficiency
            # Keep only diagonal elements of Fisher matrix
            pass
        else:
            raise ValueError(f"Unknown Fisher estimation method: {self.fisher_estimation}")


class AdaptiveNaturalGradient(NaturalGradientOptimizer):
    """
    Adaptive natural gradient with online Fisher matrix estimation
    """
    
    def __init__(self, decay_rate: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate
        self.running_fisher = None
    
    def _update_running_fisher(self, score: np.ndarray):
        """Update running average of Fisher matrix"""
        # TODO: Implement exponential moving average
        # F_t = decay * F_{t-1} + (1-decay) * s_t s_t^T
        pass


class KroneckerFactoredNaturalGradient:
    """
    K-FAC: Kronecker-Factored Approximate Curvature
    Efficient natural gradient for neural networks
    """
    
    def __init__(self, learning_rate: float = 0.01, damping: float = 1e-3,
                 update_frequency: int = 10):
        self.learning_rate = learning_rate
        self.damping = damping
        self.update_frequency = update_frequency
        
        # Kronecker factors for each layer
        self.A_factors = {}  # Input covariances
        self.S_factors = {}  # Output covariances
        
        self.history = {
            'loss': [],
            'gradient_norm': [],
            'natural_gradient_norm': []
        }
    
    def optimize(self, model: NeuralNetworkModel, data: np.ndarray,
                initial_params: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run K-FAC optimization"""
        
        # TODO: Implement K-FAC algorithm
        # 1. Compute activations and pre-activations
        # 2. Update Kronecker factors (A and S matrices)
        # 3. Compute K-FAC natural gradient
        # 4. Update parameters
        
        pass
    
    def _update_kronecker_factors(self, activations: List[np.ndarray], 
                                 gradients: List[np.ndarray]):
        """Update Kronecker factors for each layer"""
        # TODO: Update A_i = E[a_i a_i^T] and S_i = E[s_i s_i^T]
        # where a_i are layer inputs and s_i are gradient w.r.t. pre-activations
        pass
    
    def _compute_kfac_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Compute K-FAC natural gradient"""
        # TODO: Apply (A ⊗ S)^{-1} to gradient
        # Use Kronecker product properties: (A ⊗ B)^{-1} = A^{-1} ⊗ B^{-1}
        pass


def compare_optimizers(model: ProbabilisticModel, data: np.ndarray,
                      optimizers: Dict[str, object],
                      initial_params: np.ndarray) -> Dict:
    """Compare different optimization methods"""
    
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"Running {name}...")
        start_time = time.time()
        
        if hasattr(optimizer, 'optimize'):
            final_params, history = optimizer.optimize(model, data, initial_params.copy())
        else:
            # Handle standard gradient descent for comparison
            final_params, history = standard_gradient_descent(
                model, data, initial_params.copy(), optimizer
            )
        
        end_time = time.time()
        
        results[name] = {
            'final_params': final_params,
            'history': history,
            'runtime': end_time - start_time,
            'final_log_likelihood': model.log_likelihood(final_params, data)
        }
    
    return results


def standard_gradient_descent(model: ProbabilisticModel, data: np.ndarray,
                            initial_params: np.ndarray, 
                            optimizer_config: Dict) -> Tuple[np.ndarray, Dict]:
    """Standard gradient descent for comparison"""
    
    params = initial_params.copy()
    history = {'log_likelihood': [], 'gradient_norm': []}
    
    learning_rate = optimizer_config.get('learning_rate', 0.01)
    max_iterations = optimizer_config.get('max_iterations', 1000)
    
    for iteration in range(max_iterations):
        # TODO: Implement standard gradient descent
        # params -= learning_rate * gradient
        pass
    
    return params, history


def plot_optimization_comparison(results: Dict, title: str = "Optimization Comparison"):
    """Plot convergence comparison between optimization methods"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Log-likelihood convergence
    ax = axes[0]
    for name, result in results.items():
        history = result['history']
        if 'log_likelihood' in history:
            ax.plot(history['log_likelihood'], label=name)
    ax.set_title('Log-Likelihood')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.legend()
    ax.grid(True)
    
    # Gradient norm
    ax = axes[1]
    for name, result in results.items():
        history = result['history']
        if 'gradient_norm' in history:
            ax.semilogy(history['gradient_norm'], label=name)
    ax.set_title('Gradient Norm')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||∇L||')
    ax.legend()
    ax.grid(True)
    
    # Runtime comparison
    ax = axes[2]
    names = list(results.keys())
    runtimes = [results[name]['runtime'] for name in names]
    ax.bar(names, runtimes)
    ax.set_title('Runtime')
    ax.set_ylabel('Time (seconds)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_gaussian_natural_gradient():
    """
    Exercise 1: Natural gradient for Gaussian distribution
    
    Tasks:
    1. Complete GaussianModel implementation
    2. Implement Fisher Information Matrix computation
    3. Compare natural gradient vs standard gradient
    4. Verify invariance to reparameterization
    """
    
    print("=== Exercise 1: Gaussian Natural Gradient ===")
    
    # TODO: Create synthetic Gaussian data
    # Test natural gradient on parameter estimation
    # Compare convergence with standard gradient descent
    
    pass


def exercise_2_categorical_classification():
    """
    Exercise 2: Natural gradient for categorical models
    
    Tasks:
    1. Complete CategoricalModel implementation
    2. Test on multinomial classification problem
    3. Compare with standard gradient descent
    4. Study effect of Fisher matrix conditioning
    """
    
    print("=== Exercise 2: Categorical Classification ===")
    
    # TODO: Test natural gradient on classification
    # Compare convergence properties
    
    pass


def exercise_3_neural_network_natural_gradient():
    """
    Exercise 3: Natural gradient for neural networks
    
    Tasks:
    1. Implement empirical Fisher matrix computation
    2. Test on simple neural network classification
    3. Compare with standard backpropagation
    4. Study computational complexity
    """
    
    print("=== Exercise 3: Neural Network Natural Gradient ===")
    
    # TODO: Apply natural gradient to neural networks
    # Compare efficiency and convergence
    
    pass


def exercise_4_kfac_implementation():
    """
    Exercise 4: K-FAC for efficient natural gradients
    
    Tasks:
    1. Implement Kronecker factorization
    2. Test on feedforward networks
    3. Compare computational cost with full Fisher matrix
    4. Study approximation quality
    """
    
    print("=== Exercise 4: K-FAC Implementation ===")
    
    # TODO: Implement K-FAC algorithm
    # Test efficiency and approximation quality
    
    pass


def exercise_5_fisher_estimation_methods():
    """
    Exercise 5: Different Fisher matrix estimation methods
    
    Tasks:
    1. Compare exact vs empirical Fisher matrices
    2. Test diagonal approximations
    3. Study online estimation methods
    4. Analyze computational trade-offs
    """
    
    print("=== Exercise 5: Fisher Estimation Methods ===")
    
    # TODO: Compare different Fisher estimation approaches
    # Study accuracy vs efficiency trade-offs
    
    pass


def exercise_6_geometric_insights():
    """
    Exercise 6: Geometric understanding of natural gradients
    
    Tasks:
    1. Visualize optimization paths in parameter space
    2. Compare gradient vs natural gradient directions
    3. Study invariance properties
    4. Understand Riemannian optimization perspective
    """
    
    print("=== Exercise 6: Geometric Insights ===")
    
    # TODO: Visualize geometric properties of natural gradients
    # Understand Riemannian perspective
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_gaussian_natural_gradient()
    exercise_2_categorical_classification()
    exercise_3_neural_network_natural_gradient()
    exercise_4_kfac_implementation()
    exercise_5_fisher_estimation_methods()
    exercise_6_geometric_insights()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Geometric interpretation of natural gradients")
    print("2. Role of Fisher Information Matrix in optimization")
    print("3. Computational trade-offs in second-order methods")
    print("4. Applications to probabilistic models and neural networks")