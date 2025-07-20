"""
Adaptive Gradient Methods Implementation Exercises

This module contains implementation templates for various adaptive learning rate optimization algorithms.
Your task is to implement these methods from scratch and understand their theoretical properties.

Author: ML From Scratch Curriculum  
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod
import warnings


class AdaptiveOptimizerBase(ABC):
    """Base class for adaptive optimizers"""
    
    def __init__(self, learning_rate: float = 0.001, eps: float = 1e-8):
        self.learning_rate = learning_rate
        self.eps = eps  # Numerical stability
        self.history = {'loss': [], 'x': [], 'lr_effective': []}
        self.step_count = 0
    
    @abstractmethod
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Perform one optimization step"""
        pass
    
    def reset(self):
        """Reset optimizer state"""
        self.history = {'loss': [], 'x': [], 'lr_effective': []}
        self.step_count = 0


class AdaGrad(AdaptiveOptimizerBase):
    """AdaGrad: Adaptive Gradient Algorithm"""
    
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        super().__init__(learning_rate, eps)
        self.G = None  # Accumulated squared gradients
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement AdaGrad update rule.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.G is None:
            self.G = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement AdaGrad
        # Formula:
        # G_t = G_{t-1} + g_t^2  (element-wise)
        # x_{t+1} = x_t - η / sqrt(G_t + ε) * g_t  (element-wise)
        
        raise NotImplementedError("Implement AdaGrad update")
    
    def reset(self):
        super().reset()
        self.G = None


class RMSprop(AdaptiveOptimizerBase):
    """RMSprop: Root Mean Square Propagation"""
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.9, eps: float = 1e-8):
        super().__init__(learning_rate, eps)
        self.gamma = gamma  # Decay factor for moving average
        self.v = None  # Exponential moving average of squared gradients
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement RMSprop update rule.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.v is None:
            self.v = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement RMSprop
        # Formula:
        # v_t = γ * v_{t-1} + (1-γ) * g_t^2
        # x_{t+1} = x_t - η / sqrt(v_t + ε) * g_t
        
        raise NotImplementedError("Implement RMSprop update")
    
    def reset(self):
        super().reset()
        self.v = None


class Adam(AdaptiveOptimizerBase):
    """Adam: Adaptive Moment Estimation"""
    
    def __init__(self, learning_rate: float = 0.001, 
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate, eps)
        self.beta1 = beta1  # First moment decay
        self.beta2 = beta2  # Second moment decay
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement Adam update rule.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement Adam
        # Formula:
        # m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        # v_t = β₂ * v_{t-1} + (1-β₂) * g_t^2
        # m̂_t = m_t / (1 - β₁^t)  # Bias correction
        # v̂_t = v_t / (1 - β₂^t)  # Bias correction
        # x_{t+1} = x_t - η * m̂_t / (sqrt(v̂_t) + ε)
        
        raise NotImplementedError("Implement Adam update")
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None


class AMSGrad(AdaptiveOptimizerBase):
    """AMSGrad: Fixing Adam's convergence issues"""
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  # First moment estimate
        self.v = None  # Second moment estimate  
        self.v_hat_max = None  # Maximum of bias-corrected v
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement AMSGrad update rule.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
            self.v_hat_max = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement AMSGrad
        # Similar to Adam but with maximum operation:
        # v̂_t = max(v̂_{t-1}, v_t / (1 - β₂^t))
        # x_{t+1} = x_t - η * m̂_t / (sqrt(v̂_t) + ε)
        
        raise NotImplementedError("Implement AMSGrad update")
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None
        self.v_hat_max = None


class AdaBelief(AdaptiveOptimizerBase):
    """AdaBelief: Adapting Stepsizes by the Belief in Gradient Direction"""
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None  # First moment estimate (momentum)
        self.s = None  # Second moment of prediction error
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement AdaBelief update rule.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(x)
            self.s = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement AdaBelief
        # Formula:
        # m_t = β₁ * m_{t-1} + (1-β₁) * g_t
        # s_t = β₂ * s_{t-1} + (1-β₂) * (g_t - m_t)^2  # Prediction error
        # m̂_t = m_t / (1 - β₁^t)
        # ŝ_t = s_t / (1 - β₂^t)  
        # x_{t+1} = x_t - η * m̂_t / (sqrt(ŝ_t) + ε)
        
        raise NotImplementedError("Implement AdaBelief update")
    
    def reset(self):
        super().reset()
        self.m = None
        self.s = None


class AdamW(AdaptiveOptimizerBase):
    """AdamW: Adam with Decoupled Weight Decay"""
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate, eps)
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement AdamW update rule with decoupled weight decay.
        
        Args:
            x: Current parameters
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.m is None:
            self.m = np.zeros_like(x)
            self.v = np.zeros_like(x)
        
        self.step_count += 1
        
        # TODO: Implement AdamW
        # Same as Adam but weight decay is applied separately:
        # x_{t+1} = x_t - η * (m̂_t / (sqrt(v̂_t) + ε) + λ * x_t)
        # where λ is weight_decay
        
        raise NotImplementedError("Implement AdamW update")
    
    def reset(self):
        super().reset()
        self.m = None
        self.v = None


class Lookahead:
    """Lookahead wrapper for any optimizer"""
    
    def __init__(self, base_optimizer: AdaptiveOptimizerBase, 
                 k: int = 5, alpha: float = 0.5):
        """
        Args:
            base_optimizer: Fast optimizer (Adam, SGD, etc.)
            k: Update frequency for slow weights
            alpha: Interpolation factor
        """
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.slow_weights = None
        self.fast_weights = None
        self.step_count = 0
    
    def step(self, x: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """
        Implement Lookahead update rule.
        
        Args:
            x: Current parameters (slow weights)
            gradient: Gradient at current parameters
            
        Returns:
            Updated parameters
        """
        if self.slow_weights is None:
            self.slow_weights = x.copy()
            self.fast_weights = x.copy()
        
        # TODO: Implement Lookahead
        # 1. Update fast weights using base optimizer
        # 2. Every k steps, interpolate: slow = slow + α(fast - slow)
        # 3. Reset fast weights to slow weights
        
        raise NotImplementedError("Implement Lookahead wrapper")
    
    def reset(self):
        self.base_optimizer.reset()
        self.slow_weights = None
        self.fast_weights = None
        self.step_count = 0


class AdaptiveTestFunctions:
    """Test functions specifically designed for adaptive methods"""
    
    @staticmethod
    def sparse_gradient_function(x: np.ndarray, sparsity: float = 0.8) -> Tuple[float, np.ndarray]:
        """
        Function with sparse gradients (good for adaptive methods).
        
        Args:
            x: Input vector
            sparsity: Fraction of zero gradients
            
        Returns:
            Function value and sparse gradient
        """
        # TODO: Create function with sparse gradient structure
        # Hint: Randomly zero out gradient components
        
        raise NotImplementedError("Implement sparse gradient function")
    
    @staticmethod
    def varying_curvature_function(x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Function with varying curvature across dimensions.
        
        Args:
            x: Input vector
            
        Returns:
            Function value and gradient
        """
        # TODO: Create function where different dimensions have very different curvatures
        # This will test adaptive learning rate effectiveness
        
        raise NotImplementedError("Implement varying curvature function")
    
    @staticmethod
    def adam_failure_function(x: float, t: int) -> Tuple[float, float]:
        """
        Simple function where Adam fails to converge (from Reddi et al. 2018).
        
        Args:
            x: Scalar parameter
            t: Time step
            
        Returns:
            Function value and gradient
        """
        # TODO: Implement the function from the Adam non-convergence example
        # f_t(x) = 1010*x if t mod 3 == 1, else -x
        
        raise NotImplementedError("Implement Adam failure function")


def compare_adaptive_methods(x_init: np.ndarray,
                           objective_fn: Callable,
                           max_iterations: int = 1000,
                           learning_rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Compare different adaptive methods on the same optimization problem.
    
    Args:
        x_init: Initial point
        objective_fn: Objective function
        max_iterations: Maximum iterations
        learning_rates: Custom learning rates for each method
        
    Returns:
        Dictionary with results for each method
    """
    if learning_rates is None:
        learning_rates = {
            'AdaGrad': 0.1,
            'RMSprop': 0.001, 
            'Adam': 0.001,
            'AMSGrad': 0.001,
            'AdaBelief': 0.001
        }
    
    # TODO: Implement comparison of adaptive methods
    # 1. Create optimizer instances
    # 2. Run optimization with each method
    # 3. Collect convergence curves and final results
    # 4. Analyze computational efficiency
    
    raise NotImplementedError("Implement adaptive methods comparison")


def hyperparameter_sensitivity_analysis():
    """
    Analyze sensitivity of adaptive methods to hyperparameters.
    """
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 40)
    
    # TODO: Implement sensitivity analysis
    # 1. Test different values of β₁, β₂ for Adam
    # 2. Test different learning rates
    # 3. Test different ε values
    # 4. Visualize results
    
    raise NotImplementedError("Implement hyperparameter sensitivity analysis")


def numerical_stability_test():
    """
    Test numerical stability of adaptive methods.
    """
    print("Numerical Stability Test")
    print("=" * 30)
    
    # TODO: Test numerical stability
    # 1. Create functions with extreme gradient values
    # 2. Test different ε values
    # 3. Check for NaN/Inf values
    # 4. Compare single vs double precision
    
    raise NotImplementedError("Implement numerical stability test")


def adaptive_vs_nonadaptive_generalization():
    """
    Study generalization differences between adaptive and non-adaptive methods.
    """
    print("Generalization Study: Adaptive vs Non-Adaptive")
    print("=" * 50)
    
    # TODO: Implement generalization study
    # 1. Create train/validation split of optimization problem
    # 2. Compare final loss values on both sets
    # 3. Study sharpness of found minima
    # 4. Analyze learning curves
    
    raise NotImplementedError("Implement generalization study")


def learning_rate_evolution_visualization(optimizers: Dict[str, AdaptiveOptimizerBase],
                                        objective_fn: Callable,
                                        x_init: np.ndarray,
                                        max_iterations: int = 1000):
    """
    Visualize how effective learning rates evolve during training.
    
    Args:
        optimizers: Dictionary of optimizer instances
        objective_fn: Objective function
        x_init: Initial point
        max_iterations: Maximum iterations
    """
    # TODO: Implement learning rate evolution visualization
    # 1. Track effective learning rate per parameter over time
    # 2. Plot evolution for different optimizers
    # 3. Show relationship to gradient magnitude/variance
    
    raise NotImplementedError("Implement learning rate visualization")


def bias_correction_analysis():
    """
    Analyze the effect of bias correction in Adam-family methods.
    """
    print("Bias Correction Analysis")
    print("=" * 30)
    
    # TODO: Implement bias correction analysis
    # 1. Compare Adam with and without bias correction
    # 2. Show bias evolution over time
    # 3. Analyze effect on early training dynamics
    
    raise NotImplementedError("Implement bias correction analysis")


def sparse_gradients_experiment():
    """
    Test adaptive methods on problems with sparse gradients.
    """
    print("Sparse Gradients Experiment")
    print("=" * 35)
    
    # TODO: Implement sparse gradients experiment
    # 1. Create optimization problems with different sparsity levels
    # 2. Compare adaptive vs non-adaptive methods
    # 3. Analyze why adaptive methods excel here
    
    raise NotImplementedError("Implement sparse gradients experiment")


def adam_convergence_failure_demo():
    """
    Demonstrate Adam's convergence failure on the example from Reddi et al.
    """
    print("Adam Convergence Failure Demonstration")
    print("=" * 45)
    
    # TODO: Implement convergence failure demo
    # 1. Use the function from adam_failure_function
    # 2. Show Adam oscillating/diverging
    # 3. Show AMSGrad converging
    # 4. Explain the theory behind the failure
    
    raise NotImplementedError("Implement Adam failure demonstration")


if __name__ == "__main__":
    print("Adaptive Gradient Methods Implementation Exercises")
    print("=" * 60)
    
    # Example usage and testing
    x_init = np.array([1.0, 1.0])
    
    # TODO: Add your test cases here
    # 1. Test each adaptive optimizer on various functions
    # 2. Compare convergence properties
    # 3. Study hyperparameter sensitivity
    # 4. Analyze numerical stability
    # 5. Test on sparse gradient problems
    # 6. Demonstrate Adam's convergence issues
    
    print("Complete the implementation and run your experiments!")