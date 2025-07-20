"""
Solution implementations for Rademacher Complexity exercises.

This file provides complete implementations of all TODO items in exercise.py.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings


class FunctionClass:
    """
    Base class for function classes in Rademacher complexity analysis.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, f_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Evaluate function on data.
        
        Args:
            f_params: Function parameters
            X: Input data
            
        Returns:
            Function values
        """
        raise NotImplementedError
    
    def sample_function(self) -> np.ndarray:
        """Sample random function parameters."""
        raise NotImplementedError


class LinearFunctions(FunctionClass):
    """
    Linear functions: f(x) = w^T x with ||w|| ≤ B
    """
    
    def __init__(self, dimension: int, bound: float = 1.0):
        super().__init__(f"Linear Functions ||w|| ≤ {bound}")
        self.dimension = dimension
        self.bound = bound
    
    def evaluate(self, f_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate linear function."""
        return X @ f_params
    
    def sample_function(self) -> np.ndarray:
        """Sample random linear function."""
        w = np.random.randn(self.dimension)
        # Normalize to satisfy bound constraint
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            return self.bound * w / w_norm
        return w
    
    def theoretical_rademacher_complexity(self, X: np.ndarray) -> float:
        """
        Compute theoretical Rademacher complexity.
        
        For linear functions with ||w|| ≤ B:
        R̂_S(F) = B * E[||∑_i σ_i x_i||] / m
        
        Args:
            X: Data matrix (m × d)
            
        Returns:
            Theoretical Rademacher complexity
        """
        m = X.shape[0]
        
        # Monte Carlo estimate of E[||∑_i σ_i x_i||]
        n_samples = 1000
        expectations = []
        
        for _ in range(n_samples):
            sigma = np.random.choice([-1, 1], size=m)
            weighted_sum = np.sum(sigma[:, np.newaxis] * X, axis=0)
            expectations.append(np.linalg.norm(weighted_sum))
        
        expected_norm = np.mean(expectations)
        return self.bound * expected_norm / m


class FiniteFunctionClass(FunctionClass):
    """
    Finite function class with |F| functions.
    """
    
    def __init__(self, functions: List[np.ndarray], name: str = "Finite Class"):
        super().__init__(name)
        self.functions = functions
        self.size = len(functions)
    
    def evaluate(self, f_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Evaluate function by index."""
        idx = int(f_params[0]) % self.size  # Ensure valid index
        return self.functions[idx]
    
    def sample_function(self) -> np.ndarray:
        """Sample random function index."""
        return np.array([np.random.randint(self.size)])
    
    def theoretical_rademacher_complexity(self, X: np.ndarray) -> float:
        """
        Compute theoretical Rademacher complexity using Massart's lemma.
        
        For finite class: R̂_S(F) ≤ √(2 log |F|) * max_f ||f||_2 / m
        
        Args:
            X: Data matrix (not used for finite classes)
            
        Returns:
            Theoretical Rademacher complexity bound
        """
        m = len(X) if X.ndim > 1 else len(X)
        
        # Compute maximum L2 norm among functions
        max_norm = 0
        for f in self.functions:
            max_norm = max(max_norm, np.linalg.norm(f))
        
        # Massart's lemma bound
        return np.sqrt(2 * np.log(self.size)) * max_norm / m


def empirical_rademacher_complexity(function_class: FunctionClass,
                                  X: np.ndarray,
                                  n_samples: int = 1000) -> Tuple[float, float]:
    """
    Estimate empirical Rademacher complexity via Monte Carlo.
    
    R̂_S(F) = E_σ[sup_{f∈F} (1/m) ∑_i σ_i f(x_i)]
    
    Args:
        function_class: Function class to analyze
        X: Data points (m × d)
        n_samples: Number of Monte Carlo samples
        
    Returns:
        (mean_estimate, std_estimate)
    """
    m = X.shape[0]
    suprema = []
    
    for _ in range(n_samples):
        # Sample Rademacher variables
        sigma = np.random.choice([-1, 1], size=m)
        
        # Find supremum over function class
        max_value = -np.inf
        
        # Sample functions from the class
        n_function_samples = 200  # Number of functions to sample
        for _ in range(n_function_samples):
            f_params = function_class.sample_function()
            f_values = function_class.evaluate(f_params, X)
            
            # Compute empirical average
            empirical_avg = np.mean(sigma * f_values)
            max_value = max(max_value, empirical_avg)
        
        suprema.append(max_value)
    
    return np.mean(suprema), np.std(suprema)


def rademacher_generalization_bound(empirical_risk: float,
                                  rademacher_complexity: float,
                                  confidence: float = 0.05,
                                  sample_size: int = None) -> float:
    """
    Compute Rademacher generalization bound.
    
    With probability ≥ 1-δ:
    R(f) ≤ R̂(f) + 2R̂_m(F) + √(log(2/δ)/(2m))
    
    Args:
        empirical_risk: Empirical risk R̂(f)
        rademacher_complexity: Rademacher complexity R̂_m(F)
        confidence: Confidence level δ
        sample_size: Sample size m
        
    Returns:
        Upper bound on true risk
    """
    if sample_size is None:
        # Default sample size if not provided
        sample_size = 100
    
    # Confidence term
    confidence_term = np.sqrt(np.log(2 / confidence) / (2 * sample_size))
    
    # Rademacher bound
    return empirical_risk + 2 * rademacher_complexity + confidence_term


class RadmacherComplexityAnalyzer:
    """
    Analyze Rademacher complexity and its properties.
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze_scaling_with_sample_size(self, function_class: FunctionClass,
                                       data_generator: Callable[[int], np.ndarray],
                                       sample_sizes: List[int],
                                       n_trials: int = 100) -> Dict:
        """
        Analyze how Rademacher complexity scales with sample size.
        
        Args:
            function_class: Function class to analyze
            data_generator: Generates data of given size
            sample_sizes: Sample sizes to test
            n_trials: Number of trials per sample size
            
        Returns:
            Dictionary with scaling analysis results
        """
        results = {
            'sample_sizes': sample_sizes,
            'complexities': [],
            'std_errors': [],
            'theoretical_bounds': []
        }
        
        for m in sample_sizes:
            complexities_at_m = []
            
            for trial in range(n_trials):
                X = data_generator(m)
                complexity, _ = empirical_rademacher_complexity(
                    function_class, X, n_samples=50
                )
                complexities_at_m.append(complexity)
            
            mean_complexity = np.mean(complexities_at_m)
            std_error = np.std(complexities_at_m) / np.sqrt(n_trials)
            
            results['complexities'].append(mean_complexity)
            results['std_errors'].append(std_error)
            
            # Theoretical bound (if available)
            if hasattr(function_class, 'theoretical_rademacher_complexity'):
                X_sample = data_generator(m)
                theoretical = function_class.theoretical_rademacher_complexity(X_sample)
                results['theoretical_bounds'].append(theoretical)
            else:
                results['theoretical_bounds'].append(None)
        
        return results
    
    def compare_theoretical_empirical(self, function_class: FunctionClass,
                                    X: np.ndarray,
                                    n_monte_carlo: int = 1000) -> Dict:
        """
        Compare theoretical and empirical Rademacher complexity.
        
        Args:
            function_class: Function class with theoretical bound
            X: Data matrix
            n_monte_carlo: Number of Monte Carlo samples
            
        Returns:
            Comparison results
        """
        # Empirical estimate
        empirical, empirical_std = empirical_rademacher_complexity(
            function_class, X, n_monte_carlo
        )
        
        # Theoretical bound
        if hasattr(function_class, 'theoretical_rademacher_complexity'):
            theoretical = function_class.theoretical_rademacher_complexity(X)
        else:
            theoretical = None
        
        return {
            'empirical': empirical,
            'empirical_std': empirical_std,
            'theoretical': theoretical,
            'ratio': empirical / theoretical if theoretical else None,
            'sample_size': X.shape[0]
        }
    
    def study_composition_properties(self, base_classes: List[FunctionClass],
                                   X: np.ndarray) -> Dict:
        """
        Study Rademacher complexity under function composition.
        
        Analyze:
        1. Scaling: R_m(cF) = |c|R_m(F)
        2. Convex hull: R_m(conv(F)) = R_m(F)
        3. Sum: R_m(F1 + F2) ≤ R_m(F1) + R_m(F2)
        
        Args:
            base_classes: Base function classes
            X: Data points
            
        Returns:
            Composition property results
        """
        results = {}
        
        # Test scaling property
        if len(base_classes) > 0:
            base_class = base_classes[0]
            base_complexity, _ = empirical_rademacher_complexity(base_class, X, 100)
            
            # Create scaled version
            class ScaledClass(FunctionClass):
                def __init__(self, base_class, scale):
                    super().__init__(f"Scaled {base_class.name}")
                    self.base_class = base_class
                    self.scale = scale
                
                def evaluate(self, f_params, X):
                    return self.scale * self.base_class.evaluate(f_params, X)
                
                def sample_function(self):
                    return self.base_class.sample_function()
            
            scale = 2.0
            scaled_class = ScaledClass(base_class, scale)
            scaled_complexity, _ = empirical_rademacher_complexity(scaled_class, X, 100)
            
            results['scaling_property'] = {
                'base_complexity': base_complexity,
                'scaled_complexity': scaled_complexity,
                'scale_factor': scale,
                'theoretical_ratio': scale,
                'empirical_ratio': scaled_complexity / base_complexity if base_complexity > 0 else None
            }
        
        # Test sum property (simplified)
        if len(base_classes) >= 2:
            class1, class2 = base_classes[0], base_classes[1]
            
            complexity1, _ = empirical_rademacher_complexity(class1, X, 100)
            complexity2, _ = empirical_rademacher_complexity(class2, X, 100)
            
            # Create sum class
            class SumClass(FunctionClass):
                def __init__(self, class1, class2):
                    super().__init__(f"Sum {class1.name} + {class2.name}")
                    self.class1 = class1
                    self.class2 = class2
                
                def evaluate(self, f_params, X):
                    # Split parameters
                    mid = len(f_params) // 2
                    params1 = f_params[:mid] if mid > 0 else self.class1.sample_function()
                    params2 = f_params[mid:] if len(f_params) > mid else self.class2.sample_function()
                    
                    return (self.class1.evaluate(params1, X) + 
                            self.class2.evaluate(params2, X))
                
                def sample_function(self):
                    params1 = self.class1.sample_function()
                    params2 = self.class2.sample_function()
                    return np.concatenate([params1, params2])
            
            sum_class = SumClass(class1, class2)
            sum_complexity, _ = empirical_rademacher_complexity(sum_class, X, 100)
            
            results['sum_property'] = {
                'complexity1': complexity1,
                'complexity2': complexity2,
                'sum_complexity': sum_complexity,
                'theoretical_bound': complexity1 + complexity2,
                'bound_satisfied': sum_complexity <= complexity1 + complexity2 + 0.1
            }
        
        return results


def symmetrization_lemma_verification(function_class: FunctionClass,
                                    distribution: Callable[[], np.ndarray],
                                    sample_size: int = 100,
                                    n_trials: int = 1000) -> Dict:
    """
    Verify symmetrization lemma empirically.
    
    Lemma: E[sup_f |E[f] - E_S[f]|] ≤ 2R_m(F)
    
    Args:
        function_class: Function class to test
        distribution: Data distribution
        sample_size: Sample size for each trial
        n_trials: Number of trials
        
    Returns:
        Verification results
    """
    generalization_gaps = []
    
    for trial in range(n_trials):
        # Generate sample
        S = np.array([distribution() for _ in range(sample_size)])
        
        # Estimate generalization gap
        max_gap = 0
        
        for _ in range(50):  # Sample functions
            f_params = function_class.sample_function()
            
            # Empirical average on S
            f_values_S = function_class.evaluate(f_params, S)
            empirical_avg = np.mean(f_values_S)
            
            # True expectation (estimate with large sample)
            large_sample = np.array([distribution() for _ in range(1000)])
            f_values_large = function_class.evaluate(f_params, large_sample)
            true_expectation = np.mean(f_values_large)
            
            gap = abs(empirical_avg - true_expectation)
            max_gap = max(max_gap, gap)
        
        generalization_gaps.append(max_gap)
    
    # Estimate Rademacher complexity
    sample_for_rad = np.array([distribution() for _ in range(sample_size)])
    rademacher_complexity, _ = empirical_rademacher_complexity(
        function_class, sample_for_rad, n_samples=200
    )
    
    expected_gap = np.mean(generalization_gaps)
    rademacher_bound = 2 * rademacher_complexity
    
    return {
        'expected_generalization_gap': expected_gap,
        'rademacher_bound': rademacher_bound,
        'lemma_satisfied': expected_gap <= rademacher_bound + 0.1,
        'generalization_gap': expected_gap,
        'bound_tightness': expected_gap / rademacher_bound if rademacher_bound > 0 else None
    }


class GaussianComplexity:
    """
    Gaussian complexity computation and comparison with Rademacher.
    """
    
    def __init__(self):
        pass
    
    def compute_gaussian_complexity(self, function_class: FunctionClass,
                                  X: np.ndarray,
                                  n_samples: int = 1000) -> float:
        """
        Compute Gaussian complexity.
        
        γ_m(F) = E_g[sup_{f∈F} (1/m) ∑_i g_i f(x_i)]
        where g_i ~ N(0,1) are independent.
        
        Args:
            function_class: Function class
            X: Data points
            n_samples: Number of Gaussian samples
            
        Returns:
            Gaussian complexity estimate
        """
        m = X.shape[0]
        suprema = []
        
        for _ in range(n_samples):
            # Sample Gaussian variables
            g = np.random.randn(m)
            
            # Find supremum over function class
            max_value = -np.inf
            
            # Sample functions
            for _ in range(200):
                f_params = function_class.sample_function()
                f_values = function_class.evaluate(f_params, X)
                
                empirical_avg = np.mean(g * f_values)
                max_value = max(max_value, empirical_avg)
            
            suprema.append(max_value)
        
        return np.mean(suprema)
    
    def compare_rademacher_gaussian(self, function_class: FunctionClass,
                                  X: np.ndarray) -> Dict:
        """
        Compare Rademacher and Gaussian complexities.
        
        Theory: R_m(F) ≤ √(π/2) γ_m(F)
        
        Args:
            function_class: Function class
            X: Data points
            
        Returns:
            Comparison results
        """
        # Compute both complexities
        rademacher, _ = empirical_rademacher_complexity(function_class, X, 500)
        gaussian = self.compute_gaussian_complexity(function_class, X, 500)
        
        theoretical_ratio = np.sqrt(np.pi / 2)
        empirical_ratio = rademacher / gaussian if gaussian > 0 else None
        
        return {
            'rademacher': rademacher,
            'gaussian': gaussian,
            'theoretical_bound': theoretical_ratio,
            'empirical_ratio': empirical_ratio,
            'bound_satisfied': empirical_ratio <= theoretical_ratio + 0.1 if empirical_ratio else None
        }


def local_rademacher_complexity(function_class: FunctionClass,
                              X: np.ndarray,
                              center_function: np.ndarray,
                              radius: float,
                              n_samples: int = 1000) -> float:
    """
    Compute local Rademacher complexity.
    
    R_m(F ∩ B(f̂, r)) for functions within radius r of center.
    
    Args:
        function_class: Function class
        X: Data points
        center_function: Center function parameters
        radius: Ball radius
        n_samples: Monte Carlo samples
        
    Returns:
        Local Rademacher complexity
    """
    m = X.shape[0]
    suprema = []
    
    # Center function values
    center_values = function_class.evaluate(center_function, X)
    
    for _ in range(n_samples):
        # Sample Rademacher variables
        sigma = np.random.choice([-1, 1], size=m)
        
        max_value = -np.inf
        
        # Sample functions within the ball
        for _ in range(200):
            f_params = function_class.sample_function()
            f_values = function_class.evaluate(f_params, X)
            
            # Check if function is within radius
            distance = np.linalg.norm(f_values - center_values)
            if distance <= radius:
                empirical_avg = np.mean(sigma * f_values)
                max_value = max(max_value, empirical_avg)
        
        if max_value > -np.inf:
            suprema.append(max_value)
    
    return np.mean(suprema) if suprema else 0.0


def chaining_bound_verification():
    """
    Verify Dudley's chaining bound.
    
    Implement generic chaining for simple function classes and
    verify against direct Rademacher computation.
    
    Returns:
        Verification results
    """
    # Simple example: finite function class
    functions = [np.random.randn(20) for _ in range(8)]
    finite_class = FiniteFunctionClass(functions)
    
    # Generate data
    X = np.random.randn(20, 1)  # Dummy X for finite class
    
    # Direct Rademacher complexity
    direct_complexity, _ = empirical_rademacher_complexity(finite_class, X, 500)
    
    # Chaining bound (simplified Dudley bound for finite classes)
    # For finite class: bound ≤ √(log |F|) * max_norm / √m
    max_norm = max(np.linalg.norm(f) for f in functions)
    m = X.shape[0]
    chaining_bound = np.sqrt(np.log(len(functions))) * max_norm / np.sqrt(m)
    
    return {
        'empirical_complexity': direct_complexity,
        'chaining_bound': chaining_bound,
        'bound_satisfied': direct_complexity <= chaining_bound + 0.1,
        'bound_tightness': direct_complexity / chaining_bound if chaining_bound > 0 else None
    }


class StabilityRadmacherConnection:
    """
    Explore connection between algorithmic stability and Rademacher complexity.
    """
    
    def __init__(self):
        pass
    
    def analyze_stable_algorithm(self, algorithm: Callable,
                                data_generator: Callable,
                                stability_parameter: float,
                                sample_sizes: List[int]) -> Dict:
        """
        Analyze Rademacher complexity of stable algorithms.
        
        If algorithm has uniform stability β, then the class of functions
        it can output has Rademacher complexity ≤ β.
        
        Args:
            algorithm: Learning algorithm
            data_generator: Generates training data
            stability_parameter: Stability parameter β
            sample_sizes: Sample sizes to test
            
        Returns:
            Stability-Rademacher analysis results
        """
        results = {
            'sample_sizes': sample_sizes,
            'complexity_bounds': [],
            'empirical_stabilities': [],
            'theoretical_bound': stability_parameter
        }
        
        for m in sample_sizes:
            # Generate dataset
            X, y = data_generator()
            X = X[:m]
            y = y[:m]
            
            # Learn function
            learned_params = algorithm(X, y)
            
            # Estimate empirical stability
            stability_estimates = []
            for _ in range(20):
                # Perturb one example
                X_pert = X.copy()
                y_pert = y.copy()
                idx = np.random.randint(m)
                
                # Replace with new example
                new_x, new_y = data_generator()
                X_pert[idx] = new_x[0]
                y_pert[idx] = new_y[0]
                
                # Learn on perturbed dataset
                learned_params_pert = algorithm(X_pert, y_pert)
                
                # Compute stability (simplified as parameter difference)
                stability = np.linalg.norm(learned_params - learned_params_pert)
                stability_estimates.append(stability)
            
            empirical_stability = np.mean(stability_estimates)
            results['empirical_stabilities'].append(empirical_stability)
            
            # Rademacher complexity bound
            complexity_bound = min(stability_parameter, empirical_stability)
            results['complexity_bounds'].append(complexity_bound)
        
        return results


def kernel_rademacher_complexity(kernel_matrix: np.ndarray,
                               radius: float = 1.0) -> float:
    """
    Compute Rademacher complexity for kernel methods.
    
    For RKHS ball of radius R:
    R̂_m(B_R) ≤ R√(tr(K)/m)
    
    Args:
        kernel_matrix: Kernel matrix K
        radius: RKHS ball radius
        
    Returns:
        Kernel Rademacher complexity bound
    """
    m = kernel_matrix.shape[0]
    trace_K = np.trace(kernel_matrix)
    
    return radius * np.sqrt(trace_K / m)


def neural_network_rademacher_analysis(layer_widths: List[int],
                                     spectral_norms: List[float],
                                     input_bound: float,
                                     sample_size: int) -> float:
    """
    Analyze Rademacher complexity of neural networks.
    
    Use spectral norm bounds for deep networks.
    
    Args:
        layer_widths: Width of each layer
        spectral_norms: Spectral norm bound for each layer
        input_bound: Bound on input norm
        sample_size: Sample size
        
    Returns:
        Neural network Rademacher complexity bound
    """
    # Product of spectral norms
    spectral_product = np.prod(spectral_norms)
    
    # Depth-dependent factor (simplified)
    depth = len(layer_widths) - 1
    depth_factor = np.sqrt(2 * np.log(depth)) if depth > 1 else 1.0
    
    # Width-dependent factor
    max_width = max(layer_widths)
    width_factor = np.sqrt(np.log(max_width))
    
    # Final bound (simplified version of sophisticated bounds)
    return (spectral_product * input_bound * depth_factor * width_factor / 
            np.sqrt(sample_size))


def visualize_rademacher_complexity(function_classes: List[FunctionClass],
                                  sample_sizes: np.ndarray,
                                  data_generator: Callable):
    """
    Visualize Rademacher complexity vs sample size.
    
    Create plots showing:
    1. Empirical complexity vs sample size
    2. Theoretical bounds
    3. Comparison across function classes
    
    Args:
        function_classes: Classes to compare
        sample_sizes: Range of sample sizes
        data_generator: Generates data
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, func_class in enumerate(function_classes):
        empirical_complexities = []
        theoretical_bounds = []
        
        for m in sample_sizes:
            # Generate data
            X = data_generator(int(m))
            
            # Empirical complexity
            emp_complex, _ = empirical_rademacher_complexity(func_class, X, 100)
            empirical_complexities.append(emp_complex)
            
            # Theoretical bound
            if hasattr(func_class, 'theoretical_rademacher_complexity'):
                theoretical = func_class.theoretical_rademacher_complexity(X)
                theoretical_bounds.append(theoretical)
            else:
                theoretical_bounds.append(None)
        
        color = colors[i % len(colors)]
        
        # Plot empirical
        plt.plot(sample_sizes, empirical_complexities, 
                'o-', color=color, label=f'{func_class.name} (Empirical)')
        
        # Plot theoretical if available
        if all(b is not None for b in theoretical_bounds):
            plt.plot(sample_sizes, theoretical_bounds, 
                    '--', color=color, alpha=0.7, 
                    label=f'{func_class.name} (Theoretical)')
    
    plt.xlabel('Sample Size (m)')
    plt.ylabel('Rademacher Complexity')
    plt.title('Rademacher Complexity vs Sample Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()


# Export all solution implementations
__all__ = [
    'FunctionClass', 'LinearFunctions', 'FiniteFunctionClass',
    'empirical_rademacher_complexity', 'rademacher_generalization_bound',
    'RadmacherComplexityAnalyzer', 'symmetrization_lemma_verification',
    'GaussianComplexity', 'local_rademacher_complexity',
    'chaining_bound_verification', 'StabilityRadmacherConnection',
    'kernel_rademacher_complexity', 'neural_network_rademacher_analysis',
    'visualize_rademacher_complexity'
]