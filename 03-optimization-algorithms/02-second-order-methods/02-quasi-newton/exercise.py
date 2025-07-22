"""
Quasi-Newton Methods Implementation Exercise

Implement BFGS, L-BFGS, and other quasi-Newton methods.
Focus on understanding second-order approximations without explicit Hessian computation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional, Dict, Deque
from abc import ABC, abstractmethod
from collections import deque
import time


class OptimizationProblem:
    """Base class for optimization problems"""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """Compute objective function value"""
        pass
    
    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        pass
    
    def hessian(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Compute Hessian (if available for comparison)"""
        return None
    
    @abstractmethod
    def optimal_point(self) -> np.ndarray:
        """Return optimal point (if known)"""
        pass


class QuadraticProblem(OptimizationProblem):
    """Quadratic function for testing convergence properties"""
    
    def __init__(self, dim: int, condition_number: float = 10.0):
        super().__init__(dim)
        self.condition_number = condition_number
        
        # TODO: Generate positive definite matrix with specified condition number
        # Use eigenvalue decomposition to control conditioning
        self.A = None
        self.b = None
        self._generate_problem_data()
    
    def _generate_problem_data(self):
        """Generate well-conditioned or ill-conditioned quadratic problem"""
        # TODO: Create matrix A with eigenvalues from 1 to condition_number
        # Generate random b vector
        pass
    
    def objective(self, x: np.ndarray) -> float:
        # TODO: Implement f(x) = 1/2 * x^T A x - b^T x
        pass
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement ∇f(x) = A x - b
        pass
    
    def hessian(self, x: np.ndarray) -> np.ndarray:
        # Hessian is constant for quadratic functions
        return self.A
    
    def optimal_point(self) -> np.ndarray:
        # TODO: Solve A x* = b
        pass


class RosenbrockProblem(OptimizationProblem):
    """Extended Rosenbrock function for testing non-convex optimization"""
    
    def __init__(self, dim: int):
        super().__init__(dim)
        if dim < 2:
            raise ValueError("Rosenbrock requires at least 2 dimensions")
    
    def objective(self, x: np.ndarray) -> float:
        # TODO: Implement Rosenbrock function
        # f(x) = sum_{i=0}^{n-2} [100(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
        pass
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement gradient of Rosenbrock
        pass
    
    def optimal_point(self) -> np.ndarray:
        return np.ones(self.dim)


class LogisticRegressionProblem(OptimizationProblem):
    """L2-regularized logistic regression"""
    
    def __init__(self, n_samples: int, dim: int, regularization: float = 0.01):
        super().__init__(dim)
        self.n_samples = n_samples
        self.regularization = regularization
        
        # TODO: Generate synthetic binary classification data
        self.features = None
        self.labels = None
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic classification dataset"""
        # TODO: Create separable but challenging dataset
        pass
    
    def objective(self, x: np.ndarray) -> float:
        # TODO: Implement logistic loss + L2 regularization
        pass
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement gradient
        pass
    
    def optimal_point(self) -> np.ndarray:
        # Use high-precision numerical solution
        from scipy.optimize import minimize
        # TODO: Cache numerical solution
        pass


class QuasiNewtonOptimizer(ABC):
    """Base class for quasi-Newton methods"""
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6,
                 line_search: bool = True):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.line_search = line_search
        
        # History tracking
        self.history = {
            'objective': [],
            'gradient_norm': [],
            'distance_to_opt': [],
            'step_size': [],
            'hessian_condition': []
        }
    
    @abstractmethod
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        """Update Hessian approximation given step s and gradient change y"""
        pass
    
    @abstractmethod
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        """Compute search direction given current gradient"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset Hessian approximation"""
        pass
    
    def optimize(self, problem: OptimizationProblem, x0: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Run quasi-Newton optimization"""
        
        x = x0.copy()
        self.reset()
        
        # Initial gradient
        grad_prev = problem.gradient(x)
        optimal_point = problem.optimal_point()
        
        for iteration in range(self.max_iterations):
            # TODO: Implement quasi-Newton iteration
            # 1. Get search direction from Hessian approximation
            # 2. Perform line search to find step size
            # 3. Update parameters
            # 4. Compute new gradient
            # 5. Update Hessian approximation using s = x_new - x_old, y = grad_new - grad_old
            # 6. Check convergence
            # 7. Record metrics
            
            pass
        
        return x, self.history
    
    def _line_search(self, problem: OptimizationProblem, x: np.ndarray, 
                    direction: np.ndarray, gradient: np.ndarray) -> float:
        """Wolfe line search"""
        # TODO: Implement strong Wolfe conditions line search
        # 1. Armijo condition: f(x + α*p) <= f(x) + c1*α*∇f^T*p
        # 2. Curvature condition: |∇f(x + α*p)^T*p| <= c2*|∇f^T*p|
        # 3. Use backtracking or interpolation
        pass


class BFGSOptimizer(QuasiNewtonOptimizer):
    """
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm
    
    Maintains full n×n Hessian approximation B_k or its inverse H_k = B_k^{-1}
    """
    
    def __init__(self, store_inverse: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.store_inverse = store_inverse
        self.H = None  # Inverse Hessian approximation
        self.B = None  # Hessian approximation
    
    def reset(self):
        """Initialize Hessian approximation to identity"""
        self.H = None
        self.B = None
    
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        """BFGS update formula"""
        
        # TODO: Implement BFGS update
        # Check curvature condition: s^T y > 0
        # If storing inverse Hessian H:
        #   H_{k+1} = (I - ρ s y^T) H_k (I - ρ y s^T) + ρ s s^T
        #   where ρ = 1 / (y^T s)
        # If storing Hessian B:
        #   B_{k+1} = B_k + (y y^T)/(y^T s) - (B_k s s^T B_k)/(s^T B_k s)
        
        pass
    
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        """Compute search direction p = -H * g"""
        
        # TODO: Initialize H to identity if first iteration
        # Return -H @ gradient
        
        pass


class LBFGSOptimizer(QuasiNewtonOptimizer):
    """
    L-BFGS (Limited-memory BFGS) algorithm
    
    Stores only m recent (s, y) pairs instead of full Hessian approximation.
    Memory requirement: O(md) instead of O(d²)
    """
    
    def __init__(self, memory_size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.s_history = deque(maxlen=memory_size)  # Step vectors
        self.y_history = deque(maxlen=memory_size)  # Gradient differences
        self.rho_history = deque(maxlen=memory_size)  # 1/(y^T s) values
    
    def reset(self):
        """Clear history"""
        self.s_history.clear()
        self.y_history.clear()
        self.rho_history.clear()
    
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        """Store (s, y) pair and compute ρ = 1/(y^T s)"""
        
        # TODO: Implement L-BFGS history update
        # 1. Check curvature condition: s^T y > 0
        # 2. Store s, y, and ρ = 1/(y^T s)
        # 3. Automatically manage memory limit via deque
        
        pass
    
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        """Two-loop recursion for L-BFGS direction"""
        
        # TODO: Implement L-BFGS two-loop recursion
        # Algorithm:
        # 1. First loop (backward): compute α_i and update q
        # 2. Initial Hessian scaling: r = H_0 * q
        # 3. Second loop (forward): compute β_i and update r
        # 4. Return -r as search direction
        
        if len(self.s_history) == 0:
            return -gradient
        
        # Two-loop recursion
        q = gradient.copy()
        alphas = []
        
        # First loop (backward through history)
        for i in range(len(self.s_history) - 1, -1, -1):
            # TODO: Implement backward loop
            pass
        
        # Initial Hessian approximation (usually scaled identity)
        r = self._get_initial_hessian_scaling() * q
        
        # Second loop (forward through history)  
        for i in range(len(self.s_history)):
            # TODO: Implement forward loop
            pass
        
        return -r
    
    def _get_initial_hessian_scaling(self) -> float:
        """Compute scaling for initial Hessian approximation"""
        # TODO: Use γ = (y^T s) / (y^T y) from most recent update
        # This provides better scaling than identity
        if len(self.y_history) == 0:
            return 1.0
        
        y = self.y_history[-1]
        s = self.s_history[-1]
        return np.dot(y, s) / np.dot(y, y)


class DFPOptimizer(QuasiNewtonOptimizer):
    """
    DFP (Davidon-Fletcher-Powell) algorithm
    
    Historical quasi-Newton method, predecessor to BFGS.
    Generally inferior to BFGS but useful for comparison.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.H = None  # Inverse Hessian approximation
    
    def reset(self):
        self.H = None
    
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        """DFP update formula"""
        
        # TODO: Implement DFP update
        # H_{k+1} = H_k + (s s^T)/(s^T y) - (H_k y y^T H_k)/(y^T H_k y)
        # Check curvature condition and handle numerical issues
        
        pass
    
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        # TODO: Initialize H if needed and return -H @ gradient
        pass


class SROptimizer(QuasiNewtonOptimizer):
    """
    SR1 (Symmetric Rank-1) update
    
    Simpler update formula but doesn't guarantee positive definiteness.
    Useful for understanding quasi-Newton method principles.
    """
    
    def __init__(self, skip_threshold: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.skip_threshold = skip_threshold
        self.B = None  # Hessian approximation
    
    def reset(self):
        self.B = None
    
    def update_hessian_approximation(self, s: np.ndarray, y: np.ndarray):
        """SR1 update formula"""
        
        # TODO: Implement SR1 update
        # B_{k+1} = B_k + ((y - B_k s)(y - B_k s)^T) / ((y - B_k s)^T s)
        # Skip update if denominator is too small (near singularity)
        
        pass
    
    def get_search_direction(self, gradient: np.ndarray) -> np.ndarray:
        # TODO: Solve B * p = -g using appropriate linear solver
        # Handle potential indefiniteness of B
        pass


def compare_quasi_newton_methods(problem: OptimizationProblem,
                               methods: Dict[str, QuasiNewtonOptimizer],
                               x0: np.ndarray) -> Dict:
    """Compare different quasi-Newton methods"""
    
    results = {}
    
    for name, optimizer in methods.items():
        print(f"Running {name}...")
        start_time = time.time()
        x_final, history = optimizer.optimize(problem, x0)
        end_time = time.time()
        
        results[name] = {
            'final_point': x_final,
            'history': history,
            'runtime': end_time - start_time,
            'final_objective': problem.objective(x_final),
            'final_gradient_norm': np.linalg.norm(problem.gradient(x_final)),
            'iterations': len(history['objective'])
        }
    
    return results


def memory_usage_study(problem_dimensions: List[int]) -> Dict:
    """Study memory usage scaling of different quasi-Newton methods"""
    
    results = {
        'dimensions': problem_dimensions,
        'bfgs_memory': [],
        'lbfgs_memory': [],
        'theoretical_bfgs': [],
        'theoretical_lbfgs': []
    }
    
    for dim in problem_dimensions:
        # TODO: Create problems of different dimensions
        # Measure actual memory usage of BFGS vs L-BFGS
        # Compare with theoretical O(d²) vs O(md) scaling
        
        pass
    
    return results


def convergence_rate_analysis(problem: OptimizationProblem,
                            optimizer: QuasiNewtonOptimizer,
                            x0: np.ndarray) -> Dict:
    """Analyze convergence rate properties"""
    
    x_final, history = optimizer.optimize(problem, x0)
    
    # TODO: Analyze convergence rate
    # 1. Fit exponential/polynomial models to convergence curves
    # 2. Estimate convergence rate
    # 3. Compare with theoretical predictions
    
    analysis = {
        'linear_rate': None,
        'superlinear_evidence': None,
        'quadratic_phase': None
    }
    
    return analysis


def condition_number_experiment(base_condition: float, 
                              multiples: List[float]) -> Dict:
    """Study effect of problem conditioning on quasi-Newton methods"""
    
    results = {}
    
    for mult in multiples:
        condition_number = base_condition * mult
        print(f"Testing condition number: {condition_number}")
        
        # TODO: Create quadratic problems with different condition numbers
        # Test BFGS and L-BFGS performance
        # Record convergence rate vs conditioning
        
        pass
    
    return results


def plot_convergence_comparison(results: Dict, problem_name: str):
    """Create comprehensive convergence plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Objective convergence
    ax = axes[0, 0]
    for name, result in results.items():
        history = result['history']
        if 'objective' in history:
            ax.semilogy(history['objective'], label=name)
    ax.set_title('Objective Convergence')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('f(x) - f*')
    ax.legend()
    ax.grid(True)
    
    # Gradient norm
    ax = axes[0, 1]
    for name, result in results.items():
        history = result['history']
        if 'gradient_norm' in history:
            ax.semilogy(history['gradient_norm'], label=name)
    ax.set_title('Gradient Norm')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||∇f(x)||')
    ax.legend()
    ax.grid(True)
    
    # Distance to optimum
    ax = axes[0, 2]
    for name, result in results.items():
        history = result['history']
        if 'distance_to_opt' in history:
            ax.semilogy(history['distance_to_opt'], label=name)
    ax.set_title('Distance to Optimum')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||x - x*||')
    ax.legend()
    ax.grid(True)
    
    # Runtime comparison
    ax = axes[1, 0]
    names = list(results.keys())
    runtimes = [results[name]['runtime'] for name in names]
    iterations = [results[name]['iterations'] for name in names]
    ax.bar(names, runtimes)
    ax.set_title('Total Runtime')
    ax.set_ylabel('Time (seconds)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Iterations to convergence
    ax = axes[1, 1]
    ax.bar(names, iterations)
    ax.set_title('Iterations to Convergence')
    ax.set_ylabel('Iterations')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Performance summary
    ax = axes[1, 2]
    final_objectives = [results[name]['final_objective'] for name in names]
    ax.bar(names, final_objectives)
    ax.set_title('Final Objective Value')
    ax.set_ylabel('f(x_final)')
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.suptitle(f'Quasi-Newton Methods Comparison: {problem_name}')
    plt.show()


# ============================================================================
# EXERCISES
# ============================================================================

def exercise_1_bfgs_implementation():
    """
    Exercise 1: Implement and test BFGS algorithm
    
    Tasks:
    1. Complete BFGSOptimizer implementation
    2. Test on quadratic and Rosenbrock functions
    3. Verify superlinear convergence
    4. Compare storing H vs B
    """
    
    print("=== Exercise 1: BFGS Implementation ===")
    
    # TODO: Test BFGS on different problems
    # Verify convergence properties
    # Compare with gradient descent and Newton's method
    
    pass


def exercise_2_lbfgs_efficiency():
    """
    Exercise 2: L-BFGS memory efficiency
    
    Tasks:
    1. Complete LBFGSOptimizer with two-loop recursion
    2. Study memory usage vs problem dimension
    3. Test effect of memory parameter m
    4. Compare with full BFGS on large problems
    """
    
    print("=== Exercise 2: L-BFGS Memory Efficiency ===")
    
    # TODO: Implement L-BFGS and study memory efficiency
    # Test on high-dimensional problems
    
    pass


def exercise_3_method_comparison():
    """
    Exercise 3: Compare different quasi-Newton methods
    
    Tasks:
    1. Implement DFP and SR1 methods
    2. Compare all methods on various problems
    3. Study convergence rates and robustness
    4. Analyze when each method works best
    """
    
    print("=== Exercise 3: Method Comparison ===")
    
    # TODO: Comprehensive comparison of quasi-Newton methods
    # Analyze strengths and weaknesses of each
    
    pass


def exercise_4_line_search_impact():
    """
    Exercise 4: Effect of line search on quasi-Newton methods
    
    Tasks:
    1. Implement Wolfe line search
    2. Compare exact vs inexact line search
    3. Study effect on Hessian approximation quality
    4. Test different line search parameters
    """
    
    print("=== Exercise 4: Line Search Analysis ===")
    
    # TODO: Study line search effects on quasi-Newton methods
    # Compare different line search strategies
    
    pass


def exercise_5_conditioning_analysis():
    """
    Exercise 5: Effect of problem conditioning
    
    Tasks:
    1. Test on problems with different condition numbers
    2. Study how ill-conditioning affects convergence
    3. Compare with Newton's method sensitivity
    4. Investigate preconditioning strategies
    """
    
    print("=== Exercise 5: Conditioning Analysis ===")
    
    # TODO: Systematic study of conditioning effects
    # Compare robustness of different quasi-Newton methods
    
    pass


def exercise_6_practical_considerations():
    """
    Exercise 6: Practical implementation considerations
    
    Tasks:
    1. Handle numerical edge cases (e.g., negative curvature)
    2. Implement restart strategies
    3. Study parallel computation opportunities
    4. Create robust, production-ready implementation
    """
    
    print("=== Exercise 6: Practical Considerations ===")
    
    # TODO: Address practical implementation challenges
    # Create robust, efficient implementations
    
    pass


if __name__ == "__main__":
    # Run all exercises
    exercise_1_bfgs_implementation()
    exercise_2_lbfgs_efficiency()
    exercise_3_method_comparison()
    exercise_4_line_search_impact()
    exercise_5_conditioning_analysis()
    exercise_6_practical_considerations()
    
    print("\nAll exercises completed!")
    print("Key insights to understand:")
    print("1. Trade-offs between memory and convergence speed")
    print("2. When quasi-Newton methods excel over first/second-order methods")
    print("3. Importance of line search for Hessian approximation quality")
    print("4. Robustness vs efficiency considerations in practice")