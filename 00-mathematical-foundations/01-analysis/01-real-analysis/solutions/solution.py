"""
Real Analysis Solutions - Reference Implementation

This file contains complete solutions to the exercises. 
Try to implement everything yourself first before looking at these!
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt


class MetricSpace:
    """Base class for metric spaces."""
    
    def __init__(self, distance_func: Callable[[np.ndarray, np.ndarray], float]):
        self.d = distance_func
    
    def verify_metric_properties(self, points: List[np.ndarray], tolerance: float = 1e-10) -> bool:
        """Verify the four metric properties."""
        n = len(points)
        
        # Check all combinations of points
        for i in range(n):
            for j in range(n):
                # Property 1: Non-negativity
                if self.d(points[i], points[j]) < -tolerance:
                    return False
                
                # Property 2: Identity of indiscernibles
                if i == j:
                    if abs(self.d(points[i], points[j])) > tolerance:
                        return False
                else:
                    if self.d(points[i], points[j]) <= tolerance:
                        # Different points should have positive distance
                        if not np.allclose(points[i], points[j], atol=tolerance):
                            return False
                
                # Property 3: Symmetry
                if abs(self.d(points[i], points[j]) - self.d(points[j], points[i])) > tolerance:
                    return False
                
                # Property 4: Triangle inequality
                for k in range(n):
                    if self.d(points[i], points[k]) > self.d(points[i], points[j]) + self.d(points[j], points[k]) + tolerance:
                        return False
        
        return True


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean (L2) distance."""
    return np.sqrt(np.sum((x - y)**2))


def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Manhattan (L1) distance."""
    return np.sum(np.abs(x - y))


def chebyshev_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Chebyshev (L∞) distance."""
    return np.max(np.abs(x - y))


class Sequence:
    """Class for analyzing sequences in metric spaces."""
    
    def __init__(self, terms: Callable[[int], np.ndarray], metric_space: MetricSpace):
        self.terms = terms
        self.metric_space = metric_space
    
    def check_convergence(self, candidate_limit: np.ndarray, epsilon: float = 1e-6, 
                         max_n: int = 10000) -> Tuple[bool, Optional[int]]:
        """Check if sequence converges to the candidate limit."""
        for n in range(1, max_n + 1):
            term_n = self.terms(n)
            distance = self.metric_space.d(term_n, candidate_limit)
            
            # Check if we've found N such that all subsequent terms are within epsilon
            if distance < epsilon:
                # Verify next few terms to be sure
                all_close = True
                for m in range(n, min(n + 100, max_n + 1)):
                    if self.metric_space.d(self.terms(m), candidate_limit) >= epsilon:
                        all_close = False
                        break
                
                if all_close:
                    return True, n
        
        return False, None
    
    def is_cauchy(self, epsilon: float = 1e-6, max_n: int = 10000) -> Tuple[bool, Optional[int]]:
        """Check if sequence is Cauchy."""
        # For efficiency, we check if terms stabilize
        for n in range(1, max_n):
            term_n = self.terms(n)
            
            # Check if all pairs of terms after n are within epsilon
            is_cauchy_from_n = True
            for i in range(10):  # Check next 10 terms
                for j in range(i + 1, 10):
                    if n + i > max_n or n + j > max_n:
                        break
                    term_i = self.terms(n + i)
                    term_j = self.terms(n + j)
                    if self.metric_space.d(term_i, term_j) >= epsilon:
                        is_cauchy_from_n = False
                        break
                if not is_cauchy_from_n:
                    break
            
            if is_cauchy_from_n:
                return True, n
        
        return False, None


class ContinuousFunction:
    """Class for analyzing continuous functions between metric spaces."""
    
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], 
                 domain_metric: MetricSpace, codomain_metric: MetricSpace):
        self.f = f
        self.domain_metric = domain_metric
        self.codomain_metric = codomain_metric
    
    def check_continuity_at_point(self, x0: np.ndarray, epsilon: float = 0.1, 
                                  delta_search_iters: int = 100) -> Tuple[bool, Optional[float]]:
        """Check if f is continuous at x0 using epsilon-delta definition."""
        f_x0 = self.f(x0)
        
        # Binary search for delta
        delta_min, delta_max = 0, 1
        
        for _ in range(delta_search_iters):
            delta = (delta_min + delta_max) / 2
            
            # Test continuity with current delta
            works = True
            
            # Sample points in delta-neighborhood
            for _ in range(100):
                # Generate random direction
                if x0.shape[0] == 1:
                    direction = np.array([1.0])
                else:
                    direction = np.random.randn(*x0.shape)
                    direction = direction / np.linalg.norm(direction)
                
                # Scale to be within delta
                scale = np.random.uniform(0, delta)
                x = x0 + scale * direction
                
                # Check if f(x) is within epsilon of f(x0)
                if self.codomain_metric.d(self.f(x), f_x0) >= epsilon:
                    works = False
                    break
            
            if works:
                delta_min = delta
            else:
                delta_max = delta
        
        # Final verification
        final_delta = delta_min
        if final_delta > 1e-10:
            return True, final_delta
        else:
            return False, None
    
    def check_uniform_continuity(self, domain_points: List[np.ndarray], 
                                epsilon: float = 0.1) -> Tuple[bool, Optional[float]]:
        """Check if f is uniformly continuous on given domain points."""
        # Find the minimum delta that works for all points
        min_delta = float('inf')
        
        for x in domain_points:
            is_cont, delta = self.check_continuity_at_point(x, epsilon)
            if not is_cont:
                return False, None
            min_delta = min(min_delta, delta)
        
        # Verify this delta works uniformly
        for i in range(len(domain_points)):
            for j in range(i + 1, len(domain_points)):
                if self.domain_metric.d(domain_points[i], domain_points[j]) < min_delta:
                    f_dist = self.codomain_metric.d(
                        self.f(domain_points[i]), 
                        self.f(domain_points[j])
                    )
                    if f_dist >= epsilon:
                        return False, None
        
        return True, min_delta


class FixedPointIterator:
    """Implements fixed point iteration for contraction mappings."""
    
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], metric_space: MetricSpace):
        self.f = f
        self.metric_space = metric_space
    
    def estimate_lipschitz_constant(self, sample_points: List[np.ndarray]) -> float:
        """Estimate the Lipschitz constant of f."""
        L = 0
        n = len(sample_points)
        
        for i in range(n):
            for j in range(i + 1, n):
                x, y = sample_points[i], sample_points[j]
                if not np.allclose(x, y):
                    ratio = self.metric_space.d(self.f(x), self.f(y)) / self.metric_space.d(x, y)
                    L = max(L, ratio)
        
        return L
    
    def iterate(self, x0: np.ndarray, max_iters: int = 1000, 
                tolerance: float = 1e-8) -> Tuple[np.ndarray, List[float], bool]:
        """Implement fixed point iteration."""
        x = x0.copy()
        distances = []
        
        for i in range(max_iters):
            x_next = self.f(x)
            dist = self.metric_space.d(x_next, x)
            distances.append(dist)
            
            if dist < tolerance:
                return x_next, distances, True
            
            x = x_next
        
        return x, distances, False
    
    def verify_banach_theorem(self, x0: np.ndarray, sample_points: List[np.ndarray],
                             max_iters: int = 1000) -> dict:
        """Verify the Banach fixed point theorem experimentally."""
        # Check if f is a contraction
        L = self.estimate_lipschitz_constant(sample_points)
        is_contraction = L < 1
        
        # Find fixed point
        fixed_point, distances, converged = self.iterate(x0, max_iters)
        
        # Check uniqueness by starting from different points
        unique = True
        if converged:
            for _ in range(5):
                random_start = sample_points[np.random.randint(len(sample_points))]
                other_fixed, _, other_converged = self.iterate(random_start, max_iters)
                if other_converged:
                    if self.metric_space.d(fixed_point, other_fixed) > 1e-6:
                        unique = False
                        break
        
        # Check convergence rate
        convergence_rate = None
        if len(distances) > 10:
            # Estimate rate from ratio of consecutive distances
            ratios = [distances[i+1] / distances[i] for i in range(5, 10) if distances[i] > 1e-10]
            if ratios:
                convergence_rate = np.mean(ratios)
        
        return {
            'is_contraction': is_contraction,
            'lipschitz_constant': L,
            'converged': converged,
            'fixed_point': fixed_point if converged else None,
            'unique_fixed_point': unique and converged,
            'convergence_rate': convergence_rate,
            'theoretical_rate': L if is_contraction else None
        }


class GradientDescent:
    """Gradient descent with convergence analysis."""
    
    def __init__(self, loss_func: Callable[[np.ndarray], float], 
                 grad_func: Callable[[np.ndarray], np.ndarray]):
        self.loss = loss_func
        self.grad = grad_func
    
    def optimize(self, x0: np.ndarray, learning_rate: float = 0.01,
                max_iters: int = 1000, tolerance: float = 1e-6) -> dict:
        """Implement gradient descent with convergence analysis."""
        x = x0.copy()
        trajectory = [x.copy()]
        losses = [self.loss(x)]
        grad_norms = []
        step_sizes = []
        
        for i in range(max_iters):
            # Compute gradient
            g = self.grad(x)
            grad_norm = np.linalg.norm(g)
            grad_norms.append(grad_norm)
            
            # Check convergence
            if grad_norm < tolerance:
                break
            
            # Update parameters
            x_new = x - learning_rate * g
            step_size = np.linalg.norm(x_new - x)
            step_sizes.append(step_size)
            
            x = x_new
            trajectory.append(x.copy())
            losses.append(self.loss(x))
        
        return {
            'trajectory': trajectory,
            'losses': losses,
            'grad_norms': grad_norms,
            'step_sizes': step_sizes,
            'converged': grad_norms[-1] < tolerance if grad_norms else False,
            'iterations': len(trajectory) - 1
        }
    
    def analyze_convergence_rate(self, results: dict) -> dict:
        """Analyze the convergence rate from optimization results."""
        trajectory = results['trajectory']
        
        # Estimate optimal point as final point
        x_star = trajectory[-1]
        
        # Compute distances to optimum
        distances = [np.linalg.norm(x - x_star) for x in trajectory[:-1]]
        
        # Skip if converged too quickly
        if len(distances) < 5:
            return {'convergence_type': 'immediate', 'rate': None}
        
        # Check for linear convergence: ||x_{k+1} - x*|| ≤ r * ||x_k - x*||
        linear_ratios = []
        for i in range(len(distances) - 1):
            if distances[i] > 1e-10:
                ratio = distances[i+1] / distances[i]
                if ratio < 1:
                    linear_ratios.append(ratio)
        
        # Check for quadratic convergence: ||x_{k+1} - x*|| ≤ C * ||x_k - x*||^2
        quadratic = False
        if len(distances) > 2:
            quadratic_ratios = []
            for i in range(len(distances) - 1):
                if distances[i] > 1e-10:
                    ratio = distances[i+1] / (distances[i]**2)
                    if ratio < 1e6:  # Reasonable bound
                        quadratic_ratios.append(ratio)
            
            # If ratios are roughly constant, it's quadratic
            if quadratic_ratios and np.std(quadratic_ratios) < 10:
                quadratic = True
        
        if quadratic:
            return {
                'convergence_type': 'quadratic',
                'rate': None,
                'quadratic_constant': np.mean(quadratic_ratios) if quadratic_ratios else None
            }
        elif linear_ratios:
            avg_ratio = np.mean(linear_ratios)
            return {
                'convergence_type': 'linear',
                'rate': avg_ratio,
                'theoretical_rate': None  # Would need to know condition number
            }
        else:
            return {
                'convergence_type': 'sublinear',
                'rate': None
            }


def visualize_metric_balls(metrics: List[Tuple[str, MetricSpace]], 
                          center: np.ndarray = np.array([0, 0]), 
                          radius: float = 1.0):
    """Visualize unit balls for different metrics in 2D."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    # Create grid of points
    x = np.linspace(center[0] - 2*radius, center[0] + 2*radius, 200)
    y = np.linspace(center[1] - 2*radius, center[1] + 2*radius, 200)
    X, Y = np.meshgrid(x, y)
    
    for ax, (name, metric_space) in zip(axes, metrics):
        # Compute distance from center for each point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                Z[i, j] = metric_space.d(point, center)
        
        # Plot unit ball
        contour = ax.contour(X, Y, Z, levels=[radius], colors='blue', linewidths=2)
        ax.contourf(X, Y, Z, levels=[0, radius], colors=['lightblue'], alpha=0.5)
        
        # Mark center
        ax.plot(center[0], center[1], 'ro', markersize=8)
        
        ax.set_title(f'{name} Ball (r={radius})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_continuity_breakdown():
    """Create examples showing how discontinuity affects optimization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Continuous function: smooth quadratic
    def continuous_f(x):
        return x[0]**2
    
    def continuous_grad(x):
        return np.array([2*x[0]])
    
    # 2. Jump discontinuity
    def discontinuous_f(x):
        return x[0]**2 if x[0] < 0 else x[0]**2 + 1
    
    def discontinuous_grad(x):
        return np.array([2*x[0]])  # Gradient ignores jump
    
    # 3. Non-differentiable: absolute value
    def nondiff_f(x):
        return np.abs(x[0])
    
    def nondiff_grad(x):
        return np.array([np.sign(x[0])]) if x[0] != 0 else np.array([0])
    
    # Run gradient descent on each
    x0 = np.array([2.0])
    lr = 0.1
    
    for ax, (f, grad, name) in zip(axes, [
        (continuous_f, continuous_grad, "Continuous"),
        (discontinuous_f, discontinuous_grad, "Jump Discontinuity"),
        (nondiff_f, nondiff_grad, "Non-differentiable")
    ]):
        # Plot function
        x_plot = np.linspace(-3, 3, 1000)
        y_plot = [f(np.array([xi])) for xi in x_plot]
        ax.plot(x_plot, y_plot, 'b-', label='f(x)', linewidth=2)
        
        # Run gradient descent
        gd = GradientDescent(f, grad)
        results = gd.optimize(x0, learning_rate=lr, max_iters=50)
        
        # Plot trajectory
        traj_x = [x[0] for x in results['trajectory']]
        traj_y = [f(x) for x in results['trajectory']]
        ax.plot(traj_x, traj_y, 'ro-', markersize=5, alpha=0.7, label='GD path')
        ax.plot(traj_x[0], traj_y[0], 'go', markersize=10, label='Start')
        ax.plot(traj_x[-1], traj_y[-1], 'rs', markersize=10, label='End')
        
        ax.set_title(f'{name}\nConverged: {results["converged"]}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Real Analysis Solutions - Testing implementations")
    
    # Test metrics
    print("\n1. Testing Metric Spaces")
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    print(f"Euclidean distance: {euclidean_distance(x, y):.4f}")
    print(f"Manhattan distance: {manhattan_distance(x, y):.4f}")
    print(f"Chebyshev distance: {chebyshev_distance(x, y):.4f}")
    
    # Verify metric properties
    points = [np.random.randn(3) for _ in range(5)]
    for name, dist in [("Euclidean", euclidean_distance), 
                       ("Manhattan", manhattan_distance),
                       ("Chebyshev", chebyshev_distance)]:
        space = MetricSpace(dist)
        print(f"{name} satisfies metric properties: {space.verify_metric_properties(points)}")
    
    # Test sequence convergence
    print("\n2. Testing Sequence Convergence")
    def sequence_term(n):
        return np.array([1/n, 1/n**2])
    
    space = MetricSpace(euclidean_distance)
    seq = Sequence(sequence_term, space)
    
    limit = np.array([0, 0])
    converges, N = seq.check_convergence(limit, epsilon=1e-3)
    print(f"Sequence converges to (0,0): {converges}, N = {N}")
    
    is_cauchy, N_cauchy = seq.is_cauchy(epsilon=1e-3)
    print(f"Sequence is Cauchy: {is_cauchy}, N = {N_cauchy}")
    
    # Test fixed point iteration
    print("\n3. Testing Fixed Point Iteration")
    def fixed_point_func(x):
        return 0.5 * x + 1
    
    space_1d = MetricSpace(lambda x, y: abs(x[0] - y[0]))
    iterator = FixedPointIterator(fixed_point_func, space_1d)
    
    sample_points = [np.array([float(i)]) for i in range(-5, 6)]
    L = iterator.estimate_lipschitz_constant(sample_points)
    print(f"Lipschitz constant: {L:.4f}")
    
    x0 = np.array([0.0])
    fixed_point, distances, converged = iterator.iterate(x0)
    print(f"Fixed point: {fixed_point[0]:.6f}, Converged: {converged}")
    
    # Verify Banach theorem
    results = iterator.verify_banach_theorem(x0, sample_points)
    print(f"Banach theorem verification: {results}")
    
    # Test gradient descent
    print("\n4. Testing Gradient Descent")
    def quadratic_loss(x):
        return 0.5 * np.sum(x**2)
    
    def quadratic_grad(x):
        return x
    
    gd = GradientDescent(quadratic_loss, quadratic_grad)
    x0 = np.array([5.0, 3.0])
    results = gd.optimize(x0, learning_rate=0.5, max_iters=100)
    
    print(f"Final loss: {results['losses'][-1]:.6f}")
    print(f"Converged: {results['converged']}")
    print(f"Iterations: {results['iterations']}")
    
    analysis = gd.analyze_convergence_rate(results)
    print(f"Convergence analysis: {analysis}")
    
    # Visualizations
    print("\n5. Creating Visualizations")
    metrics = [
        ("Euclidean", MetricSpace(euclidean_distance)),
        ("Manhattan", MetricSpace(manhattan_distance)),
        ("Chebyshev", MetricSpace(chebyshev_distance))
    ]
    visualize_metric_balls(metrics)
    
    demonstrate_continuity_breakdown()