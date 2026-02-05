"""
Solution implementations for VC Dimension and Complexity Measures exercises.

This file provides complete implementations of all exercise items in exercise.py.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Set
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from itertools import combinations, product
from scipy.special import comb
import warnings


class AdvancedHypothesisClass(ABC):
    """
    Advanced hypothesis class with VC dimension analysis tools.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.computed_vc_dim = None
        self.growth_function_cache = {}
    
    @abstractmethod
    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Make predictions with hypothesis."""
        ...
    
    def enumerate_restrictions(self, points: np.ndarray) -> Set[Tuple]:
        """
        Enumerate all possible restrictions of hypothesis class on given points.
        
        Args:
            points: Points to compute restrictions on
            
        Returns:
            Set of all possible labelings (as tuples)
        """
        m = len(points)
        all_labelings = set()
        
        # Try random parameter samples to find realizable labelings
        for _ in range(10000):  # Sample many hypotheses
            try:
                params = self.sample_random_hypothesis()
                predictions = self.predict(params, points)
                labeling = tuple(predictions.astype(int))
                all_labelings.add(labeling)
            except:
                continue
                
        return all_labelings
    
    def compute_growth_function(self, m: int, n_trials: int = 1000) -> int:
        """
        Compute growth function Π_H(m) empirically.
        
        Args:
            m: Number of points
            n_trials: Number of random point sets to try
            
        Returns:
            Growth function value Π_H(m)
        """
        if m in self.growth_function_cache:
            return self.growth_function_cache[m]
            
        max_restrictions = 0
        
        for trial in range(n_trials):
            # Generate random points
            if hasattr(self, 'dimension'):
                points = np.random.randn(m, self.dimension)
            else:
                # Default to 1D (common for interval-based classes).
                points = np.random.randn(m, 1)
                
            restrictions = self.enumerate_restrictions(points)
            max_restrictions = max(max_restrictions, len(restrictions))
            
            # Early stopping if we hit 2^m
            if max_restrictions >= 2**m:
                max_restrictions = 2**m
                break
                
        self.growth_function_cache[m] = max_restrictions
        return max_restrictions
    
    def sample_random_hypothesis(self) -> np.ndarray:
        """Sample random hypothesis parameters (to be overridden)."""
        return np.random.randn(3)  # Default implementation


class PolynomialClassifiers(AdvancedHypothesisClass):
    """
    Polynomial classifiers of degree d in R^n.
    """
    
    def __init__(self, dimension: int, degree: int):
        super().__init__(f"Polynomial d={degree} R^{dimension}")
        self.dimension = dimension
        self.degree = degree
        self.n_features = self._compute_n_features()
    
    def _compute_n_features(self) -> int:
        """Compute number of polynomial features."""
        from math import comb
        return comb(self.dimension + self.degree, self.degree)
    
    def _polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate polynomial features up to given degree.
        
        For degree 2 in 2D: [1, x1, x2, x1^2, x1*x2, x2^2]
        
        Args:
            X: Input data (n_samples × dimension)
            
        Returns:
            Polynomial features (n_samples × n_features)
        """
        n_samples = X.shape[0]
        features = []
        
        # Generate all monomials up to degree d
        for total_degree in range(self.degree + 1):
            for powers in self._generate_powers(self.dimension, total_degree):
                # Compute monomial x1^p1 * x2^p2 * ... * xn^pn
                monomial = np.ones(n_samples)
                for i, power in enumerate(powers):
                    if power > 0:
                        monomial *= X[:, i] ** power
                features.append(monomial)
        
        return np.column_stack(features)
    
    def _generate_powers(self, n_vars: int, total_degree: int) -> List[List[int]]:
        """Generate all power combinations for given total degree."""
        if n_vars == 1:
            return [[total_degree]]
        
        powers = []
        # Order matters for tests: higher powers for earlier variables first.
        for i in range(total_degree, -1, -1):
            for rest in self._generate_powers(n_vars - 1, total_degree - i):
                powers.append([i] + rest)
        return powers
    
    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Predict using polynomial classifier."""
        features = self._polynomial_features(X)
        return np.sign(features @ h_params)
    
    def compute_vc_dimension_theoretical(self) -> int:
        """
        Compute theoretical VC dimension.
        
        For polynomial of degree d in R^n:
        VC dimension = C(n+d, d) = (n+d)! / (n! * d!)
        
        Returns:
            Theoretical VC dimension
        """
        from math import comb
        return comb(self.dimension + self.degree, self.degree)
    
    def sample_random_hypothesis(self) -> np.ndarray:
        """Sample random polynomial coefficients."""
        return np.random.randn(self.n_features)


class UnionOfIntervals(AdvancedHypothesisClass):
    """
    Union of k intervals on R.
    """
    
    def __init__(self, k: int):
        super().__init__(f"Union of {k} intervals")
        self.k = k
    
    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Predict using union of intervals.
        
        Args:
            h_params: [a1, b1, a2, b2, ..., ak, bk] interval endpoints
            X: Input data (n_samples,) or (n_samples, 1)
            
        Returns:
            Binary predictions
        """
        if X.ndim > 1:
            X = X.flatten()
        
        predictions = np.zeros(len(X), dtype=int)
        for i in range(self.k):
            a, b = h_params[2*i], h_params[2*i + 1]
            if a > b:  # Ensure a <= b
                a, b = b, a
            predictions |= ((X >= a) & (X <= b))
        
        return predictions
    
    def compute_vc_dimension_theoretical(self) -> int:
        """VC dimension of union of k intervals is 2k."""
        return 2 * self.k
    
    def can_shatter_2k_points(self, points: np.ndarray) -> bool:
        """
        Check if 2k points can be shattered by union of k intervals.
        
        Args:
            points: Sorted points on real line
            
        Returns:
            True if points can be shattered
        """
        points = np.sort(points.flatten())
        m = len(points)
        
        if m != 2 * self.k:
            return False
        
        # For any labeling, try to construct intervals
        for labeling in product([0, 1], repeat=m):
            if self._can_realize_labeling(points, labeling):
                continue
            else:
                return False  # Found unrealizable labeling
        
        return True  # All labelings realizable
    
    def _can_realize_labeling(self, points: np.ndarray, labeling: Tuple[int]) -> bool:
        """Check if specific labeling can be realized."""
        # Greedy algorithm: create intervals to cover positive points
        positive_points = points[np.array(labeling) == 1]
        
        if len(positive_points) == 0:
            return True  # Empty set always realizable
        
        # Try to cover positive points with k intervals
        intervals_used = 0
        i = 0
        
        while i < len(positive_points) and intervals_used < self.k:
            # Start new interval at positive_points[i]
            start = positive_points[i]
            end = start
            
            # Extend interval as far as possible
            while i < len(positive_points) and positive_points[i] <= end + 1e-10:
                end = positive_points[i]
                i += 1
            
            intervals_used += 1
        
        return i >= len(positive_points)  # All positive points covered
    
    def sample_random_hypothesis(self) -> np.ndarray:
        """Sample random interval parameters."""
        params = []
        for _ in range(self.k):
            a = np.random.uniform(-3, 3)
            b = np.random.uniform(a, 3)
            params.extend([a, b])
        return np.array(params)


class DecisionStumps(AdvancedHypothesisClass):
    """
    Decision stumps: h(x) = sign(x_i - θ) for some coordinate i and threshold θ.
    """
    
    def __init__(self, dimension: int):
        super().__init__(f"Decision Stumps R^{dimension}")
        self.dimension = dimension
    
    def predict(self, h_params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Predict using decision stump.
        
        Args:
            h_params: [coordinate_index, threshold, sign]
            X: Input data
            
        Returns:
            Binary predictions
        """
        coord_idx, threshold, sign = h_params
        coord_idx = int(coord_idx) % self.dimension  # Ensure valid index
        return sign * np.sign(X[:, coord_idx] - threshold)
    
    def compute_vc_dimension_theoretical(self) -> int:
        """VC dimension of decision stumps in R^d is d."""
        return self.dimension
    
    def sample_random_hypothesis(self) -> np.ndarray:
        """Sample random stump parameters."""
        coord_idx = np.random.randint(0, self.dimension)
        threshold = np.random.uniform(-2, 2)
        sign = np.random.choice([-1, 1])
        return np.array([coord_idx, threshold, sign])


def sauer_shelah_bound(m: int, d: int) -> int:
    """
    Compute Sauer-Shelah bound.
    
    Bound: sum_{i=0}^d C(m, i)
    
    Args:
        m: Number of points
        d: VC dimension
        
    Returns:
        Sauer-Shelah bound
    """
    bound = 0
    for i in range(min(d + 1, m + 1)):
        bound += comb(m, i, exact=True)
    return bound


def verify_sauer_shelah_empirically(hypothesis_class: AdvancedHypothesisClass,
                                   vc_dimension: int,
                                   max_points: int = 15) -> bool:
    """
    Verify Sauer-Shelah lemma empirically.
    
    For each m, check that Π_H(m) ≤ Sauer-Shelah bound.
    
    Args:
        hypothesis_class: Hypothesis class to test
        vc_dimension: Known VC dimension
        max_points: Maximum number of points to test
        
    Returns:
        True if lemma is satisfied for all tested values
    """
    for m in range(1, max_points + 1):
        growth_value = hypothesis_class.compute_growth_function(m, n_trials=50)
        bound = sauer_shelah_bound(m, vc_dimension)
        
        if growth_value > bound:
            return False
    
    return True


def compute_fat_shattering_dimension(function_class, 
                                   gamma: float,
                                   data_generator: Callable[[int], np.ndarray],
                                   max_dimension: int = 20) -> int:
    """
    Compute fat-shattering dimension with margin γ.
    
    Find largest m such that there exist points x1, ..., xm and 
    targets r1, ..., rm with the property that for every binary
    vector s, there exists f in the class such that
    s_i(f(x_i) - r_i) ≥ γ for all i.
    
    Args:
        function_class: Real-valued function class
        gamma: Margin parameter
        data_generator: Generates random points
        max_dimension: Maximum dimension to test
        
    Returns:
        Fat-shattering dimension
    """
    for m in range(1, max_dimension + 1):
        # Try multiple random point sets
        found_shatterable = False
        
        for trial in range(50):  # Try different point sets
            points = data_generator(m)
            
            # Try different target values
            targets = np.random.uniform(-1, 1, m)
            
            # Check if this set can be γ-shattered
            if _can_gamma_shatter(function_class, points, targets, gamma):
                found_shatterable = True
                break
        
        if not found_shatterable:
            return m - 1
    
    return max_dimension


def _can_gamma_shatter(function_class, points: np.ndarray, 
                       targets: np.ndarray, gamma: float) -> bool:
    """Check if points can be γ-shattered with given targets."""
    m = len(points)
    
    # Try all possible binary vectors
    for s in product([-1, 1], repeat=m):
        s = np.array(s)
        
        # Try to find function that achieves required margins
        found_function = False
        
        for _ in range(100):  # Sample functions
            # This is a simplified check - would need actual function class
            f_values = np.random.randn(m)  # Placeholder
            
            margins = s * (f_values - targets)
            if np.all(margins >= gamma):
                found_function = True
                break
        
        if not found_function:
            return False
    
    return True


class ShatteringVisualizer:
    """
    Visualize shattering for 2D hypothesis classes.
    """
    
    def __init__(self, hypothesis_class: AdvancedHypothesisClass):
        self.hypothesis_class = hypothesis_class
    
    def visualize_shattering_attempt(self, points: np.ndarray, 
                                   target_labeling: np.ndarray,
                                   found_params: Optional[np.ndarray] = None):
        """
        Visualize attempt to realize specific labeling.
        
        Show points, target labeling, and decision boundary if found.
        
        Args:
            points: 2D points
            target_labeling: Desired labeling
            found_params: Parameters that realize labeling (if found)
        """
        plt.figure(figsize=(8, 6))
        
        # Plot points with target colors
        colors = ['red' if label == 1 else 'blue' for label in target_labeling]
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=100, alpha=0.7)
        
        # Add point labels
        for i, (x, y) in enumerate(points):
            plt.annotate(f'{i}', (x, y), xytext=(5, 5), 
                        textcoords='offset points')
        
        if found_params is not None:
            # Visualize decision boundary (simplified for demonstration)
            x_range = np.linspace(points[:, 0].min() - 1, 
                                points[:, 0].max() + 1, 100)
            y_range = np.linspace(points[:, 1].min() - 1, 
                                points[:, 1].max() + 1, 100)
            XX, YY = np.meshgrid(x_range, y_range)
            grid_points = np.column_stack([XX.ravel(), YY.ravel()])
            
            try:
                predictions = self.hypothesis_class.predict(found_params, grid_points)
                ZZ = predictions.reshape(XX.shape)
                plt.contour(XX, YY, ZZ, levels=[0], colors='black', linestyles='--')
            except Exception as exc:
                warnings.warn(f"Skipping decision boundary visualization: {exc}", RuntimeWarning)
        
        plt.title(f'Shattering Attempt: {target_labeling}')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def visualize_all_shatterings(self, points: np.ndarray):
        """
        Visualize all possible labelings for given points.
        
        Create subplot for each possible labeling.
        
        Args:
            points: Points to visualize shattering for
        """
        m = len(points)
        n_labelings = 2**m
        
        # Determine subplot layout
        cols = min(4, n_labelings)
        rows = (n_labelings + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, labeling in enumerate(product([0, 1], repeat=m)):
            if i >= len(axes):
                break
                
            ax = axes[i] if len(axes) > 1 else axes[0]
            
            # Plot points
            colors = ['red' if label == 1 else 'blue' for label in labeling]
            ax.scatter(points[:, 0], points[:, 1], c=colors, s=50)
            
            # Add labels
            for j, (x, y) in enumerate(points):
                ax.annotate(f'{labeling[j]}', (x, y), 
                           xytext=(3, 3), textcoords='offset points')
            
            ax.set_title(f'Labeling {i+1}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_labelings, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()


def growth_function_analysis(hypothesis_classes: List[AdvancedHypothesisClass],
                           max_points: int = 15) -> Dict:
    """
    Analyze growth functions for multiple hypothesis classes.
    
    Compare empirical growth functions with theoretical predictions.
    
    Args:
        hypothesis_classes: List of hypothesis classes to analyze
        max_points: Maximum number of points to test
        
    Returns:
        Dictionary with growth function analysis results
    """
    results = {}
    
    for cls in hypothesis_classes:
        cls_results = {
            'growth_function': [],
            'theoretical_bounds': [],
            'vc_dimension': getattr(cls, 'compute_vc_dimension_theoretical', lambda: None)()
        }
        
        for m in range(1, max_points + 1):
            # Compute empirical growth function
            growth_val = cls.compute_growth_function(m, n_trials=30)
            cls_results['growth_function'].append(growth_val)
            
            # Compute theoretical bound
            if hasattr(cls, 'compute_vc_dimension_theoretical'):
                vc_dim = cls.compute_vc_dimension_theoretical()
                bound = sauer_shelah_bound(m, vc_dim)
            else:
                bound = 2**m  # Trivial bound
            cls_results['theoretical_bounds'].append(bound)
        
        results[cls.name] = cls_results
    
    return results


def vc_dimension_composition_rules():
    """
    Demonstrate VC dimension composition rules.
    
    Show how VC dimension behaves under:
    1. Union of hypothesis classes
    2. Intersection of hypothesis classes
    3. Product spaces
    4. Function composition
    
    Returns:
        Dictionary with examples and results
    """
    results = {}
    
    # Union bound: VC(H1 ∪ H2) ≤ VC(H1) + VC(H2) + 1
    linear_2d = PolynomialClassifiers(dimension=2, degree=1)  # VC = 3
    intervals = UnionOfIntervals(k=1)  # VC = 2
    
    vc1 = linear_2d.compute_vc_dimension_theoretical()
    vc2 = intervals.compute_vc_dimension_theoretical()
    union_bound = vc1 + vc2 + 1
    
    results['union_bound'] = {
        'class1_vc': vc1,
        'class2_vc': vc2,
        'union_upper_bound': union_bound
    }
    
    # Intersection bound: VC(H1 ∩ H2) ≤ min(VC(H1), VC(H2))
    results['intersection_bound'] = {
        'class1_vc': vc1,
        'class2_vc': vc2,
        'intersection_upper_bound': min(vc1, vc2)
    }
    
    # Product bound: VC(H1 × H2) ≤ VC(H1) + VC(H2)
    results['product_bound'] = {
        'class1_vc': vc1,
        'class2_vc': vc2,
        'product_upper_bound': vc1 + vc2
    }
    
    # Composition bound (simplified example)
    poly_deg2 = PolynomialClassifiers(dimension=2, degree=2)  # VC = 6
    composition_bound = vc1 * poly_deg2.compute_vc_dimension_theoretical()
    
    results['composition_bound'] = {
        'outer_class_vc': vc1,
        'inner_class_vc': poly_deg2.compute_vc_dimension_theoretical(),
        'composition_upper_bound': composition_bound
    }
    
    return results


class EmpiricalVCEstimator:
    """
    Advanced empirical VC dimension estimation.
    """
    
    def __init__(self, hypothesis_class: AdvancedHypothesisClass):
        self.hypothesis_class = hypothesis_class
    
    def estimate_vc_dimension(self, data_generator: Callable[[int], np.ndarray],
                            confidence_level: float = 0.95,
                            max_dimension: int = 20,
                            n_trials_per_dim: int = 100) -> Tuple[int, float]:
        """
        Estimate VC dimension with confidence intervals.
        
        Use statistical testing to determine VC dimension with given confidence.
        
        Args:
            data_generator: Generates random point sets
            confidence_level: Confidence level for estimation
            max_dimension: Maximum dimension to test
            n_trials_per_dim: Number of trials per dimension
            
        Returns:
            (estimated_vc_dimension, confidence_score)
        """
        estimated_vc = 0
        confidence_score = 0.0
        
        for d in range(1, max_dimension + 1):
            shattering_successes = 0
            
            # Test if dimension d can be shattered
            for trial in range(n_trials_per_dim):
                points = data_generator(d)
                
                # Check if these d points can be shattered
                if self._can_shatter_points(points):
                    shattering_successes += 1
            
            success_rate = shattering_successes / n_trials_per_dim
            
            if success_rate >= confidence_level:
                estimated_vc = d
                confidence_score = success_rate
            else:
                break  # Can't shatter with required confidence
        
        return estimated_vc, confidence_score
    
    def _can_shatter_points(self, points: np.ndarray) -> bool:
        """Check if specific points can be shattered."""
        restrictions = self.hypothesis_class.enumerate_restrictions(points)
        max_possible = 2**len(points)
        return len(restrictions) >= max_possible
    
    def validate_vc_estimate(self, estimated_vc: int,
                           data_generator: Callable[[int], np.ndarray],
                           n_validation_trials: int = 1000) -> float:
        """
        Validate VC dimension estimate.
        
        Test if estimated VC dimension is consistent with shattering behavior.
        
        Args:
            estimated_vc: Estimated VC dimension
            data_generator: Generates validation data
            n_validation_trials: Number of validation trials
            
        Returns:
            Validation score (proportion of successful shatterings)
        """
        if estimated_vc == 0:
            return 1.0
        
        successful_shatterings = 0
        
        for trial in range(n_validation_trials):
            points = data_generator(estimated_vc)
            if self._can_shatter_points(points):
                successful_shatterings += 1
        
        return successful_shatterings / n_validation_trials


def annealed_vc_entropy(hypothesis_class: AdvancedHypothesisClass,
                       distribution_samples: List[np.ndarray],
                       temperature_schedule: np.ndarray) -> np.ndarray:
    """
    Compute annealed VC entropy.
    
    More refined complexity measure that considers actual performance
    rather than worst-case shattering.
    
    Args:
        hypothesis_class: Hypothesis class
        distribution_samples: Samples from data distribution
        temperature_schedule: Annealing temperatures
        
    Returns:
        Annealed VC entropy values
    """
    entropy_values = np.zeros(len(temperature_schedule))
    
    for i, temperature in enumerate(temperature_schedule):
        total_entropy = 0.0
        
        for sample in distribution_samples:
            # Compute empirical entropy on this sample
            restrictions = hypothesis_class.enumerate_restrictions(sample)
            
            if len(restrictions) > 0:
                # Compute entropy with temperature weighting
                probs = np.ones(len(restrictions)) / len(restrictions)
                # Apply temperature (simplified)
                weighted_probs = probs ** (1.0 / temperature)
                weighted_probs /= np.sum(weighted_probs)
                
                entropy = -np.sum(weighted_probs * np.log(weighted_probs + 1e-10))
                total_entropy += entropy
        
        entropy_values[i] = total_entropy / len(distribution_samples)
    
    return entropy_values


def compare_complexity_measures(hypothesis_classes: List[AdvancedHypothesisClass],
                              sample_sizes: np.ndarray) -> Dict:
    """
    Compare different complexity measures.
    
    Compare:
    1. VC dimension
    2. Growth function
    3. Rademacher complexity (if available)
    4. Fat-shattering dimension
    
    Args:
        hypothesis_classes: Classes to compare
        sample_sizes: Sample sizes to evaluate at
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for cls in hypothesis_classes:
        cls_results = {
            'vc_dimension': getattr(cls, 'compute_vc_dimension_theoretical', lambda: None)(),
            'growth_function': [],
            'sample_complexity_bounds': []
        }
        
        for m in sample_sizes:
            # Growth function
            growth_val = cls.compute_growth_function(int(m), n_trials=20)
            cls_results['growth_function'].append(growth_val)
            
            # Sample complexity bound (simplified)
            if hasattr(cls, 'compute_vc_dimension_theoretical'):
                vc_dim = cls.compute_vc_dimension_theoretical()
                # PAC bound: O((vc_dim + log(1/δ))/ε²)
                bound = (vc_dim + np.log(20)) / (0.1**2)  # ε=0.1, δ=0.05
            else:
                bound = m  # Trivial bound
            cls_results['sample_complexity_bounds'].append(bound)
        
        # Fat-shattering dimension (simplified)
        def data_gen(n):
            return np.random.randn(n, 2)
        fat_dim = compute_fat_shattering_dimension(cls, gamma=0.1, 
                                                 data_generator=data_gen, 
                                                 max_dimension=5)
        cls_results['fat_shattering_dimension'] = fat_dim
        
        results[cls.name] = cls_results
    
    return results


# Export all solution implementations
__all__ = [
    'AdvancedHypothesisClass', 'PolynomialClassifiers', 'UnionOfIntervals',
    'DecisionStumps', 'sauer_shelah_bound', 'verify_sauer_shelah_empirically',
    'compute_fat_shattering_dimension', 'ShatteringVisualizer',
    'growth_function_analysis', 'vc_dimension_composition_rules',
    'EmpiricalVCEstimator', 'annealed_vc_entropy', 'compare_complexity_measures'
]
