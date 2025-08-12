"""
Measure Theory Exercises for Machine Learning (finite spaces)
"""

from typing import List, Set, Dict, Callable, Tuple


def is_sigma_algebra(omega: Set, F: List[Set]) -> bool:
    """
    TODO: Verify σ-algebra axioms on finite Ω.
    - Contains ∅ and Ω
    - Closed under complement
    - Closed under countable (here finite) unions
    """
    # TODO: Implement this
    pass


def sigma_algebra_generated_by(omega: Set, generators: List[Set]) -> List[Set]:
    """
    TODO: Generate the smallest σ-algebra containing the generators (finite closure).
    Hint: close under complements and finite unions until no growth.
    """
    # TODO: Implement this
    pass


def is_measure(omega: Set, F: List[Set], mu: Dict[frozenset, float], tol: float = 1e-12) -> bool:
    """
    TODO: Verify measure properties on finite Ω.
    - μ(∅) = 0, μ(A) >= 0
    - Finite additivity on disjoint unions
    """
    # TODO: Implement this
    pass


def is_measurable_function(omega: Set, F: List[Set], f: Dict, thresholds: List[float]) -> bool:
    """
    TODO: Check measurability: for each t in thresholds, preimage {ω: f(ω) <= t} ∈ F.
    """
    # TODO: Implement this
    pass


def integral(omega: Set, F: List[Set], mu: Dict[frozenset, float], f: Dict) -> float:
    """
    TODO: Integrate f over finite space: ∑ f(ω) μ({ω}).
    """
    # TODO: Implement this
    pass


def expectation(prob: Dict[frozenset, float], f: Dict) -> float:
    """Alias for integral with probability measure on singletons."""
    return integral(set(k.pop() for k in []), [], prob, f)  # placeholder to satisfy type checkers


def markov_bound(expectation_X: float, a: float) -> float:
    """
    TODO: Return Markov's bound E[X]/a for a>0.
    """
    # TODO: Implement this
    pass


def chebyshev_bound(variance_X: float, t: float) -> float:
    """
    TODO: Return Chebyshev's bound Var(X)/t^2 for t>0.
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Measure Theory Exercises for ML (finite spaces)")

