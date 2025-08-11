"""
Topology Exercises for Machine Learning (finite spaces)
"""

from typing import List, Set, Callable


def is_topology(X: Set, tau: List[Set]) -> bool:
    """
    TODO: Verify topology axioms on finite set X.
    """
    # TODO: Implement this
    pass


def closure(X: Set, tau: List[Set], A: Set) -> Set:
    """
    TODO: Compute closure of A as the smallest closed set containing A.
    (In finite spaces, intersect all closed sets containing A.)
    """
    # TODO: Implement this
    pass


def interior(X: Set, tau: List[Set], A: Set) -> Set:
    """
    TODO: Compute interior of A as the largest open set contained in A.
    """
    # TODO: Implement this
    pass


def boundary(X: Set, tau: List[Set], A: Set) -> Set:
    """
    TODO: ∂A = cl(A) \ int(A).
    """
    # TODO: Implement this
    pass


def is_continuous(X: Set, tau_X: List[Set], Y: Set, tau_Y: List[Set], f: Callable) -> bool:
    """
    TODO: Check continuity by preimage of open sets: f^{-1}(U) open in X for all U open in Y.
    """
    # TODO: Implement this
    pass


def is_compact(X: Set, tau: List[Set]) -> bool:
    """
    TODO: Check compactness: for every open cover, a finite subcover exists.
    For finite spaces, this is always True; implement a direct check.
    """
    # TODO: Implement this
    pass


def is_connected(X: Set, tau: List[Set]) -> bool:
    """
    TODO: Check connectedness: there are no nonempty disjoint open sets whose union is X.
    """
    # TODO: Implement this
    pass


if __name__ == "__main__":
    print("Topology Exercises for ML (finite spaces)")

