"""
Topology Solutions (finite spaces) - Reference Implementation
"""

from typing import List, Set, Callable


def is_topology(X: Set, tau: List[Set]) -> bool:
    Xs = set(X)
    Tau = [set(U) for U in tau]
    if set() not in Tau:
        return False
    if Xs not in Tau:
        return False
    # Unions
    for U in Tau:
        for V in Tau:
            if (U | V) not in Tau:
                return False
    # Finite intersections
    for U in Tau:
        for V in Tau:
            if (U & V) not in Tau:
                return False
    return True


def closure(X: Set, tau: List[Set], A: Set) -> Set:
    Xs = set(X)
    Tau = [set(U) for U in tau]
    # Closed sets are complements of open sets
    closed_sets = [Xs - U for U in Tau]
    candidates = [C for C in closed_sets if set(A).issubset(C)]
    if not candidates:
        return set(A)
    # Intersection of all closed supersets
    Cl = candidates[0].copy()
    for C in candidates[1:]:
        Cl &= C
    return Cl


def interior(X: Set, tau: List[Set], A: Set) -> Set:
    Tau = [set(U) for U in tau]
    candidates = [U for U in Tau if set(U).issubset(A)]
    if not candidates:
        return set()
    Int = candidates[0].copy()
    for U in candidates[1:]:
        Int |= U
    return Int


def boundary(X: Set, tau: List[Set], A: Set) -> Set:
    return closure(X, tau, A) - interior(X, tau, A)


def is_continuous(X: Set, tau_X: List[Set], Y: Set, tau_Y: List[Set], f: Callable) -> bool:
    TauX = [set(U) for U in tau_X]
    for U in tau_Y:
        pre = {x for x in X if f(x) in U}
        if set(pre) not in TauX:
            return False
    return True


def is_compact(X: Set, tau: List[Set]) -> bool:
    # Every finite topological space is compact
    return True


def is_connected(X: Set, tau: List[Set]) -> bool:
    Tau = [set(U) for U in tau]
    Xs = set(X)
    # Look for separation: U,V open, disjoint, nonempty, U∪V=X
    for U in Tau:
        for V in Tau:
            if U and V and (U & V) == set() and (U | V) == Xs:
                return False
    return True


