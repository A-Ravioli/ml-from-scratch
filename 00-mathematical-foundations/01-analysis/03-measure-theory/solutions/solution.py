"""
Measure Theory Solutions (finite spaces) - Reference Implementation
"""

from typing import List, Set, Dict, Callable


def is_sigma_algebra(omega: Set, F: List[Set]) -> bool:
    O = set(omega)
    coll = [set(A) for A in F]
    # 1) Contains empty and omega
    if set() not in coll:
        return False
    if O not in coll:
        return False
    # 2) Closed under complement
    for A in coll:
        if (O - A) not in coll:
            return False
    # 3) Closed under unions (finite suffice)
    for A in coll:
        for B in coll:
            if (A | B) not in coll:
                return False
    return True


def sigma_algebra_generated_by(omega: Set, generators: List[Set]) -> List[Set]:
    O = set(omega)
    family = set(map(frozenset, [set(), O] + [set(g) for g in generators]))
    changed = True
    while changed:
        changed = False
        current = list(family)
        # complements
        for A in current:
            comp = frozenset(O - set(A))
            if comp not in family:
                family.add(comp)
                changed = True
        # unions
        current = list(family)
        for A in current:
            for B in current:
                U = frozenset(set(A) | set(B))
                if U not in family:
                    family.add(U)
                    changed = True
    return [set(A) for A in family]


def is_measure(omega: Set, F: List[Set], mu: Dict[frozenset, float], tol: float = 1e-12) -> bool:
    coll = set(map(frozenset, [set(A) for A in F]))
    # Non-negativity and null empty
    if abs(mu.get(frozenset(), 0.0)) > tol:
        return False
    for A in coll:
        if mu.get(A, 0.0) < -tol:
            return False
    # Finite additivity on disjoint unions
    fam = list(coll)
    for i in range(len(fam)):
        for j in range(len(fam)):
            if i == j:
                continue
            A, B = fam[i], fam[j]
            if (set(A) & set(B)):
                continue
            if frozenset(set(A) | set(B)) in coll:
                lhs = mu.get(frozenset(set(A) | set(B)), 0.0)
                rhs = mu.get(A, 0.0) + mu.get(B, 0.0)
                if abs(lhs - rhs) > 1e-10:
                    return False
    return True


def is_measurable_function(omega: Set, F: List[Set], f: Dict, thresholds: List[float]) -> bool:
    coll = set(map(frozenset, [set(A) for A in F]))
    for t in thresholds:
        pre = {w for w in omega if f[w] <= t}
        if frozenset(pre) not in coll:
            return False
    return True


def integral(omega: Set, F: List[Set], mu: Dict[frozenset, float], f: Dict) -> float:
    # Expect mu is defined on singletons
    total = 0.0
    for w in omega:
        total += float(f[w]) * float(mu.get(frozenset({w}), 0.0))
    return total


def expectation(prob: Dict[frozenset, float], f: Dict) -> float:
    # Extract omega from prob keys (singletons)
    omega = set()
    for k in prob.keys():
        s = set(k)
        if len(s) == 1:
            omega |= s
    return integral(omega, [], prob, f)


def markov_bound(expectation_X: float, a: float) -> float:
    if a <= 0:
        raise ValueError("a must be > 0")
    return expectation_X / a


def chebyshev_bound(variance_X: float, t: float) -> float:
    if t <= 0:
        raise ValueError("t must be > 0")
    return variance_X / (t * t)


