"""
Test suite for Topology implementations (finite spaces).
"""

import pytest

from exercise import (
    is_topology, closure, interior, boundary,
    is_continuous, is_compact, is_connected
)


class TestTopologyAxioms:
    def test_discrete_and_trivial_topology(self):
        X = {1, 2, 3}
        # Discrete: all subsets
        P = []
        from itertools import chain, combinations

        def powerset(s):
            s = list(s)
            return [set(c) for r in range(len(s)+1) for c in combinations(s, r)]

        Tau_discrete = powerset(X)
        assert is_topology(X, Tau_discrete)

        # Trivial: only ∅ and X
        Tau_trivial = [set(), set(X)]
        assert is_topology(X, Tau_trivial)


class TestOperators:
    def test_closure_interior_boundary(self):
        X = {1, 2, 3}
        Tau = [set(), {1, 2, 3}, {1, 2}]
        A = {1}
        clA = closure(X, Tau, A)
        intA = interior(X, Tau, A)
        bdA = boundary(X, Tau, A)

        assert clA.issuperset(A)
        assert intA.issubset(A)
        assert bdA == clA - intA


class TestContinuityCompactnessConnectedness:
    def test_continuity(self):
        X = {0, 1}
        Tau_X = [set(), X, {0}, {1}]
        Y = {"a", "b"}
        Tau_Y = [set(), Y, {"a"}, {"b"}]

        def f(x):
            return "a" if x == 0 else "b"

        assert is_continuous(X, Tau_X, Y, Tau_Y, f)

    def test_compact_and_connected(self):
        X = {1, 2}
        Tau_trivial = [set(), set(X)]
        assert is_compact(X, Tau_trivial)
        assert is_connected(X, Tau_trivial)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


