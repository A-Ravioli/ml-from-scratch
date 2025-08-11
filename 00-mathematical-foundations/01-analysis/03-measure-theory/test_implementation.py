"""
Test suite for Measure Theory implementations (finite spaces).
"""

import numpy as np
import pytest

from exercise import (
    is_sigma_algebra, sigma_algebra_generated_by, is_measure,
    is_measurable_function, integral, expectation,
    markov_bound, chebyshev_bound
)


class TestSigmaAlgebra:
    def test_sigma_algebra_axioms(self):
        omega = {1, 2, 3}
        F = [set(), {1, 2, 3}, {1}, {2, 3}]
        assert is_sigma_algebra(omega, F)

    def test_generated_sigma(self):
        omega = {1, 2, 3, 4}
        gens = [{1, 2}]
        sigma = sigma_algebra_generated_by(omega, gens)
        coll = list(map(frozenset, sigma))
        # Should contain ∅, Ω, {1,2}, {3,4}
        assert frozenset() in coll
        assert frozenset(omega) in coll
        assert frozenset({1, 2}) in coll
        assert frozenset({3, 4}) in coll


class TestMeasure:
    def test_measure_axioms(self):
        omega = {"a", "b"}
        F = [set(), omega, {"a"}, {"b"}]
        mu = {frozenset(): 0.0, frozenset(omega): 1.0, frozenset({"a"}): 0.3, frozenset({"b"}): 0.7}
        assert is_measure(omega, F, mu)

    def test_integral_and_expectation(self):
        omega = {0, 1, 2}
        prob = {frozenset({0}): 0.2, frozenset({1}): 0.5, frozenset({2}): 0.3}
        f = {0: 1.0, 1: 3.0, 2: 5.0}
        E = expectation(prob, f)
        assert abs(E - (0.2*1 + 0.5*3 + 0.3*5)) < 1e-12


class TestMeasurability:
    def test_measurable_function(self):
        omega = {0, 1, 2}
        F = [set(), omega, {0}, {1, 2}]
        f = {0: -1.0, 1: 0.2, 2: 3.5}
        assert is_sigma_algebra(omega, F)
        assert is_measurable_function(omega, F, f, thresholds=[-1, 0, 1, 3])


class TestInequalities:
    def test_markov_and_chebyshev(self):
        assert abs(markov_bound(2.0, 4.0) - 0.5) < 1e-12
        assert abs(chebyshev_bound(1.0, 2.0) - 0.25) < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 


