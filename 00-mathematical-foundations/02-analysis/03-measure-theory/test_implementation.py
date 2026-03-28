from exercise import chebyshev_bound, expectation, integral, is_sigma_algebra, markov_bound


def test_sigma_algebra_basic():
    omega = {0, 1}
    F = [set(), omega, {0}, {1}]
    assert is_sigma_algebra(omega, F)


def test_integral_and_expectation():
    omega = {0, 1}
    F = [set(), omega, {0}, {1}]
    mu = {frozenset(): 0.0, frozenset(omega): 1.0, frozenset({0}): 0.25, frozenset({1}): 0.75}
    f = {0: 2.0, 1: 4.0}
    assert integral(omega, F, mu, f) == 3.5
    assert expectation({frozenset({0}): 0.25, frozenset({1}): 0.75}, f) == 3.5


def test_probability_bounds():
    assert markov_bound(10.0, 2.0) == 5.0
    assert chebyshev_bound(4.0, 2.0) == 1.0
