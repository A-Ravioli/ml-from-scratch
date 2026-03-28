import numpy as np

from exercise import (
    FiniteFunctionClass,
    GaussianComplexity,
    LinearFunctions,
    RadmacherComplexityAnalyzer,
    empirical_rademacher_complexity,
    rademacher_generalization_bound,
    symmetrization_lemma_verification,
)


def test_linear_function_sampling_and_evaluation():
    np.random.seed(0)
    function_class = LinearFunctions(dimension=3, bound=2.0)
    weights = function_class.sample_function()
    X = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])

    values = function_class.evaluate(weights, X)

    assert values.shape == (2,)
    assert np.linalg.norm(weights) <= 2.0 + 1e-8


def test_finite_class_and_empirical_rademacher_complexity():
    np.random.seed(1)
    functions = [
        np.array([1.0, -1.0, 0.5]),
        np.array([-0.5, 0.25, 1.0]),
    ]
    function_class = FiniteFunctionClass(functions)
    X = np.zeros((3, 2))

    assert np.allclose(function_class.evaluate(np.array([1]), X), functions[1])

    mean_complexity, std_complexity = empirical_rademacher_complexity(
        function_class, X, n_samples=20
    )

    assert mean_complexity >= 0.0
    assert std_complexity >= 0.0
    assert function_class.theoretical_rademacher_complexity(X) > 0.0


def test_generalization_bound_and_auxiliary_analyses():
    np.random.seed(2)
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]])
    function_class = LinearFunctions(dimension=2, bound=1.0)

    analyzer = RadmacherComplexityAnalyzer()
    comparison = analyzer.compare_theoretical_empirical(
        function_class, X, n_monte_carlo=20
    )
    assert comparison["sample_size"] == len(X)
    assert comparison["empirical"] >= 0.0

    bound = rademacher_generalization_bound(
        empirical_risk=0.1,
        rademacher_complexity=0.2,
        confidence=0.1,
        sample_size=20,
    )
    assert bound > 0.1

    gaussian = GaussianComplexity().compute_gaussian_complexity(
        function_class, X, n_samples=20
    )
    assert gaussian >= 0.0

    verification = symmetrization_lemma_verification(
        function_class,
        distribution=lambda: np.random.randn(2),
        sample_size=4,
        n_trials=5,
    )
    assert "lemma_satisfied" in verification
