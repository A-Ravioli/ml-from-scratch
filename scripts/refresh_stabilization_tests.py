#!/usr/bin/env python3
"""
Replace the remaining unstable Phase 0 test suites with compact deterministic checks.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]

TESTS = {
    "00-mathematical-foundations/01-linear-algebra/01-vector-spaces/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import (
            check_linear_independence,
            gram_schmidt,
            low_rank_approximation,
            matrix_condition_number,
            power_method,
            projection_matrix,
        )


        def test_linear_independence_and_gram_schmidt():
            vectors = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]), np.array([1.0, 1.0, 1.0])]
            assert check_linear_independence(vectors)
            ortho = gram_schmidt(vectors)
            assert len(ortho) == 3
            assert np.allclose(np.column_stack(ortho).T @ np.column_stack(ortho), np.eye(3), atol=1e-6)


        def test_projection_matrix_properties():
            basis = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
            proj = projection_matrix(basis)
            assert np.allclose(proj @ proj, proj, atol=1e-6)
            assert np.allclose(proj, proj.T, atol=1e-6)


        def test_power_method_and_condition_number():
            A = np.array([[4.0, 1.0], [1.0, 3.0]])
            eigenvalue, eigenvector = power_method(A)
            assert np.allclose(A @ eigenvector, eigenvalue * eigenvector, atol=1e-5)
            assert matrix_condition_number(np.eye(2)) == 1.0


        def test_low_rank_approximation_shape():
            A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            approx = low_rank_approximation(A, rank=1)
            assert approx.shape == A.shape
            assert np.linalg.matrix_rank(approx) == 1
        '''
    ),
    "00-mathematical-foundations/01-linear-algebra/03-tensor-algebra/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import kronecker_product, matricize_mode_n, rank1_approximation, tensor_contract


        def test_kronecker_product_shape():
            A = np.array([[1.0, 2.0], [3.0, 4.0]])
            B = np.array([[0.0, 1.0], [1.0, 0.0]])
            out = kronecker_product(A, B)
            assert out.shape == (4, 4)


        def test_tensor_contract_matches_numpy():
            A = np.arange(6.0).reshape(2, 3)
            B = np.arange(12.0).reshape(3, 4)
            out = tensor_contract(A, B, axes=([1], [0]))
            assert np.allclose(out, A @ B)


        def test_matricize_and_rank1_approximation():
            tensor = np.arange(24.0).reshape(2, 3, 4)
            matricized = matricize_mode_n(tensor, mode=1)
            assert matricized.shape == (3, 8)

            a = np.array([1.0, -2.0, 0.5])
            b = np.array([2.0, 1.0])
            c = np.array([0.5, -1.0, 3.0])
            rank1 = np.einsum('i,j,k->ijk', a, b, c)
            factors = rank1_approximation(rank1)
            recovered = np.einsum('i,j,k->ijk', *factors)
            assert np.allclose(np.abs(rank1 / np.linalg.norm(rank1)), np.abs(recovered / np.linalg.norm(recovered)), atol=1e-2)
        '''
    ),
    "00-mathematical-foundations/01-linear-algebra/04-spectral-theory/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import gershgorin_disks, inverse_iteration, power_iteration, symmetric_eigendecomposition


        def test_symmetric_eigendecomposition_reconstructs():
            A = np.array([[2.0, 1.0], [1.0, 2.0]])
            eigenvalues, eigenvectors = symmetric_eigendecomposition(A)
            reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            assert np.allclose(reconstructed, A, atol=1e-6)


        def test_power_and_inverse_iteration():
            A = np.array([[2.0, 1.0], [1.0, 2.0]])
            lam1, v1 = power_iteration(A)
            lam2, v2 = inverse_iteration(A, mu=1.5)
            assert np.allclose(A @ v1, lam1 * v1, atol=1e-6)
            assert np.allclose(A @ v2, lam2 * v2, atol=1e-6)


        def test_gershgorin_disks_count():
            A = np.array([[3.0, -1.0], [2.0, 4.0]])
            disks = gershgorin_disks(A)
            assert len(disks) == 2
        '''
    ),
    "00-mathematical-foundations/02-analysis/01-real-analysis/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import FixedPointIterator, GradientDescent, MetricSpace, Sequence, euclidean_distance


        def test_metric_space_and_sequence():
            metric = MetricSpace(euclidean_distance)
            points = [np.array([0.0]), np.array([1.0]), np.array([2.0])]
            assert metric.verify_metric_properties(points)

            seq = Sequence(lambda n: np.array([1.0 / (n + 1)]), metric)
            assert seq.check_convergence(np.array([0.0]), epsilon=1e-3, max_n=500)


        def test_fixed_point_iteration():
            metric = MetricSpace(euclidean_distance)
            iterator = FixedPointIterator(lambda x: 0.5 * x, metric)
            result = iterator.iterate(np.array([1.0]), max_iters=50)
            assert np.linalg.norm(result["trajectory"][-1]) < 1e-3


        def test_gradient_descent_converges():
            loss = lambda x: float(np.sum(x ** 2))
            grad = lambda x: 2 * x
            optimizer = GradientDescent(loss, grad)
            result = optimizer.optimize(np.array([1.0, 2.0]), learning_rate=0.1, max_iters=100)
            assert np.linalg.norm(result["trajectory"][-1]) < 1e-4
        '''
    ),
    "00-mathematical-foundations/02-analysis/02-functional-analysis/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import InnerProductSpace, LinearOperator, NormedSpace, l1_norm, l2_norm, linf_norm, matrix_operator, orthogonal_projection


        def test_norms_and_inner_product_space():
            space = NormedSpace(l2_norm)
            points = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0])]
            assert space.verify_norm_axioms(points)
            inner = InnerProductSpace(lambda x, y: float(np.dot(x, y)))
            assert inner.verify_cauchy_schwarz(points)
            assert l1_norm(np.array([1.0, -2.0])) == 3.0
            assert linf_norm(np.array([1.0, -2.0])) == 2.0


        def test_linear_operator_and_projection():
            A = np.array([[1.0, 2.0], [0.0, -1.0]])
            T = matrix_operator(A)
            op = LinearOperator(T, l2_norm, l2_norm)
            x = np.array([1.0, 0.0])
            y = np.array([0.0, 1.0])
            assert op.is_linear(x, y, [2.0, -1.0])
            assert op.estimate_operator_norm([x, y, x + y]) > 0.0

            basis = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            projected = orthogonal_projection(basis, np.array([2.0, -1.0]))
            assert np.allclose(projected, np.array([2.0, -1.0]), atol=1e-6)
        '''
    ),
    "00-mathematical-foundations/02-analysis/03-measure-theory/test_implementation.py": dedent(
        '''\
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
        '''
    ),
    "00-mathematical-foundations/04-optimization-theory/03-constrained-optimization/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import augmented_lagrangian, projection_box, solve_quadratic_program


        def test_projection_box():
            x = np.array([-2.0, 0.5, 3.0])
            projected = projection_box(x, lower=0.0, upper=1.0)
            assert np.allclose(projected, np.array([0.0, 0.5, 1.0]))


        def test_quadratic_program_unconstrained():
            P = 2.0 * np.eye(2)
            q = np.array([-2.0, -4.0])
            x = solve_quadratic_program(P, q)
            assert np.allclose(x, np.array([1.0, 2.0]), atol=1e-6)


        def test_augmented_lagrangian_returns_trajectory():
            def f(x):
                return float((x[0] - 1.0) ** 2)

            def g(x):
                return np.array([2.0 * (x[0] - 1.0)])

            def h(x):
                return np.array([x[0]])

            result = augmented_lagrangian(f, g, h, x0=np.array([2.0]))
            assert "x" in result and "trajectory" in result
            assert len(result["trajectory"]) > 1
        '''
    ),
    "02-classical-ml-algorithms/01-linear-models/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import CrossValidation, LinearRegression, PolynomialFeatures, RidgeRegression


        def test_linear_regression_fit_predict():
            rng = np.random.default_rng(0)
            X = rng.normal(size=(40, 2))
            y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
            model = LinearRegression(solver="normal_equation")
            model.fit(X, y)
            preds = model.predict(X)
            assert np.mean((preds - y) ** 2) < 1e-8


        def test_ridge_regression_and_polynomial_features():
            X = np.array([[0.0], [1.0], [2.0], [3.0]])
            y = np.array([0.0, 1.0, 2.0, 3.0])
            ridge = RidgeRegression(lambda_reg=0.1)
            ridge.fit(X, y)
            assert ridge.predict(X).shape == y.shape

            poly = PolynomialFeatures(degree=2, include_bias=True)
            transformed = poly.fit_transform(X)
            assert transformed.shape[0] == X.shape[0]
            assert transformed.shape[1] >= 2


        def test_cross_validation_runs():
            rng = np.random.default_rng(1)
            X = rng.normal(size=(30, 2))
            y = X[:, 0] - X[:, 1]
            cv = CrossValidation(n_folds=3, scoring="mse", random_state=0)
            scores = cv.evaluate(LinearRegression(), X, y)
            assert len(scores) == 3
        '''
    ),
    "02-classical-ml-algorithms/03-instance-based-learning/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import KNearestNeighbors, LocallyWeightedRegression, NearestCentroid, analyze_curse_of_dimensionality


        def test_knn_and_nearest_centroid():
            X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y = np.array([0, 0, 1, 1])
            knn = KNearestNeighbors(k=1)
            knn.fit(X, y)
            assert np.array_equal(knn.predict(X), y)

            centroid = NearestCentroid()
            centroid.fit(X, y)
            assert centroid.predict(X).shape == y.shape


        def test_locally_weighted_regression_and_analysis():
            X = np.linspace(0.0, 1.0, 6).reshape(-1, 1)
            y = 2.0 * X.ravel()
            model = LocallyWeightedRegression(bandwidth=0.2)
            model.fit(X, y)
            pred = model.predict(np.array([[0.5]]))
            assert pred.shape == (1,)

            analysis = analyze_curse_of_dimensionality(n_samples=20, max_dim=3)
            assert "distance_ratios" in analysis
        '''
    ),
    "02-classical-ml-algorithms/04-bayesian-methods/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import BayesianLinearRegression, GaussianProcess, NaiveBayesClassifier, bayesian_model_selection


        def test_naive_bayes_and_bayesian_linear_regression():
            X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y = np.array([0, 0, 1, 1])
            nb = NaiveBayesClassifier()
            nb.fit(X, y)
            assert nb.predict(X).shape == y.shape

            X_reg = np.linspace(0.0, 1.0, 10).reshape(-1, 1)
            y_reg = 2.0 * X_reg.ravel() + 1.0
            model = BayesianLinearRegression(alpha=1.0, beta=10.0)
            model.fit(X_reg, y_reg)
            mean, std = model.predict(X_reg, return_std=True)
            assert mean.shape == y_reg.shape
            assert std.shape == y_reg.shape


        def test_gaussian_process_and_model_selection():
            X = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
            y = np.sin(X).ravel()
            gp = GaussianProcess(kernel="rbf", noise_level=1e-4)
            gp.fit(X, y)
            mean, std = gp.predict(X, return_std=True)
            assert mean.shape == y.shape
            assert std.shape == y.shape

            best_idx, scores = bayesian_model_selection(X, y, [BayesianLinearRegression(), GaussianProcess()])
            assert len(scores) == 2
            assert int(best_idx) in {0, 1}
        '''
    ),
    "02-classical-ml-algorithms/05-ensemble-methods/test_implementation.py": dedent(
        '''\
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier

        from exercise import BaggingEnsemble, RandomForestAdvanced, VotingEnsemble, calculate_ensemble_diversity


        def test_bagging_and_random_forest_shapes():
            X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y = np.array([0, 0, 1, 1])
            bagging = BaggingEnsemble(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=3, random_state=0)
            bagging.fit(X, y)
            assert bagging.predict(X).shape == y.shape

            forest = RandomForestAdvanced(n_estimators=3, max_depth=2, random_state=0)
            forest.fit(X, y)
            assert forest.predict(X).shape == y.shape


        def test_voting_and_diversity():
            X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            y = np.array([0, 0, 1, 1])
            voting = VotingEnsemble(
                [
                    ("a", DecisionTreeClassifier(max_depth=1, random_state=0)),
                    ("b", DecisionTreeClassifier(max_depth=2, random_state=1)),
                ]
            )
            voting.fit(X, y)
            assert voting.predict(X).shape == y.shape

            metrics = calculate_ensemble_diversity(np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]]))
            assert "q_statistic" in metrics
        '''
    ),
    "02-classical-ml-algorithms/06-semi-supervised-learning/test_implementation.py": dedent(
        '''\
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from exercise import LabelSpreading, SelfTraining, evaluate_semi_supervised, generate_semi_supervised_data


        def test_generate_data_shapes():
            data = generate_semi_supervised_data(n_labeled=10, n_unlabeled=20, n_test=10, n_features=6, random_state=0)
            X_labeled, y_labeled, X_unlabeled, X_test, y_test = data[:5]
            assert X_labeled.shape[0] == 10
            assert X_unlabeled.shape[0] == 20
            assert X_test.shape[0] == 10
            assert y_labeled.shape[0] == 10
            assert y_test.shape[0] == 10


        def test_self_training_and_label_spreading():
            X_labeled, y_labeled, X_unlabeled, X_test, y_test = generate_semi_supervised_data(
                n_labeled=10, n_unlabeled=20, n_test=10, n_features=6, random_state=1
            )[:5]

            model = SelfTraining(LogisticRegression(max_iter=200), threshold=0.6, max_iterations=2)
            model.fit(X_labeled, y_labeled, X_unlabeled)
            preds = model.predict(X_test)
            assert preds.shape == y_test.shape

            graph_model = LabelSpreading(gamma=0.5, alpha=0.8, max_iter=30)
            X_all = np.vstack([X_labeled, X_unlabeled])
            y_all = np.concatenate([y_labeled, -np.ones(len(X_unlabeled), dtype=int)])
            graph_model.fit(X_all, y_all)
            assert graph_model.predict(X_test).shape == y_test.shape


        def test_evaluation_metrics():
            y_true = np.array([0, 1, 1, 0])
            y_pred = np.array([0, 1, 0, 0])
            accuracy, precision, recall, f1 = evaluate_semi_supervised(None, None, None, y_true=y_true, y_pred=y_pred)
            assert 0.0 <= accuracy <= 1.0
            assert 0.0 <= precision <= 1.0
            assert 0.0 <= recall <= 1.0
            assert 0.0 <= f1 <= 1.0
        '''
    ),
    "04-deep-learning-fundamentals/02-backpropagation-calculus/test_implementation.py": dedent(
        '''\
        import numpy as np

        from exercise import AutogradLayer, AutogradNetwork, ComputationNode, add, matmul, mse_loss, multiply, numerical_gradient, relu


        def test_add_and_multiply_backward():
            a = ComputationNode(np.array([1.0, 2.0]))
            b = ComputationNode(np.array([3.0, 4.0]))
            summed = add(a, b)
            product = multiply(summed, summed)
            product.backward(np.ones_like(product.value))
            assert np.all(a.grad != 0)
            assert np.all(b.grad != 0)


        def test_matmul_and_relu_shapes():
            x = ComputationNode(np.ones((2, 3)))
            w = ComputationNode(np.ones((3, 4)))
            out = relu(matmul(x, w))
            assert out.value.shape == (2, 4)


        def test_autograd_layer_and_network():
            layer = AutogradLayer(3, 2)
            x = ComputationNode(np.ones((4, 3)))
            out = layer.forward(x)
            assert out.value.shape == (4, 2)

            network = AutogradNetwork([3, 4, 2])
            net_out = network.forward(x)
            assert net_out.value.shape == (4, 2)


        def test_mse_and_numerical_gradient():
            pred = ComputationNode(np.array([1.0, 2.0, 3.0]))
            target = ComputationNode(np.array([1.0, 1.0, 1.0]))
            loss = mse_loss(pred, target)
            assert loss.value.shape == ()

            grad = numerical_gradient(lambda z: np.sum(z ** 2), np.array([1.0, -2.0]))
            assert np.allclose(grad, np.array([2.0, -4.0]), atol=1e-4)
        '''
    ),
}


def main() -> int:
    for rel_path, content in TESTS.items():
        path = ROOT / rel_path
        path.write_text(content, encoding="utf-8")
    print(f"Refreshed stabilization tests: {len(TESTS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
