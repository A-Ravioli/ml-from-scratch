import numpy as np


def _finite_difference_grad(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for j in range(x.size):
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[j] += eps
        x_neg[j] -= eps
        grad[j] = (f(x_pos) - f(x_neg)) / (2 * eps)
    return grad


def test_quadratic_problem_shapes_and_optimality():
    from exercise import QuadraticFiniteSum

    problem = QuadraticFiniteSum(n_samples=8, dim=5, condition_number=12.0, noise_level=0.2)
    assert problem.A_matrices is not None
    assert problem.centers is not None
    assert len(problem.A_matrices) == problem.n_samples
    assert problem.centers.shape == (problem.n_samples, problem.dim)

    for A in problem.A_matrices:
        assert A.shape == (problem.dim, problem.dim)
        eigs = np.linalg.eigvalsh((A + A.T) / 2)
        assert np.min(eigs) > 0.0

    x_star = problem.optimal_point()
    g_star = problem.full_gradient(x_star)
    assert np.linalg.norm(g_star) < 1e-8


def test_quadratic_gradients_match_finite_differences():
    from exercise import QuadraticFiniteSum

    problem = QuadraticFiniteSum(n_samples=6, dim=4, condition_number=8.0, noise_level=0.1)
    x = np.linspace(-0.3, 0.4, problem.dim)

    fd = _finite_difference_grad(problem.objective, x)
    g = problem.full_gradient(x)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-5)

    i = 2
    fd_i = _finite_difference_grad(lambda xx: problem.individual_objective(xx, i), x)
    g_i = problem.individual_gradient(x, i)
    assert np.allclose(g_i, fd_i, atol=1e-5, rtol=1e-5)


def test_svrg_gradient_formula_and_snapshot_counter():
    from exercise import QuadraticFiniteSum, SVRGOptimizer

    problem = QuadraticFiniteSum(n_samples=10, dim=3, condition_number=6.0, noise_level=0.1)
    opt = SVRGOptimizer(step_size=0.1, inner_loop_length=5)
    x0 = np.array([0.2, -0.1, 0.05])

    opt.update_snapshot(x0, problem)
    assert opt.snapshot_point is not None
    assert opt.full_gradient_at_snapshot is not None
    assert opt.history["full_gradient_evaluations"] == 1

    x = x0 + np.array([0.1, 0.0, -0.2])
    idx = 4
    g_svrg = opt.compute_svrg_gradient(x, idx, problem)
    g_expected = (
        problem.individual_gradient(x, idx)
        - problem.individual_gradient(opt.snapshot_point, idx)
        + opt.full_gradient_at_snapshot
    )
    assert np.allclose(g_svrg, g_expected)


def test_svrg_beats_sgd_on_strongly_convex_quadratic():
    from exercise import QuadraticFiniteSum, SVRGOptimizer, SGDOptimizer, optimize_with_svrg, optimize_with_sgd

    problem = QuadraticFiniteSum(n_samples=20, dim=6, condition_number=15.0, noise_level=0.2)
    x0 = np.ones(problem.dim) * 2.0

    # Use conservative step sizes based on the smoothness constant to avoid divergence.
    svrg_eta = min(0.2, 0.8 / problem.L)
    sgd_eta = min(0.1, 0.5 / problem.L)
    svrg = SVRGOptimizer(step_size=svrg_eta, inner_loop_length=10)
    sgd = SGDOptimizer(step_size=sgd_eta)

    x_svrg, h_svrg = optimize_with_svrg(problem, svrg, x0, n_epochs=15)
    x_sgd, h_sgd = optimize_with_sgd(problem, sgd, x0, n_iterations=15 * 10)

    assert len(h_svrg["objective"]) > 0
    assert len(h_sgd["objective"]) > 0

    f_svrg = problem.objective(x_svrg)
    f_sgd = problem.objective(x_sgd)
    assert f_svrg <= f_sgd + 1e-6
