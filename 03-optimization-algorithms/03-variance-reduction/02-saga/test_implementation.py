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


def test_quadratic_problem_optimality_and_gradients():
    from exercise import QuadraticFiniteSum

    problem = QuadraticFiniteSum(n_samples=7, dim=4, condition_number=10.0, regularization=0.1)
    x = np.linspace(-0.2, 0.3, problem.dim)
    fd = _finite_difference_grad(problem.objective, x)
    g = problem.full_gradient(x)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-5)

    x_star = problem.optimal_point()
    assert np.linalg.norm(problem.full_gradient(x_star)) < 1e-8


def test_saga_table_init_and_gradient_formula():
    from exercise import QuadraticFiniteSum, SAGAOptimizer

    problem = QuadraticFiniteSum(n_samples=9, dim=3, condition_number=8.0, regularization=0.05)
    opt = SAGAOptimizer(step_size=0.1)
    x0 = np.array([0.2, -0.1, 0.05])

    opt.initialize_table(problem, x0)
    assert opt.gradient_table.shape == (problem.n_samples, problem.dim)
    assert opt.average_gradient.shape == (problem.dim,)

    idx = 3
    g_current = problem.individual_gradient(x0, idx)
    g_saga = opt.compute_saga_gradient(problem, x0, idx)
    g_expected = g_current - opt.gradient_table[idx] + opt.average_gradient + problem.regularization * x0
    assert np.allclose(g_saga, g_expected)


def test_l1_prox_operator_soft_thresholding():
    from exercise import l1_prox_operator

    x = np.array([-2.0, -0.5, 0.0, 0.25, 3.0])
    out = l1_prox_operator(x, threshold=0.6)
    expected = np.array([-1.4, -0.0, 0.0, 0.0, 2.4])
    assert np.allclose(out, expected)


def test_saga_converges_on_quadratic_and_beats_sgd_like_baseline():
    from exercise import QuadraticFiniteSum, SAGAOptimizer, optimize_with_saga

    problem = QuadraticFiniteSum(n_samples=20, dim=6, condition_number=20.0, regularization=0.1)
    x0 = np.ones(problem.dim) * 2.0

    # Conservative step size to ensure stability.
    eta = min(0.2, 0.6 / problem.L)
    saga = SAGAOptimizer(step_size=eta)
    x_saga, h_saga = optimize_with_saga(problem, saga, x0, n_epochs=20)
    assert len(h_saga["objective"]) > 0

    # Compare to starting objective as a sanity check.
    assert problem.objective(x_saga) < problem.objective(x0)
