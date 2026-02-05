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


def test_nonconvex_quadratic_gradients_and_batch_gradient():
    from exercise import NonConvexQuadratic

    problem = NonConvexQuadratic(n_samples=12, dim=4, negative_curvature_ratio=0.4)
    x = np.linspace(-0.2, 0.3, problem.dim)

    g = problem.full_gradient(x)
    fd = _finite_difference_grad(problem.objective, x)
    assert np.allclose(g, fd, atol=1e-5, rtol=1e-5)

    batch = np.array([0, 3, 5, 7, 9])
    gb = problem.batch_gradient(x, batch)
    g_manual = np.mean([problem.individual_gradient(x, int(i)) for i in batch], axis=0)
    assert np.allclose(gb, g_manual)


def test_spider_estimator_update_and_step():
    from exercise import NonConvexQuadratic, SPIDEROptimizer

    problem = NonConvexQuadratic(n_samples=20, dim=5, negative_curvature_ratio=0.3)
    x0 = np.ones(problem.dim) * 0.5

    opt = SPIDEROptimizer(batch_size_estimator=10, batch_size_update=5, update_frequency=3, step_size=0.2)
    x1 = opt.step(problem, x0, x_previous=None)
    assert opt.estimator is not None
    assert opt.iteration_count == 1
    assert x1.shape == x0.shape

    x2 = opt.step(problem, x1, x_previous=x0)
    assert opt.iteration_count == 2
    assert opt.estimator is not None


def test_optimize_with_spider_runs_and_improves_objective_on_easy_instance():
    from exercise import NonConvexQuadratic, SPIDEROptimizer, optimize_with_spider

    problem = NonConvexQuadratic(n_samples=30, dim=6, negative_curvature_ratio=0.1)
    x0 = np.ones(problem.dim) * 2.0
    opt = SPIDEROptimizer(batch_size_estimator=15, batch_size_update=5, update_frequency=5, step_size=0.1)

    f0 = problem.objective(x0)
    x_final, history = optimize_with_spider(problem, opt, x0, n_iterations=200, tolerance=1e-6, track_progress=True)
    assert len(history["objective"]) > 0
    assert problem.objective(x_final) <= f0 + 1e-6

