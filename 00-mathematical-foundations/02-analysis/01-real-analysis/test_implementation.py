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
    final_point, trajectory, _ = result
    assert np.linalg.norm(final_point) < 1e-3
    assert len(trajectory) > 0


def test_gradient_descent_converges():
    loss = lambda x: float(np.sum(x ** 2))
    grad = lambda x: 2 * x
    optimizer = GradientDescent(loss, grad)
    result = optimizer.optimize(np.array([1.0, 2.0]), learning_rate=0.1, max_iters=100)
    assert np.linalg.norm(result["trajectory"][-1]) < 1e-4
