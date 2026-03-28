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
