"""Checks for exercise.py (nonconvex optimization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The test functions
are checked at hand-computed points and against central differences; perturbed gradient descent
on a function with a saddle whose stable manifold plain gradient descent never leaves; the
uniform-ball noise by its radius distribution; the landscape and PCA tools on problems whose
answers are known exactly. grads are wrapped in call budgets so loops can't run away.
"""
import numpy as np


class _Budget:
    def __init__(self, fn, limit, what="grad"):
        self.fn, self.limit, self.what, self.calls = fn, limit, what, 0

    def __call__(self, x):
        self.calls += 1
        assert self.calls <= self.limit, f"{self.what} was called more than {self.limit} times: does your loop respect max_iters?"
        return self.fn(x)


def _fd_grad(f, x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _saddle(v):  # 1/2 x^2 - 1/2 y^2 + 1/4 y^4: a saddle at 0, minima at (0, +-1) with f = -1/4
    return float(0.5 * v[0] ** 2 - 0.5 * v[1] ** 2 + 0.25 * v[1] ** 4)


def _saddle_grad(v):
    return np.array([v[0], -v[1] + v[1] ** 3])


# ---------------------------------------------------------------- stage 1: test functions


def test_stage1_rosenbrock_values(nb):
    assert np.isclose(nb.rosenbrock(np.array([-1.2, 1.0])), 24.2), (
        f"Rosenbrock at (-1.2, 1): 100 (1 - 1.44)^2 + (1 + 1.2)^2 = 19.36 + 4.84 = 24.2; got {nb.rosenbrock(np.array([-1.2, 1.0]))}")
    assert nb.rosenbrock(np.ones(5)) == 0.0 and np.allclose(nb.rosenbrock_grad(np.ones(5)), 0), "minimum 0 at all ones, gradient 0"
    x = np.array([0.5, -1.0, 2.0, 0.0])
    want = sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(3))
    got = nb.rosenbrock(x)
    assert np.isclose(got, want), (
        f"the generalized (chained) Rosenbrock sums b (x_{{i+1}} - x_i^2)^2 + (a - x_i)^2 over consecutive pairs "
        f"i = 0..n-2: {want} at {x}; got {got}")
    assert np.isclose(nb.rosenbrock(np.array([2.0, 4.0]), a=2.0, b=5.0), 0.0), "with a = 2 the minimum is at (2, 4)"
    assert isinstance(got, float), "return a Python float"


def test_stage1_rosenbrock_gradient(nb):
    rng = np.random.default_rng(0)
    for n in (2, 3, 7):
        x = rng.normal(size=n)
        g = np.asarray(nb.rosenbrock_grad(x, a=0.5, b=20.0))
        fd = _fd_grad(lambda v: nb.rosenbrock(v, a=0.5, b=20.0), x)
        assert g.shape == (n,) and np.allclose(g, fd, rtol=1e-5, atol=1e-5), (
            f"n = {n}: your gradient {np.round(g, 4)} differs from central differences {np.round(fd, 4)}. the middle "
            "coordinates appear in TWO terms (as x_i in term i, as x_{i+1} in term i-1): add both contributions.")


def test_stage1_rastrigin(nb):
    assert nb.rastrigin(np.zeros(3)) == 0.0, "Rastrigin is 0 at the origin"
    assert np.isclose(nb.rastrigin(np.array([1.0, 1.0])), 2.0), "at (1, 1): 20 + 2 - 10 (cos 2 pi + cos 2 pi) = 2"
    assert np.isclose(nb.rastrigin(np.array([0.5])), 20.25), "at 0.5: 10 + 0.25 - 10 cos(pi) = 20.25"
    assert np.isclose(nb.rastrigin(np.array([0.5]), A=3.0), 6.25), "with A = 3: 3 + 0.25 + 3 = 6.25"
    x = np.random.default_rng(1).uniform(-3, 3, size=5)
    assert np.allclose(nb.rastrigin_grad(x, A=4.0), _fd_grad(lambda v: nb.rastrigin(v, A=4.0), x), atol=1e-5), (
        "rastrigin_grad should be 2 x + 2 pi A sin(2 pi x)")
    big = np.random.default_rng(2).normal(size=100_000)
    assert np.isfinite(nb.rastrigin(big)) and nb.rastrigin_grad(big).shape == (100_000,), "work for any n (vectorized)"


# ---------------------------------------------------------------- stage 2: perturbed gradient descent


def test_stage2_escapes_a_saddle(nb):
    stuck = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 3_000), np.array([1.0, 0.0]), lr=0.1,
                                          noise_radius=0.0, max_iters=2_000, rng=0)
    assert abs(stuck.fun) < 1e-6 and stuck.converged, (
        f"with no noise this is gradient descent, and from (1, 0) it never leaves the line y = 0: it converges to the "
        f"SADDLE at the origin (f = 0, a vanishing gradient); got {stuck}")
    esc = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 3_000), np.array([1.0, 0.0]), lr=0.1,
                                        noise_radius=0.01, max_iters=2_000, rng=0)
    assert np.isclose(esc.fun, -0.25, atol=1e-4) and abs(abs(esc.x[1]) - 1) < 0.02, (
        f"with a little noise added when the gradient is small, the iterate leaves the saddle along the y-direction "
        f"(negative curvature) and reaches a minimum (0, +-1) with f = -0.25; got {esc}")


def test_stage2_noise_is_uniform_in_the_ball(nb):
    for n, r in ((2, 0.3), (3, 1.0)):
        steps = []
        x_prev = [np.zeros(n)]

        def cb(x, k):
            steps.append(np.linalg.norm(x - x_prev[0]))
            x_prev[0] = np.array(x)

        nb.perturbed_gradient_descent(lambda v: 0.0, _Budget(lambda v: np.zeros(n), 4_010), np.zeros(n), noise_radius=r,
                                      max_iters=4_000, rng=1, callback=cb)
        steps = np.array(steps)
        assert steps.max() <= r + 1e-12, f"every kick must stay inside the ball of radius {r}; the largest was {steps.max()}"
        frac = np.mean(steps <= r * 0.5 ** (1 / n))
        assert abs(frac - 0.5) < 0.04, (
            f"for a point uniform in a {n}-D ball of radius r, P(|xi| <= r 0.5^(1/{n})) = 1/2 (half the volume); you "
            f"got {frac:.3f}. use radius r U^(1/n): r U puts too many points near the centre, and a fixed radius r "
            "samples the sphere, not the ball.")


def test_stage2_conventions(nb):
    gb = _Budget(_saddle_grad, 60)
    ks, xs = [], []
    res = nb.perturbed_gradient_descent(_saddle, gb, np.array([1.0, 0.5]), lr=0.1, max_iters=50, rng=3,
                                        callback=lambda x, k: (ks.append(k), xs.append(np.array(x))))
    assert res.converged == bool(np.linalg.norm(_saddle_grad(res.x)) < 0.001), (
        "converged must be judged at the returned (best) iterate: ||grad(best x)|| < grad_threshold")
    jit0 = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 60), np.array([0.0, 1.0]), noise_radius=0.2,
                                         max_iters=30, rng=5)
    assert jit0.converged == bool(np.linalg.norm(_saddle_grad(jit0.x)) < 0.001), (
        "started at a minimum and kicked away: converged refers to the best iterate (the minimum itself), not "
        "the last, kicked one")
    assert ks == list(range(1, 51)) and res.n_iters == 50, "run all max_iters steps, calling callback(x, k) with k = 1..50"
    assert len(res.history) == 51 and np.isclose(res.history[0], _saddle(np.array([1.0, 0.5]))), (
        "history = [f(x0), f(x1), ...]: 51 values")
    assert np.allclose(xs[0], [0.9, 0.5 - 0.1 * (-0.5 + 0.125)]), (
        f"far from stationary points it's a plain gradient step: x1 = x0 - 0.1 grad(x0) = (0.9, 0.5375); got {xs[0]}")
    best = int(np.argmin(res.history))
    assert np.isclose(res.fun, res.history[best]) and res.fun == min(res.history), (
        "return the BEST iterate (lowest f) and its value")
    jit = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 60), np.array([0.0, 1.0]), noise_radius=0.2,
                                        max_iters=30, rng=5)
    assert jit.fun == min(jit.history) and jit.fun < jit.history[-1] and np.isclose(_saddle(jit.x), jit.fun), (
        f"started AT a minimum, the noise keeps kicking the iterate away, so the last iterate is worse than the best "
        f"one: return the best iterate seen (x0 included) and its value; got fun = {jit.fun}, last f = "
        f"{jit.history[-1]}, best in history = {min(jit.history)}")
    a = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 30), np.array([0.0, 0.0]), max_iters=20, rng=7)
    b = nb.perturbed_gradient_descent(_saddle, _Budget(_saddle_grad, 30), np.array([0.0, 0.0]), max_iters=20, rng=7)
    assert a.history == b.history, "the same rng seed must give the same run (use as_rng(rng), not np.random)"


# ---------------------------------------------------------------- stage 3: loss landscapes


def test_stage3_quadratic_slice(nb):
    A = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
    f = lambda x: float(x @ A @ x)
    x0 = np.array([0.1, -0.2, 0.3])
    d1, d2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 1.0])
    al, be, Z = nb.loss_landscape_2d(f, x0, d1, d2, span=0.5, resolution=11)
    assert np.allclose(al, np.linspace(-0.5, 0.5, 11)) and np.allclose(be, al), "alphas = betas = linspace(-span, span, resolution)"
    Z = np.asarray(Z)
    assert Z.shape == (11, 11), f"Z should be (resolution, resolution) = (11, 11); got {Z.shape}"
    for i, j in ((0, 0), (2, 9), (10, 4), (5, 5)):
        want = f(x0 + al[i] * d1 + be[j] * d2)
        assert np.isclose(Z[i, j], want), (
            f"Z[{i}, {j}] should be f(x0 + alphas[{i}] d1 + betas[{j}] d2) = {want:.6f}; got {Z[i, j]:.6f}"
            + (" (rows and columns swapped?)" if np.isclose(Z[j, i], want) else ""))
    assert np.isclose(Z[5, 5], f(x0)), "the centre of the grid is x0 itself"


def test_stage3_defaults(nb):
    al, be, Z = nb.loss_landscape_2d(lambda x: float(np.sum(x)), np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert len(al) == 51 and np.isclose(al[0], -1) and np.isclose(al[-1], 1), "defaults: span 1, resolution 51"
    assert np.allclose(np.asarray(Z), al[:, None] + be[None, :]), "f = x1 + x2 gives Z[i, j] = alphas[i] + betas[j]"


# ---------------------------------------------------------------- stage 4: trajectory PCA


def test_stage4_planar_path(nb):
    rng = np.random.default_rng(4)
    Q, _ = np.linalg.qr(rng.normal(size=(10, 2)))
    t = np.linspace(0, 1, 30)
    path2 = np.stack([3 * (1 - t) ** 2, np.sin(4 * t) * (1 - t)], axis=1)  # ends at (0, 0)
    xs = np.ones(10) + path2 @ Q.T
    d1, d2, coords, explained = nb.trajectory_pca(xs)
    d1, d2, coords, explained = map(np.asarray, (d1, d2, coords, explained))
    assert np.isclose(d1 @ d1, 1) and np.isclose(d2 @ d2, 1) and abs(d1 @ d2) < 1e-10, "d1 and d2 must be orthonormal"
    assert coords.shape == (30, 2) and np.allclose(coords[-1], 0), (
        "coords has one row per iterate, and the final point (the centre) has coordinates (0, 0)")
    recon = xs[-1] + coords @ np.stack([d1, d2])
    assert np.allclose(recon, xs, atol=1e-10), (
        "this path lies in a plane, so x_final + coords @ [d1; d2] must reproduce every iterate exactly. center the "
        "iterates at the LAST one (not the mean) before the SVD.")
    assert np.isclose(explained.sum(), 1.0) and explained[0] >= explained[1], (
        f"a planar path is fully explained by two directions, the first explaining the most: got {explained}")


def test_stage4_first_direction_is_dominant(nb):
    rng = np.random.default_rng(5)
    n = 8
    main = rng.normal(size=n)
    main /= np.linalg.norm(main)
    xs = np.array([(5 - k) * main + 0.05 * rng.normal(size=n) for k in range(6)])
    d1, d2, coords, explained = nb.trajectory_pca(xs)
    assert abs(np.asarray(d1) @ main) > 0.99, "the first direction should align (up to sign) with the path's main direction"
    assert np.asarray(explained)[0] > 0.98, f"almost all of the movement is along one direction; got explained = {explained}"
    xs3 = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    _, _, _, ex3 = nb.trajectory_pca(xs3)
    D = xs3 - xs3[-1]
    s3 = np.linalg.svd(D, compute_uv=False)
    want = s3[:2] ** 2 / np.sum(s3**2)
    assert np.allclose(ex3, want), (
        f"this path uses three directions, so the top two explain less than everything: explained should be "
        f"{np.round(want, 4)} (s_i^2 over the sum of ALL squared singular values); got {np.round(ex3, 4)}")
