"""Checks for exercise.py (kernels & RKHS).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`.
scikit-learn's pairwise kernels are used only as oracles.
"""
import tracemalloc

import numpy as np
import pytest


def _data(n, d, seed):
    return np.random.default_rng(seed).normal(size=(n, d))


# ---------------------------------------------------------------- stage 1: Gram matrices


def test_stage1_hand_example(nb):
    x, y = np.array([[1.0, 2.0]]), np.array([[3.0, -1.0]])
    checks = [
        ("linear_kernel", nb.linear_kernel(x, y), 1.0, "x . y = 3 - 2 = 1"),
        ("polynomial_kernel", nb.polynomial_kernel(x, y, degree=2), 4.0, "(1 * (x . y) + 1)^2 = 4"),
        ("rbf_kernel", nb.rbf_kernel(x, y, gamma=0.5), np.exp(-6.5), "exp(-0.5 * ||x - y||^2) = exp(-0.5 * 13)"),
        ("laplacian_kernel", nb.laplacian_kernel(x, y, gamma=0.5), np.exp(-2.5), "exp(-0.5 * ||x - y||_1) = exp(-0.5 * 5)"),
    ]
    for name, got, want, why in checks:
        got = np.asarray(got)
        assert got.shape == (1, 1), f"{name}: one row against one row gives a (1, 1) Gram matrix, got {got.shape}"
        assert np.isclose(got[0, 0], want), f"{name}: {why} = {want:.6g}, got {got[0, 0]:.6g}"


def test_stage1_matches_sklearn(nb):
    from sklearn.metrics import pairwise as P
    X, Y = _data(9, 4, 0), _data(6, 4, 1)
    pairs = [
        (nb.linear_kernel(X, Y), P.linear_kernel(X, Y), "linear_kernel"),
        (nb.polynomial_kernel(X, Y, degree=3, gamma=0.3, coef0=2.0), P.polynomial_kernel(X, Y, 3, 0.3, 2.0),
         "polynomial_kernel(degree=3, gamma=0.3, coef0=2)"),
        (nb.rbf_kernel(X, Y, gamma=0.7), P.rbf_kernel(X, Y, gamma=0.7), "rbf_kernel(gamma=0.7)"),
        (nb.laplacian_kernel(X, Y, gamma=0.4), P.laplacian_kernel(X, Y, gamma=0.4), "laplacian_kernel(gamma=0.4)"),
    ]
    for got, want, name in pairs:
        assert np.shape(got) == (9, 6), f"{name}: X (9, 4), Y (6, 4) -> (9, 6), got {np.shape(got)}"
        assert np.allclose(got, want), f"{name} disagrees with the formula (checked against scikit-learn)"


def test_stage1_default_Y_is_X(nb):
    X = _data(7, 3, 2)
    for name in ["linear_kernel", "polynomial_kernel", "rbf_kernel", "laplacian_kernel"]:
        fn = getattr(nb, name)
        K = np.asarray(fn(X))
        assert K.shape == (7, 7) and np.allclose(K, fn(X, X)), f"{name}(X) should equal {name}(X, X)"
    K = np.asarray(nb.rbf_kernel(X, gamma=2.0))
    assert np.array_equal(np.diag(K), np.ones(7)) and np.array_equal(K, K.T), (
        "rbf_kernel(X) should be exactly symmetric with exactly 1 on the diagonal: pass Y=None through to "
        "pairwise_distances, which guarantees an exactly zero diagonal")


def test_stage1_laplacian_memory(nb):
    X, Y = _data(300, 60, 3), _data(300, 60, 4)
    nb.laplacian_kernel(X[:5], Y[:5])  # warm up
    tracemalloc.start()
    nb.laplacian_kernel(X, Y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    full = 300 * 300 * 60 * 8
    assert peak < full / 4, (
        f"laplacian_kernel peaked at {peak / 1e6:.1f} MB; an (n, m, d) array of differences is {full / 1e6:.0f} MB. "
        "accumulate |X[:, j] - Y[:, j]| into an (n, m) matrix one feature at a time instead")


# ---------------------------------------------------------------- stage 2: explicit features


def test_stage2_hand_example(nb):
    got = np.asarray(nb.quadratic_features(np.array([[1.0, 2.0]])))
    r2 = np.sqrt(2)
    want = [1.0, r2, 2 * r2, 1.0, 4.0, 2 * r2]
    assert got.shape == (1, 6), f"d = 2 gives (d+1)(d+2)/2 = 6 features, got shape {got.shape}"
    assert np.allclose(got, [want]), (
        f"phi(1, 2) = [1, sqrt2 * 1, sqrt2 * 2, 1^2, 2^2, sqrt2 * 1 * 2] = {np.round(want, 4)}, got {np.round(got, 4)}. "
        "order: the constant, the sqrt2 x_i, the squares, then the sqrt2 x_i x_j for i < j")


@pytest.mark.parametrize("d", [1, 3, 5])
def test_stage2_inner_products_are_the_kernel(nb, d):
    X, Y = _data(8, d, 10 + d), _data(5, d, 20 + d)
    PX, PY = np.asarray(nb.quadratic_features(X)), np.asarray(nb.quadratic_features(Y))
    assert PX.shape == (8, (d + 1) * (d + 2) // 2), f"d = {d}: expected {(d + 1) * (d + 2) // 2} features"
    want = (X @ Y.T + 1) ** 2
    assert np.allclose(PX @ PY.T, want), (
        f"d = {d}: phi(x) . phi(y) must equal (x . y + 1)^2 for every pair. expand (x . y + 1)^2 = 1 + 2 sum x_i y_i "
        "+ sum x_i^2 y_i^2 + 2 sum_{i<j} x_i x_j y_i y_j: that's where the sqrt2's come from")


# ---------------------------------------------------------------- stage 3: get_kernel


def test_stage3_dispatch_and_binding(nb):
    X, Y = _data(6, 3, 5), _data(4, 3, 6)
    K = nb.get_kernel("rbf", gamma=0.3, degree=7, coef0=-1.0)
    assert callable(K), "get_kernel returns a function"
    assert np.allclose(K(X, Y), nb.rbf_kernel(X, Y, gamma=0.3)), (
        "get_kernel('rbf', gamma=0.3, degree=7, coef0=-1) behaves like rbf_kernel(X, Y, gamma=0.3): parameters "
        "the kernel doesn't take are ignored")
    assert np.allclose(K(X), nb.rbf_kernel(X, gamma=0.3)), "the returned function's Y defaults to None (i.e. X)"
    P = nb.get_kernel("poly", degree=2, gamma=0.5, coef0=3.0)
    assert np.allclose(P(X, Y), nb.polynomial_kernel(X, Y, degree=2, gamma=0.5, coef0=3.0)), "'poly' binds all three"
    assert np.allclose(nb.get_kernel("poly")(X, Y), nb.polynomial_kernel(X, Y)), (
        "parameters you don't pass keep the kernel's own defaults")
    assert np.allclose(nb.get_kernel("linear", gamma=9.0)(X, Y), X @ Y.T), "'linear' takes no parameters"
    assert np.allclose(nb.get_kernel("laplacian", gamma=2.0)(X, Y), nb.laplacian_kernel(X, Y, gamma=2.0))


def test_stage3_callables_and_errors(nb):
    def my_kernel(X, Y=None):
        return np.ones((len(X), len(X if Y is None else Y)))

    assert nb.get_kernel(my_kernel, gamma=5.0) is my_kernel, "a callable kernel is returned as-is"
    with pytest.raises(ValueError):
        nb.get_kernel("sigmoid")
        pytest.fail("an unknown kernel name should raise ValueError")


# ---------------------------------------------------------------- stage 4: PSD and centring


def test_stage4_is_psd(nb):
    X = _data(40, 2, 7)
    assert nb.is_psd(nb.rbf_kernel(X)) is True, "an RBF Gram matrix is PSD"
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    assert nb.is_psd(-D) is False, "minus the distance matrix has zero trace and isn't zero, so it has a negative eigenvalue"
    assert nb.is_psd(np.tanh(X @ X.T - 1.0)) is False, "the 'sigmoid kernel' tanh(x . y - 1) is not PSD here"
    A = np.array([[2.0, 1.0], [0.0, 2.0]])
    assert nb.is_psd(A) is False, "a non-symmetric matrix is not PSD (by this definition), even with positive eigenvalues"
    w, V = np.linalg.eigh(nb.rbf_kernel(X, gamma=0.01))
    near = V @ np.diag(np.where(np.arange(40) == 0, -1e-12, w)) @ V.T
    near = (near + near.T) / 2
    assert nb.is_psd(near) is True, (
        "an eigenvalue of -1e-12 is round-off, not a violation: allow -tol * max(1, |largest eigenvalue|)")
    assert nb.is_psd(np.diag([1.0, -0.5])) is False, "a clearly negative eigenvalue fails"
    big = np.diag([1e7, 5e6, -1e-3])
    assert nb.is_psd(big) is True, (
        "-1e-3 is round-off next to an eigenvalue of 1e7: the tolerance scales with max(1, |largest eigenvalue|)")
    assert nb.is_psd(np.zeros((0, 0))) is True, "an empty Gram matrix is (vacuously) PSD; don't index into no eigenvalues"


def test_stage4_closure_properties(nb):
    X = _data(30, 3, 8)
    K1, K2 = nb.rbf_kernel(X, gamma=0.5), nb.polynomial_kernel(X, degree=2)
    for name, K in [("K1 + K2", K1 + K2), ("K1 * K2 (elementwise)", K1 * K2), ("3 K1", 3 * K1), ("exp(K_linear)", np.exp(X @ X.T / 4))]:
        assert nb.is_psd(K), f"{name} should be PSD: sums, positive multiples, elementwise products and exp of kernels are kernels"


def test_stage4_center_gram(nb):
    from sklearn.preprocessing import KernelCenterer
    X = _data(12, 4, 9)
    Kc = np.asarray(nb.center_gram(X @ X.T))
    Xc = X - X.mean(axis=0)
    assert np.allclose(Kc, Xc @ Xc.T), (
        "for the linear kernel, centring the Gram matrix must give the Gram matrix of the centred data, Xc Xc^T")
    K = nb.rbf_kernel(X)
    Kc = np.asarray(nb.center_gram(K))
    assert np.allclose(Kc, KernelCenterer().fit_transform(K)), "center_gram(K) = H K H with H = I - 11^T/n"
    assert np.allclose(Kc.sum(axis=0), 0) and np.allclose(Kc.sum(axis=1), 0), "every row and column of H K H sums to 0"
    assert np.allclose(nb.center_gram(Kc), Kc), "centring twice changes nothing (H is a projection)"


# ---------------------------------------------------------------- stage 5: the representer theorem


def test_stage5_worked_example(nb):
    X, y = np.array([[0.0], [1.0]]), np.array([1.0, 2.0])
    K = lambda A, B=None: nb.rbf_kernel(A, B, gamma=1.0)  # noqa: E731
    alpha = np.asarray(nb.interpolate(X, y, K))
    assert np.allclose(alpha, [0.305597, 1.887587], atol=1e-6), (
        f"solve [[1, e^-1], [e^-1, 1]] alpha = [1, 2]: alpha = [0.3056, 1.8876], got {alpha}")
    f = np.asarray(nb.kernel_expansion(alpha, X, np.array([[0.5], [0.0]]), K))
    assert np.allclose(f, [1.708054, 1.0], atol=1e-6), f"f(0.5) = (a1 + a2) e^-0.25 = 1.7081 and f(0) = 1; got {f}"
    assert np.isclose(nb.rkhs_norm(alpha, K(X)), np.sqrt(4.080771), atol=1e-6), (
        "||f||^2 = alpha^T K alpha = alpha^T y = 4.0808 for an interpolant")


def test_stage5_matches_min_norm_in_feature_space(nb):
    rng = np.random.default_rng(11)
    X, y = rng.normal(size=(4, 2)), rng.normal(size=4)
    K = lambda A, B=None: nb.polynomial_kernel(A, B, degree=2)  # noqa: E731
    alpha = np.asarray(nb.interpolate(X, y, K))
    Phi = nb.quadratic_features(X)
    w = np.linalg.pinv(Phi) @ y  # the minimum-norm weights in the 6-dimensional feature space
    Xn = rng.normal(size=(7, 2))
    f = np.asarray(nb.kernel_expansion(alpha, X, Xn, K))
    assert f.shape == (7,), f"one value per new point, got {f.shape}"
    assert np.allclose(f, nb.quadratic_features(Xn) @ w), (
        "the representer theorem: the minimum-norm interpolant in feature space, w = pinv(Phi) y, is the same "
        "function as sum_i alpha_i k(x_i, .) with K alpha = y")
    assert np.isclose(nb.rkhs_norm(alpha, K(X)), np.linalg.norm(w)), "and its RKHS norm is ||w||"
    assert np.allclose(nb.kernel_expansion(alpha, X, X, K), y), "the interpolant passes through every training point"


def test_stage5_singular_gram(nb):
    X = np.array([[0.0], [1.0], [1.0], [2.5]])  # a repeated point: K is singular
    y = np.array([0.0, 1.0, 1.0, -1.0])
    K = lambda A, B=None: nb.rbf_kernel(A, B, gamma=1.0)  # noqa: E731
    alpha = np.asarray(nb.interpolate(X, y, K))
    assert np.all(np.isfinite(alpha)), "a repeated point makes K singular: use least squares (np.linalg.lstsq), not inv"
    assert np.allclose(nb.kernel_expansion(alpha, X, X, K), y, atol=1e-8), "it still interpolates the (consistent) data"
