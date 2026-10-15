"""Checks for exercise.py (matrix calculus).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The gradient
tests compare against this file's own finite differences (not yours), so a bug in your
`numerical_gradient` can't hide a bug in a hand-derived gradient.
"""
import numpy as np


def _fd(f, X, h=1e-6):
    """Reference central differences, entry by entry."""
    X = np.array(X, dtype=float)
    G = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        old = X[idx]
        X[idx] = old + h
        fp = f(X)
        X[idx] = old - h
        fm = f(X)
        X[idx] = old
        G[idx] = (fp - fm) / (2 * h)
    return G


def _close(got, want, label, hint, rtol=1e-5, atol=1e-7):
    got = np.asarray(got, dtype=float)
    want = np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{label}: the gradient must have the shape of the variable, {want.shape}; got {got.shape}"
    err = np.abs(got - want).max()
    assert np.allclose(got, want, rtol=rtol, atol=atol), (
        f"{label}: your gradient differs from finite differences by up to {err:.2e}. {hint}")


def _same(got, other, atol=1e-8):
    """True if `got` has other's shape and values (used only to pick a more specific hint)."""
    got, other = np.asarray(got, dtype=float), np.asarray(other, dtype=float)
    return got.shape == other.shape and np.allclose(got, other, atol=atol)


def _logsumexp(z):
    m = np.max(z)
    return m + np.log(np.sum(np.exp(z - m)))


RNG = np.random.default_rng(0)
A_NS = RNG.normal(size=(4, 4))          # non-symmetric on purpose
A_RECT = RNG.normal(size=(6, 4))
B_VEC = RNG.normal(size=6)
X_VEC = RNG.normal(size=4)


# ---------------------------------------------------------------- stage 1: numerical gradients


def test_stage1_vector_input(nb):
    x = np.array([0.5, -1.0, 2.0])
    g = np.asarray(nb.numerical_gradient(lambda v: np.sum(v**3), x))
    assert g.shape == (3,), f"the gradient of a function of a (3,) array has shape (3,), got {g.shape}"
    assert np.allclose(g, 3 * x**2, atol=1e-6), f"d/dx sum(x^3) = 3x^2 = {3 * x**2}, got {g}"


def test_stage1_matrix_input(nb):
    X = np.random.default_rng(1).normal(size=(3, 4))
    g = np.asarray(nb.numerical_gradient(lambda M: np.sum(np.sin(M)), X))
    assert g.shape == (3, 4), (
        f"X has shape (3, 4), so the gradient must too; got {g.shape}. loop over every index with "
        "np.ndindex(X.shape) instead of assuming a vector.")
    assert np.allclose(g, np.cos(X), atol=1e-8), "d/dX sum(sin(X)) = cos(X), entrywise"


def test_stage1_central_difference_with_given_h(nb):
    g = np.asarray(nb.numerical_gradient(lambda v: v[0] ** 3, np.array([1.0]), h=0.1))
    assert np.isclose(g[0], 3.01, atol=1e-12), (
        f"with h = 0.1, the CENTRAL difference of x^3 at x = 1 is ((1.1)^3 - (0.9)^3) / 0.2 = 3.01 "
        f"exactly; got {g[0]:.6f}. "
        + ("3.31 means you used a forward difference (f(x+h) - f(x)) / h." if np.isclose(g[0], 3.31)
           else "check that you use the h argument and divide by 2h."))


def test_stage1_integer_input_and_no_mutation(nb):
    x = np.array([1, 2, 3])
    g = np.asarray(nb.numerical_gradient(lambda v: v @ v, x))
    assert np.allclose(g, [2.0, 4.0, 6.0], atol=1e-6), (
        f"d/dx x.x = 2x = [2, 4, 6] for x = [1, 2, 3] (an INTEGER array), got {g}. adding h = 1e-6 to an "
        "integer array does nothing: work on np.array(X, dtype=float).")
    X = np.random.default_rng(2).normal(size=(2, 2))
    X0 = X.copy()
    nb.numerical_gradient(lambda M: np.sum(M**2), X)
    assert np.array_equal(X, X0), "numerical_gradient must not change X; perturb a copy and restore each entry"


def test_stage1_non_separable_functions(nb):
    # functions where every entry interacts with the others: a perturbation that isn't undone
    # before moving to the next entry shows up as a wrong gradient here
    M = np.random.default_rng(20).normal(size=(3, 4))
    g = np.asarray(nb.numerical_gradient(lambda X: np.sum((X @ X.T) ** 2), M))
    want = 4 * (M @ M.T) @ M
    assert np.allclose(g, want, rtol=1e-5, atol=1e-6), (
        "for f(M) = sum((M M^T)^2), whose gradient is 4 (M M^T) M, your numerical gradient is off by up to "
        f"{np.abs(g - want).max():.2e}. after nudging entry idx up and down, RESTORE it to its old value "
        "before moving on, or every later entry is differentiated at the wrong point.")
    a = np.array([0.5, -1.0, 2.0, 1.5])
    v = np.array([1.2, 0.7, -0.9, 1.1])
    f = lambda w: (a @ w) ** 2 * np.prod(w)
    want = 2 * (a @ v) * a * np.prod(v) + (a @ v) ** 2 * np.prod(v) / v
    g = np.asarray(nb.numerical_gradient(f, v))
    assert np.allclose(g, want, rtol=1e-6), (
        f"for f(v) = (a.v)^2 prod(v), expected {np.round(want, 6).tolist()}, got {np.round(g, 6).tolist()}")


# ---------------------------------------------------------------- stage 2: gradient checker


def test_stage2_non_separable_correct_gradient(nb):
    M = np.random.default_rng(21).normal(size=(3, 4))
    err = nb.check_gradient(lambda X: np.sum((X @ X.T) ** 2), lambda X: 4 * (X @ X.T) @ X, M)
    assert err < 1e-8, f"a correct gradient of sum((M M^T)^2) should give a tiny relative error, got {err:.2e}"


def test_stage2_passes_h_through(nb):
    x = np.array([1.0, -2.0, 0.5])
    f, g = (lambda v: np.sum(v**3)), (lambda v: 3 * v**2)
    # with step h the central difference of v^3 is exactly 3 v^2 + h^2 in every entry
    g_fd = 3 * x**2 + 0.01
    want = np.linalg.norm(g(x) - g_fd) / (np.linalg.norm(g(x)) + np.linalg.norm(g_fd))
    got = nb.check_gradient(f, g, x, h=0.1)
    assert np.isclose(got, want, rtol=1e-6), (
        f"check_gradient(..., h=0.1) on f = sum(v^3) should give relative error {want:.6e} (the central "
        f"difference with h = 0.1 is 3v^2 + 0.01), got {got:.6e}. pass h on to numerical_gradient.")


def test_stage2_grad_f_gets_a_copy(nb):
    def grad_that_scribbles(v):
        out = 2 * v
        v[:] = 0.0  # a badly behaved gradient function that overwrites its input
        return out

    x = np.array([1.0, 2.0, -1.0])
    x0 = x.copy()
    err = nb.check_gradient(lambda v: v @ v, grad_that_scribbles, x)
    assert np.array_equal(x, x0), "check_gradient must not let grad_f change the caller's X: pass it a copy"
    assert err < 1e-8, (
        f"the gradient is correct, so the error should be tiny, got {err:.2e}: grad_f overwrote X before the "
        "finite differences ran. give grad_f its own copy of X.")


def test_stage2_correct_gradient_passes(nb):
    f = lambda v: np.sum(np.exp(v)) + v @ v
    x = np.random.default_rng(3).normal(size=5)
    err = nb.check_gradient(f, lambda v: np.exp(v) + 2 * v, x)
    assert err < 1e-8, f"a correct gradient should give a relative error around 1e-10, got {err:.2e}"


def test_stage2_exact_relative_error(nb):
    f = lambda v: v @ v
    x = np.array([1.0, -2.0, 0.5])
    err = nb.check_gradient(f, lambda v: 4 * v, x)  # twice the true gradient 2v
    assert np.isclose(err, 1 / 3, atol=1e-7), (
        f"with g = 2 * g_fd, the relative error ||g - g_fd|| / (||g|| + ||g_fd||) = 1/(2 + 1) = 1/3; got {err:.6f}. "
        "use exactly the contract's formula, with g = grad_f(X) and g_fd from numerical_gradient.")


def test_stage2_zero_gradient_and_shapes(nb):
    err = nb.check_gradient(lambda M: 0.0, lambda M: np.zeros_like(M), np.ones((2, 3)))
    assert err == 0.0 or err < 1e-12, (
        f"when both gradients are zero the error should be 0, not nan; got {err}. "
        "the denominator is max(||g|| + ||g_fd||, 1e-12).")
    X = np.random.default_rng(4).normal(size=(3, 2))
    X0 = X.copy()
    err = nb.check_gradient(lambda M: np.sum(M**3), lambda M: 3 * M**2, X)
    assert err < 1e-8 and np.array_equal(X, X0), "check_gradient should work for matrix inputs and leave X unchanged"


# ---------------------------------------------------------------- stage 3: vector gradients


def test_stage3_grad_linear(nb):
    a = np.array([1.0, -2.0, 3.0, 0.5])
    _close(nb.grad_linear(a, X_VEC), _fd(lambda x: a @ x, X_VEC), "grad_linear", "d(a^T x)/dx = a.")


def test_stage3_grad_quadratic_nonsymmetric(nb):
    got = nb.grad_quadratic(A_NS, X_VEC)
    want = _fd(lambda x: x @ A_NS @ x, X_VEC)
    hint = "the gradient of x^T A x is (A + A^T) x."
    if _same(got, 2 * A_NS @ X_VEC):
        hint += " you returned 2 A x, which is only right when A is symmetric; this A isn't."
    _close(got, want, "grad_quadratic", hint)


def test_stage3_grad_least_squares_worked_example(nb):
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([1.0, 1.0])
    x = np.array([1.0, -1.0])
    got = np.asarray(nb.grad_least_squares(A, b, x))
    assert np.allclose(got, [-16.0, -24.0]), (
        f"worked example: A = [[1,2],[3,4]], b = [1,1], x = [1,-1] gives r = Ax - b = [-2,-2] and "
        f"gradient 2 A^T r = [-16, -24]; got {got.tolist()}")
    _close(nb.grad_least_squares(A_RECT, B_VEC, X_VEC),
           _fd(lambda x: np.sum((A_RECT @ x - B_VEC) ** 2), X_VEC),
           "grad_least_squares on a (6, 4) A", "2 A^T (A x - b): note the transpose, which makes the shape (n,).")


def test_stage3_grad_norm(nb):
    got = nb.grad_norm(X_VEC)
    hint = "the gradient of ||x|| (not squared!) is x / ||x||, a unit vector."
    if _same(got, 2 * X_VEC):
        hint += " you returned 2x, the gradient of ||x||^2."
    _close(got, _fd(lambda x: np.linalg.norm(x), X_VEC), "grad_norm", hint)


# ---------------------------------------------------------------- stage 4: matrix gradients


def test_stage4_grad_trace(nb):
    X = np.random.default_rng(5).normal(size=(4, 4))
    got = nb.grad_trace(A_NS, X)
    hint = "d tr(AX)/dX_ij = A_ji, so the gradient is A^T."
    if _same(got, A_NS):
        hint += " you returned A itself."
    _close(got, _fd(lambda M: np.trace(A_NS @ M), X), "grad_trace", hint)


def test_stage4_grad_bilinear(nb):
    a, b = np.array([1.0, 2.0, -1.0]), np.array([0.5, -3.0])
    X = np.random.default_rng(6).normal(size=(3, 2))
    got = nb.grad_bilinear(a, b, X)
    _close(got, _fd(lambda M: a @ M @ b, X), "grad_bilinear",
           "d(a^T X b)/dX_ij = a_i b_j: the outer product a b^T, shape (3, 2) like X.")


def test_stage4_grad_matrix_least_squares(nb):
    rng = np.random.default_rng(7)
    X, Y, W = rng.normal(size=(8, 3)), rng.normal(size=(8, 2)), rng.normal(size=(3, 2))
    _close(nb.grad_matrix_least_squares(X, Y, W), _fd(lambda M: np.sum((X @ M - Y) ** 2), W),
           "grad_matrix_least_squares",
           "it's the vector least-squares gradient applied to every column of W at once: 2 X^T (X W - Y).")


def test_stage4_grad_logdet(nb):
    X = np.random.default_rng(8).normal(size=(4, 4)) + 2 * np.eye(4)
    got = nb.grad_logdet(X)
    hint = "d log|det X| / dX = X^{-T}, the inverse TRANSPOSED."
    if _same(got, np.linalg.inv(X)):
        hint += " you returned X^{-1}, which is only right for symmetric X."
    _close(got, _fd(lambda M: np.log(abs(np.linalg.det(M))), X), "grad_logdet", hint)


# ---------------------------------------------------------------- stage 5: ML loss gradients


def test_stage5_softmax_cross_entropy(nb):
    z = np.array([2.0, 1.0, 0.1])
    got = np.asarray(nb.grad_softmax_cross_entropy(z, 0))
    p = np.exp(z) / np.exp(z).sum()
    assert np.allclose(got, p - np.array([1.0, 0.0, 0.0])), (
        f"for z = [2, 1, 0.1] and y = 0 the gradient is softmax(z) - e_0 = {np.round(p - [1, 0, 0], 4).tolist()}, "
        f"got {np.round(got, 4).tolist()}")
    z = np.random.default_rng(9).normal(size=6) * 3
    _close(nb.grad_softmax_cross_entropy(z, 4), _fd(lambda v: _logsumexp(v) - v[4], z),
           "grad_softmax_cross_entropy", "softmax(z) minus the one-hot vector of the true class.")
    big = np.asarray(nb.grad_softmax_cross_entropy(np.array([1000.0, 0.0]), 1))
    assert np.all(np.isfinite(big)) and np.allclose(big, [1.0, -1.0]), (
        "with logits [1000, 0] the gradient should be [1, -1]; you got overflow. subtract max(z) "
        "before exponentiating (see ch 00 · topic 03).")


def test_stage5_logistic(nb):
    rng = np.random.default_rng(10)
    X = rng.normal(size=(20, 3))
    y = (rng.random(20) < 0.5).astype(float)
    w = rng.normal(size=3)

    def loss(v):
        z = X @ v
        return np.mean(np.logaddexp(0.0, z) - y * z)

    got = nb.grad_logistic(X, y, w)
    hint = "X^T (sigmoid(X w) - y) / n: the loss is a MEAN over the n samples."
    if _same(got, 20 * _fd(loss, w), atol=1e-5):
        hint += " yours is n times too big: you summed instead of averaging."
    _close(got, _fd(loss, w), "grad_logistic", hint)
