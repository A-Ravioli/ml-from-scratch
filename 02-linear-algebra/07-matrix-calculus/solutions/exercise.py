import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import solve  # yours, from topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # matrix calculus · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `numerical_gradient` and
    `check_gradient` become `mlfs.numgrad`, the gradient checker every later chapter uses to
    test its derivatives. the ten `grad_*` functions are your hand-derived gradients: derive
    each one on paper first, then code it.

    | stage | you build |
    |---|---|
    | 1 | `numerical_gradient`: central differences for any array shape |
    | 2 | `check_gradient`: the relative error between two gradients |
    | 3 | four gradients with respect to a vector |
    | 4 | four gradients with respect to a matrix |
    | 5 | the gradients of softmax cross-entropy and logistic loss |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def numerical_gradient(f, X: np.ndarray, *, h: float = 1e-6) -> np.ndarray:
    """stage 1: the gradient of a scalar function f at X by central differences.

    f takes an array shaped like X and returns a float. X can have any shape (vector, matrix,
    ...); the result has X's shape, entry [i] = (f(X + h e_i) - f(X - h e_i)) / (2h).
    X is not modified (perturb a float copy, one entry at a time).
    """
    Xw = np.array(X, dtype=float)
    G = np.zeros_like(Xw)
    for idx in np.ndindex(Xw.shape):
        old = Xw[idx]
        Xw[idx] = old + h
        f_plus = f(Xw)
        Xw[idx] = old - h
        f_minus = f(Xw)
        Xw[idx] = old
        G[idx] = (f_plus - f_minus) / (2 * h)
    return G


@app.cell(hide_code=True)
def _():
    checks.run(1, numerical_gradient=numerical_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def check_gradient(f, grad_f, X: np.ndarray, *, h: float = 1e-6) -> float:
    """stage 2: relative error between an analytic gradient and central differences.

    g = grad_f(X), g_fd = numerical_gradient(f, X, h=h); returns the float
    ||g - g_fd|| / max(||g|| + ||g_fd||, 1e-12) (Frobenius norms, any shape).
    rule of thumb: < 1e-7 is right, > 1e-3 is a bug.
    """
    X = np.array(X, dtype=float)
    g = np.asarray(grad_f(X.copy()), dtype=float)
    g_fd = numerical_gradient(f, X, h=h)
    return float(np.linalg.norm(g - g_fd) / max(np.linalg.norm(g) + np.linalg.norm(g_fd), 1e-12))


@app.cell(hide_code=True)
def _():
    checks.run(2, check_gradient=check_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def grad_linear(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """stage 3: gradient of f(x) = a^T x with respect to x (n,). returns (n,)."""
    return np.array(a, dtype=float)


@app.function
def grad_quadratic(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """stage 3: gradient of f(x) = x^T A x with respect to x (n,), for any square A (n, n). returns (n,)."""
    return (A + A.T) @ x


@app.function
def grad_least_squares(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """stage 3: gradient of f(x) = ||A x - b||^2 with respect to x (n,); A (m, n), b (m,). returns (n,)."""
    return 2.0 * A.T @ (A @ x - b)


@app.function
def grad_norm(x: np.ndarray) -> np.ndarray:
    """stage 3: gradient of f(x) = ||x||_2 (not squared) with respect to x (n,), x != 0. returns (n,)."""
    return x / np.linalg.norm(x)


@app.cell(hide_code=True)
def _():
    checks.run(3, grad_linear=grad_linear, grad_quadratic=grad_quadratic,
               grad_least_squares=grad_least_squares, grad_norm=grad_norm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def grad_trace(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """stage 4: gradient of f(X) = tr(A X) with respect to X (n, m); A (m, n). returns (n, m)."""
    return np.array(A.T, dtype=float)


@app.function
def grad_bilinear(a: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """stage 4: gradient of f(X) = a^T X b with respect to X (m, n); a (m,), b (n,). returns (m, n)."""
    return np.outer(a, b)


@app.function
def grad_matrix_least_squares(X: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """stage 4: gradient of f(W) = ||X W - Y||_F^2 with respect to W (d, k); X (n, d), Y (n, k). returns (d, k)."""
    return 2.0 * X.T @ (X @ W - Y)


@app.function
def grad_logdet(X: np.ndarray) -> np.ndarray:
    """stage 4: gradient of f(X) = log|det X| with respect to an invertible X (n, n). returns (n, n).

    use your solve (from mlfs.linalg) rather than forming an inverse by hand.
    """
    n = X.shape[0]
    return solve(X.T, np.eye(n))  # X^{-T}


@app.cell(hide_code=True)
def _():
    checks.run(4, grad_trace=grad_trace, grad_bilinear=grad_bilinear,
               grad_matrix_least_squares=grad_matrix_least_squares, grad_logdet=grad_logdet)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def grad_softmax_cross_entropy(z: np.ndarray, y: int) -> np.ndarray:
    """stage 5: gradient of f(z) = -log softmax(z)[y] = logsumexp(z) - z[y] with respect to the logits z (k,).

    y is the integer index of the true class. returns (k,). must not overflow for large logits.
    """
    e = np.exp(z - np.max(z))
    p = e / e.sum()
    p[y] -= 1.0
    return p


@app.function
def grad_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """stage 5: gradient of the mean logistic loss with respect to w (d,).

    f(w) = (1/n) sum_i [log(1 + exp(x_i^T w)) - y_i x_i^T w], X (n, d), labels y (n,) in {0, 1}.
    returns (d,).
    """
    z = X @ w
    p = np.exp(-np.logaddexp(0.0, -z))  # sigmoid(z), computed without overflow
    return X.T @ (p - y) / X.shape[0]


@app.cell(hide_code=True)
def _():
    checks.run(5, grad_softmax_cross_entropy=grad_softmax_cross_entropy, grad_logistic=grad_logistic)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
