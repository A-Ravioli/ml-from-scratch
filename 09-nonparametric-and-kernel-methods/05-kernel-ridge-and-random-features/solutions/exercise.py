import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, Transformer, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.kernels import get_kernel  # yours, from topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # kernel ridge & random features · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you a
    class or function to write, and checks it live as you edit. `KernelRidge`,
    `RandomFourierFeatures` and `Nystroem` become part of `mlfs.kernels`; they use your
    `get_kernel` from topic 03 (launch with `MLFS_REFERENCE=kernels marimo edit exercise.py`
    if that isn't finished).

    | stage | you build |
    |---|---|
    | 1 | `KernelRidge`: regularized least squares in an RKHS, in closed form → `mlfs.kernels` |
    | 2 | `RandomFourierFeatures`: explicit random features whose inner products approximate the RBF kernel → `mlfs.kernels` |
    | 3 | `Nystroem`: features from a random subset of landmark points → `mlfs.kernels` |
    | 4 | `primal_ridge`, `gram_error`: linear ridge on approximate features, and how good the approximation is |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class KernelRidge(Regressor):
    """kernel ridge regression: f(x) = sum_i dual_coef_i k(x_i, x), with dual_coef_ = (K + alpha I)^-1 y (no intercept)."""

    def __init__(self, alpha=1.0, *, kernel="rbf", gamma=1.0, degree=3, coef0=1.0):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        """stage 1: store X_fit_ and kernel_, and solve (K + alpha I) dual_coef_ = y; return self."""
        X, y = check_X_y(X, y)
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        self.kernel_ = get_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        self.X_fit_ = X
        K = self.kernel_(X)
        A = K + self.alpha * np.eye(len(X))
        try:
            self.dual_coef_ = np.linalg.solve(A, y.astype(float))
        except np.linalg.LinAlgError:  # alpha = 0 with repeated points: K is singular, use least squares
            self.dual_coef_ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
        return self

    def predict(self, X):
        """stage 1: K(X, X_fit_) @ dual_coef_, shape (m,)."""
        X, _ = check_X_y(X)
        return self.kernel_(X, self.X_fit_) @ self.dual_coef_


@app.cell(hide_code=True)
def _():
    checks.run(1, KernelRidge=KernelRidge)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class RandomFourierFeatures(Transformer):
    """random features z(x) = sqrt(2/D) cos(x W + b) with z(x) . z(y) ~ exp(-gamma ||x - y||^2)."""

    def __init__(self, n_components=100, *, gamma=1.0, rng=None):
        self.n_components = n_components
        self.gamma = gamma
        self.rng = rng

    def fit(self, X, y=None):
        """stage 2: draw random_weights_ (d, D) from N(0, 2 gamma) and random_offset_ (D,) from U[0, 2 pi) with as_rng(rng); return self."""
        X, _ = check_X_y(X)
        gen = as_rng(self.rng)
        D = self.n_components
        self.random_weights_ = np.sqrt(2.0 * self.gamma) * gen.standard_normal((X.shape[1], D))
        self.random_offset_ = gen.uniform(0.0, 2.0 * np.pi, size=D)
        return self

    def transform(self, X):
        """stage 2: sqrt(2 / D) cos(X @ random_weights_ + random_offset_), shape (n, D)."""
        X, _ = check_X_y(X)
        if X.shape[1] != self.random_weights_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, fit saw {self.random_weights_.shape[0]}")
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.random_weights_ + self.random_offset_)


@app.cell(hide_code=True)
def _():
    checks.run(2, RandomFourierFeatures=RandomFourierFeatures)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class Nystroem(Transformer):
    """Nyström features z(x) = K(x, L) K_LL^(-1/2) from m landmark rows L of the training data, so z z^T ~ K."""

    def __init__(self, n_components=100, *, kernel="rbf", gamma=1.0, degree=3, coef0=1.0, rng=None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rng = rng

    def fit(self, X, y=None):
        """stage 3: pick min(n_components, n) landmark rows without replacement (component_indices_, components_) and store normalization_ = K_LL^(-1/2); return self."""
        X, _ = check_X_y(X)
        gen = as_rng(self.rng)
        m = min(self.n_components, len(X))
        self.kernel_ = get_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        self.component_indices_ = gen.choice(len(X), size=m, replace=False)
        self.components_ = X[self.component_indices_]
        w, U = np.linalg.eigh(self.kernel_(self.components_))
        w = np.maximum(w, 1e-12)
        self.normalization_ = (U / np.sqrt(w)) @ U.T
        return self

    def transform(self, X):
        """stage 3: K(X, components_) @ normalization_, shape (n, m)."""
        X, _ = check_X_y(X)
        return self.kernel_(X, self.components_) @ self.normalization_


@app.cell(hide_code=True)
def _():
    checks.run(3, Nystroem=Nystroem)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def primal_ridge(Z, y, alpha):
    """stage 4: ridge weights w (D,) on features Z (n, D): solve (Z^T Z + alpha I) w = Z^T y (no intercept)."""
    Z = np.asarray(Z, dtype=float)
    return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ np.asarray(y, dtype=float))


@app.function
def gram_error(Z, K):
    """stage 4: relative Frobenius error ||Z Z^T - K||_F / ||K||_F of an approximate feature map."""
    Z = np.asarray(Z, dtype=float)
    K = np.asarray(K, dtype=float)
    return float(np.linalg.norm(Z @ Z.T - K) / np.linalg.norm(K))


@app.cell(hide_code=True)
def _():
    checks.run(4, primal_ridge=primal_ridge, gram_error=gram_error, KernelRidge=KernelRidge, Nystroem=Nystroem,
               RandomFourierFeatures=RandomFourierFeatures)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import KernelRidge, Nystroem` gives you your
    own code. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
