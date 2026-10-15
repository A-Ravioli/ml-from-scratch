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
    # kernel ridge & random features · build

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
        raise NotImplementedError("stage1: implement KernelRidge.fit")

    def predict(self, X):
        """stage 1: K(X, X_fit_) @ dual_coef_, shape (m,)."""
        raise NotImplementedError("stage1: implement KernelRidge.predict")


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
        raise NotImplementedError("stage2: implement RandomFourierFeatures.fit")

    def transform(self, X):
        """stage 2: sqrt(2 / D) cos(X @ random_weights_ + random_offset_), shape (n, D)."""
        raise NotImplementedError("stage2: implement RandomFourierFeatures.transform")


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
        raise NotImplementedError("stage3: implement Nystroem.fit")

    def transform(self, X):
        """stage 3: K(X, components_) @ normalization_, shape (n, m)."""
        raise NotImplementedError("stage3: implement Nystroem.transform")


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
    raise NotImplementedError("stage4: implement primal_ridge")


@app.function
def gram_error(Z, K):
    """stage 4: relative Frobenius error ||Z Z^T - K||_F / ||K||_F of an approximate feature map."""
    raise NotImplementedError("stage4: implement gram_error")


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
