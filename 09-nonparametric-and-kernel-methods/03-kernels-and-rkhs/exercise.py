import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.utils import pairwise_distances  # yours, from ch 00 · topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # kernels & RKHS · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. the kernel library becomes part of
    `mlfs.kernels`: your SVM (topic 04), kernel ridge regression and Nyström features
    (topic 05) all build their Gram matrices with it, through `get_kernel`.

    | stage | you build |
    |---|---|
    | 1 | `linear_kernel`, `polynomial_kernel`, `rbf_kernel`, `laplacian_kernel`: Gram matrices without loops over pairs → `mlfs.kernels` |
    | 2 | `quadratic_features`: an explicit feature map whose inner products *are* a kernel |
    | 3 | `get_kernel`: one way for every estimator to name its kernel → `mlfs.kernels` |
    | 4 | `is_psd`, `center_gram`: checking Mercer's condition, and centring in feature space → `mlfs.kernels` |
    | 5 | `interpolate`, `kernel_expansion`, `rkhs_norm`: the representer theorem, in code |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def linear_kernel(X, Y=None):
    """stage 1: the Gram matrix <x_i, y_j>, shape (n, m); Y defaults to X."""
    raise NotImplementedError("stage1: implement linear_kernel")


@app.function
def polynomial_kernel(X, Y=None, *, degree=3, gamma=1.0, coef0=1.0):
    """stage 1: (gamma <x, y> + coef0)^degree for every pair of rows, shape (n, m)."""
    raise NotImplementedError("stage1: implement polynomial_kernel")


@app.function
def rbf_kernel(X, Y=None, *, gamma=1.0):
    """stage 1: exp(-gamma ||x - y||^2) for every pair of rows, shape (n, m)."""
    raise NotImplementedError("stage1: implement rbf_kernel")


@app.function
def laplacian_kernel(X, Y=None, *, gamma=1.0):
    """stage 1: exp(-gamma ||x - y||_1) for every pair of rows, shape (n, m), without an (n, m, d) array."""
    raise NotImplementedError("stage1: implement laplacian_kernel")


@app.cell(hide_code=True)
def _():
    checks.run(
        1,
        linear_kernel=linear_kernel,
        polynomial_kernel=polynomial_kernel,
        rbf_kernel=rbf_kernel,
        laplacian_kernel=laplacian_kernel,
    )
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def quadratic_features(X):
    """stage 2: phi(X), shape (n, (d+1)(d+2)/2), with phi(x) . phi(y) = (x . y + 1)^2: [1, sqrt2 x_i, x_i^2, sqrt2 x_i x_j (i < j)]."""
    raise NotImplementedError("stage2: implement quadratic_features")


@app.cell(hide_code=True)
def _():
    checks.run(2, quadratic_features=quadratic_features)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def get_kernel(kernel, **params):
    """stage 3: a function K(X, Y=None) for 'linear' | 'poly' | 'rbf' | 'laplacian' with the relevant params bound; a callable is returned as-is."""
    raise NotImplementedError("stage3: implement get_kernel")


@app.cell(hide_code=True)
def _():
    checks.run(3, get_kernel=get_kernel, laplacian_kernel=laplacian_kernel, polynomial_kernel=polynomial_kernel, rbf_kernel=rbf_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def is_psd(K, *, tol=1e-08):
    """stage 4: True if K is symmetric and its smallest eigenvalue is >= -tol * max(1, |largest eigenvalue|)."""
    raise NotImplementedError("stage4: implement is_psd")


@app.function
def center_gram(K):
    """stage 4: H K H with H = I - 11^T / n: the Gram matrix of the features after subtracting their mean."""
    raise NotImplementedError("stage4: implement center_gram")


@app.cell(hide_code=True)
def _():
    checks.run(4, is_psd=is_psd, center_gram=center_gram, polynomial_kernel=polynomial_kernel, rbf_kernel=rbf_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def interpolate(X, y, kernel):
    """stage 5: the coefficients alpha (n,) of the minimum-RKHS-norm interpolant f = sum_i alpha_i k(x_i, .), from K alpha = y (least squares if K is singular)."""
    raise NotImplementedError("stage5: implement interpolate")


@app.function
def kernel_expansion(alpha, X_train, X_new, kernel):
    """stage 5: f(x) = sum_i alpha_i k(x_i, x) at each row of X_new, shape (m,)."""
    raise NotImplementedError("stage5: implement kernel_expansion")


@app.function
def rkhs_norm(alpha, K):
    """stage 5: ||f||_H = sqrt(alpha^T K alpha) for f = sum_i alpha_i k(x_i, .), with K the Gram matrix of the x_i."""
    raise NotImplementedError("stage5: implement rkhs_norm")


@app.cell(hide_code=True)
def _():
    checks.run(5, interpolate=interpolate, kernel_expansion=kernel_expansion, rkhs_norm=rkhs_norm,
               quadratic_features=quadratic_features, polynomial_kernel=polynomial_kernel, rbf_kernel=rbf_kernel)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import get_kernel, rbf_kernel` gives you
    your own kernels. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
