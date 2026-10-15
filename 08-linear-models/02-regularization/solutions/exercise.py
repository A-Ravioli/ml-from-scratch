import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.linalg import cho_solve, cholesky  # yours, from ch 02 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. `Ridge`, `Lasso`, `ElasticNet`,
    `ridge_path` and `lasso_path` join `mlfs.linear`. they use your `Regressor` and
    `check_X_y` (ch 07) and your Cholesky (ch 02); if those aren't finished, launch with
    `MLFS_REFERENCE=base,linalg marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `Ridge`: the closed-form ridge estimator, intercept unpenalized → `mlfs.linear` |
    | 2 | `ridge_path`, `ridge_dof`: the whole path at once from one SVD, and its effective degrees of freedom → `mlfs.linear` |
    | 3 | `soft_threshold`, `coordinate_descent`: the lasso / elastic-net solver |
    | 4 | `Lasso`, `ElasticNet`: estimators on top of it → `mlfs.linear` |
    | 5 | `lasso_alpha_max`, `lasso_path`: where sparsity starts, and a warm-started path → `mlfs.linear` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.function
def center(X, y, fit_intercept):
    """given: (Xc, yc, x_mean, y_mean). with fit_intercept, subtract the column means of X and
    the mean of y; without, return them unchanged with zero means. the intercept of a
    centered fit is y_mean - x_mean @ coef (lesson, topic 01)."""
    X, y = check_X_y(X, y)
    y = y.astype(np.float64)
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), float(y.mean())
        return X - x_mean, y - y_mean, x_mean, y_mean
    return X, y, np.zeros(X.shape[1]), 0.0


@app.function
def linear_predict(model, X):
    """given: X @ model.coef_ + model.intercept_, after checking the number of features."""
    X, _ = check_X_y(X)
    if X.shape[1] != len(model.coef_):
        raise ValueError(f"fitted with {len(model.coef_)} features, got {X.shape[1]}")
    return X @ model.coef_ + model.intercept_


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class Ridge(Regressor):
    """ridge regression: minimizes ||y - X w - b||^2 + alpha ||w||^2 (b unpenalized)."""

    def __init__(self, alpha=1.0, *, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """stage 1: the closed-form ridge solution; returns self.

        center with the given center(X, y, fit_intercept) helper, then solve
        (Xc^T Xc + alpha I) w = Xc^T yc with your cholesky and cho_solve (mlfs.linalg).
        coef_ = w (d,), intercept_ = y_mean - x_mean @ w (a float). alpha < 0 raises ValueError.
        """
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)
        A = Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1])
        self.coef_ = np.asarray(cho_solve(cholesky(A), Xc.T @ yc), dtype=np.float64)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        """stage 1: X @ coef_ + intercept_ (use the given linear_predict)."""
        return linear_predict(self, X)


@app.cell(hide_code=True)
def _():
    checks.run(1, Ridge=Ridge)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def ridge_path(X, y, alphas):
    """stage 2: ridge coefficients for every alpha, from ONE thin SVD of X.

    X (n, d), y (n,), alphas: a sequence of k values >= 0. no intercept (center the data
    yourself first). with X = U diag(s) V^T (np.linalg.svd(X, full_matrices=False)), the
    solution for alpha is V diag(s / (s^2 + alpha)) U^T y. singular values below
    1e-12 * s.max() are rounding noise standing in for exact zeros: use a factor of 0 for them
    (so alpha = 0 gives the minimum-norm least-squares solution). returns coefs (k, d).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    keep = s > 1e-12 * s.max()
    Uty = U.T @ y
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1, 1)
    shrink = np.where(keep, s / (s**2 + alphas + ~keep), 0.0)  # (k, r); "+ ~keep" avoids 0/0
    return (shrink * Uty) @ Vt


@app.function
def ridge_dof(X, alphas):
    """stage 2: the effective degrees of freedom of ridge, sum_j s_j^2 / (s_j^2 + alpha), for each
    alpha (s = singular values of X, the same 1e-12 * s.max() cutoff as ridge_path).
    returns a float array (k,)."""
    s = np.linalg.svd(np.asarray(X, dtype=np.float64), compute_uv=False)
    s = s[s > 1e-12 * s.max()]
    alphas = np.asarray(alphas, dtype=np.float64).reshape(-1, 1)
    return (s**2 / (s**2 + alphas)).sum(axis=1)


@app.cell(hide_code=True)
def _():
    checks.run(2, ridge_path=ridge_path, ridge_dof=ridge_dof, Ridge=Ridge)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def soft_threshold(z, t):
    """stage 3: sign(z) * max(|z| - t, 0), elementwise (z a float or an array, t >= 0)."""
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


@app.function
def coordinate_descent(X, y, l1, l2=0.0, *, w0=None, max_iters=1000, tol=1e-06):
    """stage 3: minimize (1/2n)||y - X w||^2 + l1 ||w||_1 + (l2/2) ||w||^2 by cyclic coordinate descent.

    no intercept (the callers center). start from w0 (default zeros (d,)); keep the residual
    r = y - X w up to date. one sweep visits j = 0, 1, ..., d-1 and sets
        w_j = soft_threshold(X_j^T r / n + z_j w_j, l1) / (z_j + l2),   z_j = ||X_j||^2 / n
    (a column of zeros gets w_j = 0). stop after the first sweep in which no coordinate
    changed by more than tol, or after max_iters sweeps. returns (w (d,), n_sweeps).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d) if w0 is None else np.array(w0, dtype=np.float64)
    r = y - X @ w
    z = (X**2).sum(axis=0) / n
    sweeps = 0
    while sweeps < max_iters:
        sweeps += 1
        biggest = 0.0
        for j in range(d):
            old = w[j]
            if z[j] == 0.0:
                new = 0.0
            else:
                rho = X[:, j] @ r / n + z[j] * old
                new = float(soft_threshold(rho, l1)) / (z[j] + l2)
            if new != old:
                r -= X[:, j] * (new - old)
                w[j] = new
                biggest = max(biggest, abs(new - old))
        if biggest <= tol:
            break
    return w, sweeps


@app.cell(hide_code=True)
def _():
    checks.run(3, soft_threshold=soft_threshold, coordinate_descent=coordinate_descent)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class Lasso(Regressor):
    """the lasso: minimizes (1/2n)||y - X w - b||^2 + alpha ||w||_1 (b unpenalized)."""

    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iters=1000, tol=1e-06):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X, y):
        """stage 4: center, run coordinate_descent with l1 = alpha and l2 = 0, store coef_,
        intercept_ (a float) and n_iter_ (sweeps); returns self. alpha < 0 raises ValueError."""
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)
        self.coef_, self.n_iter_ = coordinate_descent(Xc, yc, self.alpha, 0.0, max_iters=self.max_iters,
                                                      tol=self.tol)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        """stage 4: X @ coef_ + intercept_."""
        return linear_predict(self, X)


@app.class_definition
class ElasticNet(Regressor):
    """minimizes (1/2n)||y - Xw - b||^2 + alpha l1_ratio ||w||_1 + alpha (1 - l1_ratio)/2 ||w||^2."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, *, fit_intercept=True, max_iters=1000, tol=1e-06):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X, y):
        """stage 4: like Lasso.fit with l1 = alpha * l1_ratio and l2 = alpha * (1 - l1_ratio).
        alpha < 0 or l1_ratio outside [0, 1] raises ValueError."""
        if self.alpha < 0 or not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError(f"need alpha >= 0 and 0 <= l1_ratio <= 1, got {self.alpha}, {self.l1_ratio}")
        Xc, yc, x_mean, y_mean = center(X, y, self.fit_intercept)
        l1, l2 = self.alpha * self.l1_ratio, self.alpha * (1.0 - self.l1_ratio)
        self.coef_, self.n_iter_ = coordinate_descent(Xc, yc, l1, l2, max_iters=self.max_iters, tol=self.tol)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        """stage 4: X @ coef_ + intercept_."""
        return linear_predict(self, X)


@app.cell(hide_code=True)
def _():
    checks.run(4, Lasso=Lasso, ElasticNet=ElasticNet)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def lasso_alpha_max(X, y):
    """stage 5: the smallest alpha at which the lasso solution (no intercept) is all zeros:
    max_j |X_j^T y| / n, as a float."""
    X = np.asarray(X, dtype=np.float64)
    return float(np.max(np.abs(X.T @ np.asarray(y, dtype=np.float64))) / len(X))


@app.function
def lasso_path(X, y, alphas, *, max_iters=1000, tol=1e-06):
    """stage 5: lasso coefficients for every alpha (no intercept), warm-started.

    solve for the alphas from LARGEST to smallest, starting each coordinate_descent from the
    previous alpha's solution (the first from zeros). return coefs (len(alphas), d) with row
    i belonging to alphas[i], in the order the alphas were GIVEN.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    coefs = np.zeros((len(alphas), X.shape[1]))
    w = np.zeros(X.shape[1])
    for i in np.argsort(-alphas, kind="stable"):
        w, _ = coordinate_descent(X, y, float(alphas[i]), 0.0, w0=w, max_iters=max_iters, tol=tol)
        coefs[i] = w
    return coefs


@app.cell(hide_code=True)
def _():
    checks.run(5, lasso_alpha_max=lasso_alpha_max, lasso_path=lasso_path, coordinate_descent=coordinate_descent)
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
