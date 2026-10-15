import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.linalg import cho_solve, cholesky, qr, solve_triangular  # yours, from ch 02
    from mlfs.optim import gradient_descent  # yours, from ch 06 · topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # linear regression · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. stage 3 starts `mlfs.linear`, the
    module every model in this chapter joins. it uses your `Regressor` and `check_X_y`
    (ch 07), your Cholesky and QR (ch 02) and your `gradient_descent` (ch 06); if any of
    those aren't finished, launch with `MLFS_REFERENCE=base,linalg,optim marimo edit exercise.py`.

    | stage | you build |
    |---|---|
    | 1 | `mse_loss`, `mse_gradient`: the least-squares objective and its gradient |
    | 2 | `solve_normal_equations`, `solve_qr`: two exact solvers, one stable and one not |
    | 3 | `LinearRegression` (solvers `"normal"` and `"qr"`), with the intercept done by centering → `mlfs.linear` |
    | 4 | `solve_gd`: least squares by gradient descent (`solver="gd"`) |
    | 5 | `gaussian_nll`, `noise_variance`: least squares as maximum likelihood |
    | 6 | `polynomial_basis`, `rbf_basis`: linear models of nonlinear functions |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def mse_loss(w, X, y):
    """stage 1: the least-squares loss L(w) = ||X w - y||^2 / (2n), as a Python float.

    w: float (d,); X: float (n, d); y: float (n,). no intercept here: if you want one, X
    carries a column of ones.
    """
    r = X @ w - y
    return float(r @ r) / (2 * len(y))


@app.function
def mse_gradient(w, X, y):
    """stage 1: the gradient of mse_loss with respect to w: X^T (X w - y) / n, shape (d,).

    one matrix-vector product each way: never build an (n, d, d) array or loop over rows.
    """
    return X.T @ (X @ w - y) / len(y)


@app.cell(hide_code=True)
def _():
    checks.run(1, mse_loss=mse_loss, mse_gradient=mse_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def solve_normal_equations(X, y):
    """stage 2: the least-squares w from the normal equations X^T X w = X^T y.

    X: float (n, d) with full column rank, y: float (n,). factor X^T X = L L^T with your
    cholesky and solve with cho_solve (both from mlfs.linalg). if X^T X isn't (numerically)
    positive definite, cholesky raises np.linalg.LinAlgError: let it propagate. returns w (d,).
    """
    return cho_solve(cholesky(X.T @ X), X.T @ y)


@app.function
def solve_qr(X, y):
    """stage 2: the same w from a QR factorization of X, never forming X^T X.

    X = Q R with your qr (mlfs.linalg, reduced: Q (n, d), R (d, d) upper triangular); then
    solve R w = Q^T y with solve_triangular(R, Q.T @ y, lower=False). this is exactly what
    your ch 02 lstsq does. returns w (d,).
    """
    Q, R = qr(X)
    return solve_triangular(R, Q.T @ y, lower=False)


@app.cell(hide_code=True)
def _():
    checks.run(2, solve_normal_equations=solve_normal_equations, solve_qr=solve_qr)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class LinearRegression(Regressor):
    """ordinary least squares: y ~ X @ coef_ + intercept_, with three solvers.

    hyperparameters: fit_intercept, solver ("normal", "qr" or "gd") and, for "gd" only, lr,
    max_iters and tol. fitted attributes: coef_ (d,), intercept_ (a float, 0.0 without an
    intercept) and n_iter_ (gradient steps taken; 0 for the exact solvers).
    """

    def __init__(self, *, fit_intercept=True, solver="normal", lr=0.01, max_iters=1000, tol=1e-06):
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X, y):
        """stage 3: fit by least squares and return self.

        X, y = check_X_y(X, y), y as float. an unknown solver raises ValueError.
        with fit_intercept, CENTER first: subtract the column means x_mean from X and the mean
        y_mean from y, solve for coef_ on the centered data (no ones column), then
        intercept_ = y_mean - x_mean @ coef_. without it, solve on X, y directly and
        intercept_ = 0.0. the solve: solver "normal" -> solve_normal_equations, "qr" ->
        solve_qr, "gd" -> solve_gd(X, y, lr=..., max_iters=..., tol=...) (stage 4), whose
        .x is coef_ and .n_iters is n_iter_.
        """
        if self.solver not in ("normal", "qr", "gd"):
            raise ValueError(f"solver must be 'normal', 'qr' or 'gd', got {self.solver!r}")
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), float(y.mean())
            Xc, yc = X - x_mean, y - y_mean
        else:
            x_mean, y_mean = np.zeros(X.shape[1]), 0.0
            Xc, yc = X, y
        self.n_iter_ = 0
        if self.solver == "normal":
            coef = solve_normal_equations(Xc, yc)
        elif self.solver == "qr":
            coef = solve_qr(Xc, yc)
        else:
            res = solve_gd(Xc, yc, lr=self.lr, max_iters=self.max_iters, tol=self.tol)
            coef, self.n_iter_ = res.x, res.n_iters
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X):
        """stage 3: X @ coef_ + intercept_, shape (n,). X = check_X_y(X)[0]; a different
        number of columns than in fit raises ValueError."""
        X, _ = check_X_y(X)
        if X.shape[1] != len(self.coef_):
            raise ValueError(f"fitted with {len(self.coef_)} features, got {X.shape[1]}")
        return X @ self.coef_ + self.intercept_


@app.cell(hide_code=True)
def _():
    checks.run(3, LinearRegression=LinearRegression)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def solve_gd(X, y, *, lr=0.01, max_iters=1000, tol=1e-06, w0=None):
    """stage 4: minimize mse_loss by gradient descent from w0 (default: zeros (d,)).

    call your gradient_descent (mlfs.optim) with f = mse_loss and grad = mse_gradient (as
    functions of w alone) and pass lr, max_iters and tol through: it stops when
    ||gradient|| <= tol or after max_iters steps, and it stops early if the loss blows up.
    returns its OptimResult (x = w (d,), fun, n_iters, converged, history).
    """
    w0 = np.zeros(X.shape[1]) if w0 is None else np.asarray(w0, dtype=np.float64)
    return gradient_descent(lambda w: mse_loss(w, X, y), lambda w: mse_gradient(w, X, y), w0,
                            lr=lr, max_iters=max_iters, tol=tol)


@app.cell(hide_code=True)
def _():
    checks.run(4, solve_gd=solve_gd, LinearRegression=LinearRegression, mse_loss=mse_loss, mse_gradient=mse_gradient)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def gaussian_nll(y, y_pred, sigma2):
    """stage 5: the mean negative log-likelihood of y under y_i ~ N(y_pred_i, sigma2).

    mean over i of 0.5 log(2 pi sigma2) + (y_i - y_pred_i)^2 / (2 sigma2), as a float.
    sigma2 > 0 (a float); y and y_pred (n,).
    """
    r = np.asarray(y, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(0.5 * np.log(2 * np.pi * sigma2) + np.mean(r**2) / (2 * sigma2))


@app.function
def noise_variance(y, y_pred, n_params=0):
    """stage 5: the residual sum of squares divided by (n - n_params), as a float.

    n_params=0 gives the maximum-likelihood estimate RSS / n (biased low); n_params = the
    number of fitted coefficients (including the intercept) gives the unbiased RSS / (n - p).
    n_params >= n raises ValueError.
    """
    r = np.asarray(y, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    n = len(r)
    if n_params >= n:
        raise ValueError(f"need more rows ({n}) than fitted parameters ({n_params})")
    return float(r @ r) / (n - n_params)


@app.cell(hide_code=True)
def _():
    checks.run(5, gaussian_nll=gaussian_nll, noise_variance=noise_variance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def polynomial_basis(x, degree):
    """stage 6: the features x, x^2, ..., x^degree of a 1-D input.

    x: float (n,) -> float (n, degree), column j is x ** (j + 1). no constant column: the
    model's intercept plays that role. degree 0 gives an (n, 0) array.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[:, None] ** np.arange(1, degree + 1)


@app.function
def rbf_basis(x, centers, width):
    """stage 6: Gaussian bumps: entry (i, j) is exp(-(x_i - centers_j)^2 / (2 width^2)).

    x: float (n,), centers: float (m,), width > 0 -> float (n, m). no loops: broadcast.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    c = np.asarray(centers, dtype=np.float64).ravel()
    return np.exp(-((x[:, None] - c[None, :]) ** 2) / (2.0 * width**2))


@app.cell(hide_code=True)
def _():
    checks.run(6, polynomial_basis=polynomial_basis, rbf_basis=rbf_basis, LinearRegression=LinearRegression)
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
