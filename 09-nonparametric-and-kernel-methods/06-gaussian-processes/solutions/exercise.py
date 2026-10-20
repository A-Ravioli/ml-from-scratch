import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.optim import lbfgs  # yours, from ch 06 · topic 05
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00

    JITTER = 1e-8  # added to the noise variance on the diagonal, so Cholesky never sees a singular matrix


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gaussian processes · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions and methods to write, and checks them live as you edit.
    `GaussianProcessRegressor` and `expected_improvement` complete `mlfs.kernels`; the
    hyperparameters are learned with your L-BFGS from chapter 06.

    | stage | you build |
    |---|---|
    | 1 | `stationary_kernel`: RBF and Matérn covariance functions with a length scale and a variance |
    | 2 | `GaussianProcessRegressor.fit`, `predict`: the posterior mean, standard deviation and covariance → `mlfs.kernels` |
    | 3 | `GaussianProcessRegressor.sample_y`: functions drawn from the prior and the posterior |
    | 4 | `log_marginal_likelihood`, `lml_gradient`: the evidence and its analytic gradient |
    | 5 | `optimize_theta`: hyperparameters by maximizing the evidence with L-BFGS, with restarts |
    | 6 | `expected_improvement`: the acquisition function of Bayesian optimization → `mlfs.kernels` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def stationary_kernel(X, Y=None, *, kind="rbf", lengthscale=1.0, variance=1.0):
    """stage 1: the covariance matrix (n, m) of a 'rbf', 'matern32' or 'matern52' kernel with the given length scale and variance; Y defaults to X."""
    X = np.asarray(X, dtype=float)
    r = pairwise_distances(X, None if Y is None else np.asarray(Y, dtype=float)) / lengthscale
    if kind == "rbf":
        return variance * np.exp(-0.5 * r**2)
    if kind == "matern32":
        s = math.sqrt(3) * r
        return variance * (1 + s) * np.exp(-s)
    if kind == "matern52":
        s = math.sqrt(5) * r
        return variance * (1 + s + s**2 / 3) * np.exp(-s)
    raise ValueError(f"kind must be 'rbf', 'matern32' or 'matern52', got {kind!r}")


@app.function
def kernel_lengthscale_grad(X, *, kind="rbf", lengthscale=1.0, variance=1.0):
    """(given) dK/d(log lengthscale) for stationary_kernel(X): the matrix you need in stage 4."""
    r = pairwise_distances(np.asarray(X, dtype=float)) / lengthscale
    if kind == "rbf":
        return variance * np.exp(-0.5 * r**2) * r**2
    if kind == "matern32":
        s = math.sqrt(3) * r
        return variance * s**2 * np.exp(-s)
    if kind == "matern52":
        s = math.sqrt(5) * r
        return variance * s**2 * (1 + s) / 3 * np.exp(-s)
    raise ValueError(f"unknown kind {kind!r}")


@app.cell(hide_code=True)
def _():
    checks.run(1, stationary_kernel=stationary_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.class_definition
class GaussianProcessRegressor(Regressor):
    """GP regression with a zero-mean prior on the centred targets, a stationary kernel and Gaussian noise.

    hyperparameters theta = log([lengthscale, variance, noise]) (noise is a VARIANCE). fitted
    attributes: X_train_, y_mean_, y_train_ (centred), theta_, lengthscale_, variance_, noise_,
    L_ (lower Cholesky factor of K + (noise + JITTER) I) and alpha_ = (K + (noise + JITTER) I)^-1 y_train_.
    """

    def __init__(self, *, kernel="rbf", lengthscale=1.0, variance=1.0, noise=0.01, optimize=True, n_restarts=0,
                 rng=None):
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.optimize = optimize
        self.n_restarts = n_restarts
        self.rng = rng

    def set_theta(self, theta):
        """(given) store theta_ and its three hyperparameters, and factorize K + (noise + JITTER) I into L_ and alpha_."""
        self.theta_ = np.asarray(theta, dtype=float).copy()
        self.lengthscale_, self.variance_, self.noise_ = np.exp(self.theta_)
        K = stationary_kernel(self.X_train_, kind=self.kernel, lengthscale=self.lengthscale_, variance=self.variance_)
        self.L_ = np.linalg.cholesky(K + (self.noise_ + JITTER) * np.eye(len(K)))
        self.alpha_ = np.linalg.solve(self.L_.T, np.linalg.solve(self.L_, self.y_train_))
        return self

    def fit(self, X, y):
        """stage 2: store X_train_, y_mean_ and the centred y_train_; start from theta = log([lengthscale, variance, noise]), improve it with optimize_theta if self.optimize, then set_theta; return self."""
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_mean_ = float(np.mean(y))
        self.y_train_ = y.astype(float) - self.y_mean_
        theta = np.log([self.lengthscale, self.variance, self.noise])
        if self.optimize:
            theta = self.optimize_theta(theta)
        return self.set_theta(theta)

    def predict(self, X, *, return_std=False, return_cov=False):
        """stage 2: the posterior mean of f at X (m,), plus its std (m,) or covariance (m, m) if asked; the prior if unfitted."""
        X, _ = check_X_y(X)
        if not hasattr(self, "L_"):  # unfitted: the prior
            ls, var = self.lengthscale, self.variance
            mean = np.zeros(len(X))
            if return_cov:
                return mean, stationary_kernel(X, kind=self.kernel, lengthscale=ls, variance=var)
            if return_std:
                return mean, np.full(len(X), math.sqrt(var))
            return mean
        Ks = stationary_kernel(self.X_train_, X, kind=self.kernel, lengthscale=self.lengthscale_, variance=self.variance_)
        mean = Ks.T @ self.alpha_ + self.y_mean_
        if not (return_std or return_cov):
            return mean
        V = np.linalg.solve(self.L_, Ks)
        if return_cov:
            Kss = stationary_kernel(X, kind=self.kernel, lengthscale=self.lengthscale_, variance=self.variance_)
            return mean, Kss - V.T @ V
        return mean, np.sqrt(np.clip(self.variance_ - np.sum(V**2, axis=0), 0, None))

    def sample_y(self, X, n_samples=1, *, rng=None):
        """stage 3: (n_samples, m) draws of f at X from the posterior (or the prior if unfitted), via an eigendecomposition of the covariance."""
        mean, cov = self.predict(X, return_cov=True)
        w, U = np.linalg.eigh((cov + cov.T) / 2)
        root = U * np.sqrt(np.clip(w, 0, None))
        z = as_rng(rng).standard_normal((n_samples, len(mean)))
        return mean + z @ root.T

    def log_marginal_likelihood(self, theta=None):
        """stage 4: log p(y_train_ | X, theta) for theta = log([lengthscale, variance, noise]) (default: the fitted theta_); -inf if the Cholesky fails."""
        theta = self.theta_ if theta is None else np.asarray(theta, dtype=float)
        ls, var, noise = np.exp(theta)
        K = stationary_kernel(self.X_train_, kind=self.kernel, lengthscale=ls, variance=var)
        n = len(K)
        try:
            L = np.linalg.cholesky(K + (noise + JITTER) * np.eye(n))
        except np.linalg.LinAlgError:
            return -np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train_))
        return float(-0.5 * self.y_train_ @ a - np.sum(np.log(np.diag(L))) - 0.5 * n * math.log(2 * math.pi))

    def lml_gradient(self, theta):
        """stage 4: the gradient (3,) of log_marginal_likelihood with respect to theta = log([lengthscale, variance, noise])."""
        theta = np.asarray(theta, dtype=float)
        ls, var, noise = np.exp(theta)
        K = stationary_kernel(self.X_train_, kind=self.kernel, lengthscale=ls, variance=var)
        n = len(K)
        L = np.linalg.cholesky(K + (noise + JITTER) * np.eye(n))
        Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
        a = Kinv @ self.y_train_
        inner = np.outer(a, a) - Kinv
        dK = [kernel_lengthscale_grad(self.X_train_, kind=self.kernel, lengthscale=ls, variance=var), K,
              noise * np.eye(n)]
        return np.array([0.5 * np.sum(inner * d) for d in dK])

    def optimize_theta(self, theta0):
        """stage 5: maximize the log marginal likelihood with lbfgs (max_iters=200) from theta0 and from n_restarts starts drawn uniformly within ±3 of theta0 (as_rng(rng)); return the best theta."""

        def f(t):
            with np.errstate(all="ignore"):  # trial steps can be extreme: let them fail as inf, quietly
                return -self.log_marginal_likelihood(t)

        def g(t):
            try:
                with np.errstate(all="ignore"):
                    return -self.lml_gradient(t)
            except np.linalg.LinAlgError:
                return np.zeros(3)

        gen = as_rng(self.rng)
        theta0 = np.asarray(theta0, dtype=float)
        starts = [theta0] + [theta0 + gen.uniform(-3, 3, size=3) for _ in range(self.n_restarts)]
        best, best_val = theta0, f(theta0)
        for start in starts:
            res = lbfgs(f, g, start, max_iters=200)
            if np.isfinite(res.fun) and res.fun < best_val:
                best, best_val = np.asarray(res.x, dtype=float), res.fun
        return best


@app.cell(hide_code=True)
def _():
    checks.run(2, GaussianProcessRegressor=GaussianProcessRegressor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, GaussianProcessRegressor=GaussianProcessRegressor, stationary_kernel=stationary_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, GaussianProcessRegressor=GaussianProcessRegressor, stationary_kernel=stationary_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, GaussianProcessRegressor=GaussianProcessRegressor)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def expected_improvement(mu, sigma, best, *, xi=0.01):
    """stage 6: E[max(f - best - xi, 0)] for f ~ N(mu, sigma^2) (maximization): (mu - best - xi) Phi(z) + sigma phi(z), 0 where sigma = 0 (by convention), never negative."""
    mu, sigma = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float))
    imp = mu - best - xi
    pos = sigma > 0
    z = np.where(pos, imp / np.where(pos, sigma, 1.0), 0.0)
    cdf = 0.5 * np.vectorize(math.erfc, otypes=[float])(-z / math.sqrt(2))  # 1 + erf(x) cancels for x << 0
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    ei = np.where(pos, np.maximum(imp * cdf + sigma * pdf, 0.0), 0.0)
    return float(ei) if ei.ndim == 0 else ei


@app.cell(hide_code=True)
def _():
    checks.run(6, expected_improvement=expected_improvement)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import GaussianProcessRegressor,
    expected_improvement` gives you your own GP. open the lab (`marimo edit lab.py`), then the
    chapter's capstone, which uses both to tune a model by Bayesian optimization.
    """)
    return


if __name__ == "__main__":
    app.run()
