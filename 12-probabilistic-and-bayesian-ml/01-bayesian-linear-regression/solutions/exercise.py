import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Regressor, check_X_y  # yours, from chapter 07
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # bayesian linear regression · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions to write, and checks them live as you edit. the model is
    $y = \Phi w + \varepsilon$ with prior $w \sim \mathcal N(0, \alpha^{-1} I)$ and noise
    $\varepsilon \sim \mathcal N(0, \beta^{-1} I)$: $\alpha$ is the prior precision, $\beta$ the
    noise precision. `BayesianLinearRegression` becomes part of `mlfs.bayes`.

    | stage | you build |
    |---|---|
    | 1 | `polynomial_basis`, `gaussian_basis`: the feature maps $\Phi$ |
    | 2 | `posterior`, `sequential_posterior`: the Gaussian posterior over weights, in one go and one batch at a time |
    | 3 | `predictive`: the posterior predictive mean and variance at new inputs |
    | 4 | `log_evidence`: the marginal likelihood $\log p(y \mid \alpha, \beta)$ |
    | 5 | `BayesianLinearRegression` → `mlfs.bayes`: an estimator, with evidence maximization |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def polynomial_basis(x, degree):
    """stage 1: x (n,) -> (n, degree + 1) with columns x**0, x**1, ..., x**degree."""
    x = np.asarray(x, dtype=float).ravel()
    return x[:, None] ** np.arange(degree + 1)


@app.function
def gaussian_basis(x, centers, width):
    """stage 1: x (n,) -> (n, 1 + k): a constant column, then exp(-(x - c_j)^2 / (2 width^2)) per center."""
    x = np.asarray(x, dtype=float).ravel()
    c = np.asarray(centers, dtype=float).ravel()
    bumps = np.exp(-((x[:, None] - c[None, :]) ** 2) / (2.0 * width**2))
    return np.hstack([np.ones((x.size, 1)), bumps])


@app.cell(hide_code=True)
def _():
    checks.run(1, polynomial_basis=polynomial_basis, gaussian_basis=gaussian_basis)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def posterior(Phi, y, alpha, beta):
    """stage 2: posterior (mean (d,), cov (d, d)) of w under the prior N(0, alpha^-1 I) and noise precision beta."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
    cov = np.linalg.inv(A)
    cov = (cov + cov.T) / 2
    mean = beta * cov @ (Phi.T @ y)
    return mean, cov


@app.function
def sequential_posterior(prior_mean, prior_cov, Phi, y, beta):
    """stage 2: update a general Gaussian prior N(prior_mean, prior_cov) with a batch (Phi, y); returns (mean, cov)."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    P0 = np.linalg.inv(np.asarray(prior_cov, dtype=float))
    A = P0 + beta * Phi.T @ Phi
    cov = np.linalg.inv(A)
    cov = (cov + cov.T) / 2
    mean = cov @ (P0 @ np.asarray(prior_mean, dtype=float) + beta * Phi.T @ y)
    return mean, cov


@app.cell(hide_code=True)
def _():
    checks.run(2, posterior=posterior, sequential_posterior=sequential_posterior)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def predictive(Phi, mean, cov, beta):
    """stage 3: predictive (mean (n,), var (n,)) at the rows of Phi; var includes the noise 1 / beta."""
    Phi = np.asarray(Phi, dtype=float)
    mu = Phi @ mean
    var = 1.0 / beta + np.einsum("ij,jk,ik->i", Phi, cov, Phi)
    return mu, var


@app.cell(hide_code=True)
def _():
    checks.run(3, predictive=predictive)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def log_evidence(Phi, y, alpha, beta):
    """stage 4: log p(y | alpha, beta) with w integrated out, as a float (use the posterior, not an n x n matrix)."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, d = Phi.shape
    mean, _ = posterior(Phi, y, alpha, beta)
    A = alpha * np.eye(d) + beta * Phi.T @ Phi
    _, logdet_A = np.linalg.slogdet(A)
    E = 0.5 * beta * np.sum((y - Phi @ mean) ** 2) + 0.5 * alpha * mean @ mean
    return float(0.5 * d * math.log(alpha) + 0.5 * n * math.log(beta) - E
                 - 0.5 * logdet_A - 0.5 * n * math.log(2 * math.pi))


@app.cell(hide_code=True)
def _():
    checks.run(4, log_evidence=log_evidence, posterior=posterior)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.class_definition
class BayesianLinearRegression(Regressor):
    """Bayesian linear regression with prior w ~ N(0, alpha^-1 I) and noise precision beta.

    no intercept: put a constant column in X. fitted attributes: mean_ (d,), cov_ (d, d),
    alpha_, beta_ (the hyperparameters used, re-estimated if optimize=True) and n_iter_.
    """

    def __init__(self, *, alpha=1.0, beta=1.0, optimize=False, max_iters=100):
        """stage 5: store the hyperparameters unchanged."""
        self.alpha = alpha
        self.beta = beta
        self.optimize = optimize
        self.max_iters = max_iters

    def fit(self, X, y):
        """stage 5: posterior under (alpha, beta); with optimize=True first run the evidence fixed-point updates."""
        X, y = check_X_y(X, y)
        y = y.astype(float)
        n = X.shape[0]
        alpha, beta = float(self.alpha), float(self.beta)
        self.n_iter_ = 0
        if self.optimize:
            eig0 = np.linalg.eigvalsh(X.T @ X)
            for it in range(self.max_iters):
                mean, _ = posterior(X, y, alpha, beta)
                lam = beta * eig0
                gamma = float(np.sum(lam / (alpha + lam)))
                tiny = 1e-300  # guards for degenerate fits (all-zero weights, exact interpolation)
                alpha_new = float(np.clip(gamma / max(float(mean @ mean), tiny), 1e-12, 1e12))
                rss = float(np.sum((y - X @ mean) ** 2))
                beta_new = float(np.clip(max(n - gamma, tiny) / max(rss, tiny), 1e-12, 1e12))
                self.n_iter_ = it + 1
                done = abs(alpha_new - alpha) <= 1e-10 * alpha and abs(beta_new - beta) <= 1e-10 * beta
                alpha, beta = alpha_new, beta_new
                if done:
                    break
        self.alpha_, self.beta_ = alpha, beta
        self.mean_, self.cov_ = posterior(X, y, alpha, beta)
        return self

    def predict(self, X, *, return_std=False):
        """stage 5: the predictive mean (n,), or (mean, std) where std includes the noise."""
        X, _ = check_X_y(X)
        mu, var = predictive(X, self.mean_, self.cov_, self.beta_)
        return (mu, np.sqrt(var)) if return_std else mu

    def log_evidence(self, X, y):
        """stage 5: log p(y | X) at the fitted alpha_, beta_ (the constructor's alpha, beta before fit)."""
        X, y = check_X_y(X, y)
        alpha = getattr(self, "alpha_", self.alpha)
        beta = getattr(self, "beta_", self.beta)
        return log_evidence(X, y.astype(float), alpha, beta)

    def sample_weights(self, n=1, *, rng=None):
        """stage 5: n draws from the weight posterior, shape (n, d)."""
        rng = as_rng(rng)
        L = np.linalg.cholesky(self.cov_ + 1e-12 * np.eye(len(self.mean_)))
        return self.mean_ + rng.standard_normal((n, len(self.mean_))) @ L.T


@app.cell(hide_code=True)
def _():
    checks.run(5, BayesianLinearRegression=BayesianLinearRegression, gaussian_basis=gaussian_basis,
               polynomial_basis=polynomial_basis)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, your estimator is part of `mlfs.bayes` (try
    `from mlfs.bayes import BayesianLinearRegression` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
