import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import grad, numerical_jacobian  # yours, from ch 03 · topics 02 and 04
    from mlfs.linalg import solve  # yours, from ch 02 · topic 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # exponential families & fisher information · build

    work through the stages in order. you'll build one class, `ExponentialFamily`, that
    represents the Bernoulli, Poisson, exponential and Gaussian distributions (and many more)
    in a single form, $p(x \mid \eta) = h(x)\exp(\eta^\top T(x) - A(\eta))$. from the
    log-partition function $A$ alone, with your forward-mode autodiff from ch 03, you'll get
    means, Fisher information, a generic maximum-likelihood fitter, and the Cramér–Rao bound.

    a family is described by three functions and a starting point:

    - `sufficient_stats(x)`: data `(n,)` → $T(x)$, shape `(n, k)`;
    - `log_partition(eta)`: $A(\eta)$ for `eta` of shape `(k,)`, written with operations your
      `Dual` numbers support (`+ - * / **`, `np.exp`, `np.log`, `np.log1p`), so that
      `mlfs.autodiff.grad` can differentiate it;
    - `log_base_measure(x)`: $\log h(x)$, shape `(n,)`;
    - `eta0`: a valid natural parameter to start iterative fitting from.

    | stage | you build |
    |---|---|
    | 1 | `ExponentialFamily.__init__`, `log_prob`: the canonical form |
    | 2 | `bernoulli_family`, `poisson_family`, `exponential_family`, `gaussian_family`: four distributions in that form |
    | 3 | `mean_params`, `fisher_information`: derivatives of the log-partition function |
    | 4 | `fit`: a generic maximum-likelihood fitter (damped Newton) |
    | 5 | `score`, `cramer_rao_bound`: the score function, and the best variance any unbiased estimator can achieve |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class ExponentialFamily:
    """an exponential family p(x | eta) = h(x) exp(eta . T(x) - A(eta)) with k natural parameters."""

    def __init__(self, sufficient_stats, log_partition, log_base_measure, eta0):
        """stage 1: store the three functions as attributes T, A, log_h, eta0 as a float array (k,), and dim = k."""
        raise NotImplementedError("stage1: implement ExponentialFamily.__init__")

    def log_prob(self, x, eta) -> np.ndarray:
        """stage 1: log p(x_i | eta) for every x_i in x (n,): log_h(x) + T(x) @ eta - A(eta). returns (n,)."""
        raise NotImplementedError("stage1: implement ExponentialFamily.log_prob")

    def mean_params(self, eta) -> np.ndarray:
        """stage 3: E[T(X)] under eta, which equals grad A(eta). use mlfs.autodiff.grad. returns (k,)."""
        raise NotImplementedError("stage3: implement ExponentialFamily.mean_params")

    def fisher_information(self, eta) -> np.ndarray:
        """stage 3: the Fisher information of ONE observation, I(eta) = Hessian of A = Cov[T(X)]. returns (k, k).

        differentiate your mean_params numerically (mlfs.autodiff.numerical_jacobian) and symmetrize.
        """
        raise NotImplementedError("stage3: implement ExponentialFamily.fisher_information")

    def fit(self, x, *, eta0=None, max_iters=100, tol=1e-10) -> np.ndarray:
        """stage 4: the maximum-likelihood natural parameter for data x (n,), by damped Newton's method.

        the average log-likelihood is l(eta) = eta . tbar - A(eta) + const, with tbar = mean of T(x). its gradient
        is g = tbar - mean_params(eta) and its Hessian is -fisher_information(eta). start at eta0 (default
        self.eta0). each iteration: stop if ||g|| <= tol; else step = solve(fisher_information(eta), g); halve the
        step (at most 50 times) until l(eta + step) is finite and >= l(eta); take it. at most max_iters iterations.
        returns eta (k,).
        """
        raise NotImplementedError("stage4: implement ExponentialFamily.fit")

    def score(self, x, eta) -> np.ndarray:
        """stage 5: the score of each observation, d/d eta log p(x_i | eta) = T(x_i) - mean_params(eta). returns (n, k)."""
        raise NotImplementedError("stage5: implement ExponentialFamily.score")

    def cramer_rao_bound(self, eta, n: int) -> np.ndarray:
        """stage 5: the Cramér-Rao lower bound on the covariance of any unbiased estimator of eta from n
        i.i.d. observations: inverse(fisher_information(eta)) / n. returns (k, k)."""
        raise NotImplementedError("stage5: implement ExponentialFamily.cramer_rao_bound")


@app.cell(hide_code=True)
def _():
    checks.run(1, ExponentialFamily=ExponentialFamily)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def bernoulli_family() -> ExponentialFamily:
    """stage 2: Bernoulli(p) with natural parameter eta = log(p / (1 - p)): T(x) = x, A = log(1 + e^eta), h = 1. eta0 = [0]."""
    raise NotImplementedError("stage2: implement bernoulli_family")


@app.function
def poisson_family() -> ExponentialFamily:
    """stage 2: Poisson(lam) with eta = log(lam): T(x) = x, A = e^eta, log h(x) = -log(x!). eta0 = [0]."""
    raise NotImplementedError("stage2: implement poisson_family")


@app.function
def exponential_family() -> ExponentialFamily:
    """stage 2: Exponential(rate) with eta = -rate < 0: T(x) = x, A = -log(-eta), h = 1 (x >= 0). eta0 = [-1]."""
    raise NotImplementedError("stage2: implement exponential_family")


@app.function
def gaussian_family() -> ExponentialFamily:
    """stage 2: N(mu, sigma2) with eta = (mu / sigma2, -1 / (2 sigma2)): T(x) = (x, x^2),
    A = -eta1^2 / (4 eta2) - log(-2 eta2) / 2, log h = -log(2 pi) / 2. eta0 = [0, -0.5] (the standard normal)."""
    raise NotImplementedError("stage2: implement gaussian_family")


@app.cell(hide_code=True)
def _():
    checks.run(2, bernoulli_family=bernoulli_family, poisson_family=poisson_family,
               exponential_family=exponential_family, gaussian_family=gaussian_family)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **stage 3:** implement `ExponentialFamily.mean_params` and `ExponentialFamily.fisher_information` in the class above.
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, ExponentialFamily=ExponentialFamily, bernoulli_family=bernoulli_family,
               poisson_family=poisson_family, gaussian_family=gaussian_family)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **stage 4:** implement `ExponentialFamily.fit` in the class above.
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, ExponentialFamily=ExponentialFamily, bernoulli_family=bernoulli_family,
               poisson_family=poisson_family, exponential_family=exponential_family,
               gaussian_family=gaussian_family)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **stage 5:** implement `ExponentialFamily.score` and `ExponentialFamily.cramer_rao_bound` in the class above.
    """)
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, ExponentialFamily=ExponentialFamily, bernoulli_family=bernoulli_family,
               poisson_family=poisson_family, gaussian_family=gaussian_family)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab (`marimo edit lab.py`) and find out which
    estimators reach the Cramér–Rao bound, and which ones beat it.
    """)
    return


if __name__ == "__main__":
    app.run()
