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
    # bayesian linear regression · build

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
    raise NotImplementedError("stage1: implement polynomial_basis")


@app.function
def gaussian_basis(x, centers, width):
    """stage 1: x (n,) -> (n, 1 + k): a constant column, then exp(-(x - c_j)^2 / (2 width^2)) per center."""
    raise NotImplementedError("stage1: implement gaussian_basis")


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
    raise NotImplementedError("stage2: implement posterior")


@app.function
def sequential_posterior(prior_mean, prior_cov, Phi, y, beta):
    """stage 2: update a general Gaussian prior N(prior_mean, prior_cov) with a batch (Phi, y); returns (mean, cov)."""
    raise NotImplementedError("stage2: implement sequential_posterior")


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
    raise NotImplementedError("stage3: implement predictive")


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
    raise NotImplementedError("stage4: implement log_evidence")


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
        raise NotImplementedError("stage5: implement BayesianLinearRegression.__init__")

    def fit(self, X, y):
        """stage 5: posterior under (alpha, beta); with optimize=True first run the evidence fixed-point updates."""
        raise NotImplementedError("stage5: implement BayesianLinearRegression.fit")

    def predict(self, X, *, return_std=False):
        """stage 5: the predictive mean (n,), or (mean, std) where std includes the noise."""
        raise NotImplementedError("stage5: implement BayesianLinearRegression.predict")

    def log_evidence(self, X, y):
        """stage 5: log p(y | X) at the fitted alpha_, beta_ (the constructor's alpha, beta before fit)."""
        raise NotImplementedError("stage5: implement BayesianLinearRegression.log_evidence")

    def sample_weights(self, n=1, *, rng=None):
        """stage 5: n draws from the weight posterior, shape (n, d)."""
        raise NotImplementedError("stage5: implement BayesianLinearRegression.sample_weights")


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
