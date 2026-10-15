import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.unsupervised import KMeans  # yours, from topic 01
    from mlfs.utils import as_rng, logsumexp  # yours, from ch 00

    COVARIANCE_TYPES = ("full", "diag", "spherical")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # gaussian mixtures & EM · build

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. the covariance of component $j$ is
    stored as a $(d, d)$ matrix (`"full"`), a vector of $d$ variances (`"diag"`), or a single
    variance (`"spherical"`); every function takes `covariance_type` to know which.

    | stage | you build |
    |---|---|
    | 1 | `gaussian_log_density`: $\log\mathcal N(x_i \mid \mu_j, \Sigma_j)$ for all points and components |
    | 2 | `e_step`: responsibilities and the mean log-likelihood, computed stably |
    | 3 | `m_step`: weights, means and covariances from responsibilities |
    | 4 | `GaussianMixture.fit` and friends: EM as an `Estimator` → `mlfs.unsupervised` |
    | 5 | `GaussianMixture.sample`, `GaussianMixture.bic`: generate data, choose $k$ |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def gaussian_log_density(X, means, covariances, covariance_type="full"):
    """stage 1: log N(x_i | means[j], covariances[j]) for every point i and component j, (n, k)."""
    raise NotImplementedError("stage1: implement gaussian_log_density")


@app.cell(hide_code=True)
def _():
    checks.run(1, gaussian_log_density=gaussian_log_density)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def e_step(X, weights, means, covariances, covariance_type="full"):
    """stage 2: (responsibilities (n, k) with rows summing to 1, mean log-likelihood of X as a float)."""
    raise NotImplementedError("stage2: implement e_step")


@app.cell(hide_code=True)
def _():
    checks.run(2, e_step=e_step, gaussian_log_density=gaussian_log_density)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def m_step(X, resp, covariance_type="full", reg_covar=1e-6):
    """stage 3: (weights (k,), means (k, d), covariances) maximizing the expected log-likelihood, + reg_covar on variances."""
    raise NotImplementedError("stage3: implement m_step")


@app.cell(hide_code=True)
def _():
    checks.run(3, m_step=m_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class GaussianMixture(Estimator):
    """a mixture of k Gaussians fitted by expectation-maximization."""

    def __init__(self, n_components=1, *, covariance_type="full", max_iters=100, tol=1e-06, reg_covar=1e-06,
                 init="kmeans", rng=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.init = init
        self.rng = rng

    def _initial_resp(self, X, rng):
        """starting responsibilities (n, k): one-hot k-means labels, or random rows normalized to 1."""
        n, k = len(X), self.n_components
        if self.init == "kmeans":
            labels = KMeans(k, n_init=1, rng=rng).fit(X).labels_
            resp = np.zeros((n, k))
            resp[np.arange(n), labels] = 1.0
            return resp
        if self.init == "random":
            resp = rng.random((n, k))
            return resp / resp.sum(axis=1, keepdims=True)
        raise ValueError(f"init must be 'kmeans' or 'random', got {self.init!r}")

    def fit(self, X, y=None):
        """stage 4: initial M-step from _initial_resp, then E/M steps until the mean log-likelihood gains < tol."""
        raise NotImplementedError("stage4: implement GaussianMixture.fit")

    def predict_proba(self, X):
        """stage 4: responsibilities of the fitted mixture for X, (n, k)."""
        raise NotImplementedError("stage4: implement GaussianMixture.predict_proba")

    def predict(self, X):
        """stage 4: the most responsible component for each point, int (n,)."""
        raise NotImplementedError("stage4: implement GaussianMixture.predict")

    def score_samples(self, X):
        """stage 4: log p(x_i) under the fitted mixture, (n,)."""
        raise NotImplementedError("stage4: implement GaussianMixture.score_samples")

    def score(self, X, y=None):
        """stage 4: the mean log-likelihood of X."""
        raise NotImplementedError("stage4: implement GaussianMixture.score")

    def sample(self, n=1, *, rng=None):
        """stage 5: (X (n, d), labels (n,)): a component from weights_, then a draw from its Gaussian."""
        raise NotImplementedError("stage5: implement GaussianMixture.sample")

    def n_parameters(self):
        """the number of free parameters: k - 1 weights, k*d means, and the covariances."""
        k, d = self.means_.shape
        cov = {"full": k * d * (d + 1) // 2, "diag": k * d, "spherical": k}[self.covariance_type]
        return (k - 1) + k * d + cov

    def bic(self, X):
        """stage 5: Bayesian information criterion -2 log L + p log n on X (lower is better)."""
        raise NotImplementedError("stage5: implement GaussianMixture.bic")


@app.cell(hide_code=True)
def _():
    checks.run(4, GaussianMixture=GaussianMixture, e_step=e_step, m_step=m_step)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, GaussianMixture=GaussianMixture)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `GaussianMixture` is part of `mlfs.unsupervised`. open the
    lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
