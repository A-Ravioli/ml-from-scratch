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
    # gaussian mixtures & EM · build (reference solution)

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
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    n, d = X.shape
    out = np.empty((n, len(means)))
    for j, mu in enumerate(means):
        diff = X - mu
        if covariance_type == "full":
            L = np.linalg.cholesky(covariances[j])
            z = np.linalg.solve(L, diff.T)
            maha = np.sum(z**2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
        elif covariance_type == "diag":
            var = covariances[j]
            maha = np.sum(diff**2 / var, axis=1)
            logdet = np.sum(np.log(var))
        elif covariance_type == "spherical":
            var = covariances[j]
            maha = np.sum(diff**2, axis=1) / var
            logdet = d * np.log(var)
        else:
            raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, got {covariance_type!r}")
        out[:, j] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return out


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
    log_joint = gaussian_log_density(X, means, covariances, covariance_type) + np.log(weights)
    log_px = logsumexp(log_joint, axis=1)
    return np.exp(log_joint - log_px[:, None]), float(np.mean(log_px))


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
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    nk = resp.sum(axis=0) + 10 * np.finfo(float).eps
    weights = nk / n
    means = resp.T @ X / nk[:, None]
    if covariance_type == "full":
        covs = np.empty((len(nk), d, d))
        for j in range(len(nk)):
            diff = X - means[j]
            covs[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + reg_covar * np.eye(d)
    elif covariance_type in ("diag", "spherical"):
        sq = np.stack([resp[:, j] @ (X - means[j]) ** 2 for j in range(len(nk))])
        covs = sq / nk[:, None] + reg_covar
        if covariance_type == "spherical":
            covs = covs.mean(axis=1)
    else:
        raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, got {covariance_type!r}")
    return weights, means, covs


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
        X, _ = check_X_y(X)
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError(f"covariance_type must be one of {COVARIANCE_TYPES}, got {self.covariance_type!r}")
        if not 1 <= self.n_components <= len(X):
            raise ValueError(f"n_components must be in [1, {len(X)}], got {self.n_components}")
        rng = as_rng(self.rng)
        resp = self._initial_resp(X, rng)
        self.weights_, self.means_, self.covariances_ = m_step(X, resp, self.covariance_type, self.reg_covar)
        self.lower_bound_history_ = []
        self.converged_ = False
        for self.n_iter_ in range(1, self.max_iters + 1):
            resp, ll = e_step(X, self.weights_, self.means_, self.covariances_, self.covariance_type)
            self.lower_bound_history_.append(ll)
            self.weights_, self.means_, self.covariances_ = m_step(X, resp, self.covariance_type, self.reg_covar)
            if len(self.lower_bound_history_) > 1 and abs(ll - self.lower_bound_history_[-2]) < self.tol:
                self.converged_ = True
                break
        return self

    def predict_proba(self, X):
        """stage 4: responsibilities of the fitted mixture for X, (n, k)."""
        X, _ = check_X_y(X)
        return e_step(X, self.weights_, self.means_, self.covariances_, self.covariance_type)[0]

    def predict(self, X):
        """stage 4: the most responsible component for each point, int (n,)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        """stage 4: log p(x_i) under the fitted mixture, (n,)."""
        X, _ = check_X_y(X)
        log_joint = gaussian_log_density(X, self.means_, self.covariances_, self.covariance_type) + np.log(self.weights_)
        return logsumexp(log_joint, axis=1)

    def score(self, X, y=None):
        """stage 4: the mean log-likelihood of X."""
        return float(np.mean(self.score_samples(X)))

    def sample(self, n=1, *, rng=None):
        """stage 5: (X (n, d), labels (n,)): a component from weights_, then a draw from its Gaussian."""
        rng = as_rng(rng)
        k, d = self.means_.shape
        labels = rng.choice(k, size=n, p=self.weights_)
        z = rng.standard_normal((n, d))
        X = np.empty((n, d))
        for j in range(k):
            m = labels == j
            if self.covariance_type == "full":
                X[m] = self.means_[j] + z[m] @ np.linalg.cholesky(self.covariances_[j]).T
            else:
                X[m] = self.means_[j] + z[m] * np.sqrt(self.covariances_[j])
        return X, labels

    def n_parameters(self):
        """the number of free parameters: k - 1 weights, k*d means, and the covariances."""
        k, d = self.means_.shape
        cov = {"full": k * d * (d + 1) // 2, "diag": k * d, "spherical": k}[self.covariance_type]
        return (k - 1) + k * d + cov

    def bic(self, X):
        """stage 5: Bayesian information criterion -2 log L + p log n on X (lower is better)."""
        X, _ = check_X_y(X)
        return float(-2.0 * np.sum(self.score_samples(X)) + self.n_parameters() * np.log(len(X)))


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
