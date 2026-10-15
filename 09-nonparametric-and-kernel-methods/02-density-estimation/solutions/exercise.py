import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.base import Estimator, check_X_y  # yours, from ch 07 · topic 01
    from mlfs.model_selection import KFold  # yours, from ch 07 · topic 02
    from mlfs.utils import as_rng, logsumexp, pairwise_distances  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # density estimation · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    functions and classes to write, and checks them live as you edit. `KernelDensity` and
    `select_bandwidth` become part of `mlfs.kernels`.

    | stage | you build |
    |---|---|
    | 1 | `histogram_density`: the oldest density estimator, and its two knobs |
    | 2 | `log_kernel`: normalized Gaussian, Epanechnikov and top-hat kernels in $d$ dimensions, in log space |
    | 3 | `KernelDensity.fit`, `score_samples`, `score`: the kernel density estimator → `mlfs.kernels` |
    | 4 | `KernelDensity.sample`: drawing new points from the estimate → `mlfs.kernels` |
    | 5 | `silverman_bandwidth`, `select_bandwidth`: a rule of thumb, and cross-validated likelihood → `mlfs.kernels` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def histogram_density(x_train, x_eval, bin_width, origin=0.0):
    """stage 1: the histogram density at each point of x_eval (m,), from 1-D x_train (n,), with bins [origin + j*h, origin + (j+1)*h)."""
    x_train = np.asarray(x_train, dtype=float).ravel()
    x_eval = np.asarray(x_eval, dtype=float).ravel()
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    train_bins = np.sort(np.floor((x_train - origin) / bin_width))
    eval_bins = np.floor((x_eval - origin) / bin_width)
    counts = np.searchsorted(train_bins, eval_bins, side="right") - np.searchsorted(train_bins, eval_bins, side="left")
    return counts / (len(x_train) * bin_width)


@app.cell(hide_code=True)
def _():
    checks.run(1, histogram_density=histogram_density)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def log_kernel(sq_dist, bandwidth, d, kernel="gaussian"):
    """stage 2: log K_h at squared distances sq_dist (any shape) for a d-dimensional kernel that integrates to 1; -inf outside its support."""
    sq_dist = np.asarray(sq_dist, dtype=float)
    h = float(bandwidth)
    if h <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    u2 = sq_dist / h**2
    log_unit_ball = 0.5 * d * math.log(math.pi) - math.lgamma(0.5 * d + 1)  # log volume of the unit d-ball
    if kernel == "gaussian":
        return -0.5 * u2 - 0.5 * d * math.log(2 * math.pi) - d * math.log(h)
    if kernel == "epanechnikov":
        log_norm = math.log((d + 2) / 2) - log_unit_ball - d * math.log(h)
        with np.errstate(divide="ignore"):
            return np.where(u2 < 1, np.log(np.clip(1 - u2, 0, None)) + log_norm, -np.inf)
    if kernel == "tophat":
        return np.where(u2 < 1, -log_unit_ball - d * math.log(h), -np.inf)
    raise ValueError(f"kernel must be 'gaussian', 'epanechnikov' or 'tophat', got {kernel!r}")


@app.cell(hide_code=True)
def _():
    checks.run(2, log_kernel=log_kernel)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class KernelDensity(Estimator):
    """the kernel density estimate p(x) = (1/n) sum_i K_h(x - x_i)."""

    def __init__(self, bandwidth=1.0, *, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y=None):
        """stage 3: store the training rows as X_ (n, d) after check_X_y, validating bandwidth and kernel; return self."""
        X, _ = check_X_y(X)
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")
        if self.kernel not in ("gaussian", "epanechnikov", "tophat"):
            raise ValueError(f"unknown kernel {self.kernel!r}")
        self.X_ = X
        return self

    def score_samples(self, X):
        """stage 3: (m,) log density log p(x) at each row of X, via logsumexp over the training rows."""
        X, _ = check_X_y(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(f"X has {X.shape[1]} features, the estimate has {self.X_.shape[1]}")
        D2 = pairwise_distances(X, self.X_, squared=True)
        logk = log_kernel(D2, self.bandwidth, self.X_.shape[1], self.kernel)
        return logsumexp(logk, axis=1) - math.log(len(self.X_))

    def score(self, X, y=None):
        """stage 3: the total log-likelihood sum_i log p(x_i) of the rows of X (a float)."""
        return float(np.sum(self.score_samples(X)))

    def sample(self, n=1, *, rng=None):
        """stage 4: (n, d) draws from the estimate (gaussian kernel only): a random training row plus N(0, h^2 I) noise."""
        if self.kernel != "gaussian":
            raise NotImplementedError("sampling is implemented for the gaussian kernel only")
        gen = as_rng(rng)
        idx = gen.integers(len(self.X_), size=n)
        return self.X_[idx] + self.bandwidth * gen.standard_normal((n, self.X_.shape[1]))


@app.cell(hide_code=True)
def _():
    checks.run(3, KernelDensity=KernelDensity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, KernelDensity=KernelDensity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def silverman_bandwidth(X):
    """stage 5: the normal-reference rule h = (4 / (d + 2))^(1/(d+4)) n^(-1/(d+4)) sigma, sigma the mean per-feature std (ddof=1)."""
    X, _ = check_X_y(X)
    n, d = X.shape
    sigma = float(np.mean(np.std(X, axis=0, ddof=1)))
    return (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4)) * sigma


@app.function
def select_bandwidth(X, bandwidths, *, cv=5, rng=None):
    """stage 5: (best bandwidth, mean_scores (len(bandwidths),)): mean held-out log-likelihood per point of a gaussian KDE, over shuffled KFold(cv) folds."""
    X, _ = check_X_y(X)
    splits = list(KFold(cv, shuffle=True, rng=as_rng(rng)).split(X))
    scores = np.empty(len(bandwidths))
    for j, h in enumerate(bandwidths):
        fold_scores = [np.mean(KernelDensity(h).fit(X[tr]).score_samples(X[te])) for tr, te in splits]
        scores[j] = np.mean(fold_scores)
    return float(np.asarray(bandwidths, dtype=float)[int(np.argmax(scores))]), scores


@app.cell(hide_code=True)
def _():
    checks.run(5, silverman_bandwidth=silverman_bandwidth, select_bandwidth=select_bandwidth)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `from mlfs.kernels import KernelDensity, select_bandwidth`
    gives you your own code. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
