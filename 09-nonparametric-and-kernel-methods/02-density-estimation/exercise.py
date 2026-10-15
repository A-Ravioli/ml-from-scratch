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
    # density estimation · build

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
    raise NotImplementedError("stage1: implement histogram_density")


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
    raise NotImplementedError("stage2: implement log_kernel")


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
        raise NotImplementedError("stage3: implement KernelDensity.fit")

    def score_samples(self, X):
        """stage 3: (m,) log density log p(x) at each row of X, via logsumexp over the training rows."""
        raise NotImplementedError("stage3: implement KernelDensity.score_samples")

    def score(self, X, y=None):
        """stage 3: the total log-likelihood sum_i log p(x_i) of the rows of X (a float)."""
        raise NotImplementedError("stage3: implement KernelDensity.score")

    def sample(self, n=1, *, rng=None):
        """stage 4: (n, d) draws from the estimate (gaussian kernel only): a random training row plus N(0, h^2 I) noise."""
        raise NotImplementedError("stage4: implement KernelDensity.sample")


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
    raise NotImplementedError("stage5: implement silverman_bandwidth")


@app.function
def select_bandwidth(X, bandwidths, *, cv=5, rng=None):
    """stage 5: (best bandwidth, mean_scores (len(bandwidths),)): mean held-out log-likelihood per point of a gaussian KDE, over shuffled KFold(cv) folds."""
    raise NotImplementedError("stage5: implement select_bandwidth")


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
