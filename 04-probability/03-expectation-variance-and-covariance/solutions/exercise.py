import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # expectation, variance & covariance · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    something to write, and checks your code live as you edit. `RunningStats` and `covariance`
    become part of `mlfs.sampling`: later chapters use them to track training statistics,
    normalize data on the fly, and estimate covariance matrices.

    | stage | you build |
    |---|---|
    | 1 | `expectation`, `mc_estimate`: exact expectations (LOTUS) and Monte Carlo ones with error bars |
    | 2 | `naive_variance`, `two_pass_variance`: the textbook formula, and the one that works |
    | 3 | `RunningStats`: Welford/Chan streaming mean and variance, one sample or a batch at a time |
    | 4 | `covariance`, `correlation`: covariance matrices, two-pass |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def expectation(values, probs, f=None) -> float:
    """stage 1: E[f(X)] for a discrete X with P(X = values[i]) = probs[i] (LOTUS).

    values: (k,) numbers; probs: (k,) non-negative, summing to 1. f: a vectorized function
    (array -> array), or None for f(x) = x. returns a Python float.
    """
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    fx = values if f is None else np.asarray(f(values), dtype=float)
    return float(np.sum(fx * probs))


@app.function
def mc_estimate(samples) -> tuple[float, float]:
    """stage 1: Monte Carlo estimate of E[X] from i.i.d. samples (n,), n >= 2, with its standard error.

    returns (mean, standard_error) where standard_error = s / sqrt(n) and s is the sample
    standard deviation with ddof=1. both Python floats.
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    mean = x.mean()
    s = math.sqrt(np.sum((x - mean) ** 2) / (n - 1))
    return float(mean), s / math.sqrt(n)


@app.cell(hide_code=True)
def _():
    checks.run(1, expectation=expectation, mc_estimate=mc_estimate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def naive_variance(x) -> float:
    """stage 2: the textbook one-pass formula E[X^2] - E[X]^2 for the population variance of x (n,).

    compute mean(x**2) - mean(x)**2 exactly as written. (it's here to fail: see the lesson.)
    returns a Python float.
    """
    x = np.asarray(x, dtype=float)
    return float(np.mean(x**2) - np.mean(x) ** 2)


@app.function
def two_pass_variance(x, ddof: int = 0) -> float:
    """stage 2: the variance of x (n,) in two passes: first the mean, then the mean squared deviation.

    sum((x - mean)^2) / (n - ddof). ddof=0: population variance; ddof=1: unbiased sample variance.
    returns a Python float.
    """
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    return float(np.sum(d * d) / (len(x) - ddof))


@app.cell(hide_code=True)
def _():
    checks.run(2, naive_variance=naive_variance, two_pass_variance=two_pass_variance)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.class_definition
class RunningStats:
    """Streaming mean and variance (Welford's update, generalized to batches by Chan et al.)."""

    def __init__(self, shape=()):
        """stage 3: an empty accumulator for samples of shape `shape`.

        sets count = 0 (int), mean = zeros(shape) (float array), and m2 = zeros(shape): the
        running SUM of squared deviations from the mean.
        """
        self.shape = tuple(shape)
        self.count = 0
        self.mean = np.zeros(self.shape)
        self.m2 = np.zeros(self.shape)

    def update(self, x):
        """stage 3: add one sample (shape `shape`) or a batch (k, *shape) of samples.

        merge the batch's own count, mean and m2 into the running ones (Chan's formula):
          delta = mean_b - mean;  n = count + n_b
          mean += delta * n_b / n;  m2 += m2_b + delta^2 * count * n_b / n
        a single sample is a batch with n_b = 1, mean_b = x, m2_b = 0 (Welford's update).
        raises ValueError if x has the wrong shape. returns self.
        """
        x = np.asarray(x, dtype=float)
        if x.shape == self.shape:
            x = x[None]
        elif x.shape[1:] != self.shape:
            raise ValueError(f"expected a sample of shape {self.shape} or a batch (k, *{self.shape}), got {x.shape}")
        n_b = x.shape[0]
        if n_b == 0:
            return self
        mean_b = x.mean(axis=0)
        m2_b = np.sum((x - mean_b) ** 2, axis=0)
        n = self.count + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * (n_b / n)
        self.m2 = self.m2 + m2_b + delta**2 * (self.count * n_b / n)
        self.count = n
        return self

    @property
    def var(self):
        """population variance (ddof=0) of everything seen so far: m2 / count (nan if empty)."""
        return self.m2 / self.count if self.count > 0 else np.full(self.shape, np.nan)

    @property
    def sample_var(self):
        """unbiased sample variance (ddof=1): m2 / (count - 1) (nan if count < 2)."""
        return self.m2 / (self.count - 1) if self.count > 1 else np.full(self.shape, np.nan)

    @property
    def std(self):
        """population standard deviation, sqrt(var)."""
        return np.sqrt(self.var)


@app.cell(hide_code=True)
def _():
    checks.run(3, RunningStats=RunningStats)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def covariance(X, *, ddof: int = 1) -> np.ndarray:
    """stage 4: the covariance matrix of the rows of X (n, d): returns (d, d).

    two passes: center the columns (subtract the column means), then Xc.T @ Xc / (n - ddof).
    ddof=1 (default) is the unbiased estimate, like np.cov; ddof=0 the population version.
    a 1-D X (n,) is treated as one variable (n, 1).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Xc = X - X.mean(axis=0)
    return Xc.T @ Xc / (X.shape[0] - ddof)


@app.function
def correlation(X) -> np.ndarray:
    """stage 4: the correlation matrix (d, d) of the rows of X (n, d): cov_ij / (sd_i sd_j).

    ones on the diagonal, entries in [-1, 1]. use your covariance (the ddof cancels).
    """
    C = covariance(X)
    sd = np.sqrt(np.diag(C))
    return C / np.outer(sd, sd)


@app.cell(hide_code=True)
def _():
    checks.run(4, covariance=covariance, correlation=correlation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `RunningStats` and `covariance` are part of `mlfs.sampling`
    (try `from mlfs.sampling import RunningStats` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
