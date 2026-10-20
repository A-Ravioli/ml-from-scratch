import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.linalg import cho_solve, cholesky, solve_triangular  # yours, from chapter 02
    from mlfs.sampling import normal  # yours, from topic 02


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # the multivariate gaussian · build (reference solution)

    work through the stages in order. you're building one class, `MultivariateNormal`, a
    method or two per stage. it stands on two things you already built: `cholesky`,
    `cho_solve` and `solve_triangular` from `mlfs.linalg` (chapter 02), and `normal` from
    `mlfs.sampling` (topic 02). it becomes part of `mlfs.sampling`, and later chapters use it
    for Gaussian classifiers, Gaussian processes, Kalman filters and variational inference.

    | stage | you build |
    |---|---|
    | 1 | `__init__`, `sample`: store and factor the covariance; sample with Cholesky |
    | 2 | `logpdf`, `pdf`, `entropy`: densities without ever forming an inverse |
    | 3 | `marginal`: forget some coordinates |
    | 4 | `condition`: observe some coordinates, update the rest |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class MultivariateNormal:
    """The Gaussian N(mean, cov) on R^d."""

    def __init__(self, mean, cov):
        """stage 1: store mean (d,) and cov (d, d) as float arrays (copies), dim = d, and the
        lower Cholesky factor chol = mlfs.linalg.cholesky(cov), so chol @ chol.T == cov.

        raises ValueError if the shapes don't match; cholesky raises np.linalg.LinAlgError if
        cov isn't symmetric positive definite.
        """
        self.mean = np.array(mean, dtype=float).reshape(-1)
        self.cov = np.array(cov, dtype=float)
        self.dim = self.mean.shape[0]
        if self.cov.shape != (self.dim, self.dim):
            raise ValueError(f"mean has {self.dim} entries but cov has shape {self.cov.shape}")
        self.chol = cholesky(self.cov)

    def sample(self, n: int = 1, *, rng=None) -> np.ndarray:
        """stage 1: n independent draws, shape (n, d).

        x = mean + L z with z ~ N(0, I) drawn by YOUR mlfs.sampling.normal. for a whole batch
        Z (n, d) of standard normals that's mean + Z @ L.T.
        """
        z = normal(0.0, 1.0, (n, self.dim), rng=rng)
        return self.mean + z @ self.chol.T

    def logpdf(self, x):
        """stage 2: the log density at x (d,) -> float, or at each row of x (n, d) -> (n,).

        log p(x) = -1/2 [ (x - mu)^T S^-1 (x - mu) + log det S + d log(2 pi) ].
        with S = L L^T: solve L z = x - mu (a triangular solve), so the quadratic form is
        |z|^2, and log det S = 2 sum(log diag L). never form S^-1.
        """
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        X = np.atleast_2d(x)
        if X.shape[1] != self.dim:
            raise ValueError(f"expected points of dimension {self.dim}, got shape {x.shape}")
        Z = solve_triangular(self.chol, (X - self.mean).T, lower=True)  # (d, n)
        maha = np.sum(Z**2, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(self.chol)))
        out = -0.5 * (maha + logdet + self.dim * math.log(2 * math.pi))
        return float(out[0]) if single else out

    def pdf(self, x):
        """stage 2: the density, exp(logpdf(x)). same shapes as logpdf."""
        lp = self.logpdf(x)
        return math.exp(lp) if isinstance(lp, float) else np.exp(lp)

    def entropy(self) -> float:
        """stage 2: the differential entropy in nats, d/2 (1 + log 2 pi) + 1/2 log det S."""
        logdet = 2.0 * np.sum(np.log(np.diag(self.chol)))
        return float(0.5 * self.dim * (1.0 + math.log(2 * math.pi)) + 0.5 * logdet)

    def marginal(self, idx):
        """stage 3: the distribution of the coordinates x[idx], as a MultivariateNormal.

        idx is a sequence of ints (in any order, which the result keeps): the mean is mean[idx]
        and the covariance is the submatrix cov[idx][:, idx]. marginalizing is just selecting.
        """
        idx = list(idx)
        return MultivariateNormal(self.mean[idx], self.cov[np.ix_(idx, idx)])

    def condition(self, idx, values):
        """stage 4: the distribution of the OTHER coordinates given x[idx] = values.

        with a = idx (observed) and b = the remaining coordinates IN THEIR ORIGINAL ORDER:
          mean_b|a = mean_b + S_ba S_aa^-1 (values - mean_a)
          cov_b|a  = S_bb - S_ba S_aa^-1 S_ab
        use cho_solve with the Cholesky factor of S_aa instead of an inverse, and symmetrize
        cov_b|a ((C + C.T) / 2) against rounding. raises ValueError if idx covers every coordinate.
        """
        a = list(idx)
        values = np.asarray(values, dtype=float).reshape(-1)
        b = [i for i in range(self.dim) if i not in set(a)]
        if not b:
            raise ValueError("conditioning on every coordinate leaves nothing to describe")
        S_aa = self.cov[np.ix_(a, a)]
        S_ba = self.cov[np.ix_(b, a)]
        L_aa = cholesky(S_aa)
        mean = self.mean[b] + S_ba @ cho_solve(L_aa, values - self.mean[a])
        C = self.cov[np.ix_(b, b)] - S_ba @ cho_solve(L_aa, S_ba.T)
        return MultivariateNormal(mean, (C + C.T) / 2)

    def __repr__(self) -> str:
        return f"MultivariateNormal(dim={self.dim}, mean={np.round(self.mean, 4).tolist()})"


@app.cell(hide_code=True)
def _():
    checks.run(1, MultivariateNormal=MultivariateNormal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.cell(hide_code=True)
def _():
    checks.run(2, MultivariateNormal=MultivariateNormal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.cell(hide_code=True)
def _():
    checks.run(3, MultivariateNormal=MultivariateNormal)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.cell(hide_code=True)
def _():
    checks.run(4, MultivariateNormal=MultivariateNormal)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `MultivariateNormal` is part of `mlfs.sampling` (try
    `from mlfs.sampling import MultivariateNormal` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
