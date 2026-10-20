import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np

    from mlfs import checks, lesson
    from mlfs.autodiff import grad  # yours, from chapter 03
    from mlfs.optim import Adam, Param  # yours, from chapter 06
    from mlfs.sampling import MultivariateNormal  # yours, from chapter 04
    from mlfs.utils import as_rng, logsumexp  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # variational inference · build

    work through the stages in order. instead of sampling the posterior, variational
    inference picks the member $q$ of a simple family (here: Gaussians) that is closest to it
    in KL divergence, by maximizing the **ELBO** $\mathcal L(q) = \mathbb E_q[\log p(z)] + H(q)$.
    `cavi_gmm` and `reparam_vi` become part of `mlfs.infer`.

    | stage | you build |
    |---|---|
    | 1 | `elbo_diag`: a Monte Carlo estimate of the ELBO for a diagonal Gaussian $q$ |
    | 2 | `cavi_gmm` → `mlfs.infer`: coordinate-ascent mean-field VI for a Bayesian Gaussian mixture |
    | 3 | `elbo_grad_diag`: the reparameterization gradient of the ELBO |
    | 4 | `reparam_vi` → `mlfs.infer`: stochastic gradient ascent on the ELBO with your Adam |
    | 5 | `elbo_grad_full`: full-covariance Gaussians via a Cholesky factor (`family='full'`) |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def elbo_diag(log_prob, mean, log_std, eps):
    """stage 1: ELBO estimate for q = N(mean, diag(exp(log_std)^2)) from standard normal draws eps (S, d)."""
    raise NotImplementedError("stage1: implement elbo_diag")


@app.cell(hide_code=True)
def _():
    checks.run(1, elbo_diag=elbo_diag)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def gmm_elbo(x, m, s2, phi, prior_var):
    """the ELBO of the 1-D Bayesian GMM (unit-variance components, uniform weights), all constants included."""
    x = np.asarray(x, dtype=float)
    n, K = phi.shape
    e_log_prior = np.sum(-0.5 * np.log(2 * np.pi * prior_var) - (s2 + m**2) / (2 * prior_var))
    e_log_c = -n * math.log(K)
    sq = x[:, None] ** 2 - 2 * x[:, None] * m[None, :] + (s2 + m**2)[None, :]
    e_log_lik = np.sum(phi * (-0.5 * np.log(2 * np.pi) - 0.5 * sq))
    h_mu = np.sum(0.5 * np.log(2 * np.pi * np.e * s2))
    with np.errstate(divide="ignore", invalid="ignore"):
        h_c = -np.sum(np.where(phi > 0, phi * np.log(phi), 0.0))
    return float(e_log_prior + e_log_c + e_log_lik + h_mu + h_c)


@app.function
def cavi_gmm(x, n_components, *, prior_var=10.0, max_iters=100, tol=1e-08, rng=None):
    """stage 2: CAVI for a 1-D Bayesian GMM (unit variances, mu_k ~ N(0, prior_var)); returns dict(m, s2, phi, elbo)."""
    raise NotImplementedError("stage2: implement cavi_gmm")


@app.cell(hide_code=True)
def _():
    checks.run(2, cavi_gmm=cavi_gmm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def elbo_grad_diag(grad_log_prob, mean, log_std, eps):
    """stage 3: reparameterization gradient of elbo_diag w.r.t. (mean, log_std), each of shape (d,)."""
    raise NotImplementedError("stage3: implement elbo_grad_diag")


@app.cell(hide_code=True)
def _():
    checks.run(3, elbo_grad_diag=elbo_grad_diag)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def reparam_vi(log_prob, dim, *, grad_log_prob=None, family="diagonal", n_samples=16, lr=0.01, n_iters=2000, rng=None):
    """stage 4: fit q = N(mean, cov) by maximizing the reparameterized ELBO with Adam; returns (MultivariateNormal, elbo list)."""
    raise NotImplementedError("stage4: implement reparam_vi")


@app.cell(hide_code=True)
def _():
    checks.run(4, reparam_vi=reparam_vi, elbo_grad_diag=elbo_grad_diag)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def elbo_grad_full(grad_log_prob, mean, L, eps):
    """stage 5: gradient of the ELBO for q = N(mean, L L^T), z = mean + L eps, w.r.t. mean (d,) and lower-triangular L (d, d)."""
    raise NotImplementedError("stage5: implement elbo_grad_full")


@app.cell(hide_code=True)
def _():
    checks.run(5, reparam_vi=reparam_vi, elbo_grad_full=elbo_grad_full)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `cavi_gmm` and `reparam_vi` are part of `mlfs.infer` (try
    `from mlfs.infer import reparam_vi` in a scratch cell). then open the lab:
    `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
