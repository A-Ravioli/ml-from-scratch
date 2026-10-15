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
    # variational inference · build (reference solution)

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
    mean = np.asarray(mean, dtype=float)
    log_std = np.asarray(log_std, dtype=float)
    z = mean + np.exp(log_std) * np.asarray(eps, dtype=float)
    d = mean.size
    entropy = float(np.sum(log_std)) + 0.5 * d * (1.0 + math.log(2 * math.pi))
    return float(np.mean([log_prob(zs) for zs in z])) + entropy


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
    x = np.asarray(x, dtype=float).ravel()
    rng = as_rng(rng)
    n, K = x.size, n_components
    m = x[rng.choice(n, size=K, replace=False)].copy()
    s2 = np.ones(K)
    elbo = []
    phi = np.full((n, K), 1.0 / K)
    for _ in range(max_iters):
        logits = x[:, None] * m[None, :] - 0.5 * (s2 + m**2)[None, :]
        phi = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        nk = phi.sum(axis=0)
        s2 = 1.0 / (1.0 / prior_var + nk)
        m = s2 * (phi * x[:, None]).sum(axis=0)
        elbo.append(gmm_elbo(x, m, s2, phi, prior_var))
        if len(elbo) > 1 and abs(elbo[-1] - elbo[-2]) < tol:
            break
    return {"m": m, "s2": s2, "phi": phi, "elbo": elbo}


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
    mean = np.asarray(mean, dtype=float)
    log_std = np.asarray(log_std, dtype=float)
    eps = np.asarray(eps, dtype=float)
    std = np.exp(log_std)
    z = mean + std * eps
    G = np.array([grad_log_prob(zs) for zs in z])  # (S, d)
    return G.mean(axis=0), (G * eps * std).mean(axis=0) + 1.0


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
    rng = as_rng(rng)
    if grad_log_prob is None:
        grad_log_prob = lambda z: grad(log_prob, z)
    const = 0.5 * dim * (1.0 + math.log(2 * math.pi))
    mean = Param(np.zeros(dim))
    scale = Param(np.zeros(dim) if family == "diagonal" else np.zeros((dim, dim)))
    if family not in ("diagonal", "full"):
        raise ValueError("family must be 'diagonal' or 'full'")
    opt = Adam([mean, scale], lr=lr)
    history = []
    for _ in range(n_iters):
        eps = rng.standard_normal((n_samples, dim))
        if family == "diagonal":
            z = mean.data + np.exp(scale.data) * eps
            entropy = float(np.sum(scale.data)) + const
            g_mean, g_scale = elbo_grad_diag(grad_log_prob, mean.data, scale.data, eps)
        else:
            L = np.tril(scale.data, -1) + np.diag(np.exp(np.diag(scale.data)))
            z = mean.data + eps @ L.T
            entropy = float(np.sum(np.diag(scale.data))) + const
            g_mean, g_L = elbo_grad_full(grad_log_prob, mean.data, L, eps)
            g_scale = np.tril(g_L, -1) + np.diag(np.diag(g_L) * np.diag(L))  # chain rule through exp on the diagonal
        history.append(float(np.mean([log_prob(zs) for zs in z])) + entropy)
        mean.grad = -g_mean  # Adam minimizes: descend on -ELBO
        scale.grad = -g_scale
        opt.step()
    if family == "diagonal":
        cov = np.diag(np.exp(2 * scale.data))
    else:
        L = np.tril(scale.data, -1) + np.diag(np.exp(np.diag(scale.data)))
        cov = L @ L.T
    return MultivariateNormal(mean.data.copy(), cov), history


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
    mean = np.asarray(mean, dtype=float)
    L = np.asarray(L, dtype=float)
    eps = np.asarray(eps, dtype=float)
    z = mean + eps @ L.T
    G = np.array([grad_log_prob(zs) for zs in z])  # (S, d)
    g_L = np.tril(G.T @ eps / len(eps)) + np.diag(1.0 / np.diag(L))
    return G.mean(axis=0), g_L


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
