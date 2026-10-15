import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    from scipy.special import ndtr

    from mlfs import checks, lesson
    from mlfs.utils import as_rng  # yours, from chapter 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # stability, margins & PAC-Bayes · build (reference solution)

    work through the stages in order. you'll build the machinery behind the tightest bounds
    in the chapter: the binary KL divergence and its inverse, the PAC-Bayes-kl bound, the
    exact risk of a *stochastic* linear classifier with Gaussian weights and its KL to a
    Gaussian prior, and a bound for a trained linear model optimized over a grid of prior
    widths. the last stage builds the two other modern routes to generalization, margins and
    algorithmic stability.

    | stage | you build |
    |---|---|
    | 1 | `kl_bernoulli` → `mlfs.theory`: $\text{kl}(q\,\|\,p)$ between two coins |
    | 2 | `kl_inverse` → `mlfs.theory`: the largest $p$ with $\text{kl}(q\,\|\,p) \le c$, by bisection |
    | 3 | `pac_bayes_bound` → `mlfs.theory`, `mcallester_bound`: Maurer's PAC-Bayes-kl bound, and its square-root relaxation |
    | 4 | `gibbs_error`, `isotropic_gaussian_kl`: the stochastic classifier's risk, exactly, and its KL to the prior |
    | 5 | `pac_bayes_linear`: a PAC-Bayes bound for a trained linear classifier |
    | 6 | `linear_margin_bound`, `loo_stability`: margins and stability, measured |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def kl_bernoulli(q, p):
    """stage 1: kl(q || p) = q ln(q/p) + (1-q) ln((1-q)/(1-p)) in nats, with 0 ln 0 = 0; inf if p hits 0 or 1 where q doesn't."""
    q, p = float(q), float(p)
    out = 0.0
    for a, b in ((q, p), (1 - q, 1 - p)):
        if a > 0:
            if b <= 0:
                return math.inf
            out += a * math.log(a / b)
    return max(out, 0.0)


@app.cell(hide_code=True)
def _():
    checks.run(1, kl_bernoulli=kl_bernoulli)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def kl_inverse(q, c, *, tol=1e-10):
    """stage 2: the largest p in [q, 1] with kl_bernoulli(q, p) <= c, by bisection to width tol (returns the bracket's upper end)."""
    if c <= 0:
        return float(q)
    lo, hi = float(q), 1.0
    if kl_bernoulli(q, hi) <= c:
        return 1.0
    for _ in range(200):  # 200 halvings shrink [0, 1] far below any useful tol
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(q, mid) <= c:
            lo = mid
        else:
            hi = mid
    return hi


@app.cell(hide_code=True)
def _():
    checks.run(2, kl_inverse=kl_inverse, kl_bernoulli=kl_bernoulli)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def pac_bayes_bound(emp_risk, kl, n, delta):
    """stage 3: Maurer/Seeger PAC-Bayes-kl bound, kl_inverse(emp_risk, (kl + ln(2 sqrt(n) / delta)) / n)."""
    return kl_inverse(emp_risk, (kl + math.log(2 * math.sqrt(n) / delta)) / n)


@app.function
def mcallester_bound(emp_risk, kl, n, delta):
    """stage 3: the Pinsker relaxation, emp_risk + sqrt((kl + ln(2 sqrt(n) / delta)) / (2n))."""
    return emp_risk + math.sqrt((kl + math.log(2 * math.sqrt(n) / delta)) / (2 * n))


@app.cell(hide_code=True)
def _():
    checks.run(3, pac_bayes_bound=pac_bayes_bound, mcallester_bound=mcallester_bound, kl_inverse=kl_inverse,
               kl_bernoulli=kl_bernoulli)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def gibbs_error(w, sigma, X, y):
    """stage 4: exact mean 0-1 error of sign(w' . x) with w' ~ N(w, sigma^2 I), labels y in {-1, +1}: mean Phi(-y w.x / (sigma ||x||))."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    norms = np.linalg.norm(X, axis=1)
    margin = np.zeros(len(y))
    nz = norms > 0
    margin[nz] = y[nz] * (X[nz] @ np.asarray(w, dtype=float)) / (sigma * norms[nz])
    return float(np.mean(ndtr(-margin)))  # x = 0: the prediction is a coin flip, error 1/2


@app.function
def isotropic_gaussian_kl(mu_q, sigma_q, mu_p, sigma_p):
    """stage 4: KL(N(mu_q, sigma_q^2 I) || N(mu_p, sigma_p^2 I)) in nats."""
    mu_q = np.asarray(mu_q, dtype=float)
    d = mu_q.size
    r = (sigma_q / sigma_p) ** 2
    return float(0.5 * d * (r - 1 - math.log(r)) + np.sum((mu_q - np.asarray(mu_p, dtype=float)) ** 2) / (2 * sigma_p**2))


@app.cell(hide_code=True)
def _():
    checks.run(4, gibbs_error=gibbs_error, isotropic_gaussian_kl=isotropic_gaussian_kl)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def pac_bayes_linear(w, X, y, *, sigmas, prior_sigma=None, prior_mean=None, delta=0.05):
    """stage 5: PAC-Bayes bound for Q = N(w, s^2 I) vs P = N(prior_mean or 0, (prior_sigma or s)^2 I), best s in sigmas, union bound delta / len(sigmas) -> dict."""
    w = np.asarray(w, dtype=float)
    mu_p = np.zeros_like(w) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    n, k = len(y), len(sigmas)
    risks, kls, bounds = [], [], []
    for s in sigmas:
        sp = s if prior_sigma is None else prior_sigma
        risks.append(gibbs_error(w, s, X, y))
        kls.append(isotropic_gaussian_kl(w, s, mu_p, sp))
        bounds.append(pac_bayes_bound(risks[-1], kls[-1], n, delta / k))
    i = int(np.argmin(bounds))
    return {"sigma": float(sigmas[i]), "emp_risk": risks[i], "kl": kls[i], "bound": bounds[i],
            "bounds": np.array(bounds), "emp_risks": np.array(risks), "kls": np.array(kls)}


@app.cell(hide_code=True)
def _():
    checks.run(5, pac_bayes_linear=pac_bayes_linear, gibbs_error=gibbs_error, isotropic_gaussian_kl=isotropic_gaussian_kl,
               pac_bayes_bound=pac_bayes_bound)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 6")
    return


@app.function
def linear_margin_bound(margins, gamma, B, R, delta):
    """stage 6: ramp-loss margin bound, mean(clip(1 - margins / gamma, 0, 1)) + 2 B R / (gamma sqrt(n)) + sqrt(ln(1/delta) / (2n))."""
    m = np.asarray(margins, dtype=float)
    n = m.size
    ramp = np.clip(1 - m / gamma, 0.0, 1.0)
    return float(ramp.mean() + 2 * B * R / (gamma * math.sqrt(n)) + math.sqrt(math.log(1 / delta) / (2 * n)))


@app.function
def loo_stability(fit_predict, X, y, X_eval, *, n_remove=20, indices=None, rng=None):
    """stage 6: max over removed training points i (indices, or n_remove random ones) of max_x |f_S(x) - f_{S without i}(x)| on X_eval."""
    X, y = np.asarray(X), np.asarray(y)
    if indices is None:
        indices = as_rng(rng).choice(len(y), size=min(n_remove, len(y)), replace=False)
    full = np.asarray(fit_predict(X, y, X_eval), dtype=float)
    worst = 0.0
    for i in indices:
        keep = np.arange(len(y)) != i
        pred = np.asarray(fit_predict(X[keep], y[keep], X_eval), dtype=float)
        worst = max(worst, float(np.max(np.abs(pred - full))))
    return worst


@app.cell(hide_code=True)
def _():
    checks.run(6, linear_margin_bound=linear_margin_bound, loo_stability=loo_stability)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `kl_bernoulli`, `kl_inverse` and `pac_bayes_bound` are part of
    `mlfs.theory`. then open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
