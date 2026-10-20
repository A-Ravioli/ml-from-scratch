# gaussian processes · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

compute the scaled distances once: `r = pairwise_distances(X, Y) / lengthscale` (with `Y` passed through, so `None` stays `None`). then each kernel is a formula in `r`: for the Matérns, set `s = sqrt(3) * r` or `s = sqrt(5) * r` first.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if kind == "matern52":
    s = math.sqrt(5) * r
    return variance * (1 + s + s**2 / 3) * np.exp(-s)
```

$\frac53 r^2 = \frac{s^2}{3}$ when $s = \sqrt5\,r$.

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

`fit`: `check_X_y`, store `X_train_`, `y_mean_ = mean(y)`, `y_train_ = y - y_mean_`; `theta = np.log([self.lengthscale, self.variance, self.noise])`; if `self.optimize`, `theta = self.optimize_theta(theta)`; then `return self.set_theta(theta)`. `predict`: `Ks = stationary_kernel(self.X_train_, X, ...)` with the *fitted* `lengthscale_` and `variance_`, mean `Ks.T @ self.alpha_ + self.y_mean_`. for the uncertainty, `V = np.linalg.solve(self.L_, Ks)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if return_cov:
    Kss = stationary_kernel(X, kind=self.kernel, lengthscale=self.lengthscale_, variance=self.variance_)
    return mean, Kss - V.T @ V
return mean, np.sqrt(np.clip(self.variance_ - np.sum(V**2, axis=0), 0, None))
```

for an unfitted model (no `L_` attribute yet), return zeros and the prior: `np.full(m, sqrt(variance))` or the prior kernel matrix, using the `__init__` values.

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

`mean, cov = self.predict(X, return_cov=True)` handles both the prior and the posterior. symmetrize (`(cov + cov.T) / 2`), take `w, U = np.linalg.eigh(...)`, and build the root `U * np.sqrt(np.clip(w, 0, None))` (column $k$ scaled by $\sqrt{w_k}$).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
z = as_rng(rng).standard_normal((n_samples, len(mean)))
return mean + z @ root.T
```

each row of `z @ root.T` is `root @ z_row`, a draw from $\mathcal N(0, \Sigma)$.

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

rebuild $K_y = K + (\sigma_n^2 + \text{JITTER})I$ from `theta` (`ls, var, noise = np.exp(theta)`), Cholesky it inside `try`/`except np.linalg.LinAlgError`, solve for `a = K_y^{-1} y_train_` with two triangular-style solves, and combine $-\frac12 y^\top a - \sum\log L_{ii} - \frac n2\log 2\pi$. for the gradient you need $K_y^{-1}$ itself (solve against the identity) and the three matrices $\partial K_y/\partial\theta_j$: `kernel_lengthscale_grad(...)`, `K`, and `noise * np.eye(n)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
inner = np.outer(a, a) - Kinv
return np.array([0.5 * np.sum(inner * dK) for dK in (dK_ls, K, noise * np.eye(n))])
```

`np.sum(A * B)` is $\mathrm{tr}(AB)$ for symmetric $A$, $B$.

</details>

## stage 5

<details><summary>hint 1: the approach</summary>

define `f(t) = -self.log_marginal_likelihood(t)` and `g(t) = -self.lml_gradient(t)` (return zeros if the gradient's Cholesky fails). build the list of starts: `theta0`, then `n_restarts` of `theta0 + gen.uniform(-3, 3, size=3)` from one `gen = as_rng(self.rng)`. run `lbfgs(f, g, start, max_iters=200)` from each and keep the result with the smallest `fun`, comparing with `f(theta0)` too.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
best, best_val = theta0, f(theta0)
for start in starts:
    res = lbfgs(f, g, start, max_iters=200)
    if np.isfinite(res.fun) and res.fun < best_val:
        best, best_val = np.asarray(res.x, dtype=float), res.fun
return best
```

wrap the bodies of `f` and `g` in `np.errstate(all="ignore")`: line-search trial steps can make `exp(theta)` overflow, and that should just count as a failed step.

</details>

## stage 6

<details><summary>hint 1: the approach</summary>

broadcast `mu` and `sigma` to a common shape, compute `imp = mu - best - xi`, and use a mask `pos = sigma > 0` so you only divide where it's safe: `z = np.where(pos, imp / np.where(pos, sigma, 1.0), 0.0)`. then $\Phi(z) = \frac12\mathrm{erfc}(-z/\sqrt2)$ from `math.erfc` (not `1 + erf`, which loses every digit for very negative $z$) and $\varphi(z)$ from the formula.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
cdf = 0.5 * np.vectorize(math.erfc, otypes=[float])(-z / math.sqrt(2))
pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
ei = np.where(pos, np.maximum(imp * cdf + sigma * pdf, 0.0), 0.0)
return float(ei) if ei.ndim == 0 else ei
```

</details>
