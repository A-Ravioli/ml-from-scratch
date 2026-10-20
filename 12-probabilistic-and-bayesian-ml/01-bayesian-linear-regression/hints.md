# bayesian linear regression · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

both are one broadcast. for polynomials, a column `x[:, None]` raised to a row of powers `np.arange(degree + 1)` gives the whole matrix. for bumps, `x[:, None] - centers[None, :]` is the `(n, k)` matrix of differences; then stack a column of ones in front with `np.hstack`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float).ravel()
return x[:, None] ** np.arange(degree + 1)

c = np.asarray(centers, dtype=float).ravel()
bumps = np.exp(-((x[:, None] - c[None, :]) ** 2) / (2.0 * width**2))
return np.hstack([np.ones((x.size, 1)), bumps])
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

build the precision first, $A = \alpha I + \beta\Phi^\top\Phi$ (shape `(d, d)`), invert it to get the covariance, and the mean is $\beta S\Phi^\top y$. `sequential_posterior` is the same with $\alpha I$ replaced by the prior precision $S_0^{-1}$ and an extra $S_0^{-1}m_0$ inside the mean.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
A = alpha * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi
cov = np.linalg.inv(A)
cov = (cov + cov.T) / 2
mean = beta * cov @ (Phi.T @ y)

P0 = np.linalg.inv(prior_cov)
cov = np.linalg.inv(P0 + beta * Phi.T @ Phi)
mean = cov @ (P0 @ prior_mean + beta * Phi.T @ y)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the mean is `Phi @ mean`. for the variance you need $\phi_i^\top S\phi_i$ for every row $i$: that's the diagonal of `Phi @ S @ Phi.T`, but compute it without the $n \times n$ matrix: multiply `Phi @ S` elementwise by `Phi` and sum each row. then add `1 / beta`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
mu = Phi @ mean
var = 1.0 / beta + np.sum((Phi @ cov) * Phi, axis=1)
return mu, var
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

get $m_N$ from your `posterior`, rebuild $A$, and use `np.linalg.slogdet(A)` (it returns `(sign, logdet)`). then assemble the five terms: $\frac d2\log\alpha$, $\frac n2\log\beta$, $-E(m_N)$, $-\frac12\log|A|$, $-\frac n2\log 2\pi$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
n, d = Phi.shape
mean, _ = posterior(Phi, y, alpha, beta)
A = alpha * np.eye(d) + beta * Phi.T @ Phi
_, logdet_A = np.linalg.slogdet(A)
E = 0.5 * beta * np.sum((y - Phi @ mean) ** 2) + 0.5 * alpha * mean @ mean
return float(0.5 * d * math.log(alpha) + 0.5 * n * math.log(beta) - E
             - 0.5 * logdet_A - 0.5 * n * math.log(2 * math.pi))
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`fit`: `X, y = check_X_y(X, y)`, then (if `optimize`) a `for it in range(self.max_iters)` loop: posterior at the current $(\alpha, \beta)$; $\lambda$ = `beta * np.linalg.eigvalsh(X.T @ X)` (compute the eigenvalues once, outside the loop); $\gamma$; the new $\alpha, \beta$; break when both relative changes are below $10^{-10}$. store `alpha_`, `beta_`, `n_iter_`, then the final posterior as `mean_`, `cov_`. `predict` is your `predictive` with `beta_` and a square root. `sample_weights`: `np.linalg.cholesky(cov_)` and `as_rng(rng).standard_normal((n, d))`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
eig0 = np.linalg.eigvalsh(X.T @ X)
for it in range(self.max_iters):
    mean, _ = posterior(X, y, alpha, beta)
    lam = beta * eig0
    gamma = np.sum(lam / (alpha + lam))
    alpha_new = np.clip(gamma / max(mean @ mean, 1e-300), 1e-12, 1e12)
    beta_new = np.clip(max(n - gamma, 1e-300) / max(np.sum((y - X @ mean) ** 2), 1e-300), 1e-12, 1e12)
    self.n_iter_ = it + 1
    done = abs(alpha_new - alpha) <= 1e-10 * alpha and abs(beta_new - beta) <= 1e-10 * beta
    alpha, beta = alpha_new, beta_new
    if done:
        break
```

</details>
