# gaussian mixtures & EM · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

loop over the $k$ components (they're few) and vectorize over the $n$ points. for each, `diff = X - means[j]`. full: `L = np.linalg.cholesky(cov)`, solve `L z = diff.T` (`np.linalg.solve(L, diff.T)`; you built a triangular solver in chapter 02, NumPy's general solve is fine here), Mahalanobis = column sums of `z**2`, log det = `2 * log(diag(L)).sum()`. diag: divide `diff**2` by the variance vector. spherical: divide by the single variance, and the log det is `d * log(var)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
out[:, j] = -0.5 * (d * np.log(2 * np.pi) + logdet + maha)
```

where for `"full"`: `maha = (np.linalg.solve(L, diff.T) ** 2).sum(0)`, `logdet = 2 * np.log(np.diag(L)).sum()`; for `"diag"`: `maha = (diff**2 / var).sum(1)`, `logdet = np.log(var).sum()`; for `"spherical"`: `maha = (diff**2).sum(1) / var`, `logdet = d * np.log(var)`.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

add `np.log(weights)` (shape `(k,)`, broadcasts over rows) to the log densities to get $\log\pi_j + \log\mathcal N$. the log-likelihood of each point is `logsumexp` of that along `axis=1`; subtract it (as a column) before exponentiating, and every row sums to 1.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
log_joint = gaussian_log_density(X, means, covariances, covariance_type) + np.log(weights)
log_px = logsumexp(log_joint, axis=1)
return np.exp(log_joint - log_px[:, None]), float(log_px.mean())
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`nk = resp.sum(0) + 10 * np.finfo(float).eps`, `weights = nk / n`, `means = resp.T @ X / nk[:, None]`. for a full covariance, weight each centred row by its responsibility: `(resp[:, j, None] * diff).T @ diff / nk[j]`. for diag, `resp[:, j] @ diff**2 / nk[j]` gives the $d$ variances directly. then add `reg_covar` to the variances (the diagonal for full).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
if covariance_type == "full":
    covs = np.stack([(resp[:, j, None] * (X - means[j])).T @ (X - means[j]) / nk[j] + reg_covar * np.eye(d)
                     for j in range(k)])
else:
    covs = np.stack([resp[:, j] @ (X - means[j]) ** 2 for j in range(k)]) / nk[:, None] + reg_covar
    if covariance_type == "spherical":
        covs = covs.mean(axis=1)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

in `fit`: validate, make `rng = as_rng(self.rng)`, get `resp = self._initial_resp(X, rng)`, run one `m_step` to get starting parameters, then loop `for it in range(1, self.max_iters + 1)`: E-step (append the mean log-likelihood to `self.lower_bound_history_`), M-step, and break once the last two history values differ by less than `self.tol` (set `converged_ = True`). store `n_iter_`. the other methods are one or two lines each on top of `e_step`, `gaussian_log_density` and `logsumexp`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.weights_, self.means_, self.covariances_ = m_step(X, resp, self.covariance_type, self.reg_covar)
self.lower_bound_history_, self.converged_ = [], False
for self.n_iter_ in range(1, self.max_iters + 1):
    resp, ll = e_step(X, self.weights_, self.means_, self.covariances_, self.covariance_type)
    self.lower_bound_history_.append(ll)
    self.weights_, self.means_, self.covariances_ = m_step(X, resp, self.covariance_type, self.reg_covar)
    if len(self.lower_bound_history_) > 1 and abs(ll - self.lower_bound_history_[-2]) < self.tol:
        self.converged_ = True
        break
return self
```

`score_samples` is `logsumexp(gaussian_log_density(...) + np.log(self.weights_), axis=1)`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`sample`: draw all the labels at once with `rng.choice(k, size=n, p=self.weights_)` and all the standard normal noise at once, `z = rng.standard_normal((n, d))`; then for each component, transform its rows: `mean + z @ L.T` for full (`L` the Cholesky factor), `mean + z * sqrt(var)` otherwise. `bic` is `-2 * score_samples(X).sum() + self.n_parameters() * np.log(n)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
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
```

</details>
