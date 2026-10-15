# dimensionality reduction · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`_, s, Vt = np.linalg.svd(X - mean, full_matrices=False)`. the variances are `s**2 / (n - 1)` and the ratios divide by their *total* (all of them). resolve `n_components` to an int `k`: `None` → `min(n, d)`; a float in $(0, 1)$ → the first index where `np.cumsum(ratio)` reaches it, plus one (`np.searchsorted` finds it); an int must be in `[1, min(n, d)]`. then keep `Vt[:k]` through `flip_signs`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
var = s**2 / (n - 1)
ratio = var / var.sum()
if k is None:
    k = min(n, d)
elif isinstance(k, float) and 0 < k < 1:
    k = int(np.searchsorted(np.cumsum(ratio), k - 1e-12) + 1)
elif not (isinstance(k, (int, np.integer)) and 1 <= k <= min(n, d)):
    raise ValueError(...)
self.components_ = flip_signs(Vt[:k])
self.explained_variance_, self.explained_variance_ratio_ = var[:k], ratio[:k]
```

(the `- 1e-12` keeps a ratio that's 0.9 up to rounding from needing one more component.)

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

scores are `(X - self.mean_) @ self.components_.T`, shape `(n, k)`. whitening divides each column by `np.sqrt(self.explained_variance_)` (broadcasting over rows). `inverse_transform` goes the other way: undo the whitening first, then `Z @ self.components_ + self.mean_`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def transform(self, X):
    Z = (check_X_y(X)[0] - self.mean_) @ self.components_.T
    return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

def inverse_transform(self, X):
    Z = np.asarray(X, dtype=float)
    if self.whiten:
        Z = Z * np.sqrt(self.explained_variance_)
    return Z @ self.components_ + self.mean_
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

reuse the SVD: the eigenvalues of $S = X_c^\top X_c/n$ are `s**2 / n`. the noise variance is `(total - lam[:k].sum()) / (d - k)` with `total = (Xc**2).sum() / n` (the trace of $S$), which also handles $n < d$. `W_ = U_k * sqrt(lam[:k] - sigma2)` with `U_k = flip_signs(Vt[:k]).T` (columns scaled by broadcasting). for `transform`, solve with $M = W^\top W + \sigma^2I$ rather than inverting; for `score_samples`, build $C$, take its Cholesky factor and follow topic 02's log density.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
M = self.W_.T @ self.W_ + self.noise_variance_ * np.eye(k)
Z = np.linalg.solve(M, self.W_.T @ (X - self.mean_).T).T          # transform

C = self.W_ @ self.W_.T + self.noise_variance_ * np.eye(d)
L = np.linalg.cholesky(C)
z = np.linalg.solve(L, (X - self.mean_).T)
logp = -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (z**2).sum(0))   # score_samples

z = rng.standard_normal((n, k)); eps = rng.standard_normal((n, d))
return z @ self.W_.T + self.mean_ + np.sqrt(self.noise_variance_) * eps                  # sample
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`kernel_matrix` is one line per kernel: `X @ Y.T`, `(gamma * X @ Y.T + coef0) ** degree`, `np.exp(-gamma * pairwise_distances(X, Y, squared=True))`. in `fit`, centre with broadcasting: `K - col_means[None, :] - row_means[:, None] + overall_mean`. `np.linalg.eigh` returns eigenvalues in *ascending* order, so reverse them before taking the top `k`, and pass the eigenvectors (as rows) through `flip_signs`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
K = self._kernel(X, X)
self.X_fit_, self.K_col_means_, self.K_mean_ = X, K.mean(axis=0), float(K.mean())
Kc = K - self.K_col_means_[None, :] - K.mean(axis=1)[:, None] + self.K_mean_
w, V = np.linalg.eigh(Kc)
top = np.argsort(w)[::-1][: self.n_components]
self.lambdas_ = w[top]
self.alphas_ = flip_signs(V[:, top].T).T
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`Kx = self._kernel(X, self.X_fit_)` is $(m, n)$. centre it like the training matrix, but the column means and overall mean are the stored *training* ones; only the row means come from `Kx` itself. then multiply by `self.alphas_` and divide each column by `sqrt(lambdas_)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
Kc = Kx - Kx.mean(axis=1)[:, None] - self.K_col_means_[None, :] + self.K_mean_
return Kc @ self.alphas_ / np.sqrt(self.lambdas_)
```

(the reference also maps components with eigenvalue $\approx 0$ to 0 instead of dividing by 0.)

</details>
