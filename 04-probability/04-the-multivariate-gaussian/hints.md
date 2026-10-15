# the multivariate gaussian · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`__init__` is bookkeeping: `np.array(mean, dtype=float)` makes a copy, `self.dim` is its length, check that `cov` is `(d, d)`, and `self.chol = cholesky(self.cov)` (imported from `mlfs.linalg` in the setup cell). `sample` is "white noise, then the affine map".

</details>

<details><summary>hint 2: the approach</summary>

draw all the noise at once: `z = normal(0.0, 1.0, (n, self.dim), rng=rng)` gives an `(n, d)` array of standard normals. each row should become `mean + L @ row`. in row form, for the whole array, that's `mean + z @ L.T` (broadcasting adds `mean` to every row).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def sample(self, n=1, *, rng=None):
    z = normal(0.0, 1.0, (n, self.dim), rng=rng)
    return self.mean + z @ self.chol.T
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

handle both input shapes with one code path: remember whether `x.ndim == 1`, then work with `X = np.atleast_2d(x)` (shape `(n, d)`), and convert back at the end (a float for a single point).

</details>

<details><summary>hint 2: the approach</summary>

`solve_triangular(self.chol, (X - self.mean).T, lower=True)` solves $Lw = x - \mu$ for every point at once (the right-hand side is `(d, n)`, one column per point). the squared Mahalanobis distances are the column sums of `w**2`. `logdet = 2 * np.sum(np.log(np.diag(self.chol)))`. the entropy uses the same `logdet`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
Z = solve_triangular(self.chol, (X - self.mean).T, lower=True)
maha = np.sum(Z**2, axis=0)
logdet = 2.0 * np.sum(np.log(np.diag(self.chol)))
out = -0.5 * (maha + logdet + self.dim * math.log(2 * math.pi))
return float(out[0]) if single else out
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the marginal's mean is `self.mean[idx]`. its covariance needs the rows *and* columns in `idx`.

</details>

<details><summary>hint 2: the approach</summary>

`self.cov[np.ix_(idx, idx)]` selects the submatrix with rows `idx` and columns `idx`, in the order given. return a fresh `MultivariateNormal` built from those two.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
idx = list(idx)
return MultivariateNormal(self.mean[idx], self.cov[np.ix_(idx, idx)])
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

first get the index lists: `a = list(idx)` and `b = [i for i in range(self.dim) if i not in a]` (which keeps the original order). then pull out the blocks `S_aa`, `S_ba`, `S_bb` with `np.ix_`.

</details>

<details><summary>hint 2: the approach</summary>

factor `L_aa = cholesky(S_aa)` once. `cho_solve(L_aa, values - self.mean[a])` is $\Sigma_{aa}^{-1}(v - \mu_a)$, and `cho_solve(L_aa, S_ba.T)` is $\Sigma_{aa}^{-1}\Sigma_{ab}$ (`cho_solve` accepts a matrix right-hand side). multiply each on the left by `S_ba`. raise `ValueError` if `b` is empty.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
L_aa = cholesky(S_aa)
mean = self.mean[b] + S_ba @ cho_solve(L_aa, values - self.mean[a])
C = self.cov[np.ix_(b, b)] - S_ba @ cho_solve(L_aa, S_ba.T)
return MultivariateNormal(mean, (C + C.T) / 2)
```

</details>
