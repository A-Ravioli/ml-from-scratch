# kernel ridge & random features · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

in `fit`: `X, y = check_X_y(X, y)`, `self.kernel_ = get_kernel(self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)`, `K = self.kernel_(X)`, then solve the $n \times n$ system $(K + \alpha I)\beta = y$. in `predict`: the cross-Gram matrix between the new points and `X_fit_`, times `dual_coef_`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.dual_coef_ = np.linalg.solve(K + self.alpha * np.eye(len(X)), y.astype(float))
...
return self.kernel_(X, self.X_fit_) @ self.dual_coef_
```

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

one generator, `gen = as_rng(self.rng)`, for both draws. the frequencies are a $(d, D)$ matrix whose entries are $\mathcal N(0, 2\gamma)$: standard normals times $\sqrt{2\gamma}$. the offsets are `gen.uniform(0, 2 * np.pi, size=D)`. `transform` is one line: scale, cosine, matrix product, broadcast add.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.random_weights_ + self.random_offset_)
```

nothing random happens in `transform`: it reuses the frequencies drawn in `fit`.

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

`m = min(self.n_components, len(X))`; landmarks `idx = gen.choice(len(X), size=m, replace=False)`; `W = self.kernel_(X[idx])`. for $W^{-1/2}$: `w, U = np.linalg.eigh(W)`, clip `w` at `1e-12`, then $U\,\mathrm{diag}(w^{-1/2})\,U^\top$. `transform` is `self.kernel_(X, self.components_) @ self.normalization_`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
w, U = np.linalg.eigh(self.kernel_(self.components_))
self.normalization_ = (U / np.sqrt(np.maximum(w, 1e-12))) @ U.T
```

`U / np.sqrt(w)` divides column $k$ of $U$ by $\sqrt{w_k}$, which is $U\,\mathrm{diag}(w^{-1/2})$ without building the diagonal matrix.

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

`primal_ridge` is the ridge normal equations in the $D$-dimensional feature space: a $D \times D$ system. `gram_error` is two Frobenius norms (`np.linalg.norm` of a matrix is Frobenius by default).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
return np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ y)
```

and `np.linalg.norm(Z @ Z.T - K) / np.linalg.norm(K)`.

</details>
