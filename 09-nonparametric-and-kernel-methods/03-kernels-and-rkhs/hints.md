# kernels & RKHS · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: the approach</summary>

linear: `X @ Y.T` (with `Y = X` when `Y is None`). polynomial: build on your linear kernel. RBF: `np.exp(-gamma * pairwise_distances(X, Y, squared=True))`, passing `Y` through *unchanged* (so `None` stays `None`). Laplacian: start from `np.zeros((n, m))` and add `np.abs(X[:, j, None] - Y[None, :, j])` for each feature `j`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def polynomial_kernel(X, Y=None, *, degree=3, gamma=1.0, coef0=1.0):
    return (gamma * linear_kernel(X, Y) + coef0) ** degree
```

`X[:, j, None]` is a column $(n, 1)$ and `Y[None, :, j]` a row $(1, m)$; their difference broadcasts to $(n, m)$.

</details>

## stage 2

<details><summary>hint 1: the approach</summary>

build the four blocks and `np.hstack` them: a column of ones, `np.sqrt(2) * X`, `X ** 2`, and the cross terms. `i, j = np.triu_indices(d, k=1)` lists every pair $i < j$ in the required order, so the cross terms are `np.sqrt(2) * X[:, i] * X[:, j]`, an $(n, \binom d2)$ block in one line.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
i, j = np.triu_indices(X.shape[1], k=1)
return np.hstack([np.ones((len(X), 1)), np.sqrt(2) * X, X**2, np.sqrt(2) * X[:, i] * X[:, j]])
```

</details>

## stage 3

<details><summary>hint 1: the approach</summary>

keep a table from name to (function, names of the parameters it takes): `"rbf": (rbf_kernel, ("gamma",))`, `"poly": (polynomial_kernel, ("degree", "gamma", "coef0"))`, and so on. filter `params` down to the allowed names, then return a small function (a closure) that calls the kernel with them.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
fn, names = table[kernel]
bound = {k: v for k, v in params.items() if k in names}

def K(X, Y=None):
    return fn(X, Y, **bound)

return K
```

handle `callable(kernel)` first (return it), and raise `ValueError` for names not in the table.

</details>

## stage 4

<details><summary>hint 1: the approach</summary>

`is_psd`: return `False` for non-square input; check symmetry with `np.allclose(K, K.T, rtol=0, atol=tol * scale)` where `scale = max(1, max|K_ij|)`; then `w = np.linalg.eigvalsh(K)` (ascending) and compare `w[0]` with `-tol * max(1, abs(w[-1]))`. return a real `bool`. `center_gram`: subtract the column means (`K.mean(axis=0, keepdims=True)`) and the row means, and add back the overall mean.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
```

this is $HKH$ written out (on paper 4): no need to build $H$ or multiply $n \times n$ matrices.

</details>

## stage 5

<details><summary>hint 1: the approach</summary>

`kernel` is a function: `K = kernel(X)` is the training Gram matrix, and `kernel(X_new, X_train)` the $(m, n)$ cross-Gram matrix. the interpolant's coefficients solve $K\alpha = y$; `np.linalg.lstsq(K, y, rcond=None)[0]` works even when $K$ is singular. the norm is $\sqrt{\alpha^\top K\alpha}$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def kernel_expansion(alpha, X_train, X_new, kernel):
    return kernel(X_new, X_train) @ alpha
```

in `rkhs_norm`, clip $\alpha^\top K\alpha$ at 0 before the square root: round-off can make it $-10^{-17}$.

</details>
