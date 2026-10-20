# manifold learning · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

get the *squared* distances first: `pairwise_distances(X, squared=True)` for `"euclidean"`, `D**2` for `"precomputed"` (after checking it's square and symmetric). double centring is broadcasting: subtract the column means (as a row) and the row means (as a column), add the overall mean, multiply by $-\frac12$. then `np.linalg.eigh` (ascending!), take the top `n_components`, and scale each eigenvector by the square root of its eigenvalue (clip negative eigenvalues at 0). `flip_signs` works on rows, so transpose in and out.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
B = -0.5 * (D2 - D2.mean(axis=0)[None, :] - D2.mean(axis=1)[:, None] + D2.mean())
w, V = np.linalg.eigh(B)
top = np.argsort(w)[::-1][: self.n_components]
self.eigenvalues_ = w[top]
self.embedding_ = flip_signs(V[:, top].T).T * np.sqrt(np.maximum(w[top], 0.0))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`np.argsort(D, axis=1)[:, 1:k + 1]` gives each point's $k$ nearest neighbours (column 0 is the point itself). start from a matrix full of `np.inf`, write the neighbour distances in with fancy indexing, symmetrize with `np.minimum(G, G.T)`, and zero the diagonal. Floyd–Warshall is a single loop over the intermediate vertex `m`, with an `(n, n)` broadcast inside: `G[:, m, None] + G[None, m, :]` is the length of every path $i \to m \to j$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
nbrs = np.argsort(D, axis=1)[:, 1 : n_neighbors + 1]
G = np.full((n, n), np.inf)
rows = np.repeat(np.arange(n), n_neighbors)
G[rows, nbrs.ravel()] = D[rows, nbrs.ravel()]
G = np.minimum(G, G.T)
np.fill_diagonal(G, 0.0)
for m in range(n):
    G = np.minimum(G, G[:, m, None] + G[None, m, :])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`fit` is three steps: `geodesic_distances(X, self.n_neighbors)`, a check that every entry is finite (`np.isfinite(...).all()`, else `ValueError`), and your `MDS` with `dissimilarity="precomputed"` on the result.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
self.dist_matrix_ = geodesic_distances(X, self.n_neighbors)
if not np.all(np.isfinite(self.dist_matrix_)):
    raise ValueError("the neighbourhood graph is disconnected: increase n_neighbors")
self.embedding_ = MDS(self.n_components, dissimilarity="precomputed").fit(self.dist_matrix_).embedding_
return self
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

keep three arrays of shape `(n,)`: `beta` (start at 1), and a bracket `lo` (start `-inf`) and `hi` (start `+inf`). each round: compute every row's probabilities at once (logits `-beta[:, None] * D2`, the diagonal set to `-inf`, minus each row's max, exponentiate, normalize), their entropies, and stop if all are within `tol` of `log(perplexity)`. where the entropy is too high, `lo = beta` (beta must grow); elsewhere `hi = beta`. the new beta is the midpoint if both ends are finite, else double (no upper end yet) or halve (no lower end yet).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for _ in range(max_iters):
    logits = np.where(off_diagonal, -beta[:, None] * D2, -np.inf)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    H = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0), axis=1)
    if np.all(np.abs(H - target) < tol):
        break
    too_flat = H > target
    lo = np.where(too_flat, beta, lo)
    hi = np.where(too_flat, hi, beta)
    beta = np.where(np.isinf(hi), beta * 2, np.where(np.isinf(lo), beta / 2, (lo + hi) / 2))
return P
```

`joint_probabilities` is then `(Pc + Pc.T) / (2 * n)`.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`num = 1 / (1 + pairwise_distances(Y, Y, squared=True))` with its diagonal zeroed; `Q = num / num.sum()`. the KL is `sum(P * log(P / Q))` where terms with `P == 0` count as 0 (`np.maximum(P, 1e-300)` inside the log does that without a slow boolean mask). for the gradient, build `W = (P - Q) * num`; the sum $\sum_jW_{ij}(y_i - y_j)$ for every $i$ at once is `W.sum(axis=1)[:, None] * Y - W @ Y`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
num = 1.0 / (1.0 + pairwise_distances(Y, Y, squared=True))
np.fill_diagonal(num, 0.0)
Q = np.maximum(num / num.sum(), 1e-300)
W = (P - Q) * num
grad = 4.0 * (W.sum(axis=1)[:, None] * Y - W @ Y)
if not kl:
    return None, grad
return float(np.sum(P * np.log(np.maximum(P, 1e-300) / Q))), grad
```

</details>

## stage 6

<details><summary>hint 1: a nudge</summary>

compute `P` once, then initialize `Y = 1e-4 * rng.standard_normal((n, self.n_components))`, `update = zeros`, `gains = ones`. in each of the `n_iters` iterations: `early = it < 250`; the gradient with `P * self.early_exaggeration` if early, else `P`; momentum 0.5 if early, else 0.8; update the gains, then the step, then `Y`, then subtract `Y.mean(axis=0)`. finally `kl_divergence_` from `tsne_kl_gradient(Y, P)` with the plain `P`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for it in range(self.n_iters):
    early = it < 250
    _, grad = tsne_kl_gradient(Y, P * self.early_exaggeration if early else P, kl=False)
    momentum = 0.5 if early else 0.8
    gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
    gains = np.maximum(gains, 0.01)
    update = momentum * update - self.lr * gains * grad
    Y = Y + update
    Y = Y - Y.mean(axis=0)
```

</details>
