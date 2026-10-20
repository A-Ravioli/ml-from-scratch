# graph & density clustering · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`np.unique(labels, return_inverse=True)` maps any labels to `0..k-1` in sorted order; `np.add.at(C, (a, b), 1)` fills the table. for ARI, one small helper `pairs(m) = sum(m * (m - 1) / 2)` applied to the table, its row sums and its column sums gives the three pair counts. for NMI, divide the table by $n$ to get the joint distribution, its row and column sums are the marginals, and sum $p\log(p/(p_uq_v))$ only over nonzero cells.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
C = contingency_matrix(labels_true, labels_pred).astype(float)
n = C.sum()
index, rows, cols = pairs(C), pairs(C.sum(1)), pairs(C.sum(0))
expected = rows * cols / (n * (n - 1) / 2)
top = 0.5 * (rows + cols) - expected
return 1.0 if top == 0 else (index - expected) / top
```

```python
P = C / C.sum(); pu, pv = P.sum(1), P.sum(0)
nz = P > 0
mi = np.sum(P[nz] * np.log(P[nz] / np.outer(pu, pv)[nz]))
return mi / (0.5 * (entropy(pu) + entropy(pv)))      # 1.0 if both entropies are 0
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

for the kNN affinity, set the diagonal of the distance matrix to `inf` before `argsort`, so a point isn't its own neighbour; fill a 0/1 matrix `A` with fancy indexing, then `W = (A + A.T) / 2`. in `fit`: degrees `d = W.sum(1)`, `s = 1 / sqrt(d)`, and `M = s[:, None] * W * s[None, :]` is $D^{-1/2}WD^{-1/2}$. its *largest* eigenvectors are the ones you want (the smallest of $L_{\text{sym}} = I - M$); `eigh` sorts ascending. normalize each row of the $(n, k)$ matrix, then `KMeans(self.n_clusters, n_init=10, rng=self.rng)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
W = affinity_matrix(X, self.affinity, self.gamma, self.n_neighbors)
s = 1.0 / np.sqrt(np.maximum(W.sum(axis=1), 1e-12))
w, V = np.linalg.eigh(s[:, None] * W * s[None, :])
U = V[:, np.argsort(w)[::-1][: self.n_clusters]]
U = U / np.linalg.norm(U, axis=1, keepdims=True)
self.affinity_matrix_, self.embedding_ = W, U
self.labels_ = KMeans(self.n_clusters, n_init=10, rng=self.rng).fit(U).labels_
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

precompute, once: the boolean matrix `pairwise_distances(X) <= eps`, each point's neighbour list (`np.flatnonzero` of its row), and `core = row sums >= min_samples` (the point itself is in its own row). then a loop over points in index order: skip labelled or non-core points; otherwise label it with a new cluster id and expand with a stack. when you pop a point, label each still-unlabelled neighbour with the current id, and push it only if it's core.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
labels = np.full(n, -1)
cluster = 0
for i in range(n):
    if labels[i] != -1 or not core[i]:
        continue
    labels[i] = cluster
    stack = [i]
    while stack:
        p = stack.pop()
        for q in neighbours[p]:
            if labels[q] == -1:
                labels[q] = cluster
                if core[q]:
                    stack.append(q)
    cluster += 1
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

keep: the distance matrix `D` (squared distances for Ward) with `inf` on the diagonal; `size` (points per row's cluster); `node` (the tree id each row currently represents, initially `arange(n)`); and the member lists of the active clusters. each of the $n - 1$ steps: `i, j = np.unravel_index(np.argmin(D), D.shape)` with `i < j`; record the sorted pair `(node[i], node[j])` and the height; compute the Lance–Williams row `new` for all clusters at once (vector operations with `D[i]`, `D[j]` and `size`); write it into row and column `i`; set row and column `j` to `inf`; update `size[i]`, the members, and `node[i] = n + step`. take a snapshot of the members when `n_clusters` clusters remain.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
ni, nj, dij = size[i], size[j], D[i, j]
new = {"single": np.minimum(D[i], D[j]),
       "complete": np.maximum(D[i], D[j]),
       "average": (ni * D[i] + nj * D[j]) / (ni + nj),
       "ward": ((ni + size) * D[i] + (nj + size) * D[j] - size * dij) / (ni + nj + size)}[self.linkage]
D[i, :] = D[:, i] = new
D[i, i] = np.inf
D[j, :] = D[:, j] = np.inf
size[i] = ni + nj
```

(inactive rows hold `inf`, so their entries in `new` stay `inf` and they're never chosen again.) number the final clusters by their smallest member: `sorted(groups, key=min)`.

</details>
