# k-means · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`assign_to_nearest` is three lines: the $(n, k)$ matrix of squared distances from `pairwise_distances(X, centers, squared=True)`, its row-wise `argmin`, and fancy indexing `d2[np.arange(n), labels]` to pick each row's minimum. for `update_centers`, `np.bincount(labels, minlength=k)` counts each cluster's points. for the sums, sort the points by label (`order = np.argsort(labels, kind="stable")`): each cluster is then a contiguous block of `X[order]`, and `np.add.reduceat(X[order], starts, axis=0)` sums every block at once, where `starts` are the blocks' first rows (`cumsum` of the counts, shifted). pass only the starts of non-empty clusters. this uses $O(nd)$ memory, unlike a $(k, n)$ one-hot matrix, which matters when $n$ and $k$ are large. (`np.add.at(sums, labels, X)` also works, but it's an order of magnitude slower on wide data like MNIST; plain `sums[labels] += X` silently drops repeated indices.)

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
counts = np.bincount(labels, minlength=len(centers))
new = np.array(centers, dtype=float, copy=True)   # empty clusters keep their old centre
filled = counts > 0
order = np.argsort(labels, kind="stable")
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
new[filled] = np.add.reduceat(X[order], starts[filled], axis=0) / counts[filled, None]
return new
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep one array `d2` (shape `(n,)`): each point's squared distance to the nearest centre chosen so far. after choosing the first centre, it's the squared distance to that centre; after each new centre $c$, it's `np.minimum(d2, squared distance to c)`. the chosen points have `d2 == 0`, so they can never be picked again. if every remaining `d2` is 0 (fewer distinct points than $k$), fall back to uniform probabilities rather than dividing by zero.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
rng = as_rng(rng)
idx = [int(rng.integers(n))]
d2 = pairwise_distances(X, X[idx], squared=True)[:, 0]
for _ in range(1, n_clusters):
    j = int(rng.choice(n, p=d2 / d2.sum()))
    idx.append(j)
    d2 = np.minimum(d2, pairwise_distances(X, X[[j]], squared=True)[:, 0])
return X[idx].copy()
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

structure `fit` as two nested loops. the outer one runs `n_init` times (once if `self.init` is an array) and keeps the best `(inertia, centers, labels, n_iter)`; create the generator *once*, before it, with `rng = as_rng(self.rng)`, and pass it to `self._initial_centers(X, rng)`. the inner loop is `for n_iter in range(1, self.max_iters + 1)`: assign, update, measure the squared shift of the centres, break when it's at most `self.tol * np.mean(np.var(X, axis=0))`. after the inner loop, assign once more to get the final labels and inertia. `predict`, `transform` and `score` are each one line using `assign_to_nearest` or `pairwise_distances`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for _ in range(runs):
    centers = self._initial_centers(X, rng)
    for n_iter in range(1, self.max_iters + 1):
        labels, _ = assign_to_nearest(X, centers)
        new = update_centers(X, labels, centers)
        shift = np.sum((new - centers) ** 2)
        centers = new
        if shift <= tol:
            break
    labels, d2 = assign_to_nearest(X, centers)
    if best is None or d2.sum() < best[0]:
        best = (float(d2.sum()), centers, labels, n_iter)
self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

relabel to `0..k-1` with `np.unique(labels, return_inverse=True)`, build the one-hot matrix $M$ $(n, k)$ and the distance matrix $D$ $(n, n)$, and compute `S = D @ M`: `S[i, j]` is the total distance from point $i$ to cluster $j$. then $a(i)$ = `S[i, own] / (count[own] - 1)` (the point's distance to itself is 0, so it adds nothing), and the mean distances to the other clusters are `S / counts` with the own column set to `inf` before taking the row minimum.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
own = counts[inv]                                  # size of each point's cluster
a = np.where(own > 1, S[np.arange(n), inv] / np.maximum(own - 1, 1), 0.0)
means = S / counts
means[np.arange(n), inv] = np.inf
b = means.min(axis=1)
s = np.where(own > 1, (b - a) / np.maximum(a, b), 0.0)
return float(s.mean())
```

</details>
