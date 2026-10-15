# interpreting models · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

write a small `score(Xs)` that calls `estimator.score(Xs, y)` or `scoring(estimator, Xs, y)`, and compute the baseline once. then two loops: over features, and over repeats. per feature, take a fresh copy of `X`; per repeat, overwrite column `j` with `X[gen.permutation(n), j]` (the original column, permuted) and store `base - score(copy)`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
gen = as_rng(rng)
base = score(X)
imp = np.zeros((d, n_repeats))
for j in range(d):
    Xp = X.copy()
    for r in range(n_repeats):
        Xp[:, j] = X[gen.permutation(len(X)), j]
        imp[j, r] = base - score(Xp)
return {"importances_mean": imp.mean(1), "importances_std": imp.std(1), "importances": imp}
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`np.repeat(X, g, axis=0)` repeats each row $g$ times in a row (rows $0, 0, 0, 1, 1, 1, \dots$); `np.tile(grid, n)` gives grid values in the matching order. overwrite column `feature` with the tiled grid, call `model_response` once, and reshape to $(n, g)$. partial dependence is the mean over axis 0.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
Xg = np.repeat(X, len(grid), axis=0)
Xg[:, feature] = np.tile(grid, len(X))
return model_response(estimator, Xg, response).reshape(len(X), len(grid))
...
vals = np.unique(column)
if len(vals) <= grid_resolution:
    return vals.astype(float)
return np.unique(np.quantile(column, np.linspace(0.05, 0.95, grid_resolution)))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`masked_predictions`: broadcasting does the work. `np.where(masks[:, None, :], x[None, None, :], Xb[None, :, :])` has shape $(m, |B|, d)$: for every mask and background row, $x$'s values where the mask is True. reshape to $(m|B|, d)$, predict once, reshape back to $(m, |B|)$ and average over the background. exact Shapley: `itertools.product([False, True], repeat=d)` lists all masks; evaluate `value` on all of them once, keep a dict from mask (as a tuple) to its value, and for each $j$ and each mask without $j$, add the weight times $v(S \cup j) - v(S)$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
masks = np.array(list(itertools.product([False, True], repeat=d)), dtype=bool)
v = np.asarray(value(masks), dtype=float)
index = {tuple(mk): k for k, mk in enumerate(masks)}
for j in range(d):
    for k, mk in enumerate(masks):
        if mk[j]:
            continue
        with_j = mk.copy(); with_j[j] = True
        s = int(mk.sum())
        w = math.factorial(s) * math.factorial(d - s - 1) / math.factorial(d)
        phi[j] += w * (v[index[tuple(with_j)]] - v[k])
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

for the exact branch, list every mask with `0 < sum < d` and give it weight `(d - 1) / (comb(d, s) * s * (d - s))`. in `sample_coalitions`, `gen.choice(np.arange(1, d), size=n_samples, p=p / p.sum())` draws the sizes; a uniformly random subset of size $s$ per row is "the $s$ smallest of $d$ random numbers": rank random keys per row and keep ranks `< s`. in `kernel_shap`'s sampling branch, call it and give every row weight 1. then the elimination: with $Z$ the 0/1 mask matrix, regress `v - f0 - Z[:, -1] * (fx - f0)` on `Z[:, :-1] - Z[:, -1:]` by weighted least squares.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# sample_coalitions:
ranks = np.argsort(np.argsort(gen.random((n_samples, d)), axis=1), axis=1)
return ranks < s[:, None]
# kernel_shap:
v = masked_predictions(predict, Xb, x, masks)
Z = masks.astype(float)
target = v - f0 - Z[:, -1] * (fx - f0)
A = Z[:, :-1] - Z[:, -1:]
sw = np.sqrt(w)
head = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)[0]
return np.append(head, (fx - f0) - head.sum())
```

</details>
