# hypothesis testing · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

build a boolean array "is this null value as extreme as the observed one?", with one rule per alternative, then count the `True`s. the $+1$ goes in both the numerator and the denominator.

</details>

<details><summary>hint 2: the approach</summary>

`tol = 1e-12 * max(1, abs(observed))`. greater: `null >= observed - tol`. less: `null <= observed + tol`. two-sided: `np.abs(null) >= abs(observed) - tol`. anything else: `raise ValueError`. then `(1 + extreme.sum()) / (1 + len(null))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
null = np.asarray(null_stats, dtype=float)
tol = 1e-12 * max(1.0, abs(observed))
if alternative == "greater":
    extreme = null >= observed - tol
elif alternative == "less":
    extreme = null <= observed + tol
elif alternative == "two-sided":
    extreme = np.abs(null) >= abs(observed) - tol
else:
    raise ValueError(f"unknown alternative {alternative!r}")
return float((1 + np.sum(extreme)) / (1 + len(null)))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

two branches. unpaired: concatenate, permute, split at `len(x)`. paired: for each pair, flip a fair coin to decide whether to swap. either way, call the statistic once per permutation and store the result.

</details>

<details><summary>hint 2: the approach</summary>

unpaired: `pooled = np.concatenate([x, y])` once, then in the loop `perm = rng.permutation(pooled)` and `statistic(perm[:len(x)], perm[len(x):])`. paired: check the lengths, then in the loop `swap = rng.random(len(x)) < 0.5` and `statistic(np.where(swap, y, x), np.where(swap, x, y))`. don't forget `statistic = mean_difference if statistic is None else statistic`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = as_rng(rng)
statistic = mean_difference if statistic is None else statistic
x, y = np.asarray(x), np.asarray(y)
out = np.empty(n_permutations)
if paired:
    if len(x) != len(y):
        raise ValueError("paired needs equal lengths")
    for b in range(n_permutations):
        swap = rng.random(len(x)) < 0.5
        out[b] = statistic(np.where(swap, y, x), np.where(swap, x, y))
else:
    pooled = np.concatenate([x, y])
    for b in range(n_permutations):
        perm = rng.permutation(pooled)
        out[b] = statistic(perm[:len(x)], perm[len(x):])
return out
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`permutation_test` is three lines: compute the observed statistic, draw the null with `permutation_null` (passing every argument through), and call `p_value`. `rejection_rate` is a loop of simulated experiments, counting `p <= alpha`.

</details>

<details><summary>hint 2: the approach</summary>

resolve the default statistic *before* computing the observed value, so `observed` and the null use the same one. in `rejection_rate`, normalize `rng` once and pass the same generator to both `simulate(rng)` and `test(x, y, rng)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def permutation_test(x, y, statistic=None, *, n_permutations=10000, alternative="two-sided", paired=False, rng=None):
    stat = mean_difference if statistic is None else statistic
    observed = float(stat(np.asarray(x), np.asarray(y)))
    null = permutation_null(x, y, stat, n_permutations=n_permutations, paired=paired, rng=rng)
    return observed, p_value(observed, null, alternative)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

Bonferroni is one comparison. for BH, work with the sorted p-values but remember where each came from (`np.argsort`), so you can write the decisions back in the original order.

</details>

<details><summary>hint 2: the approach</summary>

`order = np.argsort(p)`, `below = p[order] <= alpha * np.arange(1, m + 1) / m`. if any are `True`, `k` is one more than the index of the *last* `True` (`np.nonzero(below)[0].max() + 1`). set `reject[order[:k]] = True`. in both functions, raise `ValueError` if any p-value is NaN, and return an empty boolean array for no p-values (Bonferroni would otherwise divide by zero).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
p = np.asarray(p_values, dtype=float)
if np.isnan(p).any():
    raise ValueError("p-values contain NaN")
m = len(p)
order = np.argsort(p, kind="stable")
below = p[order] <= alpha * np.arange(1, m + 1) / m
reject = np.zeros(m, dtype=bool)
if below.any():
    k = int(np.nonzero(below)[0].max()) + 1
    reject[order[:k]] = True
return reject
```

</details>
