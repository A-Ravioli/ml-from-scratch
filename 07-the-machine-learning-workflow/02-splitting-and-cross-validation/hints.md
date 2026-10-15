# splitting & cross-validation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

work with row *indices*, not with the arrays: decide `train` and `test` (two int arrays) once, then index every array with them. that's what keeps `X_train[i]` paired with `y_train[i]`.

</details>

<details><summary>hint 2: the approach</summary>

turn `test_size` into a count first: `math.ceil(test_size * n)` for a float in $(0, 1)$, `int(test_size)` for an int; then check `1 <= n_test <= n - 1`. without stratification: `perm = as_rng(rng).permutation(n)`, `test, train = perm[:n_test], perm[n_test:]` (or the last `n_test` rows when `shuffle=False`). with stratification: compute each class's ideal count `n_test * n_c / n`, take floors, give the leftover rows to the largest fractional parts (`np.argsort(-(ideal - floor), kind="stable")[:extra]`), then pick that many random rows from each class.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
classes, cls = np.unique(np.asarray(stratify), return_inverse=True)
ideal = n_test * np.bincount(cls) / n
take = np.floor(ideal).astype(int)
take[np.argsort(-(ideal - take), kind="stable")[: n_test - take.sum()]] += 1
test = np.concatenate([gen.permutation(np.flatnonzero(cls == c))[:t] for c, t in enumerate(take)])
mask = np.zeros(n, dtype=bool); mask[test] = True
train = np.flatnonzero(~mask)          # permute both if you like
out = []
for a in arrays:
    a = np.asarray(a)
    out += [a[train], a[test]]
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`split` is a generator: use `yield`, not `return`. compute the fold sizes first (all `n // k`, plus one for the first `n % k`), then walk through `idx` (either `np.arange(n)` or a permutation) with a running start position.

</details>

<details><summary>hint 2: the approach</summary>

for each fold size: `test = np.sort(idx[start:start + size])`; the training rows are all the others, which a boolean mask gives you in sorted order: `mask = np.ones(n, bool); mask[test] = False; train = np.flatnonzero(mask)`. validate `2 <= k <= n` at the top of `split` (not in `__init__`: the protocol keeps `__init__` free of logic).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
sizes = np.full(k, n // k)
sizes[: n % k] += 1
start = 0
for size in sizes:
    test = np.sort(idx[start:start + size])
    mask = np.ones(n, dtype=bool)
    mask[test] = False
    yield np.flatnonzero(mask), test
    start += size
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

both splitters can be written as "compute a fold number for every row, then yield `(rows not in fold f, rows in fold f)` for each `f`". only the fold numbers differ.

</details>

<details><summary>hint 2: the approach</summary>

stratified: encode labels as integers with `np.unique(y, return_inverse=True)`, start from `idx` (`np.arange(n)` or a permutation), order it by class with a **stable** sort, `order = idx[np.argsort(cls[idx], kind="stable")]`, then `fold[order] = np.arange(n) % k`. grouped: `np.unique(groups, return_inverse=True, return_counts=True)` gives each row's group number and each group's size; loop over groups in the order `np.argsort(sizes, kind="stable")[::-1]`, assigning each to `np.argmin(fold_rows)` and adding its size to that fold; then `fold = fold_of_group[group_of_row]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# GroupKFold.split, after the checks
fold_rows = np.zeros(k, dtype=int)
fold_of_group = np.empty(len(sizes), dtype=int)
for grp in np.argsort(sizes, kind="stable")[::-1]:
    f = int(np.argmin(fold_rows))          # the lightest fold; ties -> lowest number
    fold_of_group[grp] = f
    fold_rows[f] += sizes[grp]
fold = fold_of_group[g]
for f in range(k):
    yield np.flatnonzero(fold != f), np.flatnonzero(fold == f)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

draw it for $n = 12$, $k = 3$: `test_size = 3`, and the test blocks are rows 3–5, 6–8, 9–11. each training set is everything before its block.

</details>

<details><summary>hint 2: the approach</summary>

the first test block starts at `n - k * test_size` (the leftover `n % (k + 1)` rows all go to the first training set). step through the starts with `range(n - k * test_size, n, test_size)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
test_size = n // (k + 1)
for start in range(n - k * test_size, n, test_size):
    yield np.arange(start), np.arange(start, start + test_size)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

normalize `cv` first, so the loop only ever sees `(train, test)` pairs: an int becomes `KFold(cv)`, anything with a `split` method is called as `cv.split(X, y, groups)`, and anything else is already an iterable of pairs.

</details>

<details><summary>hint 2: the approach</summary>

convert `X = np.asarray(X)` (and `y`, unless it's `None`) once, so `X[train]` works for lists too. in the loop: `est = clone(estimator)`, `est.fit(X[train], None if y is None else y[train])`, then score with `scoring(est, X_te, y_te)` if `scoring` is given, else `est.score(X_te, y_te)`. collect into lists and convert to arrays at the end.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for train, test in splits:
    est = clone(estimator)
    est.fit(X[train], None if y is None else y[train])
    test_scores.append(score(est, test))       # a small inner helper that picks the scorer
    if return_train_score:
        train_scores.append(score(est, train))
    if return_estimators:
        fitted.append(est)
out = {"test_score": np.array(test_scores)}
```

</details>
