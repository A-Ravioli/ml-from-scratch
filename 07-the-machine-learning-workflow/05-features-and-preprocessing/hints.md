# features & preprocessing · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

everything is a column-wise reduction in `fit` (`X.mean(axis=0)`, `X.std(axis=0)`, `X.min(axis=0)`, `X.max(axis=0)`) and broadcasting in `transform`: an `(n, d)` array minus a `(d,)` array subtracts from every row.

</details>

<details><summary>hint 2: a nudge</summary>

handle the switches by choosing the statistics, not by branching in `transform`: with `with_mean=False`, store `mean_ = np.zeros(d)`; with `with_std=False`, `scale_ = np.ones(d)`. then `transform` is always `(X - mean_) / scale_` and `inverse_transform` always `X * scale_ + mean_`. zero spreads: `np.where(std == 0, 1.0, std)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# MinMaxScaler.transform
lo, hi = self.feature_range
rng = self.data_max_ - self.data_min_
rng = np.where(rng == 0, 1.0, rng)
return lo + (X - self.data_min_) / rng * (hi - lo)
```

the inverse solves that for `X`: `(Xs - lo) / (hi - lo) * rng + data_min_`.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

work one input column at a time and `np.hstack` the blocks at the end. for a column, you need each row's position in the sorted `categories_[j]`, and whether the value is there at all.

</details>

<details><summary>hint 2: the approach</summary>

`idx = np.searchsorted(cats, col)` gives the position where each value *would* go; clip it to `len(cats) - 1` and check `cats[idx] == col` to know which values are real categories. build `block = np.zeros((n, len(cats)))` and set `block[rows_known, idx_known] = 1`. unknowns: raise if `handle_unknown == "error"`, else just leave their rows at zero. `drop="first"`: append `block[:, 1:]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for j, cats in enumerate(self.categories_):
    col = X[:, j]
    idx = np.minimum(np.searchsorted(cats, col), len(cats) - 1)
    known = cats[idx] == col
    if not known.all() and self.handle_unknown == "error":
        raise ValueError("unseen category")
    block = np.zeros((len(X), len(cats)))
    block[np.flatnonzero(known), idx[known]] = 1.0
    blocks.append(block[:, 1:] if self.drop == "first" else block)
return np.hstack(blocks)
```

for `inverse_transform`, walk through the blocks with a running column offset; `np.argmax(block, axis=1)` picks the category, and rows with no nonzero entry need the special cases.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

for each column, `seen = X[~np.isnan(X[:, j]), j]` are the observed values; the statistic comes from those.

</details>

<details><summary>hint 2: the approach</summary>

`np.mean`, `np.median` on `seen`; for `most_frequent`, `values, counts = np.unique(seen, return_counts=True)` and `values[np.argmax(counts)]` (`np.unique` sorts, and `argmax` takes the first maximum, so ties go to the smallest value). in `transform`, copy with `np.array(X, dtype=float)` and fill all NaNs at once: `rows, cols = np.nonzero(np.isnan(X)); X[rows, cols] = self.statistics_[cols]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if self.strategy == "constant":
    self.statistics_ = np.full(X.shape[1], 0.0 if self.fill_value is None else float(self.fill_value))
    return self
```

and remember the empty-column case before computing a mean: `len(seen) == 0` raises `ValueError`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`itertools.combinations_with_replacement(range(2), 2)` gives `(0, 0), (0, 1), (1, 1)`: the monomials $x_0^2, x_0x_1, x_1^2$ as lists of feature indices. `np.bincount(combo, minlength=d)` turns one into its exponent row.

</details>

<details><summary>hint 2: the approach</summary>

loop `k` from 0 (or 1 without the bias) to `degree`, and within each `k` over the combinations, collecting exponent rows; `np.array(rows).reshape(-1, d)` is `powers_` (the reshape keeps the shape right even if `rows` is empty). `transform`: `X[:, None, :] ** powers_[None, :, :]` has shape `(n, n_out, d)`; take the product over the last axis.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
combos = itertools.combinations if self.interaction_only else itertools.combinations_with_replacement
rows = [np.bincount(np.array(c, dtype=int), minlength=d)
        for k in range(0 if self.include_bias else 1, self.degree + 1)
        for c in combos(range(d), k)]
self.powers_ = np.array(rows, dtype=int).reshape(-1, d)
```

(`np.array((), dtype=int)` handles the empty combination of degree 0: all exponents 0, the bias column.)

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`fit` is a loop over `self.steps[:-1]` that keeps replacing `Xt`, then one call on `self.steps[-1][1]`. the given `transform_before_last(X)` does the prediction-time half: use it in `predict`, `predict_proba` and `score`.

</details>

<details><summary>hint 2: the approach</summary>

validate first: `names = [n for n, _ in self.steps]` must have no duplicates; every step before the last needs `hasattr(step, "fit") and hasattr(step, "transform")`. then `Xt = step.fit_transform(Xt, y)` for those steps, and `self.steps[-1][1].fit(Xt, y)`. for `make_pipeline`, count each lowercased class name first; names that occur more than once get `-1`, `-2`, ... in order.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def predict(self, X):
    return self.steps[-1][1].predict(self.transform_before_last(X))

def transform(self, X):
    return self.steps[-1][1].transform(self.transform_before_last(X))
```

</details>

## stage 6

<details><summary>hint 1: a nudge</summary>

`get_params` starts from `{"steps": self.steps}` and adds `f"{name}__{key}"` for every `key, value` in each step's own `get_params()`. that's all nesting needs: a nested pipeline's own keys already contain `"__"`.

</details>

<details><summary>hint 2: the approach</summary>

in `set_params`, split each key once with `key.partition("__")`. `"steps"` replaces the list. an unknown head raises `ValueError`. with a separator, forward the rest: `self.named_steps[name].set_params(**{rest: value})`. without one, rebuild the list with that step swapped: `[(n, value if n == name else s) for n, s in self.steps]`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for key, value in params.items():
    if key == "steps":
        self.steps = value
        continue
    name, sep, rest = key.partition("__")
    if name not in self.named_steps:
        raise ValueError(f"no step {name!r}")
    if sep:
        self.named_steps[name].set_params(**{rest: value})
    else:
        self.steps = [(n, value if n == name else s) for n, s in self.steps]
return self
```

</details>
