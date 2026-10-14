# hyperparameter search · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`itertools.product([1, 2], ["x", "y"])` gives `(1, "x"), (1, "y"), (2, "x"), (2, "y")`: the last list already varies fastest. you just need the lists in sorted-key order, and to zip each tuple back with the keys.

</details>

<details><summary>hint 2: the approach</summary>

`__iter__` is a generator: loop over `self.grids()`, sort each dict's keys, and `yield dict(zip(keys, combo))` for every `combo` in `itertools.product(*(grid[k] for k in keys))`. an empty dict gives `product()` with no arguments, which yields one empty tuple, so you get `{}` for free.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def __len__(self):
    return sum(int(np.prod([len(v) for v in grid.values()])) for grid in self.grids())
```

`np.prod([])` is `1.0`, which handles the empty dict.

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

normalize `self.cv` to a *list* of `(train, test)` pairs first (an int becomes `KFold(k)`; a splitter's `split(X, y)` becomes a list; anything else is already pairs). then every candidate gets `cv=splits`.

</details>

<details><summary>hint 2: the approach</summary>

for each `params` in `self.candidate_params()`: `est = clone(self.estimator).set_params(**params)`, `scores = cross_validate(est, X, y, cv=splits, scoring=self.scoring)["test_score"]`, and append `params`, `np.mean(scores)`, `np.std(scores)` to the three lists of `cv_results_`. then `np.argmax` of the means (first on ties), and the refit if `self.refit`. `predict` and `score` go through the given `self.best()`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
self.best_index_ = int(np.argmax(results["mean_test_score"]))
self.best_params_ = results["params"][self.best_index_]
self.best_score_ = results["mean_test_score"][self.best_index_]
if self.refit:
    self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
    if hasattr(self.best_estimator_, "classes_"):
        self.classes_ = self.best_estimator_.classes_
return self
```

and `GridSearchCV.candidate_params` is `list(ParameterGrid(self.param_grid))`.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

make the generator once, before the loop: `gen = as_rng(self.rng)`. (an int seed passed to `as_rng` inside the loop would restart the same stream every time, and all candidates would be identical.)

</details>

<details><summary>hint 2: the approach</summary>

for each of `n_iter` candidates, build a dict over `sorted(self.param_distributions)`: `dist(gen)` if `callable(dist)`, else `dist[int(gen.integers(len(dist)))]`. sorting the names makes the stream of random numbers, and so the candidates, independent of how the dict was written.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
out = []
for _ in range(self.n_iter):
    params = {}
    for name in sorted(dists):
        dist = dists[name]
        params[name] = dist(gen) if callable(dist) else dist[int(gen.integers(len(dist)))]
    out.append(params)
return out
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

a search is an estimator. what does `cross_validate` do with an estimator?

</details>

<details><summary>hint 2: the approach</summary>

`cross_validate` clones the search for each outer fold, calls its `fit` on the outer training rows (which runs the inner CV and the refit), and scores it on the outer test rows with `scoring` (or `search.score`). that's nested cross-validation.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
return cross_validate(search, X, y, cv=cv, scoring=scoring)["test_score"]
```

if this fails inside `clone`, check that your search's `__init__` stores every argument unchanged (topic 01).

</details>
