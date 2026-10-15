# what is learning? · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

do the conversions first, then the checks, in the order the docstring lists them. `np.asarray(X, dtype=np.float64)` already raises `ValueError` for things that aren't numbers, so you get that check for free.

</details>

<details><summary>hint 2: the approach</summary>

after converting: reshape a 1-D `X` with `X.reshape(-1, 1)`; then reject `X.ndim != 2`, `X.shape[0] == 0`, and `not np.all(np.isfinite(X))` (that catches NaN and both infinities). for `y`, use `np.asarray(y)` *without* a dtype (strings must survive), `ravel()` a `(n, 1)` column, reject any other `y.ndim != 1` and a length mismatch, and only check finiteness when `y.dtype.kind == "f"`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
X = np.asarray(X, dtype=np.float64)
if X.ndim == 1:
    X = X.reshape(-1, 1)
if X.ndim != 2 or X.shape[0] == 0 or not np.all(np.isfinite(X)):
    raise ValueError("...")            # say which one went wrong
if y is None:
    return X, None
y = np.asarray(y)
if y.ndim == 2 and y.shape[1] == 1:
    y = y.ravel()
# then: y.ndim != 1, len(y) != len(X), float y with NaN/inf -> ValueError
return X, y
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`inspect.signature(type(self).__init__).parameters` is an ordered mapping from parameter name to `Parameter`. each `Parameter` has a `.kind`; skip `self` and the kinds `VAR_POSITIONAL` (`*args`) and `VAR_KEYWORD` (`**kwargs`). a class with no `__init__` of its own inherits `object.__init__`, whose signature is `(self, /, *args, **kwargs)`, so the same filter gives an empty dict.

</details>

<details><summary>hint 2: the approach</summary>

`get_params`: loop over the names, check `hasattr(self, name)` (raise `AttributeError` if not), collect `getattr(self, name)`. `set_params`: get the valid names from `self.get_params()`, then for each `key, value`, split once on `"__"` with `key.partition("__")`. an unknown head raises `ValueError`; a nested key recurses into `getattr(self, head).set_params(**{rest: value})`; a plain key is `setattr`. return `self`.

for `clone`, write a small inner function `copy_value(v)` that handles the four cases, then `type(estimator)(**{k: copy_value(v) for k, v in params.items() if "__" not in k})`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def copy_value(value):
    if hasattr(value, "get_params") and not isinstance(value, type):
        return clone(value)                      # an estimator: recurse
    if isinstance(value, (list, tuple)):
        return type(value)(copy_value(v) for v in value)
    if isinstance(value, dict):
        return {k: copy_value(v) for k, v in value.items()}
    return copy.deepcopy(value)
```

the `isinstance(value, type)` test stops `clone` from trying to "clone" a class that was passed as a hyperparameter (a class also has a `get_params` attribute, but it's unbound).

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

these methods don't know what model they belong to: they only call `self.predict`, `self.predict_proba`, `self.fit` and `self.transform`, which the subclass provides. that's the point of a base class.

</details>

<details><summary>hint 2: the approach</summary>

`Regressor.score`: convert `y` and `self.predict(X)` to 1-D float arrays, compute `ss_res` and `ss_tot`, handle `ss_tot == 0` before dividing. `Classifier.predict`: `np.argmax(proba, axis=1)` gives column indices; index `np.asarray(self.classes_)` with them. `Classifier.score`: `np.mean(self.predict(X) == y)`, as a float. `fit_transform`: one line.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# Regressor.score
y = np.asarray(y, dtype=np.float64).ravel()
y_pred = np.asarray(self.predict(X), dtype=np.float64).ravel()
ss_res, ss_tot = np.sum((y - y_pred) ** 2), np.sum((y - y.mean()) ** 2)
if ss_tot == 0:
    return 1.0 if ss_res == 0 else 0.0
return float(1 - ss_res / ss_tot)

# Classifier.predict
return np.asarray(self.classes_)[np.argmax(np.asarray(self.predict_proba(X)), axis=1)]
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`np.unique(y, return_counts=True)` gives the sorted classes and how often each appears, in the same order: that's `classes_` and (after dividing by the total) `class_prior_` in one call.

</details>

<details><summary>hint 2: the approach</summary>

`predict_proba` builds an `(n, k)` array for each strategy. `most_frequent`: zeros with a column of ones at `np.argmax(self.class_prior_)` (argmax picks the first, i.e. smallest, label on ties). `prior`: `np.tile(self.class_prior_, (n, 1))`. `uniform`: `np.full((n, k), 1 / k)`. `stratified`: draw `rng_.choice(k, size=n, p=class_prior_)` and put a 1 in each row at the drawn column. `predict` overrides only `uniform`; everything else is `super().predict(X)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# DummyClassifier.predict_proba, strategy "stratified"
proba = np.zeros((n, k))
proba[np.arange(n), self.rng_.choice(k, size=n, p=self.class_prior_)] = 1.0

# DummyClassifier.predict
if self.strategy == "uniform":
    return np.asarray(self.classes_)[self.rng_.integers(len(self.classes_), size=len(X))]
return super().predict(X)
```

make `rng_` in `fit` (`self.rng_ = as_rng(self.rng)`), not in `predict`: with an int seed, every call to `as_rng` would restart the same stream.

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

lookup table: numpy rows aren't hashable, `tuple(row)` is. count labels per row in a dict of count vectors, then keep the argmax of each. stump: for one feature, sort the labels by the feature's values once; then "how many positives are among the $i$ smallest values" for every $i$ at once is a cumulative sum.

</details>

<details><summary>hint 2: the approach</summary>

stump, per feature `j`: `order = np.argsort(X[:, j], kind="stable")`, `xs = X[order, j]`, `pos = (label index == 1)[order]`. `pos_left = np.concatenate([[0], np.cumsum(pos)])` (length $n+1$) and `neg_left = np.arange(n + 1) - pos_left`. the valid splits are `i = 0` and every `i` with `xs[i - 1] < xs[i]`: `np.concatenate([[0], np.flatnonzero(xs[1:] > xs[:-1]) + 1])`. the mistakes for sign $+1$ are `pos_left[i] + (total_neg - neg_left[i])`; for sign $-1$, `neg_left[i] + (total_pos - pos_left[i])`. threshold for split `i > 0` is `(xs[i - 1] + xs[i]) / 2`, and `-inf` for `i = 0`.

</details>

<details><summary>hint 3: nearly the answer</summary>

the tie-breaking rule is a lexicographic minimum, so let Python compare tuples for you:

```python
i = int(np.argmin(errs))                    # first = smallest threshold for this (j, sign)
cand = (int(errs[i]), j, float(thresholds[i]), -sign)   # -sign: +1 beats -1 on ties
if best is None or cand < best:
    best = cand
```

after the loops, unpack `best` into `feature_`, `threshold_`, `sign_` and `train_error_ = mistakes / n`. `predict_proba` is `right = sign_ * (X[:, feature_] - threshold_) > 0` stacked as `[~right, right]`.

</details>
