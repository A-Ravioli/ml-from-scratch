# bias–variance & complexity · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`check_X_y` already turns a 1-D `x` into a column, so after it `X[:, 0]` is the feature. the whole fit is two lines: build the design matrix, solve least squares.

</details>

<details><summary>hint 2: the approach</summary>

`np.vander(x, degree + 1, increasing=True)` gives the columns $1, x, \dots, x^d$ (without `increasing=True` they come out highest power first). `np.linalg.lstsq(A, y, rcond=None)` returns a tuple whose first element is the solution. `predict` builds the same matrix for the new inputs and multiplies by `coef_`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
X, y = check_X_y(X, y)
if X.shape[1] != 1:
    raise ValueError("one feature only")
A = np.vander(X[:, 0], self.degree + 1, increasing=True)
self.coef_ = np.linalg.lstsq(A, y.astype(float), rcond=None)[0]
return self
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

collect the predictions in an array of shape `(n_rounds, m)`, one row per round. once you have it, bias² and variance are a mean and a variance along axis 0, followed by a mean over the test points.

</details>

<details><summary>hint 2: the approach</summary>

make one generator, `gen = as_rng(rng)`, *outside* the loop and pass it to `sample_data` every round (a new `as_rng(rng)` each round would restart an int seed and draw the same training set every time). in the loop: `X, y = sample_data(gen)` and `preds[r] = clone(estimator).fit(X, y).predict(X_test)`. then `mean = preds.mean(axis=0)`, `bias2 = np.mean((mean - f_test) ** 2)`, `variance = np.mean(preds.var(axis=0))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
return {"bias2": float(bias2), "variance": float(variance), "noise": float(noise_var),
        "expected_error": float(bias2 + variance + noise_var)}
```

a sanity check you can run yourself: `np.mean((preds - f_test) ** 2)` should equal `bias2 + variance` to rounding error.

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`splits = materialize_splits(cv, X, y)` gives a list of `(train, test)` pairs. the sizes come from the smallest training fold: `n_max = min(len(tr) for tr, _ in splits)`.

</details>

<details><summary>hint 2: the approach</summary>

convert each fraction to `max(1, int(f * n_max))` (and reject fractions outside $(0, 1]$). allocate two arrays of shape `(len(sizes), len(splits))`. for every fold `j` and size `i`, call your `cross_validate` with a one-split cv, `cv=[(tr[:size], te)]`, and `return_train_score=True`: its `"train_score"[0]` and `"test_score"[0]` are the two entries.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for j, (tr, te) in enumerate(splits):
    for i, size in enumerate(sizes):
        out = cross_validate(estimator, X, y, cv=[(tr[:size], te)], scoring=scoring, return_train_score=True)
        train_scores[i, j] = out["train_score"][0]
        val_scores[i, j] = out["test_score"][0]
return sizes, train_scores, val_scores
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

it's stage 3's loop turned sideways: instead of varying the training size, vary one hyperparameter, and let `cross_validate` loop over the folds.

</details>

<details><summary>hint 2: the approach</summary>

materialize the splits **once**, before the loop. for each value: `est = clone(estimator).set_params(**{param_name: value})` (the `**{...}` turns a name held in a variable into a keyword argument), then `cross_validate(est, X, y, cv=splits, scoring=scoring, return_train_score=True)`, and append its two score arrays.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
splits = materialize_splits(cv, X, y)
train_scores, val_scores = [], []
for value in param_range:
    est = clone(estimator).set_params(**{param_name: value})
    out = cross_validate(est, X, y, cv=splits, scoring=scoring, return_train_score=True)
    train_scores.append(out["train_score"])
    val_scores.append(out["test_score"])
return np.array(train_scores), np.array(val_scores)
```

</details>
