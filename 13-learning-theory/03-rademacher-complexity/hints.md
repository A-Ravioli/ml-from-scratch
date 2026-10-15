# rademacher complexity · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

make `P = np.atleast_2d(...)` so one hypothesis becomes a `(1, n)` matrix. a `(draws, n)` matrix of signs `S` times `P.T` gives every correlation (times $n$) at once, shape `(draws, hypotheses)`. take the max over hypotheses (axis 1), average over draws, divide by $n$. if memory worries you, process the draws in chunks of a few hundred.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
gen = as_rng(rng)
sigma = 2.0 * gen.integers(0, 2, size=(n_draws, n)) - 1.0
return float(np.mean(np.max(sigma @ P.T, axis=1)) / n)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

deduplicate first with `np.unique(P, axis=0)`; then $N$ is the number of rows left and $r$ is the largest row norm (`np.linalg.norm(P, axis=1).max()`). `math.log(1) = 0` handles the single-hypothesis case for free.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
P = np.unique(np.atleast_2d(np.asarray(predictions, dtype=float)), axis=0)
return float(np.linalg.norm(P, axis=1).max() * math.sqrt(2 * math.log(len(P))) / P.shape[1])
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

map the constraint norm to its dual with a small dict, `{1: np.inf, 2: 2, np.inf: 1}`. one matrix product `sigma @ X` gives $v = \sum_i\sigma_ix_i$ for every draw, shape `(draws, d)`; `np.linalg.norm(v, ord=q, axis=1)` gives one norm per draw. the Jensen bound is `B * sqrt((X**2).sum()) / n`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
dual = {1: np.inf, 2: 2, np.inf: 1}[norm]
sigma = 2.0 * as_rng(rng).integers(0, 2, size=(n_draws, n)) - 1.0
return float(B * np.mean(np.linalg.norm(sigma @ X, ord=dual, axis=1)) / n)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

a one-liner: `emp_risk + 2 * rademacher + 3 * math.sqrt(math.log(2 / delta) / (2 * n))`. the input `rademacher` is already the loss class's complexity; the halving for 0-1 loss is the caller's job.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
return float(emp_risk + 2 * rademacher + 3 * math.sqrt(math.log(2 / delta) / (2 * n)))
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

one generator outside the loop. per draw: `sigma = 2.0 * gen.integers(0, 2, size=n) - 1.0`, `h = fit_predict(X, sigma)`, then `np.mean(sigma * h)`. pass `X` exactly as you received it.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
corrs = []
for _ in range(n_draws):
    sigma = 2.0 * gen.integers(0, 2, size=len(X)) - 1.0
    h = np.asarray(fit_predict(X, sigma), dtype=float).ravel()
    corrs.append(np.mean(sigma * h))
return float(np.mean(corrs))
```

</details>
