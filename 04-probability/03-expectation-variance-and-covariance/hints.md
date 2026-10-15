# expectation, variance & covariance · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

LOTUS is a weighted sum: apply `f` to the *values* (not to the mean), multiply by the probabilities, add up. `f=None` means the identity.

</details>

<details><summary>hint 2: the approach</summary>

`expectation`: `fx = values if f is None else f(values)`, then `np.sum(fx * probs)`. `mc_estimate`: the mean is `x.mean()`. the sample standard deviation is `sqrt(sum((x - mean)**2) / (n - 1))`, and the standard error divides that by `sqrt(n)`. return Python floats.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(samples, dtype=float)
n = len(x)
mean = x.mean()
s = math.sqrt(np.sum((x - mean) ** 2) / (n - 1))
return float(mean), s / math.sqrt(n)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`naive_variance` is one line, and it's *supposed* to be the bad formula: `np.mean(x**2) - np.mean(x)**2`. resist the urge to fix it; the test checks that it fails on offset data.

</details>

<details><summary>hint 2: the approach</summary>

`two_pass_variance`: pass 1 computes `x.mean()`. pass 2 computes the deviations `d = x - mean`, then `np.sum(d * d) / (n - ddof)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
d = x - x.mean()
return float(np.sum(d * d) / (len(x) - ddof))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

reduce everything to the batch case. in `update`, if `x.shape == self.shape` it's one sample: turn it into a batch of one with `x[None]`. if `x.shape[1:] != self.shape` it's neither, so raise `ValueError`. then compute the batch's own `n_b`, `mean_b` and `m2_b` with two passes along axis 0, and merge.

</details>

<details><summary>hint 2: the approach</summary>

the merge, with the *old* count `self.count`:

```
n     = self.count + n_b
delta = mean_b - self.mean
mean  = self.mean + delta * n_b / n
m2    = self.m2 + m2_b + delta**2 * self.count * n_b / n
```

update `self.count` last. an empty batch (`n_b == 0`) should change nothing (and would divide by zero if `self.count` is 0 too).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
if x.shape == self.shape:
    x = x[None]
elif x.shape[1:] != self.shape:
    raise ValueError(f"bad shape {x.shape}")
n_b = x.shape[0]
if n_b == 0:
    return self
mean_b = x.mean(axis=0)
m2_b = np.sum((x - mean_b) ** 2, axis=0)
n = self.count + n_b
delta = mean_b - self.mean
self.mean = self.mean + delta * (n_b / n)
self.m2 = self.m2 + m2_b + delta**2 * (self.count * n_b / n)
self.count = n
return self
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

center each *column*: `X.mean(axis=0)` is the vector of column means, and `X - X.mean(axis=0)` broadcasts it across the rows. then one matrix product gives every pairwise sum of products at once.

</details>

<details><summary>hint 2: the approach</summary>

`Xc.T @ Xc` is `(d, d)`, with entry `(i, j)` equal to `sum_k Xc[k, i] * Xc[k, j]`. divide by `n - ddof`. turn a 1-D input into a column first with `X[:, None]`. for `correlation`, take `sd = np.sqrt(np.diag(C))` and divide `C` by `np.outer(sd, sd)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
X = np.asarray(X, dtype=float)
if X.ndim == 1:
    X = X[:, None]
Xc = X - X.mean(axis=0)
return Xc.T @ Xc / (X.shape[0] - ddof)
```

</details>
