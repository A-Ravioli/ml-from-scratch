# limit theorems · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

running means are running sums divided by how many terms each sum has. NumPy has a function for running sums, and `np.arange(1, n + 1)` gives the counts.

</details>

<details><summary>hint 2: the approach</summary>

`running_mean`: `np.cumsum(x) / np.arange(1, len(x) + 1)`. `sample_means`: normalize the generator, draw all the data at once with `sampler((n_reps, n), rng)`, which gives one row per repetition, and average each row with `.mean(axis=1)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def sample_means(sampler, n, n_reps, *, rng=None):
    rng = np.random.default_rng(rng)
    x = np.asarray(sampler((n_reps, n), rng), dtype=float)
    return x.mean(axis=1)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`math.erf` takes one number. `np.vectorize(math.erf)` turns it into a function that works elementwise on arrays. for `ks_distance`, sort the sample first: everything is about the staircase's height at the sorted points.

</details>

<details><summary>hint 2: the approach</summary>

with `z` sorted and `F = normal_cdf(z)`, and `i = np.arange(1, n + 1)`: the gaps just after each jump are `i / n - F`, the gaps just before are `F - (i - 1) / n`. the distance is the largest of all of them.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
z = np.sort(np.asarray(z, dtype=float))
n = len(z)
F = normal_cdf(z)
i = np.arange(1, n + 1)
return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`clt_distances` is a loop over `ns` that reuses your earlier functions: `sample_means` for the means, then standardize, then `ks_distance`. normalize `rng` once, before the loop, and pass the same generator to every `sample_means` call.

</details>

<details><summary>hint 2: the approach</summary>

the standardization is `z = math.sqrt(n) * (m - mu) / sigma`, where `m` is the array of `n_reps` means. the Berry–Esseen bound is a one-liner; use `np.sqrt` so that `n` can be an array.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = np.random.default_rng(rng)
out = []
for n in ns:
    m = sample_means(sampler, int(n), n_reps, rng=rng)
    out.append(ks_distance(math.sqrt(n) * (m - mu) / sigma))
return np.array(out)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

same loop shape as `clt_distances`, but instead of a KS distance, record the fraction of means that land more than `eps` away from `mu`.

</details>

<details><summary>hint 2: the approach</summary>

`np.mean(np.abs(m - mu) > eps)` is the fraction of `True`s: the Monte Carlo estimate of the probability. use `>`, not `>=`: with coin flips, the mean can land exactly on `mu ± eps`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = np.random.default_rng(rng)
out = []
for n in ns:
    m = sample_means(sampler, int(n), n_reps, rng=rng)
    out.append(np.mean(np.abs(m - mu) > eps))
return np.array(out, dtype=float)
```

</details>
