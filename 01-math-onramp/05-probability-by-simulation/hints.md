# probability by simulation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

try `list(itertools.product(range(1, 7), repeat=2))[:8]` in a cell: it lists outcomes as tuples, `(1, 1), (1, 2), …`. for each one, which sum does it add to, and where should that count go?

</details>

<details><summary>hint 2: the approach</summary>

`dice_sum_pmf`: `sums = np.arange(n_dice, 6 * n_dice + 1)` and `counts = np.zeros(len(sums))`. the smallest sum, `n_dice`, sits at index 0, so a sum `s` sits at index `s - n_dice`. loop over the outcomes, add 1 at that index, and finally divide by the number of outcomes, `6 ** n_dice`.

`roll_dice`: `rng = np.random.default_rng(rng)`, then one call to `rng.integers` with `size=(n, n_dice)`. remember the upper bound is excluded.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
sums = np.arange(n_dice, 6 * n_dice + 1)
counts = np.zeros(len(sums))
for outcome in itertools.product(range(1, 7), repeat=n_dice):
    counts[sum(outcome) - n_dice] += 1
return sums, counts / 6**n_dice
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`samples == 3` is an array of `True`/`False`, and `np.sum` of it counts the `True`s. expectation and variance are both "multiply by the probabilities, then add".

</details>

<details><summary>hint 2: the approach</summary>

`empirical_pmf`: a list comprehension `[np.sum(samples == v) for v in values]`, turned into a float array and divided by `len(samples)`. `expectation`: `np.sum(values * probs)`. `variance`: compute `mu = expectation(values, probs)`, then `np.sum(probs * (values - mu) ** 2)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
values = np.asarray(values, dtype=float)
mu = expectation(values, probs)
return float(np.sum(np.asarray(probs, dtype=float) * (values - mu) ** 2))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

two numbers: the plain average, and (sample standard deviation) divided by (square root of how many samples). NumPy's `np.std` has a `ddof` argument that changes the $n$ in its denominator to $n - \text{ddof}$.

</details>

<details><summary>hint 2: the approach</summary>

`mean = x.mean()`, `s = np.std(x, ddof=1)`, `se = s / np.sqrt(len(x))`. return both as plain floats.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
n = len(x)
return float(x.mean()), float(np.std(x, ddof=1) / np.sqrt(n))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

it's three lines, and you've written the hard part already. `sample` and `f` are functions you were handed: call them.

</details>

<details><summary>hint 2: the approach</summary>

turn `rng` into a generator (`np.random.default_rng(rng)` handles None, a seed, or a generator, and returns a generator unchanged), draw `sample(n, rng)`, apply `f` to the draws, and hand the result to your `mean_and_se`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rng = np.random.default_rng(rng)
draws = sample(n, rng)
return mean_and_se(f(draws))
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

you need two small functions to hand to `mc_estimate`: one that throws `n` darts (rows of $(x, y)$), and one that scores each dart 4 or 0.

</details>

<details><summary>hint 2: the approach</summary>

define `sample(m, gen)` returning `gen.uniform(-1.0, 1.0, size=(m, 2))`, and `f(points)` returning `4.0 * (points[:, 0]**2 + points[:, 1]**2 <= 1)` (multiplying a boolean array by 4.0 turns `True` into 4.0 and `False` into 0.0). then `return mc_estimate(f, sample, n, rng=rng)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def sample(m, gen):
    return gen.uniform(-1.0, 1.0, size=(m, 2))

def f(points):
    return 4.0 * (points[:, 0] ** 2 + points[:, 1] ** 2 <= 1.0)

return mc_estimate(f, sample, n, rng=rng)
```

</details>
