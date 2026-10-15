# PAC learning · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

check the inputs first, then compute the real number $m^\ast$ from the formula and return `math.ceil(m_star)` as an `int`. for the logarithm of the class size use `math.log(class_size)`: it works on Python integers of any size, while `np.log` converts to a float first and overflows past about $10^{308}$. write $\ln(|\mathcal H|/\delta)$ as `math.log(class_size) + math.log(1 / delta)` so you never divide a huge integer.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
log_h = math.log(class_size)
if realizable:
    m = (log_h + math.log(1 / delta)) / epsilon
else:
    m = (math.log(2) + log_h + math.log(1 / delta)) / (2 * epsilon**2)
return int(math.ceil(m))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the candidates are `np.append(np.unique(x), np.inf)`. for each candidate $c$ you need two counts: positives with $x < c$ and negatives with $x \ge c$. if `pos` is the sorted array of positive $x$ values, `np.searchsorted(pos, c, side="left")` is the number of them strictly below $c$, and it accepts a whole array of candidates at once. `np.argmin` returns the first minimizer, which is the smallest candidate because they're sorted.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
candidates = np.append(np.unique(x), np.inf)
pos, neg = np.sort(x[y]), np.sort(x[~y])
errors = np.searchsorted(pos, candidates, side="left") + (len(neg) - np.searchsorted(neg, candidates, side="left"))
return float(candidates[np.argmin(errors)])
```

(convert `y` to bools first, so `x[y]` and `x[~y]` select positives and negatives.)

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

clip both thresholds into $[0, 1]$ with `min(max(t, 0.0), 1.0)` (this sends `inf` to 1); their distance is the probability that the two classifiers disagree. the failure probability is a one-liner, plus the special case $t^\star + \varepsilon \ge 1$.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
disagree = abs(min(max(t, 0.0), 1.0) - min(max(t_star, 0.0), 1.0))
return float(noise + (1 - 2 * noise) * disagree)
```

and `return 0.0 if t_star + epsilon >= 1 else float((1 - epsilon) ** m)`.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

create the generator once, `gen = as_rng(rng)`, before the loop. in each trial: `x = gen.random(m)`, then `y = (x >= t_star) ^ (gen.random(m) < noise)` (XOR flips the selected labels), then fit and compare the excess risk with $\varepsilon$. with `noise=0` the second draw flips nothing, which is fine.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
best = threshold_risk(t_star, t_star, noise)
failures = 0
for _ in range(n_trials):
    x = gen.random(m)
    y = (x >= t_star) ^ (gen.random(m) < noise)
    failures += threshold_risk(erm_threshold(x, y), t_star, noise) - best > epsilon
return failures / n_trials
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

one generator for the whole scan (`gen = as_rng(rng)`, passed on as `rng=gen`, so successive sample sizes use fresh random numbers). loop over `sorted(int(v) for v in m_grid)` and return as soon as the estimate is at most `delta`. falling off the end of the loop means no grid value worked.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
gen = as_rng(rng)
for m in sorted(int(v) for v in m_grid):
    if estimate_failure_probability(m, epsilon, t_star=t_star, noise=noise, n_trials=n_trials, rng=gen) <= delta:
        return m
return None
```

</details>
