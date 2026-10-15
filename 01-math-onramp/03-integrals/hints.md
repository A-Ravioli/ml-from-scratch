# integrals · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`np.arange(n)` gives the strip numbers $0, 1, \ldots, n-1$ as one array. strip $i$ starts at $a + ih$. if you can write the sample points as one array, `f` evaluates all of them in one call.

</details>

<details><summary>hint 2: the approach</summary>

with `h = (b - a) / n` and `i = np.arange(n)`, the sample points are `a + i * h` (left), `a + (i + 1) * h` (right), or `a + (i + 0.5) * h` (midpoint). use `if`/`elif`/`else` on `rule`, with `raise ValueError(...)` in the `else`. the area is `h * np.sum(f(points))`, wrapped in `float(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
h = (b - a) / n
i = np.arange(n)
if rule == "left":
    points = a + i * h
elif rule == "right":
    points = a + (i + 1) * h
elif rule == "midpoint":
    points = a + (i + 0.5) * h
else:
    raise ValueError(f"unknown rule {rule!r}")
return float(h * np.sum(f(points)))
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

$n$ strips have $n + 1$ edges. `np.linspace(a, b, n + 1)` makes exactly those edges, from $a$ to $b$ inclusive.

</details>

<details><summary>hint 2: the approach</summary>

every grid value counts once, except the two ends, which count half. so take the sum of all the values and subtract half of the first and half of the last: `np.sum(y) - (y[0] + y[-1]) / 2`. multiply by $h$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = np.linspace(a, b, n + 1)
y = f(xs)
h = (b - a) / n
return float(h * (np.sum(y) - (y[0] + y[-1]) / 2))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

write out the indices for $n = 6$: $y_0, \ldots, y_6$, with weights $1, 4, 2, 4, 2, 4, 1$. which indices get a 4? which get a 2? Python slices with a step, like `y[1:6:2]`, pick out every other entry.

</details>

<details><summary>hint 2: the approach</summary>

the weight-4 points are the odd indices $1, 3, \ldots, n-1$: `y[1:-1:2]` (start at 1, stop before the last, step 2). the weight-2 points are the even indices strictly inside, $2, 4, \ldots, n-2$: `y[2:-1:2]`. the ends are `y[0]` and `y[-1]`. check `n % 2 != 0` first and raise `ValueError`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = np.linspace(a, b, n + 1)
y = f(xs)
h = (b - a) / n
odd = np.sum(y[1:-1:2])
even = np.sum(y[2:-1:2])
return float(h / 3 * (y[0] + 4 * odd + 2 * even + y[-1]))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

the area of trapezoid $k$ uses the values at its two ends, $y_{k-1}$ and $y_k$. `y[:-1]` is every left end and `y[1:]` is every right end, so you can compute all $n$ trapezoids at once.

</details>

<details><summary>hint 2: the approach</summary>

`strips = h * (y[:-1] + y[1:]) / 2` has $n$ entries. `np.cumsum(strips)` gives the running totals after 1, 2, …, $n$ strips. the running area at the very first grid point is 0, so put a 0 in front with `np.concatenate([[0.0], ...])`. now `F` has $n + 1$ entries, one per grid point.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = np.linspace(a, b, n + 1)
y = f(xs)
h = (b - a) / n
strips = h * (y[:-1] + y[1:]) / 2
F = np.concatenate([[0.0], np.cumsum(strips)])
return xs, F
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

change variables: $u_i = \log n_i$, $v_i = \log \text{error}_i$. now it's a plain "fit a line through points" problem, and the answer you want is minus the line's slope.

</details>

<details><summary>hint 2: the approach</summary>

subtract the averages first: `du = u - u.mean()` and `dv = v - v.mean()`. the slope is `np.sum(du * dv) / np.sum(du**2)`. return `float(-slope)`. any log base works (the ratio doesn't depend on it), but `np.log` is simplest.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
u = np.log(np.asarray(ns, dtype=float))
v = np.log(np.asarray(errors, dtype=float))
du = u - u.mean()
slope = np.sum(du * (v - v.mean())) / np.sum(du**2)
return float(-slope)
```

</details>
