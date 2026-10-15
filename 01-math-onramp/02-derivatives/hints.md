# derivatives · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

sketch two points on a curve, $(a, f(a))$ and $(b, f(b))$. the rise is the change in height, the run is the change in $x$. both must be measured in the same direction, from $a$ to $b$.

</details>

<details><summary>hint 2: the approach</summary>

rise is `f(b) - f(a)`, run is `b - a`. divide. NumPy arithmetic works the same on floats and arrays, so you don't need a loop.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
return (f(b) - f(a)) / (b - a)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

each method is a secant slope through two points. which two? forward looks ahead ($x$ and $x + h$), backward looks behind ($x - h$ and $x$), central looks both ways ($x - h$ and $x + h$).

</details>

<details><summary>hint 2: the approach</summary>

after `x = np.asarray(x, dtype=float)`, write three `if method == ...:` branches, each returning `secant_slope(f, left_point, right_point)`. after them, `raise ValueError(...)`: you only get there if no branch matched. because `secant_slope` divides by the actual distance between the points, the central branch automatically divides by $2h$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if method == "forward":
    return secant_slope(f, x, x + h)
if method == "backward":
    return secant_slope(f, x - h, x)
if method == "central":
    return secant_slope(f, x - h, x + h)
raise ValueError(f"unknown method {method!r}")
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

both functions work like `compose` from topic 01: define a function inside, and return it. the inner function just writes out the rule from the lesson, using the four functions you were given.

</details>

<details><summary>hint 2: the approach</summary>

product rule: $(fg)' = f'g + fg'$, so the inner function returns `df(x) * g(x) + f(x) * dg(x)`. chain rule: $(f(g(x)))' = f'(g(x))\,g'(x)$. compute the inner value `g(x)` first, feed it to `df`, and multiply by `dg(x)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def dc(x):
    return df(g(x)) * dg(x)

return dc
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

your `derivative` already handles a whole array of points at once, and so does `df` (it's vectorized). so you can compute all the errors in one go and take the maximum.

</details>

<details><summary>hint 2: the approach</summary>

`fd = derivative(f, xs, h=h, method="central")` and `analytic = np.asarray(df(xs), dtype=float)`. the error array is `np.abs(analytic - fd) / np.maximum(1.0, np.abs(fd))`. careful: `np.maximum` (elementwise, two arrays) is not `np.max` (the largest entry of one array). you need both: `np.maximum` for the denominators, `np.max` for the final answer.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = np.asarray(xs, dtype=float)
fd = derivative(f, xs, h=h, method="central")
analytic = np.asarray(df(xs), dtype=float)
err = np.abs(analytic - fd) / np.maximum(1.0, np.abs(fd))
return float(np.max(err))
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

it's one line of arithmetic. three function values: one step right, the middle (counted twice, with a minus sign), and one step left.

</details>

<details><summary>hint 2: the approach</summary>

the forward slope at $x$ is $(f(x + h) - f(x))/h$ and the backward slope is $(f(x) - f(x - h))/h$. subtract the backward slope from the forward slope and divide by $h$ once more. simplify, and you get one fraction over $h^2$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
return (f(x + h) - 2 * f(x) + f(x - h)) / h**2
```

</details>
