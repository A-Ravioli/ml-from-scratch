# functions, exponents & logs · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

in Python, a function defined inside another function can use the outer function's arguments. so inside `compose(f, g)` you can write `def h(x): ...` and use `f` and `g` in its body. then `return h`, without parentheses.

for `bisect_inverse`, draw it: an increasing curve, a horizontal line at height $y$, and a bracket $[lo, hi]$ on the $x$-axis. where is the crossing if the curve at the midpoint is below the line?

</details>

<details><summary>hint 2: the approach</summary>

`compose`: the inner function is one line, `return f(g(x))`.

`bisect_inverse`: first `if y < f(lo) or y > f(hi): raise ValueError("...")`. then a `for _ in range(200):` loop. at the top of each pass, `break` if `hi - lo <= tol`. otherwise compute the midpoint and move *one* end of the bracket to it: `lo` if $f(mid) < y$, `hi` otherwise. after the loop, return `float((lo + hi) / 2)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for _ in range(200):
    if hi - lo <= tol:
        break
    mid = (lo + hi) / 2
    if f(mid) < y:
        lo = mid
    else:
        hi = mid
return float((lo + hi) / 2)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

write out the terms for $x = 2$: $1,\ 2,\ 2,\ \frac43,\ \frac23, \ldots$. how do you get each term from the one before it? (divide the second by the first, the third by the second, …)

</details>

<details><summary>hint 2: the approach</summary>

term $k$ is term $k-1$ times $x/k$. keep two variables, `term` (starting at 1) and `total` (starting at 1, the $k = 0$ term). loop `for k in range(1, n_terms):`, which runs $n\_terms - 1$ times, adding one new term each time. that makes $n\_terms$ terms in total. if `x` is an array, `np.ones_like(x)` makes a starting `term` of the right shape, and all the arithmetic works entry by entry.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
term = np.ones_like(x)
total = term.copy()
for k in range(1, n_terms):
    term = term * x / k
    total = total + term
return total
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

$e^{2.3} = e^2 \cdot e^{0.3}$. which of the two factors is easy for the series? how can you compute the other one with a single `**`?

</details>

<details><summary>hint 2: the approach</summary>

`k = np.round(x)` gives the nearest whole number to every entry. `r = x - k` is then between $-\frac12$ and $\frac12$. compute `E = exp_series(1.0, 20)` (that's $e$) and return `E**k * exp_series(r, 20)`. everything works on arrays without a loop.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
k = np.round(x)
r = x - k
E = exp_series(1.0, 20)
return E**k * exp_series(r, 20)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

this is a loop that updates one number and remembers every value it takes. the first value in the result is `x0` itself, before any step.

</details>

<details><summary>hint 2: the approach</summary>

`xs = [x0]`, `x = x0`, then `for _ in range(n_steps):` update `x = x - 1 + y * exp(-x)` and `xs.append(x)`. return `np.array(xs, dtype=float)`. with `n_steps = 0` the loop doesn't run and you return `[x0]`. (if `exp` hands back a 0-d array, wrapping the update in `float(...)` keeps `x` a plain number.)

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = [float(x0)]
x = float(x0)
for _ in range(n_steps):
    x = float(x - 1 + y * exp(-x))
    xs.append(x)
return np.array(xs, dtype=float)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

try `np.frexp(1000.0)` in a cell. it gives `(0.9765625, 10)`, because $1000 = 0.9765625 \times 2^{10}$. so $\log 1000 = \log 0.9765625 + 10 \log 2$. which of those two logs is easy for Newton? (frexp is a shortcut, not magic: halving 1000 ten times gives 0.9765625, with no rounding at all.)

</details>

<details><summary>hint 2: the approach</summary>

check `np.any(y <= 0)` first and raise `ValueError`. then `m, p = np.frexp(y)`. run the Newton update on the whole array `m` at once, starting from `x = m - 1`: `x = x - 1 + m * exp(-x)`, six times (your `exp` works on arrays). get $\log 2$ as the last entry of `newton_log_steps(2.0, 0.7, 6)`. return `x + p * ln2`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
m, p = np.frexp(y)
x = m - 1
for _ in range(6):
    x = x - 1 + m * exp(-x)
ln2 = newton_log_steps(2.0, 0.7, 6)[-1]
return x + p * ln2
```

</details>
