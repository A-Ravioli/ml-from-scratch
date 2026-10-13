# taylor approximation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

you need the terms $(x - x_0)^k/k!$ for $k = 0, 1, \dots, K$. each one is the previous one times $(x - x_0)/k$, so you never need `math.factorial` or a large power.

</details>

<details><summary>hint 2: the approach</summary>

convert `x` with `np.asarray(x, dtype=float)` and set `d = x - x0`. start with `term = np.ones_like(d)` and `total = np.zeros_like(d)`. loop over `enumerate(derivs)`: for `k > 0` update `term = term * d / k`, then add `c * term` to the total. return a float when `x` was a float.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
for k, c in enumerate(derivs):
    if k > 0:
        term = term * d / k
    total = total + c * term
return float(total) if total.ndim == 0 else total
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the bound for a degree-`order` polynomial uses the *next* derivative: power `order + 1`, factorial `(order + 1)!`. the slope is an ordinary least-squares line fit in log-log coordinates.

</details>

<details><summary>hint 2: the approach</summary>

`lagrange_bound`: `M * np.abs(x - x0) ** (order + 1) / math.factorial(order + 1)`. `loglog_slope`: with `u = log(hs)` and `v = log(errors)`, the least-squares slope is $\sum(u - \bar u)(v - \bar v)/\sum(u - \bar u)^2$. don't just use the two endpoints.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
u = np.log(np.asarray(hs, dtype=float))
v = np.log(np.asarray(errors, dtype=float))
uc = u - u.mean()
return float(uc @ (v - v.mean()) / (uc @ uc))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`quadratic_model` returns a function. define `m(x)` inside it, using the `x0`, `f0`, `g`, `H` from the enclosing call (a closure), and return `m`.

</details>

<details><summary>hint 2: the approach</summary>

inside `m`, let `d = x - x0`. the linear term is `d @ g`, which works for `d` of shape `(n,)` (a number) or `(k, n)` (a length-$k$ array). the quadratic term needs $d_i^\top Hd_i$ for each row: `np.einsum("...i,ij,...j->...", d, H, d)` handles both shapes.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
def m(x):
    d = np.asarray(x, dtype=float) - x0
    out = f0 + d @ g + 0.5 * np.einsum("...i,ij,...j->...", d, H, d)
    return float(out) if np.ndim(out) == 0 else out
return m
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

this is `quadratic_model` with ingredients you compute yourself. for lower orders, just pass zeros for the parts you don't include.

</details>

<details><summary>hint 2: the approach</summary>

check `order in (0, 1, 2)` and raise `ValueError` otherwise. `g = numerical_gradient(f, x0)` if `order >= 1` else `np.zeros(n)`; `H = numerical_hessian(f, x0)` if `order == 2` else `np.zeros((n, n))`. return `quadratic_model(x0, float(f(x0)), g, H)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x0 = np.asarray(x0, dtype=float)
n = x0.size
g = numerical_gradient(f, x0) if order >= 1 else np.zeros(n)
H = numerical_hessian(f, x0) if order == 2 else np.zeros((n, n))
return quadratic_model(x0, float(f(x0)), g, H)
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`agreement_ratio` is one line: actual reduction over predicted reduction. `trust_radius` is a loop over the radii that remembers the last good one and stops at the first bad one.

</details>

<details><summary>hint 2: the approach</summary>

normalize the direction, `u = direction / norm(direction)`. set `best = 0.0`. for each `r` in `radii`: if `abs(agreement_ratio(f, model, x0, x0 + r * u) - 1) > tol`, `break`; otherwise `best = r`. return `best`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
u = np.asarray(direction, dtype=float)
u = u / np.linalg.norm(u)
best = 0.0
for r in radii:
    if abs(agreement_ratio(f, model, x0, x0 + r * u) - 1.0) > tol:
        break
    best = float(r)
return best
```

</details>
