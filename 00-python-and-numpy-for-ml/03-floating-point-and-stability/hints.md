# floating point & stability · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

for `decompose`, think of what you'd do by hand with scientific notation: if the number is 10 or more, divide by 10 and add one to the exponent; if it's less than 1, multiply by 10 and subtract one. do the same in base 2. for `machine_epsilon`, the whole difficulty is making sure the arithmetic really happens in the requested dtype.

</details>

<details><summary>hint 2: the approach</summary>

`decompose`: after the checks, set `sign = 1 if x < 0 else 0`, `m = abs(x)`, `exponent = 0`. then `while m >= 2: m /= 2; exponent += 1` and `while m < 1: m *= 2; exponent -= 1`. only one of the two loops will do anything.

`machine_epsilon`: `dtype(1)` makes a NumPy scalar of that type (`np.float32(1)`), and arithmetic between two `float32` scalars stays `float32`. if you write `1 + eps / 2` with Python floats anywhere, the computation silently becomes `float64`. so make `one`, `two` and `eps` all `dtype(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
one, two, eps = dtype(1), dtype(2), dtype(1)
while one + eps / two != one:
    eps = eps / two
return float(eps)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the two numerators of the school formula are $-b + \sqrt{d}$ and $-b - \sqrt{d}$. for $b > 0$, which one subtracts two nearly equal numbers? compute only the other one with the formula, then get the second root from the product of the roots, $c/a$.

</details>

<details><summary>hint 2: the approach</summary>

after the checks, `q = -(b + math.copysign(math.sqrt(disc), b)) / 2`. `copysign(r, b)` is `r` with the sign of `b`, so `b` and the root always have the same sign and add up without cancelling. the roots are `q / a` and `c / q`. if `q == 0` (only possible when `b == 0` and `c == 0`), both roots are 0. return `(min(...), max(...))`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
q = -(b + math.copysign(math.sqrt(disc), b)) / 2.0
if q == 0:
    return 0.0, 0.0
r1, r2 = q / a, c / q
return (min(r1, r2), max(r1, r2))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

never call `np.exp` on something that can be large and positive. $e^{-|x|}$ is always between 0 and 1. can you write the sigmoid and the softplus using only that?

</details>

<details><summary>hint 2: the approach</summary>

- `sigmoid`: `e = np.exp(-np.abs(x))`; for `x >= 0` the answer is `1 / (1 + e)`, for `x < 0` it's `e / (1 + e)` (idea 3 shows why both are the sigmoid). combine with `np.where(x >= 0, ..., ...)`; both branches only use `e`, so neither overflows.
- `softplus`: `np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))`.
- `log_sigmoid`: `-softplus(-x)`.

convert the input with `np.asarray(x, dtype=float)` first, so plain numbers and integer arrays work too.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
e = np.exp(-np.abs(x))
return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$ with $m$ the maximum along the axis. compute `m` with `keepdims=True` so that `x - m` broadcasts, whatever the axis.

</details>

<details><summary>hint 2: the approach</summary>

1. `m = np.max(x, axis=axis, keepdims=True)`, then `m = np.where(np.isfinite(m), m, 0.0)` (all-`-inf` slices would otherwise give `-inf - (-inf) = nan`).
2. `s = np.sum(np.exp(x - m), axis=axis, keepdims=True)`.
3. `out = np.log(s) + m`, inside `with np.errstate(divide="ignore"):` (log of 0 is the correct `-inf`).
4. if not `keepdims`, `out = np.squeeze(out, axis=axis)` (with `axis=None` this drops every axis).

</details>

<details><summary>hint 3: nearly the answer</summary>

the last line, `return out[()]`, turns a 0-dimensional array into a plain NumPy number and leaves bigger arrays unchanged, so `logsumexp(x)` returns a single float like SciPy's does.

```python
with np.errstate(divide="ignore"):
    out = np.log(s) + m
if not keepdims:
    out = np.squeeze(out, axis=axis)
return out[()]
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

both functions are two or three lines. softmax: shift, exponentiate, normalize, all along `axis` with `keepdims=True`. log-softmax: one subtraction using your `logsumexp`.

</details>

<details><summary>hint 2: the approach</summary>

`z = x - np.max(x, axis=axis, keepdims=True)`, `e = np.exp(z)`, return `e / np.sum(e, axis=axis, keepdims=True)`. for log-softmax, return `x - logsumexp(x, axis=axis, keepdims=True)`. don't take `np.log` of a softmax.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
e = np.exp(x - np.max(x, axis=axis, keepdims=True))
return e / np.sum(e, axis=axis, keepdims=True)
```

</details>
