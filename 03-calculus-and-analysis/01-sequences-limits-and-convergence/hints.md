# sequences, limits & convergence · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

think about it from the end of the array. $N$ has to come after every term that is *outside* the band, so the interesting index is the last bad one, not the first good one.

</details>

<details><summary>hint 2: the approach</summary>

build a boolean mask `inside = np.abs(a - limit) < eps` (strict). find the indices where it's `False` with `np.flatnonzero(~inside)`. if there are none, $N = 0$. otherwise $N$ is the last such index plus one (which is $K$ when the last term itself is outside).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
a = np.asarray(a, dtype=float)
outside = np.flatnonzero(~(np.abs(a - limit) < eps))
return 0 if outside.size == 0 else int(outside[-1] + 1)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the largest $|a_m - a_n|$ over a set of numbers is just its maximum minus its minimum. so you need $\max(a_{N:})$ and $\min(a_{N:})$ for every $N$, the running max and min of the array read from the right.

</details>

<details><summary>hint 2: the approach</summary>

reverse the array, `rev = a[::-1]`. then `np.maximum.accumulate(rev)[j]` is the max of the last $j + 1$ terms. do the same with `np.minimum.accumulate`, subtract, and reverse the result so that entry $N$ corresponds to the tail starting at $N$.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
rev = np.asarray(a, dtype=float)[::-1]
return (np.maximum.accumulate(rev) - np.minimum.accumulate(rev))[::-1]
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

both estimators only need the consecutive ratios $e_{k+1}/e_k$. with NumPy slicing, `e[1:] / e[:-1]` gives all of them at once.

</details>

<details><summary>hint 2: the approach</summary>

let `r = np.log(e[1:] / e[:-1])` (length $n - 1$). the order estimate $q_k$ is the *next* log-ratio divided by the current one: `r[1:] / r[:-1]` (length $n - 2$). check for non-positive errors first and raise `ValueError`. for the rates, raise each old error to the power $q$: `e[1:] / e[:-1] ** q`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# order_estimates
e = np.asarray(errors, dtype=float)
if np.any(e <= 0):
    raise ValueError("errors must be positive")
r = np.log(e[1:] / e[:-1])
return r[1:] / r[:-1]

# rate_estimates
e = np.asarray(errors, dtype=float)
return e[1:] / e[:-1] ** q
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

all three are loops that append to a Python list and convert it with `np.array` at the end. decide carefully *where* the stopping test goes: Newton and secant check $|f(x_k)| \le$ tol **before** computing the next iterate; bisection records the midpoint, then checks whether to stop.

</details>

<details><summary>hint 2: the approach</summary>

**bisection**: compute `fa, fb` once (raise if `fa * fb > 0`). each iteration: `m = (a + b) / 2`, append it, `fm = f(m)`; stop if `fm == 0` or `(b - a) / 2 <= tol`; otherwise, if `fa * fm < 0` the root is in `[a, m]` so set `b, fb = m, fm`, else `a, fa = m, fm`.

**newton**: start with `xs = [x0]`; loop up to `max_iters` times: evaluate `fx = f(xs[-1])`, break if `abs(fx) <= tol`, else append `xs[-1] - fx / df(xs[-1])`.

**secant**: keep `f_prev, f_cur` for the last two iterates so each step costs one new evaluation; break if `abs(f_cur) <= tol` or `f_cur == f_prev`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
# secant
xs = [float(x0), float(x1)]
f_prev, f_cur = f(x0), f(x1)
for _ in range(max_iters):
    if abs(f_cur) <= tol or f_cur == f_prev:
        break
    x_new = xs[-1] - f_cur * (xs[-1] - xs[-2]) / (f_cur - f_prev)
    xs.append(x_new)
    f_prev, f_cur = f_cur, f(x_new)
return np.array(xs)
```

</details>
