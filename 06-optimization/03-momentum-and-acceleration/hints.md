# momentum & acceleration · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

start from your `gradient_descent` loop. the only new state is the previous iterate `x_prev`, initialized to `x0` itself so that the first velocity `x - x_prev` is zero.

</details>

<details><summary>hint 2: the approach</summary>

each step: `x_new = x - lr * g + momentum * (x - x_prev)`, then `x_prev = x` and `x = x_new`. a tuple assignment does both at once: `x, x_prev = x - lr * g + momentum * (x - x_prev), x` (the right side is evaluated first). everything else (history, callback, divergence check, `OptimResult`) is as in topic 02; import `OptimResult` from `mlfs.optim`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.array(x0, dtype=float); x_prev = x
fx = float(f(x)); history = [fx]
g = np.asarray(grad(x), dtype=float); k = 0
while np.linalg.norm(g) > tol and k < max_iters:
    x, x_prev = x - lr * g + momentum * (x - x_prev), x
    fx = float(f(x)); history.append(fx); k += 1
    if callback is not None:
        callback(x, k)
    if not np.isfinite(fx):
        break
    g = np.asarray(grad(x), dtype=float)
return OptimResult(x, fx, k, bool(np.isfinite(fx) and np.linalg.norm(g) <= tol), history)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the loop is the same as heavy ball's, but each step has two parts: build the look-ahead point `y` from the momentum, then take a gradient step *from `y`* using the gradient *at `y`*. the gradient at `x` is only for the stopping test.

</details>

<details><summary>hint 2: the approach</summary>

increment `k` at the start of the step, so the first step has `k = 1` and `beta = (k - 1) / (k + 2) = 0`. `beta = (k - 1) / (k + 2) if momentum is None else momentum`. then `y = x + beta * (x - x_prev)` and `x, x_prev = y - lr * grad(y), x`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
while np.linalg.norm(g) > tol and k < max_iters:
    k += 1
    beta = (k - 1) / (k + 2) if momentum is None else momentum
    y = x + beta * (x - x_prev)
    x, x_prev = y - lr * np.asarray(grad(y), dtype=float), x
    fx = float(f(x)); history.append(fx)
    if callback is not None:
        callback(x, k)
    if not np.isfinite(fx):
        break
    g = np.asarray(grad(x), dtype=float)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

for every eigenvalue you need the two roots of $z^2 - az + b = 0$, with $a$ and $b$ from the lesson's table. the roots can be complex, so compute the discriminant as a complex number before taking its square root.

</details>

<details><summary>hint 2: the approach</summary>

with `lam = np.asarray(eigs, dtype=float)`, build arrays `a` and `b` (for heavy ball, `b = momentum * np.ones_like(lam)`). then `disc = np.sqrt((a * a - 4 * b).astype(complex))`, the roots are `(a ± disc) / 2`, and the rate per eigenvalue is the larger `np.abs`. return the max over eigenvalues as a float. `tuned_momentum` is the two formulas with `q = sqrt(L / mu)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
disc = np.sqrt((a * a - 4 * b).astype(complex))
r = np.maximum(np.abs((a + disc) / 2), np.abs((a - disc) / 2))
return float(np.max(r))

# tuned_momentum
q = np.sqrt(L / mu)
if nesterov:
    return 1.0 / L, float((q - 1) / (q + 1))
return float(4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2), float(((q - 1) / (q + 1)) ** 2)
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

a tridiagonal matrix is three diagonals: build each with `np.eye(n, k=...)`. the lower bound is a one-line formula.

</details>

<details><summary>hint 2: the approach</summary>

`T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)`; `A = (L / 4) * T`; `b = np.zeros(n)` with `b[0] = L / 4`. the bound is `3 * L * R**2 / (32 * (k + 1) ** 2)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
b = np.zeros(n)
b[0] = L / 4
return (L / 4) * T, b
```

</details>
