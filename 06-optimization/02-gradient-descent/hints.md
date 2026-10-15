# gradient descent · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`OptimResult.__init__` is five assignments with conversions (`np.asarray(x, dtype=float)`, `float`, `int`, `bool`, a list of floats). for the loop, keep three things up to date: the iterate `x`, its value `fx`, and its gradient `g`. compute `g` once per iterate; the loop condition uses it, and so does the step.

</details>

<details><summary>hint 2: the approach</summary>

start with `x = np.array(x0, dtype=float)` (a copy), `fx = f(x)`, `history = [fx]`, `g = grad(x)`, `k = 0`. then `while norm(g) > tol and k < max_iters:` step with `x = x - lr * g` (a new array), update `fx`, append it, `k += 1`, call the callback, `break` if `fx` isn't finite, and recompute `g`. after the loop, `converged` is whether the final `g` passes the test (and `fx` is finite).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.array(x0, dtype=float)
fx = float(f(x)); history = [fx]
g = np.asarray(grad(x), dtype=float); k = 0
while np.linalg.norm(g) > tol and k < max_iters:
    alpha = lr                      # stage 3 changes this line
    x = x - alpha * g
    fx = float(f(x)); history.append(fx); k += 1
    if callback is not None:
        callback(x, k)
    if not np.isfinite(fx):
        break
    g = np.asarray(grad(x), dtype=float)
converged = bool(np.isfinite(fx) and np.linalg.norm(g) <= tol)
return OptimResult(x, fx, k, converged, history)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

compute the slope `gx . direction` once (flatten both, so matrices work too). then loop at most `max_iters` times: evaluate `f` at the trial point, compare with `fx + c * alpha * slope`, and either return `alpha` or multiply it by `rho`.

</details>

<details><summary>hint 2: the approach</summary>

`slope = float(np.ravel(gx) @ np.ravel(direction))`, `alpha = alpha0`. in the loop: `if f(x + alpha * d) <= fx + c * alpha * slope: return alpha`, else `alpha *= rho`. after the loop, `return 0.0`. a `nan` value fails the comparison automatically; wrap the evaluation in `np.errstate(over="ignore", invalid="ignore")` to silence overflow warnings.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
d = np.asarray(direction, dtype=float)
slope = float(np.ravel(gx) @ d.ravel())
alpha = float(alpha0)
for _ in range(max_iters):
    with np.errstate(over="ignore", invalid="ignore"):
        f_new = float(f(x + alpha * d))
    if f_new <= fx + c * alpha * slope:
        return alpha
    alpha *= rho
return 0.0
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

only the choice of `alpha` changes. you already have `x`, `fx` and `g` in the loop: hand them to your line search along the direction `-g`, starting from `alpha0=lr`.

</details>

<details><summary>hint 2: the approach</summary>

`alpha = backtracking_line_search(f, x, fx, g, -g, alpha0=lr)`. if it returns `0.0`, no step along `-g` gives a measurable decrease: `break` out of the loop before stepping (the final `converged` test then reports whatever the gradient says).

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
if line_search:
    alpha = backtracking_line_search(f, x, fx, g, -g, alpha0=lr)
    if alpha == 0.0:
        break
else:
    alpha = lr
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

minimize the one-variable quadratic $\phi(\alpha) = f(x - \alpha g)$: its derivative is linear in $\alpha$. the iteration is then gradient descent with that step, recomputed every iteration.

</details>

<details><summary>hint 2: the approach</summary>

$\phi'(\alpha) = -g^\top g + \alpha\,g^\top Ag$, so `alpha = (g @ g) / (g @ A @ g)`, guarded by `if g @ g == 0: return 0.0`. in the loop: `g = A @ x - b`, `x = x - exact_step(A, g) * g`, append a new array each time.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = [np.array(x0, dtype=float)]
for _ in range(n_iters):
    g = A @ xs[-1] - b
    xs.append(xs[-1] - exact_step(A, g) * g)
return np.array(xs)
```

</details>
