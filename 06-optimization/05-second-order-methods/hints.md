# second-order methods · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

the loop is your `gradient_descent` loop; only the direction and the step length change. compute the direction by solving a linear system (with `solve` from `mlfs.linalg`), then decide whether to trust it.

</details>

<details><summary>hint 2: the approach</summary>

`H = hess(x)`; `d = solve(H + damping * np.eye(n), -g)` inside `try/except np.linalg.LinAlgError` (fallback `d = -g`). with a line search: if `g @ d >= 0` (or `d` isn't finite), `d = -g`; then `alpha = backtracking_line_search(f, x, fx, g, d, alpha0=1.0)` and `break` if it's 0. without one, `alpha = 1.0`. step `x = x + alpha * d`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
while np.linalg.norm(g) > tol and k < max_iters:
    H = np.asarray(hess(x), dtype=float)
    try:
        d = np.asarray(solve(H + damping * np.eye(n), -g), dtype=float)
    except np.linalg.LinAlgError:
        d = -g
    if line_search:
        if not np.all(np.isfinite(d)) or g @ d >= 0:
            d = -g
        alpha = backtracking_line_search(f, x, fx, g, d, alpha0=1.0)
        if alpha == 0.0:
            break
    else:
        alpha = 1.0
    x = x + alpha * d
    ...  # history, k, callback, finiteness check, g = grad(x), as in gradient_descent
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the line search keeps a bracket `[lo, hi]` with `hi = inf` at the start. each trial either fails Armijo (too long: lower `hi`), fails the curvature condition (too short: raise `lo`), or passes both. the update is one line with two outer products; `bfgs` is the `newton` loop with `d = -Hinv @ g` and an update after each step.

</details>

<details><summary>hint 2: the approach</summary>

line search: `slope = gx @ d`; for each trial compute `f_new = f(x + alpha * d)`; if not finite or `f_new > fx + c1 * alpha * slope`: `hi = alpha`; else `g_new = grad(x + alpha * d)`, and if `g_new @ d < c2 * slope`: `lo = alpha`, else return `(alpha, f_new, g_new)`. next `alpha = (lo + hi) / 2` if `hi` is finite else `2 * alpha`. update: `rho = 1 / (y @ s)`, `V = I - rho * np.outer(s, y)`, return `V @ Hinv @ V.T + rho * np.outer(s, s)`. in `bfgs`, reuse `f_new` and `g_new` from the line search, and rescale `Hinv = (s @ y) / (y @ y) * I` just before the first update.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
while np.linalg.norm(g) > tol and k < max_iters:
    d = -Hinv @ g
    if g @ d >= 0:
        Hinv, d = np.eye(n), -g
    alpha, f_new, g_new = wolfe_line_search(f, grad, x, fx, g, d)
    if alpha == 0.0:
        break
    s, y = alpha * d, g_new - g
    x, fx, g = x + s, f_new, g_new
    ...  # history, k, callback
    if s @ y > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
        if not updated:
            Hinv, updated = (s @ y) / (y @ y) * np.eye(n), True
        Hinv = bfgs_update(Hinv, s, y)
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

`two_loop` never builds a matrix: it walks the pairs twice, first from newest to oldest (saving one number per pair), then from oldest to newest (using those numbers in reverse). `lbfgs` is your `bfgs` with the matrix replaced by two deques.

</details>

<details><summary>hint 2: the approach</summary>

`pairs = list(zip(s_hist, y_hist))`. first loop over `reversed(pairs)`: `a = (s @ q) / (y @ s)`, `q = q - a * y`, append `a`. scale `q` by `gamma = (s @ y) / (y @ y)` of `pairs[-1]` (if any). second loop over `zip(pairs, reversed(alphas))`: `b = (y @ q) / (y @ s)`, `q = q + (a - b) * s`. in `lbfgs`, `S, Y = deque(maxlen=history_size), deque(maxlen=history_size)`; `d = -two_loop(g, S, Y)`; on a non-descent direction clear both deques.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
pairs = list(zip(s_hist, y_hist))
q = np.array(g, dtype=float)
alphas = []
for s, y in reversed(pairs):
    a = (s @ q) / (y @ s)
    q = q - a * y
    alphas.append(a)
if pairs:
    s, y = pairs[-1]
    q = (s @ y) / (y @ y) * q
for (s, y), a in zip(pairs, reversed(alphas)):
    b = (y @ q) / (y @ s)
    q = q + (a - b) * s
return q
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

build the keyword arguments every method shares once (`max_iters`, `tol`, `callback`, plus `**options`), then pick the function. Newton is the odd one out: it also takes `hess` as a positional argument.

</details>

<details><summary>hint 2: the approach</summary>

`common = dict(max_iters=max_iters, tol=tol, callback=callback, **options)`. a dict maps `'gd'`, `'heavy_ball'`, `'nesterov'`, `'bfgs'`, `'lbfgs'` to the functions; `'newton'` checks `hess is not None` and calls `newton(f, grad, hess, x0, **common)`. anything else: `raise ValueError(...)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
common = dict(max_iters=max_iters, tol=tol, callback=callback, **options)
if method == "newton":
    if hess is None:
        raise ValueError("method='newton' needs hess=")
    return newton(f, grad, hess, x0, **common)
table = {"gd": gradient_descent, "heavy_ball": heavy_ball, "nesterov": nesterov, "bfgs": bfgs, "lbfgs": lbfgs}
if method not in table:
    raise ValueError(f"unknown method {method!r}")
return table[method](f, grad, x0, **common)
```

</details>
