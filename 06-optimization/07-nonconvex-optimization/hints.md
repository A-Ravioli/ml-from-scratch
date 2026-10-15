# nonconvex optimization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

use slices instead of loops: `x[:-1]` is every $x_i$ with a successor, `x[1:]` every $x_{i+1}$. for the Rosenbrock gradient, write the residual `r = x[1:] - x[:-1] ** 2` once, then add the contribution of each term to the two coordinates it involves.

</details>

<details><summary>hint 2: the approach</summary>

$f = \sum b\,r_i^2 + (a - x_i)^2$. differentiating term $i$ by $x_i$ gives $-4bx_ir_i - 2(a - x_i)$; differentiating it by $x_{i+1}$ gives $2br_i$. so `g[:-1] = -4 * b * x[:-1] * r - 2 * (a - x[:-1])` and then `g[1:] += 2 * b * r` (`+=`, not `=`). Rastrigin is one line each: `A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))` and `2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
x = np.asarray(x, dtype=float)
r = x[1:] - x[:-1] ** 2
g = np.zeros_like(x)
g[:-1] = -4 * b * x[:-1] * r - 2 * (a - x[:-1])
g[1:] += 2 * b * r
return g
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

the loop runs exactly `max_iters` times (unless `f` blows up). each iteration computes the gradient once and does *either* a kick *or* a gradient step. keep track of the best `(x, f)` seen so far, starting with `x0`.

</details>

<details><summary>hint 2: the approach</summary>

for the kick: `u = rng.normal(size=x.shape)`, `xi = u / norm(u) * noise_radius * rng.random() ** (1 / n)`, with `rng = as_rng(rng)` created once at the start. after the move: append `f(x)`, increment `k`, call the callback, break if not finite, update the best. at the end, `converged` is whether `norm(grad(best_x)) < grad_threshold`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
while k < max_iters:
    g = np.asarray(grad(x), dtype=float)
    if np.linalg.norm(g) < grad_threshold:
        u = rng.normal(size=x.shape)
        x = x + u / np.linalg.norm(u) * noise_radius * rng.random() ** (1.0 / n)
    else:
        x = x - lr * g
    fx = float(f(x)); history.append(fx); k += 1
    if callback is not None:
        callback(x, k)
    if not np.isfinite(fx):
        break
    if fx < best_f:
        best_x, best_f = x, fx
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

two `np.linspace` grids and a double loop (or a nested list comprehension) that evaluates `f` at `x0 + a * d1 + b * d2`. the outer index goes with `d1`.

</details>

<details><summary>hint 2: the approach</summary>

`alphas = betas = np.linspace(-span, span, resolution)`; `Z = np.array([[f(x0 + a * d1 + b * d2) for b in betas] for a in alphas])` puts `alphas[i]` in row `i` and `betas[j]` in column `j`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
alphas = np.linspace(-span, span, resolution)
betas = np.linspace(-span, span, resolution)
Z = np.array([[f(x0 + a * d1 + b * d2) for b in betas] for a in alphas], dtype=float)
return alphas, betas, Z
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

subtract the last iterate from every row, then an SVD of that matrix gives the principal directions as rows of `Vt`, and the singular values give how much each one explains.

</details>

<details><summary>hint 2: the approach</summary>

`D = xs - xs[-1]`, `_, s, Vt = np.linalg.svd(D, full_matrices=False)`, `d1, d2 = Vt[0], Vt[1]`, `coords = D @ np.stack([d1, d2], axis=1)`, `explained = s[:2] ** 2 / np.sum(s ** 2)`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
xs = np.asarray(xs, dtype=float)
D = xs - xs[-1]
_, s, Vt = np.linalg.svd(D, full_matrices=False)
d1, d2 = Vt[0], Vt[1]
coords = D @ np.stack([d1, d2], axis=1)
return d1, d2, coords, s[:2] ** 2 / np.sum(s**2)
```

</details>
