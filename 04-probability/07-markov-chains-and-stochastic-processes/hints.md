# markov chains & stochastic processes · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

allocate the output `(n_steps + 1, S)`, put `p0` in row 0, and fill each next row from the previous one.

</details>

<details><summary>hint 2: the approach</summary>

row `t + 1` is `out[t] @ P`: a row vector times the matrix. (`P @ out[t]` would use the columns of `P`.)

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
out = np.empty((n_steps + 1, P.shape[0]))
out[0] = np.asarray(p0, dtype=float)
for t in range(n_steps):
    out[t + 1] = out[t] @ P
return out
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

precompute the cumulative sums of every row once, `cdf = np.cumsum(P, axis=1)`, and all the uniforms you'll need, `u = rng.random(n_steps)`. then each step is one `np.searchsorted` on the current state's row.

</details>

<details><summary>hint 2: the approach</summary>

validate first: square, no negative entries, `np.allclose(P.sum(axis=1), 1, atol=1e-8)`; otherwise raise `ValueError`. in the loop, `row = cdf[path[t]]` and `path[t + 1] = np.searchsorted(row, u[t] * row[-1], side="right")`. multiplying by `row[-1]` handles cumulative sums that round to just under 1, as in topic 02.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
cdf = np.cumsum(P, axis=1)
u = rng.random(n_steps)
path = np.empty(n_steps + 1, dtype=np.int64)
path[0] = x0
for t in range(n_steps):
    row = cdf[path[t]]
    path[t + 1] = np.searchsorted(row, u[t] * row[-1], side="right")
return path
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

$\pi P = \pi$ transposes to $(P^\top - I)\pi^\top = 0$, a homogeneous linear system with infinitely many solutions (any multiple of $\pi$). one extra equation, $\sum_i \pi_i = 1$, pins it down.

</details>

<details><summary>hint 2: the approach</summary>

stack the matrix `P.T - np.eye(S)` on top of a row of ones, and the right-hand side `zeros(S)` on top of a single 1. the system is `(S + 1) × S` and consistent, so `np.linalg.lstsq(A, b, rcond=None)[0]` solves it exactly. clip tiny negative entries (rounding) and renormalize.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
A = np.vstack([P.T - np.eye(S), np.ones((1, S))])
b = np.zeros(S + 1)
b[-1] = 1.0
pi = np.linalg.lstsq(A, b, rcond=None)[0]
pi = np.clip(pi, 0.0, None)
return pi / pi.sum()
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`total_variation` is one line (half the sum of absolute differences). `spectral_gap`: take `np.abs(np.linalg.eigvals(P))`, sort it in decreasing order, and the answer is `1 - mods[1]`.

</details>

<details><summary>hint 2: the approach</summary>

for `mixing_time`, keep a matrix `D` whose row `x` is the distribution after `t` steps from state `x`. it starts as `np.eye(S)` (at `t = 0` you know exactly where you are). at each `t`, compute the TV distance of every row from `pi` at once, `0.5 * np.sum(np.abs(D - pi), axis=1)`, and stop if the *largest* is at most `eps`. otherwise `D = D @ P`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
pi = stationary_distribution(P)
D = np.eye(P.shape[0])
for t in range(max_steps + 1):
    if 0.5 * np.max(np.sum(np.abs(D - pi), axis=1)) <= eps:
        return t
    D = D @ P
return None
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

a Brownian path on a grid is a running sum of independent Gaussian increments. draw them all at once with your `normal`, shape `(n_paths, n_steps)`.

</details>

<details><summary>hint 2: the approach</summary>

the increments have *variance* `dt`, so standard deviation `math.sqrt(dt)`. take `np.cumsum(..., axis=1)` (along time, within each path) and put a column of zeros in front with `np.concatenate`.

</details>

<details><summary>hint 3: nearly the answer</summary>

```python
inc = normal(0.0, math.sqrt(dt), (n_paths, n_steps), rng=rng)
return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
```

</details>
