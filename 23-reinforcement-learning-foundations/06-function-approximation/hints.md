# function approximation · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

work in units of tile widths: after clipping, `u = (x - low) / width` is a vector of shape $(d,)$, and `u + offsets` (broadcast to `(n_tilings, d)`) gives each tiling's continuous coordinates; `np.floor(...).astype(int)` gives tile coordinates. `np.ravel_multi_index(coords.T, (T,) * d)` turns each row into one index in $[0, T^d)$; add `i * T ** d` for tiling $i$. the default offsets are one broadcast expression in `i` (a column) and `j` (a row).

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# __init__
i, j = np.arange(self.n_tilings)[:, None], np.arange(d)[None, :]
self.offsets = (i * (2 * j + 1) / self.n_tilings) % 1.0      # or as_rng(rng).random((n_tilings, d))

# __call__
x = np.clip(np.atleast_1d(np.asarray(obs, dtype=float)), self.low, self.high)
u = (x - self.low) / self.width
coords = np.minimum(np.floor(u + self.offsets).astype(int), self.tiles_per_dim - 1)
flat = np.ravel_multi_index(coords.T, (self.tiles_per_dim,) * len(self.low))
return np.arange(self.n_tilings) * self.tiles_per_dim ** len(self.low) + flat
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

keep the state as a length-2 float array `self.state`. in `step`, compute the new velocity first (with the *old* position in the cosine), clip it, then the new position from the new velocity, clip it, and zero the velocity if the position hit $-1.2$. return `self.state.copy()`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
x, v = self.state
v = float(np.clip(v + 0.001 * (action - 1) - 0.0025 * np.cos(3 * x), -0.07, 0.07))
x = float(np.clip(x + v, -1.2, 0.5))
if x <= -1.2:
    v = 0.0
self.state = np.array([x, v])
self.t += 1
terminated = bool(x >= 0.5)
truncated = bool(not terminated and self.max_steps is not None and self.t >= self.max_steps)
return self.state.copy(), -1.0, terminated, truncated, {}
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

it's your `sarsa` with `Q[s]` replaced by `w[idx].sum(axis=0)` (the action values of a state with active features `idx`) and `Q[s, a]` by `w[idx, a].sum()`. the update adds the same amount, `alpha / len(idx) * delta`, to every active weight of the action taken: `w[idx, a] += ...`. carry `idx` and `a` from one step to the next.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
obs2, r, terminated, truncated, _ = env.step(a)
if terminated:
    target = r
else:
    idx2 = featurizer(obs2)
    a2 = epsilon_greedy(w[idx2].sum(axis=0), epsilon, rng=rng)
    target = r + gamma * w[idx2, a2].sum()
w[idx, a] += alpha / len(idx) * (target - w[idx, a].sum())
done = terminated or truncated
if not done:
    idx, a = idx2, a2
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`np.diag(d)` builds $D$; then $A$ and $b$ are one line each. `expected_td` calls `td_matrices` once, copies `w0` with `np.array(w0, dtype=float)` (or starts from zeros), and appends each new iterate to a list, starting with the first.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
D = np.diag(d)
A = Phi.T @ D @ (Phi - gamma * P @ Phi)
b = Phi.T @ D @ r

w = np.zeros(A.shape[0]) if w0 is None else np.array(w0, dtype=float)
out = [w]
for _ in range(n_iters):
    w = w + alpha * (b - A @ w)
    out.append(w)
return np.array(out)
```

</details>
