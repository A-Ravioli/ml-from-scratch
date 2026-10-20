# MDPs & bellman equations · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`Discrete.contains` has to reject bools before anything else (`isinstance(True, int)` is True in Python), and it has to reject floats like `1.0` even though `1.0 == 1`. accept `int` and `np.integer`, and 0-d integer arrays. for `Box`, `np.broadcast(np.asarray(low), np.asarray(high)).shape` infers the shape, and `np.broadcast_to(...).copy()` gives an owned float array. `Env.reset` is two lines: one test, one assignment.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# Discrete.contains
if isinstance(x, (bool, np.bool_)):
    return False
if isinstance(x, np.ndarray):
    if x.shape != () or not np.issubdtype(x.dtype, np.integer):
        return False
    x = int(x)
if not isinstance(x, (int, np.integer)):
    return False
return 0 <= int(x) < self.n

# Env.reset
if seed is not None or getattr(self, "np_random", None) is None:
    self.np_random = np.random.default_rng(seed)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

build a deterministic "next state" table `nxt` of shape $(S, A)$ first: for each state and each of the four moves, the cell you'd land in, or the same state if the move leaves the grid, enters a wall, or starts in an absorbing (goal or wall) state. then a one-hot array `det` of shape $(S, A, S)$, and the slip mixture `(1 - slip) * det + slip * det.mean(axis=1, keepdims=True)`: the mean over the action axis is "a uniformly random action".

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
for s in range(S):
    r, c = divmod(s, width)
    for a, (dr, dc) in enumerate(moves):
        rr, cc = r + dr, c + dc
        ok = 0 <= rr < height and 0 <= cc < width and (rr, cc) not in walls
        nxt[s, a] = rr * width + cc if ok and s not in absorbing else s
det = np.zeros((S, A, S))
det[np.arange(S)[:, None], np.arange(A)[None, :], nxt] = 1.0
P = (1 - slip) * det + slip * det.mean(axis=1, keepdims=True)
R = np.full((S, A), step_reward)
R[absorbing] = 0.0      # absorbing = goal and wall states; their P rows already loop to themselves
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

sample the next state by inverting the CDF: `cdf = np.cumsum(self.P[s, a])`, then the first index where `cdf` exceeds `u = self.np_random.random()` (`np.searchsorted(cdf, u, side="right")`, clipped to `n_states - 1` against rounding). in `rollout`, check the `max_steps` cap at the top of the loop, before asking the policy for an action, and keep `terminated` and `truncated` separate.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
# GridWorld.step
if not self.action_space.contains(action):
    raise ValueError(...)
s = self.state
cdf = np.cumsum(self.P[s, action])
s2 = int(min(np.searchsorted(cdf, self.np_random.random(), side="right"), self.n_states - 1))
self.state, self.t = s2, self.t + 1
terminated = bool(self.terminal[s2])
truncated = bool(not terminated and self.max_steps is not None and self.t >= self.max_steps)
return s2, float(self.R[s, action]), terminated, truncated, {}

# rollout
while not (terminated or truncated):
    if max_steps is not None and len(actions) >= max_steps:
        truncated = True
        break
    action = policy(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    ...
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

collapse the policy into the model first: $P_\pi$ = `np.einsum("sa,sat->st", policy, P)` and $R_\pi$ = `(policy * R).sum(axis=1)`. then it's a fixed-point loop with a `for` over `range(max_iters)` and a `break` on convergence. `q_from_v` is one line: `P @ V` contracts the last axis of `P`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
V = np.zeros(len(R_pi))
for _ in range(max_iters):
    V_new = R_pi + gamma * P_pi @ V
    done = np.max(np.abs(V_new - V)) < tol
    V = V_new
    if done:
        break
return V
```

</details>
