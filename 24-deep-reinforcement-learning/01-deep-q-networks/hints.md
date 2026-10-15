# deep Q-networks · hints

this chapter is spec-driven: the lesson's spec boxes are the contract, and each stage gets one hint. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

compute the two accelerations from the *current* state first, then do the four Euler updates with the old values on the right-hand side (unpack `x, x_dot, theta, theta_dot = self.state` once and build the new array at the end). twelve degrees is `12 * 2 * np.pi / 360`. the flags must be Python `bool`s: wrap the comparisons in `bool(...)`, since `abs(np.float64) > 2.4` is a `np.bool_`. truncation is `not terminated and self.t >= self.max_steps`. return `self.state.copy()`, never the array itself.

</details>

## stage 2

<details><summary>hint</summary>

keep two integers, `pos` (where the next write goes) and `size` (how many are stored). `add` writes `self.obs[pos] = obs` (assigning into a preallocated row copies the data), then `pos = (pos + 1) % capacity` and `size = min(size + 1, capacity)`. `sample` draws `idx = self.rng.integers(0, self.size, size=batch_size)` and returns `{key: torch.as_tensor(array[idx])}`; fancy indexing returns a copy, so the tensors don't alias the buffer.

</details>

## stage 3

<details><summary>hint</summary>

```python
q = q_net(batch["obs"]).gather(1, batch["actions"].long().unsqueeze(1)).squeeze(1)
with torch.no_grad():
    next_q = target_net(batch["next_obs"])                        # (B, A)
    chooser = q_net(batch["next_obs"]) if double else next_q
    best = chooser.argmax(dim=1, keepdim=True)                    # (B, 1)
    y = batch["rewards"] + gamma * (1 - batch["terminated"]) * next_q.gather(1, best).squeeze(1)
```

then `F.smooth_l1_loss(q, y)` or `F.mse_loss(q, y)`. for the dueling head, run the shared layers to a hidden vector `h`, then `v = self.value(h)` `(B, 1)` and `a = self.advantage(h)` `(B, A)`, and return `v + a - a.mean(dim=1, keepdim=True)`.

</details>

## stage 4

<details><summary>hint</summary>

the skeleton, after building `q_net`, `target_net = copy.deepcopy(q_net)`, Adam, and the buffer (all inside `torch.random.fork_rng(devices=[])` after `torch.manual_seed(seed)`):

```python
obs, _ = env.reset(seed=int(rng.integers(2**31)))
for t in range(n_steps):
    eps = linear_schedule(eps_start, eps_end, eps_decay_steps, t)
    with torch.no_grad():
        q = q_net(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
    a = epsilon_greedy(q, eps, rng=rng)
    next_obs, r, terminated, truncated, _ = env.step(a)
    buf.add(obs, a, r, next_obs, terminated)          # terminated, not done
    ...                                               # bookkeeping; reset on terminated or truncated
    if t + 1 >= learning_starts and (t + 1) % train_freq == 0:
        ...                                           # one gradient step on buf.sample(batch_size)
        ...                                           # count it; hard or soft target update
```

</details>
