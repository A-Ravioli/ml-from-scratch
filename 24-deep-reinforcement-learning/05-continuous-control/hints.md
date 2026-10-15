# continuous control · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

in `step`: unpack `th, thdot = self.state`; `u = float(np.clip(np.ravel(action)[0], -2, 2))`; compute the cost from `th`, `thdot`, `u` *first*; then `thdot = np.clip(thdot + (3 * g / (2 * l) * np.sin(th) + 3 / (m * l**2) * u) * dt, -8, 8)` and `th = th + thdot * dt`. return `-float(cost)`, `False`, and `truncated = self.t >= self.max_steps` after incrementing the counter.

</details>

## stage 2

<details><summary>hint</summary>

```python
mu, log_std = self.net(obs).chunk(2, dim=-1)
std = log_std.clamp(-20, 2).exp()
u = mu if deterministic else mu + std * torch.randn_like(mu)
log_prob = Normal(mu, std).log_prob(u).sum(-1)
log_prob = log_prob - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
return self.act_limit * torch.tanh(u), log_prob
```

`TwinQ` is two `MLP(obs_dim + act_dim, hidden_sizes, 1)`s applied to `torch.cat([obs, act], -1)`, each squeezed to `(B,)`.

</details>

## stage 3

<details><summary>hint</summary>

each loss is one line: `y = rewards + gamma * (1 - terminated) * (torch.min(q1n, q2n) - alpha * next_log_prob)` inside `torch.no_grad()`; `(alpha * log_prob - torch.min(q1, q2)).mean()`; `-(log_alpha.exp() * (log_prob.detach() + target_entropy)).mean()`.

</details>

## stage 4

<details><summary>hint</summary>

per step after the warm-up: sample a batch; with `no_grad`, `a2, lp2 = actor(next_obs)` and `q1t, q2t = critic_target(next_obs, a2)`; the target from stage 3 (with the current α, a float); critic loss `mse(q1, y) + mse(q2, y)` and its step; then `a_new, lp = actor(obs)`, `q1n, q2n = critic(obs, a_new)`, the actor loss and its step (zero the critic's grads before *its* next backward, since this backward also fills them); the temperature step with `lp`; finally `ema_update(critic_target, critic, decay=1 - tau)`.

</details>
