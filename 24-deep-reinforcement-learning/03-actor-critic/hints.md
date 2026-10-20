# actor-critic · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
with torch.no_grad():
    nonterminal = 1.0 - terminated.float()
    adv = torch.zeros_like(rewards)
    gae, next_value = torch.zeros_like(rewards[0]), last_value
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nonterminal[t] * next_value - values[t]
        gae = delta + gamma * lam * nonterminal[t] * gae
        adv[t] = gae
        next_value = values[t]
```

the same loop works for `(T,)` and `(T, N)`: `rewards[t]` is a scalar or an `(N,)` row. detach `values` first.

</details>

## stage 2

<details><summary>hint</summary>

two `MLP(obs_dim, hidden_sizes, out, activation="tanh")`s, one with `act_dim` outputs and one with 1, plus `self.log_std = nn.Parameter(torch.zeros(act_dim))` if continuous. in `forward`: `Independent(Normal(mean, self.log_std.exp().expand_as(mean)), 1)` or `Categorical(logits=...)`, and `self.critic(obs).squeeze(-1)`. `act` reshapes the observation to `(1, obs_dim)`, samples under `no_grad`, and unpacks the first row.

</details>

## stage 3

<details><summary>hint</summary>

per step: `dist, value = model(torch.as_tensor(np.stack(self.obs), dtype=torch.float32))` under `no_grad`; `actions = dist.sample()`; then loop over the envs. for env `i`: step it, add the reward to `running[i]`, and if `terminated or truncated`: if it was only truncated, add `gamma * model(cut_obs)[1]` to its reward; set `dones[i] = 1`; record and zero `running[i]`; replace the observation with `env.reset()[0]`. append the *old* observations (the ones the actions were taken in) to the batch, then move on with the new ones. after the loop, stack each list with `torch.stack` and compute `last_value` from the current observations.

</details>

## stage 4

<details><summary>hint</summary>

`a2c_loss`: `pg = -(advantages.detach() * dist.log_prob(actions)).mean()`, `vl = ((values - returns.detach()) ** 2).mean()`, `ent = dist.entropy().mean()`, and `loss = pg + vf_coef * vl - ent_coef * ent`; report `.item()`s. in `train_a2c`, flatten the `(T, N, ...)` batch to `(T * N, ...)` before calling the model, and use `nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)` between `backward()` and `step()`.

</details>
