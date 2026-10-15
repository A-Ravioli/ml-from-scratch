# policy gradients · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

one backward loop with a running sum: `acc = rewards[t] + gamma * acc; out[t] = acc` for `t` from the end. allocate `out` first with the right type: `torch.zeros_like(rewards, dtype=torch.float32)` (or the input's float dtype) for a tensor, `np.zeros(len(rewards))` for an array. assigning into `out` leaves `rewards` untouched.

</details>

## stage 2

<details><summary>hint</summary>

`forward` is one line: `Categorical(logits=self.net(obs))`. in `act`, inside `torch.no_grad()`, turn the ndarray into a float32 tensor of shape `(1, obs_dim)`, call `self(...)`, `.sample()`, and return `int(sample[0])`. `MLP(obs_dim, hidden_sizes, n_actions, activation="tanh")` from `mlfs.torchnn` builds the network.

</details>

## stage 3

<details><summary>hint</summary>

```python
w = torch.as_tensor(returns, dtype=log_probs.dtype).detach()
if baseline is not None:
    w = w - torch.as_tensor(baseline, dtype=log_probs.dtype).detach()
return -(w * log_probs).mean()
```

</details>

## stage 4

<details><summary>hint</summary>

per batch: run `batch_episodes` episodes with `policy.act`, keeping lists of observations, actions and rewards; compute each episode's returns with `discounted_returns` (bootstrapping first, if truncated and you have a value net: `rewards[-1] += gamma * V(last_obs)`), and concatenate. then one forward pass over the whole batch, `log_probs = policy(obs).log_prob(actions)`, the loss, and one Adam step. with a value net, compute `values = value_net(obs).squeeze(-1)` once, use it as the baseline, and take a separate Adam step on `((values - returns) ** 2).mean()`. wrap everything in `with torch.random.fork_rng(devices=[]):` after `torch.manual_seed(seed)`.

</details>
