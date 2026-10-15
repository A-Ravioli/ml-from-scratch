# optimization in practice · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`groups = getattr(optimizer, "param_groups", None)` tells the two kinds apart without importing torch. if there are groups, read `groups[0]["lr"]` and write `group["lr"]` for every group; otherwise use `optimizer.lr`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def set_lr(optimizer, lr):
    groups = getattr(optimizer, "param_groups", None)
    if groups is not None:
        for group in groups:
            group["lr"] = float(lr)
    else:
        optimizer.lr = float(lr)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`LRScheduler.__init__`: store the optimizer, `self.base_lr = get_lr(optimizer)`, `self.last_step = 0`, then `set_lr(optimizer, self.lr_at(0))`. `step()`: increment, then `set_lr(self.optimizer, self.lr_at(self.last_step))`. in the subclasses, assign `self.step_size`, `self.gamma` (or the cosine settings) *first*, then call `super().__init__(optimizer)`. `CosineSchedule.lr_at` has three cases: `step < warmup_steps`, `step >= total_steps`, and the cosine in between.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def lr_at(self, step):
    if step < self.warmup_steps:
        return self.base_lr * step / self.warmup_steps
    if step >= self.total_steps:
        return self.min_lr
    t = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
    return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

collect `grads = [p.grad for p in params if p.grad is not None]` (this also materializes a generator once), compute `total = sqrt(sum of np.sum(g**2))`, and if `total > max_norm`, multiply each array in place: `g *= max_norm / (total + 1e-6)`. return `total`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
grads = [p.grad for p in params if p.grad is not None]
total = math.sqrt(sum(float(np.sum(np.square(g))) for g in grads))
if total > max_norm:
    scale = max_norm / (total + 1e-6)
    for g in grads:
        g *= scale
return total
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

save `model.state_dict()`, `get_lr(optimizer)` and (if `type(optimizer.state) is dict`) a `copy.deepcopy` of the optimizer's state; restore all three in a `finally`. for minibatches, keep an iterator over `minibatch_indices(len(X), batch_size, shuffle=True, rng=rng)` and make a new one when `next(it, None)` returns `None`. each step: compute the step's learning rate, `set_lr`, then the five-line update; record `lr` and `loss.item()`; `break` if the loss isn't finite or exceeds `4 * best`. `suggest_lr`: one loop computing the bias-corrected EMA, then `lrs[argmin] / 10`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
for i in range(n_steps):
    lr = min_lr * (max_lr / min_lr) ** (i / max(n_steps - 1, 1))
    idx = next(batches, None)
    if idx is None:
        batches = iter(minibatch_indices(len(X), batch_size, shuffle=True, rng=rng))
        idx = next(batches)
    set_lr(optimizer, lr)
    optimizer.zero_grad()
    loss = loss_fn(model(X[idx]), y[idx])
    loss.backward()
    optimizer.step()
    lrs.append(lr)
    losses.append(loss.item())
    best = min(best, losses[-1])
    if not math.isfinite(losses[-1]) or losses[-1] > 4 * best:
        break
```

</details>
