# data parallelism · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

keep `buf[i][c]`, rank $i$'s copy of chunk $c$: `[[c.copy() for c in np.array_split(a.ravel(), p)] for a in arrays]`. in each step, first collect what every rank sends (copies), then apply the receives, so a rank never forwards something it received in the same step:

```python
for s in range(p - 1):                                    # reduce-scatter
    sent = [buf[i][(i - s) % p].copy() for i in range(p)]
    for i in range(p):
        c = (i - s) % p
        buf[(i + 1) % p][c] = buf[(i + 1) % p][c] + sent[i]
for s in range(p - 1):                                    # all-gather
    sent = [((i + 1 - s) % p, buf[i][(i + 1 - s) % p].copy()) for i in range(p)]
    for i, (c, x) in enumerate(sent):
        buf[(i + 1) % p][c] = x
```

finish with `np.concatenate(buf[i]).reshape(shape)` for each rank.

</details>

## stage 2

<details><summary>hint</summary>

three formulas from the lesson. for `ddp_efficiency`, the exposed communication is `max(0.0, t_comm - overlap * t_compute)`.

</details>

## stage 3

<details><summary>hint</summary>

set `p.grad = None` for every parameter, compute `N = sum(len(x) for x, _ in micro_batches)`, then for each micro-batch `loss = loss_fn(model(x), y) * len(x) / N; loss.backward()` and add `loss.item()` to a running total, which you return.

</details>

## stage 4

<details><summary>hint</summary>

`perm = np.random.default_rng(seed + epoch).permutation(n)`; the padded length is `-(-n // world_size) * world_size`; `perm = np.resize(perm, padded)` (it repeats `perm` cyclically, which also covers padding longer than `n`); return `perm[rank::world_size]`.

</details>

## stage 5

<details><summary>hint</summary>

`average_gradients`: `grads = [p.grad for p in model.parameters() if p.grad is not None]`, `flat = torch.cat([g.reshape(-1) for g in grads])`, `dist.all_reduce(flat)`, `flat /= world_size`, then walk an offset through `flat`, copying `flat[o:o + g.numel()].view_as(g)` into each `g`. `ddp_train`: `dist.broadcast(p.data, src=0)` for every parameter, then the usual loop over `shard_indices(...)` in slices of `batch_size`, with `average_gradients(model, world_size)` between `backward()` and `step()`. to debug, run a two-process test from a terminal and read the rank's traceback that the check prints.

</details>
