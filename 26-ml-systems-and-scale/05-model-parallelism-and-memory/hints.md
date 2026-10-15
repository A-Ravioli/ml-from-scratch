# model parallelism & memory · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

per-parameter bytes: params and grads `bytes_per_param`; optimizer `{"adamw": 8, "momentum": 4, "sgd": 0}[optimizer] + (4 if bytes_per_param < 4 else 0)`. with `shard = -(-n_params // world_size)`, each part counts `shard` parameters when its stage shards it (`zero_stage >= 1` for the optimizer, `>= 2` for grads, `>= 3` for params) and `n_params` otherwise.

</details>

## stage 2

<details><summary>hint</summary>

`saved_tensor_bytes`: inside `with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):` call `fn(*args)`; `pack(t)` records `seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()` unless the pointer is one of `exclude`'s, and returns `t`.

`checkpoint`: define a `torch.autograd.Function` inside the function (so it can see `fn`). `forward(ctx, *inputs)`: `ctx.rng = torch.get_rng_state()`, `ctx.save_for_backward(*inputs)`, return `fn(*inputs)` under `torch.no_grad()`. `backward(ctx, grad_out)`: detach the saved inputs and `requires_grad_` them as the originals did; inside `with torch.random.fork_rng(devices=[]):` restore `ctx.rng` and recompute under `torch.enable_grad()`; then `torch.autograd.backward(out, grad_out)` and return each input's `.grad` (or `None`). call it with `Recompute.apply(*args)`.

</details>

## stage 3

<details><summary>hint</summary>

with `k = h // p`: `(fc1.weight[r*k:(r+1)*k].detach().clone(), fc1.bias[r*k:(r+1)*k].detach().clone(), fc2.weight[:, r*k:(r+1)*k].detach().clone())`. each partial is `F.gelu(x @ w1.T + b1) @ w2.T`; sum them and add `b2` once.

</details>

## stage 4

<details><summary>hint</summary>

make a `2 * (m + p - 1)` by `p` grid of `None`, then for every stage `s` and micro-batch `j` set `grid[s + j][s] = ("F", j)` and `grid[(m + p - 1) + (p - 1 - s) + j][s] = ("B", j)`.

</details>

## stage 5

<details><summary>hint</summary>

in `__init__`: `self.params = list(params)`, `N = sum(p.numel() ...)`, `S = -(-N // world_size)`, `shard = (min(rank*S, N), min(rank*S + S, N))`, zero moments of length `S`, step count 0. in `step()` (under `torch.no_grad()`): build a zero vector of length `S * world_size`, fill its first `N` entries with the concatenated flattened grads (zeros for `None`); if `world_size > 1`, `dist.all_reduce` it and divide by `world_size`. build the same padded vector of parameters, take your slice `[rank*S, rank*S + S)` of both, apply AdamW to it (only where a gradient exists: keep a per-element step count and a mask built the same way as the gradient vector, and use `torch.where`), then `dist.all_gather` a list of `world_size` slices (or, with one rank, write the slice back), concatenate, and copy consecutive pieces back into each parameter with `p.copy_(flat[o:o + p.numel()].view_as(p))`.

</details>
