# federated averaging · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`torch.cat([p.detach().reshape(-1) for p in model.parameters()])` is already a copy. for `set_weights`, walk the parameters with a running offset and `p.copy_(weights[offset:offset + p.numel()].reshape(p.shape))` inside `torch.no_grad()`. for the average, stack the vectors into a `(m, P)` tensor and multiply by the normalized weights: `(n / n.sum()) @ stacked` (match the dtypes first).

</details>

## stage 2

<details><summary>hint</summary>

create `rng = np.random.default_rng(seed)` once, before the epoch loop, and draw `order = rng.permutation(n)` at the start of every epoch. inside, `for start in range(0, n, bs): idx = order[start:start + bs]`. after `loss.backward()`, update each parameter under `torch.no_grad()` with `p.add_(p.grad, alpha=-lr)` (or use `torch.optim.SGD`), and zero the gradients before the next step.

</details>

## stage 3

<details><summary>hint</summary>

IID is `np.array_split(rng.permutation(n), n_clients)`. for shards, `order = np.argsort(y, kind="stable")`, `shards = np.array_split(order, n_clients * s)`, `deal = rng.permutation(len(shards))`, and client `k` gets `np.concatenate([shards[j] for j in deal[k*s:(k+1)*s]])`.

</details>

## stage 4

<details><summary>hint</summary>

snapshot `w = get_weights(model)` at the start of each round, and keep it unchanged while the clients run. inside the client loop: `set_weights(model, w)`, call `local_update`, form `delta = w_k - w`, optionally compress it, and record `len(y_k)`. after the loop, `set_weights(model, w + fedavg_aggregate(deltas, sizes))`. keep running totals for bytes and steps and append them to the history after each round; one message is `w.numel() * w.element_size()` bytes.

</details>

## stage 5

<details><summary>hint</summary>

`lo, hi = v.min(), v.max()`; return early if they're equal. with `levels = 2**bits - 1`, `x = (v - lo) / (hi - lo) * levels`, `low = torch.floor(x)`, and `u = torch.as_tensor(rng.random(v.shape), dtype=v.dtype)`, the quantized level is `low + (u < x - low)`; map back with `lo + q * (hi - lo) / levels`. clamp `q` to `[0, levels]` to be safe against rounding at the top end.

</details>
