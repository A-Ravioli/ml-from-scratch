# heterogeneity · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

wrap the whole split in `for _ in range(max_tries):` and `return` as soon as `min(len(p) for p in parts) >= min_size`; raise after the loop. inside, for each class: `idx = rng.permutation(np.flatnonzero(y == c))`, `p = rng.dirichlet(np.full(n_clients, alpha))`, `cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)`, and `np.split(idx, cuts)` gives one chunk per client. for `heterogeneity`, the global distribution is the size-weighted average of the rows of `label_distribution`, and TV is `0.5 * np.abs(row - glob).sum()`.

</details>

## stage 2

<details><summary>hint</summary>

for the fixed point, build each client's affine map by accumulating `S = I + G + ... + G^(K-1)` and `G^K` in one loop over the local steps, with `G = I - lr * (A + mu * I)`. then `T_k = G^K + lr * mu * S` and `s_k = lr * S @ A @ c`. average with the weights and solve `(I - T_bar) w = s_bar` with `np.linalg.solve`. in `quadratic_fedavg`, keep the round's starting point `w` fixed while each client steps from a copy of it; the proximal gradient is `mu * (v - w)`.

</details>

## stage 3

<details><summary>hint</summary>

keep `ci` as an `(n_clients, d)` array of zeros and `c` as a zero vector. per round and client: start `v = x.copy()`; for each local step `v -= lr * (A @ (v - c_k) - ci[k] + c)`; then `new_ci = ci[k] - c + (x - v) / (local_steps * lr)`. accumulate `p_k * (v - x)` and `p_k * (new_ci - ci[k])`, store `new_ci`, and apply both accumulated changes to `x` and `c` only after all clients have run (every client in a round must see the same `x` and `c`).

</details>

## stage 4

<details><summary>hint</summary>

copy your `client_update` and add two lines: at entry, `anchor = [p.detach().clone() for p in model.parameters()]`; in every step, `loss = loss + 0.5 * mu * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))` before `backward()`.

</details>

## stage 5

<details><summary>hint</summary>

`local = copy.deepcopy(model)`. for head-only, `head = [m for m in local.modules() if isinstance(m, nn.Linear)][-1]` and update only `head.parameters()` (the others still get gradients; just don't apply them). keep a permutation and a position: `if pos + bs > n: order, pos = rng.permutation(n), 0`, then take `order[pos:pos + bs]` and advance `pos`. for `per_client_accuracy`, if `models` is a single `nn.Module`, repeat it for every client.

</details>
