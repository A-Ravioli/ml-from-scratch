# efficient attention & inference · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`LayerKVCache` keeps `self.k = self.v = None`; `update` stores the first pair as is and afterwards does `torch.cat([self.k, k], dim=-2)` (same for `v`); `seq_len` is a `@property` returning `0 if self.k is None else self.k.shape[-2]`. `KVCache` holds `self.layers = [LayerKVCache() for _ in range(n_layers)]` (a comprehension, so the objects are distinct).

</details>

## stage 2

<details><summary>hint</summary>

```python
cache = KVCache(len(model.blocks))
logits, _ = model(idx, cache=cache)                 # prefill
for step in range(max_new_tokens):
    nxt = sample_next_token(logits[:, -1], ...)[:, None]
    out.append(nxt)
    if step < max_new_tokens - 1:
        logits, _ = model(nxt, cache=cache)         # decode one token
```

check `getattr(model, "block_size", None)` before starting, and wrap the loop in `model.eval()` / `try` / `finally: model.train(was_training)`; decorate the function with `@torch.no_grad()` (under `@app.function`).

</details>

## stage 3

<details><summary>hint</summary>

keep `m` (shape `(..., Tq, 1)`, starts at `-inf`), `l` (zeros, same shape) and `acc` (zeros, `(..., Tq, d_v)`). per block: `s = q @ kb.transpose(-2, -1) / sqrt(d)`, mask with `k_pos > q_pos` where `q_pos = arange(Tq) + Tk - Tq`, `m_new = torch.maximum(m, s.amax(-1, keepdim=True))`, `m_safe = where(isinf(m_new), 0, m_new)`, `p = exp(s - m_safe)`, `scale = exp(m - m_safe)`, then `l = l * scale + p.sum(-1, keepdim=True)` and `acc = acc * scale + p @ vb`. return `acc / l` (0 where `l == 0`).

</details>

## stage 4

<details><summary>hint</summary>

`x[:, :, None].expand(B, H, n_rep, T, dh).reshape(B, H * n_rep, T, dh)` repeats each head in place (`torch.repeat_interleave(x, n_rep, dim=1)` does the same). the byte count is a product of seven numbers, the first being 2 (keys and values).

</details>

## stage 5

<details><summary>hint</summary>

non-causal: `kv = phi(k).transpose(-2, -1) @ v` and `z = phi(k).sum(-2)`, then `(phi(q) @ kv) / ((phi(q) * z[..., None, :]).sum(-1, keepdim=True) + eps)`. causal: prefix sums `torch.cumsum(phi(k)[..., :, :, None] * v[..., :, None, :], dim=-3)` and `torch.cumsum(phi(k), dim=-2)`, then contract each position's `phi(q_i)` with its own prefix. the step function does the same update once: `S + outer(phi(k), v)`, `z + phi(k)`, then the output.

</details>
