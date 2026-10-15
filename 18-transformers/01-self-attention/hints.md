# self-attention · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
Tq, Tk = scores.shape[-2:]
allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril(diagonal=Tk - Tq)   # if is_causal
# combine with a bool mask by &; add a float mask to the scores
scores = scores.masked_fill(~allowed, float("-inf"))
row_ok = (scores > float("-inf")).any(-1, keepdim=True)    # rows with something to attend to
weights = torch.softmax(scores.masked_fill(~row_ok, 0.0), dim=-1).masked_fill(~row_ok, 0.0)
```

then `F.dropout(weights, p=dropout_p)` (only when `dropout_p > 0`) for the copy you multiply by `v`, and return the undropped weights. `causal_mask` is one line with `.tril()`.

</details>

## stage 2

<details><summary>hint</summary>

`x.view(B, T, n_heads, d // n_heads).transpose(1, 2)` and back with `x.transpose(1, 2).reshape(B, T, H * dh)`. the transpose makes the tensor non-contiguous, so merging needs `reshape` (or `.contiguous().view`), not `view`.

</details>

## stage 3

<details><summary>hint</summary>

```python
src = x if context is None else context
q = split_heads(self.q_proj(x), self.n_heads)
k = split_heads(self.k_proj(src), self.n_heads)
v = split_heads(self.v_proj(src), self.n_heads)
if cache is not None:
    k, v = cache.update(k, v)
out, w = scaled_dot_product_attention(q, k, v, mask=mask, is_causal=is_causal,
                                      dropout_p=self.dropout if self.training else 0.0)
```

store `w.detach()` in `self.last_weights` when `self.save_weights`, then `self.out_proj(merge_heads(out))`. set `self.save_weights = False` and `self.last_weights = None` in `__init__`.

</details>
