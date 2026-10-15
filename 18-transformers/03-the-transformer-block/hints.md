# the transformer block · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`self.fc1 = nn.Linear(d_model, d_ff or 4 * d_model)`, `self.fc2 = nn.Linear(d_ff, d_model)`, `self.act = nn.GELU()` (or `nn.ReLU()`), `self.dropout = nn.Dropout(dropout)`; check `activation` before building anything. `forward` is one line: `self.dropout(self.fc2(self.act(self.fc1(x))))`.

</details>

## stage 2

<details><summary>hint</summary>

```python
if self.norm_first:
    x = x + self.drop(self.attn(self.ln1(x), mask=mask, is_causal=is_causal, cache=cache))
    if self.has_cross:
        x = x + self.drop(self.cross_attn(self.ln_cross(x), context))
    return x + self.mlp(self.ln2(x))
x = self.ln1(x + self.drop(self.attn(x, mask=mask, is_causal=is_causal, cache=cache)))
...   # the same pattern for cross-attention and the MLP, the norm after the add
```

`self.drop = nn.Dropout(dropout)`; `FeedForward` already applies dropout to its own output. create `ln_cross` and `cross_attn` only when `cross_attention=True`, pass `mask=context_mask` to `cross_attn`, and check the context arguments at the top of `forward`.

</details>

## stage 3

<details><summary>hint</summary>

register the positions once: `self.register_buffer("pos", sinusoidal_positions(max_len, d_model), persistent=False)`, and embed with `emb(idx) + self.pos[:idx.shape[1]]`. in `encode`, turn `src_mask` (B, S) into `src_mask[:, None, None, :]` and pass it as every encoder block's `mask`; in `decode`, call each decoder block as `block(h, memory, is_causal=True, context_mask=memory_mask[:, None, None, :])` (or `None`), and have `forward` pass `src_mask` to `decode`. the final norms are `nn.LayerNorm(d_model)` for pre-LN and `nn.Identity()` for post-LN.

</details>

## stage 4

<details><summary>hint</summary>

for each block, collect `p.grad.pow(2).sum()` over `block.parameters()` whose `.grad` isn't `None`, add them up, take `math.sqrt(float(total))`; append `0.0` when the list is empty.

</details>
