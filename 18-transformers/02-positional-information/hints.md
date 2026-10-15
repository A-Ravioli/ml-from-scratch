# positional information · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

build the angles as an outer product: `pos = torch.arange(T)[:, None]`, `two_i = torch.arange(0, d, 2)`, `angle = pos / 10000 ** (two_i / d)`, shape `(T, ceil(d/2))`. fill a zero tensor with `pe[:, 0::2] = sin(angle)` and `pe[:, 1::2] = cos(angle)[:, :d // 2]` (the slice handles odd `d`). computing in float64 and casting to float32 at the end keeps the large positions accurate.

</details>

## stage 2

<details><summary>hint</summary>

in `__init__`: `inv_freq = base ** (-torch.arange(0, head_dim, 2) / head_dim)`; `angles = torch.outer(torch.arange(n), inv_freq)`; `emb = torch.cat([angles, angles], -1)`; keep `emb.cos()` and `emb.sin()` as float32 plain attributes (not buffers, so a bf16 cast can't touch them), moving them to the input's device when it differs. in `forward`, rebuild the tables if `offset + T` is past their end, slice rows `offset:offset + T`, cast to `x.dtype`, and return `x * cos + torch.cat([-x2, x1], -1) * sin` with `x1, x2` the two halves of the last dimension.

</details>

## stage 3

<details><summary>hint</summary>

slopes for a power of two `n`: `start = 2 ** (-8 / n)`, then `[start ** (h + 1) for h in range(n)]`. otherwise `m = 2 ** floor(log2(n))` and `slopes(m) + slopes(2 * m)[0::2][:n - m]`. for the bias, `dist = (i[:, None] - i[None, :]).float()` with `i = torch.arange(T)`, then `-slopes[:, None, None] * dist`, and `masked_fill(dist < 0, -inf)`.

</details>

## stage 4

<details><summary>hint</summary>

hold each layer's pieces in an `nn.ModuleDict` (`ln1`, `qkv` = `Linear(d, 3d)`, `out`, `ln2`, `mlp`) inside an `nn.ModuleList`. per layer: `q, k, v = layer["qkv"](layer["ln1"](x)).chunk(3, -1)`, `split_heads` each, rotate `q` and `k` if `'rope'`, then `scaled_dot_product_attention(q, k, v, mask=bias)` for `'alibi'` (its bias already blocks the future) or `is_causal=True` otherwise, `merge_heads`, project, add; then the MLP with its own residual. positions for `'learned'` and `'sinusoidal'` are added once, to the embeddings.

</details>
