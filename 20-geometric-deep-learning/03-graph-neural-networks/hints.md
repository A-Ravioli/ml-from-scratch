# graph neural networks · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

sum: `src.new_zeros((dim_size, *src.shape[1:])).index_add(0, index, src)` (the out-of-place `index_add` keeps autograd happy). mean: divide by the group sizes, `bincount(index, minlength=dim_size)` clamped to at least 1, reshaped to broadcast over the feature axes. max: `scatter_reduce(0, idx, src, reduce="amax", include_self=True)` on a tensor filled with `-inf`, where `idx` is `index` reshaped and expanded to `src`'s shape; then `torch.where(count > 0, out, 0)`.

</details>

## stage 2

<details><summary>hint</summary>

`src, dst = add_self_loops(edge_index, n)`; `deg = degree(dst, n, x.dtype)`; `norm = deg.pow(-0.5)[src] * deg.pow(-0.5)[dst]` (one number per edge); `h = x @ W` first (cheaper when `out_dim < in_dim`); then `scatter_sum(norm[:, None] * h[src], dst, n) + b`. self-loops guarantee every degree is at least 1, so `pow(-0.5)` never divides by zero.

</details>

## stage 3

<details><summary>hint</summary>

`agg = scatter_mean(x[edge_index[0]], edge_index[1], n)` is the neighbour mean (0 where there are no incoming edges); the output is `lin_root(x) + lin_nbr(agg)` with the bias on only one of the two linear layers.

</details>

## stage 4

<details><summary>hint</summary>

softmax: `m = scatter_max(src.detach(), index, n)` (the max is only a stabilizing constant, no gradient needed), `e = (src - m[index]).exp()`, `e / scatter_sum(e, index, n)[index]`. GAT: add self-loops; `h = lin(x).reshape(n, heads, out_dim)`; per-node scores `s = (h * att_src).sum(-1)` and `t = (h * att_dst).sum(-1)`, both `(N, heads)`; edge scores `leaky_relu(s[src] + t[dst], 0.2)`; `alpha = scatter_softmax(scores, dst, n)` then `F.dropout(alpha, p, training=self.training)`; `out = scatter_sum(alpha[..., None] * h[src], dst, n)`, reshape to `(N, heads * out_dim)` or average over heads.

</details>

## stage 5

<details><summary>hint</summary>

the pools are `scatter_sum` / `scatter_mean` with `index = batch` and `dim_size = num_graphs` (computed from `batch.max() + 1` when not given). for `batch_graphs`, keep a running `offset`: append `ei + offset`, a `batch` vector `torch.full((n_g,), g)`, then add `n_g` to the offset; concatenate at the end (`dim=1` for the edge indices).

</details>
