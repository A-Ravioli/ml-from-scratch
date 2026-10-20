# equivariant networks · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`src, dst = edge_index`; `diff = pos[dst] - pos[src]`; the squared distance is `(diff * diff).sum(-1, keepdim=True)`. for the radius graph, `torch.cdist(pos, pos)` gives all distances; keep pairs below the cutoff that share a `batch` id (`batch[:, None] == batch[None, :]`), drop the diagonal, and `keep.nonzero(as_tuple=True)` returns (row, column) pairs in row-major order: use the rows as sources and the columns as targets and the result is already sorted.

</details>

## stage 2

<details><summary>hint</summary>

per edge, `m = phi_e(cat([h[dst], h[src], d2, edge_attr?]))` with `phi_e` taking `2H + 1 + edge_dim` inputs. positions: `count = index_add` of ones at `dst`, clamped at 1; `shift = zeros_like(pos).index_add(0, dst, diff * phi_x(m)) / count[:, None]` with `phi_x` ending in `nn.Linear(H, 1)`. features: `agg = index_add` of `m` at `dst`, then `h + phi_h(cat([h, agg]))`. every learned quantity depends on `h` and `d2` only; `pos` enters only through `diff`.

</details>

## stage 3

<details><summary>hint</summary>

`h = embed(x)`; loop `h, pos = layer(h, pos, edge_index, edge_attr)`; `node = head(h)`; then the per-molecule sum `torch.zeros(num_graphs, out_dim).index_add(0, batch, node)`, with `num_graphs = batch.max() + 1` when not given. the final positions aren't used by the readout.

</details>
