# GNN expressivity · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

convert each `edge_index` to a NumPy array first. keep one colour array per graph (all zeros at the start). in each round, build every node's signature `(own colour, tuple(sorted(colours of in-neighbours)))` for *all* graphs, then make one palette `{sig: k for k, sig in enumerate(sorted(set_of_all_signatures))}` and map every node through it. Python loops over nodes and edges are fine here: WL is combinatorial and these graphs are small. stack the rounds into `(n_iters + 1, N)`.

</details>

## stage 2

<details><summary>hint</summary>

different `num_nodes` → `False`. otherwise colour both graphs *together* with `wl_colors([g1, g2], n_iters)`, and for each round compare the sorted colour vectors (`torch.sort(c).values`), which compares the histograms. any mismatch → `False`; none → `True`.

</details>

## stage 3

<details><summary>hint</summary>

`agg = torch.zeros_like(x).index_add(0, edge_index[1], x[edge_index[0]])` is the raw neighbour sum. `self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))`. for `eps`: `nn.Parameter(torch.tensor(float(eps)))` if `train_eps`, else `self.register_buffer("eps", torch.tensor(float(eps)))`. return `self.mlp((1 + self.eps) * x + agg)`.

</details>

## stage 4

<details><summary>hint</summary>

keep a list of readouts: before each layer, append `torch.zeros(num_graphs, h.shape[1]).index_add(0, batch, h)` (the per-graph sum), run the layer (ReLU after every layer but the last), and append the final layer's sum too. the head is `nn.Linear(in_dim + n_layers * hidden_dim, out_dim)` applied to `torch.cat(readouts, dim=1)`.

</details>
