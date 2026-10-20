# sets · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

make `m = mask.unsqueeze(-1)` (shape `(B, N, 1)`, broadcasts over features). sum: `(h * m).sum(1)`. mean: that, divided by `m.sum(1).clamp(min=1)` (the clamp turns an empty set's 0/0 into 0/1). max: `h.masked_fill(~m, float("-inf")).amax(1)`, then replace the rows of empty sets (`~m.any(1)`) by zeros with `torch.where`. with `mask=None`, build an all-`True` mask and reuse the same code.

</details>

## stage 2

<details><summary>hint</summary>

two MLPs and one pool: `self.phi = MLP(in_dim, [hidden_dim], hidden_dim)`, `self.rho = MLP(hidden_dim, [hidden_dim], out_dim)`, and `forward` is `self.rho(masked_pool(self.phi(x), mask, self.aggregation))`. a `Linear` acts on the last axis, so `phi` applied to `(B, N, in_dim)` processes every element with the same weights.

</details>

## stage 3

<details><summary>hint</summary>

two `nn.Linear(in_dim, out_dim)` layers, one with a bias ($\Lambda$, $b$) and one without ($\Gamma$). `y = lam(x) + gam(masked_pool(x, mask, agg)).unsqueeze(1)`: the pooled `(B, F)` vector is broadcast back to every element. finish with `y * mask.unsqueeze(-1)` when a mask is given.

</details>

## stage 4

<details><summary>hint</summary>

attention: `scores = q @ k.transpose(-1, -2) / math.sqrt(d)`, `masked_fill(~mask.unsqueeze(1), -inf)`; for sets with no real element, replace their score rows by zeros (`torch.where` on `mask.any(-1)[:, None, None]`) before the softmax, and multiply the weights by that same flag after it. pooling: queries are `self.q(self.seeds)` expanded to the batch; reshape `(B, L, dim)` → `(B, L, heads, dim/heads)` → transpose to `(B, heads, L, d)` → merge to `(B·heads, L, d)`, repeat the mask `heads` times with `repeat_interleave(heads, dim=0)`, attend, undo the reshapes, apply the output projection.

</details>
