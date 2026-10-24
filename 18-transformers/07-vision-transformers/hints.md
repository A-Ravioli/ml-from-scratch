# vision transformers · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`x.reshape(B, C, H // p, p, W // p, p)` gives axes `(B, C, grid_row, i, grid_col, j)`; `permute(0, 2, 4, 1, 3, 5)` puts the grid first and `(C, i, j)` last; a final `reshape(B, N, C * p * p)` flattens both. `PatchEmbedding.forward` is `self.proj(x).flatten(2).transpose(1, 2)`.

</details>

## stage 2

<details><summary>hint</summary>

```python
t = self.patch_embed(x)                                        # (B, N, d)
if self.cls_token is not None:
    t = torch.cat([self.cls_token.expand(len(t), -1, -1), t], dim=1)
t = t + self.pos_emb
for block in self.blocks:
    t = block(t)                                               # no mask
return self.norm(t)
```

create the parameters with `nn.Parameter(torch.zeros(...))` and initialize them with `nn.init.trunc_normal_(..., std=0.02)`; `forward` pools `t[:, 0]` or `t.mean(1)` and applies `self.head`.

</details>

## stage 3

<details><summary>hint</summary>

set `block.attn.save_weights = True` for every block, run `model(x)` under `torch.no_grad()` in eval mode, and read `block.attn.last_weights` `(B, H, T, T)`. slice off the [CLS] row and column (`[:, :, 1:, 1:]`), renormalize rows, and multiply by the `(N, N)` matrix `torch.cdist(coords, coords)` where `coords` are the grid positions times the patch size. sum over keys, average over images and queries, stack the layers. undo `save_weights` in a `finally`.

</details>
