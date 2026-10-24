# contrastive learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

```python
z = F.normalize(torch.cat([z1, z2]), dim=1)           # (2B, d)
logits = z @ z.T / temperature
logits = logits.masked_fill(torch.eye(len(z), dtype=torch.bool), float("-inf"))
targets = (torch.arange(len(z)) + len(z1)) % len(z)
return F.cross_entropy(logits, targets)
```

use `masked_fill` (which returns a new tensor), not an in-place `fill_diagonal_` on a tensor autograd needs.

</details>

## stage 2

<details><summary>hint</summary>

draw every random quantity as a `(B,)` tensor. crop: `w = sqrt(area * ratio).clamp(max=1)`, `h = sqrt(area / ratio).clamp(max=1)`, centre `cx = w/2 + (1 - w) * rand`. in `F.affine_grid`'s normalised coordinates ($[-1, 1]$) the box is `theta = [[w * flip, 0, 2*cx - 1], [0, h, 2*cy - 1]]`, with `flip = ±1`; then `F.grid_sample(x, F.affine_grid(theta, (B, C, size, size), align_corners=False), align_corners=False)`. for the colour steps, compute the jittered batch for everyone and pick per image with `torch.where(mask.view(B, 1, 1, 1), jittered, original)`; the same trick does grayscale. gray: `(x * torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)).sum(1, keepdim=True)`.

</details>

## stage 3

<details><summary>hint</summary>

`self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))`, and `forward` returns `self.net(x)`.

</details>

## stage 4

<details><summary>hint</summary>

order matters for reproducibility: `torch.manual_seed(seed)` and `rng = np.random.default_rng(seed)` first, then build the head and the augmentation. one Adam over `list(encoder.parameters()) + list(head.parameters())`. per epoch: `encoder.train(); head.train(); perm = rng.permutation(n)`, then `for start in range(0, n - bs + 1, bs)`: take `X[perm[start:start + bs]].to(device)`, make two views with two calls to the augmentation, and step on `info_nce_loss(head(encoder(v1)), head(encoder(v2)), temperature=temperature)`.

</details>
