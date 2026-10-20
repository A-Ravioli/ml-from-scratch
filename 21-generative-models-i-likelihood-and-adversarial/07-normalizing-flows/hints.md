# normalizing flows · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

one network `net: dim -> hidden -> hidden -> 2*dim` whose output you `.chunk(2, dim=1)` into raw $s$ and $t$; set `s = scale * torch.tanh(raw_s)` with `self.scale = nn.Parameter(torch.ones(dim))`, and multiply both by `(1 - mask)` so they vanish on the pass-through coordinates. forward: `x_m = x * mask`, `y = x_m + (1 - mask) * (x * s.exp() + t)`, `log_det = s.sum(1)`. inverse: `y_m = y * mask`, the same `s, t` from `y_m`, `x = y_m + (1 - mask) * (y - t) * (-s).exp()`, `log_det = -s.sum(1)`. zero-initializing the network's last layer makes the layer start as the identity.

</details>

## stage 2

<details><summary>hint</summary>

`self.layers = nn.ModuleList(AffineCoupling(dim, (torch.arange(dim) + k) % 2, hidden_dim=hidden_dim) for k in range(n_layers))`. `forward` loops over the layers accumulating `log_det`; `inverse` loops over `reversed(self.layers)` calling `layer.inverse` and keeping only the first output; `log_prob` is `standard_normal_log_prob(z) + log_det`; `sample` draws `torch.randn((n, self.dim), generator=generator)` and calls `inverse` under `torch.no_grad()`.

</details>

## stage 3

<details><summary>hint</summary>

convert `X` with `torch.as_tensor(X, dtype=torch.float32)`; `g = torch.Generator().manual_seed(seed)`; each step: `idx = torch.randint(len(X), (batch_size,), generator=g)`, `loss = -flow.log_prob(X[idx]).mean()`, `opt.zero_grad(); loss.backward(); opt.step()`, append `loss.item()`.

</details>

## stage 4

<details><summary>hint</summary>

`p = alpha + (1 - 2 * alpha) * x`, `y = torch.log(p) - torch.log1p(-p)`, `log_det = (math.log(1 - 2 * alpha) - torch.log(p) - torch.log1p(-p)).flatten(1).sum(1)`. for the bits: under `torch.no_grad()`, `u = dequantize(x_levels, n_levels=n_levels, generator=generator)`, `y, ld = logit_transform(u, alpha=alpha)`, `log_p = flow.log_prob(y) + ld` (keep `y` in the data's shape), `D = y[0].numel()`, and `bits_per_dim(discrete_nll(log_p, D, n_levels=n_levels), D)`.

</details>
