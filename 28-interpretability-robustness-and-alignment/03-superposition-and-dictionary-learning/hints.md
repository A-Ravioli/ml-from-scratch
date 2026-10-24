# superposition & dictionary learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

two draws from the same generator: `values = torch.rand(n, n_features, generator=generator)` and a mask `torch.rand(n, n_features, generator=generator) >= sparsity` (True = keep, which happens with probability $1 - S$). multiply them. check the range of `sparsity` first.

</details>

## stage 2

<details><summary>hint</summary>

`self.W = nn.Parameter(torch.empty(n_hidden, n_features))` then `nn.init.xavier_normal_(self.W)`; `self.b = nn.Parameter(torch.zeros(n_features))`; `self.register_buffer("importance", imp)` makes it move with `.to(device)` without being trained. for a batch of row vectors, $W^\top W x$ is `(x @ self.W.T) @ self.W`.

</details>

## stage 3

<details><summary>hint</summary>

the loss is `(model.importance * (x - model(x)) ** 2).sum(-1).mean()`. in the training loop, call `sparse_feature_batch` *inside* the loop. for the dimensionality: `norms = W.norm(dim=0)`, `unit = W / norms.clamp_min(1e-12)`, then `(unit.T @ W) ** 2` is the matrix of $(\hat W_i \cdot W_j)^2$; sum its rows, divide `norms ** 2` by that, and use `torch.where(norms > 1e-8, D, 0)` for dropped features.

</details>

## stage 4

<details><summary>hint</summary>

`dec = torch.randn(d_in, d_hidden); dec = dec / dec.norm(dim=0, keepdim=True)`; `W_dec = nn.Parameter(dec)`, `W_enc = nn.Parameter(dec.clone())`. `dictionary()` divides by `self.W_dec.norm(dim=0, keepdim=True)` on every call, so decode always uses unit columns. the loss: `((x_hat - x) ** 2).sum(-1).mean() + self.l1_coeff * f.abs().sum(-1).mean()`.

</details>

## stage 5

<details><summary>hint</summary>

`idx = torch.randint(0, acts.shape[0], (batch_size,), generator=generator)` each step, then `sae(acts[idx])[2]` is the loss. for MMCS normalise both matrices' columns; `(true_unit.T @ learned_unit)` is `(n, m)`; take `.max(dim=1).values.mean()`. in `sae_stats`, `active = f > 0`: `l0` is `active.sum(1).float().mean()`, `dead` is `(~active.any(0)).float().mean()`.

</details>
