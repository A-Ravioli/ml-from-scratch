# in-context learning as meta-learning · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`sample_linear_tasks`: `w = torch.randn(B, d, generator=g)`, `xs = x_scale * torch.randn(B, n, d, generator=g)`, `ys = (xs @ w.unsqueeze(-1)).squeeze(-1) + noise_std * torch.randn(B, n, generator=g)`. `least_squares_predictions`: work in float64; for `i` in `1..n-1`, `Xi = X[:, :i]`, `Yi = Y[:, :i, None]`; `w = torch.linalg.pinv(Xi) @ Yi` (minimum norm, batched) or `torch.linalg.solve(Xi.mT @ Xi + lam * I, Xi.mT @ Yi)`; prediction `(X[:, i:i+1] @ w)[:, 0, 0]`; cast back at the end. `position_mse`: `((preds - ys) ** 2).mean(0)`.

</details>

## stage 2

<details><summary>hint</summary>

`gd_predictions`: for each `i`, start `w = zeros(B, d)` and repeat `w = w - lr * (Xi.mT @ (Xi @ w[..., None] - Yi[..., None]))[..., 0]` `n_steps` times. `lsa_predict`: `E = torch.cat([xs, ys[..., None]], -1)` `(B, n, d+1)`, `e_q = torch.cat([x_query, zeros(B, 1)], -1)`; scores `E @ (W_kq @ e_q[..., None])` `(B, n, 1)`; values `E @ W_pv.T`; `(e_q + (scores * values).sum(1))[:, -1]`. `gd_lsa_weights`: zeros, identity in `W_kq[:d, :d]`, `lr` in `W_pv[d, d]`.

</details>

## stage 3

<details><summary>hint</summary>

`interleave`: allocate zeros `(B, 2n, d+1)`, then `z[:, 0::2, :d] = xs` and `z[:, 1::2, d] = ys`. in `forward`: `h = self.read_in(interleave(xs, ys)) + self.pos[:2 * n]`, then `h = block(h, is_causal=True)` for each block, `self.read_out(self.ln_f(h))[:, 0::2, 0]`. check `d == self.dim` and `n <= self.max_points` first.

</details>

## stage 4

<details><summary>hint</summary>

`generator = torch.Generator().manual_seed(seed)`, `opt = torch.optim.Adam(model.parameters(), lr=lr)`, `model.train()`; per step `b = sample_linear_tasks(batch_size, n_points, model.dim, noise_std=noise_std, generator=generator)`, `loss = ((model(b["xs"], b["ys"]) - b["ys"]) ** 2).mean()`, `zero_grad`/`backward`/`step`, record `loss.item()`. `model.eval()` before returning.

</details>
