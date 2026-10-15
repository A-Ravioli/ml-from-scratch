# the fisher metric · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`p = torch.softmax(logits, dim=-1)`, then `torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)` handles any number of leading batch dimensions. for the Gaussian, build `torch.diag(torch.stack([torch.exp(-2 * log_std), torch.tensor(2.0)]))` after converting `log_std` with `torch.as_tensor(..., dtype=torch.float64)`; on paper 2 says where the 2 comes from.

</details>

## stage 2

<details><summary>hint</summary>

for `fisher_exact`: `theta = theta.detach().clone().requires_grad_(True)`, then for each `x` compute `lp = log_prob(theta, x)`, `(s,) = torch.autograd.grad(lp, theta)`, and add `lp.detach().exp() * torch.outer(s, s)`. for `kl_hessian`: `torch.autograd.functional.hessian(lambda t: kl(theta0, t), theta0.clone())` with a detached `theta0`.

</details>

## stage 3

<details><summary>hint</summary>

build the Jacobians first: run `logits = model(X)` once, then for each example `n` and class `k` call `torch.autograd.grad(logits[n, k], params, retain_graph=True)` and concatenate the flattened pieces into row `J[n, k]` of an `(N, K, P)` tensor. then `true_fisher` is `torch.einsum("nkp,nkl,nlq->pq", J, H, J) / N` with `H = categorical_fisher(logits)` (detached), and the empirical Fisher uses the per-example scores `G[n] = (onehot(y_n) - p_n) @ J[n]`, giving `G.T @ G / N`.

</details>

## stage 4

<details><summary>hint</summary>

`pullback_fisher` is one line: `jacobian.T @ fisher @ jacobian`. for the length, `d = path[1:] - path[:-1]` and `mid = (path[1:] + path[:-1]) / 2`; loop over the pairs, add `sqrt(d_t @ fisher_fn(m_t) @ d_t)` (clamp tiny negative round-off to 0 first), and return a `float`.

</details>
