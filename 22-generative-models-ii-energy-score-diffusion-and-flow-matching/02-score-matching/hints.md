# score matching · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

turn `sigma` into a `(B,)` tensor first: `sig = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)`, then `sig.expand(B)` if it's 0-d. reshape it to `(B, 1, ..., 1)` for broadcasting: `s_ = sig.view((B,) + (1,) * (x.dim() - 1))`. then `z = torch.randn(x.shape, generator=generator).to(x.device)`, `x_tilde = x + s_ * z`, `target = -z / s_`, `score = score_fn(x_tilde, sig)`, and `(0.5 * sig ** 2 * ((score - target) ** 2).flatten(1).sum(1)).mean()`.

</details>

## stage 2

<details><summary>hint</summary>

`x = x.detach().requires_grad_(True)`; `s = score_fn(x)` once; `v = torch.randn((n_projections, *x.shape), generator=generator).to(x.device)`. for each `k`: `sv = (s * v[k]).flatten(1).sum(1)` (shape `(B,)`), `(gv,) = torch.autograd.grad(sv.sum(), x, create_graph=True)`, and add `((gv * v[k]).flatten(1).sum(1) + 0.5 * sv ** 2).mean()` to a running total; divide by `n_projections`. `create_graph=True` is what lets the trace term reach the parameters.

</details>

## stage 3

<details><summary>hint</summary>

`geometric_sigmas`: `torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), n))`. the network: an `nn.Sequential` of `n_layers` `(Linear, SiLU)` pairs starting from width `data_dim + 1`, then `Linear(hidden_dim, data_dim)`; in `forward`, make `sigma` a `(B,)` tensor as in stage 1, `h = torch.cat([x, sig.log()[:, None]], 1)`, return `self.net(h) / sig[:, None]`. `fit_dsm`: `xb = X[torch.randint(len(X), (batch_size,), generator=g)]`, `sb = sigmas[torch.randint(len(sigmas), (batch_size,), generator=g)]`, `loss = dsm_loss(model, xb, sb, generator=g)`, then the usual Adam step.

</details>

## stage 4

<details><summary>hint</summary>

convert `sigmas` to a list of floats; `sigma_L = sigmas[-1]`. copy `x0`, then under `torch.no_grad()`: for each `s` in order, `alpha = eps * s ** 2 / sigma_L ** 2`, `sig = torch.full((B,), s, device=x.device)`, and `steps_per_sigma` times draw `z = torch.randn(x.shape, generator=generator).to(x.device)` and set `x = x + 0.5 * alpha * score_fn(x, sig) + math.sqrt(alpha) * z`.

</details>
