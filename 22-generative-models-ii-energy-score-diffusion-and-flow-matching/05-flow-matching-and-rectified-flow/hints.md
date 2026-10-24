# flow matching & rectified flow · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`if x0 is None: x0 = torch.randn(x1.shape, generator=generator).to(x1.device)`; then `t = torch.rand(x1.shape[0], generator=generator).to(x1.device)`; reshape it for broadcasting, `tt = t.view((-1,) + (1,) * (x1.dim() - 1))`; `x_t = (1 - tt) * x0 + tt * x1`; return `F.mse_loss(model(x_t, t, y), x1 - x0)` (the model gets the `(B,)` `t`, not `tt`).

</details>

## stage 2

<details><summary>hint</summary>

the same pattern as topic 04's ODE sampler: eval mode, `torch.no_grad()`, `try/finally` to restore the mode. `x = torch.randn(shape, generator=generator) if x0 is None else x0.detach().clone()`, moved to `device`; `f = lambda t, v: model(v, torch.full((B,), float(t), device=device), y)`; `x = odeint(f, x, torch.tensor([0.0, 1.0]), method=method, step_size=(1 + 1e-6) / n_steps)[-1]`.

</details>

## stage 3

<details><summary>hint</summary>

convert `X` (and `X0` if given) with `torch.as_tensor(..., dtype=torch.float32)`; per step `idx = torch.randint(len(X), (batch_size,), generator=g)`, then `flow_matching_loss(model, X[idx], x0=None if X0 is None else X0[idx], generator=g)`, and the usual Adam step. call `model.train()` before the loop.

</details>

## stage 4

<details><summary>hint</summary>

pairs: turn an int `shape` into `(shape,)`; `x0 = torch.randn((n, *shape), generator=generator).to(device)`; `x1 = sample_flow(model, (n, *shape), n_steps=n_steps, x0=x0, y=y, device=device)`. straightness: eval mode, no gradients; `x = x0.clone()`, and for `k in range(n_steps)`: `v = model(x, torch.full((B,), k / n_steps), y)`, keep `v`, `x = x + v / n_steps`; then `disp = x - x0` and average `((disp - v) ** 2).flatten(1).sum(1).mean()` over the kept velocities; return a python float.

</details>
