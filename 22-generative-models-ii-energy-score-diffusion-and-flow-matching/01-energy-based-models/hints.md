# energy-based models · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

the score: inside `score_fn`, `with torch.enable_grad():` make `x = x.detach().requires_grad_(True)`, compute `(g,) = torch.autograd.grad(energy_fn(x).sum(), x)`, and return `-g.detach()`. summing the energies gives each example's gradient in one call, because example $b$'s energy depends only on row $b$. the sampler: copy `x0` (`x = x0.detach().clone()`), then under `torch.no_grad()` loop `n_steps` times: `z = torch.randn(x.shape, generator=generator).to(x.device)` and `x = x + 0.5 * step_size * score_fn(x) + math.sqrt(step_size) * z`.

</details>

## stage 2

<details><summary>hint</summary>

write a helper that returns `(E(v), grad E(v))` for a batch, detached. keep the current energies and gradients between steps so you evaluate the energy only at the proposal. with `g = grad E(x)` and `g_p = grad E(x')`: `log_q(to, frm, grad_frm) = -((to - frm + 0.5 * h * grad_frm) ** 2).flatten(1).sum(1) / (2 * h)`, `log_alpha = e - e_p + log_q(x, x', g_p) - log_q(x', x, g)`, `acc = torch.log(u) < log_alpha`, and update `x`, `e`, `g` with `torch.where` (reshape `acc` to `(B, 1, ...)` for `x` and `g`). accumulate `acc.float().mean().item()` over steps and divide by `n_steps`.

</details>

## stage 3

<details><summary>hint</summary>

`EnergyMLP`: build a list `[nn.Linear(d, hidden_dim), nn.SiLU()]` `n_layers` times (with `d = data_dim` the first time, `hidden_dim` after), append `nn.Linear(hidden_dim, 1)`, wrap it in `nn.Sequential`, and `.squeeze(-1)` the output. `ebm_loss` is one line: `er.mean() - ef.mean() + alpha * (er.pow(2).mean() + ef.pow(2).mean())`.

</details>

## stage 4

<details><summary>hint</summary>

`lo, hi = X.min(0).values, X.max(0).values`; box noise is `lo + (hi - lo) * torch.rand((n, D), generator=g)`. per iteration: `real = X[torch.randint(len(X), (batch_size,), generator=g)]`; `idx = torch.randint(buffer_size, (batch_size,), generator=g)`; `start = buffer[idx]`; `fresh = torch.rand(batch_size, generator=g) < reinit_prob`; `start = torch.where(fresh[:, None], box_noise(batch_size), start)`; `fake = langevin_dynamics(score_from_energy(energy), start, step_size=..., n_steps=langevin_steps, generator=g)`; `buffer[idx] = fake`; then the loss on `energy(real)` and `energy(fake)`, `zero_grad`, `backward`, `step`, and append `loss.item()` and the two mean energies.

</details>
