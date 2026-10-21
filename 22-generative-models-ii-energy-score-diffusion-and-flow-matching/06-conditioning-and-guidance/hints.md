# conditioning & guidance · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`mask = (torch.rand(len(y), generator=generator) < p_uncond).to(y.device)`, then `torch.where(mask, torch.full_like(y, null_label), y)`. `torch.where` returns a new tensor, so `y` is untouched.

</details>

## stage 2

<details><summary>hint</summary>

a small `nn.Module` subclass whose `__init__` stores `self.model = model` (assigning a module to an attribute registers it as a submodule), the scale and the null label. in `forward`: `null = torch.full_like(y, self.null_label)`, `out = self.model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null]))`, `cond, uncond = out.chunk(2)`, `return uncond + w * (cond - uncond)`. for `y is None`, call the model once with the null label for every row. `cfg_model` just returns an instance.

</details>

## stage 3

<details><summary>hint</summary>

another wrapper module holding `eps_model` and `classifier`. in `forward`: `eps = self.eps_model(x, t)`; if `y` is `None` return it. otherwise `with torch.enable_grad():` take `xg = x.detach().requires_grad_(True)`, `logp = torch.log_softmax(self.classifier(xg, t), dim=1)`, `(grad,) = torch.autograd.grad(logp.gather(1, y.view(-1, 1)).sum(), xg)`; then `ab = alpha_bar_at(self.schedule, t)` reshaped to `(B, 1, ...)`, and return `eps - w * (1 - ab).sqrt() * grad`.

</details>

## stage 4

<details><summary>hint</summary>

the loop of `fit_flow_matching` with two extra lines: `labels = drop_labels(y[idx], p_uncond, n_classes, generator=g)` right after drawing `idx`, then `flow_matching_loss(model, X[idx], y=labels, generator=g)` or `ddpm_loss(model, schedule, X[idx], y=labels, generator=g)` depending on `objective`. check `objective` before the loop so a typo fails immediately.

</details>
