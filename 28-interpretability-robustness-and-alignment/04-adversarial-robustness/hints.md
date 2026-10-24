# adversarial robustness · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

save the mode, `model.eval()`, and inside `with torch.enable_grad():` make `xg = x.detach().clone().requires_grad_(True)` and take `grad = torch.autograd.grad(loss_fn(model(xg), y), xg)[0]`; restore the mode in `finally`. then `x_adv = x.detach() + epsilon * grad.sign()`, `x_adv.clamp(*clamp)` unless `clamp` is None.

</details>

## stage 2

<details><summary>hint</summary>

write one `project(x_adv)` helper: `delta = x_adv - x0`; `linf`: `delta.clamp(-eps, eps)`; `l2`: per-example norms `n = delta.flatten(1).norm(dim=1)` reshaped to broadcast, `delta * torch.clamp(eps / n, max=1)`; then clamp `x0 + delta`. the random `l2` start is a Gaussian direction normalised per example times `eps * u ** (1 / d)` with `u` uniform and `d` the number of pixels. draw on the generator's device (CPU) and `.to(x.device)`. in the loop, take the gradient as in stage 1, step, project.

</details>

## stage 3

<details><summary>hint</summary>

`x_adv = pgd_attack(model, x, y, epsilon, alpha=alpha, n_steps=n_steps, loss_fn=loss_fn, generator=generator)` (it restores the mode, so the model is back in whatever mode it was); then `model.train()`, the loss on `x_adv`, `zero_grad`, `backward`, `step`, `return loss.item()`. for `robust_accuracy`, attack each batch first, then `model.eval()` and count correct predictions under `no_grad`; restore the mode in `finally`.

</details>

## stage 4

<details><summary>hint</summary>

each corruption is one line: `x + 0.08 * s * torch.randn(x.shape, generator=generator)`, `m + (x - m) * (1 - 0.15 * s)` with `m = x.mean(dim=(1, 2, 3), keepdim=True)`, `x + 0.1 * s`, and for the shift `out = torch.zeros_like(x); out[..., s:] = x[..., :W - s]`. clamp at the end. `corruption_accuracy` loops over kinds and severities in eval mode under `no_grad`.

</details>
