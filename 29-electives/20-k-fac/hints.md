# K-FAC · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`pi = torch.sqrt((torch.trace(A) / A.shape[0]) / (torch.trace(G) / G.shape[0]))` and `r = math.sqrt(damping)`; return `torch.linalg.inv(A + pi * r * eye_a)` and `torch.linalg.inv(G + r / pi * eye_g)`, with `eye_a = torch.eye(A.shape[0], dtype=A.dtype)` (and the same for `G`). handle `damping == 0` separately (no `pi` needed). `kron_precondition` is the single line `G_inv @ grad @ A_inv`.

</details>

## stage 2

<details><summary>hint</summary>

register a forward hook on every `nn.Linear` (`[m for m in model.modules() if isinstance(m, nn.Linear)]`) that appends `inp[0].detach()` to `inputs` and the output tensor itself (not detached) to `outputs`; run `logp = torch.log_softmax(model(X), -1)`; remove the hooks in a `finally`. then `A = abar.T @ abar / N` with `abar = torch.cat([a, ones(N, 1)], 1)`. for `G`: with labels, `grads = torch.autograd.grad(logp.gather(1, labels[:, None]).sum(), outputs)` and `G = g.T @ g / N` per layer; without, loop over classes `k`, take `torch.autograd.grad(logp[:, k].sum(), outputs, retain_graph=True)`, scale each row by `sqrt(p[:, k])` and accumulate `(w * g).T @ (w * g) / N`.

</details>

## stage 3

<details><summary>hint</summary>

for each linear layer, `grad = torch.cat([layer.weight.grad, layer.bias.grad[:, None]], 1)` and `U = kron_precondition(grad, *damped_inverses(A, G, damping))`. if `kl_clip` is set, `vg = sum((U * grad).sum() for each layer)` and `nu = min(1, sqrt(kl_clip / (lr**2 * vg)))`. then under `torch.no_grad()`: `layer.weight -= lr * nu * U[:, :-1]` and `layer.bias -= lr * nu * U[:, -1]`.

</details>

## stage 4

<details><summary>hint</summary>

in `step`: (1) on statistics steps, pick the labels (`torch.multinomial(softmax(model(X)), 1, generator=self.generator)` under `no_grad` for `"sample"`, `None` for `"exact"`, `y` for `"empirical"`), call `kfac_factors`, and either store the result or blend it into `self.factors` with `ema_decay`. (2) on inverse steps (or if `self.inverses is None`), `self.inverses = [damped_inverses(A, G, self.damping) for A, G in self.factors]`. (3) `model.zero_grad()`, mean cross-entropy, `.backward()`. (4) precondition each layer with its stored inverses, compute `nu` as in stage 3, then `v = momentum * v + nu * U` and subtract `lr * v` (weights from `v[:, :-1]`, biases from `v[:, -1]`). (5) `self.steps += 1` and return `loss.item()`.

</details>
