# infinite width: NTK & mean field · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

write a helper that builds the Jacobian: `out = model(x).reshape(-1)`, then for each `i`, `g = torch.autograd.grad(out[i], params, retain_graph=True, allow_unused=True)` (an unused parameter gives `None`: use zeros), flatten each piece with `.reshape(-1)` and `torch.cat` them into one row; `torch.stack` the rows. with `params = [p for p in model.parameters() if p.requires_grad]`, the kernel is `(J1 @ J2.T).detach()`, reusing `J1` when `x2` is `None`.

</details>

## stage 2

<details><summary>hint</summary>

branch on `torch.is_tensor(K_test_train)`. in the tensor branch convert the other two with `torch.as_tensor(..., dtype=K_test_train.dtype)` and use `torch.linalg.solve(K + ridge * torch.eye(n), y)`; in the array branch the same with `np.linalg.solve`. `solve` handles `y` of shape `(n,)` and `(n, k)` alike.

</details>

## stage 3

<details><summary>hint</summary>

`self.W = nn.Parameter(torch.randn(width, in_dim))`, `self.a = nn.Parameter(torch.randn(width))`, and in `forward`, `scale * (torch.relu(x @ self.W.T) @ self.a)`. for the kernel: `dots = X1 @ X2.T`, `norms = X1.norm(dim=1)[:, None] * X2.norm(dim=1)[None, :]`, `cos = (dots / norms).clamp(-1, 1)` (guard zero norms), `t = torch.arccos(cos)`, then the two terms of the formula, in float64.

</details>

## stage 4

<details><summary>hint</summary>

convert everything to float64 tensors, `lam, V = torch.linalg.eigh(K)`, `r0 = V.T @ (y - f0_train)`, `decay = (1 - lr * lam) ** steps`. train: `y - V @ (decay * r0)`. test: `f0_test + K_test_train @ (V @ (gain * r0))` with `gain = (1 - decay) / lam`, replaced by `lr * steps` where `|lam|` is below about `1e-12 * lam.max()`. use `torch.where` with a safe denominator so you never divide by zero.

</details>

## stage 5

<details><summary>hint</summary>

in the loop: `loss = 0.5 * ((model(x).reshape(-1) - y) ** 2).sum()`, record `float(loss.detach())`, `grads = torch.autograd.grad(loss, params)`, then under `torch.no_grad()`, `p -= lr * g` for each pair. after the loop, record the final loss once more. for `relative_change`, accumulate `((p - p0) ** 2).sum()` and `(p0 ** 2).sum()` over `named_parameters()` and return the square root of their ratio.

</details>
