# MAML · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

`init_mlp`: loop over `zip(sizes[:-1], sizes[1:])`; `(2 * torch.rand(fan_in, fan_out, generator=generator) - 1) * bound` with `bound = 1 / math.sqrt(fan_in)` gives the uniform range, and `.requires_grad_()` makes a leaf. `mlp_forward`: `h = h @ W + b.unsqueeze(-2)` works for both layouts, because `b.unsqueeze(-2)` is `(1, d_out)` for shared biases and `(T, 1, d_out)` for per-task ones, and `@` broadcasts the leading task dimension. `sample_sine_tasks`: write a helper `lo + (hi - lo) * torch.rand(shape, generator=generator)`, draw `amp (T,)`, `phase (T,)`, `xs (T, n_support, 1)` and `xq (T, n_query, 1)`, then `amp[:, None, None] * torch.sin(x - phase[:, None, None])`.

</details>

## stage 2

<details><summary>hint</summary>

`fast = [p.expand(T, *p.shape) for p in params]` (first replace any `p` without `requires_grad` by `p.detach().requires_grad_()`). then, `n_steps` times: `loss = loss_fn(forward(fast, xs), ys).sum()`, `grads = torch.autograd.grad(loss, fast, create_graph=not first_order)`, `fast = [w - inner_lr * g for w, g in zip(fast, grads)]`. no `.data`, no in-place ops, no optimizer.

</details>

## stage 3

<details><summary>hint</summary>

`maml_loss` is two lines: `fast = adapt(params, batch["xs"], batch["ys"], ...)` with every option passed through, then `loss_fn(forward(fast, batch["xq"]), batch["yq"]).mean()`. `linear_meta_gradient`: `H = 2 * Xs.T @ Xs / n`; run the $k$ inner steps on a copy of `theta` with the gradient `2 * Xs.T @ (Xs @ th - ys) / n`; `g = 2 * Xq.T @ (Xq @ th - yq) / m`; return `g`, or `np.linalg.matrix_power(np.eye(d) - inner_lr * H, n_steps) @ g`.

</details>

## stage 4

<details><summary>hint</summary>

`theta = [p.detach().clone().requires_grad_() for p in params]`; `generator = torch.Generator().manual_seed(seed)`; build the optimizer on `theta` once. in the loop: `batch = sample_batch(generator)`, `loss = maml_loss(theta, batch, ...)`, `opt.zero_grad()`, `loss.backward()`, `opt.step()`, `history.append(loss.item())`. return `(theta, history)`.

</details>
