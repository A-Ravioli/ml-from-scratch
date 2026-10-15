# natural gradient · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`x = torch.linalg.solve(metric, grad)` gives $M^{-1}g$; then $d^* = -r\,x/\sqrt{g\cdot x}$. for `natural_gradient`, build `A = fisher + damping * torch.eye(P, dtype=fisher.dtype)` (a new tensor, not `+=`) and return `torch.linalg.solve(A, grad)`.

</details>

## stage 2

<details><summary>hint</summary>

start from `theta = theta0.detach().clone()`, keep a list `path = [theta.clone()]`, and in each step compute `theta = theta - lr * natural_gradient(grad_fn(theta), fisher_fn(theta), damping=damping)` (a new tensor each time, so earlier path entries don't change). return `torch.stack(path)`.

</details>

## stage 3

<details><summary>hint</summary>

CG: `x = 0`, `r = p = b`, `rr = r @ r`; loop `max_iters` times: break if `rr <= tol`; `Ap = mvp(p)`; `alpha = rr / (p @ Ap)`; `x += alpha p`; `r -= alpha Ap`; `rr_new = r @ r`; `p = r + (rr_new / rr) p`; `rr = rr_new`.

Fisher-vector product: `logits = model(X)`; `w = torch.zeros_like(logits, requires_grad=True)`; `u = torch.autograd.grad(logits, params, grad_outputs=w, create_graph=True)`; flatten `u`, then `Jv = torch.autograd.grad(u_flat @ v, w)[0]` (shape `(N, K)`). with `p = softmax(logits)` (no grad): `HJv = p * Jv - p * (p * Jv).sum(-1, keepdim=True)`. finally `torch.autograd.grad(logits, params, grad_outputs=HJv)`, flattened, divided by `N`, plus `damping * v`.

</details>

## stage 4

<details><summary>hint</summary>

`loss = cross_entropy(model(X), y)`; `g = flatten(torch.autograd.grad(loss, params))`; `step = conjugate_gradient(lambda v: fisher_vector_product(model, X, v, damping=damping), g, max_iters=cg_iters)`. then, under `torch.no_grad()`, walk through `params` with a running offset `i` and do `p -= lr * step[i:i + p.numel()].reshape(p.shape)`. return `loss.item()`.

</details>
