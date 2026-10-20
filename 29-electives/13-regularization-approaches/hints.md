# regularization approaches · hints

open a hint only after you've been stuck for a while. this chapter is spec-driven: one hint per stage.

## stage 1

<details><summary>hint</summary>

the simplest correct version loops: for each example, compute `logp = F.log_softmax(model(x[None])[0][cols], -1)` (`cols` = the classes, or all), then for each class `c` take `torch.autograd.grad(logp[c], params, retain_graph=True)`, square, weight by `logp[c].exp().item()` (exact) or keep only the true label's term (empirical, with the label's position inside `cols`), and add to running sums; divide by the number of examples at the end. the fast version uses `torch.func`: with `params = {name: p.detach()}`, define `log_probs(params, x)` via `functional_call(model, params, (x[None],))[0]`, and `vmap(jacrev(log_probs), in_dims=(None, 0))(params, X_chunk)` gives every example's Jacobian, a dict of tensors `(b, C, *shape)`; combine with weights `(b, C)` using `einsum("bc,bc...->...", w, jac ** 2)`.

</details>

## stage 2

<details><summary>hint</summary>

keep two lists, `anchors` and `fishers`, of dicts keyed by parameter name. `consolidate`: take `task["X"][:n]` and `task["y"][:n]`, call `fisher_diagonal` with `classes=task["classes"] if scenario == "task" else None`, and snapshot `{name: p.detach().clone()}`; for online, replace both lists with single entries, combining `gamma * old + new`. `penalty`: start from `torch.zeros(())`, look up `dict(model.named_parameters())`, and add `(f * (p - anchor) ** 2).sum()` per pair and name; multiply by `0.5 * lam` at the end.

</details>

## stage 3

<details><summary>hint</summary>

store dicts keyed by parameter name: `start`, `prev`, `w` (reset in `begin_task`), `omega` (created once, all zeros) and `anchor` (set in `consolidate`, `None` until then). in `after_step`, under `torch.no_grad()`: `if p.grad is not None: w[n] -= p.grad * (p - prev[n])`, then `prev[n] = p.detach().clone()` for every parameter. `consolidate`: `omega[n] += w[n] / ((p - start[n]) ** 2 + xi)`. `penalty` returns `c * sum((omega[n] * (p - anchor[n]) ** 2).sum())`, or a zero tensor while `anchor` is `None`.

</details>
