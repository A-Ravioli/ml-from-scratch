# privacy & fairness · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}` (and the buffers likewise). define `one_loss(p, xi, yi)` that runs `functional_call(model, (p, buffers), (xi.unsqueeze(0),))` and returns `loss_fn(out, yi.unsqueeze(0))`. then `vmap(grad(one_loss), in_dims=(None, 0, 0))(params, x, y)`: `None` means "the same parameters for every example".

</details>

## stage 2

<details><summary>hint</summary>

compute the mean loss under `no_grad` first. with `grads` from stage 1, the per-example norms are `torch.sqrt(sum(g.reshape(B, -1).pow(2).sum(1) for g in grads.values()))`, and `factor = C / torch.clamp(norms, min=C)` is $\min(1, C/\|g_i\|)$ without dividing by zero. for each parameter: `(g * factor.view(B, 1, ...)).sum(0)`, plus `sigma * C * torch.randn(p.shape, generator=generator)`, divided by `B`, assigned to `p.grad`. then `optimizer.step()`.

</details>

## stage 3

<details><summary>hint</summary>

for each order $\alpha$, build the $\alpha + 1$ log-terms `lgamma(a+1) - lgamma(k+1) - lgamma(a-k+1) + (a-k) log1p(-q) + k log q + (k*k - k) / (2 sigma**2)` and log-sum-exp them (subtract the max before exponentiating); divide by $\alpha - 1$. handle $q = 0$, $\sigma = 0$ and $q = 1$ first. the epsilon is one line of NumPy over the array of orders.

</details>

## stage 4

<details><summary>hint</summary>

the AUROC is your chapter 07 `roc_auc_score` with members labelled 1 and the score $-\text{loss}$. for the low-FPR rate: sort the non-member losses, `m = math.floor(fpr * n)`, `tau = s[m]` (zero-based, so that's $s_{(m+1)}$) or `inf` if `m >= n`, and `np.mean(member_losses < tau)`.

</details>

## stage 5

<details><summary>hint</summary>

loop over `np.unique(groups)`; the approval rate of a group is `y_pred[groups == g].mean()`, its TPR the mean of `y_pred` over the group's positives, its FPR over its negatives. for the thresholds, sort the group's positive scores in descending order and take the `ceil(target * n_pos)`-th one.

</details>
