# uncertainty & OOD · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

work in $\beta = 1/T$, starting at 1. each iteration: `p = softmax(beta * z)` row-wise (use `scipy.special.logsumexp` for stability), `ez = (p * z).sum(1)`, gradient `mean(ez - z[range(n), y])`, Hessian `mean((p * z**2).sum(1) - ez**2)`; Newton step `beta -= g / h`, halving the step while it would make $\beta \le 0$ or raise the NLL. stop early when the gradient is tiny; return `1 / beta`.

</details>

## stage 2

<details><summary>hint</summary>

`s = np.sort(scores)` (a copy), `k = math.ceil((n + 1) * (1 - alpha))`, return `s[k - 1]` or `inf` if `k > n`. for the sets: `scores = 1 - cal_probs[np.arange(n), cal_labels]`, then `(1 - test_probs) <= q` is already the boolean set matrix.

</details>

## stage 3

<details><summary>hint</summary>

`q = conformal_quantile(np.abs(cal_y - cal_pred), alpha)` and return `(test_pred - q, test_pred + q)`. that's the whole method: the guarantee comes from the quantile, not from the model.

</details>

## stage 4

<details><summary>hint</summary>

an entropy helper that treats $0 \log 0$ as 0: `-(np.where(q > 0, q * np.log(np.where(q > 0, q, 1)), 0)).sum(-1)`. total is the entropy of `probs.mean(0)`, aleatoric the mean over axis 0 of the members' entropies. for the AUROC, label the in-distribution scores 0 and the OOD scores 1 and call your chapter 07 `roc_auc_score`.

</details>
