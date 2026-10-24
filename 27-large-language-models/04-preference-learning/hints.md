# preference learning · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`logp = F.log_softmax(model(input_ids)[0].float(), -1)`; `picked = logp.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)` (clamping makes the $-100$s valid indices; you zero them next); multiply by `(targets != -100)`. `sequence_logprobs` is `.sum(-1)` of that.

</details>

## stage 2

<details><summary>hint</summary>

`h = self.backbone.hidden_states(input_ids)`. the index of the last True in each row is `T - 1 - mask.flip(-1).long().argmax(-1)` (`argmax` returns the first maximum, and flipping turns "last True" into "first True"). gather with `h[torch.arange(B), last]`, apply the head, `squeeze(-1)`. Bradley–Terry is one line with `F.logsigmoid`.

</details>

## stage 3

<details><summary>hint</summary>

`chosen = pi_c - ref_c.detach()`, `rejected = pi_r - ref_r.detach()`, `loss = -F.logsigmoid(beta * (chosen - rejected)).mean()`, and return `(beta * chosen).detach()` and `(beta * rejected).detach()` with it.

</details>

## stage 4

<details><summary>hint</summary>

`reference.eval()` and the two reference `sequence_logprobs` calls inside `with torch.no_grad():`; then `policy.train()`, the two policy calls, `dpo_loss`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`. return `{"loss": loss.item(), "accuracy": (r_c > r_r).float().mean().item(), "margin": (r_c - r_r).mean().item()}`.

</details>

## stage 5

<details><summary>hint</summary>

`r = -beta * (policy_logps - ref_logps) * mask`, then find each row's last True as in stage 2 and `r[torch.arange(B), last] += scores` (clone first if you want to be safe). `best_of_n_kl` is `math.log(n) - (n - 1) / n` after the check.

</details>
