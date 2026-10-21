# decoding & inference · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

a boolean "seen" mask: `torch.zeros_like(logits, dtype=torch.bool).scatter(-1, prev_tokens, True)` (scattering the same index twice still gives True, so repeats count once). then on a clone, `out[seen] = torch.where(out[seen] > 0, out[seen] / penalty, out[seen] * penalty)`. min-p: `probs = F.softmax(logits, -1)`, `keep = probs >= min_p * probs.max(-1, keepdim=True).values`, `logits.masked_fill(~keep, float("-inf"))`.

</details>

## stage 2

<details><summary>hint</summary>

loop over the drafts: `u = torch.rand((), generator=generator).item()`; accept if `u * q[i, x] < p[i, x]` (that's $u < p/q$ without dividing by a small number). on rejection, `residual = (p[i] - q[i]).clamp(min=0)`, normalize, and return `(i, int(torch.multinomial(residual, 1, generator=generator)))`. after the loop, return `(k, ...)` with a sample from `p[k]`.

</details>

## stage 3

<details><summary>hint</summary>

keep `seq` (the tokens so far). in each round: append `n = min(k, remaining)` draft tokens one at a time to a copy, remembering each step's draft distribution `q` (a one-hot of the argmax at temperature 0). run the target once on the extended sequence and keep its last `n + 1` positions' logits: position `len(seq) - 1 + i` predicts draft `i`. temperature 0: count how many drafts equal the target's argmax and take the argmax at the first mismatch (or the bonus position). otherwise softmax both at the temperature and call `speculative_accept`. append the accepted drafts plus the new token, add to the proposed/accepted counters, and truncate to `max_new_tokens` at the end. wrap the loop in `try/finally` to restore the models' modes.

</details>

## stage 4

<details><summary>hint</summary>

keep a Python list `generated`. each step: `mask = torch.as_tensor(allowed(list(generated)), dtype=torch.bool)`; `break` if `not mask.any()`; take the last logits of `model(seq[:, -block_size:])`, `masked_fill(~mask, -inf)`, then argmax or `torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=generator)`; append to both `generated` and `seq`.

</details>
