# alignment · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

work with log-weights: `logits = log(ref) + r / beta` where `ref > 0` and `-inf` elsewhere, subtract `logits.max()`, exponentiate, normalise. for the KL, sum only over entries with `policy > 0`: `sum(pi * (log(pi) - log(ref)))`. the expectations are dot products.

</details>

## stage 2

<details><summary>hint</summary>

loop over the distinct scores in increasing order (`np.unique`), keeping `below` = the reference mass of all lower scores. for a score level with mass `m`, the probability that the best of $n$ is at this level is `(below + m) ** n - below ** n`; give each item at the level its share `p_i / m` of it; then `below += m`.

</details>

## stage 3

<details><summary>hint</summary>

for each non-goal state and action, the next state is `max(s - 1, 0)` or `s + 1`; set `P[s, a, nxt] = 1`, the goal reward when `nxt` is the goal, and the bonus when `nxt == bonus_pos` (and `s != bonus_pos`). the goal row points to itself with no reward. `gaming_gap` is two calls of `value_iteration` and one of `policy_evaluation`; `rollout_states` follows `argmax(policy[s])` and `argmax(P[s, a])`.

</details>

## stage 4

<details><summary>hint</summary>

run every check once, keeping `(category, passed)` pairs. `rng = as_rng(rng)` once, then for each category (in sorted order) and for `"overall"`, `accuracy_ci(np.array(passes, dtype=float), n_boot=n_boot, rng=rng)`. `compare_agents` builds two pass/fail arrays over the same cases and calls `paired_accuracy_diff(a, b, n_boot=n_boot, rng=rng)`.

</details>
