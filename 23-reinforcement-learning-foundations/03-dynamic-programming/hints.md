# dynamic programming · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`greedy_policy`: start from `np.zeros(Q.shape)` and put a 1 at `(row, argmax of that row)` for every row at once with fancy indexing; `np.argmax` already breaks ties towards the lowest index. `bellman_optimality` is your `q_from_v` followed by a max over the action axis.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
policy = np.zeros(Q.shape)
policy[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1.0

return np.max(q_from_v(P, R, V, gamma), axis=1)
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

three pieces in a loop: evaluate (`policy_evaluation`), compute `Q` (`q_from_v`), compare. the comparison is between `Q.max(axis=1)` and the value of the action the policy currently takes, `Q[np.arange(S), np.argmax(policy, axis=1)]`. only the states where the first beats the second by more than `tol` get the new greedy action; if there are none, you're done.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
V = policy_evaluation(P, R, np.full((S, A), 1.0 / A), gamma, tol=tol)
policy = greedy_policy(q_from_v(P, R, V, gamma))
for _ in range(1000):
    V = policy_evaluation(P, R, policy, gamma, tol=tol)
    Q = q_from_v(P, R, V, gamma)
    better = Q.max(axis=1) > Q[np.arange(S), np.argmax(policy, axis=1)] + tol
    if not better.any():
        break
    policy = policy.copy()
    policy[better] = greedy_policy(Q)[better]
return policy, V
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the same loop shape as `policy_evaluation` (a `for` over `range(max_iters)`, compute the new values from the old, test the largest change, swap, break), with `bellman_optimality` as the update. after the loop, one more `q_from_v` and `greedy_policy` give the policy.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
V = np.zeros(R.shape[0])
for _ in range(max_iters):
    V_new = bellman_optimality(P, R, V, gamma)
    done = np.max(np.abs(V_new - V)) < tol
    V = V_new
    if done:
        break
return greedy_policy(q_from_v(P, R, V, gamma)), V
```

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

each iteration: greedy policy for the current `V`, collapse it into `P_pi` and `R_pi` (as in `policy_evaluation`), remember `V_old = V`, run `m` sweeps of `V = R_pi + gamma * P_pi @ V`, and stop if `V` moved by less than `tol` over the whole iteration. count iterations with an integer and stop at `max_iters`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
V, n = np.zeros(R.shape[0]), 0
while n < max_iters:
    n += 1
    policy = greedy_policy(q_from_v(P, R, V, gamma))
    P_pi = np.einsum("sa,sat->st", policy, P)
    R_pi = np.sum(policy * R, axis=1)
    V_old = V
    for _ in range(m):
        V = R_pi + gamma * P_pi @ V
    if np.max(np.abs(V - V_old)) < tol:
        break
return greedy_policy(q_from_v(P, R, V, gamma)), V, n
```

</details>
