# dynamic programming

## why this matters

topic 02 answered "how good is this policy?". the question you actually care about is "what's the *best* policy?". for a gridworld you could guess; for a slippery grid with walls, or a game with thousands of states, you can't, and trying every policy is hopeless: with 4 actions and 100 states there are $4^{100}$ deterministic policies.

dynamic programming (DP) solves this exactly, in time polynomial in the number of states, *if you know the model* $P$ and $R$. two algorithms do it, from opposite ends: **policy iteration** improves a whole policy at a time, and **value iteration** improves the values one backup at a time. both rest on the same two facts: an optimal value function satisfies a fixed-point equation, and acting greedily with respect to a value function never makes things worse.

you'll rarely run DP on a real problem: real problems don't come with $P$ as an array, and their state spaces are too big. but DP is the conceptual core of RL. Monte Carlo, TD learning and Q-learning (topics 04–05) are DP with the model replaced by samples; DQN (ch 24) is value iteration with a neural network; and on every small problem in this course, DP gives the exact answer that learning algorithms are tested against.

## warm-up: predict first

1. you evaluate the random policy on a gridworld, then act greedily on its values. how good is the resulting policy: barely better than random, or close to optimal?
2. policy iteration on an MDP with 50 states and 4 actions. how many improvement steps does it take: about 4, about 50, or about $4^{50}$?
3. value iteration with $\gamma = 0.99$ versus $\gamma = 0.9$ on a task with no terminal states: how many times more iterations?
4. value iteration is still running: its values are far from converged. is its greedy policy also far from optimal?
5. a chain: a reward of $+1$ two steps to the left, $+10$ six steps to the right. which way does the optimal policy go?

## prerequisites check

- [ ] you can evaluate a policy with `policy_evaluation` and compute action values with `q_from_v` (→ topic 02)
- [ ] you know the Bellman expectation equation and why its iteration is a $\gamma$-contraction (→ topic 02, on paper 3)
- [ ] you know what a fixed point is and why a contraction has exactly one (Banach's theorem, → ch 03 · topic 01)

## core ideas

### idea 1: optimal values and the greedy policy

the **optimal value functions** are the best values any policy can achieve, state by state:

$$V^*(s) = \max_\pi V^\pi(s),\qquad Q^*(s, a) = \max_\pi Q^\pi(s, a) = R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V^*(s').$$

it's not obvious that a single policy can be best in *every* state at once, but for finite MDPs it can, and a deterministic one suffices. once you know $Q^*$, acting optimally is trivial: in each state take an action with the largest $Q^*(s, a)$. that is, the **greedy** policy with respect to $Q^*$ is optimal. since the best action's value is the state's value, $V^*(s) = \max_aQ^*(s, a)$, and substituting gives the **Bellman optimality equation**:

$$V^*(s) = \max_a\Big[R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)\,V^*(s')\Big].$$

compare it with topic 02's expectation equation: the average over the policy's actions became a max. that makes the equation nonlinear (no more linear solve), but the operator on the right, $(TV)(s) = \max_a[R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V(s')]$, the **optimality backup**, is still a $\gamma$-contraction in the max norm (on paper 1). so it has exactly one fixed point, $V^*$, and iterating it converges there from anywhere.

> **spec.** `greedy_policy(Q)` returns a float array of the same shape as `Q` $(S, A)$ with a single 1 per row, at the argmax of that row, ties to the lowest action (`np.argmax`). it must not modify `Q`. `bellman_optimality(P, R, V, gamma=0.99)` returns $TV$, shape $(S,)$ (use your `q_from_v`).

> **▶ build: stage 1.** implement `greedy_policy` and `bellman_optimality`.

### idea 2: policy iteration

why does acting greedily help? the **policy improvement theorem**: if $\pi'$ is greedy with respect to $Q^\pi$, then $V^{\pi'}(s) \ge V^\pi(s)$ in every state. the intuition: taking the greedy action for *one* step and then following $\pi$ is at least as good as following $\pi$, by construction; doing it at every step only compounds the gain (on paper 2). and if the greedy policy is no better anywhere, then $V^\pi$ already satisfies the Bellman optimality equation, so $\pi$ is optimal.

that gives **policy iteration**:

1. **evaluate**: $V \leftarrow V^\pi$ (topic 02's `policy_evaluation`).
2. **improve**: $\pi \leftarrow$ greedy with respect to $Q = R + \gamma PV$ (`q_from_v`).
3. stop when the greedy step changes nothing.

each improvement strictly improves the policy unless it's already optimal, and there are finitely many deterministic policies, so policy iteration terminates with an optimal policy after finitely many steps. in practice "finitely many" means *very* few, typically a handful even for large problems (warm-up 2), because each improvement step changes the policy in every state at once. the price is that every step contains a full policy evaluation.

one practical detail. the evaluation is iterative, so $V$ carries an error of order `tol`; two actions that are truly tied can look different by $10^{-12}$, and a greedy step that flips between them would never stop. so the improvement only switches a state's action when another action is better by more than `tol`.

> **spec.** `policy_iteration(P, R, gamma=0.99, *, tol=1e-8)`: start from the greedy policy for the *uniformly random* policy's values (evaluating the random policy first keeps $\gamma = 1$ safe: it reaches the goal). then repeat: $V$ = `policy_evaluation(P, R, policy, gamma, tol=tol)`, $Q$ = `q_from_v(P, R, V, gamma)`; in every state where $\max_aQ(s, a) > Q(s, \pi(s)) + \text{tol}$, switch to the greedy action (ties to the lowest); stop when no state switches (and after at most 1,000 improvements, as a safety net). return `(policy, V)` with a one-hot policy $(S, A)$.

> **🔮 predict:** on the $1\times3$ corridor, the random policy has $V = (-12, -8, 0)$. what policy does one improvement step give, and is it optimal?

> **▶ build: stage 2.** implement `policy_iteration`.

### idea 3: value iteration

policy iteration waits for each evaluation to converge before improving. **value iteration** doesn't wait at all: it applies the optimality backup directly,

$$V_{k+1} = TV_k,\qquad V_{k+1}(s) = \max_a\Big[R(s, a) + \gamma\sum_{s'}P(s'\mid s, a)V_k(s')\Big],$$

and reads off the greedy policy at the end. one sweep costs $O(S^2A)$, the same as one sweep of policy evaluation. because $T$ is a $\gamma$-contraction, $\|V_k - V^*\|_\infty \le \gamma^k\|V_0 - V^*\|_\infty$: the error shrinks by a factor $\gamma$ per sweep, so reaching accuracy $\epsilon$ takes about $\ln(1/\epsilon)/(1 - \gamma)$ sweeps in the worst case. that's ten times more sweeps for $\gamma = 0.99$ than for $\gamma = 0.9$ (warm-up 3). in episodic tasks with absorbing goals the picture is better: value information only has to propagate as far as the longest path to the goal.

when do you stop? you don't know $V^*$, but the contraction lets you bound the error from the last change: if $\|V_{k+1} - V_k\|_\infty < \epsilon$, then $\|V_{k+1} - V^*\|_\infty \le \frac{\gamma\epsilon}{1 - \gamma}$ (on paper 3), and the greedy policy loses at most $\frac{2\gamma\epsilon}{1 - \gamma}$ in any state. note the $1/(1 - \gamma)$: with $\gamma$ close to 1, a small change between sweeps doesn't mean you're close. and the policy is the thing you actually want: it usually becomes optimal long before the values settle (warm-up 4, and the lab).

> **spec.** `value_iteration(P, R, gamma=0.99, *, tol=1e-8, max_iters=10000)`: start from $V = 0$; apply `bellman_optimality` until the largest absolute change is below `tol` or `max_iters` backups have been done; return `(greedy_policy(q_from_v(P, R, V, gamma)), V)` with the last $V$.

> **▶ build: stage 3.** implement `value_iteration`.

### idea 4: one family, and generalized policy iteration

policy iteration and value iteration are two ends of one family. **modified policy iteration** improves greedily, then runs only $m$ sweeps of policy evaluation (starting from the current values) before improving again. with $m = 1$, one greedy step followed by one evaluation sweep of the greedy policy is exactly one optimality backup: value iteration. with $m \to \infty$ the evaluation is exact: policy iteration. in between, a few cheap evaluation sweeps per (more expensive) greedy step usually beat both ends (Puterman & Shin, 1978).

this is a special case of the pattern Sutton & Barto call **generalized policy iteration**: keep *some* value estimate and *some* policy, push the values towards the policy's values (evaluation) and the policy towards greediness (improvement), in any proportion, and the two settle together at the optimum. nearly every RL algorithm in this course is an instance. Monte Carlo control evaluates with sampled returns, SARSA and Q-learning with single sampled backups, actor–critic methods (ch 24) with a neural critic and a gradient step for the policy. the sweeps don't even have to be synchronous: updating states one at a time, in any order, still converges as long as every state keeps being updated (asynchronous DP).

> **spec.** `modified_policy_iteration(P, R, gamma=0.99, *, m=5, tol=1e-8, max_iters=10000)`: start from $V = 0$. each iteration: $\pi$ = greedy for $Q = $ `q_from_v(P, R, V, gamma)`, then $m$ sweeps of $V \leftarrow R_\pi + \gamma P_\pi V$; stop when the iteration changed $V$ by less than `tol` (max norm), or after `max_iters` iterations. return `(policy, V, n_iters)`: the greedy policy for the final $V$, the final $V$, and the number of iterations (an int).

> **🔮 predict:** on a 50-state MDP with no terminal states and $\gamma = 0.999$, how many iterations does value iteration need to reach `tol = 1e-8`? and how many policy evaluations does policy iteration need?

> **▶ build: stage 4.** implement `modified_policy_iteration`.

## worked example

the $1\times3$ corridor from topic 02 (goal at state 2, $-1$ per step), $\gamma = 1$.

**value iteration** from $V_0 = (0, 0, 0)$. each backup takes the best action's one-step lookahead: $V_1 = (-1, -1, 0)$, then $V_2(0) = -1 + \max(V_1(0), V_1(1)) = -2$ and $V_2(1) = -1 + \max(V_1(0), V_1(1), V_1(2)) = -1$, so $V_2 = (-2, -1, 0)$. one more backup changes nothing: $V_3 = V_2$, the change is 0, and value iteration stops after 3 backups with $V^* = (-2, -1, 0)$ and the greedy policy "right" in states 0 and 1 (the goal's actions tie, so it gets action 0).

**the backup of the random policy's values.** with $V = (-12, -8, 0)$: $TV(0) = -1 + \max(-12, -8) = -9$ and $TV(1) = -1 + \max(-12, -8, 0) = -1$, so $TV = (-9, -1, 0)$.

**policy iteration.** evaluating the random policy gives $(-12, -8, 0)$; the greedy actions for those values are "right" in both states (that's the predict callout: one improvement step from the *random* policy already gives the optimal policy here). evaluating "right, right" gives $(-2, -1, 0)$; no action beats "right" by more than `tol`, so it stops after that single evaluation. your functions must reproduce all of this.

## on paper

1. **the optimality backup is a contraction.** show $|\max_af(a) - \max_ag(a)| \le \max_a|f(a) - g(a)|$, and use it to prove $\|TV - TU\|_\infty \le \gamma\|V - U\|_\infty$.

<details><summary>solution</summary>

let $a^*$ maximize $f$. then $\max f - \max g \le f(a^*) - g(a^*) \le \max_a|f(a) - g(a)|$; swap $f$ and $g$ for the other direction. with $f(a) = R(s, a) + \gamma P(\cdot\mid s, a)^\top V$ and $g$ the same with $U$: $|TV(s) - TU(s)| \le \max_a\gamma|P(\cdot\mid s, a)^\top(V - U)| \le \gamma\|V - U\|_\infty$, since each $P(\cdot\mid s, a)$ is a probability vector.

</details>

2. **policy improvement.** let $\pi'$ be greedy for $Q^\pi$, so $Q^\pi(s, \pi'(s)) \ge V^\pi(s)$ for all $s$. show $V^{\pi'} \ge V^\pi$.

<details><summary>solution</summary>

in operator form, the assumption is $T_{\pi'}V^\pi \ge V^\pi$, where $T_{\pi'}V = R_{\pi'} + \gamma P_{\pi'}V$. $T_{\pi'}$ is monotone ($V \ge U$ implies $T_{\pi'}V \ge T_{\pi'}U$, because $P_{\pi'}$ has nonnegative entries). applying it repeatedly: $V^\pi \le T_{\pi'}V^\pi \le T_{\pi'}^2V^\pi \le \dots \to V^{\pi'}$, the fixed point of $T_{\pi'}$. if the inequality is strict in some state, so is the conclusion there.

</details>

3. **when to stop value iteration.** show that $\|V_{k+1} - V_k\|_\infty \le \epsilon$ implies $\|V_{k+1} - V^*\|_\infty \le \frac{\gamma\epsilon}{1 - \gamma}$.

<details><summary>solution</summary>

$\|V_{k+1} - V^*\| = \|TV_k - TV^*\| \le \gamma\|V_k - V^*\| \le \gamma(\|V_k - V_{k+1}\| + \|V_{k+1} - V^*\|)$. rearranging, $(1 - \gamma)\|V_{k+1} - V^*\| \le \gamma\epsilon$.

</details>

4. **a chain with two rewards.** states $0, 1, \dots, 10$; states 0 and 10 are terminal; you start in state 2; moving into 0 pays $+1$, moving into 10 pays $+10$, every other step pays 0. for which $\gamma$ is "left" optimal from state 2?

<details><summary>solution</summary>

going left, the reward arrives on the 2nd step: value $\gamma\cdot1$. going right, on the 8th step: $\gamma^7\cdot10$. left wins when $\gamma > 10\gamma^7$, i.e. $\gamma^6 < 0.1$, i.e. $\gamma < 0.1^{1/6} \approx 0.681$. (states further right switch at larger $\gamma$; the lab draws the whole picture.)

</details>

## experiment

`lab.py` investigates three things with your code:

1. how many iterations value iteration, modified policy iteration and policy iteration need as $\gamma \to 1$.
2. when value iteration's greedy policy becomes optimal, compared with when its values converge.
3. how the optimal policy on a chain with a near small reward and a far big one depends on $\gamma$.

## check your understanding

1. why is it enough to search over deterministic policies?

<details><summary>answer</summary>

the greedy policy for $Q^*$ is deterministic and optimal in every state. a stochastic policy's value is an average of action values, which can't exceed the maximum. (in games against an adversary, or with partial observability, stochastic policies can be strictly better; not in a fully observed MDP.)

</details>

2. (warm-up 1.) why is the greedy policy for the *random* policy's values often already optimal or close to it in a gridworld?

<details><summary>answer</summary>

the random policy's values, although far from $V^*$, still decrease with distance to the goal (with $\gamma = 1$, topic 02's lab), so their greedy direction already points towards the goal. one improvement step uses all of that information at once in every state.

</details>

3. policy iteration needs only a handful of improvements. why isn't it always faster than value iteration?

<details><summary>answer</summary>

each improvement includes a full policy evaluation: a linear solve ($O(S^3)$) or an iterative evaluation that itself needs about $\ln(1/\epsilon)/(1 - \gamma)$ sweeps. counting sweeps, PI does about as much work as VI per improvement. modified policy iteration balances the two.

</details>

4. the greedy policy from value iteration is optimal long before $V$ converges. can you detect the moment it becomes optimal?

<details><summary>answer</summary>

not for free: the policy's stability isn't proof. you can bound its loss by $2\gamma\epsilon/(1 - \gamma)$ from the last change $\epsilon$, or evaluate the greedy policy and check that one greedy step changes nothing (that's policy iteration's stopping test).

</details>

5. what goes wrong with value iteration if $\gamma = 1$ and some policy can collect positive reward forever?

<details><summary>answer</summary>

$V^*$ is infinite, $T$ isn't a contraction, and the values grow without bound: the `max_iters` cap is what stops it. with $\gamma = 1$ you need every policy (or at least the optimal one, with all rewards negative) to reach a terminal state.

</details>

6. why must policy iteration switch actions only when the improvement exceeds `tol`?

<details><summary>answer</summary>

the evaluated $V$ is only accurate to about `tol`; truly tied actions get tiny spurious differences. if the improvement step always took the argmax, it could flip between tied actions forever, and "the policy didn't change" would never become true.

</details>

## common bugs

- **in-place updates without meaning to.** symptom: results depend on the state order. computing `V_new` from `V` and swapping is synchronous DP; updating `V[s]` in place is Gauss–Seidel. both converge, but tests that count iterations assume synchronous sweeps.
- **flip-flopping on ties.** symptom: `policy_iteration` never terminates (or hits its safety cap) on a gridworld. only switch actions for a real improvement.
- **stopping on a small change with $\gamma \approx 1$.** symptom: values that are "converged" but wrong by $\gamma\epsilon/(1 - \gamma)$. use a tighter `tol` or the bound.
- **returning the policy for the previous $V$.** symptom: off-by-one disagreements with the checks. return the greedy policy for the *final* values.
- **evaluating an improper policy with $\gamma = 1$.** symptom: `policy_evaluation` runs to `max_iters` and returns huge negative numbers. start policy iteration from a policy that terminates (the greedy policy of the random policy).

## going deeper

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapter 4 (dynamic programming, generalized policy iteration). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Bellman (1957), *Dynamic Programming*, Princeton University Press. value iteration and the principle of optimality.
- Howard (1960), *Dynamic Programming and Markov Processes*, MIT Press. policy iteration.
- Puterman & Shin (1978), "Modified policy iteration algorithms for discounted Markov decision problems", *Management Science* 24(11). [doi:10.1287/mnsc.24.11.1127](https://doi.org/10.1287/mnsc.24.11.1127)
- Puterman (1994), *Markov Decision Processes*, Wiley, chapter 6 (discounted MDPs: value, policy and modified policy iteration, error bounds). [doi:10.1002/9780470316887](https://doi.org/10.1002/9780470316887)
- Bertsekas (2019), *Reinforcement Learning and Optimal Control*, Athena Scientific. DP as the bridge between control and RL.

## connections

- ← builds on: policy evaluation and `q_from_v` (topic 02), contractions and the Banach fixed-point theorem (ch 03 · topic 01), linear solves (ch 02).
- → used in: the exact solutions every later learner is checked against (topics 04–06 and the capstone), Monte Carlo control and Q-learning as sampled generalized policy iteration (topics 04–05), fitted value iteration and DQN (ch 24 · topic 01), Dyna-Q and planning with a learned model (ch 24 · topic 06).
