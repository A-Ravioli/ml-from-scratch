# temporal-difference learning

## why this matters

Monte Carlo waits. it can't update anything until the episode is over, its estimates carry the noise of every random event in the rest of the episode, and in a task that never ends it can't learn at all. dynamic programming doesn't wait: it updates a state's value from the *estimated* values of its successors. but it needs the model.

**temporal-difference** (TD) learning takes the best of both: learn from sampled experience like MC, and update from estimates of successor states like DP (**bootstrapping**), after every single step. Sutton (1988) introduced it; Q-learning (Watkins, 1989) made it a control algorithm, and it's still, with a neural network in place of the table, the core of DQN (ch 24). this topic builds the whole family: TD(0) for prediction; SARSA, Q-learning and expected SARSA for control; n-step methods and TD($\lambda$) in between TD and Monte Carlo. on the way you'll meet the cliff, where the algorithm that learns the optimal path earns less reward than the one that doesn't, and the distinction between termination and truncation stops being pedantic.

## warm-up: predict first

1. you drive home. at the halfway point you hit a traffic jam, and your estimate of the total trip goes from 30 to 45 minutes. MC updates its estimate of "time from the office" when you get home. when does TD update it?
2. a robot walks next to a cliff while exploring 10% of the time at random. does the *best* path for it stay next to the edge?
3. Q-learning's target uses $\max_aQ(s', a)$, not the action it will actually take. does it need importance sampling to be off-policy?
4. TD(0) bootstraps after one step, Monte Carlo after none (it waits for the end). what's in between, and which should be best?
5. an episode ends because time ran out. what should the TD target be?

## prerequisites check

- [ ] you can run episodes through the `Env` protocol and you know `terminated` from `truncated` (→ topic 02)
- [ ] you know the Bellman expectation and optimality equations (→ topics 02–03)
- [ ] you've built Monte Carlo prediction and control with `epsilon_greedy` (→ topic 04)
- [ ] you know stochastic approximation in the form of the incremental mean $x \leftarrow x + \alpha(\text{target} - x)$ (→ topic 01, ch 06 · topic 04)

## core ideas

### idea 1: TD(0) and the cliff

Monte Carlo moves $V(S_t)$ towards the return $G_t$. the Bellman equation says $V^\pi(s) = \mathbb E[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s]$, so the one-step quantity $R_{t+1} + \gamma V(S_{t+1})$ is also a target with (approximately) the right mean, and it's available one step later. **TD(0)** updates after every step:

$$V(S_t) \leftarrow V(S_t) + \alpha\big[\underbrace{R_{t+1} + \gamma V(S_{t+1}) - V(S_t)}_{\delta_t,\ \text{the TD error}}\big].$$

the target uses the current estimate $V(S_{t+1})$, so it's biased while $V$ is wrong, but its variance is much smaller than the return's: it depends on one random transition, not on the whole rest of the episode. in practice TD usually learns faster than MC on Markov problems (lab, experiment 2). it also learns online, from incomplete episodes, and in continuing tasks. (warm-up 1: TD updates at the halfway point, the moment the jam changes its estimate of the remaining time.)

what's $V(S_{t+1})$ when the episode stopped? if it **terminated**, the next state is terminal and worth 0 by definition: target $R_{t+1}$. if it was **truncated**, the next state is an ordinary state whose future you didn't see, and the right target is still $R_{t+1} + \gamma V(S_{t+1})$ (warm-up 5). that's the reason `step` returns two flags: with bootstrapping, a time limit costs nothing; treating it as termination teaches the agent that time-outs are cliffs (lab, experiment 3).

speaking of cliffs: **cliff walking** (Sutton & Barto, Example 6.6) is a $4\times12$ grid with the start in the bottom-left corner, the goal in the bottom-right, and the ten cells between them a cliff. every step costs $-1$; stepping into the cliff costs $-100$ and sends you back to the start (it doesn't end the episode). the optimal path runs along the edge: 13 steps, return $-13$.

> **spec.** `CliffWalking()` is a `GridWorld(4, 12, start=(3, 0), goal=(3, 11), max_steps=1000)` (a generous time limit, so no episode can run forever) whose transitions *into* a cliff cell $(3, 1), \dots, (3, 10)$ go to the start state with reward $-100$; the cliff cells themselves are unreachable, absorbing, reward 0. the list of cliff states goes in `self.cliff`. `td0_prediction(env, policy, n_episodes, *, alpha=0.1, gamma=0.99, rng=None)`: $V$ zeros; seed the env from `rng` on the first episode as in topic 04; after every step update $V(s)$ with target $r$ if terminated, else $r + \gamma V(s')$ (also when truncated). returns $V$.

> **▶ build: stage 1.** implement `CliffWalking` and `td0_prediction`.

### idea 2: SARSA and Q-learning

for control, learn $Q$ instead of $V$ (topic 04's reason: improvement without a model), act ε-greedily, and update after every step. two choices for the target give two classic algorithms.

**SARSA** (Rummery & Niranjan, 1994) uses the next action the agent *actually takes*: after observing $S_t, A_t, R_{t+1}, S_{t+1}$ it picks $A_{t+1}$ ε-greedily and updates

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\big],$$

(the name is the quintuple it uses). it's **on-policy**: it learns $Q^\pi$ for the ε-greedy policy it follows, which keeps improving: generalized policy iteration one step at a time.

**Q-learning** (Watkins, 1989) uses the best next action instead:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\big[R_{t+1} + \gamma\max_aQ(S_{t+1}, a) - Q(S_t, A_t)\big].$$

that's a sampled version of value iteration's optimality backup, and it learns $Q^*$ whatever policy generates the data, as long as every pair keeps being visited (and the step sizes decay suitably): it's **off-policy** with no importance sampling (warm-up 3), because the target doesn't depend on which action the behaviour takes next.

on the cliff the difference is stark. Q-learning learns the values of the *optimal* policy, so its greedy path hugs the edge; but the agent actually *behaves* ε-greedily, and a random step from the edge goes off the cliff. SARSA learns the values of the ε-greedy policy it follows, including its random steps, so it learns that the edge is dangerous *for an agent that explores*, and prefers a safer path a row or two up (warm-up 2). which is "better" depends on whether you care about the policy you'll deploy (greedy, no exploration) or the reward you collect while learning.

> **spec.** `sarsa(env, n_episodes, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None)` and `q_learning(...)` (same parameters): $Q$ zeros `(n_states, n_actions)`; env seeding as before; actions from `epsilon_greedy(Q[s], epsilon, rng=rng)`. SARSA picks the first action right after `reset`, and after each step picks the next action *unless the episode terminated* (it's needed for the target, also on truncation); the target is $r$ if terminated. Q-learning picks each action at the start of the step. both return `(Q, episode_returns)`, the undiscounted sum of each episode's rewards.

> **🔮 predict:** on the cliff with $\varepsilon = 0.1$, which earns more per episode during learning, SARSA or Q-learning, and by roughly how much?

> **▶ build: stage 2.** implement `sarsa` and `q_learning`.

### idea 3: expected SARSA

SARSA's target has noise from two sources: the environment's transition, and the agent's own random choice of $A_{t+1}$. the second is avoidable, because the agent knows its policy. **expected SARSA** averages over it:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\Big[R_{t+1} + \gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1}, a) - Q(S_t, A_t)\Big].$$

for the ε-greedy policy, $\pi(a\mid s) = \varepsilon/|\mathcal A|$ for every action plus $(1 - \varepsilon)$ shared equally among the greedy actions. its expected update is the same as SARSA's, but with less variance, so it tolerates larger step sizes (van Seijen et al., 2009); in the lab it beats both SARSA and Q-learning on the cliff. it also generalizes Q-learning: with a greedy target policy, $\sum_a\pi(a\mid s')Q(s', a) = \max_aQ(s', a)$. so expected SARSA with a target policy different from the behaviour is the general off-policy one-step TD method, and Q-learning is its greedy special case.

> **spec.** `expected_sarsa(env, n_episodes, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None)`: like Q-learning (action at the start of each step), with the target $r + \gamma\sum_a\pi(a\mid s')Q(s', a)$ for the ε-greedy $\pi$ (ties share the greedy mass equally), or $r$ if terminated. returns `(Q, episode_returns)`.

> **▶ build: stage 3.** implement `expected_sarsa`.

### idea 4: n-step bootstrapping

TD(0) bootstraps after one reward; Monte Carlo after all of them. in between (warm-up 4) is the **n-step return**:

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1}R_{t+n} + \gamma^nQ(S_{t+n}, A_{t+n}),$$

where the bootstrap term is dropped if the episode terminated before $t + n$ (then $G_{t:t+n} = G_t$, the full return). **n-step SARSA** moves $Q(S_t, A_t)$ towards $G_{t:t+n}$, which it can only do $n$ steps later, so the algorithm keeps a short buffer of states, actions and rewards and updates the pair from $n$ steps ago at each step (Sutton & Barto, §7.2). larger $n$ means less bias (more real rewards, less reliance on the estimate) and more variance (more random rewards); also faster propagation: one episode updates the values of the last $n$ steps before a reward, not just one. the best $n$ is usually in the middle, and depends on the step size.

the bookkeeping is the hard part. with time $t$ counting steps taken and $T$ the episode's length (unknown until it ends), the pair to update at step $t$ is $\tau = t - n + 1$; its return sums rewards $R_{\tau+1}, \dots, R_{m}$ with $m = \min(\tau + n, T)$, and adds $\gamma^{m-\tau}Q(S_m, A_m)$ unless the episode *terminated* at $m = T$. once $\tau = T - 1$, every pair of the episode has been updated. a truncated episode bootstraps from $Q(S_T, A_T)$, so the agent picks $A_T$ even though it won't take it.

> **spec.** `n_step_sarsa(env, n_episodes, n=4, *, alpha=0.5, gamma=1.0, epsilon=0.1, rng=None)`: as SARSA (same order of action draws: first action after reset, then the next action after each step unless terminated), with n-step targets as above. with `n=1` it must give exactly the same results as your `sarsa`. returns `(Q, episode_returns)`.

> **🔮 predict:** a chain with one action and rewards $1, 2, 3, 4$; $n = 2$, $\gamma = 0.5$, $\alpha = 1$, $Q$ starting at 0. what is $Q$ after one episode? (the worked example has the answer.)

> **▶ build: stage 4.** implement `n_step_sarsa`.

### idea 5: TD(λ) and eligibility traces

why choose one $n$? the **λ-return** averages all of them with geometrically decaying weights,

$$G_t^\lambda = (1 - \lambda)\sum_{n\ge1}\lambda^{n-1}G_{t:t+n},$$

$\lambda = 0$ giving TD(0) and $\lambda = 1$ giving Monte Carlo. computing it directly (the "forward view") needs the future again. the **backward view** gets the same effect online with an **eligibility trace**: a vector $e$ with one entry per state that records how recently and how often each state was visited. at every step,

$$e \leftarrow \gamma\lambda\,e,\qquad e(S_t) \leftarrow e(S_t) + 1,\qquad V \leftarrow V + \alpha\,\delta_t\,e,$$

with $\delta_t$ the TD(0) error. each TD error is broadcast backwards to all recently visited states, in proportion to their traces. for offline updates the two views are equivalent (Sutton & Barto, §12.2); online they're close. one danger: these **accumulating** traces add 1 on every visit, so a state revisited many times in a short span gets a trace above 1, and with a large $\alpha$ the updates overshoot and can diverge (the lab shows it). replacing traces (set to 1 instead of adding) and "dutch" traces (true online TD($\lambda$), van Seijen & Sutton, 2014) fix that.

> **spec.** `td_lambda_prediction(env, policy, n_episodes, *, lam=0.9, alpha=0.1, gamma=0.99, rng=None)`: TD($\lambda$) with accumulating traces as above, $e$ reset to zeros at the start of every episode, $\delta$ computed like TD(0) (no bootstrap after termination); env seeding as before. with `lam=0` it must equal your `td0_prediction`.

> **▶ build: stage 5.** implement `td_lambda_prediction`.

## worked example

**TD(0).** on the $1\times3$ corridor with the policy "always right", $\alpha = 0.5$, $\gamma = 1$, $V$ starting at 0. episode 1: from 0 to 1, $\delta = -1 + V(1) - V(0) = -1$, so $V(0) = -0.5$; from 1 to the goal (terminated), $\delta = -1 - V(1) = -1$, so $V(1) = -0.5$. episode 2: $\delta = -1 + (-0.5) - (-0.5) = -1$, so $V(0) = -1$; then $\delta = -1 - (-0.5) = -0.5$, so $V(1) = -0.75$. (the true values are $-2$ and $-1$.)

**the three targets.** from state 0, either action leads to state 1 with reward 0; in state 1, action 0 pays $+1$ and action 1 pays $-1$, and the episode ends. with $\alpha = 1$, once $Q(1, \cdot) = (1, -1)$ is learned, the next update of $Q(0, a)$ sets it to the target: Q-learning, $\max(1, -1) = 1$; SARSA, $+1$ or $-1$ depending on the next action sampled; expected SARSA with $\varepsilon = 1$, $\tfrac12(1) + \tfrac12(-1) = 0$, and with $\varepsilon = 0.5$, $0.75(1) + 0.25(-1) = 0.5$.

**n-step returns.** a chain with one action and rewards $1, 2, 3, 4$, then the end; $n = 2$, $\gamma = 0.5$, $\alpha = 1$, $Q$ at 0. in the first episode each pair gets $R_{t+1} + 0.5R_{t+2} + 0.25Q(S_{t+2})$ with $Q$ still 0, and the last two stop at the terminal: $Q = (1 + 1, 2 + 1.5, 3 + 2, 4) = (2, 3.5, 5, 4)$. in the second episode $Q(S_0) = 2 + 0.25\cdot5 = 3.25$. with $n \ge$ the episode length, n-step SARSA is Monte Carlo.

**traces.** a chain $0 \to 1 \to 2 \to$ end with rewards $0, 0, 1$, $\gamma = 1$, $\alpha = 0.5$. the first two TD errors are 0; the last is $\delta = 1$, when the traces are $(\lambda^2, \lambda, 1)$. so one episode gives $V = 0.5\,(\lambda^2, \lambda, 1)$: $(0.5, 0.5, 0.5)$ for $\lambda = 1$, $(0.125, 0.25, 0.5)$ for $\lambda = 0.5$, $(0, 0, 0.5)$ for TD(0). your code must reproduce all of these.

## on paper

1. **the TD error and the MC error.** with $V$ held fixed during an episode, show that the Monte Carlo error is a discounted sum of TD errors: $G_t - V(S_t) = \sum_{k=t}^{T-1}\gamma^{k-t}\delta_k$.

<details><summary>solution</summary>

$G_t - V(S_t) = R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) = \delta_t + \gamma(G_{t+1} - V(S_{t+1}))$. unroll to the end, where $G_T - V(S_T) = 0 - 0$. this identity is what makes eligibility traces work: with $\lambda = 1$, distributing each $\delta_k$ back to earlier states with weight $\gamma^{k-t}$ adds up to the MC update.

</details>

2. **the cliff, by hand.** with $\varepsilon = 0.1$ and four actions, an agent on the edge row takes a random action with probability $0.1$, and one of four random actions is "down". how much does one step along the edge cost in expectation, and why does that make the edge path worse than the path one row up for the ε-greedy policy?

<details><summary>solution</summary>

$P(\text{fall}) = 0.1 \cdot \tfrac14 = 0.025$ per step on the edge; a fall costs $-100$ plus the steps to come back. over the 10 edge cells that's about $10 \times 0.025 \times (100 + \text{a dozen steps}) \approx 28$ extra cost, versus 2 extra steps for walking one row higher (where a random "down" just returns you to the edge row). so for the ε-greedy policy the safer path is better; for the greedy policy the edge path is optimal. SARSA evaluates the first, Q-learning the second.

</details>

3. **expected SARSA with a greedy target is Q-learning.** show that if $\pi(\cdot\mid s')$ puts all its mass on the greedy actions, the expected SARSA target equals the Q-learning target.

<details><summary>solution</summary>

$\sum_a\pi(a\mid s')Q(s', a) = \sum_{a \in \arg\max}\pi(a\mid s')\max_bQ(s', b) = \max_bQ(s', b)$, since the greedy actions all have the max value and their probabilities sum to 1.

</details>

4. **λ-return weights.** show that the weights $(1 - \lambda)\lambda^{n-1}$ sum to 1, and that in an episode that terminates after $T - t$ steps, all the weight on $n \ge T - t$ goes to the full return $G_t$ (with total weight $\lambda^{T-t-1}$).

<details><summary>solution</summary>

$\sum_{n\ge1}(1 - \lambda)\lambda^{n-1} = (1 - \lambda)/(1 - \lambda) = 1$. for $n \ge T - t$, $G_{t:t+n} = G_t$; the weights for those $n$ sum to $(1 - \lambda)\sum_{n\ge T-t}\lambda^{n-1} = \lambda^{T-t-1}$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. SARSA, Q-learning and expected SARSA on the cliff: reward while learning, and the greedy path each one learns.
2. TD($\lambda$) on the 19-state random walk: the error after 10 episodes for a grid of $\lambda$ and $\alpha$.
3. what TD(0) learns when a time limit is reported as termination.

## check your understanding

1. TD's target is biased (it uses the current estimate). why is TD(0) still guaranteed to converge for a fixed policy?

<details><summary>answer</summary>

its expected update is a step of the Bellman expectation operator, a contraction, so the bias shrinks as the estimates improve; with step sizes satisfying $\sum\alpha = \infty$, $\sum\alpha^2 < \infty$ (or a small constant, in mean) tabular TD(0) converges to $V^\pi$ (Sutton, 1988; Dayan, 1992). the bias is a transient, not a fixed offset.

</details>

2. (the predict callout in idea 2.) SARSA versus Q-learning online on the cliff?

<details><summary>answer</summary>

SARSA earns about $-25$ per episode and Q-learning about $-50$ (lab, experiment 1): Q-learning's edge path loses about one in four episodes to a random step off the cliff. with $\varepsilon \to 0$ both converge to the optimal $-13$.

</details>

3. why can Q-learning be off-policy without importance sampling, while off-policy Monte Carlo needs it?

<details><summary>answer</summary>

MC's target is a return generated by the behaviour policy's future actions, so it must be reweighted. Q-learning's target $r + \gamma\max_aQ(s', a)$ only uses the transition from $(s, a)$, which doesn't depend on the policy, plus a max over the estimates: no future behaviour actions appear in it.

</details>

4. your `n_step_sarsa` with `n=1` must equal `sarsa` exactly. why is that a stronger test than "both learn the cliff"?

<details><summary>answer</summary>

it checks the bookkeeping of the n-step buffer (which pair is updated when, which actions are drawn and in what order, the terminal and truncation cases) against an independent implementation, with no statistical slack.

</details>

5. accumulating traces can diverge with a large $\alpha$. why does a random walk make that especially likely?

<details><summary>answer</summary>

a random walk revisits the same states many times within a short span, so their traces accumulate well above 1 (up to about $1/(1 - \gamma\lambda)$). the effective step for that state is $\alpha e(s)$, which can exceed 1 and overshoot, and each overshoot creates a larger TD error.

</details>

6. what does Q-learning's max do when the $Q$ estimates are noisy?

<details><summary>answer</summary>

$\mathbb E[\max_aQ] \ge \max_a\mathbb E[Q]$: the max of noisy estimates is biased upward (maximization bias, Sutton & Barto §6.7). it can make Q-learning prefer actions with high variance. double Q-learning (van Hasselt, 2010) uses one estimate to pick the action and another to evaluate it; DQN's double-Q target in ch 24 is the same idea.

</details>

## common bugs

- **bootstrapping from a terminal state.** symptom: values drift by $\gamma V(\text{terminal})$, which isn't always 0 if you ever update it. use target $r$ when `terminated`.
- **not bootstrapping on truncation.** symptom: values near the time limit are too optimistic (or pessimistic). only `terminated` drops the bootstrap.
- **SARSA updating with the action after next.** symptom: SARSA behaves erratically. pick $A_{t+1}$ once, use it for the target, and then *take* it.
- **n-step off-by-ones.** symptom: `n=1` doesn't match `sarsa`. check the index arithmetic against the worked example.
- **forgetting to reset traces.** symptom: TD($\lambda$) leaks credit across episodes. zero $e$ at every reset.
- **ε-greedy probabilities that don't sum to 1.** symptom: expected SARSA's values drift. with ties, split $1 - \varepsilon$ among the greedy actions.

## going deeper

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapters 6 (TD learning, SARSA, Q-learning, expected SARSA, the cliff), 7 (n-step bootstrapping) and 12 (eligibility traces). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Sutton (1988), "Learning to predict by the methods of temporal differences", *Machine Learning* 3. TD(λ). [doi:10.1007/BF00115009](https://doi.org/10.1007/BF00115009)
- Watkins & Dayan (1992), "Q-learning", *Machine Learning* 8. the convergence proof. [doi:10.1007/BF00992698](https://doi.org/10.1007/BF00992698)
- Rummery & Niranjan (1994), "On-line Q-learning using connectionist systems", Cambridge University Engineering Department technical report CUED/F-INFENG/TR 166. where SARSA comes from.
- van Seijen, van Hasselt, Whiteson & Wiering (2009), "A theoretical and empirical analysis of Expected Sarsa", *IEEE ADPRL*. [doi:10.1109/ADPRL.2009.4927542](https://doi.org/10.1109/ADPRL.2009.4927542)
- van Hasselt (2010), "Double Q-learning", *NeurIPS*. maximization bias and its fix.
- van Seijen & Sutton (2014), "True online TD(λ)", *ICML*. traces that match the forward view online.

## connections

- ← builds on: the `Env` protocol and the two stopping flags (topic 02), Bellman equations and value iteration (topics 02–03), Monte Carlo returns, `epsilon_greedy` and generalized policy iteration (topic 04), incremental means (topic 01).
- → used in: linear semi-gradient TD and SARSA with tile coding (topic 06), the capstone's tabular agents, DQN's Q-learning target and double Q-learning (ch 24 · topic 01), n-step returns and GAE's λ-weighting of TD errors in actor–critic methods (ch 24 · topic 03), Dyna-Q (ch 24 · topic 06).
