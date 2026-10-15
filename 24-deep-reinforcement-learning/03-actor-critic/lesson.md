# actor-critic

## why this matters

REINFORCE judges every action by the return that followed it: a sum of dozens or hundreds of random rewards. topic 02's lab showed what that costs: a single episode's gradient often points the wrong way, and the baseline helped only once a value network had learned something. the next step is to lean on that value network much harder. instead of waiting for the whole return, judge an action by what happened *immediately* plus what the value network predicts for the rest: $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$, the TD error you met in ch 23. that's an **actor–critic** method: the actor is the policy, the critic is a value function that grades it.

the trade-off is the one TD always makes: far less variance, some bias (the critic is wrong, especially early). **generalized advantage estimation** (Schulman et al., 2016) turns that trade-off into a single knob, λ, that slides from one-step TD (λ = 0) to Monte Carlo (λ = 1), and it's how every modern policy-gradient method (A2C, PPO, the RLHF of ch 27) computes advantages. on the engineering side, actor–critic methods run many environments in lockstep, which is where the terminated/truncated distinction stops being an edge case: with auto-resetting environments, a time limit happens somewhere in almost every batch, and the obvious implementation silently bootstraps one episode's value from the *next* episode's first state. in the lab you'll measure GAE's bias–variance trade-off exactly, then sweep λ on CartPole.

## warm-up: predict first

1. the critic is perfect, $V = V^\pi$. is the TD error $r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ an unbiased estimate of the advantage $A^\pi(s_t, a_t)$?
2. λ = 1 gives Monte Carlo advantages $G_t - V(s_t)$. is that unbiased even if the critic is wrong?
3. eight environments run in lockstep; env 3 hits its time limit at step 17 and auto-resets. what's wrong with computing GAE across step 17 as if nothing happened?
4. why add an *entropy bonus* to the loss, and what happens if it's too large?
5. a Gaussian policy over a 2-D action. what's its log-probability of an action, in terms of the per-dimension log-probabilities?

## prerequisites check

- [ ] you know TD errors, n-step returns and λ-returns (→ ch 23 · topic 05)
- [ ] you know REINFORCE, baselines and `reinforce_loss` (→ topic 02)
- [ ] you know the multivariate Gaussian with diagonal covariance and its entropy (→ ch 04, ch 05)
- [ ] you can bootstrap a truncated episode (→ ch 23 · topic 05, topic 01)

## core ideas

### idea 1: generalized advantage estimation

the policy gradient only needs the weight of each action to be an estimate of how much better it was than average: the **advantage** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$. topic 02 used $G_t - V(s_t)$. with a critic you can build cheaper estimates. the one-step TD error

$$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

has expectation exactly $A^\pi(s_t, a_t)$ if $V = V^\pi$ (warm-up 1; on paper 1), and its variance comes from one transition. the $k$-step estimates in between,

$$\hat A_t^{(k)} = \sum_{l=0}^{k-1}\gamma^l\delta_{t+l} = r_{t+1} + \dots + \gamma^{k-1}r_{t+k} + \gamma^kV(s_{t+k}) - V(s_t),$$

trade more real rewards (less reliance on the critic, more variance) against fewer. **GAE** averages all of them with weights $(1 - \lambda)\lambda^{k-1}$, exactly as the λ-return averages $n$-step returns, and the sum telescopes into a discounted sum of TD errors:

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l},\qquad\text{computed backwards as}\qquad \hat A_t = \delta_t + \gamma\lambda\,\hat A_{t+1}.$$

λ = 0 gives $\delta_t$ (low variance, biased by the critic's errors); λ = 1 gives $G_t - V(s_t)$ (unbiased whatever the critic, warm-up 2, but with the full return's variance). in practice λ ≈ 0.9–0.97.

over a rollout of $T$ steps you compute this backwards from $t = T - 1$, bootstrapping the last step from `last_value`, the critic's value of the state after the rollout. where an episode **ended** inside the rollout (flag $d_t = 1$), two things are cut: the bootstrap ($\delta_t = r_{t+1} - V(s_t)$, no $V(s_{t+1})$, which belongs to the next episode) and the accumulation ($\hat A_t$ doesn't include $\hat A_{t+1}$):

$$\delta_t = r_{t+1} + \gamma(1 - d_t)V(s_{t+1}) - V(s_t),\qquad\hat A_t = \delta_t + \gamma\lambda(1 - d_t)\hat A_{t+1}.$$

the **returns** $\hat A_t + V(s_t)$ (a λ-return) are the critic's regression targets. both are labels: compute them without an autograd graph.

> **spec.** `compute_gae(rewards, values, terminated, last_value, *, gamma=0.99, lam=0.95)`: tensors of shape `(T,)` or `(T, N)` (N parallel environments); `terminated[t] = 1` means an episode ended after step $t$ (cut as above); `last_value` has shape `()` or `(N,)`. returns `(advantages, returns)` with `returns = advantages + values`, same shape, computed under `torch.no_grad()`.

> **▶ build: stage 1.** implement `compute_gae`.

### idea 2: one module, two heads

an `ActorCritic` holds both networks and returns both outputs from one call: the action distribution and the state value. sharing hidden layers between them saves computation but couples their gradients (the value loss can drown out the policy's); for small problems two separate MLPs are simpler and more robust, and that's what we'll use.

for **continuous actions** the policy is a Gaussian: the actor outputs the mean $\mu_\theta(s) \in \mathbb R^{d}$, and a learnable, state-independent vector $\log\sigma$ (initialized to 0, so $\sigma = 1$) sets the spread. with a diagonal covariance the dimensions are independent, so the log-probability of an action is the *sum* of the per-dimension log-probabilities (warm-up 5), and the entropy is the sum of per-dimension entropies, $\sum_i\tfrac12\log(2\pi e\sigma_i^2)$. `torch.distributions.Independent(Normal(mu, sigma), 1)` does the summing for you, so `log_prob` returns one number per sample, shape `(B,)`, exactly like a `Categorical`'s, and the rest of the code doesn't care which kind of policy it has. learning $\log\sigma$ rather than $\sigma$ keeps it positive for free.

> **spec.** `ActorCritic(obs_dim, act_dim, *, hidden_sizes=(64, 64), continuous=False)` is an `nn.Module` with a tanh MLP actor (`act_dim` outputs: logits, or the Gaussian mean) and a separate tanh MLP critic (1 output). if `continuous`, a parameter `log_std` of shape `(act_dim,)`, initialized to 0. `forward(obs)` takes `(B, obs_dim)` and returns `(dist, value)`: a `Categorical`, or `Independent(Normal(mean, exp(log_std)), 1)`, both with `log_prob` and `entropy` of shape `(B,)`, and `value` of shape `(B,)`. `act(obs)` takes one ndarray observation and returns `(action, log_prob, value)`: the action a Python `int` (discrete) or an ndarray `(act_dim,)` (continuous), the other two Python floats, all computed without gradients.

> **▶ build: stage 2.** implement `ActorCritic`.

### idea 3: many environments, and time limits

on-policy methods can't reuse old data, so they need lots of fresh data per update, and it should be as decorrelated as possible. the standard trick (Mnih et al., 2016) is to run $N$ copies of the environment in lockstep: at every step one forward pass over the $N$ observations picks $N$ actions, each environment steps, and any environment whose episode ends is reset immediately. after $T$ steps you have a $(T, N)$ batch from $N$ different episodes at different stages, and the state after the last step gives `last_value` for the bootstrap.

auto-resetting creates the trap of warm-up 3. say env $i$ is **truncated** after step $t$. the next observation in the batch is the first state of a *new* episode, so you must cut GAE there ($d_t = 1$). but cutting means "the future is worth nothing", which is right for termination and wrong for truncation. the fix is to bootstrap explicitly, *before* resetting: compute $V(s_{\text{cut}})$ for the observation the episode was cut off in, add $\gamma V(s_{\text{cut}})$ to the reward of that step, and then cut. GAE then sees exactly the target $r + \gamma V(s_{\text{cut}})$ that an uninterrupted rollout would have used (on paper 3). for a **terminated** episode nothing is added: the future really is zero.

> **spec.** `RolloutCollector(envs, model, *, gamma=0.99, seed=None)`: `envs` is a list of $N$ environments, `model` an `ActorCritic`-like module (`model(obs) -> (dist, value)`). `__init__` draws $N$ seeds with `np.random.default_rng(seed).integers(2**31, size=N)` and resets env $i$ with the $i$-th; it keeps the current observations, a running return per env, and a list `episode_returns`. `collect(n_steps)` runs `n_steps` lockstep steps: one batched forward pass (no gradients) to sample actions, then each env steps; when an episode ends it appends the episode's undiscounted return (without any bootstrap) to `episode_returns` (envs in index order), resets that env with `env.reset()`, and, if it was truncated and not terminated, adds `gamma * V(cut-off obs)` to that step's reward. it returns a dict of tensors: `obs` `(T, N, obs_dim)` (the observation each action was taken in), `actions`, `log_probs`, `values`, `rewards`, `dones` (1.0 where an episode ended for either reason), all `(T, N)` (continuous actions `(T, N, act_dim)`), and `last_value` `(N,)`, the value of each env's observation after the last step. a later `collect` continues where the last one stopped. also keep `episode_ends`: for each episode in `episode_returns`, the total number of environment steps taken so far over all envs, counted as $N$ times the number of lockstep steps including the current one.

> **🔮 predict:** a single environment truncated at its third step, rewards 1, 1, 1, $V(s_3) = 30$, $\gamma = 0.5$, λ = 1. after folding the bootstrap into the third reward, what returns does GAE produce for the three steps? (the worked example has them.)

> **▶ build: stage 3.** implement `RolloutCollector`.

### idea 4: advantage actor–critic (A2C)

with advantages and returns in hand, one update minimizes

$$\mathcal L = \underbrace{-\frac1B\sum_i\hat A_i\log\pi_\theta(a_i\mid s_i)}_{\text{policy}} + c_v\underbrace{\frac1B\sum_i\big(V_\phi(s_i) - \hat V_i\big)^2}_{\text{value}} - c_e\underbrace{\frac1B\sum_i\mathcal H\big[\pi_\theta(\cdot\mid s_i)\big]}_{\text{entropy}}$$

over the flattened $(T\cdot N)$ batch, with the advantages and returns detached. the **entropy bonus** (warm-up 4) rewards keeping the policy spread out: policy gradients tend to collapse onto whichever action got lucky early, and once a probability is near 0 the action is never sampled again, so the gradient can't bring it back. a small $c_e$ (0.01) delays that collapse; a large one keeps the policy random forever. clipping the global gradient norm (to 0.5) guards against the occasional huge step that a bad batch would otherwise cause.

this is **A2C**, the synchronous version of Mnih et al.'s A3C (asynchronous workers each updating a shared network). the synchronous version turned out to be as good and much simpler, and it's what PPO (next topic) builds on: same rollouts, same GAE, a different loss.

> **spec.** `a2c_loss(dist, actions, advantages, returns, values, *, vf_coef=0.5, ent_coef=0.01)` returns `(loss, info)`: the loss above with $c_v$ = `vf_coef` and $c_e$ = `ent_coef` (the value term is the plain mean squared error), `advantages` and `returns` detached, and `info` a dict of Python floats `policy_loss`, `value_loss` (the MSE, without the coefficient), `entropy`. `train_a2c(make_env, total_steps, *, n_envs=8, n_steps=16, gamma=0.99, lam=0.95, lr=7e-4, vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, hidden_sizes=(64, 64), continuous=False, seed=None)` builds `n_envs` environments with `make_env()`, an `ActorCritic` and Adam, and a `RolloutCollector` (with `seed`); then for `max(1, total_steps // (n_envs * n_steps))` iterations: collect `n_steps`, compute GAE, one gradient step on `a2c_loss` over the whole batch with the gradient norm clipped to `max_grad_norm`. torch is seeded from `seed` before building the model. returns `(model, history)` with the collector's `episode_returns` and `episode_ends` and per-update lists `policy_loss`, `value_loss`, `entropy`.

> **🔮 predict:** A2C on CartPole with λ = 0 (one-step TD advantages) and with λ = 1 (Monte Carlo advantages). which learns faster?

> **▶ build: stage 4.** implement `a2c_loss` and `train_a2c`.

## worked example

**GAE by hand.** rewards $(1, 0, 2)$, values $(0.5, 1, -1)$, no episode ends, `last_value` $= 3$, $\gamma = 0.9$, λ = 0.5 (so $\gamma\lambda = 0.45$). the TD errors are

$$\delta_0 = 1 + 0.9\cdot1 - 0.5 = 1.4,\quad \delta_1 = 0 + 0.9\cdot(-1) - 1 = -1.9,\quad \delta_2 = 2 + 0.9\cdot3 - (-1) = 5.7,$$

and backwards: $\hat A_2 = 5.7$, $\hat A_1 = -1.9 + 0.45\cdot5.7 = 0.665$, $\hat A_0 = 1.4 + 0.45\cdot0.665 = 1.69925$. the returns are $\hat A + V = (2.19925, 1.665, 4.7)$.

**a cut.** rewards $(1, 1, 1)$, values $(2, 3, 4)$, an episode ended after step 1, `last_value` 5, $\gamma = \lambda = 1$: $\delta_2 = 1 + 5 - 4 = 2$, $\delta_1 = 1 - 3 = -2$ (no bootstrap), $\delta_0 = 1 + 3 - 2 = 2$; $\hat A = (2 + (-2), -2, 2) = (0, -2, 2)$: the accumulation stops at the cut too.

**a time limit.** one environment, reward 1 per step, truncated at its third step in a state with $V = 30$, $\gamma = 0.5$. the collector stores rewards $(1, 1, 1 + 0.5\cdot30) = (1, 1, 16)$ and `dones` $(0, 0, 1)$. GAE with λ = 1 then gives returns $16$, $1 + 0.5\cdot16 = 9$, $1 + 0.5\cdot9 = 5.5$: exactly $1 + 0.5 + 0.25 + 0.125\cdot30$ for the first step, the return of the uninterrupted problem with the critic standing in for the unseen future.

your code must reproduce all three.

## on paper

1. **the TD error estimates the advantage.** show that if $V = V^\pi$, then $\mathbb E[\delta_t\mid s_t, a_t] = A^\pi(s_t, a_t)$. what goes wrong if $V \neq V^\pi$?

<details><summary>solution</summary>

$\mathbb E[r_{t+1} + \gamma V^\pi(s_{t+1})\mid s_t, a_t] = Q^\pi(s_t, a_t)$ by the Bellman equation, so $\mathbb E[\delta_t\mid s_t, a_t] = Q^\pi - V^\pi = A^\pi$. with an approximate $V$, the error is $\gamma\,\mathbb E[V(s_{t+1}) - V^\pi(s_{t+1})] - (V(s_t) - V^\pi(s_t))$; the second term is action-independent (a baseline, harmless in expectation), but the first biases the gradient. GAE with λ shrinks that bias term by weighting later critic errors by $(\gamma\lambda)^k$.

</details>

2. **the telescoping sum.** show that $\sum_{l=0}^{k-1}\gamma^l\delta_{t+l} = \sum_{l=0}^{k-1}\gamma^lr_{t+l+1} + \gamma^kV(s_{t+k}) - V(s_t)$, and deduce that GAE with λ = 1 on an episode that terminates is $G_t - V(s_t)$.

<details><summary>solution</summary>

each $\gamma^lV(s_{t+l+1})$ from $\delta_{t+l}$ cancels with $-\gamma^{l+1}V(s_{t+l+1})$ from $\delta_{t+l+1}$; only the first $-V(s_t)$ and the last $\gamma^kV(s_{t+k})$ survive. with λ = 1, sum to the end of the episode, where the bootstrap is 0: $\hat A_t = G_t - V(s_t)$.

</details>

3. **the time-limit trick.** an episode is truncated after step $t$ in state $s'$. show that GAE on the modified data ($r'_{t+1} = r_{t+1} + \gamma V(s')$, $d_t = 1$) gives the same $\hat A_t$, $\hat A_{t-1}, \dots$ as an uninterrupted rollout in which the agent's future after $s'$ contributes exactly $V(s')$, i.e. where $\hat A_{t+1} = 0$.

<details><summary>solution</summary>

modified: $\delta'_t = r_{t+1} + \gamma V(s') - V(s_t)$ and $\hat A_t = \delta'_t$ (cut). uninterrupted with $\hat A_{t+1} = 0$: $\delta_t = r_{t+1} + \gamma V(s') - V(s_t)$ and $\hat A_t = \delta_t + \gamma\lambda\cdot0$. the same, and all earlier steps only see $\hat A_t$ and their own TD errors, so they match too. the only thing lost is the real data after $s'$, which nobody has.

</details>

4. **the entropy of a Gaussian policy.** for $\pi = \mathcal N(\mu, \operatorname{diag}(\sigma^2))$ in $d$ dimensions, show $\mathcal H = \sum_i(\tfrac12\log(2\pi e) + \log\sigma_i)$. what does the gradient of the entropy bonus do to $\log\sigma$?

<details><summary>solution</summary>

the dimensions are independent, so the entropy is the sum of 1-D entropies, $\tfrac12\log(2\pi e\sigma_i^2) = \tfrac12\log(2\pi e) + \log\sigma_i$. $\partial\mathcal H/\partial\log\sigma_i = 1$: the bonus pushes every $\log\sigma_i$ up at a constant rate $c_e$, independent of the state. the policy gradient pulls it down where precision pays; the balance sets the exploration noise.

</details>

## experiment

`lab.py` investigates:

1. the bias and variance of GAE's targets as a function of λ, on a random walk whose true values are known, with a perfect, a shrunk and a noisy critic.
2. A2C on CartPole for five values of λ, over several seeds.

## check your understanding

1. why do A2C's advantages come from the critic's values computed *during* the rollout, while the value loss uses values recomputed at update time?

<details><summary>answer</summary>

the advantages and returns are labels for this batch, fixed once computed (no gradients). the value loss needs $V_\phi(s)$ *with* gradients, from a fresh forward pass. in A2C the network hasn't changed in between, so the numbers agree; in PPO (several epochs per batch) they don't, and the rollout values become "old values".

</details>

2. why is λ = 0 especially bad early in training?

<details><summary>answer</summary>

the critic starts random, so $\delta_t$ is mostly the critic's error: the actor is graded by noise. information about actual rewards enters only one step at a time, and with $\gamma = 0.99$ it takes many updates to propagate. larger λ lets real rewards from many steps ahead reach the advantage directly (lab, experiment 2).

</details>

3. (the predict callout in idea 3.) the returns of the truncated episode?

<details><summary>answer</summary>

$(5.5, 9, 16)$: the stored third reward is $1 + 0.5\cdot30 = 16$, and GAE with λ = 1 discounts it back: $1 + 0.5\cdot16 = 9$, $1 + 0.5\cdot9 = 5.5$.

</details>

4. you forget the time-limit trick and just set `dones = terminated or truncated`. on CartPole, what goes wrong, and when?

<details><summary>answer</summary>

nothing, until the agent gets good: then every episode ends at the 500-step limit, and the last steps of each episode are treated as terminal with value 0. the critic learns that states late in an episode are worth little, the advantages near the limit become negative, and the agent is pushed away from behaviour that keeps it alive; learning curves wobble near the top. it's a classic "it learns, but plateaus below the max" bug.

</details>

5. why use $N$ environments instead of one environment for $N$ times as many steps?

<details><summary>answer</summary>

the batch then contains $N$ independent episodes at different stages instead of one long correlated stretch, so each update's gradient is less correlated with the last; and the forward pass for action selection is batched, which is far faster per step. for an on-policy method it's the cheapest way to decorrelate data without a replay buffer.

</details>

6. A2C's policy loss uses $\hat A$ with gradients cut. what would happen if gradients flowed through $\hat A$ into the critic?

<details><summary>answer</summary>

the policy loss would push the critic to change $V$ so as to make $-\hat A\log\pi$ small, e.g. making advantages of likely actions large, which has nothing to do with predicting returns. the critic stops being a good baseline, and the actor's gradient becomes biased in a way no one designed.

</details>

## common bugs

- **GAE that ignores `dones`.** symptom: advantages leak across episode boundaries; values of first states get credit for the previous episode's end.
- **bootstrapping from the reset observation.** symptom: subtly wrong values near time limits. bootstrap from the observation the episode was cut off in, *before* resetting.
- **values of shape `(B, 1)`.** symptom: the value loss broadcasts `(B, 1) - (B,)` to `(B, B)` and trains on nonsense without an error. squeeze the critic's output.
- **summing log-probabilities over the batch for a Gaussian.** symptom: works for discrete, breaks for continuous. use `Independent(..., 1)`, so `log_prob` sums over action dimensions only.
- **forgetting `no_grad` during rollouts.** symptom: memory grows with every step, then the backward pass goes through the whole rollout.
- **an entropy coefficient that's too large.** symptom: entropy stays at its maximum and returns stay random.

## going deeper

- Schulman, Moritz, Levine, Jordan & Abbeel (2016), "High-dimensional continuous control using generalized advantage estimation", *ICLR*. GAE. [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)
- Mnih et al. (2016), "Asynchronous methods for deep reinforcement learning", *ICML*. A3C, and the entropy bonus. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)
- Konda & Tsitsiklis (2000), "Actor-critic algorithms", *NeurIPS* 12. the two-timescale analysis of learning a critic and an actor at once. [papers.nips.cc/paper/1786](https://papers.nips.cc/paper/1786-actor-critic-algorithms)
- Pardo, Tavakoli, Levdik & Kormushev (2018), "Time limits in reinforcement learning", *ICML*. partial-episode bootstrapping. [arXiv:1712.00378](https://arxiv.org/abs/1712.00378)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §13.5 (actor–critic methods) and §12.1 (the λ-return). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

## connections

- ← builds on: TD errors and λ-returns (ch 23 · topic 05), REINFORCE and baselines (topic 02), `CartPole` and bootstrapping on truncation (topic 01), diagonal Gaussians (ch 04).
- → used in: PPO is A2C's rollouts and GAE with a clipped loss and several epochs per batch (topic 04); the capstone's PPO agent; `compute_gae` computes the per-token advantages of RLHF (ch 27 · topic 04).
