# continuous control

## why this matters

a robot arm's joints take torques, a car takes a steering angle, a drone takes four rotor speeds: real control problems have continuous actions. DQN can't handle them, because its target needs $\max_aQ(s, a)$, and maximizing over a continuous action at every step is an optimization problem in its own right. A2C and PPO can (a Gaussian policy, topic 03), but they're on-policy: every transition is used for a few epochs and discarded, which is fine in a simulator that produces a million steps a minute and painful on a physical robot.

the methods of this topic are **off-policy actor–critics for continuous actions**: a Q-function learned from a replay buffer, as in DQN, plus an actor that does the maximization for you by gradient ascent on $Q$. DDPG (Lillicrap et al., 2016) does it with a deterministic actor; TD3 (Fujimoto et al., 2018) fixes DDPG's overestimation with two critics; **soft actor–critic** (SAC; Haarnoja et al., 2018) adds the idea that makes it the default for continuous control: optimize reward *plus entropy*, so the policy stays as random as it can afford to be. that one change gives exploration that adapts itself, robustness, and stable learning, at the price of a new hyperparameter, the **temperature** α, which SAC can tune automatically. you'll build the classic pendulum swing-up task, the squashed Gaussian actor and twin critics, SAC's three losses and the agent; in the lab you'll turn the temperature knob and watch what it does to a bimodal reward and to the pendulum.

## warm-up: predict first

1. a policy outputs $u \sim \mathcal N(\mu, \sigma^2)$ and acts with $a = \tanh(u)$, so actions stay in $(-1, 1)$. is $\log\pi(a)$ the Gaussian's log-density at $u$?
2. a reward has two equally good actions, $a = -0.5$ and $a = +0.5$, with a valley between them. where does a Gaussian policy that maximizes reward put its mean? one that maximizes reward plus entropy?
3. two independent noisy critics estimate the same $Q$. is the *min* of the two an underestimate, an overestimate, or unbiased?
4. the pendulum's episodes are always cut off after 200 steps. which flag ends them, and what does SAC's target do with it?
5. a temperature of α = 0 turns SAC into what?

## prerequisites check

- [ ] you've built DQN's replay buffer, target networks and `ema_update` soft updates (→ topic 01)
- [ ] you know Gaussian policies and `log_prob` summed over action dimensions (→ topic 03)
- [ ] you know the change-of-variables formula for densities (→ ch 04, ch 21 · flows)
- [ ] you know the reparameterization trick (→ ch 12 · topic 05, ch 21 · VAEs)
- [ ] you know entropy and the Boltzmann distribution $p \propto e^{f/T}$ (→ ch 05)

## core ideas

### idea 1: the pendulum

the classic continuous-control test bed is the **inverted pendulum swing-up**: a pendulum hangs from a pivot, and a motor applies a torque $u \in [-2, 2]$. the goal is to swing it up and balance it upright, but the motor is too weak to lift it directly: the agent has to pump energy in by swinging back and forth first, then catch it at the top. the state is the angle $\theta$ (0 = upright) and angular velocity $\dot\theta$, with the dynamics of a rod of mass $m = 1$ and length $l = 1$ under gravity $g = 10$:

$$\dot\theta_{t+1} = \operatorname{clip}\Big(\dot\theta_t + \Big(\frac{3g}{2l}\sin\theta_t + \frac{3}{ml^2}u_t\Big)\Delta t,\ -8, 8\Big),\qquad \theta_{t+1} = \theta_t + \dot\theta_{t+1}\Delta t,$$

with $\Delta t = 0.05$ (semi-implicit Euler: the angle uses the *new* velocity). the reward is a quadratic cost, $r_t = -\big(\bar\theta_t^2 + 0.1\dot\theta_t^2 + 0.001u_t^2\big)$, with $\bar\theta$ the angle wrapped into $[-\pi, \pi)$; it's at most 0 (upright, still, no effort) and about $-16$ hanging down. the agent observes $(\cos\theta, \sin\theta, \dot\theta)$, not $\theta$ itself: the angle is circular, and $\cos/\sin$ avoids a fake discontinuity at $\pm\pi$. episodes never terminate; they're truncated after 200 steps, so every transition bootstraps (warm-up 4). a good policy earns about $-150$ per episode (most of it the unavoidable cost of the swing-up), a random one about $-1200$. this is exactly Gymnasium's `Pendulum-v1`.

> **spec.** `angle_normalize(x)` returns `((x + pi) % (2 pi)) - pi`. `Pendulum(*, max_steps=200)` subclasses `mlfs.rl.Env` with `observation_space = Box((-1, -1, -8), (1, 1, 8))` and `action_space = Box(-2, 2, (1,))`; the state is `self.state = (theta, theta_dot)` (a float array) with step counter `self.t`. `reset(seed=None)` draws the state with `self.np_random.uniform((-pi, -1), (pi, 1))` and returns `(obs, {})` with obs $= (\cos\theta, \sin\theta, \dot\theta)$. `step(action)` takes an array of shape `(1,)` (or a float), clips the torque to $[-2, 2]$, computes the reward from the state *before* the step, applies the dynamics above, and returns `(obs, reward, False, truncated, {})` with `truncated` when `t` reaches `max_steps`.

> **▶ build: stage 1.** implement `angle_normalize` and `Pendulum`.

### idea 2: actors for bounded actions, and two critics

**DDPG's idea.** with a deterministic actor $a = \mu_\theta(s)$, "maximize $Q$ over actions" becomes "move the actor uphill on $Q$": $\nabla_\theta Q(s, \mu_\theta(s)) = \nabla_aQ(s, a)\,\nabla_\theta\mu_\theta(s)$, the **deterministic policy gradient** (Silver et al., 2014). the critic learns from replay like DQN, with the target's max replaced by the actor's choice: $y = r + \gamma Q'(s', \mu'(s'))$. exploration comes from noise added to the actions.

**TD3's fixes.** DDPG overestimates badly, for the same reason Q-learning does (topic 01), and worse, because the actor actively seeks out actions where the critic's errors are positive. TD3 learns **two** critics and uses the smaller of their estimates in the target, $y = r + \gamma\min_{j}Q'_j(s', a')$: if one critic is too optimistic about an action, the other usually isn't (warm-up 3: the min of two noisy unbiased estimates is biased *downwards*, which is the safer error). SAC inherits the twin critics.

**the squashed Gaussian.** SAC's actor is stochastic: a network outputs a mean $\mu(s)$ and a log standard deviation $\log\sigma(s)$ (clamped to $[-20, 2]$ to keep it sane), samples $u = \mu + \sigma\epsilon$ with $\epsilon \sim \mathcal N(0, I)$, and squashes: $a = a_{\max}\tanh(u)$, so actions always respect the bounds. the log-probability of the *squashed* action needs the change-of-variables correction (warm-up 1):

$$\log\pi(a\mid s) = \log\mathcal N(u;\mu, \sigma^2) - \sum_i\log\big(1 - \tanh^2(u_i)\big).$$

$1 - \tanh^2u$ underflows to 0 for $|u| \gtrsim 10$, so compute it in log space with the identity $\log(1 - \tanh^2u) = 2\big(\log2 - u - \operatorname{softplus}(-2u)\big)$ (on paper 1). writing $u = \mu + \sigma\epsilon$ (the reparameterization trick) makes the action a differentiable function of the actor's parameters, which is how SAC's actor loss gets its gradient. the scale $a_{\max}$ only shifts the log-density by a constant, $-\log a_{\max}$ per dimension; Spinning Up's convention, which the spec follows (and the checks pin), leaves it out.

> **spec.** `SquashedGaussianActor(obs_dim, act_dim, *, hidden_sizes=(256, 256), act_limit=1.0)` is an `nn.Module`: an `MLP` with `2 * act_dim` outputs, the mean and the log-std (clamped to $[-20, 2]$). `forward(obs, deterministic=False)` returns `(action, log_prob)`: `action` $= a_{\max}\tanh(u)$ of shape `(B, act_dim)` with $u = \mu + \sigma\epsilon$ (reparameterized; $u = \mu$ if `deterministic`), and `log_prob` of shape `(B,)` with the tanh correction above (computed stably) and no $-\log a_{\max}$ term. `TwinQ(obs_dim, act_dim, *, hidden_sizes=(256, 256))` holds two separate `MLP`s from `obs_dim + act_dim` to 1; `forward(obs, act)` concatenates and returns `(q1, q2)`, each `(B,)`.

> **▶ build: stage 2.** implement `SquashedGaussianActor` and `TwinQ`.

### idea 3: maximum entropy, and SAC's three losses

SAC optimizes a different objective: reward plus the policy's entropy at every state visited,

$$J(\pi) = \mathbb E_\pi\Big[\sum_t\gamma^t\big(r_{t+1} + \alpha\,\mathcal H[\pi(\cdot\mid s_t)]\big)\Big].$$

the agent is paid for acting randomly, at an exchange rate α (the **temperature**). where one action is clearly best, the reward term dominates and the policy sharpens; where several actions are about equally good, the entropy term keeps them all alive (warm-up 2). the optimal policy of this objective is a Boltzmann distribution over the soft Q-values, $\pi(a\mid s) \propto \exp(Q(s, a)/\alpha)$ (on paper 2), which is where the name "temperature" comes from; α → 0 recovers ordinary RL (warm-up 5: SAC with α = 0 is a TD3-like stochastic actor–critic).

the soft Bellman backup includes the entropy of the next action: with $a' \sim \pi(\cdot\mid s')$ freshly sampled from the current actor,

$$y = r + \gamma(1 - d)\Big(\min_jQ_{\bar\phi_j}(s', a') - \alpha\log\pi(a'\mid s')\Big),$$

($-\log\pi(a'\mid s')$ is a one-sample estimate of the entropy at $s'$; $d$ is `terminated`, so truncated episodes bootstrap as always.) the three losses are:

- **critics**: $\sum_j\hat{\mathbb E}[(Q_{\phi_j}(s, a) - y)^2]$, with $y$ computed without gradients and $a$ from the buffer;
- **actor**: $\hat{\mathbb E}_s\big[\alpha\log\pi(\tilde a\mid s) - \min_jQ_{\phi_j}(s, \tilde a)\big]$, with $\tilde a$ a fresh *reparameterized* sample from the current actor, so the gradient flows through $Q$ into the actor (the stochastic version of DDPG's update);
- **temperature** (optional): treat the entropy as a constraint, "at least $\bar{\mathcal H}$ on average", and adjust α by gradient descent on $-\hat{\mathbb E}\big[\alpha(\log\pi(\tilde a\mid s) + \bar{\mathcal H})\big]$ (Haarnoja et al., 2018b). if the policy's entropy $-\log\pi$ is above the target, the gradient lowers α; below it, α rises. a common target is $\bar{\mathcal H} = -\dim(\mathcal A)$. learning $\log\alpha$ keeps α positive.

> **spec.** `sac_critic_target(rewards, terminated, next_q1, next_q2, next_log_prob, *, gamma=0.99, alpha=0.2)` returns $y$ above, computed under `torch.no_grad()`. `sac_actor_loss(q1, q2, log_prob, alpha)` returns $\operatorname{mean}(\alpha\log\pi - \min(Q_1, Q_2))$. `sac_alpha_loss(log_alpha, log_prob, target_entropy)` returns $-\operatorname{mean}\big(e^{\log\alpha}(\log\pi + \bar{\mathcal H})\big)$ with `log_prob` detached.

> **🔮 predict:** two critics say $Q'_1 = 2$, $Q'_2 = 1.5$ for the next state, whose sampled action has $\log\pi = -1$; $r = 1$, $\gamma = 0.9$, α = 0.2. what's the target? is it larger or smaller than the non-soft target $r + \gamma\min Q'$, and why?

> **▶ build: stage 3.** implement the three losses.

### idea 4: the SAC agent

the agent combines everything: act with a fresh sample from the actor (that's the exploration), store transitions with `terminated`, and after a warm-up of `learning_starts` steps with uniformly random actions (so the critics see the whole action box before the actor narrows it), take one gradient step on each loss per environment step, then soft-update the target critics, $\bar\phi \leftarrow (1 - \tau)\bar\phi + \tau\phi$ with $\tau = 0.005$ (your `ema_update` with `decay = 1 - tau`). there's no target actor: the next action in the target comes from the current one. off-policy learning with replay makes SAC far more sample-efficient than PPO (thousands rather than hundreds of thousands of steps on the pendulum), at the cost of a gradient step (three, in fact) per environment step.

> **spec.** `train_sac(env, n_steps, *, gamma=0.99, tau=0.005, lr=3e-4, batch_size=256, buffer_size=100_000, learning_starts=1000, alpha=0.2, target_entropy=None, hidden_sizes=(256, 256), seed=None)` runs exactly `n_steps` steps of a Box-observation, Box-action env (symmetric bounds; `act_limit` = the largest `action_space.high`). `alpha` is a float (fixed) or `'auto'` (learned, starting at α = 1, with $\bar{\mathcal H}$ = `target_entropy`, default $-\dim(\mathcal A)$). for $t <$ `learning_starts` the action is `env.action_space.sample(rng)`; afterwards a stochastic sample from the actor. a continuous `ReplayBuffer` stores `terminated`. once `t + 1 >= learning_starts`, every step: one Adam step (`lr`) each on the critic loss, the actor loss and (if auto) the temperature loss, then a soft target update with `tau`. reproducible from `seed` (NumPy generator for the env, the warm-up actions and the buffer; torch seeded before building the networks). returns `(actor, history)` with `episode_returns`, `episode_ends`, `alpha` (the temperature at each episode end after learning starts), `critic` (the `TwinQ`) and `critic_target` (its target copy).

> **🔮 predict:** on the pendulum, which temperature learns the swing-up best: α = 0, 0.2, 1, or the automatically tuned one?

> **▶ build: stage 4.** implement `train_sac`.

## worked example

**the squashed density.** a 1-D actor with $\mu = 0$, $\sigma = 1$ samples $u = 0.5$. the Gaussian log-density is $-\tfrac12\log2\pi - \tfrac12(0.5)^2 = -1.0439$. the action is $\tanh(0.5) = 0.4621$, and $\log(1 - 0.4621^2) = \log0.7864 = -0.2403$ (the stable form gives $2(0.6931 - 0.5 - \operatorname{softplus}(-1)) = 2(0.6931 - 0.5 - 0.3133) = -0.2403$ too). so $\log\pi(a) = -1.0439 + 0.2403 = -0.8037$: squashing *raises* the density near 0, where tanh compresses the line, and (much more) near $\pm1$.

**the critic target.** $r = (1, 0.5)$, terminated $= (0, 1)$, $Q'_1 = (2, 3)$, $Q'_2 = (1.5, 4)$, $\log\pi(a'\mid s') = (-1, 0.5)$, $\gamma = 0.9$, α = 0.2: $y_0 = 1 + 0.9(\min(2, 1.5) - 0.2\cdot(-1)) = 1 + 0.9\cdot1.7 = 2.53$, and $y_1 = 0.5$ (terminated).

**the actor loss.** $Q_1 = (1, 3)$, $Q_2 = (2, 2)$, $\log\pi = (-1, 0.5)$, α = 0.1: $\tfrac12\big[(0.1\cdot(-1) - 1) + (0.1\cdot0.5 - 2)\big] = \tfrac12(-1.1 - 1.95) = -1.525$. the gradient reaches only the smaller $Q$ of each pair.

**the temperature loss.** α = 0.5, $\log\pi = (-1, -3)$ (entropy estimate 2), $\bar{\mathcal H} = -1$: $-0.5\cdot\tfrac12\big((-1 - 1) + (-3 - 1)\big) = 1.5$, and its gradient with respect to $\log\alpha$ is also $+1.5$: descent lowers α, because the policy is more random than the target asks.

your code must reproduce these numbers.

## on paper

1. **the tanh correction, stably.** derive $\log\pi(a) = \log\mathcal N(u) - \log(1 - \tanh^2u)$ for $a = \tanh u$, and prove $\log(1 - \tanh^2u) = 2(\log2 - u - \operatorname{softplus}(-2u))$.

<details><summary>solution</summary>

change of variables: $p_a(a) = p_u(u)/|da/du|$ with $da/du = 1 - \tanh^2u$. for the identity: $1 - \tanh^2u = \operatorname{sech}^2u = 4/(e^u + e^{-u})^2 = 4e^{-2u}/(1 + e^{-2u})^2$, so $\log(1 - \tanh^2u) = 2\log2 - 2u - 2\log(1 + e^{-2u}) = 2(\log2 - u - \operatorname{softplus}(-2u))$. for large positive $u$, softplus$(-2u) \to 0$ and the expression is $\approx 2\log2 - 2u$: finite, no underflow. (for large negative $u$, softplus$(-2u) \approx -2u$, giving $\approx 2\log2 + 2u$, also fine.)

</details>

2. **the Boltzmann policy.** for a single state with soft values $Q(a)$, show that the policy maximizing $\mathbb E_\pi[Q(a)] + \alpha\mathcal H[\pi]$ is $\pi(a) \propto e^{Q(a)/\alpha}$, and that the maximum is $\alpha\log\sum_ae^{Q(a)/\alpha}$.

<details><summary>solution</summary>

$\mathbb E_\pi[Q] + \alpha\mathcal H[\pi] = -\alpha\,\text{KL}(\pi\,\|\,\pi^*) + \alpha\log Z$ with $\pi^* = e^{Q/\alpha}/Z$ and $Z = \sum_ae^{Q(a)/\alpha}$: expand $\text{KL}(\pi\,\|\,\pi^*) = \sum\pi\log\pi - \sum\pi(Q/\alpha - \log Z)$. the KL is minimized (at 0) by $\pi = \pi^*$, leaving $\alpha\log Z$, the "soft max" of $Q$. as α → 0 it approaches $\max_aQ(a)$ and $\pi^*$ the greedy policy. SAC's actor loss is (an estimate of) exactly this KL, projected onto squashed Gaussians.

</details>

3. **the min of two critics.** let $\hat Q_1, \hat Q_2$ be independent estimates of the same value $q$, each $\mathcal N(q, \sigma^2)$. compute $\mathbb E[\min(\hat Q_1, \hat Q_2)]$.

<details><summary>solution</summary>

$\min(X, Y) = \tfrac12(X + Y - |X - Y|)$, and $\mathbb E|X - Y| = 2\sigma/\sqrt\pi$, so $\mathbb E\min = q - \sigma/\sqrt\pi \approx q - 0.56\sigma$: an underestimate of the same size as the max's overestimate (topic 01, on paper 2). the actor's search for high-$Q$ actions pushes the other way, and the two roughly cancel; underestimation is also the safer failure, because an agent that underrates an action just doesn't take it, while one that overrates it takes it and feeds the error back through bootstrapping.

</details>

4. **why the temperature loss has that sign.** show that gradient descent on $L(\alpha) = -\mathbb E[\alpha(\log\pi + \bar{\mathcal H})]$ (with $\log\pi$ held fixed) raises α when the policy's entropy $\mathbb E[-\log\pi]$ is below the target $\bar{\mathcal H}$, and lowers it when above.

<details><summary>solution</summary>

$\partial L/\partial\alpha = -\mathbb E[\log\pi] - \bar{\mathcal H} = \mathcal H_\pi - \bar{\mathcal H}$. descent moves α by $-\eta(\mathcal H_\pi - \bar{\mathcal H})$: down when the entropy exceeds the target, up when it falls short. it's dual gradient descent on the constraint $\mathcal H_\pi \ge \bar{\mathcal H}$, with α the Lagrange multiplier. (with $\log\alpha$ as the parameter the gradient is multiplied by α, which only rescales the step.)

</details>

## experiment

`lab.py` investigates the temperature:

1. a one-step task with two equally good actions and a valley between them: what does SAC's policy look like for α = 0, 0.05, 0.2 and 1?
2. SAC on your `Pendulum` with α = 0, 0.2, 1 and the automatic temperature.

## check your understanding

1. why does SAC have no target *actor*, while DDPG and TD3 do?

<details><summary>answer</summary>

DDPG's target uses $\mu'(s')$, a deterministic action; a lagged actor smooths the target the way the lagged critic does. SAC's target uses a *sample* from the current stochastic actor plus its entropy term; the sampling noise already smooths the target over actions (TD3 adds noise to its target action for the same reason, "target policy smoothing"), and Haarnoja et al. found a target actor unnecessary.

</details>

2. what would go wrong if the actor loss used `.sample()` instead of a reparameterized sample?

<details><summary>answer</summary>

`.sample()` cuts the gradient path from $Q(s, \tilde a)$ back to the actor's parameters, so only the $\alpha\log\pi$ term would produce gradients: the actor would maximize entropy and ignore the reward. (you could use a score-function estimator instead, as REINFORCE does, but with much higher variance; the reparameterized gradient uses $\nabla_aQ$ directly.)

</details>

3. (the predict callout in idea 3.) the target?

<details><summary>answer</summary>

$y = 1 + 0.9(1.5 + 0.2) = 2.53$, larger than the non-soft $1 + 0.9\cdot1.5 = 2.35$, because the next state's policy is random enough to earn an entropy bonus: $-\alpha\log\pi = 0.2$. soft values include the future entropy the agent will be paid.

</details>

4. why the warm-up with uniformly random actions?

<details><summary>answer</summary>

the critics need data across the whole action box before the actor starts exploiting their (initially random) gradients. a fresh actor's actions cluster around $\tanh(\mu(s))$ for a near-random $\mu$; without the warm-up, the critic would extrapolate into unseen actions, and the actor would chase those extrapolations.

</details>

5. Pendulum never terminates. does it matter whether you store `terminated` or `terminated or truncated`?

<details><summary>answer</summary>

yes: storing `done` would mark every 200th transition as terminal, so the critic learns that the state at step 200 is worth nothing, wherever the pendulum is. the values become time-dependent in a way the observation can't explain (it has no clock), which adds noise to every target. with `terminated` (always False here) every transition bootstraps, as it should.

</details>

6. with a fixed α, the right value depends on the reward scale. why, and how does auto-tuning avoid it?

<details><summary>answer</summary>

α converts entropy (nats) into reward units, so multiplying all rewards by 10 is equivalent to dividing α by 10. a fixed α that's right for one task is wrong for another. auto-tuning fixes the *entropy* (a target in nats, $-\dim\mathcal A$) instead, and lets α absorb the reward scale.

</details>

## common bugs

- **the log-probability of $u$ instead of $a$.** symptom: wrong entropy, a temperature that tunes itself to nonsense. subtract the tanh correction.
- **`log(1 - tanh(u) ** 2)` without care.** symptom: `-inf` and `nan` once actions saturate. use the softplus identity.
- **forgetting `min` in the target or the actor loss.** symptom: steady overestimation; Q-values grow while returns don't.
- **critic target with gradients.** symptom: the target network's parameters change through backprop, or the critic chases its own tail. compute $y$ under `no_grad`.
- **stale actor gradients in the critic.** the actor loss backpropagates into the critics too; zero the critics' gradients before their own step (or freeze them during the actor step).
- **α as the parameter.** symptom: α goes negative and the agent is paid for being deterministic. learn $\log\alpha$.

## going deeper

- Haarnoja, Zhou, Abbeel & Levine (2018), "Soft actor-critic: off-policy maximum entropy deep reinforcement learning with a stochastic actor", *ICML*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Haarnoja et al. (2018b), "Soft actor-critic algorithms and applications". automatic temperature tuning. [arXiv:1812.05905](https://arxiv.org/abs/1812.05905)
- Fujimoto, van Hoof & Meger (2018), "Addressing function approximation error in actor-critic methods", *ICML*. TD3. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- Lillicrap et al. (2016), "Continuous control with deep reinforcement learning", *ICLR*. DDPG. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Silver et al. (2014), "Deterministic policy gradient algorithms", *ICML*. [proceedings.mlr.press/v32/silver14](https://proceedings.mlr.press/v32/silver14.html)
- Ziebart (2010), *Modeling purposeful adaptive behavior with the principle of maximum causal entropy*, PhD thesis, Carnegie Mellon University. the maximum-entropy framework.
- Achiam (2018), *Spinning Up in Deep RL*, "Soft Actor-Critic". the conventions used here. [spinningup.openai.com](https://spinningup.openai.com/en/latest/algorithms/sac.html)

## connections

- ← builds on: replay, target networks, double Q-learning and `ema_update` (topic 01), Gaussian policies (topic 03), the reparameterization trick (ch 12, ch 21), change of variables (ch 21 · normalizing flows), entropy and Boltzmann distributions (ch 05).
- → used in: offline RL starts from SAC/DQN-style critics and adds a conservative penalty (topic 07); the Boltzmann policy $\pi \propto e^{Q/\alpha}$ and the KL-regularized objective reappear as the closed form behind RLHF and DPO (ch 27 · topic 04).
