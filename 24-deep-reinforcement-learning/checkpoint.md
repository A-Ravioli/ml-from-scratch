# 24 · deep reinforcement learning · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. name DQN's two stabilizing ingredients and the failure each one prevents.

<details><summary>answer</summary>

experience replay breaks the correlation between consecutive transitions (and reuses data), so SGD sees something close to i.i.d. minibatches; the target network freezes the regression targets for many steps, so raising $Q(s, a)$ doesn't immediately raise its own target through generalization (the feedback loop that made topic 01's "no target network" values explode into the millions). (topic 01, ideas 2–3)

</details>

2. you store `terminated or truncated` in the replay buffer. what happens to the values on a task with a time limit?

<details><summary>answer</summary>

the last transition before every time-out is treated as terminal (target $r$ instead of $r + \gamma\max Q$), so states near time-outs look worse (for positive rewards) than they are, and the agent learns that time-outs are cliffs. in the checks' single-state task the learned value is 2.78 instead of 5. only `terminated` cuts the bootstrap. (topic 01, idea 2)

</details>

3. what does double DQN change, and what bias does it address?

<details><summary>answer</summary>

the target's action is chosen by the online network and evaluated by the target network, $y = r + \gamma Q_{\bar\theta}(s', \arg\max_aQ_\theta(s', a))$, instead of both by the target network. it removes the compounding of maximization bias: the max of noisy estimates is biased upwards, and a separate evaluator doesn't share the selecting network's noise. (topic 01, idea 3)

</details>

4. state the score-function (log-derivative) identity and explain why REINFORCE doesn't need to differentiate the environment.

<details><summary>answer</summary>

$\nabla_\theta\mathbb E_{x\sim p_\theta}[f(x)] = \mathbb E_{x\sim p_\theta}[f(x)\nabla_\theta\log p_\theta(x)]$. for trajectories, $\log p_\theta(\tau)$ is a sum of policy log-probabilities and dynamics log-probabilities; only the policy terms depend on $\theta$, so the gradient needs $\nabla\log\pi_\theta(a_t\mid s_t)$ and sampled returns, nothing about the dynamics. (topic 02, idea 3)

</details>

5. why doesn't subtracting a baseline $b(s)$ bias the policy gradient, and why does it reduce variance on CartPole?

<details><summary>answer</summary>

$\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)] = \nabla\sum_a\pi(a\mid s) = 0$, so $\mathbb E[b(s)\nabla\log\pi] = 0$. on CartPole every return is positive, so without a baseline every episode votes for its own actions and the sign of its gradient is decided by which way its random actions leaned; with $b \approx V$, weights are negative for worse-than-usual outcomes, and topic 02's lab saw the variance drop 4.5× and the cosine with the true gradient rise from near 0 to 0.8. (topic 02, idea 3 and lab)

</details>

6. what are GAE's two limits, λ = 0 and λ = 1, and what does each trade?

<details><summary>answer</summary>

λ = 0: the one-step TD error $\delta_t$, low variance but biased by the critic's errors. λ = 1: $G_t - V(s_t)$, unbiased whatever the critic but with the full return's variance. GAE sums $(\gamma\lambda)^l\delta_{t+l}$ in between; practical values are 0.9–0.97. (topic 03, idea 1)

</details>

7. with auto-resetting parallel environments, how do you handle an episode that is *truncated* at step $t$?

<details><summary>answer</summary>

before resetting, compute the critic's value of the observation the episode was cut off in, add $\gamma V(s_{\text{cut}})$ to that step's reward, and set the cut flag ($d_t = 1$) so GAE doesn't bootstrap from the next episode's first state. for a terminated episode, add nothing. (topic 03, idea 3)

</details>

8. write PPO's clipped objective. for an action with positive advantage whose ratio is already 1.3 ($\varepsilon = 0.2$), what is the gradient through that sample?

<details><summary>answer</summary>

$L^{\text{CLIP}} = \hat{\mathbb E}[\min(r\hat A, \operatorname{clip}(r, 1 - \varepsilon, 1 + \varepsilon)\hat A)]$. at $r = 1.3 > 1.2$ with $\hat A > 0$ the clipped term $1.2\hat A$ is the smaller one and constant in $\theta$: zero gradient. (topic 04, idea 1)

</details>

9. does PPO's clip keep every ratio inside $[1 - \varepsilon, 1 + \varepsilon]$? what does?

<details><summary>answer</summary>

no: it removes each sample's own incentive to move further, but other samples' gradients (through shared weights) and optimizer momentum keep moving it; topic 04's lab saw the largest ratio reach 1.42 after 40 epochs, and 4.9 without the clip. a KL constraint with a line search (TRPO), or early stopping on the approximate KL, bounds the change. (topic 04, lab)

</details>

10. what is the natural gradient step of TRPO, and how is $F^{-1}g$ computed without forming $F$?

<details><summary>answer</summary>

$\Delta = \sqrt{2\delta/(g^\top F^{-1}g)}\,F^{-1}g$, with $F$ the Hessian of the mean KL at the old parameters (the Fisher information). conjugate gradient solves $Fx = g$ using only Fisher-vector products, each computed as the gradient of $(\nabla\text{KL})\cdot v$ (two backward passes). (topic 04, idea 3)

</details>

11. why does SAC's actor use a reparameterized sample, and what is the tanh correction?

<details><summary>answer</summary>

the actor loss $\mathbb E[\alpha\log\pi(\tilde a\mid s) - \min_jQ_j(s, \tilde a)]$ needs gradients through $Q$ into the actor's parameters; writing $\tilde a = a_{\max}\tanh(\mu + \sigma\epsilon)$ makes the action differentiable. squashing changes the density: $\log\pi(a) = \log\mathcal N(u;\mu, \sigma^2) - \sum_i\log(1 - \tanh^2u_i)$, computed stably as $2(\log2 - u - \operatorname{softplus}(-2u))$. (topic 05, idea 2)

</details>

12. what does the temperature α do in SAC, and what does automatic tuning fix?

<details><summary>answer</summary>

it sets the exchange rate between reward and entropy: large α keeps the policy random (topic 05's α = 5 couldn't hold the pendulum still), α = 0 is ordinary RL. auto-tuning fixes a target entropy (e.g. $-\dim\mathcal A$ nats) instead of α, making the setting independent of the reward scale. (topic 05, idea 3 and lab)

</details>

13. in MCTS, how are values backed up in a two-player zero-sum game, and why are visit counts the output?

<details><summary>answer</summary>

values are stored from the point of view of the player to move at each node; on the way up the sign flips at every level (negamax). visit counts integrate value and reliability: PUCT keeps visiting a move only as long as it keeps looking good, so counts are robust, and they form the policy target AlphaZero trains on. (topic 06, ideas 2–3)

</details>

14. why does behavior cloning's error grow faster than linearly in the horizon, and how does DAgger fix it?

<details><summary>answer</summary>

a mistake takes the clone to states the expert never visited, where it has no training data and errs more, so errors compound: expected cost up to $\varepsilon T^2$. DAgger rolls out the learner, asks the expert to label the states the learner actually visits, aggregates, and retrains, so the training distribution includes the learner's own states: $O(\varepsilon T)$. (topic 07, ideas 1–2)

</details>

15. why does plain offline DQN fail, and what does CQL's penalty do?

<details><summary>answer</summary>

the target's max queries actions the dataset rarely or never contains; their values are untrained extrapolations, the max seeks the largest errors, bootstrapping propagates them, and no new data ever corrects them (topic 07's offline DQN reached Q-values of $5\times10^5$). CQL adds $\alpha\,\mathbb E[\log\sum_ae^{Q(s, a)} - Q(s, a_{\text{data}})]$, pushing down soft-max values and up the dataset's actions, so unsupported actions can't win the max. (topic 07, idea 3)

</details>

## 2. by hand

1. **targets.** (a) a DQN batch has $Q_\theta(s, a) = 3$, $r = 0.5$, terminated, Huber loss: what's the loss? (b) not terminated, $Q_{\bar\theta}(s') = (2, 4)$, $Q_\theta(s') = (5, 1)$, $\gamma = 0.5$: the plain and the double target? (c) a SAC target with $r = 1$, $\gamma = 0.9$, α = 0.2, $Q'_1 = 2$, $Q'_2 = 1.5$, $\log\pi(a'\mid s') = -1$?

<details><summary>solution</summary>

(a) error $3 - 0.5 = 2.5 > 1$: Huber $= 2.5 - 0.5 = 2$. (b) plain: $0.5 + 0.5\cdot4 = 2.5$; double: the online net picks $a^* = 0$, the target net evaluates it: $0.5 + 0.5\cdot2 = 1.5$. (c) $1 + 0.9(\min(2, 1.5) + 0.2) = 2.53$. (topics 01 and 05, worked examples)

</details>

2. **GAE.** rewards $(1, 0, 2)$, values $(0.5, 1, -1)$, no episode ends, `last_value` 3, $\gamma = 0.9$, λ = 0.5. compute the advantages and returns. then: if the episode had been *truncated* after step 2 in a state with $V = 3$, what would the collector store as step 2's reward, and what would `dones` be?

<details><summary>solution</summary>

$\delta = (1.4, -1.9, 5.7)$; $\hat A_2 = 5.7$, $\hat A_1 = -1.9 + 0.45\cdot5.7 = 0.665$, $\hat A_0 = 1.4 + 0.45\cdot0.665 = 1.69925$; returns $(2.19925, 1.665, 4.7)$. truncated: reward $2 + 0.9\cdot3 = 4.7$ and `dones` $= (0, 0, 1)$, which gives exactly the same $\hat A_2 = 4.7 - (-1) = 5.7$. (topic 03, worked example)

</details>

3. **the clip and the bandit gradient.** (a) ratios $(1.5, 0.5, 0.5, 1.5, 1.1)$, advantages $(2, 1, -1, -2, 1)$, $\varepsilon = 0.2$: the PPO loss and its gradient with respect to the log-probabilities. (b) a two-armed softmax bandit with logits $(0, 0)$ paying 1 and 0: the exact gradient of the expected reward, and the single-sample REINFORCE estimates for each arm with baseline 0.5.

<details><summary>solution</summary>

(a) surrogates $(2.4, 0.5, -0.8, -3, 1.1)$, loss $-0.04$; gradient $(0, -0.1, 0, 0.6, -0.22)$ (zero where the clipped term is active). (b) $\nabla J = (0.25, -0.25)$; with $b = 0.5$, arm 0 gives $0.5\cdot(0.5, -0.5)$ and arm 1 gives $-0.5\cdot(-0.5, 0.5)$, both exactly $(0.25, -0.25)$. (topics 02 and 04, worked examples)

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **the actor–critic core** of topics 03–04: `compute_gae`, `ActorCritic`, `RolloutCollector` (with the time-limit trick) and `ppo_clip_loss`. work in scratch directories, starting from the *untouched* stub notebooks and the topics' checks, so your own solved notebooks stay where they are. from the repo root:

```bash
for T in 03-actor-critic 04-trust-regions-and-ppo; do
  D=/tmp/rebuild-24-$T; mkdir -p $D
  git show HEAD:24-deep-reinforcement-learning/$T/exercise.py > $D/exercise.py   # the stubs as shipped
  cp 24-deep-reinforcement-learning/$T/test_exercise.py $D/
done
```

(if you've committed your own solutions, regenerate the stubs instead: copy the topic's `solutions/` to the scratch directory and run `python scripts/stubify.py <scratch dir>`.) open each scratch `exercise.py` with `marimo edit`, implement the stages from memory, and run, from the repo root:

```bash
pytest /tmp/rebuild-24-03-actor-critic/test_exercise.py -k "stage1 or stage2 or stage3"
pytest /tmp/rebuild-24-04-trust-regions-and-ppo/test_exercise.py -k "stage1"
```

you pass if every check is green within the hour. (the checks verify GAE exactly by hand, including cuts at episode ends, step the collector through a scripted environment that terminates and one that is truncated, and check the clipped loss's value and gradient exactly, so "almost right" fails. topic 04's notebook imports topic 03 through `mlfs.deeprl`, which the tests resolve to the reference, so only the code you rebuilt is tested.)

**harder variant** (for testing out of the chapter with confidence): in a further 60 minutes, rebuild topic 01's `ReplayBuffer`, `QNetwork` and `dqn_loss`, and topic 06's `MCTS`, and pass:

```bash
pytest /tmp/rebuild-24-01-deep-q-networks/test_exercise.py -k "stage2 or stage3"
pytest /tmp/rebuild-24-06-model-based-rl-and-planning/test_exercise.py -k "stage3"
```

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer names the topic and idea it comes from.
