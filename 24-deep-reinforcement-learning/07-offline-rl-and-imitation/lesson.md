# offline RL & imitation

## why this matters

every method in this chapter learns by trial and error: act, see what happens, adjust. in many of the problems where RL would be most valuable (driving, medicine, industrial control, recommending things to people), trial and error is expensive or dangerous, and what you have instead is a **log**: demonstrations from experts, or records of whatever policy was running before. learning from a fixed dataset, with no further interaction, is **offline RL**; learning to copy demonstrations is **imitation learning**.

both look like supervised learning, and both break in the same characteristic way: **distribution shift**. a policy trained on an expert's states is evaluated on *its own* states, and the first small mistake takes it somewhere the expert never went, where it has no idea what to do, so the next mistake is likelier, and errors compound: Ross & Bagnell (2010) showed the expected cost of behavior cloning can grow quadratically with the horizon. offline RL has the mirror image: Q-learning's max picks actions the dataset never tried, whose values are pure extrapolation, and the policy happily exploits those errors. this topic builds behavior cloning, **DAgger** (which fixes imitation's shift by asking the expert about the learner's own states), and **conservative Q-learning** (which fixes offline RL's by pushing down the values of actions the data doesn't support). in the lab you'll watch a cloned cart–pole controller fall over at a time that depends on how much it was shown, and see what conservatism buys on a dataset from a noisy controller.

## warm-up: predict first

1. a cloned policy agrees with the expert 99% of the time on the expert's states. over a 500-step episode, should you expect it to perform at 99% of the expert's level?
2. you have more demonstrations. does that fix the problem in warm-up 1, or only delay it?
3. an offline dataset never contains action 1 in state $s$. what does the Q-learning target $\max_aQ(s', a)$ do with $Q(s, 1)$'s random initial value?
4. why can't you just run ordinary DQN on an offline replay buffer? it's off-policy, after all.
5. how would you make Q-values *pessimistic* about actions the data doesn't cover, without knowing which ones those are?

## prerequisites check

- [ ] you can train a `CategoricalPolicy` and know cross-entropy as negative log-likelihood (→ topic 02, ch 08)
- [ ] you know DQN's loss, `ReplayBuffer` and maximization bias (→ topic 01)
- [ ] you know importance sampling and why off-policy evaluation is hard (→ ch 23 · topic 04)
- [ ] you know `logsumexp` and why it's a smooth max (→ ch 01, ch 08)

## core ideas

### idea 1: behavior cloning

**behavior cloning** (BC) treats imitation as classification: collect the expert's (observation, action) pairs, and train the policy to maximize the log-probability of the expert's action, the cross-entropy loss $-\frac1N\sum_i\log\pi_\theta(a_i\mid s_i)$. it's simple, needs no rewards and no environment access during training, and with enough data it often works well: Pomerleau's ALVINN drove a van on public roads with a cloned network in 1989.

the catch is what the training distribution covers. BC minimizes errors on the *expert's* state distribution $d_{\pi^*}$; at test time the cloned policy visits its own, $d_{\pi_\theta}$. if the clone makes a mistake with probability $\varepsilon$ at each step of a horizon-$T$ task, and a mistake takes it to states where it may keep failing, the expected cost grows like $\varepsilon T^2$ rather than $\varepsilon T$ (Ross & Bagnell, 2010; on paper 1). the 99%-accurate clone of warm-up 1 can be much worse than 99% as good, and more demonstrations of the same expert only shrink $\varepsilon$, they don't cover the states the clone wanders into (warm-up 2). the expert never needs to recover from mistakes, so its demonstrations never show how.

> **spec.** `behavior_cloning(policy, obs, actions, *, epochs=50, lr=1e-2, batch_size=64, seed=None)` trains `policy` (an `nn.Module` with `policy(obs) -> Categorical`) in place with Adam (`lr`) on minibatches of the mean negative log-likelihood of the expert's actions, shuffling each epoch with `np.random.default_rng(seed).permutation`. returns a list with each epoch's mean loss over the dataset. `run_policy(env, act, n_episodes, *, seed=None)` runs `n_episodes` episodes with `act(obs) -> action` (the first reset seeded with `seed`) and returns `(returns, visited)`: the undiscounted returns, and every observation an action was taken in (float32 arrays), in order.

> **▶ build: stage 1.** implement `behavior_cloning` and `run_policy`.

### idea 2: DAgger

if the problem is that the training data comes from the wrong distribution, collect data from the right one. **DAgger** (dataset aggregation; Ross, Gordon & Bagnell, 2011) does it iteratively. start with BC on expert rollouts. then repeat: roll out the *current learner*, record the states it visits, ask the expert what it would have done in each, add those labelled states to the dataset, and retrain on everything collected so far. the learner's mistakes now produce training data about exactly the states mistakes lead to, and the expert's labels show how to recover. Ross et al. prove that the cost grows only linearly in $T$ (with a policy class that can fit the aggregated data).

the price is an **interactive expert**: someone (or something) must label arbitrary states the learner wanders into, which is easy when the expert is a planner, a simulator oracle or a hand-written controller, and harder when it's a human driver who would have to say "steer left now" while watching the learner drive badly. variants mix expert and learner actions during rollouts (the original's $\beta_i$ schedule), or only query where the learner is uncertain.

> **spec.** `dagger(env, expert, policy, *, n_iterations=5, episodes_per_iteration=5, epochs=30, lr=1e-2, batch_size=64, seed=None)`: `expert(obs) -> action`. iteration 0 rolls out the expert with `run_policy`; iterations $\ge1$ roll out the learner with `policy.act` (it samples from its distribution). every visited observation is labelled with `expert(obs)` and added to one growing dataset; after each iteration, `behavior_cloning` trains `policy` (continuing from its current weights) on the whole dataset. seeds for the rollouts and for training come from `np.random.default_rng(seed)`, and torch is seeded from `seed`. returns `(policy, history)` with `history['returns']` (the mean return of each iteration's rollouts) and `history['dataset_size']` (after each iteration).

> **🔮 predict:** on a tightrope where the expert always steps back to the middle (so its demonstrations never show a position two steps off-centre), what will the cloned policy do the first time a gust and a mistake put it there?

> **▶ build: stage 2.** implement `dagger`.

### idea 3: offline RL and conservative Q-learning

with rewards in the log, you could hope to do better than the policy that collected it: run Q-learning on the fixed dataset. DQN is off-policy, so in principle this works (warm-up 4). in practice it fails badly (Fujimoto, Meger & Precup, 2019; Kumar et al., 2019), because the off-policy *correction* isn't the issue, *coverage* is. the target $r + \gamma\max_{a'}Q(s', a')$ evaluates actions at $s'$ that the dataset may never have tried; their Q-values were never trained, so they're whatever the network extrapolates (warm-up 3). the max seeks out the largest of those errors, the bootstrapped targets propagate them, and since the agent never acts, nothing ever corrects them. online, an overestimated action gets tried and found out; offline, it just stays overestimated, and the greedy policy picks it.

**conservative Q-learning** (CQL; Kumar et al., 2020) adds a regularizer that pushes Q-values down everywhere and back up on the actions the data actually contains:

$$\mathcal L(\theta) = \mathcal L_{\text{TD}}(\theta) + \alpha\,\hat{\mathbb E}_{(s,a)\sim\mathcal D}\Big[\log\sum_{a'}e^{Q_\theta(s, a')} - Q_\theta(s, a)\Big].$$

the log-sum-exp is a soft maximum over all actions, so minimizing it lowers the largest Q-values, whichever action they belong to; the $-Q(s, a)$ term raises the dataset's action. at the optimum, actions the data supports keep values near their TD targets, while unsupported actions are pushed below them, and the greedy policy stays on the data's support (warm-up 5). Kumar et al. show the learned values lower-bound the true values of the policy. α trades conservatism (stay close to the behaviour) against improvement (trust the TD targets); the gradient of the penalty with respect to the Q-values is $\alpha(\operatorname{softmax}(Q(s, \cdot)) - \text{onehot}(a))$, per sample (on paper 3).

> **spec.** `cql_penalty(q_values, actions, *, alpha=1.0)`: `q_values` `(B, n_actions)`, `actions` `(B,)`; returns $\alpha\cdot\operatorname{mean}(\operatorname{logsumexp}_aQ - Q[a])$. `train_offline_q(buffer, obs_dim, n_actions, n_updates, *, gamma=0.99, lr=1e-3, batch_size=64, target_update=100, cql_alpha=0.0, hidden_sizes=(64, 64), double=True, seed=None)` trains a `QNetwork` (torch seeded from `seed`) for `n_updates` Adam steps on minibatches from a fixed `ReplayBuffer`: loss `dqn_loss(q_net, target_net, batch, gamma=gamma, double=double)` plus, if `cql_alpha > 0`, `cql_penalty(q_net(obs), actions, alpha=cql_alpha)`; a hard target update every `target_update` steps. returns `(q_net, history)` with per-update lists `td_loss` and `cql` (0 when the penalty is off).

> **🔮 predict:** the dataset only ever takes action 0, which pays 0.5. after training with CQL (α = 1), where do $Q(s, 0)$ and $Q(s, 1)$ end up, and why isn't $Q(s, 1) = -\infty$?

> **▶ build: stage 3.** implement `cql_penalty` and `train_offline_q`.

## worked example

**the CQL penalty.** two samples, three actions, α = 0.5. sample 1: $Q = (1, 2, 0)$, data action 1: $\operatorname{logsumexp} = \log(e + e^2 + 1) = 2.4076$, minus $Q = 2$: 0.4076. sample 2: $Q = (0, 0, 0)$, data action 2: $\log3 - 0 = 1.0986$. the penalty is $0.5\cdot(0.4076 + 1.0986)/2 = 0.3766$. its gradient on sample 1's Q-values is $\tfrac{0.5}{2}(\operatorname{softmax}(1, 2, 0) - (0, 1, 0)) = 0.25\cdot(0.2447, -0.3348, 0.0900) = (0.0612, -0.0837, 0.0225)$: gradient descent raises $Q(s, 1)$ and lowers the other two. note the penalty is positive even when the data action is the best one (sample 1): it's 0 only if the data action's Q is infinitely larger than the others, which is why α must be balanced against the TD loss.

**compounding errors.** a clone that errs with probability $\varepsilon = 0.01$ per step and, after an error, never recovers, survives 500 steps with probability $0.99^{500} = 0.0066$; its expected number of steps before the first error is about 100. the expert survives every time. that's warm-up 1's answer for the worst case; how bad it is in practice depends on how often an error is recoverable.

your code must reproduce the penalty and its gradient.

## on paper

1. **the $T^2$ bound.** a policy errs with probability $\varepsilon$ at each step *on the expert's distribution*; after its first error it's in unfamiliar states and pays cost 1 per step for the rest of the episode. show that the expected cost over horizon $T$ is at most about $\varepsilon T^2/2$, and compare with $\varepsilon T$ for a policy whose errors don't change the state distribution.

<details><summary>solution</summary>

the first error happens at step $t$ with probability at most $\varepsilon$ (union bound over the steps it could be first), and costs at most $T - t$. summing, $\sum_{t=1}^T\varepsilon(T - t) \approx \varepsilon T^2/2$. if errors don't affect future states (i.i.d. supervised learning), each step independently costs $\varepsilon$ in expectation: $\varepsilon T$. Ross & Bagnell (2010) give the precise bound $T^2\varepsilon$ for BC and show it's tight; DAgger achieves $O(T\varepsilon)$ because its training distribution includes the learner's own states.

</details>

2. **why the max loves untrained actions.** let $Q(s, a) = q + e_a$ where the errors $e_a$ on untrained actions are i.i.d. with mean 0 and standard deviation $\sigma$, and on the single trained action $e = 0$. with $k$ untrained actions, what does $\mathbb E[\max_aQ(s, a)]$ look like for large $k$?

<details><summary>solution</summary>

the max is at least $q$ (the trained action) and, for Gaussian errors, $\mathbb E[\max_ae_a] \approx \sigma\sqrt{2\ln k}$ for large $k$, so the target is overestimated by roughly that much whenever the untrained errors exceed 0. with continuous actions $k$ is effectively infinite: some action in the continuum always looks great. online, those actions would be tried; offline, the overestimate feeds every bootstrapped target.

</details>

3. **the CQL gradient.** show that $\partial/\partial Q(s, a')\,[\log\sum_be^{Q(s, b)} - Q(s, a)] = \operatorname{softmax}(Q(s, \cdot))_{a'} - \mathbb 1[a' = a]$. at what Q-values is the gradient zero?

<details><summary>solution</summary>

$\partial\log\sum_be^{Q_b}/\partial Q_{a'} = e^{Q_{a'}}/\sum_be^{Q_b}$, and $\partial(-Q_a)/\partial Q_{a'} = -\mathbb 1[a' = a]$. the penalty's gradient alone vanishes only when the softmax puts all its mass on the data action, i.e. when $Q(s, a) - Q(s, a') \to \infty$ for all $a' \neq a$. in practice the TD loss holds $Q(s, a)$ near its target, and for an action that never appears in the data only the penalty acts on it, pushing it down steadily until the network's shared parameters (and the TD loss on related states) balance the push.

</details>

4. **DAgger's dataset.** after $N$ iterations with $m$ episodes each, which state distributions does DAgger's dataset contain, and why is training on all of them (rather than only the latest) important?

<details><summary>solution</summary>

the expert's ($d_{\pi^*}$, iteration 0) and those of every learner so far, $d_{\pi_1}, \dots, d_{\pi_{N-1}}$, in equal parts. training on the aggregate is a "follow the leader" strategy in an online learning game, which is what gives the no-regret guarantee; training only on the latest data could oscillate: the new policy fixes the latest learner's states and forgets earlier ones, visits them again, and so on.

</details>

## experiment

`lab.py` investigates, on your `CartPole` with a hand-written expert controller:

1. behavior cloning from 1, 3 or 10 expert episodes against DAgger: how long the clones balance, and how the failures are spread over time.
2. offline RL from a dataset collected by a noisy version of the expert: offline DQN, CQL and behavior cloning.

## check your understanding

1. why does adding noise to the *expert's* actions while collecting demonstrations (and recording the expert's intended action as the label) help behavior cloning?

<details><summary>answer</summary>

the noise pushes the expert into slightly off-course states, and the labels show how it recovers from them: the demonstrations now cover a neighbourhood of the expert's trajectories instead of a thin line (Laskey et al., 2017, "DART"). it's a cheap, non-interactive approximation of what DAgger does.

</details>

2. (the predict callout in idea 2.) what does the clone do two steps off-centre?

<details><summary>answer</summary>

whatever its network happens to output for an input it has never seen: for one-hot observations that's mostly the biases learned from the common states ("stay"), plus untrained weights. with nothing pulling it back, gusts push it further, and it falls. DAgger's learner visits those states and the expert's labels teach it to step back.

</details>

3. why does offline RL need something like CQL even with a perfect function approximator and infinite data from the behaviour policy?

<details><summary>answer</summary>

infinite data from a behaviour policy that never takes action $a$ in state $s$ still says nothing about $Q(s, a)$. coverage, not sample size, is the issue: Q-learning's max will query exactly the uncovered actions. only with full coverage (every action has positive probability everywhere) does standard Q-learning converge offline.

</details>

4. (the predict callout in idea 3.) where do the Q-values end up?

<details><summary>answer</summary>

$Q(s, 0)$ stays near its TD target 0.5 (the penalty pulls it up slightly, the TD loss holds it); $Q(s, 1)$ is pushed down, and in the tabular case it never stops: the penalty's gradient on it is $\alpha\operatorname{softmax}_1/B \approx \alpha e^{Q(s,1) - Q(s,0)}/B$, so the gap grows only logarithmically in the number of updates, but it keeps growing ($-\infty$ is the limit, approached very slowly). in a network the shared weights and the TD loss on other inputs slow it further, which is why after a finite run it sits some distance below.

</details>

5. when is plain behavior cloning the right offline method, and when would you prefer offline RL?

<details><summary>answer</summary>

BC when the data comes from a good, consistent expert: it's simple and stable. offline RL when the data is mixed or mediocre (several policies, exploration noise, failures) and has rewards: it can stitch together the good parts of different trajectories and improve on the behaviour, which BC, copying the average behaviour, can't. a strong baseline in practice is filtered BC: clone only the best episodes.

</details>

6. DAgger needs an interactive expert. what does offline RL need instead, and what does it lose?

<details><summary>answer</summary>

rewards in the dataset and no further interaction at all. it loses the ability to ask about states the data doesn't cover; it can only be cautious there (CQL), which caps how far it can improve on the behaviour policy.

</details>

## common bugs

- **labelling DAgger data with the learner's actions.** symptom: DAgger is no better than BC (it's imitating itself). the labels must come from the expert.
- **retraining on the newest data only.** symptom: performance oscillates between iterations.
- **evaluating the clone on the expert's states.** symptom: 99% accuracy and a policy that falls over. evaluate by running it.
- **CQL with the wrong sign.** symptom: the unsupported actions' values climb and the greedy policy leaves the data. the penalty is logsumexp *minus* the data action's Q.
- **CQL's α too large.** symptom: the policy can't improve on the behaviour at all; values collapse well below the returns.
- **sampling the buffer's empty slots.** symptom: zero observations in batches. sample only stored transitions.

## going deeper

- Pomerleau (1989), "ALVINN: an autonomous land vehicle in a neural network", *NeurIPS* 1. behavior cloning on the road.
- Ross & Bagnell (2010), "Efficient reductions for imitation learning", *AISTATS*. the $T^2$ bound. [proceedings.mlr.press/v9/ross10a](https://proceedings.mlr.press/v9/ross10a.html)
- Ross, Gordon & Bagnell (2011), "A reduction of imitation learning and structured prediction to no-regret online learning", *AISTATS*. DAgger. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
- Kumar, Zhou, Tucker & Levine (2020), "Conservative Q-learning for offline reinforcement learning", *NeurIPS*. [arXiv:2006.04779](https://arxiv.org/abs/2006.04779)
- Fujimoto, Meger & Precup (2019), "Off-policy deep reinforcement learning without exploration", *ICML*. extrapolation error and BCQ. [arXiv:1812.02900](https://arxiv.org/abs/1812.02900)
- Levine, Kumar, Tucker & Fu (2020), "Offline reinforcement learning: tutorial, review, and perspectives on open problems". [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)
- Laskey, Lee, Fox, Dragan & Goldberg (2017), "DART: noise injection for robust imitation learning", *CoRL*. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)

## connections

- ← builds on: `CategoricalPolicy` and cross-entropy (topic 02), `ReplayBuffer`, `QNetwork` and `dqn_loss` (topic 01), off-policy evaluation and coverage (ch 23 · topic 04), `CartPole` (topic 01).
- → used in: supervised fine-tuning of language models is behavior cloning on demonstrations, with the same compounding-error problem at generation time (ch 27 · topic 03); RLHF's KL penalty to the reference model is a conservatism term in the same spirit as CQL (ch 27 · topic 04); distribution shift returns in robustness (ch 28).
