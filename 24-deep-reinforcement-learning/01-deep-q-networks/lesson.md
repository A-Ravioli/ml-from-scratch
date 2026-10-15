# deep Q-networks

## why this matters

chapter 23 ended with tile coding: a hand-designed featurization that let a linear Q-learner drive a car up a hill. it worked because you, the designer, knew which features mattered. the moment observations become rich (pixels, many joint angles, a board position) that stops scaling, and the obvious fix, a neural network in place of the table, has a bad reputation: Q-learning with nonlinear function approximation can diverge, and the deadly triad you met in ch 23 · topic 06 (function approximation + bootstrapping + off-policy data) is exactly what a neural Q-learner is.

in 2015 Mnih et al. made it work anyway, on 49 Atari games from raw pixels, with one architecture and one set of hyperparameters. the innovation was two pieces of plumbing that keep the triad in check: an **experience replay** buffer that turns the agent's stream of correlated transitions into something close to i.i.d. minibatches, and a **target network**, a lagged copy of the Q-network that the regression targets come from, so the agent isn't chasing a target that moves every time it takes a step. in this topic you build both, the loss with its double-DQN and dueling refinements, and the agent loop around them, on a cart–pole you write yourself. in the lab you'll take the plumbing away again and watch what breaks (one of the two matters far more than the other on this task, and when it's gone the estimated values climb into the millions, for a task whose returns can never exceed 100).

## warm-up: predict first

1. a cart–pole earns $+1$ per step and is cut off after 500 steps. with $\gamma = 0.99$, what's the largest value $Q(s, a)$ can possibly have?
2. you train a Q-network on transitions in the order the agent experienced them, one gradient step per transition. what goes wrong that doesn't go wrong in supervised learning?
3. the target $r + \gamma\max_{a'}Q(s', a')$ uses the same network you're updating. why might that be a problem even with a tiny learning rate?
4. you have noisy, unbiased estimates of ten numbers that are all truly 0. is the max of the estimates an unbiased estimate of the max (which is 0)?
5. a time limit ends an episode. what do you store in the replay buffer's `terminated` field?

## prerequisites check

- [ ] you know Q-learning's target and why it's off-policy (→ ch 23 · topic 05)
- [ ] you know the deadly triad and semi-gradient TD (→ ch 23 · topic 06)
- [ ] you can write an `mlfs.rl.Env` and you know `terminated` from `truncated` (→ ch 23 · topic 02)
- [ ] you can build an `MLP`, take optimizer steps in PyTorch, and soft-update with `ema_update` (→ ch 15 · topic 07)
- [ ] you know the Huber loss and why it's robust to outliers (→ ch 08)

## core ideas

### idea 1: a continuous-state task: the cart–pole

tabular methods index a table by the state. in a **continuous** state space no state is ever visited twice, so a table learns nothing; you need a function $Q_\theta(s, a)$ that generalizes from visited states to nearby ones. the classic test bed is the **cart–pole** (Barto, Sutton & Anderson, 1983): a pole hinged on a cart that moves along a track. each step you push the cart left or right with a fixed force; the episode ends when the pole tips past 12° or the cart leaves the track ($|x| > 2.4$ m). the reward is $+1$ per step, so the return counts how long you balanced, and the Gymnasium version (`CartPole-v1`) truncates at 500 steps.

the physics is a pair of coupled ODEs for the cart position $x$ and pole angle $\theta$ (from vertical), with cart mass $m_c = 1$, pole mass $m_p = 0.1$, pole half-length $l = 0.5$, gravity $g = 9.8$ and force $F = \pm10$. writing $M = m_c + m_p$ and

$$\text{tmp} = \frac{F + m_pl\,\dot\theta^2\sin\theta}{M},\qquad \ddot\theta = \frac{g\sin\theta - \cos\theta\cdot\text{tmp}}{l\left(\frac43 - \frac{m_p\cos^2\theta}{M}\right)},\qquad \ddot x = \text{tmp} - \frac{m_pl\,\ddot\theta\cos\theta}{M},$$

these are the frictionless equations of Barto et al. as corrected by Florian (2005/2007). they're integrated with one explicit Euler step of $\tau = 0.02$ s: $x \leftarrow x + \tau\dot x$, $\dot x \leftarrow \dot x + \tau\ddot x$, and the same for $\theta$ (each position update uses the *old* velocity). that's exactly what Gymnasium's `CartPole-v1` does, so numbers you measure here are comparable with published ones.

one practical note: everything in this chapter speaks your `mlfs.rl` protocol, and Gymnasium isn't installed. if you do have it, an adapter is three lines, because the `reset`/`step` signatures already match: wrap `gymnasium.make("CartPole-v1")` in a class whose `observation_space` and `action_space` expose `.shape` and `.n`, whose `reset(seed=...)` and `step(a)` forward to the Gymnasium env, and which converts the reward to `float` and the flags to `bool`.

> **spec.** `CartPole(*, max_steps=500)` subclasses `mlfs.rl.Env`. `observation_space = Box(-high, high)` with `high = (4.8, inf, 0.4189, inf)` (twice the termination limits), `action_space = Discrete(2)`: 0 pushes left ($F = -10$), 1 pushes right. the current state is `self.state`, a float array $(x, \dot x, \theta, \dot\theta)$, and the step counter `self.t`. `reset(seed=None)` calls `Env.reset`, draws the state with `self.np_random.uniform(-0.05, 0.05, size=4)`, sets `t = 0` and returns `(state.copy(), {})`. `step(a)` applies the equations above, then returns `(state.copy(), 1.0, terminated, truncated, {})` with `terminated` when $|x| > 2.4$ or $|\theta| > 12°$, and `truncated` when `t` reaches `max_steps` without termination. invalid actions raise `ValueError`.

> **▶ build: stage 1.** implement `CartPole`.

### idea 2: experience replay

online Q-learning updates on each transition as it arrives. with a network that's a disaster for two reasons. consecutive transitions are **strongly correlated** (the pole is leaning left for twenty steps in a row), so successive gradient steps all push the same way and the network forgets what it learned about the other side: SGD assumes roughly i.i.d. samples, and this stream is anything but. and each transition is used **once**, then thrown away, which wastes experience that cost real interaction.

**experience replay** (Lin, 1992) fixes both. store every transition $(s, a, r, s', \text{terminated})$ in a large buffer; at every training step, sample a random minibatch from it. random sampling breaks the correlations, each transition is reused many times, and the training distribution changes slowly (it's an average over the last $N$ transitions' worth of policies). replay is only possible because Q-learning is **off-policy**: its target doesn't care which policy generated the transition. you couldn't replay old data into SARSA or REINFORCE without correcting for the policy change.

implementation matters because the buffer is touched every step: preallocate NumPy arrays of size `capacity` once and write into them at a moving position, wrapping around so the oldest transition is overwritten first (a **ring buffer**).

the field you store is `terminated`, **not** `terminated or truncated` (warm-up 5). the target will multiply the bootstrap term by $1 - \text{terminated}$. a time-limit cutoff isn't a terminal state, so its transition must bootstrap like any other; the "done" flag of older code silently taught agents that time-outs are cliffs.

> **spec.** `ReplayBuffer(capacity, obs_shape, *, action_shape=(), action_dtype='int64', seed=None)`: preallocated arrays for `obs` and `next_obs` (float32, shape `(capacity, *obs_shape)`), actions (`action_dtype`, shape `(capacity, *action_shape)`), rewards and terminated (float32). `add(obs, action, reward, next_obs, terminated)` copies the transition in at the current position, overwriting the oldest once full. `len(buf)` is the number stored. `sample(batch_size)` draws indices uniformly **with replacement** from the stored transitions, using a `np.random.default_rng(seed)` created in `__init__`, and returns a dict of torch tensors with keys `obs`, `actions`, `rewards`, `next_obs`, `terminated`. sampling an empty buffer raises `ValueError`.

> **▶ build: stage 2.** implement `ReplayBuffer`.

### idea 3: the Q-network, the target network and the loss

the **Q-network** maps an observation to one Q-value per action: $Q_\theta(s) \in \mathbb R^{|\mathcal A|}$. one forward pass gives every action's value, so the greedy action is an argmax and the max in the target is cheap. for a minibatch $\{(s_i, a_i, r_i, s'_i, d_i)\}$ the loss is a regression of $Q_\theta(s_i, a_i)$ onto a target,

$$y_i = r_i + \gamma(1 - d_i)\max_{a'}Q_{\bar\theta}(s'_i, a'),\qquad \mathcal L(\theta) = \frac1B\sum_i\ell\big(Q_\theta(s_i, a_i) - y_i\big),$$

with $d_i$ the stored `terminated` flag. three details carry the whole method.

**the target network** $\bar\theta$ is a copy of $\theta$ that's updated only occasionally: every $C$ gradient steps ($\bar\theta \leftarrow \theta$, a hard update), or a little every step ($\bar\theta \leftarrow (1 - \tau)\bar\theta + \tau\theta$, a soft update, which is your `ema_update` with `decay = 1 - tau`). if the target used $\theta$ itself, every gradient step that raises $Q_\theta(s, a)$ would also raise $Q_\theta(s', \cdot)$ for similar states (the network generalizes), which raises the target, which calls for a bigger $Q_\theta(s, a)$: a feedback loop (warm-up 3). freezing the target for $C$ steps turns each period into an ordinary supervised regression with fixed labels. the price: a value can only propagate one backup per target update, so $C$ also limits how fast values grow.

**the semi-gradient.** the target is a label, not part of the model: compute it under `torch.no_grad()`. differentiating through it would turn Q-learning into a residual-gradient method with different (and usually worse) behaviour (ch 23 · topic 06).

**the Huber loss.** $\ell(e) = \tfrac12e^2$ for $|e| \le 1$ and $|e| - \tfrac12$ beyond (PyTorch's `smooth_l1_loss`). TD errors early in training can be huge, and a squared loss turns one huge error into one huge gradient step. Huber clips the gradient to $\pm1$ per sample.

two refinements. **double DQN** (van Hasselt, Guez & Silver, 2016) attacks *maximization bias*. the max of noisy estimates is biased upwards (warm-up 4: $\mathbb E[\max_a\hat Q_a] \ge \max_a\mathbb E[\hat Q_a]$), and Q-learning feeds that bias back into its own targets. double DQN decouples choosing the action from evaluating it: the online network picks $a^* = \arg\max_{a'}Q_\theta(s', a')$ and the target network evaluates it, $y = r + \gamma(1 - d)Q_{\bar\theta}(s', a^*)$. if one network's noise makes a bad action look good, the other usually doesn't share it.

**dueling networks** (Wang et al., 2016) change the architecture, not the loss. in many states the choice of action barely matters, and learning each $Q(s, a)$ separately wastes data. a dueling head splits the last layer into a state value $V(s)$ and advantages $A(s, a)$, and combines them as

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal A|}\sum_{a'}A(s, a').$$

subtracting the mean advantage makes the split identifiable: without it, adding a constant to $A$ and subtracting it from $V$ gives the same $Q$, and the two streams can drift.

> **spec.** `QNetwork(obs_dim, n_actions, *, hidden_sizes=(128, 128), dueling=False)` is an `nn.Module` mapping `(B, obs_dim)` to `(B, n_actions)`. plain: an `mlfs.torchnn.MLP`. dueling: shared hidden layers, then an `nn.Linear` value head (1 output) and an `nn.Linear` advantage head (`n_actions` outputs), combined as above. `dqn_loss(q_net, target_net, batch, *, gamma=0.99, double=False, huber=True)` takes a batch dict as returned by `ReplayBuffer.sample` and returns the scalar mean loss: Huber (`F.smooth_l1_loss`, threshold 1) or MSE between $Q_\theta(s, a)$ and $y$; the target is computed without gradients; with `double=True` the online net picks $a^*$ and the target net evaluates it.

> **🔮 predict:** what is $\mathcal L$ for $Q_\theta(s, a) = 3$, $r = 0.5$, a transition that terminated, and Huber loss? what if it hadn't terminated, with $\max_{a'}Q_{\bar\theta}(s', a') = 10$ and $\gamma = 0.5$? (the worked example has a full batch.)

> **▶ build: stage 3.** implement `QNetwork` and `dqn_loss`.

### idea 4: the agent

the full agent interleaves acting and learning. at each environment step: pick an action ε-greedily from $Q_\theta(s)$ (your `epsilon_greedy` from ch 23, with ε annealed linearly from 1 to a small floor so early experience is diverse); step the environment; store the transition; and, once the buffer holds `learning_starts` transitions, take a gradient step on a sampled minibatch every `train_freq` steps; every `target_update` gradient steps, copy $\theta$ into $\bar\theta$ (or soft-update every step). when an episode ends (terminated **or** truncated), reset and carry on; the buffer doesn't care about episode boundaries.

`learning_starts` exists because the first minibatches would otherwise be drawn from a handful of transitions, and the network would overfit them before it has seen anything else. note the asymmetry with the buffer's flag: the *episode* ends on either flag, but the *bootstrap* is cut only on `terminated`.

seeding matters more in RL than anywhere else (Henderson et al., 2018). make every run reproducible from one integer: a NumPy generator for exploration and the buffer, the environment seeded once on the first reset, and torch's generator for the network's initialization (wrapping the run in `torch.random.fork_rng()` leaves the global RNG as you found it).

> **spec.** `linear_schedule(start, end, duration, t)` returns `start + (end - start) * min(t / duration, 1)`. `train_dqn(env, n_steps, *, gamma=0.99, lr=1e-3, buffer_size=50_000, batch_size=64, learning_starts=1000, train_freq=1, target_update=500, tau=None, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000, hidden_sizes=(64, 64), double=False, dueling=False, huber=True, seed=None)` runs exactly `n_steps` environment steps on a Box-observation, Discrete-action env. ε at step $t$ (0-based) is `linear_schedule(eps_start, eps_end, eps_decay_steps, t)`; actions come from `epsilon_greedy`; Adam with `lr`; a gradient step after env step $t$ whenever `t + 1 >= learning_starts` and `(t + 1) % train_freq == 0`; a hard target update every `target_update` gradient steps, or `ema_update(target, q_net, decay=1 - tau)` after every gradient step if `tau` is given. everything is reproducible from `seed`. returns `(q_net, history)` with `history['episode_returns']` (undiscounted return of each finished episode) and `history['episode_ends']` (the 1-based env step at which each ended).

> **🔮 predict:** on a task with a single state, reward $+1$ per step, $\gamma = 0.8$ and episodes truncated after 5 steps, what Q-value does a correct agent learn? what does an agent that stores `terminated or truncated` learn?

> **▶ build: stage 4.** implement `linear_schedule` and `train_dqn`.

## worked example

**the loss on a batch of three.** three states with one-hot observations and two actions. the online net's table is $Q = \big((1, 2), (0.5, -1), (3, 0)\big)$ and the target net's $Q_{\bar\theta} = \big((0, 4), (2, 1), (-1, 0.5)\big)$ (rows are states). the batch, with $\gamma = 0.5$:

| $s$ | $a$ | $r$ | $s'$ | terminated | $Q_\theta(s, a)$ | target $y$ | error $e$ |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 0 | 2 | $1 + 0.5\max(2, 1) = 2$ | 0 |
| 1 | 0 | −1 | 2 | 0 | 0.5 | $-1 + 0.5\max(-1, 0.5) = -0.75$ | 1.25 |
| 2 | 0 | 0.5 | 0 | 1 | 3 | $0.5$ (no bootstrap) | 2.5 |

the MSE loss is $(0 + 1.5625 + 6.25)/3 = 2.6042$. the Huber loss is $(0 + (1.25 - 0.5) + (2.5 - 0.5))/3 = 0.9167$: both nonzero errors are in the linear regime. note the gradient: with MSE the third sample pushes $Q_\theta(2, 0)$ down with force $2.5\cdot\tfrac23$; with Huber, $1\cdot\tfrac13$.

**double DQN.** online $Q_\theta(s') = (5, 1)$, target $Q_{\bar\theta}(s') = (2, 3)$, $r = 0$, $\gamma = 1$, $Q_\theta(s, a) = 0$. plain DQN's target is $\max(2, 3) = 3$ (loss 9 with MSE). double DQN lets the online net choose $a^* = 0$ and the target net evaluate it: target 2, loss 4.

your code must reproduce these numbers.

## on paper

1. **the largest possible Q-value.** with reward $+1$ per step and discount $\gamma$, show that $Q(s, a) \le 1/(1 - \gamma)$ for every state when time limits are bootstrapped through, and compute it for $\gamma = 0.99$. what does a value estimate of 120 tell you?

<details><summary>solution</summary>

the return from any state is $\sum_{k\ge0}\gamma^kR_{t+k+1} \le \sum_k\gamma^k = 1/(1 - \gamma) = 100$ for $\gamma = 0.99$. bootstrapping through truncation means the values describe the time-unlimited task, so 100 is the supremum (and with a policy that never falls, the value is 100). an estimate of 120 is impossible: it's pure overestimation, maximization bias amplified by bootstrapping. it's a useful sanity check to log the max Q-value during training.

</details>

2. **maximization bias.** let $\hat Q_1, \hat Q_2$ be independent estimates of two actions whose true values are both 0, each $\mathcal N(0, \sigma^2)$. compute $\mathbb E[\max(\hat Q_1, \hat Q_2)]$. then let a second, independent pair $\hat Q'_1, \hat Q'_2$ evaluate the action the first pair picks: what's the expectation of that double estimate?

<details><summary>solution</summary>

$\max(X, Y) = \tfrac12(X + Y + |X - Y|)$ and $X - Y \sim \mathcal N(0, 2\sigma^2)$, so $\mathbb E\max = \tfrac12\mathbb E|X - Y| = \tfrac12\sigma\sqrt2\sqrt{2/\pi} = \sigma/\sqrt\pi \approx 0.56\sigma$. the double estimate is $\hat Q'_{a^*}$ with $a^*$ chosen by the first pair; since $\hat Q'$ is independent of $a^*$, $\mathbb E[\hat Q'_{a^*}] = 0$: unbiased. (in general the double estimator can underestimate, but it never compounds an upward bias the way the max does.)

</details>

3. **the dueling identifiability problem.** show that without the mean subtraction, $(V + c, A - c)$ gives the same $Q$ as $(V, A)$ for any constant $c$, and that with it, $V(s) = \tfrac{1}{|\mathcal A|}\sum_aQ(s, a)$.

<details><summary>solution</summary>

$V + c + A(a) - c = V + A(a)$. with the subtraction, $\sum_aQ(s, a) = |\mathcal A|V(s) + \sum_aA(s, a) - \sum_aA(s, a) = |\mathcal A|V(s)$, so $V$ is pinned to the mean of $Q$, and shifting $A$ by a constant has no effect on $Q$ at all. (Wang et al. also tried subtracting the max, which makes $V = \max_aQ$; the mean is more stable in practice.)

</details>

4. **how fast can values propagate?** with hard target updates every $C$ gradient steps and the network fitting each period's targets exactly, after $k$ target updates what is the largest Q-value a reward-$1$-per-step task can have learned (starting from $Q = 0$)? how many gradient steps does reaching 90% of $1/(1 - \gamma)$ take for $\gamma = 0.99$, $C = 500$?

<details><summary>solution</summary>

each period is one Bellman backup: $Q_k \le 1 + \gamma Q_{k-1}$, so $Q_k \le (1 - \gamma^k)/(1 - \gamma)$. 90% needs $\gamma^k \le 0.1$, $k \ge \ln 10/0.01 \approx 230$ target updates, about 115,000 gradient steps. a short training run with a long target period simply can't represent the true values yet, which is one reason the lab uses $C = 100$.

</details>

## experiment

`lab.py` runs your `train_dqn` on your `CartPole`:

1. ablating replay and the target network: four variants, several seeds each, learning curves and final greedy returns.
2. how good are the value estimates? the network's $Q(s, a)$ along greedy rollouts against the discounted returns actually obtained, for DQN and double DQN.

## check your understanding

1. why does replay require an off-policy algorithm?

<details><summary>answer</summary>

the buffer holds transitions generated by many older policies. Q-learning's target $r + \gamma\max_{a'}Q(s', a')$ only uses the transition itself and the current estimates, so it's valid whichever policy chose $a$. an on-policy target (SARSA's next action, a Monte Carlo return) is tied to the behaviour policy, so stale data would need importance weights.

</details>

2. why sample *with* replacement, and does it matter?

<details><summary>answer</summary>

it makes sampling O(batch) with no bookkeeping, and with a buffer of tens of thousands of transitions and batches of 64, duplicates within a batch are rare.

</details>

3. (the predict callout in idea 4.) one state, reward 1, $\gamma = 0.8$, truncation after 5 steps: what's learned?

<details><summary>answer</summary>

correct: $Q = 1 + 0.8Q$, so $Q = 5$; the cutoff is irrelevant to the values. storing `done` makes one transition in five a "terminal" with target 1, so $Q = 0.8(1 + 0.8Q) + 0.2\cdot1$, giving $Q = 1/0.36 \approx 2.78$. the checks catch exactly this.

</details>

4. a DQN's ε-greedy returns plateau at 200 while its greedy policy balances for 500 steps. which number is "the" performance?

<details><summary>answer</summary>

they answer different questions. training returns include exploration (ε random actions knock the pole over); the greedy policy is what you'd deploy. report greedy evaluation over several episodes and seeds, with uncertainty, and say which one you're plotting.

</details>

5. why might a *larger* target period $C$ learn *slower* but more *stably*?

<details><summary>answer</summary>

each period is one approximate Bellman backup, so values propagate at most one backup per $C$ gradient steps (on paper 4). but within a period the regression target is fixed, so the optimization is ordinary supervised learning with no feedback loop. a smaller $C$ propagates faster and drifts closer to the unstable "no target network" regime.

</details>

6. double DQN changes one line. why does it help more on some tasks than others?

<details><summary>answer</summary>

maximization bias grows with the noise in the estimates and the number of actions competing in the max, and it compounds through bootstrapping over long horizons. with two actions, deterministic dynamics and a short effective horizon the bias is small; with many actions and noisy rewards it can dominate (van Hasselt et al. report overestimates of several hundred percent on some Atari games).

</details>

7. Huber or MSE: which is more robust to a rare, huge TD error, and what does it cost?

<details><summary>answer</summary>

Huber: its gradient is bounded by 1 per sample, so one outlier can't take a huge step. the cost is slower correction of genuinely large errors (they get a constant-size gradient instead of a proportional one), and an implicit change of the regression target from the mean towards something between the mean and the median.

</details>

## common bugs

- **storing `done` instead of `terminated`.** symptom: values too low (for positive rewards) and an agent that learns to fear time-outs. the checks' single-state task learns 2.8 instead of 5.
- **forgetting `no_grad` on the target.** symptom: slower, less stable learning, sometimes divergence; the target network's parameters get gradients.
- **`gather` with the wrong shape.** `q.gather(1, actions)` needs `actions` as an int64 column `(B, 1)`; passing `(B,)` errors, and broadcasting `q[:, actions]` silently builds a `(B, B)` matrix.
- **Q-values for the wrong observation.** after `env.step`, update `obs = next_obs` *before* resetting on episode end, and store the transition with the old `obs`.
- **training before the buffer has data.** symptom: early collapse. respect `learning_starts`.
- **aliasing the target network.** `target_net = q_net` makes them the same object; use `copy.deepcopy` (or a fresh net plus `load_state_dict`).

## going deeper

- Mnih et al. (2015), "Human-level control through deep reinforcement learning", *Nature* 518. DQN: replay, target networks, the Atari protocol. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
- Lin (1992), "Self-improving reactive agents based on reinforcement learning, planning and teaching", *Machine Learning* 8. experience replay. [doi:10.1007/BF00992699](https://doi.org/10.1007/BF00992699)
- van Hasselt, Guez & Silver (2016), "Deep reinforcement learning with double Q-learning", *AAAI*. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)
- Wang et al. (2016), "Dueling network architectures for deep reinforcement learning", *ICML*. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)
- Hessel et al. (2018), "Rainbow: combining improvements in deep reinforcement learning", *AAAI*. which of the many DQN extensions matter, by ablation. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)
- Barto, Sutton & Anderson (1983), "Neuronlike adaptive elements that can solve difficult learning control problems", *IEEE Transactions on Systems, Man, and Cybernetics* 13(5). the cart–pole task. [doi:10.1109/TSMC.1983.6313077](https://doi.org/10.1109/TSMC.1983.6313077)
- Florian (2005/2007), "Correct equations for the dynamics of the cart-pole system", technical report, Center for Cognitive and Neural Studies, Romania. [coneural.org/florian/papers/05_cart_pole.pdf](https://coneural.org/florian/papers/05_cart_pole.pdf)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §6.7 (maximization bias) and §16.5 (DQN). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)

## connections

- ← builds on: Q-learning, maximization bias and double Q-learning (ch 23 · topic 05), the deadly triad and semi-gradient TD (ch 23 · topic 06), the `Env` protocol and `epsilon_greedy` (ch 23 · topics 02 and 04), `MLP` and `ema_update` (ch 15 · topic 07), the Huber loss (ch 08).
- → used in: `CartPole` is the test bed for REINFORCE, A2C and PPO (topics 02–04); the replay buffer stores SAC's continuous actions (topic 05) and the offline datasets of topic 07, whose CQL penalty is added to this topic's `dqn_loss`; `ema_update` target networks return in SAC (topic 05) and, as EMA teachers, in BYOL (ch 25).
