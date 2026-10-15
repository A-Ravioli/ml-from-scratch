# policy gradients

## why this matters

a Q-learner acts by taking $\arg\max_aQ(s, a)$. that one operation causes three problems. it needs a max over actions, which is easy with two buttons and hopeless with a continuous torque. it produces deterministic policies, but some problems need randomness (rock–paper–scissors, or any state you can't tell apart from another one that needs a different action). and a tiny change in the Q-values can flip the argmax and change the behaviour completely, which makes learning jumpy.

**policy gradient** methods skip the values and optimize the policy directly: parameterize $\pi_\theta(a\mid s)$ as a network that outputs a distribution, write the expected return $J(\theta)$ as a function of $\theta$, and climb its gradient. the surprise is that this gradient can be estimated from sampled episodes alone, without a model of the environment and without differentiating through it: the **score-function** (or likelihood-ratio) trick turns "how would the return change if I changed the policy" into "make the actions that led to high returns more likely". the algorithm, **REINFORCE** (Williams, 1992), is ten lines. its weakness is variance: the gradient estimate from one episode is so noisy that most of this topic, and the next two, are about taming it. the first and cheapest tool is a **baseline**, and in the lab you'll measure how much it buys.

## warm-up: predict first

1. you can't differentiate the environment. how can you possibly compute $\nabla_\theta\,\mathbb E[\text{return}]$?
2. an episode earns return 100. should the agent make *every* action of that episode more likely?
3. subtracting a constant $b$ from every return changes the loss. does it change the expected gradient?
4. a two-armed bandit pays 1 for arm 0 and 0 for arm 1; your softmax policy picks each with probability 1/2. what's the gradient of the expected reward with respect to the two logits?
5. one episode's gradient estimate on CartPole: how well do you think it points in the direction of the true gradient (cosine similarity)?

## prerequisites check

- [ ] you know returns, discounting and Monte Carlo estimation (→ ch 23 · topics 02 and 04)
- [ ] you know the softmax, cross-entropy and its gradient (→ ch 08, ch 14)
- [ ] you can sample from and take log-probabilities of a `torch.distributions` object (→ ch 21)
- [ ] you know what makes an estimator unbiased and what its variance is (→ ch 05)
- [ ] you have a `CartPole` (→ topic 01)

## core ideas

### idea 1: returns, from every step

the objective is the expected return of the policy from the start, $J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[G_0]$, where $\tau = (s_0, a_0, r_1, s_1, \dots)$ is an episode (a **trajectory**) and $G_t = \sum_{k\ge0}\gamma^kr_{t+k+1}$. you'll need $G_t$ at *every* step $t$ of an episode, not just $G_0$, and the recursion $G_t = r_{t+1} + \gamma G_{t+1}$ (with $G_T = 0$ after the last step) computes all of them in one backward pass, in $O(T)$ rather than the $O(T^2)$ of summing each one separately.

why every step: the action at step $t$ can't affect rewards that came before it. so when you judge $a_t$, only the rewards from $t$ on are evidence (warm-up 2). using $G_t$, the **reward-to-go**, instead of $G_0$ for every action removes noise without adding bias (on paper 3).

indexing convention: in code, `rewards[t]` is the reward *received after* action `t` (what the math calls $r_{t+1}$), so `G[t] = rewards[t] + gamma * G[t+1]`.

> **spec.** `discounted_returns(rewards, gamma)`: `rewards` is a 1-D ndarray or tensor holding one episode's rewards; returns $G$ of the same type and shape (floating point, even for integer rewards) with `G[t] = rewards[t] + gamma * G[t+1]` and `G[-1] = rewards[-1]`. don't modify the input.

> **▶ build: stage 1.** implement `discounted_returns`.

### idea 2: a policy is a distribution

for discrete actions the natural policy is a network that outputs one **logit** per action, turned into probabilities by a softmax: $\pi_\theta(a\mid s) = \operatorname{softmax}(f_\theta(s))_a$. `torch.distributions.Categorical(logits=...)` wraps that, and gives you the two operations everything else needs: `sample()` to act, and `log_prob(a)` to learn. working with log-probabilities and logits (never with probabilities you then take the log of) keeps things numerically stable: `Categorical` computes log-softmax internally.

the policy must be **stochastic** while learning. exploration comes for free (actions with probability 0.2 get tried a fifth of the time), and, more fundamentally, the gradient you're about to derive needs the policy to put probability on the actions it evaluates. as training goes on the probabilities usually sharpen towards a near-deterministic policy on their own.

a design choice you'll see everywhere: tanh hidden activations for small policy networks. they're bounded and smooth, which keeps the logits from blowing up early in training; ReLU works too.

> **spec.** `CategoricalPolicy(obs_dim, n_actions, *, hidden_sizes=(64, 64))` is an `nn.Module`: an MLP from `obs_dim` to `n_actions` logits. `forward(obs)` takes `(B, obs_dim)` and returns a `torch.distributions.Categorical` with batch shape `(B,)`. `act(obs)` takes one observation as an ndarray, *samples* an action under `torch.no_grad()` (using torch's global generator) and returns it as a Python `int`.

> **▶ build: stage 2.** implement `CategoricalPolicy`.

### idea 3: the policy gradient

you want $\nabla_\theta J$, but $J$ is an expectation over trajectories whose *distribution* depends on $\theta$, through dynamics you can't differentiate. the **log-derivative trick** gets around it. for any distribution $p_\theta(x)$ and function $f$,

$$\nabla_\theta\,\mathbb E_{x\sim p_\theta}[f(x)] = \int f(x)\nabla_\theta p_\theta(x)\,dx = \int f(x)\,p_\theta(x)\nabla_\theta\log p_\theta(x)\,dx = \mathbb E_{x\sim p_\theta}\big[f(x)\nabla_\theta\log p_\theta(x)\big],$$

an expectation you can estimate by sampling (warm-up 1). apply it to trajectories: $\log p_\theta(\tau) = \log p(s_0) + \sum_t[\log\pi_\theta(a_t\mid s_t) + \log p(s_{t+1}\mid s_t, a_t)]$, and the dynamics terms don't depend on $\theta$, so their gradients vanish. with reward-to-go (idea 1),

$$\nabla_\theta J(\theta) = \mathbb E_{\tau\sim\pi_\theta}\Big[\sum_t\gamma^t\,G_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big].$$

(the $\gamma^t$ factor is almost always dropped in practice, and we drop it too. be aware of what that costs: the resulting update is no longer the gradient of the discounted objective, and in general it isn't the gradient of *any* objective (Nota & Thomas, 2020). it's a biased direction that happens to work well, partly because it doesn't shrink the weight of late steps towards zero.) read the formula as weighted maximum likelihood: push up the log-probability of every action taken, in proportion to how good what followed was.

**baselines.** subtract any $b(s_t)$ that doesn't depend on the action:

$$\nabla_\theta J = \mathbb E\Big[\sum_t\big(G_t - b(s_t)\big)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big].$$

this is still exact (warm-up 3), because $\mathbb E_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)] = \nabla_\theta\sum_a\pi_\theta(a\mid s) = \nabla_\theta1 = 0$. but it can change the *variance* enormously. on CartPole every return is positive, so without a baseline every action is reinforced and the learning signal is the small difference between "reinforced a lot" and "reinforced a bit". with $b(s) \approx V^\pi(s)$, the weight $G_t - V(s_t)$ says whether the action did better or worse than usual from that state, and its sign carries the information.

in code you never write the gradient. you write a **surrogate loss** whose gradient is the (negative) estimate, and let autograd do the rest:

$$\mathcal L(\theta) = -\frac1N\sum_i\big(G_i - b_i\big)\log\pi_\theta(a_i\mid s_i).$$

its value means nothing (it's not the return, and it can go up while the policy improves); only its gradient matters. $G$ and $b$ are constants here: if $b$ comes from a value network, gradients must not flow into it through this loss.

> **spec.** `reinforce_loss(log_probs, returns, *, baseline=None)`: `log_probs` is a tensor `(N,)` (with grad), `returns` a tensor or array `(N,)`, `baseline` `None`, a float, or a tensor/array `(N,)`. returns the scalar $-\operatorname{mean}((G - b)\log\pi)$ with $G$ and $b$ detached.

> **🔮 predict:** in warm-up 4's bandit, with a baseline of 0.5 (the expected reward), what gradient estimate does a *single* sample give, for each of the two possible actions? (the worked example has the answer, and it's surprising.)

> **▶ build: stage 3.** implement `reinforce_loss`.

### idea 4: REINFORCE

the algorithm: run the current policy for a batch of episodes; compute every step's reward-to-go; take a gradient step on the surrogate loss; repeat. it's **on-policy**: the gradient formula is an expectation under the *current* policy, so each batch is used for one step and thrown away (contrast with replay in topic 01). that's the price of not learning values: every update needs fresh data.

the baseline we'll use is a learned **value network** $V_\phi(s)$, regressed onto the returns with its own loss $\frac1N\sum_i(V_\phi(s_i) - G_i)^2$ and its own optimizer. it's a critic in all but name; topic 03 makes it one.

with a value network you can also fix the time-limit problem properly. a truncated episode's returns are cut off: the final steps look worse than they are because their future wasn't observed. with $V_\phi$ you can **bootstrap** the missing tail: add $\gamma V_\phi(s_T)$ to the last reward before computing returns, exactly as TD does (ch 23 · topic 05). without a value network there's nothing to bootstrap from, and you live with the bias.

> **spec.** `train_reinforce(env, n_episodes, *, gamma=0.99, lr=1e-2, batch_episodes=1, hidden_sizes=(32,), baseline=None, lr_value=1e-2, seed=None)` trains a `CategoricalPolicy` (with `hidden_sizes`) on a Box-observation, Discrete-action env for exactly `n_episodes` episodes, in batches of `batch_episodes` (the last batch may be smaller), one Adam step (`lr`) per batch on `reinforce_loss` with reward-to-go returns. `baseline=None` uses no baseline; `baseline='value'` adds an MLP value network (same `hidden_sizes`, tanh, one output) trained by MSE on the returns with its own Adam (`lr_value`), used as the baseline, and bootstraps truncated (not terminated) episodes by adding $\gamma V(s_T)$ to their last reward; any other value raises `ValueError`. seed the env once (first reset) from a NumPy generator made from `seed`, and torch from `seed` before building the networks. returns `(policy, history)` with `history['episode_returns']` (undiscounted, one per episode) and, for `baseline='value'`, `history['value_net']`.

> **🔮 predict:** REINFORCE on CartPole, with and without the value baseline. which learns faster, and does the baseline matter more early or late in training?

> **▶ build: stage 4.** implement `train_reinforce`.

## worked example

**returns.** rewards $(1, 2, 3)$, $\gamma = 0.5$: $G_2 = 3$, $G_1 = 2 + 0.5\cdot3 = 3.5$, $G_0 = 1 + 0.5\cdot3.5 = 2.75$.

**the loss.** log-probabilities $(-0.5, -1, -2)$, returns $(1, 2, -1)$: $\mathcal L = -\tfrac13(1\cdot(-0.5) + 2\cdot(-1) + (-1)\cdot(-2)) = 0.1667$, and $\partial\mathcal L/\partial\log\pi_i = -G_i/3 = (-\tfrac13, -\tfrac23, \tfrac13)$: gradient *descent* raises the log-probabilities of the first two actions and lowers the third's. with a baseline of 1 the weights become $(0, 1, -2)$ and $\mathcal L = -1$.

**a bandit, exactly.** two arms paying 1 and 0, logits $\theta = (0, 0)$, so $\pi = (\tfrac12, \tfrac12)$ and $J = \tfrac12$. for a softmax, $\nabla_\theta\log\pi_a = e_a - \pi$. the exact gradient is $\nabla J = \sum_a\pi_ar_a(e_a - \pi) = \tfrac12(\tfrac12, -\tfrac12) = (0.25, -0.25)$.

without a baseline, a single sample gives $r_a(e_a - \pi)$: $(0.5, -0.5)$ if arm 0 is drawn, $(0, 0)$ if arm 1 is. right on average, but each sample is off by a factor of 2 or 0. with the baseline $b = J = 0.5$: arm 0 gives $0.5\cdot(0.5, -0.5) = (0.25, -0.25)$, arm 1 gives $-0.5\cdot(-0.5, 0.5) = (0.25, -0.25)$. **every** sample is exactly the true gradient: the baseline took the variance to zero. (that's special to two arms, where the score vectors of the two actions are parallel; with more arms the variance shrinks but doesn't vanish.)

your code must reproduce these numbers.

## on paper

1. **the score has mean zero.** show that $\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}[\nabla_\theta\log\pi_\theta(a\mid s)] = 0$, and conclude that any action-independent baseline leaves the policy gradient unbiased.

<details><summary>solution</summary>

$\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s) = \sum_a\nabla_\theta\pi_\theta(a\mid s) = \nabla_\theta\sum_a\pi_\theta(a\mid s) = \nabla_\theta1 = 0$. then $\mathbb E[b(s_t)\nabla\log\pi(a_t\mid s_t)] = \mathbb E_{s_t}\big[b(s_t)\,\mathbb E_{a_t}[\nabla\log\pi(a_t\mid s_t)\mid s_t]\big] = 0$. if $b$ depended on $a_t$ the inner expectation wouldn't factor, which is why an action-dependent "baseline" needs a correction term.

</details>

2. **the softmax score.** for $\pi = \operatorname{softmax}(\theta)$, show $\partial\log\pi_a/\partial\theta_j = \mathbb 1[a = j] - \pi_j$, and use it to derive the exact bandit gradient $\partial J/\partial\theta_j = \pi_j(r_j - J)$.

<details><summary>solution</summary>

$\log\pi_a = \theta_a - \log\sum_ke^{\theta_k}$, so $\partial/\partial\theta_j = \mathbb 1[a = j] - e^{\theta_j}/\sum_ke^{\theta_k}$. then $\partial J/\partial\theta_j = \sum_a\pi_ar_a(\mathbb 1[a = j] - \pi_j) = \pi_jr_j - \pi_jJ$. this is the gradient-bandit algorithm of Sutton & Barto §2.8; note that $J$ itself appears as the natural baseline.

</details>

3. **reward-to-go is unbiased.** show that for $k < t$, $\mathbb E[r_{k+1}\nabla_\theta\log\pi_\theta(a_t\mid s_t)] = 0$, so replacing $G_0$ by $G_t$ as the weight of step $t$ removes only zero-mean terms.

<details><summary>solution</summary>

condition on everything up to $s_t$ (which includes $r_{k+1}$ for $k < t$): $\mathbb E[r_{k+1}\nabla\log\pi(a_t\mid s_t)] = \mathbb E\big[r_{k+1}\,\mathbb E[\nabla\log\pi(a_t\mid s_t)\mid s_0, \dots, s_t]\big] = 0$ by problem 1. the past rewards act like a (history-dependent) baseline, and dropping them removes their variance contribution.

</details>

4. **the best constant baseline.** for a single-step problem with score $g = \nabla_\theta\log\pi(a)$ and reward $r$, find the constant $b$ that minimizes the total variance $\mathbb E\|(r - b)g\|^2 - \|\nabla J\|^2$. is it the expected reward?

<details><summary>solution</summary>

$\frac{d}{db}\mathbb E[(r - b)^2\|g\|^2] = -2\mathbb E[(r - b)\|g\|^2] = 0$ gives $b^* = \mathbb E[r\|g\|^2]/\mathbb E[\|g\|^2]$: the expected reward *weighted by the squared score*, not $\mathbb E[r]$ itself. they coincide when $\|g\|^2$ is the same for every action, as in the symmetric bandit of the worked example. in practice $V(s)$ is used because it's nearly as good and much easier to learn.

</details>

## experiment

`lab.py` investigates, with your code and your `CartPole`:

1. how noisy is a policy gradient estimate? at a fixed policy, single-episode gradient estimates with total return, reward-to-go and a value baseline, compared with the true gradient.
2. REINFORCE learning curves with and without the value baseline, over several seeds.

## check your understanding

1. why is the value of the surrogate loss meaningless as a progress measure?

<details><summary>answer</summary>

it's $-\operatorname{mean}(w\log\pi)$ with weights from the current batch: its scale depends on the returns (which grow as the policy improves), its sign on the baseline, and the log-probabilities shrink in magnitude as the policy becomes confident. only its gradient at the current $\theta$ is meaningful. track the returns instead.

</details>

2. why can't REINFORCE reuse old episodes the way DQN reuses its buffer?

<details><summary>answer</summary>

the gradient is an expectation under the *current* policy. episodes from an older policy come from a different distribution; using them as if they were on-policy biases the gradient. you'd need importance weights $\pi_\theta(a\mid s)/\pi_{\text{old}}(a\mid s)$ per step (whose product over an episode has huge variance). topic 04's PPO reuses a batch for a few epochs with exactly such a ratio, clipped.

</details>

3. (the predict callout in idea 3.) the single-sample estimates with baseline 0.5?

<details><summary>answer</summary>

both actions give exactly $(0.25, -0.25)$, the true gradient: zero variance. with two arms the score vectors $e_a - \pi$ of the two actions are negatives of each other, and the baseline makes the weights $r_a - b$ opposite in sign too, so the products coincide.

</details>

4. why do policy-gradient methods typically use larger batches than DQN?

<details><summary>answer</summary>

a single episode's gradient is extremely noisy (lab, experiment 1), and an on-policy method can't average over a big buffer of past data. averaging over many episodes (or many parallel environments, topic 03) is the main way to reduce the variance of each step.

</details>

5. what goes wrong if gradients flow from `reinforce_loss` into the value network used as the baseline?

<details><summary>answer</summary>

the policy loss becomes $-\operatorname{mean}((G - V_\phi)\log\pi)$ in $\phi$ too, so the value net is pushed to make $V_\phi$ larger wherever $\log\pi$ is very negative, i.e. it optimizes something that has nothing to do with predicting returns. the baseline then stops being a good estimate of $V$, and the variance reduction disappears.

</details>

6. a sparse task pays $+1$ only on success, which a random policy almost never reaches. what does REINFORCE do?

<details><summary>answer</summary>

almost nothing: every episode has return 0, every weight is 0, the gradient is 0. policy gradients need occasional successes to learn from; exploration (entropy bonuses, curricula, demonstrations) has to come from somewhere else. value-based methods with replay at least propagate a rare success to many states.

</details>

## common bugs

- **using $G_0$ for every step.** symptom: learns, but much more slowly. use reward-to-go.
- **summing instead of averaging.** symptom: the effective learning rate scales with episode length, so it changes as the policy improves. take the mean.
- **sign error.** symptom: the return goes *down*. the loss is the *negative* of the objective.
- **not detaching the baseline.** symptom: the value network's loss doesn't decrease, and the baseline stops helping.
- **sampling in `act` with gradients on.** symptom: memory grows every step. act under `torch.no_grad()`, and recompute log-probabilities for the whole batch at update time.
- **`log(probs)` instead of `log_prob`.** symptom: `nan` once a probability underflows to 0. use logits and `Categorical.log_prob`.

## going deeper

- Williams (1992), "Simple statistical gradient-following algorithms for connectionist reinforcement learning", *Machine Learning* 8. REINFORCE and baselines. [doi:10.1007/BF00992696](https://doi.org/10.1007/BF00992696)
- Sutton, McAllester, Singh & Mansour (2000), "Policy gradient methods for reinforcement learning with function approximation", *NeurIPS* 12. the policy gradient theorem with function approximation. [papers.nips.cc/paper/1713](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
- Nota & Thomas (2020), "Is the policy gradient a gradient?", *AAMAS*. what dropping $\gamma^t$ does. [arXiv:1906.07073](https://arxiv.org/abs/1906.07073)
- Greensmith, Bartlett & Baxter (2004), "Variance reduction techniques for gradient estimates in reinforcement learning", *JMLR* 5. optimal baselines. [jmlr.org/papers/v5/greensmith04a.html](https://www.jmlr.org/papers/v5/greensmith04a.html)
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., §2.8 (gradient bandits) and chapter 13 (policy gradient methods). [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Mohamed, Rosca, Figurnov & Mnih (2020), "Monte Carlo gradient estimation in machine learning", *JMLR* 21. the score-function estimator next to the reparameterization trick you met in ch 12 and ch 21. [arXiv:1906.10652](https://arxiv.org/abs/1906.10652)
- Achiam (2018), *Spinning Up in Deep RL*, "Part 3: Intro to policy optimization". a careful derivation of reward-to-go and baselines. [spinningup.openai.com](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

## connections

- ← builds on: Monte Carlo returns (ch 23 · topic 04), gradient bandits and baselines (ch 23 · topic 01), softmax and cross-entropy (ch 08), the score function vs the reparameterization trick (ch 12 · topic 05, ch 21), `MLP` (ch 15), `CartPole` (topic 01).
- → used in: actor–critic methods replace the Monte Carlo return with a bootstrapped advantage (topic 03); PPO reuses each batch with a clipped importance ratio (topic 04); the same score-function gradient trains language models from human feedback (ch 27 · topic 04).
