# 23 · reinforcement learning foundations

> learning from interaction, from bandits to function approximation.

every model so far learned from a dataset someone handed it. this chapter's agents collect their own data, by acting, and their choices decide what they get to learn from. that feedback loop is what makes reinforcement learning hard and interesting: an agent has to explore to find out what's good, but every exploratory action costs reward; a good action now may pay off only much later; and the data it learns from keeps changing as its behaviour improves. you'll start with the simplest version of the problem (a slot machine with several arms) and end with an agent that teaches itself to drive an underpowered car up a hill from raw position and velocity. on the way you'll meet the exact theory (Markov decision processes, Bellman equations, dynamic programming) and the three families of learning methods built on it (Monte Carlo, temporal-difference and approximate), and you'll be surprised by an "optimal" algorithm that loses to a simpler one at a realistic horizon, a value iteration whose policy is right a thousand iterations before its values are, an importance-sampling estimator, right on average, that lands below the truth in nearly 9 runs out of 10, an agent that learns the best path along a cliff and earns less for it, and a seven-state problem with all-zero rewards on which a noise-free learning rule diverges.

**you will build:** a bandit testbed with ε-greedy, UCB and Thompson sampling agents and the Lai–Robbins bound; the gym-style environment protocol (`Discrete`, `Box`, `Env`, `rollout`), a gridworld that exposes its model `P`, `R`, and iterative policy evaluation; greedy policies, policy iteration, value iteration and modified policy iteration; blackjack, ε-greedy exploration, Monte Carlo prediction and control, and off-policy evaluation by ordinary and weighted importance sampling; cliff walking, TD(0), SARSA, Q-learning, expected SARSA, n-step SARSA and TD($\lambda$) with eligibility traces; tile coding, mountain car, linear semi-gradient SARSA, and the expected dynamics of semi-gradient TD. module: **`mlfs.rl`** (topics 01–06), whose environments and protocol chapter 24's deep RL agents run on.
**prerequisites:** chapter 04 (Markov chains, running means, Hoeffding's inequality), chapter 05 (Beta–Bernoulli updating and KL divergence, for bandits), chapter 06 (stochastic approximation and step sizes), chapter 02 (linear solves and eigenvalues), chapter 03 · topic 01 (contractions and fixed points), and `as_rng` from chapter 00. no neural networks: everything here is NumPy.
**time · scaffolding:** about 85 hours · L3 (independent): every function comes with its signature and a one-line docstring; the exact conventions (tie-breaking, the order of random draws, how environments are seeded) are in the lesson's spec boxes. hints have two levels. the capstone gives you the stubs and the exact-solution plumbing; you write the experiment harness and design a game of your own.
**newly unlocked library components:** after this chapter you may use Gymnasium's environments, spaces and wrappers (e.g. `CartPole`, `LunarLander`, time-limit and observation wrappers) in later chapters' labs and experiments, since you've built the protocol, tabular and classic-control environments and the interaction loop yourself; your `Env`, `Discrete` and `Box` follow the same `reset`/`step` API, so agents written against your protocol can drive Gymnasium environments (chapter 24 wraps them); the converse doesn't hold: your environments aren't drop-in Gymnasium environments (no `dtype` on the spaces, no `close`, `unwrapped` or `metadata`). inside `mlfs` modules, later chapters use *your* protocol and tabular tools through `mlfs.rl`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md): chapter 24 runs its DQN, policy-gradient and actor–critic agents on your `Env`/`Box`/`Discrete` protocol and `rollout`, explores with your `epsilon_greedy`, and uses your tabular environments (`GridWorld`, `CliffWalking`) and DP solvers as ground truth.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [bandits](01-bandits/) | `BernoulliBandit`, `GaussianBandit`, `run_bandit`, `EpsilonGreedy`, `UCB`, `ThompsonSampling` → `mlfs.rl`, `lai_robbins_constant` | at 3,000 rounds ε-greedy (0.1) beats UCB, the algorithm with the logarithmic guarantee; the hardest gap is neither the smallest nor the largest; Thompson's regret stays far *below* the Lai–Robbins "lower bound" for 20,000 rounds |
| 02 | [MDPs & bellman equations](02-mdps-and-bellman-equations/) | `Discrete`, `Box`, `Env`, `GridWorld`, `rollout`, `policy_evaluation`, `q_from_v` → `mlfs.rl` | with $\gamma = 0.5$ the random walker's values can't see the goal; paying $+1$ per step makes "walk into a wall forever" the best policy; a time limit biases Monte Carlo returns by 8 steps |
| 03 | [dynamic programming](03-dynamic-programming/) | `greedy_policy`, `policy_iteration`, `value_iteration` → `mlfs.rl`, `bellman_optimality`, `modified_policy_iteration` | value iteration needs 18,460 sweeps at $\gamma = 0.999$, policy iteration 4 evaluations at every $\gamma$; value iteration's policy is optimal after 3 backups, its values after 1,838 |
| 04 | [monte carlo methods](04-monte-carlo-methods/) | `Blackjack`, `epsilon_greedy`, `mc_control`, `off_policy_mc_prediction` → `mlfs.rl`, `mc_prediction` | 500,000 games of MC control disagree with the exact optimum in 15 states and lose only 0.0044 per unit bet; ordinary importance sampling, right on average, is below the truth in nearly 9 runs out of 10 |
| 05 | [temporal-difference learning](05-temporal-difference-learning/) | `CliffWalking`, `td0_prediction`, `sarsa`, `q_learning`, `expected_sarsa`, `n_step_sarsa` → `mlfs.rl`, `td_lambda_prediction` | Q-learning learns the optimal path and earns −47 per episode, SARSA a safer one and earns −28; accumulating traces diverge for large $\lambda$ and $\alpha$; reporting time-outs as terminal halves every value |
| 06 | [function approximation](06-function-approximation/) | `TileCoder`, `MountainCar`, `linear_sarsa` → `mlfs.rl`, `td_matrices`, `expected_td` | optimism alone (ε = 0) gets the car up the hill; 8 offset tilings beat a single grid with the same number of features by 4×; on Baird's counterexample noise-free TD diverges for $\gamma$ above 0.883 |

## capstone

*design your own environment* (a small game) and solve it with tabular and approximate methods. you build **Catch** (a falling ball, a paddle, and wind) as a gym-style environment with its exact model, an environment checker that catches the six classic environment bugs (from flags of the wrong type to dynamics that disagree with the model), an evaluation harness with error bars, and features that let a linear learner generalize across positions; then you solve Catch exactly with value iteration (optimal return 0.680) and compare Q-learning, SARSA, Monte Carlo control and linear SARSA against it over seeds, and do the same for a game of your own design. small CPU config: a few seconds; full config: about 20 seconds.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 14 concept questions, three by-hand problems (policy evaluation and improvement on a corridor, TD and TD($\lambda$) updates, the variance of an importance-sampling estimate), and a 60-minute blind rebuild of the TD control family.

## notation

| symbol | meaning |
|---|---|
| $\mathcal S$, $\mathcal A$, $S$, $A$ | state and action sets; their sizes (`n_states`, `n_actions`) |
| $S_t$, $A_t$, $R_{t+1}$ | state, action and the reward that follows them, at time $t$ |
| $P(s'\mid s, a)$, $R(s, a)$ | transition probabilities (`P`, shape $(S, A, S)$) and expected rewards (`R`, shape $(S, A)$) |
| $\gamma$ | discount factor; $1/(1 - \gamma)$ is the effective horizon |
| $G_t$, $G_{t:t+n}$, $G_t^\lambda$ | return, n-step return, λ-return |
| $\pi(a\mid s)$, $b(a\mid s)$ | target policy, behaviour policy (arrays of shape $(S, A)$, or callables obs → action) |
| $V^\pi$, $Q^\pi$, $V^*$, $Q^*$ | state- and action-value functions of $\pi$, and the optimal ones |
| $P_\pi$, $R_\pi$, $T$, $T_\pi$ | the policy's transition matrix and reward vector; the optimality and expectation Bellman operators |
| $\delta_t$ | TD error $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ |
| $\alpha$, $\varepsilon$, $\lambda$, $n$ | step size, exploration rate, trace decay, number of bootstrapping steps |
| $e$ | eligibility trace |
| $\rho_{t:T-1}$, $W$ | importance-sampling ratio $\prod\pi/b$ and its running product |
| $\mu_a$, $\mu^*$, $\Delta_a$, $N_a$, $Q_a$ | bandit arm means, the best mean, gaps, pull counts, estimates |
| $\bar R_T$ | cumulative pseudo-regret after $T$ rounds |
| $x(s)$, $\Phi$, $w$ | feature vector, feature matrix (rows $x(s)^\top$), weights |
| $D$, $d$ | state weighting $\operatorname{diag}(d)$ in the expected TD update |
| $A$, $b$ | in topic 06: the expected TD update $w \leftarrow w + \alpha(b - Aw)$ (not the action set) |

## how this connects

- builds on: Markov chains (ch 04 · topic 07), running means, concentration and Beta–Bernoulli posteriors (ch 04–05), linear solves and eigenvalues (ch 02), contractions and fixed points (ch 03 · topic 01), stochastic approximation and step sizes (ch 06), linear models (ch 08), `as_rng` (ch 00).
- used later in: deep Q-networks, replay buffers and double Q-learning (ch 24 · topic 01), REINFORCE's Monte Carlo returns and baselines (ch 24 · topic 02), actor–critic methods and GAE's λ-weighted TD errors (ch 24 · topic 03), PPO's importance ratio (ch 24 · topic 04), MCTS's UCB-style tree policy and Dyna-Q (ch 24 · topic 06), offline RL and off-policy evaluation (ch 24 · topic 07), and RL from human feedback for language models (ch 27 · topic 04).

## core references

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press. the textbook this chapter follows; free online. [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Lattimore & Szepesvári (2020), *Bandit Algorithms*, Cambridge University Press. [tor-lattimore.com/downloads/book/book.pdf](https://tor-lattimore.com/downloads/book/book.pdf)
- Puterman (1994), *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, Wiley. [doi:10.1002/9780470316887](https://doi.org/10.1002/9780470316887)
- Szepesvári (2010), *Algorithms for Reinforcement Learning*, Morgan & Claypool. a compact, rigorous survey. [doi:10.2200/S00268ED1V01Y201005AIM009](https://doi.org/10.2200/S00268ED1V01Y201005AIM009)
- Bertsekas & Tsitsiklis (1996), *Neuro-Dynamic Programming*, Athena Scientific. the theory of approximate DP and TD with function approximation.
- Towers et al. (2024), "Gymnasium: a standard interface for reinforcement learning environments". the environment protocol. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
