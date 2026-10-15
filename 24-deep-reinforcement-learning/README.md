# 24 · deep reinforcement learning

> scale RL with neural networks, and do it reproducibly.

chapter 23's agents kept a number for every state, or a hand-designed feature for every situation. this chapter replaces the tables with neural networks, and in doing so meets every way deep RL goes wrong: values that climb into the millions, gradient estimates that point the wrong way half the time, policies that collapse after one bad update, critics that are confidently wrong, offline datasets that a Q-learner exploits into nonsense, and learning curves that say more about the random seed than about the algorithm. you'll build the methods that work anyway (DQN, REINFORCE, A2C with GAE, TRPO's natural gradient and PPO, SAC, MCTS with Dyna and AlphaZero-lite, behavior cloning, DAgger and CQL) on environments you write yourself, and you'll measure what each ingredient actually buys. on the way you'll see a target network matter far more than replay, a baseline turn single-episode gradients from coin flips into reliable votes, the PPO clip slow the policy's drift without stopping it, a policy at temperature 0.2 spread itself uniformly rather than cover two good actions, AlphaZero-lite fall short of plain random rollouts at laptop scale, and stay there with four times the training, and offline DQN collapse on a dataset whose behaviour policy balanced the pole 475 steps out of 500.

**you will build:** `CartPole` and a replay buffer, Q-networks (plain and dueling), the (double) DQN loss and agent; discounted returns, a categorical policy, the REINFORCE loss and agent with a learned baseline; generalized advantage estimation, an actor–critic module for discrete and continuous actions, a vectorized rollout collector that handles time limits correctly, and A2C; the clipped surrogate, explained variance, conjugate gradients, Fisher-vector products and the natural-gradient step, the PPO update and agent; `Pendulum`, the squashed-Gaussian actor and twin critics, SAC's three losses and the agent with automatic temperature; Dyna-Q, Connect Four, PUCT Monte Carlo tree search, and AlphaZero-lite's network, loss and self-play; behavior cloning, DAgger, the CQL penalty and offline Q-learning. module: **`mlfs.deeprl`** (topics 01–07), whose `ppo_clip_loss` and `compute_gae` chapter 27 uses for RLHF.
**prerequisites:** chapter 23 (the `Env` protocol, terminated vs truncated, `epsilon_greedy`, Q-learning, TD(λ), Monte Carlo methods, UCB), chapter 15 (`MLP`, `ema_update`, training loops), chapter 05 (bootstrap intervals and permutation tests, used from topic 01's lab to the capstone), chapters 12 and 21 (the reparameterization trick, change of variables).
**time · scaffolding:** about 115 hours · L4 (spec-driven): each topic's lesson has **spec** boxes that fix the names, arguments and conventions (the order of random draws where exactness matters, which flag cuts the bootstrap, the shapes of everything); the notebooks give you signatures with one-line docstrings and the checks, and the internal structure is yours to design. one hint per stage. the capstone gives you a brief and checks.
**newly unlocked library components:** after this chapter you may use reference RL libraries' algorithm implementations for comparison and heavy lifting in later chapters (e.g. Stable-Baselines3's or CleanRL's DQN, PPO and SAC), and Gymnasium environments through a thin adapter (topic 01, idea 1), since you've built the agents, buffers, losses and environments yourself; `torch.distributions.Independent`, `Normal` and `Categorical` were allowed from topic 02 on as the distributions you know from ch 04 and 21. inside `mlfs`, later chapters use only the contract in [`CONTRACTS.md`](../CONTRACTS.md): chapter 27 trains language models with your `ppo_clip_loss` and `compute_gae`.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [deep Q-networks](01-deep-q-networks/) | `CartPole`, `ReplayBuffer`, `QNetwork`, `dqn_loss` → `mlfs.deeprl`, `train_dqn` | without a target network the Q-values reach two million (no state is worth more than 100) and nothing is learned, while removing replay barely matters on CartPole; a policy that balances for 500 steps underestimates its own value, failing ones overestimate theirs |
| 02 | [policy gradients](02-policy-gradients/) | `discounted_returns`, `CategoricalPolicy`, `reinforce_loss` → `mlfs.deeprl`, `train_reinforce` | single-episode gradients point almost exactly along or exactly against the true gradient, half and half without a baseline (mean cosine −0.16), mostly along with one (0.82); the baseline only starts paying after ~250 episodes |
| 03 | [actor-critic](03-actor-critic/) | `compute_gae`, `ActorCritic` → `mlfs.deeprl`, `RolloutCollector`, `a2c_loss`, `train_a2c` | Monte Carlo targets (λ = 1) have the worst error with *every* critic on the random walk; in A2C, λ = 0 barely learns CartPole while 0.9, 0.97 and 1 are indistinguishable over four seeds |
| 04 | [trust regions & PPO](04-trust-regions-and-ppo/) | `ppo_clip_loss`, `explained_variance` → `mlfs.deeprl`, `conjugate_gradient`, `fisher_vector_product`, `natural_gradient_step`, `ppo_update`, `train_ppo` | the clip doesn't hold ratios inside [0.8, 1.2] (the largest reaches 1.42 after 40 epochs), but it cuts the KL 50×; removing it halves PPO's learning speed at 20 epochs per batch, while gradient clipping and lr annealing change nothing measurable |
| 05 | [continuous control](05-continuous-control/) | `Pendulum`, `SquashedGaussianActor`, `TwinQ` → `mlfs.deeprl`, `sac_critic_target`, `sac_actor_loss`, `sac_alpha_loss`, `train_sac` | at α ≥ 0.2 SAC's policy on a two-peaked reward goes uniform instead of covering both peaks; the pendulum is solved even at α = 0, and only α = 5 fails |
| 06 | [model-based RL & planning](06-model-based-rl-and-planning/) | `dyna_q`, `ConnectFour`, `MCTS` → `mlfs.deeprl`, `PolicyValueNet`, `alphazero_loss`, `self_play` | 50 planning steps per real step learn the Dyna maze in 3 episodes instead of 27; a 400-simulation search beats a 100-simulation one in 14 of 16 games; AlphaZero-lite at laptop scale plateaus below random rollouts (scores 0.25–0.49 over 5 training seeds, no better after 4× the training) |
| 07 | [offline RL & imitation](07-offline-rl-and-imitation/) | `behavior_cloning`, `run_policy`, `dagger`, `cql_penalty` → `mlfs.deeprl`, `train_offline_q` | a clone of one expert episode balances for a median of 112 steps, a clone of two for 500; offline DQN on a 20,000-step dataset from a controller that balances 475 steps falls in 9, with Q-values of half a million; CQL balances for 500 |

## capstone

*PPO, reproducibly*: you build the **acrobot** (the two-link swing-up task with the published dynamics and RK4 integration), greedy evaluation with standard errors, learning curves with bootstrap bands over seeds, a two-sample comparison with a bootstrap interval and a permutation test, and a multi-seed runner; then you show over 10 seeds that your PPO solves CartPole (greedy return 500 on every seed) and the acrobot (−124.6, 95% CI [−208.9, −80.9], dragged down by one seed whose greedy policy never swings up), compare it with A2C (no detectable difference: p = 0.93), and test an implementation detail of your choice with a hypothesis stated beforehand. small CPU config: about 10 seconds; full config: about 10 minutes. the curriculum's alternative, *AlphaZero-lite for Connect Four*, is the main stretch goal.

see [`capstone/brief.md`](capstone/brief.md). the chapter is followed by research milestone [**R3**](../milestones/R3.md): reproduce an RL paper at small scale.

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (DQN, double-DQN and SAC targets; GAE with a truncation; the PPO clip and a bandit's policy gradient), and a 60-minute blind rebuild of the actor–critic core: GAE, `ActorCritic`, the rollout collector and the clipped loss.

## notation

| symbol | meaning |
|---|---|
| $s, a, r, s'$ | state (observation), action, reward, next state of a transition |
| $d$ | the `terminated` flag (cuts the bootstrap); truncation never does |
| $\gamma$, $\lambda$ | discount; GAE / λ-return parameter |
| $Q_\theta$, $Q_{\bar\theta}$ | online and target Q-networks; $\bar\theta$ updated by copying or $\bar\theta \leftarrow (1 - \tau)\bar\theta + \tau\theta$ |
| $\pi_\theta(a\mid s)$, $\pi_{\text{old}}$ | the policy and the policy that collected the batch |
| $V_\phi(s)$ | the critic (state value) |
| $G_t$ | return (reward-to-go) from step $t$ |
| $\delta_t$, $\hat A_t$ | TD error $r_{t+1} + \gamma(1 - d_t)V(s_{t+1}) - V(s_t)$; GAE advantage $\sum_l(\gamma\lambda)^l\delta_{t+l}$ |
| $r_t(\theta)$, $\varepsilon$ | probability ratio $\pi_\theta/\pi_{\text{old}}$; PPO's clip range |
| $F$, $g$, $\delta$ | Fisher information (Hessian of the mean KL), policy gradient, TRPO's KL radius |
| $\mathcal H$, $\alpha$, $\bar{\mathcal H}$ | entropy; SAC's temperature (also CQL's weight, by context); target entropy |
| $u$, $a_{\max}$ | pre-squash Gaussian sample and action bound of the squashed actor, $a = a_{\max}\tanh u$ |
| $N(a), W(a), Q(a), P(a)$ | MCTS visit counts, total values, mean values and priors at a node; $c_{\text{puct}}$ the exploration constant |
| $\pi$, $z$ | AlphaZero's visit-count target and game outcome (for the player to move) |
| $\mathcal D$, $d_{\pi}$ | an offline dataset; the state distribution a policy visits |

## how this connects

- builds on: Q-learning, TD(λ), Monte Carlo and the deadly triad (ch 23 · topics 04–06), the `Env` protocol and time limits (ch 23 · topic 02), UCB (ch 23 · topic 01), `MLP` and `ema_update` (ch 15 · topic 07), bootstrap intervals and permutation tests (ch 05), the reparameterization trick and change of variables (ch 12, ch 21), KL divergence and conjugate gradients (ch 05, ch 06).
- used later in: RL from human feedback, where `ppo_clip_loss` and `compute_gae` train a language model against a reward model (ch 27 · topic 04); supervised fine-tuning as behavior cloning (ch 27 · topic 03); search at inference time (ch 27); distribution shift and robustness (ch 28); research milestone R3 and the reproduction protocol (ch 30 · topic 02).

## core references

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapters 8, 13 and 16. [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Mnih et al. (2015), "Human-level control through deep reinforcement learning", *Nature* 518. [doi:10.1038/nature14236](https://doi.org/10.1038/nature14236)
- Schulman, Wolski, Dhariwal, Radford & Klimov (2017), "Proximal policy optimization algorithms". [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Haarnoja, Zhou, Abbeel & Levine (2018), "Soft actor-critic", *ICML*. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Silver et al. (2017), "Mastering the game of Go without human knowledge", *Nature* 550. [doi:10.1038/nature24270](https://doi.org/10.1038/nature24270)
- Levine, Kumar, Tucker & Fu (2020), "Offline reinforcement learning: tutorial, review, and perspectives on open problems". [arXiv:2005.01643](https://arxiv.org/abs/2005.01643)
- Henderson et al. (2018), "Deep reinforcement learning that matters", *AAAI*. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Achiam (2018), *Spinning Up in Deep RL*. [spinningup.openai.com](https://spinningup.openai.com)
