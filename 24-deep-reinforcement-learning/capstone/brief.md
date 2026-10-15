# 24 · capstone: PPO, reproducibly

## goal

a single deep RL learning curve is mostly noise. two correct implementations, or one implementation run on two sets of five seeds, can look like different algorithms (Henderson et al., 2018), and "my method beats the baseline" is only a claim once it comes with an interval over seeds. this capstone is about doing deep RL the way the chapter's subtitle promised: *reproducibly*.

you'll take your PPO from topic 04 and show, with evidence that would survive review, that it solves two control tasks: CartPole (topic 01) and a harder one you build here, the **acrobot**, a two-link pendulum that has to swing its tip above a bar using a weak motor at the elbow. along the way you build the evaluation and statistics every deep RL result needs (greedy evaluation with error bars, learning curves with bootstrap bands over seeds, a two-sample comparison with an interval and a permutation test) and a runner that trains one configuration over many seeds. then you use them to compare PPO with A2C and to test one implementation detail of your choice, stating your hypothesis before you run it.

this pulls in the whole chapter: the `Env` protocol and time limits (topic 01), policy gradients and their variance (topic 02), GAE, `ActorCritic`, `RolloutCollector` and `train_a2c` (topic 03), `train_ppo` and its details (topic 04), and the honest-evaluation habits of every lab. the curriculum also offers *AlphaZero-lite for Connect Four*; it's this capstone's main stretch goal, on your topic 06 code.

## the task you build: the acrobot

two links of length 1 and mass 1 hang from a fixed pivot; the elbow motor applies a torque of $-1$, 0 or $+1$ (actions 0, 1, 2). the state is the two joint angles $\theta_1$ (shoulder, from hanging straight down) and $\theta_2$ (elbow, relative to the first link) and their velocities. the episode **terminates** when the tip rises above the bar, $-\cos\theta_1 - \cos(\theta_1 + \theta_2) > 1$; every step pays $-1$ except the terminating one, which pays 0; episodes are **truncated** after 500 steps. the observation is $(\cos\theta_1, \sin\theta_1, \cos\theta_2, \sin\theta_2, \dot\theta_1, \dot\theta_2)$.

the dynamics are the "book" equations of Sutton & Barto (1998, §11.3; Sutton, 1996), exactly as in Gymnasium's `Acrobot-v1`. with link masses $m = 1$, lengths $l = 1$, centres of mass at $l_c = 0.5$, moments of inertia $I = 1$ and $g = 9.8$:

$$d_1 = m_1l_{c1}^2 + m_2(l_1^2 + l_{c2}^2 + 2l_1l_{c2}\cos\theta_2) + I_1 + I_2,\qquad d_2 = m_2(l_{c2}^2 + l_1l_{c2}\cos\theta_2) + I_2,$$

$$\phi_2 = m_2l_{c2}g\cos(\theta_1 + \theta_2 - \tfrac\pi2),\qquad \phi_1 = -m_2l_1l_{c2}\dot\theta_2^2\sin\theta_2 - 2m_2l_1l_{c2}\dot\theta_2\dot\theta_1\sin\theta_2 + (m_1l_{c1} + m_2l_1)g\cos(\theta_1 - \tfrac\pi2) + \phi_2,$$

$$\ddot\theta_2 = \frac{\tau + \frac{d_2}{d_1}\phi_1 - m_2l_1l_{c2}\dot\theta_1^2\sin\theta_2 - \phi_2}{m_2l_{c2}^2 + I_2 - d_2^2/d_1},\qquad \ddot\theta_1 = -\frac{d_2\ddot\theta_2 + \phi_1}{d_1}.$$

one step integrates these with a single classical Runge–Kutta (RK4) step of $\Delta t = 0.2$, then wraps both angles into $[-\pi, \pi)$ and clips the velocities to $\pm4\pi$ and $\pm9\pi$. the start state is uniform in $[-0.1, 0.1]^4$. a random policy almost never swings up within 500 steps (return $-500$); a good one takes about 80 steps.

## data

none: everything is simulated. the tests use the acrobot, tiny one-step tasks and a short CartPole run, with fixed seeds.

## configurations

| config | where | sizes | runtime (laptop CPU, one thread) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 2 seeds; 8,000 steps per run; 3 evaluation episodes | about 10 seconds |
| **full** (your write-up) | default | 10 seeds; CartPole 100,000 steps, acrobot 150,000 steps per run; 20 greedy evaluation episodes | about 10 minutes with the reference `mlfs.deeprl` |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required code implemented and its checks green:
   - `Acrobot(*, max_steps=500)`: the environment above as an `mlfs.rl.Env`, with `observation_space` `Box(6)`, `action_space` `Discrete(3)`, the state in `self.state` (an array $(\theta_1, \theta_2, \dot\theta_1, \dot\theta_2)$) and the step counter in `self.t`; `reset` draws the state with `self.np_random.uniform(-0.1, 0.1, size=4)`; invalid actions raise `ValueError`.
   - `greedy(model)` → a callable obs → action for an `ActorCritic`: the most likely action (a Python `int`), or the Gaussian's mean for a continuous policy. `evaluate_policy(make_env, act, n_episodes, *, seed=0)` → `(mean, standard error)` of the undiscounted returns over `n_episodes` of one fresh `make_env()`, seeding only the first reset; the standard error is `std(ddof=1) / sqrt(n)`.
   - `learning_curves(histories, grid, *, window=10)` → an array `(n_runs, len(grid))`: for each run (a history with `episode_returns` and `episode_ends`), the mean return of the last `window` episodes that had ended by each grid step (`nan` before the first). `mean_ci(curves, *, n_boot=2000, alpha=0.05, rng=None)` → `(mean, lo, hi)` per column: the mean over runs and a percentile bootstrap interval that resamples runs, ignoring `nan`s.
   - `compare_seeds(a, b, *, n_boot=10000, rng=None)` → `{'diff', 'ci', 'p'}`: the difference of means, a percentile bootstrap interval that resamples the two groups *independently*, and the p-value of your `mlfs.stats.permutation_test`.
   - `run_seeds(make_env, train_fn, total_steps, seeds, *, eval_episodes=20, **kwargs)` → `{'histories', 'final'}`: calls `train_fn(make_env, total_steps, seed=s, **kwargs)` (e.g. `train_ppo` or `train_a2c`) for each seed, and evaluates each trained model greedily with `evaluate_policy(..., seed=10_000 + s)`; `final` is an array of the seeds' mean evaluation returns.
2. the experiments in the notebook: PPO on CartPole; PPO and A2C on the acrobot; one ablation of your choice, with the hypothesis written in the write-up before you run it.
3. the write-up at the end of the notebook, answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`, including the slow metric test) |
| `Acrobot` | matches an independent implementation of the published equations to $10^{-9}$ on single steps; termination, reward and truncation as specified |
| evaluation and statistics | exact on hand-built cases; bootstrap bands of the right width; `compare_seeds` resamples the groups independently |
| `run_seeds` with your PPO | greedy success above 0.9 on a one-step task; median greedy return ≥ 150 on CartPole after 20,000 steps (3 seeds) |
| the write-up | every number with its interval and number of seeds; the ablation's hypothesis stated before its result |

for reference, the reference solution's full config (seeds 0–9) measures:

| experiment | greedy return, mean over 10 seeds [95% bootstrap CI] | worst / best seed |
|---|---|---|
| PPO on CartPole (100k steps, lr $10^{-3}$) | 500.0 [500.0, 500.0] | 500 / 500 |
| PPO on the acrobot (150k steps, lr $10^{-3}$) | −124.6 [−208.9, −80.9] | −500 / −77 |
| A2C on the acrobot (150k steps) | −107.5 [−151.0, −83.1] | −297 / −78 |
| PPO without advantage normalization, acrobot | −83.6 [−85.9, −81.3] | −90 / −78 |

PPO − A2C on the acrobot: −17.1, 95% CI [−120.0, 50.9], permutation p = 0.93: no detectable difference at this budget. the PPO number is dragged down by one seed (seed 8) that learned poorly (training return about −230; sampling from its policy scores −223) and whose greedy policy never swings up at all (−500); the other nine seeds average about −83. the example ablation (no advantage normalization) came out *better* than the default, with a much tighter interval, but the difference (−41.0, CI [−125.7, 3.4], p = 0.64) is not significant either. explaining (and testing) results like these is the heart of the write-up.

## stretch goals

- **AlphaZero-lite for full-size Connect Four.** extend topic 06's `train_alphazero` with Dirichlet noise at the root, left–right symmetry augmentation, a small convolutional `PolicyValueNet`, and an arena that only promotes a new network if it beats the old one; measure strength against MCTS with 100, 400 and 1,600 rollout simulations, with Wilson intervals over ≥ 100 games per point.
- **the greedy-evaluation trap.** investigate the acrobot seed whose greedy policy scores −500 while sampling from the same policy scores about −220: what does the deterministic policy get stuck doing? does a small ε help? which deployment protocol would you report, and why?
- **a continuous acrobot or pendulum with PPO.** use `ActorCritic(continuous=True)` and compare against your SAC from topic 05 at equal environment steps, with intervals.
- **statistical power.** from your 10 PPO and 10 A2C final scores, estimate how many seeds you'd need to detect a difference of 10 return points with 80% power (simulate by resampling), and compare with Colas et al. (2018).
- **aggregate metrics.** replace the mean with the interquartile mean and report stratified bootstrap intervals, as recommended by Agarwal et al. (2021).

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), "Deep reinforcement learning that matters", *AAAI*. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), "Deep reinforcement learning at the edge of the statistical precipice", *NeurIPS*. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
- Colas, Sigaud & Oudeyer (2018), "How many random seeds? Statistical power analysis in deep reinforcement learning experiments". [arXiv:1806.08295](https://arxiv.org/abs/1806.08295)
- Sutton (1996), "Generalization in reinforcement learning: successful examples using sparse coarse coding", *NeurIPS* 8. the acrobot task as used in RL.
- Sutton & Barto (1998), *Reinforcement Learning: An Introduction*, 1st ed., MIT Press, §11.3. the "book" acrobot dynamics.
- Schulman, Wolski, Dhariwal, Radford & Klimov (2017), "Proximal policy optimization algorithms". [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Towers et al. (2024), "Gymnasium: a standard interface for reinforcement learning environments". `Acrobot-v1`, the version reproduced here. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
