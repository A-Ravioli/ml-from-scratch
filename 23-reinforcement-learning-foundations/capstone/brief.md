# 23 · capstone: design your own environment

## goal

every topic in this chapter handed you an environment. real RL work starts one step earlier: someone has to turn a problem into an MDP, with states that are Markov, rewards that say what you actually want, and an implementation that matches its own model. most RL bugs live there, not in the learning algorithm, and they're silent: an agent will happily learn whatever your buggy environment rewards.

in this capstone you build a small game twice. first a specified one, **Catch**, together with the tools that make environment design safe: a checker that tests any environment against the protocol and its own model, an evaluation harness with error bars, and features for an approximate learner. then a game **of your own design**, which you check, solve exactly, and attack with tabular and approximate learners from this chapter. this pulls in every topic: the bandit-style exploration of topic 01 (ε-greedy and optimism), the `Env` protocol and model arrays of topic 02, value iteration as ground truth (topic 03), Monte Carlo and TD control (topics 04–05), and features with linear SARSA (topic 06).

## the game you're given: Catch

a ball starts in the top row of a `rows × cols` grid, in a uniformly random column; the paddle starts in the middle of the bottom row (`cols // 2`). each step the paddle moves left, stays or moves right (actions 0, 1, 2; it stops at the walls), then the ball falls one row, and a gust of **wind** pushes it one column left or right with probability `wind / 2` each (it stops at the walls too). when the ball reaches the bottom row the episode ends: $+1$ if the paddle is under it, $-1$ otherwise. every other step pays 0. with the defaults (6 rows, 5 columns, wind 0.2) the optimal expected return is **0.680**; never moving gets about $-0.6$.

it's tabular: obs $= (\text{ball\_row}\cdot\text{cols} + \text{ball\_col})\cdot\text{cols} + \text{paddle}$, so there are $\text{rows}\cdot\text{cols}^2$ states, and `P`, `R` and `terminal` follow topic 02's conventions (terminal states, the ones with the ball on the bottom row, absorb with zero reward; `R[s, a]` is the expected reward, so the landing step's `R` averages over the wind).

## data

none: everything is simulated. tests use `Catch` with its default and small sizes and fixed seeds.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 3 seeds; checkpoints 25, 100, 400 episodes; 200 evaluation episodes | a few seconds |
| **full** (your write-up) | default | 10 seeds; checkpoints 25 to 1,600 episodes; 2,000 evaluation episodes | about 20 seconds with the reference `mlfs.rl` |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required code implemented and its checks green:
   - `Catch(rows=6, cols=5, *, wind=0.2)`: the game as an `Env` with `encode`/`decode`, `n_states`, `n_actions`, spaces, `terminal`, `P` and `R`; `reset` draws the ball's column with `self.np_random.integers(cols)`; `step` draws the wind with one `self.np_random.random()` (left if below `wind / 2`, right if below `wind`).
   - `check_env(env, *, n_steps=20000, seed=0)` → a list of problems (strings; `[]` means it passed). it must catch at least: a malformed `reset` or `step` result, observations outside the observation space, rewards that aren't Python floats, flags that aren't Python bools, `P` rows that aren't distributions, terminal states that don't absorb, an environment that isn't reproducible under `reset(seed=...)` (e.g. one that uses global `np.random`), and an environment whose sampled transitions or rewards disagree with its own `P` and `R` (test pairs with at least 30 samples, allowing about 5 standard errors).
   - `evaluate_policy(env, policy, n_episodes, *, seed=0)` → `(mean return, standard error)` as floats, seeding the first episode with `seed`; and `q_policy(Q)`, the greedy policy of a table as a callable.
   - `CatchFeatures(env)`: a featurizer for `linear_sarsa` with two active features per state, one for (ball row, ball column − paddle) and one for (ball column − paddle) alone, and `n_features = rows·(2 cols − 1) + (2 cols − 1)`.
2. **your own game**, as `MyGame` in the notebook: an `Env` subclass that passes your `check_env`. it must be small enough to solve exactly if it's tabular (keep `P` under a few million entries), or come with another way to know how good a policy is.
3. the experiments: the exact optimum of both games, and learning curves (greedy return against training episodes, with uncertainty over seeds) for at least one tabular and one approximate learner on each.
4. the write-up at the end of the notebook, answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `Catch` | its model gives the optimum 0.680 under value iteration; episodes last `rows − 1` steps |
| `check_env` | no false alarms on `Catch` (and a small, windier variant); catches each of six broken environments |
| `evaluate_policy` | exact on deterministic cases; the standard error is `std(ddof=1) / sqrt(n)` |
| `CatchFeatures` | linear SARSA with them ($\alpha = 0.2$, ε = 0.1) reaches a median greedy return above 0.45 over 5 seeds within 150 episodes |
| your game | passes your `check_env`; every number in the write-up is reproducible from the notebook |

for reference, with the full config the reference solution's learners reach, after 1,600 episodes, greedy returns of about 0.62–0.64 (Q-learning, SARSA, MC control and linear SARSA all within noise of each other, against the optimum 0.68); after 100 episodes, linear SARSA with `CatchFeatures` is at about 0.54 while tabular Q-learning is still near 0.1.

## designing your game: ideas and rules

- **ideas.** a variant of Catch (bombs to dodge, two balls, a paddle with momentum); a tiny dice game (Pig to a small target, "hold or roll"); a gridworld with keys and doors (the key in the state); a one-pile Nim against a fixed random opponent; an inventory problem (order stock, pay for shortages and storage); a small maze with a moving guard.
- **keep it Markov.** whatever the dynamics and rewards depend on must be in the observation. if your game has a hidden variable (whose turn, what's been collected), put it in the state.
- **say what you want with rewards, and nothing else.** remember topic 02's lab: a positive reward per step in an episodic task teaches the agent to never finish. test your reward design on the optimal policy: does it do what you'd want?
- **terminate or truncate deliberately.** a game over is termination; a time limit is truncation.
- **seed everything** through `self.np_random`.
- **size.** for value iteration, keep $S^2A$ below a few million.

## stretch goals

- **reward shaping.** add a potential-based shaping reward $F = \gamma\Phi(s') - \Phi(s)$ to your game (Ng, Harada & Russell, 1999) and show that the optimal policy doesn't change while learning speeds up; then add a non-potential shaping reward and find a policy it breaks.
- **exploration.** compare ε-greedy with optimistic initialization (initial $Q$ above the largest possible return) on your game.
- **features that fail.** design a featurization of your game that aliases two states with different optimal actions, and show where the approximate learner's policy goes wrong.
- **a harder game.** make your game too large for value iteration and use the approximate learner with a proper evaluation (enough episodes for a tight error bar) as the yardstick.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, chapters 3–10. [incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html)
- Osband et al. (2020), "Behaviour suite for reinforcement learning", *ICLR*. small, diagnostic environments (Catch is one of them) designed to isolate one difficulty at a time. [arXiv:1908.03568](https://arxiv.org/abs/1908.03568)
- Ng, Harada & Russell (1999), "Policy invariance under reward transformations: theory and application to reward shaping", *ICML*.
- Towers et al. (2024), "Gymnasium: a standard interface for reinforcement learning environments". the protocol, and its environment checker. [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)
- Henderson et al. (2018), "Deep reinforcement learning that matters", *AAAI*. why seeds, error bars and baselines matter in RL evaluation. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
