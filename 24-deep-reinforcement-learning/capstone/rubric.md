# 24 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, including the slow metric threshold | all pass, plus your own checks: e.g. the acrobot's energy without torque staying near-constant over a short horizon, or your PPO reproducing the same returns from the same seed on a rerun |
| **results** | no numbers | final returns only, or one seed | learning curves with bootstrap bands and greedy evaluations with intervals over ≥ 10 seeds for PPO on CartPole and on the acrobot, and A2C on the acrobot | the same, plus a baseline that anchors the scale (random policy, or the hand-written controller of topic 07) and the number of environment steps each method needs to reach a stated return |
| **analysis** | none | describes the curves | explains the PPO–A2C difference with the chapter's ideas (sample reuse, the clip, the variance of the advantages) and states a hypothesis for the ablation *before* running it | also tests an explanation with a further experiment (e.g. A2C with PPO's larger batches, or PPO with one epoch), and reports a power estimate: how many seeds a difference of the observed size would need |
| **clarity** | hard to follow | readable | every figure labelled, every number with its uncertainty and number of seeds, all five prompts answered | a reader could reproduce every number from the write-up (seeds, budgets, hyperparameters, code version) and would know which differences are real and which are noise |

## what graders look for

- **seeds, not anecdotes.** one learning curve is mostly noise (Henderson et al., 2018). results over many seeds, with intervals, and claims no stronger than the intervals allow.
- **greedy evaluation, separately from training curves.** training returns include exploration; the deployed policy is the greedy one. report both, and say which is which.
- **a pre-registered ablation.** the hypothesis comes before the run. "no significant difference" with a wide interval is an honest result, not a failure; say what effect size the experiment could have detected.
- **the environment checked against its source.** the acrobot's equations come from a published description; the tests check one step, and a careful write-up checks more (an energy argument, a comparison with a long rollout).
- **honest failures.** a seed that never learned, a baseline that did better than expected, an ablation that changed nothing: reported and discussed.
