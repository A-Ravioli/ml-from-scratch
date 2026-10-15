# 23 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and your own game passes your `check_env` | all pass, plus your own checks: e.g. `evaluate_policy` of value iteration's policy agreeing with $V^*$ within its error bar on *your* game, and a deliberately broken copy of your game that `check_env` catches |
| **results** | no numbers | final returns only | learning curves for a tabular and an approximate learner on both games, with mean ± standard error over seeds and the exact optimum as a reference line | the same, plus a baseline (random or "do nothing") and a sensitivity check (step size or ε) for at least one learner |
| **analysis** | none | describes the curves | explains the rankings with the chapter's ideas: bootstrapping versus Monte Carlo variance, on- versus off-policy, what the features generalize across | also explains where a learner falls short of the optimum and *why* (exploration that stops too early, aliased states, a step size too large to settle), with an experiment that tests the explanation |
| **clarity** | hard to follow | readable | the game is specified precisely enough to reimplement; labelled figures; all five prompts answered | a reader could reproduce every number (sizes, seeds, step sizes, episodes) and would agree that the game is Markov and the rewards say what you meant |

## what graders look for

- **a precise game.** states, actions, dynamics, rewards and episode end, written so that someone else could implement the same MDP, and an argument that the state is Markov.
- **ground truth first.** an exact optimum (or a convincing upper bound) before any learning curve. a learning curve without a reference line can't say whether 0.6 is good.
- **error bars that mean something.** returns over seeds, the standard error, and enough evaluation episodes that the evaluation noise is smaller than the differences you claim.
- **honest failures.** a learner that never reaches the optimum, a feature set that confuses two states, a reward that taught the wrong behaviour: reported and explained.
- **design reasoning.** why this game, what makes it hard for each method, and what you'd change next.
