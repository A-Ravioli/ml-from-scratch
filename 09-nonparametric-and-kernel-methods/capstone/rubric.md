# 09 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end | all pass, plus your own checks (e.g. your EI against a Monte Carlo estimate of $\mathbb E\max(f - f_\text{best}, 0)$, your objective against a hand-written CV loop, `bayes_opt` on a function whose maximum you know in 3-D) |
| **results** | no numbers | the best CV RMSE of one run | median and spread over seeds for both searches at several budgets, the grid optimum as a reference, and the test RMSE of the tuned model with a CI against the defaults and the mean baseline | the same, plus the number of evaluations each method needs to get within a stated tolerance of the optimum, and a paired comparison over seeds (BO and random search share seeds) |
| **analysis** | none | describes the plots | explains *why* BO wins (or doesn't) with the chapter's ideas: the GP posterior, EI's balance of mean and uncertainty, where the evaluations went, the shape of the landscape | also tests an explanation: a smaller box, a different kernel or $\xi$, more initial points, or a noisy objective, and reports what changed |
| **clarity** | hard to follow | readable | a short report with one figure and one table; all five prompts answered | a reader could reproduce every number (config, seeds, budget), and the conclusion is stated with its uncertainty ("BO reaches within 0.5% of the grid optimum in a median of 12 evaluations; random search needs more than 30") |

## what graders look for

- **a fair comparison.** both searches get the same budget, the same box, the same objective (the same folds) and several seeds; the comparison is reported as a distribution, not one lucky run.
- **a reference point.** "BO found 0.598" means little without "the best of a 256-point grid is 0.5976" and "the defaults give 0.733 on the test set".
- **honest optimism.** the best CV score of a search is the maximum of noisy estimates; the locked test set, used once, says how optimistic it was.
- **cost in the right units.** evaluations of the expensive objective, not wall-clock time of the GP: the GP's own cost is negligible here and the whole point is to save objective evaluations.
- **limits stated.** what the GP assumed (a stationary, smooth function on the log-scaled box; nearly noise-free evaluations) and when that breaks (categorical hyperparameters, conditional ones, very noisy CV scores, many dimensions).
