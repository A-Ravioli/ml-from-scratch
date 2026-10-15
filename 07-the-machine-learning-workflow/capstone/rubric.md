# 07 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end | all pass, plus your own checks (e.g. your `KNNRegressor` against a brute-force loop on real rows, `cross_val_predict` against a hand-written loop, `rmse_ci` coverage on a simulation where you know the truth) |
| **results** | no numbers | test RMSE only | every model's CV RMSE with a CI and a baseline, the tuned model's nested CV estimate, and one test-set number with a CI, in dollars as well as in the target's units | the same, plus paired comparisons for every claim that one model beats another, and a check of how much the conclusions move with a different split or fold seed |
| **analysis** | none | describes the tables | explains each result with the chapter's ideas: scaling and distances, the winner's curse (`best_score_` versus nested CV), the bias–variance trade-off in $k$, what leakage would and wouldn't change | also identifies what the evaluation *doesn't* capture (spatial dependence between districts, the capped target, the variance of CV itself) and tests at least one of them |
| **clarity** | hard to follow | readable | a short report with one results table and one figure; all five prompts answered | a reader could reproduce every number (config, seeds, folds), and the conclusion is stated with its uncertainty ("RMSE \$60k ± \$2k on held-out districts, 48% lower than predicting the mean") |

## what graders look for

- **a baseline next to every number.** "RMSE 0.60" means nothing without "the mean predictor gets 1.15 on the same rows".
- **uncertainty that matches the claim.** a difference between two models is supported by a paired CI or test on the same rows, not by two overlapping CIs eyeballed side by side.
- **choices made on the training data only.** the test set is used once, for the final model. any number that depended on looking at the test set (experiment D's "best $k$ on the test set") is labelled as optimistic.
- **leakage handled by structure.** preprocessing lives inside a pipeline, not in a cell that runs before the cross-validation.
- **honest limits.** a good report says what its confidence intervals leave out: the bootstrap over rows treats districts as independent, which neighbouring districts aren't.
