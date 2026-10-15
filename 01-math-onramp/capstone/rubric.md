# 01 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass; standardization uses training statistics only | all pass, plus your own extra checks (e.g. your `mse_grad` against finite differences on the real data, your closed form against `np.linalg.lstsq`) |
| **results** | no baseline | test MSE reported next to the "predict the mean" baseline | meets the ≤ 0.55 threshold on the real data, with a standard error, an RMSE in dollars and $R^2$ | also compares gradient descent with the closed form on the test set, and says whether any difference exceeds the uncertainty |
| **analysis** | none | describes the loss curve | explains the learning-rate limit and the no-standardization ablation with topic 06's curvature argument | adds a careful reading of the weights (and why correlated features make it risky) and a failure analysis using the residual plot (the \$500k cap, which districts are worst) |
| **clarity** | hard to follow | readable | clear figures with labeled axes, and a narrative that answers the five prompts | a classmate could reproduce every number from the write-up alone (split seed, learning rate, steps, config) |

## what graders look for

- **leakage.** standardizing with statistics of the whole dataset (train and test together) lets the test set influence training. it barely changes the numbers here, but it's the habit that matters: it scores at most 1 on correctness.
- **honest uncertainty.** with about 4,000 test districts, the standard error of the test MSE is around 0.017. a difference of 0.001 between gradient descent and the closed form is noise, and the write-up should say so rather than declare a winner.
- **the right comparison.** "MSE 0.49" means little on its own. "MSE 0.49 against 1.33 for always predicting the average, so the model explains about 63% of the variance" means a lot.
- **understanding the training.** the best write-ups explain why a learning rate of 0.3 diverges even on standardized features: strongly related features (rooms, bedrooms, population and households all measure district size) make the loss surface steeper in one combined direction than in any single feature direction.
- **humility about the model.** a straight line can't capture "near the coast is expensive" or the \$500k cap. naming those failures, with evidence from the residuals, is worth more than a slightly lower MSE.
