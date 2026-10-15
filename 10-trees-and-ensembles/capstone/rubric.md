# 10 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end with the benchmark protocol (same rows, same features, iterations chosen on validation for both) | all pass, plus your own checks: e.g. `relative_gap_ci` coverage on a simulation where you know the true gap, or your booster's first tree against a CART tree on the bin codes on the real data |
| **results** | no numbers | a test RMSE for your model only | the table of experiment E: baseline, benchmark and yours with CIs, the paired gap with its CI and the verdict, in dollars as well as in the target's units | the same, plus the validation-set history of every choice (features, search) and a check of how much the verdict moves with a different split seed or search seed |
| **analysis** | none | describes the tables | explains each result with the chapter's ideas: why ratios help trees, shrinkage and the number of trees, why the paired CI is so much narrower than the two separate ones, why the validation winner looks better than it is | also says what the benchmark *doesn't* show (one dataset, one split, the benchmark's untuned defaults, single-core timings) and tests at least one of them (a stretch goal) |
| **clarity** | hard to follow | readable | a short report with one results table and one figure; all five prompts answered | a reader could reproduce every number (config, seeds, search space), and the conclusion is stated with its uncertainty ("within 3%: gap +0.6%, 95% CI −0.8% to +2.0%") |

## what graders look for

- **one look at the test set.** every choice is made on the validation rows; the test rows are touched once, in experiment E. a model "improved" after seeing test numbers isn't a benchmark result any more.
- **the same rules for both contestants.** same training rows, same features, the number of iterations chosen the same way. any advantage you give your model (a search, extra features) is stated, and ideally given to the benchmark too.
- **uncertainty that matches the claim.** "within 3%" is a claim about a difference, so it needs a CI of the difference on the same rows, not two overlapping CIs.
- **costs next to scores.** runtime per fit for both models, and where your booster spends its time.
- **honest limits.** a good report says what the bootstrap over test rows leaves out: the randomness of the split and the search, the dependence between neighbouring districts, the capped target.
