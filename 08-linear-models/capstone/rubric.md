# 08 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end | all pass, plus your own checks (e.g. your vectorizer against a hand-counted message from the real data, your pipeline's CV predictions against a hand-written loop, `metric_with_ci` coverage for F1 on a simulation) |
| **results** | no numbers | test accuracy only | accuracy, precision, recall, F1 and AUC on the test set, each with a CI, next to the majority baseline, plus the CV choice of $\alpha$ | the same, plus a paired comparison for every claim that one variant beats another, and the recall at a fixed, low false-positive rate (the number a spam filter's user cares about) |
| **analysis** | none | describes the tables | explains the results with the chapter's ideas: why the penalty is needed (separability), what TF-IDF changes, which ablations are within the fold-to-fold noise, what the top features and the worst mistakes show | also identifies what the evaluation *doesn't* capture (spam that changes over time, dataset artifacts in the top features, the small number of test spam messages) and tests at least one of them |
| **clarity** | hard to follow | readable | a short report with one results table and the error analysis; all five prompts answered | a reader could reproduce every number (config, seeds, folds), and the conclusion is stated with its uncertainty ("catches 89% of spam [84%, 93%] while flagging 0.4% of real messages") |

## what graders look for

- **the right metric for the question.** with 13% spam, a model that never flags anything is 87% accurate. precision and recall on the spam class (or the recall at a fixed false-positive rate) carry the information; accuracy alone doesn't.
- **choices made on the training data only.** $\alpha$ and every ablation are decided by cross-validation; the test set is used once. if the chosen $\alpha$ is at the edge of the grid, the grid was extended.
- **uncertainty that matches the claim.** "bigrams help" needs a difference larger than the fold-to-fold spread, or a paired test, not two overlapping means.
- **reading the model.** the top features are inspected and discussed critically: which reflect spam in general, and which reflect how this particular dataset was collected (UK premium-rate numbers, `£`, `txt`).
- **honest limits.** the CIs cover resampling of test messages, not a new collection year, a new country, or spammers adapting to the filter.
