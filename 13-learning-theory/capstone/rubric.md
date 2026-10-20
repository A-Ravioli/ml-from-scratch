# 13 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and the full config runs end to end on the real data | all pass, plus your own checks: e.g. a coverage simulation showing your margin or PAC-Bayes bound holds on synthetic data at the promised rate, your Gibbs risk against Monte Carlo over weights, your RKHS norm against an explicit feature map for a polynomial kernel |
| **results** | no numbers | test errors only | for all six (task, model) pairs: train and test errors, a CI on the test error, the test-set bound and every applicable uniform bound, with the bounds clearly marked vacuous or not | the same, plus how the bounds and the measured gap change with $n$ (a small learning curve for at least one model), or a second choice of prior / grid showing how sensitive the PAC-Bayes number is |
| **analysis** | none | restates the table | explains *why* each bound is tight or loose using the chapter's terms (union bounds, Sauer, worst-case constants, KL to the prior, normalized margins, random-label fit), and why the models without bounds still generalize | also tests an explanation: e.g. shows that a data-dependent prior shrinks the KL, that standardizing or projecting the features changes the VC-vs-PAC-Bayes ranking, or that the SVM's margin bound improves with a smaller $C$ at a measured cost in test error |
| **clarity** | hard to follow | readable | a short report with one table and one figure; all five prompts answered with numbers and uncertainties | a reader could reproduce every number (sizes, seeds, $\delta$, grids, hyperparameters), every bound states its version and assumptions, and the conclusion is stated precisely (for example: "with 60,000 training digits, PAC-Bayes bounds the error of the Gibbs version of the logistic regression by 0.232, 1.6× its measured test error of 0.149; for the deterministic classifier the tightest training-data bound is 0.463, 4.6× its measured 0.100 (95% CI 0.094–0.106)") |

## what graders look for

- **measured, not estimated.** every empirical number comes from held-out data, with a confidence interval or an exact binomial bound; nothing is "about" or "should be".
- **the version of every bound.** which constants, which loss, which grid, which $\delta$, and whether a union bound was paid for choosing $\gamma$ or $\sigma$ after seeing the data.
- **vacuous stated plainly.** a bound above 0.5 on a balanced binary task says nothing; say so, and say which term made it so.
- **no certified claims from proxies.** the random-label "proxy bound" is evidence, not a guarantee; it must be labelled as such.
- **honest limits.** hyperparameters chosen by looking at test results, priors that saw the data, and non-iid data all void the guarantees; the write-up should say which of these the experiment respects and which it doesn't.
