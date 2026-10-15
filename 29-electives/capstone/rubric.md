# 29 · capstone rubric

score each criterion 0–3 for the portfolio as a whole. a solid portfolio scores at least 9 out of 12, with no zeros. the weakest entry sets the score: two strong entries and one careless one is a 2, not a 3.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | tests fail | most tests pass | all tests pass, and every entry is built with `portfolio_entry` from per-seed scores (no hand-typed means) | all pass, plus your own checks: e.g. the sign-flip test's size at *your* seed count simulated, or one entry's interval recomputed by hand from the rendered numbers |
| **results vs baselines** | no baseline, or numbers without uncertainty | a baseline per elective, but means only, or one seed | one entry per completed elective (≥ 2), each with paired seeds, an interval, the exact p and its floor, and the budget's unit stated where there is a budget | the same, plus a second, stronger baseline in at least one elective (with Holm across that entry's comparisons), and a budget table in both units where the unit could change the winner |
| **analysis** | none | restates the tables | answers all five prompts per elective: why this baseline, what the budget leaves out, what the verdict rests on (interval or p), how the simulator or semi-synthetic world favours one side, what would flip the result | also tests one of those claims with an extra run: more seeds, the other unit, the harder variant of the world (causal: real covariate dependence; continual: bytes; your block's equivalent), and reports whether the verdict survived |
| **clarity** | hard to follow | readable | each entry readable on its own: question, method, baseline, metric and direction, seeds, budget, verdict; one figure per project | a reader who skipped the electives could tell which claims are solid, which are inconclusive and why, and could rerun every number from the seeds and configs given |

## what graders look for

- **the baseline a skeptic would choose.** the strongest simple method, tuned as carefully as yours: seasonal-naive for forecasting, fine-tuning *and* a small replay buffer for continual learning, an RBF-kernel SVM for quantum kernels, popularity and ALS for recommenders. a weak baseline makes every verdict meaningless.
- **paired seeds, really paired.** same seeds, data orders, splits or draws for method and baseline, and the pairing done by seed, not by list position.
- **the verdict from the interval.** "better" when the interval excludes zero on the good side; "inconclusive" said plainly otherwise, with the effect size the experiment could have detected. with five seeds a two-sided sign-flip p can't go below 0.0625; a report that calls this "not significant" without saying so has misread its own test.
- **budgets with units.** memory in floats *or* bytes, compute in steps, FLOPs *or* seconds, and a sentence on what the budget doesn't count (replay's doubled compute, a generator's training time, communication rounds).
- **the thumb on the scale.** every simulator and semi-synthetic world is built by someone who knows the method. the best reports name how theirs favours the method (faithfulness by construction, classes that are blobs around their means, a sequence task made for long memory) and say where the ranking would flip.
- **honest failures.** an elective where the method lost, or where the evidence is inconclusive, reported as such, is worth more than a third "better".
