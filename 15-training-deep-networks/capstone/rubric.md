# 15 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | fewer than half the bugs fixed | most checks pass | all checks pass, and the full config reaches $\ge$ 0.975 test accuracy on real MNIST | all of that, and every fix is minimal and matches the docstring's specification (no checks satisfied by special-casing, no unrelated rewrites) |
| **results** | no test number | a test accuracy without uncertainty or baseline | the fixed pipeline's test accuracy with a 95% interval and the majority-class baseline, and all 10 per-bug measurements from full-config runs | all of that plus the seed spread of the fixed pipeline (≥ 3 seeds), and a statement of which per-bug effects are distinguishable from seed noise and the test set's sampling error |
| **analysis** | bugs listed without explanation | each bug's cause explained | each bug's symptom, cause, fix and measured effect, with the mechanism behind the effect (why that number, not just that number) | all of that plus at least one interaction between bugs or with the configuration, and a ranking of the bugs by cost versus by visibility, with what you'd change in your own training code |
| **clarity** | hard to follow | readable | one consistent entry per bug and labelled figures (curves that show the symptoms) | a reader could re-introduce any bug and reproduce your number from the write-up alone (config, seeds, runtime, machine) |

## what graders look for

- **symptom before cause.** the best write-ups say what was *observed* first (a `nan` in the first training loss, a learning rate of $3 \times 10^{-34}$ at epoch 2, validation accuracy that changes when you evaluate twice), and only then the line that caused it. that's the skill being practised.
- **honest null results.** some bugs have no measurable effect in this configuration. "re-introducing it changes the test accuracy by 0.0008, inside the interval" is a correct and valuable result; explain *why* it's harmless here and when it wouldn't be.
- **measurements, not impressions.** every effect is a full-config number next to the fixed pipeline's, with the test set used only for the final report (choosing which bug "matters most" by test accuracy across ten runs is fine as a measurement, but not for tuning anything).
- **mechanisms.** "accuracy dropped" is a symptom; "the learning rate was divided by 2 every 4 *batches*, so after one epoch it was $10^{-34}$ and the weights froze where they were after about 40 steps" is an explanation.
- **prevention.** for each bug, a sentence on the check, assertion or logged statistic that would have made it loud. the strongest write-ups notice which bugs no generic diagnostic could catch, and what kind of test catches those.
