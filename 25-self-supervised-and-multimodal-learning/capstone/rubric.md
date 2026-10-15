# 25 · capstone rubric

score each criterion 0–3. a solid capstone scores at least 9 out of 12, with no zeros.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **correctness** | checks fail | most checks pass | all checks pass, including the learning thresholds | all pass, plus your own checks (e.g. a test that your augmentation never changes a label, or that a frozen encoder stays bit-identical) |
| **results** | no from-scratch baseline, or no full-config run | SimCLR and from-scratch numbers at all three label fractions, single runs | every cell of the table with its spread over label subsets and the number of runs; random-init baselines for the frozen protocols | as 2, plus at least one source of variation the default study doesn't cover (a second pretraining seed, or fine-tuning seeds at 100%) |
| **analysis** | numbers without interpretation | describes the curves | explains the crossover (or its absence) with reference to what each protocol measures, and says which differences exceed the noise | as 2, plus a pre-registered ablation (hypothesis written before the run) reported with uncertainty, and an honest account of what the budget (pretraining epochs, fixed step counts) might be hiding |
| **clarity** | hard to follow | readable | one clear figure and one table, every number with its interval and n | a peer could reproduce the study from the write-up alone: config, seeds, runtimes and hardware stated |

notes for grading yourself:

- "SimCLR helps" is not a result; "fine-tuning from SimCLR beats training from scratch by X ± Y points at 1% of the labels (3 label subsets), and by Z (1 run) at 100%" is.
- the spread over label subsets is not the whole uncertainty: with one pretraining run, you can't say how much a different pretraining seed would move the SimCLR rows. say so.
- equal *steps* per fraction is a choice. a from-scratch network might close the gap at 100% with more steps; if your conclusion depends on the budget, state it.
