# 05 · capstone: a model-comparison toolkit

## goal

"model B gets 85.1% and model A gets 82.0%, so B is better." every ML paper, leaderboard and pull request makes claims like that, and most of them skip the question this chapter is about: *would the ranking survive a different test set?* in this capstone you build the toolkit that answers it, and that the rest of the course uses for every evaluation (it is the evaluation half of `mlfs.stats` put to work):

- a **confidence interval for any metric**, by bootstrapping test examples (topic 03);
- a **paired** interval for the difference between two models scored on the *same* test examples (topic 03's lab showed why pairing matters);
- a **paired permutation test** and **McNemar's exact test** for "are these two models equally good?" (topic 04);
- a **many-models comparison** against a baseline, with Benjamini–Hochberg or Bonferroni control (topic 04).

then you use it on six classifiers for a real problem and write up what you can and can't conclude. estimation (topic 01), the logic of likelihoods and tests (topics 02 and 04), and the habit of asking how much information the data really carry (topics 05–06) all come into the write-up.

## data

the **California housing** data: 20,433 census districts from the 1990 US census (rows with missing values dropped), with 8 numeric features (location, house age, rooms, bedrooms, population, households, median income). the label is binary: **is the district's median house value above the training set's median?** load it with

```python
from mlfs import datasets
X, value, names = datasets.california_housing()   # X float64 (n, 8), value in $100k
```

the first call downloads the file once and caches it under `~/.cache/mlfs/`. with `MLFS_OFFLINE=1`, or without a network, you get a synthetic stand-in of 1,000 rows with the same columns, so the notebook always runs. the tests never touch either: they generate labels and the predictions of imaginary classifiers whose accuracies and error correlations are set exactly.

the notebook splits the rows at random, 70% train and 30% test, and standardizes the features with the training statistics. the six models are given, each a few lines of NumPy: majority class, a one-feature income rule, nearest centroid, 1-NN, 15-NN, and 15-NN on location (longitude, latitude) only. they're not the point; they're something to compare. you'll build better classifiers in chapters 08–10.

## configurations

| config | data | test examples | resampling | runtime (reference, laptop CPU) |
|---|---|---|---|---|
| **small** (tests, `verify.py`, offline) | synthetic stand-in, or the tests' own simulated classifiers | 300 | 2,000 bootstrap resamples, 2,000 permutations | about 4 s for the notebook |
| **full** (your write-up) | California housing | 6,130 | the same | about 8 s for the notebook (1 s for the six intervals, 1.6 s for `compare_models`) |

the notebook picks the config automatically from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `accuracy(y_true, y_pred)` and `metric_ci(metric, y_true, y_out, *, alpha=0.05, n_boot=2000, method='bca', rng=None)` → `(estimate, lo, hi)`
   - `paired_difference_ci(metric, y_true, out_a, out_b, *, ...)` → `(diff, lo, hi)`
   - `paired_permutation_test(metric, y_true, out_a, out_b, *, n_permutations=2000, rng=None)` → `(diff, p)` and `mcnemar_test(y_true, pred_a, pred_b)` → `(b, c, p)`
   - `compare_models(metric, y_true, outputs, baseline, *, alpha=0.05, correction='bh', ...)` → one row per model
2. the write-up section at the end of the notebook, answering its six prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| interval width for an accuracy on 2,000 examples | within 15% of the normal-theory width $2 \cdot 1.96\sqrt{p(1-p)/n}$ (`test_stage1_metric_ci_width_and_center`) |
| coverage of 95% intervals | at least 88% over 150 simulated test sets (`test_stage1_metric_ci_coverage`), and at least 85 of 100 for paired differences (`test_stage2_covers_the_truth`) |
| paired interval width with correlated errors | within 15% of the paired width, and under 60% of the unpaired one (`test_stage2_sign_and_pairing`) |
| size of the paired permutation test | at most 5% + 5 standard errors for two equally good models (`test_stage3_size_and_power`) |
| power | detects a 3-point improvement on 2,000 examples with correlated errors in at least 80% of simulations (the reference: 98.5%) |
| on California housing | report every model's accuracy with a 95% interval, and a `compare_models` table against the income rule with BH at 0.05 |

what the reference gets (95% BCa intervals; differences are paired, against the income rule):

| model | full: accuracy [95% CI] | full: difference [95% CI], BH-significant? | small (offline): accuracy |
|---|---|---|---|
| majority | 0.501 [0.488, 0.515] | −0.240 [−0.256, −0.222], yes | 0.520 |
| income rule | 0.741 [0.729, 0.751] | (baseline) | 0.830 |
| nearest centroid | 0.751 [0.739, 0.761] | +0.010 [−0.001, +0.020], **no** (p = 0.058) | 0.877 |
| 1-NN | 0.820 [0.810, 0.829] | +0.079 [+0.066, +0.092], yes | 0.770 |
| 15-NN | 0.851 [0.842, 0.859] | +0.110 [+0.098, +0.121], yes | 0.880 |
| 15-NN, location only | 0.884 [0.876, 0.892] | +0.143 [+0.131, +0.157], yes | 0.560 |

two things worth noticing before you write. on the full data, k-NN on **location alone** beats k-NN on all eight features. and nearest centroid's one-point gain over the income rule is *not* significant, even with 6,130 test examples, while 15-NN's three-point gain over 1-NN is: the paired interval is $[+0.023, +0.041]$, and McNemar counts 527 examples where only 15-NN is right against 337 where only 1-NN is. explaining both is part of the write-up. (the synthetic stand-in has no geography, so there location is useless: 0.560.)

## stretch goals

- **other metrics.** your functions take any `metric(y_true, y_out)`. add balanced accuracy or F1, and (once you have probabilistic classifiers in ch 08) AUC and log loss. which metrics need the tuple bootstrap rather than a mean of per-example scores?
- **training-set variability.** the intervals only resample the test set. refit 15-NN and 1-NN on 20 bootstrap resamples of the *training* set and see how much the accuracy difference moves (this is the variance Dietterich (1998) and Bouthillier et al. (2021) worry about).
- **spatial splits.** split the districts by geography instead of at random (say, by longitude bands) and rerun the comparison. what happens to "15-NN, location only", and why?
- **a sequential comparison.** evaluate on growing prefixes of the test set and stop as soon as $p \le 0.05$. simulate how often that procedure declares a difference between two *equally good* models (topic 04's optional stopping).

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Dietterich (1998), "Approximate statistical tests for comparing supervised classification learning algorithms", *Neural Computation* 10(7): 1895–1923. McNemar's test and the 5×2 cv test. [doi:10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)
- Dror, Baumer, Shlomov & Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", ACL 2018. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", MLSys 2021. how much of a benchmark difference is training randomness. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall/CRC, chapters 14–15: BCa intervals and permutation tests.
- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3): 291–297. the source of the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
