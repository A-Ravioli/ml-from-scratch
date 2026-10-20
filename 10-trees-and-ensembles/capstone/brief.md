# 10 · capstone: a tabular benchmark

## goal

race your own gradient booster against a production one. scikit-learn's `HistGradientBoostingRegressor` is built on the same ideas as your topic-04 booster (quantile bins, gradient and hessian histograms, shrinkage), written in Cython with years of tuning. your job: get within **3%** of its test RMSE on California housing, using only your NumPy code, and *prove* it with a confidence interval rather than a single lucky number.

a benchmark is an experiment, so the rules matter as much as the model: both contestants see the same training rows, every choice (features, number of trees, hyperparameters) is made on the same validation rows, the test rows are opened once, and the verdict accounts for the uncertainty of a finite test set. along the way you'll use most of the chapter: a feature that trees can't build themselves (topic 01's lesson on axis-aligned splits), early stopping and shrinkage (topic 04), and permutation importance (topic 05), plus chapter 05's bootstrap and chapter 07's protocol of train/validation/test.

## data

- **California housing** via `mlfs.datasets.california_housing()`: 20,433 districts from the 1990 census (rows with missing values dropped), 8 numeric features (longitude, latitude, median age, total rooms, total bedrooms, population, households, median income), target = median house value in units of \$100,000 (capped at 5.0). it's cached in `~/.cache/mlfs/`.
- the notebook splits it once with a fixed permutation (seed 0): **60% train** (12,260 rows), **20% validation** (4,087), **20% test** (4,086).
- with `MLFS_OFFLINE=1` (tests, `verify.py`) you get a 1,000-row synthetic stand-in with a linear target: a smoke test only.
- **tests** use small synthetic datasets generated in `test_capstone.py` (Friedman's benchmark #1 and simulated errors).

## the benchmark protocol

- **the benchmark:** `HistGradientBoostingRegressor` with its default hyperparameters (learning rate 0.1, leaves of up to 31, no L2 penalty, 256 bins), `max_iter=N_TREES`, `early_stopping=False`, `random_state=0`, fitted on the training rows; its number of iterations is chosen as the minimum of its validation RMSE over `staged_predict`, exactly like yours. it runs on **one CPU core** (`threadpoolctl.threadpool_limits(1)`), so the timings compare algorithms rather than core counts.
- **same features:** whatever features your final model uses (raw, or raw + ratios), the benchmark gets the same.
- **your model:** your `mlfs.trees.GradientBoostingRegressor`, any hyperparameters, chosen on the validation rows only.
- **the verdict:** the relative gap $\text{RMSE}_\text{yours}/\text{RMSE}_\text{benchmark} - 1$ on the test rows, with a 95% **paired** bootstrap CI over test rows. **within** the margin if the CI's upper end is at most +3%; **outside** if its lower end is above +3%; otherwise **inconclusive**.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 1,000 synthetic rows; up to 150 trees; 3 search candidates; 500 bootstrap draws | about 5 seconds |
| **full** (your write-up) | default | 20,433 rows; up to 1,000 trees; 10 search candidates; 2,000 bootstrap draws | about 4–5 minutes, three quarters of it the random search |

the notebook picks the config from `datasets.is_offline()`. the full-config time was measured on a busy 8-core laptop with every fit on one core: each candidate of the search takes 10–45 s depending on depth and number of trees, the benchmark 2–4 s.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `add_ratio_features(X)`: $(n, 8) \to (n, 11)$ with rooms per household, bedrooms per room and people per household appended (0 where a denominator is 0)
   - `staged_rmse(model, X, y)`, `best_iteration(curve)` and `predict_with_n_trees(model, X, n_trees)`: early stopping on a validation set without refitting
   - `random_search(make_model, space, X_tr, y_tr, X_val, y_val, *, n_iter, rng)`: candidates drawn uniformly per key, each fitted on train and early-stopped on validation; results sorted best first, each with its params, fitted model, number of trees, validation RMSE and time
   - `rmse_ci(y_true, y_pred, *, n_boot, alpha, rng)`, `relative_gap_ci(y_true, y_pred, y_ref, *, n_boot, alpha, rng)` (paired) and `verdict(gap_ci, margin)`
2. experiments B–F in the notebook (A, the benchmark and a default booster, is given): features, random search, the final models, one look at the test set, and permutation importance.
3. the write-up section at the end, answering its five prompts.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `rmse_ci` | 95% intervals contain the true RMSE in at least 50 of 60 simulations |
| `relative_gap_ci` | on almost perfectly paired errors 25% apart, a CI narrower than 3 points around +25% |
| your pipeline on Friedman #1 | random search + early stopping within 5% of the benchmark's test RMSE |
| the full benchmark | a verdict of **within** at the 3% margin, with every number in your write-up reproducible from the notebook |

for reference, the solution's full config gives:

| model | validation RMSE | test RMSE (95% CI) |
|---|---|---|
| predict the training mean | 1.169 | 1.150 (1.123–1.176) |
| benchmark, raw features (435 iterations) | 0.4749 | |
| yours, default (lr 0.1, depth 6), raw features (691 trees) | 0.4794 | |
| yours, default, raw + ratios (375 trees) | 0.4601 | |
| benchmark, raw + ratios (385 iterations) | 0.4568 | **0.4394** (0.4205–0.4591) |
| yours, tuned, raw + ratios (877 trees) | 0.4547 | **0.4419** (0.4218–0.4618) |

the best candidate of the search was learning rate 0.05, depth 7, `min_child_weight` 5, $\lambda = 0$, `subsample` 0.7. the relative gap on the test set is **+0.55%**, 95% CI **[−0.78%, +1.98%]**: **within** the 3% margin. in dollars, the typical error is about \$44,000 per district, against \$115,000 for predicting the mean. the ratio features were worth about 4% of validation RMSE to your default booster (0.4794 → 0.4601), more than all the tuning that followed (0.4601 → 0.4547). permutation importance puts latitude and longitude far ahead (test $R^2$ drops of about 1.5 and 1.2 when shuffled), then median income (0.3) and people per household.

two things about these numbers. first, the two models' test RMSEs have CIs about 0.04 wide, but the CI of their *gap* is only about 3 points: both models make their big errors on the same districts, and the paired bootstrap knows it. comparing the two separate CIs would make a 3% margin impossible to establish with 4,000 test rows. second, your tuned model beat the benchmark on the validation set (0.4547 against 0.4568) but not on the test set: the validation number is the minimum over 10 searched candidates *and* over 1,000 tree counts, and is optimistic by construction (chapter 07's winner's curse). that's why the verdict uses the test set.

## stretch goals

- **a fairer fight.** give the benchmark the same search budget (10 candidates over its own `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`). does the verdict survive?
- **leaf-wise trees.** LightGBM and `HistGradientBoostingRegressor` grow the leaf with the best gain first, up to a number of leaves, instead of level by level to a depth. add a `max_leaves` option to a copy of your `HistGradientTree` (a priority queue of candidate splits) and compare.
- **speed.** profile one fit of your booster. how much time goes into `gradient_histograms`, the split scan, `predict` on the training rows, and Python overhead? try updating $F$ from the leaf assignment of the training rows instead of calling `predict` (with `subsample < 1` you still need it for the rows left out).
- **the capped target.** about 5% of districts have the capped value 5.0. how do both models do on them? would a model trained on log-prices do better on the uncapped ones?
- **spatial honesty.** neighbouring districts share prices. repeat the comparison with a split by geographic cluster (chapter 07's `GroupKFold` idea). how much do the RMSEs change, and does the verdict?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3): 291–297. the source of the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
- Chen & Guestrin (2016), "XGBoost: a scalable tree boosting system", *KDD*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Ke et al. (2017), "LightGBM: a highly efficient gradient boosting decision tree", *Advances in Neural Information Processing Systems* 30. the design `HistGradientBoostingRegressor` follows.
- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. [jmlr.org/papers/v13/bergstra12a](https://www.jmlr.org/papers/v13/bergstra12a.html)
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall/CRC: paired comparisons and bootstrap confidence intervals.
- Grinsztajn, Oyallon & Varoquaux (2022), "Why do tree-based models still outperform deep learning on typical tabular data?", *NeurIPS Datasets and Benchmarks*. a careful benchmark protocol for tabular data, California housing included. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815)
