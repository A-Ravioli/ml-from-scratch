# 07 · capstone: an end-to-end tabular project

## goal

predict the median house value of a California district from eight numbers, and report how well you do in a way that would survive a skeptical reviewer. the modelling is deliberately simple (a $k$-nearest-neighbour regressor and least squares, both of which you've met). the point of this capstone is the *evaluation*: every number comes with a baseline and an uncertainty estimate, every learned step lives inside the cross-validation folds, every choice is evaluated on data it didn't see, and the test set is opened exactly once.

this pulls the whole chapter together: the estimator protocol and dummy baselines (topic 01), a locked test split and $k$-fold CV (topic 02), RMSE and $R^2$ through scorers (topic 03), a validation curve over $k$ (topic 04), a scaler inside a pipeline (topic 05), and a grid search with nested CV (topic 06), plus chapter 05's bootstrap and permutation test.

## data

- **California housing** via `mlfs.datasets.california_housing()` (1990 census, 20,433 districts after dropping rows with missing values; 8 numeric features: longitude, latitude, median age, total rooms, total bedrooms, population, households, median income; target: median house value in units of \$100,000, capped at 5.0), downloaded once and cached under `~/.cache/mlfs/`.
- with `MLFS_OFFLINE=1` (tests, `verify.py`), a 1,000-row synthetic stand-in with the same shape. its target is a *linear* function of the features plus noise, so least squares wins there, and the conclusions of the real data don't carry over: it's a smoke test.
- **tests** use only small synthetic datasets generated in `test_capstone.py`.

## configurations

| config | where | sizes | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 1,000 synthetic rows (800 train / 200 test); $k \in \{1, 3, 5, 10, 20, 40\}$; 1,000 bootstrap and permutation draws | a few seconds |
| **full** (your write-up) | default | 20,433 rows (16,346 train / 4,087 test); $k$ up to 80; 2,000 draws | 2–3 minutes, most of it kNN in the nested CV |

the notebook picks the config from `datasets.is_offline()`.

## deliverables

1. `capstone.py` with the required functions implemented and their checks green:
   - `KNNRegressor(k=5, *, chunk_size=512)`: a `Regressor` predicting the mean target of the $k$ nearest training rows, with distances computed `chunk_size` query rows at a time (your `pairwise_distances` from chapter 00), so memory stays $O(\text{chunk} \times n_\text{train})$
   - `LeastSquaresRegressor(fit_intercept=True)`: ordinary least squares with `coef_` and `intercept_`
   - `cross_val_predict(estimator, X, y, *, cv=5)`: out-of-fold predictions, one per row, each from a clone that never saw that row
   - `rmse_ci(y_true, y_pred, *, n_boot, alpha, rng)` and `paired_comparison(y_true, pred_a, pred_b, *, n_boot, n_permutations, alpha, rng)`: an RMSE with a bootstrap CI, and a paired bootstrap CI plus permutation test for the difference in MSE between two models on the same rows
   - `tuned_knn(ks, *, cv, chunk_size)`: an unfitted `GridSearchCV` over `knnregressor__k` for `make_pipeline(StandardScaler(), KNNRegressor())`, scored by negative MSE
2. the write-up section at the end of the notebook: an evaluation report answering its five prompts (the result, what mattered, the cost of choosing, leakage, limits). experiments A–E are given and run on your functions.

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` pass | all (`pytest test_capstone.py`) |
| `KNNRegressor` | matches brute-force nearest neighbours for $k = 1, 4, 9$; peak memory under a quarter of the full distance matrix |
| `rmse_ci` | 95% intervals contain the true RMSE in at least 50 of 60 simulated datasets |
| `paired_comparison` | a clear improvement gives a CI excluding 0 and $p < 0.01$; a negligible one a CI containing 0 and $p > 0.05$ |
| `tuned_knn` | on held-out synthetic data with badly scaled features, $R^2 > 0.75$ |
| full-config report | every number in your write-up is reproducible from the notebook |

for reference, in the full config the solution gets a cross-validated RMSE of **1.155** for the mean baseline, **0.701** for least squares, **0.971** for $k$-NN on raw features and **0.626** for $k$-NN on standardized features (in units of \$100,000); the search picks $k = 10$; nested CV estimates **0.630**; and the one look at the test set gives an RMSE of **0.601** (95% CI 0.578–0.625, $R^2 = 0.73$) against **1.150** for the baseline.

experiment D is meant to show you effects of very different sizes. scaling outside the folds changes nothing you can see (0.6263 both ways). the target-encoded location feature, fitted before the CV, makes least squares look like the best model of all (**0.483**), while the honest pipeline gets **0.586**: each training district's own price is part of its cell's mean, and with cells of a few districts that's the answer leaking in. peeking at the test set over $k = 1..80$ finds $k = 12$ at 0.5999 against 0.6005 for the CV choice: with 4,000 test rows and a smooth curve the optimism is real but tiny, and it would grow with more choices and fewer test rows.

one number in experiment B looks like a bug and isn't: least squares versus the degree-2 model has a permutation $p$ of 1.000 and a CI from about −0.08 to +0.08. the degree-2 model makes a handful of catastrophic out-of-fold errors (squared features extrapolated far outside the training range), and those few rows dominate both the bootstrap and the sign-flip test: almost every random sign assignment gives a mean difference at least as large as the observed one. a paired test can only be as stable as the per-row differences it averages; look at their distribution before trusting either number.

## stretch goals

- **spatial leakage.** districts close to each other have similar prices, and $k$-NN on longitude and latitude exploits that. cluster the districts geographically (or bin them into grid cells) and use `GroupKFold` with the clusters as groups. how much does the CV RMSE change, and which question does each number answer?
- **better features.** rooms, bedrooms and population are totals per district; ratios (rooms per household, bedrooms per room, people per household) are more meaningful. do they help least squares? $k$-NN? report the differences with `paired_comparison`.
- **the capped target.** about 5% of districts are capped at 5.0. how do your models do on them? what does the cap do to the RMSE and to $R^2$?
- **a fairer polynomial.** the degree-2 least-squares model has a few catastrophic out-of-fold errors (look at its wide CI). find the rows, explain them (extrapolation of squared features), and fix them (clipping, or scaling plus ridge from chapter 08 once you've built it).
- **repeated CV.** rerun experiment A with 5 different fold seeds. how much do the CV RMSEs move, compared with the width of your bootstrap CIs? (Bengio and Grandvalet, 2004, explain why the bootstrap over rows understates the CV's variance.)

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3): 291–297. the source of the California housing data. [doi:10.1016/S0167-7152(96)00140-X](https://doi.org/10.1016/S0167-7152(96)00140-X)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapter 7 (model assessment and selection). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Cawley & Talbot (2010), "On over-fitting in model selection and subsequent selection bias in performance evaluation", *JMLR* 11: 2079–2107. [jmlr.org/papers/v11/cawley10a](https://www.jmlr.org/papers/v11/cawley10a.html)
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall/CRC, chapters 13–15: bootstrap confidence intervals and permutation tests.
- Roberts et al. (2017), "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", *Ecography* 40(8): 913–929. for the spatial stretch goal. [doi:10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881)
