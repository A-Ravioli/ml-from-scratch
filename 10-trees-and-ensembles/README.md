# 10 · trees & ensembles

> build the models that still win on tabular data.

a decision tree asks one yes/no question about one feature at a time. on its own it's a mediocre model: grown deep it memorizes noise, and a 5% change in the data rearranges it. but trees are the raw material of the models that still win most competitions and run most production systems on tables of numbers: random forests, which average hundreds of deliberately different deep trees, and gradient boosting, which adds shallow trees one at a time, each correcting the last. in this chapter you build all of them, from the split search up, in vectorized NumPy that's fast enough to be useful: your CART trees match scikit-learn's split for split (except where two splits tie exactly), your forest's out-of-bag estimate tracks its test accuracy for free, your AdaBoost reproduces scikit-learn's estimator weights to rounding, and your histogram booster reproduces scikit-learn's `HistGradientBoosting*` to float32 rounding on exactly binnable data and ends up within a percent of scikit-learn's `HistGradientBoostingRegressor` on California housing. on the way you'll be surprised by a fully grown tree that's worse than a single split, by trees that get *worse* and a forest that gets *better*, by a test error that keeps falling after the training error hits zero while the log loss gets worse, by a booster that needs 128 bins on one dataset, and by the most important feature by impurity importance being pure noise.

**you will build:** impurity functions on additive statistics, a fully vectorized best-split search, CART classification and regression trees grown with an explicit stack, minimal cost-complexity pruning and impurity importances; bootstrap sampling, random forests for classification and regression with out-of-bag estimates; SAMME AdaBoost with staged scores and margins; quantile binning, gradient/hessian histograms, second-order split gains, a histogram tree with histogram subtraction, and XGBoost-style gradient boosting for regression and binary classification with shrinkage and row subsampling; permutation importance, ICE curves and partial dependence, exact Shapley values and KernelSHAP. modules: **`mlfs.trees`** (topics 01–04) and **`mlfs.inspection`** (topic 05).
**prerequisites:** chapter 07 (the estimator protocol and `Classifier`/`Regressor` bases, splitting, metrics and scorers, bias–variance), chapter 05 (entropy, the bootstrap), chapter 06 (gradient descent and Newton's method, for topic 04), chapter 02 (weighted least squares, for KernelSHAP), chapter 04 (variance of sums), and `as_rng` from chapter 00. chapter 08 isn't needed. everything is NumPy.
**time · scaffolding:** about 65 hours · L3 (independent): every function and method comes with its signature and a one-line docstring; the exact conventions (tie-breaking, node arrays, the order of random draws, SAMME's scaling) are in the lesson's spec boxes. hints have two levels. the capstone gives you the data loading and the benchmark run; you write the statistics, the search, the experiments and the report.
**newly unlocked library components:** after this chapter you may use scikit-learn's `tree` (`DecisionTreeClassifier`/`Regressor`), `ensemble` (`RandomForest*`, `AdaBoostClassifier`, `GradientBoosting*`, `HistGradientBoosting*`) and `inspection` (`permutation_importance`, `partial_dependence`), and gradient-boosting libraries built on the same algorithms (XGBoost, LightGBM), in labs and experiments, since you've built what they implement. inside `mlfs` modules, later chapters use *your* models through `mlfs.trees` and `mlfs.inspection`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md): later chapters may use your forests and boosters as tabular baselines and your inspection tools to explain models, and chapter 11's isolation forest reuses the random-tree ideas of topics 01–02.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [decision trees](01-decision-trees/) | `gini`, `entropy`, `squared_error`, `best_split`, `BaseDecisionTree`, `DecisionTreeClassifier`, `DecisionTreeRegressor` → `mlfs.trees` | with 20% label noise the fully grown tree tests at 0.664, worse than one split (0.700), as $2\varepsilon(1 - \varepsilon)$ predicts; refits on 95% of the rows disagree on 15% of test points; a straight diagonal boundary costs 50 leaves |
| 02 | [bagging & random forests](02-bagging-and-random-forests/) | `bootstrap_sample`, `RandomForestClassifier`, `RandomForestRegressor` → `mlfs.trees` | the forest's variance follows $\rho\sigma^2 + (1 - \rho)\sigma^2/B$ to a floor; random-forest trees are *more* variable than bagged ones yet have the lower floor; bagging cuts a stump's error by 9% and a full tree's by 58% |
| 03 | [boosting](03-boosting/) | `samme_update`, `exponential_loss`, `AdaBoostClassifier` → `mlfs.trees` | the test error falls from 0.149 to 0.133 *after* the training error hits zero, while the smallest margin keeps growing; 22 of the 30 heaviest points are the mislabeled ones; accuracy keeps improving while the log loss climbs from 0.32 to 0.54 |
| 04 | [gradient boosting](04-gradient-boosting/) | `quantile_bin_edges`, `bin_data`, `gradient_histograms`, `split_gain`, `leaf_weight`, `best_hist_split`, `HistGradientTree`, `subsample_rows`, `GradientBoostingRegressor`, `GradientBoostingClassifier` → `mlfs.trees` | learning rate 1 bottoms out at RMSE 0.580, 0.03 reaches 0.496; California needs 128 bins to be within 1% of 256 (16 bins cost 12%); Newton leaves reach the best log loss in 16 trees, gradient leaves never do in 400 |
| 05 | [interpreting models](05-interpreting-models/) | `permutation_importance`, `ice_curves`, `partial_dependence`, `kernel_shap` → `mlfs.inspection`, `default_grid`, `masked_predictions`, `shapley_values_exact`, `sample_coalitions` | impurity importance ranks a pure-noise continuous feature first and the only real signal fourth; a feature's near copy halves its permutation importance while dropping either costs nothing; a feature that's half the model has a flat partial dependence |

## capstone

*a tabular benchmark*: your gradient booster against scikit-learn's `HistGradientBoostingRegressor` on California housing (20,433 districts), with every choice made on a validation set, one look at the test set and a paired bootstrap CI on the RMSE gap. ratio features (which trees can't build themselves) and a random search take your booster's validation RMSE from 0.479 to 0.455; on the test set it gets **0.4419** against the benchmark's **0.4394**, a gap of **+0.55%** (95% CI −0.78% to +1.98%): within the 3% margin. small CPU config: about 5 seconds (offline). full config: about 4–5 minutes on one core, most of it the search.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a split's Gini decreases, two rounds of AdaBoost, one second-order boosting step), and a 60-minute blind rebuild of CART and the histogram booster.

## notation

| symbol | meaning |
|---|---|
| $X$, $x_i$, $x_{ij}$; $n$, $d$ | the data matrix $(n, d)$, row $i$, feature $j$ of row $i$; numbers of rows and features |
| $w_i$, $W$, $W_L$, $W_R$ | sample weights; summed weights of a node and of its left and right children |
| $I(\cdot)$, $p_k$ | a node's impurity (Gini, entropy or variance); class proportions in a node |
| $\Delta(j, t)$ | the weighted impurity decrease of the split "$x_j \le t$" |
| $R(t)$, $R(T_t)$, $\alpha_\text{eff}$ | weighted impurity of node $t$, summed over the leaves of its subtree; the effective alpha of pruning |
| $B$, $\rho$, $\sigma^2$ | number of trees in a forest; correlation between two trees' predictions; one tree's variance |
| $h_m$, $\alpha_m$, $\varepsilon_m$ | AdaBoost's $m$-th weak learner (in $\{-1, +1\}$), its vote and its weighted error |
| $f(x)$, $F(x)$ | an ensemble's score (AdaBoost's $\sum\alpha_mh_m$; gradient boosting's sum of trees, a log-odds for classification) |
| $g_i$, $h_i$; $G$, $H$ | gradient and hessian of the loss at row $i$; their sums over a node or a leaf |
| $\eta$, $\lambda$, $\gamma$ | learning rate (shrinkage); L2 penalty on leaf values; minimum gain per split |
| $\text{PD}_j$, $\phi_j$, $v(S)$ | partial dependence on feature $j$; feature $j$'s Shapley value; the value of a coalition $S$ of features |

## how this connects

- builds on: the estimator protocol, metrics, scorers and bias–variance (ch 07), the bootstrap and entropy (ch 05), Newton's method (ch 06), variance of correlated sums (ch 04), weighted least squares (ch 02), vectorized NumPy and `as_rng` (ch 00).
- used later in: isolation forests for anomaly detection (ch 11 · topic 06); boosting and margins in learning theory (ch 13); early stopping and learning-rate schedules for neural networks (ch 15); tree ensembles as the tabular baseline that deep models are measured against throughout the course; feature attribution and calibration (ch 28).

## core references

- Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression Trees*, Wadsworth (now CRC Press). [doi:10.1201/9781315139470](https://doi.org/10.1201/9781315139470)
- Breiman (2001), "Random forests", *Machine Learning* 45(1): 5–32. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- Freund & Schapire (1997), "A decision-theoretic generalization of on-line learning and an application to boosting", *Journal of Computer and System Sciences* 55(1): 119–139. [doi:10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
- Friedman (2001), "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29(5): 1189–1232. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
- Chen & Guestrin (2016), "XGBoost: a scalable tree boosting system", *KDD*. [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapters 9, 10, 15 and 16. [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
