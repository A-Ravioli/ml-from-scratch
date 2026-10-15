# 07 · the machine learning workflow

> set up learning problems correctly and evaluate them without fooling yourself.

this is where the course turns from mathematics to machine learning, and it starts with the part that's easiest to get wrong: measuring. a model is only as good as the evidence that it works on data it hasn't seen, and that evidence is surprisingly easy to fake by accident. a lookup table scores 100% on its training data and nothing on anything else. a feature selector run before cross-validation turns coin flips into 94% "accuracy". a fraud detector is 99% accurate by never flagging fraud. a hyperparameter search reports the luckiest of twenty noisy scores as if it were the truth. you'll reproduce every one of these failures with your own code, and build the machinery that prevents them: the estimator protocol every later model speaks, splitters and cross-validation, a full metrics library, bias–variance diagnostics, leak-proof pipelines and honest hyperparameter search. from here on, every number you report comes with a baseline and an uncertainty estimate.

**you will build:** `check_X_y`; the `Estimator` protocol (`get_params`/`set_params`, `clone`) and its `Regressor`, `Classifier` and `Transformer` bases; `DummyRegressor` and `DummyClassifier`; a lookup table and an exact-ERM decision stump; `train_test_split`, `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `cross_validate`, `cross_val_score`; accuracy, confusion matrices, precision/recall/F1 with every averaging, ROC and precision–recall curves with AUC and AP, log loss, MSE/MAE/$R^2$, calibration curves, ECE and reliability diagrams, and scorers; a polynomial regressor, a simulation-based bias–variance decomposition, learning and validation curves; `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`, `PolynomialFeatures`, `Pipeline` and `make_pipeline`; `ParameterGrid`, `GridSearchCV`, `RandomizedSearchCV` and nested cross-validation. modules: **`mlfs.base`** (topic 01), **`mlfs.model_selection`** (topics 02, 04, 06), **`mlfs.metrics`** (topic 03) and **`mlfs.preprocessing`** (topic 05). every classical model in chapters 08–13 subclasses your base classes, and every evaluation in the rest of the course runs through your splitters, metrics, pipelines and searches.
**prerequisites:** chapter 00 (classes, vectorized NumPy, `as_rng`, `pairwise_distances`), chapter 02 (least squares), chapter 04 (expectation and variance, Hoeffding's inequality), chapter 05 (bias and variance of estimators, cross-entropy and KL, the bootstrap and permutation tests). chapter 06 helps for the conditioning argument in topic 05 but isn't required.
**time · scaffolding:** about 60 hours · L2 (supported): every function and method comes with its signature and a docstring with shapes; hints have three levels. the capstone gives you the data loading and the evaluation report; you write the models, the out-of-fold predictions, the statistics and the tuned pipeline.
**newly unlocked library components:** after this chapter you may use scikit-learn's `model_selection` (splitters, `cross_validate`, `GridSearchCV`, `RandomizedSearchCV`), `metrics`, `preprocessing` (scalers, encoders, `PolynomialFeatures`), `impute.SimpleImputer`, `pipeline.Pipeline` and `dummy` in labs and experiments, since you've built what they implement. inside `mlfs` modules, later chapters use *your* versions through `mlfs.base`, `mlfs.model_selection`, `mlfs.metrics` and `mlfs.preprocessing`, per the contract in [`CONTRACTS.md`](../CONTRACTS.md). scikit-learn's models (linear models, neighbours, trees, ...) stay locked until you've built each one.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build | the lab's surprise |
|---|---|---|---|
| 01 | [what is learning?](01-what-is-learning/) | `check_X_y`, `Estimator`, `clone`, `Regressor`, `Classifier`, `Transformer`, `DummyRegressor`, `DummyClassifier` → `mlfs.base`, `LookupTableClassifier`, `StumpClassifier` | the memorizer is perfect on training data and *exactly* the majority-class dummy on test data; with 8 possible inputs it's optimal; on coin flips the best of 200 features' stumps "learns" 75% |
| 02 | [splitting & cross-validation](02-splitting-and-cross-validation/) | `train_test_split`, `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `cross_validate`, `cross_val_score` → `mlfs.model_selection` | feature selection before splitting turns pure noise into 94% CV accuracy, while scaling before splitting changes nothing; recordings of the same patient give 100%; independent time series give 70% |
| 03 | [metrics](03-metrics/) | `accuracy_score`, `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `log_loss`, `mean_squared_error`, `mean_absolute_error`, `r2_score`, `calibration_curve`, `expected_calibration_error`, `reliability_diagram`, `make_scorer`, `get_scorer` → `mlfs.metrics` | an accuracy-maximizing model is 99.2% accurate and catches one fraud in five; ROC AUC stays at 0.95–0.96 as fraud gets rarer while AP falls from 0.96 to 0.16; three models with identical accuracy and AUC have log losses from 0.42 to 0.64 |
| 04 | [bias–variance & complexity](04-bias-variance-and-complexity/) | `PolynomialRegressor`, `bias_variance_decomposition`, `learning_curve`, `validation_curve` → `mlfs.model_selection` | a biased constant beats an unbiased line; a degree-12 fit to 30 points has variance about $1.5\times10^6$ times the noise at the lab's seed ($10^4$–$10^8$ across seeds), concentrated at the edges; more data doesn't move a straight line's learning curves off their plateau |
| 05 | [features & preprocessing](05-features-and-preprocessing/) | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer`, `PolynomialFeatures`, `Pipeline`, `make_pipeline` → `mlfs.preprocessing` | recording income in dollars drops kNN to near chance, and recording it in millions makes raw kNN beat standardized kNN; one typo in 600 rows costs min–max scaling 15 points |
| 06 | [hyperparameter search](06-hyperparameter-search/) | `ParameterGrid`, `GridSearchCV`, `RandomizedSearchCV`, `nested_cross_validate` → `mlfs.model_selection` | with 9 trials, random search beats the grid in 19 of 20 runs; on coin flips the search reports 57% and nested CV 48%; the top 5% is reached in about 60 random trials, in any dimension |

## capstone

*an end-to-end tabular project* with baselines, CV, CIs, and an honest evaluation report. you predict California house values with a chunked $k$-NN regressor and least squares, behind a locked test set: cross-validated RMSEs with bootstrap CIs for six models and two baselines, paired comparisons with permutation tests, a scaler-plus-kNN pipeline tuned by grid search and evaluated by nested CV, three deliberate ways of fooling yourself (one invisible, one that makes the leaky model look best: 0.48 against an honest 0.59), and one final look at the test set (RMSE 0.60 against the baseline's 1.15, in units of \$100,000). small CPU config: a few seconds (offline). full config: 2–3 minutes.

see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip it. it has 15 concept questions, three by-hand problems (a stump and its training error, a ROC curve and its AUC, a bias–variance computation), and a 60-minute blind rebuild of the estimator protocol, `cross_validate` and `GridSearchCV`.

## notation

| symbol | meaning |
|---|---|
| $X$ (`X`), $x_i$, $x_{ij}$ | the data matrix $(n, d)$, row $i$ (one example), feature $j$ of example $i$ |
| $y$, $y_i$, $\hat y_i$ | targets (labels or real values) and predictions |
| $n$, $d$ | number of examples, number of features |
| $P(x, y)$ | the (unknown) data-generating distribution |
| $\mathcal H$, $h$, $\hat h$ | hypothesis class, a hypothesis, the one chosen by learning |
| $\ell(y, \hat y)$, $R(h)$, $\hat R(h)$ | loss, risk (expected loss), empirical risk (training loss) |
| $k$ | the number of CV folds, *or* the number of neighbours in $k$-NN: the context says which |
| $\theta$ | a setting of hyperparameters |
| TP, FP, TN, FN; $P$, $N$ | confusion counts; numbers of positives and negatives |
| TPR, FPR, $\pi$ | true and false positive rates, prevalence of the positive class |
| $f(x)$, $\varepsilon$, $\sigma^2$ | the true regression function, noise, its variance |
| $\bar h(x)$ | the average prediction at $x$ over training sets |
| $\mu_j$, $\sigma_j$ | a feature's training mean and standard deviation (in scalers) |
| $\phi(x)$ | a feature map (polynomial features) |

## how this connects

- builds on: classes, vectorized NumPy and `as_rng` (ch 00), least squares (ch 02), conditioning (ch 06), expectation, variance and concentration (ch 04), estimators, cross-entropy, the bootstrap and permutation tests (ch 05).
- used later in: every model in chapters 08–13 (they subclass `Regressor`, `Classifier` or `Transformer` and are evaluated with your splitters, scorers and searches), the text-classifier capstone (ch 08), learning theory's generalization bounds (ch 13, which turns topic 01's union bound into a theory), validation splits, early stopping and hyperband (ch 15), calibration and conformal prediction (ch 28), and the experimental hygiene of the research chapters (ch 30).

## core references

- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer, chapters 2 and 7 (supervised learning; model assessment and selection, including the wrong and right way to do cross-validation). [doi:10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, chapters 2–5 and 11 (ERM, the bias–complexity trade-off, model selection and validation). [cs.huji.ac.il/~shais/UnderstandingMachineLearning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- Buitinck et al. (2013), "API design for machine learning software: experiences from the scikit-learn project". the estimator, transformer and pipeline protocol. [arXiv:1309.0238](https://arxiv.org/abs/1309.0238)
- Kaufman, Rosset, Perlich & Stitelman (2012), "Leakage in data mining: formulation, detection, and avoidance", *ACM Transactions on Knowledge Discovery from Data* 6(4). [doi:10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579)
- Bergstra & Bengio (2012), "Random search for hyper-parameter optimization", *JMLR* 13: 281–305. [jmlr.org/papers/v13/bergstra12a](https://www.jmlr.org/papers/v13/bergstra12a.html)
- Cawley & Talbot (2010), "On over-fitting in model selection and subsequent selection bias in performance evaluation", *JMLR* 11: 2079–2107. [jmlr.org/papers/v11/cawley10a](https://www.jmlr.org/papers/v11/cawley10a.html)
