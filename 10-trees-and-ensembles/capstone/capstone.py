import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor  # the benchmark (an oracle, not a building block)
    from threadpoolctl import threadpool_limits  # run the benchmark on one core, like your NumPy booster

    from mlfs import checks, datasets
    from mlfs.inspection import permutation_importance  # yours, from topic 05
    from mlfs.stats import bootstrap_ci  # yours, from ch 05
    from mlfs.trees import GradientBoostingRegressor  # yours, from topic 04
    from mlfs.utils import as_rng  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_TREES = 150 if SMALL else 1000        # upper limit; early stopping picks the number
    N_ITER = 3 if SMALL else 10             # random-search candidates
    N_BOOT = 500 if SMALL else 2000
    MARGIN = 0.03                           # "within 3% of HistGradientBoosting"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 10 · capstone: a tabular benchmark

    read [`brief.md`](brief.md) first. your gradient booster from topic 04 against
    scikit-learn's `HistGradientBoostingRegressor` on California housing: same data, same
    validation set for every choice, one look at the test set, and a verdict with a
    confidence interval.

    | stage | you build |
    |---|---|
    | 1 | `add_ratio_features`: three ratios trees can't build on their own |
    | 2 | `staged_rmse`, `best_iteration`, `predict_with_n_trees`: early stopping on a validation set |
    | 3 | `random_search`: tune the booster on the validation set |
    | 4 | `rmse_ci`, `relative_gap_ci`, `verdict`: the benchmark, with uncertainty |

    the data loading and experiment A (the benchmark and a default booster) are given; you
    write experiments B–F in the cells marked for them. the write-up prompts are at the end.
    """)
    return


@app.function
def add_ratio_features(X):
    """stage 1: (n, 8) California columns -> (n, 11): append rooms/households, bedrooms/rooms, population/households (0 where a denominator is 0)."""
    raise NotImplementedError("stage1: implement add_ratio_features")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", add_ratio_features=add_ratio_features)
    return


@app.function
def staged_rmse(model, X, y):
    """stage 2: the RMSE after each tree, a float array (n_trees,), from model.staged_predict(X)."""
    raise NotImplementedError("stage2: implement staged_rmse")


@app.function
def best_iteration(curve):
    """stage 2: (n_trees, rmse) at the minimum of a staged RMSE curve; n_trees counts from 1, ties go to fewer trees."""
    raise NotImplementedError("stage2: implement best_iteration")


@app.function
def predict_with_n_trees(model, X, n_trees):
    """stage 2: the prediction (n,) of the first n_trees trees (n_trees >= 1), via staged_predict."""
    raise NotImplementedError("stage2: implement predict_with_n_trees")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", staged_rmse=staged_rmse, best_iteration=best_iteration,
               predict_with_n_trees=predict_with_n_trees)
    return


@app.function
def random_search(make_model, space, X_tr, y_tr, X_val, y_val, *, n_iter=10, rng=None):
    """stage 3: n_iter candidates (each key drawn uniformly from its list), fitted on train and early-stopped on val.

    returns a list of dicts {"params", "model", "n_trees", "val_rmse", "seconds"} sorted by val_rmse (best first),
    where "model" is the fitted model (all its trees; use the first n_trees).
    """
    raise NotImplementedError("stage3: implement random_search")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", random_search=random_search)
    return


@app.function
def rmse_ci(y_true, y_pred, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (rmse, lo, hi): the RMSE and a percentile bootstrap CI over rows (mlfs.stats.bootstrap_ci)."""
    raise NotImplementedError("stage4: implement rmse_ci")


@app.function
def relative_gap_ci(y_true, y_pred, y_ref, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (gap, lo, hi) for gap = rmse(y_pred) / rmse(y_ref) - 1, with a PAIRED bootstrap CI (rows resampled jointly)."""
    raise NotImplementedError("stage4: implement relative_gap_ci")


@app.function
def verdict(gap_ci, margin=0.03):
    """stage 4: "within" if the CI's upper end <= margin, "outside" if its lower end > margin, else "inconclusive"."""
    raise NotImplementedError("stage4: implement verdict")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", rmse_ci=rmse_ci, relative_gap_ci=relative_gap_ci, verdict=verdict)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data, and three disjoint splits (given)

    California housing (1990 census): 8 numeric features per district, target = median house
    value in units of \$100,000. 60% of the rows train every model, 20% (validation) make
    every choice (number of trees, features, hyperparameters), and 20% (test) are locked
    until experiment E. small config: the 1,000-row synthetic stand-in.
    """)
    return


@app.cell
def _():
    X_all, y_all, feature_names = datasets.california_housing()
    _idx = np.random.default_rng(0).permutation(len(y_all))
    _n = len(y_all)
    _te, _va, _tr = _idx[: int(0.2 * _n)], _idx[int(0.2 * _n): int(0.4 * _n)], _idx[int(0.4 * _n):]
    splits = {"train": _tr, "val": _va, "test": _te}
    mo.md(f"{_n:,} districts · train {len(_tr):,} · validation {len(_va):,} · test {len(_te):,} (locked) · "
          f"config: **{'small (offline)' if SMALL else 'full'}**")
    return X_all, feature_names, splits, y_all


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: the benchmark and your default booster

    the benchmark is `HistGradientBoostingRegressor` with its default hyperparameters
    (learning rate 0.1, leaves of up to 31, $\lambda = 0$, 256 bins), with up to `N_TREES`
    iterations and the number chosen on the validation set, exactly as for your models. your
    booster starts from learning rate 0.1, depth 6, $\lambda = 1$. both run on one CPU core (the
    benchmark's OpenMP threads are limited to 1), so the timings compare algorithms, not core counts.
    """)
    return


@app.cell
def _(X_all, splits, y_all):
    _tr, _va = splits["train"], splits["val"]
    _t = time.perf_counter()
    with threadpool_limits(1):
        bench = HistGradientBoostingRegressor(max_iter=N_TREES, early_stopping=False, random_state=0).fit(
            X_all[_tr], y_all[_tr])
        bench_curve = staged_rmse(bench, X_all[_va], y_all[_va])
    bench_n, bench_val = best_iteration(bench_curve)
    bench_s = time.perf_counter() - _t
    _t = time.perf_counter()
    default = GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=6).fit(X_all[_tr], y_all[_tr])
    default_curve = staged_rmse(default, X_all[_va], y_all[_va])
    default_n, default_val = best_iteration(default_curve)
    default_s = time.perf_counter() - _t
    mean_val = float(np.sqrt(np.mean((y_all[_va] - y_all[_tr].mean()) ** 2)))
    _fig, _ax = plt.subplots(figsize=(7.5, 4))
    _ax.semilogx(np.arange(1, len(bench_curve) + 1), bench_curve, label="HistGradientBoosting (benchmark)")
    _ax.semilogx(np.arange(1, len(default_curve) + 1), default_curve, label="yours (lr 0.1, depth 6)")
    _ax.axhline(mean_val, color="gray", ls=":", label="predict the training mean")
    _ax.set(xlabel="trees", ylabel="validation RMSE ($100k)", ylim=(None, mean_val * 1.05))
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        f"| model | best trees | validation RMSE | fit + curve time |\n|---|---|---|---|\n"
        f"| mean of training targets | – | {mean_val:.4f} | – |\n"
        f"| HistGradientBoosting | {bench_n} | {bench_val:.4f} | {bench_s:.1f} s |\n"
        f"| yours, default | {default_n} | {default_val:.4f} | {default_s:.1f} s |")])
    return bench_val, default_val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: features trees can't build

    topic 01's lab: a tree needs many leaves for any boundary that isn't parallel to an axis,
    and a ratio of two features is such a boundary. rooms, bedrooms and population are
    *totals* per district; per household they mean something. same default booster, raw
    versus raw + ratios, judged on the validation set.
    """)
    return


@app.cell
def _(X_all, default_val, splits, y_all):
    # experiment B (yours to write): X_ratio = add_ratio_features(X_all); fit the same default booster
    # (n_estimators=N_TREES, learning_rate=0.1, max_depth=6) on the training rows, early-stop it on the
    # validation rows with staged_rmse + best_iteration, and compare with default_val. keep the better
    # feature set as X_use, and use_ratios = True if it's the one with ratios.
    X_use, use_ratios = X_all, False
    mo.md(f"experiment B isn't written yet (validation RMSE with raw features: {default_val:.4f}).")
    return X_use, use_ratios


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: random search on the validation set

    `N_ITER` candidates from the space below, each fitted on the training rows with up to
    `N_TREES` trees and early-stopped on the validation rows. this is the slow part of the
    full config (a few minutes).
    """)
    return


@app.cell
def _(X_use, splits, y_all):
    # experiment C (yours to write): a search space, e.g. learning_rate in [0.05, 0.1], max_depth in [5..8],
    # min_child_weight, reg_lambda and subsample; make_model(**params) returning
    # GradientBoostingRegressor(n_estimators=N_TREES, rng=0, **params); then random_search on the training
    # and validation rows of X_use with n_iter=N_ITER. show the results as a table, with their times.
    search = []
    mo.md("experiment C isn't written yet.")
    return (search,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment D: the final models

    the best candidate (fitted on the training rows during the search) is cut at its
    validation-chosen number of trees; the benchmark is refitted on the same features and cut
    at *its* validation-chosen number of iterations. nothing here has seen the test rows.
    """)
    return


@app.cell
def _(X_use, search, splits, y_all):
    # experiment D (yours to write): best = search[0] and its fitted model, cut at best["n_trees"]; refit the
    # benchmark (HistGradientBoostingRegressor, defaults, max_iter=N_TREES, early_stopping=False, inside
    # threadpool_limits(1)) on the SAME features and pick its number of iterations on the validation rows.
    best, final, bench_final, bench_final_n = None, None, None, None
    mo.md("experiment D isn't written yet.")
    return bench_final, bench_final_n, best, final


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment E: one look at the test set

    test RMSEs with 95% bootstrap CIs, the relative gap
    $\text{RMSE}_\text{yours}/\text{RMSE}_\text{benchmark} - 1$ with a paired bootstrap CI,
    and the verdict against the margin (3%).
    """)
    return


@app.cell
def _(X_all, X_use, bench_final, bench_final_n, best, final, splits, y_all):
    # experiment E (yours to write), the only cell that touches the test rows: predictions of the mean
    # baseline, the benchmark (predict_with_n_trees) and your final model; rmse_ci for each;
    # relative_gap_ci(y_test, yours, benchmark); and verdict(gap, MARGIN). report RMSEs in dollars too.
    _waiting = len(splits["test"])
    mo.md(f"experiment E isn't written yet ({_waiting:,} test rows are waiting).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment F: what did your model learn?

    permutation importance of your final model on the test rows (your topic-05 code), and
    the runtimes of the whole notebook so far.
    """)
    return


@app.cell
def _(X_use, best, feature_names, final, splits, use_ratios, y_all):
    # experiment F (yours to write): permutation_importance (mlfs.inspection) of your final model, cut at
    # best["n_trees"], on the test rows, as a bar chart with error bars; and a table of the runtimes.
    _n_features = X_use.shape[1]
    mo.md(f"experiment F isn't written yet ({_n_features} features to rank).")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer in a few sentences each, with numbers from the experiments above (and their
    uncertainty). [`rubric.md`](rubric.md) says what a strong answer contains.

    1. **the result.** how far is your booster from the benchmark on the test set, with what
       uncertainty, and what's the verdict? translate the RMSE into dollars.
    2. **what mattered.** of the choices you made (ratio features, number of trees,
       hyperparameters), which moved the validation RMSE the most? show the numbers.
    3. **fairness.** in what ways is this comparison fair, and in what ways isn't it (tuning
       budget, features, the benchmark's defaults, leaf-wise vs depth-wise trees)? what would
       you change to make it fairer?
    4. **cost.** how long did your booster take compared with the benchmark, and where does
       the time go? (profile one fit if you can.)
    5. **limits.** what does the bootstrap CI over test rows *not* capture? (think of the split
       seed, the search's randomness, neighbouring districts and the capped target.)
    """)
    return


if __name__ == "__main__":
    app.run()
