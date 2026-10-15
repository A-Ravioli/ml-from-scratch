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
    # 10 · capstone: a tabular benchmark (reference solution)

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

    the data loading is given. the experiments (A–F) run on your functions as soon as they're
    green; the write-up prompts are at the end.
    """)
    return


@app.function
def add_ratio_features(X):
    """stage 1: (n, 8) California columns -> (n, 11): append rooms/households, bedrooms/rooms, population/households (0 where a denominator is 0)."""
    X = np.asarray(X, dtype=float)

    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    return np.column_stack([X, ratio(X[:, 3], X[:, 6]), ratio(X[:, 4], X[:, 3]), ratio(X[:, 5], X[:, 6])])


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", add_ratio_features=add_ratio_features)
    return


@app.function
def staged_rmse(model, X, y):
    """stage 2: the RMSE after each tree, a float array (n_trees,), from model.staged_predict(X)."""
    y = np.asarray(y, dtype=float)
    return np.array([np.sqrt(np.mean((np.asarray(p) - y) ** 2)) for p in model.staged_predict(X)])


@app.function
def best_iteration(curve):
    """stage 2: (n_trees, rmse) at the minimum of a staged RMSE curve; n_trees counts from 1, ties go to fewer trees."""
    i = int(np.argmin(curve))
    return i + 1, float(curve[i])


@app.function
def predict_with_n_trees(model, X, n_trees):
    """stage 2: the prediction (n,) of the first n_trees trees (n_trees >= 1), via staged_predict."""
    if n_trees < 1:
        raise ValueError(f"n_trees must be >= 1, got {n_trees}")
    for k, p in enumerate(model.staged_predict(X), start=1):
        if k == n_trees:
            return np.array(p, copy=True)
    raise ValueError(f"the model has fewer than {n_trees} trees")


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
    gen = as_rng(rng)
    keys = sorted(space)
    results = []
    for _ in range(n_iter):
        params = {k: space[k][int(gen.integers(len(space[k])))] for k in keys}
        t0 = time.perf_counter()
        model = make_model(**params).fit(X_tr, y_tr)
        n_trees, val = best_iteration(staged_rmse(model, X_val, y_val))
        results.append({"params": params, "model": model, "n_trees": n_trees, "val_rmse": val,
                        "seconds": time.perf_counter() - t0})
    return sorted(results, key=lambda r: r["val_rmse"])


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", random_search=random_search)
    return


@app.function
def rmse_ci(y_true, y_pred, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (rmse, lo, hi): the RMSE and a percentile bootstrap CI over rows (mlfs.stats.bootstrap_ci)."""
    sq = (np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2
    lo, hi = bootstrap_ci(sq, lambda s: np.sqrt(np.mean(s)), n_boot=n_boot, alpha=alpha, rng=rng)
    return float(np.sqrt(sq.mean())), float(lo), float(hi)


@app.function
def relative_gap_ci(y_true, y_pred, y_ref, *, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (gap, lo, hi) for gap = rmse(y_pred) / rmse(y_ref) - 1, with a PAIRED bootstrap CI (rows resampled jointly)."""
    e = (np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)) ** 2
    r = (np.asarray(y_ref, dtype=float) - np.asarray(y_true, dtype=float)) ** 2

    def gap(a, b):
        return np.sqrt(np.mean(a) / np.mean(b)) - 1.0

    lo, hi = bootstrap_ci((e, r), gap, n_boot=n_boot, alpha=alpha, rng=rng)
    return float(gap(e, r)), float(lo), float(hi)


@app.function
def verdict(gap_ci, margin=0.03):
    """stage 4: "within" if the CI's upper end <= margin, "outside" if its lower end > margin, else "inconclusive"."""
    _, lo, hi = gap_ci
    if hi <= margin:
        return "within"
    if lo > margin:
        return "outside"
    return "inconclusive"


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
    _tr, _va = splits["train"], splits["val"]
    X_ratio = add_ratio_features(X_all)
    _m = GradientBoostingRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=6).fit(X_ratio[_tr], y_all[_tr])
    ratio_n, ratio_val = best_iteration(staged_rmse(_m, X_ratio[_va], y_all[_va]))
    use_ratios = ratio_val < default_val
    X_use = X_ratio if use_ratios else X_all
    mo.md(f"validation RMSE: raw features **{default_val:.4f}**, with ratios **{ratio_val:.4f}** "
          f"({ratio_n} trees). chosen for everything that follows: **{'raw + ratios' if use_ratios else 'raw'}**.")
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
    _tr, _va = splits["train"], splits["val"]
    space = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 6, 7, 8],
        "min_child_weight": [1.0, 5.0, 20.0],
        "reg_lambda": [0.0, 1.0, 5.0],
        "subsample": [0.7, 0.85, 1.0],
    }

    def make_model(**params):
        return GradientBoostingRegressor(n_estimators=N_TREES, rng=0, **params)

    _t = time.perf_counter()
    search = random_search(make_model, space, X_use[_tr], y_all[_tr], X_use[_va], y_all[_va], n_iter=N_ITER, rng=1)
    search_s = time.perf_counter() - _t
    _rows = "\n".join(
        f"| {r['val_rmse']:.4f} | {r['n_trees']} | " + ", ".join(f"{k}={v}" for k, v in r["params"].items())
        + f" | {r['seconds']:.1f} s |" for r in search)
    mo.md("| validation RMSE | trees | parameters | time |\n|---|---|---|---|\n" + _rows
          + f"\n\nsearch time: **{search_s:.0f} s**.")
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
    _tr, _va = splits["train"], splits["val"]
    best = search[0]
    final = best["model"]
    _t = time.perf_counter()
    with threadpool_limits(1):
        bench_final = HistGradientBoostingRegressor(max_iter=N_TREES, early_stopping=False, random_state=0).fit(
            X_use[_tr], y_all[_tr])
        bench_final_n, bench_final_val = best_iteration(staged_rmse(bench_final, X_use[_va], y_all[_va]))
    bench_final_s = time.perf_counter() - _t
    mo.md(f"yours: {best['params']}, **{best['n_trees']}** trees, validation RMSE **{best['val_rmse']:.4f}** "
          f"(fit {best['seconds']:.1f} s). benchmark on the same features: **{bench_final_n}** iterations, validation "
          f"RMSE **{bench_final_val:.4f}** ({bench_final_s:.1f} s).")
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
    _tr, _te = splits["train"], splits["test"]
    y_te = y_all[_te]
    pred_mean = np.full(len(_te), y_all[_tr].mean())
    with threadpool_limits(1):
        pred_bench = predict_with_n_trees(bench_final, X_use[_te], bench_final_n)
    pred_yours = predict_with_n_trees(final, X_use[_te], best["n_trees"])
    cis = {name: rmse_ci(y_te, p, n_boot=N_BOOT, rng=k) for k, (name, p) in enumerate(
        [("mean of training targets", pred_mean), ("HistGradientBoosting", pred_bench), ("yours", pred_yours)])}
    gap = relative_gap_ci(y_te, pred_yours, pred_bench, n_boot=N_BOOT, rng=7)
    result = verdict(gap, MARGIN)
    _rows = "\n".join(f"| {k} | {r:.4f} | [{lo:.4f}, {hi:.4f}] |" for k, (r, lo, hi) in cis.items())
    mo.md("| model | test RMSE | 95% CI |\n|---|---|---|\n" + _rows
          + f"\n\nrelative gap (yours vs benchmark): **{100 * gap[0]:+.2f}%**, 95% CI "
          f"[{100 * gap[1]:+.2f}%, {100 * gap[2]:+.2f}%] → verdict at a {100 * MARGIN:.0f}% margin: **{result}**. "
          f"in dollars: your typical error is about ${1e5 * cis['yours'][0]:,.0f} per district.")
    return pred_yours, y_te


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
    _te = splits["test"]
    _names = list(feature_names) + (["rooms/household", "bedrooms/room", "people/household"] if use_ratios else [])

    class _Cut:  # the final model, cut at its chosen number of trees, as an estimator with a score
        def predict(self, X):
            return predict_with_n_trees(final, X, best["n_trees"])

        def score(self, X, y):
            return float(1 - np.sum((y - self.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2))

    _pi = permutation_importance(_Cut(), X_use[_te], y_all[_te], n_repeats=3, rng=0)
    _order = np.argsort(_pi["importances_mean"])
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.barh(np.array(_names)[_order], _pi["importances_mean"][_order], xerr=_pi["importances_std"][_order])
    _ax.set(xlabel="test R² drop when shuffled", title="permutation importance of your final model")
    _fig.tight_layout()
    _fig
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
