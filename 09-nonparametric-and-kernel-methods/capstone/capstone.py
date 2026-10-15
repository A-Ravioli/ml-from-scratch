import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.kernels import GaussianProcessRegressor, KernelRidge, expected_improvement  # yours, from topics 05-06
    from mlfs.model_selection import KFold  # yours, from ch 07 · topic 02
    from mlfs.preprocessing import StandardScaler  # yours, from ch 07 · topic 05
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.utils import as_rng  # yours, from ch 00 · topic 05

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_TRAIN = 300 if SMALL else 1500  # districts used for tuning (5-fold CV inside the objective)
    N_TEST = 200 if SMALL else 4000  # locked test districts, used once at the end
    BUDGET = 12 if SMALL else 30  # objective evaluations per search
    N_INIT = 4 if SMALL else 5  # random evaluations before the GP takes over
    N_SEEDS = 2 if SMALL else 6  # repetitions of each search
    GRID = 6 if SMALL else 16  # dense reference grid: GRID x GRID evaluations
    BOUNDS = np.array([[-6.0, 2.0], [-4.0, 2.0]])  # log10 alpha, log10 gamma of the kernel ridge model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 · capstone: Bayesian optimization

    read [`brief.md`](brief.md) first. you'll tune the two hyperparameters of a kernel ridge
    regression model on California housing, where every evaluation is a 5-fold
    cross-validation, with a Gaussian process and expected improvement, and measure whether it
    beats random search with the same budget.

    | stage | you build |
    |---|---|
    | 1 | `propose`: the candidate with the highest expected improvement under a fitted GP |
    | 2 | `bayes_opt`: the Bayesian-optimization loop over a box |
    | 3 | `random_search`: the baseline, with the same interface |
    | 4 | `krr_objective`: hyperparameters → cross-validated score of your `KernelRidge` |
    | 5 | `compare_searches`: both searches over several seeds, as best-so-far traces |

    the data loading is given. the experiments at the end run on your functions; the write-up
    prompts follow them.
    """)
    return


@app.function
def propose(gp, candidates, best, *, xi=0.01):
    """stage 1: (index of the candidate with the highest expected improvement over best, EI of every candidate (m,)); ties go to the first."""
    raise NotImplementedError("stage1: implement propose")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", propose=propose)
    return


@app.function
def bayes_opt(objective, bounds, *, n_init=5, n_iter=20, n_candidates=1000, xi=0.01, rng=None):
    """stage 2: maximize objective(x) over the box bounds (d, 2) with a Matérn-5/2 GP on the unit cube and expected improvement -> dict(X, y, best_x, best_y, trace)."""
    raise NotImplementedError("stage2: implement bayes_opt")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", bayes_opt=bayes_opt)
    return


@app.function
def random_search(objective, bounds, *, n_iter=25, rng=None):
    """stage 3: objective at n_iter points drawn uniformly in the box -> the same dict as bayes_opt."""
    raise NotImplementedError("stage3: implement random_search")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", random_search=random_search)
    return


@app.function
def krr_objective(X, y, *, cv=5, rng=None):
    """stage 4: a function params -> -(cross-validated RMSE) of KernelRidge(alpha=10**params[0], gamma=10**params[1]) on targets centred per fold, with folds fixed once by KFold(cv, shuffle=True, rng=as_rng(rng))."""
    raise NotImplementedError("stage4: implement krr_objective")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", krr_objective=krr_objective)
    return


@app.function
def compare_searches(objective, bounds, *, budget=25, n_init=5, n_seeds=5, rng=None):
    """stage 5: run bayes_opt (n_init + the rest) and random_search with the same budget for n_seeds seeds -> dict(bo=(n_seeds, budget) traces, random=(n_seeds, budget) traces, bo_runs, random_runs)."""
    raise NotImplementedError("stage5: implement compare_searches")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", compare_searches=compare_searches)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    California housing: 8 standardized features, target in \$100,000. `N_TRAIN` districts are
    used for tuning; every objective evaluation is a 5-fold cross-validation on them.
    `N_TEST` other districts are locked away for one final check. small config: the
    synthetic stand-in.
    """)
    return


@app.cell
def _():
    _X, _y, _ = datasets.california_housing()
    _perm = np.random.default_rng(0).permutation(len(_X))
    _tr, _te = _perm[:N_TRAIN], _perm[N_TRAIN:N_TRAIN + N_TEST]
    _scaler = StandardScaler().fit(_X[_tr])
    X_train, y_train = _scaler.transform(_X[_tr]), _y[_tr]
    X_test, y_test = _scaler.transform(_X[_te]), _y[_te]
    objective = krr_objective(X_train, y_train, cv=5, rng=1)
    mo.md(f"tuning on **{len(X_train):,}** districts, testing on **{len(X_test):,}** (locked). search box: "
          f"log10 α ∈ {BOUNDS[0].tolist()}, log10 γ ∈ {BOUNDS[1].tolist()}; budget {BUDGET} evaluations, {N_SEEDS} seeds.")
    return X_test, X_train, objective, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: the landscape, by brute force (given)

    a `GRID` × `GRID` grid of evaluations, to know what "good" looks like. this is what you
    *can't* afford when one evaluation takes an hour; here it takes a fraction of a second.
    """)
    return


@app.cell
def _(objective):
    grid_a = np.linspace(*BOUNDS[0], GRID)
    grid_g = np.linspace(*BOUNDS[1], GRID)
    grid_scores = np.array([[objective(np.array([a, g])) for g in grid_g] for a in grid_a])
    grid_best = float(grid_scores.max())
    _i, _j = np.unravel_index(np.argmax(grid_scores), grid_scores.shape)
    mo.md(f"grid optimum: CV RMSE **{-grid_best:.4f}** at log10 α = {grid_a[_i]:.2f}, log10 γ = {grid_g[_j]:.2f}; "
          f"worst grid point {-grid_scores.min():.3f}; median {-np.median(grid_scores):.3f}.")
    return grid_a, grid_best, grid_g, grid_scores


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: Bayesian optimization against random search

    both searches, `N_SEEDS` seeds each, `BUDGET` evaluations each, on your objective.
    left: the best CV RMSE found so far (median and range over seeds) against the number of
    evaluations. right: where one Bayesian-optimization run and one random-search run
    evaluated, over the grid's landscape.
    """)
    return


@app.cell
def _(grid_a, grid_best, grid_g, grid_scores, objective):
    results = compare_searches(objective, BOUNDS, budget=BUDGET, n_init=N_INIT, n_seeds=N_SEEDS, rng=2)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    _n = np.arange(1, BUDGET + 1)
    for _name, _c in (("bo", "C3"), ("random", "C0")):
        _t = -results[_name]
        _a1.plot(_n, np.median(_t, axis=0), color=_c, lw=2, label={"bo": "Bayesian optimization", "random": "random search"}[_name])
        _a1.fill_between(_n, _t.min(axis=0), _t.max(axis=0), color=_c, alpha=0.15)
    _a1.axhline(-grid_best, color="k", ls=":", lw=1, label=f"grid optimum ({GRID}×{GRID} evaluations)")
    _a1.axvline(N_INIT + 0.5, color="gray", lw=0.8)
    _a1.set(xlabel="evaluations", ylabel="best CV RMSE so far ($100k)", ylim=(-grid_best - 0.01, -grid_best + 0.25))
    _a1.legend(fontsize=8)
    _cs = _a2.contourf(grid_g, grid_a, np.maximum(-grid_scores, -grid_best - 1e-9).clip(None, -grid_best + 0.5), levels=25,
                       cmap="viridis_r")
    _fig.colorbar(_cs, ax=_a2, label="CV RMSE (clipped)")
    _bo, _rs = results["bo_runs"][0]["X"], results["random_runs"][0]["X"]
    _a2.plot(_rs[:, 1], _rs[:, 0], "wo", ms=5, mec="k", label="random search")
    _a2.plot(_bo[:N_INIT, 1], _bo[:N_INIT, 0], "s", color="orange", ms=6, mec="k", label="BO: initial design")
    _a2.plot(_bo[N_INIT:, 1], _bo[N_INIT:, 0], "r^", ms=6, mec="k", label="BO: chosen by EI")
    _a2.set(xlabel="log10 γ", ylabel="log10 α", title="seed 0")
    _a2.legend(fontsize=7, loc="lower left")
    _fig.tight_layout()
    _final = {k: -results[k][:, -1] for k in ("bo", "random")}
    _half = {k: -results[k][:, BUDGET // 2 - 1] for k in ("bo", "random")}
    _hits = {k: int(np.sum(_final[k] <= -grid_best + 0.005)) for k in _final}
    _paired = _final["random"] - _final["bo"]  # per seed (the two searches share seeds): > 0 means BO found better
    mo.vstack([_fig, mo.md(
        f"after {BUDGET} evaluations, best CV RMSE: BO median **{np.median(_final['bo']):.4f}** "
        f"(range {_final['bo'].min():.4f}–{_final['bo'].max():.4f}), random median **{np.median(_final['random']):.4f}** "
        f"(range {_final['random'].min():.4f}–{_final['random'].max():.4f}); grid optimum {-grid_best:.4f}. "
        f"after {BUDGET // 2}: BO {np.median(_half['bo']):.4f}, random {np.median(_half['random']):.4f}. "
        f"runs within 0.005 of the grid optimum: BO {_hits['bo']}/{N_SEEDS}, random {_hits['random']}/{N_SEEDS}. "
        f"paired by seed, random minus BO: {', '.join(f'{d:+.4f}' for d in _paired)} (BO better in "
        f"{int(np.sum(_paired > 0))}/{N_SEEDS} seeds, mean difference {_paired.mean():+.4f}).")])
    return (results,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: one look at the test set

    refit `KernelRidge` on all `N_TRAIN` tuning districts with: the class defaults
    ($\alpha = 1$, $\gamma = 1$), the best hyperparameters of the first random-search run, and
    of the first Bayesian-optimization run. test RMSE on the locked districts, with 95%
    bootstrap intervals, and the mean predictor as the baseline.
    """)
    return


@app.cell
def _(X_test, X_train, results, y_test, y_train):
    def _test_rmse(params):
        a, g = 10.0 ** np.asarray(params)
        m = KernelRidge(a, gamma=g).fit(X_train, y_train - y_train.mean())
        err2 = (m.predict(X_test) + y_train.mean() - y_test) ** 2
        lo, hi = bootstrap_ci(err2, lambda e: float(np.sqrt(np.mean(e))), n_boot=2000, rng=3)
        return float(np.sqrt(np.mean(err2))), lo, hi

    _rows = {
        "mean of y (baseline)": (float(np.sqrt(np.mean((y_train.mean() - y_test) ** 2))), np.nan, np.nan),
        "defaults (α = 1, γ = 1)": _test_rmse([0.0, 0.0]),
        "random search, seed 0": _test_rmse(results["random_runs"][0]["best_x"]),
        "Bayesian optimization, seed 0": _test_rmse(results["bo_runs"][0]["best_x"]),
    }
    _table = "| model | test RMSE | 95% CI |\n|---|---|---|\n" + "\n".join(
        f"| {k} | {r:.4f} | " + ("–" if np.isnan(lo) else f"{lo:.4f}–{hi:.4f}") + " |" for k, (r, lo, hi) in _rows.items())
    _bo = results["bo_runs"][0]
    mo.vstack([mo.md(_table), mo.md(
        f"BO's choice: log10 α = {_bo['best_x'][0]:.2f}, log10 γ = {_bo['best_x'][1]:.2f} (CV RMSE {-_bo['best_y']:.4f}).")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer in a few sentences each, with numbers from the experiments above (and their
    spread over seeds):

    1. **the result.** how much better than the defaults is the tuned model on the test set,
       in dollars, with its uncertainty? how close did each search get to the grid optimum,
       and after how many evaluations?
    2. **why it works.** in experiment B's right panel, where did EI send the evaluations,
       and why there? when did it explore rather than exploit?
    3. **the baseline.** is random search a weak baseline here? what about the shape of this
       landscape helps or hurts each method? would the gap grow or shrink with 5
       hyperparameters, or with a smaller box?
    4. **the cost of choosing.** the best CV score of a search is the maximum of many noisy
       estimates (chapter 07, topic 06). how optimistic is it, judging by the test RMSE?
    5. **limits.** what did the GP assume about the objective (smoothness, stationarity, noise),
       and where might those assumptions fail for real hyperparameter tuning?
    """)
    return


if __name__ == "__main__":
    app.run()
