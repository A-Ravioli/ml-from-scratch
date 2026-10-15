import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import checks, datasets
    from mlfs.stats import benjamini_hochberg, bonferroni, bootstrap_ci, permutation_test  # yours, topics 03-04
    from mlfs.utils import as_rng, pairwise_distances  # yours, from ch 00

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # small CPU config (offline, tests, verify) vs full config (the real California housing data)
    OFFLINE = datasets.is_offline()
    N_BOOT = 2000
    N_PERM = 2000


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 · capstone: a model-comparison toolkit (reference solution)

    read [`brief.md`](brief.md) first. you'll build the toolkit that every evaluation in the
    rest of the course runs through: a confidence interval for any metric, a **paired**
    confidence interval and a **paired** permutation test for the difference between two
    models scored on the same test set, McNemar's exact test, and a function that compares
    many models against a baseline with multiple-comparison control. everything rests on
    your `mlfs.stats`. the data, the models and the experiments are written for you; the
    write-up at the end is yours.

    a **metric** is any function `metric(y_true, y_out) -> float` computed from per-example
    arrays: accuracy from predicted labels, or (later in the course) AUC from scores, log loss
    from probabilities. "outputs" are 1-D arrays, one entry per test example.

    | stage | you build |
    |---|---|
    | 1 | `accuracy`, `metric_ci`: a metric with a bootstrap confidence interval |
    | 2 | `paired_difference_ci`: an interval for the difference between two models on the same examples |
    | 3 | `paired_permutation_test`, `mcnemar_test`: is the difference real? |
    | 4 | `compare_models`: many models against a baseline, with multiple-comparison control |

    when the checks are green, scroll down to run the comparison and write up what you find.
    """)
    return


@app.function
def accuracy(y_true, y_pred) -> float:
    """stage 1: the fraction of examples where y_pred equals y_true (both (n,))."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


@app.function
def metric_ci(metric, y_true, y_out, *, alpha=0.05, n_boot=2000, method='bca', rng=None) -> tuple:
    """stage 1: (metric(y_true, y_out), lo, hi): the metric on the test set, with a bootstrap confidence
    interval from mlfs.stats.bootstrap_ci, resampling EXAMPLES (the pair (y_true, y_out) jointly)."""
    y_true, y_out = np.asarray(y_true), np.asarray(y_out)
    est = float(metric(y_true, y_out))
    lo, hi = bootstrap_ci((y_true, y_out), metric, n_boot=n_boot, alpha=alpha, method=method, rng=rng)
    return est, float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", accuracy=accuracy, metric_ci=metric_ci)
    return


@app.function
def paired_difference_ci(metric, y_true, out_a, out_b, *, alpha=0.05, n_boot=2000, method='bca', rng=None) -> tuple:
    """stage 2: (diff, lo, hi) for diff = metric(y_true, out_a) - metric(y_true, out_b), with a bootstrap
    interval that resamples examples, keeping each example's (y_true, out_a, out_b) together."""
    y_true, out_a, out_b = np.asarray(y_true), np.asarray(out_a), np.asarray(out_b)

    def diff(y, a, b):
        return float(metric(y, a) - metric(y, b))

    lo, hi = bootstrap_ci((y_true, out_a, out_b), diff, n_boot=n_boot, alpha=alpha, method=method, rng=rng)
    return diff(y_true, out_a, out_b), float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", paired_difference_ci=paired_difference_ci, accuracy=accuracy)
    return


@app.function
def paired_permutation_test(metric, y_true, out_a, out_b, *, n_permutations=2000, rng=None) -> tuple:
    """stage 3: (diff, p): a two-sided paired permutation test of "models a and b are equally good",
    using mlfs.stats.permutation_test with paired=True: under the null, swapping a's and b's outputs on any
    example changes nothing. the statistic is metric(y_true, a') - metric(y_true, b')."""
    y_true = np.asarray(y_true)
    return permutation_test(np.asarray(out_a), np.asarray(out_b),
                            lambda a, b: float(metric(y_true, a) - metric(y_true, b)),
                            n_permutations=n_permutations, paired=True, rng=rng)


@app.function
def mcnemar_test(y_true, pred_a, pred_b) -> tuple:
    """stage 3: McNemar's exact test for two classifiers' predictions. returns (b, c, p).

    b = #examples a gets right and b gets wrong, c = the reverse. under the null each of the b + c
    discordant examples is equally likely to go either way: p = min(1, 2 P(Binomial(b + c, 1/2) <= min(b, c))),
    computed exactly (math.comb), and p = 1 when b + c = 0.
    """
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    ra, rb = pred_a == y_true, pred_b == y_true
    b, c = int(np.sum(ra & ~rb)), int(np.sum(~ra & rb))
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return b, c, float(min(1.0, 2 * tail / 2**n))


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", paired_permutation_test=paired_permutation_test,
               mcnemar_test=mcnemar_test, accuracy=accuracy)
    return


@app.function
def compare_models(metric, y_true, outputs, baseline, *, alpha=0.05, correction='bh', n_boot=2000, n_permutations=2000, rng=None) -> list:
    """stage 4: compare every model in outputs (dict name -> (n,) outputs) with outputs[baseline].

    returns one dict per non-baseline model, in the dict's order, with keys
      "model", "score" (its metric), "diff" (score - baseline score), "lo", "hi" (paired_difference_ci at
      level alpha), "p" (paired_permutation_test), and "reject" (bool).
    "reject" applies the correction to all the p-values together: 'bh' (benjamini_hochberg at alpha),
    'bonferroni', or 'none' (p <= alpha). an unknown baseline or correction raises ValueError.
    one generator (as_rng(rng)) is shared by every resampling call.
    """
    if baseline not in outputs:
        raise ValueError(f"baseline {baseline!r} isn't one of the models {list(outputs)}")
    if correction not in ("bh", "bonferroni", "none"):
        raise ValueError(f"correction must be 'bh', 'bonferroni' or 'none', got {correction!r}")
    rng = as_rng(rng)
    y_true = np.asarray(y_true)
    base = outputs[baseline]
    rows = []
    for name, out in outputs.items():
        if name == baseline:
            continue
        diff, lo, hi = paired_difference_ci(metric, y_true, out, base, alpha=alpha, n_boot=n_boot, rng=rng)
        _, p = paired_permutation_test(metric, y_true, out, base, n_permutations=n_permutations, rng=rng)
        rows.append({"model": name, "score": float(metric(y_true, np.asarray(out))), "diff": diff,
                     "lo": lo, "hi": hi, "p": p})
    ps = np.array([r["p"] for r in rows])
    if correction == "bh":
        reject = benjamini_hochberg(ps, alpha)
    elif correction == "bonferroni":
        reject = bonferroni(ps, alpha)
    else:
        reject = ps <= alpha
    for r, rej in zip(rows, reject):
        r["reject"] = bool(rej)
    return rows


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", compare_models=compare_models, accuracy=accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    the California housing data: one row per census district (1990), with 8 numeric features
    such as median income, house age, location and population. the task: **is the district's
    median house value above the median of the training districts?** a balanced binary
    classification problem. offline, `mlfs.datasets` substitutes a small synthetic stand-in
    with the same columns. the rows are split at random, 70% to train and 30% to test, and the
    features are standardized with the training set's means and standard deviations.
    """)
    return


@app.cell
def _():
    X_all, value, names = datasets.california_housing()
    _rng = np.random.default_rng(0)
    _perm = _rng.permutation(len(X_all))
    _cut = int(0.7 * len(X_all))
    _tr, _te = _perm[:_cut], _perm[_cut:]
    threshold = float(np.median(value[_tr]))
    y_train, y_test = (value[_tr] > threshold).astype(int), (value[_te] > threshold).astype(int)
    _mu, _sd = X_all[_tr].mean(axis=0), X_all[_tr].std(axis=0)
    X_train, X_test = (X_all[_tr] - _mu) / _sd, (X_all[_te] - _mu) / _sd
    mo.md(f"{'**offline stand-in**' if OFFLINE else '**California housing**'}: {len(X_all):,} districts, "
          f"{len(names)} features ({', '.join(names)}). train {len(y_train):,} · test {len(y_test):,} · "
          f"label = median value above {threshold * 1e5:,.0f} dollars · fraction positive in test {y_test.mean():.3f}")
    return X_test, X_train, names, y_test, y_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the models (given)

    six classifiers, from trivial to reasonable, each a few lines of NumPy: the majority class;
    a one-feature rule (median income above a threshold chosen on the training set); nearest
    centroid; and $k$-nearest neighbours with $k = 1$ and $k = 15$, on all features, plus
    15-NN on location (longitude, latitude) alone. none of them is the point: they're
    something to compare. chapters 08–10 build better ones.
    """)
    return


@app.cell
def _(X_test, X_train, names, y_test, y_train):
    def _knn(Xtr, ytr, Xte, k):
        out = np.empty(len(Xte), dtype=int)
        for _s in range(0, len(Xte), 500):  # chunks keep the distance matrix small
            _d = pairwise_distances(Xte[_s:_s + 500], Xtr, squared=True)
            _nn = np.argpartition(_d, k - 1, axis=1)[:, :k]
            out[_s:_s + 500] = (ytr[_nn].mean(axis=1) > 0.5).astype(int)
        return out

    def _stump(xtr, ytr, xte):
        _cands = np.quantile(xtr, np.linspace(0.01, 0.99, 99))
        _acc = [np.mean((xtr > t) == ytr) for t in _cands]
        return (xte > _cands[int(np.argmax(_acc))]).astype(int)

    def _centroid(Xtr, ytr, Xte):
        _c = np.stack([Xtr[ytr == 0].mean(axis=0), Xtr[ytr == 1].mean(axis=0)])
        return np.argmin(pairwise_distances(Xte, _c), axis=1)

    _t0 = time.perf_counter()
    _inc, _loc = names.index("median_income"), [names.index("longitude"), names.index("latitude")]
    preds = {
        "majority": np.full(len(y_test), int(y_train.mean() > 0.5)),
        "income rule": _stump(X_train[:, _inc], y_train, X_test[:, _inc]),
        "nearest centroid": _centroid(X_train, y_train, X_test),
        "1-NN": _knn(X_train, y_train, X_test, 1),
        "15-NN": _knn(X_train, y_train, X_test, 15),
        "15-NN, location only": _knn(X_train[:, _loc], y_train, X_test[:, _loc], 15),
    }
    model_secs = time.perf_counter() - _t0
    mo.md(f"fitted and evaluated {len(preds)} models in {model_secs:.1f} s.")
    return (preds,)


@app.cell
def _(preds, y_test):
    # every model's test accuracy with a 95% BCa interval (your metric_ci)
    _t0 = time.perf_counter()
    _rows, _ests = [], {}
    for _k, (_name, _p) in enumerate(preds.items()):
        _ests[_name] = metric_ci(accuracy, y_test, _p, n_boot=N_BOOT, rng=_k)
        _a, _lo, _hi = _ests[_name]
        _rows.append(f"| {_name} | {_a:.4f} | [{_lo:.4f}, {_hi:.4f}] | {_hi - _lo:.4f} |")
    _fig, _ax = plt.subplots(figsize=(7, 3.2))
    _names = list(_ests)
    _mid = np.array([_ests[_n][0] for _n in _names])
    _err = np.array([[_ests[_n][0] - _ests[_n][1], _ests[_n][2] - _ests[_n][0]] for _n in _names]).T
    _ax.errorbar(_mid, np.arange(len(_names)), xerr=_err, fmt="o", capsize=3)
    _ax.set_yticks(np.arange(len(_names)), _names)
    _ax.invert_yaxis()
    _ax.set(xlabel="test accuracy (95% BCa interval)")
    mo.vstack([_fig, mo.md(f"computed in {time.perf_counter() - _t0:.1f} s\n\n| model | accuracy | 95% CI | width |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell
def _(preds, y_test):
    # everything against the income rule, with Benjamini-Hochberg (your compare_models)
    _t0 = time.perf_counter()
    _rows = compare_models(accuracy, y_test, preds, "income rule", n_boot=N_BOOT, n_permutations=N_PERM, rng=1)
    _md = "\n".join(f"| {_r['model']} | {_r['score']:.4f} | {_r['diff']:+.4f} | [{_r['lo']:+.4f}, {_r['hi']:+.4f}] | "
                    f"{_r['p']:.4f} | {'yes' if _r['reject'] else 'no'} |" for _r in _rows)
    mo.md(f"baseline: **income rule**, Benjamini–Hochberg at 0.05 ({time.perf_counter() - _t0:.1f} s)\n\n"
          "| model | accuracy | difference | 95% paired CI | permutation p | significant? |\n"
          "|---|---|---|---|---|---|\n" + _md)
    return


@app.cell
def _(preds, y_test):
    # one comparison up close: 15-NN vs 1-NN, paired vs unpaired
    _a, _b = preds["15-NN"], preds["1-NN"]
    _diff, _lo, _hi = paired_difference_ci(accuracy, y_test, _a, _b, n_boot=N_BOOT, rng=2)
    _, _p = paired_permutation_test(accuracy, y_test, _a, _b, n_permutations=N_PERM, rng=3)
    _bb, _cc, _pm = mcnemar_test(y_test, _a, _b)
    _ea, _la, _ha = metric_ci(accuracy, y_test, _a, n_boot=N_BOOT, rng=4)
    _eb, _lb, _hb = metric_ci(accuracy, y_test, _b, n_boot=N_BOOT, rng=5)
    _ra, _rb = (_a == y_test).astype(float), (_b == y_test).astype(float)
    _corr = float(np.corrcoef(_ra, _rb)[0, 1])
    _se_unpaired = math.sqrt(_ra.var() / len(_ra) + _rb.var() / len(_rb))
    mo.md(f"""
    **15-NN vs 1-NN** on the same {len(y_test):,} test examples:

    | | value |
    |---|---|
    | 15-NN accuracy (95% CI) | {_ea:.4f} [{_la:.4f}, {_ha:.4f}] |
    | 1-NN accuracy (95% CI) | {_eb:.4f} [{_lb:.4f}, {_hb:.4f}] |
    | difference, paired 95% CI | {_diff:+.4f} [{_lo:+.4f}, {_hi:+.4f}] |
    | difference, unpaired normal 95% CI (ignores the pairing) | {_diff:+.4f} [{_diff - 1.96 * _se_unpaired:+.4f}, {_diff + 1.96 * _se_unpaired:+.4f}] |
    | paired permutation test p | {_p:.4f} |
    | McNemar: 15-NN right & 1-NN wrong / the reverse / exact p | {_bb} / {_cc} / {_pm:.4f} |
    | correlation of the two models' per-example correctness | {_corr:.3f} |
    """)
    return


@app.cell
def _(preds, y_test):
    # how big must the test set be? the paired CI width for 15-NN vs the income rule, on random subsets
    _rng = np.random.default_rng(6)
    _sizes = [s for s in (100, 200, 400, 800, 1600, 3200, 6400) if s <= len(y_test)]
    _w = []
    for _m in _sizes:
        _idx = _rng.choice(len(y_test), _m, replace=False)
        _, _lo, _hi = paired_difference_ci(accuracy, y_test[_idx], preds["15-NN"][_idx], preds["income rule"][_idx],
                                           n_boot=1000, method="percentile", rng=_rng)
        _w.append(_hi - _lo)
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _ax.loglog(_sizes, _w, "o-", label="paired 95% CI width")
    _ax.loglog(_sizes, _w[0] * np.sqrt(_sizes[0] / np.array(_sizes)), "k--", lw=0.8, label="slope -1/2")
    _ax.set(xlabel="test examples used", ylabel="CI width (as a fraction)")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| 15-NN vs income rule | " + " | ".join(f"n = {_m:,}" for _m in _sizes) + " |\n|---|"
                           + "---|" * len(_sizes) + "\n| paired 95% CI width | " + " | ".join(f"{_x:.4f}" for _x in _w) + " |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** every model's accuracy with its 95% interval. which models are
       *significantly* better than the income rule after Benjamini–Hochberg, and by how much
       (with the paired interval)? is anything worse than the majority baseline?
    2. **paired vs unpaired.** for 15-NN vs 1-NN, compare the width of the paired interval for
       the difference with the unpaired one, and explain the ratio with the correlation of the
       models' per-example correctness. on the full data the two individual intervals don't
       even overlap, so pairing changes no *conclusion* here. now look at nearest centroid vs
       the income rule, whose individual intervals overlap heavily: compare the paired and
       unpaired intervals for that difference. on the full data both still include 0, so
       pairing narrows the interval without flipping the verdict. how much narrower is it, and
       how much larger a test set would the unpaired analysis need to match it? (on the offline
       stand-in, 300 test examples, the numbers differ: report what you see.)
    3. **three tests, one question.** compare the paired permutation p-value, McNemar's p-value
       and the paired bootstrap interval for the same comparison. do they agree? when could they
       disagree?
    4. **test-set size.** from the width-vs-size plot, how many test examples would you need to
       resolve a 1-point accuracy difference between these two kinds of model? how does that
       scale?
    5. **what the intervals don't cover.** the intervals describe resampling the *test set*.
       name two sources of uncertainty they leave out (think about the training set, the random
       split, and the geography of California: neighbouring districts are similar, and the
       split is random over districts). how would you estimate one of them?
    6. **multiple comparisons.** you compared five models with the baseline. what changes with
       Bonferroni instead of BH, or with no correction? if you'd tried 50 hyperparameter settings
       of k-NN and reported the best one's p-value, what would be wrong?
    """)
    return


if __name__ == "__main__":
    app.run()
