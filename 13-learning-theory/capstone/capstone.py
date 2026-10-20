import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time
    import warnings

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from scipy.special import ndtr

    from mlfs import checks, datasets
    from mlfs.kernels import SVC, KNeighborsClassifier, get_kernel  # yours, from ch 09
    from mlfs.linear import LogisticRegression  # yours, from ch 08 · topic 04
    from mlfs.preprocessing import StandardScaler  # yours, from ch 07 · topic 05
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.theory import finite_class_bound, pac_bayes_bound, vc_bound  # yours, from this chapter
    from mlfs.utils import as_rng  # yours, from ch 00

    # small CPU config (offline: tests, verify) vs full config (your write-up)
    SMALL = datasets.is_offline()
    N_MNIST = 400 if SMALL else 2000  # training digits (odd vs even)
    N_MNIST_TEST = 400 if SMALL else 10000
    N_CAL = 400 if SMALL else 4000  # training districts (value above the training median?)
    N_CAL_TEST = 400 if SMALL else 4000
    N_DRAWS = 2 if SMALL else 5  # random-label fits per model
    N_BOOT = 300 if SMALL else 2000
    DELTA = 0.05
    GAMMAS = 2.0 ** -np.arange(0, 13)  # margin grid, fixed before seeing data (normalized margins live in [-1, 1])
    SIGMAS = np.geomspace(1e-3, 1.0, 31)  # posterior-width grid for PAC-Bayes, fixed before seeing data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 · capstone: a generalization report

    read [`brief.md`](brief.md) first. for three models from part II (logistic regression, an
    RBF-kernel SVM and $k$-nearest neighbours) on two real tasks (MNIST odd vs even, and
    California housing binarized at the median value), you'll **measure** the train–test gap
    with its uncertainty and **compute** every bound from this chapter that applies, then write
    up how they compare.

    | stage | you build |
    |---|---|
    | 1 | `measure_gap`: fit, train and test error, a bootstrap CI and the exact test-set bound |
    | 2 | `margin_bounds`: the ramp-loss margin bound and the PAC-Bayes bound, from normalized margins |
    | 3 | `linear_bounds`: VC, bit-counting, margin and PAC-Bayes bounds for a fitted linear classifier |
    | 4 | `svm_bounds`: the margin bound in the RKHS of a fitted kernel SVM |
    | 5 | `random_label_correlation`: how well a learner fits random labels (a Rademacher proxy) |

    the data loading is given. the experiments at the end run on your functions; the write-up
    prompts follow them.
    """)
    return


@app.function
def measure_gap(model, X_train, y_train, X_test, y_test, *, delta=0.05, n_boot=1000, rng=None):
    """stage 1: fit model; dict(train_error, test_error, gap, test_ci (95% percentile bootstrap of the test error), test_bound (exact binomial upper bound at delta))."""
    raise NotImplementedError("stage1: implement measure_gap")


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", measure_gap=measure_gap)
    return


@app.function
def margin_bounds(norm_margins, *, delta=0.05, gammas=GAMMAS, sigmas=SIGMAS):
    """stage 2: from normalized margins y u.x/||x|| (||u|| = 1) -> dict(margin_bound, margin_gamma, pac_bayes_bound, pac_bayes_sigma), each optimized over its grid with a union bound."""
    raise NotImplementedError("stage2: implement margin_bounds")


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", margin_bounds=margin_bounds)
    return


@app.function
def linear_bounds(model, X, y, *, delta=0.05, bits=32):
    """stage 3: for a fitted linear classifier (coef_, intercept_, classes_): dict(train_error, vc_bound, finite_bound (weights rounded to float{bits}), norm_margins, + margin_bounds keys)."""
    raise NotImplementedError("stage3: implement linear_bounds")


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", linear_bounds=linear_bounds)
    return


@app.function
def svm_bounds(svc, X, y, *, delta=0.05):
    """stage 4: for a fitted kernel SVC: dict(train_error, rkhs_norm, norm_margins, margin_bound, margin_gamma), margins normalized by ||(w, b)|| and ||(phi(x), 1)||."""
    raise NotImplementedError("stage4: implement svm_bounds")


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", svm_bounds=svm_bounds)
    return


@app.function
def random_label_correlation(make_model, X, *, n_draws=5, rng=None):
    """stage 5: mean over n_draws of (1/n) sum_i s_i h(x_i), where h = make_model() fitted to random signs s in {-1, +1}^n."""
    raise NotImplementedError("stage5: implement random_label_correlation")


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", random_label_correlation=random_label_correlation)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    **MNIST, odd vs even**: `N_MNIST` training digits drawn at random from the 60,000, pixels
    scaled to $[0, 1]$, labels $\pm1$; tested on `N_MNIST_TEST` of the official test digits.
    **California housing, binarized**: is a district's value above the median of the training
    districts? `N_CAL` training districts, `N_CAL_TEST` others for testing, features
    standardized with a scaler fitted on the training districts. small config: the synthetic
    stand-ins.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    _rng = np.random.default_rng(0)
    _i = _rng.choice(len(_d["X_train"]), size=min(N_MNIST, len(_d["X_train"])), replace=False)
    _j = _rng.choice(len(_d["X_test"]), size=min(N_MNIST_TEST, len(_d["X_test"])), replace=False)
    _flat = lambda A: A.reshape(len(A), -1).astype(float) / 255.0  # noqa: E731
    _pm = lambda y: np.where(y % 2 == 1, 1, -1)  # noqa: E731
    mnist = (_flat(_d["X_train"][_i]), _pm(_d["y_train"][_i]), _flat(_d["X_test"][_j]), _pm(_d["y_test"][_j]))
    _X, _y, _ = datasets.california_housing()
    _p = np.random.default_rng(1).permutation(len(_X))
    _n_te = min(N_CAL_TEST, len(_X) // 2)
    _te, _tr = _p[:_n_te], _p[_n_te:_n_te + N_CAL]
    _sc = StandardScaler().fit(_X[_tr])
    _thr = np.median(_y[_tr])
    cal = (_sc.transform(_X[_tr]), np.where(_y[_tr] > _thr, 1, -1), _sc.transform(_X[_te]), np.where(_y[_te] > _thr, 1, -1))
    mo.md(f"MNIST: {len(mnist[1]):,} train / {len(mnist[3]):,} test, {mnist[0].shape[1]} features; "
          f"California: {len(cal[1]):,} train / {len(cal[3]):,} test, {cal[0].shape[1]} features.")
    return cal, mnist


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the models (given)

    the same three model families on both tasks, with hyperparameters fixed in advance (a
    bound only holds for choices made without looking at the data it's computed on):

    - **logistic regression**, $\alpha = 10^{-4}$ (VC dimension $d + 1$; bounds: VC, 32-bit counting, margin, PAC-Bayes);
    - **RBF SVM**, $C = 5$, $\gamma = 0.02$ on MNIST and $0.5$ on California (VC dimension infinite; bound: RKHS margin);
    - **15-nearest neighbours** (VC dimension infinite; no uniform bound applies).

    every model also gets the test-set bound and the random-label correlation.
    """)
    return


@app.function
def make_models(task):
    """the three models for 'mnist' or 'california' (given)."""
    g = 0.02 if task == "mnist" else 0.5
    return {
        "logistic": lambda: LogisticRegression(alpha=1e-4),
        "RBF SVM": lambda: SVC(C=5.0, gamma=g, rng=0),
        "15-NN": lambda: KNeighborsClassifier(15),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment A: the report

    for every task and model: the measured train and test errors with a 95% bootstrap CI on
    the test error, the test-set bound, every applicable uniform bound, and the random-label
    correlation $\hat\rho$ (`N_DRAWS` fits on random labels). the last column is the
    Rademacher-style "proxy bound" $\text{train error} + \hat\rho + 3\sqrt{\ln(2/\delta)/(2n)}$,
    which is **not** a certified bound (the random-label fit only lower-bounds the class's
    complexity).
    """)
    return


@app.cell
def _(cal, mnist):
    report = {}
    _t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # logistic regression on separable random labels warns about convergence
        for _task, (_Xtr, _ytr, _Xte, _yte) in [("mnist", mnist), ("california", cal)]:
            for _k, (_name, _make) in enumerate(make_models(_task).items()):
                _model = _make()
                _row = measure_gap(_model, _Xtr, _ytr, _Xte, _yte, delta=DELTA, n_boot=N_BOOT, rng=10 + _k)
                if _name == "logistic":
                    _row.update({k: v for k, v in linear_bounds(_model, _Xtr, _ytr, delta=DELTA).items() if k != "train_error"})
                    # the PAC-Bayes bound is about the Gibbs classifier N(u, sigma^2 I): measure ITS test error
                    _mte = linear_bounds(_model, _Xte, _yte, delta=DELTA)["norm_margins"]
                    _row["gibbs_test"] = float(np.mean(ndtr(-_mte / _row["pac_bayes_sigma"])))
                    _row["pac_bayes_w"] = min(1.0, 2 * _row["pac_bayes_bound"])  # L(w) <= 2 L(Q)
                elif _name == "RBF SVM":
                    _row.update({k: v for k, v in svm_bounds(_model, _Xtr, _ytr, delta=DELTA).items() if k != "train_error"})
                _row.setdefault("vc_bound", math.inf)  # RBF SVM and 15-NN: infinite VC dimension
                _row["rho"] = random_label_correlation(_make, _Xtr, n_draws=N_DRAWS, rng=20 + _k)
                _row["proxy_bound"] = _row["train_error"] + _row["rho"] + 3 * math.sqrt(math.log(2 / DELTA) / (2 * len(_ytr)))
                report[(_task, _name)] = _row

    def _f(r, key):
        return ("∞" if r[key] == math.inf else f"{r[key]:.3f}") if key in r else "–"

    _lines = []
    for (_task, _name), _r in report.items():
        _lines.append(
            f"| {_task} | {_name} | {_r['train_error']:.3f} | {_r['test_error']:.3f} ({_r['test_ci'][0]:.3f}–{_r['test_ci'][1]:.3f}) | "
            f"{_r['gap']:+.3f} | {_r['test_bound']:.3f} | {_f(_r, 'vc_bound')} | {_f(_r, 'finite_bound')} | "
            f"{_f(_r, 'margin_bound')} | {_f(_r, 'pac_bayes_bound')} | {_f(_r, 'gibbs_test')} | {_f(_r, 'pac_bayes_w')} | "
            f"{_r['rho']:.3f} | {_r['proxy_bound']:.3f} |")
    mo.md("| task | model | train err | test err (95% CI) | gap | test-set bound | VC | 32-bit | margin | PAC-Bayes (Gibbs) | "
          "Gibbs test err | PAC-Bayes for w | ρ̂ | proxy |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(_lines)
          + f"\n\n(all bounds at δ = {DELTA}; – = no bound of this kind applies; computed in {time.perf_counter() - _t0:.0f} s.) "
          "the PAC-Bayes bound is for the **stochastic** Gibbs classifier N(u, σ²I) at the chosen σ: compare it with the "
          "Gibbs test error, not with w's. for w itself the valid statement is L(w) ≤ 2 L(Q), the next column.")
    return (report,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment B: where the margin and PAC-Bayes bounds come from

    the normalized margins $y\,u^\top\tilde x/\|\tilde x\|$ of the logistic regression and the
    SVM on each task (histograms), with the margin $\gamma$ and PAC-Bayes $\sigma$ each bound
    chose. a bound built on margins can only be good if most training points have margins
    well above the chosen $\gamma$, and if that $\gamma$ isn't too small for $n$. when the
    chosen $\gamma$ is 1, the largest value in the grid, the optimizer has given up: every
    smaller $\gamma$ makes $2/(\gamma\sqrt n)$ even larger, and the training margins sit far
    below all of them, so the bound is vacuous.
    """)
    return


@app.cell
def _(report):
    _fig, _axes = plt.subplots(1, 2, figsize=(11.5, 4))
    for _ax, _task in zip(_axes, ["mnist", "california"]):
        _all = np.concatenate([report[(_task, m)]["norm_margins"] for m in ("logistic", "RBF SVM")])
        _lo, _hi = np.quantile(_all, [0.005, 0.995])
        _bins = np.linspace(min(_lo, 0.0), _hi, 70)
        for _name, _c in [("logistic", "C0"), ("RBF SVM", "C3")]:
            _r = report[(_task, _name)]
            _ax.hist(np.clip(_r["norm_margins"], _bins[0], _bins[-1]), bins=_bins, alpha=0.5, color=_c, density=True,
                     label=f"{_name}: chosen γ = {_r['margin_gamma']:.3g}")
        _ax.axvline(0, color="k", lw=0.8)
        _ax.set(title=f"{_task}: normalized training margins", xlabel="y u·x̃ / ‖x̃‖  (clipped to the central 99%)",
                ylabel="density")
        _ax.legend(fontsize=7, title=f"PAC-Bayes σ (logistic) = {report[(_task, 'logistic')]['pac_bayes_sigma']:.3g}",
                   title_fontsize=7)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment C: more data, same model

    the bounds for one model as the training set grows: logistic regression on MNIST odd vs
    even trained on the first $n$ of a fixed shuffle of all training digits, for five values
    of $n$ up to 60,000 (offline: up to the stand-in's 2,000), tested on the full test set.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1).astype(float) / 255.0
    _y = np.where(_d["y_train"] % 2 == 1, 1, -1)
    _Xte = _d["X_test"].reshape(len(_d["X_test"]), -1).astype(float) / 255.0
    _yte = np.where(_d["y_test"] % 2 == 1, 1, -1)
    _perm = np.random.default_rng(2).permutation(len(_X))
    _sizes = sorted({min(n, len(_X)) for n in (500, 2000, 8000, 30000, 60000)})
    curve = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _n in _sizes:
            _m = LogisticRegression(alpha=1e-4)
            _g = measure_gap(_m, _X[_perm[:_n]], _y[_perm[:_n]], _Xte, _yte, delta=DELTA, n_boot=N_BOOT, rng=_n)
            _b = linear_bounds(_m, _X[_perm[:_n]], _y[_perm[:_n]], delta=DELTA)
            _mte = linear_bounds(_m, _Xte, _yte, delta=DELTA)["norm_margins"]
            _gibbs = float(np.mean(ndtr(-_mte / _b["pac_bayes_sigma"])))
            curve.append({"n": _n, **_g, **{k: v for k, v in _b.items() if k not in ("norm_margins", "train_error")},
                          "gibbs_test": _gibbs})
    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ns = [c["n"] for c in curve]
    _ax.semilogx(_ns, [c["test_error"] for c in curve], "ko-", label="test error (measured)")
    _ax.fill_between(_ns, [c["test_ci"][0] for c in curve], [c["test_ci"][1] for c in curve], color="k", alpha=0.15)
    _ax.semilogx(_ns, [c["train_error"] for c in curve], "k^:", label="train error")
    _ax.semilogx(_ns, [c["test_bound"] for c in curve], "C0o-", label="test-set bound")
    _ax.semilogx(_ns, [min(c["pac_bayes_bound"], 1.2) for c in curve], "C2s--", label="PAC-Bayes (prior at 0), Gibbs classifier")
    _ax.semilogx(_ns, [c["gibbs_test"] for c in curve], "C2s:", label="Gibbs classifier's test error")
    _ax.semilogx(_ns, [min(c["margin_bound"], 1.2) for c in curve], "C4v--", label="margin")
    _ax.semilogx(_ns, [min(c["finite_bound"], 1.2) for c in curve], "C1d--", label="32-bit counting")
    _ax.semilogx(_ns, [min(c["vc_bound"], 1.2) for c in curve], "C3x--", label="VC")
    _ax.axhline(0.5, color="0.5", ls=":", lw=1)
    _ax.set(xlabel="training digits n", ylabel="error / bound (clipped at 1.2)", ylim=(0, 1.25),
            title="logistic regression, MNIST odd vs even")
    _ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _fig.tight_layout()
    _rows = "\n".join(f"| {c['n']:,} | {c['train_error']:.3f} | {c['test_error']:.3f} | {c['test_bound']:.3f} | "
                      f"{c['gibbs_test']:.3f} | {c['pac_bayes_bound']:.3f} | {min(1.0, 2 * c['pac_bayes_bound']):.3f} | "
                      f"{c['margin_bound']:.3f} | {c['finite_bound']:.3f} | {c['vc_bound']:.3f} |" for c in curve)
    mo.vstack([_fig, mo.md("| n | train err | test err | test-set bound | Gibbs test err | PAC-Bayes (Gibbs) | PAC-Bayes for w "
                           "(2×) | margin | 32-bit | VC |\n|---|---|---|---|---|---|---|---|---|---|\n" + _rows)])
    return (curve,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    answer in a few sentences each, with numbers from the experiments above (and their
    uncertainty):

    1. **the measurements.** for each task, which model generalizes best, and is the
       difference between models larger than the test-error CIs? how large is each model's
       train–test gap, and which gaps are distinguishable from zero?
    2. **the bounds.** rank the bounds for logistic regression on each task from tightest to
       loosest. which are vacuous? for the tightest uniform bound, how far is it above the
       measured test error, and where does that slack come from (name the terms)?
    3. **models without a bound.** the SVM's margin bound and 15-NN's lack of any bound: does
       the absence of a guarantee mean poor generalization here? what does the random-label
       correlation say about each model, and why does it not predict the real gap?
    4. **the honest number.** if you had to promise a stakeholder an error rate for each
       model, which number would you give, and what did it cost (data, assumptions)?
    5. **limits.** list three assumptions behind these bounds (iid, fixed hyperparameters,
       the grids, data-independent priors...) and say which ones this experiment might
       violate, and how you'd fix it.
    """)
    return


if __name__ == "__main__":
    app.run()
