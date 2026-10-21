import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    from scipy.special import logsumexp

    from mlfs import checks, lesson
    from mlfs.metrics import roc_auc_score  # yours, from ch 07 · 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # uncertainty & OOD · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `fit_temperature`: one parameter that fixes a model's overconfidence → `mlfs.safety` |
    | 2 | `conformal_quantile`, `conformal_prediction_sets`: prediction sets with guaranteed coverage → `mlfs.safety` |
    | 3 | `conformal_intervals`: the same guarantee for regression → `mlfs.safety` |
    | 4 | `ensemble_uncertainty`, `ood_auroc`: splitting uncertainty into its two kinds, and detecting unusual inputs |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def fit_temperature(logits, labels, *, max_iters=100):
    """stage 1: T > 0 minimising the mean NLL of softmax(logits / T); Newton's method on beta = 1/T from T = 1."""
    raise NotImplementedError("stage1: implement fit_temperature")


@app.cell(hide_code=True)
def _():
    checks.run(1, fit_temperature=fit_temperature)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def conformal_quantile(scores, alpha):
    """stage 2: the ceil((n+1)(1-alpha))-th smallest of the n scores (inf if that rank exceeds n)."""
    raise NotImplementedError("stage2: implement conformal_quantile")


@app.function
def conformal_prediction_sets(cal_probs, cal_labels, test_probs, alpha=0.1):
    """stage 2: bool (n_test, k): class c is in the set iff 1 - p_c <= the conformal quantile of the 1 - p_y scores."""
    raise NotImplementedError("stage2: implement conformal_prediction_sets")


@app.cell(hide_code=True)
def _():
    checks.run(2, conformal_quantile=conformal_quantile, conformal_prediction_sets=conformal_prediction_sets)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def conformal_intervals(cal_pred, cal_y, test_pred, alpha=0.1):
    """stage 3: (lo, hi) = test_pred -/+ the conformal quantile of the |cal_y - cal_pred| scores."""
    raise NotImplementedError("stage3: implement conformal_intervals")


@app.cell(hide_code=True)
def _():
    checks.run(3, conformal_intervals=conformal_intervals)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def ensemble_uncertainty(probs):
    """stage 4: probs (M, n, k) -> {'mean': (n, k), 'total': H(mean), 'aleatoric': mean_m H(p_m), 'epistemic': total - aleatoric}, nats."""
    raise NotImplementedError("stage4: implement ensemble_uncertainty")


@app.function
def ood_auroc(scores_in, scores_out):
    """stage 4: ROC AUC of telling out-of-distribution (positives) from in-distribution inputs by score (higher = more OOD)."""
    raise NotImplementedError("stage4: implement ood_auroc")


@app.cell(hide_code=True)
def _():
    checks.run(4, ensemble_uncertainty=ensemble_uncertainty, ood_auroc=ood_auroc)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import fit_temperature, conformal_prediction_sets, conformal_intervals`
    gives you calibrated probabilities and distribution-free prediction sets (MAPIE-style
    libraries are unlocked). topic 06 turns to what a model reveals about its training data.
    """)
    return


if __name__ == "__main__":
    app.run()
