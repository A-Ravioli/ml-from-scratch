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
    # uncertainty & OOD · build (reference solution)

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
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if z.ndim != 2 or y.shape != (z.shape[0],):
        raise ValueError(f"logits must be (n, k) and labels (n,), got {z.shape} and {y.shape}")
    zy = z[np.arange(len(y)), y]

    def nll(b):
        return float(np.mean(logsumexp(b * z, axis=1) - b * zy))

    beta = 1.0
    for _ in range(max_iters):
        p = np.exp(beta * z - logsumexp(beta * z, axis=1, keepdims=True))
        ez = (p * z).sum(1)
        grad = np.mean(ez - zy)  # d NLL / d beta = E_p[z] - z_y
        hess = np.mean((p * z * z).sum(1) - ez ** 2)  # = Var_p[z] >= 0: the NLL is convex in beta
        if abs(grad) < 1e-10:
            break
        step = grad / hess if hess > 1e-12 else math.copysign(beta / 2, grad)
        f0, new = nll(beta), beta - step
        while new <= 0 or nll(new) > f0 + 1e-12:  # keep beta > 0 and never increase the loss
            step /= 2
            new = beta - step
            if abs(step) < 1e-15:
                new = beta
                break
        beta = min(new, 1e6)
    return float(1.0 / beta)


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
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha) - 1e-9)  # (n+1)(1-alpha) can land a hair above an integer in floats
    return float(np.inf) if k > n else float(s[k - 1])


@app.function
def conformal_prediction_sets(cal_probs, cal_labels, test_probs, alpha=0.1):
    """stage 2: bool (n_test, k): class c is in the set iff 1 - p_c <= the conformal quantile of the 1 - p_y scores."""
    cal_probs, test_probs = np.asarray(cal_probs), np.asarray(test_probs)
    cal_labels = np.asarray(cal_labels).astype(int)
    scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]
    q = conformal_quantile(scores, alpha)
    return (1.0 - test_probs) <= q


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
    q = conformal_quantile(np.abs(np.asarray(cal_y, dtype=np.float64) - np.asarray(cal_pred, dtype=np.float64)), alpha)
    test_pred = np.asarray(test_pred, dtype=np.float64)
    return test_pred - q, test_pred + q


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
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 3:
        raise ValueError(f"probs must be (members, n, classes), got shape {p.shape}")

    def entropy(q):
        return -np.sum(np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0)), 0.0), axis=-1)

    mean = p.mean(0)
    total = entropy(mean)
    aleatoric = entropy(p).mean(0)
    return {"mean": mean, "total": total, "aleatoric": aleatoric, "epistemic": total - aleatoric}


@app.function
def ood_auroc(scores_in, scores_out):
    """stage 4: ROC AUC of telling out-of-distribution (positives) from in-distribution inputs by score (higher = more OOD)."""
    s_in, s_out = np.asarray(scores_in, dtype=np.float64), np.asarray(scores_out, dtype=np.float64)
    y = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
    return float(roc_auc_score(y, np.concatenate([s_in, s_out])))


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
