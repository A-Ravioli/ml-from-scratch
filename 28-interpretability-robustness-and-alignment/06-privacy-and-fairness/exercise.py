import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import numpy as np
    import torch
    from torch.func import functional_call, grad, vmap

    from mlfs import checks, lesson
    from mlfs.metrics import roc_auc_score  # yours, from ch 07 · 03


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # privacy & fairness · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `per_sample_gradients`: one gradient per example, in one vectorised pass → `mlfs.safety` |
    | 2 | `dp_sgd_step`: clip each example's gradient, add calibrated noise → `mlfs.safety` |
    | 3 | `rdp_subsampled_gaussian`, `dp_sgd_epsilon`: the privacy accountant that turns noise into an epsilon |
    | 4 | `loss_threshold_attack`, `tpr_at_fpr`: a membership-inference attack, and how to score it → `mlfs.safety` |
    | 5 | `demographic_parity_difference`, `equalized_odds_difference`, `equal_opportunity_thresholds`: group fairness → `mlfs.safety` |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def per_sample_gradients(model, loss_fn, x, y):
    """stage 1: {name: (B, *param.shape)} gradient of loss_fn(model(x[i:i+1]), y[i:i+1]) per example, via torch.func."""
    raise NotImplementedError("stage1: implement per_sample_gradients")


@app.cell(hide_code=True)
def _():
    checks.run(1, per_sample_gradients=per_sample_gradients)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def dp_sgd_step(model, loss_fn, optimizer, x, y, *, max_grad_norm=1.0, noise_multiplier=1.0, generator=None):
    """stage 2: clip each example's gradient to max_grad_norm, sum, add N(0, (noise_multiplier max_grad_norm)^2), divide by B, step."""
    raise NotImplementedError("stage2: implement dp_sgd_step")


@app.cell(hide_code=True)
def _():
    checks.run(2, dp_sgd_step=dp_sgd_step, per_sample_gradients=per_sample_gradients)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def rdp_subsampled_gaussian(q, noise_multiplier, orders):
    """stage 3: RDP epsilon(alpha) of one Poisson-subsampled Gaussian step (sensitivity 1) at integer orders alpha >= 2."""
    raise NotImplementedError("stage3: implement rdp_subsampled_gaussian")


@app.function
def dp_sgd_epsilon(q, noise_multiplier, steps, delta, *, orders=None):
    """stage 3: epsilon after `steps` subsampled Gaussian steps: min over orders (None: 2..255) of steps * RDP + log(1/delta) / (alpha - 1)."""
    raise NotImplementedError("stage3: implement dp_sgd_epsilon")


@app.cell(hide_code=True)
def _():
    checks.run(3, rdp_subsampled_gaussian=rdp_subsampled_gaussian, dp_sgd_epsilon=dp_sgd_epsilon)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def loss_threshold_attack(member_losses, nonmember_losses):
    """stage 4: ROC AUC of flagging members (positives) by LOW loss (Yeom et al., 2018); ties count 1/2."""
    raise NotImplementedError("stage4: implement loss_threshold_attack")


@app.function
def tpr_at_fpr(member_losses, nonmember_losses, fpr=0.01):
    """stage 4: fraction of members with loss < tau, tau the loosest threshold flagging at most fpr of non-members."""
    raise NotImplementedError("stage4: implement tpr_at_fpr")


@app.cell(hide_code=True)
def _():
    checks.run(4, loss_threshold_attack=loss_threshold_attack, tpr_at_fpr=tpr_at_fpr)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def demographic_parity_difference(y_pred, groups):
    """stage 5: max - min over groups of P(y_pred = 1 | group)."""
    raise NotImplementedError("stage5: implement demographic_parity_difference")


@app.function
def equalized_odds_difference(y_true, y_pred, groups):
    """stage 5: max of the TPR gap and the FPR gap across groups (gap = max - min); ValueError if a group lacks a class."""
    raise NotImplementedError("stage5: implement equalized_odds_difference")


@app.function
def equal_opportunity_thresholds(scores, y_true, groups, target_tpr):
    """stage 5: {group: threshold}, the largest t with TPR(score >= t) >= target_tpr in that group (Hardt et al.)."""
    raise NotImplementedError("stage5: implement equal_opportunity_thresholds")


@app.cell(hide_code=True)
def _():
    checks.run(5, demographic_parity_difference=demographic_parity_difference,
               equalized_odds_difference=equalized_odds_difference,
               equal_opportunity_thresholds=equal_opportunity_thresholds)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import per_sample_gradients, dp_sgd_step, loss_threshold_attack` gives
    you private training and the attack that audits it (Opacus and Fairlearn are unlocked).
    topic 07 asks the hardest question of the chapter: what if the model does exactly what
    you asked for?
    """)
    return


if __name__ == "__main__":
    app.run()
