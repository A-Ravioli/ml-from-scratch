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
    # privacy & fairness · build (reference solution)

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
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def one_loss(p, xi, yi):
        out = functional_call(model, (p, buffers), (xi.unsqueeze(0),))
        return loss_fn(out, yi.unsqueeze(0))

    return vmap(grad(one_loss), in_dims=(None, 0, 0))(params, x, y)


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
    with torch.no_grad():
        mean_loss = float(loss_fn(model(x), y))
    grads = per_sample_gradients(model, loss_fn, x, y)
    B = x.shape[0]
    norms = torch.sqrt(sum(g.reshape(B, -1).pow(2).sum(1) for g in grads.values()))
    factor = max_grad_norm / torch.clamp(norms, min=max_grad_norm)  # min(1, C / ||g_i||)
    for name, p in model.named_parameters():
        if name not in grads:
            continue
        g = grads[name]
        summed = (g * factor.view(B, *([1] * (g.dim() - 1)))).sum(0)
        noise = torch.randn(p.shape, generator=generator).to(device=p.device, dtype=p.dtype)
        p.grad = (summed + noise_multiplier * max_grad_norm * noise) / B
    optimizer.step()
    return mean_loss


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
    out = []
    for a in orders:
        a = int(a)
        if a < 2:
            raise ValueError(f"orders must be integers >= 2, got {a}")
        if q == 0:
            out.append(0.0)
        elif noise_multiplier == 0:
            out.append(math.inf)
        elif q == 1:
            out.append(a / (2 * noise_multiplier ** 2))
        else:  # A_a = sum_k C(a, k) (1-q)^(a-k) q^k exp((k^2 - k) / (2 sigma^2)), in log space
            logs = [math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1) + (a - k) * math.log1p(-q)
                    + k * math.log(q) + (k * k - k) / (2 * noise_multiplier ** 2) for k in range(a + 1)]
            m = max(logs)
            out.append((m + math.log(sum(math.exp(v - m) for v in logs))) / (a - 1))
    return np.array(out)


@app.function
def dp_sgd_epsilon(q, noise_multiplier, steps, delta, *, orders=None):
    """stage 3: epsilon after `steps` subsampled Gaussian steps: min over orders (None: 2..255) of steps * RDP + log(1/delta) / (alpha - 1)."""
    orders = np.arange(2, 256) if orders is None else np.asarray(orders)
    rdp = steps * rdp_subsampled_gaussian(q, noise_multiplier, orders)
    return float(np.min(rdp + math.log(1 / delta) / (orders - 1)))


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
    m = np.asarray(member_losses, dtype=np.float64)
    n = np.asarray(nonmember_losses, dtype=np.float64)
    labels = np.concatenate([np.ones(len(m)), np.zeros(len(n))])
    return float(roc_auc_score(labels, -np.concatenate([m, n])))


@app.function
def tpr_at_fpr(member_losses, nonmember_losses, fpr=0.01):
    """stage 4: fraction of members with loss < tau, tau the loosest threshold flagging at most fpr of non-members."""
    m = np.asarray(member_losses, dtype=np.float64)
    s = np.sort(np.asarray(nonmember_losses, dtype=np.float64))
    k = math.floor(fpr * len(s) + 1e-9)  # at most k non-members may be flagged
    tau = np.inf if k >= len(s) else s[k]
    return float(np.mean(m < tau))


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
    y_pred, groups = np.asarray(y_pred), np.asarray(groups)
    rates = [np.mean(y_pred[groups == g] == 1) for g in np.unique(groups)]
    return float(max(rates) - min(rates))


@app.function
def equalized_odds_difference(y_true, y_pred, groups):
    """stage 5: max of the TPR gap and the FPR gap across groups (gap = max - min); ValueError if a group lacks a class."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    tprs, fprs = [], []
    for g in np.unique(groups):
        yt, yp = y_true[groups == g], y_pred[groups == g]
        if not (yt == 1).any() or not (yt == 0).any():
            raise ValueError(f"group {g!r} needs both positive and negative examples")
        tprs.append(np.mean(yp[yt == 1] == 1))
        fprs.append(np.mean(yp[yt == 0] == 1))
    return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))


@app.function
def equal_opportunity_thresholds(scores, y_true, groups, target_tpr):
    """stage 5: {group: threshold}, the largest t with TPR(score >= t) >= target_tpr in that group (Hardt et al.)."""
    scores, y_true, groups = np.asarray(scores, dtype=np.float64), np.asarray(y_true), np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        pos = np.sort(scores[(groups == g) & (y_true == 1)])[::-1]
        if len(pos) == 0:
            raise ValueError(f"group {g!r} has no positive examples")
        k = math.ceil(target_tpr * len(pos) - 1e-9)
        out[g.item() if hasattr(g, "item") else g] = float(pos[max(k, 1) - 1])
    return out


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
