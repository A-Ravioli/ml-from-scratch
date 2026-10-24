import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, lesson


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # adversarial robustness · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `fgsm`: the one-step attack → `mlfs.safety` |
    | 2 | `pgd_attack`: projected gradient descent in an $\ell_\infty$ or $\ell_2$ ball → `mlfs.safety` |
    | 3 | `adversarial_training_step`, `robust_accuracy`: Madry-style adversarial training and its evaluation |
    | 4 | `corrupt`, `corruption_accuracy`: natural distribution shift, for comparison |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def fgsm(model, x, y, epsilon, *, loss_fn=None, clamp=(0.0, 1.0)):
    """stage 1: clamp(x + epsilon * sign(grad_x loss_fn(model(x), y))); eval mode (restored); detached."""
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():  # works inside torch.no_grad() too
            xg = x.detach().clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(loss_fn(model(xg), y), xg)
    finally:
        model.train(was_training)
    x_adv = x.detach() + epsilon * grad.sign()
    if clamp is not None:
        x_adv = x_adv.clamp(*clamp)
    return x_adv.detach()


@app.cell(hide_code=True)
def _():
    checks.run(1, fgsm=fgsm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def pgd_attack(model, x, y, epsilon, *, alpha=None, n_steps=10, norm="linf", random_start=True, loss_fn=None,
               clamp=(0.0, 1.0), generator=None):
    """stage 2: n_steps of ascent (sign step for linf, normalised gradient for l2), each projected onto the epsilon ball and clamp."""
    if norm not in ("linf", "l2"):
        raise ValueError(f"norm must be 'linf' or 'l2', got {norm!r}")
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    alpha = 2.5 * epsilon / max(n_steps, 1) if alpha is None else alpha
    x0 = x.detach()
    dims = tuple(range(1, x0.dim()))

    def project(x_adv):
        delta = x_adv - x0
        if norm == "linf":
            delta = delta.clamp(-epsilon, epsilon)
        else:
            n = delta.flatten(1).norm(dim=1).view(-1, *([1] * len(dims)))
            delta = delta * torch.clamp(epsilon / n.clamp_min(1e-12), max=1.0)
        out = x0 + delta
        return out.clamp(*clamp) if clamp is not None else out

    if random_start:
        if norm == "linf":
            delta = (torch.rand(x0.shape, generator=generator) * 2 - 1) * epsilon
        else:
            d = torch.randn(x0.shape, generator=generator)
            d = d / d.flatten(1).norm(dim=1).view(-1, *([1] * len(dims)))
            r = torch.rand(x0.shape[0], generator=generator) ** (1.0 / x0[0].numel())
            delta = d * (epsilon * r).view(-1, *([1] * len(dims)))
        x_adv = project(x0 + delta.to(device=x0.device, dtype=x0.dtype))
    else:
        x_adv = project(x0.clone())
    was_training = model.training
    model.eval()
    try:
        for _ in range(n_steps):
            with torch.enable_grad():
                x_adv = x_adv.detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(loss_fn(model(x_adv), y), x_adv)
            if norm == "linf":
                step = grad.sign()
            else:
                step = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1, *([1] * len(dims)))
            x_adv = project(x_adv.detach() + alpha * step)
    finally:
        model.train(was_training)
    return x_adv.detach()


@app.cell(hide_code=True)
def _():
    checks.run(2, pgd_attack=pgd_attack, fgsm=fgsm)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def adversarial_training_step(model, optimizer, x, y, *, epsilon, alpha=None, n_steps=7, loss_fn=None,
                              generator=None):
    """stage 3: linf PGD examples against the current model (eval mode), then one optimizer step on their loss in train mode."""
    loss_fn = F.cross_entropy if loss_fn is None else loss_fn
    x_adv = pgd_attack(model, x, y, epsilon, alpha=alpha, n_steps=n_steps, loss_fn=loss_fn, generator=generator)
    model.train()
    loss = loss_fn(model(x_adv), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@app.function
def robust_accuracy(model, x, y, attack=None, *, batch_size=256):
    """stage 3: accuracy of model on attack(model, xb, yb) (clean if attack is None), batch by batch, in eval mode."""
    was_training = model.training
    correct = 0
    try:
        for i in range(0, len(x), batch_size):
            xb, yb = x[i:i + batch_size], y[i:i + batch_size]
            if attack is not None:
                xb = attack(model, xb, yb)
            model.eval()
            with torch.no_grad():
                correct += (model(xb).argmax(1) == yb).sum().item()
    finally:
        model.train(was_training)
    return correct / len(x)


@app.cell(hide_code=True)
def _():
    checks.run(3, adversarial_training_step=adversarial_training_step, robust_accuracy=robust_accuracy,
               pgd_attack=pgd_attack)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def corrupt(x, kind, severity, *, generator=None):
    """stage 4: a corrupted copy of images x (B, C, H, W) in [0, 1]; kind in gaussian_noise/contrast/brightness/shift, severity 0..5."""
    if not 0 <= severity <= 5:
        raise ValueError(f"severity must be in 0..5, got {severity}")
    x = x.detach()
    if kind == "gaussian_noise":
        noise = torch.randn(x.shape, generator=generator).to(device=x.device, dtype=x.dtype)
        out = x + 0.08 * severity * noise
    elif kind == "contrast":
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        out = mean + (x - mean) * (1 - 0.15 * severity)
    elif kind == "brightness":
        out = x + 0.1 * severity
    elif kind == "shift":
        s = int(severity)
        out = torch.zeros_like(x)
        out[..., s:] = x[..., :x.shape[-1] - s]
    else:
        raise ValueError(f"unknown corruption {kind!r}")
    return out.clamp(0.0, 1.0)


@app.function
def corruption_accuracy(model, x, y, kinds, severities, *, generator=None):
    """stage 4: {kind: ndarray of accuracies, one per severity}, eval mode (restored), corruptions drawn with generator."""
    was_training = model.training
    model.eval()
    try:
        out = {}
        with torch.no_grad():
            for k in kinds:
                out[k] = np.array([(model(corrupt(x, k, s, generator=generator)).argmax(1) == y).float().mean().item()
                                   for s in severities])
        return out
    finally:
        model.train(was_training)


@app.cell(hide_code=True)
def _():
    checks.run(4, corrupt=corrupt, corruption_accuracy=corruption_accuracy)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import fgsm, pgd_attack` gives you both attacks (torchattacks-style
    libraries are unlocked). topic 05 asks a quieter question: when the model is wrong, does
    it at least know it might be?
    """)
    return


if __name__ == "__main__":
    app.run()
