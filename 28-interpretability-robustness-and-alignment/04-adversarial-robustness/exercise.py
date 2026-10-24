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
    # adversarial robustness · build

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
    raise NotImplementedError("stage1: implement fgsm")


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
    raise NotImplementedError("stage2: implement pgd_attack")


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
    raise NotImplementedError("stage3: implement adversarial_training_step")


@app.function
def robust_accuracy(model, x, y, attack=None, *, batch_size=256):
    """stage 3: accuracy of model on attack(model, xb, yb) (clean if attack is None), batch by batch, in eval mode."""
    raise NotImplementedError("stage3: implement robust_accuracy")


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
    raise NotImplementedError("stage4: implement corrupt")


@app.function
def corruption_accuracy(model, x, y, kinds, severities, *, generator=None):
    """stage 4: {kind: ndarray of accuracies, one per severity}, eval mode (restored), corruptions drawn with generator."""
    raise NotImplementedError("stage4: implement corruption_accuracy")


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
