import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy

    import marimo as mo
    import numpy as np
    import torch
    from scipy import stats as sps

    from mlfs import checks, lesson
    from mlfs.ssl import linear_probe  # yours, from ch 25 · 01
    from mlfs.utils import as_rng  # yours, from ch 00


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # attribution & probing · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `saliency`: the gradient of a class logit with respect to the input → `mlfs.safety` |
    | 2 | `integrated_gradients`: the path integral that makes attributions add up → `mlfs.safety` |
    | 3 | `control_labels`, `probe_selectivity`: linear probes, and a control task that keeps them honest |
    | 4 | `rank_correlation`, `randomize_layers`, `cascading_randomization`: the sanity checks for saliency maps |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def saliency(model, x, target):
    """stage 1: |d model(x)[b, target_b] / d x|, same shape as x; eval mode (restored after); parameter grads untouched."""
    raise NotImplementedError("stage1: implement saliency")


@app.cell(hide_code=True)
def _():
    checks.run(1, saliency=saliency)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def integrated_gradients(model, x, target, *, baseline=None, n_steps=50):
    """stage 2: (x - baseline) * trapezoid average of the gradient at baseline + k/n_steps (x - baseline), k = 0..n_steps."""
    raise NotImplementedError("stage2: implement integrated_gradients")


@app.cell(hide_code=True)
def _():
    checks.run(2, integrated_gradients=integrated_gradients)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def control_labels(types, n_classes, *, rng=None):
    """stage 3: one uniformly random label in range(n_classes) per distinct type (Hewitt & Liang); (n,) int64."""
    raise NotImplementedError("stage3: implement control_labels")


@app.function
def probe_selectivity(train_feats, train_labels, test_feats, test_labels, train_types, test_types, *, rng=None):
    """stage 3: linear-probe accuracy on the task and on a control task (one label table for both splits); dict."""
    raise NotImplementedError("stage3: implement probe_selectivity")


@app.cell(hide_code=True)
def _():
    checks.run(3, control_labels=control_labels, probe_selectivity=probe_selectivity)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def rank_correlation(a, b):
    """stage 4: Spearman correlation of two equal-size arrays (flattened, ties get average ranks)."""
    raise NotImplementedError("stage4: implement rank_correlation")


@app.function
def randomize_layers(model, n, *, generator=None):
    """stage 4: deep copy with the LAST n parameterised modules re-initialised (reset_parameters, seeded by generator);
    the original and the global RNG state are left unchanged."""
    raise NotImplementedError("stage4: implement randomize_layers")


@app.function
def cascading_randomization(model, attribute, x, target, *, generator=None):
    """stage 4: Spearman(|attr| of model, |attr| after randomizing the top k layers) for k = 1..L; ndarray (L,)."""
    raise NotImplementedError("stage4: implement cascading_randomization")


@app.cell(hide_code=True)
def _():
    checks.run(4, rank_correlation=rank_correlation, randomize_layers=randomize_layers,
               cascading_randomization=cascading_randomization, saliency=saliency)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.safety import saliency, integrated_gradients` gives you both attribution
    methods (Captum's `Saliency` and `IntegratedGradients` are unlocked). topic 02 moves from
    the input to the inside of the model: caching and patching its activations.
    """)
    return


if __name__ == "__main__":
    app.run()
