import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from mlfs import checks, lesson
    from mlfs.linear import SoftmaxRegression  # yours, from ch 08 · topic 05


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # evaluating representations · build

    work through the stages in order. each stage shows its part of the lesson and checks your
    code live as you edit. the lesson's **spec** boxes are the contract; the internal
    structure is yours to design.

    | stage | you build |
    |---|---|
    | 1 | `extract_features`: run a frozen encoder over a loader, in eval mode → `mlfs.ssl` |
    | 2 | `linear_probe`: standardise, fit a softmax regression, report test accuracy → `mlfs.ssl` |
    | 3 | `knn_eval`: cosine-similarity weighted kNN, in memory-bounded chunks → `mlfs.ssl` |
    | 4 | `fine_tune`: a linear head on top of the encoder, trained end to end or frozen |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def extract_features(encoder, loader, *, device="cpu"):
    """stage 1: (feats float32 (n, d), labels (n,)) ndarrays from encoder over (x, y) batches; eval mode + no_grad, modes restored."""
    raise NotImplementedError("stage1: implement extract_features")


@app.cell(hide_code=True)
def _():
    checks.run(1, extract_features=extract_features)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def linear_probe(train_feats, train_labels, test_feats, test_labels, *, alpha=0.0001, max_iters=1000):
    """stage 2: standardise with TRAIN statistics, fit SoftmaxRegression(alpha, max_iters), return test accuracy (float)."""
    raise NotImplementedError("stage2: implement linear_probe")


@app.cell(hide_code=True)
def _():
    checks.run(2, linear_probe=linear_probe)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def knn_eval(train_feats, train_labels, test_feats, test_labels, *, k=20, temperature=0.07):
    """stage 3: cosine-similarity kNN, neighbour votes weighted exp(sim / temperature); test accuracy (float)."""
    raise NotImplementedError("stage3: implement knn_eval")


@app.cell(hide_code=True)
def _():
    checks.run(3, knn_eval=knn_eval)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def fine_tune(encoder, feature_dim, n_classes, loader, *, epochs=1, lr=1e-3, freeze=False, device="cpu", seed=None):
    """stage 4: nn.Sequential(encoder, Linear(feature_dim, n_classes)) trained with Adam(lr) for `epochs`; freeze keeps the encoder untouched."""
    raise NotImplementedError("stage4: implement fine_tune")


@app.cell(hide_code=True)
def _():
    checks.run(4, fine_tune=fine_tune)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, `extract_features`, `linear_probe` and `knn_eval` are part of
    `mlfs.ssl`: every later topic in this chapter (and chapter 28) evaluates its
    representations with them. open the lab: `marimo edit lab.py`.
    """)
    return


if __name__ == "__main__":
    app.run()
