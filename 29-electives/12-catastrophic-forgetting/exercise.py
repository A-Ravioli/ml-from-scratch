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
    # catastrophic forgetting · build

    spec-driven (L4): the **spec** boxes in the lesson define every function, the checks enforce
    them, and the internals are yours. you're building a small continual-learning benchmark
    harness. topics 13 and 14 plug their strategies (EWC, SI, replay) into it, so keep to the spec.

    a **task** is a dict `{"X": float32 tensor (n, d), "y": int64 tensor (n,), "classes": tuple of ints}`.
    a **model** is any `nn.Module` mapping `(B, d)` inputs to `(B, n_classes)` logits.

    | stage | you build |
    |---|---|
    | 1 | `split_tasks`, `permuted_tasks`: task sequences from one labelled dataset |
    | 2 | `average_accuracy`, `backward_transfer`, `forward_transfer`, `forgetting`: metrics on the accuracy matrix |
    | 3 | `scenario_logits`, `evaluate`: task-incremental vs class-incremental evaluation |
    | 4 | `train_task`: fine-tune on one task, with hooks for penalties, per-step callbacks and replay |
    | 5 | `run_sequence`: train through a task sequence and fill in the accuracy matrix |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.function
def split_tasks(X, y, class_groups):
    """stage 1: one task per group of classes, keeping only that group's examples (in their original order)."""
    raise NotImplementedError("stage1: implement split_tasks")


@app.function
def permuted_tasks(X, y, n_tasks, *, seed=0):
    """stage 1: task 0 is the data as is; task k > 0 applies a fixed pixel permutation that depends only on (seed, k)."""
    raise NotImplementedError("stage1: implement permuted_tasks")


@app.cell(hide_code=True)
def _():
    checks.run(1, split_tasks=split_tasks, permuted_tasks=permuted_tasks)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def average_accuracy(R):
    """stage 2: mean accuracy over all tasks after training on the last one (mean of R's last row)."""
    raise NotImplementedError("stage2: implement average_accuracy")


@app.function
def backward_transfer(R):
    """stage 2: mean over tasks j < T-1 of R[T-1, j] - R[j, j]; negative means forgetting."""
    raise NotImplementedError("stage2: implement backward_transfer")


@app.function
def forward_transfer(R, baseline):
    """stage 2: mean over tasks j > 0 of R[j-1, j] - baseline[j] (accuracy on a task before training on it)."""
    raise NotImplementedError("stage2: implement forward_transfer")


@app.function
def forgetting(R):
    """stage 2: mean over tasks j < T-1 of (best earlier accuracy max_{l < T-1} R[l, j]) - R[T-1, j]."""
    raise NotImplementedError("stage2: implement forgetting")


@app.cell(hide_code=True)
def _():
    checks.run(2, average_accuracy=average_accuracy, backward_transfer=backward_transfer,
               forward_transfer=forward_transfer, forgetting=forgetting)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def scenario_logits(logits, classes, scenario):
    """stage 3: "class": logits unchanged; "task": every column not in `classes` set to -inf."""
    raise NotImplementedError("stage3: implement scenario_logits")


@app.function
def evaluate(model, task, *, scenario="class", batch_size=1024):
    """stage 3: accuracy (a float in [0, 1]) of the argmax of scenario_logits on the task; no gradients, mode restored."""
    raise NotImplementedError("stage3: implement evaluate")


@app.cell(hide_code=True)
def _():
    checks.run(3, scenario_logits=scenario_logits, evaluate=evaluate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.function
def train_task(model, task, *, scenario="class", epochs=1, lr=0.05, batch_size=64, seed=0,
               optimizer="sgd", penalty=None, after_step=None, replay=None):
    """stage 4: minibatch SGD (or Adam) on the task's cross-entropy (+ penalty(model)); returns the per-step losses."""
    raise NotImplementedError("stage4: implement train_task")


@app.cell(hide_code=True)
def _():
    checks.run(4, train_task=train_task, evaluate=evaluate)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.function
def run_sequence(model, train_tasks, test_tasks, *, train_fn=None, scenario="class", **train_kwargs):
    """stage 5: train on each task in turn; returns (R, baseline): R[i, j] = accuracy on test task j after task i."""
    raise NotImplementedError("stage5: implement run_sequence")


@app.cell(hide_code=True)
def _():
    checks.run(5, run_sequence=run_sequence, train_task=train_task, evaluate=evaluate,
               split_tasks=split_tasks)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. topics 13 and 14 load this
    notebook (your copy) for their labs, so the harness you just built is the one that will
    measure EWC, SI and replay.
    """)
    return


if __name__ == "__main__":
    app.run()
