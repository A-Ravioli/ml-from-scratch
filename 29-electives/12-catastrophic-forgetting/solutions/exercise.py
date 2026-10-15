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
    X = np.asarray(X)
    y = np.asarray(y)
    flat = X.reshape(len(X), -1).astype(np.float32)
    tasks = []
    for group in class_groups:
        group = tuple(int(c) for c in group)
        keep = np.isin(y, group)
        tasks.append({
            "X": torch.from_numpy(flat[keep].copy()),
            "y": torch.from_numpy(y[keep].astype(np.int64)),
            "classes": group,
        })
    return tasks


@app.function
def permuted_tasks(X, y, n_tasks, *, seed=0):
    """stage 1: task 0 is the data as is; task k > 0 applies a fixed pixel permutation that depends only on (seed, k)."""
    X = np.asarray(X)
    y = np.asarray(y)
    flat = X.reshape(len(X), -1).astype(np.float32)
    d = flat.shape[1]
    rng = np.random.default_rng(seed)
    perms = [np.arange(d)] + [rng.permutation(d) for _ in range(n_tasks - 1)]
    classes = tuple(int(c) for c in np.unique(y))
    yt = torch.from_numpy(y.astype(np.int64))
    return [{"X": torch.from_numpy(flat[:, p].copy()), "y": yt.clone(), "classes": classes} for p in perms]


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
    R = np.asarray(R, dtype=float)
    return float(R[-1].mean())


@app.function
def backward_transfer(R):
    """stage 2: mean over tasks j < T-1 of R[T-1, j] - R[j, j]; negative means forgetting."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        raise ValueError("backward transfer needs at least two tasks")
    return float(np.mean(R[-1, :-1] - np.diag(R)[:-1]))


@app.function
def forward_transfer(R, baseline):
    """stage 2: mean over tasks j > 0 of R[j-1, j] - baseline[j] (accuracy on a task before training on it)."""
    R = np.asarray(R, dtype=float)
    b = np.asarray(baseline, dtype=float)
    T = R.shape[0]
    if T < 2:
        raise ValueError("forward transfer needs at least two tasks")
    return float(np.mean([R[j - 1, j] - b[j] for j in range(1, T)]))


@app.function
def forgetting(R):
    """stage 2: mean over tasks j < T-1 of (best earlier accuracy max_{l < T-1} R[l, j]) - R[T-1, j]."""
    R = np.asarray(R, dtype=float)
    T = R.shape[0]
    if T < 2:
        raise ValueError("forgetting needs at least two tasks")
    return float(np.mean([R[:T - 1, j].max() - R[T - 1, j] for j in range(T - 1)]))


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
    if scenario == "class":
        return logits
    if scenario == "task":
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[list(classes)] = False
        return logits.masked_fill(mask, float("-inf"))
    raise ValueError(f"scenario must be 'class' or 'task', got {scenario!r}")


@app.function
def evaluate(model, task, *, scenario="class", batch_size=1024):
    """stage 3: accuracy (a float in [0, 1]) of the argmax of scenario_logits on the task; no gradients, mode restored."""
    was_training = model.training
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(task["y"]), batch_size):
            out = scenario_logits(model(task["X"][i:i + batch_size]), task["classes"], scenario)
            correct += int((out.argmax(1) == task["y"][i:i + batch_size]).sum())
    model.train(was_training)
    return correct / len(task["y"])


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
    if replay is not None and scenario != "class":
        raise ValueError("replay is only supported with scenario='class'")
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"optimizer must be 'sgd' or 'adam', got {optimizer!r}")
    model.train()
    g = torch.Generator().manual_seed(seed)
    n = len(task["y"])
    losses = []
    for _ in range(epochs):
        order = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x, y = task["X"][idx], task["y"][idx]
            if replay is not None:
                extra = replay()
                if extra is not None:
                    x = torch.cat([x, extra[0]])
                    y = torch.cat([y, extra[1]])
            loss = F.cross_entropy(scenario_logits(model(x), task["classes"], scenario), y)
            if penalty is not None:
                loss = loss + penalty(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if after_step is not None:
                after_step(model)
            losses.append(loss.item())
    return losses


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
    if len(train_tasks) != len(test_tasks):
        raise ValueError("need one test task per training task")
    T = len(train_tasks)
    baseline = np.array([evaluate(model, tt, scenario=scenario) for tt in test_tasks])
    R = np.zeros((T, T))
    for t, task in enumerate(train_tasks):
        if train_fn is None:
            train_task(model, task, scenario=scenario, **{"seed": t, **train_kwargs})
        else:
            train_fn(model, task, t)
        R[t] = [evaluate(model, tt, scenario=scenario) for tt in test_tasks]
    return R, baseline


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
