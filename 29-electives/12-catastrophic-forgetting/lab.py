import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(min(2, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    TRAIN = dict(optimizer="adam", lr=1e-3, batch_size=64, epochs=1)


@app.function
def new_model(seed, hidden=200):
    """the network used throughout this block: 784 -> 200 -> 200 -> 10, ReLU."""
    torch.manual_seed(seed)
    return MLP(784, [hidden, hidden], 10)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # catastrophic forgetting · lab

    every experiment here runs on **your** harness: `split_tasks`, `permuted_tasks`,
    `train_task`, `evaluate`, `run_sequence` and the metrics. the network is a 784–200–200–10
    MLP trained with Adam (learning rate $10^{-3}$, batch 64, one epoch per task). for each
    experiment: write down a prediction, lock it in, run it, then explain what happened. compare
    with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute on a laptop CPU.
    """)
    return


@app.cell
def _():
    _d = datasets.mnist()
    Xtr, ytr = _d["X_train"] / 255.0, _d["y_train"]
    Xte, yte = _d["X_test"] / 255.0, _d["y_test"]
    split_train = ex.split_tasks(Xtr, ytr, GROUPS)
    split_test = ex.split_tasks(Xte, yte, GROUPS)
    mo.md(f"split-MNIST: {len(split_train)} tasks, "
          f"{', '.join(str(len(t['y'])) for t in split_train)} training examples per task.")
    return Xte, Xtr, split_test, split_train, yte, ytr


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how fast is a task forgotten?

    train on task 0 (digits 0 vs 1) until it's solved, then start training on task 1 (2 vs 3),
    and measure task 0's test accuracy after every optimizer step. do it twice: in the
    **class-incremental** scenario (one 10-way softmax, no task identity) and the
    **task-incremental** one (the softmax restricted to the current task's two classes, in
    training and at test time).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "in each scenario, how many steps of training on task 1 (batches of 64) does it take until task 0's "
        "accuracy falls below 50%? (one epoch of task 1 is about 190 steps)"
    )
    p1
    return (p1,)


@app.cell
def _(p1, split_test, split_train):
    ui.gate(p1)
    curves = {}
    for _sc in ("class", "task"):
        _m = new_model(0)
        ex.train_task(_m, split_train[0], scenario=_sc, seed=0, **TRAIN)
        _acc = [ex.evaluate(_m, split_test[0], scenario=_sc)]

        def _track(model, _sc=_sc, _acc=_acc):
            _acc.append(ex.evaluate(model, split_test[0], scenario=_sc))

        ex.train_task(_m, split_train[1], scenario=_sc, seed=1, after_step=_track, **TRAIN)
        curves[_sc] = np.array(_acc)
    _fig, _ax = plt.subplots(figsize=(7, 3.4))
    for _sc, _c in curves.items():
        _ax.plot(np.arange(len(_c)), _c, label=f"{_sc}-incremental")
    _ax.set(xlabel="optimizer steps on task 1", ylabel="task 0 test accuracy", ylim=(-0.03, 1.03),
            title="task 0 (0 vs 1) while training on task 1 (2 vs 3)")
    _ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    _ax.legend()
    _fig.tight_layout()
    _below = {k: int(np.argmax(c < 0.5)) if (c < 0.5).any() else None for k, c in curves.items()}
    mo.vstack([_fig, mo.md(
        "| scenario | before task 1 | after 10 steps | after 50 steps | end of task 1 | first step below 50% |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(
            f"| {k} | {c[0]:.3f} | {c[min(10, len(c) - 1)]:.3f} | {c[min(50, len(c) - 1)]:.3f} | {c[-1]:.3f} | "
            f"{_below[k] if _below[k] is not None else 'never'} |" for k, c in curves.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the class-incremental run, nothing happens for a few steps and then task 0
    collapses within a few dozen. what is the network learning in those steps that destroys
    task 0, and why does it need so few examples? in the task-incremental run the *same weights*
    are being changed just as much: why doesn't task 0 suffer?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the whole sequence, two scenarios

    now all five split-MNIST tasks, in order, for three seeds. you'll see the accuracy matrix
    $R$ for seed 0 and the four metrics for every seed. as a reference point, **joint
    training** (one epoch over all five tasks' data, shuffled together) shows what the network can
    do when nothing is sequential.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "after all five tasks, what average accuracy (ACC) do you expect in the class-incremental scenario? "
        "in the task-incremental one? (joint training gets about 96%)"
    )
    p2
    return (p2,)


@app.cell
def _(p2, split_test, split_train):
    ui.gate(p2)
    results = {}
    for _sc in ("class", "task"):
        results[_sc] = []
        for _seed in range(3):
            _R, _b = ex.run_sequence(new_model(_seed), split_train, split_test, scenario=_sc, **TRAIN)
            results[_sc].append((_R, _b))
    _joint_task = {"X": torch.cat([t["X"] for t in split_train]), "y": torch.cat([t["y"] for t in split_train]),
                   "classes": tuple(range(10))}
    _jm = new_model(0)
    ex.train_task(_jm, _joint_task, seed=0, **TRAIN)
    joint_acc = float(np.mean([ex.evaluate(_jm, t) for t in split_test]))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for _ax, (_sc, _runs) in zip(_axes, results.items()):
        _R = _runs[0][0]
        _ax.imshow(_R, vmin=0, vmax=1, cmap="viridis")
        for _i in range(5):
            for _j in range(5):
                _ax.text(_j, _i, f"{_R[_i, _j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if _R[_i, _j] < 0.6 else "black")
        _ax.set(title=f"{_sc}-incremental, seed 0", xlabel="evaluated on task j", ylabel="after training task i",
                xticks=range(5), yticks=range(5))
    _fig.tight_layout()

    def _row(sc):
        _m = np.array([[ex.average_accuracy(R), ex.backward_transfer(R), ex.forward_transfer(R, b), ex.forgetting(R)]
                       for R, b in results[sc]])
        return f"| {sc} | " + " | ".join(f"{mu:.3f} ± {sd:.3f}" for mu, sd in zip(_m.mean(0), _m.std(0, ddof=1))) + " |"

    mo.vstack([_fig, mo.md(
        "| scenario | ACC | BWT | FWT | forgetting |\n|---|---|---|---|---|\n" + _row("class") + "\n" + _row("task")
        + f"\n\nmean ± standard deviation over seeds 0–2. joint training (seed 0): ACC = {joint_acc:.3f}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the class-incremental ACC is suspiciously close to a simple fraction. which
    one, and what does that say about what the network predicts at the end? the task-incremental
    network learned exactly the same tasks in the same order: list the two things that make its
    job easier at test time. why is FWT close to zero in both scenarios?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: gradual forgetting, and the learning rate

    permuted MNIST is the other classic benchmark: ten tasks, each the full 10-class problem
    with the pixels shuffled by a different fixed permutation (5,000 training and 1,000 test
    images per task, to keep it quick). every task uses all ten outputs, so task identity
    doesn't change the output layer. pick a learning rate below and watch task 0 through the
    sequence. the experiment runs all three rates for comparison.
    """)
    return


@app.cell
def _():
    lr_pick = mo.ui.dropdown(options={"1e-4": 1e-4, "1e-3": 1e-3, "1e-2": 1e-2}, value="1e-3",
                             label="learning rate to highlight")
    p3 = ui.predict(
        "with Adam at learning rate 1e-2 instead of 1e-3, does each new task get learned better or worse? does "
        "task 0 end up better or worse?"
    )
    mo.vstack([p3, lr_pick])
    return lr_pick, p3


@app.cell
def _(Xte, Xtr, p3, yte, ytr):
    ui.gate(p3)
    _ptr = ex.permuted_tasks(Xtr[:5000], ytr[:5000], 10, seed=0)
    _pte = ex.permuted_tasks(Xte[:1000], yte[:1000], 10, seed=0)
    perm_runs = {}
    for _lr in (1e-4, 1e-3, 1e-2):
        perm_runs[_lr] = ex.run_sequence(new_model(0), _ptr, _pte, **dict(TRAIN, lr=_lr))
    return (perm_runs,)


@app.cell
def _(lr_pick, perm_runs):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for _lr, (_R, _b) in perm_runs.items():
        _hi = _lr == lr_pick.value
        _a1.plot(range(10), _R[:, 0], "o-", lw=2.5 if _hi else 1, alpha=1 if _hi else 0.45, label=f"lr {_lr:g}")
        _a2.plot(range(10), np.diag(_R), "o-", lw=2.5 if _hi else 1, alpha=1 if _hi else 0.45, label=f"lr {_lr:g}")
    _a1.set(xlabel="after training task i", ylabel="accuracy on task 0", ylim=(0, 1), title="the first task")
    _a2.set(xlabel="task i", ylabel="accuracy on task i right after it", ylim=(0, 1), title="each new task")
    _a1.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(
        "| learning rate | ACC | BWT | mean new-task accuracy | task 0 at the end |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {lr:g} | {ex.average_accuracy(R):.3f} | {ex.backward_transfer(R):.3f} | {np.mean(np.diag(R)):.3f} | "
            f"{R[-1, 0]:.3f} |" for lr, (R, b) in perm_runs.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is forgetting gradual here and abrupt in experiment 1? going from $10^{-3}$
    to $10^{-2}$, what happens to how well each new task is learned, and what happens to
    forgetting? what does that suggest about *how far* the weights travel versus *how far they
    need to*? (this is the stability–plasticity dilemma in its simplest form; topics 13 and 14
    attack it directly.)
    """)
    return


if __name__ == "__main__":
    app.run()
