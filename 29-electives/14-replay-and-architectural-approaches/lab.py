import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.stats import bootstrap_ci
    from mlfs.torchnn import MLP
    from mlfs.utils import image_grid

    # your own implementations: this topic's exercise, and topic 12's harness
    # (the reference solutions when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    _t12 = mo.notebook_dir().parent / "12-catastrophic-forgetting"
    cl = notebooks.load(notebooks.resolve(_t12, "exercise"))
    try:  # smoke-test topic 12's harness; fall back to the reference if it isn't finished
        _tiny = cl.split_tasks(np.eye(4, dtype=np.float32), np.array([0, 1, 2, 3]), [(0, 1), (2, 3)])
        cl.permuted_tasks(np.eye(4, dtype=np.float32), np.array([0, 1, 2, 3]), 2, seed=0)
        _R, _b = cl.run_sequence(torch.nn.Linear(4, 4), _tiny, _tiny, scenario="task", lr=0.1)
        cl.average_accuracy(_R), cl.backward_transfer(_R), cl.forward_transfer(_R, _b), cl.forgetting(_R)
        CL_FALLBACK = False
    except NotImplementedError:
        cl = notebooks.load(notebooks.resolve(_t12, "exercise", solution=True))
        CL_FALLBACK = True

    torch.set_num_threads(min(2, torch.get_num_threads()))  # never more than the environment allows (CI sets 1)
    GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    TRAIN = dict(optimizer="adam", lr=1e-3, batch_size=64, epochs=1)
    BUDGET = 50 * 784  # the project's memory budget: 39,200 floats, the size of 50 MNIST images


@app.function
def new_model(seed, hidden=200):
    """the block's network: 784 -> 200 -> 200 -> 10, ReLU."""
    torch.manual_seed(seed)
    return MLP(784, [hidden, hidden], 10)


@app.function
def replay_run(memories, seed, train_tasks, test_tasks):
    """class-incremental run where each step appends k replayed examples from every (memory, k) in `memories`.

    memories are filled with each task's training data after the task (buffers: add, generators: fit).
    returns (R, floats stored at the end).
    """
    def replay():
        xs, ys = zip(*(mem.sample(k) for mem, k in memories))
        return torch.cat(xs), torch.cat(ys)

    def train_fn(model, task, t):
        cl.train_task(model, task, seed=t, replay=replay if (t > 0 and memories) else None, **TRAIN)
        for mem, _ in memories:
            if hasattr(mem, "fit"):
                mem.fit(task["X"], task["y"])
            else:
                mem.add(task["X"], task["y"])

    R, _ = cl.run_sequence(new_model(seed), train_tasks, test_tasks, train_fn=train_fn)
    return R, sum(mem.n_floats() for mem, _ in memories)


@app.function
def summary(values):
    """'mean ± sd' over seeds."""
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}" if len(values) > 1 else f"{values[0]:.3f}"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # replay & architectural approaches · lab

    every experiment here uses **your** buffers, generator and gated network, plugged into
    **your** topic-12 harness (`train_task(..., replay=...)`). same network and training as
    before: 784–200–200–10, Adam at $10^{-3}$, batch 64, one epoch per task; each replayed batch
    adds 64 remembered examples to the 64 new ones. unless stated otherwise the scenario is the
    hard one, **class-incremental** split-MNIST, where fine-tuning ends at ACC ≈ 0.20 and EWC
    and SI didn't help. the lab ends with the block's project. the whole lab takes about two
    minutes on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    (mo.callout(mo.md("using the **reference** topic-12 harness: finish topic 12 to use yours."), kind="warn")
     if CL_FALLBACK else None)
    return


@app.cell
def _():
    _d = datasets.mnist()
    split_train = cl.split_tasks(_d["X_train"] / 255.0, _d["y_train"], GROUPS)
    split_test = cl.split_tasks(_d["X_test"] / 255.0, _d["y_test"], GROUPS)
    return split_test, split_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much memory does replay need?

    a reservoir buffer of $M$ images, for $M$ from 10 (one image per class, on average) to 1,000
    (2% of the training set), three seeds each.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "class-incremental ACC with a reservoir of 10 images? of 200? of 1,000? (fine-tuning: 0.20; joint "
        "training on everything: 0.96)"
    )
    p1
    return (p1,)


@app.cell
def _(p1, split_test, split_train):
    ui.gate(p1)
    sizes = [10, 50, 200, 1000]
    size_acc = {}
    for _M in sizes:
        size_acc[_M] = [cl.average_accuracy(replay_run([(ex.ReservoirBuffer(_M, seed=_s), 64)], _s, split_train,
                                                       split_test)[0]) for _s in range(3)]
    ft_acc = [cl.average_accuracy(replay_run([], _s, split_train, split_test)[0]) for _s in range(3)]
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.errorbar(sizes, [np.mean(size_acc[m]) for m in sizes], yerr=[np.std(size_acc[m], ddof=1) for m in sizes],
                 fmt="o-", capsize=3, label="reservoir replay")
    _ax.axhline(np.mean(ft_acc), color="gray", ls="--", label="fine-tuning")
    _ax.set(xscale="log", xlabel="images stored (M)", ylabel="class-incremental ACC", ylim=(0, 1),
            title="split-MNIST, mean ± sd over 3 seeds")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| memory | floats | ACC |\n|---|---|---|\n| none (fine-tuning) | 0 | " + summary(ft_acc)
                           + " |\n" + "\n".join(f"| reservoir, M = {m} | {m * 784:,} | {summary(size_acc[m])} |"
                                                for m in sizes))])
    return ft_acc, size_acc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** ten images move ACC by about ten points, and the first few hundred images do
    most of the work. replaying an image costs the network almost nothing to fit, so what is the replay
    actually doing to the output layer? how far is $M = 1{,}000$ from joint training, and why
    doesn't more replay close the gap for free?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: a model of the data instead of the data

    replace the buffer with your `GaussianGenerator`: one diagonal Gaussian per class, a mean and
    a standard deviation per pixel, 10 classes × 2 × 784 = 15,680 floats. that's the memory of 20
    images. below: what it generates, and how it compares with a 20-image buffer (same memory) and
    a 200-image buffer (10× the memory).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "the Gaussian's samples look like static with a faint digit in it. will replaying them beat a buffer of 20 "
        "real images? of 200?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, size_acc, split_test, split_train):
    ui.gate(p2)
    _show = ex.GaussianGenerator(seed=0)
    for _t in split_train:
        _show.fit(_t["X"], _t["y"])
    _xs, _ys = _show.sample(16)
    _order = torch.argsort(_ys)
    _fig, _ax = plt.subplots(figsize=(6, 1.8))
    image_grid(_xs[_order].clamp(0, 1).reshape(-1, 28, 28).numpy(), nrow=8, ax=_ax,
               titles=[str(int(v)) for v in _ys[_order]])
    _fig.tight_layout()
    gauss_acc = [cl.average_accuracy(replay_run([(ex.GaussianGenerator(seed=_s), 64)], _s, split_train, split_test)[0])
                 for _s in range(3)]
    _res20 = [cl.average_accuracy(replay_run([(ex.ReservoirBuffer(20, seed=_s), 64)], _s, split_train, split_test)[0])
              for _s in range(3)]
    mo.vstack([_fig, mo.md(
        "| memory | floats | ACC |\n|---|---|---|\n"
        f"| reservoir, 20 images | 15,680 | {summary(_res20)} |\n"
        f"| Gaussian per class | 15,680 | {summary(gauss_acc)} |\n"
        f"| reservoir, 200 images | 156,800 | {summary(size_acc[200])} |")])
    return (gauss_acc,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the generated "digits" are poor, so why are they so effective? what does
    class-incremental fine-tuning get wrong (look back at topic 12's accuracy matrix), and what
    property of a replayed example is enough to fix it? where would this cheap generator break
    down: on CIFAR-10, or with a convolutional network?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: isolating parameters instead of remembering data

    `GatedMLP` gives every task a fixed random subset of each hidden layer (a fraction `keep`
    of the units) and trains with the topic-12 task-incremental loss. it stores no data at all,
    but it needs to be told the task at test time, so it only applies to the task-incremental
    scenario. `keep = 1` is plain fine-tuning (every task uses every unit).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "task-incremental fine-tuning ends near ACC 0.92. what will gating with keep = 0.5 and keep = 0.2 give, "
        "and which is better?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, split_test, split_train):
    ui.gate(p3)
    gate_acc = {}
    for _keep in (1.0, 0.5, 0.2):
        gate_acc[_keep] = []
        for _s in range(3):
            torch.manual_seed(_s)
            _m = ex.GatedMLP(784, [200, 200], 10, len(split_train), keep=_keep, seed=_s)
            _R = np.zeros((len(split_train), len(split_train)))
            for _t, _task in enumerate(split_train):
                _m.set_task(_t)
                cl.train_task(_m, _task, scenario="task", seed=_t, **TRAIN)
                for _j, _tt in enumerate(split_test):
                    _m.set_task(_j)
                    _R[_t, _j] = cl.evaluate(_m, _tt, scenario="task")
            gate_acc[_keep].append((cl.average_accuracy(_R), cl.backward_transfer(_R)))
    mo.md("| keep | hidden units per task | task-incremental ACC | BWT |\n|---|---|---|---|\n" + "\n".join(
        f"| {k:g} | {int(round(k * 200))} of 200 | {summary([a for a, _ in v])} | {summary([b for _, b in v])} |"
        for k, v in gate_acc.items()) + "\n\nmean ± sd over seeds 0–2.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with `keep = 0.2` two tasks share on average 20% of each other's units, yet
    forgetting nearly vanishes. what is still shared between tasks, and why doesn't it matter
    much here? what's the cost of gating as the number of tasks grows, and why can't it work
    class-incrementally?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## project: beat fine-tuning on split-MNIST with a fixed memory budget

    the rules: class-incremental split-MNIST, the network, optimizer and single epoch above, and
    at most **39,200 floats** of memory (`BUDGET`, the size of 50 images) counted by
    `n_floats()`. anything that stores more is disqualified: a 784–200–200–10 network has
    199,210 weights, so EWC's anchor plus Fisher (398,420 floats even online) is 10× over budget.
    you report class-incremental ACC over five seeds, with a paired bootstrap confidence
    interval for the improvement over fine-tuning.

    edit `my_strategy` below: it returns a list of `(memory, k)` pairs, each memory with
    `add` or `fit`, `sample(k)` and `n_floats()` (`k` examples from it are appended to every
    batch). it starts as the obvious baseline, a 50-image reservoir that uses the whole budget.
    beat it: with what experiments 1 and 2 showed, by a different replay ratio, by splitting the
    budget between kinds of memory, or with a memory you design yourself (class means? a low-rank
    Gaussian? images stored as uint8?). write up what you tried, including what didn't work, in
    the cell at the end.

    four caveats belong in that write-up. the **unit decides the winner**: MNIST pixels are
    uint8, so a budget of 156,800 *bytes* (the same 39,200 float32s) holds 200 uint8 images, which
    score a little higher than this lab's float32 generator (0.81 against 0.78–0.79); counted in bytes, the generator is about 2.5× cheaper than
    the buffer it matches, not 10×. `n_floats()` is **self-reported**, so the check is only as
    honest as your memory class. replay **doubles the compute** per step, which the budget doesn't
    count. and a percentile bootstrap over **5 paired seeds** under-covers: treat the interval as
    optimistic.
    """)
    return


@app.function
def my_strategy(seed):
    """the project's strategy: a list of (memory, examples replayed per step) pairs."""
    return [(ex.ReservoirBuffer(50, seed=seed), 64)]


@app.cell
def _():
    p4 = ui.predict(
        "what class-incremental ACC will your strategy reach over five seeds, and how wide will the 95% interval "
        "for its improvement over fine-tuning be?"
    )
    p4
    return (p4,)


@app.cell
def _(ft_acc, p4, split_test, split_train):
    ui.gate(p4)
    _mine, _floats = [], []
    _ft = list(ft_acc)  # fine-tuning seeds 0-2 from experiment 1; add seeds 3-4
    _ft += [cl.average_accuracy(replay_run([], _s, split_train, split_test)[0]) for _s in range(3, 5)]
    for _s in range(5):
        _R, _fl = replay_run(my_strategy(_s), _s, split_train, split_test)
        _mine.append(cl.average_accuracy(_R))
        _floats.append(_fl)
    _mine, _ft = np.array(_mine), np.array(_ft)
    _within = max(_floats) <= BUDGET
    _lo, _hi = bootstrap_ci((_mine, _ft), lambda a, b: float(np.mean(a - b)), n_boot=5000,
                            rng=np.random.default_rng(0))
    mo.vstack([
        mo.callout(mo.md(f"memory used: **{max(_floats):,} floats** of {BUDGET:,}: "
                         + ("within budget." if _within else "**over budget: disqualified.**")),
                   kind="success" if _within else "danger"),
        mo.md("| strategy | ACC per seed | mean ± sd |\n|---|---|---|\n"
              f"| fine-tuning | {', '.join(f'{v:.3f}' for v in _ft)} | {summary(_ft)} |\n"
              f"| yours | {', '.join(f'{v:.3f}' for v in _mine)} | {summary(_mine)} |\n\n"
              f"improvement: **{np.mean(_mine - _ft):+.3f}**, 95% paired bootstrap CI "
              f"[{_lo:+.3f}, {_hi:+.3f}] over seeds 0–4."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain (the project write-up):** what did you try, what did each attempt score, and
    why do you think the winner won? with five seeds, how small an improvement could you
    reliably detect? which parts of your result would carry over to a harder benchmark
    (split CIFAR-10, more tasks, a CNN), and which wouldn't? the elective portfolio capstone
    asks for this write-up.
    """)
    return


if __name__ == "__main__":
    app.run()
