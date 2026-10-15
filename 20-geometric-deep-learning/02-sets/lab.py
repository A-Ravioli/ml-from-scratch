import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import notebooks, ui
    from mlfs.torchnn import MLP  # yours, from ch 15 · topic 07

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    torch.set_num_threads(4)
    TASKS = {
        "sum": lambda x, m: (x * m).sum(1),
        "max": lambda x, m: x.masked_fill(~m, -1.0).amax(1),
        "count > 0.5": lambda x, m: ((x > 0.5) & m).float().sum(1),
    }


@app.function
def make_sets(n, sizes, task, seed):
    """n sets of numbers uniform on [0, 1], sizes uniform in `sizes` (inclusive range), padded; returns x (n, N, 1), mask, y (n, 1)."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = sizes
    k = torch.randint(lo, hi + 1, (n,), generator=g)
    x = torch.rand(n, hi, generator=g)
    mask = torch.arange(hi)[None] < k[:, None]
    return x[..., None], mask, TASKS[task](x, mask)[:, None]


@app.class_definition
class AttentionSet(torch.nn.Module):
    """phi (per element) -> your AttentionPool (1 seed) -> rho: a minimal set transformer encoder-free variant."""

    def __init__(self, hidden=32, heads=4):
        super().__init__()
        self.phi = MLP(1, [hidden], hidden)
        self.pool = ex.AttentionPool(hidden, heads=heads, n_seeds=1)
        self.rho = MLP(hidden, [hidden], 1)

    def forward(self, x, mask=None):
        return self.rho(self.pool(self.phi(x), mask)[:, 0])


@app.function
def train_set_model(model, task, *, steps=800, sizes=(1, 10), seed=0):
    """Adam on MSE with fresh random sets every step (batch 128)."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, m, y = make_sets(128, sizes, task, seed=10_000 * seed + step)
        loss = F.mse_loss(model(x, m), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@app.function
def rmse(model, task, sizes, seed=12345):
    x, m, y = make_sets(2000, sizes, task, seed)
    with torch.no_grad():
        return float(((model(x, m) - y) ** 2).mean().sqrt())


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # sets · lab

    every experiment here uses **your** `DeepSets`, `masked_pool` and `AttentionPool`. the data
    are sets of 1 to 10 numbers drawn uniformly from $[0, 1]$, and three tasks: the **sum** of
    the set, its **max**, and the **count** of elements above 0.5. each model is trained for
    800 Adam steps on fresh random sets. for each experiment: write down a prediction, lock it
    in, run it, then explain what happened. compare with `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: which aggregation can learn which task?

    three DeepSets, identical except for the aggregation (sum, mean, max), each trained on each
    of the three tasks: 9 models. the table shows the test RMSE on new sets of the same sizes
    (1 to 10). as a yardstick, predicting the average target gives an RMSE of about 1.6 (sum),
    0.21 (max) and 1.9 (count).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "which of the 9 (aggregation, task) pairs will fail, i.e. stay near the yardstick? which aggregation is "
        "best at the max task?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    exp1 = {}
    for _task in TASKS:
        for _agg in ("sum", "mean", "max"):
            torch.manual_seed(0)
            exp1[(_agg, _task)] = train_set_model(ex.DeepSets(1, 32, 1, aggregation=_agg), _task)
    _rows = []
    for _agg in ("sum", "mean", "max"):
        _rows.append(f"| {_agg} | " + " | ".join(f"{rmse(exp1[(_agg, t)], t, (1, 10)):.3f}" for t in TASKS) + " |")
    mo.md("| aggregation | " + " | ".join(TASKS) + " |\n|---|---|---|---|\n" + "\n".join(_rows))
    return (exp1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** for each failure, name the information the aggregation throws away that the
    task needs. a mean-pooled network sees "the average $\phi(x)$": why is that enough for the
    max but not for the sum? why can't max-pooling count, whatever $\phi$ it learns?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: bigger sets than you trained on

    the same 9 models, tested on sets of **50 to 100** elements, which none of them has seen.
    the plot shows predictions against the truth for the sum-aggregation and max-aggregation
    networks on the max task.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "on sets of 50–100 elements, which models keep working? in particular: the sum-aggregation network learned the "
        "max task well in experiment 1. does it still predict the max correctly on bigger sets?"
    )
    p2
    return (p2,)


@app.cell
def _(exp1, p2):
    ui.gate(p2)
    _rows = []
    for _agg in ("sum", "mean", "max"):
        _rows.append(f"| {_agg} | " + " | ".join(
            f"{rmse(exp1[(_agg, t)], t, (1, 10)):.3f} → {rmse(exp1[(_agg, t)], t, (50, 100)):.3f}" for t in TASKS) + " |")
    _x, _m, _y = make_sets(400, (50, 100), "max", seed=7)
    _fig, _ax = plt.subplots(figsize=(5, 4))
    with torch.no_grad():
        for _agg, _c in (("sum", "C1"), ("max", "C0")):
            _ax.scatter(_y[:, 0], exp1[(_agg, "max")](_x, _m)[:, 0], s=8, c=_c, label=f"{_agg} aggregation", alpha=0.6)
    _ax.plot([0.9, 1.0], [0.9, 1.0], "k--", lw=1, label="perfect")
    _ax.set(xlabel="true max (sets of 50–100)", ylabel="predicted", title="the max task, out of distribution")
    _ax.legend()
    _fig.tight_layout()
    mo.vstack([mo.md("RMSE, sizes 1–10 → sizes 50–100:\n\n| aggregation | " + " | ".join(TASKS)
                     + " |\n|---|---|---|---|\n" + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does the sum-aggregation network have to do internally to compute a
    max, and why does that trick depend on the set size? why does the max-aggregation network
    extrapolate perfectly on the max task? which of the three tasks is "size-free", and does
    the aggregation that extrapolates best match the task's own structure?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: does attention fix it?

    the set transformer replaces the fixed sum/mean/max with *learned* attention pooling. we
    train `phi → AttentionPool (4 heads) → rho` on the three tasks, in distribution (sizes 1–10).
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "attention pooling learns which elements to weight. will it match the best fixed aggregation on all three "
        "tasks, including the sum and the count?"
    )
    p3
    return (p3,)


@app.cell
def _(exp1, p3):
    ui.gate(p3)
    _rows = []
    _best = {t: min(("sum", "mean", "max"), key=lambda a: rmse(exp1[(a, t)], t, (1, 10))) for t in TASKS}
    _att = {}
    for _task in TASKS:
        torch.manual_seed(0)
        _att[_task] = train_set_model(AttentionSet(), _task)
    _rows.append("| attention pooling | " + " | ".join(f"{rmse(_att[t], t, (1, 10)):.3f}" for t in TASKS) + " |")
    _rows.append("| best fixed aggregation | " + " | ".join(
        f"{rmse(exp1[(_best[t], t)], t, (1, 10)):.3f} ({_best[t]})" for t in TASKS) + " |")
    mo.md("| model | " + " | ".join(TASKS) + " |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the stage 4 checks showed that attention pooling gives the same output for a
    set and for the same set listed twice. connect that property to this table. what would you
    add to the attention model so it could count? (the set transformer paper's answer, and a
    cheaper one.)
    """)
    return


if __name__ == "__main__":
    app.run()
