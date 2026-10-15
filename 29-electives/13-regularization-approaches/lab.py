import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.stats import spearmanr

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP

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

    torch.set_num_threads(2)
    GROUPS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    TRAIN = dict(optimizer="adam", lr=1e-3, batch_size=64, epochs=1)
    N_FISHER = 200


@app.function
def new_model(seed, hidden=200):
    """the block's network: 784 -> 200 -> 200 -> 10, ReLU."""
    torch.manual_seed(seed)
    return MLP(784, [hidden, hidden], 10)


@app.function
def strategy(method, strength, scenario):
    """a train_fn for run_sequence: plain fine-tuning, EWC(lam=strength) or SI(c=strength)."""
    reg = None
    if method == "ewc":
        reg = ex.EWC(strength, n_samples=N_FISHER)
    elif method == "si":
        reg = ex.SI(strength)

    def train_fn(model, task, t):
        if method == "si":
            reg.begin_task(model)
        cl.train_task(model, task, scenario=scenario, seed=t, penalty=None if reg is None else reg.penalty,
                      after_step=reg.after_step if method == "si" else None, **TRAIN)
        if method == "ewc":
            reg.consolidate(model, task, scenario=scenario)
        elif method == "si":
            reg.consolidate(model)

    return train_fn


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # regularization approaches · lab

    every experiment here uses **your** `fisher_diagonal`, `EWC` and `SI`, plugged into
    **your** topic-12 harness through `train_task(..., penalty=..., after_step=...)`. the network
    and training are the same as in topic 12: a 784–200–200–10 MLP, Adam at $10^{-3}$, batch 64,
    one epoch per task. EWC estimates each Fisher from 200 examples. for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. compare with
    `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes on a laptop CPU.
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
    Xtr, ytr = _d["X_train"] / 255.0, _d["y_train"]
    Xte, yte = _d["X_test"] / 255.0, _d["y_test"]
    perm_train = cl.permuted_tasks(Xtr[:10000], ytr[:10000], 5, seed=0)
    perm_test = cl.permuted_tasks(Xte[:2000], yte[:2000], 5, seed=0)
    split_train = cl.split_tasks(Xtr, ytr, GROUPS)
    split_test = cl.split_tasks(Xte, yte, GROUPS)
    return perm_test, perm_train, split_test, split_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: the stability–plasticity tradeoff

    five permuted-MNIST tasks (10,000 training and 2,000 test images each). sweep EWC's
    $\lambda$ and SI's $c$ over several orders of magnitude, and record three numbers per run:
    **ACC** (final average accuracy), **BWT** (how much old tasks lost), and the **learning
    accuracy**, the mean of $R$'s diagonal: how well each task was learned when it was the
    current one. strength 0 is plain fine-tuning.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "as lambda grows from 0 to 1e5, what happens to BWT, to the learning accuracy, and to ACC? is there a best "
        "lambda, and roughly where?"
    )
    p1
    return (p1,)


@app.cell
def _(p1, perm_test, perm_train):
    ui.gate(p1)
    sweeps = {"ewc": [0, 10, 100, 1e3, 1e4, 1e5], "si": [0, 1, 10, 100, 1e3]}
    _cache = {}
    sweep_res = {}
    for _method, _vals in sweeps.items():
        sweep_res[_method] = []
        for _v in _vals:
            _key = ("ft", 0) if _v == 0 else (_method, _v)
            if _key not in _cache:
                _R, _ = cl.run_sequence(new_model(0), perm_train, perm_test, train_fn=strategy(*_key, "class"))
                _cache[_key] = (cl.average_accuracy(_R), cl.backward_transfer(_R), float(np.mean(np.diag(_R))))
            sweep_res[_method].append(_cache[_key])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for _ax, (_method, _vals) in zip(_axes, sweeps.items()):
        _r = np.array(sweep_res[_method])
        _x = np.arange(len(_vals))
        _ax.plot(_x, _r[:, 2], "o-", label="learning accuracy (plasticity)")
        _ax.plot(_x, 1 + _r[:, 1], "s-", label="1 + BWT (stability)")
        _ax.plot(_x, _r[:, 0], "^-", lw=2.5, label="ACC")
        _ax.set_xticks(_x, [f"{v:g}" for v in _vals])
        _ax.set(xlabel="λ" if _method == "ewc" else "c", title=_method.upper(), ylim=(0.5, 1.02))
    _axes[0].set_ylabel("accuracy")
    _axes[0].legend(loc="lower left", fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| method | strength | ACC | BWT | learning accuracy |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {m} | {v:g} | {a:.3f} | {b:.3f} | {la:.3f} |" for m, vals in sweeps.items()
        for v, (a, b, la) in zip(vals, sweep_res[m])))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** describe the two curves that cross. what limits ACC on the left of the best
    strength, and what limits it on the right? the strengths that work for EWC and for SI differ
    by orders of magnitude: is one method "stronger", or are the numbers in different units?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the same methods, a different scenario

    split-MNIST, in both scenarios from topic 12, three seeds each. fine-tuning against EWC
    ($\lambda = 10^4$) and SI ($c = 100$), strengths that work well on this benchmark.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "fine-tuning reaches ACC ≈ 0.90 task-incrementally and ≈ 0.20 class-incrementally. what ACC do you expect "
        "for EWC and SI in each scenario?"
    )
    p2
    return (p2,)


@app.cell
def _(p2, split_test, split_train):
    ui.gate(p2)
    _settings = [("fine-tuning", "ft", 0), ("EWC λ=1e4", "ewc", 1e4), ("SI c=100", "si", 100)]
    _rows = []
    for _label, _method, _s in _settings:
        _cells = []
        for _sc in ("task", "class"):
            _accs = [cl.average_accuracy(cl.run_sequence(new_model(_seed), split_train, split_test, scenario=_sc,
                                                         train_fn=strategy(_method, _s, _sc))[0])
                     for _seed in range(3)]
            _cells.append(f"{np.mean(_accs):.3f} ± {np.std(_accs, ddof=1):.3f}")
        _rows.append(f"| {_label} | {_cells[0]} | {_cells[1]} |")
    mo.md("| method | task-incremental ACC | class-incremental ACC |\n|---|---|---|\n" + "\n".join(_rows)
          + "\n\nmean ± standard deviation over seeds 0–2.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in the class-incremental scenario, what would EWC have to protect to keep
    task 0 alive, and why can't a penalty on *weights* provide it? think about what task 0's
    Fisher says about the output units of classes 2–9, at the moment task 0 is consolidated.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: which weights matter?

    train one network on the first permuted-MNIST task while SI tracks its path integral, then
    compute the Fisher. do the two importance measures agree? then test them: add Gaussian noise
    (standard deviation 0.1) to the 1% of weights each measure ranks highest, and compare with
    the same noise on the 1% with the *largest magnitude*, on 1% chosen at random, and on the 1%
    the Fisher ranks *lowest* among weights whose Fisher isn't exactly zero.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the Fisher (curvature) and SI's Omega (loss decrease along the path) are computed in completely different "
        "ways. how well will their rankings of the 180,000 weights agree? and which 1% of weights hurts most to perturb?"
    )
    p3
    return (p3,)


@app.cell
def _(p3, perm_test, perm_train):
    ui.gate(p3)
    _m = new_model(0)
    _si = ex.SI(1.0)
    _si.begin_task(_m)
    cl.train_task(_m, perm_train[0], seed=0, after_step=_si.after_step, **TRAIN)
    _si.consolidate(_m)
    _F = ex.fisher_diagonal(_m, perm_train[0]["X"][:500])
    _names = list(_F)
    _params = dict(_m.named_parameters())
    _fv = torch.cat([_F[n].flatten() for n in _names])
    _ov = torch.cat([_si.omega[n].flatten() for n in _names])
    _wv = torch.cat([_params[n].detach().flatten() for n in _names])
    _rho_so = spearmanr(_fv.numpy(), _ov.numpy())[0]
    _both = (_fv > 0) & (_ov > 0)
    _rho_pos = spearmanr(_fv[_both].numpy(), _ov[_both].numpy())[0]
    _rho_w = spearmanr(_fv.numpy(), _wv.abs().numpy())[0]
    _base = cl.evaluate(_m, perm_test[0])
    _state = {k: v.clone() for k, v in _m.state_dict().items()}
    _k = int(0.01 * len(_fv))
    _noise = 0.1 * torch.randn(len(_fv), generator=torch.Generator().manual_seed(1))
    _scores = {"Fisher, top 1%": _fv, "SI Ω, top 1%": _ov, "magnitude, top 1%": _wv.abs(),
               "random 1%": torch.rand(len(_fv), generator=torch.Generator().manual_seed(2)),
               "Fisher, bottom 1% of those with F > 0": torch.where(_fv > 0, -_fv, torch.tensor(-float("inf")))}
    _after = {}
    for _label, _score in _scores.items():
        _mask = torch.zeros(len(_fv))
        _mask[torch.topk(_score, _k).indices] = 1.0
        _off = 0
        with torch.no_grad():
            for _n in _names:
                _p = _params[_n]
                _p.add_((_noise * _mask)[_off:_off + _p.numel()].view_as(_p))
                _off += _p.numel()
        _after[_label] = cl.evaluate(_m, perm_test[0])
        _m.load_state_dict(_state)
    _pos = torch.nonzero((_fv > 0) & (_ov > 0)).flatten()
    _sub = _pos[torch.randperm(len(_pos), generator=torch.Generator().manual_seed(3))[:5000]]
    _fig, _ax = plt.subplots(figsize=(4.6, 4.2))
    _ax.loglog(_fv[_sub].numpy(), _ov[_sub].numpy(), ".", ms=2, alpha=0.4)
    _ax.set(xlabel="Fisher diagonal", ylabel="SI Ω", title=f"5,000 random weights with both > 0\nSpearman ρ = {_rho_pos:.2f}")
    _fig.tight_layout()
    _zero = int((_fv == 0).sum())
    mo.vstack([_fig, mo.md(
        f"Spearman rank correlation of the Fisher with SI's Ω: **{_rho_so:.2f}** over all weights, **{_rho_pos:.2f}** "
        f"over the {int(_both.sum()):,} where both are positive; of the Fisher with weight magnitude: "
        f"**{_rho_w:.2f}**. {_zero:,} of {len(_fv):,} weights have a Fisher of exactly 0. task-0 test accuracy "
        f"before any noise: {_base:.3f}.\n\n"
        "| noise on | task-0 accuracy |\n|---|---|\n"
        + "\n".join(f"| {k} | {v:.3f} |" for k, v in _after.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why should "how much did this weight contribute to lowering the loss" and
    "how curved is the loss in this weight's direction" pick out the same weights? why is
    magnitude such a poor stand-in for importance? what does the bottom-1% row say about how
    much of the network EWC leaves free for the next task?
    """)
    return


if __name__ == "__main__":
    app.run()
