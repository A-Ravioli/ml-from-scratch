import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import functools

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP  # yours, from chapter 15

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    # and your FedAvg simulator from topic 27 (electives own no mlfs module, so load it directly)
    def _smoke(m):  # raises NotImplementedError if a stage you need here isn't done yet
        _y = np.repeat(np.arange(2), 4)
        m.partition_shards(_y, 2, 2, seed=0)
        _X = torch.randn(8, 2)
        m.fedavg(torch.nn.Linear(2, 2), [(_X[:4], torch.tensor(_y[:4])), (_X[4:], torch.tensor(_y[4:]))], rounds=1,
                 client_fraction=1.0, batch_size=None, local_update=m.client_update, seed=0)

    prev = notebooks.load_topic("27-federated-averaging", key="fedavg", smoke=_smoke)
    fl = prev.module
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # heterogeneity · lab

    every experiment here runs **your** code: your partitions and FedProx from this topic,
    and your `fedavg` from topic 27. for each one: write down a prediction, lock it in, run
    it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards. the whole lab takes about two minutes on one CPU
    core.

    the setup is topic 27's: 10,000 MNIST training images over 100 clients, the MLP
    784 → 64 → 10, 10 clients per round, batch 10, lr 0.05, and 2,000 test images.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell
def _():
    _d = datasets.mnist()
    _n = min(10_000, len(_d["y_train"]))
    X = torch.tensor(_d["X_train"][:_n].reshape(_n, -1) / 255.0, dtype=torch.float32)
    y = torch.tensor(_d["y_train"][:_n])
    _nt = min(2_000, len(_d["y_test"]))
    X_test = torch.tensor(_d["X_test"][:_nt].reshape(_nt, -1) / 255.0, dtype=torch.float32)
    y_test = torch.tensor(_d["y_test"][:_nt])
    _yn = y.numpy()
    splits = {
        "IID (Dirichlet α = 100)": ex.partition_dirichlet(_yn, 100, 100.0, seed=0),
        "Dirichlet α = 1": ex.partition_dirichlet(_yn, 100, 1.0, seed=0),
        "Dirichlet α = 0.1": ex.partition_dirichlet(_yn, 100, 0.1, seed=0),
        "shards, 2 labels per client": fl.partition_shards(_yn, 100, 2, seed=0),
        "shards, 1 label per client": fl.partition_shards(_yn, 100, 1, seed=0),
    }

    def accuracy(model):
        with torch.no_grad():
            return (model(X_test).argmax(1) == y_test).float().mean().item()

    def new_model(seed):
        torch.manual_seed(seed)
        return MLP(784, [64], 10)

    _rows = [f"| {k} | {ex.heterogeneity(p, _yn, 10):.2f} | {min(map(len, p))}–{max(map(len, p))} | "
             f"{np.median([len(np.unique(_yn[q])) for q in p]):.0f} |" for k, p in splits.items()]
    mo.md("| split | heterogeneity | client sizes | median labels per client |\n|---|---|---|---|\n" + "\n".join(_rows))
    return X, accuracy, new_model, splits, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much skew does FedAvg tolerate?

    your `fedavg` with $E = 5$ local epochs for 30 rounds on each split above, two seeds each.
    the heterogeneity score (your `heterogeneity`: the size-weighted total-variation distance
    between each client's label mix and the global one) runs from 0.04 (IID) to 0.89 (one
    label per client).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "rank the five splits by final test accuracy. at which heterogeneity does FedAvg start to lose more than "
        "5 points, and how badly does it do with one label per client?"
    )
    p1
    return (p1,)


@app.cell
def _(X, accuracy, new_model, p1, splits, y):
    ui.gate(p1)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _rows = []
    _yn = y.numpy()
    for _name, _parts in splits.items():
        _data = [(X[p], y[p]) for p in _parts]
        _hs = [fl.fedavg(new_model(s), _data, rounds=30, client_fraction=0.1, local_epochs=5, batch_size=10, lr=0.05,
                         eval_fn=accuracy, seed=s) for s in range(2)]
        _acc = np.array([h["eval"] for h in _hs])
        _axes[0].plot(_hs[0]["eval_round"], _acc.mean(0), lw=1.5, label=_name)
        _het = ex.heterogeneity(_parts, _yn, 10)
        _final = _acc[:, -5:].mean(1)  # mean of the last 5 evaluations, per seed: less noisy than the last one
        _axes[1].errorbar(_het, _final.mean(), yerr=np.abs(_final[0] - _final[1]) / 2, fmt="o", capsize=4)
        _axes[1].annotate(_name.split(" (")[0], (_het, _final.mean()), textcoords="offset points", xytext=(6, 4),
                          fontsize=8)
        _rows.append(f"| {_name} | {_het:.2f} | {_final.mean():.3f} ({', '.join(f'{v:.3f}' for v in _final)}) | "
                     f"{np.abs(np.diff(_acc[:, -10:], axis=1)).mean():.3f} |")
    _axes[0].set(xlabel="communication rounds", ylabel="test accuracy (mean of 2 seeds)")
    _axes[0].legend(fontsize=8)
    _axes[1].set(xlabel="heterogeneity (weighted TV distance)", ylabel="accuracy, mean of rounds 26–30",
                 xlim=(-0.05, 1.05))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| split | heterogeneity | accuracy, rounds 26–30 (per seed) | mean round-to-round change, last 10 rounds |\n"
                           "|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is FedAvg nearly unaffected by moderate skew (α = 1), when every
    client's label mix is visibly different? what changes qualitatively when each client sees
    only one label: what does a client's update look like after 50 local steps on a single
    digit, and what does averaging 10 of them give (think about which output biases and
    which last-layer rows move)? why is the round-to-round wobble a symptom of the same
    problem?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: client drift, exactly

    now a setting where you can compute everything: 10 clients, each with a quadratic loss
    $\tfrac12(w - c_k)^\top A_k(w - c_k)$ in 5 dimensions, random positive-definite $A_k$
    (eigenvalues between 0.2 and 2) and random optima $c_k$. full participation, local
    gradient descent with lr 0.1. for each number of local steps $K$ we compute the point
    FedAvg converges to with your `quadratic_fixed_point`, the same for FedProx at the
    $\mu$ chosen below, and run your `quadratic_scaffold` for 400 rounds, then measure the
    distance to the true optimum.
    """)
    return


@app.cell
def _():
    mu_slider = mo.ui.slider(0.0, 3.0, value=1.0, step=0.1, label="FedProx μ")
    p2 = ui.predict(
        "as K grows from 1 to 200, how does FedAvg's error behave: grow without bound, grow and level off, or stay "
        "near zero? does FedProx's error go to zero for large μ? and SCAFFOLD's?"
    )
    mo.vstack([mu_slider, p2])
    return mu_slider, p2


@app.cell
def _(mu_slider, p2):
    ui.gate(p2)
    _rng = np.random.default_rng(0)
    _As = []
    for _ in range(10):
        _Q, _ = np.linalg.qr(_rng.normal(size=(5, 5)))
        _As.append(_Q @ np.diag(_rng.uniform(0.2, 2.0, 5)) @ _Q.T)
    _As = np.array(_As)
    _cs = _rng.normal(size=(10, 5)) * 2
    _opt = ex.global_optimum(_As, _cs)
    _Ks = [1, 2, 5, 10, 20, 50, 100, 200]
    _err = {"FedAvg": [], f"FedProx μ = {mu_slider.value:g}": [], "SCAFFOLD (400 rounds)": []}
    for _K in _Ks:
        _err["FedAvg"].append(np.linalg.norm(ex.quadratic_fixed_point(_As, _cs, local_steps=_K, lr=0.1) - _opt))
        _err[f"FedProx μ = {mu_slider.value:g}"].append(np.linalg.norm(
            ex.quadratic_fixed_point(_As, _cs, local_steps=_K, lr=0.1, mu=mu_slider.value) - _opt))
        _tr = ex.quadratic_scaffold(_As, _cs, np.zeros(5), rounds=400, local_steps=_K, lr=0.1)
        _err["SCAFFOLD (400 rounds)"].append(np.linalg.norm(_tr[-1] - _opt))
    _avg_opt = np.linalg.norm(_cs.mean(0) - _opt)
    _fig, _ax = plt.subplots(figsize=(7, 4))
    for _name, _e in _err.items():
        _ax.plot(_Ks, _e, "o-", label=_name)
    _ax.axhline(_avg_opt, color="k", ls=":", lw=1, label="distance of mean(c_k) from the optimum")
    _ax.set(xscale="log", xlabel="local steps per round K (log scale)", ylabel="‖limit − optimum‖")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = [f"| {K} | " + " | ".join(f"{_err[n][i]:.2e}" for n in _err) + " |" for i, K in enumerate(_Ks)]
    mo.vstack([_fig, mo.md("| K | " + " | ".join(_err) + " |\n|---|---|---|---|\n" + "\n".join(_rows)
                           + f"\n\nthe average of the local optima is {_avg_opt:.3f} away from the global optimum.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** where does FedAvg's error level off, and why exactly there? why is the
    error exactly zero at $K = 1$? move the μ slider: why can't any finite μ make FedProx's
    limit exact (what does the proximal term pull towards, and is that the global optimum)?
    SCAFFOLD's local steps follow a corrected gradient $g_k(y) - c_k + c$: at SCAFFOLD's
    fixed point, what does each correction equal?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: does FedProx fix the broken case?

    back to MNIST with one label per client, the case FedAvg handled worst. your
    `fedprox_client_update` plugs into your `fedavg` as `local_update`; $E = 5$, 30 rounds,
    two seeds, $\mu \in \{0, 0.1, 1\}$ ($\mu = 0$ is FedAvg).
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how many points of test accuracy does the best μ add over plain FedAvg on the one-label split?")
    p3
    return (p3,)


@app.cell
def _(X, accuracy, new_model, p3, splits, y):
    ui.gate(p3)
    _data = [(X[p], y[p]) for p in splits["shards, 1 label per client"]]
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _mu in [0.0, 0.1, 1.0]:
        _upd = functools.partial(ex.fedprox_client_update, mu=_mu)
        _hs = [fl.fedavg(new_model(s), _data, rounds=30, client_fraction=0.1, local_epochs=5, batch_size=10, lr=0.05,
                         local_update=_upd, eval_fn=accuracy, seed=s) for s in range(2)]
        _acc = np.array([h["eval"] for h in _hs])
        _ax.plot(_hs[0]["eval_round"], _acc.mean(0), label=f"μ = {_mu:g}")
        _final = _acc[:, -5:].mean(1)
        _rows.append(f"| {_mu:g} | {_final.mean():.3f} ({', '.join(f'{v:.3f}' for v in _final)}) |")
    _ax.set(xlabel="communication rounds", ylabel="test accuracy (mean of 2 seeds)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| μ | accuracy, rounds 26–30 (per seed) |\n|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** experiment 2 says FedProx reduces drift for large $K$. why doesn't that
    translate into accuracy here? (compare the size of the correction with the size of the
    problem: what does μ = 1 do to a client's *useful* progress, and is the one-label
    problem mainly drift in the sense of experiment 2, or something else?) what was FedProx
    actually designed to handle (reread idea 4: partial work and stragglers)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 4: one model for everyone, or one each?

    each client's data is split 80/20 into local training and local test data, so a
    client's test set follows *its own* label mix. we train one global model with your
    `fedavg` ($E = 5$, 30 rounds), then compare, averaged over the 100 clients:

    - **global**: the federated model as is;
    - **fine-tuned**: your `fine_tune`, 20 SGD steps on the client's training data (all layers, and head only);
    - **local only**: a fresh model trained on the client's data alone for 20 epochs.
    """)
    return


@app.cell
def _():
    p4 = ui.predict(
        "on the 2-labels-per-client split, which of the four wins per client, and by how much? and on IID data?"
    )
    p4
    return (p4,)


@app.cell
def _(X, accuracy, new_model, p4, splits, y):
    ui.gate(p4)
    _rows = []
    _res = {}
    for _name in ["shards, 2 labels per client", "Dirichlet α = 1", "IID (Dirichlet α = 100)"]:
        _rng = np.random.default_rng(0)
        _tr, _te = [], []
        for _p in splits[_name]:
            _p = _rng.permutation(_p)
            _k = max(1, int(0.8 * len(_p)))
            _tr.append((X[_p[:_k]], y[_p[:_k]]))
            _te.append((X[_p[_k:]], y[_p[_k:]]) if _k < len(_p) else (X[_p[:1]], y[_p[:1]]))
        _glob = new_model(0)
        _h = fl.fedavg(_glob, _tr, rounds=30, client_fraction=0.1, local_epochs=5, batch_size=10, lr=0.05,
                       eval_fn=accuracy, eval_every=30, seed=0)
        _local = []
        for _k, (_Xk, _yk) in enumerate(_tr):
            _m = new_model(100 + _k)
            fl.client_update(_m, _Xk, _yk, epochs=20, batch_size=10, lr=0.05, seed=_k)
            _local.append(_m)
        _res[_name] = {
            "global": ex.per_client_accuracy(_glob, _te),
            "fine-tuned (all)": ex.per_client_accuracy(
                [ex.fine_tune(_glob, a, b, steps=20, lr=0.05, seed=k) for k, (a, b) in enumerate(_tr)], _te),
            "fine-tuned (head)": ex.per_client_accuracy(
                [ex.fine_tune(_glob, a, b, steps=20, lr=0.05, head_only=True, seed=k) for k, (a, b) in enumerate(_tr)],
                _te),
            "local only": ex.per_client_accuracy(_local, _te),
        }
        _rows.append(f"| {_name} | {_h['eval'][-1]:.3f} | " + " | ".join(
            f"{v.mean():.3f} ± {v.std(ddof=1) / np.sqrt(len(v)):.3f}" for v in _res[_name].values()) + " |")
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _w = 0.2
    for _j, _method in enumerate(["global", "fine-tuned (all)", "fine-tuned (head)", "local only"]):
        _ax.bar(np.arange(3) + (_j - 1.5) * _w, [_res[n][_method].mean() for n in _res], _w, label=_method)
    _ax.set_xticks(range(3), list(_res))
    _ax.set(ylabel="mean per-client test accuracy", ylim=(0.5, 1.0))
    _ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| split | global model, global test set | global | fine-tuned (all) | fine-tuned (head) | local only |\n"
                           "|---|---|---|---|---|---|\n" + "\n".join(_rows)
                           + "\n\nper-client columns: mean ± standard error over the 100 clients' local test sets.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on the 2-label split, why does a model trained on 80 images of two
    digits do so well *on that client*, and what is it useless for? why does fine-tuning help
    a lot there, a little at α = 1, and not at all (or slightly hurt) on IID data? the global
    model's accuracy on the shared test set and its mean per-client accuracy are different
    numbers: which one should a federated system report, and to whom?
    """)
    return


if __name__ == "__main__":
    app.run()
