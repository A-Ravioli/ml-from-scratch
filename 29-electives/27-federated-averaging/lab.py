import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.torchnn import MLP  # yours, from chapter 15

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # federated averaging · lab

    every experiment here runs **your** `fedavg` on MNIST split across 100 simulated phones.
    for each one: write down a prediction, lock it in, run it, then explain what happened.
    compare your explanations with `solutions/lab_notes.md` afterwards. the whole lab takes
    about two minutes on one CPU core.

    the setup, used throughout: the first 10,000 MNIST training images, split over 100
    clients of 100 images each, either **IID** (`partition_iid`) or **pathologically
    non-IID** (`partition_shards`: two label-sorted shards per client, so most clients see
    only two digits). the model is an MLP 784 → 64 → 10 (50,890 parameters, so one model
    upload is about 204 kB). test accuracy is measured on 2,000 MNIST test images.
    """)
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
    clients = {
        "IID": [(X[p], y[p]) for p in ex.partition_iid(_n, 100, seed=0)],
        "shards (non-IID)": [(X[p], y[p]) for p in ex.partition_shards(y.numpy(), 100, seed=0)],
    }

    def accuracy(model):
        with torch.no_grad():
            return (model(X_test).argmax(1) == y_test).float().mean().item()

    def new_model(seed):
        torch.manual_seed(seed)
        return MLP(784, [64], 10)

    mo.md(f"{_n:,} training images over 100 clients; {_nt:,} test images. "
          f"labels per client under shards: median "
          f"{int(np.median([len(np.unique(c[1].numpy())) for c in clients['shards (non-IID)']]))}.")
    return accuracy, clients, new_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: local steps vs communication rounds

    each round, 10 clients (`client_fraction=0.1`) download the model, run `E` local epochs
    of SGD (batch 10, lr 0.05) and upload. with 100 images per client, one epoch is 10
    local steps, so `E = 1, 5, 20` means 10, 50, 200 local steps per round. we run 25
    rounds, two seeds each, and look at test accuracy against **rounds** (what communication
    costs) and against **total local steps** (what computation costs).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "how many rounds does E = 20 need to reach 85% (IID) compared with E = 1? and on the non-IID shards, does "
        "more local work still help, or does it start to hurt?"
    )
    p1
    return (p1,)


@app.cell
def _(accuracy, clients, new_model, p1):
    ui.gate(p1)
    _epochs = [1, 5, 20]
    _targets = {"IID": 0.85, "shards (non-IID)": 0.70}
    _runs = {}
    for _split, _data in clients.items():
        for _E in _epochs:
            _runs[_split, _E] = [ex.fedavg(new_model(s), _data, rounds=25, client_fraction=0.1, local_epochs=_E,
                                           batch_size=10, lr=0.05, eval_fn=accuracy, seed=s) for s in range(2)]
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    _rows = []
    for _i, _split in enumerate(clients):
        for _E in _epochs:
            _hs = _runs[_split, _E]
            _acc = np.mean([h["eval"] for h in _hs], axis=0)
            _axes[_i, 0].plot(_hs[0]["eval_round"], _acc, "o-", ms=3, label=f"E = {_E}")
            _axes[_i, 1].plot(_hs[0]["client_steps"], _acc[1:], "o-", ms=3, label=f"E = {_E}")  # rounds 1..25
            _hit = [next((r for r, a in zip(h["eval_round"], h["eval"]) if a >= _targets[_split]), None) for h in _hs]
            _rows.append(f"| {_split} | {_E} | {', '.join('>25' if r is None else str(r) for r in _hit)} | "
                         f"{np.mean([h['eval'][-1] for h in _hs]):.3f} ({', '.join(f'{h['eval'][-1]:.3f}' for h in _hs)}) |")
        _axes[_i, 0].axhline(_targets[_split], color="k", ls=":", lw=1)
        _axes[_i, 1].axhline(_targets[_split], color="k", ls=":", lw=1)
        _axes[_i, 0].set(ylabel=f"{_split}: test accuracy")
        _axes[_i, 1].set(xscale="log")
        _axes[_i, 0].legend(fontsize=8)
    _axes[1, 0].set(xlabel="communication rounds")
    _axes[1, 1].set(xlabel="total local SGD steps across all clients, after rounds 1-25 (log scale)")
    _fig.suptitle("mean of 2 seeds; dotted line: target accuracy")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| split | E | rounds to target (per seed) | final accuracy (per seed) |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** on IID data, why does more local work cut the number of rounds so
    sharply, and why are the returns diminishing? look at the IID curves against total local
    steps (top right): what does their shape say about what a round of FedAvg *is* when the
    clients agree? on the non-IID shards every client's local optimum is a "2-digit
    classifier": why does averaging such models still produce a 10-digit classifier, and why
    are the non-IID curves so much bumpier? which axis, rounds or local steps, is the right
    one for a phone on a slow uplink?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many bits does an update need?

    IID clients, `E = 5`, 25 rounds, two seeds. each client's upload goes through your
    `stochastic_quantize` with `b` bits per entry (the download stays at full precision). we
    plot accuracy against the megabytes **uploaded**.
    """)
    return


@app.cell
def _():
    bits_choice = mo.ui.multiselect(["32 (no compression)", "8", "4", "2", "1"],
                                    value=["32 (no compression)", "8", "4", "2", "1"], label="bits per entry")
    p2 = ui.predict(
        "with 2 bits per entry (16× fewer bytes than float32), how much test accuracy do you lose after 25 rounds? "
        "and with 1 bit?"
    )
    mo.vstack([bits_choice, p2])
    return bits_choice, p2


@app.cell
def _(accuracy, bits_choice, clients, new_model, p2):
    ui.gate(p2)

    def _quantizer(b):
        return lambda delta, rng: ex.stochastic_quantize(delta, b, rng=rng)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _rows = []
    for _name in bits_choice.value:
        _b = int(_name.split()[0])
        _hs = [ex.fedavg(new_model(s), clients["IID"], rounds=25, client_fraction=0.1, local_epochs=5,
                         batch_size=10, lr=0.05, compress=None if _b == 32 else _quantizer(_b), eval_fn=accuracy,
                         seed=s) for s in range(2)]
        _mb = np.array(_hs[0]["bytes_up"]) / 1e6  # after rounds 1..25
        _ax.plot(_mb, np.mean([h["eval"][1:] for h in _hs], axis=0), "o-", ms=3, label=f"{_b} bits")
        _to85 = [next((u for u, a in zip(h["bytes_up"], h["eval"][1:]) if a >= 0.85), None) for h in _hs]
        _rows.append(f"| {_b} | {_hs[0]['bytes_up'][-1] / 1e6:.2f} | "
                     f"{', '.join('—' if u is None else f'{u / 1e6:.2f}' for u in _to85)} | "
                     f"{np.mean([h['eval'][-1] for h in _hs]):.3f} ({', '.join(f'{h['eval'][-1]:.3f}' for h in _hs)}) |")
    _ax.set(xscale="log", xlabel="megabytes uploaded after rounds 1-25 (all clients, cumulative; log scale)",
            ylabel="test accuracy")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| bits | MB uploaded in 25 rounds | MB uploaded to reach 85% (per seed) | final accuracy (per seed) |\n|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the quantizer is unbiased, so averaging 10 quantized updates is an unbiased
    estimate of the average update. why does that make low-bit uploads so cheap in accuracy?
    the error of one entry scales with the spread $\max\Delta - \min\Delta$ of the whole
    update: what kind of update would make 1-bit quantization hopeless, and what could you
    do about it (think: quantize per layer, or rotate first)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: how many clients per round?

    IID clients, `E = 5`, 25 rounds, one seed per setting, with
    `client_fraction` $C \in \{0.01, 0.05, 0.1, 0.5\}$ (1, 5, 10 or 50 clients per round).
    a larger cohort costs proportionally more bytes per round.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("going from 10 to 50 clients per round (5× the traffic per round), how much faster does accuracy rise?")
    p3
    return (p3,)


@app.cell
def _(accuracy, clients, new_model, p3):
    ui.gate(p3)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    _rows = []
    for _C in [0.01, 0.05, 0.1, 0.5]:
        _h = ex.fedavg(new_model(0), clients["IID"], rounds=25, client_fraction=_C, local_epochs=5, batch_size=10,
                       lr=0.05, eval_fn=accuracy, seed=0)
        _axes[0].plot(_h["eval_round"], _h["eval"], "o-", ms=3, label=f"C = {_C}")
        _axes[1].plot(np.array(_h["bytes_up"]) / 1e6, _h["eval"][1:], "o-", ms=3, label=f"C = {_C}")
        _r85 = next((r for r, a in zip(_h["eval_round"], _h["eval"]) if a >= 0.85), None)
        _rows.append(f"| {_C} | {max(1, round(_C * 100))} | {'>25' if _r85 is None else _r85} | "
                     f"{_h['eval'][-1]:.3f} | {_h['bytes_up'][-1] / 1e6:.1f} |")
    _axes[0].set(xlabel="communication rounds", ylabel="test accuracy")
    _axes[1].set(xscale="log", xlabel="megabytes uploaded (log scale)")
    _axes[0].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| C | clients per round | rounds to 85% | final accuracy | MB uploaded |\n|---|---|---|---|---|\n"
                           + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** averaging more client updates reduces the *variance* of each round's
    step, but not its *bias* or its length. why does that give diminishing returns beyond a
    handful of clients on IID data? per megabyte, which cohort size wins? (keep this
    question in mind for topic 29: with differential privacy, a large cohort is what drowns
    out the noise.)
    """)
    return


if __name__ == "__main__":
    app.run()
