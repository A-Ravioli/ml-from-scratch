import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.nn import Linear, ReLU, Sequential, cross_entropy  # yours, from ch 14 · topic 04
    from mlfs.optim import SGD  # yours, from ch 06 · topic 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    _d = datasets.mnist()  # real MNIST, or a small synthetic stand-in when offline
    _X = _d["X_train"].reshape(len(_d["X_train"]), -1) / 255.0
    _mu, _sd = _X.mean(), _X.std()
    X_all, y_all = (_X - _mu) / _sd, _d["y_train"]
    X_test = (_d["X_test"].reshape(len(_d["X_test"]), -1) / 255.0 - _mu) / _sd
    y_test = _d["y_test"]
    DEVICE = ex.get_device()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # bridging to PyTorch · lab

    every experiment here uses **your** `to_torch`, `MLP`, `make_loader`, `evaluate` and
    `ema_update` from `exercise.py`, next to your own `mlfs` engine, on MNIST digits
    (standardized pixels; offline, a synthetic stand-in with the same shapes). the best
    accelerator `get_device()` found on this machine: **{DEVICE}**. for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. compare your
    explanations with `solutions/lab_notes.md` afterwards. the whole lab takes about a minute.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: two engines, one trajectory

    a $784 \to 128 \to 10$ network is built with your `mlfs.nn` and converted with
    `to_torch`. then both copies are trained side by side for 300 SGD steps (learning rate
    0.1) on the *same* minibatches: once with the torch copy in float64, once converted to
    float32. after steps 1, 10, 100 and 300 we measure the largest difference between any
    weight in the two copies.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "after 300 steps, how far apart are the two float64 copies? and the float32 copy: does the gap stay at "
        "float32 rounding (~1e-7), or grow until the two models are unrelated?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _checkpoints = (1, 10, 30, 100, 300)
    _res = {}
    for _dtype in (torch.float64, torch.float32):
        _r = np.random.default_rng(0)
        _mine = Sequential(Linear(784, 128, rng=_r), ReLU(), Linear(128, 10, rng=_r))
        _theirs = ex.to_torch(_mine).to(_dtype)
        _om, _ot = SGD(_mine.parameters(), lr=0.1), torch.optim.SGD(_theirs.parameters(), lr=0.1)
        _r, _gaps = np.random.default_rng(0), []
        for _step in range(1, 301):
            _idx = _r.integers(0, min(10000, len(X_all)), 64)
            _om.zero_grad()
            cross_entropy(_mine(X_all[_idx]), y_all[_idx]).backward()
            _om.step()
            _ot.zero_grad()
            torch.nn.functional.cross_entropy(_theirs(torch.tensor(X_all[_idx], dtype=_dtype)),
                                              torch.tensor(y_all[_idx])).backward()
            _ot.step()
            if _step in _checkpoints:
                _gaps.append(max(float(np.abs(p.data - q.detach().double().numpy()).max())
                                 for p, q in zip(_mine.parameters(), _theirs.parameters())))
        _res[str(_dtype).replace("torch.", "")] = _gaps
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    for _k, _g in _res.items():
        _ax.loglog(_checkpoints, _g, "o-", label=f"torch copy in {_k}")
    _ax.set(xlabel="SGD step", ylabel="max |w_mlfs - w_torch|")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| torch dtype | " + " | ".join(f"step {c}" for c in _checkpoints) + " |\n|---|"
                           + "---|" * len(_checkpoints) + "\n"
                           + "\n".join(f"| {k} | " + " | ".join(f"{g:.1e}" for g in v) + " |" for k, v in _res.items()))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why aren't the float64 copies *exactly* equal (think about the order in
    which a matrix product adds its terms)? the float32 copy starts $10^{-8}$ away: where does
    that come from, and how fast does the gap grow? would you expect the two trajectories to
    stay this close for 100,000 steps, and does it matter if they don't? what has this
    experiment proved about your engine that a single gradient check couldn't?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how fast is each engine?

    one training step (forward, backward, SGD update) of a $784 \to w \to w \to 10$ ReLU
    network on a batch of 256 images, for widths $w$ from 64 to 2,048: your `mlfs` engine
    (NumPy, float64), PyTorch on the CPU in float32 and float64, and PyTorch on the
    accelerator if this machine has one. each time is the best of 5 runs after a warm-up
    (they depend on your machine, and on what else it's doing).
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "PyTorch is written in C++. how much faster than your NumPy engine is it on the CPU for a small network "
        "(w = 64)? for a big one (w = 2048)? is a GPU always faster than the CPU?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)

    def _best_time(fn, reps=5):
        fn()  # warm-up
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    _Xb, _yb = X_all[:256], y_all[:256]
    _widths = [64, 256, 1024, 2048]
    _engines = ["mlfs (NumPy, float64)", "torch CPU float32", "torch CPU float64"]
    if DEVICE.type != "cpu":
        _engines.append(f"torch {DEVICE.type} float32")
    _times = {e: [] for e in _engines}
    for _w in _widths:
        _r = np.random.default_rng(0)
        _m = Sequential(Linear(784, _w, rng=_r), ReLU(), Linear(_w, _w, rng=_r), ReLU(), Linear(_w, 10, rng=_r))
        _o = SGD(_m.parameters(), lr=0.01)

        def _mlfs_step(m=_m, o=_o):
            o.zero_grad()
            cross_entropy(m(_Xb), _yb).backward()
            o.step()

        _times[_engines[0]].append(_best_time(_mlfs_step))
        for _e in _engines[1:]:
            _dev = "cpu" if "CPU" in _e else DEVICE
            _dt = torch.float64 if "float64" in _e else torch.float32
            _t = ex.MLP(784, [_w, _w], 10).to(_dev, _dt)
            _ot = torch.optim.SGD(_t.parameters(), lr=0.01)
            _xb, _ybt = torch.tensor(_Xb, dtype=_dt, device=_dev), torch.tensor(_yb, device=_dev)

            def _torch_step(t=_t, ot=_ot, xb=_xb, yb=_ybt, dev=_dev):
                ot.zero_grad()
                torch.nn.functional.cross_entropy(t(xb), yb).backward()
                ot.step()
                if getattr(dev, "type", dev) == "mps":
                    torch.mps.synchronize()  # wait for the GPU, or we'd time only the launch
                elif getattr(dev, "type", dev) == "cuda":
                    torch.cuda.synchronize()

            _times[_e].append(_best_time(_torch_step))
    _fig, _ax = plt.subplots(figsize=(6.5, 3.8))
    for _e, _ts in _times.items():
        _ax.loglog(_widths, np.array(_ts) * 1e3, "o-", label=_e, base=2)
    _ax.set(xlabel="width w", ylabel="ms per training step (batch 256)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _rows = "\n".join(f"| {e} | " + " | ".join(f"{t * 1e3:.2f}" for t in ts) + " |" for e, ts in _times.items())
    _ratio = " · ".join(f"w={w}: {m / t:.1f}×" for w, m, t in zip(_widths, _times[_engines[0]], _times[_engines[1]]))
    mo.vstack([_fig, mo.md("| engine | " + " | ".join(f"w = {w}" for w in _widths) + " |\n|---|"
                           + "---|" * len(_widths) + "\n" + _rows
                           + f"\n\nspeedup of torch CPU float32 over your engine: {_ratio}.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** at large widths almost all the time is in a few matrix products. which
    library actually computes them, for NumPy and for PyTorch on the CPU? so what is the
    speedup of float32 over float64 really measuring? at small widths, what dominates instead
    (count the Python-level operations per step in your engine)? why is the accelerator
    *slower* than the CPU for the smallest network, and roughly at what width does it start
    to win?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: an exponential moving average of the weights

    a $784 \to 256 \to 256 \to 10$ `MLP` trained with SGD (learning rate 0.1, momentum 0.9,
    batch 128) for 4 epochs on 20,000 images (offline: 2,000). after every optimizer step,
    your `ema_update(ema, model, decay=0.99)` folds the current weights into a second copy
    of the network. after each epoch we score both on the test set with your `evaluate`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "the EMA copy is never trained: it only averages the trained model's recent weights. will it score better or "
        "worse than the model itself, and by how much?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    torch.manual_seed(0)
    _n = min(20000, len(X_all))
    _model = ex.MLP(784, [256, 256], 10)
    _ema = copy.deepcopy(_model)
    _opt = torch.optim.SGD(_model.parameters(), lr=0.1, momentum=0.9)
    _train = ex.make_loader(X_all[:_n], y_all[:_n], batch_size=128, seed=0)
    _test = ex.make_loader(X_test, y_test, shuffle=False, batch_size=1000)
    _rows, _raw, _avg = [], [], []
    for _epoch in range(4):
        _model.train()
        for _xb, _ybb in _train:
            _opt.zero_grad()
            torch.nn.functional.cross_entropy(_model(_xb), _ybb).backward()
            _opt.step()
            ex.ema_update(_ema, _model, decay=0.99)
        _raw.append(ex.evaluate(_model, _test)["accuracy"])
        _avg.append(ex.evaluate(_ema, _test)["accuracy"])
        _rows.append(f"| {_epoch + 1} | {_raw[-1]:.4f} | {_avg[-1]:.4f} | {_avg[-1] - _raw[-1]:+.4f} |")
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.plot(range(1, 5), _raw, "o-", label="trained model")
    _ax.plot(range(1, 5), _avg, "s-", label="EMA of its weights (decay 0.99)")
    _ax.set(xlabel="epoch", ylabel="test accuracy", xticks=[1, 2, 3, 4])
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("| epoch | model | EMA | EMA − model |\n|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** with decay 0.99, roughly how many recent steps does the EMA average over?
    with a constant learning rate, SGD's weights keep bouncing around a minimum (topic 04,
    on paper 1): what does averaging the bounces do? how is that related to decaying the
    learning rate? why must `ema_update` *copy* the batch-norm buffers instead of averaging
    them (this MLP has none, but chapter 21's and 24's models do)?
    """)
    return


if __name__ == "__main__":
    app.run()
