import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.train import CosineSchedule  # yours, from ch 15 · 04
    from mlfs.transformer import GPT, get_batch, kv_cache_bytes  # yours, from ch 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    _text = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    TOK = CharTokenizer().train(_text)
    _ids = np.asarray(TOK.encode(_text), dtype=np.int64)
    TRAIN, VAL = _ids[: int(0.9 * len(_ids))], _ids[int(0.9 * len(_ids)):]
    V, CTX, STEPS = TOK.vocab_size, 64, 400


@app.function
def train(model, *, lr=3e-3, steps=STEPS, seed=0, record=None):
    """AdamW (warmup 50, cosine to lr/10), batches of 32 x 64 characters; record(model, step) every 20 steps."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = CosineSchedule(opt, steps, warmup_steps=50, min_lr=lr / 10)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(TRAIN, CTX, 32, generator=g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if record is not None and step % 20 == 0:
            record(model, step)
    return model.eval()


@app.function
@torch.no_grad()
def val_ce(model, n_batches=20):
    """validation cross-entropy in nats per character (no aux loss), the same 20 batches for every model."""
    g = torch.Generator().manual_seed(123)
    model.eval()
    out = []
    for _ in range(n_batches):
        x, y = get_batch(VAL, CTX, 32, generator=g)
        logits, _ = model(x)
        out.append(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
    return float(np.mean(out))


@app.function
def n_params(model):
    return sum(p.numel() for p in model.parameters())


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # modern LLM architecture · lab

    every experiment trains **your** `Llama` (and, for comparison, your chapter 18 `GPT`) on
    tiny Shakespeare characters (a synthetic stand-in offline): {len(TRAIN):,} training
    characters, {STEPS} steps of 32 × {CTX}, about ten seconds per model on a laptop CPU, so
    every number is a single short run, one seed. differences of a few hundredths of a nat can be noise
    (`solutions/lab_notes.md` gives the seed-to-seed spread). for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. about two minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: GPT-2 recipe against Llama recipe

    two models of almost the same size (2 layers, width 64, 4 heads): your chapter 18 GPT
    (learned positions, LayerNorm with biases, GELU MLP of width 256) and your Llama (RoPE,
    RMSNorm, SwiGLU of width 176, no biases, full multi-head attention). same data, same
    optimizer, same steps, each at three peak learning rates.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("at learning rate 3e-3, how much lower (in nats per character) will the Llama's validation "
                    "loss be? and when each model gets its best learning rate?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    lrs = [1e-3, 3e-3, 1e-2]
    arch = {}
    for _name, _make in {"GPT (ch 18)": lambda: GPT(V, block_size=CTX, n_layers=2, n_heads=4, d_model=64),
                         "Llama, n_kv_heads=4": lambda: ex.Llama(V, block_size=CTX, n_layers=2, n_heads=4,
                                                                 n_kv_heads=4, d_model=64)}.items():
        arch[_name] = {}
        for _lr in lrs:
            torch.manual_seed(0)
            _m = _make()
            _t0 = time.perf_counter()
            train(_m, lr=_lr)
            arch[_name][_lr] = (_m, n_params(_m), val_ce(_m), time.perf_counter() - _t0)
    _llama = arch["Llama, n_kv_heads=4"][3e-3][0]
    _sample = TOK.decode(_llama.generate(torch.tensor([TOK.encode(TOK.decode(TRAIN[:6].tolist()))]), 120,
                                         temperature=0.8, generator=torch.Generator().manual_seed(0))[0].tolist())
    _fig, _ax = plt.subplots(figsize=(5, 3))
    for _name, _r in arch.items():
        _ax.plot(lrs, [_r[lr][2] for lr in lrs], "o-", label=_name)
    _ax.set(xscale="log", xlabel="peak learning rate", ylabel="validation loss (nats/char)")
    _ax.legend(fontsize=8)
    _best = {k: min(r.values(), key=lambda t: t[2])[2] for k, r in arch.items()}
    mo.vstack([_fig, mo.md(
        "| model | parameters | " + " | ".join(f"lr {lr:g}" for lr in lrs) + " |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {k} | {r[lrs[0]][1]:,} | " + " | ".join(f"{r[lr][2]:.3f}" for lr in lrs) + " |"
                    for k, r in arch.items())
        + f"\n\nat lr 3e-3 the gap is **{arch['GPT (ch 18)'][3e-3][2] - arch['Llama, n_kv_heads=4'][3e-3][2]:.3f}** "
        f"nats; between the best runs of each it's **{_best['GPT (ch 18)'] - _best['Llama, n_kv_heads=4']:.3f}**.\n\n"
        f"a sample from the Llama (lr 3e-3, temperature 0.8):\n\n```\n{_sample}\n```")])
    return (arch,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what would you have concluded from the 3e-3 column alone? why does a
    comparison of two architectures need each one's own learning rate (and why might the two
    prefer different ones)? the models still differ in four ways (positions, normalization,
    MLP, biases): how would you find out which one matters? why is a single seed not enough
    to rank two models that differ by a few hundredths?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: how many key/value heads do you need?

    the same Llama (lr 3e-3) with 4 query heads and 4 (MHA), 2 (GQA) or 1 (MQA) key/value heads. next
    to the loss, the KV cache each choice needs: for this toy model, and for a 7B-class
    configuration (32 layers, 32 query heads of width 128, a 4,096-token context, 16-bit
    values) from your chapter 18 `kv_cache_bytes`.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("going from 4 key/value heads to 1, how much validation loss do you lose? how much cache "
                    "memory do you save?")
    p2
    return (p2,)


@app.cell
def _(arch, p2):
    ui.gate(p2)
    gqa = {4: arch["Llama, n_kv_heads=4"][3e-3]}
    for _kv in (2, 1):
        torch.manual_seed(0)
        _m = train(ex.Llama(V, block_size=CTX, n_layers=2, n_heads=4, n_kv_heads=_kv, d_model=64))
        gqa[_kv] = (_m, n_params(_m), val_ce(_m), 0.0)
    _rows = []
    for _kv, (_m, _n, _l, _) in gqa.items():
        _toy = kv_cache_bytes(2, _kv, 16, CTX, bytes_per_value=4)
        _big = kv_cache_bytes(32, 8 * _kv, 128, 4096)  # the 7B-class model keeps the same ratio of q to kv heads
        _rows.append(f"| {_kv} ({ {4: 'MHA', 2: 'GQA', 1: 'MQA'}[_kv]}) | {_n:,} | {_l:.3f} | {_toy / 1024:.0f} KiB | "
                     f"{_big / 2**30:.2f} GiB |")
    mo.md("| key/value heads (of 4 query heads) | parameters | validation loss | toy cache per sequence | "
          "7B-class cache per sequence (32, 16 or 8 kv heads) |\n|---|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does sharing keys and values cost so little quality, while the cache
    shrinks in proportion? at batch 32 and 32k tokens of context, how many GiB would the
    MHA cache of the 7B-class model take, and what does that do to the batch sizes you can
    serve (chapter 26 · 06)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: load balancing

    a Llama whose MLPs are your `MoE`: 8 SwiGLU experts of width 64, top-2 routing, trained
    with three load-balancing coefficients. every 20 steps your layer's `last_counts`
    records how many token slots each expert received. the plot shows the **effective number
    of experts** $\exp(H(f))$, where $f$ is the share of slots per expert: 8 when perfectly
    balanced, 2 when every token goes to the same two experts (with top-2 routing that's the floor).
    """)
    return


@app.cell
def _():
    p3 = ui.predict("without the load-balancing loss (coefficient 0), what happens to the experts' shares? does "
                    "balancing cost or buy validation loss?")
    p3
    return (p3,)


@app.cell
def _():
    coef_pick = mo.ui.dropdown({"0": 0.0, "0.01": 0.01, "0.1": 0.1}, value="0", label="bar chart for coefficient")
    coef_pick
    return (coef_pick,)


@app.cell
def _(p3):
    ui.gate(p3)
    moe_runs = {}
    for _coef in (0.0, 0.01, 0.1):
        _trace = []

        def _rec(model, step, _trace=_trace):
            _f = [b.mlp.last_counts.float() / b.mlp.last_counts.sum() for b in model.blocks]
            _trace.append((step, [float(torch.exp(-(f * torch.log(f.clamp_min(1e-12))).sum())) for f in _f]))

        torch.manual_seed(0)
        _m = ex.Llama(V, block_size=CTX, n_layers=2, n_heads=4, n_kv_heads=2, d_model=64, d_ff=64, n_experts=8,
                      top_k=2, aux_loss_coef=_coef)
        train(_m, record=_rec)
        _x, _ = get_batch(VAL, CTX, 32, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            _m(_x)
        _shares = [(b.mlp.last_counts.float() / b.mlp.last_counts.sum()).numpy() for b in _m.blocks]
        moe_runs[_coef] = dict(trace=_trace, val=val_ce(_m), shares=_shares, params=n_params(_m))
    return (moe_runs,)


@app.cell
def _(coef_pick, moe_runs):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for (_coef, _r), _c in zip(moe_runs.items(), ["C3", "C0", "C2"]):
        _steps = [s for s, _ in _r["trace"]]
        for _layer, _ls in ((0, "-"), (1, "--")):
            _a1.plot(_steps, [e[_layer] for _, e in _r["trace"]], _ls, color=_c,
                     label=f"coef {_coef}, layer {_layer}")
    _a1.set(xlabel="training step", ylabel="effective experts exp(H)", ylim=(0, 8.5))
    _a1.legend(fontsize=7, ncol=2)
    _sh = moe_runs[coef_pick.value]["shares"]
    _w = 0.4
    for _layer in range(2):
        _a2.bar(np.arange(8) + (_layer - 0.5) * _w, _sh[_layer], _w, label=f"layer {_layer}")
    _a2.axhline(1 / 8, color="k", ls=":", lw=1)
    _a2.set(xlabel="expert", ylabel="share of token slots", title=f"after training, coefficient {coef_pick.value}")
    _a2.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        "| aux coefficient | validation loss | effective experts (layer 0, layer 1) at the end |\n|---|---|---|\n"
        + "\n".join(f"| {c} | {r['val']:.3f} | {r['trace'][-1][1][0]:.2f}, {r['trace'][-1][1][1]:.2f} |"
                    for c, r in moe_runs.items())
        + f"\n\neach MoE model has {moe_runs[0.0]['params']:,} parameters, but a token passes through only 2 of 8 "
          "experts: about 25k active MLP parameters per layer against 34k for experiment 2's dense SwiGLU.")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does an unregularized router drift towards a few experts (think about
    which experts get gradient), and why is that bad even when the loss looks fine? why
    does the auxiliary loss use the router's *probabilities* $P_e$ and not only the counts
    $f_e$? at this scale the MoE doesn't beat the dense model: what would have to change for
    sparsity to pay off?
    """)
    return


if __name__ == "__main__":
    app.run()
