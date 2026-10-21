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

    from mlfs import notebooks, ui
    from mlfs.transformer import GPT  # yours, from ch 18 · 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    V, T, P = 48, 32, 10  # vocabulary, training length, period of the probe sequences
    STEPS = 800


@app.function
def periodic_batch(B, generator):
    """B sequences of T random tokens that repeat with a random period between 4 and 16."""
    x = torch.randint(0, V, (B, T), generator=generator)
    lens = torch.randint(4, 17, (B,), generator=generator)
    for b in range(B):
        x[b] = x[b, :int(lens[b])].repeat(T // int(lens[b]) + 1)[:T]
    return x


@app.function
def probe_batch(B, seed):
    """B sequences of 2P tokens: P random tokens, then the same P again."""
    x = torch.randint(0, V, (B, 2 * P), generator=torch.Generator().manual_seed(seed))
    x[:, P:] = x[:, :P]
    return x


@app.function
def second_half_loss(logits, probe):
    """next-token loss on the repeated half (predicting tokens P+1 .. 2P-1)."""
    return F.cross_entropy(logits[:, P:-1].reshape(-1, V), probe[:, P + 1:].reshape(-1)).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # mechanistic interpretability · lab

    the experiments open up a 2-layer GPT (your chapter 18 model, width 64, 4 heads per
    layer) trained on **synthetic** sequences: random tokens from a vocabulary of 48 that
    repeat with a random period between 4 and 16. the first period is unpredictable; after
    that, every token can be predicted by finding the previous occurrence of the current token
    and copying whatever came next. the period changes from sequence to sequence, so "look
    $k$ positions back" doesn't work: the model has to match *content*. your
    `attention_patterns`, `induction_score`, `prev_token_score`, `ablate_heads` and
    `position_patching` find out how it does it. for each experiment: write down a prediction,
    lock it in, run it, then explain. about a minute on a laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how does in-context copying arrive?

    800 training steps (batches of 64, AdamW, lr $10^{-3}$). every 20 steps, a fixed probe
    set (64 sequences of 10 random tokens repeated once) measures the loss on the repeated
    half, and your scores read the heads: the best induction score in layer 1 and the best
    previous-token score in layer 0.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("will the loss on the repeated half fall gradually over the 800 steps, or suddenly? when the "
                    "induction score appears, what will the loss do at the same moment?")
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    model = GPT(V, block_size=T, n_layers=2, n_heads=4, d_model=64)
    _opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    _g = torch.Generator().manual_seed(0)
    probe = probe_batch(64, 123)
    history = {"step": [], "loss": [], "induction": [], "prev": []}
    for _step in range(STEPS + 1):
        if _step % 20 == 0:
            model.eval()
            with torch.no_grad():
                _l = second_half_loss(model(probe)[0], probe)
            _pats = ex.attention_patterns(model, probe)
            history["step"].append(_step)
            history["loss"].append(_l)
            history["induction"].append(ex.induction_score(_pats[1], P).max().item())
            history["prev"].append(ex.prev_token_score(_pats[0]).max().item())
            model.train()
        _x = periodic_batch(64, _g)
        _loss = F.cross_entropy(model(_x[:, :-1])[0].reshape(-1, V), _x[:, 1:].reshape(-1))
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    model.eval()
    _fig, _ax = plt.subplots(figsize=(7, 3.3))
    _ax.plot(history["step"], history["loss"], "k-", label="loss on the repeated half (left)")
    _ax.axhline(np.log(V), color="gray", ls=":", lw=0.8)
    _ax.set(xlabel="training step", ylabel="loss (nats)")
    _ax2 = _ax.twinx()
    _ax2.plot(history["step"], history["induction"], "C3-", label="best induction score, layer 1")
    _ax2.plot(history["step"], history["prev"], "C0--", label="best previous-token score, layer 0")
    _ax2.set(ylabel="score", ylim=(0, 1))
    _h1, _l1 = _ax.get_legend_handles_labels()
    _h2, _l2 = _ax2.get_legend_handles_labels()
    _ax.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    _rows = [i for i, s in enumerate(history["step"]) if s in (0, 100, 140, 180, 220, 260, 300, 400, 800)]
    mo.vstack([mo.md(f"trained in {time.perf_counter() - _t0:.0f} s."), _fig,
               mo.md("| step | repeated-half loss | induction (L1) | prev-token (L0) |\n|---|---|---|---|\n"
                     + "\n".join(f"| {history['step'][i]} | {history['loss'][i]:.3f} | {history['induction'][i]:.2f}"
                                 f" | {history['prev'][i]:.2f} |" for i in _rows))])
    return model, probe


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the loss sit near $\log 48 = 3.87$ for a while and then drop within
    a hundred steps or so? why would the induction score in layer 1 and the previous-token
    score in layer 0 rise *together*, rather than one first? (what is either one good for
    without the other?)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: reading the heads

    the trained model's attention on one probe sequence (10 random tokens, then the same 10),
    and every head's two scores over the probe set.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("which layer will hold the induction head, and what will a layer-0 head need to do for it to "
                    "work? sketch the attention pattern of an induction head on a repeated sequence.")
    p2
    return (p2,)


@app.cell
def _(model, p2, probe):
    ui.gate(p2)
    _pats = ex.attention_patterns(model, probe)
    scores = {(l, h): (ex.prev_token_score(_pats[l])[h].item(), ex.induction_score(_pats[l], P)[h].item())
              for l in range(2) for h in range(4)}
    prev_head = max([k for k in scores if k[0] == 0], key=lambda k: scores[k][0])
    ind_head = max([k for k in scores if k[0] == 1], key=lambda k: scores[k][1])
    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for _ax, (_l, _h), _title in zip(_axes, [prev_head, ind_head], ["previous-token head", "induction head"]):
        _ax.imshow(_pats[_l][0, _h].numpy(), cmap="viridis", vmin=0, vmax=1)
        _ax.set(title=f"layer {_l} head {_h}: {_title}", xlabel="key position", ylabel="query position",
                xticks=range(0, 2 * P, 2), yticks=range(0, 2 * P, 2))
        _ax.axvline(P - 0.5, color="w", lw=0.5)
        _ax.axhline(P - 0.5, color="w", lw=0.5)
    mo.vstack([_fig, mo.md("| layer | head | prev-token score | induction score |\n|---|---|---|---|\n"
                           + "\n".join(f"| {l} | {h} | {s[0]:.2f} | {s[1]:.2f} |" for (l, h), s in scores.items()))])
    return ind_head, prev_head, scores


@app.cell
def _():
    head_pick = mo.ui.dropdown([f"layer {l}, head {h}" for l in range(2) for h in range(4)], value="layer 1, head 0",
                               label="look at any head")
    head_pick
    return (head_pick,)


@app.cell
def _(head_pick, model, p2, probe):
    ui.gate(p2)
    _l, _h = int(head_pick.value.split()[1][0]), int(head_pick.value.split()[-1])
    _w = ex.attention_patterns(model, probe)[_l]
    _fig, _ax = plt.subplots(figsize=(3.8, 3.6))
    _ax.imshow(_w[1, _h].numpy(), cmap="viridis", vmin=0, vmax=1)
    _ax.set(title=head_pick.value, xlabel="key position", ylabel="query position", xticks=range(0, 2 * P, 2),
            yticks=range(0, 2 * P, 2))
    mo.vstack([_fig, mo.md(f"previous-token score {ex.prev_token_score(_w)[_h]:.2f}, induction score "
                           f"{ex.induction_score(_w, P)[_h]:.2f} (over the whole probe set; the picture is sequence 1)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the induction head attends from each position in the second half to one
    position *after* the earlier copy of the current token. its query is "the token I am now";
    what must the key at the target position contain for the match to work, and which head
    wrote it there? why can't a single layer of attention implement this?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: are those heads the cause?

    two interventions. **ablation:** your `ablate_heads` zeroes the best induction head, then
    all four heads of layer 1, then the best previous-token head, then all four heads of
    layer 0, and measures the repeated-half loss. **patching:** in a corrupted copy of the probe, the token at position
    $j + 1 = 4$ (in the first half) is replaced, so at position $P + j = 13$ the model should
    now predict the replacement $y$ instead of the true next token $x_4$; the metric is
    $\text{logit}(x_4) - \text{logit}(y)$ at position 13. your `position_patching` restores
    the clean activation of the embedding, block 0 or block 1 at one position at a time.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("ablating the induction head breaks copying. what happens if you ablate the previous-token "
                    "head in layer 0 instead? and in the patching grid, which positions of block 0 will restore the "
                    "clean prediction?")
    p3
    return (p3,)


@app.cell
def _(ind_head, model, p3, prev_head, probe, scores):
    ui.gate(p3)
    _cases = {"none": [], f"best induction head {ind_head}": [ind_head],
              "all four layer-1 heads": [(1, h) for h in range(4)],
              f"best prev-token head {prev_head}": [prev_head], "all four layer-0 heads": [(0, h) for h in range(4)]}
    ablation = {}
    with torch.no_grad():
        for _name, _heads in _cases.items():
            ablation[_name] = second_half_loss(ex.ablate_heads(model, probe, _heads)[0], probe)
    _j = 3
    _clean = probe[:16]
    _corrupt = _clean.clone()
    _corrupt[:, _j + 1] = (_clean[:, _j + 1] + 1 + torch.arange(16) % (V - 1)) % V
    _true, _repl = _clean[:, _j + 1], _corrupt[:, _j + 1]

    def _metric(out):
        _lg = out[0][:, P + _j]
        return (_lg.gather(1, _true[:, None]) - _lg.gather(1, _repl[:, None])).mean()

    _names = ["tok_emb", "blocks.0", "blocks.1"]
    grid = ex.position_patching(model, _clean, _corrupt, _metric, _names)
    with torch.no_grad():
        _base = (float(_metric(model(_clean))), float(_metric(model(_corrupt))))
    _fig, _ax = plt.subplots(figsize=(8, 2.4))
    _im = _ax.imshow(grid, cmap="RdBu", vmin=-abs(grid).max(), vmax=abs(grid).max(), aspect="auto")
    _ax.set(yticks=range(3), yticklabels=_names, xticks=range(2 * P), xlabel="patched position")
    _fig.colorbar(_im, label="logit(x4) - logit(y) at 13")
    mo.vstack([mo.md("| ablated | repeated-half loss |\n|---|---|\n"
                     + "\n".join(f"| {k} | {v:.3f} |" for k, v in ablation.items())),
               mo.md(f"metric on the clean run **{_base[0]:.2f}**, on the corrupted run **{_base[1]:.2f}**:"), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does knocking out the best induction head barely matter, while
    knocking out one layer-0 head, which never attends to the earlier copy, hurts much
    more? (look at the score table: how many heads in each layer do each job?) in the patching
    grid, why does block 0 restore the prediction at position 4 (and not at 3, the matching
    token) while block 1 restores it only at position 13? what does that say about *where* the
    information travels?
    """)
    return


if __name__ == "__main__":
    app.run()
