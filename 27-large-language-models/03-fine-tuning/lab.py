import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch import nn

    from mlfs import datasets, notebooks, ui
    from mlfs.llm import Llama  # yours, from topic 02
    from mlfs.text import BPETokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import estimate_loss, generate_cached, train_lm  # yours, from ch 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    CUT = int(0.9 * len(TEXT))
    # byte-level tokenizer (256 bytes, no merges) plus one special token that ends every document and response
    TOK = BPETokenizer(257, special_tokens=("<|endoftext|>",)).train(TEXT[:1000])
    EOT = TOK.encode("<|endoftext|>")[0]
    PRE_STEPS, SFT_STEPS = (200, 80) if OFFLINE else (800, 300)


@app.function
def make_tasks(text, cut, seed=0):
    """two synthetic instruction datasets, as {task: (train, test)} lists of (kind, prompt, response).

    'continue' (format): finish a line of the play from its first half; the base model can already continue
    text, so fine-tuning mostly teaches the chat format. test lines come from the held-out tenth of the text.
    'letters' (skill): capitalize a word, count its letters, or name its first letter; test words never appear
    in training prompts. both are generated here: there is no instruction dataset on this machine.
    """
    rng = np.random.default_rng(seed)

    def lines(t):
        return [ln.strip() for ln in t.splitlines() if len(ln.split()) >= 8 and not ln.strip().endswith(":")]

    def cont(t, n):
        ls = lines(t)
        out = []
        for i in rng.choice(len(ls), min(n, len(ls)), replace=False):
            w = ls[i].split()
            out.append(("continue", "Continue: " + " ".join(w[:len(w) // 2]), " ".join(w[len(w) // 2:])))
        return out

    words = sorted({w for w in re.findall(r"[a-z]+", text) if 3 <= len(w) <= 6})
    rng.shuffle(words)
    kinds = {"capitals": (lambda w: f"Write {w} in capitals.", str.upper),
             "count": (lambda w: f"How many letters are in {w}?", lambda w: str(len(w))),
             "first": (lambda w: f"What is the first letter of {w}?", lambda w: w[0])}

    def letters(ws, n):
        out = []
        for _ in range(n):
            k = list(kinds)[int(rng.integers(len(kinds)))]
            w = ws[int(rng.integers(len(ws)))]
            out.append((k, kinds[k][0](w), kinds[k][1](w)))
        return out

    n_test = min(150, len(words) // 4)
    return {"continue": (cont(text[:cut], 3000), cont(text[cut:], 150)),
            "letters": (letters(words[n_test:], 3000), letters(words[:n_test], 150))}


@app.function
def sft(model, train, steps, lr, *, batch_size=32, seed=0):
    """AdamW on the trainable parameters, response-only loss, batches sampled from `train`."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    g = np.random.default_rng(seed)
    model.train()
    for _ in range(steps):
        batch = [ex.build_sft_example(TOK, p, r) for _, p, r in (train[i] for i in g.integers(0, len(train), batch_size))]
        x, y = ex.collate_sft(batch)
        loss = ex.sft_loss(model(x)[0], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@app.function
@torch.no_grad()
def evaluate(model, test, held_out, n_generate=60):
    """(held-out response loss, exact-match accuracy of greedy answers, LM loss on held-out Shakespeare)."""
    model.eval()
    x, y = ex.collate_sft([ex.build_sft_example(TOK, p, r) for _, p, r in test])
    loss = ex.sft_loss(model(x)[0], y).item()
    hits = 0
    for _, p, r in test[:n_generate]:
        prompt = TOK.encode(f"User: {p}\nAssistant:")
        out = generate_cached(model, torch.tensor([prompt]), min(40, model.block_size - len(prompt)),
                              temperature=0.0)[0, len(prompt):].tolist()
        out = out[:out.index(EOT)] if EOT in out else out
        hits += TOK.decode(out).strip() == r
    lm = estimate_loss(model, held_out, model.block_size, batch_size=16, n_batches=10)
    return loss, hits / min(n_generate, len(test)), lm


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # fine-tuning · lab

    every experiment fine-tunes a small `Llama` (your topic 02 model: 3 layers, width 96)
    with **your** `apply_lora`, `build_sft_example`, `collate_sft`, `sft_loss`, `merge_lora`
    and `nf4_quantize`. first it's pretrained on tiny Shakespeare bytes, with your
    end-of-text token between speeches ({PRE_STEPS} steps; offline, a synthetic stand-in and
    fewer steps). the instruction data is **synthetic** (`make_tasks`): there is no
    instruction dataset on this machine. every number is one seed. for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. about three minutes.
    """)
    return


@app.cell
def _():
    speeches = [s for s in TEXT[:CUT].split("\n\n") if s.strip()]
    _ids = np.asarray(TOK.encode("<|endoftext|>".join(speeches)), dtype=np.int64)
    held = np.asarray(TOK.encode(TEXT[CUT:]), dtype=np.int64)
    torch.manual_seed(0)
    base = Llama(TOK.vocab_size, block_size=128, n_layers=3, n_heads=4, n_kv_heads=2, d_model=96)
    _t0 = time.perf_counter()
    train_lm(base, _ids, 128, batch_size=32, max_steps=PRE_STEPS, lr=3e-3, warmup_steps=50,
             eval_interval=PRE_STEPS, eval_batches=5)
    tasks = make_tasks(TEXT, CUT)
    _ex = tasks["letters"][0][0]
    mo.md(f"pretrained in {time.perf_counter() - _t0:.0f} s: {sum(p.numel() for p in base.parameters()):,} "
          f"parameters, held-out loss {estimate_loss(base, held, 128, batch_size=16, n_batches=10):.3f} nats per "
          f"byte. a training example, as the model sees it:\n\n```\nUser: {_ex[1]}\nAssistant: {_ex[2]}"
          "<|endoftext|>\n```")
    return base, held, tasks


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 1: rank against quality

    the same base fine-tuned {SFT_STEPS} steps on each task: fully (every parameter, lr
    $10^{{-3}}$), and with your LoRA at ranks 1, 4 and 16 on every linear layer (alpha 16,
    lr $3 \times 10^{{-3}}$). three numbers per run: the loss on held-out responses, the
    fraction of 60 held-out prompts answered exactly (greedy), and the language-modelling
    loss on held-out Shakespeare (did fine-tuning damage what pretraining taught?).
    """)
    return


@app.cell
def _():
    p1 = ui.predict("LoRA rank 1 trains 1.6% as many parameters as full fine-tuning. on which task will it come "
                    "close to full fine-tuning, and on which won't it? which run forgets the most Shakespeare?")
    p1
    return (p1,)


@app.cell
def _(base, held, p1, tasks):
    ui.gate(p1)
    configs = [("full", None), ("LoRA r=1", 1), ("LoRA r=4", 4), ("LoRA r=16", 16)]
    rank_rows = []
    _base_eval = {t: evaluate(base, te, held) for t, (_, te) in tasks.items()}
    for _task, (_tr, _te) in tasks.items():
        for _name, _r in configs:
            _m = copy.deepcopy(base)
            if _r is not None:
                ex.apply_lora(_m, r=_r, alpha=16.0)
            _n = sum(p.numel() for p in _m.parameters() if p.requires_grad)
            sft(_m, _tr, SFT_STEPS, 1e-3 if _r is None else 3e-3)
            rank_rows.append((_task, _name, _n, *evaluate(_m, _te, held)))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for _ax, _task in zip(_axes, tasks):
        _rows = [r for r in rank_rows if r[0] == _task]
        _ax.semilogx([r[2] for r in _rows[1:]], [r[3] for r in _rows[1:]], "o-", label="LoRA r = 1, 4, 16")
        _ax.axhline(_rows[0][3], color="C1", ls="--", label=f"full fine-tuning ({_rows[0][2]:,} params)")
        _ax.axhline(_base_eval[_task][0], color="gray", ls=":", label="base model")
        _ax.set(title=f"task: {_task}", xlabel="trainable parameters", ylabel="held-out response loss")
        _ax.set_xticks([r[2] for r in _rows[1:]], [f"r={c[1]}\n{r[2]:,}" for c, r in zip(configs[1:], _rows[1:])])
        _ax.minorticks_off()
        _ax.legend(fontsize=7)
    mo.vstack([_fig, mo.md(
        "| task | run | trainable parameters | response loss | exact match | Shakespeare loss |\n|---|---|---|---|---|---|\n"
        + "\n".join(f"| {t} | base model | 0 | {b[0]:.3f} | {b[1]:.2f} | {b[2]:.3f} |" for t, b in _base_eval.items()) + "\n"
        + "\n".join(f"| {t} | {n} | {k:,} | {l:.3f} | {'–' if t == 'continue' else f'{a:.2f}'} | {s:.3f} |"
                    for t, n, k, l, a, s in rank_rows)
        + "\n\n(exact match is meaningless for 'continue': nobody finishes a line of Shakespeare word for word. the "
          "LoRA runs on 'letters' scatter a lot from seed to seed; see the lab notes.)")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does rank barely matter for one task and a lot for the other? what does
    the base model already know how to do in each case? relate the Shakespeare column to
    Biderman et al. (2024), "LoRA learns less and forgets less". what would you try before
    concluding that LoRA can't learn the second task (think steps, learning rate, targets)?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: QLoRA and merging

    QLoRA freezes the base in 4 bits and trains LoRA on top. here every linear weight of the
    base is replaced by your `nf4_quantize` (blocks of 64) or by a uniform 4-bit grid with
    the same blocks, the Shakespeare loss of each quantized base is measured, and LoRA rank
    16 is trained on the NF4 base for the letters task. last, your `merge_lora` folds a LoRA
    model back into plain weights: same outputs, original size and speed.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("how much does rounding the base to 4 bits hurt its Shakespeare loss, NF4 against uniform? "
                    "does LoRA on the 4-bit base learn the task as well as on the full-precision one?")
    p2
    return (p2,)


@app.cell
def _(base, held, p2, tasks):
    ui.gate(p2)

    def _quantized(fn):
        _m = copy.deepcopy(base)
        with torch.no_grad():
            for _mod in _m.modules():
                if isinstance(_mod, nn.Linear) and _mod is not _m.lm_head:
                    _mod.weight.copy_(fn(_mod.weight))
        return _m

    def _uniform4(w, block_size=64):
        _b = w.reshape(-1, block_size)
        _s = _b.abs().amax(1, keepdim=True).clamp_min(1e-12) / 7
        return (torch.round(_b / _s).clamp(-7, 7) * _s).reshape(w.shape)

    _q = {"float32": base, "NF4 (your nf4_quantize)": _quantized(ex.nf4_quantize),
          "uniform 4-bit": _quantized(_uniform4)}
    _lm = {k: estimate_loss(m, held, 128, batch_size=16, n_batches=10) for k, m in _q.items()}
    _ws = [(_a.weight.detach(), _b.weight.detach()) for _a, _b in zip(base.modules(), _q["NF4 (your nf4_quantize)"].modules())
           if isinstance(_a, nn.Linear) and _a is not base.lm_head]
    _wu = [_b.weight.detach() for _b in _q["uniform 4-bit"].modules() if isinstance(_b, nn.Linear) and _b is not _q["uniform 4-bit"].lm_head]
    _norm = sum(float((w ** 2).sum()) for w, _ in _ws)
    _err = {"float32": 0.0, "NF4 (your nf4_quantize)": sum(float(((q - w) ** 2).sum()) for w, q in _ws) / _norm,
            "uniform 4-bit": sum(float(((u - w) ** 2).sum()) for (w, _), u in zip(_ws, _wu)) / _norm}
    _tr, _te = tasks["letters"]
    _ql = ex.apply_lora(copy.deepcopy(_q["NF4 (your nf4_quantize)"]), r=16, alpha=16.0)
    sft(_ql, _tr, SFT_STEPS, 3e-3)
    _ql_eval = evaluate(_ql, _te, held)
    _lo = ex.apply_lora(copy.deepcopy(base), r=16, alpha=16.0)
    sft(_lo, _tr, SFT_STEPS, 3e-3)
    _lo_eval = evaluate(_lo, _te, held)
    _prompt = torch.tensor([TOK.encode(f"User: {_te[0][1]}\nAssistant:")])
    with torch.no_grad():
        _before = _lo(_prompt)[0]
    _n_lora = sum(p.numel() for p in _lo.parameters())

    def _speed(m):
        _ts = []
        for _ in range(3):
            _t0 = time.perf_counter()
            generate_cached(m, _prompt, 60, temperature=0.0)
            _ts.append(time.perf_counter() - _t0)
        return min(_ts)

    _t_lora = _speed(_lo)
    ex.merge_lora(_lo)
    with torch.no_grad():
        _diff = (_lo(_prompt)[0] - _before).abs().max().item()
    _t_merged = _speed(_lo)
    mo.md("| base weights | relative squared weight error | Shakespeare loss |\n|---|---|---|\n"
          + "\n".join(f"| {k} | {100 * _err[k]:.2f}% | {v:.3f} |" for k, v in _lm.items())
          + f"\n\nLoRA r=16 on the letters task: on the float32 base, response loss {_lo_eval[0]:.3f}, exact match "
          f"{_lo_eval[1]:.2f}; on the NF4 base (QLoRA), {_ql_eval[0]:.3f} and {_ql_eval[1]:.2f}. "
          f"4-bit weights take 4.5 bits per weight with a 32-bit scale per block of 64 (QLoRA then quantizes the scales too), against 32 for float32.\n\n"
          f"merging: largest change in the logits {_diff:.1e}; parameters {_n_lora:,} with LoRA, "
          f"{sum(p.numel() for p in _lo.parameters()):,} after merging; 60 tokens of greedy generation "
          f"{1e3 * _t_lora:.0f} ms with LoRA layers, {1e3 * _t_merged:.0f} ms merged.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does NF4 round trained weights more accurately than a uniform grid
    with the same blocks (what does their distribution look like?), and why does that barely
    show in the loss of this model? why can LoRA on a quantized base still learn: where do its
    gradients flow, and in what precision are $A$ and $B$? merging removes the extra matmuls:
    why can't you merge a QLoRA model and keep the 4-bit savings?
    """)
    return


if __name__ == "__main__":
    app.run()
