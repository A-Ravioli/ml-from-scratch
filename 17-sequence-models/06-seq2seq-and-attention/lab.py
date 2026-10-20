import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import datetime
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december"]
    DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def _ordinal(d):
        return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"

    # six ways of writing a date; the target is always ISO 8601, yyyy-mm-dd
    FORMATS = {
        "march 3, 1987": lambda t: f"{MONTHS[t.month - 1]} {t.day}, {t.year}",
        "3 mar 1987": lambda t: f"{t.day} {MONTHS[t.month - 1][:3]} {t.year}",
        "03/03/1987": lambda t: f"{t.month:02d}/{t.day:02d}/{t.year}",
        "3.3.1987": lambda t: f"{t.day}.{t.month}.{t.year}",
        "tuesday, march 3, 1987": lambda t: f"{DAYS[t.weekday()]}, {MONTHS[t.month - 1]} {t.day}, {t.year}",
        "the 3rd of march 1987": lambda t: f"the {_ordinal(t.day)} of {MONTHS[t.month - 1]} {t.year}",
    }

    def _make(n, rng):
        lo, hi = datetime.date(1950, 1, 1).toordinal(), datetime.date(2049, 12, 31).toordinal()
        out = []
        for _ in range(n):
            t = datetime.date.fromordinal(int(rng.integers(lo, hi + 1)))
            f = list(FORMATS)[int(rng.integers(len(FORMATS)))]
            out.append((FORMATS[f](t), t.isoformat(), f))
        return out

    _rng = np.random.default_rng(0)
    TRAIN = _make(20_000, _rng)
    _seen = {s for s, _, _ in TRAIN}
    TEST = [it for it in _make(1_000, _rng) if it[0] not in _seen][:600]  # no test string appears in training
    PAD, BOS, EOS = 0, 1, 2
    SRC_VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted({c for s, _, _ in TRAIN + TEST for c in s})
    TGT_VOCAB = ["<pad>", "<bos>", "<eos>"] + sorted("0123456789-")
    S2I = {c: i for i, c in enumerate(SRC_VOCAB)}
    T2I = {c: i for i, c in enumerate(TGT_VOCAB)}
    FIELDS = {"year": slice(0, 4), "month": slice(5, 7), "day": slice(8, 10)}


@app.function
def encode_batch(items):
    """(src (B, S) padded, src_mask (B, S), tgt_in (B, 11) = bos + 10 chars, tgt_out (B, 11) = 10 chars + eos)."""
    S = max(len(s) for s, _, _ in items)
    src = torch.zeros(len(items), S, dtype=torch.long)
    mask = torch.zeros(len(items), S, dtype=torch.bool)
    for b, (s, _, _) in enumerate(items):
        src[b, : len(s)] = torch.tensor([S2I[c] for c in s])
        mask[b, : len(s)] = True
    tgt = torch.tensor([[BOS] + [T2I[c] for c in t] + [EOS] for _, t, _ in items])
    return src, mask, tgt[:, :-1], tgt[:, 1:]


@app.function
def field_accuracy(model, items):
    """exact match of the whole date and of each field, decoding greedily with your greedy_decode."""
    src, mask, _, tout = encode_batch(items)
    out = ex.greedy_decode(model, src, BOS, EOS, max_len=12, src_mask=mask)
    pred = [o[1:] + [PAD] * (11 - len(o[1:])) for o in out]
    pred = torch.tensor([p[:11] for p in pred])
    res = {name: (pred[:, sl] == tout[:, sl]).all(1).float().mean().item() for name, sl in FIELDS.items()}
    res["exact"] = (pred == tout).all(1).float().mean().item()
    return res


@app.function
def train_dates(attention, steps, *, seed=0, log_every=100):
    """your Seq2Seq (embed 32, hidden 128) trained with Adam on random minibatches of 64; returns (model, log)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    model = ex.Seq2Seq(len(SRC_VOCAB), len(TGT_VOCAB), embed_dim=32, hidden_dim=128, attention=attention)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    log = []
    for step in range(1, steps + 1):
        src, mask, tin, tout = encode_batch([TRAIN[i] for i in rng.integers(0, len(TRAIN), 64)])
        loss = F.cross_entropy(model(src, tin, src_mask=mask).reshape(-1, len(TGT_VOCAB)), tout.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            log.append((step, field_accuracy(model, TEST[:200])))
    return model, log


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # seq2seq & attention · lab

    every experiment here uses **your** `Seq2Seq`, `BahdanauAttention`, `greedy_decode` and
    `beam_search` from `exercise.py`, on a synthetic translation task: dates written in six
    styles ("march 3, 1987", "3 mar 1987", "03/03/1987", "3.3.1987", "tuesday, march 3, 1987",
    "the 3rd of march 1987"), between 1950 and 2049, character by character, to be rewritten as
    ISO dates, "1987-03-03". 20,000 training pairs, 600 test pairs, none of which occurs verbatim in
    the training set (about 9% of randomly drawn test strings would). for each experiment: write
    down a prediction, lock it in, run it, then explain what happened. compare your explanations
    with `solutions/lab_notes.md` afterwards. the lab takes about a minute and a half on an idle
    laptop CPU.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: with and without attention

    the same encoder–decoder (character embeddings of size 32, GRUs with 128 units), trained
    with teacher forcing, Adam (learning rate $3 \times 10^{-3}$) and batches of 64: once with
    attention for 400 steps, once without for 1,500 steps. every 100 steps we decode 200 test
    dates greedily and score the year, the month, the day and the whole date.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "without attention, the decoder starts from the encoder's last state and must get everything from that one "
        "vector. which part of the date will it learn first, and how many steps will it need compared with attention?"
    )
    p1
    return (p1,)


@app.cell
def _(p1):
    ui.gate(p1)
    _t0 = time.perf_counter()
    att_model, _log_att = train_dates(True, 400)
    plain_model, _log_plain = train_dates(False, 1500)
    _secs = time.perf_counter() - _t0
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    for _ax, (_name, _log) in zip(_axes, [("with attention", _log_att), ("without attention", _log_plain)]):
        _steps = [s for s, _ in _log]
        for _k, _style in [("year", "C0"), ("month", "C1"), ("day", "C2"), ("exact", "k")]:
            _ax.plot(_steps, [r[_k] for _, r in _log], "o-" if _k == "exact" else "-", color=_style, ms=3,
                     label=_k if _k != "exact" else "whole date")
        _ax.set(title=_name, xlabel="training step", ylim=(-0.02, 1.02))
        _ax.legend(fontsize=8)
    _axes[0].set_ylabel("test accuracy (greedy)")
    _final = {"with attention (400 steps)": field_accuracy(att_model, TEST),
              "without attention (1,500 steps)": field_accuracy(plain_model, TEST)}
    _per_format = []
    for _f in FORMATS:
        _items = [it for it in TEST if it[2] == _f]
        _per_format.append(f"| {_f} | {len(_items)} | {field_accuracy(att_model, _items)['exact']:.2f} | "
                           f"{field_accuracy(plain_model, _items)['exact']:.2f} |")
    mo.vstack([_fig, mo.md(
        f"{_secs:.0f} s of training\n\n| model | year | month | day | whole date |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {k} | {v['year']:.2f} | {v['month']:.2f} | {v['day']:.2f} | **{v['exact']:.2f}** |"
                    for k, v in _final.items())
        + "\n\n| format | test dates | with attention | without |\n|---|---|---|---|\n" + "\n".join(_per_format))])
    return att_model, plain_model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** without attention, why does the year come first and the day last? (where
    is each field in the source strings, and how far is it from the encoder's final state?)
    with attention, the decoder can read any source position at every step: why does that
    make all three fields equally easy, and the whole task about ten times faster to learn?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the attention looks at

    the attention weights `last_attention` of the model with attention, for a few test dates:
    one row per output character, one column per input character.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "nobody told the model which input characters correspond to which output characters. when it writes the "
        "month digits of 'tuesday, march 3, 1987', where will it look? and for '03/03/1987'?"
    )
    p2
    return (p2,)


@app.cell
def _():
    pick = mo.ui.dropdown(list(FORMATS), value="tuesday, march 3, 1987", label="format of the first example")
    pick
    return (pick,)


@app.cell
def _(att_model, p2, pick):
    ui.gate(p2)
    _first = next(it for it in TEST if it[2] == pick.value)
    _others = [next(it for it in TEST if it[2] == f) for f in ["03/03/1987", "the 3rd of march 1987"] if f != pick.value]
    _items = [_first] + _others[:2]
    _fig, _axes = plt.subplots(1, len(_items), figsize=(4.2 * len(_items), 3.6))
    for _ax, _it in zip(np.atleast_1d(_axes), _items):
        _src, _mask, _tin, _ = encode_batch([_it])
        _out = ex.greedy_decode(att_model, _src, BOS, EOS, max_len=12, src_mask=_mask)[0]
        with torch.no_grad():
            att_model(_src, torch.tensor([_out[:-1]]), src_mask=_mask)
        _w = att_model.last_attention[0].numpy()
        _ax.imshow(_w, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        _ax.set_xticks(range(len(_it[0])), list(_it[0]), fontsize=7)
        _ax.set_yticks(range(len(_out) - 1), [TGT_VOCAB[i] if i > 2 else "⏎" for i in _out[1:]], fontsize=8)
        _ax.set_title(f"{_it[0]!r}", fontsize=8)
    _fig.supxlabel("input characters", fontsize=9)
    _fig.supylabel("output characters", fontsize=9)
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the attention for the year digits sits on the year characters in every
    format: why does the model learn an alignment it was never shown? what does it look at to
    write the month digits when the month is spelled out, and what does that tell you about
    what the encoder states at those positions contain? where is the attention fuzzy, and why
    might that be harmless?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: greedy decoding against beam search

    both models from experiment 1 decode 120 test dates with beam sizes 1 (greedy), 3 and 10
    (length penalty 1; every correct output has the same 11 tokens, so the length penalty can only matter
    for malformed outputs). the step
    function re-runs your `Seq2Seq` on each prefix and returns the log-softmax of the last step.
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "beam search keeps 10 hypotheses instead of 1. how much will it improve the exact-match accuracy of the "
        "attention model? of the under-trained model without attention?"
    )
    p3
    return (p3,)


@app.cell
def _(att_model, p3, plain_model):
    ui.gate(p3)
    _t0 = time.perf_counter()
    _items = TEST[:120]
    _rows = []
    for _name, _m in [("with attention", att_model), ("without attention", plain_model)]:
        _m.eval()
        _accs = []
        for _beam in [1, 3, 10]:
            _hits = 0
            for _it in _items:
                _src, _mask, _, _tout = encode_batch([_it])

                def _step(prefixes, _src=_src, _mask=_mask, _m=_m):
                    with torch.no_grad():
                        _n = len(prefixes)
                        _lg = _m(_src.expand(_n, -1), torch.tensor(prefixes), src_mask=_mask.expand(_n, -1))
                    return F.log_softmax(_lg[:, -1], dim=-1)

                _best = ex.beam_search(_step, BOS, EOS, beam_size=_beam, max_len=12)[0][0]
                _hits += _best[1:] == _tout[0].tolist()
            _accs.append(_hits / len(_items))
        _m.train()
        _rows.append(f"| {_name} | " + " | ".join(f"{a:.3f}" for a in _accs) + " |")
    mo.md(f"{time.perf_counter() - _t0:.0f} s\n\n| model | beam 1 (greedy) | beam 3 | beam 10 |\n|---|---|---|---|\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** beam search finds outputs with higher total log-probability than greedy
    decoding. why can't that rescue a model that doesn't *know* the day? when does beam search
    help a lot (think of translation between natural languages, where many outputs are
    plausible and an early word constrains the rest)? chapter 27 uses your `beam_search` on
    language models, where it has a well-known failure mode: what might it be?
    """)
    return


if __name__ == "__main__":
    app.run()
