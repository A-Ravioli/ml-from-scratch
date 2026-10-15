import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from mlfs import checks, datasets
    from mlfs.stats import bootstrap_ci  # yours, from ch 05 · topic 03
    from mlfs.text import CharTokenizer, NGramLM, tune_lambdas  # yours, from topics 01-02
    from mlfs.torchnn import get_device  # yours, from ch 15 · topic 07
    from mlfs.train import CosineSchedule  # yours, from ch 15 · topic 04

    plt.rcParams["figure.autolayout"] = True

    # small CPU config (offline, tests, verify) vs full config (the real tiny Shakespeare). set
    # MLFS_CAPSTONE=small to force the small config on the real data.
    OFFLINE = datasets.is_offline()
    FULL = not OFFLINE and os.environ.get("MLFS_CAPSTONE", "full") != "small"
    CONFIG = (dict(embed_dim=64, hidden_dim=512, num_layers=2, dropout=0.25, steps=2500, batch_size=64, seq_len=128,
                   lr=2e-3, warmup=100, eval_every=250, ngram_n=7, device=str(get_device()))
              if FULL else
              dict(embed_dim=32, hidden_dim=128, num_layers=2, dropout=0.0, steps=300, batch_size=32, seq_len=64,
                   lr=3e-3, warmup=20, eval_every=100, ngram_n=4, device="cpu"))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 17 · capstone: a character-level LSTM language model (reference solution)

    read [`brief.md`](brief.md) first. you'll train a two-layer LSTM to predict the next
    character of Shakespeare, measure it in bits per character with an honest error bar,
    compare it with an n-gram model built from your own `mlfs.text.NGramLM`, and generate text
    with temperature sampling. the data loading and the split are written for you; the model,
    the training loop, the evaluation and the sampler are yours.

    | stage | you build |
    |---|---|
    | 1 | `CharLSTM`: embedding → `nn.LSTM` → dropout → linear, with a carried state |
    | 2 | `get_batch`: random windows of the training text, inputs and next-character targets |
    | 3 | `train_lm`: AdamW, warmup + cosine schedule, clipping, validation, best checkpoint |
    | 4 | `char_nll`, `bpc_ci`: per-character losses with the state carried, and a block-bootstrap interval |
    | 5 | `sample`: temperature sampling, greedy at temperature 0 |

    when the checks are green, scroll down to train, evaluate and write up.
    """)
    return


@app.class_definition
class CharLSTM(nn.Module):
    """embedding -> nn.LSTM (batch-first) -> dropout -> linear over the vocabulary."""

    def __init__(self, vocab_size, *, embed_dim=64, hidden_dim=512, num_layers=2, dropout=0.0):
        """stage 1: `embed`, `lstm` (nn.LSTM with dropout between layers), `drop` (nn.Dropout), `head` (nn.Linear(H, V))."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, state=None):
        """stage 1: idx (B, T) int64 -> (logits (B, T, V), state (h, c)); state=None starts from zeros."""
        out, state = self.lstm(self.embed(idx), state)
        return self.head(self.drop(out)), state


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", CharLSTM=CharLSTM)
    return


@app.function
def get_batch(ids, batch_size, seq_len, *, generator=None):
    """stage 2: B random windows: x = ids[s:s+L], y = ids[s+1:s+L+1], start s uniform in [0, len - L - 1]; int64 (B, L)."""
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    starts = torch.randint(0, len(ids) - seq_len, (batch_size,), generator=generator)
    offsets = torch.arange(seq_len)
    x = ids[starts[:, None] + offsets]
    y = ids[starts[:, None] + offsets + 1]
    return x, y


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", get_batch=get_batch)
    return


@app.function
def train_lm(model, train_ids, *, steps, batch_size=64, seq_len=128, lr=2e-3, weight_decay=0.01, warmup=100, clip=1.0,
             val_ids=None, eval_every=None, device="cpu", seed=0, callback=None):
    """stage 3: AdamW + CosineSchedule (warmup), clipping; every eval_every steps the val loss, keeping the best weights; returns history."""
    g = torch.Generator().manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineSchedule(opt, total_steps=steps, warmup_steps=warmup, min_lr=lr / 10)
    history = {"step": [], "train_loss": [], "val_step": [], "val_loss": [], "lr": []}
    if val_ids is not None:
        val_g = torch.Generator().manual_seed(seed + 1)  # the same validation windows every time
        val_batches = [get_batch(val_ids, batch_size, seq_len, generator=val_g) for _ in range(8)]
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        model.train()
        x, y = get_batch(train_ids, batch_size, seq_len, generator=g)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step()
        history["step"].append(step)
        history["train_loss"].append(loss.item())
        if val_ids is not None and eval_every and (step % eval_every == 0 or step == steps):
            model.eval()
            with torch.no_grad():
                v = float(np.mean([F.cross_entropy(model(xv.to(device))[0].reshape(-1, logits.shape[-1]),
                                                   yv.to(device).reshape(-1)).item() for xv, yv in val_batches]))
            history["val_step"].append(step)
            history["val_loss"].append(v)
            if v < best:
                best, best_state = v, copy.deepcopy(model.state_dict())
        if callback is not None and callback(step, history):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", train_lm=train_lm, CharLSTM=CharLSTM, get_batch=get_batch)
    return


@app.function
def char_nll(model, ids, *, seq_len=256, device="cpu"):
    """stage 4: (len(ids) - 1,) float64 array: -ln P(ids[t] | ids[:t]) for t >= 1, one stream, chunks of seq_len, state carried."""
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.int64)
    was_training = model.training
    model.eval()
    out, state = [], None
    try:
        with torch.no_grad():
            for i in range(0, len(ids) - 1, seq_len):
                x = ids[i:i + seq_len][None].to(device)
                y = ids[i + 1:i + 1 + seq_len].to(device)
                logits, state = model(x[:, : len(y)], state)
                out.append(F.cross_entropy(logits[0], y, reduction="none").cpu().double().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(out) if out else np.zeros(0)


@app.function
def bpc_ci(nll, *, block_size=2000, n_boot=2000, alpha=0.05, rng=None):
    """stage 4: (bits per char, lo, hi): mean nll / ln 2, with a bootstrap CI that resamples contiguous blocks of ~block_size."""
    nll = np.asarray(nll, dtype=np.float64)
    n_blocks = max(len(nll) // block_size, 2)
    blocks = np.array_split(nll, n_blocks)  # every character counts; blocks differ in size by at most one
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=np.float64)
    lo, hi = bootstrap_ci((sums, counts), lambda s, c: s.sum() / c.sum() / math.log(2), n_boot=n_boot, alpha=alpha,
                          rng=rng)
    return float(nll.mean() / math.log(2)), float(lo), float(hi)


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", char_nll=char_nll, bpc_ci=bpc_ci, CharLSTM=CharLSTM)
    return


@app.function
def sample(model, prompt_ids, n_tokens, *, temperature=1.0, generator=None, device="cpu"):
    """stage 5: n_tokens new ids (a list), each drawn from softmax(logits / temperature); temperature 0 is greedy."""
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            x = torch.as_tensor(list(prompt_ids), dtype=torch.int64)[None].to(device)
            logits, state = model(x)
            last = logits[0, -1]
            for _ in range(n_tokens):
                if temperature == 0:
                    nxt = int(last.argmax())
                else:
                    probs = torch.softmax(last.cpu().double() / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, 1, generator=generator))
                out.append(nxt)
                logits, state = model(torch.tensor([[nxt]], device=device), state)
                last = logits[0, -1]
    finally:
        model.train(was_training)
    return out


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", sample=sample, CharLSTM=CharLSTM)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## the data (given)

    tiny Shakespeare, split by position exactly like chapter 04's capstone: the first 80%
    trains, the next 10% is validation (choosing the checkpoint and the n-gram weights), the last
    10% is the test set, used once. characters are ids through your `CharTokenizer`, trained on
    the whole text so that every character has an id (the vocabulary is just the 65 symbols).
    """)
    return


@app.cell
def _():
    text = datasets.tiny_shakespeare()
    tok = CharTokenizer().train(text)
    ids_all = np.array(tok.encode(text), dtype=np.int64)
    _a, _b = int(0.8 * len(ids_all)), int(0.9 * len(ids_all))
    train_ids, val_ids, test_ids = ids_all[:_a], ids_all[_a:_b], ids_all[_b:]
    mo.md(f"{'**tiny Shakespeare**' if not OFFLINE else '**synthetic stand-in (offline)**'}: {len(text):,} characters, "
          f"vocabulary {tok.vocab_size} · train {len(train_ids):,} · validation {len(val_ids):,} · test {len(test_ids):,}"
          f" · config: **{'full' if FULL else 'small'}** {CONFIG}")
    return test_ids, text, tok, train_ids, val_ids


@app.cell
def _(test_ids, text, train_ids, val_ids):
    # baseline (given): your interpolated NGramLM on the same split, one sequence per line, weights tuned
    # on validation. per-character losses are aligned with the text ('</s>' stands for the newline).
    _t0 = time.perf_counter()
    _a, _b = len(train_ids), len(train_ids) + len(val_ids)
    ngram = NGramLM(CONFIG["ngram_n"]).fit(text[:_a].split("\n"))
    tune_lambdas(ngram, text[_a:_b].split("\n"))

    def ngram_nll(block):
        out = []
        for line in block.split("\n"):
            padded = ["<s>"] * (ngram.n - 1) + list(line) + ["</s>"]
            out += [-ngram.logprob(padded[t - ngram.n + 1:t], padded[t]) for t in range(ngram.n - 1, len(padded))]
        return np.array(out[: len(block)])  # the last piece has no newline: drop its '</s>'

    ngram_val, ngram_test = ngram_nll(text[_a:_b]), ngram_nll(text[_b:])
    mo.md(f"order-{ngram.n} interpolated n-gram (EM weights {np.round(ngram.lambdas, 2).tolist()}): validation "
          f"**{ngram_val.mean() / math.log(2):.3f}** bits per character, test **{ngram_test.mean() / math.log(2):.3f}** "
          f"({time.perf_counter() - _t0:.0f} s)")
    return ngram_test, ngram_val


@app.cell
def _(tok, train_ids, val_ids):
    # training (yours, from stage 3). the full config takes about a quarter of an hour on a laptop GPU (MPS).
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    model = CharLSTM(tok.vocab_size, embed_dim=CONFIG["embed_dim"], hidden_dim=CONFIG["hidden_dim"],
                     num_layers=CONFIG["num_layers"], dropout=CONFIG["dropout"])
    history = train_lm(model, train_ids, steps=CONFIG["steps"], batch_size=CONFIG["batch_size"],
                       seq_len=CONFIG["seq_len"], lr=CONFIG["lr"], warmup=CONFIG["warmup"], val_ids=val_ids,
                       eval_every=CONFIG["eval_every"], device=CONFIG["device"], seed=0)
    train_secs = time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _k = 25
    _ax.plot(history["step"][_k - 1:], np.convolve(history["train_loss"], np.ones(_k) / _k, "valid") / math.log(2),
             label="training (moving average)")
    _ax.plot(history["val_step"], np.array(history["val_loss"]) / math.log(2), "o-", label="validation windows")
    _ax.set(xlabel="step", ylabel="bits per character")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(f"{sum(p.numel() for p in model.parameters()):,} parameters · {train_secs / 60:.1f} min on "
                           f"{CONFIG['device']} · best validation-window loss "
                           f"{min(history['val_loss']) / math.log(2):.3f} bits per character")])
    return history, model, train_secs


@app.cell
def _(model, ngram_test, ngram_val, test_ids, train_ids, val_ids):
    # evaluation (yours, from stage 4): every character of the validation and test blocks, with the state carried
    # from the 256 characters before the block, and block-bootstrap 95% intervals.
    _prime = 256

    def block_nll(block, before):
        return char_nll(model, np.concatenate([before[-_prime:], block]), device=CONFIG["device"])[_prime - 1:]

    lstm_val = block_nll(val_ids, train_ids)
    lstm_test = block_nll(test_ids, val_ids)
    _rows, _bs = [], 2000 if FULL else 500
    for _name, _v, _t in [("LSTM", lstm_val, lstm_test), (f"n-gram (order {CONFIG['ngram_n']})", ngram_val, ngram_test)]:
        _bv, _bt = bpc_ci(_v, block_size=_bs, rng=0), bpc_ci(_t, block_size=_bs, rng=0)
        _rows.append(f"| {_name} | {_bv[0]:.3f} [{_bv[1]:.3f}, {_bv[2]:.3f}] | {2 ** _bv[0]:.2f} | "
                     f"{_bt[0]:.3f} [{_bt[1]:.3f}, {_bt[2]:.3f}] | {2 ** _bt[0]:.2f} |")
    _d = bpc_ci(lstm_test - ngram_test, block_size=_bs, rng=0)
    mo.md("| model | validation bits/char [95% CI] | perplexity | **test** bits/char [95% CI] | perplexity |\n"
          "|---|---|---|---|---|\n" + "\n".join(_rows)
          + f"\n\npaired difference on test, LSTM − n-gram: **{_d[0]:+.3f}** bits per character [{_d[1]:+.3f}, {_d[2]:+.3f}]"
          + "\n\nchapter 04's best add-k n-gram (order 4, k = 0.1) on the same split: 2.51 bits per character on "
            "validation, 2.83 on test.")
    return lstm_test, lstm_val


@app.cell
def _():
    temperature = mo.ui.slider(steps=[0.0, 0.5, 0.8, 1.0, 1.5], value=0.8, label="temperature", show_value=True)
    temperature
    return (temperature,)


@app.cell
def _(model, temperature, text, tok):
    # samples (given): 500 characters from a prompt, and how much of the output is copied from the training text
    _start = "ROMEO:\n" if set("ROMEO:\n") <= set(tok.chars) else text[:20]  # the offline stand-in has no capitals
    _prompt = tok.encode(_start)
    _out = tok.decode(sample(model, _prompt, 500, temperature=temperature.value,
                             generator=torch.Generator().manual_seed(0), device=CONFIG["device"]))
    _train_text = text[: int(0.8 * len(text))]
    _chunks = [_out[_i:_i + 20] for _i in range(0, len(_out) - 20, 10)]
    _copied = np.mean([_c in _train_text for _c in _chunks]) if _chunks else 0.0
    mo.md(f"temperature {temperature.value}:\n\n```\n{_start}{_out}\n```\n\nfraction of 20-character chunks found "
          f"verbatim in the training text: **{_copied:.2f}**")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace this cell with your write-up (grade it with [`rubric.md`](rubric.md)). answer at
    least:

    1. **results.** validation and test bits per character (and perplexity) of your LSTM and of
       the n-gram baseline, with the block-bootstrap intervals, and the paired difference on
       test. compare with chapter 04's add-k n-gram (2.83 bits per character on test). say what
       the uncertainty does and doesn't cover.
    2. **why blocks?** compute the interval again with `block_size=1` (an i.i.d. bootstrap over
       characters). how much narrower is it, and why is it wrong? what does the gap between
       validation and test (for both models) tell you about the blocks' independence?
    3. **training.** the learning curves: where does validation stop improving, and what did
       the best-checkpoint rule do? one ablation from: no dropout, a one-layer LSTM, a shorter
       `seq_len` (32), or no warmup. report its validation bits per character against the
       baseline run.
    4. **context.** using `char_nll`, measure the LSTM's loss on the validation block with the
       state *reset* every 32 characters versus carried. how much does context beyond 32
       characters buy? compare with what the n-gram's order buys.
    5. **samples.** samples at temperatures 0, 0.5, 1.0 and 1.5. describe what greedy decoding
       does, what high temperature does, and use the copying check to discuss memorization.
       how do the samples compare with chapter 04's n-gram samples?
    6. **connections.** where did each topic of the chapter enter: the language-model view and
       perplexity (01), the tokenizer (02), embeddings (03), BPTT and its truncation (04), the
       LSTM (05), and decoding (06)?
    """)
    return


if __name__ == "__main__":
    app.run()
