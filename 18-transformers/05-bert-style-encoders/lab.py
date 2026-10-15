import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import math
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.linear import LogisticRegression  # yours, from ch 08 · topic 04
    from mlfs.text import CharTokenizer  # yours, from ch 17 · topic 02

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    MASK, PAD, CLS, N_SPECIAL = 0, 1, 2, 3
    MAX_LEN, PRETRAIN_STEPS, FT_STEPS = 64, 600, 120
    BUDGETS = (40, 160, 640, None)  # None = every labelled training message
    ENC = dict(max_len=MAX_LEN, n_layers=2, n_heads=4, d_model=64, dropout=0.1)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # BERT-style encoders · lab

    every experiment here uses **your** `mask_tokens`, `BERT`, `pretrain_mlm`, `BERTClassifier`
    and `finetune` from `exercise.py`, on the SMS spam collection (5,574 text messages, 13% spam),
    read character by character. for each experiment: write down a prediction, lock it in, run it,
    then explain what happened. compare your explanations with `solutions/lab_notes.md`
    afterwards. the lab takes two to three minutes on a laptop CPU.
    """)
    return


@app.cell
def _():
    texts, labels = datasets.sms_spam()
    tok = CharTokenizer().train("".join(texts))
    vocab = tok.vocab_size + N_SPECIAL  # ids 0-2 are [MASK], [PAD], [CLS]; characters start at 3

    def encode(batch):
        X = torch.full((len(batch), MAX_LEN), PAD, dtype=torch.long)
        X[:, 0] = CLS
        for i, s in enumerate(batch):
            ids = [c + N_SPECIAL for c in tok.encode(s)][: MAX_LEN - 1]
            X[i, 1:1 + len(ids)] = torch.tensor(ids)
        return X, X != PAD

    _perm = np.random.default_rng(0).permutation(len(texts))
    _n_test = len(texts) // 5
    test_idx, train_idx = _perm[:_n_test], _perm[_n_test:]
    X_all, M_all = encode(texts)
    y_all = torch.tensor(labels, dtype=torch.long)
    mo.md(f"{'SMS spam' if not datasets.is_offline() else 'offline stand-in'}: {len(texts):,} messages, "
          f"{labels.mean():.1%} spam, {tok.vocab_size} distinct characters; {len(train_idx):,} training and "
          f"{len(test_idx):,} test messages, truncated to {MAX_LEN - 1} characters after [CLS]")
    return M_all, X_all, test_idx, tok, train_idx, vocab, y_all


@app.function
def balanced_accuracy(y_true, y_pred):
    """the mean of the per-class recalls: 0.5 for a constant prediction, whatever the class balance."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in np.unique(y_true)]))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what masked-language-model pretraining buys

    pretrain one small `BERT` (2 layers, width 64) with your `pretrain_mlm` on the *text* of the
    4,460 training messages, labels unused (600 steps). then fine-tune a `BERTClassifier` (the
    [CLS] token) for 120 steps, starting either from the pretrained encoder or from a fresh one,
    with only 40, 160, 640 or all the training labels. the score is balanced accuracy on the
    test messages (0.5 = guessing). the third line is your logistic regression from chapter 08 on
    character counts, the kind of baseline that's easy to forget.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "with 40 labelled messages, how much does pretraining help (balanced accuracy, from scratch vs pretrained)? with "
        "all 4,460? and where does a bag-of-characters logistic regression land?"
    )
    p1
    return (p1,)


@app.cell
def _(M_all, X_all, p1, train_idx, vocab):
    ui.gate(p1)
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    pretrained = ex.BERT(vocab, **ENC)
    mlm_curve = ex.pretrain_mlm(pretrained, X_all[train_idx], mask_token_id=MASK, special_ids=(PAD, CLS),
                                attention_mask=M_all[train_idx], steps=PRETRAIN_STEPS, batch_size=64, lr=2e-3, seed=0)
    pretrain_secs = time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(6, 2.8))
    _ax.plot(np.convolve(mlm_curve, np.ones(25) / 25, "valid"))
    _ax.axhline(math.log(vocab), color="gray", ls="--", lw=0.8)
    _ax.set(xlabel="step", ylabel="MLM loss (nats)")
    mo.vstack([_fig, mo.md(f"MLM loss: {np.mean(mlm_curve[:25]):.2f} → {np.mean(mlm_curve[-50:]):.2f} nats "
                           f"(ln V = {math.log(vocab):.2f}) · {pretrain_secs:.0f} s")])
    return (pretrained,)


@app.cell
def _(M_all, X_all, pretrained, test_idx, train_idx, vocab, y_all):
    import copy

    _t0 = time.perf_counter()
    rows = []
    _counts = np.log1p(np.stack([np.bincount(x[m].numpy(), minlength=vocab) for x, m in zip(X_all, M_all)]))
    _counts = (_counts - _counts[train_idx].mean(0)) / (_counts[train_idx].std(0) + 1e-6)  # standardized log-counts
    for _n in BUDGETS:
        _sub = train_idx[:_n] if _n else train_idx
        _scores = {}
        for _name in ("from scratch", "pretrained"):
            torch.manual_seed(1)
            _enc = copy.deepcopy(pretrained) if _name == "pretrained" else ex.BERT(vocab, **ENC)
            _clf = ex.BERTClassifier(_enc, 2)
            _epochs = max(1, math.ceil(FT_STEPS * 32 / len(_sub)))
            ex.finetune(_clf, X_all[_sub], y_all[_sub], attention_mask=M_all[_sub], epochs=min(_epochs, 120),
                        batch_size=32, lr=5e-4, seed=0)
            with torch.no_grad():
                _pred = _clf(X_all[test_idx], M_all[test_idx]).argmax(-1).numpy()
            _scores[_name] = balanced_accuracy(y_all[test_idx].numpy(), _pred)
        if len(np.unique(y_all[_sub].numpy())) > 1:
            _lr = LogisticRegression(alpha=0.01).fit(_counts[_sub], y_all[_sub].numpy())
            _scores["char-count logistic regression"] = balanced_accuracy(y_all[test_idx].numpy(),
                                                                          _lr.predict(_counts[test_idx]))
        rows.append((_n or len(train_idx), _scores))
    mo.md("| labelled messages | from scratch | pretrained | char-count logistic regression |\n|---|---|---|---|\n"
          + "\n".join(f"| {n:,} | {s['from scratch']:.3f} | {s['pretrained']:.3f} | "
                      f"{s.get('char-count logistic regression', float('nan')):.3f} |" for n, s in rows)
          + f"\n\nfine-tuning: {time.perf_counter() - _t0:.0f} s")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** pretraining never saw a label. what did learning to fill in missing
    characters teach the encoder that makes spam detection easier from 40 examples? why does the
    gap shrink as labels grow? and what does the logistic regression tell you about how hard this
    task is, and about which baseline a paper should have reported?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what the pretrained encoder predicts

    take test messages, hide 15% of their characters (your `mask_tokens`, always with
    [MASK] here), and ask the pretrained model's MLM head for each hidden character. compare
    with the best context-free guess, the most frequent character.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("what fraction of the hidden characters will the pretrained model get exactly right?")
    p2
    return (p2,)


@app.cell
def _(M_all, X_all, p2, pretrained, test_idx, tok, vocab):
    ui.gate(p2)
    _X, _M = X_all[test_idx[:400]], M_all[test_idx[:400]]
    _g = torch.Generator().manual_seed(0)
    _inp, _lab = ex.mask_tokens(_X, mask_token_id=MASK, vocab_size=vocab, special_ids=(PAD, CLS),
                                generator=_g)
    _sel = _lab != -100
    _inp = torch.where(_sel, torch.full_like(_X, MASK), _X)  # always [MASK] for this test
    with torch.no_grad():
        _pred = pretrained.mlm_logits(pretrained(_inp, _M)).argmax(-1)
    _acc = (_pred[_sel] == _X[_sel]).float().mean().item()
    _common = torch.bincount(_X[_M & (_X > CLS)]).argmax()
    _base = (_X[_sel] == _common).float().mean().item()
    _show = []
    for _i in range(3):
        _row = [tok.decode([int(t) - 3]) if not s else "_" for t, s in zip(_X[_i, 1:], _sel[_i, 1:]) if t != PAD]
        _fill = [tok.decode([int(p) - 3]) if int(p) >= 3 else "?" for p, s in zip(_pred[_i, 1:], _sel[_i, 1:]) if s]
        _show.append("".join(_row) + "\n  filled in: " + "".join(_fill))
    mo.md(f"exact recovery of hidden characters: **{_acc:.1%}** (most frequent character "
          f"{tok.decode([int(_common) - 3])!r}: {_base:.1%})\n\n```\n" + "\n\n".join(_show) + "\n```")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** compare the recovery rate with the most-frequent-character guess, and look at
    the MLM loss curve from experiment 1. how much has 600 steps of pretraining really taught the
    model about filling in characters? then reconcile that with experiment 1: what could a
    barely-trained encoder give the classifier that a fresh one doesn't? (lab notes: what
    5× more pretraining does.) why is masked-character prediction a task a bidirectional model
    can't cheat on, unlike next-character prediction?
    """)
    return


if __name__ == "__main__":
    app.run()
