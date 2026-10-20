import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import notebooks, ui

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    def _audio_smoke(m):  # raises NotImplementedError if a topic 24 stage this lab needs isn't done yet
        _x = m.harmonic_vowel(120.0, (730, 1090, 2440), 0.05, 8000)
        m.power_to_db(m.mel_spectrogram(_x, 8000, n_fft=256, hop_length=160, n_mels=40), top_db=None)

    # your topic 24 features: the reference under pytest/verify (or MLFS_REFERENCE=audio24), and whenever a
    # topic 24 stage it needs is unfinished
    prev = notebooks.load_topic("24-audio-representations", key="audio24", smoke=_audio_smoke)
    audio = prev.module
    torch.set_num_threads(1)  # small models: one thread is fastest


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # speech recognition with CTC · lab

    every experiment here runs **your** code: your `ctc_loss`, decoders, error rate and
    `CTCRecognizer` from this topic, on features from your topic 24 `mel_spectrogram`. for
    each one: write down a prediction, lock it in, run it, then explain what happened. the
    lab takes 30–90 s on one CPU core (four small models are trained).

    **the audio is synthetic.** each utterance is 3–6 "phonemes" at 8 kHz: the five vowels
    /a e i o u/ from topic 24's source–filter model (one random pitch per utterance, 90–220 Hz,
    ±5% per vowel), and three fricatives, white noise shaped by a single wide resonance
    ("s" at 3.4 kHz, "sh" at 2.3 kHz, "f" broad around 1.5 kHz). phonemes last 80–200 ms,
    separated by 30–60 ms gaps, with 50–150 ms of silence at both ends and a little background
    noise. with probability 0.3 a phoneme repeats its predecessor ("a a"). features: your log
    mel spectrogram, 40 bands from 32 ms windows every 20 ms, normalized per utterance.
    """)
    return


@app.cell
def _():
    prev.callout()
    return


@app.cell
def _():
    SR = 8000
    HOP = 160
    PHONES = ["a", "e", "i", "o", "u", "s", "sh", "f"]  # class k + 1; class 0 is the blank
    FRICATIVES = {"s": (3400.0, 700.0), "sh": (2300.0, 600.0), "f": (1500.0, 3000.0)}
    VOWELS = audio.vowel_formants()

    def fricative(name, n, rng):
        spec = np.fft.rfft(rng.normal(size=n))
        centre, bw = FRICATIVES[name]
        spec *= audio.formant_gain(np.fft.rfftfreq(n, 1 / SR), [centre], [bw])
        x = np.fft.irfft(spec, n)
        return x * 0.05 / np.sqrt(np.mean(x ** 2))

    def utterance(labels, rng):
        """waveform and the (start, end) sample range of every phoneme."""
        f0 = rng.uniform(90, 220)
        parts, spans, pos = [], [], 0

        def push(seg):
            nonlocal pos
            parts.append(seg)
            pos += len(seg)

        push(np.zeros(int(rng.uniform(0.05, 0.15) * SR)))
        for lab in labels:
            dur = rng.uniform(0.08, 0.2)
            n = int(dur * SR)
            name = PHONES[lab - 1]
            if name in VOWELS:
                seg = audio.harmonic_vowel(f0 * rng.uniform(0.95, 1.05), VOWELS[name], dur, SR,
                                           rng=int(rng.integers(1 << 30)), rms=0.1 * rng.uniform(0.6, 1.4))[:n]
            else:
                seg = fricative(name, n, rng)
            ramp = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.01 * SR))
            spans.append((pos, pos + n))
            push(seg * ramp)
            push(np.zeros(int(rng.uniform(0.03, 0.06) * SR)))
        push(np.zeros(int(rng.uniform(0.05, 0.15) * SR)))
        x = np.concatenate(parts)
        return x + 0.003 * rng.normal(size=len(x)), spans

    def features(x):
        f = audio.power_to_db(audio.mel_spectrogram(x, SR, n_fft=256, hop_length=HOP, n_mels=40), top_db=None)
        return ((f - f.mean(1, keepdims=True)) / (f.std(1, keepdims=True) + 1e-5)).T.astype(np.float32)

    def corpus(n, seed, p_repeat=0.3):
        rng = np.random.default_rng(seed)
        X, Y, S = [], [], []
        for _ in range(n):
            labs = [int(rng.integers(1, 9))]
            for _ in range(int(rng.integers(3, 7)) - 1):
                if rng.random() < p_repeat:
                    labs.append(labs[-1])
                else:
                    labs.append(int(rng.choice([c for c in range(1, 9) if c != labs[-1]])))
            x, spans = utterance(labs, rng)
            X.append(features(x))
            Y.append(labs)
            S.append(spans)
        return X, Y, S

    _t = time.time()
    Xtr, Ytr, _ = corpus(400, 0)
    Xte, Yte, Ste = corpus(150, 1)
    REP = [i for i, y in enumerate(Yte) if any(a == b for a, b in zip(y, y[1:]))]
    NOREP = [i for i in range(len(Yte)) if i not in REP]
    mo.md(f"corpus built in {time.time() - _t:.1f} s: {len(Xtr)} training and {len(Xte)} test utterances, "
          f"{np.mean([len(x) for x in Xtr]):.0f} frames and {np.mean([len(y) for y in Ytr]):.1f} phonemes on "
          f"average; {len(REP)} of the test utterances contain a repeated neighbour.")
    return HOP, NOREP, PHONES, REP, Ste, Xte, Xtr, Yte, Ytr


@app.cell
def _(HOP, PHONES, Ste, Xte, Yte):
    _fig, _ax = plt.subplots(figsize=(10, 3))
    _ax.imshow(Xte[0].T, origin="lower", aspect="auto", cmap="magma")
    for _lab, (_a, _b) in zip(Yte[0], Ste[0]):
        _ax.axvspan(_a / HOP, _b / HOP, color="w", alpha=0.12)
        _ax.text((_a + _b) / 2 / HOP, 41, PHONES[_lab - 1], ha="center", va="bottom")
    _ax.set(xlabel="frame (20 ms)", ylabel="mel band", ylim=(0, 45),
            title="test utterance 0: normalized log mel features, phoneme spans shaded")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(Xtr, Ytr):
    def batch(X, Y, idx):
        T, U = max(len(X[i]) for i in idx), max(len(Y[i]) for i in idx)
        xb = np.zeros((len(idx), T, X[0].shape[1]), np.float32)
        yb = np.zeros((len(idx), U), np.int64)
        for k, i in enumerate(idx):
            xb[k, :len(X[i])], yb[k, :len(Y[i])] = X[i], Y[i]
        return (torch.tensor(xb), torch.tensor(yb), torch.tensor([len(X[i]) for i in idx]),
                torch.tensor([len(Y[i]) for i in idx]))

    def no_blank(lp):
        """take the blank away: blank log-prob fixed at -30, the labels renormalized among themselves."""
        return torch.cat([torch.full_like(lp[..., :1], -30.0), lp[..., 1:].log_softmax(-1)], dim=-1)

    def train(seed, blank=True, epochs=16):
        torch.manual_seed(seed)
        model = ex.CTCRecognizer(40, 9, hidden_size=64, num_layers=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(Xtr))
            for s in range(0, len(order), 32):
                xb, yb, il, tl = batch(Xtr, Ytr, order[s:s + 32])
                lp = model(xb)
                loss = ex.ctc_loss(lp if blank else no_blank(lp), yb, il, tl)
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        return model

    def posteriors(model, X, blank=True):
        with torch.no_grad():
            out = [model(torch.tensor(x)[None])[0] for x in X]
        return out if blank else [no_blank(lp) for lp in out]

    return posteriors, train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: what is the blank for?

    two versions of the same model, each trained with your `ctc_loss` for 16 epochs from two
    seeds. **with blank** is ordinary CTC. **without blank** fixes the blank's log-probability
    at −30 (probability $10^{-13}$) and renormalizes the eight phoneme classes among
    themselves, so every frame must be labeled with a phoneme, and a repeated pair can only
    be aligned by paying 30 nats for one blank frame between them. both are decoded greedily
    and scored with your `error_rate`, separately on test utterances with and without a
    repeated neighbour.
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "without a usable blank, how much worse is the phoneme error rate on utterances WITH repeats? and on "
        "utterances WITHOUT repeats, where the blank shouldn't be needed?"
    )
    p1
    return (p1,)


@app.cell
def _(NOREP, REP, Xte, Yte, p1, posteriors, train):
    ui.gate(p1)
    _t = time.time()
    models = {(b, s): train(s, blank=b) for b in (True, False) for s in (0, 1)}
    post = {k: posteriors(m, Xte, blank=k[0]) for k, m in models.items()}
    greedy = {k: [ex.greedy_decode(lp) for lp in v] for k, v in post.items()}
    _rows = []
    for _b in (True, False):
        _cells = []
        for _subset in (range(len(Yte)), REP, NOREP):
            _r = [ex.error_rate([Yte[i] for i in _subset], [greedy[(_b, s)][i] for i in _subset]) for s in (0, 1)]
            _cells.append(f"{np.mean(_r):.3f} ({_r[0]:.3f}, {_r[1]:.3f})")
        _rows.append(f"| {'with blank' if _b else 'without blank'} | " + " | ".join(_cells) + " |")
    _merged = np.mean([sum(len(greedy[(False, s)][i]) < len(Yte[i]) for i in REP) / len(REP) for s in (0, 1)])
    mo.md("| model | PER, all | PER, utterances with repeats | PER, without repeats |\n|---|---|---|---|\n"
          + "\n".join(_rows)
          + f"\n\nper-seed values in brackets. without blank, {_merged:.0%} of the repeat utterances come out "
          f"shorter than their reference. four models trained in {time.time() - _t:.0f} s.")
    return greedy, post


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't the blank-free model output "a a" even when it has learned both
    a's perfectly? (what does `ctc_collapse` do with a run of a-frames?) more surprising: it
    is also much worse on utterances *without* repeats. which frames of an utterance does it
    have to label with *some* phoneme, and what does that do to its decisions near the
    silences and gaps? what are the two jobs the blank does?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: what does the trained model output?

    the with-blank model (seed 0) on a test utterance (pick one with the slider): the probability of each class at each
    frame, with the true phoneme spans shaded. the table counts, over the whole test set, the
    fraction of frames whose most likely class is blank, against the fraction of frames that
    actually lie inside a phoneme.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "about 70% of the frames lie inside a phoneme. what fraction of frames does the trained model label blank "
        "(argmax)? where inside each phoneme do the label spikes sit?"
    )
    utt = mo.ui.slider(0, 149, value=0, label="test utterance to plot")
    mo.vstack([p2, utt])
    return p2, utt


@app.cell
def _(HOP, PHONES, Ste, Xte, Yte, p2, post, utt):
    ui.gate(p2)
    _inside = []
    for _x, _spans in zip(Xte, Ste):
        _m = np.zeros(len(_x), bool)
        for _a, _b in _spans:
            _m[int(round(_a / HOP)):int(round(_b / HOP)) + 1] = True
        _inside.append(_m)
    _inside_frac = np.concatenate(_inside).mean()
    _rows = []
    for _s in (0, 1):
        _am = np.concatenate([lp.argmax(-1).numpy() for lp in post[(True, _s)]])
        _rows.append(f"| {_s} | {(_am == 0).mean():.3f} | {_inside_frac:.3f} | {(_am != 0).sum() / sum(map(len, Yte)):.2f} |")
    _u = utt.value
    _lp = post[(True, 0)][_u].exp().numpy()
    _fig, _ax = plt.subplots(figsize=(10, 3.4))
    for _lab, (_a, _b) in zip(Yte[_u], Ste[_u]):
        _ax.axvspan(_a / HOP, _b / HOP, color="0.85")
        _ax.text((_a + _b) / 2 / HOP, 1.03, PHONES[_lab - 1], ha="center")
    _ax.plot(_lp[:, 0], "k-", lw=1.2, label="blank")
    for _k in range(1, 9):
        if _lp[:, _k].max() > 0.2:
            _ax.plot(_lp[:, _k], lw=1.5, label=PHONES[_k - 1])
    _ax.set(xlabel="frame (20 ms)", ylabel="probability", ylim=(0, 1.12),
            title=f"with-blank model, test utterance {_u} (grey: true phoneme spans)")
    _ax.legend(fontsize=8, ncol=5, loc="center right")
    _fig.tight_layout()
    mo.vstack([mo.md("| seed | frames labeled blank (argmax) | frames inside a phoneme | non-blank frames per phoneme |\n"
                     "|---|---|---|---|\n" + "\n".join(_rows)), _fig])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** nothing in the loss asks for blanks *inside* phonemes, so why does the model
    put almost all its probability on blank there and fire each label in a single spike? (on
    paper 3: the gradient pushes $y_t$ toward the posterior over alignments. which alignments
    have the most probability early in training, when $y_t$ is nearly uniform, and what does
    that do to the next update?) what does the spike's position tell you, and not tell you,
    about *when* the phoneme happened?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: greedy or beam?

    the same with-blank models, decoded with your `greedy_decode` and with your
    `prefix_beam_search` (beam 8, no language model).
    """)
    return


@app.cell
def _():
    p3 = ui.predict("how much lower is the phoneme error rate with prefix beam search than with greedy decoding?")
    p3
    return (p3,)


@app.cell
def _(Yte, greedy, p3, post):
    ui.gate(p3)
    _t = time.time()
    _rows = []
    for _s in (0, 1):
        _beam = [list(ex.prefix_beam_search(lp, beam_size=8)[0][0]) for lp in post[(True, _s)]]
        _g = greedy[(True, _s)]
        _diff = sum(list(a) != list(b) for a, b in zip(_g, _beam))
        _rows.append(f"| {_s} | {ex.error_rate(Yte, _g):.4f} | {ex.error_rate(Yte, _beam):.4f} | {_diff} of {len(Yte)} |")
    mo.md("| seed | PER, greedy | PER, prefix beam (8) | utterances where they differ |\n|---|---|---|---|\n"
          + "\n".join(_rows) + f"\n\nbeam search over {2 * len(Yte)} utterances took {time.time() - _t:.0f} s.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the worked example showed greedy decoding picking the wrong answer. why
    does that almost never happen here? (connect it to experiment 2: when one path carries
    nearly all of a labeling's probability, how do best path and best labeling compare?) what
    would a beam search need, beyond the acoustic model's posteriors, to make a real
    difference?
    """)
    return


if __name__ == "__main__":
    app.run()
