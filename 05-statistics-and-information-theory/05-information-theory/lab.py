import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import math
    import zlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    from mlfs import datasets, notebooks, ui

    plt.rcParams["figure.autolayout"] = True  # tight layout: keep axis labels inside the figure

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # information theory · lab

    every experiment here runs **your** code from `exercise.py`. for each one: write down a
    prediction, lock it in, run it, then explain what happened. compare your explanations with
    `solutions/lab_notes.md` afterwards.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how close can a compressor get to the entropy?

    a source emits independent bits, 1 with probability $p$ (the slider). your `huffman_code`
    compresses it, first one symbol at a time, then in **blocks** of $k$ symbols (an alphabet of
    $2^k$ blocks, each with the product probability). for each $k$, the table shows the
    expected bits per source symbol, and the rate actually achieved by your `encode` on a
    random message of 25,200 symbols (decoded again with your `decode` as a check).
    """)
    return


@app.cell
def _():
    p1 = ui.predict(
        "for p = 0.1 the entropy is 0.469 bits per symbol. how many bits per symbol does Huffman coding "
        "use one symbol at a time? with blocks of 8 symbols?"
    )
    p1
    return (p1,)


@app.cell
def _():
    p_one = mo.ui.slider(start=0.02, stop=0.5, step=0.02, value=0.1, label="p = P(symbol = 1)", show_value=True)
    p_one
    return (p_one,)


@app.cell
def _(p1, p_one):
    ui.gate(p1)
    _p = p_one.value
    _H = ex.entropy([1 - _p, _p], base=2)
    _msg = (np.random.default_rng(0).random(25_200) < _p).astype(int)  # divisible by every k from 1 to 10
    _ks = list(range(1, 11))
    _rows, _exp, _ach = [], [], []
    for _k in _ks:
        _blocks = list(itertools.product([0, 1], repeat=_k))
        _probs = [_p ** sum(_b) * (1 - _p) ** (_k - sum(_b)) for _b in _blocks]
        _code = ex.huffman_code(_probs)
        _L = ex.expected_length(_probs, _code) / _k
        _ids = _msg.reshape(-1, _k) @ (2 ** np.arange(_k - 1, -1, -1))  # block -> index in _blocks
        _bits = ex.encode(_ids, _code)
        assert list(ex.decode(_bits, _code)) == list(_ids), "decode(encode(message)) didn't round-trip"
        _exp.append(_L)
        _ach.append(len(_bits) / len(_msg))
        _rows.append(f"| {_k} | {2 ** _k} | {_L:.4f} | {_ach[-1]:.4f} | {_H + 1 / _k:.4f} |")
    _fig, _ax = plt.subplots(figsize=(6.5, 3.4))
    _ax.plot(_ks, _exp, "o-", label="expected bits/symbol (Huffman on k-blocks)")
    _ax.plot(_ks, _ach, "x", label="achieved on a 25,200-symbol message")
    _ax.axhline(_H, color="k", ls="--", lw=0.8, label=f"entropy H = {_H:.3f}")
    _ax.plot(_ks, [_H + 1 / _k for _k in _ks], ":", color="gray", label="bound H + 1/k")
    _ax.set(xlabel="block size k", ylabel="bits per source symbol", ylim=(0, 1.1))
    _ax.legend(fontsize=7)
    mo.vstack([_fig, mo.md(f"entropy of the source: **{_H:.4f} bits per symbol**.\n\n"
                           "| block size k | alphabet size | expected bits/symbol | achieved | bound H + 1/k |\n"
                           "|---|---|---|---|---|\n" + "\n".join(_rows))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    now real-ish text: `datasets.tiny_shakespeare()` (a synthetic stand-in of random English
    words when you're offline). your `entropy` of the character frequencies, your Huffman code
    on characters and on character pairs, and Python's `zlib` (the algorithm inside gzip and
    PNG), all in bits per character.
    """)
    return


@app.cell
def _(p1):
    ui.gate(p1)
    _text = datasets.tiny_shakespeare()[:200_000]
    _chars = sorted(set(_text))
    _idx = {c: i for i, c in enumerate(_chars)}
    _ids = np.array([_idx[c] for c in _text])
    _counts = np.bincount(_ids, minlength=len(_chars))
    _H1 = ex.entropy(_counts, base=2)
    _code1 = ex.huffman_code(_counts)
    _rate1 = len(ex.encode(_ids, _code1)) / len(_ids)
    _pairs = _ids[: len(_ids) // 2 * 2].reshape(-1, 2) @ np.array([len(_chars), 1])
    _pc = np.bincount(_pairs, minlength=len(_chars) ** 2)
    _used = np.nonzero(_pc)[0]
    _code2 = ex.huffman_code(_pc[_used])
    _remap = {v: i for i, v in enumerate(_used)}
    _rate2 = len(ex.encode([_remap[v] for v in _pairs], _code2)) / (2 * len(_pairs))
    _H2 = ex.entropy(_pc, base=2) / 2
    _z = 8 * len(zlib.compress(_text.encode("utf-8"), 9)) / len(_text)
    mo.md(f"""
    {'**offline stand-in**' if datasets.is_offline() else '**tiny Shakespeare**'}: {len(_text):,} characters, {len(_chars)} distinct.

    | method | bits per character |
    |---|---|
    | no compression (8-bit bytes) | 8.000 |
    | uniform code over the alphabet, log₂ {len(_chars)} | {math.log2(len(_chars)):.3f} |
    | entropy of the character frequencies | {_H1:.3f} |
    | your Huffman code, one character at a time | {_rate1:.3f} |
    | entropy of character-pair frequencies, per character | {_H2:.3f} |
    | your Huffman code on character pairs | {_rate2:.3f} |
    | zlib, level 9 | {_z:.3f} |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does Huffman coding of single bits give no compression at all, and why
    do blocks fix it? (where does the "+1" in $H \le L < H + 1$ come from, and what happens to
    it per symbol?) on the text, zlib beats "the entropy" of the character frequencies. how is
    that possible, and which entropy is the real limit?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: dependence out of thin air

    $X$ and $Y$ are **independent**, each uniform over $k$ categories, so $I(X; Y) = 0$. you
    observe $n$ pairs, tabulate the counts, and estimate the mutual information with your
    `mutual_information` applied to the count table (the "plug-in" estimate). each cell
    averages 200 repetitions.
    """)
    return


@app.cell
def _():
    p2 = ui.predict(
        "with k = 10 categories each and n = 100 samples, what's the average estimated mutual information "
        "(true value 0)? with n = 10,000?"
    )
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    _rng = np.random.default_rng(1)
    _rows = []
    _ns = [50, 100, 1000, 10_000]
    for _k in (2, 5, 10):
        _cells = []
        for _n in _ns:
            _est = []
            for _ in range(200):
                _x, _y = _rng.integers(0, _k, _n), _rng.integers(0, _k, _n)
                _table = np.zeros((_k, _k))
                np.add.at(_table, (_x, _y), 1)
                _est.append(ex.mutual_information(_table))
            _cells.append(f"{np.mean(_est):.4f} ({(_k - 1) ** 2 / (2 * _n):.4f})")
        _rows.append(f"| {_k} | " + " | ".join(_cells) + " |")
    mo.md("mean plug-in estimate in nats, true value 0 (in brackets: the approximation $(k - 1)^2 / 2n$)\n\n"
          "| categories k | " + " | ".join(f"n = {_n:,}" for _n in _ns) + " |\n|---|" + "---|" * len(_ns) + "\n"
          + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why is the plug-in estimate of a quantity whose true value is 0 always
    positive? why does the bias grow with the number of cells $k^2$ and shrink with $n$? if you
    ranked 1,000 candidate features by their estimated mutual information with a label, which
    features would this bias favor?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: which way round is KL?

    $p$ is a mixture of two well-separated Gaussians (peaks at $\pm 2$). you approximate it
    with a **single** Gaussian $q_{\mu,\sigma}$, choosing $(\mu, \sigma)$ on a grid to minimize
    either the **forward** KL, $\mathrm{KL}(p \,\|\, q)$, or the **reverse** KL,
    $\mathrm{KL}(q \,\|\, p)$. both are computed by your `kl_divergence` on a fine grid of
    $x$ values. (fitting by maximum likelihood minimizes the forward one; variational
    inference, in ch 12, minimizes the reverse one.)
    """)
    return


@app.cell
def _():
    p3 = ui.predict(
        "where does each fitted Gaussian end up: centered between the two peaks, or on one of them? "
        "how wide is each?"
    )
    p3
    return (p3,)


@app.cell
def _(p3):
    ui.gate(p3)
    _x = np.linspace(-7, 7, 1401)
    _g = lambda m, s: np.exp(-0.5 * ((_x - m) / s) ** 2) / s
    _p = 0.5 * _g(-2.0, 0.6) + 0.5 * _g(2.0, 0.6)
    _mus, _sigmas = np.linspace(-3, 3, 61), np.linspace(0.3, 4.0, 38)
    _fwd = np.full((61, 38), np.inf)
    _rev = np.full((61, 38), np.inf)
    for _i, _m in enumerate(_mus):
        for _j, _s in enumerate(_sigmas):
            _q = _g(_m, _s)
            _fwd[_i, _j] = ex.kl_divergence(_p, _q)
            _rev[_i, _j] = ex.kl_divergence(_q, _p)
    _bf = np.unravel_index(np.argmin(_fwd), _fwd.shape)
    _br = np.unravel_index(np.argmin(_rev), _rev.shape)
    _dx = _x[1] - _x[0]
    _fig, _ax = plt.subplots(figsize=(7.5, 3.4))
    _ax.fill_between(_x, _p / (_p.sum() * _dx), alpha=0.3, color="gray", label="p (two peaks)")
    _qf = _g(_mus[_bf[0]], _sigmas[_bf[1]])
    _qr = _g(_mus[_br[0]], _sigmas[_br[1]])
    _ax.plot(_x, _qf / (_qf.sum() * _dx), lw=2, label=f"min KL(p || q): mu = {_mus[_bf[0]]:.1f}, sigma = {_sigmas[_bf[1]]:.1f}")
    _ax.plot(_x, _qr / (_qr.sum() * _dx), lw=2, label=f"min KL(q || p): mu = {_mus[_br[0]]:.1f}, sigma = {_sigmas[_br[1]]:.1f}")
    _ax.set(xlabel="x", ylabel="density")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"| fit | mu | sigma | KL(p ‖ q) | KL(q ‖ p) |\n|---|---|---|---|---|\n"
        f"| forward (min KL(p ‖ q)) | {_mus[_bf[0]]:.2f} | {_sigmas[_bf[1]]:.2f} | {_fwd[_bf]:.3f} | {_rev[_bf]:.3f} |\n"
        f"| reverse (min KL(q ‖ p)) | {_mus[_br[0]]:.2f} | {_sigmas[_br[1]]:.2f} | {_fwd[_br]:.3f} | {_rev[_br]:.3f} |")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** in $\mathrm{KL}(p \,\|\, q) = \sum p\log(p/q)$, what happens to the cost
    where $p$ is large but $q$ is tiny? and in $\mathrm{KL}(q \,\|\, p)$, where $q$ is large but
    $p$ is tiny? use that to explain "mode covering" versus "mode seeking". why does the
    forward fit put most of its mass where $p$ has almost none? which behavior would you want
    for a generative model of images, and which for an approximate posterior?
    """)
    return


if __name__ == "__main__":
    app.run()
