import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    from collections import Counter

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import GPT, train_lm  # yours, from ch 18 · 04

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    TEXT = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    HELD_OUT = TEXT[-len(TEXT) // 10:]  # never used to build the crawl
    SOURCE = TEXT[: -len(TEXT) // 10]


@app.function
def make_crawl(text, n_pages=3000, seed=0):
    """a synthetic web crawl built from `text`: (pages, kinds, origin, articles); origin[i] indexes `articles` (-1: none).

    45% original articles (half wrapped in a site's navigation and footer), 15% exact mirrors (half of them re-spaced or
    re-cased), 15% syndicated near-copies (a few words changed, another site's boilerplate), 13% templated shop pages,
    12% junk (hashtag spam, cookie walls, number dumps). copies pick their article with Zipf-like popularity, so a few
    articles are copied dozens of times, as viral pages are on the real web.
    """
    rng = np.random.default_rng(seed)
    sites = ["thebard.example", "quillpress.example", "globe-reads.example", "folio.example", "stagedaily.example"]
    items = ["lantern", "quill", "doublet", "goblet", "dagger", "crown", "candle", "lute", "cloak", "map"]
    cities = ["verona", "padua", "messina", "elsinore", "athens", "venice", "milan", "navarre"]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    pages, kinds, origin, originals = [], [], [], []

    def boilerplate(body):
        site = sites[int(rng.integers(len(sites)))]
        return (f"Home | Plays | Sonnets | About {site}\n{body}\nShare this page. Subscribe to our newsletter. "
                f"(c) {int(rng.integers(2005, 2025))} {site}. All rights reserved.")

    def popular():
        w = 1.0 / np.arange(1, len(originals) + 1) ** 1.1
        return int(rng.choice(len(originals), p=w / w.sum()))

    for _ in range(n_pages):
        u = rng.random()
        if u < 0.45 or not originals:
            start = int(rng.integers(0, len(lines) - 40))
            body = "\n".join(lines[start:start + int(rng.integers(8, 30))])
            originals.append(body)
            pages.append(boilerplate(body) if rng.random() < 0.5 else body)
            kinds.append("article")
            origin.append(len(originals) - 1)
        elif u < 0.60:
            j = popular()
            r = rng.random()
            pages.append(originals[j] if r < 0.5 else originals[j].replace("\n", "\n\n") if r < 0.8 else originals[j].upper())
            kinds.append("exact copy")
            origin.append(j)
        elif u < 0.75:
            j = popular()
            words = originals[j].split(" ")
            for i in rng.choice(len(words), size=max(1, len(words) // 40), replace=False):
                words[i] = words[i].upper()
            pages.append(boilerplate(" ".join(words)))
            kinds.append("near copy")
            origin.append(j)
        elif u < 0.88:
            item, city = items[int(rng.integers(len(items)))], cities[int(rng.integers(len(cities)))]
            price = int(rng.integers(5, 200))
            pages.append(f"Buy a {item} online in {city}. The best {item} prices in {city}, from {price} ducats. "
                         f"Free delivery anywhere in {city} on orders over 50 ducats. Our {item} range is hand made "
                         f"by craftsmen and every {item} comes with a full guarantee. Order your {item} today and "
                         f"join thousands of happy customers in {city} and beyond. Returns are free within thirty days.")
            kinds.append("template")
            origin.append(-1)
        else:
            r = rng.random()
            if r < 0.4:
                w_ = items[int(rng.integers(len(items)))]
                pages.append(" ".join([f"#{w_}", "cheap", w_, "#deal", "...", "click"] * int(rng.integers(4, 12))))
            elif r < 0.7:
                pages.append("\n".join(["Accept all cookies", "Manage preferences"] * int(rng.integers(3, 8))))
            else:
                pages.append(" ".join(str(int(x)) for x in rng.integers(0, 10**6, size=int(rng.integers(30, 80)))))
            kinds.append("junk")
            origin.append(-1)
    return pages, kinds, origin, originals


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # pretraining data & compute · lab

    every experiment here runs **your** `shingles`, `minhash_signature`, `minhash_dedup` and
    `quality_filter`. the web crawl is **synthetic**: there's no crawl on this machine and
    nothing gets downloaded, so `make_crawl` builds one from tiny Shakespeare (real text, a
    synthetic stand-in offline) the way real crawls look: original articles, mirrors,
    syndicated copies wrapped in other sites' boilerplate, templated shop pages, and junk.
    you know exactly how each page was made, which a real crawl never tells you. for each
    experiment: write down a prediction, lock it in, run it, then explain what happened.
    compare with `solutions/lab_notes.md` afterwards. about two minutes.
    """)
    return


@app.cell
def _():
    pages, kinds, origin, articles = make_crawl(SOURCE)
    _c = Counter(kinds)
    mo.md(f"the crawl: **{len(pages):,} pages**: " + ", ".join(f"{v:,} {k}" for k, v in _c.items()) + ".\n\n"
          f"a page, chosen at random:\n\n```\n{pages[7][:400]}\n```")
    return articles, kinds, origin, pages


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: how much of a crawl survives?

    the pipeline: your `quality_filter`, then exact deduplication (hash the lowercased,
    whitespace-normalised text), then your `minhash_dedup` (threshold 0.8, 32 bands of 4).
    the table counts the pages of each kind that survive each stage.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("what fraction of the 3,000 pages survive all three stages? which stage removes the most, and "
                    "which kind of copy is hardest to remove?")
    p1
    return (p1,)


@app.cell
def _(kinds, p1, pages):
    ui.gate(p1)
    _q = [ex.quality_filter(p) for p in pages]
    passed = [i for i, (_k, _) in enumerate(_q) if _k]
    _seen, exact = set(), []
    for _i in passed:
        _h = " ".join(pages[_i].lower().split())
        if _h not in _seen:
            _seen.add(_h)
            exact.append(_i)
    _keep = ex.minhash_dedup([pages[i] for i in exact])
    final = [exact[i] for i in _keep]
    _stages = {"crawl": range(len(pages)), "quality filter": passed, "exact dedup": exact, "MinHash dedup": final}
    _kinds = ["article", "exact copy", "near copy", "template", "junk"]
    _rows = "\n".join(f"| {s} | " + " | ".join(str(sum(kinds[i] == k for i in idx)) for k in _kinds)
                      + f" | **{len(idx):,}** |" for s, idx in _stages.items())
    _reasons = Counter(r for _, rs in _q for r in rs)
    _fig, _ax = plt.subplots(figsize=(6, 3))
    _bottom = np.zeros(len(_stages))
    for _k in _kinds:
        _v = np.array([sum(kinds[i] == _k for i in idx) for idx in _stages.values()])
        _ax.bar(list(_stages), _v, bottom=_bottom, label=_k)
        _bottom += _v
    _ax.set(ylabel="pages")
    _ax.set_ylim(0, 1.25 * len(pages))
    _ax.legend(fontsize=8, ncol=5, loc="upper center")
    mo.vstack([_fig, mo.md(
        "| stage | " + " | ".join(_kinds) + " | total |\n|---|" + "---|" * (len(_kinds) + 1) + "\n" + _rows +
        f"\n\n**{100 * len(final) / len(pages):.0f}%** of the pages survive. the quality filter's reasons: "
        + ", ".join(f"{k} {v}" for k, v in _reasons.most_common()) + ".")])
    return exact, final, passed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** exact hashing catches mirrors but almost no near copies: why? what
    happened to the templated shop pages, and is that what you want? some original
    **articles** were removed as near-duplicates too: of what? a few dozen copies survive
    MinHash: what do they have in common? (compare a surviving copy with its original using
    `ex.jaccard(ex.shingles(a), ex.shingles(b))`, and look at their lengths.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: the S-curve

    a pair with Jaccard $s$ becomes a candidate if all $r$ rows of at least one of $b$
    bands agree: probability $1 - (1 - s^r)^b$. here the probability is measured on pairs of
    sets with known Jaccard, 200 seeds of your `minhash_signature` each, and compared with
    the formula for three ways to split 128 hashes into bands.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("with 32 bands of 4 rows, at what Jaccard is a pair a candidate half the time? is it near "
                    "the 0.8 dedup threshold?")
    p2
    return (p2,)


@app.cell
def _(p2):
    ui.gate(p2)
    sims = np.linspace(0.05, 0.95, 13)
    configs = [(64, 2), (32, 4), (16, 8)]
    measured = {c: [] for c in configs}
    for _s in sims:
        _m = int(round(400 * _s))
        _u = (400 - _m) // 2
        _a = {f"x{i}" for i in range(_m)} | {f"a{i}" for i in range(_u)}
        _b = {f"x{i}" for i in range(_m)} | {f"b{i}" for i in range(_u)}
        _sigs = [(ex.minhash_signature(_a, seed=s), ex.minhash_signature(_b, seed=s)) for s in range(200)]
        for (_bands, _r) in configs:
            _hit = [any(np.array_equal(x[i * _r:(i + 1) * _r], y[i * _r:(i + 1) * _r]) for i in range(_bands))
                    for x, y in _sigs]
            measured[(_bands, _r)].append(np.mean(_hit))
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _grid = np.linspace(0, 1, 200)
    for (_bands, _r), _col in zip(configs, ["C0", "C1", "C2"]):
        _ax.plot(_grid, 1 - (1 - _grid**_r) ** _bands, color=_col, label=f"b={_bands}, r={_r}: theory")
        _ax.plot(sims, measured[(_bands, _r)], "o", color=_col, ms=4)
        _ax.axvline((1 / _bands) ** (1 / _r), color=_col, ls=":", lw=1)
    _ax.axvline(0.8, color="k", lw=1, ls="--", label="dedup threshold 0.8")
    _ax.set(xlabel="Jaccard similarity s", ylabel="P(candidate pair)")
    _ax.legend(fontsize=8)
    _half = {c: (1 / c[0]) ** (1 / c[1]) for c in configs}
    mo.vstack([_fig, mo.md("dots: measured with your signatures; lines: $1 - (1 - s^r)^b$; dotted lines: "
                           "$(1/b)^{1/r}$, where each curve reaches about $1 - 1/e = 0.63$ (it crosses one half a little lower): "
                           + ", ".join(f"b={b}, r={r}: {v:.2f}" for (b, r), v in _half.items()) + ".")])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the default (32 bands of 4) put the steep part of the curve far
    *below* the 0.8 threshold, and what does that cost? which split would you use for a
    threshold of 0.5? what does the verification step in `minhash_dedup` protect you from?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: does deduplication matter for training?

    two identical small GPTs (your chapter 18 model: 2 layers, width 128, characters) trained
    for the same 800 steps: one on every page that passed the quality filter, duplicates and
    all, the other on the MinHash-deduplicated pages. both are evaluated on held-out
    Shakespeare that never appears in the crawl, and on the text of the ten **most copied**
    articles versus 100 articles copied once. one seed per model: differences of a few
    hundredths of a nat are within run-to-run noise.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("the raw corpus repeats some articles over a hundred times. how much lower is that model's loss on "
                    "them than on rarely copied ones? which model does better on held-out text?")
    p3
    return (p3,)


@app.cell
def _(articles, final, origin, p3, pages, passed):
    ui.gate(p3)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    tok = CharTokenizer().train("".join(sorted(set("".join(pages) + HELD_OUT))))
    _corpora = {"with duplicates": [pages[i] for i in passed], "deduplicated": [pages[i] for i in final]}
    _held = np.asarray(tok.encode(HELD_OUT[:60_000]), dtype=np.int64)
    _copies = Counter(origin[i] for i in passed if origin[i] >= 0)
    _ranked = [j for j, _ in _copies.most_common()]
    _popular, _rare = _ranked[:10], [j for j in _ranked if _copies[j] == 1][:100]
    def _article_loss(model, which):
        """mean per-character loss over the whole text of each article (in windows of 64), averaged over articles."""
        _out = []
        with torch.no_grad():
            for _j in which:
                _ids = torch.tensor(tok.encode(articles[_j]))
                _n = (len(_ids) - 1) // 64
                _x, _y = _ids[:64 * _n].view(_n, 64), _ids[1:64 * _n + 1].view(_n, 64)
                _out.append(model(_x, _y)[1].item())
        return float(np.mean(_out))

    train_rows = []
    for _name, _docs in _corpora.items():
        _data = np.asarray(tok.encode("\n\n".join(_docs)), dtype=np.int64)
        torch.manual_seed(0)
        _m = GPT(tok.vocab_size, block_size=64, n_layers=2, n_heads=4, d_model=128)
        _h = train_lm(_m, _data, 64, val_data=_held, batch_size=32, max_steps=800, lr=3e-3, warmup_steps=50,
                      eval_interval=800, eval_batches=20)
        train_rows.append((_name, len(_data), _h["train_loss"][-1], _h["val_loss"][-1],
                           _article_loss(_m, _popular), _article_loss(_m, _rare)))
    _seen_chars = 800 * 32 * 64
    _uniq = train_rows[1][1]
    mo.md("| training corpus | characters | train loss | held-out loss | 10 most-copied articles | 100 "
          "rarely copied articles |\n|---|---|---|---|---|---|\n"
          + "\n".join(f"| {n} | {c:,} | {a:.3f} | {b:.3f} | {p:.3f} | {r:.3f} |" for n, c, a, b, p, r in train_rows)
          + f"\n\nthe most-copied article appears {_copies[_ranked[0]]} times among the filtered pages. "
          f"the deduplicated corpus is {100 * _uniq / train_rows[0][1]:.0f}% of the size. each run sees "
          f"{_seen_chars:,} characters: {_seen_chars / train_rows[0][1]:.2f} epochs of the raw corpus, "
          f"{_seen_chars / _uniq:.2f} of the deduplicated one, which your `effective_tokens` values at "
          f"{ex.effective_tokens(_uniq, _seen_chars) / _uniq:.2f} epochs of fresh data.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why can't you compare the two training losses? what is the model trained on
    duplicates spending its capacity on, and how do the last two columns show it?
    connect this with the memorisation results of Lee et al. (2022) and Carlini et al. (2023).
    """)
    return


if __name__ == "__main__":
    app.run()
