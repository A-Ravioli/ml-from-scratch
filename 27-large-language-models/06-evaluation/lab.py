import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import string
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import datasets, notebooks, ui
    from mlfs.llm import Llama  # yours, from topic 02
    from mlfs.text import CharTokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import train_lm  # yours, from ch 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    CUT = int(0.9 * len(TEXT))
    TOK = CharTokenizer().train(TEXT + string.ascii_letters + string.digits + string.punctuation + "\n")  # extras matter only offline
    STEPS = 150 if OFFLINE else 500
    N_ITEMS = 60 if OFFLINE else 200


@app.function
def make_cloze(text, n, seed=0):
    """a synthetic benchmark: the first half of a held-out line, and its true second half among 3 halves of other lines."""
    rng = np.random.default_rng(seed)
    lines = [ln.strip() for ln in text.splitlines() if 8 <= len(ln.split()) <= 14 and not ln.strip().endswith(":")]
    picks = rng.choice(len(lines), min(n, len(lines)), replace=False)
    items = []
    for i in picks:
        w = lines[i].split()
        h = len(w) // 2
        others = [lines[j].split() for j in rng.choice(len(lines), 3, replace=False) if j != i][:3]
        while len(others) < 3:
            others.append(lines[int(rng.integers(len(lines)))].split())
        endings = [" ".join(w[h:])] + [" ".join(o[len(o) // 2:]) for o in others]
        order = rng.permutation(4)
        items.append({"prefix": " ".join(w[:h]), "endings": [endings[k] for k in order],
                      "answer": int(np.argmax(order == 0)), "line": lines[i]})
    return items


@app.function
def render(items, template):
    """turn cloze items into evaluate_multiple_choice items under one prompt template."""
    out = []
    for it in items:
        e = it["endings"]
        if template == "raw":
            out.append({"question": it["prefix"], "choices": [" " + x for x in e], "answer": it["answer"]})
        elif template == "instruction":
            out.append({"question": f"Complete the line.\nLine: {it['prefix']}\nCompletion:",
                        "choices": [" " + x for x in e], "answer": it["answer"]})
        elif template == "speaker":
            out.append({"question": f"FIRST CITIZEN:\n{it['prefix']}", "choices": [" " + x for x in e],
                        "answer": it["answer"]})
        elif template == "lettered":
            opts = "\n".join(f"{L}. {x}" for L, x in zip("ABCD", e))
            out.append({"question": f"{it['prefix']}\n{opts}\nAnswer:", "choices": [" A", " B", " C", " D"],
                        "answer": it["answer"]})
    return out


@app.function
def per_item(model, items, *, normalize=True):
    """0/1 correctness per item (evaluate_multiple_choice on one item at a time)."""
    return np.array([ex.evaluate_multiple_choice(model, TOK, [it], normalize=normalize) for it in items])


@app.function
def train_model(text, seed=0):
    torch.manual_seed(seed)
    m = Llama(TOK.vocab_size, block_size=256, n_layers=4, n_heads=4, n_kv_heads=2, d_model=128)
    ids = np.asarray(TOK.encode(text), dtype=np.int64)
    train_lm(m, ids, 128, batch_size=32, max_steps=STEPS, lr=3e-3, warmup_steps=50, eval_interval=10**9,
             eval_batches=1, seed=seed)
    return m.eval()


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    # evaluation · lab

    every number here comes from **your** `loglikelihood`, `evaluate_multiple_choice`,
    `accuracy_ci`, `paired_accuracy_diff`, `contamination_flags` and `pairwise_judge`. the
    model is a character-level `Llama` (4 layers, width 128) trained on the first 90% of tiny
    Shakespeare ({STEPS} steps; a synthetic stand-in offline). the benchmark is **synthetic**:
    {N_ITEMS} *cloze* items built from held-out lines (the first half of a line, and its true
    ending among the endings of three other lines; chance is 25%). every interval is a 95%
    bootstrap over items. for each experiment: write down a prediction, lock it in, run it,
    then explain what happened. about three minutes.
    """)
    return


@app.cell
def _():
    _t0 = time.perf_counter()
    model = train_model(TEXT[:CUT])
    items = make_cloze(TEXT[CUT:], N_ITEMS)
    _it = items[0]
    mo.md(f"trained in {time.perf_counter() - _t0:.0f} s. an item: *{_it['prefix']} ...* → " + "; ".join(
        f"{'**' if k == _it['answer'] else ''}{e}{'**' if k == _it['answer'] else ''}" for k, e in enumerate(_it["endings"])))
    return items, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: one model, one benchmark, five scores

    the same model and items, scored five ways: the bare line prefix ("raw"), the raw format
    without per-token normalization, an instruction-style prompt, the prefix after a speaker
    name, and a lettered multiple-choice format where the model must produce " A" to " D"
    (the format of MMLU).
    """)
    return


@app.cell
def _():
    p1 = ui.predict("how far apart will the highest and lowest of the five accuracies be? which format will be "
                    "closest to chance?")
    p1
    return (p1,)


@app.cell
def _(items, model, p1):
    ui.gate(p1)
    variants = {"raw": ("raw", True), "raw, unnormalized": ("raw", False), "instruction": ("instruction", True),
                "after a speaker name": ("speaker", True), "lettered (A-D)": ("lettered", True)}
    scores = {k: per_item(model, render(items, t), normalize=nm) for k, (t, nm) in variants.items()}
    _ci = {k: ex.accuracy_ci(v, rng=0) for k, v in scores.items()}
    _fig, _ax = plt.subplots(figsize=(6.5, 3))
    _names = list(_ci)
    _ax.errorbar(range(len(_names)), [c[0] for c in _ci.values()],
                 yerr=[[c[0] - c[1] for c in _ci.values()], [c[2] - c[0] for c in _ci.values()]], fmt="o", capsize=4)
    _ax.axhline(0.25, color="gray", ls=":", label="chance")
    _ax.set_xticks(range(len(_names)), _names, rotation=20, fontsize=8)
    _ax.set(ylabel="accuracy", ylim=(0, 1))
    _ax.legend(fontsize=8)
    _d = ex.paired_accuracy_diff(scores["raw"], scores["instruction"], rng=0)
    mo.vstack([_fig, mo.md("| format | accuracy | 95% interval |\n|---|---|---|\n"
                           + "\n".join(f"| {k} | {a:.3f} | [{lo:.3f}, {hi:.3f}] |" for k, (a, lo, hi) in _ci.items())
                           + f"\n\nraw minus instruction, paired: {_d[0]:+.3f} [{_d[1]:+.3f}, {_d[2]:+.3f}].")])
    return (scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** the model's knowledge didn't change between rows: what did? why does the
    lettered format, the format of the most famous LLM benchmark, collapse to chance for this
    model (what would the model need to have learned to answer " B")? why does
    normalization matter when the endings have different lengths? which row would you report,
    and how would you report it honestly?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 2: contamination

    a second model, identical except that its training text also contains every benchmark
    line, ten times (the test set leaked into the "web crawl", as happens when benchmarks are
    posted online). both models are scored in the raw format, and your `contamination_flags`
    (word 8-grams) checks each item against each training set.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("how much higher will the contaminated model score? will its held-out loss on other "
                    "Shakespeare change?")
    p2
    return (p2,)


@app.cell
def _(items, model, p2, scores):
    ui.gate(p2)
    _leak = "\n".join(it["line"] for it in items) + "\n"
    _dirty_text = TEXT[:CUT] + ("\n" + _leak) * 10
    dirty = train_model(_dirty_text)
    _raw = render(items, "raw")
    _dirty_scores = per_item(dirty, _raw)
    _d = ex.paired_accuracy_diff(_dirty_scores, scores["raw"], rng=0)
    _texts = [it["line"] for it in items]
    _flag_clean = ex.contamination_flags(_texts, [TEXT[:CUT]], n=8)
    _flag_dirty = ex.contamination_flags(_texts, [_dirty_text], n=8)
    _other = make_cloze(TEXT[CUT:], N_ITEMS, seed=1)
    _other = [it for it in _other if it["line"] not in {x["line"] for x in items}]
    _held = np.asarray(TOK.encode(TEXT[CUT:]), dtype=np.int64)

    def _loss(m):
        g = torch.Generator().manual_seed(0)
        s = torch.randint(0, len(_held) - 129, (20,), generator=g)
        x = torch.stack([torch.as_tensor(_held[i:i + 129]) for i in s.tolist()])
        with torch.no_grad():
            return F.cross_entropy(m(x[:, :-1])[0].reshape(-1, TOK.vocab_size), x[:, 1:].reshape(-1)).item()

    mo.md(f"| | clean model | contaminated model |\n|---|---|---|\n"
          f"| accuracy on the benchmark | {scores['raw'].mean():.3f} | **{_dirty_scores.mean():.3f}** |\n"
          f"| accuracy on {len(_other)} fresh items | {per_item(model, render(_other, 'raw')).mean():.3f} | "
          f"{per_item(dirty, render(_other, 'raw')).mean():.3f} |\n"
          f"| loss on held-out text (nats/char) | {_loss(model):.3f} | {_loss(dirty):.3f} |\n"
          f"| items flagged by 8-gram overlap with its training text | {_flag_clean.sum()} of {len(items)} | "
          f"{_flag_dirty.sum()} of {len(items)} |\n\n"
          f"contaminated minus clean on the benchmark, paired: {_d[0]:+.3f} [{_d[1]:+.3f}, {_d[2]:+.3f}]. the leaked "
          f"copies are {100 * 10 * len(_leak) / len(_dirty_text):.1f}% of the contaminated training text, and the "
          f"contaminated model saw about {STEPS * 32 * 128 / len(_dirty_text):.2f} epochs of it.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** each benchmark line was in the training stream about 19 times (ten copies,
    almost two epochs). why does the score jump, and why not all the way to 100% (how much
    can a 0.7M-parameter model memorize in 500 steps)? the held-out loss and the fresh
    items barely move: what does that say about using a model's general quality to detect
    contamination? the overlap check found every leaked item here: when would 8-gram matching
    miss a contaminated item, or flag a clean one? what can an evaluator do when the training
    data isn't public?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 3: the model as a judge

    120 pairs: a held-out line and the same line with its words shuffled. two ways to let the
    model judge which is better. **asking**: the prompt "Which line is better? 1: ... 2: ...
    Better:" and the model's probability of " 1" against " 2". **likelihood**: the line with the
    higher per-character log-likelihood (your `loglikelihood`). each judge is used in both
    orders and through your `pairwise_judge`.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("asked directly, how often will the model prefer the real line? how often will it prefer "
                    "whichever line is shown first?")
    p3
    return (p3,)


@app.cell
def _(model, p3):
    ui.gate(p3)
    _rng = np.random.default_rng(3)
    _lines = [it["line"] for it in make_cloze(TEXT[CUT:], 120, seed=2)]
    _pairs = []
    for _ln in _lines:
        _w = _ln.split()
        _pairs.append((_ln, " ".join(_rng.permutation(_w))))

    def asked(prompt, first, second):
        q = f"Which line is better?\n1: {first}\n2: {second}\nBetter:"
        l1, _ = ex.loglikelihood(model, TOK, q, " 1")
        l2, _ = ex.loglikelihood(model, TOK, q, " 2")
        return float(1 / (1 + np.exp(l2 - l1)))

    def likelihood(prompt, first, second):
        s1 = ex.loglikelihood(model, TOK, "\n", first)[0] / len(first)
        s2 = ex.loglikelihood(model, TOK, "\n", second)[0] / len(second)
        return float(s1 > s2) if s1 != s2 else 0.5

    judge_rows = []
    for _name, _judge in (("asking", asked), ("likelihood", likelihood)):
        _fwd = np.array([_judge("", a, b) for a, b in _pairs])  # real line shown first
        _bwd = np.array([_judge("", b, a) for a, b in _pairs])  # real line shown second
        _deb = np.array([ex.pairwise_judge(_judge, "", a, b) for a, b in _pairs])
        judge_rows.append((_name, np.mean(_fwd > 0.5), np.mean(_bwd < 0.5), np.mean(np.r_[_fwd, 1 - _bwd] > 0.5),
                           np.mean(np.r_[_fwd, _bwd]), np.mean(_deb > 0.5), np.mean(_deb == 0.5)))
    mo.md("| judge | right, real first | right, real second | right overall | P(first is better), mean | "
          "right after pairwise_judge | ties after pairwise_judge |\n|---|---|---|---|---|---|---|\n"
          + "\n".join(f"| {n} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {e:.2f} | {f:.2f} |"
                      for n, a, b, c, d, e, f in judge_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what is the "asking" judge actually measuring? why does swapping the order
    expose it, and what does `pairwise_judge` report for a judge with no signal? the
    likelihood judge never sees a question at all: why is it so much better here, and why
    can't you use it to judge a large model's answers to open questions? which biases of real
    LLM judges (position, length, self-preference) would survive the swap?
    """)
    return


if __name__ == "__main__":
    app.run()
