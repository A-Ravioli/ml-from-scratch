import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import os
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn.functional as F

    from mlfs import checks, datasets
    from mlfs.llm import (  # yours, from this chapter's topics
        Llama, accuracy_ci, build_sft_example, collate_sft, contamination_flags, dpo_step, minhash_dedup,
        paired_accuracy_diff, quality_filter, sft_loss)
    from mlfs.text import BPETokenizer  # yours, from ch 17 · 02
    from mlfs.torchnn import get_device  # yours, from ch 15 · 07
    from mlfs.transformer import estimate_loss, generate_cached, train_lm  # yours, from ch 18

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    EOT = "<|endoftext|>"
    # small CPU config (offline, tests, verify, no GPU, or MLFS_CAPSTONE=small) vs full config (tiny Shakespeare on
    # cuda or mps). the full config took 23 minutes on an Apple M2 GPU: see brief.md.
    OFFLINE = datasets.is_offline()
    FULL = (not OFFLINE and str(get_device()) != "cpu" and os.environ.get("MLFS_CAPSTONE", "full") != "small")
    DEVICE = str(get_device()) if FULL else "cpu"
    if FULL:
        CONFIG = dict(model=dict(d_model=192, n_layers=6, n_heads=6, n_kv_heads=2), block_size=256,
                      pre_steps=1200, pre_batch=32, pre_lr=2e-3, sft_steps=600, sft_lr=1e-3, n_tasks=6000,
                      n_pref_prompts=800, k=8, dpo_steps=300, dpo_lr=5e-5, beta=0.1, n_eval=300,
                      seeds=[0, 1, 2], eval_batches=40)
    else:
        CONFIG = dict(model=dict(d_model=32, n_layers=1, n_heads=2, n_kv_heads=1), block_size=128,
                      pre_steps=60, pre_batch=8, pre_lr=3e-3, sft_steps=40, sft_lr=2e-3, n_tasks=400,
                      n_pref_prompts=20, k=4, dpo_steps=15, dpo_lr=1e-4, beta=0.1, n_eval=20,
                      seeds=[0], eval_batches=4)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 27 · capstone: your own assistant pipeline (reference solution)

    read [`brief.md`](brief.md) first. you'll take a language model from raw text to an aligned
    assistant and document what you made: pretrain a small `Llama` on a filtered, deduplicated
    corpus, teach it to follow instructions, align it to a hidden preference with DPO, evaluate
    every stage honestly, and write a model card. the functions are the pipeline; the runs
    and the model card below them are the capstone.

    | stage | you build |
    |---|---|
    | 1 | `prepare_corpus`: speeches → quality filter → MinHash dedup → tokens with end-of-text markers |
    | 2 | `make_tasks`, `gold_reward`: the synthetic assistant tasks and the hidden "human" preference |
    | 3 | `respond`, `sft`: answering a prompt, and supervised fine-tuning on the tasks |
    | 4 | `preference_pairs`, `dpo_train`: preferences from the gold rule, and DPO against a frozen reference |
    | 5 | `evaluate_assistant`: exact-match skills, gold reward, win rate and held-out loss, with intervals |
    | 6 | `model_card`: the documentation that ships with the model |

    when the checks are green, scroll down: run the pipeline, then write the model card.
    """)
    return


@app.function
def prepare_corpus(text, tokenizer, *, min_words=5, filter=True, dedup=True, seed=0):
    """stage 1: blank-line-separated speeches -> quality_filter(min_words) -> minhash_dedup -> ids joined by EOT; (ids int64, stats)."""
    docs = [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]
    kept = [d for d in docs if quality_filter(d, min_words=min_words)[0]] if filter else docs
    n_filtered = len(docs) - len(kept)
    unique = [kept[i] for i in minhash_dedup(kept, seed=seed)] if dedup else kept
    ids = np.asarray(tokenizer.encode(EOT.join(unique) + EOT), dtype=np.int64)
    stats = {"n_docs": len(docs), "n_filtered": n_filtered, "n_duplicates": len(kept) - len(unique),
             "n_kept": len(unique), "n_tokens": int(len(ids))}
    return ids, stats


@app.cell(hide_code=True)
def _():
    checks.run(1, _tests="test_capstone.py", prepare_corpus=prepare_corpus)
    return


@app.function
def make_tasks(text, n, *, split="train", seed=0):
    """stage 2: n synthetic tasks dict(kind, prompt, answer): 'continue' (half a line), 'capitals', 'count' (a word); split by line/word hash."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))

    def in_split(s):  # a stable 80/20 split by content, so train and test never share a line or a word
        return (sum(map(ord, s)) * 2654435761 % 2**32) % 5 != 0 if split == "train" else \
            (sum(map(ord, s)) * 2654435761 % 2**32) % 5 == 0

    lines = sorted({ln.strip() for ln in text.splitlines()
                    if 8 <= len(ln.split()) <= 14 and not ln.strip().endswith(":") and in_split(ln.strip())})
    words = sorted({w for w in re.findall(r"[a-z]+", text) if 3 <= len(w) <= 7 and in_split(w)})
    tasks = []
    for _ in range(n):
        kind = ("continue", "capitals", "count")[int(rng.integers(3))]
        if kind == "continue":
            w = lines[int(rng.integers(len(lines)))].split()
            h = len(w) // 2
            tasks.append({"kind": kind, "prompt": "Continue: " + " ".join(w[:h]), "answer": " ".join(w[h:])})
        else:
            word = words[int(rng.integers(len(words)))]
            if kind == "capitals":
                tasks.append({"kind": kind, "prompt": f"Write {word} in capitals.", "answer": word.upper()})
            else:
                tasks.append({"kind": kind, "prompt": f"How many letters are in {word}?", "answer": str(len(word))})
    return tasks


@app.function
def gold_reward(task, response, vocab):
    """stage 2: hidden preference: skills 1.0 if exact else 0.0; continue: valid-word x distinct-word fraction - 0.15 |words - target words|; empty -1."""
    if task["kind"] != "continue":
        return 1.0 if response.strip() == task["answer"] else 0.0
    w = re.findall(r"[A-Za-z']+", response)
    if not w:
        return -1.0
    valid = np.mean([x.lower() in vocab for x in w])
    distinct = len({x.lower() for x in w}) / len(w)
    return float(valid * distinct - 0.15 * abs(len(w) - len(task["answer"].split())))


@app.cell(hide_code=True)
def _():
    checks.run(2, _tests="test_capstone.py", make_tasks=make_tasks, gold_reward=gold_reward)
    return


@app.function
@torch.no_grad()
def respond(model, tokenizer, prompt, *, n=1, max_new_tokens=32, temperature=1.0, generator=None, device="cpu"):
    """stage 3: n sampled responses (cut at EOT) to 'User: {prompt}\\nAssistant:'; list of (token ids incl. EOT if emitted, text)."""
    eot = tokenizer.encode(EOT)[0]
    head = tokenizer.encode(f"User: {prompt}\nAssistant:")
    room = min(max_new_tokens, model.block_size - len(head))
    x = torch.tensor([head] * n, device=device)
    rows = generate_cached(model, x, room, temperature=temperature, generator=generator)[:, len(head):].tolist()
    out = []
    for r in rows:
        r = r[:r.index(eot) + 1] if eot in r else r
        out.append((r, tokenizer.decode([t for t in r if t != eot]).strip()))
    return out


@app.function
def sft(model, tokenizer, tasks, *, steps, lr=1e-3, batch_size=32, seed=0, device="cpu"):
    """stage 3: AdamW on build_sft_example batches (response-only loss); returns the list of per-step losses; leaves eval mode."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    model.train()
    losses = []
    for _ in range(steps):
        batch = [build_sft_example(tokenizer, tasks[i]["prompt"], tasks[i]["answer"], max_len=model.block_size)
                 for i in rng.integers(0, len(tasks), batch_size)]
        x, y = collate_sft(batch)
        loss = sft_loss(model(x.to(device))[0], y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


@app.cell(hide_code=True)
def _():
    checks.run(3, _tests="test_capstone.py", respond=respond, sft=sft)
    return


@app.function
def preference_pairs(model, tokenizer, tasks, vocab, *, k=8, seed=0, device="cpu"):
    """stage 4: k samples per task, paired (0,1), (2,3), ...; the higher gold reward is chosen (ties dropped); list of dict(prompt, chosen, rejected) token lists."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pairs = []
    for t in tasks:
        samples = respond(model, tokenizer, t["prompt"], n=k, generator=g, device=device)
        for a, b in zip(samples[0::2], samples[1::2]):
            ra, rb = gold_reward(t, a[1], vocab), gold_reward(t, b[1], vocab)
            if ra == rb:
                continue
            win, lose = (a, b) if ra > rb else (b, a)
            pairs.append({"prompt": t["prompt"], "chosen": win[0], "rejected": lose[0]})
    return pairs


@app.function
def dpo_train(policy, reference, tokenizer, pairs, *, steps, beta=0.1, lr=5e-5, batch_size=16, seed=0, device="cpu"):
    """stage 4: dpo_step on batches of pairs against the frozen reference; returns the list of dpo_step metric dicts."""
    for p in reference.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    def batch(key, idx):
        rows = []
        for i in idx:
            head = tokenizer.encode(f"User: {pairs[i]['prompt']}\nAssistant:")
            resp = pairs[i][key]
            rows.append(((head + resp)[:-1], [-100] * (len(head) - 1) + resp))
        x, y = collate_sft(rows)
        return x.to(device), y.to(device)

    history = []
    for _ in range(steps):
        idx = rng.integers(0, len(pairs), batch_size)
        xc, yc = batch("chosen", idx)
        xr, yr = batch("rejected", idx)
        history.append(dpo_step(policy, reference, {"chosen_ids": xc, "chosen_targets": yc, "rejected_ids": xr,
                                                    "rejected_targets": yr}, opt, beta=beta))
    policy.eval()
    return history


@app.cell(hide_code=True)
def _():
    checks.run(4, _tests="test_capstone.py", preference_pairs=preference_pairs, dpo_train=dpo_train, gold_reward=gold_reward)
    return


@app.function
def evaluate_assistant(model, tokenizer, tasks, vocab, *, reference=None, held_out=None, seed=0, device="cpu",
                       eval_batches=20):
    """stage 5: dict of (value, lo, hi): skill_acc (greedy exact match), gold (one sample per continue task), win_rate vs reference (ties 1/2), and lm_loss."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    skills = [t for t in tasks if t["kind"] != "continue"]
    conts = [t for t in tasks if t["kind"] == "continue"]
    correct = np.array([respond(model, tokenizer, t["prompt"], temperature=0.0, device=device)[0][1] == t["answer"]
                        for t in skills], dtype=float)
    gold = np.array([gold_reward(t, respond(model, tokenizer, t["prompt"], generator=g, device=device)[0][1], vocab)
                     for t in conts])
    rng = np.random.default_rng(seed)
    boot = gold[rng.integers(0, len(gold), (2000, len(gold)))].mean(1)
    out = {"skill_acc": accuracy_ci(correct, rng=seed), "gold": (float(gold.mean()), float(np.quantile(boot, 0.025)),
                                                                   float(np.quantile(boot, 0.975)))}
    if reference is not None:
        g2 = torch.Generator(device="cpu").manual_seed(seed)
        ref_gold = np.array([gold_reward(t, respond(reference, tokenizer, t["prompt"], generator=g2, device=device)[0][1],
                                         vocab) for t in conts])
        wins = (gold > ref_gold) + 0.5 * (gold == ref_gold)
        out["win_rate"] = accuracy_ci(wins, rng=seed)
    if held_out is not None:
        loss = estimate_loss(model, held_out, model.block_size, batch_size=8, n_batches=eval_batches, device=device)
        out["lm_loss"] = (float(loss), float("nan"), float("nan"))
    return out


@app.cell(hide_code=True)
def _():
    checks.run(5, _tests="test_capstone.py", evaluate_assistant=evaluate_assistant, gold_reward=gold_reward,
               make_tasks=make_tasks, respond=respond)
    return


@app.function
def model_card(name, config, results, notes=""):
    """stage 6: markdown card: model details, intended use, training data, training procedure (a readable list), evaluation (every result with its interval), limitations (incl. every stage-to-stage drop in skill_acc or rise in lm_loss), ethical considerations."""
    def fmt(v):
        val, lo, hi = v
        return f"{val:.3f}" if np.isnan(lo) else f"{val:.3f} [{lo:.3f}, {hi:.3f}]"

    def readable(value):
        if isinstance(value, dict):
            return ", ".join(f"{k} {readable(v)}" for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return ", ".join(readable(v) for v in value)
        return str(value)

    procedure = "\n".join(f"- {k.replace('_', ' ')}: {readable(v)}" for k, v in config.items())
    order = list(results)
    costs = []
    for metric, worse, what in (("skill_acc", lambda a, b: b < a, "skill accuracy fell"),
                                ("lm_loss", lambda a, b: b > a, "held-out loss rose")):
        have = [st for st in order if metric in results[st]]
        for a, b in zip(have, have[1:]):
            va, vb = results[a][metric][0], results[b][metric][0]
            if worse(va, vb):
                costs.append(f"{what} from {va:.3f} ({a}) to {vb:.3f} ({b})")
    cost_text = ("measured costs of post-training: " + "; ".join(costs) + ". ") if costs else ""

    stages = sorted({k for r in results.values() for k in r})
    rows = "\n".join(f"| {stage} | " + " | ".join(fmt(r[m]) if m in r else "–" for m in stages) + " |"
                     for stage, r in results.items())
    model = config.get("model", {})
    return f"""# model card: {name}

## model details

a Llama-style decoder-only language model (RMSNorm, SwiGLU, rotary positions, grouped-query attention) with
{model.get('n_layers', '?')} layers, width {model.get('d_model', '?')}, {model.get('n_heads', '?')} query heads and
{model.get('n_kv_heads', '?')} key/value heads, context {config.get('block_size', '?')} byte-level tokens. trained from
scratch in this course's chapter 27 capstone.

## intended use

a teaching artifact: studying the pretraining → supervised fine-tuning → preference optimization pipeline at small
scale. not intended for any real use: it writes Shakespeare-flavoured text and answers three toy instruction types.

## training data

pretraining: tiny Shakespeare (about 1M characters of plays), split into speeches, filtered with heuristic quality
rules and MinHash-deduplicated. instruction and preference data are **synthetic**: generated from templates
(finish a line, capitalize a word, count its letters), with preferences labelled by a hidden rule standing in for
human judgement. no personal data.

## training procedure

{procedure}

## evaluation

| stage | {' | '.join(stages)} |
|---|{'---|' * len(stages)}
{rows}

95% bootstrap intervals over evaluation items in brackets; skill accuracy is greedy exact match on held-out words,
gold is the hidden preference rule on one sampled answer per held-out line, win rate is against the previous stage
(ties count 1/2), lm loss is nats per byte on held-out text.

## limitations

{cost_text}tiny model and data: frequent misspellings and nonsense; skills generalize only partially to unseen
words; the preference signal is a hand-written rule, so "aligned" means aligned to that rule, which can be gamed
(repetition, length). results are from the runs reported here; seeds are listed where more than one was run, and GPU
runs aren't bit-reproducible, so a rerun moves every number within (and sometimes beyond) its interval.

the evaluation doesn't cover: answers to any instruction outside the three templates, multi-turn use, factual
accuracy, harmful or biased content, robustness to rephrased prompts, or agreement with real human preferences (the
"gold" rule is a stand-in written for this capstone).

## ethical considerations

trained on public-domain plays; generated text can still be offensive or archaic in ways the filter doesn't catch.
the preference data encode one rule's notion of a good answer, not a population's.

{notes}
"""


@app.cell(hide_code=True)
def _():
    checks.run(6, _tests="test_capstone.py", model_card=model_card)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## the pipeline

    config: **{"full" if FULL else "small"}** on **{DEVICE}** (offline: {OFFLINE}). the full config takes
    about 25 minutes on an Apple-silicon GPU (`brief.md` has the measured times); the small one, used by the
    tests, `verify.py` and CPU-only machines, about a minute.
    """)
    return


@app.cell
def _():
    t_start = time.perf_counter()
    text = datasets.tiny_shakespeare()
    cut = int(0.9 * len(text))
    tok = BPETokenizer(257, special_tokens=(EOT,)).train(text[:1000])  # bytes + the end-of-text token
    vocab = set(re.findall(r"[a-z']+", text[:cut].lower()))
    if OFFLINE:  # the synthetic stand-in has no blank lines and isn't English (the filter would reject it all)
        _lines = text[:cut].splitlines()
        _pre_text = "\n\n".join("\n".join(_lines[i:i + 8]) for i in range(0, len(_lines), 8))
    else:
        _pre_text = text[:cut]
    train_ids, corpus_stats = prepare_corpus(_pre_text, tok, filter=not OFFLINE)
    held_out = np.asarray(tok.encode(text[cut:]), dtype=np.int64)
    train_tasks = make_tasks(text[:cut], CONFIG["n_tasks"], split="train")
    test_tasks = make_tasks(text[cut:], CONFIG["n_eval"], split="test", seed=1)
    _cont = [t["prompt"].removeprefix("Continue: ") + " " + t["answer"] for t in test_tasks if t["kind"] == "continue"]
    n_leaked = int(contamination_flags(_cont, [text[:cut]], n=8).sum())
    pref_tasks = [t for t in make_tasks(text[:cut], CONFIG["n_pref_prompts"] * 4, split="train", seed=2)
                  if t["kind"] == "continue"][:CONFIG["n_pref_prompts"]]
    mo.md(f"corpus: {corpus_stats}. {len(train_tasks)} training tasks, {len(test_tasks)} test tasks (lines from the "
          f"held-out tenth, words disjoint from the training words; {n_leaked} of {len(_cont)} test lines share an "
          f"8-gram with the pretraining text), {len(pref_tasks)} preference prompts.")
    return corpus_stats, held_out, pref_tasks, t_start, test_tasks, tok, train_ids, train_tasks, vocab


@app.cell
def _(held_out, tok, train_ids):
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    base = Llama(tok.vocab_size, block_size=CONFIG["block_size"], **CONFIG["model"]).to(DEVICE)
    pre_hist = train_lm(base, train_ids, CONFIG["block_size"], val_data=held_out, batch_size=CONFIG["pre_batch"],
                        max_steps=CONFIG["pre_steps"], lr=CONFIG["pre_lr"], warmup_steps=min(100, CONFIG["pre_steps"] // 5),
                        eval_interval=max(CONFIG["pre_steps"] // 10, 1), eval_batches=CONFIG["eval_batches"],
                        device=DEVICE, seed=0)
    t_pre = time.perf_counter() - _t0
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.plot(pre_hist["step"], pre_hist["train_loss"], label="train")
    _ax.plot(pre_hist["step"], pre_hist["val_loss"], label="held out")
    _ax.set(xlabel="step", ylabel="loss (nats per byte)", title="pretraining")
    _ax.legend()
    _sample = respond(base, tok, "ROMEO:", n=1, max_new_tokens=120, generator=torch.Generator().manual_seed(0),
                      device=DEVICE)[0][1]
    mo.vstack([_fig, mo.md(f"pretrained {sum(p.numel() for p in base.parameters()):,} parameters in {t_pre / 60:.1f} min; "
                           f"held-out loss {pre_hist['val_loss'][-1]:.3f}. the base model, prompted with a chat turn, "
                           f"doesn't know the format yet:\n\n```\n{_sample}\n```")])
    return base, pre_hist, t_pre


@app.cell
def _(base, held_out, pref_tasks, test_tasks, tok, train_tasks, vocab):
    runs = {}
    _t0 = time.perf_counter()
    for _seed in CONFIG["seeds"]:
        torch.manual_seed(_seed)
        _sft = copy.deepcopy(base)
        _sft_losses = sft(_sft, tok, train_tasks, steps=CONFIG["sft_steps"], lr=CONFIG["sft_lr"], seed=_seed,
                          device=DEVICE)
        _pairs = preference_pairs(_sft, tok, pref_tasks, vocab, k=CONFIG["k"], seed=_seed, device=DEVICE)
        _dpo = copy.deepcopy(_sft)
        _hist = dpo_train(_dpo, _sft, tok, _pairs, steps=CONFIG["dpo_steps"], beta=CONFIG["beta"], lr=CONFIG["dpo_lr"],
                          seed=_seed, device=DEVICE)
        runs[_seed] = {
            "sft_losses": _sft_losses, "n_pairs": len(_pairs), "dpo": _hist,
            "results": {"base": evaluate_assistant(base, tok, test_tasks, vocab, held_out=held_out, seed=_seed,
                                                   device=DEVICE, eval_batches=CONFIG["eval_batches"]),
                        "sft": evaluate_assistant(_sft, tok, test_tasks, vocab, reference=base, held_out=held_out,
                                                  seed=_seed, device=DEVICE, eval_batches=CONFIG["eval_batches"]),
                        "dpo": evaluate_assistant(_dpo, tok, test_tasks, vocab, reference=_sft, held_out=held_out,
                                                  seed=_seed, device=DEVICE, eval_batches=CONFIG["eval_batches"])},
            "models": (_sft, _dpo)}
        _skills = [t for t in test_tasks if t["kind"] != "continue"]
        _right = {name: np.array([respond(m, tok, t["prompt"], temperature=0.0, device=DEVICE)[0][1] == t["answer"]
                                  for t in _skills]) for name, m in (("sft", _sft), ("dpo", _dpo))}
        runs[_seed]["skill_diff"] = paired_accuracy_diff(_right["dpo"], _right["sft"], rng=_seed)
    t_post = time.perf_counter() - _t0
    return runs, t_post


@app.cell
def _(runs, t_post):
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9, 3))
    for _seed, _r in runs.items():
        _a1.plot(np.convolve(_r["sft_losses"], np.ones(10) / 10, mode="valid"), label=f"seed {_seed}")
        _a2.plot([h["margin"] for h in _r["dpo"]], label=f"seed {_seed}")
    _a1.set(xlabel="SFT step", ylabel="response loss (10-step mean)")
    _a2.set(xlabel="DPO step", ylabel="implicit reward margin")
    _a1.legend(fontsize=8)
    _metrics = ["skill_acc", "gold", "win_rate", "lm_loss"]
    _rows = []
    for _seed, _r in runs.items():
        for _stage, _res in _r["results"].items():
            _rows.append(f"| {_seed} | {_stage} | " + " | ".join(
                ("–" if _m not in _res else f"{_res[_m][0]:.3f}" + ("" if np.isnan(_res[_m][1]) else
                                                                   f" [{_res[_m][1]:.3f}, {_res[_m][2]:.3f}]"))
                for _m in _metrics) + " |")
    mo.vstack([_fig, mo.md(f"post-training for {len(runs)} seed(s): {t_post / 60:.1f} min. preference pairs: "
                           + ", ".join(str(r["n_pairs"]) for r in runs.values()) + ".\n\n| seed | stage | "
                           + " | ".join(_metrics) + " |\n|---|---|" + "---|" * len(_metrics) + "\n" + "\n".join(_rows)
                           + "\n\nskill accuracy, DPO minus SFT (paired bootstrap over the same items): "
                           + "; ".join(f"seed {s}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for s, (d, lo, hi) in
                                       ((s, r["skill_diff"]) for s, r in runs.items())) + ".")])
    return


@app.cell
def _(runs, test_tasks, tok):
    _sft, _dpo = runs[CONFIG["seeds"][0]]["models"]
    _g1, _g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    _show = [t for t in test_tasks if t["kind"] == "continue"][:3] + [t for t in test_tasks if t["kind"] != "continue"][:3]
    mo.md("illustrative answers (seed 0, one sample each: a rerun, especially on a GPU, gives different ones):\n\n"
          "| prompt | reference answer | SFT | DPO |\n|---|---|---|---|\n" + "\n".join(
        f"| {t['prompt']} | {t['answer']} | {respond(_sft, tok, t['prompt'], generator=_g1, device=DEVICE)[0][1]} | "
        f"{respond(_dpo, tok, t['prompt'], generator=_g2, device=DEVICE)[0][1]} |" for t in _show))
    return


@app.cell
def _(corpus_stats, pre_hist, runs, t_pre, t_post, t_start):
    _res = runs[CONFIG["seeds"][0]]["results"]
    _spread = {m: [r["results"]["dpo"][m][0] for r in runs.values()] for m in ("skill_acc", "gold", "win_rate")}
    card = model_card(
        "shakespeare-assistant-tiny", dict(CONFIG, device=DEVICE, corpus=corpus_stats), _res,
        notes=(f"runs: pretraining once ({t_pre / 60:.1f} min, final held-out loss {pre_hist['val_loss'][-1]:.3f}); "
               f"post-training with seeds {CONFIG['seeds']} ({t_post / 60:.1f} min). spread of the DPO model over seeds: "
               + ", ".join(f"{m} {min(v):.3f}–{max(v):.3f}" for m, v in _spread.items())
               + ". skill accuracy, DPO minus SFT (paired): " + "; ".join(
                   f"seed {s} {r['skill_diff'][0]:+.3f} [{r['skill_diff'][1]:+.3f}, {r['skill_diff'][2]:+.3f}]"
                   for s, r in runs.items())
               + f". total {(time.perf_counter() - t_start) / 60:.1f} min on {DEVICE}."))
    mo.md(card)
    return (card,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## write-up

    replace each prompt with your answer, using the numbers above (and your reruns). the model card
    above is generated; edit its limitations and notes so that they describe *your* model.

    1. **the data.** how much of the corpus did filtering and deduplication remove, and what kind of
       text was it? show two examples of each.
    2. **each stage's effect.** what did SFT change (format, skills, held-out loss), and what did DPO
       change on top? separate effects larger than the seed spread from those within it.
    3. **alignment to what.** DPO optimized a rule you wrote. find one way the rule can be gamed and
       check whether your DPO model games it (look at its answers, not only its scores).
    4. **evaluation honesty.** which of your numbers would you trust in a paper, and which wouldn't
       you? what would you need to change (items, seeds, judges) to trust the rest?
    5. **the model card.** what does your card leave out that a user of a real model would need?
    """)
    return


if __name__ == "__main__":
    app.run()
