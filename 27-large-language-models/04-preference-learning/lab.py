import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import copy
    import math
    import re
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from mlfs import datasets, notebooks, ui
    from mlfs.deeprl import ppo_clip_loss  # yours, from ch 24 · 04
    from mlfs.llm import Llama, build_sft_example, collate_sft, sft_loss  # yours, from topics 02-03
    from mlfs.text import BPETokenizer  # yours, from ch 17 · 02
    from mlfs.transformer import generate_cached, train_lm  # yours, from ch 18

    # your own implementation from exercise.py (the reference when MLFS_SOLUTIONS=1)
    ex = notebooks.load_sibling("exercise")

    plt.rcParams["figure.autolayout"] = True
    torch.set_num_threads(min(4, torch.get_num_threads()))
    OFFLINE = datasets.is_offline()
    TEXT = datasets.tiny_shakespeare()  # a synthetic stand-in offline
    CUT = int(0.9 * len(TEXT))
    TOK = BPETokenizer(257, special_tokens=("<|endoftext|>",)).train(TEXT[:1000])
    EOT = TOK.encode("<|endoftext|>")[0]
    VOCAB = set(re.findall(r"[a-z']+", TEXT[:CUT].lower()))
    # (pretrain, SFT, preference prompts, PPO iterations, DPO steps): offline runs are small, for CI
    CFG = dict(pre=200, sft=80, pref_prompts=150, ppo_iters=15, dpo_steps=50) if OFFLINE else \
        dict(pre=800, sft=300, pref_prompts=600, ppo_iters=50, dpo_steps=150)


@app.function
def continue_prompts(text, n, seed):
    """(prompt, reference continuation) pairs: 'Continue: <first half of a line>' and its second half."""
    rng = np.random.default_rng(seed)
    lines = [ln.strip() for ln in text.splitlines() if len(ln.split()) >= 8 and not ln.strip().endswith(":")]
    out = []
    for i in rng.choice(len(lines), min(n, len(lines)), replace=False):
        w = lines[i].split()
        out.append(("Continue: " + " ".join(w[:len(w) // 2]), " ".join(w[len(w) // 2:])))
    return out


@app.function
def gold_reward(prompt, response):
    """the hidden 'true' preference (a rule written for this lab, standing in for human judgement).

    real words, not repeated, about as many as the prompt's half-line: (fraction of words in the play's vocabulary)
    x (fraction of distinct words) - 0.15 |words - prompt words|; an empty answer scores -1.
    """
    w = re.findall(r"[A-Za-z']+", response)
    if not w:
        return -1.0
    valid = np.mean([x.lower() in VOCAB for x in w])
    distinct = len({x.lower() for x in w}) / len(w)
    return float(valid * distinct - 0.15 * abs(len(w) - (len(prompt.split()) - 1)))


@app.function
@torch.no_grad()
def sample_responses(model, prompts, k, *, seed=0, max_new=24):
    """k samples (temperature 1) per prompt: a list over prompts of lists of (tokens incl. EOT if emitted, text)."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    for p in prompts:
        pr = TOK.encode(f"User: {p}\nAssistant:")
        rows = generate_cached(model, torch.tensor([pr] * k), max_new, generator=gen)[:, len(pr):].tolist()
        res = []
        for r in rows:
            r = r[:r.index(EOT) + 1] if EOT in r else r
            res.append((r or [EOT], TOK.decode([t for t in r if t != EOT]).strip()))
        out.append(res)
    return out


@app.function
def lm_batch(prompts, responses):
    """(input_ids, targets) for the causal LM: prompt template + response tokens, loss on the response."""
    rows = []
    for p, r in zip(prompts, responses):
        h = TOK.encode(f"User: {p}\nAssistant:")
        rows.append(((h + r)[:-1], [-100] * (len(h) - 1) + r))
    return collate_sft(rows)


@app.function
def rm_batch(prompts, responses):
    """(input_ids, attention_mask) for the reward model: the whole conversation, right-padded."""
    seqs = [TOK.encode(f"User: {p}\nAssistant:") + r for p, r in zip(prompts, responses)]
    x = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
    m = torch.zeros_like(x, dtype=torch.bool)
    for i, s in enumerate(seqs):
        x[i, :len(s)], m[i, :len(s)] = torch.tensor(s), True
    return x, m


@app.function
@torch.no_grad()
def measure(policy, reference, rm, prompts, *, seed=7, k=4):
    """sample k responses per prompt from policy: (mean proxy reward, mean gold reward, KL estimate, a sample)."""
    ss = sample_responses(policy, prompts, k, seed=seed)
    plist = [p for p in prompts for _ in range(k)]
    resp, texts = [s[0] for row in ss for s in row], [s[1] for row in ss for s in row]
    x, y = lm_batch(plist, resp)
    kl = ex.sequence_logprobs(policy.eval(), x, y) - ex.sequence_logprobs(reference, x, y)
    xr, mr = rm_batch(plist, resp)
    return (rm(xr, mr).mean().item(), float(np.mean([gold_reward(p, t) for p, t in zip(plist, texts)])),
            kl.mean().item(), texts[0])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # preference learning · lab

    the experiments align a small `Llama` with **your** `RewardModel`, `bradley_terry_loss`,
    `kl_penalized_rewards`, `best_of_n_kl`, `dpo_step` and `sequence_logprobs`. the model is
    pretrained on tiny Shakespeare bytes and fine-tuned to finish half-lines ("Continue: ...").
    there are no human preferences on this machine, so they're **synthetic**: a hidden rule,
    `gold_reward`, plays the human (real words, not repeated, a sensible length), and labels
    pairs of the model's own answers, with noise. a reward model trained on those labels is
    the **proxy**; the gold rule is how you find out what optimizing the proxy really did
    (Gao et al., 2022). every curve is one run. for each experiment: write down a prediction,
    lock it in, run it, then explain what happened. about three minutes.
    """)
    return


@app.cell
def _():
    _speeches = [s for s in TEXT[:CUT].split("\n\n") if s.strip()]
    _ids = np.asarray(TOK.encode("<|endoftext|>".join(_speeches)), dtype=np.int64)
    _t0 = time.perf_counter()
    torch.manual_seed(0)
    sft = Llama(TOK.vocab_size, block_size=128, n_layers=3, n_heads=4, n_kv_heads=2, d_model=96)
    train_lm(sft, _ids, 128, batch_size=32, max_steps=CFG["pre"], lr=3e-3, warmup_steps=50,
             eval_interval=CFG["pre"], eval_batches=5)
    train_prompts = continue_prompts(TEXT[:CUT], 3000, 1)
    _opt = torch.optim.AdamW(sft.parameters(), lr=1e-3, weight_decay=0.0)
    _g = np.random.default_rng(0)
    sft.train()
    for _ in range(CFG["sft"]):
        _x, _y = collate_sft([build_sft_example(TOK, p, r) for p, r in
                              (train_prompts[i] for i in _g.integers(0, len(train_prompts), 32))])
        _loss = sft_loss(sft(_x)[0], _y)
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    sft.eval()
    for _p in sft.parameters():
        _p.requires_grad_(False)  # the SFT model is the frozen reference from here on
    _demo = sample_responses(sft, [train_prompts[0][0]], 3, seed=0)[0]
    mo.md(f"pretrained and fine-tuned in {time.perf_counter() - _t0:.0f} s. three answers to "
          f"*{train_prompts[0][0]}*, with their gold rewards:\n\n"
          + "\n".join(f"- `{t}` → {gold_reward(train_prompts[0][0], t):.2f}" for _, t in _demo))
    return sft, train_prompts


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## experiment 1: a reward model, and best-of-n against it

    eight answers per prompt for 600 training prompts, paired up and labelled by the gold rule
    through a Bradley–Terry coin: answer $a$ wins with probability $\sigma((g_a - g_b)/0.1)$.
    your `RewardModel` (initialized from the SFT model) is trained on 85% of the pairs with
    your `bradley_terry_loss` and checked on the rest. then **best-of-n**: for 60 held-out
    prompts, draw $n$ answers and keep the one the proxy likes best. its KL from the SFT
    model is your `best_of_n_kl(n)`. the plot shows the proxy reward and the gold reward of
    the chosen answers, and, for comparison, best-of-n chosen by the gold rule itself.
    """)
    return


@app.cell
def _():
    p1 = ui.predict("the reward model agrees with the gold rule on most pairs. as n grows from 1 to 64, what "
                    "happens to the proxy reward of the best-of-n answer, and to its gold reward?")
    p1
    return (p1,)


@app.cell
def _(p1, sft, train_prompts):
    ui.gate(p1)
    _rng = np.random.default_rng(0)
    _pp = [p for p, _ in train_prompts[:CFG["pref_prompts"]]]
    pairs = []
    for _p, _ss in zip(_pp, sample_responses(sft, _pp, 8, seed=1)):
        for _a, _b in ((0, 1), (2, 3), (4, 5), (6, 7)):
            _ga, _gb = gold_reward(_p, _ss[_a][1]), gold_reward(_p, _ss[_b][1])
            if _ga == _gb:
                continue
            _win = _a if _rng.random() < 1 / (1 + math.exp(-(_ga - _gb) / 0.1)) else _b
            _lose = _b if _win == _a else _a
            pairs.append((_p, _ss[_win][0], _ss[_lose][0], _ga if _win == _a else _gb, _gb if _win == _a else _ga))
    _agree = np.mean([w > l for *_, w, l in pairs])
    _split = int(0.85 * len(pairs))
    train_pairs, _test_pairs = pairs[:_split], pairs[_split:]
    torch.manual_seed(0)
    _backbone = copy.deepcopy(sft)
    for _q in _backbone.parameters():
        _q.requires_grad_(True)
    rm = ex.RewardModel(_backbone, 96)
    _opt = torch.optim.AdamW(rm.parameters(), lr=1e-4)
    rm.train()
    for _ in range(400):
        _bi = _rng.integers(0, len(train_pairs), 32)
        _xc, _mc = rm_batch([train_pairs[i][0] for i in _bi], [train_pairs[i][1] for i in _bi])
        _xr, _mr = rm_batch([train_pairs[i][0] for i in _bi], [train_pairs[i][2] for i in _bi])
        _loss = ex.bradley_terry_loss(rm(_xc, _mc), rm(_xr, _mr))
        _opt.zero_grad()
        _loss.backward()
        _opt.step()
    rm.eval()
    for _q in rm.parameters():
        _q.requires_grad_(False)
    with torch.no_grad():
        _xc, _mc = rm_batch([q[0] for q in _test_pairs], [q[1] for q in _test_pairs])
        _xr, _mr = rm_batch([q[0] for q in _test_pairs], [q[2] for q in _test_pairs])
        _rm_acc = (rm(_xc, _mc) > rm(_xr, _mr)).float().mean().item()
    _test = [p for p, _ in continue_prompts(TEXT[CUT:], 60, 2)]
    _ts = sample_responses(sft, _test, 64, seed=3)
    with torch.no_grad():
        _proxy = [rm(*rm_batch([p] * len(s), [x[0] for x in s])).numpy() for p, s in zip(_test, _ts)]
    _gold = [np.array([gold_reward(p, x[1]) for x in s]) for p, s in zip(_test, _ts)]
    rm_stats = (float(np.mean(np.concatenate(_proxy))), float(np.std(np.concatenate(_proxy))))
    ns = [1, 2, 4, 8, 16, 32, 64]
    bon = {"kl": [ex.best_of_n_kl(n) for n in ns],
           "proxy": [np.mean([pm[:n].max() for pm in _proxy]) for n in ns],
           "gold": [np.mean([gm[:n][np.argmax(pm[:n])] for gm, pm in zip(_gold, _proxy)]) for n in ns],
           "oracle": [np.mean([gm[:n].max() for gm in _gold]) for n in ns]}
    _fig, _ax = plt.subplots(figsize=(6, 3.4))
    _ax.plot(bon["kl"], bon["gold"], "o-", color="C0", label="gold reward, picked by the proxy")
    _ax.plot(bon["kl"], bon["oracle"], "s--", color="C2", label="gold reward, picked by gold (oracle)")
    _ax2 = _ax.twinx()
    _ax2.plot(bon["kl"], bon["proxy"], "^:", color="C3", label="proxy reward (right axis)")
    _ax.set(xlabel="KL(best-of-n || SFT) = log n - (n-1)/n  (nats)", ylabel="gold reward")
    _ax2.set_ylabel("proxy reward", color="C3")
    _h1, _l1 = _ax.get_legend_handles_labels()
    _h2, _l2 = _ax2.get_legend_handles_labels()
    _ax.legend(_h1 + _h2, _l1 + _l2, fontsize=7, loc="lower right")
    mo.vstack([mo.md(f"{len(pairs):,} pairs; the noisy labels agree with the gold rule on **{100 * _agree:.0f}%** of "
                     f"them. the reward model ranks **{100 * _rm_acc:.0f}%** of the {len(_test_pairs)} held-out pairs "
                     "the way their labels do."), _fig,
               mo.md("| n | KL | proxy | gold (proxy picks) | gold (oracle picks) |\n|---|---|---|---|---|\n"
                     + "\n".join(f"| {n} | {k:.2f} | {p:.2f} | {g:.3f} | {o:.3f} |" for n, k, p, g, o in
                                 zip(ns, bon["kl"], bon["proxy"], bon["gold"], bon["oracle"])))])
    return rm, rm_stats, train_pairs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** why does the proxy reward keep rising while the gold reward of the same
    answers stops rising (and here even falls)? which answers does a large $n$ select, and
    why are the proxy's errors concentrated exactly there? what does the gap to the oracle
    curve say about how much better a better reward model could do?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 2: RLHF with PPO

    now the policy itself is trained against the proxy. every iteration samples 4 answers to
    each of 16 prompts, scores them with the proxy, turns scores into per-token rewards with
    your `kl_penalized_rewards` (penalty $\beta$ per nat of log-ratio to the SFT model),
    computes returns to go, and takes two PPO steps (Adam, lr $2 \times 10^{{-4}}$) with your chapter 24 `ppo_clip_loss`. the
    proxy scores are standardized first (mean and standard deviation of the proxy over the SFT
    model's answers in experiment 1), so $\beta$ is in units of "reward standard deviations per
    nat". {CFG["ppo_iters"]} iterations with $\beta = 0$ (no leash) and $\beta = 0.1$. every 5
    iterations, 32 fixed held-out prompts measure the proxy reward, the gold reward and the KL.
    """)
    return


@app.cell
def _():
    p2 = ui.predict("with no KL penalty, what will the policy end up writing? where on the KL axis does the gold "
                    "reward peak, if it does?")
    p2
    return (p2,)


@app.cell
def _(p2, rm, rm_stats, sft, train_prompts):
    ui.gate(p2)
    _eval_prompts = [p for p, _ in continue_prompts(TEXT[CUT:], 32, 5)]  # held out: never trained on
    ppo_hist = {}
    for _beta in (0.0, 0.1):
        _pol = copy.deepcopy(sft)
        for _q in _pol.parameters():
            _q.requires_grad_(True)
        _opt = torch.optim.Adam(_pol.parameters(), lr=2e-4)
        _rng = np.random.default_rng(0)
        _hist = [(0, *measure(_pol, sft, rm, _eval_prompts))]
        for _it in range(1, CFG["ppo_iters"] + 1):
            _ps = [train_prompts[i][0] for i in _rng.integers(CFG["pref_prompts"], len(train_prompts), 16)]
            _ss = sample_responses(_pol, _ps, 4, seed=_it)
            _plist = [p for p in _ps for _ in range(4)]
            _resp = [s[0] for row in _ss for s in row]
            _x, _y = lm_batch(_plist, _resp)
            _mask = _y != -100
            with torch.no_grad():
                _old = ex.token_logprobs(_pol.eval(), _x, _y)
                _ref = ex.token_logprobs(sft, _x, _y)
                _score = (rm(*rm_batch(_plist, _resp)) - rm_stats[0]) / rm_stats[1]
            _rew = ex.kl_penalized_rewards(_score, _old, _ref, _mask, beta=_beta)
            _ret = _rew.flip(1).cumsum(1).flip(1)  # return to go, no discounting
            _adv = (_ret - _ret[_mask].mean()) / (_ret[_mask].std() + 1e-8)
            _pol.train()
            for _ in range(2):
                _new = ex.token_logprobs(_pol, _x, _y)
                _loss, _ = ppo_clip_loss(_new[_mask], _old[_mask], _adv[_mask])
                _opt.zero_grad()
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(_pol.parameters(), 1.0)
                _opt.step()
            if _it % 5 == 0:
                _hist.append((_it, *measure(_pol, sft, rm, _eval_prompts)))
        ppo_hist[_beta] = _hist
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for (_beta, _h), _c in zip(ppo_hist.items(), ["C3", "C0"]):
        _a1.plot([r[3] for r in _h], [r[2] for r in _h], "o-", color=_c, label=f"PPO, beta = {_beta}")
        _a2.plot([r[3] for r in _h], [r[1] for r in _h], "o-", color=_c, label=f"PPO, beta = {_beta}")
    _a1.set(xlabel="KL from the SFT model (nats)", ylabel="gold reward", title="what you wanted")
    _a2.set(xlabel="KL from the SFT model (nats)", ylabel="proxy reward", title="what you optimized")
    _a1.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| beta | iteration | proxy | gold | KL | a held-out answer |\n|---|---|---|---|---|---|\n"
                           + "\n".join(f"| {b} | {it} | {p:.2f} | {g:.2f} | {k:.1f} | `{s[:40]}` |"
                                       for b, h in ppo_hist.items() for it, p, g, k, s in (h[::2] if (len(h) - 1) % 2 == 0 else h[::2] + [h[-1]])))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** what does the unleashed policy discover, and why does the proxy reward it?
    (the reward model never saw anything like it during training.) why does the KL penalty
    prevent it, and what does it cost? if you only had the proxy curve (right), when would
    you have stopped training?
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## experiment 3: DPO, no reward model at all

    your `dpo_step` trains the SFT model directly on the same preference pairs (frozen SFT
    reference, batches of 32, lr $10^{{-4}}$), with $\beta = 0.1$ and $\beta = 0.01$, for
    {CFG["dpo_steps"]} steps. every 25 steps the same 32 held-out prompts measure gold reward,
    proxy reward and KL.
    """)
    return


@app.cell
def _():
    p3 = ui.predict("DPO has no reward model to exploit. can it over-optimize? how will beta = 0.01 differ from 0.1?")
    p3
    return (p3,)


@app.cell
def _(p3, rm, sft, train_pairs, train_prompts):
    ui.gate(p3)
    _eval_prompts = [p for p, _ in continue_prompts(TEXT[CUT:], 32, 5)]  # held out: never trained on
    dpo_hist = {}
    for _beta in (0.1, 0.01):
        _pol = copy.deepcopy(sft)
        for _q in _pol.parameters():
            _q.requires_grad_(True)
        _opt = torch.optim.Adam(_pol.parameters(), lr=1e-4)
        _rng = np.random.default_rng(0)
        _hist = [(0, *measure(_pol, sft, rm, _eval_prompts))]
        for _st in range(1, CFG["dpo_steps"] + 1):
            _bi = _rng.integers(0, len(train_pairs), 32)
            _xc, _yc = lm_batch([train_pairs[i][0] for i in _bi], [train_pairs[i][1] for i in _bi])
            _xr, _yr = lm_batch([train_pairs[i][0] for i in _bi], [train_pairs[i][2] for i in _bi])
            ex.dpo_step(_pol, sft, {"chosen_ids": _xc, "chosen_targets": _yc, "rejected_ids": _xr,
                                    "rejected_targets": _yr}, _opt, beta=_beta)
            if _st % 25 == 0:
                _hist.append((_st, *measure(_pol, sft, rm, _eval_prompts)))
        dpo_hist[_beta] = _hist
    _fig, _ax = plt.subplots(figsize=(6, 3.3))
    for (_beta, _h), _c in zip(dpo_hist.items(), ["C0", "C3"]):
        _ax.plot([r[3] for r in _h], [r[2] for r in _h], "o-", color=_c, label=f"DPO, beta = {_beta}")
    _ax.set(xscale="symlog", xlabel="KL from the SFT model (nats, symlog)", ylabel="gold reward")
    _ax.legend(fontsize=8)
    mo.vstack([_fig, mo.md("| beta | step | proxy | gold | KL | a held-out answer |\n|---|---|---|---|---|---|\n"
                           + "\n".join(f"| {b} | {st} | {p:.2f} | {g:.2f} | {k:.1f} | `{s[:40]!r}` |"
                                       for b, h in dpo_hist.items() for st, p, g, k, s in h))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    📝 **explain:** DPO's loss only asks that the chosen answer gain log-probability *relative*
    to the rejected one. what stops it from pushing both down, and what does a small $\beta$
    do to that brake? why does the KL keep growing even though the preference pairs never
    change? compare the best gold reward each method reached and the KL it took.
    """)
    return


if __name__ == "__main__":
    app.run()
