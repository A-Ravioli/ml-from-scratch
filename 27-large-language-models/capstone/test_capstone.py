"""Checks for capstone.py (your own assistant pipeline).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. everything runs on the CPU with tiny
Llamas and a byte-level tokenizer on text written here; nothing is downloaded and there are no quality thresholds
that need a real training run. the corpus step for filtering, deduplication and end-of-text markers; the task
generator for correct answers and disjoint splits, and the gold rule against its formula; responding, SFT
(it learns, uses lr, steps and the response-only loss, leaves eval mode), preference pairs (always ordered by the
gold rule, ties dropped, k used), DPO training (frozen reference, beta used, learning); the evaluation's
structure and identities (a model against itself wins exactly half); the model card's sections and table.
"""
import copy
import math
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlfs.llm import Llama, build_sft_example, collate_sft, sft_loss  # noqa: E402
from mlfs.testing import bounded  # noqa: E402
from mlfs.text import BPETokenizer  # noqa: E402

EOT = "<|endoftext|>"
WORDS = ("crown sword river castle winter falcon mirror lantern harbor meadow candle thunder garden silver bridge "
         "orchard tempest willow marble shadow anchor ember").split()


def _text(seed=0, n_lines=400):
    rng = np.random.default_rng(seed)
    lines = [" ".join(rng.choice(WORDS, size=int(rng.integers(8, 13)))) for _ in range(n_lines)]
    return "\n".join(lines)


def _tok():
    return BPETokenizer(257, special_tokens=(EOT,)).train("abc")


def _model(seed=0, block=96):
    torch.manual_seed(seed)
    return Llama(257, block_size=block, n_layers=1, n_heads=2, n_kv_heads=1, d_model=32)


VOCAB = set(WORDS)


# ---------------------------------------------------------------- stage 1: corpus


def test_stage1_corpus(nb):
    good = ["the harbor was quiet when the boats came back from the river at night",
            "a falcon watched from the castle wall while the candles burned in the town",
            "the winter garden lay under silver frost and the willow bent over the bridge"]
    junk = ["# # # 12 34", "ok"]
    text = "\n\n".join(good + [good[0]] + junk + [good[1].upper()])
    tok = _tok()
    ids, stats = nb.prepare_corpus(text, tok)
    assert ids.dtype == np.int64 and ids.ndim == 1
    assert stats["n_docs"] == 7 and stats["n_filtered"] == 2 and stats["n_duplicates"] == 2 and stats["n_kept"] == 3, (
        f"7 speeches: 2 fail the quality filter (too short / symbols), 2 are duplicates (an exact copy and a re-cased "
        f"one), 3 remain; got {stats}")
    eot = tok.encode(EOT)[0]
    assert ids[-1] == eot and int((ids == eot).sum()) == 3, "each kept speech is followed by one end-of-text token"
    assert stats["n_tokens"] == len(ids)
    assert tok.decode(ids.tolist()) == EOT.join(good) + EOT, "kept speeches in their original order, joined by EOT"
    _, no_dedup = nb.prepare_corpus(text, tok, dedup=False)
    assert no_dedup["n_duplicates"] == 0 and no_dedup["n_kept"] == 5, "dedup=False keeps the copies"
    _, strict = nb.prepare_corpus(text, tok, min_words=15)
    assert strict["n_filtered"] > stats["n_filtered"], "min_words is passed to the quality filter (15 words removes more)"
    from mlfs.llm import minhash_dedup
    rng = np.random.default_rng(2)
    base = [str(w) for w in rng.choice(WORDS, size=60)]
    for n_edit in range(1, 8):  # find a near-copy whose fate depends on the hash functions (the seed)
        near = list(base)
        for i in rng.choice(60, size=n_edit, replace=False):
            near[i] = "zq" + near[i]
        pair = [" ".join(base), " ".join(near)]
        fates = [len(minhash_dedup(pair, seed=s)) for s in range(8)]
        if len(set(fates)) > 1:
            for sd, f in enumerate(fates):
                got = nb.prepare_corpus("\n\n".join(pair), tok, seed=sd)[1]["n_duplicates"]
                assert got == 2 - f, f"seed={sd}: pass seed to minhash_dedup (duplicates {got}, want {2 - f})"
            break
    _, nofilter = nb.prepare_corpus(text, tok, filter=False)
    assert nofilter["n_filtered"] == 0, "filter=False skips the quality filter"


# ---------------------------------------------------------------- stage 2: tasks and gold rule


def test_stage2_tasks(nb):
    text = _text()
    tr = nb.make_tasks(text, 300, split="train", seed=0)
    te = nb.make_tasks(text, 300, split="test", seed=0)
    assert len(tr) == 300 and {t["kind"] for t in tr} == {"continue", "capitals", "count"}
    for t in tr[:100]:
        if t["kind"] == "capitals":
            w = re.match(r"Write (\w+) in capitals\.", t["prompt"]).group(1)
            assert t["answer"] == w.upper()
        elif t["kind"] == "count":
            w = re.match(r"How many letters are in (\w+)\?", t["prompt"]).group(1)
            assert t["answer"] == str(len(w))
        else:
            assert t["prompt"].startswith("Continue: ")
            line = t["prompt"].removeprefix("Continue: ") + " " + t["answer"]
            assert line in text.splitlines(), "a continue task splits a real line into prompt and answer"
            assert abs(len(t["prompt"].split()) - 1 - len(t["answer"].split())) <= 1, "split at the middle word"
    words = lambda ts: {re.findall(r"(\w+)", t["prompt"])[1 if t["kind"] == "capitals" else -1]
                        for t in ts if t["kind"] != "continue"}
    lines = lambda ts: {t["prompt"] for t in ts if t["kind"] == "continue"}
    assert not (words(tr) & words(te)), "train and test skill words are disjoint"
    assert not (lines(tr) & lines(te)), "train and test lines are disjoint"
    assert nb.make_tasks(text, 50, seed=3) == nb.make_tasks(text, 50, seed=3), "deterministic for a seed"
    assert nb.make_tasks(text, 50, seed=3) != nb.make_tasks(text, 50, seed=4)
    with pytest.raises(ValueError):
        nb.make_tasks(text, 5, split="dev")


def test_stage2_gold(nb):
    cap = {"kind": "capitals", "prompt": "Write crown in capitals.", "answer": "CROWN"}
    assert nb.gold_reward(cap, " CROWN ", VOCAB) == 1.0 and nb.gold_reward(cap, "CROWNS", VOCAB) == 0.0, (
        "skills: 1.0 for the exact answer (surrounding whitespace ignored), else 0.0")
    cont = {"kind": "continue", "prompt": "Continue: a b c", "answer": "river castle winter"}
    got = nb.gold_reward(cont, "river river zzz, castle", VOCAB)
    want = (3 / 4) * (3 / 4) - 0.15 * abs(4 - 3)
    assert math.isclose(got, want), (
        f"continue: (fraction of words in vocab) x (fraction of distinct words) - 0.15 |words - answer words| = "
        f"0.75 x 0.75 - 0.15 = {want:.4f}; got {got}")
    assert nb.gold_reward(cont, "  ", VOCAB) == -1.0, "an empty answer scores -1"
    assert math.isclose(nb.gold_reward(cont, "River Castle Winter", VOCAB), 1.0), "case-insensitive vocabulary"
    assert math.isclose(nb.gold_reward(cont, "River river castle", VOCAB), 2 / 3), (
        "distinct words are counted case-insensitively: 'River' and 'river' are one word (2 of 3 distinct)")


# ---------------------------------------------------------------- stage 3: respond and SFT


def test_stage3_respond(nb):
    m, tok = _model(), _tok()
    m.train()
    out = nb.respond(m, tok, "Write crown in capitals.", n=3, max_new_tokens=10, generator=torch.Generator().manual_seed(0))
    assert len(out) == 3 and all(isinstance(t, list) and isinstance(s, str) for t, s in out)
    eot = tok.encode(EOT)[0]
    for t, s in out:
        assert len(t) <= 10 and (eot not in t or t[-1] == eot), "tokens stop at (and include) the first EOT"
        assert s == tok.decode([x for x in t if x != eot]).strip(), "the text is the decoded tokens without EOT"
    g1 = nb.respond(m, tok, "Continue: crown", temperature=0.0)
    assert g1 == nb.respond(m, tok, "Continue: crown", temperature=0.0), "temperature 0 is deterministic"
    head = tok.encode("User: Continue: crown\nAssistant:")
    want = m.generate(torch.tensor([head]), 32, temperature=0.0)[0, len(head):].tolist()
    want = want[:want.index(eot) + 1] if eot in want else want
    assert g1[0][0] == want, "respond continues 'User: {prompt}\\nAssistant:' (greedy here, 32 tokens by default)"


def test_stage3_sft(nb):
    tok = _tok()
    tasks = [{"kind": "capitals", "prompt": f"Write {w} in capitals.", "answer": w.upper()} for w in ("crown", "sword")]
    m = _model(1)
    before = copy.deepcopy(m)
    losses = bounded(nb.sft, m, tok, tasks, steps=300, lr=3e-3, batch_size=8, what="sft", budget=100_000_000)
    assert len(losses) == 300 and all(isinstance(x, float) for x in losses), "one float loss per step"
    assert np.mean(losses[-10:]) < 0.3 * np.mean(losses[:10]), f"SFT should learn two fixed answers: {losses[0]:.2f} -> {losses[-1]:.2f}"
    assert not m.training, "sft leaves the model in eval mode"
    toks, got = nb.respond(m, tok, "Write crown in capitals.", temperature=0.0)[0]
    assert got == "CROWN", f"after SFT the model answers a training prompt exactly; got {got!r}"
    eot = tok.encode(EOT)[0]
    assert toks[-1] == eot and toks.count(eot) == 1, (
        "the trained model emits the end-of-text token after its answer: cut the tokens there (keeping the EOT)")
    x, y = collate_sft([build_sft_example(tok, t["prompt"], t["answer"]) for t in tasks])
    with torch.no_grad():
        assert sft_loss(m(x)[0], y).item() < 0.2, "the response tokens are what's learned"
    one = [tasks[0]]
    probe = _model(6)
    first = nb.sft(probe, tok, one, steps=1, lr=0.0, batch_size=4)[0]
    x1, y1 = collate_sft([build_sft_example(tok, one[0]["prompt"], one[0]["answer"], max_len=probe.block_size)] * 4)
    with torch.no_grad():
        want = sft_loss(probe(x1)[0], y1).item()
    assert math.isclose(first, want, rel_tol=1e-5), (
        f"the SFT loss is the response-only loss of build_sft_example / collate_sft batches: {want:.5f}; got {first:.5f}. "
        "training on the prompt tokens too gives a different number")
    frozen = _model(1)
    snap = copy.deepcopy(frozen.state_dict())
    nb.sft(frozen, tok, tasks, steps=3, lr=0.0)
    assert all(torch.equal(snap[k], v) for k, v in frozen.state_dict().items()), "lr is used (lr=0 changes nothing)"
    p1 = nb.sft(_model(2), tok, tasks, steps=5, seed=0)
    p2 = nb.sft(_model(2), tok, tasks, steps=5, seed=0)
    assert p1 == p2, "the same seed gives the same batches and losses"


# ---------------------------------------------------------------- stage 4: preferences and DPO


def test_stage4_pairs(nb):
    tok = _tok()
    m = _model(3)
    tasks = [{"kind": "continue", "prompt": "Continue: crown sword river", "answer": "castle winter falcon"},
             {"kind": "capitals", "prompt": "Write crown in capitals.", "answer": "CROWN"}]
    pairs = bounded(nb.preference_pairs, m, tok, tasks, VOCAB, k=6, seed=0, what="preference_pairs", budget=50_000_000)
    eot = tok.encode(EOT)[0]
    assert len(pairs) <= 3 * 1, "k=6 samples make 3 pairs per task; an untrained model ties on the skill task (0 vs 0)"
    for p in pairs:
        assert set(p) >= {"prompt", "chosen", "rejected"}
        task = next(t for t in tasks if t["prompt"] == p["prompt"])
        dec = lambda ids: tok.decode([i for i in ids if i != eot]).strip()
        assert nb.gold_reward(task, dec(p["chosen"]), VOCAB) > nb.gold_reward(task, dec(p["rejected"]), VOCAB), (
            "the chosen response has the higher gold reward (ties are dropped)")
    two = nb.preference_pairs(m, tok, tasks[:1] * 4, VOCAB, k=2, seed=1)
    many = nb.preference_pairs(m, tok, tasks[:1] * 4, VOCAB, k=16, seed=1)
    assert len(two) <= 4 and len(many) > 4, (
        f"k is used: k=2 gives at most one pair per task (4 here), k=16 up to 8 per task; got {len(two)} and {len(many)}")
    assert nb.preference_pairs(m, tok, tasks, VOCAB, k=6, seed=0) == pairs, "deterministic for a seed"


def test_stage4_dpo(nb):
    tok = _tok()
    pairs = [{"prompt": "Continue: crown", "chosen": tok.encode(" river castle") + tok.encode(EOT),
              "rejected": tok.encode(" zzq qqz") + tok.encode(EOT)},
             {"prompt": "Continue: sword", "chosen": tok.encode(" winter falcon") + tok.encode(EOT),
              "rejected": tok.encode(" xxv vvx") + tok.encode(EOT)}]
    ref = _model(4)
    ref.train()
    snap = copy.deepcopy(ref.state_dict())
    pol = copy.deepcopy(ref)
    hist = bounded(nb.dpo_train, pol, ref, tok, pairs, steps=25, beta=0.5, lr=3e-3, batch_size=4, what="dpo_train",
                   budget=50_000_000)
    assert len(hist) == 25 and all("loss" in h and "margin" in h for h in hist), "one dpo_step metrics dict per step"
    assert abs(hist[0]["loss"] - math.log(2)) < 1e-4, "the policy starts equal to the reference: first loss log 2"
    assert hist[-1]["loss"] < 0.3 and hist[-1]["margin"] > 0, f"DPO learns the preference; last step {hist[-1]}"
    assert all(torch.equal(snap[k], v) for k, v in ref.state_dict().items()), "the reference never changes"
    assert not any(p.requires_grad for p in ref.parameters()), "freeze the reference's parameters"
    assert not pol.training, "dpo_train leaves the policy in eval mode"
    tiny = copy.deepcopy(_model(4))
    h2 = nb.dpo_train(tiny, _model(4), tok, pairs, steps=10, beta=1e-6, lr=3e-3, batch_size=4)
    assert abs(h2[-1]["loss"] - math.log(2)) < 1e-3, (
        "with beta = 1e-6 the implicit rewards stay ~0 and the loss stays at log 2: is beta passed to dpo_step?")


# ---------------------------------------------------------------- stage 5: evaluation


def test_stage5_evaluate(nb):
    tok = _tok()
    m = _model(5)
    text = _text(1)
    tasks = nb.make_tasks(text, 12, split="test", seed=0)
    held = np.asarray(tok.encode(text[:3000]), dtype=np.int64)
    res = bounded(nb.evaluate_assistant, m, tok, tasks, VOCAB, reference=copy.deepcopy(m), held_out=held, seed=0,
                  eval_batches=2, what="evaluate_assistant", budget=50_000_000)
    assert set(res) == {"skill_acc", "gold", "win_rate", "lm_loss"}, f"keys: {sorted(res)}"
    for k in ("skill_acc", "gold", "win_rate"):
        v, lo, hi = res[k]
        assert lo <= v <= hi, f"{k}: (value, lo, hi) with lo <= value <= hi; got {res[k]}"
    assert res["win_rate"][0] == 0.5, (
        "a model against an identical copy, sampled with the same seed, ties every time: win rate exactly 0.5 (ties "
        "count 1/2)")
    skills = [t for t in tasks if t["kind"] != "continue"]
    want = np.mean([nb.respond(m, tok, t["prompt"], temperature=0.0)[0][1] == t["answer"] for t in skills])
    assert math.isclose(res["skill_acc"][0], want), "skill accuracy is greedy exact match on the skill tasks"
    conts = [t for t in tasks if t["kind"] == "continue"]
    g = torch.Generator().manual_seed(0)
    want_gold = np.mean([nb.gold_reward(t, nb.respond(m, tok, t["prompt"], generator=g)[0][1], VOCAB) for t in conts])
    assert math.isclose(res["gold"][0], want_gold, rel_tol=1e-6), (
        "gold: one sampled answer per continue task, generator seeded with `seed`, in task order")
    assert res["lm_loss"][0] > 0 and math.isnan(res["lm_loss"][1])
    bare = nb.evaluate_assistant(m, tok, tasks, VOCAB, seed=0)
    assert set(bare) == {"skill_acc", "gold"}, "win_rate only with a reference, lm_loss only with held_out data"


# ---------------------------------------------------------------- stage 6: model card


def test_stage6_model_card(nb):
    results = {"base": {"skill_acc": (0.1, 0.05, 0.15), "lm_loss": (2.5, float("nan"), float("nan"))},
               "dpo": {"skill_acc": (0.62, 0.55, 0.69), "win_rate": (0.58, 0.51, 0.65)}}
    config = {"model": {"d_model": 192, "n_layers": 6, "n_heads": 6, "n_kv_heads": 2}, "block_size": 256, "beta": 0.1}
    card = nb.model_card("tiny-bard", config, results, notes="three seeds")
    assert isinstance(card, str) and card.lstrip().startswith("#") and "tiny-bard" in card.splitlines()[0]
    for section in ("model details", "intended use", "training data", "training procedure", "evaluation",
                    "limitations", "ethical considerations"):
        assert re.search(rf"^## {section}", card, flags=re.M | re.I), f"the card needs a '## {section}' section"
    assert "0.620 [0.550, 0.690]" in card and "0.580 [0.510, 0.650]" in card, (
        "every result appears in the evaluation table with its interval, 3 decimals: '0.620 [0.550, 0.690]'")
    assert "2.500" in card and "2.500 [nan" not in card, "a result without an interval is shown as the value alone"
    assert "synthetic" in card.lower(), "say plainly which data are synthetic"
    assert "192" in card and "three seeds" in card, "the configuration and the notes are included"
    proc = card.split("## training procedure")[1].split("##")[0]
    assert "{" not in proc and re.search(r"^- ", proc, flags=re.M), (
        "the training procedure is a readable list ('- block size: 256'), not a raw dict or JSON dump")
    full = {"base": {"skill_acc": (0.0, 0.0, 0.0), "lm_loss": (1.791, float("nan"), float("nan"))},
            "sft": {"skill_acc": (0.852, 0.80, 0.90), "lm_loss": (2.760, float("nan"), float("nan"))},
            "dpo": {"skill_acc": (0.793, 0.74, 0.85), "lm_loss": (3.207, float("nan"), float("nan"))}}
    lim = nb.model_card("m", config, full).split("## limitations")[1].split("## ethical")[0]
    for v in ("0.852", "0.793", "1.791", "2.760", "3.207"):
        assert v in lim, (
            f"the limitations state every stage-to-stage drop in skill accuracy and rise in held-out loss with the "
            f"numbers ({v} is missing)")
    assert "0.000" not in lim, "a rise in skill accuracy (base -> sft) isn't a cost"
    assert "doesn't cover" in lim or "does not cover" in lim, "say what the evaluation doesn't cover"
