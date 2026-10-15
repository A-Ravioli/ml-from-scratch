"""Checks for 18 · 04 building GPT.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. batches are checked
against their definition; the model for the causal-LM protocol the rest of the course relies on
(pre-shifted targets, -100 ignored, causality, tied weights, GPT-2 initialization, block_size, and a
duck-typed KV cache that must reproduce full-sequence logits); sampling against the softmax with
temperature, top-k and nucleus filtering; generation against a hand-written loop; training against a
Markov chain whose entropy rate is known exactly. training runs under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _warm_up():
    """torch imports parts of itself on first use; do that here, outside any line budget."""
    m = nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters())
    F.cross_entropy(m(torch.zeros(3, 2)), torch.zeros(3, dtype=torch.long)).backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    torch.multinomial(torch.ones(3) / 3, 1, generator=torch.Generator().manual_seed(0))


_warm_up()


def _close(a, b, atol=1e-5):
    return torch.allclose(a, b, atol=atol, rtol=1e-4)


def _gpt(nb, **kw):
    args = dict(block_size=16, n_layers=2, n_heads=2, d_model=16)
    args.update(kw)
    return nb.GPT(kw.pop("vocab_size", 11), **{k: v for k, v in args.items() if k != "vocab_size"})


# ---------------------------------------------------------------- stage 1: batches


def test_stage1_get_batch(nb):
    data = np.arange(100, 150)
    g = torch.Generator().manual_seed(0)
    x, y = nb.get_batch(data, 8, 12, generator=g)
    assert x.dtype == torch.int64 and tuple(x.shape) == (12, 8) and tuple(y.shape) == (12, 8), (
        f"x and y are int64 (batch_size, block_size) = (12, 8); got {x.dtype} {tuple(x.shape)}")
    assert torch.equal(x - x[:, :1], torch.arange(8).expand(12, 8)), "each row of x is a contiguous window"
    assert torch.equal(y, x + 1), "y is x shifted by one: y[b, t] is the token after x[b, t] (pre-shifted targets)"
    starts = set()
    for _ in range(300):
        xb, _ = nb.get_batch(torch.arange(100, 150), 8, 12, generator=g)
        starts |= set((xb[:, 0] - 100).tolist())
    assert min(starts) == 0 and max(starts) == 50 - 8 - 1, (
        f"starts are uniform over 0 .. len - block_size - 1 = 41 (so y stays inside the data); saw {min(starts)}.."
        f"{max(starts)}")
    a, _ = nb.get_batch(data, 8, 12, generator=torch.Generator().manual_seed(5))
    b, _ = nb.get_batch(data, 8, 12, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b), "the same generator seed gives the same batch"


# ---------------------------------------------------------------- stage 2: the model


def test_stage2_structure_and_init(nb):
    torch.manual_seed(0)
    m = nb.GPT(50, block_size=32, n_layers=3, n_heads=4, d_model=64)
    for name in ("tok_emb", "pos_emb", "blocks", "ln_f", "lm_head"):
        assert hasattr(m, name), f"GPT needs a submodule named {name!r}"
    assert isinstance(m.blocks, nn.ModuleList) and len(m.blocks) == 3 and m.n_layers == 3, (
        "blocks is an nn.ModuleList of n_layers TransformerBlocks; n_layers is an attribute")
    assert all(b.norm_first for b in m.blocks), "GPT uses pre-LN blocks"
    assert m.lm_head.weight is m.tok_emb.weight, "tie_weights=True: lm_head.weight IS tok_emb.weight (one tensor)"
    n = sum(p.numel() for p in m.parameters())
    block = sum(p.numel() for p in m.blocks[0].parameters())
    want = 50 * 64 + 32 * 64 + 3 * block + 2 * 64
    assert n == want, (
        f"with tied weights the parameters are the token and position embeddings, the blocks and ln_f: {want:,}; "
        f"got {n:,} (an untied or biased head adds {50 * 64:,})")
    n65 = sum(p.numel() for p in nb.GPT(65).parameters())
    assert n65 == 834_432, f"worked example: GPT(65) with the default sizes has 834,432 parameters; got {n65:,}"
    untied = nb.GPT(50, block_size=32, n_layers=3, n_heads=4, d_model=64, tie_weights=False)
    assert untied.lm_head.weight is not untied.tok_emb.weight, "tie_weights=False keeps a separate head"
    assert nb.GPT(50, pos_encoding="sinusoidal").pos_emb is None, "pos_encoding='sinusoidal': pos_emb is None"
    std = m.tok_emb.weight.std().item()
    assert 0.017 < std < 0.023, f"GPT-2 initialization: embeddings and linear weights ~ N(0, 0.02); tok_emb std {std:.4f}"
    proj = torch.cat([b.attn.out_proj.weight.flatten() for b in m.blocks] + [b.mlp.fc2.weight.flatten() for b in m.blocks])
    want_std = 0.02 / math.sqrt(2 * 3)
    assert abs(proj.std().item() - want_std) < 0.15 * want_std, (
        f"the residual output layers (attn.out_proj and mlp.fc2) use std 0.02 / sqrt(2 n_layers) = {want_std:.4f}; "
        f"got {proj.std().item():.4f}")


def test_stage2_forward_loss_and_protocol(nb):
    torch.manual_seed(1)
    m = _gpt(nb, vocab_size=11).eval()
    idx = torch.randint(0, 11, (3, 10))
    logits, loss = m(idx)
    assert tuple(logits.shape) == (3, 10, 11) and loss is None, "forward(idx) -> (logits (B, T, V), None)"
    y = torch.randint(0, 11, (3, 10))
    _, loss = m(idx, y)
    assert math.isclose(loss.item(), F.cross_entropy(logits.reshape(-1, 11), y.reshape(-1)).item(), rel_tol=1e-5), (
        "loss is the mean cross-entropy of logits against the (already shifted) targets: don't shift them again")
    assert abs(loss.item() - math.log(11)) < 0.15, (
        f"with the small GPT-2 initialization the initial loss is close to ln V = {math.log(11):.3f}; got {loss.item():.3f}")
    y2 = y.clone()
    y2[:, :4] = -100
    _, loss2 = m(idx, y2)
    want2 = F.cross_entropy(logits[:, 4:].reshape(-1, 11), y[:, 4:].reshape(-1))
    assert _close(loss2, want2), "targets equal to -100 are ignored (prompt positions, padding)"
    idx2 = idx.clone()
    idx2[:, 7:] = (idx2[:, 7:] + 1) % 11
    assert _close(m(idx2)[0][:, :7], logits[:, :7]), "GPT is causal: tokens 7.. can't change the logits at 0..6"
    h = m.hidden_states(idx)
    assert tuple(h.shape) == (3, 10, 16) and _close(m.lm_head(h), logits), (
        "hidden_states(idx) is what the head reads (after ln_f): lm_head(hidden_states(idx)) == logits")
    with pytest.raises(ValueError):
        m(torch.randint(0, 11, (1, 17)))


def test_stage2_embedding_dropout(nb):
    torch.manual_seed(7)
    m = nb.GPT(11, block_size=16, n_layers=1, n_heads=2, d_model=16, dropout=0.5).train()
    for mod in m.blocks.modules():  # silence every dropout inside the blocks
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0
    idx = torch.randint(0, 11, (2, 8))
    assert not _close(m(idx)[0], m(idx)[0]), (
        "GPT applies dropout (with the model's dropout rate) to the embeddings (token + position) before the blocks")
    m.eval()
    assert _close(m(idx)[0], m(idx)[0]), "and not in eval mode"


class _Layer:
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        self.k = k if self.k is None else torch.cat([self.k, k], -2)
        self.v = v if self.v is None else torch.cat([self.v, v], -2)
        return self.k, self.v


class _Cache:
    """duck-typed KVCache: cache[i] is layer i's store; seq_len counts tokens cached so far."""

    def __init__(self, n):
        self.layers = [_Layer() for _ in range(n)]

    def __getitem__(self, i):
        return self.layers[i]

    @property
    def seq_len(self):
        return 0 if self.layers[0].k is None else self.layers[0].k.shape[-2]


def test_stage2_cache_reproduces_full_logits(nb):
    for pos in ("learned", "sinusoidal"):
        torch.manual_seed(2)
        m = _gpt(nb, pos_encoding=pos).eval()
        idx = torch.randint(0, 11, (2, 9))
        full, _ = m(idx)
        cache = _Cache(m.n_layers)
        parts = [m(idx[:, :4], cache=cache)[0]]
        for t in range(4, 9):
            parts.append(m(idx[:, t:t + 1], cache=cache)[0])
        assert cache.seq_len == 9, "block i must pass its own cache[i] to its attention"
        got = torch.cat(parts, 1)
        assert _close(got, full, atol=1e-4), (
            f"pos_encoding={pos!r}: a prompt of 4 tokens, then one token at a time through the cache, must give the same "
            f"logits as one pass over all 9. the new tokens' positions start at cache.seq_len (read it BEFORE running "
            f"the blocks, which update the cache); positions restarting at 0 is the classic bug")
        with pytest.raises(ValueError):
            c = _Cache(m.n_layers)
            m(torch.randint(0, 11, (1, 10)), cache=c)
            m(torch.randint(0, 11, (1, 7)), cache=c)  # 17 > block_size


# ---------------------------------------------------------------- stage 3: sampling


def _freq(nb, logits, n=6000, **kw):
    out = nb.sample_next_token(logits.expand(n, -1).clone(), generator=torch.Generator().manual_seed(0), **kw)
    assert out.dtype == torch.int64 and tuple(out.shape) == (n,), f"(B,) int64; got {out.dtype} {tuple(out.shape)}"
    return torch.bincount(out, minlength=logits.shape[-1]).double() / n


def test_stage3_temperature_and_argmax(nb):
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    for T in (0.5, 1.0, 2.0):
        f, p = _freq(nb, logits, temperature=T), torch.softmax(logits[0].double() / T, 0)
        assert torch.allclose(f, p, atol=0.025), f"temperature {T}: frequencies {f.numpy().round(3)} vs softmax(logits/T) {p.numpy().round(3)}"
    rows = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 0.0]])
    assert torch.equal(nb.sample_next_token(rows, temperature=0.0), torch.tensor([1, 0])), "temperature 0 is argmax, per row"
    a = nb.sample_next_token(torch.zeros(20, 9), generator=torch.Generator().manual_seed(3))
    b = nb.sample_next_token(torch.zeros(20, 9), generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b), "the generator makes sampling reproducible"


def test_stage3_top_k_and_top_p(nb):
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = p.log()[None]
    f = _freq(nb, logits, top_k=2)
    assert f[2:].sum() == 0 and torch.allclose(f[:2], torch.tensor([0.625, 0.375]).double(), atol=0.025), (
        f"top_k=2 keeps the 2 largest logits and renormalizes: (0.625, 0.375, 0, 0); got {f.numpy().round(3)}")
    f = _freq(nb, logits, top_p=0.75)
    assert f[2:].sum() == 0 and torch.allclose(f[:2], torch.tensor([0.625, 0.375]).double(), atol=0.025), (
        f"top_p=0.75 keeps the smallest set of most likely tokens whose mass reaches 0.75: {{0, 1}} (0.5 isn't enough, "
        f"0.8 is); got {f.numpy().round(3)}")
    f = _freq(nb, logits, top_p=0.9)
    assert f[3] == 0 and f[2] > 0.1, f"top_p=0.9 keeps tokens 0, 1, 2 (mass 0.95); got {f.numpy().round(3)}"
    f = _freq(nb, logits, top_p=0.3)
    assert f[0] == 1.0, f"top_p=0.3: the top token alone already has mass 0.5 >= 0.3, so it is always chosen; got {f.numpy()}"
    f = _freq(nb, logits, top_k=3, top_p=0.6, temperature=1.0)
    assert f[2:].sum() == 0, "top_k first, then top_p on the renormalized survivors: (0.53, 0.32, 0.16) -> keep 2"
    perm = torch.tensor([2, 0, 3, 1])
    f = _freq(nb, logits[:, perm], top_p=0.75)
    assert f[0] == 0 and f[2] == 0 and f[1] > 0.5, "top-p must work whatever order the vocabulary is in (sort, then undo)"
    two = torch.stack([p.log(), p.flip(0).log()])
    out = nb.sample_next_token(two.repeat(500, 1), top_k=1, generator=torch.Generator().manual_seed(0))
    assert (out[0::2] == 0).all() and (out[1::2] == 3).all(), "filtering is per row"


def test_stage3_boundaries(nb):
    uniform = torch.zeros(1, 4)  # probabilities exactly 0.25 each
    out = nb.sample_next_token(uniform.expand(4000, 4).clone(), top_p=0.5, generator=torch.Generator().manual_seed(0))
    assert len(out.unique()) == 2, (
        f"top_p=0.5 over four tokens of probability exactly 0.25: the smallest set with mass >= 0.5 has 2 tokens "
        f"(mass reaching p counts: compare the mass BEFORE each token with >=); sampled {len(out.unique())} distinct")
    with pytest.raises(ValueError):
        nb.sample_next_token(torch.zeros(1, 4), top_p=0.0)
    ties = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = nb.sample_next_token(ties.expand(3000, 4).clone(), top_k=2, generator=torch.Generator().manual_seed(1))
    assert set(out.tolist()) == {0, 1, 2}, (
        "top_k keeps every logit >= the k-th largest, so ties at the cut are all kept (here 3 tokens with top_k=2)")


# ---------------------------------------------------------------- stage 4: generation


def test_stage4_generate(nb):
    torch.manual_seed(4)
    m = _gpt(nb, block_size=8).train()
    idx = torch.randint(0, 11, (2, 3))
    out = m.generate(idx, 12, temperature=0.0)
    assert tuple(out.shape) == (2, 15) and torch.equal(out[:, :3], idx), (
        "generate returns the prompt followed by max_new_tokens new tokens: (B, T + max_new_tokens)")
    assert m.training, "generate restores the model's training mode"
    seq = idx.clone()
    m.eval()
    with torch.no_grad():
        for _ in range(12):
            nxt = m(seq[:, -8:])[0][:, -1].argmax(-1)
            seq = torch.cat([seq, nxt[:, None]], 1)
    assert torch.equal(out, seq), (
        "greedy generation must follow the model's argmax given the last block_size tokens: crop the context to "
        "block_size (here 8) so generation can continue past it")
    a = m.generate(idx, 10, generator=torch.Generator().manual_seed(1), top_k=5)
    b = m.generate(idx, 10, generator=torch.Generator().manual_seed(1), top_k=5)
    assert torch.equal(a, b), "sampling with the same generator is reproducible: pass it to sample_next_token"
    d = _gpt(nb, dropout=0.5).train()
    assert torch.equal(d.generate(idx, 6, temperature=0.0), d.generate(idx, 6, temperature=0.0)), (
        "generate runs in eval mode (no dropout)")


# ---------------------------------------------------------------- stage 5: training


def _chain_data(n, seed):
    rng = np.random.default_rng(100)
    P = np.full((6, 6), 0.02)
    for i in range(6):
        P[i, rng.choice(6, size=2, replace=False)] += [0.8, 0.15]
    P /= P.sum(1, keepdims=True)
    r = np.random.default_rng(seed)
    x = [0]
    u = r.random(n)
    for t in range(n - 1):
        x.append(int(np.searchsorted(P[x[-1]].cumsum(), u[t])))
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(abs(w - 1))])
    pi /= pi.sum()
    return np.minimum(np.array(x), 5), float(-(pi[:, None] * P * np.log(P)).sum())


def test_stage5_estimate_loss(nb):
    torch.manual_seed(5)
    m = _gpt(nb, vocab_size=6, dropout=0.3).train()
    data, _ = _chain_data(500, 0)
    a = nb.estimate_loss(m, data, 8, batch_size=4, n_batches=5)
    b = nb.estimate_loss(m, data, 8, batch_size=4, n_batches=5)
    assert isinstance(a, float) and a == b, "estimate_loss returns a float, the same every call (eval mode, fixed batches)"
    assert m.training, "estimate_loss restores the model's mode"
    g = torch.Generator().manual_seed(0)
    m.eval()
    with torch.no_grad():
        want = np.mean([m(*nb.get_batch(data, 8, 4, generator=g))[1].item() for _ in range(5)])
    assert math.isclose(a, want, rel_tol=1e-5), (
        f"the mean loss over n_batches batches from get_batch with torch.Generator().manual_seed(0); expected "
        f"{want:.5f}, got {a:.5f}")


def test_stage5_train_lm(nb):
    data, H = _chain_data(6000, 1)
    val, _ = _chain_data(2000, 2)

    def run(seed, steps=120):
        torch.manual_seed(seed)
        m = nb.GPT(6, block_size=16, n_layers=1, n_heads=2, d_model=32)
        hist = bounded(nb.train_lm, m, data, 16, val_data=val, batch_size=16, max_steps=steps, lr=3e-3, warmup_steps=10,
                       eval_interval=50, eval_batches=2, seed=seed, budget=80_000_000, what="train_lm")
        return m, hist

    m, hist = run(0)
    for key in ("step", "train_loss", "val_loss"):
        assert key in hist and len(hist[key]) == 3, f"history[{key!r}] has one entry per evaluation: steps 50, 100, 120"
    assert hist["step"] == [50, 100, 120], f"evaluate every eval_interval steps and at the last step; got {hist['step']}"
    assert not m.training, "train_lm leaves the model in eval mode"
    finals = [hist["val_loss"][-1]] + [run(s)[1]["val_loss"][-1] for s in (1, 2)]
    assert np.median(finals) < H + 0.2, (
        f"the chain's entropy rate is {H:.3f} nats per token; after 120 steps the validation loss should be within 0.2 "
        f"of it (median of 3 seeds). got {np.round(finals, 3)}")
    _, a = run(3, steps=20)
    _, b = run(3, steps=20)
    assert np.allclose(a["train_loss"], b["train_loss"]), "the same seed (and model init) gives the same run"


def test_stage5_step_size_and_decay_groups(nb):
    data, _ = _chain_data(3000, 3)
    torch.manual_seed(6)
    m = nb.GPT(6, block_size=16, n_layers=1, n_heads=2, d_model=32)
    w0 = m.blocks[0].attn.q_proj.weight.detach().clone()
    g0 = m.blocks[0].ln1.weight.detach().clone()
    nb.train_lm(m, data, 16, batch_size=8, max_steps=3, lr=1e-2, warmup_steps=1, weight_decay=0.0, grad_clip=None,
                eval_interval=3, eval_batches=1)
    step = (m.blocks[0].attn.q_proj.weight.detach() - w0).abs().median().item()
    assert 0.3e-2 < step < 3e-2, (
        f"with lr=1e-2, a few AdamW steps move each weight by about the learning rate (Adam's first steps have size "
        f"~lr per parameter); the median change was {step:.2e}. check that AdamW gets lr and that the schedule warms up "
        f"TO lr (your CosineSchedule, stepped once per optimizer step)")
    torch.manual_seed(6)
    m2 = nb.GPT(6, block_size=16, n_layers=1, n_heads=2, d_model=32)
    with torch.no_grad():
        for p in m2.parameters():
            p.requires_grad_(False)
        m2.blocks[0].ln1.weight.requires_grad_(True)
        m2.blocks[0].attn.q_proj.weight.requires_grad_(True)
        m2.blocks[0].ln1.weight.fill_(1.0)
    ln_before = m2.blocks[0].ln1.weight.detach().clone()
    nb.train_lm(m2, data, 16, batch_size=8, max_steps=3, lr=1e-3, warmup_steps=1, weight_decay=900.0, grad_clip=None,
                eval_interval=3, eval_batches=1)
    q_after = m2.blocks[0].attn.q_proj.weight.detach()
    ln_after = m2.blocks[0].ln1.weight.detach()
    assert q_after.abs().mean() < 0.5 * w0.abs().mean(), (
        "weight decay applies to >= 2-D parameters: with lr=1e-3 and weight_decay=900 (decoupled: each step multiplies by 1 - lr * wd) the attention weights must shrink a lot")
    assert (ln_after - ln_before).abs().max() < 0.1, (
        "weight decay must NOT apply to 1-D parameters (layer-norm gains, biases): with weight_decay=900 the layer-norm "
        "gain moved far from 1. put parameters with p.dim() < 2 in a param group with weight_decay=0")


def test_stage5_schedule_optimizer_and_seed(nb):
    data, _ = _chain_data(3000, 4)
    seen = {}
    orig_adamw, orig_clip = torch.optim.AdamW, torch.nn.utils.clip_grad_norm_

    class Recorder(orig_adamw):
        def __init__(self, params, *a, **k):
            super().__init__(params, *a, **k)
            seen["betas"] = self.param_groups[0]["betas"]

    def clip(params, max_norm, *a, **k):
        seen.setdefault("clip", []).append(max_norm)
        return orig_clip(params, max_norm, *a, **k)

    torch.optim.AdamW, torch.nn.utils.clip_grad_norm_ = Recorder, clip  # watch what train_lm passes
    try:
        torch.manual_seed(8)
        m = nb.GPT(6, block_size=8, n_layers=1, n_heads=2, d_model=16)
        h = nb.train_lm(m, data, 8, batch_size=4, max_steps=20, lr=1e-2, warmup_steps=5, grad_clip=0.7,
                        eval_interval=1, eval_batches=1, seed=0)
        clips_first = list(seen.get("clip", []))
        seen.pop("clip", None)
        torch.manual_seed(8)
        nb.train_lm(nb.GPT(6, block_size=8, n_layers=1, n_heads=2, d_model=16), data, 8, batch_size=4, max_steps=3,
                    grad_clip=None, eval_interval=3, eval_batches=1)
        clips_none = list(seen.get("clip", []))
    finally:
        torch.optim.AdamW, torch.nn.utils.clip_grad_norm_ = orig_adamw, orig_clip
    assert "lr" in h and len(h["lr"]) == 20, "history['lr'] holds the learning rate used at each evaluated step"
    lr = np.asarray(h["lr"])
    assert lr[0] == 0.0 and math.isclose(lr[5], 1e-2, rel_tol=1e-6), (
        f"linear warmup from 0: step 1 uses lr 0 and step warmup + 1 uses the full lr (your CosineSchedule, stepped "
        f"once per optimizer step, recording the lr each step used); got {lr[:7]}")
    assert 1e-3 <= lr[-1] < 1.3e-3, f"the cosine decays to lr / 10 = 1e-3 at max_steps, not to 0; last step used {lr[-1]:.2e}"
    assert tuple(seen.get("betas", ())) == (0.9, 0.95), f"AdamW with betas (0.9, 0.95); got {seen.get('betas')}"
    assert len(clips_first) == 20 and all(c == 0.7 for c in clips_first), (
        f"clip the gradient norm at grad_clip with torch.nn.utils.clip_grad_norm_ (called through the module, e.g. "
        f"nn.utils.clip_grad_norm_), once per step; saw {clips_first[:3]}... ({len(clips_first)} calls)")
    assert not clips_none, "grad_clip=None means no clipping"
    runs = []
    for seed in (0, 1):
        torch.manual_seed(8)
        runs.append(nb.train_lm(nb.GPT(6, block_size=8, n_layers=1, n_heads=2, d_model=16), data, 8, batch_size=4,
                                max_steps=6, eval_interval=2, eval_batches=1, seed=seed)["train_loss"])
    assert not np.allclose(runs[0], runs[1]), "seed drives batch sampling: different seeds give different runs"
    torch.manual_seed(8)
    short = nb.train_lm(nb.GPT(6, block_size=8, n_layers=1, n_heads=2, d_model=16), data, 8, batch_size=4,
                        max_steps=10, warmup_steps=100, eval_interval=5, eval_batches=1)
    assert len(short["step"]) == 2, "warmup_steps longer than max_steps must still run (clamp the warmup)"
