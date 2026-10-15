"""Checks for capstone.py (a character-level LSTM language model).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. data are synthetic: a
Markov chain over 8 symbols whose entropy rate is known exactly, so a trained model can be judged
against the best possible loss. the model is checked for carrying its state exactly, batches against
their definition, training against the chain's entropy, its schedule and its best-checkpoint rule,
evaluation against a single full forward pass, the interval against block structure, and sampling
against the softmax with temperature. training runs under `mlfs.testing.bounded`.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _warm_up():
    """torch imports parts of itself on first use (millions of lines); do that here, outside any line budget."""
    m = nn.LSTM(2, 2, batch_first=True)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.zeros(1, 2, 2))[0].sum().backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    torch.multinomial(torch.ones(3) / 3, 1, generator=torch.Generator().manual_seed(0))


_warm_up()


def _chain(seed=0):
    """transition matrix of an order-1 chain over 8 symbols: each symbol has 2 likely successors."""
    rng = np.random.default_rng(100)  # the chain is fixed; only the samples depend on seed
    P = np.full((8, 8), 0.01)
    for i in range(8):
        a, b = rng.choice(8, size=2, replace=False)
        P[i, a] += 0.7
        P[i, b] += 0.25
    return P / P.sum(1, keepdims=True)


def _sample_chain(n, seed):
    P = _chain()
    rng = np.random.default_rng(seed)
    x = [0]
    for _ in range(n - 1):
        x.append(int(rng.choice(8, p=P[x[-1]])))
    return np.array(x, dtype=np.int64)


def _entropy_rate():
    P = _chain()
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi /= pi.sum()
    return float(-(pi[:, None] * P * np.log(P)).sum())  # nats per symbol


class _FixedLogits(nn.Module):
    """a 'model' that always predicts the same logits; its state counts the tokens it has read."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.as_tensor(logits, dtype=torch.float32)

    def forward(self, idx, state=None):
        count = (0 if state is None else state) + idx.shape[1]
        return self.logits.expand(idx.shape[0], idx.shape[1], -1).clone(), count


# ---------------------------------------------------------------- stage 1: the model


def test_stage1_structure_and_shapes(nb):
    m = nb.CharLSTM(11, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.3)
    assert isinstance(m.embed, nn.Embedding) and isinstance(m.lstm, nn.LSTM) and isinstance(m.head, nn.Linear)
    assert m.lstm.batch_first and m.lstm.num_layers == 2 and m.lstm.hidden_size == 16, "a batch-first 2-layer nn.LSTM"
    assert m.lstm.dropout == 0.3 and isinstance(m.drop, nn.Dropout) and m.drop.p == 0.3, (
        "dropout between the LSTM layers (nn.LSTM's dropout argument) and on the LSTM's output (self.drop)")
    logits, (h, c) = m(torch.randint(0, 11, (3, 7)))
    assert tuple(logits.shape) == (3, 7, 11) and tuple(h.shape) == (2, 3, 16) and tuple(c.shape) == (2, 3, 16), (
        f"logits (B, T, V) and the LSTM's state (h, c), each (num_layers, B, H); got {tuple(logits.shape)}")


def test_stage1_state_is_carried_exactly(nb):
    torch.manual_seed(0)
    m = nb.CharLSTM(9, embed_dim=8, hidden_dim=12, num_layers=2, dropout=0.5).eval()
    x = torch.randint(0, 9, (2, 10))
    full, _ = m(x)
    a, state = m(x[:, :4])
    b, _ = m(x[:, 4:], state)
    assert torch.allclose(full, torch.cat([a, b], 1), atol=1e-6), (
        "running the sequence in two pieces, passing the state from the first into the second, must give exactly the "
        "logits of one run over the whole sequence (and in eval mode, dropout must be off)")
    m.train()
    assert not torch.allclose(m(x)[0], m(x)[0]), "in training mode dropout is active"
    one = nb.CharLSTM(9, embed_dim=8, hidden_dim=12, num_layers=1, dropout=0.5).train()
    assert not torch.allclose(one(x)[0], one(x)[0]), (
        "with one layer nn.LSTM applies no dropout of its own: self.drop on the LSTM's output must still be active in "
        "training mode")


# ---------------------------------------------------------------- stage 2: batches


def test_stage2_windows(nb):
    ids = np.arange(100, 160)
    g = torch.Generator().manual_seed(0)
    x, y = nb.get_batch(ids, 16, 8, generator=g)
    assert x.dtype == torch.int64 and tuple(x.shape) == (16, 8) and tuple(y.shape) == (16, 8), (
        f"x and y are int64 (B, L) = (16, 8); got {x.dtype} {tuple(x.shape)}")
    starts = x[:, 0] - 100
    assert torch.equal(x - 100, starts[:, None] + torch.arange(8)), "each row of x is a contiguous window of ids"
    assert torch.equal(y, x + 1), "y is x shifted by one position: the next character at every position"
    seen = set()
    for _ in range(200):
        xb, _ = nb.get_batch(ids, 16, 8, generator=g)
        seen |= set((xb[:, 0] - 100).tolist())
    assert min(seen) == 0 and max(seen) == 60 - 8 - 1, (
        f"starts are uniform over every valid position, 0 .. len - L - 1 = 51 (so y stays inside the text); saw "
        f"{min(seen)} .. {max(seen)}")
    x2, _ = nb.get_batch(ids, 16, 8, generator=torch.Generator().manual_seed(0))
    assert torch.equal(x, x2), "the generator makes batches reproducible"


# ---------------------------------------------------------------- stage 3: training


def _fit(nb, steps=150, seed=0, **kw):
    torch.manual_seed(seed)
    m = nb.CharLSTM(8, embed_dim=16, hidden_dim=32, num_layers=1)
    hist = bounded(nb.train_lm, m, _sample_chain(20_000, seed), steps=steps, batch_size=16, seq_len=24, lr=1e-2,
                   warmup=10, val_ids=_sample_chain(3_000, seed + 50), eval_every=50, seed=seed, budget=60_000_000,
                   what="train_lm", **kw)
    return m, hist


def test_stage3_learns_the_chain(nb):
    H = _entropy_rate()
    finals = []
    for s in range(5):
        _, hist = _fit(nb, steps=100, seed=s)
        finals.append(min(hist["val_loss"]))
    assert np.median(finals) < H + 0.08, (
        f"the chain's entropy rate is {H:.3f} nats per symbol, the best any model can do; after 100 steps your model's "
        f"best validation loss should be within 0.08 of it. got {np.round(finals, 3)} over 5 seeds")


def test_stage3_history_schedule_and_determinism(nb):
    m, hist = _fit(nb, steps=100)
    for key in ("step", "train_loss", "val_step", "val_loss", "lr"):
        assert key in hist, f"history has the key {key!r}"
    assert len(hist["train_loss"]) == 100 and hist["val_step"] == [50, 100], (
        f"one training loss per step, and a validation loss every eval_every = 50 steps; got val_step {hist['val_step']}")
    lr = np.asarray(hist["lr"])
    assert lr[0] == 0.0 and math.isclose(lr[10], 1e-2, rel_tol=1e-6), (
        "warmup: the learning rate of step 1 is 0 and reaches its peak after `warmup` steps (your CosineSchedule, "
        "stepped once per optimizer step; record the learning rate the step used)")
    assert math.isclose(lr[-1], 1e-3, rel_tol=0.05), f"the cosine ends near lr / 10; got {lr[-1]:.2e}"
    _, hist2 = _fit(nb, steps=100)
    assert np.allclose(hist["train_loss"], hist2["train_loss"]), "the same seed must give the same run"
    assert not m.training, "train_lm leaves the model in eval mode"


def test_stage3_keeps_the_best_weights(nb):
    def wreck(step, history):
        if step == 100:
            with torch.no_grad():
                for p in model.parameters():
                    p.zero_()
        return False

    torch.manual_seed(0)
    model = nb.CharLSTM(8, embed_dim=16, hidden_dim=32, num_layers=1)
    bounded(nb.train_lm, model, _sample_chain(5_000, 1), steps=100, batch_size=16, seq_len=16, lr=1e-2, warmup=5,
            val_ids=_sample_chain(2_000, 2), eval_every=50, callback=wreck, budget=40_000_000, what="train_lm")
    assert any(p.abs().sum() > 0 for p in model.parameters()), (
        "after training, the model must hold the weights with the best validation loss, not the last ones (here the "
        "last ones were wiped out by a callback after the final step): copy the state_dict when validation improves "
        "and load it back at the end")


# ---------------------------------------------------------------- stage 4: evaluation


def test_stage4_char_nll_carries_state(nb):
    torch.manual_seed(3)
    m = nb.CharLSTM(8, embed_dim=8, hidden_dim=16, num_layers=2, dropout=0.4).train()
    ids = _sample_chain(301, 5)
    got = nb.char_nll(m, ids, seq_len=7)
    assert isinstance(got, np.ndarray) and got.shape == (300,), f"one loss per predicted character: (300,); got {got.shape}"
    m.eval()
    with torch.no_grad():
        logits, _ = m(torch.as_tensor(ids[:-1])[None])
        want = F.cross_entropy(logits[0], torch.as_tensor(ids[1:]), reduction="none").double().numpy()
    assert np.allclose(got, want, atol=1e-5), (
        "char_nll with seq_len=7 must equal one forward pass over the whole text: carry the LSTM state from chunk to "
        "chunk (and evaluate in eval mode, without dropout)")
    m.train()
    nb.char_nll(m, ids, seq_len=50)
    assert m.training, "char_nll restores the model's mode"


def test_stage4_block_interval(nb):
    rng = np.random.default_rng(0)
    iid = rng.exponential(size=40_000)
    bpc, lo, hi = nb.bpc_ci(iid, block_size=1000, rng=0)
    assert math.isclose(bpc, iid.mean() / math.log(2)), "the estimate is the mean nll over every character, in bits"
    se = iid.std() / math.sqrt(len(iid)) / math.log(2)
    assert lo < bpc < hi and 0.7 < (hi - lo) / (2 * 1.96 * se) < 1.3, (
        f"for independent losses the 95% interval should be close to +-1.96 standard errors ({1.96 * se:.4f}); "
        f"got [{lo:.4f}, {hi:.4f}]")
    drift = np.repeat(rng.normal(1.5, 0.5, size=40), 1000) + rng.normal(0, 0.1, size=40_000)
    _, lo2, hi2 = nb.bpc_ci(drift, block_size=1000, rng=0)
    naive = 2 * 1.96 * drift.std() / math.sqrt(len(drift)) / math.log(2)
    assert hi2 - lo2 > 3 * naive, (
        f"when the loss drifts in stretches of 1,000 characters (like text whose topic changes), resampling contiguous "
        f"blocks must give a much wider interval than an i.i.d. one ({naive:.4f}); got width {hi2 - lo2:.4f}")


# ---------------------------------------------------------------- stage 5: sampling


def test_stage5_temperature(nb):
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    m = _FixedLogits(logits)
    for T in (0.5, 1.0, 2.0):
        out = bounded(nb.sample, m, [0, 1], 4000, temperature=T, generator=torch.Generator().manual_seed(1),
                      what="sample")
        assert len(out) == 4000 and all(isinstance(t, int) for t in out), "a list of n_tokens ints (prompt excluded)"
        freq = np.bincount(out, minlength=4) / 4000
        p = torch.softmax(logits / T, 0).numpy()
        assert np.allclose(freq, p, atol=0.03), (
            f"temperature {T}: tokens should follow softmax(logits / T) = {np.round(p, 3)}; got {np.round(freq, 3)}")
    assert nb.sample(m, [0], 20, temperature=0.0) == [0] * 20, "temperature 0 is greedy: always the argmax"


def test_stage5_reproducible_and_uses_the_model(nb):
    torch.manual_seed(4)
    m = nb.CharLSTM(8, embed_dim=8, hidden_dim=16, num_layers=1)
    a = nb.sample(m, [1, 2, 3], 30, generator=torch.Generator().manual_seed(7))
    b = nb.sample(m, [1, 2, 3], 30, generator=torch.Generator().manual_seed(7))
    assert a == b, "the same generator seed gives the same sample"
    greedy = nb.sample(m, [1, 2, 3], 5, temperature=0.0)
    seq = [1, 2, 3]
    m.eval()
    for _ in range(5):
        with torch.no_grad():
            seq.append(int(m(torch.tensor([seq]))[0][0, -1].argmax()))
    assert greedy == seq[3:], (
        f"greedy sampling must follow the model's argmax given the whole history (prompt included): expected "
        f"{seq[3:]}, got {greedy}. feed each new token with the carried state")
