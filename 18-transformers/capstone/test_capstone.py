"""Checks for capstone.py (nanoGPT: a GPT on tiny Shakespeare, a scaling curve and cached inference).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. data are synthetic. the
split is checked against its definition; the sliding-window evaluation against an explicit rebuild of
its windows and against a model whose loss doesn't depend on context; the interval against block
structure; the power-law fit against exact power laws; the speed measurement for which model calls it
makes (no timing thresholds).
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.transformer import GPT  # noqa: E402  (reference, from topic 04)


# ---------------------------------------------------------------- stage 1: the split


def test_stage1_split(nb):
    ids = list(range(1000, 2000))
    tr, va, te = nb.split_ids(ids)
    assert all(isinstance(a, np.ndarray) and a.dtype == np.int64 for a in (tr, va, te)), "three int64 ndarrays"
    assert (len(tr), len(va), len(te)) == (800, 100, 100), f"80/10/10 by position; got {(len(tr), len(va), len(te))}"
    assert tr[0] == 1000 and va[0] == 1800 and te[0] == 1900 and te[-1] == 1999, (
        "contiguous blocks in order: the first 80% trains, the next 10% validates, the last 10% tests (never shuffle a "
        "text before splitting: neighbouring windows overlap)")
    tr2, va2, te2 = nb.split_ids(np.arange(1115), train_frac=0.5, val_frac=0.2)
    assert (len(tr2), len(va2), len(te2)) == (557, 223, 335), "the fractions are arguments; boundaries use int()"


# ---------------------------------------------------------------- stage 2: sliding-window evaluation


class _Bigram(nn.Module):
    """a causal-LM-protocol model whose prediction depends only on the current token."""

    def __init__(self, V=7):
        super().__init__()
        self.table = nn.Embedding(V, V)

    def forward(self, idx, targets=None, *, cache=None):
        return self.table(idx), None


def test_stage2_every_token_once(nb):
    torch.manual_seed(0)
    m = _Bigram()
    ids = np.random.default_rng(0).integers(0, 7, 103)
    got = nb.token_nll(m, ids, block_size=16, stride=5, batch_size=3)
    assert isinstance(got, np.ndarray) and got.shape == (102,), f"one loss per predicted token: (102,); got {got.shape}"
    with torch.no_grad():
        want = F.cross_entropy(m.table(torch.as_tensor(ids[:-1])), torch.as_tensor(ids[1:]), reduction="none").double().numpy()
    assert np.allclose(got, want, atol=1e-6), (
        "for a model whose loss doesn't depend on context, token_nll must return exactly the per-token loss: every "
        "token scored exactly once, aligned with ids[1:]")


def test_stage2_windows_and_context(nb):
    torch.manual_seed(1)
    m = GPT(7, block_size=8, n_layers=1, n_heads=2, d_model=16, dropout=0.3).train()
    ids = np.random.default_rng(1).integers(0, 7, 31)  # 30 predictions
    got = nb.token_nll(m, ids, block_size=8, stride=3)
    m.eval()
    want = np.empty(30)
    done = 0
    starts = list(range(0, 30 - 8 + 1, 3))
    if starts[-1] + 8 < 30:
        starts.append(30 - 8)
    with torch.no_grad():
        for s in starts:
            x, y = torch.as_tensor(ids[s:s + 8])[None], torch.as_tensor(ids[s + 1:s + 9])
            row = F.cross_entropy(m(x)[0][0], y, reduction="none").double().numpy()
            want[done:s + 8] = row[done - s:]
            done = s + 8
    assert np.allclose(got, want, atol=1e-5), (
        "windows of block_size start at 0, stride, 2*stride, ... (plus one window ending exactly at the last token); "
        "the first window scores all its positions, each later window only the positions not yet scored. so every "
        "token after the first window has at least block_size - stride tokens of context. evaluate in eval mode")
    m.train()
    nb.token_nll(m, ids, block_size=8)
    assert m.training, "token_nll restores the model's mode"
    short = nb.token_nll(m.eval(), ids[:6], block_size=8)
    with torch.no_grad():
        full = F.cross_entropy(m(torch.as_tensor(ids[:5])[None])[0][0], torch.as_tensor(ids[1:6]), reduction="none")
    assert np.allclose(short, full.double().numpy(), atol=1e-5), "a text shorter than block_size is one window"


# ---------------------------------------------------------------- stage 3: the interval


def test_stage3_bpc_ci(nb):
    rng = np.random.default_rng(0)
    iid = rng.exponential(size=40_000)
    bpc, lo, hi = nb.bpc_ci(iid, block_size=1000, rng=0)
    assert math.isclose(bpc, iid.mean() / math.log(2)), "the estimate is the mean nll in bits"
    se = iid.std() / math.sqrt(len(iid)) / math.log(2)
    assert lo < bpc < hi and 0.6 < (hi - lo) / (2 * 1.96 * se) < 1.4, (
        f"for independent losses the interval is about +-1.96 standard errors ({1.96 * se:.4f}); got [{lo:.4f}, {hi:.4f}]")
    b2, lo2, hi2 = nb.bpc_ci(iid, chars_per_token=2.5, block_size=1000, rng=0)
    assert math.isclose(b2, bpc / 2.5) and math.isclose(hi2 - lo2, (hi - lo) / 2.5, rel_tol=0.05), (
        "chars_per_token converts bits per token into bits per character (for BPE tokens): divide by it")
    _, lo50, hi50 = nb.bpc_ci(iid, block_size=1000, alpha=0.5, rng=0)
    assert 0.25 < (hi50 - lo50) / (hi - lo) < 0.5, (
        f"alpha sets the level: a 50% interval is about 0.35x as wide as a 95% one; got {(hi50 - lo50) / (hi - lo):.2f}")
    drift = np.repeat(rng.normal(1.5, 0.5, size=40), 1000) + rng.normal(0, 0.1, size=40_000)
    _, lo3, hi3 = nb.bpc_ci(drift, block_size=1000, rng=0)
    naive = 2 * 1.96 * drift.std() / math.sqrt(len(drift)) / math.log(2)
    assert hi3 - lo3 > 3 * naive, (
        f"when the loss drifts in stretches of 1,000 tokens, resampling contiguous blocks must give a much wider "
        f"interval than an i.i.d. one ({naive:.4f}); got {hi3 - lo3:.4f}")
    _, lo4, hi4 = nb.bpc_ci(drift, block_size=1, rng=0)
    assert hi4 - lo4 < 1.5 * naive, (
        f"block_size=1 resamples single tokens, an i.i.d. bootstrap: its interval should be close to {naive:.4f}; got "
        f"{hi4 - lo4:.4f}. use the block_size argument")


# ---------------------------------------------------------------- stage 4: the scaling fit


def test_stage4_fit_scaling(nb):
    n = np.array([1e5, 4e5, 1.6e6, 6.4e6])
    a, b = nb.fit_scaling(n, 9.0 * n ** -0.12)
    assert math.isclose(a, 9.0, rel_tol=1e-6) and math.isclose(b, 0.12, rel_tol=1e-6), (
        f"for losses exactly 9 N^-0.12 the fit must return (9, 0.12) (b positive: loss falls with N); got ({a}, {b})")
    noisy = 5.0 * n ** -0.08 * np.exp(np.array([0.01, -0.01, 0.005, -0.005]))
    a2, b2 = nb.fit_scaling(list(n), list(noisy))
    assert 4.0 < a2 < 6.0 and 0.07 < b2 < 0.09, "a least-squares line through (log N, log L)"


# ---------------------------------------------------------------- stage 5: generation speed


class _Counting(nn.Module):
    def __init__(self, gpt):
        super().__init__()
        self.gpt, self.blocks, self.block_size, self.lengths = gpt, gpt.blocks, gpt.block_size, []

    def forward(self, idx, targets=None, *, cache=None):
        self.lengths.append(idx.shape[1])
        return self.gpt(idx, targets, cache=cache)

    @torch.no_grad()
    def generate(self, idx, n, **kw):
        for _ in range(n):
            nxt = self(idx[:, -self.block_size:])[0][:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, nxt], 1)
        return idx


def test_stage5_generation_speed(nb):
    torch.manual_seed(2)
    gpt = GPT(9, block_size=40, n_layers=1, n_heads=2, d_model=16)
    m = _Counting(gpt)
    s = nb.generation_speed(m, torch.randint(0, 9, (1, 5)), 12, cached=True)
    assert isinstance(s, float) and s > 0, f"returns tokens per second, a positive float; got {s!r}"
    assert m.lengths[0] == 5 and set(m.lengths[1:]) == {1}, (
        f"cached=True must generate with your generate_cached (prefill once, then one token per call); calls had "
        f"lengths {m.lengths}")
    m.lengths.clear()
    nb.generation_speed(m, torch.randint(0, 9, (1, 5)), 12, cached=False)
    assert m.lengths == list(range(5, 17)), (
        f"cached=False uses the model's own generate, which re-reads the whole sequence every step; lengths {m.lengths}")


def test_stage5_counts_only_new_tokens(nb):
    import time as _time

    torch.manual_seed(3)
    gpt = GPT(9, block_size=40, n_layers=1, n_heads=2, d_model=16)
    ticks = iter(range(1000))
    saved = _time.perf_counter, _time.time, _time.monotonic
    fake = lambda: float(next(ticks))  # noqa: E731
    _time.perf_counter = _time.time = _time.monotonic = fake
    try:
        s = nb.generation_speed(gpt, torch.randint(0, 9, (2, 7)), 10, cached=True)
    finally:
        _time.perf_counter, _time.time, _time.monotonic = saved
    assert math.isclose(s, 20.0), (
        f"with a clock that advances 1 s between the two readings, 2 sequences x 10 new tokens is 20 tokens per second; "
        f"got {s}. count only the generated tokens (not the 7-token prompt), for every sequence in the batch, and "
        f"read the clock once before and once after generating")
