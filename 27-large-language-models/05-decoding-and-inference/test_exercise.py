"""Checks for 27 · 05 decoding & inference.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. tiny bigram language models
written here (a table of next-token logits) stand in for real ones, so the exact output distributions are known.
repetition penalty and min-p on hand examples; the speculative acceptance step statistically (chi-squared
over many draws): its output must be distributed exactly as the target's, with the right acceptance rate and
residual; speculative decoding for exactness (greedy equals the target's greedy decoding; sampling matches
the target's joint distribution of two tokens), for k (number of target calls), the acceptance rate, length
and determinism; constrained decoding for never leaving the allowed set, for sampling the renormalized
distribution, and for what `allowed` is given.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from scipy.stats import chisquare as _chisquare  # noqa: E402
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _acc(nb, *args, **kwargs):
    """one speculative_accept call under a line budget: a loop that never ends fails instead of hanging."""
    return bounded(nb.speculative_accept, *args, what="speculative_accept", budget=50_000, **kwargs)


def _spec(nb, *args, **kwargs):
    return bounded(nb.speculative_decode, *args, what="speculative_decode", budget=400_000, **kwargs)


def _cons(nb, *args, **kwargs):
    return bounded(nb.constrained_decode, *args, what="constrained_decode", budget=400_000, **kwargs)


class _Bigram(nn.Module):
    """causal LM whose next-token logits depend only on the current token; counts forward calls."""

    def __init__(self, table, block_size=None):
        super().__init__()
        self.table = nn.Parameter(torch.as_tensor(table, dtype=torch.float32), requires_grad=False)
        self.block_size = block_size
        self.calls = 0
        self.lengths = []
        self.modes = []

    def forward(self, idx, targets=None, *, cache=None):
        self.calls += 1
        self.lengths.append(idx.shape[1])
        self.modes.append(self.training)
        if self.block_size is not None:
            assert idx.shape[1] <= self.block_size, "context longer than block_size"
        return self.table[idx], None


P_TABLE = torch.log(torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.6, 0.1, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25],
                                  [0.05, 0.05, 0.1, 0.8]]))
Q_TABLE = torch.log(torch.tensor([[0.4, 0.2, 0.2, 0.2], [0.2, 0.2, 0.5, 0.1], [0.1, 0.6, 0.2, 0.1],
                                  [0.3, 0.3, 0.2, 0.2]]))


def chisquare(counts, expected):
    """chi-squared goodness of fit, with the expected counts renormalized in float64 to the observed total."""
    counts = np.asarray(counts, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return _chisquare(counts, expected / expected.sum() * counts.sum())


# ---------------------------------------------------------------- stage 1: sampling filters


def test_stage1_repetition_penalty(nb):
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0], [1.0, 1.0, -2.0, 0.0]])
    prev = torch.tensor([[0, 1, 1], [2, 2, 2]])
    got = nb.apply_repetition_penalty(logits, prev, 2.0)
    want = torch.tensor([[1.0, -2.0, 0.5, 3.0], [1.0, 1.0, -4.0, 0.0]])
    assert torch.allclose(got, want), (
        f"tokens already seen: positive logits / penalty, negative logits * penalty (both make them less likely); a token "
        f"seen three times is penalized once. want {want.tolist()}, got {got.tolist()}")
    assert torch.equal(logits, torch.tensor([[2.0, -1.0, 0.5, 3.0], [1.0, 1.0, -2.0, 0.0]])), "don't modify the input"
    assert torch.allclose(nb.apply_repetition_penalty(logits, prev, 1.0), logits), "penalty 1 changes nothing"
    with pytest.raises(ValueError):
        nb.apply_repetition_penalty(logits, prev, 0.0)


def test_stage1_min_p(nb):
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05], [0.25, 0.25, 0.25, 0.25]])
    logits = probs.log()
    got = nb.min_p_filter(logits, 0.4)
    kept = torch.isfinite(got)
    assert kept.tolist() == [[True, True, False, False], [True, True, True, True]], (
        "min_p 0.4 keeps tokens with probability >= 0.4 x the top probability: 0.5 and 0.3 (>= 0.2) in row 0, all of a "
        f"flat row 1; got {kept.tolist()}")
    assert torch.allclose(got[kept], logits[kept]), "kept logits are unchanged; the rest become -inf"
    assert torch.isfinite(nb.min_p_filter(logits, 0.0)).all(), "min_p 0 keeps everything"
    assert torch.isfinite(nb.min_p_filter(logits, 1.0)).sum().item() == 1 + 4, "min_p 1 keeps only the top (ties all)"
    shifted = nb.min_p_filter(logits * 0.5 + 3.0, 0.4)
    assert torch.isfinite(shifted).sum() == torch.isfinite(nb.min_p_filter(logits * 0.5, 0.4)).sum(), (
        "compare probabilities (softmax), not raw logits")
    with pytest.raises(ValueError):
        nb.min_p_filter(logits, 1.5)


# ---------------------------------------------------------------- stage 2: acceptance step


def _counts(draw, n_cats, n):
    c = np.zeros(n_cats)
    for _ in range(n):
        c[draw()] += 1
    return c


def test_stage2_first_token_is_exact(nb):
    p = torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.25, 0.25, 0.25, 0.25]])
    q = torch.tensor([[0.4, 0.2, 0.2, 0.2]])
    g = torch.Generator().manual_seed(0)
    acc = [0]

    def draw():
        x = torch.multinomial(q[0], 1, generator=g)
        n, nxt = _acc(nb, p, q, x, generator=g)
        acc[0] += n
        return int(x[0]) if n == 1 else nxt

    N = 12000
    c = _counts(draw, 4, N)
    stat, pval = chisquare(c, N * p[0].numpy())
    assert pval > 1e-4, (
        f"the token emitted at the first position must be distributed exactly as the target's p = {p[0].tolist()}; got "
        f"frequencies {np.round(c / N, 3).tolist()} (chi-squared p = {pval:.1e}). accept the draft with probability "
        "min(1, p/q); on rejection sample from max(0, p - q), renormalized")
    rate = acc[0] / N
    assert abs(rate - 0.6) < 0.025, (
        f"the acceptance rate is sum_x min(p(x), q(x)) = 0.1 + 0.2 + 0.2 + 0.1 = 0.6; got {rate:.3f}")


def test_stage2_residual_and_bonus(nb):
    p = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
    q = torch.tensor([[0.8, 0.2]])
    g = torch.Generator().manual_seed(1)
    for _ in range(300):
        n, nxt = _acc(nb, p, q, torch.tensor([0]), generator=g)
        assert n in (0, 1) and isinstance(nxt, int), "returns (number accepted, next token as an int)"
        if n == 0:
            assert nxt == 1, "after rejecting token 0, the residual max(0, p - q) = (0, 0.6) puts all its mass on token 1"
    same = torch.tensor([[0.5, 0.2, 0.3], [0.5, 0.2, 0.3], [0.1, 0.1, 0.8]])
    c = np.zeros(3)
    for _ in range(4000):
        n, nxt = _acc(nb, same, same[:2], torch.tensor([0, 2]), generator=g)
        assert n == 2, "when the draft equals the target every draft token is accepted"
        c[nxt] += 1
    stat, pval = chisquare(c, 4000 * same[2].numpy())
    assert pval > 1e-4, (
        f"after accepting all k drafts, the extra (bonus) token is sampled from the target's last row p[k] = "
        f"{same[2].tolist()}; got {np.round(c / 4000, 3).tolist()}")


def test_stage2_two_drafts_joint(nb):
    # target and draft as bigram tables; drafts sampled from q; emitted pair must follow p's chain
    p_rows, q_rows = P_TABLE.softmax(-1), Q_TABLE.softmax(-1)
    g = torch.Generator().manual_seed(2)
    counts = np.zeros(16)
    N = 8000
    for _ in range(N):
        d1 = int(torch.multinomial(q_rows[0], 1, generator=g))
        d2 = int(torch.multinomial(q_rows[d1], 1, generator=g))
        p = torch.stack([p_rows[0], p_rows[d1], p_rows[d2]])
        q = torch.stack([q_rows[0], q_rows[d1]])
        n, nxt = _acc(nb, p, q, torch.tensor([d1, d2]), generator=g)
        out = [d1, d2][:n] + [nxt]
        if len(out) < 2:  # rejected at position 0: the second token comes from the target (as decoding would)
            out.append(int(torch.multinomial(p_rows[out[0]], 1, generator=g)))
        counts[4 * out[0] + out[1]] += 1
    want = (p_rows[0][:, None] * p_rows).flatten().numpy()
    stat, pval = chisquare(counts, N * want)
    assert pval > 1e-4, (
        f"two drafted tokens: the first two emitted tokens must follow the target's joint distribution (chi-squared p = "
        f"{pval:.1e}). check that position i uses p[i] and q[i], and that you stop at the first rejection")


# ---------------------------------------------------------------- stage 3: speculative decoding


def test_stage3_greedy_is_exact(nb):
    target, draft = _Bigram(P_TABLE), _Bigram(Q_TABLE)
    idx = torch.tensor([[3, 0]])
    out, rate = bounded(nb.speculative_decode, target, draft, idx, 12, k=3, temperature=0.0, what="speculative_decode")
    want = [3, 0]
    for _ in range(12):
        want.append(int(P_TABLE[want[-1]].argmax()))
    assert out.shape == (1, 14) and out[0].tolist() == want, (
        f"temperature 0: the output must equal the target's own greedy decoding {want}; got {out[0].tolist()}")
    assert 0.0 <= rate <= 1.0 and isinstance(rate, float)
    t2, d2 = _Bigram(P_TABLE), _Bigram(P_TABLE)
    out2, rate2 = _spec(nb, t2, d2, idx, 10, k=4, temperature=0.0)
    assert rate2 == 1.0, "a draft identical to the target is always accepted (acceptance rate 1.0)"
    assert t2.calls == 2, (
        f"with every draft accepted each target call yields k + 1 = 5 tokens, so 10 tokens take 2 target calls (each "
        f"verifying all k drafts in ONE forward); got {t2.calls}")


def test_stage3_k_and_length(nb):
    for k, want_calls in ((1, 5), (2, 3), (9, 1)):
        t, d = _Bigram(P_TABLE), _Bigram(P_TABLE)
        out, rate = _spec(nb, t, d, torch.tensor([[1]]), 9, k=k, temperature=0.7,
                                          generator=torch.Generator().manual_seed(0))
        assert out.shape == (1, 10), f"exactly max_new_tokens new tokens: shape (1, 10), got {tuple(out.shape)}"
        assert t.calls == want_calls and rate == 1.0, (
            f"k={k}, draft = target: each round accepts all k drafts plus a bonus token, so 9 tokens need "
            f"{want_calls} target calls; got {t.calls}. is k used?")
        assert d.calls <= 9 + k, "the draft runs once per proposed token"
    with pytest.raises(ValueError):
        _spec(nb, _Bigram(P_TABLE), _Bigram(P_TABLE), torch.tensor([[1], [2]]), 3)


def test_stage3_sampling_is_exact(nb):
    target, draft = _Bigram(P_TABLE), _Bigram(Q_TABLE)
    g = torch.Generator().manual_seed(3)
    N = 2500
    counts = np.zeros(16)
    rates = []
    for _ in range(N):
        out, rate = _spec(nb, target, draft, torch.tensor([[0]]), 2, k=2, generator=g)
        counts[4 * int(out[0, 1]) + int(out[0, 2])] += 1
        rates.append(rate)
    p = P_TABLE.softmax(-1)
    want = (p[0][:, None] * p).flatten().numpy()
    stat, pval = chisquare(counts, N * want)
    assert pval > 1e-4, (
        f"with temperature 1 the two generated tokens must follow the target's distribution exactly, whatever the draft "
        f"(chi-squared p = {pval:.1e}). got {np.round(counts / N, 3).tolist()}")
    assert 0.2 < np.mean(rates) < 0.8, "a draft that differs from the target is sometimes rejected"


def test_stage3_temperature_and_modes(nb):
    target, draft = _Bigram(P_TABLE * 3), _Bigram(P_TABLE * 3)
    target.train()
    draft.train()
    g = torch.Generator().manual_seed(4)
    N = 3000
    first = np.zeros(4)
    rates = []
    for _ in range(N):
        out, r = _spec(nb, target, draft, torch.tensor([[0]]), 1, k=1, temperature=2.0, generator=g)
        first[int(out[0, 1])] += 1
        rates.append(r)
    assert min(rates) == 1.0, (
        "draft = target, both at temperature 2: every draft must be accepted. sample the draft's tokens at the same "
        "temperature as the target, or the acceptance rate drops (the output stays exact, but you lose the speedup)")
    want = F.softmax(P_TABLE[0] * 3 / 2.0, -1).numpy()
    stat, pval = chisquare(first, N * want)
    assert pval > 1e-4, (
        f"temperature 2 divides both models' logits by 2 before the softmax; first-token frequencies "
        f"{np.round(first / N, 3).tolist()}, want {np.round(want, 3).tolist()}")
    assert target.training and draft.training, "restore the models' train/eval mode afterwards"
    assert target.modes and not any(target.modes) and not any(draft.modes), (
        "both models must run in eval mode (no dropout) during decoding: call .eval() first")
    a = _spec(nb, _Bigram(P_TABLE), _Bigram(Q_TABLE), torch.tensor([[2]]), 15,
                              generator=torch.Generator().manual_seed(9))[0]
    b = _spec(nb, _Bigram(P_TABLE), _Bigram(Q_TABLE), torch.tensor([[2]]), 15,
                              generator=torch.Generator().manual_seed(9))[0]
    assert torch.equal(a, b), "the same generator seed gives the same tokens"
    small = _Bigram(P_TABLE, block_size=6)
    out, _ = _spec(nb, small, _Bigram(Q_TABLE, block_size=6), torch.tensor([[1, 2, 3, 0, 1]]), 8,
                                   k=3, temperature=0.0)
    assert out.shape == (1, 13) and max(small.lengths) <= 6, "crop the context to the models' block_size"


# ---------------------------------------------------------------- stage 4: constrained decoding


def test_stage4_stays_in_the_allowed_set(nb):
    model = _Bigram(P_TABLE)
    even = torch.tensor([True, False, True, False])
    g = torch.Generator().manual_seed(5)
    counts = np.zeros(4)
    for _ in range(3000):
        out = _cons(nb, model, torch.tensor([[0]]), 1, lambda gen: even, generator=g)
        counts[int(out[0, 1])] += 1
    assert counts[1] == counts[3] == 0, "never emit a disallowed token"
    p = P_TABLE[0].softmax(-1).numpy()
    want = np.array([p[0], 0, p[2], 0]) / (p[0] + p[2])
    stat, pval = chisquare(counts[[0, 2]], 3000 * want[[0, 2]])
    assert pval > 1e-4, (
        f"among allowed tokens, sample the model's distribution renormalized: want {np.round(want, 3).tolist()}, got "
        f"{np.round(counts / 3000, 3).tolist()}")
    hot = np.zeros(4)
    for _ in range(3000):
        out = _cons(nb, model, torch.tensor([[0]]), 1, lambda gen: even, temperature=3.0, generator=g)
        hot[int(out[0, 1])] += 1
    ph = F.softmax(P_TABLE[0] / 3.0, -1).numpy()
    stat, pval = chisquare(hot[[0, 2]], ph[[0, 2]])
    assert pval > 1e-4, "temperature 3: renormalize softmax(logits / 3) over the allowed tokens"
    greedy = _cons(nb, model, torch.tensor([[0]]), 3, lambda gen: even, temperature=0.0)
    assert greedy[0].tolist() == [0, 2, 0, 2], "temperature 0: the most likely allowed token (0 -> 2 -> 0 -> 2; ties go to the lower id)"


def test_stage4_modes_and_cropping(nb):
    model = _Bigram(P_TABLE, block_size=4)
    model.train()
    out = _cons(nb, model, torch.tensor([[0, 1, 2, 3]]), 6, lambda gen: torch.ones(4, dtype=torch.bool),
                temperature=0.0)
    assert out.shape == (1, 10) and max(model.lengths) <= 4, "crop the context to the model's block_size"
    assert not any(model.modes) and model.training, "run the model in eval mode, then restore its mode"
    with pytest.raises(ValueError):
        _cons(nb, model, torch.tensor([[0], [1]]), 2, lambda gen: torch.ones(4, dtype=torch.bool))


def test_stage4_allowed_sees_the_prefix(nb):
    seen = []

    def spell_312(gen):
        seen.append(list(gen))
        target = [3, 1, 2]
        mask = torch.zeros(4, dtype=torch.bool)
        if len(gen) < 3:
            mask[target[len(gen)]] = True
        return mask

    model = _Bigram(P_TABLE)
    model.train()
    out = bounded(nb.constrained_decode, model, torch.tensor([[0, 0]]), 10, spell_312, what="constrained_decode",
                  generator=torch.Generator().manual_seed(0))
    assert out[0].tolist() == [0, 0, 3, 1, 2], (
        f"the grammar forces 3, 1, 2 and then allows nothing, which ends generation early; got {out[0].tolist()}")
    assert seen[:4] == [[], [3], [3, 1], [3, 1, 2]], (
        f"allowed() receives the list of tokens generated so far (not the prompt); got {seen[:4]}")
    assert model.training, "restore the model's mode"
