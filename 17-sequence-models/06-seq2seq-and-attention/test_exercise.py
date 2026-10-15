"""Checks for exercise.py (seq2seq & attention).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. Attention is checked
against its formula, masking and a hand example; the encoder-decoder against causality (teacher
forcing never peeks ahead), padding invariance, its attention maps and a reversal task it must
learn (aggregated over seeds); greedy decoding against step-by-step argmax with the full forward;
beam search against greedy decoding (beam 1), exhaustive search on small problems, its tie rule,
length normalisation and the shrinking-beam stopping rule. Loops run under `mlfs.testing.bounded`.
"""
import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from mlfs.testing import bounded  # noqa: E402


def _warm_up():
    """torch imports parts of itself on first use (millions of lines); do that here, outside any line budget."""
    p = nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam([p])
    (p ** 2).sum().backward()
    opt.step()
    nn.GRU(2, 2, batch_first=True)(torch.zeros(1, 2, 2))[0].sum().backward()
    nn.GRUCell(2, 2)(torch.zeros(1, 2)).sum().backward()


_warm_up()


# ---------------------------------------------------------------- stage 1: attention


def test_stage1_matches_formula(nb):
    torch.manual_seed(0)
    att = nb.BahdanauAttention(5, 3, 4)
    assert att.W_q.bias is None and att.v.bias is None and att.W_k.bias is not None, (
        "W_q and v have no bias; W_k has one (one bias inside the tanh is enough)")
    assert tuple(att.W_q.weight.shape) == (4, 5) and tuple(att.W_k.weight.shape) == (4, 3)
    assert tuple(att.v.weight.shape) == (1, 4)
    q, k = torch.randn(2, 5), torch.randn(2, 6, 3)
    ctx, w = att(q, k)
    assert tuple(ctx.shape) == (2, 3) and tuple(w.shape) == (2, 6), "context (B, Dk) and weights (B, T)"
    want_w = torch.empty(2, 6)
    for b in range(2):
        for j in range(6):
            want_w[b, j] = att.v.weight[0] @ torch.tanh(att.W_q.weight @ q[b] + att.W_k.weight @ k[b, j] + att.W_k.bias)
    want_w = torch.softmax(want_w, dim=1)
    assert torch.allclose(w, want_w, atol=1e-6), (
        "weights = softmax over positions of v . tanh(W_q q + W_k k_j); got different weights")
    assert torch.allclose(ctx, (want_w[:, :, None] * k).sum(1), atol=1e-6), "context = sum_j weights_j * keys_j"


def test_stage1_mask_and_hand_example(nb):
    torch.manual_seed(1)
    att = nb.BahdanauAttention(2, 2, 3)
    q, k = torch.randn(3, 2), torch.randn(3, 4, 2)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True], [False, True, False, False]])
    ctx, w = att(q, k, mask)
    assert torch.all(w[~mask] == 0), "masked positions (mask False) get exactly zero weight"
    assert torch.allclose(w.sum(1), torch.ones(3)), "the weights over the allowed positions sum to 1"
    assert torch.allclose(ctx[2], k[2, 1]), "with a single allowed position, the context is that key"
    with torch.no_grad():
        for p in att.parameters():
            p.zero_()
    ctx, w = att(q, k)
    assert torch.allclose(w, torch.full((3, 4), 0.25)), "all scores 0 -> uniform weights"
    assert torch.allclose(ctx, k.mean(1), atol=1e-6), "and the context is the mean of the keys"


# ---------------------------------------------------------------- stage 2: the encoder-decoder


def test_stage2_structure_and_shapes(nb):
    m = nb.Seq2Seq(11, 13, embed_dim=8, hidden_dim=16)
    assert isinstance(m.encoder, nn.GRU) and isinstance(m.decoder, nn.GRUCell), (
        "encoder: nn.GRU (batch_first=True); decoder: nn.GRUCell, stepped once per output token")
    assert m.encoder.batch_first, "batch-first, like everything in the course"
    assert isinstance(m.attn, nb.BahdanauAttention)
    src, tgt = torch.randint(0, 11, (3, 7)), torch.randint(0, 13, (3, 5))
    logits = m(src, tgt)
    assert tuple(logits.shape) == (3, 5, 13), f"logits (B, T, V_tgt); got {tuple(logits.shape)}"
    assert m.last_attention is not None and tuple(m.last_attention.shape) == (3, 5, 7), (
        "last_attention holds the (B, T, S) attention weights of the last forward")
    assert torch.allclose(m.last_attention.sum(-1), torch.ones(3, 5), atol=1e-5)
    plain = nb.Seq2Seq(11, 13, embed_dim=8, hidden_dim=16, attention=False)
    assert plain.attn is None and tuple(plain(src, tgt).shape) == (3, 5, 13)
    assert plain.last_attention is None, "without attention, last_attention stays None"
    assert plain.decoder.input_size == 8 and m.decoder.input_size == 8 + 16, (
        "with attention the decoder reads [embedding; context]; without, just the embedding")


def test_stage2_decoder_reads_the_context(nb):
    torch.manual_seed(5)
    m = nb.Seq2Seq(9, 9, embed_dim=8, hidden_dim=12)
    keys, state, mask = m.encode(torch.randint(0, 9, (2, 6)))
    keys = keys.detach().requires_grad_(True)
    logits, new_state, w = m.decode_step(torch.tensor([1, 1]), state.detach(), keys, mask)
    assert tuple(logits.shape) == (2, 9) and tuple(new_state.shape) == (2, 12) and tuple(w.shape) == (2, 6), (
        "decode_step returns (logits (B, V), new state (B, H), attention weights (B, S))")
    with torch.no_grad():
        _, want_w = m.attn(state, keys, mask)
    assert torch.allclose(w, want_w, atol=1e-6), (
        "Bahdanau order: attend with the PREVIOUS decoder state (the one passed in), then update the state with the "
        "context; the returned weights must be attn(state_prev, keys, mask)")
    (g,) = torch.autograd.grad(new_state.sum(), keys)
    assert g[:, 0].abs().sum() > 0, (
        "the new decoder state must depend on the attention context (the GRU cell reads [embedding; context]), so it "
        "depends on every key, including the first")


def test_stage2_teacher_forcing_is_causal(nb):
    torch.manual_seed(2)
    for att in (True, False):
        m = nb.Seq2Seq(9, 9, embed_dim=8, hidden_dim=12, attention=att)
        src = torch.randint(0, 9, (2, 6))
        tgt = torch.randint(0, 9, (2, 5))
        a = m(src, tgt)
        tgt2 = tgt.clone()
        tgt2[:, 3:] = (tgt2[:, 3:] + 1) % 9
        b = m(src, tgt2)
        assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6), (
            "the logits at step t may depend only on tgt_in[:, :t+1]: changing later tokens must not change earlier "
            "logits (one decoder step per position, in order)")
        assert not torch.allclose(a[:, 3:], b[:, 3:]), "but they do depend on the token fed at their own step"


def test_stage2_padding_is_ignored(nb):
    torch.manual_seed(3)
    m = nb.Seq2Seq(10, 10, embed_dim=8, hidden_dim=12)
    short = torch.tensor([[3, 4, 5]])
    padded = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 1]])
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    tgt = torch.tensor([[1, 2, 3, 4]])
    alone = m(short, tgt)
    batch = m(padded, tgt.expand(2, 4), src_mask=mask)
    assert torch.allclose(alone[0], batch[0], atol=1e-5), (
        "a padded source with src_mask must give exactly what the unpadded source gives: start the decoder from the "
        "encoder state at the LAST VALID position, and mask the padding out of the attention")
    assert torch.all(m.last_attention[0, :, 3:] == 0), "no attention on padding"


def _reversal_batch(g, B=64, S=5, V=6):
    src = torch.randint(3, V + 3, (B, S), generator=g)  # 0 pad, 1 bos, 2 eos
    tgt = torch.cat([torch.ones(B, 1, dtype=torch.long), src.flip(1), torch.full((B, 1), 2)], dim=1)
    return src, tgt[:, :-1], tgt[:, 1:]


def _train_reversal(nb, seed, steps=80):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = nb.Seq2Seq(9, 9, embed_dim=16, hidden_dim=48)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(steps):
        src, tin, tout = _reversal_batch(g)
        loss = nn.functional.cross_entropy(m(src, tin).reshape(-1, 9), tout.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    src, tin, tout = _reversal_batch(g, B=256)
    with torch.no_grad():
        return (m(src, tin).argmax(-1) == tout).float().mean().item()


def test_stage2_learns_to_reverse(nb):
    accs = [bounded(_train_reversal, nb, s, budget=100_000_000, what="training your Seq2Seq") for s in range(5)]
    assert np.median(accs) > 0.9, (
        f"with teacher forcing and attention, reversing 5 symbols should be learned in 80 steps; token accuracy "
        f"over 5 seeds: {np.round(accs, 3)}")


# ---------------------------------------------------------------- stage 3: greedy decoding


def test_stage3_matches_stepwise_argmax(nb):
    torch.manual_seed(4)
    m = nb.Seq2Seq(8, 8, embed_dim=8, hidden_dim=16)
    src = torch.randint(3, 8, (3, 4))
    out = bounded(nb.greedy_decode, m, src, 1, 2, max_len=6, what="greedy_decode")
    assert len(out) == 3 and all(seq[0] == 1 for seq in out), "one list per source row, each starting with bos"
    for b in range(3):
        prefix = [1]
        for _ in range(6):
            with torch.no_grad():
                nxt = int(m(src[b:b + 1], torch.tensor([prefix])).argmax(-1)[0, -1])
            prefix.append(nxt)
            if nxt == 2:
                break
        assert out[b] == prefix, (
            f"row {b}: greedy decoding should equal feeding the argmax back in, step by step: expected {prefix}, got "
            f"{out[b]}. stop after eos (included) or after max_len tokens")
    assert m.training, "greedy_decode restores the model's mode"


class _Countdown(nn.Module):
    """a fake encoder-decoder: row b emits token 3 until step src[b, 0], then eos (2), then token 3 again."""

    def encode(self, src, src_mask=None):
        return src, torch.zeros(src.shape[0]), src_mask

    def decode_step(self, y_prev, state, keys, src_mask):
        state = state + 1
        logits = torch.zeros(keys.shape[0], 4)
        logits[:, 3] = 1.0
        logits[state == keys[:, 0].float(), 2] = 5.0
        return logits, state, None


def test_stage3_rows_stop_independently(nb):
    out = bounded(nb.greedy_decode, _Countdown(), torch.tensor([[1], [3], [5]]), 1, 2, max_len=9, what="greedy_decode")
    want = [[1, 2], [1, 3, 3, 2], [1, 3, 3, 3, 3, 2]]
    assert out == want, (
        f"each row stops at its OWN first eos, while the others keep going: expected {want}, got {out}. keep a "
        f"done flag per row and stop appending to a row once it has produced eos")


def test_stage3_stops_at_eos_and_max_len(nb):
    m = nb.Seq2Seq(5, 4, embed_dim=4, hidden_dim=8, attention=False)
    with torch.no_grad():
        m.out.weight.zero_()
        m.out.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))  # eos (2) is always the argmax
    out = nb.greedy_decode(m, torch.randint(0, 5, (2, 3)), 1, 2, max_len=10)
    assert out == [[1, 2], [1, 2]], f"decoding stops right after eos, which is included; got {out}"
    with torch.no_grad():
        m.out.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 5.0]))  # never eos
    out = bounded(nb.greedy_decode, m, torch.randint(0, 5, (1, 3)), 1, 2, max_len=7, what="greedy_decode")
    assert out == [[1] + [3] * 7], f"without eos, stop after max_len generated tokens; got {out}"


# ---------------------------------------------------------------- stage 4: beam search


def _table_fn(V, seed, eos=0):
    """a deterministic 'language model': log-probs that depend on the whole prefix."""
    def step(prefixes):
        rows = []
        for p in prefixes:
            rng = np.random.default_rng(abs(hash((seed, tuple(p)))) % (2 ** 32))
            logits = rng.normal(size=V) * 2
            rows.append(logits - np.log(np.exp(logits).sum()))
        return np.array(rows)
    return step


def _exhaustive(step, V, bos, eos, max_len, alpha):
    best = []
    def lp(prefix, tok):
        return step([prefix])[0][tok]
    for L in range(1, max_len + 1):
        for toks in itertools.product(range(V), repeat=L):
            if eos in toks[:-1]:
                continue
            if L < max_len and toks[-1] != eos:
                continue
            seq, s = [bos], 0.0
            for t in toks:
                s += lp(seq, t)
                seq = seq + [t]
            best.append((seq, s / L ** alpha))
    return max(best, key=lambda r: r[1])


def test_stage4_beam_one_is_greedy(nb):
    step = _table_fn(5, 0)
    res = bounded(nb.beam_search, step, 9, 0, beam_size=1, max_len=8, what="beam_search")
    seq = [9]
    for _ in range(8):
        tok = int(np.argmax(step([seq])[0]))
        seq.append(tok)
        if tok == 0:
            break
    assert len(res) == 1 and list(res[0][0]) == seq, (
        f"beam_size=1 must be greedy decoding: expected {seq}, got {res[0][0] if res else res}")


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_stage4_wide_beam_is_exact(nb, alpha):
    for seed in range(3):
        step = _table_fn(3, seed + 1)
        want = _exhaustive(step, 3, 7, 0, 4, alpha)
        res = bounded(nb.beam_search, step, 7, 0, beam_size=200, max_len=4, length_penalty=alpha, what="beam_search")
        assert list(res[0][0]) == want[0] and math.isclose(res[0][1], want[1], rel_tol=1e-9), (
            f"length_penalty={alpha}: with a beam wide enough to keep everything, the best result must be the "
            f"exhaustive optimum {want}; got {res[0]}. score = (sum of log-probs) / (tokens after bos) ** length_penalty; "
            f"hypotheses still unfinished at max_len count too")


def test_stage4_output_format_and_order(nb):
    step = _table_fn(6, 11)
    res = nb.beam_search(step, 1, 0, beam_size=4, max_len=6)
    assert 1 <= len(res) <= 4, "at most beam_size results"
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True), "sorted best first"
    for seq, s in res:
        assert seq[0] == 1 and all(isinstance(t, int) for t in seq), "token lists of ints, starting with bos"
        lp = sum(step([seq[:i]])[0][seq[i]] for i in range(1, len(seq)))
        assert math.isclose(s, lp / (len(seq) - 1), rel_tol=1e-9), (
            f"score of {seq} should be {lp / (len(seq) - 1):.6f} (length_penalty 1: divide by the number of tokens "
            f"after bos, eos included); got {s:.6f}")
        assert seq[-1] == 0 or len(seq) - 1 == 6, "a result either ends with eos or reached max_len"
    tens = nb.beam_search(lambda p: torch.tensor(step(p)), 1, 0, beam_size=4, max_len=6)
    assert [list(a) for a, _ in tens] == [list(a) for a, _ in res], "step_fn may return a tensor too"


def test_stage4_impossible_tokens_are_skipped(nb):
    seen = []

    def step(prefixes):  # at the first step only token 1 is possible; after that, anything
        seen.extend(list(p) for p in prefixes)
        out = np.full((len(prefixes), 4), np.log(0.25))
        for i, p in enumerate(prefixes):
            if len(p) == 1:
                out[i] = -np.inf
                out[i, 1] = 0.0
        return out

    res = bounded(nb.beam_search, step, 9, 0, beam_size=3, max_len=3, what="beam_search")
    assert all(np.isfinite(s) for _, s in res), (
        f"an extension with log-probability -inf is impossible: never keep it, even when there are fewer possible "
        f"extensions than beam slots; got scores {[s for _, s in res]}")
    assert all(p[1] == 1 for p in seen if len(p) > 1), (
        f"step_fn must never be asked to extend an impossible prefix; it saw {[p for p in seen if len(p) > 1 and p[1] != 1]}")
    assert all(list(t)[1] == 1 for t, _ in res), "every result starts with the only possible first token"


def test_stage4_ties_and_shrinking_beam(nb):
    calls = []

    def flat(prefixes):  # every token equally likely, except eos which is impossible until length 3
        calls.append(len(prefixes))
        out = np.full((len(prefixes), 4), np.log(0.25))
        for i, p in enumerate(prefixes):
            if len(p) < 3:
                out[i, 0] = -np.inf
                out[i, 1:] = np.log(1 / 3)
        return out

    res = nb.beam_search(flat, 9, 0, beam_size=2, max_len=5)
    assert [list(s) for s, _ in res] == [[9, 1, 1, 0], [9, 1, 1, 1, 0]], (
        f"on ties, prefer the earlier beam, then the smaller token id: the beams go [9,1] [9,2] -> [9,1,1] [9,1,2] -> "
        f"[9,1,1,0] (finished) [9,1,1,1] -> [9,1,1,1,0]; got {res}")
    assert all(c <= 2 for c in calls), f"step_fn never sees more than beam_size prefixes; got batch sizes {calls}"
    assert len(calls) <= 5, "and it stops at max_len"

    def eos_first(prefixes):
        out = np.full((len(prefixes), 3), np.log(0.1))
        out[:, 0] = np.log(0.8)
        return out

    calls.clear()
    res = nb.beam_search(lambda p: (calls.append(len(p)), eos_first(p))[1], 5, 0, beam_size=3, max_len=10)
    assert len(calls) <= 3, (
        f"each finished hypothesis takes one of the beam_size slots, and the search stops once beam_size hypotheses "
        f"have finished; step_fn was called {len(calls)} times")
    assert list(res[0][0]) == [5, 0], f"the best is eos straight away (log 0.8 over 1 token); got {res[0]}"
