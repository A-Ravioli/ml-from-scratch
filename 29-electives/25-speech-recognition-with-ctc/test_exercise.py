"""Checks for exercise.py (speech recognition with CTC).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The alignment space is
checked against brute-force enumeration of every frame path; the forward algorithm against the sum
over those paths and against `torch.nn.functional.ctc_loss` (values and gradients); decoding against
exhaustive search on tiny vocabularies and an independent prefix beam search; the recognizer by
training it on a toy task.
"""
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mlfs.testing import bounded

torch.set_num_threads(1)  # tiny tensors: one thread is fastest, and avoids oversubscribing a busy machine


def _logp(T, V, seed, peaky=1.0):
    g = np.random.default_rng(seed)
    return np.log(g.dirichlet(np.full(V, peaky), size=T))


def _brute_paths(labels, T, V, blank=0):
    out = []
    for p in itertools.product(range(V), repeat=T):
        o, prev = [], None
        for v in p:
            if v != prev and v != blank:
                o.append(v)
            prev = v
        if o == list(labels):
            out.append(p)
    return out


def _brute_nll(lp, labels, blank=0):
    T, V = lp.shape
    paths = _brute_paths(labels, T, V, blank)
    if not paths:
        return np.inf
    vals = np.array([lp[np.arange(T), list(p)].sum() for p in paths])
    return -np.logaddexp.reduce(vals)


# ---------------------------------------------------------------- stage 1: the alignment space


def test_stage1_collapse(nb):
    cases = [([1, 1, 0, 1, 2, 2, 0], [1, 1, 2]), ([0, 0, 0], []), ([1, 0, 1], [1, 1]), ([1, 1], [1]),
             ([2, 2, 2, 1], [2, 1]), ([], []), ([0, 3, 0, 3, 3, 0, 2], [3, 3, 2])]
    for path, want in cases:
        got = nb.ctc_collapse(path)
        assert list(got) == want, (
            f"collapse({path}) should be {want}, got {got}. merge runs of repeats FIRST, then drop blanks "
            "(so 1 0 1 is two 1s, but 1 1 is one)")
    assert nb.ctc_collapse(np.array([1, 1, 0, 2])) == [1, 2], "numpy input works and the result is a list of ints"
    assert nb.ctc_collapse([3, 1, 1, 3, 2], blank=3) == [1, 2], "honour the blank argument"
    assert nb.ctc_collapse([0, 0, 3, 0], blank=3) == [0, 0], "with blank=3, label 0 is an ordinary label"


def test_stage1_expand_and_min_frames(nb):
    assert nb.ctc_expand([1, 2]) == [0, 1, 0, 2, 0], "[blank, l1, blank, l2, blank]"
    assert nb.ctc_expand([]) == [0], "an empty label sequence expands to a single blank"
    assert nb.ctc_expand([1, 1], blank=5) == [5, 1, 5, 1, 5], "honour the blank argument"
    for labels, want in [([1, 2, 3], 3), ([1, 1], 3), ([1, 1, 1], 5), ([], 0), ([1, 2, 1], 3), ([2, 2, 3, 3], 6)]:
        got = nb.min_frames(labels)
        assert got == want, (
            f"min_frames({labels}) = {want}, got {got}: one frame per label, plus one blank between every pair of "
            "EQUAL neighbours (otherwise they'd merge)")


def test_stage1_alignments_match_brute_force(nb):
    for labels, T, V in [([1], 3, 2), ([1, 2], 4, 3), ([1, 1], 4, 2), ([2, 1, 2], 5, 3), ([], 3, 3),
                         ([1, 1, 2], 6, 3), ([1, 2, 2, 1], 6, 3)]:
        got = bounded(nb.ctc_alignments, labels, T, what="ctc_alignments")
        want = sorted(_brute_paths(labels, T, V))
        got = [tuple(int(v) for v in p) for p in got]
        assert got == want, (
            f"labels {labels}, T={T}: expected the {len(want)} paths {want[:6]}{'...' if len(want) > 6 else ''}, "
            f"got {len(got)}: {got[:6]}. every returned path must collapse to the labels, and none may be "
            "missing (start in the first blank or the first label; stay, advance one, or skip a blank only "
            "between DIFFERENT labels); return them sorted")
    assert len(nb.ctc_alignments([1], 3)) == 6, "labels [1], T=3: 111, 011, 110, 001, 100, 010"
    assert nb.ctc_alignments([1, 1], 2) == [], "[1, 1] needs 3 frames, so T=2 has no alignment"
    got = sorted(tuple(int(v) for v in p) for p in nb.ctc_alignments([0, 1], 4, blank=2))
    assert got == sorted(_brute_paths([0, 1], 4, 3, blank=2)), "honour the blank argument"


# ---------------------------------------------------------------- stage 2: the forward algorithm


def test_stage2_forward_shape_and_start(nb):
    lp = _logp(5, 4, 0)
    alpha = np.asarray(nb.ctc_forward(lp, [1, 3]))
    assert alpha.shape == (5, 5), f"log alpha has shape (T, 2U + 1) = (5, 5), got {alpha.shape}"
    assert np.isclose(alpha[0, 0], lp[0, 0]) and np.isclose(alpha[0, 1], lp[0, 1]), (
        "at t = 0 a path can start in the first blank or in the first label")
    assert np.all(np.isneginf(alpha[0, 2:])), "at t = 0 every later state is unreachable: log alpha = -inf"


def test_stage2_nll_matches_sum_over_paths(nb):
    for labels, T, V, seed in [([1], 4, 3, 1), ([1, 2], 5, 3, 2), ([1, 1], 5, 3, 3), ([2, 1, 2], 6, 3, 4),
                               ([], 4, 3, 5), ([1, 1, 1], 6, 2, 6), ([3, 3, 1], 6, 4, 7)]:
        lp = _logp(T, V, seed)
        got = nb.ctc_nll(lp, labels)
        want = _brute_nll(lp, labels)
        assert np.isclose(got, want), (
            f"labels {labels}, T={T}: -log p should be {want:.6f} (summing every path), got {got:.6f}. "
            "check the skip transition (only into a label that differs from the one two states back), the "
            "start states, and that the end is the last label OR the final blank")


def test_stage2_uniform_counts_alignments(nb):
    V, T = 3, 8
    lp = np.full((T, V), -np.log(V))
    for labels in [[1, 2, 1], [1, 1], [2]]:
        n_paths = len(_brute_paths(labels, T, V))
        got = nb.ctc_nll(lp, labels)
        assert np.isclose(got, T * np.log(V) - np.log(n_paths)), (
            f"with uniform frame probabilities, p(labels) = (number of alignments) / V^T; for {labels} that's "
            f"{n_paths} / {V}^{T}, so -log p = {T * np.log(V) - np.log(n_paths):.4f}, got {got:.4f}")


def test_stage2_matches_torch_and_is_stable(nb):
    g = np.random.default_rng(8)
    for T, labels in [(30, [1, 2, 2, 3, 1]), (1000, list(g.integers(1, 6, 80)))]:
        lp = _logp(T, 6, int(g.integers(1000)))
        got = nb.ctc_nll(lp, labels)
        want = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([labels]), torch.tensor([T]),
                          torch.tensor([len(labels)]), reduction="none").item()
        assert np.isfinite(got) and np.isclose(got, want, rtol=1e-9), (
            f"T={T}, U={len(labels)}: expected {want:.6f} (torch), got {got}. with T = 1000, alpha underflows "
            "(exp(-745) is already 0 in float64): work with log alpha and np.logaddexp throughout, never np.exp")
    lp = _logp(10, 5, 9)
    want = F.ctc_loss(torch.tensor(lp)[:, None], torch.tensor([[0, 2, 2]]), torch.tensor([10]), torch.tensor([3]),
                      blank=4, reduction="none").item()
    assert np.isclose(nb.ctc_nll(lp, [0, 2, 2], blank=4), want), "honour the blank argument (blank=4, label 0 real)"


def test_stage2_impossible_and_empty(nb):
    lp = _logp(3, 3, 10)
    assert np.isposinf(nb.ctc_nll(lp, [1, 1, 2])), "[1, 1, 2] needs 4 frames; with 3 the loss is +inf"
    assert np.isclose(nb.ctc_nll(lp, [1, 2, 1]), _brute_nll(lp, [1, 2, 1])), "[1, 2, 1] fits in exactly 3 frames"
    assert np.isclose(nb.ctc_nll(lp, []), -lp[:, 0].sum()), "an empty label sequence: the all-blank path only"


# ---------------------------------------------------------------- stage 3: batched torch CTC loss


def _batch(seed, dtype=torch.float64):
    torch.manual_seed(seed)
    B, T, V = 5, 14, 6
    logits = torch.randn(B, T, V, dtype=dtype, requires_grad=True)
    targets = torch.tensor([[1, 2, 2, 3, 5], [4, 4, 9, 9, 9], [1, 9, 9, 9, 9], [2, 3, 1, 5, 9], [9, 9, 9, 9, 9]])
    in_len = torch.tensor([14, 11, 6, 14, 9])
    tgt_len = torch.tensor([5, 2, 1, 4, 0])
    return logits, targets, in_len, tgt_len


def _torch_targets(targets, tgt_len):
    t = targets.clone()
    for b in range(len(t)):
        t[b, tgt_len[b]:] = 1  # torch doesn't allow out-of-range padding values
    return t


def test_stage3_matches_torch_values(nb):
    logits, targets, in_len, tgt_len = _batch(0)
    lp = logits.log_softmax(-1)
    for red in ["none", "sum", "mean"]:
        got = nb.ctc_loss(lp, targets, in_len, tgt_len, reduction=red)
        want = F.ctc_loss(lp.transpose(0, 1), _torch_targets(targets, tgt_len), in_len, tgt_len, reduction=red)
        assert torch.is_tensor(got) and got.shape == want.shape, (
            f"reduction={red!r}: expected a tensor of shape {tuple(want.shape)}, got {getattr(got, 'shape', got)}")
        assert torch.allclose(got, want, atol=1e-8), (
            f"reduction={red!r}: expected {want.detach().numpy().round(5)}, got {got.detach().numpy().round(5)}. "
            "log_probs are (B, T, V) here (torch's F.ctc_loss wants (T, B, V)); each item uses only its first "
            "input_lengths[b] frames and target_lengths[b] labels; 'mean' divides each loss by its target length "
            "(at least 1) before averaging over the batch")
    assert torch.allclose(nb.ctc_loss(lp, targets, in_len, tgt_len), nb.ctc_loss(lp, targets, in_len, tgt_len,
                                                                                 reduction="mean")), "default: mean"


def test_stage3_padding_is_ignored(nb):
    logits, targets, in_len, tgt_len = _batch(1)
    a = nb.ctc_loss(logits.log_softmax(-1), targets, in_len, tgt_len, reduction="none")
    noisy = logits.detach().clone()
    for b in range(len(noisy)):
        noisy[b, in_len[b]:] = torch.randn_like(noisy[b, in_len[b]:]) * 5
    t2 = targets.clone()
    for b in range(len(t2)):
        t2[b, tgt_len[b]:] = 3
    b_ = nb.ctc_loss(noisy.log_softmax(-1), t2, in_len, tgt_len, reduction="none")
    assert torch.allclose(a, b_), (
        "frames past input_lengths[b] and labels past target_lengths[b] are padding: changing them must not change "
        "the loss. read each item's final alpha at ITS last frame, at ITS last two states")


def test_stage3_gradients_match_torch(nb):
    logits, targets, in_len, tgt_len = _batch(2)
    for red in ["sum", "mean"]:
        got = nb.ctc_loss(logits.log_softmax(-1), targets, in_len, tgt_len, reduction=red)
        g1, = torch.autograd.grad(got, logits)
        want = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), _torch_targets(targets, tgt_len), in_len, tgt_len,
                          reduction=red)
        g2, = torch.autograd.grad(want, logits)
        assert torch.isfinite(g1).all(), (
            "gradients contain NaN or inf. a logsumexp over entries that are ALL -inf has a NaN gradient: use a "
            "large finite negative number (like -1e30) for unreachable states instead of -inf")
        assert torch.allclose(g1, g2, atol=1e-7), (
            f"reduction={red!r}: gradients w.r.t. the logits differ from torch's (max diff "
            f"{(g1 - g2).abs().max().item():.2e}). build the loss from differentiable torch ops only (no .item(), "
            "no NumPy) so autograd sees the whole forward algorithm")


def test_stage3_float32_blank_and_infinity(nb):
    logits, targets, in_len, tgt_len = _batch(3, dtype=torch.float32)
    lp = logits.log_softmax(-1)
    got = nb.ctc_loss(lp, targets.clamp(max=4), in_len, tgt_len, blank=5, reduction="none")
    want = F.ctc_loss(lp.transpose(0, 1), _torch_targets(targets.clamp(max=4), tgt_len), in_len, tgt_len, blank=5,
                      reduction="none")
    assert got.dtype == torch.float32 and torch.allclose(got, want, rtol=1e-4), (
        "float32 input and blank=5 (labels 0..4 are real): must match torch")
    imp_in = torch.tensor([14, 3, 6, 14, 9])  # item 1 has labels [4, 4]: needs 3 frames -> fine; make it impossible
    tg = targets.clone()
    tg[2, :3] = torch.tensor([1, 1, 1])
    tl = tgt_len.clone()
    tl[2] = 3  # [1, 1, 1] needs 5 frames, item 2 has 6 -> possible; shrink to 4 frames
    imp_in[2] = 4
    loss = nb.ctc_loss(logits.log_softmax(-1), tg, imp_in, tl, reduction="none")
    assert torch.isposinf(loss[2]) and torch.isfinite(loss[[0, 1, 3, 4]]).all(), (
        f"item 2 ([1, 1, 1] in 4 frames) is impossible: its loss is +inf, the others finite; got {loss.tolist()}")
    z = nb.ctc_loss(logits.log_softmax(-1), tg, imp_in, tl, reduction="none", zero_infinity=True)
    assert z[2] == 0 and torch.allclose(z[[0, 1, 3, 4]], loss[[0, 1, 3, 4]]), "zero_infinity=True: impossible -> 0"
    g, = torch.autograd.grad(nb.ctc_loss(logits.log_softmax(-1), tg, imp_in, tl, zero_infinity=True), logits)
    assert torch.isfinite(g).all() and g[2].abs().max() < 1e-6, (
        "with zero_infinity=True the impossible item contributes neither loss nor gradient, and nothing is NaN")
    z0 = nb.ctc_loss(lp[:2], targets[:2].clamp(max=4), torch.tensor([0, 0]), torch.tensor([0, 1]), reduction="none")
    assert z0[0] == 0 and torch.isposinf(z0[1]), (
        f"zero frames: an empty transcript has probability 1 (loss 0), any label is impossible (+inf), as in torch; "
        f"got {z0.tolist()}")
    try:
        nb.ctc_loss(lp, targets, in_len, tgt_len, reduction="avg")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown reduction should raise ValueError")


# ---------------------------------------------------------------- stage 4: decoding


def test_stage4_greedy_worked_example(nb):
    lp = np.log([[0.6, 0.4], [0.6, 0.4]])
    assert list(nb.greedy_decode(lp)) == [], "best path is blank-blank (0.36), so greedy outputs nothing"
    g = np.random.default_rng(11)
    for _ in range(5):
        lp = _logp(12, 5, int(g.integers(1000)), peaky=0.3)
        path = lp.argmax(1)
        want, prev = [], None
        for v in path:
            if v != prev and v != 0:
                want.append(int(v))
            prev = v
        assert list(nb.greedy_decode(lp)) == want, "greedy = argmax of each frame, then collapse"
        assert list(nb.greedy_decode(torch.tensor(lp))) == want, "greedy_decode accepts a torch tensor too"
    lp = np.log([[0.1, 0.2, 0.7], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    assert list(nb.greedy_decode(lp, blank=2)) == [1, 0], "with blank=2: path 2 1 0 collapses to [1, 0]"


def test_stage4_beam_worked_example(nb):
    lp = np.log([[0.6, 0.4], [0.6, 0.4]])
    res = nb.prefix_beam_search(lp, beam_size=4)
    prefixes = [tuple(int(v) for v in p) for p, _ in res]
    probs = [np.exp(s) for _, s in res]
    assert prefixes[:2] == [(1,), ()], (
        f"the best LABELING is 'a' with p = 0.24 + 0.24 + 0.16 = 0.64, then '' with 0.36; got {list(zip(prefixes, probs))}")
    assert np.allclose(probs[:2], [0.64, 0.36]), f"probabilities 0.64 and 0.36, got {probs[:2]}"
    assert len(res) == 2, f"prefixes with probability 0 (like (1, 1) here) are dropped; got {prefixes}"


def _exact_labelings(lp, blank=0):
    T, V = lp.shape
    tot = {}
    for p in itertools.product(range(V), repeat=T):
        o, prev = [], None
        for v in p:
            if v != prev and v != blank:
                o.append(v)
            prev = v
        tot.setdefault(tuple(o), []).append(lp[np.arange(T), list(p)].sum())
    return {k: np.logaddexp.reduce(v) for k, v in tot.items()}


def test_stage4_unpruned_beam_is_exact(nb):
    for T, V, seed, blank in [(5, 3, 12, 0), (6, 3, 13, 0), (4, 4, 14, 0), (5, 3, 15, 2)]:
        lp = _logp(T, V, seed)
        exact = _exact_labelings(lp, blank)
        res = bounded(nb.prefix_beam_search, lp, beam_size=10_000, blank=blank, what="prefix_beam_search")
        got = {tuple(int(v) for v in p): s for p, s in res}
        assert set(got) == set(exact), (
            f"T={T}, V={V}, blank={blank}: with an unlimited beam you must find every labeling with nonzero "
            f"probability ({len(exact)}), got {len(got)}")
        for k in exact:
            assert np.isclose(got[k], exact[k]), (
                f"labeling {k}: log p should be {exact[k]:.6f} (sum over all its paths), got {got[k]:.6f}. track, per "
                "prefix, the probability of paths ending in blank and ending in its last label separately: "
                "repeating the last label only extends the prefix from the ending-in-blank part")
        scores = [s for _, s in res]
        assert scores == sorted(scores, reverse=True), "results are sorted best first"


def _ref_beam(lp, beam_size, blank=0):
    T, V = lp.shape
    beams = {(): (0.0, -np.inf)}
    for t in range(T):
        nxt = {}
        for pre, (pb, pnb) in beams.items():
            tot = np.logaddexp(pb, pnb)
            ob, onb = nxt.get(pre, (-np.inf, -np.inf))
            nxt[pre] = (np.logaddexp(ob, tot + lp[t, blank]), onb)
            for c in range(V):
                if c == blank:
                    continue
                new = pre + (c,)
                ob, onb = nxt.get(new, (-np.inf, -np.inf))
                if pre and pre[-1] == c:
                    nxt[new] = (ob, np.logaddexp(onb, pb + lp[t, c]))
                    ob, onb = nxt[pre]
                    nxt[pre] = (ob, np.logaddexp(onb, pnb + lp[t, c]))
                else:
                    nxt[new] = (ob, np.logaddexp(onb, tot + lp[t, c]))
        beams = dict(sorted(nxt.items(), key=lambda kv: -np.logaddexp(*kv[1]))[:beam_size])
    return [(p, np.logaddexp(*v)) for p, v in sorted(beams.items(), key=lambda kv: -np.logaddexp(*kv[1]))
            if np.isfinite(np.logaddexp(*v))]


def test_stage4_pruned_beam(nb):
    g = np.random.default_rng(16)
    for beam in [1, 2, 3, 5]:
        for _ in range(3):
            lp = _logp(7, 4, int(g.integers(10_000)), peaky=0.5)
            want = _ref_beam(lp, beam)
            got = bounded(nb.prefix_beam_search, lp, beam_size=beam, what="prefix_beam_search")
            assert len(got) <= beam, f"beam_size={beam}: at most {beam} results, got {len(got)}"
            got = [(tuple(int(v) for v in p), s) for p, s in got]
            assert [p for p, _ in got] == [p for p, _ in want] and np.allclose([s for _, s in got],
                                                                                [s for _, s in want]), (
                f"beam_size={beam}: expected {[(p, round(s, 4)) for p, s in want]}, got "
                f"{[(p, round(s, 4)) for p, s in got]}. after EVERY frame keep the beam_size prefixes with the "
                "largest total (ending-in-blank + ending-in-label) probability")
    lp = torch.tensor(_logp(6, 3, 17))
    assert [tuple(p) for p, _ in nb.prefix_beam_search(lp, beam_size=3)] == [p for p, _ in _ref_beam(lp.numpy(), 3)], (
        "prefix_beam_search accepts a torch tensor too")


# ---------------------------------------------------------------- stage 5: scoring and the recognizer


def test_stage5_edit_distance_and_error_rate(nb):
    cases = [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("abc", "abc", 0), ([1, 2, 3], [1, 3], 1),
             ([1, 2], [2, 1], 2), ("flaw", "lawn", 2)]
    for a, b, want in cases:
        got = nb.edit_distance(a, b)
        assert got == want, f"edit_distance({a!r}, {b!r}) = {want}, got {got}"
    refs, hyps = [[1, 2, 3], [1]], [[1, 3], [2, 2]]
    got = nb.error_rate(refs, hyps)
    assert np.isclose(got, 0.75), (
        f"corpus error rate = (1 + 2) / (3 + 1) = 0.75, got {got}. if you got 1.17, you averaged per-utterance rates; "
        "WER/PER divide the TOTAL edits by the TOTAL reference length")
    for bad in [([[1]], [[1], [2]]), ([[]], [[1]])]:
        try:
            nb.error_rate(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"error_rate{bad} should raise ValueError (length mismatch / empty references)")


def test_stage5_recognizer_architecture(nb):
    torch.manual_seed(0)
    model = nb.CTCRecognizer(10, 7, hidden_size=12, num_layers=2)
    x = torch.randn(3, 9, 10)
    out = model(x)
    assert out.shape == (3, 9, 7), f"(B, T, n_classes) = (3, 9, 7), got {tuple(out.shape)}"
    assert torch.allclose(out.exp().sum(-1), torch.ones(3, 9), atol=1e-5), "outputs are log-probabilities per frame"
    ref = sum(p.numel() for p in nn.GRU(10, 12, num_layers=2, batch_first=True, bidirectional=True).parameters())
    ref += sum(p.numel() for p in nn.Linear(24, 7).parameters())
    n = sum(p.numel() for p in model.parameters())
    assert n == ref, (
        f"expected {ref} parameters (bidirectional 2-layer GRU with hidden_size 12, then Linear(24, 7)), got {n}. "
        "honour hidden_size and num_layers")
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(model(x2)[:, 0], out[:, 0]), (
        "the first frame's output must depend on the LAST frame: the GRU is bidirectional")


def _toy(n, seed):
    g = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        labs = [int(v) for v in g.integers(1, 4, int(g.integers(2, 4)))]
        fr = [np.zeros(4)] * int(g.integers(1, 3))
        for lab in labs:
            fr += [np.eye(4)[lab]] * int(g.integers(3, 6)) + [np.zeros(4)] * int(g.integers(1, 3))
        X.append(np.array(fr) + 0.3 * g.normal(size=(len(fr), 4)))
        Y.append(labs)
    T, U = max(map(len, X)), max(map(len, Y))
    xb, yb = np.zeros((n, T, 4)), np.zeros((n, U), dtype=np.int64)
    for i in range(n):
        xb[i, :len(X[i])], yb[i, :len(Y[i])] = X[i], Y[i]
    return (torch.tensor(xb, dtype=torch.float32), torch.tensor(yb), torch.tensor([len(x) for x in X]),
            torch.tensor([len(y) for y in Y]), Y)


def _train_toy(nb, seed, xb, yb, il, tl, xt, ilt, Yt):
    torch.manual_seed(seed)
    model = nb.CTCRecognizer(4, 4, hidden_size=16, num_layers=1)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    for _ in range(80):
        loss = nb.ctc_loss(model(xb), yb, il, tl)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lp = model(xt)
    return nb.error_rate(Yt, [nb.greedy_decode(lp[i, :ilt[i]]) for i in range(len(Yt))]), loss.item()


@pytest.mark.slow
def test_stage5_recognizer_learns_toy_task(nb):
    xb, yb, il, tl, _ = _toy(32, 0)
    xt, _, ilt, _, Yt = _toy(32, 1)
    runs = [_train_toy(nb, seed, xb, yb, il, tl, xt, ilt, Yt) for seed in (0, 1)]
    per, loss = min(runs)
    assert per < 0.2, (
        f"after 80 Adam steps on a toy task (labels held for 3-5 frames, noisy one-hot features) the held-out "
        f"error rate should be well under 0.2 in at least one of two runs, got {[round(r[0], 3) for r in runs]} "
        f"(final training losses {[round(r[1], 3) for r in runs]}). if the loss falls but decoding outputs nothing, "
        "check that greedy_decode drops only the blank")
