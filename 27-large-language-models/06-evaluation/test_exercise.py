"""Checks for 27 · 06 evaluation.

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. a bigram language model with a
known table and a small character tokenizer, both written here, make every log-likelihood computable by hand.
loglikelihood for its sum, its greedy flag, separate encoding of context and continuation, truncation to the
block size and errors; multiple-choice accuracy for using the question as context and for per-token
normalization; bootstrap intervals for coverage-like widths, alpha, the rng and pairing; n-gram contamination
on hand examples; the pairwise judge for cancelling position bias.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402


class _Tok:
    """character tokenizer over a fixed alphabet; optionally merges 'ab' into one token (id 9) like BPE would."""

    alphabet = " abcdxyz"

    def __init__(self, merge=False):
        self.merge = merge

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            if self.merge and text[i:i + 2] == "ab":
                out.append(9)
                i += 2
            else:
                out.append(self.alphabet.index(text[i]))
                i += 1
        return out

    def decode(self, ids):
        return "".join("ab" if i == 9 else self.alphabet[i] for i in ids)


class _Bigram(nn.Module):
    def __init__(self, table, block_size=None):
        super().__init__()
        self.table = nn.Parameter(torch.as_tensor(table, dtype=torch.float32))
        self.block_size = block_size
        self.max_len = 0
        self.modes, self.grads = [], []

    def forward(self, idx, targets=None, *, cache=None):
        self.max_len = max(self.max_len, idx.shape[1])
        self.modes.append(self.training)
        self.grads.append(torch.is_grad_enabled())
        if self.block_size is not None:
            assert idx.shape[1] <= self.block_size, f"got {idx.shape[1]} tokens, block_size is {self.block_size}"
        return self.table[idx], None


def _table(seed=0, V=10):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(V, V, generator=g) * 2


def _ll(table, ids_ctx, ids_cont):
    lp = F.log_softmax(torch.as_tensor(table, dtype=torch.float32), -1)
    seq = ids_ctx + ids_cont
    return sum(lp[seq[i - 1], seq[i]].item() for i in range(len(ids_ctx), len(seq)))


# ---------------------------------------------------------------- stage 1: loglikelihood


def test_stage1_hand_value_and_greedy(nb):
    tab = _table(V=len(_Tok.alphabet))
    m, tok = _Bigram(tab), _Tok()
    ll, greedy = nb.loglikelihood(m, tok, "xa", "bcd")
    want = _ll(tab, tok.encode("xa"), tok.encode("bcd"))
    assert isinstance(ll, float) and math.isclose(ll, want, rel_tol=1e-5), (
        f"sum over the continuation's tokens of log p(token | everything before it): {want:.4f}; got {ll}. the first "
        "continuation token is predicted from the last context position")
    assert isinstance(greedy, bool)
    lp = F.log_softmax(tab, -1)
    path = [tok.encode("x")[0]]
    for _ in range(3):
        path.append(int(lp[path[-1]].argmax()))
    g_text = tok.decode(path[1:])
    ll_g, is_g = nb.loglikelihood(m, tok, "x", g_text)
    assert is_g is True, f"'{g_text}' is the greedy continuation of 'x': is_greedy must be True"
    other = "".join(c for c in "abcd" if c != g_text[0])[0] + g_text[1:]
    assert nb.loglikelihood(m, tok, "x", other)[1] is False, "a continuation that isn't the argmax at every step"


def test_stage1_separate_encoding_and_truncation(nb):
    tab = _table(1)
    tok = _Tok(merge=True)
    ll, _ = nb.loglikelihood(_Bigram(tab), tok, "xa", "b")
    want = _ll(tab, tok.encode("xa"), tok.encode("b"))
    assert math.isclose(ll, want, rel_tol=1e-5), (
        "encode the context and the continuation separately and concatenate the ids: encoding 'xab' in one go would "
        "merge 'ab' into one token and the continuation would vanish")
    small = _Bigram(tab, block_size=4)
    ll2, _ = nb.loglikelihood(small, _Tok(), "abcdxyzabcd", "xy")
    assert small.max_len <= 4, "never feed the model more than block_size tokens: drop the oldest context tokens"
    assert math.isclose(ll2, _ll(tab, _Tok().encode("abcdxyzabcd"), _Tok().encode("xy")), rel_tol=1e-5)
    with pytest.raises(ValueError):
        nb.loglikelihood(_Bigram(tab), _Tok(), "", "ab")
    with pytest.raises(ValueError):
        nb.loglikelihood(_Bigram(tab), _Tok(), "ab", "")
    m = _Bigram(tab)
    m.train()
    nb.loglikelihood(m, _Tok(), "ab", "c", device="cpu")
    assert m.training, "restore the model's train/eval mode"
    assert m.modes == [False], "the forward pass runs in eval mode (no dropout)"
    assert m.grads == [False], "evaluation runs under torch.no_grad(): no graph is built"
    with pytest.raises(ValueError):
        nb.loglikelihood(_Bigram(tab, block_size=4), _Tok(), "ab", "abcdxy")  # 6 continuation tokens > block_size 4


# ---------------------------------------------------------------- stage 2: multiple choice


def test_stage2_accuracy_and_normalization(nb):
    V = len(_Tok.alphabet)
    tab = torch.full((V, V), -5.0)
    a, b, c, d, x, y, sp = (_Tok.alphabet.index(ch) for ch in "abcdxy ")
    tab[x, sp] = 5.0     # after x: a space
    tab[sp, a] = 2.8     # " a..." almost as likely
    tab[sp, b] = 3.0     # " b" most likely first letter
    tab[a, c] = 6.0      # a -> c -> d -> d ... nearly certain
    tab[c, d] = 6.0
    tab[d, d] = 6.0
    m, tok = _Bigram(tab), _Tok()
    item = {"question": "x", "choices": [" b", " acdd"], "answer": 1}
    raw = [nb.loglikelihood(m, tok, "x", ch)[0] for ch in item["choices"]]
    assert raw[0] > raw[1], "(test setup) the short choice wins on total log-likelihood"
    assert raw[1] / 5 > raw[0] / 2, "(test setup) the long choice wins per token"
    assert nb.evaluate_multiple_choice(m, tok, [item]) == 1.0, (
        "normalize=True (default): compare log-likelihood per continuation token, so ' acdd' (5 tokens) beats ' b'")
    assert nb.evaluate_multiple_choice(m, tok, [item], normalize=False) == 0.0, (
        "normalize=False: compare total log-likelihood, which favours the short choice")
    items = [item, {"question": "x", "choices": [" acdd", " b"], "answer": 0},
             {"question": "x", "choices": [" b", " acdd"], "answer": 0}]
    acc = nb.evaluate_multiple_choice(m, tok, items)
    assert isinstance(acc, float) and math.isclose(acc, 2 / 3), f"accuracy over items: 2 of 3 right; got {acc}"
    # an oracle check on random items: varied questions, choices of different lengths, a tokenizer that merges 'ab'
    g = np.random.default_rng(0)
    tokm = _Tok(merge=True)
    mm = _Bigram(_table(5))
    rnd, want_norm, want_raw = [], [], []
    for _ in range(25):
        q = "".join(g.choice(list("abcdxyz"), size=int(g.integers(1, 5))))
        chs = ["".join(g.choice(list(" abcd"), size=int(g.integers(1, 7)))) for _ in range(4)]
        lls = [nb.loglikelihood(mm, tokm, q, ch)[0] for ch in chs]
        ans = int(np.argmax([l / len(tokm.encode(ch)) for l, ch in zip(lls, chs)]))  # so the right scorer gets 100%
        want_norm.append(int(np.argmax([l / len(tokm.encode(ch)) for l, ch in zip(lls, chs)])) == ans)
        want_raw.append(int(np.argmax(lls)) == ans)
        rnd.append({"question": q, "choices": chs, "answer": ans})
    assert math.isclose(nb.evaluate_multiple_choice(mm, tokm, rnd), np.mean(want_norm)), (
        "normalize=True: each choice's log-likelihood divided by its number of TOKENS (tokenizer.encode(choice)), with "
        "the item's question as the context")
    assert math.isclose(nb.evaluate_multiple_choice(mm, tokm, rnd, normalize=False), np.mean(want_raw))
    # per TOKEN, not per character: ' abab' is 3 tokens (5 characters) with this merging tokenizer, ' cd' 3 tokens
    probs = torch.full((10, 10), 1e-3)
    sp_, a_, c_, d_, x_ = 0, 1, 3, 4, 5
    probs[x_, sp_] = 1.0
    probs[sp_, 9], probs[sp_, c_] = 0.4, 0.45           # P(ab | ' '), P(c | ' ')
    probs[9, 9], probs[c_, d_] = 0.338, 0.4957          # P(ab | ab), P(d | c)
    tm = _Bigram((probs / probs.sum(1, keepdim=True)).log())
    tokm = _Tok(merge=True)
    lab, lcd = (nb.loglikelihood(tm, tokm, "x", ch)[0] for ch in (" abab", " cd"))
    assert lcd > lab and lab / 5 > lcd / 3, "(test setup)"
    assert nb.evaluate_multiple_choice(tm, tokm, [{"question": "x", "choices": [" abab", " cd"], "answer": 1}]) == 1.0, (
        "normalize divides by the number of TOKENS (equal here, so ' cd' wins on log-likelihood), not by characters "
        "(which would favour the 5-character ' abab')")
    q_item = {"question": "y", "choices": [" b", " acdd"], "answer": 1}
    tab2 = tab.clone()
    tab2[y, sp] = 5.0
    tab2[x, sp] = -5.0
    m2 = _Bigram(tab2)
    assert nb.evaluate_multiple_choice(m2, tok, [q_item]) == 1.0, "the item's question is the context"


# ---------------------------------------------------------------- stage 3: intervals


def test_stage3_accuracy_ci(nb):
    rng = np.random.default_rng(0)
    correct = rng.random(400) < 0.7
    acc, lo, hi = nb.accuracy_ci(correct, rng=0)
    assert math.isclose(acc, correct.mean()), "the point estimate is the plain accuracy"
    assert lo < acc < hi
    se = math.sqrt(acc * (1 - acc) / 400)
    assert abs((hi - lo) - 2 * 1.96 * se) < 0.012, (
        f"a 95% interval for 400 items at accuracy {acc:.2f} is about +-1.96 sqrt(p(1-p)/n) = +-{1.96 * se:.3f} wide; got "
        f"[{lo:.3f}, {hi:.3f}]")
    assert nb.accuracy_ci(correct, rng=0) == (acc, lo, hi), "the same rng seed gives the same interval"
    _, lo50, hi50 = nb.accuracy_ci(correct, alpha=0.5, rng=0)
    assert hi50 - lo50 < 0.5 * (hi - lo), "alpha=0.5 is a 50% interval: much narrower"
    _, lo_s, hi_s = nb.accuracy_ci(correct[:40], rng=0)
    p40 = correct[:40].mean()
    want40 = 2 * 1.96 * math.sqrt(max(p40 * (1 - p40), 0.05) / 40)
    assert 0.7 * want40 < hi_s - lo_s < 1.3 * want40, (
        f"40 items: the interval is about 2 x 1.96 sqrt(p(1-p)/40) = {want40:.3f} wide; got {hi_s - lo_s:.3f}")
    _, l1, h1 = nb.accuracy_ci(correct, n_boot=20, rng=0)
    _, l2, h2 = nb.accuracy_ci(correct, n_boot=20, rng=1)
    assert (l1, h1) != (l2, h2), "rng is used"
    _, l3, h3 = nb.accuracy_ci(correct, n_boot=1, rng=0)
    assert l3 == h3, "with n_boot=1 there is a single bootstrap replicate, so lo == hi: is n_boot used?"


def test_stage3_paired(nb):
    # exactly 195 items both right, 12 only a right, 3 only b right, 90 both wrong (in a shuffled order)
    a = np.array([1] * 195 + [1] * 12 + [0] * 3 + [0] * 90, dtype=bool)
    b = np.array([1] * 195 + [0] * 12 + [1] * 3 + [0] * 90, dtype=bool)
    order = np.random.default_rng(1).permutation(300)
    a, b = a[order], b[order]
    d, lo, hi = nb.paired_accuracy_diff(a, b, rng=0)
    assert math.isclose(d, 0.03), f"the point estimate is acc_a - acc_b = 0.03; got {d}"
    assert lo > 0, (
        f"a wins 12 of the 15 items where they disagree: the 95% interval for acc_a - acc_b excludes 0 (theory: "
        f"0.030 +- 0.025); got [{lo:.3f}, {hi:.3f}]. resampling the SAME items for both keeps the pairing; resampling "
        "them independently gives ~+-0.07")
    assert abs((lo + hi) / 2 - d) < 0.01, f"the interval is centred on acc_a - acc_b (not b - a); got [{lo:.3f}, {hi:.3f}]"
    assert hi - lo < 0.06
    _, lo50, hi50 = nb.paired_accuracy_diff(a, b, alpha=0.5, rng=0)
    assert hi50 - lo50 < 0.6 * (hi - lo), "alpha=0.5 gives a 50% interval: narrower"
    _, l1, h1 = nb.paired_accuracy_diff(a, b, n_boot=1, rng=0)
    assert l1 == h1, "n_boot=1: a single bootstrap replicate, so lo == hi. is n_boot used?"
    assert nb.paired_accuracy_diff(a, b, n_boot=50, rng=0) != nb.paired_accuracy_diff(a, b, n_boot=50, rng=1), (
        "rng is used: different seeds, different replicates")
    assert nb.paired_accuracy_diff(a, b, rng=3) == nb.paired_accuracy_diff(a, b, rng=3), "the same seed, the same interval"
    assert nb.paired_accuracy_diff(a, a, rng=0) == (0.0, 0.0, 0.0), "a model against itself: exactly 0"
    with pytest.raises(ValueError):
        nb.paired_accuracy_diff(a, b[:-1])


# ---------------------------------------------------------------- stage 4: contamination


def test_stage4_ngrams_and_flags(nb):
    got = nb.ngrams("To be, or NOT to be!", n=3)
    want = {("to", "be", "or"), ("be", "or", "not"), ("or", "not", "to"), ("not", "to", "be")}
    assert got == want, f"lowercase word 3-grams with punctuation stripped: want {sorted(want)}, got {sorted(got)}"
    assert nb.ngrams("only two", n=3) == set(), "a text shorter than n has no n-grams"
    assert ("o'er", "the", "hills") in nb.ngrams("O'er the hills", n=3), "apostrophes stay inside words"
    assert len(nb.ngrams(" ".join(f"w{i}" for i in range(20)))) == 8, "default n is 13: 20 words give 8 13-grams"
    train = ["the quick brown fox jumps over the lazy dog", "a completely different sentence here"]
    test = ["THE LAZY DOG sleeps", "the quick red fox", "nothing in common at all", "jumps over the"]
    flags = nb.contamination_flags(test, train, n=3)
    assert isinstance(flags, np.ndarray) and flags.dtype == bool
    assert flags.tolist() == [True, False, False, True], (
        f"'the lazy dog' (matched case-insensitively) and 'jumps over the' are 3-grams of the training text; 'the quick "
        f"red' isn't. want [True, False, False, True], got {flags.tolist()}")
    assert nb.contamination_flags(test, train, n=2).tolist() == [True, True, False, True], (
        "with n=2: 'the lazy' and 'the quick' also match: n is used")


# ---------------------------------------------------------------- stage 5: LLM-as-judge


def test_stage5_position_bias(nb):
    calls = []

    def biased(prompt, first, second):
        calls.append((first, second))
        return 0.7  # always prefers whatever is shown first

    p = nb.pairwise_judge(biased, "q", "good", "bad")
    assert math.isclose(p, 0.5), (
        f"a judge that always prefers the first answer carries no information; averaging over both orders gives 0.5, "
        f"got {p}")
    assert calls == [("good", "bad"), ("bad", "good")], f"call the judge once in each order; calls were {calls}"

    def decent(prompt, first, second):
        base = 0.8 if first == "good" else 0.2
        return min(1.0, base + 0.1)  # plus a position bias of 0.1

    assert math.isclose(nb.pairwise_judge(decent, "q", "good", "bad"), 0.8), (
        "(0.9 + 1 - 0.3) / 2 = 0.8: the 0.1 bias cancels")
    assert math.isclose(nb.pairwise_judge(decent, "q", "bad", "good"), 0.2), "and swapping the arguments gives 1 - 0.8"
