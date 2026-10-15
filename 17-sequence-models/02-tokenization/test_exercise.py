"""Checks for exercise.py (tokenization).

Every test takes `nb`, the notebook under test, and is named `test_stageN_*`. The BPE trainer is
checked against a deliberately simple reference trainer written below (GPT-2-style: byte-level,
pre-tokenized, recount every pair after every merge), merge for merge, including the tie rule; the
hand examples pin down overlapping pairs and ties. Round trips use random Unicode (emoji, CJK,
combining marks, control characters). Training and encoding run under `mlfs.testing.bounded`.
"""
import json
import os
import re
import tempfile
from collections import Counter

import numpy as np
import pytest

from mlfs.testing import bounded

SPLIT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

ENGLISH = ("the quick brown fox jumps over the lazy dog. the dog didn't care; the fox ran on, "
           "and on, and on. 12 foxes, 3 dogs! isn't that the way it's always been? ") * 6

ALPHABET = list("abcde  .,'\n") + ["é", "ü", "日", "本", "🙂", "́", "1", "2"]


def _random_text(seed, n=300, alphabet=ALPHABET):
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(alphabet, size=n))


def _random_unicode(seed, n=200):
    rng = np.random.default_rng(seed)
    pools = [range(0x20, 0x7F), range(0xA0, 0x250), range(0x0300, 0x0370), range(0x4E00, 0x4F00),
             range(0x1F600, 0x1F650), [0x0, 0x9, 0xA, 0xD, 0x200D, 0xFEFF, 0x10FFFF], range(0x0590, 0x0600)]
    out = []
    for _ in range(n):
        pool = pools[int(rng.integers(len(pools)))]
        out.append(chr(int(rng.choice(list(pool)))))
    return "".join(out)


def reference_train(text, n_merges, specials=()):
    """the plainest correct trainer: recount every pair of every chunk after every merge."""
    pieces = [text] if not specials else [p for p in re.split("(" + "|".join(map(re.escape, sorted(
        specials, key=len, reverse=True))) + ")", text) if p and p not in specials]
    chunks = Counter(c for piece in pieces for c in SPLIT.findall(piece))
    words = {c: list(c.encode("utf-8")) for c in chunks}
    merges = []
    for r in range(n_merges):
        counts = Counter()
        for c, w in words.items():
            for p in zip(w, w[1:]):
                counts[p] += chunks[c]
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        for c, w in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(256 + r)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[c] = out
    return merges


def reference_encode(text, merges):
    """apply the merges in order to each chunk's bytes (no special tokens)."""
    ids = []
    for c in SPLIT.findall(text):
        w = list(c.encode("utf-8"))
        for r, pair in enumerate(merges):
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == tuple(pair):
                    out.append(256 + r)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            w = out
        ids += w
    return ids


def _train(nb, text, vocab_size, specials=()):
    tok = nb.BPETokenizer(vocab_size, special_tokens=specials)
    out = bounded(tok.train, text, budget=6_000_000, what="BPETokenizer.train",
                  hint="(stop when you have enough merges, or when no pair is left)")
    assert out is tok, "train returns self"
    return tok


def _merges(tok):
    return [tuple(int(x) for x in p) for p in tok.merges]


# ---------------------------------------------------------------- stage 1: CharTokenizer


def test_stage1_char_tokenizer(nb):
    tok = nb.CharTokenizer()
    assert tok.train("hello world") is tok, "train returns self"
    assert tok.vocab_size == 8, f"'hello world' has 8 distinct characters (space included); got {tok.vocab_size}"
    ids = tok.encode("hello")
    assert ids == [3, 2, 4, 4, 5], (
        f"the vocabulary is the SORTED set of characters (' ', d, e, h, l, o, r, w), so 'hello' -> [3, 2, 4, 4, 5]; "
        f"got {ids}")
    assert isinstance(ids, list) and all(type(i) is int for i in ids), "encode returns a list of Python ints"
    assert tok.decode(ids) == "hello" and tok.decode([]) == "" and tok.encode("") == []
    with pytest.raises(ValueError):
        tok.encode("help")  # 'p' was never seen


def test_stage1_round_trip_unicode(nb):
    text = _random_unicode(0, 500)
    tok = nb.CharTokenizer().train(text)
    assert tok.decode(tok.encode(text)) == text, "decode(encode(text)) == text for the training text"
    assert tok.vocab_size == len(set(text))
    assert tok.decode(np.array(tok.encode(text[:20]))) == text[:20], "decode accepts any iterable of ints"


# ---------------------------------------------------------------- stage 2: pairs and merges


def test_stage2_pair_counts(nb):
    got = nb.pair_counts([1, 2, 3, 1, 2])
    assert dict(got) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}, f"adjacent pairs of [1, 2, 3, 1, 2]; got {dict(got)}"
    got = nb.pair_counts([7, 7, 7])
    assert dict(got) == {(7, 7): 2}, (
        f"[7, 7, 7] has the pair (7, 7) at positions 0 and 1: count overlapping occurrences, 2; got {dict(got)}")
    c = Counter({(1, 2): 10})
    out = nb.pair_counts([1, 2], c, 5)
    assert out is c and c[(1, 2)] == 15, "with counts given, add weight to it in place and return it"
    assert dict(nb.pair_counts([4])) == {} and dict(nb.pair_counts([])) == {}


def test_stage2_merge_pair(nb):
    cases = [([1, 2, 3, 1, 2], (1, 2), 9, [9, 3, 9]), ([7, 7, 7], (7, 7), 9, [9, 7]),
             ([7, 7, 7, 7], (7, 7), 9, [9, 9]), ([1, 2], (2, 1), 9, [1, 2]), ([], (1, 2), 9, []),
             ([1], (1, 2), 9, [1]), ([2, 1, 2, 1, 2], (1, 2), 9, [2, 9, 9])]
    for ids, pair, new, want in cases:
        orig = list(ids)
        got = bounded(nb.merge_pair, ids, pair, new, budget=200_000, what="merge_pair",
                      hint="(every pass through the loop must move the index forward)")
        assert list(got) == want, (
            f"merge_pair({ids}, {pair}, {new}) should be {want}, got {got}. scan left to right; after a merge, "
            f"skip both tokens (so [7, 7, 7] becomes [9, 7], never [9, 9])")
        assert ids == orig, "merge_pair returns a new list and leaves its input alone"


# ---------------------------------------------------------------- stage 3: pretokenize


def test_stage3_examples(nb):
    cases = {"Hello world": ["Hello", " world"], "don't stop": ["don", "'t", " stop"],
             "  two  spaces ": [" ", " two", " ", " spaces", " "], "abc123 x_y": ["abc", "123", " x", "_", "y"],
             "hi!!\n\nOK": ["hi", "!!", "\n", "\n", "OK"], "x=3.14;": ["x", "=", "3", ".", "14", ";"],
             "naïve café 你好 🙂🙂": ["naïve", " café", " 你好", " 🙂🙂"]}
    for text, want in cases.items():
        got = list(nb.pretokenize(text))
        assert got == want, (
            f"pretokenize({text!r}) should be {want}, got {got}. use the GPT2_SPLIT pattern from the setup cell: "
            f"letters, digits and other symbols each form chunks, a single leading space sticks to the next chunk")


def test_stage3_covers_everything(nb):
    for seed in range(5):
        text = _random_unicode(seed, 300)
        chunks = nb.pretokenize(text)
        assert "".join(chunks) == text, "the chunks must concatenate back to the text exactly (nothing dropped)"
        assert list(chunks) == SPLIT.findall(text)


# ---------------------------------------------------------------- stage 4: training


def test_stage4_init(nb):
    tok = nb.BPETokenizer(300, special_tokens=["<|endoftext|>"])
    assert tok.target_vocab_size == 300 and tok.special_tokens == ("<|endoftext|>",) and list(tok.merges) == []
    assert len(tok.vocab) == 257 and tok.vocab[65] == b"A" and tok.vocab[256] == b"<|endoftext|>", (
        "before training the vocabulary is the 256 single bytes, then the special tokens")
    assert tok.vocab_size == 257
    with pytest.raises(ValueError):
        nb.BPETokenizer(256, special_tokens=["<pad>"])
    d = nb.BPETokenizer()
    assert d.target_vocab_size == 512 and d.special_tokens == ()


def test_stage4_hand_examples(nb):
    tok = _train(nb, "aaabdaaabac", 259)
    assert _merges(tok) == [(97, 97), (97, 98), (256, 257)], (
        f"'aaabdaaabac': (a, a) occurs 4 times (overlapping) -> 256; then (256, a) and (a, b) tie at 2, and the "
        f"smaller pair (97, 98) wins -> 257; then (256, 257) -> 258. got {_merges(tok)}")
    assert tok.vocab[258] == b"aaab" and tok.vocab_size == 259
    tok = _train(nb, "abcd", 259)
    assert _merges(tok) == [(97, 98), (99, 100), (256, 257)], (
        f"'abcd': every pair occurs once, so ties go to the smallest pair: (97, 98) first, then (99, 100) (99 < 256), "
        f"then (256, 257); got {_merges(tok)}")
    tok = _train(nb, "aaaa", 300)
    assert _merges(tok) == [(97, 97), (256, 256)], (
        f"'aaaa' -> [256, 256] -> [257]; then no pairs are left, so training stops early; got {_merges(tok)}")
    assert tok.vocab_size == 258, "vocab_size counts what was actually learned"


def test_stage4_merges_stay_inside_chunks(nb):
    tok = _train(nb, "ab ab ab", 262)
    assert _merges(tok)[:2] == [(97, 98), (32, 256)], (
        f"chunks are 'ab', ' ab', ' ab': (a, b) occurs 3 times -> 256, then (' ', 256) twice -> 257; got {_merges(tok)}")
    assert all(not (tok.vocab[i].endswith(b"b ")) for i in tok.vocab), "no merge crosses a chunk boundary"


def test_stage4_matches_reference_trainer(nb):
    for seed in range(4):
        text = _random_text(seed, 400)
        want = reference_train(text, 60)
        got = _merges(_train(nb, text, 256 + 60))
        if got != want:
            i = next(i for i, (a, b) in enumerate(zip(got + [None] * 61, want + [None])) if a != b)
            lo = max(0, i - 2)
            raise AssertionError(
                f"seed {seed}: your merges differ from the reference trainer's from merge {i} on: merges {lo}..{i + 2} "
                f"are {got[lo:i + 3]}, the reference has {want[lo:i + 3]}. count pairs inside each pretokenized chunk, "
                f"weighted by how often the chunk occurs; pick the highest count, ties to the smallest pair")
    want = reference_train(ENGLISH, 80)
    assert _merges(_train(nb, ENGLISH, 336)) == want, "on English text too"


def test_stage4_vocab_and_specials(nb):
    text = "hello<|eot|>hello world<|eot|>"
    tok = _train(nb, text, 256 + 10 + 1, specials=("<|eot|>",))
    assert _merges(tok) == reference_train(text, 10, specials=("<|eot|>",)), (
        "special tokens are cut out of the training text before pretokenizing: they never take part in a merge")
    for r, (a, b) in enumerate(tok.merges):
        assert tok.vocab[256 + r] == tok.vocab[a] + tok.vocab[b], "vocab[256 + r] = the bytes of both halves"
    sid = 256 + len(tok.merges)
    assert tok.vocab[sid] == b"<|eot|>", "special tokens get the ids right after the merges"
    a, b = _train(nb, ENGLISH, 330), _train(nb, ENGLISH, 330)
    assert _merges(a) == _merges(b), "training is deterministic"


def _enc(tok, text):
    return bounded(tok.encode, text, budget=5_000_000, what="encode",
                   hint="(in encode_chunk, stop as soon as no adjacent pair is a known merge)")


# ---------------------------------------------------------------- stage 5: encode / decode / save / load


def test_stage5_encode_matches_merges(nb):
    tok = _train(nb, ENGLISH, 336)
    for text in [ENGLISH, "the fox didn't jump: 42 dogs?", _random_text(9, 200)]:
        got = _enc(tok, text)
        want = reference_encode(text, _merges(tok))
        assert list(got) == want, (
            f"encoding {text[:30]!r}... should give the same ids as applying the merges in order to each chunk; "
            f"got {list(got)[:12]}, want {want[:12]}. merge the adjacent pair with the LOWEST rank first")
    assert len(_enc(tok, ENGLISH)) < len(ENGLISH.encode("utf-8")) / 3, "on its training text BPE compresses a lot"
    assert list(_enc(nb.BPETokenizer(300), "hé")) == [104, 195, 169], "an untrained tokenizer encodes raw UTF-8 bytes"
    manual = nb.BPETokenizer(258)
    manual.merges = [(99, 100), (98, 99)]  # rank 0: "cd", rank 1: "bc"
    manual._build_vocab()
    got = list(bounded(manual.encode_chunk, "bcd", budget=200_000, what="encode_chunk",
                       hint="(stop when no adjacent pair is a known merge)"))
    assert got == [98, 256], (
        f"'bcd' has two applicable merges: (98, 99) on the left and (99, 100) on the right. merge the LOWEST RANK "
        f"first (the one learned earliest), (99, 100) -> 256, giving [98, 256], not the leftmost; got {got}")


def test_stage5_retraining_resets_encoding(nb):
    tok = _train(nb, ENGLISH, 300)
    text = "the dog and the fox"
    _enc(tok, text)  # fills any cache
    other = "zzyzx qqq zzyzx qqq zzyzx " * 20
    _train_again = bounded(tok.train, other, budget=6_000_000, what="BPETokenizer.train")
    got = list(_enc(tok, text))
    want = reference_encode(text, _merges(tok))
    assert got == want, (
        f"after training again on different text, encode must use the NEW merges; got {got}, want {want}. "
        f"if you cache chunk encodings, the cache must be emptied whenever the merges change (_build_vocab does this)")


def test_stage5_round_trip(nb):
    tok = _train(nb, ENGLISH + _random_text(1, 300), 400)
    for seed in range(6):
        text = _random_unicode(seed, 300)
        ids = _enc(tok, text)
        assert all(0 <= i < tok.vocab_size for i in ids)
        assert tok.decode(ids) == text, (
            f"decode(encode(text)) must give back the text exactly, for any Unicode (emoji, CJK, combining marks, "
            f"control characters); failed on {text[:40]!r}")
    assert tok.decode([]) == "" and _enc(tok, "") == []
    assert tok.decode([0xE6]) == "�", "a lone partial UTF-8 byte decodes to U+FFFD, it doesn't raise"
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


def test_stage5_special_tokens(nb):
    sp = ("<|endoftext|>", "<|end|>")
    tok = _train(nb, ENGLISH, 330, specials=sp)
    eot, end = tok.vocab_size - 2, tok.vocab_size - 1
    ids = _enc(tok, "the dog<|endoftext|><|end|>the fox<|endoftext|>")
    assert ids.count(eot) == 2 and ids.count(end) == 1, (
        f"special tokens are matched literally, longest first (<|endoftext|> is not <|end|> + 'oftext|>'); "
        f"ids {eot} and {end} expected; got {ids}")
    assert ids[-1] == eot and tok.decode(ids) == "the dog<|endoftext|><|end|>the fox<|endoftext|>"
    assert _enc(tok, "the dog") == ids[:ids.index(eot)], "the text between special tokens is encoded normally"
    assert eot not in _enc(tok, "<|endoftex"), "a partial special token is just text"
    nested = nb.BPETokenizer(260, special_tokens=("<x>", "<x><y>"))
    ids = _enc(nested, "a<x><y>b<x>")
    assert ids == [97, 257, 98, 256], (
        f"when one special token is a prefix of another, match the longest first: '<x><y>' is one token (257), not "
        f"'<x>' followed by text; got {ids}")


def test_stage5_save_load(nb):
    tok = _train(nb, ENGLISH, 350, specials=("<|endoftext|>",))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tok.json")
        tok.save(path)
        with open(path, encoding="utf-8") as f:
            json.load(f)  # valid JSON
        tok2 = nb.BPETokenizer.load(path)
    assert isinstance(tok2, nb.BPETokenizer) and _merges(tok2) == _merges(tok), "load restores the merges exactly"
    assert tok2.special_tokens == tok.special_tokens and tok2.vocab_size == tok.vocab_size
    assert tok2.target_vocab_size == tok.target_vocab_size
    text = "the dog<|endoftext|>" + _random_unicode(3, 100)
    assert _enc(tok2, text) == _enc(tok, text) and tok2.decode(_enc(tok, text)) == text, (
        "a loaded tokenizer encodes and decodes exactly like the original")
