import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import heapq
    import json
    import re
    from collections import Counter, defaultdict

    import marimo as mo

    from mlfs import checks, lesson

    # GPT-2's pre-tokenization pattern, written for Python's `re` (no \p{L}): contractions, letters with an
    # optional leading space, digits, other symbols, then whitespace. every character matches some branch,
    # so the chunks always concatenate back to the text.
    GPT2_SPLIT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # tokenization · build (reference solution)

    work through the stages in order. each stage shows its part of the lesson, gives you
    code to write, and checks it live as you edit. you'll build the two tokenizers of
    **`mlfs.text`**: a character tokenizer, and a byte-level **byte-pair encoding** (BPE)
    tokenizer in the style of GPT-2. chapter 18 trains its GPT on your BPE tokens and chapter 27
    uses its special tokens, so it has to be exact: every string round-trips, merges are
    deterministic, and a megabyte of text trains in seconds.

    | stage | you build |
    |---|---|
    | 1 | `CharTokenizer`: train / encode / decode over the characters of a text |
    | 2 | `pair_counts`, `merge_pair`: the two primitives of BPE |
    | 3 | `pretokenize`: GPT-2-style chunks that merges never cross |
    | 4 | `BPETokenizer.__init__`, `BPETokenizer.train`: learn the merges |
    | 5 | `BPETokenizer.encode`, `decode`, `save`, `load`, with special tokens |

    you can also run the checks from a terminal: `pytest -k stage1`.
    """)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 1")
    return


@app.class_definition
class CharTokenizer:
    """characters <-> ids; the vocabulary is the sorted set of characters of the training text."""

    def __init__(self):
        """stage 1: an untrained tokenizer: empty `chars` list and `stoi` dict."""
        self.chars = []
        self.stoi = {}

    def train(self, text):
        """stage 1: chars = sorted unique characters of text, stoi = char -> index; returns self."""
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        return self

    @property
    def vocab_size(self):
        """the number of distinct characters."""
        return len(self.chars)

    def encode(self, text):
        """stage 1: list of int ids; ValueError naming the first character not in the vocabulary."""
        try:
            return [self.stoi[c] for c in text]
        except KeyError as err:
            raise ValueError(f"character {err.args[0]!r} is not in the vocabulary") from None

    def decode(self, ids):
        """stage 1: the string of the characters with these ids."""
        return "".join(self.chars[int(i)] for i in ids)


@app.cell(hide_code=True)
def _():
    checks.run(1, CharTokenizer=CharTokenizer)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 2")
    return


@app.function
def pair_counts(ids, counts=None, weight=1):
    """stage 2: add weight for every adjacent pair (ids[i], ids[i+1]) to counts (a new Counter if None); return it."""
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


@app.function
def merge_pair(ids, pair, new_id):
    """stage 2: a new list with every occurrence of pair replaced by new_id, scanning left to right."""
    out, i, a, b = [], 0, pair[0], pair[1]
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


@app.cell(hide_code=True)
def _():
    checks.run(2, pair_counts=pair_counts, merge_pair=merge_pair)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 3")
    return


@app.function
def pretokenize(text):
    """stage 3: the GPT-2-style chunks of text (GPT2_SPLIT.findall); they concatenate back to text."""
    return GPT2_SPLIT.findall(text)


@app.cell(hide_code=True)
def _():
    checks.run(3, pretokenize=pretokenize)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 4")
    return


@app.class_definition
class BPETokenizer:
    """byte-level BPE: base vocabulary = the 256 bytes, then one new id per learned merge, then special tokens.

    attributes: target_vocab_size (the requested size), special_tokens (tuple of str), merges (list of
    (id1, id2) pairs; merge r creates id 256 + r), vocab (id -> bytes), special_ids (str -> id).
    """

    def __init__(self, vocab_size=512, *, special_tokens=()):
        """stage 4: store target_vocab_size, special_tokens (a tuple), merges = [], then call self._build_vocab()."""
        special_tokens = tuple(special_tokens)
        if vocab_size < 256 + len(special_tokens):
            raise ValueError(f"vocab_size must be at least 256 + {len(special_tokens)} special tokens, got {vocab_size}")
        if len(set(special_tokens)) != len(special_tokens) or not all(special_tokens):
            raise ValueError(f"special tokens must be distinct, non-empty strings, got {special_tokens}")
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges = []
        self._build_vocab()

    def _build_vocab(self):
        """vocab, special_ids, the merge ranks and the encode cache, all from self.merges."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for r, (a, b) in enumerate(self.merges):
            self.vocab[256 + r] = self.vocab[a] + self.vocab[b]
        self.special_ids = {}
        for s in self.special_tokens:
            self.special_ids[s] = len(self.vocab)
            self.vocab[len(self.vocab)] = s.encode("utf-8")
        self._ranks = {pair: r for r, pair in enumerate(self.merges)}
        self._cache = {}
        longest_first = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(map(re.escape, longest_first)) + ")") if longest_first else None

    @property
    def vocab_size(self):
        """the size of the current vocabulary: 256 + len(merges) + len(special_tokens)."""
        return len(self.vocab)

    def _split_special(self, text):
        """[(piece, is_special)] in order: special tokens matched literally, longest first."""
        if self._special_re is None:
            return [(text, False)] if text else []
        return [(p, i % 2 == 1) for i, p in enumerate(self._special_re.split(text)) if p]

    def train(self, text):
        """stage 4: learn target_vocab_size - 256 - len(special_tokens) merges (most frequent pair, ties -> smallest); returns self."""
        n_merges = self.target_vocab_size - 256 - len(self.special_tokens)
        chunks = Counter()
        for piece, special in self._split_special(text):
            if not special:  # special tokens are never part of a merge
                chunks.update(pretokenize(piece))
        words = [list(c.encode("utf-8")) for c in chunks]
        freqs = list(chunks.values())
        stats, where = Counter(), defaultdict(set)
        for i, w in enumerate(words):
            pair_counts(w, stats, freqs[i])
            for pair in zip(w, w[1:]):
                where[pair].add(i)
        heap = [(-c, pair) for pair, c in stats.items()]
        heapq.heapify(heap)
        merges = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if stats.get(pair, 0) != -neg:  # stale entry: this pair's count has changed since it was pushed
                continue
            new_id = 256 + len(merges)
            merges.append(pair)
            changed = set()
            for i in where.pop(pair):
                w = words[i]
                new = merge_pair(w, pair, new_id)
                if len(new) == len(w):  # this word no longer contains the pair
                    continue
                for p in zip(w, w[1:]):
                    stats[p] -= freqs[i]
                    changed.add(p)
                for p in zip(new, new[1:]):
                    stats[p] += freqs[i]
                    where[p].add(i)
                    changed.add(p)
                words[i] = new
            for p in changed:
                if stats[p] > 0:
                    heapq.heappush(heap, (-stats[p], p))
                else:
                    del stats[p]
        self.merges = merges
        self._build_vocab()
        return self

    def encode_chunk(self, chunk):
        """stage 5: the ids of one chunk: its UTF-8 bytes, then repeatedly merge the adjacent pair of lowest merge rank."""
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self._ranks.get(p, float("inf")))
            if pair not in self._ranks:
                break
            ids = merge_pair(ids, pair, 256 + self._ranks[pair])
        return ids

    def encode(self, text):
        """stage 5: list of ids: special tokens first, then each chunk's bytes merged by rank; text must be valid Unicode."""
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as err:  # a lone surrogate (e.g. from a broken decode) has no UTF-8 bytes
            raise ValueError(f"text is not valid Unicode: lone surrogate {text[err.start]!r} at position {err.start}") from None
        out = []
        for piece, special in self._split_special(text):
            if special:
                out.append(self.special_ids[piece])
            else:
                for chunk in pretokenize(piece):
                    if chunk not in self._cache:  # a text repeats its words: encode each distinct chunk once
                        self._cache[chunk] = self.encode_chunk(chunk)
                    out.extend(self._cache[chunk])
        return out

    def decode(self, ids):
        """stage 5: the text: concatenate the ids' bytes and decode UTF-8 (invalid bytes -> U+FFFD); ValueError on an unknown id."""
        try:
            data = b"".join(self.vocab[int(i)] for i in ids)
        except KeyError as err:
            raise ValueError(f"id {err.args[0]} is not in the vocabulary (size {self.vocab_size})") from None
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        """stage 5: write the settings and the merges to path as JSON."""
        state = {"type": "BPETokenizer", "vocab_size": self.target_vocab_size,
                 "special_tokens": list(self.special_tokens), "merges": [list(p) for p in self.merges]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path):
        """stage 5: a BPETokenizer with exactly the settings and merges saved at path."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        tok = cls(state["vocab_size"], special_tokens=tuple(state["special_tokens"]))
        tok.merges = [(int(a), int(b)) for a, b in state["merges"]]
        tok._build_vocab()
        return tok


@app.cell(hide_code=True)
def _():
    checks.run(4, BPETokenizer=BPETokenizer, pretokenize=pretokenize, pair_counts=pair_counts, merge_pair=merge_pair)
    return


@app.cell(hide_code=True)
def _():
    lesson.section("idea 5")
    return


@app.cell(hide_code=True)
def _():
    checks.run(5, BPETokenizer=BPETokenizer, pretokenize=pretokenize, merge_pair=merge_pair)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## done?

    when every stage is green, open the lab: `marimo edit lab.py`. from now on
    `from mlfs.text import BPETokenizer, CharTokenizer` gives you these tokenizers. chapter 18's
    GPT reads its training text through them, and chapter 27 adds an `<|endoftext|>` special
    token to separate documents.
    """)
    return


if __name__ == "__main__":
    app.run()
