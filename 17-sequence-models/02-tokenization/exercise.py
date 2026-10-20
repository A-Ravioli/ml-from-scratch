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
    # tokenization · build

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
        raise NotImplementedError("stage1: implement CharTokenizer.__init__")

    def train(self, text):
        """stage 1: chars = sorted unique characters of text, stoi = char -> index; returns self."""
        raise NotImplementedError("stage1: implement CharTokenizer.train")

    @property
    def vocab_size(self):
        """the number of distinct characters."""
        return len(self.chars)

    def encode(self, text):
        """stage 1: list of int ids; ValueError naming the first character not in the vocabulary."""
        raise NotImplementedError("stage1: implement CharTokenizer.encode")

    def decode(self, ids):
        """stage 1: the string of the characters with these ids."""
        raise NotImplementedError("stage1: implement CharTokenizer.decode")


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
    raise NotImplementedError("stage2: implement pair_counts")


@app.function
def merge_pair(ids, pair, new_id):
    """stage 2: a new list with every occurrence of pair replaced by new_id, scanning left to right."""
    raise NotImplementedError("stage2: implement merge_pair")


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
    raise NotImplementedError("stage3: implement pretokenize")


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
        raise NotImplementedError("stage4: implement BPETokenizer.__init__")

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
        raise NotImplementedError("stage4: implement BPETokenizer.train")

    def encode_chunk(self, chunk):
        """stage 5: the ids of one chunk: its UTF-8 bytes, then repeatedly merge the adjacent pair of lowest merge rank."""
        raise NotImplementedError("stage5: implement BPETokenizer.encode_chunk")

    def encode(self, text):
        """stage 5: list of ids: special tokens first, then each chunk's bytes merged by rank; text must be valid Unicode."""
        raise NotImplementedError("stage5: implement BPETokenizer.encode")

    def decode(self, ids):
        """stage 5: the text: concatenate the ids' bytes and decode UTF-8 (invalid bytes -> U+FFFD); ValueError on an unknown id."""
        raise NotImplementedError("stage5: implement BPETokenizer.decode")

    def save(self, path):
        """stage 5: write the settings and the merges to path as JSON."""
        raise NotImplementedError("stage5: implement BPETokenizer.save")

    @classmethod
    def load(cls, path):
        """stage 5: a BPETokenizer with exactly the settings and merges saved at path."""
        raise NotImplementedError("stage5: implement BPETokenizer.load")


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
