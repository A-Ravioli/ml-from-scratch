# tokenization · hints

open a hint only after you've been stuck for a while. each one gives away a little more.

## stage 1

<details><summary>hint 1: a nudge</summary>

`train`: `self.chars = sorted(set(text))` and `self.stoi = {c: i for i, c in enumerate(self.chars)}`. `encode` is a list comprehension over `stoi`; catch the `KeyError` of an unknown character and raise a `ValueError` that names it. `decode` indexes `chars` with `int(i)` (so NumPy ints and tensors work too) and joins.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def encode(self, text):
    try:
        return [self.stoi[c] for c in text]
    except KeyError as err:
        raise ValueError(f"character {err.args[0]!r} is not in the vocabulary") from None
```

</details>

## stage 2

<details><summary>hint 1: a nudge</summary>

`zip(ids, ids[1:])` yields every adjacent pair, overlapping ones included. for `merge_pair`, use an index `i` and a `while` loop: if `ids[i], ids[i+1]` is the pair, append `new_id` and jump by 2, otherwise append `ids[i]` and step by 1. check `i + 1 < len(ids)` before looking at `ids[i + 1]`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
out, i = [], 0
while i < len(ids):
    if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
        out.append(new_id)
        i += 2
    else:
        out.append(ids[i])
        i += 1
return out
```

</details>

## stage 3

<details><summary>hint 1: a nudge</summary>

the pattern is already compiled in the setup cell as `GPT2_SPLIT`. `findall` returns the matched chunks in order, and because every character matches some branch, nothing is skipped.

</details>

<details><summary>hint 2: nearly the answer</summary>

`return GPT2_SPLIT.findall(text)`. if your chunks don't concatenate back to the text, you've changed the pattern: compare it with the lesson's five branches.

</details>

## stage 4

<details><summary>hint 1: a nudge</summary>

`__init__`: validate (`vocab_size >= 256 + len(special_tokens)`, distinct non-empty specials), store `target_vocab_size`, `special_tokens = tuple(...)`, `merges = []`, then `self._build_vocab()`. `train`: `chunks = Counter()` over `pretokenize(piece)` for every non-special piece of `self._split_special(text)`; `words = [list(c.encode("utf-8")) for c in chunks]` with their frequencies. the simple loop: for each merge, recount with `pair_counts(w, counts, freq)` over all words, take `min(counts, key=lambda p: (-counts[p], p))`, append it to the merges and replace every word by `merge_pair(w, best, 256 + r)`. stop early if `counts` is empty. finish with `self.merges = merges; self._build_vocab(); return self`.

</details>

<details><summary>hint 2: nearly the answer</summary>

the fast version keeps `stats` (pair → count), `where` (pair → set of word indices) and a heap of `(-count, pair)`:

```python
while len(merges) < n_merges and heap:
    neg, pair = heapq.heappop(heap)
    if stats.get(pair, 0) != -neg:      # stale entry
        continue
    new_id = 256 + len(merges)
    merges.append(pair)
    changed = set()
    for i in where.pop(pair):
        w, new = words[i], merge_pair(words[i], pair, new_id)
        if len(new) == len(w):
            continue
        for p in zip(w, w[1:]):
            stats[p] -= freqs[i]; changed.add(p)
        for p in zip(new, new[1:]):
            stats[p] += freqs[i]; where[p].add(i); changed.add(p)
        words[i] = new
    for p in changed:
        if stats[p] > 0:
            heapq.heappush(heap, (-stats[p], p))
        else:
            del stats[p]
```

</details>

## stage 5

<details><summary>hint 1: a nudge</summary>

`encode_chunk`: `ids = list(chunk.encode("utf-8"))`; while there are at least two ids, find the adjacent pair with the smallest `self._ranks.get(pair, inf)`; if it isn't in `self._ranks`, stop; else `ids = merge_pair(ids, pair, 256 + self._ranks[pair])`. `encode`: for each `(piece, special)` of `self._split_special(text)`, emit `self.special_ids[piece]` for specials, else extend with the ids of every chunk of `pretokenize(piece)` (through `self._cache`). `decode`: `b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")`, turning a `KeyError` into a `ValueError`.

</details>

<details><summary>hint 2: nearly the answer</summary>

```python
def save(self, path):
    state = {"vocab_size": self.target_vocab_size, "special_tokens": list(self.special_tokens),
             "merges": [list(p) for p in self.merges]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f)

@classmethod
def load(cls, path):
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    tok = cls(state["vocab_size"], special_tokens=tuple(state["special_tokens"]))
    tok.merges = [(int(a), int(b)) for a, b in state["merges"]]
    tok._build_vocab()
    return tok
```

JSON turns tuples into lists: convert the merges back to tuples of ints, or the rank lookup won't find them.

</details>
