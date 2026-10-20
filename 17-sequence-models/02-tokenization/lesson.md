# tokenization

## why this matters

a neural language model can't read text. it reads a sequence of integers, each an index into an embedding table, and predicts the next integer. so before any modeling, you have to decide what one integer stands for. that decision is the **tokenizer**, and it shapes everything after it: how long sequences are (a transformer's cost grows with the square of the length), how big the output layer is, which strings the model can even represent, and what "perplexity" means.

the two obvious choices both fail. **words** give short sequences, but the vocabulary never ends: names, typos, "unfoolable", a new product, any text in another language, and every one of them becomes `<unk>`, a hole in the model's view of the input. **characters** never fail, but sequences get five times longer, and the model has to spend its capacity learning to spell. modern language models use something in between: **subword** units learned from data, so frequent words are one token, rare words are spelled out from frequent pieces, and nothing is ever unknown. this topic builds the most widely used scheme, **byte-level byte-pair encoding** (BPE), the tokenizer of GPT-2 and its descendants. your tokenizer will feed chapter 18's GPT and chapter 27's language models, so it has to be exact: every string in, the same string out.

## warm-up: predict first

1. how many bytes does UTF-8 use for "a", for "é", for "日", for "🙂"?
2. you run BPE on "aaaa". how many times does the pair (a, a) occur? what does the text look like after merging it once?
3. a tokenizer trained only on English text is asked to encode a sentence in Chinese. can it? will the sequence be shorter or longer than the number of characters?
4. you double a BPE vocabulary from 2,048 to 4,096 tokens. by about how much do sequences get shorter: 50%, 25%, 10%?
5. why might GPT-2 tokenize " dog" (with a leading space) and "dog" as two different tokens?

## prerequisites check

- [ ] you know how Python strings, bytes and `str.encode("utf-8")` relate (→ ch 00 · topic 01)
- [ ] you can use `collections.Counter`, dictionaries keyed by tuples, and regular expressions (→ ch 00 · topic 01)
- [ ] you know perplexity, bits per character, and why per-token perplexities can't be compared across tokenizations (→ topic 01)
- [ ] you know what a priority queue is (Python's `heapq`), for the fast trainer in idea 4 (→ ch 00 · topic 01)

## core ideas

### idea 1: a tokenizer is a vocabulary plus two functions

every tokenizer has the same three parts: a **vocabulary** (a finite list of strings, or byte strings, each with an integer id), **`encode`** (text → list of ids) and **`decode`** (ids → text). the one property that matters above all is the round trip,

$$\text{decode}(\text{encode}(t)) = t,$$

because a model trained on a lossy tokenizer can never produce, or even see, what the tokenizer threw away.

the simplest tokenizer that satisfies it on its training text is the **character tokenizer**: the vocabulary is the sorted set of distinct characters, a character's id is its position, encode looks each character up, decode joins them back. sorting matters more than it seems: it makes the ids a deterministic function of the text, so two people (or two runs) who train on the same text get the same ids, and a saved model's embedding rows mean the same thing next time. your `CharTokenizer` stores `chars` (the sorted list) and `stoi` (char → id), and refuses to encode a character it has never seen with a `ValueError` that names it. that refusal is the character tokenizer's weakness in one line: it's closed over its training text. tiny Shakespeare has 65 distinct characters; a modern tweet can have hundreds.

> **▶ build: stage 1.** implement `CharTokenizer`: `__init__`, `train`, `encode`, `decode` (the `vocab_size` property is given).

### idea 2: byte-pair encoding, two primitives

BPE started as a compression algorithm (Gage, 1994) and became a tokenizer when Sennrich, Haddow and Birch (2016) used it for translation. the idea: start from the smallest units, find the most frequent adjacent pair, give it a new id, replace every occurrence, and repeat. each **merge** shortens the text and adds one entry to the vocabulary. after a few hundred merges, common pieces ("th", " the", "ing") are single tokens, and after a few thousand, so are most common words.

two functions do all the work:

- **`pair_counts(ids, counts=None, weight=1)`** adds `weight` to the count of every adjacent pair `(ids[i], ids[i+1])`. pairs **overlap**: in `[7, 7, 7]` the pair `(7, 7)` occurs at positions 0 and 1, so it counts 2. (this is how GPT-2 and most implementations count. the `counts` and `weight` arguments let you accumulate counts over many chunks, each counted once and weighted by how often it occurs.)
- **`merge_pair(ids, pair, new_id)`** scans left to right; where `ids[i], ids[i+1]` equals the pair, it emits `new_id` and skips both, otherwise it emits `ids[i]` and moves on. so `[7, 7, 7]` becomes `[new, 7]`, never `[new, new]`: the middle 7 can only belong to one merged token. it returns a new list.

the overlap convention means a merge's count is an upper bound on how much it shortens the text, not an exact figure (on paper 2). it's only off for runs of one repeated token, which is rare, and what matters is that the trainer and the encoder agree.

> **▶ build: stage 2.** implement `pair_counts` and `merge_pair`.

### idea 3: bytes, and the chunks merges never cross

**why bytes.** if BPE starts from characters, its base vocabulary is whatever characters it saw in training, and the first emoji at test time is unknown. **byte-level** BPE (Radford et al., 2019) starts from the 256 possible bytes of the text's UTF-8 encoding instead. UTF-8 writes every Unicode character as 1 to 4 bytes (ASCII in 1, "é" in 2, "日" in 3, "🙂" in 4: warm-up 1), so *every* string encodes, in any language, and the base vocabulary is fixed at 256. merges then build up multi-byte characters and words as frequency dictates. the price is that a rare character costs several tokens, and a single token can be half a character. that's why `decode` must join all the bytes first and decode UTF-8 once, at the end, with `errors="replace"` for an id sequence that stops mid-character (a model can generate one). the one thing that can't be encoded is a string that isn't valid Unicode at all: a lone surrogate (half of a UTF-16 pair, U+D800–U+DFFF, which a broken decode can leave in a Python string) has no UTF-8 bytes. the reference `encode` raises a `ValueError` naming its position; inputs are expected to be valid Unicode.

**why pre-tokenize.** run BPE on raw text and frequent pairs that straddle word boundaries ("e t" in "the time", "d." in "said.") get merged too, so you get tokens like "e th" that mean nothing and waste vocabulary. GPT-2 first splits the text into **chunks** with a regular expression, counts pairs only inside chunks, and so never merges across a chunk boundary. its pattern, rewritten for Python's `re` (which lacks `\p{L}`; `[^\W\d_]` means "a letter"), is `GPT2_SPLIT` in the setup cell. it is *approximately* GPT-2's pattern: Python's `\d` only matches decimal digits, so other numeric characters (superscripts like ², fractions like ½, Roman numerals like Ⅻ, Unicode categories No and Nl) are grouped with letters here, where GPT-2's `\p{N}` treats them as numbers. in order, a chunk is:

1. an English contraction suffix: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`;
2. a run of letters, with at most one space in front: `" world"`;
3. a run of digits, with at most one space in front;
4. a run of other symbols (punctuation, emoji, underscore), with at most one space in front;
5. whitespace: a run that isn't followed by a non-space, so the last space before a word stays with the word, or else any whitespace run.

so `"don't stop"` → `["don", "'t", " stop"]` and `"  two  spaces "` → `[" ", " two", " ", " spaces", " "]`. the leading space is part of the word, which is why " dog" and "dog" are different tokens (warm-up 5): one is a word in the middle of a sentence, the other starts a line. every character matches one of the branches, so the chunks always concatenate back to the text, and that's what keeps the round trip exact.

> **▶ build: stage 3.** implement `pretokenize(text)`.

### idea 4: training: greedy merges, deterministic ties

**`BPETokenizer(vocab_size=512, *, special_tokens=())`** stores `target_vocab_size`, the special tokens as a tuple, and an empty merge list, then calls the given `_build_vocab()`, which derives everything else from `self.merges`: the vocabulary `vocab` (id → bytes; ids 0–255 are the single bytes, merge $r$ creates id $256 + r$ with `vocab[256 + r] = vocab[a] + vocab[b]`, then the special tokens take the next ids, in order), `special_ids`, the merge ranks and an encode cache. `vocab_size` is a property, `len(vocab)`, so an untrained tokenizer is a plain byte tokenizer with 256 + (number of specials) tokens.

**`train(text)`** learns `target_vocab_size - 256 - len(special_tokens)` merges:

1. cut the special tokens out of the text (the given `_split_special` returns `(piece, is_special)` pairs); they never take part in a merge.
2. pretokenize the remaining pieces and count the chunks. a text of a million characters has only tens of thousands of *distinct* chunks: count each one once and weight its pairs by its frequency.
3. repeatedly take the pair with the highest total count, **ties broken by the smallest pair** `(id1, id2)` in Python's tuple order, record it, and merge it in every chunk. stop when you have enough merges, or when no pair is left (a tiny text can run out; `vocab_size` then reports what was actually learned).

the tie rule makes training a deterministic function of the text. without it, the order in which a dictionary happens to iterate would decide between equal counts, and two runs (or two libraries) could learn different tokenizers from the same data. greedy merges have one more useful property: the first $k$ merges of a long run are exactly the merges of a run that stops at $k$, so smaller vocabularies are prefixes of larger ones (the lab uses this).

**making it fast.** recounting every pair of every chunk after every merge is simple and correct, and fine for the tests. on a megabyte of text it costs tens of milliseconds per merge, a minute or more for a few thousand merges. the fix is to notice that a merge only changes the chunks that contain its pair. keep the counts, plus an index from each pair to the chunks containing it; for each affected chunk, subtract its old pairs' counts and add its new ones. pick the best pair with a heap of `(-count, pair)` entries, pushing a fresh entry whenever a count changes and skipping entries whose count is stale when popped. the heap's ordering does the tie-breaking for you. the reference trains a 4,096-token vocabulary on tiny Shakespeare in under a second this way.

> **🔮 predict:** in "aaabdaaabac", which pair is merged first, and how many times does it occur? after that merge, two pairs tie for the most frequent. which one wins?

> **▶ build: stage 4.** implement `BPETokenizer.__init__` and `BPETokenizer.train`.

### idea 5: encoding by rank, special tokens, saving

**encoding** a new text has to reproduce what training did to each chunk. the clean way: split off special tokens, pretokenize, and for each chunk start from its UTF-8 bytes and repeatedly merge the adjacent pair with the **lowest merge rank** (the earliest-learned merge) until no adjacent pair is a known merge. `encode_chunk` does one chunk. this gives exactly the same result as replaying every merge in order (on paper 3), but only touches the merges that matter. texts repeat their words endlessly, so `encode` caches each distinct chunk's ids in `self._cache` (built empty by `_build_vocab`, so it's reset whenever the merges change).

**special tokens** are strings like `<|endoftext|>` that mark structure (document boundaries in chapter 27, padding, the start of a reply). they must never be split into bytes or merged with their neighbours, so `encode` matches them literally first, *longest first* (so `<|endoftext|>` isn't read as `<|end|>` followed by text when both exist), emits their reserved ids, and tokenizes only the text in between. a partial special token is just text.

**decode** concatenates the byte strings of the ids and decodes UTF-8 once, with invalid sequences turned into U+FFFD rather than an exception; an id outside the vocabulary is a `ValueError`. **save** writes the settings and the merge list as JSON (the vocabulary is derived from the merges, so that's all a tokenizer *is*), and the classmethod **load** rebuilds an identical tokenizer from the file.

> **🔮 predict:** a tokenizer trained on Shakespeare encodes a sentence of Chinese. how many tokens per character?

> **▶ build: stage 5.** implement `BPETokenizer.encode_chunk`, `encode`, `decode`, `save` and `load`.

## worked example

**training on `"aaabdaaabac"`** (one chunk: no spaces or punctuation), `vocab_size=259`, so 3 merges. bytes: a = 97, b = 98, c = 99, d = 100.

| step | sequence | pair counts (top) | merge |
|---|---|---|---|
| 0 | a a a b d a a a b a c | (97, 97): 4 overlapping, (97, 98): 2 | (97, 97) → 256 |
| 1 | 256 a b d 256 a b a c | (256, 97): 2, (97, 98): 2 (tie) | (97, 98) → 257 (smaller pair) |
| 2 | 256 257 d 256 257 a c | (256, 257): 2 | (256, 257) → 258 |
| 3 | 258 d 258 a c | | |

so `vocab[258] = b"aaab"`, and encoding `"aaabdaaabac"` with the trained tokenizer gives `[258, 100, 258, 97, 99]`: 11 bytes in 5 tokens.

**ties in `"abcd"`:** every pair occurs once, so the smallest wins: (97, 98) → 256. then the sequence is 256 c d, and the candidates are (256, 99) and (99, 100). the smaller pair is (99, 100), because 99 < 256, so it's merged next (→ 257) even though it appears later in the text, and finally (256, 257) → 258.

**overlap in `"aaaa"`:** (97, 97) occurs 3 times (positions 0, 1, 2) and merges into `[256, 256]`, then (256, 256) → 257. after that no pair is left, so asking for 300 tokens yields a vocabulary of 258.

**chunks in `"ab ab ab"`:** the chunks are `"ab"`, `" ab"`, `" ab"`. (97, 98) occurs 3 times → 256; then (32, 256), a space followed by "ab", occurs twice → 257. no merge ever contains "b" followed by a space, because "b" ends every chunk.

**bytes:** `"hé"` encodes with an untrained tokenizer to `[104, 195, 169]`: "é" is the two bytes C3 A9. decoding just `[195]` gives `"�"`. the tests check every one of these.

## on paper

1. **UTF-8 lengths.** UTF-8 uses 1 byte for code points below $2^7$, 2 below $2^{11}$, 3 below $2^{16}$, and 4 up to U+10FFFF. how many bytes is "日本" (U+65E5, U+672C)? a byte-level tokenizer trained only on English has learned no merges among CJK bytes. how many tokens is a 20-character Chinese sentence?

<details><summary>solution</summary>

each character is between $2^{11}$ and $2^{16}$, so 3 bytes: "日本" is 6 bytes. with no applicable merges, every byte is its own token: 60 tokens for 20 characters, 3 tokens per character, longer than a character tokenizer (but a character tokenizer trained on English couldn't encode it at all). this is the multilingual cost of an English-trained tokenizer, and why multilingual models train their tokenizers on multilingual text.

</details>

2. **what a merge saves.** a merge of pair $(a, b)$ with count $c$ (overlapping occurrences) is applied to a text. show that the text gets shorter by exactly $c$ tokens when $a \ne b$, and give a sequence where it gets shorter by fewer when $a = b$.

<details><summary>solution</summary>

for $a \ne b$, two occurrences of $(a, b)$ can't overlap (an overlap would need position $i+1$ to be both $b$ and $a$), so the left-to-right scan merges every occurrence, each removing one token: $c$ fewer. for $a = b$, a run of $m$ copies has $m - 1$ overlapping occurrences but only $\lfloor m/2\rfloor$ merges: "aaa" has $c = 2$ but shrinks by 1.

</details>

3. **rank-greedy encoding replays training.** training applied merges $1, \dots, M$ in order, each to all occurrences. encoding repeatedly merges the lowest-rank pair present. show both give the same result on any chunk. (hint: when you merge pair $r$, can a pair of rank below $r$ appear?)

<details><summary>solution</summary>

merging pair $r$ creates new adjacent pairs only around the new token $256 + r$, and any merge involving token $256 + r$ was learned after it existed, so it has rank greater than $r$. the other adjacent pairs were already present. so after rank-greedy merges pair $r$, no pair of rank $< r$ is present (none was, since $r$ was the lowest, and none can appear), and the sequence of merges it performs is increasing in rank. replaying the merges in order does the same thing, except that it also "tries" merges whose pair isn't present, which change nothing. both apply the same merges, in the same order, each to all occurrences, so they agree.

</details>

## experiment

`lab.py` trains your `BPETokenizer` on tiny Shakespeare and measures:

1. held-out characters per token for vocabularies from 256 to 4,096, and what the last tokens learned look like.
2. how the Shakespeare tokenizer copes with modern English, French, code, numbers, Chinese and emoji.
3. your topic-01 `NGramLM` (order 3) on character tokens and on BPE tokens of several sizes, compared fairly in bits per character.

## check your understanding

1. why must the decoder join all the bytes before decoding UTF-8, instead of decoding each token separately?

<details><summary>answer</summary>

a token can end in the middle of a multi-byte character: with few merges, "é" is two tokens, C3 and A9, neither of which is valid UTF-8 on its own. decoding token by token would produce two replacement characters instead of "é". only the concatenation of all the bytes is guaranteed to be valid UTF-8 (when the ids came from `encode`).

</details>

2. what goes wrong if the trainer counts pairs across chunk boundaries but the encoder pretokenizes?

<details><summary>answer</summary>

the trainer would learn merges like ("e", " t") that the encoder can never apply (they span two chunks), wasting vocabulary entries, and the token counts it optimized wouldn't be the ones the encoder produces. trainer and encoder must agree on the units, the same way they must agree on overlapping counts and on left-to-right merging.

</details>

3. why does a deterministic tie rule matter for a tokenizer, when the model would train fine either way?

<details><summary>answer</summary>

because the tokenizer is part of the model: the ids index the embedding table. retraining a tokenizer on the same text must give the same ids, or a saved model, a cached dataset and a fresh tokenizer silently disagree. it also makes the tokenizer testable against a reference merge for merge.

</details>

4. why can a smaller vocabulary be obtained by truncating a larger one's merge list?

<details><summary>answer</summary>

training is greedy: merge $r$ depends only on the text after merges $1, \dots, r - 1$, never on how many merges will follow. so a run asked for $k$ merges makes exactly the first $k$ choices of a longer run, and stopping early gives the same tokenizer.

</details>

5. a language model's perplexity per token gets *worse* as you increase the BPE vocabulary. is the model getting worse?

<details><summary>answer</summary>

not necessarily: each token now covers more text, so it's a harder prediction. compare total bits divided by the number of characters (or bytes) instead. the lab's experiment 3 shows per-token perplexity rising tenfold while bits per character improve.

</details>

6. what does `<|endoftext|>` buy over just putting a newline between documents?

<details><summary>answer</summary>

it can't occur inside a document (it's matched literally and gets its own id, never produced by merging ordinary text), so a model can learn "a new document starts here; ignore what came before" without any ambiguity, and a generator can stop when it emits it. a newline is everywhere in ordinary text.

</details>

## common bugs

- **counting non-overlapping pairs in the trainer but merging left to right (or the reverse).** symptom: merges that differ from the reference on texts with repeated letters ("aaaa", "!!!"), and only there.
- **breaking ties by dictionary order.** symptom: merges that match the reference on large texts and differ on small ones, where ties are common.
- **forgetting to remove special tokens before training.** symptom: merges like `<|` and `endo`; vocabulary wasted on markup.
- **decoding token by token.** symptom: U+FFFD in the middle of words with accents or emoji.
- **matching special tokens shortest first.** symptom: `<|endoftext|>` encodes as `<|end|>` plus text when both are registered.
- **a stale encode cache.** symptom: after changing `merges` (training again, or the lab's truncation), `encode` returns ids from the old merges. rebuild everything in `_build_vocab`.

## going deeper

- Sennrich, Haddow & Birch (2016), "Neural machine translation of rare words with subword units", *ACL*. BPE as a tokenizer. [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Gage (1994), "A new algorithm for data compression", *C Users Journal* 12(2): 23–38. the original byte-pair compression.
- Radford, Wu, Child, Luan, Amodei & Sutskever (2019), "Language models are unsupervised multitask learners" (GPT-2), section 2.2: byte-level BPE and its pre-tokenization. [cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Kudo & Richardson (2018), "SentencePiece: a simple and language independent subword tokenizer and detokenizer for neural text processing", *EMNLP demo*. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- Kudo (2018), "Subword regularization: improving neural network translation models with multiple subword candidates", *ACL*. the unigram tokenizer. [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)
- Rust, Pfeiffer, Vulić, Ruder & Gurevych (2021), "How good is your tokenizer? On the monolingual performance of multilingual language models", *ACL*. [arXiv:2012.15613](https://arxiv.org/abs/2012.15613)

## connections

- ← builds on: strings, bytes and regular expressions (ch 00 · topic 01), perplexity and bits per character (topic 01).
- → used in: word embeddings over a word vocabulary (topic 03); the capstone's character tokenizer; chapter 18's GPT, trained on your `BPETokenizer`; chapter 27's document separators (`<|endoftext|>`) and supervised fine-tuning formats, which rely on special tokens being matched exactly.
