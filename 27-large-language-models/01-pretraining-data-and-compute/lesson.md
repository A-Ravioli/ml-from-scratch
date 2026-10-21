# pretraining data & compute

## why this matters

a language model is mostly its data. the architecture of a modern LLM fits on a page (topic 02), and the training loop is the one you wrote in chapter 18. what separates a good base model from a bad one is the trillions of tokens it reads, and those tokens start as a web crawl: billions of HTML pages, most of which you'd never want a model to learn from. cookie banners, navigation menus, shop pages generated from one template a million times, keyword spam, and the same news story syndicated across a thousand sites. RefinedWeb's pipeline (Penedo et al., 2023) threw away almost 90% of the Common Crawl documents it started from, and the model trained on what was left beat models trained on curated corpora.

two problems dominate. **quality**: deciding, cheaply and at scale, which pages are text at all. **duplication**: the web copies itself, and a model trained on a passage a hundred times memorizes it (Carlini et al., 2023) while wasting compute that could have gone to new text. Lee et al. (2022) found single sentences repeated tens of thousands of times in C4 and showed that deduplicating made models better while emitting memorized text ten times less often. comparing every pair of a billion documents is impossible ($10^{18}$ pairs), so near-duplicate detection needs a trick: **MinHash** to sketch each document in a few hundred numbers, and **locality-sensitive hashing** to find similar sketches without comparing them all.

and once the data is clean, there's a budget question: how many parameters and how many tokens should a given amount of compute buy, and what is it worth to repeat data you've run out of? this topic builds the deduplication and filtering pipeline and the arithmetic of data mixtures and repetition.

## warm-up: predict first

1. two documents share 80% of their 5-character substrings. if you compare 128 random "fingerprints" of each, about how many will match?
2. you hash every document's normalized text and drop exact repeats. what fraction of a crawl's copies do you think that removes?
3. a corpus has 800B tokens of web text, 150B of code and 50B of books. sampling sources in proportion to size, how often does a training batch contain books? how would you give books more weight without starving the web?
4. you have 100B unique tokens and compute for 400B. is it better to train a bigger model on 100B tokens once, or repeat the data four times?

## prerequisites check

- [ ] you can compute a Jaccard similarity and know what a hash function is (→ ch 00 · 02, ch 09)
- [ ] you know the Chinchilla rule $D \approx 20N$ and $C \approx 6ND$ (→ ch 19 · 05, `mlfs.dltheory.compute_optimal`)
- [ ] you can train a small GPT with your `train_lm` (→ ch 18 · 04)
- [ ] you can reason about the probability that independent events all happen, or that at least one does (→ ch 04)

## core ideas

### idea 1: documents as sets

to ask "how similar are two pages?" you need a representation in which small edits make small differences. a bag of words ignores order; exact hashes see any edit as total difference. the classic choice is the set of **shingles**: every run of $k$ consecutive characters (or words) of the normalized text. "the cat sat" with $k = 3$ gives {the, "he ", "e c", " ca", cat, "at ", "t s", " sa", sat}. normalizing first (lowercase, collapse whitespace) makes copies that differ only in case or spacing identical. changing one word changes only the $k$-grams that overlap it, about $k + \text{len(word)}$ of them, so a long document with one edit keeps almost all its shingles.

the similarity of two sets is their **Jaccard** index,

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|},$$

1 for identical sets, 0 for disjoint ones. $k$ trades off sensitivity: with $k = 1$ any two English texts share almost every character; with $k = 20$ a single typo destroys twenty shingles. five characters (or five words for long documents) is a common compromise.

> **spec.** `shingles(text, k=5)`: lowercase, collapse every run of whitespace to one space (`" ".join(text.lower().split())`), return the set of all length-$k$ substrings; a normalized text of length at most $k$ is a single shingle (itself), an empty one has none. `jaccard(a, b)`: for two sets, $|a \cap b| / |a \cup b|$, defined as 1.0 for two empty sets; for two ndarrays (MinHash signatures, idea 2), the fraction of positions where they're equal, `ValueError` for different shapes.

> **▶ build: stage 1.** implement `shingles` and `jaccard`.

### idea 2: MinHash

storing every document's shingle set and intersecting sets pairwise is too expensive. **MinHash** (Broder, 1997) compresses a set into a short **signature** whose agreement estimates Jaccard. take a random hash function $h$ that maps every possible shingle to a distinct number, and record $\min_{x \in A} h(x)$. the key fact:

$$P\Big[\min_{x \in A} h(x) = \min_{x \in B} h(x)\Big] = J(A, B).$$

why: the element of $A \cup B$ with the smallest hash is equally likely to be any of them (the hash is a random ordering), and the two minima agree exactly when that element lies in $A \cap B$. one hash gives a coin that lands heads with probability $J$; $n$ independent hash functions give $n$ coins, and the fraction of matching positions estimates $J$ with standard deviation $\sqrt{J(1-J)/n}$, about 0.035 at $J = 0.8$ with $n = 128$. a document of any length becomes 128 integers.

in practice "a random ordering of all strings" is a **universal hash family**: hash each shingle once to an integer $x$ with a fixed function, then use $h_i(x) = (a_i x + b_i) \bmod p$ for a prime $p$ and random $a_i, b_i$ drawn from a seeded generator. two details matter. the base hash must be the same in every process: Python's `hash()` of a string is randomized per process (`PYTHONHASHSEED`), so a crawl deduplicated today wouldn't match tomorrow's; use `hashlib` (e.g. `blake2b`) or a hand-written FNV. and the arithmetic must not overflow: with $p = 2^{31} - 1$, $a x + b < 2^{62}$ fits in a `uint64`.

> **spec.** `minhash_signature(shingle_set, *, n_perm=128, seed=0)`: a `uint64` array of shape `(n_perm,)`; entry $i$ is the minimum over the set of the $i$-th of `n_perm` independent hash functions, all drawn from `seed`. deterministic across processes (no Python `hash()`), independent of iteration order. the empty set has a fixed signature (e.g. all `np.iinfo(np.uint64).max`).

> **🔮 predict:** two sets have Jaccard 1/3. over 25 seeds of 128 hashes, how much will the estimated similarity scatter?

> **▶ build: stage 2.** implement `minhash_signature`.

### idea 3: locality-sensitive hashing

signatures make each comparison cheap, but a billion documents still give $5 \times 10^{17}$ pairs. **banding** (Leskovec, Rajaraman & Ullman, ch. 3) finds the similar pairs without looking at the rest. split each signature of $n$ entries into $b$ bands of $r = n/b$ rows. hash each band (the tuple of its $r$ values) into a bucket; two documents become a **candidate pair** if they land in the same bucket in at least one band. the probability for a pair with similarity $s$ is

$$P(\text{candidate}) = 1 - (1 - s^r)^b,$$

all $r$ rows of a band agree with probability $s^r$, and at least one of $b$ bands does with the complement of "none does". plotted against $s$, it's an S-curve that rises steeply near $s^* \approx (1/b)^{1/r}$. dissimilar pairs almost never collide, similar ones almost always do, and only candidates are compared, so the work is roughly linear in the number of documents.

then **verify**: estimate each candidate's similarity from the full signatures and keep the pair only if it reaches the threshold. that removes the false positives the S-curve's lower tail lets through. finally, near-duplication isn't transitive (A ≈ B and B ≈ C don't imply A ≈ C), but for deduplication you want to keep one representative of each group of copies, so treat verified pairs as edges of a graph and take its **connected components** with union-find. keep the first document of each component.

> **spec.** `minhash_dedup(docs, *, threshold=0.8, n_perm=128, n_bands=32, k=5, seed=0)`: `ValueError` unless `n_bands` divides `n_perm`; signatures of `shingles(doc, k)` with `n_perm` and `seed`; candidate pairs are those sharing a bucket in at least one band of `n_perm / n_bands` rows (never all pairs); a candidate becomes an edge if its signatures agree on at least `threshold` of positions; clusters are connected components; returns the sorted list of indices to keep, the smallest index of each cluster.

> **🔮 predict:** with 32 bands of 4 rows, at what similarity is a pair a candidate half the time? the lab measures it.

> **▶ build: stage 3.** implement `minhash_dedup`.

### idea 4: quality filtering

deduplication won't remove a page that's junk once. the cheapest quality filters are heuristics on surface statistics, and the rules in the Gopher paper (Rae et al., 2021) are a standard starting point: a document must have enough words (between 50 and 100,000 there), a mean word length between 3 and 10 characters (catching keyword lists, tables of numbers and gibberish), at most one `#` or ellipsis per ten words (hashtag spam, truncated previews), at least 80% of words containing a letter, and few repeated lines (navigation menus and cookie walls repeat the same lines). each rule is weak alone, and each misfires sometimes: poetry has short lines, code has symbols. production pipelines add language identification, perplexity filters (keep text a small model trained on Wikipedia finds plausible) and learned classifiers (Brown et al., 2020, kept pages that looked like the curated sources), but heuristics remove the bulk cheaply. returning the list of failed rules instead of a bare boolean is what lets you audit the filter: count how often each rule fires and read a sample of what it removes.

> **spec.** `quality_filter(text, *, min_words=20, word_len=(3.0, 10.0), max_symbol_ratio=0.1, min_alpha_frac=0.8, max_dup_line_frac=0.3)` returns `(keep, reasons)`, where `reasons` lists, in this order, every rule the text fails and `keep = not reasons`. words are `text.split()`. `"too_short"`: fewer than `min_words` words. `"word_length"`: mean word length outside `word_len` (inclusive). `"symbols"`: (count of `#` + count of `...` + count of `…`) / words above `max_symbol_ratio`. `"non_alpha"`: fraction of words with at least one alphabetic character below `min_alpha_frac`. `"repeated_lines"`: among the stripped non-empty lines, the fraction identical to an earlier line above `max_dup_line_frac`. the word rules are skipped for a text with no words.

> **▶ build: stage 4.** implement `quality_filter`.

### idea 5: mixtures, repetition and compute

a pretraining corpus mixes sources: web, code, books, papers, conversation. sampling them **in proportion to size** lets the biggest source dominate; uniform sampling repeats small sources dozens of times. the standard knob is a **temperature**: sample source $i$ with probability

$$p_i \propto n_i^{1/T},$$

$T = 1$ proportional, $T \to \infty$ uniform, in between a compromise (multilingual models sample languages with $T \approx 3$–$5$, e.g. Conneau et al., 2020, used $p \propto n^{0.3}$). the Pile (Gao et al., 2020) set weights by hand and reported epochs per source: Wikipedia was seen about three times, the web once. compute it with logs: at a low temperature, token counts like $10^{12}$ raised to the power $1/T$ overflow ($T = 0.02$ gives $10^{600}$).

repetition is the other half. Chinchilla's rule (Hoffmann et al., 2022; `compute_optimal` from chapter 19) says a compute budget $C = 6ND$ is best spent on $N$ parameters and $D \approx 20N$ tokens: a 70B model wants 1.4T tokens. what if you don't have that much unique text? Muennighoff et al. (2023) trained hundreds of models on repeated data and fit how much a repeated token is worth: with $U$ unique tokens seen for $R = D/U - 1$ extra epochs, the **effective** data is

$$D' = U + U R^* \left(1 - e^{-R/R^*}\right), \qquad R^* \approx 15.4.$$

for small $R$, $D' \approx U(1 + R) = D$: the first few repeats are almost as good as fresh data (4 epochs are worth about 3.7). for large $R$ it saturates at $U(1 + R^*)$: no amount of repetition is worth more than about 16 epochs of fresh text. deduplication and repetition are two sides of one coin: duplicates are uncontrolled repetition, concentrated on whatever the web happened to copy most.

> **spec.** `mixture_weights(sizes, *, temperature=1.0)`: `ValueError` for `temperature <= 0` or negative sizes or a zero total; probabilities $p_i \propto n_i^{1/T}$ (computed in log space so huge counts don't overflow), 0 for empty sources, summing to 1. `effective_tokens(unique_tokens, total_tokens, *, r_star=15.4)`: `total_tokens` itself if it doesn't exceed `unique_tokens`, else $U + U r^* (1 - e^{-R/r^*})$ with $R = \text{total}/U - 1$; a float.

> **▶ build: stage 5.** implement `mixture_weights` and `effective_tokens`.

## worked example

**shingles and Jaccard.** "the cat sat" and "the cat sang" with $k = 3$: the first has 9 shingles (listed in idea 1), the second 10 (the same first eight, then "san", "ang"). they share 8; the union has 11; $J = 8/11 = 0.727$.

**the S-curve.** with $n = 128$, $b = 32$, $r = 4$: a pair at $s = 0.8$ has $s^4 = 0.410$ and is missed with probability $(1 - 0.410)^{32} = e^{32 \ln 0.590} \approx 5 \times 10^{-8}$. at $s = 0.5$: $s^4 = 0.0625$, $(0.9375)^{32} = 0.127$, so it's a candidate 87% of the time (and verification throws most of those out again). at $s = 0.3$: 23%. the threshold estimate $(1/32)^{1/4} = 0.42$ is where $P \approx 1 - 1/e = 0.63$; $P = 1/2$ a little lower, at $s = 0.38$.

**mixtures.** sizes 800, 150, 50 at $T = 2$: $\sqrt{n} = 28.3, 12.2, 7.1$, so $p = (0.594, 0.257, 0.149)$: books go from 5% of batches to 15%.

**repetition.** $U = 10^9$ unique tokens, $D = 4 \times 10^9$: $R = 3$, $D' = 10^9 (1 + 15.4 (1 - e^{-3/15.4})) = 10^9 (1 + 15.4 \times 0.177) = 3.73 \times 10^9$. your functions must reproduce all of these.

## on paper

1. **the MinHash identity.** let $h$ be a uniformly random injective map from $A \cup B$ to the reals. prove $P[\min_A h = \min_B h] = J(A, B)$.

<details><summary>solution</summary>

let $x^* = \arg\min_{x \in A \cup B} h(x)$. by symmetry every element of $A \cup B$ is equally likely to be $x^*$, so $P[x^* \in A \cap B] = |A \cap B| / |A \cup B|$. if $x^* \in A \cap B$, it's the minimum of both $A$ and $B$, so the minima agree. if $x^* \in A \setminus B$, then $\min_A h = h(x^*) < \min_B h$ (the minimum over $B$ is attained at some other element, and $h$ is injective), so they differ; the same for $B \setminus A$. so the minima agree exactly when $x^* \in A \cap B$.

</details>

2. **the S-curve.** (a) derive $P(\text{candidate}) = 1 - (1 - s^r)^b$. (b) show that the curve's steepest point is near $s^* = (1/b)^{1/r}$ by finding where $s^r = 1/b$, and evaluate $P$ there for large $b$.

<details><summary>solution</summary>

(a) a band matches iff all $r$ independent rows match: $s^r$. bands use disjoint rows, so they're independent; no band matches with probability $(1 - s^r)^b$. (b) at $s^r = 1/b$, $P = 1 - (1 - 1/b)^b \to 1 - 1/e \approx 0.63$ for large $b$. the curve's inflection point sits close to (slightly below) $s^*$, and the transition width shrinks as $r$ grows, which is why more rows per band make a sharper threshold.

</details>

3. **how much is repetition worth?** using $D' = U + UR^*(1 - e^{-R/R^*})$, (a) show $D' \approx D$ for $R \ll R^*$, (b) find the limit as $R \to \infty$, (c) for $U = 100$B and a budget of $D = 400$B, compute $D'$ and the fraction of the budget wasted.

<details><summary>solution</summary>

(a) $1 - e^{-x} \approx x$ for small $x$, so $D' \approx U + UR = U(1 + R) = D$. (b) $U(1 + R^*) \approx 16.4\,U$. (c) $R = 3$: $D' = 100 (1 + 15.4 \times 0.1770) = 372.6$B; about 7% of the tokens' value is lost to repetition, a small price, which is why repeating a few epochs is a reasonable response to running out of data.

</details>

## experiment

the lab builds a **synthetic** web crawl of 3,000 pages from tiny Shakespeare (originals, mirrors, syndicated copies wrapped in other sites' boilerplate, templated shop pages, junk; there's no crawl on this machine) and runs your filter, exact deduplication and MinHash deduplication over it, counting what survives. it measures the LSH S-curve with your signatures for three band splits, and it trains two small GPTs for the same number of steps, one on the crawl with its duplicates and one deduplicated, comparing their training loss, held-out loss, and loss on heavily copied versus rarely copied articles.

## check your understanding

1. why lowercase and collapse whitespace before shingling?

<details><summary>answer</summary>

mirrored pages often differ only in formatting. without normalization, "Hello World" and "hello  world" share almost no 5-grams; with it they're identical. normalization decides which differences count as "the same text".

</details>

2. a learner hashes shingles with Python's `hash()`. their dedup works in tests. what breaks in production?

<details><summary>answer</summary>

string hashes are randomized per process, so signatures computed in different processes (parallel workers, a rerun tomorrow, an index built last month) aren't comparable: copies hashed in different workers never collide, and an incremental pipeline can't match new pages against the stored signatures.

</details>

3. your crawl has a million templated product pages with Jaccard 0.9 to each other. MinHash dedup keeps one. is that right?

<details><summary>answer</summary>

for training a language model, mostly yes: a million copies of one template teach the model the template a million times. but each page carried some unique information (its product name and price), which is now gone. a pipeline that wants the facts might remove the repeated spans instead of the page (C4 dropped repeated three-sentence spans; RefinedWeb removes long repeated substrings).

</details>

4. why verify candidate pairs instead of trusting the buckets?

<details><summary>answer</summary>

the S-curve has a lower tail: pairs well below the threshold still collide in some band with noticeable probability (a 0.5-similar pair is a candidate 87% of the time with 32 bands of 4). verification with the full signature (an estimate with standard deviation ~0.04) removes them.

</details>

5. why connected components and not "remove $j$ if it's similar to some kept document"?

<details><summary>answer</summary>

copies drift: each syndication edits the previous one a little, so the far end of a chain of copies can fall below the threshold relative to the original while still being a copy. components keep one document per chain. the greedy rule depends on document order and keeps the drifted tail. (components can also merge two genuinely different documents through a chain; at a high threshold this is rare.)

</details>

6. why is "the model trained on the raw crawl has lower training loss" not evidence that it's better?

<details><summary>answer</summary>

the two models are scored on different training sets. the raw one is full of repeated text, which is easy to predict once memorized, so its training loss is low. only a held-out set, never seen by either, compares them fairly.

</details>

7. Chinchilla says 20 tokens per parameter. you have 5× less unique data than that. what are your options?

<details><summary>answer</summary>

repeat the data (up to ~4 epochs costs little, per Muennighoff et al.), train a smaller model on fewer tokens than compute-optimal, add other sources (code, which also helps reasoning; synthetic data), or relax filtering thresholds to keep more data. each trades some quality.

</details>

## common bugs

- **Python's `hash()` for shingles.** works inside one process, silently breaks across processes.
- **overflowing `uint64`.** with $p = 2^{61} - 1$, $a \cdot x$ needs 122 bits; NumPy wraps around silently, so the functions are no longer $(a x + b) \bmod p$ and lose the universal-hash guarantee (pairwise independence, which bounds collisions and correlations between positions). on random-looking shingle sets the wrapped values still look random and the estimates come out unbiased, so tests can't catch it; the failure would show only on adversarial or highly structured inputs. use a 31-bit prime or Python integers, so the guarantee holds by construction.
- **the same hash function in every position.** 128 copies of one estimate have the variance of one; check that signature entries for different positions differ.
- **comparing all pairs.** correct, and quadratic: 10,000 documents is 50 million comparisons. only compare what shares a bucket.
- **greedy instead of union-find.** results depend on document order, and drifted copy chains survive.
- **mixture weights in linear space.** `n ** (1 / T)` with $n = 10^{12}$ and $T = 0.02$ is $10^{600}$: it overflows to `inf`, and `inf / inf` is `nan`.

## going deeper

- Broder (1997), "On the resemblance and containment of documents", *Compression and Complexity of Sequences*. [doi:10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900)
- Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, ch. 3 (shingling, MinHash, LSH banding). [mmds.org](http://www.mmds.org/)
- Lee et al. (2022), "Deduplicating training data makes language models better", *ACL*. [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)
- Carlini et al. (2023), "Quantifying memorization across neural language models", *ICLR*. [arXiv:2202.07646](https://arxiv.org/abs/2202.07646)
- Rae et al. (2021), "Scaling language models: methods, analysis & insights from training Gopher" (MassiveText quality filters, appendix A). [arXiv:2112.11446](https://arxiv.org/abs/2112.11446)
- Penedo et al. (2023), "The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only". [arXiv:2306.01116](https://arxiv.org/abs/2306.01116)
- Gao et al. (2020), "The Pile: an 800GB dataset of diverse text for language modeling". [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
- Hoffmann et al. (2022), "Training compute-optimal large language models". [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Muennighoff et al. (2023), "Scaling data-constrained language models", *NeurIPS*. [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)

## connections

- ← builds on: sets and hashing (ch 00, ch 09), scaling laws and `compute_optimal` (ch 19 · 05), your GPT and `train_lm` (ch 18 · 04).
- → used in: the capstone deduplicates and filters its pretraining corpus; topic 06 reuses shingles and n-grams to detect test items leaking into training data; topic 07's retrieval index is another way of finding similar documents fast.
