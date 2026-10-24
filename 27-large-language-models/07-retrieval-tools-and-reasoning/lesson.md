# retrieval, tools & reasoning

## why this matters

a language model's knowledge is frozen in its weights at the end of training, blurred by compression, and impossible to cite. ask it about last week, about your company's documents, or about a fact it saw once, and it will often answer anyway, fluently and wrongly. **retrieval-augmented generation** (RAG; Lewis et al., 2020) gives the model an open book: embed a document collection, find the passages closest to the question, and put them in the prompt. the model's job shrinks from remembering to reading. the same move generalizes: a model that can **call tools** (a search engine, a calculator, a code interpreter) in a loop, reading each result before deciding the next step, can solve problems no single forward pass could (Yao et al., 2023; Schick et al., 2023). and spending more compute at inference time, by sampling several reasoning paths and voting, buys accuracy that would otherwise need a bigger model (Wang et al., 2023).

all three share a sobering property: the system is only as good as its weakest stage. if the retriever doesn't find the passage, no reader can answer; if one hop of a tool chain fails, the chain fails. this topic builds the retrieval stack (chunking, a training-free embedder, exact and approximate nearest-neighbour search), a RAG prompt, a minimal agent loop, and majority voting, and the lab measures how retrieval caps everything downstream.

## warm-up: predict first

1. you split documents into 200-word chunks with no overlap. a fact straddles a chunk boundary. what happens to questions about it?
2. a retriever finds the right passage for 70% of questions. what's the best answer accuracy any reader can get with those passages?
3. an approximate index searches 6% of the collection per query. what fraction of the true top 5 do you expect it to return?
4. a model answers a question correctly 40% of the time, and when it's wrong it gives one of many different wrong answers. you sample 20 answers and take the majority. better or worse than one sample?

## prerequisites check

- [ ] you know cosine similarity and why we normalize embeddings (→ ch 17 · 03)
- [ ] you can run k-means and know what its cells look like (→ ch 11 · 01, `mlfs.unsupervised.KMeans`)
- [ ] you know TF-IDF or at least term weighting (→ ch 09)
- [ ] you can format a prompt and read a model's output (→ topics 03–05)

## core ideas

### idea 1: chunking

documents are too long to embed whole (one vector for a book says little about any page) and too long to paste into a prompt. so they're cut into **chunks** of a few hundred tokens, each embedded and retrieved on its own. a fixed-size sliding window with some **overlap** is the baseline: with chunk size $C$ and overlap $O$, windows start every $C - O$ words, so any span of at most $O + 1$ words lies entirely inside some chunk (on paper 1). no overlap means facts that straddle a boundary are split in two and may be unretrievable; too much overlap multiplies the index size by $C/(C - O)$ and fills the top-$k$ with near-copies. smarter splitters cut at paragraph or sentence boundaries, but the trade-off is the same: small chunks are precise and lose context, large chunks keep context and dilute the embedding.

> **spec.** `chunk_text(text, chunk_size=200, overlap=50)`: `ValueError` unless `chunk_size >= 1` and `0 <= overlap < chunk_size`; splits on whitespace and returns the windows of `chunk_size` words (joined by single spaces) starting at word $0, s, 2s, \dots$ with $s = $ `chunk_size - overlap`, stopping after the first window that reaches the last word (the last window may be shorter); an empty text gives `[]`.

> **▶ build: stage 1.** implement `chunk_text`.

### idea 2: an embedder with no training

dense retrievers (Karpukhin et al., 2020) embed queries and passages with a trained transformer, so that a question and its answer passage end up close even with no words in common. you don't have one here, so you'll build the strong classical baseline in vector form: **TF-IDF over character n-grams**, hashed into a fixed number of buckets.

character 3- to 5-grams of the lowercased, space-padded text capture words, word pieces and inflections, so "capital" and "capitals" share most of their n-grams. the **hashing trick** (Weinberger et al., 2009) maps each n-gram to a bucket $h(g) \bmod D$ instead of keeping a vocabulary: no fitting of a vocabulary, a fixed memory footprint, and occasional collisions that grow as $D$ shrinks. use a fixed hash function such as `zlib.crc32`, not Python's per-process `hash()` (topic 01's lesson again: an index built today must match a query embedded tomorrow). weight bucket $j$ of a text by a **sublinear term frequency** times an **inverse document frequency**,

$$w_j = (1 + \log c_j)\cdot\Big(\log\frac{1 + N}{1 + \mathrm{df}_j} + 1\Big) \quad (c_j > 0),$$

so repeated n-grams count less than proportionally and n-grams that occur in every chunk ("the ") count least. normalize each vector to unit length and the dot product is the cosine similarity. this is close to what sparse retrievers like BM25 (Robertson & Zaragoza, 2009) compute, and it's still a strong baseline for keyword-heavy questions.

> **spec.** `HashingEmbedder(dim=1024, *, ngram=(3, 5))`: `fit(texts)` computes `idf_` of shape `(dim,)` as above, where n-grams are all character $n$-grams for $n$ from `ngram[0]` to `ngram[1]` of `" " + " ".join(text.lower().split()) + " "`, bucket `zlib.crc32(gram.encode("utf-8")) % dim`, and $\mathrm{df}_j$ counts texts with at least one n-gram in bucket $j$; returns self. `transform(texts)`: a float array `(len(texts), dim)` with $w_j$ as above (0 for buckets that don't occur), each row scaled to unit norm (all-zero rows stay zero).

> **▶ build: stage 2.** implement `HashingEmbedder`.

### idea 3: exact nearest neighbours

retrieval is then a nearest-neighbour search: the $k$ chunks with the highest cosine similarity to the query embedding. with unit vectors that's one matrix product, `Q @ K.T`, and a partial sort per row (`np.argpartition` finds the top $k$ in linear time; only those $k$ need sorting). for a few million vectors on one machine, exact search is fast enough and has no failure modes, which makes it the right default and the ground truth for evaluating anything approximate.

> **spec.** `cosine_top_k(query, keys, k=5)`: NumPy; `query` `(d,)` or `(q, d)`, `keys` `(n, d)` (any lengths: normalize both; zero vectors have similarity 0, not NaN); returns `(indices, scores)`, each `(k,)` for a 1-D query or `(q, k)`, sorted best first; $k$ is capped at $n$.

> **🔮 predict:** with a perfect reader, retrieval that finds the right chunk for 69% of questions, how accurate can the answers be?

> **▶ build: stage 3.** implement `cosine_top_k`.

### idea 4: approximate search with an inverted file

at a billion vectors, scanning everything per query is too slow, and approximate nearest-neighbour (ANN) indexes trade a little recall for a lot of speed. the **inverted-file index** (IVF; Jégou et al., 2011; the backbone of FAISS, Johnson et al., 2019) clusters the collection with k-means into $L$ cells (`n_lists`) and stores each vector in the list of its nearest centroid. a query is compared with the $L$ centroids first, and then only with the vectors in its `n_probe` nearest lists. a query's true neighbours mostly fall in its own cell or adjacent ones, so a few probes recover most of them, at a cost of about $L + p\,n/L$ comparisons instead of $n$. the misses are neighbours that sit just across a cell boundary from the query. production systems add compression (product quantization) and graph indexes (HNSW; Malkov & Yashunin, 2018), but the recall–cost dial is the same.

with cosine similarity, normalize vectors before clustering (k-means on unit vectors approximates clustering by angle) and use the dot product to the centroids for assignment. an IVF index must be **trained** (its centroids fit) before vectors can be added.

> **spec.** `VectorIndex(dim, *, n_lists=0, n_probe=1, seed=0)`, cosine similarity throughout. `train(vectors)`: for `n_lists > 0`, fits `mlfs.unsupervised.KMeans(n_lists, rng=seed)` on the unit-normalized vectors and keeps the normalized centroids as `centroids_` (a no-op for `n_lists = 0`); returns self. `add(vectors, ids=None)`: stores unit-normalized vectors with `ids` (default: consecutive integers continuing from the current count, which can collide with custom ids added earlier: give ids for every call or for none), assigning each to its nearest centroid; `RuntimeError` if an IVF index isn't trained, `ValueError` for the wrong dimension. property `ntotal`. `search(queries, k=5)`: `(scores (q, k), ids (q, k))` best first; exact for `n_lists = 0`, otherwise only over the vectors in the `n_probe` lists whose centroids are most similar to the query; missing slots get score $-\infty$ and id $-1$. `n_probe` may be changed between searches.

> **▶ build: stage 4.** implement `VectorIndex`.

### idea 5: RAG, agents and test-time compute

**RAG** puts the retrieved passages into the prompt, numbered so the answer can cite them, followed by the question. the model's answer can only use what's there (or what it remembers), so answer accuracy is bounded by **recall@k**, the fraction of questions whose evidence was retrieved; the gap between them is the reader's errors, often caused by **distractors**: retrieved passages that look relevant but aren't (Liu et al., 2023, show readers also lose information in the middle of long contexts).

some questions can't be answered from one retrieval: "what's the population of the capital of X?" needs the capital first, and the population passage never mentions X. an **agent loop** (ReAct; Yao et al., 2023) alternates model calls and tool calls in a transcript: the model writes `CALL search(...)`, the harness runs the tool and appends `RESULT: ...`, and the model decides what to do next, until it writes `ANSWER: ...`. two rules make the loop safe: errors from tools (and malformed replies) go back to the model as text, never crash the loop; and a **step cap** stops a model that never answers. each hop succeeds with some probability, so a chain of hops succeeds with roughly their product.

**test-time compute**: sample several reasoning paths at temperature > 0 and take the **majority** final answer (self-consistency; Wang et al., 2023). if each sample is right with probability $p$ and wrong answers are scattered, voting amplifies $p$ (Condorcet's jury theorem); but if the model is more often wrong in the *same* way, voting amplifies the error. ignore samples whose answer couldn't be parsed.

> **spec.** `build_rag_prompt(question, passages)`: `"Context:\n[1] p1\n[2] p2\n...\n\nQuestion: {question}\nAnswer:"` (one line per passage, numbered from 1; with no passages, just `"Question: {question}\nAnswer:"`). `run_agent(llm, tools, task, *, max_steps=8)`: transcript starts as `"TASK: {task}\n"`; each step appends `llm(transcript).strip()` and a newline; if the reply contains `ANSWER: x`, return `(x stripped, transcript)`; if it contains `CALL name(arg)`, append `"RESULT: {tools[name](arg)}\n"`, or `"RESULT: error: ..."` for an unknown tool, a tool that raises, or a reply with neither; after `max_steps` model calls return `(None, transcript)`. `majority_vote(answers)`: the most common non-`None` answer, ties to the one seen first; `None` if there's none.

> **🔮 predict:** two-hop questions. single-shot retrieval with the whole question, or a loop that retrieves twice: how far apart are they?

> **▶ build: stage 5.** implement `build_rag_prompt`, `run_agent` and `majority_vote`.

## worked example

**chunking.** 10 words $w_0 \dots w_9$, chunk size 4, overlap 1: windows start at 0, 3, 6: $(w_0..w_3)$, $(w_3..w_6)$, $(w_6..w_9)$; the third reaches the end, so it's the last. with 11 words a fourth window $(w_9, w_{10})$ is needed.

**TF-IDF weights.** 3 texts; a bucket that occurs in all 3 has idf $\log(4/4) + 1 = 1$; one that occurs in 1 has $\log(4/2) + 1 = 1.69$; an n-gram occurring 3 times in a text has term weight $1 + \log 3 = 2.10$.

**IVF cost.** $n = 10^6$ vectors, $L = 1{,}000$ lists, $p = 8$ probes: $1{,}000 + 8 \cdot 1{,}000 = 9{,}000$ comparisons per query, 0.9% of exact search. the best $L$ for fixed $p$ minimizes $L + pn/L$: $L = \sqrt{pn} \approx 2{,}800$.

**majority vote.** 5 samples, each right with probability 0.6, two possible answers: the majority is right with probability $\sum_{k=3}^{5}\binom{5}{k}0.6^k 0.4^{5-k} = 0.346 + 0.259 + 0.078 = 0.683$. with $p = 0.4$ it's 0.317: voting makes a below-chance model worse. your functions must reproduce the hand examples.

## on paper

1. **the overlap guarantee.** show that with chunk size $C$ and overlap $O$, every span of at most $O + 1$ consecutive words lies entirely inside at least one chunk, and that $O + 2$ isn't guaranteed.

<details><summary>solution</summary>

chunk $i$ covers words $[is, is + C)$ with $s = C - O$; consecutive chunks share the $O$ words $[(i+1)s, is + C)$. a span $[a, a + m)$ lies inside chunk $i$ where $i = \lfloor a/s \rfloor$ if $a + m \le is + C$; the worst case is $a = is + s - 1$ (the last start covered only by chunk $i$ before chunk $i+1$ begins), giving $m \le C - s + 1 = O + 1$; but then chunk $i+1$ starting at $(i+1)s = a + 1$ doesn't contain $a$. a span of $O + 2$ words starting there fits in neither.

</details>

2. **recall caps accuracy.** let $R$ be the event that the evidence is among the retrieved passages and $A$ that the answer is right. assuming the reader can't answer without evidence (no memorized knowledge, no lucky guess), show $P(A) \le P(R)$ and write the gap in terms of the reader's accuracy given evidence.

<details><summary>solution</summary>

$P(A) = P(A \mid R)P(R) + P(A \mid \bar{R})P(\bar{R}) = P(A \mid R)P(R) \le P(R)$, since $P(A \mid \bar{R}) = 0$. the gap $P(R) - P(A) = (1 - P(A \mid R))P(R)$: the reader's error rate on questions whose evidence it was given. improving the reader closes the gap; only better retrieval raises the ceiling. (a model that also remembers facts breaks the assumption, which is how RAG systems can exceed retrieval recall on popular facts.)

</details>

3. **when does voting help?** with two possible answers, $n$ samples (odd), each correct independently with probability $p$, show that $P(\text{majority correct}) \to 1$ as $n \to \infty$ if $p > 1/2$ and $\to 0$ if $p < 1/2$. what changes with many possible wrong answers?

<details><summary>solution</summary>

the number of correct votes is Binomial($n, p$) with mean $np$ and standard deviation $\sqrt{np(1-p)}$; the majority is correct when it exceeds $n/2$. by the law of large numbers the fraction of correct votes tends to $p$, so it ends up above or below $1/2$ with probability tending to 1 (Condorcet). with many scattered wrong answers, the correct answer only needs a *plurality*: it wins whenever $p$ exceeds the probability of the most common wrong answer, even for $p < 1/2$. that's why self-consistency helps on math problems (wrong answers scatter) more than on yes/no questions.

</details>

## experiment

the lab builds a **synthetic** knowledge base (invented countries, their capitals, rulers, rivers and cities, mixed with lines of Shakespeare as filler; there's no document collection on this machine), chunks and embeds it with your code, and answers questions with a toy pattern-matching reader. it plots answer accuracy against recall@k for embedders of 32 to 4,096 buckets and $k$ from 1 to 10; measures your IVF index's recall and speed against exact search as the number of probed lists grows; and compares single-shot RAG with your agent loop (driven by a scripted policy standing in for an LLM) on two-hop questions.

## check your understanding

1. why embed chunks rather than whole documents?

<details><summary>answer</summary>

one vector per document averages everything it says, so a query about one paragraph matches it weakly; and a retrieved document may not fit in the prompt. chunks give precise matches and pieces that fit, at the cost of losing context across chunk boundaries (hence overlap, and sometimes retrieving neighbours of a hit).

</details>

2. your hashing embedder with 32 buckets retrieves almost nothing correctly. why?

<details><summary>answer</summary>

thousands of distinct n-grams share 32 buckets, so every text's vector looks like every other's: the hashing collisions destroy the distinctions between names that retrieval depends on. more buckets, fewer collisions, until the vocabulary of n-grams fits.

</details>

3. an IVF index with `n_probe = 1` misses many true neighbours. where are they?

<details><summary>answer</summary>

in neighbouring cells: the query is near the boundary of its own cell (in high dimensions most points are), and some of its nearest neighbours were assigned to the centroid on the other side. probing the next few cells finds them.

</details>

4. why must tool errors be returned to the model as text instead of raising?

<details><summary>answer</summary>

the model can recover from an error (fix the argument, try another tool) only if it sees it; an exception ends the episode and loses the transcript. the harness should also never crash because the model wrote something malformed.

</details>

5. an agent's first hop succeeds 71% of the time and its second 90%. what accuracy do you expect, and what does a five-hop task look like at the same rates?

<details><summary>answer</summary>

about $0.71 \times 0.9 = 0.64$. five hops at 90% each: $0.9^5 = 0.59$; at 80% each: 0.33. errors compound, which is why long agent chains need verification, retries, or more reliable steps.

</details>

6. self-consistency improves a model's math accuracy from 55% to 70%. would it help on a benchmark where the model is confidently wrong in the same way most of the time?

<details><summary>answer</summary>

no: if one wrong answer is more likely than the right one, the majority converges to it, and voting makes things worse. voting helps when errors are diverse.

</details>

## common bugs

- **a last chunk that drops the tail of the document**, or a loop that never stops when overlap ≥ chunk size.
- **Python's `hash()` in the embedder**: vectors change between processes.
- **dot products on unnormalized vectors**: long chunks win because they're long.
- **adding to an untrained IVF index**, or clustering unnormalized vectors for a cosine index.
- **an agent loop without a step cap**, or one that crashes on an unknown tool or a malformed reply.
- **counting unparseable answers as votes** in self-consistency.

## going deeper

- Lewis et al. (2020), "Retrieval-augmented generation for knowledge-intensive NLP tasks", *NeurIPS*. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al. (2020), "Dense passage retrieval for open-domain question answering", *EMNLP*. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Robertson & Zaragoza (2009), "The probabilistic relevance framework: BM25 and beyond", *Foundations and Trends in IR*. [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019)
- Weinberger et al. (2009), "Feature hashing for large scale multitask learning", *ICML*. [arXiv:0902.2206](https://arxiv.org/abs/0902.2206)
- Jégou, Douze & Schmid (2011), "Product quantization for nearest neighbor search", *IEEE TPAMI*. [doi:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
- Johnson, Douze & Jégou (2019), "Billion-scale similarity search with GPUs" (FAISS), *IEEE Trans. Big Data*. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
- Malkov & Yashunin (2018), "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs", *IEEE TPAMI*. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Yao et al. (2023), "ReAct: synergizing reasoning and acting in language models", *ICLR*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Schick et al. (2023), "Toolformer: language models can teach themselves to use tools", *NeurIPS*. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Wang et al. (2023), "Self-consistency improves chain of thought reasoning in language models", *ICLR*. [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- Liu et al. (2023), "Lost in the middle: how language models use long contexts", *TACL 2024*. [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)

## connections

- ← builds on: k-means (ch 11 · 01), embeddings and cosine similarity (ch 17 · 03), hashing and n-grams (topic 01), prompts and decoding (topics 03–05).
- → used in: the capstone can add retrieval to its assistant as a stretch goal; chapter 28 · 07 evaluates tool-using agents for safety; chapter 30's research projects often start from a RAG baseline.
