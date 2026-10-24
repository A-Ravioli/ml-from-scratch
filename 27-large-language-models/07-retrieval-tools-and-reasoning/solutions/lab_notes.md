# retrieval, tools & reasoning · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU), one run. the knowledge base is **synthetic** (`make_kb`: 1,500 invented countries in pairs with similar names, their capitals, rulers, rivers and founding years, plus a document per capital with its population, padded with lines of tiny Shakespeare), 3,000 documents cut into 7,239 chunks. the reader (`read_capital`) and the agent's policy are scripted stand-ins for a language model; the retrieval, the index and the loop are your code. offline the lab builds 300 countries and the numbers differ.

## experiment 1: retrieval caps answer quality

**what people predict.** that a good reader can make up for a weak retriever, and that more retrieved chunks always help.

**what happens.**

| buckets | k | recall@k | answer accuracy | wrong answer from a cousin |
|---|---|---|---|---|
| 32 | 1, 3, 10 | 0.00 | 0.00 | 0.00–0.03 |
| 256 | 1 | 0.42 | 0.42 | 0.28 |
| 256 | 3 | 0.60 | 0.48 | 0.36 |
| 256 | 10 | 0.77 | 0.53 | 0.42 |
| 4,096 | 1 | 0.69 | 0.69 | 0.29 |
| 4,096 | 3 | 0.94 | 0.69 | 0.31 |
| 4,096 | 10 | **1.00** | **0.69** | 0.31 |

every point lies on or below the diagonal. with 32 buckets nothing is retrieved and nothing is answered. with 4,096 buckets and top-10, the right chunk is retrieved for every question, yet accuracy stays at 0.69, exactly the top-1 recall: for 31% of the questions the reader answers with the capital of the country's similarly named cousin.

**why.** a reader can only use what it's given, so recall@k is a ceiling (on paper 2). with 32 buckets, the tens of thousands of distinct character n-grams collide so much that every chunk's vector looks alike, and the country name, the only distinguishing signal, is lost. above the ceiling, the gap is reader error, here entirely caused by **distractors**: every country has a cousin whose name extends it (Selliaven, Selliavenia), whose chunks share almost all the query's n-grams and often rank above the right one; the reader takes the first plausible match, so once the cousin's chunk is ranked first, retrieving the right chunk further down doesn't help. raising $k$ adds recall but not accuracy, and with the weaker 256-bucket embedder it adds more distractors than evidence. a real LLM reader would compare the names more carefully than "same first five letters", and would be more robust to this particular trap, but it can't beat the recall ceiling either, and LLMs have their own distractor problems (Liu et al., 2023).

## experiment 2: approximate nearest neighbours

**what people predict.** that probing 6% of the index finds most of the top 5, and that the approximate index is much faster.

**what happens.**

| lists probed (of 64) | recall@5 vs exact | fraction probed | time for 300 queries |
|---|---|---|---|
| 1 | 0.18 | 0.016 | 138 ms |
| 2 | 0.35 | 0.031 | 374 ms |
| 4 | 0.65 | 0.062 | 840 ms |
| 8 | 0.85 | 0.125 | 1,282 ms |
| 16 | 0.94 | 0.250 | 1,680 ms |
| 64 | 1.00 | 1.000 | 3,999 ms |

exact search with your `VectorIndex` took 4,076 ms; `cosine_top_k`, one matrix product, took **160 ms**. building the IVF index (k-means on half the chunks) took 1.9 s.

probing 4 of 64 lists finds 65% of the true top 5; 16 lists, 94%. every IVF setting except the single-probe one is slower than the plain matrix product.

**why.** a query's neighbours concentrate in the cells nearest to it, so recall rises steeply with the first few probes and then flattens; the misses are neighbours just across a cell boundary, and in 4,096 dimensions most points are near a boundary, which is why one probe is so poor here. as for speed: a single matrix product over 7,239 × 4,096 runs at the machine's full arithmetic rate, while the index visits candidates with Python-level loops, `np.isin` masks and small matrix–vector products per query. IVF wins when the collection is large enough that scanning it dominates (millions of vectors), and when the probed lists are stored contiguously and searched in batches, as FAISS does in C++. for a few thousand chunks, exact search is both faster and exact; the build cost buys nothing at this scale.

## experiment 3: one retrieval, or an agent loop?

**what people predict.** that single-shot retrieval sometimes gets lucky, and that the agent roughly doubles accuracy.

**what happens.**

| method | accuracy | tool calls per question |
|---|---|---|
| single-shot RAG (top 3) | 0.07 | 1 retrieval |
| agent loop (search, search, answer) | **0.65** | 2.0 |

the agent's first hop alone (finding the capital) succeeds for 71% of the questions.

**why.** the population sits in the capital's document, which is found by the capital's name, and the question doesn't contain that name: retrieving with the question finds the country's chunks (and its cousin's), almost never the city's. the 7% that succeed are cases where the city document happened to rank high. after its first step the agent knows the capital's name and can search for it directly. its accuracy is about (first hop) × (second hop) = 0.71 × ~0.92; each hop multiplies in its own failure rate, so long chains need reliable steps, verification or retries. a real LLM would replace the scripted policy (deciding what to search for and when to stop) and bring new failures: malformed calls, giving up early, never answering (hence the step cap), or answering from memory instead of the retrieved text.
