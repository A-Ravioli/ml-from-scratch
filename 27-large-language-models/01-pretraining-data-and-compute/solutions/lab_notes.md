# pretraining data & compute · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, 4 threads), tiny Shakespeare from the cache, other jobs sharing the machine. the web crawl is **synthetic** (`make_crawl`, seed 0): its proportions of copies, templates and junk are choices made for this lab, not measurements of the real web. offline (`MLFS_OFFLINE=1`) the source text is a synthetic word salad of very short words, so the quality filter's word-length rule rejects most of it; the numbers below need the real text.

## experiment 1: how much of a crawl survives?

**what people predict.** that exact deduplication removes most copies, MinHash a few more, and that most of the crawl survives (people guess 70–90%).

**what happens.**

| stage | article | exact copy | near copy | template | junk | total |
|---|---|---|---|---|---|---|
| crawl | 1,333 | 443 | 470 | 415 | 339 | 3,000 |
| quality filter | 1,324 | 440 | 469 | 415 | 0 | 2,648 |
| exact dedup | 1,324 | 67 | 424 | 411 | 0 | 2,226 |
| MinHash dedup | 1,289 | 23 | 27 | 1 | 0 | **1,340** |

45% of the pages survive. the filter catches all 339 junk pages (symbols 152, repeated lines 112, non-alphabetic 88, too short 21; a page can fail several rules) and nine short articles. exact hashing removes 376 of the 443 mirrors and almost no near copies. MinHash removes most of the rest: 397 of 424 near copies, 410 of the 411 shop pages, and 35 original articles.

**why.** exact hashing after normalization catches mirrors that differ in case or spacing, but one changed word or a different site header makes a different hash. the 67 "exact copies" that survive it are copies of articles whose original page carried site boilerplate: the text is the same, the page isn't. MinHash sees the shared shingles. the shop pages are a single template with three slots. pairs that share an item or a city are above the 0.8 threshold (Jaccard up to 0.88), others only at 0.66–0.7, but connected components chain them all together, so the whole family collapses to one page: for a language model that's what you want (one template a hundred times teaches nothing new), though the product names and prices are lost with them. the 35 originals removed are passages that overlap other passages: `make_crawl` cuts articles at random starting lines, and some cuts land a few lines apart. the 50 copies that survive MinHash are short: the articles they copy have a median of 74 words (113 for all articles), and their Jaccard to the first page of the same article is 0.61–0.80. a 30-word header and footer is a large fraction of a short page, which pushes it below the 0.8 threshold. real pipelines strip boilerplate before deduplicating for exactly this reason.

## experiment 2: the S-curve

**what people predict.** that with a threshold of 0.8 the LSH curve rises around 0.8.

**what happens.** the measured candidate rates (200 seeds of your signatures per point) follow $1 - (1 - s^r)^b$ to within sampling error for all three splits. the threshold estimates $(1/b)^{1/r}$ are 0.12 for 64 bands of 2, **0.42 for 32 bands of 4** (the default) and 0.71 for 16 bands of 8; at those points each curve is at about $1 - 1/e = 0.63$, and it crosses one half a little lower (0.10, 0.38 and 0.67). with the default, pairs at Jaccard 0.5 are candidates 87% of the time.

**why.** the default is deliberately generous: a near-duplicate at 0.8 is missed with probability $5 \times 10^{-8}$, at the price of many false candidates between 0.4 and 0.8, which the verification step then discards by comparing full signatures. the expensive error in deduplication is a miss (a copy left in), so the curve is placed low and verification restores precision. to target a threshold of 0.5 you'd want the steep part just below 0.5, e.g. 64 bands of 2 (too loose: almost everything above 0.2 collides) or 32 bands of 4. fewer rows per band move the curve left and make it flatter; more rows make it sharper and move it right. with only 60 seeds per point an earlier version of this plot showed every point drifting the same way from the theory: the points share the same hash functions, so their errors are correlated, and 60 seeds were too few to see the curve.

## experiment 3: does deduplication matter for training?

**what people predict.** that the deduplicated model is a little better or that it makes no difference; few expect the raw model to have the lower training loss.

**what happens.** two GPTs (2 layers, width 128, characters), 800 steps each, same seed:

| training corpus | characters | train loss | held-out loss | 10 most-copied articles | 100 rarely copied articles |
|---|---|---|---|---|---|
| with duplicates | 1,683,460 | **1.438** | 1.962 | **1.441** | 1.790 |
| deduplicated | 928,057 | 1.561 | **1.875** | 1.671 | **1.698** |

the most-copied article appears 164 times among the filtered pages. the raw model's training loss is 0.12 nats lower, and its held-out loss is 0.09 nats *higher*. it predicts the most-copied articles far better than the rare ones (1.44 against 1.79), while the deduplicated model treats them alike (1.67 against 1.70). seeds 1 and 2, run separately, give the same picture: held-out loss 1.990 and 1.953 (raw) against 1.898 and 1.875 (deduplicated); most-copied against rare articles 1.46/1.82 and 1.43/1.77 (raw), 1.69/1.72 and 1.68/1.69 (deduplicated). over three seeds, deduplication improves held-out loss by 0.08–0.09 nats, a gap several times the seed-to-seed spread.

**why.** the training losses are measured on different data: the raw corpus is full of text the model has seen many times, which is easy to predict once memorized, so its low training loss measures memorization, not language modelling. the same 1.6M characters of training buy the raw model about 0.97 passes over a corpus that is 45% repeats and templates, and the deduplicated model 1.77 passes over unique text (worth 1.75 epochs of fresh data by `effective_tokens`). the raw model spends capacity on reproducing the viral articles and the shop template, which shows up as the gap between its two "articles" columns, and the held-out loss pays for it. this is the small-scale version of Lee et al. (2022), whose deduplicated models emitted memorized text about ten times less often at equal or better perplexity, and of Carlini et al. (2023), who found that memorization grows with the number of times a sequence is duplicated.
