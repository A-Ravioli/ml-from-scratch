# two-tower retrieval · lab notes

numbers below are from the reference solution with the lab's seeds (world seed 0, split seed 1), on a CPU with 4 torch threads; your code gives the same numbers if it follows the spec's order of random draws, and similar ones otherwise. timings depend on the machine. the whole lab was rerun with world seed 1 (split seed 2); every qualitative result below holds there too, and the seed-1 numbers are quoted where they help.

the world: 38,363 interactions; 28,154 train and 7,027 held-out with warm items, plus 3,182 with the 300 cold items. unaffordable items are 49% of all user–item pairs but only 1.2% of the users' true favourites.

## experiment 1: what the logQ correction is for

**what people usually predict:** a small difference either way; maybe the uncorrected model is a bit more popularity-biased, since popular items are seen more often in training.

**what happens:**

| model | final training loss | held-out recall@10 | held-out recall@100 | true-favourite recall@10 | top-10 items from the head |
|---|---|---|---|---|---|
| popularity | — | 0.278 | 0.630 | 0.113 | 100% |
| two-tower, logQ off | 4.69 | **0.028** | 0.223 | 0.114 | 11% |
| two-tower, logQ on | 4.75 | **0.352** | 0.731 | **0.276** | 93% |

without the correction the model is a disaster: its held-out recall@10 is **0.028**, a tenth of popularity's and a twelfth of the corrected model's, and only 11% of its recommendations come from the 10% most popular items, which is what you'd get by picking items at random. it's also no better than popularity against the true favourites. its training loss is *lower* than the corrected model's.

**why:** in-batch negatives are drawn in proportion to item frequency $q_j$, so a softmax over them learns $s(u, i) \approx \log p(i\mid u) - \log q_i$ (on paper 1): every item's score has its popularity divided out. that's a measure of *surprise* (how much more this user likes the item than everyone does), and ranking by it pushes the obscure items to the top, the opposite of the popularity bias the question expected. with the correction, the logits fed to the softmax are $s - \log q$, so $s$ itself absorbs the popularity term and estimates $\log p(i\mid u)$, which is what you want to rank by. the training losses can't be compared: the two models minimize different functions (one has the $-\log q$ shift inside), and the lower one belongs to the useless model. in this world consumption is driven by both exposure and taste, so the corrected model's 93% head share isn't a flaw: the users' true favourites lean popular too: 55% of them come from the head, against 10% by chance (the item biases correlate with popularity). the corrected model is used from here on.

## experiment 2: how approximate can the index be?

**what people usually predict:** end-to-end quality degrades roughly in line with the index's recall: if the index misses 40% of the true top 100, the recommendations get noticeably worse.

**what happens:**

| n_probe | index recall@100 vs exact | catalogue scored | end-to-end held-out recall@10 |
|---|---|---|---|
| 1 | 0.231 | 1.3% | 0.272 |
| 2 | 0.394 | 2.6% | 0.334 |
| 4 | 0.574 | 5.3% | 0.350 |
| 8 | 0.756 | 11.3% | 0.354 |
| 16 | 0.908 | 24.1% | 0.354 |
| 55 (exact) | 1.000 | 100% | 0.352 |

with `n_probe = 4` the index scores 5.3% of the catalogue and returns only **57%** of the exact top 100, yet end-to-end recall@10 is **0.350 against 0.352** for exact search. from `n_probe = 8` the difference is noise. only a single probe (1.3% of the catalogue) costs real accuracy. with world seed 1 it's the same story: `n_probe = 4` finds 56% of the exact top 100 and gets 0.383 against 0.372 for exact search. (search times, 0.06–0.8 s for 1,500 queries, mostly measure a Python loop and aren't meaningful at this size; the number of items scored is.)

**why:** the index loses neighbours that sit in cells whose centroids aren't among the nearest to the query, which mostly means neighbours that are relatively far from it: the lower part of the top 100. the items that end up in a top-10 list are the very nearest ones, and they almost always share the query's cell or an adjacent one. recall@100 against exact search penalizes losing rank-80 neighbours that would never have been shown. the lesson for practice: measure an index by the downstream metric at the depth you actually use, not by its recall at some large $k$, and remember that exact search is itself only a proxy (on held-out recall, exact is no better than `n_probe = 8`).

## experiment 3: items nobody has seen

**what people usually predict:** both models are bad at items without interactions; features help a bit.

**what happens:**

| model | cold items: recall@10 among cold items | warm items: held-out recall@10 |
|---|---|---|
| features + ids | **0.269** | **0.352** |
| ids only | 0.033 | 0.326 |
| random | 0.028 | 0.002 |

for the 300 cold items, the feature-based model's recall@10 (**0.269**) is close to what it achieves on *warm* items, about 9 times random. the ids-only model is at random (0.033 vs 0.028). and the features help warm items too: 0.352 against 0.326.

**why:** a cold item's id embedding never received a gradient, so it's still the tiny random vector it was initialized with ($\mathcal N(0, 0.05^2)$ per coordinate). in the ids-only model that's all there is, so its normalized embedding points in a random direction. in the feature model, the MLP's output dominates, and the item's embedding is "an item with these features", placed near warm items with similar features, which is where the users who like such items are. this is the same idea as topic 21's fold-in (place a new row using what you know about it), but it works with *zero* interactions because it uses side information instead. on warm items, the features act as a shared prior that helps items with few interactions, whose id embeddings are poorly estimated.

## experiment 4 (project): a two-stage retrieve-then-rank recommender

**what people usually predict:** the ranker helps a little; more candidates is always better (or at least never worse); a popularity feature can only help, since popularity predicts interactions so well.

**what happens:**

| system | items the ranker scores per user | held-out recall@10 | true-favourite recall@10 | unaffordable share of the top 10 | top 10 from the head |
|---|---|---|---|---|---|
| retrieval only (top 10 of 200 candidates) | — | 0.354 | 0.270 | 15.6% | 94% |
| two-stage, N = 20 | 20 | 0.321 | 0.247 | 8.9% | 95% |
| two-stage, N = 50 | 50 | 0.379 | 0.304 | 4.3% | 92% |
| two-stage, N = 100 | 100 | **0.381** | 0.308 | 4.0% | 92% |
| two-stage, N = 200 | 200 | 0.381 | 0.308 | 4.0% | 92% |
| two-stage, N = 500 | 500 | 0.381 | 0.308 | 4.0% | 92% |
| rank every warm item | 2700 | 0.378 | **0.320** | 3.6% | 90% |
| two-stage, N = 200, ranker also sees popularity | 200 | 0.324 | 0.314 | 7.4% | 83% |

with training seed 0 the ranker (training: about 3 s, list loss 1.98 → 1.41) improves held-out recall@10 from **0.354 to 0.381** and true-favourite recall from **0.270 to 0.308**, and it cuts the share of unaffordable recommendations from 15.6% to 4.0%. the gain is complete by **N = 50–100**: going to 500 changes nothing, and scoring all 2,700 warm items (27 times the work of N = 100) is no better on held-out items and slightly better on true favourites. N = 20 is *worse than retrieval alone*. giving the ranker each item's popularity drops held-out recall to **0.324**, well below the same ranker without it, while true-favourite recall stays at 0.314, and the share of head items in the lists falls from 92% to 83%.

seed 0 is the luckiest run. retraining everything (towers, index, hard negatives, ranker) with three training seeds, with 95% paired-bootstrap intervals over users:

| training seed | retrieval only | two-stage, N = 100 | gain [95% CI] | two-stage + popularity, N = 200 |
|---|---|---|---|---|
| 0 | 0.354 | 0.381 | +0.027 [+0.019, +0.034] | 0.324 |
| 1 | 0.347 | 0.360 | +0.013 [+0.005, +0.021] | 0.321 |
| 2 | 0.360 | 0.370 | +0.010 [+0.001, +0.018] | 0.322 |

the mean held-out gain is **+0.017** (range +0.010 to +0.027), every interval above zero but some barely. a reviewer's runs on other worlds (world seeds 2 and 3) gave +0.020 and +0.019. so: a small, real gain of one to three points of recall, not a dramatic one. the popularity-fed ranker is below the plain ranker on every seed (and on other worlds), though not always below retrieval alone (on world seed 3 it was 0.395 vs 0.391). with world seed 1 the qualitative pattern repeats: unaffordable share 13.9% → 6.0%, N = 20 below retrieval, saturation by N = 50, and the popularity feature lowering held-out recall with the head share falling from 92% to 86%.

**why:**

- **what the ranker adds, and why that's specific to this world.** the gain exists because the simulator has a *hard* step (price above budget costs 3 preference points) and the ranker is handed that cross feature, price − budget, exactly; on data without such a cross effect, or where the ranker sees nothing the towers don't, expect less. the towers see budget and price, but in separate vectors whose only meeting point is a dot product, and a dot product can't easily express "a large penalty whenever price exceeds budget". the ranker sees price − budget directly and learns the step, which you can read off the unaffordable column: 15.6% → 4.0% (the true favourites are 1.2% unaffordable). every unaffordable item it removes makes room for one the user might actually take.
- **why N saturates.** retrieval's job is recall: the items the ranker wants to put on top must be in the shortlist (on paper 3). the ranker mostly reorders the retriever's top few dozen (it starts from the retrieval score and corrects it), so beyond about 50–100 candidates nothing new reaches the top 10. that's the economics of two-stage systems: a cheap stage with high recall and a precise stage that only looks at a small $N$.
- **why N = 20 is worse.** $N$ counts candidates before seen items are dropped. a heavy user's top 20 retrieved items are mostly things they've already consumed (the towers learned those best). measured: after filtering, users keep 11 of their 20 candidates on average, 36% of users keep fewer than 10, and users with at least 50 training items keep about 3. retrieval-only had 200 candidates to filter from. in production you'd retrieve $N$ plus the number of items to be filtered, or filter inside the index.
- **why popularity hurts.** look at how the ranker is trained: positives are training items, which the towers were fitted on, so their retrieval scores are already high; the hard negatives are items the towers ranked highly *without* the user consuming them, which are disproportionately popular. within those lists, popularity is a sign of a *negative*, and the ranker learns to demote popular items (head share 92% → 83%). at serving time that's wrong for the held-out protocol, which rewards popularity (topic 22), but not for the users' true favourites, which is why one metric falls and the other doesn't. this is a train/serve mismatch: the ranker's training data aren't distributed like its serving data. real systems train the ranker on logged impressions (what was actually shown, and whether it was clicked) from a period the retriever wasn't trained on, and they audit features like popularity for exactly this kind of leakage.
