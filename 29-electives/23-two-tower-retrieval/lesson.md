# two-tower retrieval

## why this matters

topics 21 and 22 scored every item for every user. a real catalogue has ten million items and a request has maybe 100 milliseconds, so scoring everything with a rich model is out of the question. and a free vector per item only exists for items seen in training: a video uploaded five minutes ago has none.

production recommenders solve this with **two stages** (Covington, Adams & Sargin, 2016, describe YouTube's). a cheap **retrieval** stage narrows ten million items to a few hundred candidates; an expensive **ranking** stage orders those carefully. retrieval is usually a **two-tower** model: one neural network embeds the user, another embeds the item, and the score is their dot product. because the item side doesn't depend on the user, every item's embedding can be computed ahead of time, and "find the best items for this user" becomes "find the item vectors with the largest inner product with this query vector", a problem with fast approximate solutions.

this topic builds the towers, the in-batch softmax that trains them (and the **logQ correction**, without which it quietly learns the wrong thing), an inverted-file index, and a ranker. the lab assembles them into the block's project.

## warm-up: predict first

1. a two-tower model scores user $u$ and item $i$ as $f(u)\cdot g(i)$. why can't a model that sees $u$ and $i$ together (an MLP on their concatenated features, say) be used for retrieval over ten million items?
2. you train with a softmax over the items in the batch, using the other users' positives as negatives. popular items show up in many batches. does the model end up over- or under-scoring popular items?
3. an approximate index returns only 57% of the true top 100 neighbours. how much worse are the top-10 recommendations?
4. an item was added after training and has no interactions. what does an item tower that reads only the item's id produce for it? and one that also reads features?
5. the ranker is a more powerful model than the retriever. should it rescore 50 candidates, 500, or the whole catalogue?

## prerequisites check

- [ ] you can fit a factorization model and know its cold-start problem (→ ch 29 · topic 21)
- [ ] you can compute recall@k for top-k lists, and know why the held-out protocol favours popular items (→ ch 29 · topic 22)
- [ ] you can write a PyTorch training loop with `nn.Module`, `Adam` and the `mlfs.torchnn.MLP` you built (→ ch 15 · topic 07)
- [ ] you know softmax cross-entropy and why the gradient of a full softmax over a large vocabulary is expensive (→ ch 08 · topic 05, ch 17 · topic 03)
- [ ] you can run k-means and know what a centroid and a cell are (→ ch 11 · topic 01)

> **the data.** as in topics 21–22 the data come from a **simulator**, since no recommendation dataset is cached and nothing is downloaded. `make_catalog` builds 1,500 users and 3,000 items with the familiar ingredients (rank-8 tastes, biases, power-law exposure, consumption $\propto \text{popularity}\times e^{2\,\text{pref}}$), plus two new ones. **features**: 12 noisy linear views of each user's and item's hidden taste vector, the kind of signal genres or text embeddings carry. **a budget and a price** (standard normal, the last feature column on each side): an item priced above a user's budget loses 3 points of preference, so users almost never consume what they can't afford. that's a step function of price − budget: easy for a model that sees both numbers at once, awkward for a dot product of two separately computed vectors. the lab also removes every interaction of 10% of the items before training, to play new arrivals.

## core ideas

### idea 1: two towers

a **two-tower** (or dual-encoder) model has a user encoder $f$ and an item encoder $g$, and scores

$$s(u, i) = \frac{f(u)\cdot g(i)}{\tau}.$$

each tower here combines an **id embedding** (a free vector per user or item, like $p_u$ and $q_i$ in topic 21) with an **MLP on features**, adds them and normalizes:

$$f(u) = \operatorname{normalize}\big(E^{\text{user}}_u + \text{MLP}^{\text{user}}(x_u)\big),\qquad g(i) = \operatorname{normalize}\big(E^{\text{item}}_i + \text{MLP}^{\text{item}}(x_i)\big).$$

the id embedding memorizes what's specific to a well-known user or item; the feature MLP generalizes to ones it has seen little of, or never (warm-up 4). normalizing makes $f(u)\cdot g(i)$ a cosine in $[-1, 1]$; the **temperature** $\tau = 0.1$ scales it to logits in $\pm 10$.

the restriction is the point: user and item meet *only* in the final dot product. a model that mixes them earlier could be more accurate, but every (user, item) pair would need its own forward pass, ten million per request (warm-up 1). with two towers you embed every item once, offline, and at request time run only the user tower and search.

> **spec.** `Tower(n_ids, n_features, *, dim=32, hidden=64)` is an `nn.Module` with an `nn.Embedding(n_ids, dim)` (initialized $\mathcal N(0, 0.05^2)$) and an MLP `n_features → hidden → dim` (use your `mlfs.torchnn.MLP`). `forward(ids, features)` returns `F.normalize(embedding(ids) + mlp(features), dim=-1)`, shape `(B, dim)`. `TwoTower(n_users, n_items, user_dim, item_dim, *, dim=32, hidden=64, temperature=0.1)` has attributes `user_tower`, `item_tower` and `temperature`; `forward(user_ids, user_feats, item_ids, item_feats)` returns the `(B, B)` logits $f(u_b)\cdot g(i_{b'})/\tau$ between every user and every item of the batch.

> **▶ build: stage 1.** implement `Tower` and `TwoTower`.

### idea 2: the in-batch softmax and the logQ correction

training data are (user, item) pairs: user $u$ consumed item $i$. a natural objective is to make $i$ the most likely item for $u$ under a softmax over the whole catalogue,

$$-\log\frac{e^{s(u, i)}}{\sum_{j=1}^{n_{\text{items}}}e^{s(u, j)}},$$

but the denominator touches every item, and with ten million items so does every gradient step. the cheap substitute is the **in-batch softmax**: take a batch of $B$ pairs $(u_b, i_b)$, compute all $B \times B$ logits (one matrix product), and treat the batch's other items as the negatives for each user. the loss is cross-entropy with the diagonal as the targets. it costs one $B \times B$ matrix, and the negatives come for free, already embedded.

there's a catch (warm-up 2). the batch's items aren't a uniform sample of the catalogue: they're a sample of *interactions*, so item $j$ appears with probability proportional to its frequency $q_j$. popular items are negatives in almost every batch, rare ones almost never. a softmax trained against negatives drawn from $q$ doesn't learn $\log p(i\mid u)$; it learns, up to a constant per user,

$$s(u, i) \approx \log p(i \mid u) - \log q_i,$$

a pointwise mutual information that has had popularity divided out. that's why it under-scores popular items. the fix is the **logQ correction** of sampled softmax (Bengio & Senécal, 2008; for two towers, Yi et al., 2019): during training, subtract each *column's* log sampling probability from the logits,

$$\ell_{bb'} = s(u_b, i_{b'}) - \log q_{i_{b'}},$$

so that $s$ itself absorbs the $\log q$ term and estimates $\log p(i\mid u)$ again. at serving time you use the raw $s(u, i)$. two details. the correction is per column (per *item*): subtracting anything per row is a constant within each softmax and changes nothing. and a batch can contain the same item twice, for two different users; then for user $b$, the other copy of their own positive is not a negative, so mask it out ($-\infty$), never the diagonal.

here $q_j = \text{count}_j / \text{total}$ from the training pairs, with unseen items counted once so the log stays finite (Yi et al. estimate it on the fly for streaming data). the batch size $B$, which multiplies every $q_j$ equally, cancels in the softmax.

> **spec.** `item_log_frequency(item_ids, n_items)` returns a float32 tensor `(n_items,)` of $\log(\max(\text{count}_j, 1)/\text{len(item\_ids)})$. `in_batch_softmax_loss(logits, item_ids, *, log_q=None)`: `logits` `(B, B)`, `item_ids` `(B,)` int64; if `log_q` (shape `(B,)`, the log frequency of each column's item) is given, subtract it from every row; set entries where the column's item equals the row's item to $-\infty$, except the diagonal; return the mean cross-entropy with targets `0..B-1`.

> **🔮 predict:** without the correction, what will the median popularity of the model's top-10 items be, compared with the catalogue's median?

> **▶ build: stage 2.** implement `item_log_frequency` and `in_batch_softmax_loss`.

### idea 3: training the towers, and what features buy

the loop is ordinary supervised training. shuffle the training pairs, cut them into batches, compute the logits and the loss (with `log_q` looked up for the batch's items), and take an Adam step. after training, run each tower over all its rows once and **export** the embeddings: a user matrix and an item matrix, both with unit rows. everything downstream (exact search, the index, the ranker) works on those two arrays, not on the network.

two things to know before the lab. the training loss isn't comparable across objectives: the corrected and uncorrected models minimize different functions, so compare them on recall@k, as in topic 22. and an item that was never in a training batch still has its initial tiny random id embedding, so its output is essentially $\operatorname{normalize}(\text{MLP}(x_i))$, "an item with these features". if the features carry taste, a new item lands near the items it resembles and can be retrieved on day one: topic 21's cold-start problem, solved from the side.

> **spec.** `train_two_tower(model, user_ids, item_ids, user_feats, item_feats, *, epochs=5, batch_size=512, lr=3e-3, logq_correction=True, seed=0)`: `user_ids`, `item_ids` are the training pairs, `user_feats` `(n_users, ·)` and `item_feats` `(n_items, ·)` the full feature tables (ndarrays or tensors). use `torch.optim.Adam(model.parameters(), lr=lr)` and one `torch.Generator().manual_seed(seed)`; each epoch, `torch.randperm(n_pairs, generator=...)`, consecutive batches of `batch_size`, logits `model(u, user_feats[u], i, item_feats[i])`, loss `in_batch_softmax_loss` with `log_q = item_log_frequency(item_ids, n_items)[i]` when `logq_correction`. return the list of mean batch losses, one per epoch. `embed_all(model, user_feats, item_feats)` puts the model in eval mode and returns `(user_emb, item_emb)`, float32 ndarrays of every user's and every item's tower output (ids `0..n-1`), computed without gradients.

> **▶ build: stage 3.** implement `train_two_tower` and `embed_all`.

### idea 4: approximate nearest neighbours

with the item matrix $V$ ($n \times d$) exported, retrieval for a query vector $u$ is **maximum inner product search**: the $k$ largest entries of $Vu$. exactly, that's $O(nd)$ per query, a single matrix-vector product. at $n = 10^7$ and $d = 64$, that's $6 \times 10^8$ multiply-adds per request, too slow and too costly at scale. approximate indexes trade a little recall for a lot of speed.

the **inverted file** (IVF) is the simplest good one. offline, run k-means on the item vectors with $L$ clusters, and file each item under its nearest centroid: $L$ lists. at query time, find the `n_probe` centroids nearest the query and score only the items in those lists. with lists of size about $n/L$, a query costs $O(Ld + \texttt{n\_probe}\cdot (n/L)\,d)$: scanning the centroids plus scanning the probed lists. with one probe, $L + n/L$ is smallest at $L = \sqrt n$, about 55 lists for the lab's catalogue. you miss a true neighbour when it sits in a cell whose centroid isn't among the `n_probe` nearest to the query: near a cell boundary, typically. raising `n_probe` trades speed back for recall, continuously, from "one cell" to "exact".

for unit vectors, nearest by Euclidean distance and largest inner product coincide ($\|u - v\|^2 = 2 - 2u\cdot v$), so plain k-means and Euclidean probing are right. other families make other trades: locality-sensitive hashing uses random projections; graph indexes such as HNSW (Malkov & Yashunin, 2018) are the current speed–recall leaders; product quantization (Jégou, Douze & Schmid, 2011) compresses vectors so billions fit in memory. FAISS (Johnson, Douze & Jégou, 2019) implements them all.

> **spec.** `exact_search(queries, vectors, k)` returns `(ids, scores)`, `(Q, k)` each: the `k` largest inner products per query, best first, ties to the lower id. `IVFIndex(n_lists=32, *, n_probe=4, rng=None)` stores its arguments. `build(vectors)` runs your `KMeans(n_lists, n_init=1, max_iters=25, rng=rng)` (ch 11) and stores `centroids_` `(n_lists, d)` and `lists_`, a list of id arrays (the items in each cell); returns `self`. `search(queries, k)` probes the `n_probe` centroids in `centroids_` nearest (Euclidean) to each query, scores every item in those `lists_` by inner product, and returns the top `k` as `(ids, scores)`, ties to the lower id; slots without a candidate get id `-1` and score $-\infty$. it sets `last_n_scored_` to the mean number of items scored per query.

> **▶ build: stage 4.** implement `exact_search` and `IVFIndex`.

### idea 5: the second stage

retrieval's job is **recall**: get the items the user would like into the few hundred candidates. the ranker's job is **precision at the top**: order them so the best ten come first. seeing only a few hundred items per request, it can afford features of the user and the item *together*. in the simulator that's affordability: the towers see budget and price only in separate vectors, and a dot product struggles to express "a big penalty when price exceeds budget"; the ranker gets price − budget as one input. (the simulator's step is hard and the ranker gets exactly the right cross feature, so expect smaller gains on real data.)

here the ranker is a small MLP on four features of a pair: the retrieval score $f(u)\cdot g(i)$, price − budget, price and budget. its job is to order candidates, so train it on **candidate lists**: for each training pair $(u, i)$, the positive $i$ plus 20 **hard negatives**, unseen items the retriever put in $u$'s top 200, the very items it will sort at serving time. the loss is the softmax cross-entropy over the 21 items, with the positive as the target.

serving: retrieve $N$ candidates, drop consumed items and padding, score the rest with the ranker, return the top $k$. how big should $N$ be (warm-up 5)? big enough that the ranker's favourites are among the candidates; beyond that, extra candidates cost time and add nothing. the lab measures where that is, and finds a trap in one "obviously useful" ranker feature.

> **spec.** `ranker_features(user_emb, item_emb, user_feats, item_feats)` takes `B` aligned rows of each and returns float32 `(B, 4)`: `[u·v, price − budget, price, budget]` with `budget = user_feats[:, -1]`, `price = item_feats[:, -1]`. `Ranker(n_features=4, *, hidden=64)` is an MLP `n_features → hidden → hidden → 1`; `forward(x)` maps `(..., n_features)` to scores `(...)`. `train_ranker(ranker, X, *, epochs=5, batch_size=256, lr=2e-3, seed=0)`: `X` is `(n, L, n_features)` with list item 0 the positive; Adam, one seeded `torch.Generator`, `torch.randperm(n)` per epoch, consecutive batches, loss `F.cross_entropy(ranker(X[b]), zeros)`; returns the mean loss per epoch (weighted by batch size) and leaves the ranker in eval mode. `two_stage(user_ids, user_emb, index, score_fn, *, n_candidates=100, k=10, exclude=None)`: `cand, _ = index.search(user_emb[user_ids], n_candidates)`; `scores = score_fn(user_ids, cand)` `(Q, n_candidates)`; scores of padding (`-1`) and of items with `exclude[user, item]` set to $-\infty$; return the top `k` candidate ids per user by score `(Q, k)`, ties in candidate order.

> **🔮 predict:** $N$ counts candidates *before* seen items are dropped. what happens to a user who has consumed 200 items when $N = 20$?

> **▶ build: stage 5.** implement `ranker_features`, `Ranker`, `train_ranker` and `two_stage`.

## worked example

**the in-batch loss.** a batch of two pairs, items 7 and 3, with logits

$$\begin{bmatrix} 2 & 0 \\ 1 & 1\end{bmatrix}.$$

without correction, row 0's loss is $-\log\frac{e^2}{e^2 + e^0} = \log(1 + e^{-2}) = 0.127$, row 1's is $-\log\frac{e^1}{e^1 + e^1} = \log 2 = 0.693$, mean 0.410. now suppose item 7 is popular, $\log q_7 = \log 0.5$, and item 3 rare, $\log q_3 = \log 0.01$. subtracting per column gives

$$\begin{bmatrix} 2 + 0.693 & 0 + 4.605 \\ 1 + 0.693 & 1 + 4.605\end{bmatrix} = \begin{bmatrix} 2.693 & 4.605 \\ 1.693 & 5.605\end{bmatrix}.$$

row 0's loss becomes $\log(1 + e^{4.605 - 2.693}) = 2.05$: the rare item 3, as a negative, now weighs much more, because seeing it in the batch was a surprise. row 1's becomes $\log(1 + e^{1.693 - 5.605}) = 0.020$: user 1's positive, item 3, no longer has to beat the popular item 7 by much, since item 7 is a negative in most batches anyway. the mean is 1.03. if the batch had been items $(7, 7)$ for two different users, each row's off-diagonal entry would be masked and both losses would be exactly 0.

**ranker features.** user embedding $(1, 0)$ with budget 0.5, item embedding $(0, 1)$ with price 1.0: the features are $[0, 0.5, 1.0, 0.5]$, and the positive price − budget says the item is out of reach. a user $(0.6, 0.8)$ with budget $-1$ and an item $(0.6, 0.8)$ with price $-2$: $[1, -1, -2, -1]$. your code must reproduce the in-batch losses and the ranker features exactly.

## on paper

1. **what the uncorrected softmax learns.** suppose each training pair's negatives are $m$ items drawn independently with probabilities $q_j$, and the loss is the softmax over the positive and the negatives. argue that the population optimum satisfies $s(u, i) = \log p(i\mid u) - \log q_i + c_u$, and that subtracting $\log q$ from the logits during training removes the $-\log q_i$ term.

<details><summary>solution</summary>

sampled softmax estimates the normalizer $\sum_j e^{s(u, j)}$ from negatives drawn from $q$; an unbiased estimate weights each draw by $1/q_j$, i.e. uses $e^{s(u, j) - \log q_j}$. without the weights, the model is effectively fitting a softmax whose "catalogue" is reweighted by $q$: its distribution over items is $\propto q_j e^{s(u, j)}$. matching the data $p(i\mid u)$ requires $q_i e^{s(u, i)} \propto p(i\mid u)$, i.e. $s(u, i) = \log p(i\mid u) - \log q_i + c_u$. if you train with logits $s - \log q$, the same argument gives $s(u, i) - \log q_i = \log p(i\mid u) - \log q_i + c_u$, so $s(u, i) = \log p(i\mid u) + c_u$. (the argument is asymptotic in the number of negatives; see Bengio & Senécal, 2008.)

</details>

2. **IVF cost.** with $n$ items, $L$ lists of equal size and `n_probe` $= P$, show that the cost per query (in inner products) is $L + Pn/L$, minimized at $L = \sqrt{Pn}$ with cost $2\sqrt{Pn}$. how many inner products is that for $n = 10^7$, $P = 8$, against exact search?

<details><summary>solution</summary>

comparing the query with all centroids costs $L$; scanning $P$ lists of $n/L$ items costs $Pn/L$. the derivative $1 - Pn/L^2$ is zero at $L = \sqrt{Pn}$, where the cost is $\sqrt{Pn} + \sqrt{Pn}$. for $n = 10^7$, $P = 8$: $L \approx 8{,}900$ and about $17{,}900$ inner products, against $10^7$ for exact search, roughly 560 times fewer. (in practice lists aren't equal-sized, so the cost depends on which cells are probed, and $L$ is often set larger to keep popular cells small.)

</details>

3. **retrieval recall bounds the system.** the ranker reorders only the $N$ candidates. show that the number of relevant items in the final top 10 can't exceed the number among the $N$ candidates, and explain why that makes a high-recall index more important than a precise one.

<details><summary>solution</summary>

the final list is a subset of the candidates, so every relevant item in it was among the $N$ candidates: $|\text{top}_{10} \cap \text{rel}| \le |\text{cand}_N \cap \text{rel}|$. an item the retriever misses is gone for good, while an item it ranks badly *within* the candidates can still be rescued by the ranker. so retrieval should be tuned for recall at $N$ (how many good items make the shortlist), not for the order of its top 10, which the ranker will redo anyway.

</details>

## experiment

`lab.py` trains your towers on the simulated catalogue and asks:

1. what happens to recall and to popularity with and without the logQ correction.
2. how few items an approximate index can scan before recommendations get worse.
3. how well items with no interactions can be recommended, with features and with ids only.
4. **project:** a two-stage system: retrieval through your IVF index, then your ranker with a cross feature, compared with retrieval alone, across candidate counts, and with a popularity feature added.

## check your understanding

1. (warm-up 1.) why do two towers make retrieval over millions of items possible, and what modelling power do they give up?

<details><summary>answer</summary>

the item side doesn't depend on the user, so item embeddings are computed once and indexed; a request needs one user-tower pass and an approximate inner-product search. the price is that user and item interact only through a dot product: no feature crosses, no "if price exceeds budget" logic, nothing that needs both at once. that's what the ranker adds back.

</details>

2. why is the logQ correction applied during training but not when serving?

<details><summary>answer</summary>

it corrects a property of the *training procedure*, the non-uniform sampling of negatives. with it, $s(u, i)$ estimates $\log p(i\mid u)$ (up to a per-user constant), which is exactly what you want to rank by at serving time. subtracting $\log q$ again at serving would demote popular items for no reason.

</details>

3. (warm-up 3.) an index finds only 57% of the exact top 100. why can the top-10 recommendations still be almost as good as exact search?

<details><summary>answer</summary>

what it misses are mostly neighbours near cell boundaries, far from the query compared with its very nearest ones, i.e. the lower part of the top 100. the true top 10 sit close to the query and usually share its cell or a neighbouring one, so they're found. recall@100 against exact search punishes losing rank-80 neighbours that never mattered for a top-10 list.

</details>

4. why train the ranker on hard negatives from the retriever instead of random items?

<details><summary>answer</summary>

at serving time the ranker only ever sees retrieved candidates, which are all plausible. random negatives are mostly obviously bad; the ranker could separate them with the retrieval score alone and would learn nothing about ordering the top of a list. hard negatives match the serving distribution. (they also contain some false negatives, items the user would like but hasn't consumed, which is one reason rankers are trained on logged impressions in practice.)

</details>

## common bugs

- **subtracting `log_q` per row.** symptom: the "corrected" model behaves exactly like the uncorrected one. it's a column (item) correction: `logits - log_q[None, :]`.
- **masking the diagonal as a duplicate.** symptom: loss of $-\infty$ or NaN. a row's own positive is never masked; only *other* columns holding the same item are.
- **probing by inner product with unnormalized centroids.** symptom: the index favours large-norm cells. probe by Euclidean distance.
- **excluding seen items after taking the top $k$.** symptom: heavy users get lists shorter than $k$. drop them from the candidates before choosing the top $k$, and retrieve enough candidates to survive the filtering.

## going deeper

- Covington, Adams & Sargin (2016), "Deep neural networks for YouTube recommendations", *RecSys*. the candidate-generation + ranking architecture. [doi:10.1145/2959100.2959190](https://doi.org/10.1145/2959100.2959190)
- Yi, Yang, Hong, Cheng, Heldt, Kumthekar, Zhao, Wei & Chi (2019), "Sampling-bias-corrected neural modeling for large corpus item recommendations", *RecSys*. two towers, in-batch softmax, and the logQ correction with streaming frequency estimates. [doi:10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996)
- Bengio & Senécal (2008), "Adaptive importance sampling to accelerate training of a neural probabilistic language model", *IEEE Transactions on Neural Networks* 19(4). the importance-sampled softmax behind the correction. [doi:10.1109/TNN.2007.912312](https://doi.org/10.1109/TNN.2007.912312)
- Malkov & Yashunin (2018), "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs", *IEEE TPAMI*. HNSW. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- Jégou, Douze & Schmid (2011), "Product quantization for nearest neighbor search", *IEEE TPAMI* 33(1). IVF with compressed vectors (IVFADC). [doi:10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
- Johnson, Douze & Jégou (2019), "Billion-scale similarity search with GPUs", *IEEE Transactions on Big Data*. the FAISS library. [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

## connections

- ← builds on: factorization and cold start (ch 29 · topic 21), top-k metrics and the popularity bias of offline evaluation (ch 29 · topic 22), `mlfs.torchnn.MLP` and PyTorch training (ch 15 · topic 07), softmax cross-entropy (ch 08 · topic 05), negative sampling in skip-gram (ch 17 · topic 03), k-means (`mlfs.unsupervised.KMeans`, ch 11 · topic 01).
- → used in: the same dual-encoder + index pattern is dense retrieval for retrieval-augmented generation (embedding documents instead of items), and contrastive objectives with in-batch negatives return in self-supervised learning (CLIP-style training is a two-tower model with an in-batch softmax).
