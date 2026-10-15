# implicit feedback

## why this matters

most systems never get star ratings. a music app knows which songs you played and how often; a shop knows what you clicked. this is **implicit feedback**, and it breaks topic 21's methods in three ways. there are **no negatives**: a song you never played might be one you'd hate or one you've never heard of. the signal is **positive and noisy**: 40 plays might be a loop while you slept. and the job is different: not "how many plays?" but, out of 600 (or 60 million) items, **put ten in front of this user** and have some be right, a ranking problem judged at the top of a list.

this topic builds the metrics that judge a top-10 list, two models designed for implicit data (confidence-weighted ALS and Bayesian personalized ranking), and an evaluation protocol. the lab then shows how easily it all misleads.

## warm-up: predict first

1. 99% of the cells in an interaction matrix are empty. a model predicts "no interaction" everywhere. what's its accuracy, and is it any use?
2. a model predicts play counts with a very low RMSE on held-out (user, item) pairs that were played. will it make good top-10 lists?
3. implicit ALS factorizes the *whole* matrix, including about 460,000 empty cells, but each user update costs about as much as in topic 21. how can that be?
4. AUC is the probability that a random relevant item is scored above a random irrelevant one. a model moves one relevant item from position 400 to position 2 out of 600. does AUC notice? does NDCG@10?
5. you hold out 20% of each user's interactions and recommend the most popular items to everyone. is that a strong or a weak baseline on this protocol, and is it a strong or weak recommender?

## prerequisites check

- [ ] you can fit biased matrix factorization by ALS and fold a user in with one ridge solve (→ ch 29 · topic 21)
- [ ] you know the sigmoid and the log-likelihood of logistic regression, and how to compute $\log\sigma$ stably (→ ch 08 · topic 04, ch 00 · topic 03)
- [ ] you know ROC curves, AUC as a ranking probability, precision and recall (→ ch 07 · topic 03)
- [ ] you can write minibatch SGD and know what happens when an index repeats in `x[idx] += ...` (→ ch 06 · topic 04, ch 00 · topic 02)

> **the data.** everything comes from a **simulator** (no interaction dataset is cached and nothing is downloaded). `make_interactions` reuses topic 21's world (rank-6 tastes, biases, power-law popularity; as there, item biases are built partly from popularity, so mean preference and log-popularity correlate at about 0.6), but users *consume* instead of rating: $n_u$ items each (median about 19), drawn $\propto \text{popularity}_i \times e^{2\,\text{pref}_{ui}}$, each played $1 + \text{Poisson}(e^{\text{pref}_{ui} - 3})$ times. 4.8% of the cells are filled, and the 10% most popular items get 47% of training interactions. the simulator also returns the true preference for every cell, so the lab can score a list against what a user would *actually* like. `split_interactions` (given) hides a random 20% of the interactions as the test set.

## core ideas

### idea 1: scoring a top-k list

a recommender produces, for each user, a score for every item. the user sees the **top $k$** of the items they haven't already consumed: training items are excluded, since recommending what someone just watched is useless (and would inflate every metric). the question is how many of those $k$ are "relevant", here meaning: in the user's held-out test set.

**recall@k** is the fraction of the user's relevant items that made the list. a user with 30 relevant items can't get more than 10 into a list of 10, so the standard version normalizes by the best achievable count:

$$\text{recall@}k(u) = \frac{|\text{top}_k(u) \cap \text{rel}(u)|}{\min(k, |\text{rel}(u)|)}.$$

(this "truncated" recall, used by Liang et al., 2018, reaches 1 for a perfect list. plain recall divides by $|\text{rel}(u)|$.) it ignores order within the list: a hit at position 1 counts the same as at position 10.

**NDCG@k** (normalized discounted cumulative gain, Järvelin & Kekäläinen, 2002) rewards hits near the top. a hit at position $j$ (counting from 1) earns $1/\log_2(j + 1)$: 1 at the top, 0.63 at position 2, 0.29 at position 10. the sum is the DCG; dividing by the DCG of a perfect list (all $\min(k, |\text{rel}|)$ hits at the top) gives a number in $[0, 1]$:

$$\text{NDCG@}k(u) = \frac{\sum_{j=1}^{k}\mathbb 1[\text{top}_k(u)_j \in \text{rel}(u)] / \log_2(j + 1)}{\sum_{j=1}^{\min(k, |\text{rel}(u)|)} 1/\log_2(j + 1)}.$$

**AUC** (ch 07) looks at the whole ranking instead: the probability that a random relevant item outscores a random irrelevant one, over all candidate items, with ties counting $\tfrac12$. it can be computed from ranks (the Mann–Whitney identity): if the $n_+$ relevant items have ranks $\rho_j$ among the $n$ candidates (rank 1 = lowest score),

$$\text{AUC} = \frac{\sum_j\rho_j - n_+(n_+ + 1)/2}{n_+\,n_-}.$$

AUC treats a swap at positions 400 and 401 exactly like a swap at positions 1 and 2. the user only ever sees positions 1 to 10 (warm-up 4).

users with nothing relevant have undefined recall and NDCG: report `nan` and leave them out of averages, never count them as 0 or 1.

> **spec.** `top_k(scores, k, *, exclude=None)`: `scores` `(n_users, n_items)`, `exclude` a bool mask of the same shape (True = never recommend). returns int64 `(n_users, k)`, best first, **ties to the lower item id** (a stable sort of the negated scores, with excluded items set to $-\infty$). `recall_at_k(top, relevant)` and `ndcg_at_k(top, relevant)`: `relevant` bool `(n_users, n_items)`; return float `(n_users,)` per the formulas above, `nan` where a user has no relevant items. `user_auc(scores, relevant, *, exclude=None)`: per-user AUC over the non-excluded items, ties $\tfrac12$, `nan` if a user has no relevant or no irrelevant candidates.

> **▶ build: stage 1.** implement `top_k`, `recall_at_k`, `ndcg_at_k` and `user_auc`.

### idea 2: implicit ALS: every cell, weighted by confidence

topic 21's model was fitted to observed cells only. with implicit data that's hopeless: every observed cell says "yes", so the best fit is "everything is a yes". you need the empty cells. Hu, Koren & Volinsky (2008) proposed a clean way to use them. split each cell into a binary **preference** and a **confidence**:

$$p_{ui} = \mathbb 1[r_{ui} > 0],\qquad c_{ui} = 1 + \alpha\,r_{ui},$$

where $r_{ui}$ is the play count. every empty cell is a weak "no" (confidence 1); every played cell is a "yes" whose confidence grows with the count. then factorize the whole preference matrix, weighting each cell's squared error by its confidence:

$$J = \sum_{u,i} c_{ui}\big(p_{ui} - x_u\cdot y_i\big)^2 + \lambda\Big(\sum_u\|x_u\|^2 + \sum_i\|y_i\|^2\Big).$$

the sum is over **all** $n_{\text{users}} \times n_{\text{items}}$ cells. as in topic 21, fixing the items makes each user's problem a (weighted) ridge regression with a closed form:

$$x_u = \big(Y^\top C^u Y + \lambda I\big)^{-1}Y^\top C^u p(u),$$

with $Y$ the item factors (all items), $C^u = \operatorname{diag}(c_{u1}, \dots, c_{un})$, and $p(u)$ the user's preference vector. done naively, that's an $n_{\text{items}} \times n_{\text{items}}$ diagonal matrix per user and $O(k^2 n_{\text{items}})$ work per user, which doesn't scale. the trick (warm-up 3) is that $C^u$ is the identity plus something sparse:

$$Y^\top C^u Y = \underbrace{Y^\top Y}_{\text{same for every user}} + \;Y^\top(C^u - I)Y,$$

and $C^u - I$ is zero except on the $n_u$ items the user consumed. compute $Y^\top Y$ once per sweep; for each user add $\sum_{i\in R(u)}(c_{ui} - 1)\,y_iy_i^\top$, a sum over a few dozen items. the right-hand side is sparse too: $p_{ui} = 0$ off $R(u)$, so $Y^\top C^up(u) = \sum_{i \in R(u)}c_{ui}y_i$. the cost per user is $O(k^2 n_u + k^3)$, independent of the catalogue size. items are solved the same way with $X^\top X$.

there are no biases in this model, and none are needed: popular items get large $\|y_i\|$ because they're "yes" for many users. $\alpha$ sets how much a play counts against all the empty cells; Hu et al. used $\alpha = 40$ on TV-watching data, and the lab lets you see what it does here.

> **spec.** `ImplicitALS(rank=16, *, alpha=10.0, lam=1.0, n_iters=10, rng=None)` stores its arguments. `fit(R)` takes a dense count matrix `(n_users, n_items)`; sets `user_factors_ = rng.normal(0, 0.1, (n_users, rank))`, then `item_factors_` the same way, from `as_rng(self.rng)`; runs `n_iters` iterations of **all users, then all items**, each solved exactly by the formula above (never building an $n \times n$ confidence matrix); appends `loss(R)` to `history_` after each iteration; returns `self`. `loss(R)` is $J$. `scores()` returns $XY^\top$.

> **🔮 predict:** with $\alpha = 0$, every cell has confidence 1. what is the model fitting then, and what will it recommend?

> **▶ build: stage 2.** implement `ImplicitALS`.

### idea 3: BPR: learn the order, not the values

implicit ALS still answers a regression question ("is this cell a 1 or a 0?") and hopes the ranking comes out right. Rendle, Freudenthaler, Gantner & Schmidt-Thieme (2009) asked for the ranking directly. their assumption is modest: a user prefers an item they consumed, $i$, over one they didn't, $j$. so train on **triples** $(u, i, j)$ and make the score difference

$$x_{uij} = \hat x_{ui} - \hat x_{uj},\qquad \hat x_{ui} = p_u\cdot q_i + b_i,$$

large and positive. treating $\sigma(x_{uij})$ as the probability that $u$ prefers $i$ to $j$, **BPR** maximizes the regularized log-likelihood

$$\sum_{(u,i,j)}\log\sigma(x_{uij}) \;-\; \lambda\,\|\Theta\|^2.$$

(the user bias would cancel in $x_{uij}$, so there is none; the item bias $b_i$ stays.) there are far too many triples to enumerate, so BPR samples them: take a positive $(u, i)$, draw $j$ uniformly from the catalogue, and take a gradient *ascent* step. with $g = \frac{d}{dx}\log\sigma(x) = \sigma(-x_{uij})$, the gradients are

$$\frac{\partial}{\partial p_u} = g\,(q_i - q_j),\qquad \frac{\partial}{\partial q_i} = g\,p_u,\qquad \frac{\partial}{\partial q_j} = -g\,p_u,\qquad \frac{\partial}{\partial b_i} = g,\qquad \frac{\partial}{\partial b_j} = -g,$$

and each parameter $\theta$ moves by $\eta\,(\text{gradient} - \lambda\theta)$ (the penalty's factor 2 is folded into $\lambda$, as with topic 21's SGD). $g$ is near 1 when the pair is ranked wrong and near 0 when it's comfortably right, so the updates concentrate on mistakes. a uniformly drawn $j$ is occasionally an item the user did consume; with 5% density that's rare, and the original paper samples $j$ uniformly too.

for speed you'll process a **minibatch** of triples at once, all gradients computed from the current parameters. careful: the same user or item can appear several times in a batch, and `P[u] += delta` with repeated `u` keeps only *one* of the updates. `np.add.at(P, u, delta)` adds them all.

why does this optimize ranking? for one user, the fraction of (consumed, unconsumed) pairs with $x_{uij} > 0$ is exactly that user's AUC on the training data; $\log\sigma$ is a smooth, concave stand-in for the step function $\mathbb 1[x > 0]$. so BPR is "maximize AUC, softly". remember idea 1's warning about what AUC doesn't see.

> **spec.** `BPR(rank=16, *, lr=0.05, lam=0.01, n_epochs=20, batch_size=256, rng=None)` stores its arguments. `fit(R)`: with `rng = as_rng(self.rng)`, set `user_factors_` then `item_factors_` from `rng.normal(0, 0.1, ...)` and `item_bias_` to zeros; the positives are `np.nonzero(R > 0)`. each epoch draws `rng.permutation(n_positives)`, cuts it into consecutive batches of `batch_size`, and for each batch draws `j = rng.integers(n_items, size=len(batch))` and calls `step`. `history_` gets the epoch's mean $\log\sigma(x_{uij})$ over all its triples. `step(u, i, j)` (int arrays of equal length) applies one ascent step to all the parameters it touches, from the old values, with repeated ids accumulating, and returns the batch's mean $\log\sigma(x_{uij})$ before the step. `scores()` returns $PQ^\top + b$.

> **▶ build: stage 3.** implement `BPR`.

### idea 4: the protocol, and the baseline that refuses to lose

to compare models you need a fixed protocol: hide a random 20% of interactions, fit on the rest, and for every user with at least one hidden item, recommend the top 10 *unseen* items and score them against the hidden ones. report the mean recall@10 and NDCG@10, and also the **catalogue coverage**: the fraction of all items that appear in at least one user's list. a model that recommends the same ten items to everyone has 1.7% coverage here, which is a warning sign whatever its accuracy.

the first model to run is **popularity**: score each item by how many users consumed it, and show everyone the same top 10. it uses nothing about the user. it's also notoriously hard to beat on this protocol (Cremonesi, Koren & Turrin, 2010), for a reason worth seeing clearly. the held-out items are a random sample of what users *consumed*, and consumption was driven partly by popularity: people come across popular items, so popular items fill the test set. a scorer that predicts exposure looks like a scorer that predicts taste (warm-up 5). Steck (2011) showed how much this inflates popularity-heavy models on real data. the lab measures it on the simulator, where the users' true favourites are known.

> **spec.** `popularity_scores(R)` returns a float `(n_users, n_items)` matrix whose every row is the number of users with `R > 0` in each column. `evaluate_topk(scores, train, test, *, k=10)` takes the top `k` with `exclude = train > 0`, scores them against `relevant = test > 0`, and returns a dict: `"recall"` and `"ndcg"` (means over users with at least one test item), `"coverage"` (distinct items in those users' lists / `n_items`), `"n_users"` (how many users were evaluated).

> **🔮 predict:** BPR's first few epochs: which parameters get a nonzero gradient from *every* triple, whatever the user? what will its top-10 lists look like before the user factors have grown?

> **▶ build: stage 4.** implement `popularity_scores` and `evaluate_topk`.

## worked example

**the list metrics.** scores $(0.1, 0.5, 0.5, 0.9)$ for one user: `top_k(..., 3)` is $(3, 1, 2)$, item 1 before item 2 because ties go to the lower id. if item 3 is excluded, it's $(1, 2, 0)$. now let the list be $(3, 1, 2)$ and the relevant items $\{1, 2\}$: two hits in a list of 3, at most $\min(3, 2) = 2$ possible, so recall@3 $= 1$. the hits sit at positions 2 and 3:

$$\text{DCG} = \frac{1}{\log_2 3} + \frac{1}{\log_2 4} = 0.631 + 0.5 = 1.131,\qquad \text{IDCG} = 1 + 0.631 = 1.631,\qquad \text{NDCG@3} = 0.693.$$

**AUC.** scores $(0.9, 0.1, 0.5, 0.5)$, relevant items 0 and 2. the four (relevant, irrelevant) pairs are $(0.9, 0.1)$ ✓, $(0.9, 0.5)$ ✓, $(0.5, 0.1)$ ✓ and $(0.5, 0.5)$, a tie worth $\tfrac12$: AUC $= 3.5/4 = 0.875$. by ranks: the ascending order is $0.1, 0.5, 0.5, 0.9$ with ranks $1, 2.5, 2.5, 4$ (ties averaged), so $(4 + 2.5 - 3)/(2 \cdot 2) = 0.875$. exclude item 3 and the only irrelevant score left is 0.1: AUC $= 1$.

**one BPR step.** $p_u = (1, 0)$, $q_i = (0.5, 0.5)$, $q_j = (-0.5, 1)$, $b_i = 0.2$, $b_j = -0.1$. then $x_{uij} = 1\cdot(0.5 + 0.5) + 0\cdot(0.5 - 1) + 0.2 + 0.1 = 1.3$, the step returns $\log\sigma(1.3) = -0.241$, and $g = \sigma(-1.3) = 0.214$. with $\eta = 0.1$, $\lambda = 0.01$: $p_u \to (1, 0) + 0.1\,(0.214\,(1, -0.5) - 0.01\,(1, 0)) = (1.0204, -0.0107)$ and $b_i \to 0.2 + 0.1\,(0.214 - 0.002) = 0.2212$.

**the implicit ALS trick.** rank 1, item factors $y = (1, 2, 3)$, a user who played item 1 twice, $\alpha = 1$, $\lambda = 1$. so $c = (1, 3, 1)$ and $p = (0, 1, 0)$. $Y^\top Y = 1 + 4 + 9 = 14$, and the correction $Y^\top(C^u - I)Y = (3 - 1)\cdot 2^2 = 8$, so the system matrix is $14 + 8 + 1 = 23$ (the dense $\sum_i c_iy_i^2 + \lambda = 1 + 12 + 9 + 1$ agrees). the right-hand side is $c_1y_1 = 6$, so $x_u = 6/23 = 0.261$. your code must reproduce the metric and BPR calculations exactly.

## on paper

1. **the implicit ALS update.** derive $x_u = (Y^\top C^uY + \lambda I)^{-1}Y^\top C^up(u)$ and show that computing it costs $O(k^2n_u + k^3)$ once $Y^\top Y$ is known.

<details><summary>solution</summary>

the terms of $J$ involving $x_u$ are $\sum_i c_{ui}(p_{ui} - y_i^\top x_u)^2 + \lambda\|x_u\|^2 = (p - Yx_u)^\top C^u(p - Yx_u) + \lambda\|x_u\|^2$. the gradient is $-2Y^\top C^u(p - Yx_u) + 2\lambda x_u$; zero gives the formula. cost: $Y^\top(C^u - I)Y = \sum_{i\in R(u)}(c_{ui} - 1)y_iy_i^\top$ is $n_u$ outer products, $O(k^2n_u)$; $Y^\top C^up = \sum_{i\in R(u)}c_{ui}y_i$ is $O(kn_u)$ (since $p_{ui} = 0$ elsewhere); solving the $k \times k$ system is $O(k^3)$. $Y^\top Y$ costs $O(k^2n_{\text{items}})$ but only once per sweep.

</details>

2. **the BPR gradient.** show that $\frac{d}{dx}\log\sigma(x) = \sigma(-x)$ and derive the gradient with respect to $q_j$. why is $g$ close to 0 for well-ranked pairs?

<details><summary>solution</summary>

$\log\sigma(x) = -\log(1 + e^{-x})$, so the derivative is $e^{-x}/(1 + e^{-x}) = 1/(e^x + 1) = \sigma(-x)$. by the chain rule, $\partial\log\sigma(x_{uij})/\partial q_j = \sigma(-x_{uij})\cdot\partial x_{uij}/\partial q_j = -\sigma(-x_{uij})\,p_u$. when $x_{uij} \gg 0$ (the pair is already in the right order by a wide margin), $\sigma(-x) \approx e^{-x} \approx 0$: no more effort is spent on it. this is also why uniformly sampled negatives become uninformative late in training: most random $j$ are easy.

</details>

3. **AUC is blind to the top.** one user, 600 candidates, 5 relevant. compute the change in AUC when one relevant item moves from position 400 to position 2 (positions counted from the top), and the change in NDCG@10 if the other four relevant items are outside the top 10.

<details><summary>solution</summary>

moving one relevant item up 398 places passes 398 items; at most all of them are irrelevant, so at most 398 more correctly ordered pairs out of $5 \times 595 = 2975$: AUC rises by at most $398/2975 \approx 0.13$, and by the same amount whether the item lands at position 2 or moves from 402 to 4 (out of sight). NDCG@10 goes from 0 to $(1/\log_23)/\text{IDCG}_5 = 0.631/2.948 = 0.21$. AUC treats the move as a modest improvement anywhere in the list; NDCG cares only about whether it reached the top 10, and where.

</details>

4. **why popularity wins offline.** suppose user $u$ consumes item $i$ with probability $\propto \pi_i\,e^{s\,\text{pref}_{ui}}$, with exposure weight $\pi_i$. the held-out set is a random 20% of consumed items. show that the probability item $i$ is in $u$'s test set is increasing in $\pi_i$ even for two items with the same preference, and explain what that means for a scorer that ranks by $\pi_i$.

<details><summary>solution</summary>

a held-out item had to be consumed first, and $P(\text{consumed}) \propto \pi_i e^{s\,\text{pref}_{ui}}$; hiding is independent of the item, so $P(i \in \text{test}_u) = 0.2\,P(\text{consumed})$, increasing in $\pi_i$ at fixed preference. ranking by $\pi_i$ (popularity is an estimate of it) therefore predicts membership of the test set without knowing anything about taste. the held-out protocol measures "predict what the user will consume", which mixes taste with exposure; a recommender that changes exposure (by recommending) breaks the very mechanism that made popularity look good.

</details>

## experiment

`lab.py` runs your metrics and models on the simulated world:

1. the observed-only model with the best RMSE, against popularity, implicit ALS and BPR, by recall@10 (and the 99%-accurate model that predicts nothing).
2. every model under the standard held-out protocol and against each user's true favourites.
3. AUC versus NDCG@10 over 100 epochs of BPR, plus a confidence-weight knob for implicit ALS.

## check your understanding

1. (warm-up 1.) why is accuracy useless for implicit feedback, and what does that say about choosing a metric?

<details><summary>answer</summary>

with 99% of cells empty, "nothing" is 99% accurate and recommends nothing. the metric must match the decision: the system shows $k$ items, so measure the quality of the top $k$ (recall@k, NDCG@k), among items the user hasn't seen.

</details>

2. why does topic 21's model, fitted to observed cells only, make bad top-10 lists even with a low RMSE?

<details><summary>answer</summary>

it learned to predict the value *given* that a cell is observed, and never saw an unobserved cell. it has no idea which items a user would never touch, so it happily ranks rarely seen items with optimistic predictions at the top. ranking needs a contrast between consumed and unconsumed items, which is exactly what implicit ALS (the zeros) and BPR (the sampled $j$) add.

</details>

3. what does $\alpha$ trade off in implicit ALS?

<details><summary>answer</summary>

how much an observed play outweighs an empty cell. small $\alpha$: all cells count nearly equally and the model mostly fits "consumed or not"; large $\alpha$: the fit concentrates on reproducing heavily played items and treats the empty cells as almost free, which spreads recommendations further from the popular head (higher coverage) but can lose accuracy if play counts are only loosely related to taste, as in this simulator.

</details>

4. why does BPR have an item bias but no user bias?

<details><summary>answer</summary>

$x_{uij} = \hat x_{ui} - \hat x_{uj}$: a user bias appears in both terms and cancels, so it gets zero gradient. the item bias doesn't cancel ($b_i - b_j$), and it learns a global item ranking: essentially popularity, since popular items are positives far more often.

</details>

5. (warm-up 5.) why can a model that is good for users score *worse* than popularity on the held-out protocol?

<details><summary>answer</summary>

the held-out items were chosen by users who mostly came across popular items, so the test set over-represents the head. a model that recommends a user's true favourites, many of them niche items they've never seen, gets few test hits for them, even though those are exactly the items the user would most enjoy. the lab's experiment 2 shows both effects.

</details>

## common bugs

- **not excluding training items.** symptom: recall and NDCG near 0 for good models, because their top 10 is filled with items the user already consumed. pass `exclude=train > 0`.
- **averaging over users with no test items.** symptom: metrics shift when you change the split seed in odd ways; NaN warnings. drop `nan` users, don't count them as 0.
- **`P[u] += delta` in a minibatch.** symptom: BPR learns, but more slowly than it should, and fails the repeated-index check; NumPy fancy-index assignment keeps only the last update per repeated index. use `np.add.at`.
- **a dense confidence matrix.** symptom: implicit ALS needs gigabytes for a modest catalogue, or takes minutes. never build `np.diag(c)`; use $Y^\top Y + Y^\top(C^u - I)Y$.
- **AUC with ties broken by position.** symptom: a constant scorer gets AUC far from 0.5. average the ranks of tied scores (ties count $\tfrac12$).
- **forgetting the IDCG normalization.** symptom: NDCG above 1, or users with many relevant items dominating the mean. divide by the DCG of a perfect list of length $\min(k, |\text{rel}|)$.

## going deeper

- Hu, Koren & Volinsky (2008), "Collaborative filtering for implicit feedback datasets", *ICDM*. confidence weighting and the $Y^\top Y$ trick. [doi:10.1109/ICDM.2008.22](https://doi.org/10.1109/ICDM.2008.22)
- Rendle, Freudenthaler, Gantner & Schmidt-Thieme (2009), "BPR: Bayesian personalized ranking from implicit feedback", *UAI*. [arXiv:1205.2618](https://arxiv.org/abs/1205.2618)
- Cremonesi, Koren & Turrin (2010), "Performance of recommender algorithms on top-N recommendation tasks", *RecSys*. RMSE-optimized models lose to simple ones at top-N; popularity is a strong baseline. [doi:10.1145/1864708.1864721](https://doi.org/10.1145/1864708.1864721)
- Steck (2011), "Item popularity and recommendation accuracy", *RecSys*. how popularity bias in the data inflates offline accuracy. [doi:10.1145/2043932.2043957](https://doi.org/10.1145/2043932.2043957)
- Järvelin & Kekäläinen (2002), "Cumulated gain-based evaluation of IR techniques", *ACM TOIS* 20(4). where NDCG comes from. [doi:10.1145/582415.582418](https://doi.org/10.1145/582415.582418)
- Liang, Krishnan, Hoffman & Jebara (2018), "Variational autoencoders for collaborative filtering", *WWW*. the truncated recall@k used here, and a strong evaluation protocol. [arXiv:1802.05814](https://arxiv.org/abs/1802.05814)

## connections

- ← builds on: biased MF and alternating ridge solves (ch 29 · topic 21), AUC and recall (ch 07 · topic 03), the sigmoid and logistic likelihood (ch 08 · topic 04), stable $\log\sigma$ (ch 00 · topic 03), minibatch SGD (ch 06 · topic 04).
- → used in: topic 23 replaces the free vectors with neural towers and the sampled negative $j$ with the other items in a batch (a sampled softmax), and evaluates retrieval with your recall@k.
