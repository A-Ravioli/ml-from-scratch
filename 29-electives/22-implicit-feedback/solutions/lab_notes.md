# implicit feedback · lab notes

numbers below are from the reference solution with the lab's seeds (world seed 0, split seed 1). your code gives the same numbers if it draws its random numbers in the order the spec gives, and similar ones otherwise. the comparisons were rechecked on other world seeds (1 and 2, split seeds 2 and 3; a reviewer added 3 and 4). the popularity and observed-only flips and the BPR AUC/NDCG divergence hold on all of them; where a difference is within seed noise, the notes say so.

the world: 23,188 interactions (18,535 train, 4,653 test) from 800 users on 600 items, 4.8% of cells; the top 10% of items account for 47% of training interactions.

## experiment 1: the best RMSE, the worst list

**what people usually predict:** the model with the lowest error is at least decent at ranking; popularity, which ignores the user, comes last among the real models.

**what happens:**

| scorer | RMSE on held-out log plays | recall@10 | NDCG@10 |
|---|---|---|---|
| random | — | 0.025 | 0.019 |
| popularity | — | 0.380 | 0.349 |
| observed-only MF | **0.534** | 0.128 | 0.107 |
| implicit ALS (α = 1) | — | 0.443 | 0.408 |
| BPR (30 epochs) | — | **0.450** | **0.419** |

predicting "no interaction" everywhere is **99.0%** accurate on the unseen cells. the observed-only model predicts held-out play counts well (RMSE 0.53 in log units; the other models don't even produce counts), yet its recall@10 is a third of popularity's. on world seeds 1 and 2: recall 0.161 and 0.142 for it, against 0.395 and 0.377 for popularity and 0.44–0.47 for implicit ALS and BPR.

**why:** RMSE on observed cells answers "given that the user played this item, how much?" a top-10 list needs "which items, out of 600, will this user play at all?", and the observed-only model has never seen a single unplayed cell. everything it has been trained on is a positive, so it has no notion of what a user *wouldn't* touch. its predictions for rarely played items are mostly bias terms fitted on a handful of enthusiastic plays, so those items float to the top of everyone's list: only 15% of its recommendations come from the popular head (experiment 2), against 75–85% for the models that do well here. this is Cremonesi, Koren & Turrin's (2010) finding on real data: optimizing the error of predicted values and optimizing top-N are different problems. and the 99% accuracy is the base rate, nothing more: with 1% positives, accuracy cannot tell a recommender from an empty list.

## experiment 2: held-out items vs the truth

**what people usually predict:** a better model is better under both protocols; maybe all scores are lower against the true favourites, but the ranking of models stays put.

**what happens:**

| scorer | held-out recall@10 | held-out NDCG@10 | true-favourite recall@10 | true-favourite NDCG@10 | from top-10% items |
|---|---|---|---|---|---|
| random | 0.025 | 0.019 | 0.036 | 0.035 | 9% |
| popularity | 0.380 | 0.349 | 0.173 | 0.202 | 100% |
| observed-only MF | 0.128 | 0.107 | 0.203 | 0.220 | 15% |
| implicit ALS (α = 1) | 0.443 | 0.408 | **0.454** | **0.495** | 75% |
| BPR (30 epochs) | **0.450** | **0.419** | 0.349 | 0.387 | 85% |

two rankings flip robustly. popularity is the third-best scorer on held-out items (NDCG 0.349, within 0.07 of the best) and the *worst* non-random scorer against the true favourites (0.202). the observed-only model goes the other way: far behind popularity on held-out items, ahead of it on the truth. BPR and implicit ALS are effectively tied on held-out items (BPR is ahead here and on seeds 1–2, but implicit ALS ties or leads on seeds 3–4: held-out NDCG@10 0.433 vs 0.431 and 0.382 vs 0.370 in a reviewer's runs), while implicit ALS is clearly ahead on the truth (0.495 vs 0.387 here; 0.483 and 0.490 vs 0.368 and 0.388 on seeds 1–2). the popularity and observed-only flips repeat on every seed (popularity 0.214 and 0.200 true-favourite NDCG on seeds 1–2; observed-only 0.281 and 0.250).

**why:** the held-out set is a random sample of *consumption*, and consumption was driven by exposure (popularity) as well as taste (on paper 4). anything that predicts exposure is rewarded by the standard protocol, and exposure is most of what popularity predicts. it isn't worthless against the truth (true-favourite NDCG 0.20 against 0.035 for random), because the simulator's item quality correlates with popularity (about 0.6 between mean preference and log-popularity): popularity captures exposure and some general quality, but nothing personal, and it shows everybody the same ten items. the observed-only model has the opposite problem: it knows something about taste (it fits play counts, which depend on preference), but nothing about exposure, so it can't predict what people will *come across*. which protocol to trust depends on what you're shipping. if the recommender will itself change what people see, you care about taste, and the held-out protocol overstates popularity-heavy models (Steck, 2011). without a simulator you can approximate the truth with randomized exposure (show random items to a small slice of traffic), with propensity-weighted metrics, or, eventually, an online A/B test.

## experiment 3: AUC vs the top of the list

**what people usually predict:** BPR improves steadily, and all three numbers rise together because they're all measuring "ranking quality".

**what happens:** (this run trains with its own random stream, so its numbers differ slightly from experiment 1's BPR.)

| epoch | 1 | 3 | 5 | 10 | 20 | 30 | 50 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| AUC (held-out) | 0.857 | 0.859 | 0.860 | 0.867 | 0.894 | 0.903 | 0.902 | 0.904 | 0.902 |
| NDCG@10 (held-out) | 0.346 | 0.336 | 0.334 | 0.345 | 0.388 | 0.403 | 0.379 | 0.382 | 0.374 |
| NDCG@10 (true favourites) | 0.203 | 0.195 | 0.193 | 0.208 | 0.301 | 0.378 | 0.431 | 0.426 | 0.431 |
| share from top-10% | 100% | 100% | 100% | 100% | 93% | 86% | 80% | 75% | 74% |

for the first 10 epochs BPR *is* popularity: every list comes entirely from the top 10% of items, and all three numbers match popularity's (AUC 0.86, NDCG 0.35 and 0.20). between epochs 10 and 30 personalization kicks in and everything rises. after epoch 30 AUC is flat to the third decimal (0.902–0.904), while the lists keep changing: held-out NDCG *falls* from 0.403 to 0.374 and true-favourite NDCG *rises* from 0.378 to 0.431, as the share of head items drops from 86% to 74%. the implicit ALS dropdown shows the same tension along a different axis: as $\alpha$ grows from 1 to 40, coverage rises from 43% to 67% while both NDCGs fall (true-favourite NDCG 0.495 → 0.325) and AUC barely moves (0.886 → 0.857).

**why:** the item biases get a gradient from every triple (a popular item is a positive for many users and a random $j$ is usually unpopular), while the factors start at size 0.1 and grow slowly, the same saddle-point plateau you met in topic 21's SGD. so BPR first learns the global item order, which is popularity, and only later the per-user part. AUC averages over all (relevant, irrelevant) pairs, about $6 \times 590$ per user, and after epoch 30 BPR is mostly reshuffling items near the top of each list: a few hundred pairs change order out of thousands, so AUC can't see it, while NDCG@10 sees nothing *but* that. the two NDCGs then disagree because the reshuffling swaps popular items (which the held-out protocol rewards) for niche items the user genuinely prefers (which the truth rewards). three lessons: optimize and select models on a metric that looks at the part of the list users see; don't treat a flat AUC as "converged"; and remember that the held-out protocol has a popularity bias that can make a model look worse as it gets better for users.
