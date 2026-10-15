# matrix factorization · lab notes

numbers below are from the reference solution with the lab's seeds (world seed 0, split seed 1). your code gives the same numbers if it draws its random numbers in the order the spec gives, and similar ones otherwise. the shapes reported here were checked on three other world seeds (1, 2, 3) and hold on each. timings are from a busy laptop and will differ on yours; the ratios are what matter.

the world: 18,226 ratings from 600 users on 500 films (14,513 train, 3,713 test), median 19 ratings per user, stars 1–5 in proportions 1%, 9%, 26%, 34%, 29%. predicting the training mean everywhere gives test RMSE 1.000; the bias baseline gives **0.861**.

## experiment 1: rank vs overfitting

**what people usually predict:** test RMSE falls until the true rank (6) and then rises steadily as extra rank overfits, most steeply with no regularization.

**what happens:**

| λ \ rank | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 1.133 | 1.074 | 1.140 | 0.937 | 1.577 | 1.520 | 1.542 | 1.577 | 1.359 | 1.145 | 1.071 | 0.989 |
| 1 | 0.861 | 0.834 | 0.798 | 0.738 | 0.794 | 0.854 | 0.908 | 0.915 | 0.875 | 0.833 | 0.812 | 0.806 |
| 10 | 0.851 | 0.836 | 0.820 | 0.811 | 0.812 | 0.812 | 0.812 | 0.812 | 0.812 | 0.812 | 0.812 | 0.812 |

training RMSE falls monotonically for small $\lambda$ (to 0.002 at rank 96 with $\lambda = 0.01$: every training rating fitted exactly). test RMSE with $\lambda = 0.01$ is never better than the bias baseline, peaks at about 1.58 around ranks 8–24 and then *improves* again, to 0.99 at rank 96. $\lambda = 1$ shows the same bump in a milder form; its best value here (0.738 at rank 6) is partly luck of the initialization seed (other init seeds give 0.74–0.79 there). $\lambda = 10$ is flat: past rank 6 it doesn't matter what rank you choose. the slider finds the best $\lambda$ at rank 8 around 3–5 (0.759 at $\lambda = 4$). on world seeds 1–3 the $\lambda = 0.01$ row has the same overall shape: a peak of 1.5–1.6 at ranks 8–16 and about 0.99 at rank 96. its point-to-point wiggles are not meaningful: with almost no regularization, 10 ALS iterations haven't converged and the curve depends on the initialization seed (rank 6 gives 0.94, 1.32 or 1.27 for three init seeds; with 30 iterations rank 6 is at 0.99 and rank 96 at 0.94). read the double-descent shape, not individual points.

**why:** this is double descent (ch 07 · topic 04), and in matrix factorization you can see exactly where its peak comes from. each ALS half-step is a ridge regression with $k + 1$ unknowns and as many equations as that user or film has ratings. when $k + 1$ is close to that count, the regression is square and badly conditioned: with almost no penalty it fits the noise in a few ratings with huge coefficients, and test error explodes. the median user has 19 ratings (about 15 in the training split), and many films have fewer, so the trouble zone is ranks 8–24. far past it the system is very underdetermined, and the ridge solution with tiny $\lambda$ behaves like the minimum-norm solution, which spreads the fit over many directions and is much tamer: error comes back down. a real $\lambda$ removes the peak, because it keeps every regression well conditioned whatever its shape. the lesson for practice: tune $\lambda$ and rank *together*, and don't read "test error went up at rank 12" as "rank 12 is too much": with a tuned $\lambda$, large ranks cost almost nothing.

## experiment 2: ALS vs SGD

**what people usually predict:** SGD, the algorithm that won the Netflix Prize and that trains every neural network, is faster or at least comparable; or ALS faster by a small factor.

**what happens:** ALS reaches its best test RMSE, **0.758**, after about 12 iterations at **0.03 s per iteration**: below 0.80 after **0.09 s**. SGD with learning rate 0.05 takes **0.23 s per epoch**, gets below 0.80 only after **2.2 s** and reaches 0.755 after 38 epochs (about 7 s): the same accuracy, roughly 25 times slower in wall-clock time. most of that factor is Python overhead: SGD runs one interpreted `step` per rating, ALS a few hundred vectorized solves per iteration, so a compiled SGD (as in production libraries) would narrow the gap a lot. the per-pass progress difference below is the part that's intrinsic. with learning rate 0.01 it's still at 0.832 after 40 epochs. both SGD curves start with a plateau: after epochs 1, 5 and 10 the test RMSE is 0.881, 0.865, 0.827 (lr 0.05) and 0.910, 0.868, 0.863 (lr 0.01), right at the bias baseline's 0.861.

**why:** one ALS iteration solves every per-user and per-film subproblem *exactly*, in vectorized linear algebra; one SGD epoch takes 14,513 small noisy steps in a Python loop. on a problem this size with a model that is exactly a dot product, ALS wins easily. the plateau is the saddle point at the origin: with factors initialized at scale 0.1, the gradient of $p_u\cdot q_i$ with respect to $p_u$ is $q_i$, also of size 0.1, so the factors grow slowly (roughly exponentially, from a small start), while the biases, whose gradient is the full error, fit immediately. until the factors are big enough the model *is* the bias baseline. a larger learning rate escapes faster. what SGD buys is generality: the same loop fits any differentiable $\hat r$ (time effects, implicit-feedback terms, neural networks, the BPR loss of topic 22), handles streaming data, and needs no matrix solves. that's why the field moved to SGD once models stopped being plain factorizations.

## experiment 3: what the test set can't see

**what people usually predict:** a little worse than the test RMSE, perhaps 0.8–0.9, with no systematic direction.

**what happens:**

| model | RMSE on held-out ratings | RMSE on random pairs | mean (predicted − actual) on random pairs |
|---|---|---|---|
| bias baseline | 0.861 | 1.253 | +0.68 |
| ALS rank 8, λ = 4 | 0.759 | 1.154 | +0.61 |

on 20,000 random user–film pairs the model's RMSE is **1.15**, half again its held-out RMSE, and it over-predicts by **0.61 stars** on average. the mean observed rating is 3.79; the mean rating of random pairs is 3.01. the error is much worse on the long tail: 0.88 on head films (top 10% by popularity), 1.18 on the rest. the left histogram shows why: the ratings people choose to give are shifted a full star up from the ratings random pairs would get. on world seeds 1–3 the same comparison gives 1.16–1.19 against 0.76–0.77, with a bias of +0.63 to +0.67. the rank-6, $\lambda = 1$ model from experiment 1 tells the same story with a smaller offset: 0.74–0.79 on held-out ratings, 1.04–1.09 on random pairs, +0.44 to +0.48 stars (three init seeds).

**why:** the test set is a random 20% of the *observed* ratings, and observation was not random: users rate films they expected to like (exposure $\propto e^{\text{pref}}$) and popular films. train and test share that selection, so a model that has learned "ratings are high, especially on the films people pick" scores well on both. the cells a recommender actually needs to rank are mostly the ones nobody chose, where ratings are lower; there the model is confidently, systematically wrong. more data of the same kind doesn't help, because every new rating has the same selection bias. what helps is changing the data (randomized exposure, asking users to rate random films, as Marlin & Zemel did), or modelling the selection (propensity weighting, Schnabel et al., 2016). this is also the seed of topic 22: when you recommend, what you need is a ranking of the unchosen items, and RMSE on chosen ones is a poor guide to it.

## experiment 4: cold start

**what people usually predict:** every revealed rating helps; after 3–5 ratings a new user is served nearly as well as a regular one.

**what happens:**

| n revealed | 0 | 1 | 2 | 3 | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|
| fold-in λ = 4 | 0.994 | 0.966 | 0.966 | 0.926 | 0.921 | 0.866 | 0.866 | 0.820 |
| fold-in λ = 0.1 | 0.997 | 1.006 | 1.039 | 1.055 | 1.169 | 1.073 | 1.083 | 0.935 |

with the model's own $\lambda = 4$, each revealed rating helps a little, and after 20 the new user's RMSE (0.82) approaches the 0.76 of users the model was trained on. with $\lambda = 0.1$, revealing ratings makes predictions *worse* than knowing nothing, with the worst point at 5 ratings (1.17); even 20 ratings only bring it back to 0.94. (the cold users were chosen among users with at least 30 ratings, so they're heavier raters than average; their RMSE with zero ratings, 0.99, is worse than the population's because the item-bias fallback can't see their personal bias.)

**why:** fold-in is a ridge regression with $k + 1 = 9$ unknowns. with $n \le 9$ ratings and a tiny penalty it interpolates them exactly, so a single 5-star rating of a film with a large factor vector becomes a large taste vector, which then makes strong, wrong predictions for every other film (on paper 4). the peak near $n = 5$–9 is the same interpolation threshold as in experiment 1. for training users this matters less, because the model's $\lambda$ was tuned on them and most have enough ratings; for a new user the prior *is* the model. practical systems therefore shrink hard for new users, start from popularity or an onboarding questionnaire, and use side features (topic 23) so a new user isn't a blank vector.
