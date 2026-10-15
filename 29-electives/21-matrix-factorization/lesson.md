# matrix factorization

## why this matters

a film site has 600 users, 500 films and about 18,000 star ratings. the full table would have 300,000 cells, so **94% of it is empty**, and the empty cells are the only ones anyone cares about: "should we show this user that film?" is a question about a cell nobody has filled in. (MovieLens-100K is also 94% empty; the Netflix Prize matrix was about 99% empty.)

what makes filling it possible is that tastes aren't arbitrary: people who loved the same three slow science-fiction films tend to agree about the fourth. **matrix factorization** gives each user and each film a short vector of hidden "taste coordinates" and predicts a rating as roughly their dot product. it was the workhorse of the Netflix Prize (Koren, Bell & Volinsky, 2009) and is still the baseline every recommender is measured against.

this topic builds it twice, by alternating least squares and by stochastic gradient descent, then asks the uncomfortable question: the ratings you have are the ones people *chose* to give. what does a good test score on those tell you?

## warm-up: predict first

1. the empty cells of a rating matrix aren't zeros. why can't you just run the SVD from chapter 02 on the matrix with blanks filled with 0 (or with the mean)?
2. you factorize with rank 50 when the true tastes have rank 6, with almost no regularization. does test error keep getting worse as you raise the rank to 100?
3. ALS alternates between solving for all users and solving for all films. can the training objective ever go *up* between two of those steps?
4. a brand-new user has rated nothing. what does a factorization model predict for them, and is that a reasonable thing to show?
5. people mostly rate films they chose to watch. if you hold out 20% of their ratings at random and your model scores well on them, will it score as well on films people *didn't* choose?

## prerequisites check

- [ ] you can solve ridge regression in closed form, $w = (X^\top X + \lambda I)^{-1}X^\top y$ (→ ch 08 · topic 02)
- [ ] you know what a low-rank approximation is and how the SVD gives the best one for a *complete* matrix (→ ch 02 · topic 06)
- [ ] you can write an SGD loop and know why the learning rate matters (→ ch 06 · topic 04)
- [ ] you know why the test set must stay untouched, and what double descent looks like (→ ch 07 · topics 02 and 04)
- [ ] you pass seeds around with `as_rng` instead of using global random state (→ ch 00 · topic 05)

> **the data.** no ratings dataset is cached and nothing is downloaded, so every rating in this block comes from a **simulator**, `make_ratings` (given in the notebook), built to have what makes real ratings hard:
>
> - **latent tastes:** hidden vectors $U_u, V_i \in \mathbb R^6$ and biases; the true preference is $\text{pref}_{ui} = 3 + b_u + b_i + U_u \cdot V_i$.
> - **power-law popularity:** film $i$ has exposure weight $(1 + \text{popularity rank})^{-0.8}$; the top 10% of films get 44% of all ratings. the item biases are built partly from popularity, so popular films are also somewhat better liked on average (mean preference and log-popularity correlate at about 0.6), a deliberate but simplifying choice.
> - **missing not at random:** each user rates $n_u$ films (median 19), drawn with probability $\propto$ popularity $\times\, e^{\text{pref}_{ui}}$.
> - **noisy stars:** $\text{pref}_{ui} + \mathcal N(0, 0.5^2)$, rounded and clipped to 1–5 (shares 1%, 9%, 26%, 34%, 29%, close to MovieLens).
>
> the simulator also knows the truth for every cell (`world.preference`) and can rate random user–film pairs (`world.sample_random`), which the lab uses. `load_movielens` reads a real `u.data` file into the same arrays; nothing here depends on it.

## core ideas

### idea 1: the rating matrix, an honest split, and the baseline to beat

store ratings as three aligned arrays, `users`, `items`, `ratings`: one entry per observed cell, never the mostly empty matrix.

to evaluate, hide a random 20% of the triples, fit on the rest, and measure the **root mean squared error** on the hidden ones,

$$\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal T|}\sum_{(u,i)\in\mathcal T}(r_{ui} - \hat r_{ui})^2}.$$

(the Netflix Prize was scored this way.) a random split means a few test users or films may have no training ratings; the model must still predict something for them.

before any factorization, ask how far you get *without* modelling taste at all. ratings have strong main effects: some users rate everything high, some films are universally liked. the **bias baseline** predicts

$$\hat r_{ui} = \mu + b_u + b_i,$$

with $\mu$ the mean training rating. fit the biases by minimizing squared error plus a ridge penalty $\lambda(\sum_u b_u^2 + \sum_i b_i^2)$. the penalty matters: a film with one 5-star rating would otherwise get $b_i = 5 - \mu$, a confident claim from one data point. holding the user biases fixed, the best item bias is a *shrunk* average of residuals, and the other way round:

$$b_i = \frac{\sum_{u \in R(i)}(r_{ui} - \mu - b_u)}{\lambda + |R(i)|},\qquad b_u = \frac{\sum_{i \in R(u)}(r_{ui} - \mu - b_i)}{\lambda + |R(u)|}.$$

with no ratings the bias is 0; with many it approaches the plain average; $\lambda$ acts like $\lambda$ phantom ratings at exactly the mean. alternating the two updates is coordinate descent on a convex quadratic, so it converges to the unique ridge solution. on the simulated world this baseline already gets a test RMSE of 0.86, against 1.00 for predicting $\mu$ everywhere. every model below has to beat 0.86, not 1.00.

> **spec.** `split_ratings(users, items, ratings, *, test_frac=0.2, rng=None)` returns `((u, i, r) train, (u, i, r) test)`: each triple goes to test independently with probability `test_frac` (use `as_rng(rng)`), and the three arrays in each part stay aligned. `rmse(pred, target)` returns a float. `BiasBaseline(lam=5.0, *, n_iters=10)` stores its arguments; `fit(users, items, ratings, n_users=None, n_items=None)` (sizes default to max id + 1) sets `mu_`, `user_bias_` `(n_users,)` and `item_bias_` `(n_items,)` by `n_iters` sweeps of **item biases first, then user biases** with the formulas above, and returns `self`. `predict(users, items)` returns `mu_ + user_bias_[users] + item_bias_[items]`.

> **▶ build: stage 1.** implement `split_ratings`, `rmse` and `BiasBaseline`.

### idea 2: factorization by alternating least squares

now add taste. give user $u$ a vector $p_u \in \mathbb R^k$ and film $i$ a vector $q_i \in \mathbb R^k$:

$$\hat r_{ui} = \mu + b_u + b_i + p_u\cdot q_i.$$

stacking the vectors as rows of $P$ and $Q$, the taste part of the full matrix is $PQ^\top$, a rank-$k$ matrix. the coordinates have no names, but after fitting they often line up with something interpretable ("serious drama vs action").

why not use the SVD, which gives the best rank-$k$ approximation? because the SVD needs every entry. fill the blanks with zeros and the SVD faithfully reproduces the zeros: it spends its rank explaining which cells are missing. the right objective sums only over the observed set $\mathcal R$:

$$J = \sum_{(u,i)\in\mathcal R}\big(r_{ui} - \mu - b_u - b_i - p_u\cdot q_i\big)^2 + \lambda\Big(\sum_u \|p_u\|^2 + \sum_i\|q_i\|^2 + \sum_u b_u^2 + \sum_i b_i^2\Big).$$

this has no closed form and isn't convex (scale $p_u$ up and $q_i$ down and nothing changes), but it has a beautiful structure: **fix all the film parameters and it splits into one small ridge regression per user.** user $u$'s rating of film $i$ is then a linear function of the unknowns $(p_u, b_u)$:

$$r_{ui} - \mu - b_i \;\approx\; \begin{bmatrix} q_i^\top & 1\end{bmatrix}\begin{bmatrix} p_u \\ b_u\end{bmatrix}.$$

so stack the rows $[q_i^\top, 1]$ for the films $u$ rated into a design matrix $X_u$ ($|R(u)| \times (k+1)$), the targets $r_{ui} - \mu - b_i$ into $y_u$, and solve

$$\begin{bmatrix} p_u \\ b_u\end{bmatrix} = \big(X_u^\top X_u + \lambda I_{k+1}\big)^{-1}X_u^\top y_u.$$

the appended 1 is a trick worth remembering: it turns "a factor plus a bias" into one ridge regression with one more coefficient. fixing the users gives the mirror-image problem for each film, with rows $[p_u^\top, 1]$ and targets $r_{ui} - \mu - b_u$. **alternating least squares** (ALS) alternates the two sweeps.

each half-step minimizes $J$ *exactly* over one block with the other fixed, so $J$ never increases (warm-up 3); it converges to a stationary point, usually within a handful of iterations. the per-user solves are independent, so ALS parallelizes well (Zhou et al., 2008). a user with no ratings gets the ridge solution zero.

**rank and $\lambda$.** rank $k$ sets capacity, $\lambda$ restrains it. with $k + 1$ unknowns per user and only 19 ratings for a median user, a large rank lets each user's regression fit its own ratings perfectly. the lab maps the full picture and finds that "more rank, more overfitting" is only half the story (warm-up 2).

> **spec.** `ALS(rank=10, *, lam=1.0, n_iters=10, rng=None)` stores its arguments. `fit(users, items, ratings, n_users=None, n_items=None)`: `mu_` = mean rating; `user_factors_ = rng.normal(0, 0.1, (n_users, rank))`, **then** `item_factors_ = rng.normal(0, 0.1, (n_items, rank))` with `rng = as_rng(self.rng)`; `user_bias_`, `item_bias_` start at zero. each of `n_iters` iterations solves **every item, then every user** with the ridge solve above ($\lambda I$ on all $k + 1$ coefficients), then appends `loss(...)` on the training data to `history_`. returns `self`. `predict(users, items)` and `loss(users, items, ratings)` compute $\hat r$ and $J$.

> **🔮 predict:** after `fit`, take one user's ratings and solve their ridge regression again by hand using the fitted item factors. do you get exactly their `user_factors_` row? would the same be true for an item?

> **▶ build: stage 2.** implement `ALS`.

### idea 3: the same model by stochastic gradient descent

the Netflix Prize made a different algorithm famous. Simon Funk (2006) noticed you can fit the same model by looping over the observed ratings one at a time and nudging the four parameters it touches. for rating $r_{ui}$, compute the error $e_{ui} = r_{ui} - \hat r_{ui}$ and step down the gradient of $e_{ui}^2 + \lambda(b_u^2 + b_i^2 + \|p_u\|^2 + \|q_i\|^2)$ (the factor 2 folds into the learning rate $\eta$):

$$\begin{aligned}
b_u &\leftarrow b_u + \eta\,(e_{ui} - \lambda b_u), & p_u &\leftarrow p_u + \eta\,(e_{ui}\,q_i - \lambda p_u),\\
b_i &\leftarrow b_i + \eta\,(e_{ui} - \lambda b_i), & q_i &\leftarrow q_i + \eta\,(e_{ui}\,p_u - \lambda q_i),
\end{aligned}$$

all four computed from the *old* values (Koren, Bell & Volinsky, 2009, give exactly these updates). an epoch visits every training rating once, in a fresh random order.

each update costs $O(k)$, but in Python an epoch is a loop over every rating, and an ALS sweep jumps to the exact optimum of half the parameters while an SGD epoch takes many small noisy steps. in exchange SGD works for any differentiable model (add time effects or a neural network to $\hat r$ and the loop is unchanged) and can learn from ratings as they stream in.

two things to watch. first, **$\lambda$ means something different here.** SGD penalizes $p_u$ every time it visits one of $u$'s ratings, so its effective penalty on a user with 200 ratings is 200 times that on a user with one rating. ALS penalizes each parameter once. an SGD $\lambda$ of 0.05 and an ALS $\lambda$ of 4 can be equally strong. second, **small initial factors.** with $p_u, q_i \approx 0$ the gradient of $p_u\cdot q_i$ with respect to $p_u$ is $q_i \approx 0$ too: the factors start near a saddle point and grow slowly at first, while the biases, whose gradient doesn't vanish, do all the early work.

> **spec.** `SGDMF(rank=10, *, lr=0.01, lam=0.05, n_epochs=20, rng=None)` stores its arguments and has the same fitted attributes as `ALS`, initialized the same way (user factors, then item factors, from `as_rng(self.rng)`). `fit` then runs `n_epochs` epochs; each draws `rng.permutation(n_ratings)` and calls `step` on the ratings in that order, then appends the training RMSE to `history_`. `fit` with `n_epochs=0` just initializes. `step(u, i, r)` applies the four updates (from the old values) and returns the error $e$ as a float. `predict` as in `ALS`.

> **▶ build: stage 3.** implement `SGDMF`.

### idea 4: cold start

a model trained yesterday doesn't know today's new user, but you needn't retrain: the new user's vector is *one* ridge regression away. given the films they've rated so far, solve

$$\begin{bmatrix} p_{\text{new}} \\ b_{\text{new}}\end{bmatrix} = \big(X^\top X + \lambda I\big)^{-1}X^\top y,\qquad X = \begin{bmatrix} q_i^\top & 1\end{bmatrix}_{i \in R(\text{new})},\ y = \big(r_i - \mu - b_i\big)_{i\in R(\text{new})},$$

with the trained $Q$ and item biases fixed. this is **fold-in**, one ALS half-step for one row: a $(k+1)\times(k+1)$ solve per new rating.

with no ratings, $X$ is empty, the solution is zero and the prediction is $\mu + b_i$: the bias baseline's answer, a sensible thing to show a stranger (warm-up 4). the hard part is the next few ratings: with one rating and $k + 1 = 9$ unknowns, only $\lambda$ stops the regression inventing a confident taste vector from a single number. features of the user or film are the other classic remedy; topic 23 uses them.

> **spec.** `fold_in(model, items, ratings, *, lam=None)` uses only `model.mu_`, `model.item_factors_`, `model.item_bias_` and (when `lam` is `None`) `model.lam`, so it works for `ALS` and `SGDMF` alike. it returns `(factors, bias)`: an array `(rank,)` and a float, zeros when `items` is empty. `predict_new_user(model, factors, bias, items)` returns `mu_ + bias + item_bias_[items] + item_factors_[items] @ factors`.

> **🔮 predict:** a new user reveals one rating, 5 stars for a film whose factor vector is large. with $\lambda = 0.1$, what happens to their predicted ratings for films pointing the opposite way?

> **▶ build: stage 4.** implement `fold_in` and `predict_new_user`.

## worked example

**biases.** user 0 rated film 0 with 5 stars and film 1 with 3; user 1 rated film 0 with 4. so $\mu = 4$. with $\lambda = 1$ and one sweep, items first (user biases still 0):

$$b_{i_0} = \frac{(5 - 4) + (4 - 4)}{1 + 2} = \tfrac13,\qquad b_{i_1} = \frac{3 - 4}{1 + 1} = -\tfrac12.$$

then users, with the new item biases:

$$b_{u_0} = \frac{(5 - 4 - \tfrac13) + (3 - 4 + \tfrac12)}{1 + 2} = 0.0556,\qquad b_{u_1} = \frac{4 - 4 - \tfrac13}{1 + 1} = -\tfrac16.$$

the prediction for user 1 on film 1 is $4 - \tfrac16 - \tfrac12 = 3.33$.

**one SGD step.** $\mu = 3$, $b_u = 0.1$, $b_i = -0.2$, $p_u = (1, 2)$, $q_i = (0.5, -1)$, $\eta = 0.1$, $\lambda = 0.5$, and the rating is 4. the prediction is $3 + 0.1 - 0.2 + (0.5 - 2) = 1.4$, so $e = 2.6$. then $b_u \to 0.1 + 0.1(2.6 - 0.05) = 0.355$, $b_i \to -0.2 + 0.1(2.6 + 0.1) = 0.07$,

$$p_u \to (1, 2) + 0.1\big(2.6\,(0.5, -1) - 0.5\,(1, 2)\big) = (1.08, 1.64),\qquad q_i \to (0.5, -1) + 0.1\big(2.6\,(1, 2) - 0.5\,(0.5, -1)\big) = (0.735, -0.43).$$

if you update $q_i$ with the *new* $p_u$ you get $(0.7558, -0.4762)$ instead.

**fold-in.** a rank-1 model with $\mu = 3$, item biases 0 and item factors $q_0 = 1$, $q_1 = 2$. a new user rates film 0 with 4 and film 1 with 5; $\lambda = 1$. the rows of $X$ are $[1, 1]$ and $[2, 1]$, and $y = (1, 2)$:

$$X^\top X + I = \begin{bmatrix} 6 & 3 \\ 3 & 3\end{bmatrix},\quad X^\top y = \begin{bmatrix} 5 \\ 3\end{bmatrix}\quad\Rightarrow\quad p = \tfrac23,\ b = \tfrac13.$$

the user's predicted ratings are $3 + \tfrac13 + \tfrac23 = 4$ for film 0 and $4.67$ for film 1: both pulled towards $\mu$ compared with the actual 4 and 5, which is the shrinkage doing its job. your code must reproduce all three calculations.

## on paper

1. **the ALS user step.** write $J$ as a function of $w_u = (p_u, b_u)$ only, with everything else fixed, and derive the solution of idea 2 by setting the gradient to zero.

<details><summary>solution</summary>

the terms of $J$ that involve $w_u$ are $\sum_{i\in R(u)}(y_{ui} - x_i^\top w_u)^2 + \lambda\|w_u\|^2$ with $x_i = (q_i, 1)$ and $y_{ui} = r_{ui} - \mu - b_i$, i.e. $\|y_u - X_uw_u\|^2 + \lambda\|w_u\|^2$. the gradient is $-2X_u^\top(y_u - X_uw_u) + 2\lambda w_u$; setting it to zero gives $(X_u^\top X_u + \lambda I)w_u = X_u^\top y_u$. the matrix is positive definite for $\lambda > 0$, so the solution exists, is unique, and is the minimum. no other user's terms involve $w_u$, which is why all users can be solved independently.

</details>

2. **monotone descent.** show that the ALS training objective is non-increasing from one iteration to the next. does that mean ALS finds the global minimum?

<details><summary>solution</summary>

let $J(P, Q)$ (biases included in $P$ and $Q$). the item sweep replaces $Q$ by $Q' = \arg\min_Q J(P, Q)$, so $J(P, Q') \le J(P, Q)$; the user sweep then gives $P' = \arg\min_P J(P, Q')$, so $J(P', Q') \le J(P, Q')$. chaining, $J$ never increases, and since $J \ge 0$ it converges. but $J$ isn't jointly convex (for $k \ge 1$, $p\cdot q$ is a saddle-shaped function of $(p, q)$), so the limit is a point where neither block can improve on its own: a stationary point, possibly a saddle or a local minimum. different seeds can end in different places, which is why the lab fixes the seed.

</details>

3. **SGD's real penalty.** an epoch of SGD with constant $\eta$ approximately follows the gradient of $\sum_{(u,i)\in\mathcal R}\big[e_{ui}^2 + \lambda(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2)\big]$. rewrite its penalty in the form $\sum_u c_u\|p_u\|^2 + \dots$ and compare with ALS. for what ALS penalty do the two objectives agree?

<details><summary>solution</summary>

$\|p_u\|^2$ appears once for every rating of user $u$, so the penalty is $\lambda\sum_u|R(u)|\,\|p_u\|^2 + \lambda\sum_i|R(i)|\,\|q_i\|^2 + \dots$. ALS uses $\lambda\|p_u\|^2$ once per user. the two agree if ALS uses a per-user $\lambda_u = \lambda|R(u)|$, which is the "weighted-$\lambda$ regularization" of Zhou et al. (2008). with it, heavy users are regularized more in absolute terms, but per rating the penalty is the same.

</details>

4. **fold-in with few ratings.** a new user has $n$ ratings and the model has rank $k$. what is the fold-in solution as $\lambda \to 0$ when $n < k + 1$? and with $\lambda > 0$ and $n = 0$?

<details><summary>solution</summary>

with $n < k + 1$ the system $Xw = y$ has infinitely many exact solutions, and $(X^\top X + \lambda I)^{-1}X^\top y \to X^+y$ as $\lambda \to 0$: the minimum-norm one, which fits the $n$ ratings exactly and reproduces their noise. the predicted ratings for other films are then driven by noise in a handful of numbers, which is why weak regularization hurts new users so much. with $n = 0$, $X^\top y = 0$ and the solution is $w = 0$ for any $\lambda > 0$: the prediction is $\mu + b_i$.

</details>

## experiment

`lab.py` runs your code on the simulated world:

1. train and test RMSE as the rank grows from 1 to 96, for three values of $\lambda$.
2. ALS against SGD, by test RMSE per second.
3. the best model's error on held-out ratings versus on random user–film pairs that the simulator can rate for you.
4. how quickly fold-in serves a new user as they reveal 0 to 20 ratings.

## check your understanding

1. (warm-up 1.) why does filling the blanks with zeros and taking an SVD give a poor recommender?

<details><summary>answer</summary>

the SVD minimizes error over *every* cell, so with zeros filled in it tries to reproduce the zeros: most of its rank goes into predicting which cells are empty and pulling predictions towards 0. filling with the mean is better but still treats "unrated" as "rated average", and 94% of the cells would dominate the fit. the observed-only objective ignores missing cells entirely.

</details>

2. why can every user be solved in parallel within an ALS sweep, but not users and items at the same time?

<details><summary>answer</summary>

with the items fixed, $J$ is a sum of separate terms, one per user, that share no parameters. users and items appear together in $p_u\cdot q_i$, so solving both at once is a nonlinear problem with no closed form.

</details>

3. a user rates every film 5 stars. what does the fitted model do with them?

<details><summary>answer</summary>

their bias absorbs the generosity: $b_u$ grows (shrunk by $\lambda$), and $p_u$ stays near 0 because there's no variation for taste to explain. every prediction for them is high, and the ranking of films for them is just the ranking by $b_i$. that's the right answer: their ratings carry no information about taste.

</details>

4. why does the fold-in of an existing user reproduce their ALS factors exactly, but folding in an *item* by the same recipe after `fit` wouldn't reproduce `item_factors_`?

<details><summary>answer</summary>

within each iteration ALS solves items first and users last, so after `fit` every user row is the exact ridge solution given the final items. the items were solved against the *previous* user factors, which have since changed.

</details>

5. (warm-up 5.) the held-out ratings came from the same people choosing films in the same way as the training ratings. why is a low test RMSE on them still not evidence that the model predicts well on films people didn't choose?

<details><summary>answer</summary>

the test set is a random sample of the *observed* cells, and observation depended on the rating itself (people rate what they like and what's popular). so both train and test over-represent high ratings of popular films; a model that has learned "ratings are high" scores well on both. the cells you want to predict for recommendations are mostly the unchosen ones, where the true ratings are lower. the lab measures the gap.

</details>

## common bugs

- **regularizing the bias differently from the objective.** symptom: ALS's `history_` occasionally goes up. use $\lambda I$ on all $k + 1$ coefficients, the same penalty `loss` adds.
- **stale targets.** symptom: users or items that don't match a manual ridge solve. each sweep's targets must use the *current* other-side biases.
- **updating $q_i$ with the new $p_u$.** symptom: the worked SGD step gives $(0.7558, -0.4762)$. copy $p_u$ first.
- **misaligned split.** symptom: test RMSE near 1.5 for every model. index all three arrays with the *same* mask.
- **using the ALS $\lambda$ for SGD (or vice versa).** symptom: SGD stuck at the bias baseline, or ALS overfitting wildly; the scales differ by about the ratings per user.

## going deeper

- Koren, Bell & Volinsky (2009), "Matrix factorization techniques for recommender systems", *IEEE Computer* 42(8). the biased model, SGD and ALS, as used to win the Netflix Prize. [doi:10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
- Funk (2006), "Netflix update: try this at home", blog post. the SGD factorization that started it. [sifter.org/~simon/journal/20061211.html](https://sifter.org/~simon/journal/20061211.html)
- Zhou, Wilkinson, Schreiber & Pan (2008), "Large-scale parallel collaborative filtering for the Netflix Prize", *AAIM*. ALS with weighted-$\lambda$ regularization, parallelized. [doi:10.1007/978-3-540-68880-8_32](https://doi.org/10.1007/978-3-540-68880-8_32)
- Koren (2008), "Factorization meets the neighborhood: a multifaceted collaborative filtering model", *KDD*. biases, implicit feedback inside an explicit model (SVD++). [doi:10.1145/1401890.1401944](https://doi.org/10.1145/1401890.1401944)
- Marlin & Zemel (2009), "Collaborative prediction and ranking with non-random missing data", *RecSys*. evidence (from a survey of real users) that ratings are missing not at random. [doi:10.1145/1639714.1639717](https://doi.org/10.1145/1639714.1639717)
- Schnabel, Swaminathan, Singh, Chandak & Joachims (2016), "Recommendations as treatments: debiasing learning and evaluation", *ICML*. propensity-weighted training and evaluation for MNAR ratings. [arXiv:1602.05352](https://arxiv.org/abs/1602.05352)
- Aggarwal (2016), *Recommender Systems: The Textbook*, Springer, chapter 3 (model-based collaborative filtering, latent factor models).

## connections

- ← builds on: ridge regression (ch 08 · topic 02), low-rank approximation and the SVD (ch 02 · topic 06), SGD (ch 06 · topic 04), honest splits and double descent (ch 07 · topics 02 and 04).
- → used in: topic 22 replaces ratings by clicks and RMSE by ranking metrics, with the same alternating ridge solves inside implicit ALS; topic 23 replaces the free vectors $p_u, q_i$ by neural "towers" that also read features, which is the general answer to cold start.
