# interpreting models

## why this matters

your boosted model predicts house prices well. now someone asks: *why* does it think this district is expensive? which inputs does it rely on? would it still work if the census stopped collecting one of them? is it using something it shouldn't? a single tree can be read; a forest of 500 trees or a booster with 2,000 can't.

the tempting answer is the one every library prints for free: `feature_importances_`, the impurity decreases from topic 01. it's also, in a precise and reproducible way, misleading: it rewards features for having many distinct values, even when they're pure noise. this topic builds three **model-agnostic** tools that only ask the model for predictions, so they work for any model in the course, and teaches you what question each one answers:

- **permutation importance**: how much does performance drop if this feature's information is destroyed?
- **partial dependence** and **ICE curves**: how does the prediction change as one feature moves?
- **Shapley values** (via **KernelSHAP**): how much did each feature contribute to *this* prediction?

none of them tell you what causes what in the world. they describe the *model*. that's still a lot: they catch leaks, spurious shortcuts and features that don't do what you think.

## warm-up: predict first

1. a random forest is trained on six features: one that matters and five of pure noise, one of them continuous. which feature gets the largest impurity importance?
2. two features are near copies of each other and both matter. you shuffle one of them. how much does the score drop, compared with shuffling a feature with no twin?
3. the model is $f(x) = x_1x_2$ with $x_2$ symmetric around 0. what does the partial dependence of $x_1$ look like?
4. $f(x) = 3x_1 + 2$, with the background mean of $x_1$ equal to 1. what's the Shapley value of $x_1$ at $x_1 = 4$?
5. with 20 features, how many coalitions does an exact Shapley computation need?

## prerequisites check

- [ ] you can use the `score`, `predict` and `predict_proba` protocol, and scorers `scorer(est, X, y)` (→ ch 07 · topics 01 and 03)
- [ ] you know impurity importance and how it's computed (→ topic 01, idea 5)
- [ ] you can solve a weighted least-squares problem, with a linear constraint (→ ch 02, ch 06)
- [ ] you know what an interaction between two features is (→ topic 03, check your understanding 2)

## core ideas

### idea 1: permutation importance

to ask whether a model *relies* on feature $j$, destroy the information in it and see how much worse the model gets. shuffling the column (a random permutation of its values across rows) keeps the feature's distribution but breaks its link with the target and with the other features. the **permutation importance** of $j$ is

$$I_j = s(f, X, y) - s(f, X^{\pi_j}, y),$$

the baseline score minus the score with column $j$ permuted, averaged over a few independent permutations (their spread is a rough uncertainty). no refitting: one prediction pass per feature per repeat.

**compute it on held-out data.** on the training rows, a model that memorized noise relies on that noise, and shuffling it hurts, so training-set permutation importance inherits the overfitting. on test rows, shuffling a noise feature can't hurt a prediction that never generalized anyway.

**why impurity importance is biased.** impurity importance sums the training-impurity decreases of the splits on a feature. a fully grown tree splits until its leaves are pure, and a split on noise still decreases *training* impurity. a continuous feature offers $n - 1$ thresholds, a coin flip one; the more thresholds, the more likely one of them happens to separate the training labels well, so high-cardinality features get chosen more often, deep in the tree, and collect importance they don't deserve (Strobl et al., 2007; warm-up 1, and the lab).

**the caveat: correlated features.** if $x_3$ is a near copy of $x_1$, the model may use both; shuffling $x_1$ alone leaves its information available in $x_3$, so each looks less important than it is (warm-up 2). worse, shuffling one of two correlated features creates rows that never occur in reality (a tiny house with many bedrooms), and the model's predictions there are arbitrary. permutation importance measures how much *this model* relies on a feature, not how much information the feature carries. to ask the second question you refit without it (drop-column importance), which the lab compares.

> **spec.** `permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5, rng=None)`: `scoring=None` uses `estimator.score(X, y)`, otherwise `scoring(estimator, X, y)`. for each feature and each repeat, permute the *original* column with `as_rng(rng).permutation(n)` in a copy of `X`, and record baseline minus permuted score. returns a dict with `importances` $(d, \text{n\_repeats})$, `importances_mean` and `importances_std` (ddof 0) over the repeats. never modify the caller's `X`.

> **▶ build: stage 1.** implement `permutation_importance`.

### idea 2: partial dependence and ICE

importance says *how much*; it doesn't say *how*. the **partial dependence** of the model on feature $j$ at value $v$ is the average prediction when you set feature $j$ to $v$ in every row and leave the other features alone (Friedman, 2001):

$$\text{PD}_j(v) = \frac1n\sum_{i=1}^nf(v, x_{i,-j}).$$

it's a curve over a grid of values: by default the distinct values of the feature if there are few, else its quantiles from the 5th to the 95th percentile (the extremes are where the model has little data). for a linear model it's a straight line with the coefficient as slope; for a tree ensemble it's a staircase.

the average hides things. an **ICE curve** (individual conditional expectation, Goldstein et al., 2015) is one row's version, $v \mapsto f(v, x_{i,-j})$, and PD is the mean of the ICE curves. if all ICE curves are parallel, the feature acts additively and PD tells the whole story. if they fan out or cross, the feature interacts with others, and PD can be badly misleading: for $f = x_1x_2$ with $x_2$ symmetric, half the curves go up, half go down, and the average is flat (warm-up 3). PD shares permutation importance's weakness with correlated features: setting $x_1 = v$ in a row whose twin $x_3$ is far from $v$ asks the model about rows that don't exist.

computing it efficiently: build all $n\cdot g$ modified rows at once (repeat each row $g$ times, overwrite column $j$ with the tiled grid) and call the model **once**. a loop over grid values costs $g$ calls, which for a booster with per-call binning overhead adds up.

> **spec.** `default_grid(column, grid_resolution=50)`: the unique values if there are at most `grid_resolution`, else `np.unique` of the quantiles at `np.linspace(0.05, 0.95, grid_resolution)`. `ice_curves(estimator, X, feature, grid, response='auto')` returns $(n, g)$ from one batched call through the given `model_response` helper (`'auto'`: `predict_proba[:, 1]` for a binary classifier, else `predict`). `partial_dependence(estimator, X, feature, *, grid=None, grid_resolution=50, response='auto')` returns `(grid, averaged)`.

> **🔮 predict:** a model has learned $f(x) = x_1^2$, with $x_1$ uniform on $[-1, 1]$. what does its permutation importance say about $x_1$, and what does its partial dependence look like?

> **▶ build: stage 2.** implement `default_grid`, `ice_curves` and `partial_dependence`.

### idea 3: Shapley values

for a single prediction you want a split: of the difference between this prediction and the average prediction, how much is due to each feature? cooperative game theory solved the fair-division version of that question in 1953. Shapley considered $d$ players and a **value function** $v(S)$ for every coalition $S \subseteq \{1, \dots, d\}$, and showed there's exactly one way to split $v(\text{all}) - v(\varnothing)$ that satisfies four axioms: **efficiency** (the shares add up to the total), **symmetry** (interchangeable players get equal shares), **dummy** (a player who never changes $v$ gets 0) and **additivity** (shares of a sum of games are sums of shares). the unique answer is each player's average marginal contribution over all orders in which the players could join:

$$\phi_j = \sum_{S \subseteq \{1..d\}\setminus\{j\}}\frac{|S|!\,(d - |S| - 1)!}{d!}\Big[v(S \cup \{j\}) - v(S)\Big].$$

to use this for a model, the players are the features, and $v(S)$ must say what the model predicts at $x$ when only the features in $S$ are "known". the **interventional** choice (the one used by KernelSHAP in practice) averages over a background sample $B$ (often a hundred training rows):

$$v(S) = \frac{1}{|B|}\sum_{b\in B}f(x_S, b_{\bar S}),$$

the prediction with the features in $S$ taken from $x$ and the rest from each background row. then $v(\varnothing)$ is the average prediction over the background, $v(\text{all}) = f(x)$, and efficiency says $\sum_j\phi_j = f(x) - E_B[f]$. for a linear model $\phi_j = w_j(x_j - \bar b_j)$ (on paper 2; warm-up 4).

the catch is the sum over all $2^{d-1}$ coalitions per feature: fine for $d \le 10$ or so, hopeless for 20 (warm-up 5). you'll write the exact version first: it's the ground truth for stage 4.

> **spec.** `masked_predictions(predict, X_background, x, masks)`: `masks` is a bool array $(m, d)$; returns $v$ for each row, $(m,)$, from a single call to `predict` on an $(m\cdot|B|, d)$ array. `shapley_values_exact(value, d)`: `value` maps a bool mask array $(m, d)$ to $(m,)$; evaluate it on all $2^d$ masks at once and return the $(d,)$ Shapley values by the formula above.

> **▶ build: stage 3.** implement `masked_predictions` and `shapley_values_exact`.

### idea 4: KernelSHAP

Lundberg and Lee (2017) found that the Shapley values are also the solution of a **weighted linear regression**. look for an additive "explanation model" $g(z) = \phi_0 + \sum_jz_j\phi_j$ over binary coalition vectors $z \in \{0, 1\}^d$, fitted so that $g(z) \approx v(z)$ in the weighted least-squares sense with the **Shapley kernel**

$$\pi(z) = \frac{d - 1}{\binom{d}{|z|}\,|z|\,(d - |z|)},$$

and with the two extreme coalitions fitted exactly: $\phi_0 = v(\varnothing)$ and $\phi_0 + \sum_j\phi_j = v(\text{all})$ (their kernel weight is infinite). the solution is exactly the Shapley vector. why is that useful? because a regression doesn't need every coalition: with $d$ large you can *sample* coalitions and still get a good estimate, with the usual least-squares error that shrinks like $1/\sqrt{\#\text{samples}}$.

the recipe:

1. if $2^d - 2 \le$ `n_samples`, use every proper, non-empty coalition with its kernel weight: the answer is exact.
2. otherwise sample `n_samples` coalitions: draw a size $s \in \{1, \dots, d - 1\}$ with probability proportional to $\frac{d - 1}{s(d - s)}$ (the kernel's total mass on size $s$, since there are $\binom ds$ coalitions of that size), then a uniformly random subset of that size, and give every sample weight 1 (the sampling already did the weighting).
3. compute $v$ for every sampled coalition with `masked_predictions`.
4. enforce efficiency by elimination: substitute $\phi_d = (f(x) - \phi_0) - \sum_{j<d}\phi_j$, which turns the problem into an unconstrained weighted least squares in $\phi_1, \dots, \phi_{d-1}$; solve it with `np.linalg.lstsq` on rows scaled by $\sqrt{w}$, and recover $\phi_d$.

KernelSHAP is model-agnostic and slow: $|B|\cdot$`n_samples` predictions per explained row. for trees there's an exact polynomial-time algorithm (TreeSHAP; Lundberg et al., 2020), but the regression view is what makes Shapley values practical for any model.

> **spec.** `sample_coalitions(d, n_samples, rng=None)` returns bool masks $(n_\text{samples}, d)$ drawn as in step 2 (sizes with `as_rng(rng).choice` and the normalized probabilities, then a uniform subset of each size), never empty or full. `kernel_shap(predict, X_background, x, *, n_samples=2048, rng=None)`: `predict` maps $(m, d)$ to $(m,)$, `x` is $(d,)$. in the sampling branch it uses `sample_coalitions(d, n_samples, rng)` with weight 1 per sample. returns $\phi$ $(d,)$ with `phi.sum() + mean(predict(X_background)) == predict(x[None])[0]` exactly. with one feature, $\phi = f(x) - E_B[f]$.

> **🔮 predict:** with 13 features and 600 sampled coalitions, how close will KernelSHAP get to the exact values for a *linear* model?

> **▶ build: stage 4.** implement `sample_coalitions` and `kernel_shap`.

## worked example

a two-feature model $f(z) = z_1z_2 + z_1$, a background of two rows $B = \{(0, 0), (2, 2)\}$, and the point $x = (1, 3)$.

| coalition $S$ | rows evaluated | $v(S)$ |
|---|---|---|
| $\varnothing$ | $f(0, 0) = 0$, $f(2, 2) = 6$ | 3 |
| $\{1\}$ | $f(1, 0) = 1$, $f(1, 2) = 3$ | 2 |
| $\{2\}$ | $f(0, 3) = 0$, $f(2, 3) = 8$ | 4 |
| $\{1, 2\}$ | $f(1, 3) = 4$ | 4 |

with two players each order has weight $\frac12$:

$$\phi_1 = \tfrac12\big[(v_{\{1\}} - v_\varnothing) + (v_{\{1,2\}} - v_{\{2\}})\big] = \tfrac12[(2 - 3) + (4 - 4)] = -0.5,$$
$$\phi_2 = \tfrac12\big[(v_{\{2\}} - v_\varnothing) + (v_{\{1,2\}} - v_{\{1\}})\big] = \tfrac12[(4 - 3) + (4 - 2)] = 1.5.$$

check efficiency: $-0.5 + 1.5 = 1 = f(x) - v(\varnothing) = 4 - 3$. notice that $x_1 = 1$ gets a *negative* share although it equals the background mean of $z_1$: on this background $f$ is large only when $z_1 = 2$ meets $z_2 = 2$, so fixing $z_1 = 1$ rules out the big value 6 and lowers the expected prediction ($v_{\{1\}} = 2 < v_\varnothing = 3$). Shapley values compare with the background's *predictions*, not with its feature means. `masked_predictions`, `shapley_values_exact` and `kernel_shap` must all reproduce these numbers.

## on paper

1. **why permuted importance is ~0 for unused features.** show that if $f$ doesn't depend on $x_j$, then $I_j = 0$ for every permutation, on any data.

<details><summary>solution</summary>

permuting column $j$ changes only column $j$ of every row. if $f(x)$ doesn't depend on $x_j$, every prediction is unchanged, so the score is unchanged and $I_j = s - s = 0$ exactly (not just on average). the checks test this with a hand-made model.

</details>

2. **Shapley values of a linear model.** for $f(x) = w^\top x + c$ and the interventional value function, show $\phi_j = w_j(x_j - \bar b_j)$.

<details><summary>solution</summary>

$v(S) = \frac{1}{|B|}\sum_b\big(\sum_{k\in S}w_kx_k + \sum_{k\notin S}w_kb_k + c\big) = c + \sum_{k\in S}w_kx_k + \sum_{k\notin S}w_k\bar b_k$. so $v(S \cup \{j\}) - v(S) = w_j(x_j - \bar b_j)$ for every $S$, and the weights sum to 1. the KernelSHAP regression is then exact with any set of coalitions, since $v$ is exactly linear in $z$.

</details>

3. **the kernel's mass per size.** show that summing $\pi(z)$ over all coalitions of size $s$ gives $\frac{d - 1}{s(d - s)}$, and use it to justify the sampling recipe.

<details><summary>solution</summary>

there are $\binom ds$ coalitions of size $s$, each with weight $\frac{d - 1}{\binom ds s(d - s)}$; the product is $\frac{d - 1}{s(d - s)}$. drawing a size with probability proportional to that and then a uniform subset of the size draws each coalition with probability proportional to $\pi(z)$. a sample drawn in proportion to the weights, with equal weights, has the same expected weighted sum of squares as the full weighted problem: importance sampling with the weights moved into the sampling.

</details>

4. **interactions split evenly.** for $f(x) = x_1x_2$ with independent features of background mean 0, compute $\phi_1$ and $\phi_2$ at a point $x$.

<details><summary>solution</summary>

$v(\varnothing) = E[b_1b_2] = 0$, $v(\{1\}) = x_1E[b_2] = 0$, $v(\{2\}) = 0$, $v(\{1, 2\}) = x_1x_2$. so $\phi_1 = \frac12[(0 - 0) + (x_1x_2 - 0)] = \frac{x_1x_2}{2} = \phi_2$. the interaction's effect is shared equally, and $\phi_1$ depends on $x_2$: plotted against $x_1$, the SHAP values fan out by the sign of $x_2$ (the lab's picture). (with a finite background the means are only approximately 0.)

</details>

## experiment

`lab.py` investigates three things with your tools, on your forests and boosters:

1. impurity importance against permutation importance (on training and test rows) when only one feature matters and the noise features have different numbers of distinct values.
2. permutation importance against drop-column importance when two features are near copies.
3. partial dependence, ICE curves and KernelSHAP values for a model with an interaction.

## check your understanding

1. permutation importance can be negative. what does that mean?

<details><summary>answer</summary>

the model scored *better* with the feature shuffled: on this data, its use of the feature hurts (typically noise it overfitted), or the drop is within the noise of the permutations. a negative mean with a large spread just means "no evidence it matters".

</details>

2. why not just drop each feature and refit, instead of permuting?

<details><summary>answer</summary>

you can (drop-column importance), but it answers a different question ("how much worse would a model *without* this feature be?") and costs $d$ refits. permutation importance asks how much *this fitted model* relies on the feature, and needs no refitting.

</details>

3. a feature has a large impurity importance and zero permutation importance on the test set. what's the likely story?

<details><summary>answer</summary>

the trees split on it a lot to fit noise in the training data (it probably has many distinct values), and those splits don't generalize. it's the bias of idea 1.

</details>

4. when is partial dependence a faithful summary of a feature's effect?

<details><summary>answer</summary>

when the feature doesn't interact with the others (parallel ICE curves) and isn't strongly correlated with them (so the modified rows are realistic). otherwise look at ICE curves, or at SHAP values coloured by the suspected interacting feature.

</details>

5. Shapley values add up to $f(x) - E[f]$. why is that property useful, and what does it *not* guarantee?

<details><summary>answer</summary>

the explanation is complete: nothing is left unexplained, and contributions can be compared on the scale of the prediction. it doesn't guarantee that the explanation is causal, stable under correlated features, or what a person would call the "reason": with a near copy $x_3$ of $x_1$, the credit can be split between them in any proportion depending on how the model happened to use them.

</details>

6. why does KernelSHAP need a background dataset at all?

<details><summary>answer</summary>

to define what "feature $j$ is unknown" means: the model can't take a missing input, so unknown features are filled in from real rows and the predictions averaged. a different background (all training rows, one cluster, a single reference row) gives a different, equally valid, explanation of a different question.

</details>

## common bugs

- **shuffling in place.** symptom: the second feature's importance is computed on data whose first column is still shuffled, and the caller's `X` is scrambled. copy, and restore or re-copy per feature.
- **a shuffled column that stays shuffled.** symptom: the next feature's importance is measured with the previous feature still scrambled. restore the column (or start from a fresh copy) before moving on to the next feature. (permuting an already-permuted column again is harmless: it's still a uniform random permutation, independent of the last one; permuting the original is just easier to reproduce.)
- **importance on the training set.** symptom: noise features with large permutation importance. use held-out rows.
- **a Python loop over the grid or the background.** symptom: partial dependence and KernelSHAP that take minutes. build all the modified rows and call the model once.
- **forgetting the constraint.** symptom: SHAP values that don't add up to $f(x) - E[f]$. eliminate one variable with the efficiency constraint (or add the two extreme coalitions with a huge weight).
- **sampling coalitions uniformly, or weighting them twice.** symptom: estimates that don't converge to the exact values. the bias is real but often small (around 1% of the largest $|\phi|$ on the checks' 13-feature model), so you won't see it in one explanation: check your sampler's size histogram against $\frac{d - 1}{s(d - s)}$, and give sampled coalitions equal weights.

## going deeper

- Breiman (2001), "Random forests", *Machine Learning* 45(1): 5–32. permutation importance as introduced for forests. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
- Strobl, Boulesteix, Zeileis & Hothorn (2007), "Bias in random forest variable importance measures: illustrations, sources and a solution", *BMC Bioinformatics* 8: 25. [doi:10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25)
- Fisher, Rudin & Dominici (2019), "All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously", *JMLR* 20(177). model reliance. [jmlr.org/papers/v20/18-760](https://jmlr.org/papers/v20/18-760.html)
- Friedman (2001), "Greedy function approximation: a gradient boosting machine", *Annals of Statistics* 29(5), §8.2 (partial dependence). [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
- Goldstein, Kapelner, Bleich & Pitkin (2015), "Peeking inside the black box: visualizing statistical learning with plots of individual conditional expectation", *Journal of Computational and Graphical Statistics* 24(1): 44–65. [doi:10.1080/10618600.2014.907095](https://doi.org/10.1080/10618600.2014.907095)
- Shapley (1953), "A value for n-person games", in *Contributions to the Theory of Games II*, Princeton University Press, 307–317.
- Lundberg & Lee (2017), "A unified approach to interpreting model predictions", *NeurIPS* 30. SHAP and KernelSHAP. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- Lundberg et al. (2020), "From local explanations to global understanding with explainable AI for trees", *Nature Machine Intelligence* 2: 56–67. TreeSHAP. [doi:10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)
- Molnar (2022), *Interpretable Machine Learning*, 2nd ed., chapters on permutation importance, PDP, ICE and SHAP. [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/)

## connections

- ← builds on: impurity importance (topics 01–02), forests and boosters to interpret (topics 02–04), scorers and held-out evaluation (ch 07), weighted least squares (ch 02).
- → used in: the capstone's analysis of what your booster learned; feature attribution and saliency for neural networks (integrated gradients, which satisfies Shapley-style axioms, in ch 28); mechanistic interpretability (ch 28).
