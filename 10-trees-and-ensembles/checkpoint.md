# 10 · trees & ensembles · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. a node holds class counts $(6, 2)$. what are its Gini impurity and its entropy (in bits)?

<details><summary>answer</summary>

$p = (0.75, 0.25)$. Gini $= 1 - 0.5625 - 0.0625 = 0.375$; entropy $= -(0.75\log_20.75 + 0.25\log_20.25) \approx 0.811$ bits. both depend only on the proportions, which is why the split search can work on (weighted) counts. (topic 01, idea 1)

</details>

2. why does CART's split search only need one sort and one cumulative sum per feature, instead of recomputing both children for every threshold?

<details><summary>answer</summary>

the impurities depend on additive statistics (class counts, or $\sum w, \sum wy, \sum wy^2$). after sorting by the feature, the left child of the cut after position $i$ is the first $i + 1$ rows, so its statistics are a cumulative sum and the right child's are the total minus that; all thresholds are scored in one vectorized call. (topic 01, idea 2)

</details>

3. you take the logarithm of one feature before fitting a tree. what changes?

<details><summary>answer</summary>

on the training data, nothing: a strictly increasing transform keeps the order of the values, so every split partitions the rows the same way. only the threshold values (and hence predictions for new points that fall between training values) change slightly. (topic 01, on paper 3)

</details>

4. why grow a tree fully and prune it with cost-complexity pruning, rather than stop splitting once a split's gain is small?

<details><summary>answer</summary>

a split's value can lie in the splits it enables (XOR, or the sine wave in topic 01's lab, where depths 1 and 2 were equally bad). early stopping can't see that; pruning judges whole subtrees by $R(t) - R(T_t)$ per removed leaf, after the fact. (topic 01, idea 5)

</details>

5. averaging $B$ predictors with variance $\sigma^2$ and pairwise correlation $\rho$: what's the variance of the average, and what does it imply for random forests?

<details><summary>answer</summary>

$\rho\sigma^2 + (1 - \rho)\sigma^2/B$. more trees remove only the second term; the floor $\rho\sigma^2$ is lowered by decorrelating the trees, which is what bootstrap samples and random feature subsets per split do. (topic 02, idea 1)

</details>

6. what fraction of the rows is out of bag for one tree, and why is the OOB score a fair estimate of test performance?

<details><summary>answer</summary>

$(1 - 1/n)^n \approx e^{-1} \approx 0.37$. each row's OOB prediction averages only the trees that never saw it, so it's a held-out prediction; the OOB score is a slightly pessimistic estimate (about 37% of the trees per prediction) obtained without a validation split. (topic 02, idea 3)

</details>

7. bagging improves fully grown trees a lot and stumps hardly at all. why?

<details><summary>answer</summary>

bagging reduces variance and leaves bias untouched. fully grown trees have low bias and high variance, so averaging removes most of their error; stumps have high bias and low variance, so there's little to remove. (topic 02, lab experiment 3)

</details>

8. after one round of AdaBoost, what is the weighted error of the weak learner just used, measured on the *updated* weights? why is that the point?

<details><summary>answer</summary>

exactly 0.5: the update multiplies the misclassified weight $\varepsilon$ by $(1 - \varepsilon)/\varepsilon$, matching the correct weight $1 - \varepsilon$. the next learner can't repeat the last one; it must find something new. (topic 03, idea 1)

</details>

9. what loss does AdaBoost minimize, and what does that say about its `predict_proba`?

<details><summary>answer</summary>

the exponential loss $e^{-yf}$, by forward stagewise additive modelling. its population minimizer is half the log-odds (SAMME's $f$ is the log-odds itself), hence $P(y = 1\mid x) = \sigma(f(x))$. on finite data AdaBoost keeps inflating $|f|$, so the probabilities become overconfident (topic 03, lab experiment 3). (topic 03, idea 2)

</details>

10. why can AdaBoost's test error keep falling after its training error has reached zero?

<details><summary>answer</summary>

the exponential loss keeps rewarding larger margins $yf(x)/\sum\alpha$, so the ensemble keeps pushing up the smallest margins; margin-based bounds depend on the margin distribution and the weak learners' complexity, not on the number of rounds. with label noise this eventually breaks down. (topic 03, idea 3)

</details>

11. in XGBoost-style boosting, what value should a leaf take, and when is a split worth making?

<details><summary>answer</summary>

$w = -G/(H + \lambda)$ from the second-order expansion; a split is worth it when $\frac12\big[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\big] - \gamma > 0$ and both children satisfy `min_child_weight`. (topic 04, idea 2)

</details>

12. what are $g$ and $h$ for the log loss with logit $F$, and why does using $h$ matter on imbalanced data?

<details><summary>answer</summary>

$g = p - t$, $h = p(1 - p)$. with rare positives $p$ is small everywhere, so $h \approx p$ is small and the Newton step $-\sum g/\sum h$ is much larger than the plain gradient step (the mean of $t - p$); the hessian is a per-leaf step size. (topic 04, idea 5 and lab experiment 3)

</details>

13. why does a smaller learning rate usually reach a *lower* test error, not just the same one later?

<details><summary>answer</summary>

shrinkage is a regularizer: each tree commits only a fraction of its fit, including the noise it fitted, and later trees fitting slightly different residuals average much of it out. full steps commit the noise immediately. you pay with more trees. (topic 04, lab experiment 1)

</details>

14. why is impurity importance biased toward features with many distinct values, and what's the fix?

<details><summary>answer</summary>

deep trees fit noise, and a feature with more thresholds is more likely to offer a split that happens to reduce *training* impurity, so it's chosen more often and collects importance. use permutation importance on held-out data instead. (topic 05, idea 1)

</details>

15. the partial dependence of $x_1$ is flat. does the model ignore $x_1$?

<details><summary>answer</summary>

not necessarily: it may interact with another feature so that its effects cancel on average ($x_1x_2$ with symmetric $x_2$). check ICE curves or SHAP values coloured by the suspected partner. (topic 05, idea 2 and lab experiment 3)

</details>

## 2. by hand

1. **a split.** one feature with values $1, 2, 3, 4$ and labels $0, 0, 1, 1$ plus a fifth row $x = 5$, label 0. compute the weighted Gini decrease of every candidate threshold and pick the best.

<details><summary>answer</summary>

root: counts $(3, 2)$, $W\cdot I = 5\cdot(1 - 0.36 - 0.16) = 2.4$. thresholds: 1.5: left $(1, 0)$ gini 0, right $(2, 2)$ gini 0.5 → $2.4 - 0 - 2 = 0.4$. 2.5: left $(2, 0)$, right $(1, 2)$ gini $4/9$ → $2.4 - 3\cdot\frac49 = 1.067$. 3.5: left $(2, 1)$ gini $4/9$, right $(1, 1)$ gini 0.5 → $2.4 - 1.333 - 1 = 0.067$. 4.5: left $(2, 2)$ gini 0.5, right $(1, 0)$ → $2.4 - 2 = 0.4$. best: 2.5, decrease $1.067$.

</details>

2. **AdaBoost, two rounds.** four points, $y = (+1, +1, -1, -1)$, uniform weights. round 1's stump predicts $(+1, +1, +1, -1)$. compute $\varepsilon_1$, $\alpha_1$ and the new weights. round 2's stump predicts $(+1, -1, -1, -1)$: compute $\varepsilon_2$, $\alpha_2$, and $f$ and the training error after two rounds.

<details><summary>answer</summary>

round 1: point 3 is wrong, $\varepsilon_1 = 0.25$, $\alpha_1 = \ln3 \approx 1.099$; point 3's weight $\times3$: $(0.25, 0.25, 0.75, 0.25)/1.5 = (\frac16, \frac16, \frac12, \frac16)$. round 2: point 2 is wrong, $\varepsilon_2 = \frac16$, $\alpha_2 = \ln5 \approx 1.609$. $f = \alpha_1h_1 + \alpha_2h_2 = (2.708, -0.511, -0.511, -2.708)$: points 1, 3, 4 are right, point 2 is wrong (training error 0.25). check the bound: $2\sqrt{0.25\cdot0.75}\cdot2\sqrt{\frac16\cdot\frac56} = 0.866\cdot0.745 = 0.645 \ge 0.25$.

</details>

3. **a boosting step.** squared loss, current predictions $F = (2, 2, 2, 2)$ for targets $y = (1, 2, 3, 6)$; one depth-1 tree on a feature that separates rows $\{1, 2\}$ from $\{3, 4\}$; $\lambda = 1$, learning rate 0.5. compute the gain of that split, the two leaf values and the updated predictions.

<details><summary>answer</summary>

$g = F - y = (1, 0, -1, -4)$, $h = 1$. left: $G_L = 1$, $H_L = 2$; right: $G_R = -5$, $H_R = 2$; total $G = -4$, $H = 4$. gain $= \frac12\big[\frac13 + \frac{25}{3} - \frac{16}{5}\big] = \frac12(8.667 - 3.2) = 2.733$. leaves: $-1/3 = -0.333$ and $5/3 = 1.667$. updated $F = 2 + 0.5\cdot(-0.333, -0.333, 1.667, 1.667) = (1.833, 1.833, 2.833, 2.833)$.

</details>

## 3. blind rebuild (60 minutes)

with no notes and no looking at your old code, reimplement in a fresh copy of the notebooks:

- **CART** (topic 01, stages 2–3): `best_split` with cumulative sums over all features at once, and `BaseDecisionTree._grow` with an explicit stack, `apply` and `DecisionTreeClassifier.fit` / `predict_proba`;
- **the histogram booster** (topic 04, stages 2–4): `gradient_histograms`, `best_hist_split`, `HistGradientTree` (with histogram subtraction if you have time) and `GradientBoostingRegressor`.

then run the checks:

```bash
pytest 10-trees-and-ensembles/01-decision-trees -k "stage2 or stage3"
pytest 10-trees-and-ensembles/04-gradient-boosting -k "stage2 or stage3 or stage4"
```

(copy the notebooks somewhere first, or stash your solutions, so you really start from the stubs.)

## 4. readiness

move on to chapter 11 if you got at least 80% of sections 1 and 2 together (say, 12 of the 15 concept questions and all three problems, or 13 and two of them) and every test in section 3 passes. the same checkpoint works as a placement test: if you pass it cold, skim the lessons for the conventions (statistics-based impurities, `value_` arrays, SAMME's scaling, `bin_edges_` and `init_`), make sure your `mlfs.trees` and `mlfs.inspection` pass their tests, and go on.
