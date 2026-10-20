# 07 · the machine learning workflow · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. what is the difference between a hyperparameter and a fitted parameter, and how does the estimator protocol keep them apart?

<details><summary>answer</summary>

hyperparameters are chosen before seeing data (degree, $k$, regularization strength); fitted parameters are learned by `fit` (coefficients, stored training rows). `__init__` stores only hyperparameters, unchanged, under their argument names (which is what `get_params` reads back); everything `fit` learns goes into attributes ending in `_`. so `clone` can rebuild an unfitted copy from hyperparameters alone. (topic 01, idea 2)

</details>

2. why must `clone` build a new object from `get_params()` instead of deep-copying the estimator?

<details><summary>answer</summary>

a deep copy also copies the fitted state, so a "fresh" model in one CV fold could carry information from data that includes its test fold, and a search would start every candidate from the last one's fit. `clone` produces an unfitted model with the same settings, recursing into nested estimators. (topic 01, idea 2)

</details>

3. a classifier's training accuracy is 100%. what can you conclude about its test accuracy? what if the inputs can take only 8 distinct values and you have 10,000 examples?

<details><summary>answer</summary>

nothing, in general: a lookup table gets 100% on any training set and falls back to guessing on new inputs. with 8 possible inputs and 10,000 examples, every input has been seen many times, the table stores the majority label per input, and that's the Bayes-optimal rule: then memorizing generalizes. what matters is the size of the hypothesis class relative to the data. (topic 01, idea 5 and lab)

</details>

4. state the finite-class generalization bound and say what each term means.

<details><summary>answer</summary>

with probability at least $1 - \delta$, every $h \in \mathcal H$ satisfies $R(h) \le \hat R(h) + \sqrt{(\log|\mathcal H| + \log(2/\delta))/(2n)}$. the risk is at most the training error plus a penalty that grows with the (log) size of the class and shrinks like $1/\sqrt n$. it comes from Hoeffding's inequality for one $h$ and a union bound over all of them, which is the price of choosing. (topic 01, idea 5)

</details>

5. why does stratified $k$-fold assign rows to folds round-robin within class order?

<details><summary>answer</summary>

it guarantees that each class's count per fold differs by at most one (and fold sizes too), so every fold has the class proportions of the whole, which matters for rare classes and for metrics like precision and AUC. (topic 02, idea 3)

</details>

6. you have 8 recordings from each of 30 patients and want to predict a diagnosis for new patients. which splitter, and why?

<details><summary>answer</summary>

`GroupKFold` with the patient as the group. the test folds must contain *new patients*, as deployment will; otherwise the model is rewarded for recognizing patients it has already seen, which can give near-perfect CV scores with no real skill. (topic 02, idea 3 and lab)

</details>

7. state the leakage rule. which leaks more: standardizing on the full dataset before CV, or selecting the 20 features most correlated with the label on the full dataset before CV?

<details><summary>answer</summary>

everything learned from data (preprocessing, feature selection, imputation, tuning) must be learned on the training fold only and applied to the test fold. selection by correlation with the label leaks far more: it uses the test labels and can turn noise into near-perfect accuracy. standardizing leaks only a little information about the test inputs and is usually negligible. (topic 02, idea 5 and lab)

</details>

8. what does ROC AUC measure, without mentioning curves? why doesn't it change with the class balance?

<details><summary>answer</summary>

the probability that a random positive gets a higher score than a random negative (ties count half). TPR and FPR are each computed within one class, so the curve, and its area, don't depend on how many positives and negatives there are. (topic 03, idea 2)

</details>

9. why can precision collapse at low prevalence even when TPR and FPR stay the same?

<details><summary>answer</summary>

$\text{precision} = \frac{\text{TPR}\,\pi}{\text{TPR}\,\pi + \text{FPR}(1 - \pi)}$: as $\pi \to 0$ the false positives from the huge negative class swamp the true positives. that's why precision–recall curves and average precision are the honest view for rare positives. (topic 03, idea 3)

</details>

10. two models have the same accuracy and the same AUC. how can one still be better, and which metric shows it?

<details><summary>answer</summary>

their probabilities can differ in calibration: one says "0.99" when it should say "0.8". log loss (a proper scoring rule) and the reliability diagram / ECE show it; accuracy and AUC only see the decisions and the ranking. (topic 03, idea 5 and lab)

</details>

11. write the bias–variance decomposition of the expected squared error at a point, and say which term more data reduces.

<details><summary>answer</summary>

$\mathbb E[(y - \hat h_D(x))^2] = \sigma^2 + (f(x) - \bar h(x))^2 + \mathbb E_D[(\hat h_D(x) - \bar h(x))^2]$: noise + bias² + variance. more data reduces the variance (for least squares, average variance $\sigma^2 p/n$); the bias depends on the hypothesis class, and the noise on nothing you control. (topic 04, idea 2)

</details>

12. a model's training and validation errors converge at a plateau far above the noise level as data grows. what's the diagnosis and the remedy?

<details><summary>answer</summary>

high bias (underfitting): the model can't represent the function. more data won't help; use a richer model or better features. (topic 04, idea 3)

</details>

13. why must a pipeline call `transform`, not `fit_transform`, on its earlier steps in `predict`? and why does cross-validating a pipeline prevent leakage?

<details><summary>answer</summary>

in `predict`, the steps must apply the statistics learned in `fit`; refitting would learn from the data being predicted (and would scale a single row to zero). cross-validating the pipeline clones and refits the whole chain on each training fold, so every learned step sees training rows only. (topic 05, idea 5)

</details>

14. with a budget of 9 trials and two hyperparameters of which one matters, why does random search usually beat a 3 × 3 grid?

<details><summary>answer</summary>

the grid tries only 3 distinct values of the important hyperparameter and spends the rest of its budget varying the unimportant one; random search tries 9 distinct values of each. more generally, the chance that $n$ random trials all miss the top fraction $q$ is $(1 - q)^n$, whatever the dimension. (topic 06, idea 3)

</details>

15. why is a search's `best_score_` an optimistic estimate of the tuned model's performance, and what fixes it?

<details><summary>answer</summary>

it's the maximum of many noisy CV scores, and the maximum of noisy estimates is biased upwards (the winner's curse), more so with many candidates and little data. nested cross-validation runs the whole search inside each outer training fold and scores the refitted winner on the outer test fold, so the choice never sees the data it's evaluated on. (topic 06, idea 4)

</details>

## 2. by hand

1. **a stump.** one feature, $x = (1, 2, 3, 4, 5, 6, 7, 8)$, labels $y = (0, 1, 0, 0, 1, 1, 0, 1)$. list the candidate thresholds, find the stump (threshold and direction) with the fewest training mistakes, breaking ties by the smallest threshold and then $s = +1$, and give its training error.

<details><summary>solution</summary>

thresholds: $-\infty, 1.5, 2.5, \dots, 7.5$. with $s = +1$ (class 1 on the right), mistakes = positives on the left + negatives on the right. positives on the left for $i = 0..8$ points: $0, 0, 1, 1, 1, 2, 3, 3, 4$; negatives on the right: $4, 3, 3, 2, 1, 1, 1, 0, 0$. totals: $4, 3, 4, 3, 2, 3, 4, 3, 4$. the minimum is 2, at $i = 4$, threshold 4.5. with $s = -1$ the mistakes are $8 - $ those, at least 4. so: threshold 4.5, $s = +1$, training error $2/8 = 0.25$.

</details>

2. **a ROC curve.** scores $(0.9, 0.7, 0.7, 0.5, 0.3, 0.2)$ with labels $(1, 0, 1, 1, 0, 0)$. draw the ROC curve (note the tie), compute the AUC by the trapezoid rule, and check it as a ranking probability.

<details><summary>solution</summary>

$P = N = 3$. thresholds $\infty, 0.9, 0.7, 0.5, 0.3, 0.2$ give the points $(0, 0), (0, 1/3), (1/3, 2/3), (1/3, 1), (2/3, 1), (1, 1)$: the tied pair at 0.7 (one positive, one negative) is a single diagonal step. area: the diagonal step from $(0, 1/3)$ to $(1/3, 2/3)$ contributes $\frac13 \cdot \frac{1/3 + 2/3}{2} = \frac16$, then $\frac13 \cdot 1 + \frac13 \cdot 1 = \frac23$: AUC $= \frac56 \approx 0.833$. ranking: positives 0.9, 0.7, 0.5 against negatives 0.7, 0.3, 0.2: $0.9$ beats 3; $0.7$ beats 2 and ties 1 ($2.5$); $0.5$ beats 2. total $7.5/9 = 5/6$. ✓

</details>

3. **bias and variance of shrinkage.** you estimate a mean $\mu$ from $n$ observations with variance $\sigma^2$, using the shrunk estimator $\hat\mu_c = c\,\bar x$ for a constant $0 \le c \le 1$. compute its bias², variance and MSE (as an estimator of $\mu$). which $c$ minimizes the MSE, and when is it far from 1?

<details><summary>solution</summary>

bias $= (c - 1)\mu$, variance $= c^2\sigma^2/n$, MSE $= (1 - c)^2\mu^2 + c^2\sigma^2/n$. setting the derivative $-2(1 - c)\mu^2 + 2c\sigma^2/n$ to zero gives $c^* = \frac{\mu^2}{\mu^2 + \sigma^2/n}$. it's far below 1 when $\sigma^2/n$ is large compared with $\mu^2$: noisy data, few observations, a small true effect. that's the bias–variance trade-off of ridge regression (chapter 08) in one dimension.

</details>

## 3. blind rebuild (60 minutes)

with no notes and no looking at your old code, reimplement in a fresh copy of the notebooks:

- **the estimator protocol** (topic 01, stages 2 and 3): `Estimator.get_params`, `Estimator.set_params` (with `"step__param"`), `clone`, `Regressor.score`, `Classifier.predict` and `score`, `Transformer.fit_transform`;
- **`cross_validate`** (topic 02, stage 5), with int, splitter and list-of-pairs `cv`;
- **`GridSearchCV`** (topic 06, stages 1–2): `ParameterGrid`, then `BaseSearchCV.fit`, `predict` and `score` with shared folds, ties to the first candidate and a refit.

then run the checks:

```bash
pytest 07-the-machine-learning-workflow/01-what-is-learning -k "stage2 or stage3"
pytest 07-the-machine-learning-workflow/02-splitting-and-cross-validation -k stage5
pytest 07-the-machine-learning-workflow/06-hyperparameter-search -k "stage1 or stage2"
```

(copy the notebooks somewhere first, or stash your solutions, so you really start from the stubs.)

## 4. readiness

move on to chapter 08 if you got at least 80% of sections 1 and 2 together (say, 12 of the 15 concept questions and all three problems, or 13 and two of them) and every test in section 3 passes. the same checkpoint works as a placement test: if you pass it cold, skim the lessons for the conventions (`classes_` ordering, `"step__param"`, scorers as "higher is better"), make sure your `mlfs.base`, `mlfs.model_selection`, `mlfs.metrics` and `mlfs.preprocessing` pass their tests, and go on.
