# metrics

## why this matters

a credit-card fraud model is 99.2% accurate. it sounds superb until you learn that 99% of transactions aren't fraud, so a model that never flags anything is 99.0% accurate, and that this one catches about one fraud in five. the number was correct; it just answered the wrong question.

a metric is a choice about what "good" means, and different choices rank the same models differently. accuracy counts mistakes and ignores which kind. precision and recall separate the two kinds. ROC curves and AUC ask whether the model *ranks* positives above negatives, ignoring the threshold. precision–recall curves ask the same with the rare class in focus. log loss and calibration ask whether a predicted "70%" really means 70%. regression has its own family. none of these is "the right one": the right one is the one that matches the decision the model will be used for.

this topic builds `mlfs.metrics`, the scoreboard for the rest of the course: counting metrics, ROC and PR curves, log loss and regression metrics, calibration curves and reliability diagrams, and `make_scorer`, which turns any metric into the `scorer(estimator, X, y)` that topic 02's `cross_validate` expects.

## warm-up: predict first

1. 1% of transactions are fraudulent. what accuracy does "never flag anything" get? what recall?
2. a model's AUC is 0.96. what does that number mean, in one sentence, without the word "curve"?
3. you double the number of negatives in your test set (same distribution). which change: accuracy, precision, recall, AUC?
4. two models give the same predictions at threshold 0.5, so they have the same accuracy. can one be clearly better than the other?
5. a model predicts 0 for an example whose label is 1. what's its log loss on that example?

## prerequisites check

- [ ] you know the `predict_proba` convention: column $j$ is the probability of `classes_[j]` (→ topic 01, idea 3)
- [ ] you know `cross_validate` and what a scorer is (→ topic 02, idea 5)
- [ ] you know cross-entropy and KL divergence, and that $\mathrm{KL}(q \| p) \ge 0$ (→ ch 05 · topic 05)
- [ ] you know conditional probability and Bayes' rule (→ ch 04 · topic 01)
- [ ] you can `argsort`, `cumsum` and `bincount` without loops (→ ch 00 · topic 02)

## core ideas

### idea 1: counting mistakes

a classifier's predictions on a test set fit in a **confusion matrix**: row $i$ is the true class, column $j$ the predicted one, and $C_{ij}$ counts examples. for two classes, with "positive" the class you care about:

| | predicted negative | predicted positive |
|---|---|---|
| **truly negative** | true negatives (TN) | false positives (FP) |
| **truly positive** | false negatives (FN) | true positives (TP) |

**accuracy** is the diagonal over the total. it treats both mistakes alike, and with a rare positive class it's dominated by the easy majority (warm-up 1). two ratios separate them:

$$\text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}.$$

precision: of the examples you flagged, how many were positive. recall (also called sensitivity, or the true positive rate): of the positives, how many you flagged. they pull against each other: flag more and recall rises while precision usually falls. the **F1 score** is their harmonic mean, which is low if *either* is low:

$$F_1 = \frac{2PR}{P + R} = \frac{2\,TP}{2\,TP + FP + FN}.$$

the right-hand form shows that F1 ignores true negatives entirely, which is exactly what you want when negatives are plentiful and boring. when a denominator is zero (a model that never predicts positive has no precision), the convention here is to report 0.

with $k$ classes, compute TP, FP and FN for each class against all the others, then average:

- **macro**: the plain mean of the $k$ per-class scores. every class counts the same, so rare classes matter.
- **weighted**: the mean weighted by each class's **support**, its number of true examples.
- **micro**: add up TP, FP and FN over all classes first, then apply the formula. for single-label multiclass problems micro-precision, micro-recall and micro-F1 all equal accuracy (on paper 4).

> **▶ build: stage 1.** implement `accuracy_score`, `confusion_matrix` (rows = truth, sorted labels by default; count with `bincount`, not a Python loop), `precision_recall_fscore` with `average` in `binary`/`macro`/`micro`/`weighted`, and the three one-line wrappers `precision_score`, `recall_score`, `f1_score`.

### idea 2: ranking, ROC and AUC

most classifiers produce a **score** (a probability, a margin) and turn it into a label with a threshold $t$: positive if $s > t$, or $\ge t$. every threshold gives a different confusion matrix. instead of picking one, sweep them all. at each threshold record

$$\text{TPR} = \frac{TP}{P} \ (\text{recall}), \qquad \text{FPR} = \frac{FP}{N},$$

where $P$ and $N$ are the numbers of positives and negatives. the **ROC curve** plots TPR against FPR as $t$ falls from $+\infty$ (nothing flagged, the point $(0, 0)$) to $-\infty$ (everything flagged, $(1, 1)$). computing it takes one sort: order the examples by decreasing score, and `cumsum` the labels to get TP at every cut. tied scores must be crossed together, because no threshold separates them, so the curve takes one diagonal step per distinct score.

the **area under the curve** has a meaning that doesn't mention curves at all (warm-up 2):

$$\text{AUC} = P\big(s_+ > s_-\big) + \tfrac12 P\big(s_+ = s_-\big),$$

the probability that a randomly chosen positive gets a higher score than a randomly chosen negative (on paper 1). this is the **Mann–Whitney** statistic. it follows that AUC depends only on the *ranking*: any increasing transformation of the scores leaves it unchanged, reversing the ranking gives $1 - \text{AUC}$, random scores give 0.5, and a perfect ranking gives 1.

TPR is computed only from positives and FPR only from negatives, so neither depends on how many of each you have: **ROC curves are insensitive to the class balance** (warm-up 3). that's a strength (you can compare across datasets) and a weakness, as the next idea shows.

> **▶ build: stage 2.** implement `binary_clf_curve` (cumulative false and true positives at each distinct score, highest first), `roc_curve` (prepend $(0, 0)$ with threshold $\infty$) and `roc_auc_score` (trapezoid rule; the larger label is the positive class).

### idea 3: precision–recall, for rare positives

with 1% fraud, an FPR of 2% sounds small, but 2% of the 99% negatives is about twice as many false alarms as there are frauds in total. ROC hides this; precision shows it. by Bayes' rule, with prevalence $\pi = P/(P + N)$,

$$\text{precision} = \frac{\text{TPR}\,\pi}{\text{TPR}\,\pi + \text{FPR}\,(1 - \pi)},$$

so the same model, at the same threshold, has lower precision when positives are rarer (on paper 2). the **precision–recall curve** traces precision against recall over all thresholds. its summary, **average precision**, is the mean of the precision values reached each time recall increases:

$$\text{AP} = \sum_k (R_k - R_{k-1})\,P_k.$$

it's a step-function area with no interpolation (a trapezoid between PR points would be optimistic, because precision doesn't vary linearly between them). a random ranking has AP equal to the prevalence, not 0.5, which is the honest baseline when positives are rare.

following scikit-learn, `precision_recall_curve` returns its arrays with thresholds *increasing* (so recall decreases), plus a final point with precision 1 and recall 0 that has no threshold.

> **🔮 predict:** the same model, with the same score distributions for positives and negatives, is evaluated on data with 50% and with 0.1% positives. which of ROC AUC and AP changes, and by roughly how much?

> **▶ build: stage 3.** implement `precision_recall_curve` (from `binary_clf_curve`, reversed, with the extra point) and `average_precision_score`.

### idea 4: scoring probabilities, and regression

thresholded metrics throw away the score; AUC keeps only its order. when the *values* matter (a doctor reading "30% risk", a system that multiplies probabilities by costs), score the probabilities directly. the standard is the **log loss**, the average negative log-likelihood of the true labels:

$$\text{log loss} = -\frac1n\sum_{i=1}^{n} \log \hat p_{i, y_i},$$

where $\hat p_{i, y_i}$ is the probability the model gave to example $i$'s true class. it's the cross-entropy between the labels and the predictions (chapter 05), and it's a **proper scoring rule**: if the true probabilities are $q$, the expected log loss is minimized by reporting exactly $p = q$ (on paper 3). no amount of hedging or exaggerating helps. its weakness is warm-up 5: a confident mistake costs $-\log 0 = \infty$, so implementations clip the probability to $[\varepsilon, 1 - \varepsilon]$; with $\varepsilon = 10^{-15}$ a "certain" mistake costs about 34.5 nats.

for regression the three workhorses are

$$\text{MSE} = \frac1n\sum_i (y_i - \hat y_i)^2, \qquad \text{MAE} = \frac1n\sum_i |y_i - \hat y_i|, \qquad R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$$

MSE is in squared units and dominated by large errors; the best constant under MSE is the mean, under MAE the median (topic 01). $R^2$ is MSE relative to the mean predictor, so it's unitless and comparable across targets, but it depends on the spread of $y$ in the test set.

> **▶ build: stage 4.** implement `log_loss` (binary `y_prob` of shape `(n,)` is $P(\text{class } 1)$; clip the true-class probability to $[\varepsilon, 1 - \varepsilon]$), `mean_squared_error`, `mean_absolute_error` and `r2_score`.

### idea 5: calibration

a model is **calibrated** if, among all examples it gives probability 0.7, about 70% are positive. ranking and calibration are separate properties: you can wreck the calibration of a perfect ranker by squashing all its probabilities towards 0.5, and AUC won't notice.

to check calibration, bin the predicted probabilities into equal-width bins on $[0, 1]$, and in each non-empty bin compare the **mean predicted probability** with the **observed fraction of positives**. plotted against each other, that's a **reliability diagram**: a calibrated model sits on the diagonal. overconfident models (probabilities too close to 0 and 1) give a curve flatter than the diagonal; underconfident ones (probabilities squeezed towards 0.5) give a steep S-shape. the **expected calibration error** summarizes the gaps, weighted by how many examples fall in each bin:

$$\text{ECE} = \sum_{b} \frac{n_b}{n}\,\big|\text{acc}_b - \text{conf}_b\big|.$$

for multiclass models the common version bins the **top-class confidence** $\max_j \hat p_{ij}$ against whether the top class was correct (Guo et al., 2017). ECE is easy to read but it depends on the binning, and with few examples per bin it's noisy and biased upwards; treat it as a diagnostic, and log loss as the score.

> **🔮 predict:** you take a calibrated model and push all its probabilities towards 0 and 1 (multiply every logit by 3). which of accuracy at threshold 0.5, ROC AUC, log loss and ECE change?

> **▶ build: stage 5.** implement `calibration_curve` (bin $= $ `np.searchsorted(edges[1:-1], p)`, return non-empty bins only, with their counts), `expected_calibration_error` (binary or top-class), and `reliability_diagram`, which plots it all on a matplotlib `Axes`.

### idea 6: scorers, the glue

`cross_validate` needs `scorer(fitted_estimator, X, y) -> float`, higher is better. metrics have the signature `metric(y_true, y_out)`, and some are "lower is better". `make_scorer` bridges the two: it asks the estimator for the right kind of output (`predict` for labels, `predict_proba` for probabilities, `decision_function` for margins), calls the metric, and negates it when lower is better. that's why the course names losses as scorers with a `neg_` prefix: `neg_log_loss`, `neg_mean_squared_error`.

one subtlety connects back to topic 01. metrics like `log_loss` want labels $0..k-1$ that index the probability columns, but labels can be anything, like `"ham"` and `"spam"`. a probability scorer therefore converts $y$ to positions in `estimator.classes_` (sorted, so `"spam"` is column 1), and uses column 1 for binary problems. that convention is the reason classifiers sort `classes_`.

> **▶ build: stage 6.** implement `make_scorer(metric, *, greater_is_better=True, response="predict")` and `get_scorer(name)` for `accuracy`, `f1`, `roc_auc`, `neg_log_loss`, `neg_mean_squared_error`, `neg_mean_absolute_error`, `r2`.

## worked example

8 emails, 4 of them spam (label 1), with scores from some model:

| email | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| label | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| score | 0.9 | 0.8 | 0.7 | 0.6 | 0.55 | 0.4 | 0.3 | 0.1 |

**at threshold 0.5** the model flags emails 1–5: TP = 3 (emails 1, 2, 4), FP = 2 (3, 5), FN = 1 (7), TN = 2 (6, 8). the confusion matrix is $\begin{pmatrix}2 & 2\\ 1 & 3\end{pmatrix}$, accuracy $5/8 = 0.625$, precision $3/5 = 0.6$, recall $3/4 = 0.75$, $F_1 = 6/9 \approx 0.667$.

**ROC.** walk down the sorted scores; each positive moves the curve up by $1/P = 0.25$, each negative right by $1/N = 0.25$. the points are $(0,0), (0, .25), (0, .5), (.25, .5), (.25, .75), (.5, .75), (.75, .75), (.75, 1), (1, 1)$. the area is the sum, over the steps to the right, of the height at that step: $0.25 \times (0.5 + 0.75 + 0.75 + 1) = 0.75$. check it as a ranking probability: the positives (0.9, 0.8, 0.6, 0.3) beat 4, 4, 3 and 1 of the 4 negatives, so $12/16 = 0.75$. ✓

**average precision.** the positives are reached at ranks 1, 2, 4 and 7, with precision $1/1, 2/2, 3/4, 4/7$ there, and each adds $1/4$ to recall: $\text{AP} = \frac14(1 + 1 + 0.75 + 0.571) \approx 0.830$.

the tests check all of these numbers.

## on paper

1. **AUC is a ranking probability.** for scores without ties, show that the trapezoid area under the ROC curve equals the fraction of (positive, negative) pairs in which the positive scores higher.

<details><summary>solution</summary>

without ties, the curve moves either up ($1/P$, a positive) or right ($1/N$, a negative). the area is a sum of rectangles, one per rightward step, of width $1/N$ and height $\text{TPR} = (\#\text{positives scored above this negative})/P$. summing over negatives gives $\frac{1}{PN}\sum_{\text{negatives}} \#\{\text{positives above it}\}$, the fraction of correctly ordered pairs. with ties, a tied block is a diagonal segment, and its trapezoid contributes half of each tied pair.

</details>

2. **precision depends on prevalence.** derive $\text{precision} = \frac{\text{TPR}\,\pi}{\text{TPR}\,\pi + \text{FPR}(1 - \pi)}$. with TPR = 0.8 and FPR = 0.02, what's the precision at $\pi = 0.5$ and at $\pi = 0.001$?

<details><summary>solution</summary>

with $n$ examples, $TP = \text{TPR} \cdot \pi n$ and $FP = \text{FPR} \cdot (1 - \pi)n$; divide $TP$ by $TP + FP$ and cancel $n$. at $\pi = 0.5$: $0.4/(0.4 + 0.01) \approx 0.976$. at $\pi = 0.001$: $0.0008/(0.0008 + 0.01998) \approx 0.038$: 96% of the alarms are false.

</details>

3. **log loss is proper.** for one binary example with true probability $q$ of being positive, the expected log loss of reporting $p$ is $L(p) = -q\log p - (1 - q)\log(1 - p)$. show that it's minimized at $p = q$, and that $L(p) - L(q)$ is a KL divergence.

<details><summary>solution</summary>

$L'(p) = -q/p + (1 - q)/(1 - p) = 0$ gives $p = q$, and $L'' > 0$. and $L(p) - L(q) = q\log\frac qp + (1 - q)\log\frac{1 - q}{1 - p} = \mathrm{KL}(\text{Bern}(q)\,\|\,\text{Bern}(p)) \ge 0$, with equality only at $p = q$ (chapter 05).

</details>

4. **micro-averaging.** show that for single-label multiclass predictions, micro-precision, micro-recall and micro-F1 all equal accuracy.

<details><summary>solution</summary>

summed over classes, TP counts every correct prediction once. every wrong prediction is a false positive for the predicted class *and* a false negative for the true class, so $\sum FP = \sum FN = $ the number of mistakes $m$. then micro-precision $= \frac{TP}{TP + m}$, micro-recall the same, and micro-F1 $= \frac{2TP}{2TP + 2m}$: all equal to $TP/n$, the accuracy.

</details>

## experiment

`lab.py` uses your metrics on three questions:

1. a fraud detector with 1% positives: accuracy versus precision, recall, F1, AUC and AP, for a dummy, an accuracy-maximizing stump and a recall-friendly threshold.
2. the same model at different prevalences: ROC curves versus precision–recall curves.
3. three models with identical rankings and different probabilities: what accuracy, AUC, log loss and the reliability diagram can and can't see.

## check your understanding

1. a cancer screening test and a spam filter: which errors matter in each, and which metric would you lead with?

<details><summary>answer</summary>

screening: missing a cancer (FN) is far worse than a follow-up test (FP), so lead with recall (sensitivity), at an acceptable precision or specificity. spam: losing a real email (FP) is worse than seeing some spam, so lead with precision, at an acceptable recall. in both cases report the confusion matrix too.

</details>

2. why is the ROC curve built from *distinct* scores, with one step per tie block?

<details><summary>answer</summary>

a threshold can't separate examples with the same score, so no classifier based on the score can realize the intermediate "staircase" points. taking them would make the curve (and AUC) depend on the arbitrary order of tied examples. a diagonal step is the honest average.

</details>

3. average precision of a random ranking is the prevalence, while its AUC is 0.5. why the difference?

<details><summary>answer</summary>

with a random ranking, every prefix of the list has the base rate of positives, so precision is $\pi$ at every recall, and AP $\approx \pi$. TPR and FPR grow at the same rate for a random ranking, so the ROC curve is the diagonal with area 0.5 whatever the prevalence.

</details>

4. you multiply all of a model's logits by 3. what happens to accuracy at threshold 0.5, AUC, log loss and ECE?

<details><summary>answer</summary>

accuracy and AUC are unchanged: the sign of each logit (and so the 0.5 decision) and the ranking are preserved. the probabilities move towards 0 and 1, so the model becomes overconfident (if it was calibrated before): ECE rises and log loss gets worse, because confident mistakes are punished by $-\log p$.

</details>

5. why does `make_scorer` convert labels to positions in `classes_` for probability responses, but not for `predict`?

<details><summary>answer</summary>

`predict` returns labels, which compare directly with $y$. `predict_proba` returns columns, which are positions; metrics like `log_loss` need to know which column belongs to each example's true class, and that's its position in the sorted `classes_`. for a binary problem the same convention makes column 1 the positive class for AUC.

</details>

6. you report "test MSE 0.42". what else does a reader need to judge that number?

<details><summary>answer</summary>

the units and scale of $y$ (or $R^2$ instead), a baseline (the MSE of predicting the training mean on the same test set), and an uncertainty estimate (a bootstrap CI over test examples, or the spread over CV folds). from this chapter on, every number comes with all three.

</details>

## common bugs

- **swapped confusion matrix.** rows as predictions instead of truth. symptom: precision and recall swapped; everything else looks plausible.
- **staircase ties.** sorting by score and taking every row as its own threshold. symptom: AUC that depends on the input order of tied rows, and curves that don't match scikit-learn on rounded scores.
- **the trapezoid under the PR curve.** interpolating linearly between precision–recall points. symptom: AP a few points higher than it should be, especially with few positives.
- **the wrong column for binary probabilities.** using `predict_proba(X)[:, 0]` as the positive score. symptom: AUC near $1 - \text{AUC}$, log loss enormous.
- **`log(0)`.** forgetting to clip. symptom: an infinite or NaN log loss from a single confident mistake.
- **macro-F1 from macro P and R.** computing the harmonic mean of the averaged precision and recall instead of averaging the per-class F1 scores. symptom: macro-F1 off by a few points from scikit-learn's.

## going deeper

- Fawcett (2006), "An introduction to ROC analysis", *Pattern Recognition Letters* 27(8): 861–874. ROC curves, AUC, ties and convex hulls. [doi:10.1016/j.patrec.2005.10.010](https://doi.org/10.1016/j.patrec.2005.10.010)
- Davis & Goadrich (2006), "The relationship between precision-recall and ROC curves", *ICML*. why PR curves can't be linearly interpolated, and how they relate to ROC. [doi:10.1145/1143844.1143874](https://doi.org/10.1145/1143844.1143874)
- Saito & Rehmsmeier (2015), "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets", *PLOS ONE* 10(3): e0118432. [doi:10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)
- Gneiting & Raftery (2007), "Strictly proper scoring rules, prediction, and estimation", *Journal of the American Statistical Association* 102(477): 359–378. the theory behind log loss and the Brier score. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
- Niculescu-Mizil & Caruana (2005), "Predicting good probabilities with supervised learning", *ICML*. reliability diagrams for classic models. [doi:10.1145/1102351.1102430](https://doi.org/10.1145/1102351.1102430)
- Guo, Pleiss, Sun & Weinberger (2017), "On calibration of modern neural networks", *ICML*. top-class ECE, and why deep networks are overconfident. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

## connections

- ← builds on: the `classes_` and `predict_proba` conventions (topic 01), `cross_validate` and scorers (topic 02), cross-entropy and KL divergence (ch 05), Bayes' rule (ch 04).
- → used in: every evaluation from here on: scorers in learning curves and searches (topics 04, 06), the capstone, logistic regression's log loss (ch 08), cross-entropy training of every neural network (ch 14 onward), temperature scaling and conformal prediction (ch 28), and the evaluation harness for language models (ch 27).
