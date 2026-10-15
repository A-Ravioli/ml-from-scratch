# boosting

## why this matters

in 1988 Michael Kearns asked a question that sounds like a riddle: if you have a learning algorithm that's only *slightly* better than guessing (51% on any distribution of examples you feed it), can you turn it into one that's as accurate as you like? Robert Schapire proved in 1990 that you can, and in 1995 Yoav Freund and Schapire turned the proof into an algorithm simple enough to fit on an index card: **AdaBoost**. a few years later, Viola and Jones used it to pick a few hundred cheap features out of more than a hundred thousand candidates and built the first real-time face detector, the kind that drew boxes around faces in consumer cameras for the next decade.

random forests (topic 02) reduce variance: they average many strong, deep trees that disagree. boosting attacks the other term, **bias**: it adds weak, shallow trees *one at a time*, and each new tree is trained on a reweighted version of the data in which the examples the ensemble currently gets wrong count more. a stump can only draw one line; a hundred boosted stumps can draw almost anything.

AdaBoost also comes with two puzzles that shaped machine learning theory: why its test error often keeps falling long after its training error has hit zero, and what it's actually optimizing. you'll reproduce the first and derive the second.

## warm-up: predict first

1. a weak learner makes a weighted error of 0.4 on five points. after AdaBoost reweights the data, what is that same learner's weighted error on the new weights?
2. AdaBoost's training error reaches zero after 185 rounds. you keep going to 1,000 rounds. does the test error go up, stay flat, or go down?
3. 10% of the training labels are flipped at random. after 500 rounds, what share of the total weight sits on those 10% of points?
4. the decision function $f(x) = \sum_m\alpha_mh_m(x)$ keeps growing in magnitude as rounds are added. what happens to the quality of the probabilities $\sigma(f(x))$?
5. a perfect weak learner (weighted error 0) shows up in the first round. what weight $\alpha$ does the formula give it?

## prerequisites check

- [ ] your `DecisionTreeClassifier` accepts `sample_weight` and splits on weighted impurity (→ topic 01)
- [ ] you know the logistic function and log-odds (→ ch 04, ch 07 · topic 03)
- [ ] you can minimize a one-dimensional convex function by setting its derivative to zero (→ ch 03)
- [ ] you know what the bias and the variance of a model are (→ ch 07 · topic 04)

## core ideas

### idea 1: reweight the mistakes

AdaBoost keeps a weight $w_i$ on every training example, starting uniform ($1/n$). each round:

1. fit a weak learner $h_m$ (here: a depth-1 or depth-2 tree, fitted with `sample_weight=w`) with outputs in $\{-1, +1\}$;
2. compute its **weighted error** $\varepsilon_m = \sum_{i: h_m(x_i)\ne y_i}w_i \big/ \sum_iw_i$;
3. give it a vote $\alpha_m = \eta\ln\frac{1 - \varepsilon_m}{\varepsilon_m}$ (with learning rate $\eta$, 1 by default): large when $\varepsilon_m$ is small, zero at $\varepsilon_m = 0.5$;
4. multiply the weights of the **misclassified** examples by $e^{\alpha_m}$ and renormalize.

the final classifier is the sign of $f(x) = \sum_m\alpha_mh_m(x)$.

this is the "SAMME" form of discrete AdaBoost (Zhu et al., 2009) with two classes, the one scikit-learn implements. Freund and Schapire's original uses $\alpha_m/2$ and multiplies correct examples by $e^{-\alpha_m/2}$ and wrong ones by $e^{+\alpha_m/2}$; after renormalizing the weights are identical, and $f$ is just twice as large.

why $\ln\frac{1 - \varepsilon}{\varepsilon}$? with $\eta = 1$, the misclassified examples carried total weight $\varepsilon$ and the correct ones $1 - \varepsilon$; multiplying the first group by $\frac{1 - \varepsilon}{\varepsilon}$ gives both groups weight $1 - \varepsilon$. after renormalizing, **the learner you just used has error exactly 0.5 on the new weights** (warm-up 1). the next learner can't repeat it; it has to find something new.

two edge cases. a weak learner with $\varepsilon_m \ge 0.5$ can't help (it's no better than chance on the weighted data): stop. a perfect one ($\varepsilon_m = 0$) would get an infinite vote; follow scikit-learn and keep it with weight 1, then stop, since there's nothing left to correct.

how good does the ensemble get on the training set? the misclassification indicator is bounded by an exponential, $\mathbb 1[yf(x) \le 0] \le e^{-yf(x)/2}$, and the average of that exponential factors over the rounds (on paper 2):

$$\text{training error} \le \frac1n\sum_ie^{-y_if(x_i)/2} = \prod_m2\sqrt{\varepsilon_m(1 - \varepsilon_m)} \le \exp\Big(-2\sum_m\gamma_m^2\Big),\qquad \gamma_m = \tfrac12 - \varepsilon_m.$$

if every weak learner beats chance by a margin $\gamma$, the training error falls *exponentially* in the number of rounds. that's Kearns's riddle answered.

> **spec.** `samme_update(y, h, sample_weight, learning_rate=1.0)`: `y`, `h` arrays in $\{-1, +1\}$; returns `(alpha, new_weight, err)` with `err` the weighted error (weights normalized first), `alpha` as above, and `new_weight` the updated weights summing to 1. if `err == 0`, return `(1.0, normalized weights, 0.0)`. `exponential_loss(y, f)` returns `mean(exp(-y * f))` as a float. (SAMME's $f$ is twice the classic score, so the training-error bound above is `exponential_loss(y, f / 2)`; the lab plots it on top of the product.)

> **▶ build: stage 1.** implement `samme_update` and `exponential_loss`.

### idea 2: what AdaBoost minimizes

AdaBoost was invented as a theory-driven algorithm, and for five years nobody could say what loss it minimized. Friedman, Hastie and Tibshirani (2000) found the answer: AdaBoost is **forward stagewise additive modelling** with the **exponential loss** $L(y, f) = e^{-yf}$.

forward stagewise means: keep the current model $F_{m-1}$ fixed and add the single term $\beta h$ that reduces the loss most,

$$\min_{\beta, h}\sum_ie^{-y_i(F_{m-1}(x_i) + \beta h(x_i))} = \min_{\beta, h}\sum_iw_i\,e^{-\beta y_ih(x_i)},\qquad w_i = e^{-y_iF_{m-1}(x_i)}.$$

the weights fall out on their own: they're the current loss of each example, which is large exactly where the ensemble is wrong. for any $\beta > 0$, the best $h$ minimizes the weighted error, which is what a weighted tree approximately does. then the best $\beta$ is $\frac12\ln\frac{1 - \varepsilon}{\varepsilon}$ (on paper 1), and the new weights $w_ie^{-\beta y_ih(x_i)}$ are AdaBoost's update. SAMME's $\alpha = 2\beta$ is the same model at twice the scale.

this view says what $f$ estimates. the exponential loss is minimized, point by point, by $\frac12\ln\frac{P(y = 1\mid x)}{P(y = -1\mid x)}$ (on paper 3): half the log-odds. with SAMME's doubled scale, $f(x)$ estimates the log-odds itself, so

$$P(y = +1\mid x) \approx \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}.$$

that's your `predict_proba`. (compute it as $\frac12(1 + \tanh(f/2))$ so that large $|f|$ can't overflow.) it also explains AdaBoost's weakness: the exponential loss punishes a confidently wrong example by $e^{|f|}$, exponentially, so a mislabeled point gets ever more weight and drags tree after tree toward it (warm-up 3).

> **spec.** `AdaBoostClassifier(n_estimators=50, *, learning_rate=1.0, max_depth=1, rng=None)`, binary only (`ValueError` otherwise). `fit` maps `classes_[1]` to $+1$ and `classes_[0]` to $-1$, starts from uniform weights, and each round fits `DecisionTreeClassifier(max_depth=..., rng=gen)` with `sample_weight=w` on the class codes, converts its predictions to $\pm1$ and calls `samme_update`; a round with `err >= 0.5` is discarded and ends the loop (`ValueError` if it's the first); a round with `err == 0` is kept and ends it. it stores `estimators_`, `estimator_weights_` and `estimator_errors_` (arrays) and returns `self`. `decision_function` $= \sum_m\alpha_mh_m(x)$ $(n,)$; `predict_proba` $= [1 - \sigma(f), \sigma(f)]$; `predict` and `score` come from `Classifier`.

> **🔮 predict:** you boost 40 stumps on a problem whose true boundary is a parabola. how much better than a single stump will the test accuracy be?

> **▶ build: stage 2.** implement `AdaBoostClassifier.fit`, `decision_function` and `predict_proba`.

### idea 3: margins

the training error bound explains why AdaBoost fits the training data. it doesn't explain the puzzle that made AdaBoost famous: on many datasets the *test* error keeps falling for hundreds of rounds after the *training* error has reached zero. the model gets more complex every round, so classical intuition says it should overfit.

Schapire, Freund, Bartlett and Lee (1998) explained it with **margins**. the normalized margin of a training example,

$$\text{margin}(x_i, y_i) = \frac{y_if(x_i)}{\sum_m\alpha_m} \in [-1, 1],$$

is positive when the example is classified correctly, and its size says *how decisively*: a margin of 1 means every weak learner agrees, a margin near 0 means a narrow majority. once every margin is positive, training error is zero, but AdaBoost keeps going: the exponential loss still rewards making $yf$ larger, so it keeps pushing up the smallest margins. their theorem bounds the test error by the fraction of training examples with margin below $\theta$ plus a term that depends on $\theta$, the sample size and the complexity of the *weak* learners, but **not on the number of rounds**. larger margins, better generalization, however many trees.

the margin story isn't the whole story. Breiman (1999) built an algorithm that maximizes the minimum margin more aggressively than AdaBoost and generalizes worse, and with noisy labels AdaBoost does overfit eventually. but it's the right first explanation, and the same idea, that confident correct classification generalizes better than barely-correct classification, reappears in support vector machines (ch 09) and in modern deep learning theory (ch 19).

to watch the ensemble grow you need its decision function after every round: the **staged** decision function. it yields $f_1, f_2, \dots, f_M$, each one step of $\pm\alpha_m$ away from the previous one.

> **spec.** `staged_decision_function(X)` is a generator yielding $f$ after rounds $1, \dots, M$, each a new array $(n,)$ (don't yield the same array object updated in place). `margins(X, y)` returns $y\,f(x)/\sum_m\alpha_m$ $(n,)$ with $y = +1$ for `classes_[1]` and $-1$ otherwise.

> **🔮 predict:** the training error reaches zero at round 185 of 1,000. what happens to the *smallest* training margin over the remaining rounds, and to the test error?

> **▶ build: stage 3.** implement `staged_decision_function` and `margins`.

## worked example

five points with labels $y = (+1, +1, +1, -1, -1)$ and uniform weights $0.2$. the first stump predicts $h = (+1, +1, -1, -1, +1)$: it gets points 3 and 5 wrong.

- weighted error: $\varepsilon = 0.2 + 0.2 = 0.4$.
- vote: $\alpha = \ln\frac{0.6}{0.4} = \ln1.5 \approx 0.405$.
- reweight: points 3 and 5 are multiplied by $e^{\alpha} = 1.5$, giving $(0.2, 0.2, 0.3, 0.2, 0.3)$, total 1.2. renormalized: $(\frac16, \frac16, \frac14, \frac16, \frac14)$.
- check: the misclassified points now carry $\frac14 + \frac14 = \frac12$. the stump is a coin flip on the new weights.
- the bound after one round: $2\sqrt{0.4\cdot0.6} = 0.980$. the actual training error is $0.4$, and $\frac15\sum_ie^{-y_if(x_i)/2}$ with $f = 0.405\,h$ is $\frac15(3e^{-0.203} + 2e^{0.203}) = 0.980$: the identity holds with equality.

your `samme_update` must reproduce $\varepsilon$, $\alpha$ and the new weights; with `learning_rate=0.5` the vote halves.

## on paper

1. **the optimal step.** minimize $J(\beta) = \sum_iw_ie^{-\beta y_ih(x_i)}$ over $\beta$, for normalized weights and a fixed $h$ with weighted error $\varepsilon$.

<details><summary>solution</summary>

split the sum: $J(\beta) = (1 - \varepsilon)e^{-\beta} + \varepsilon e^{\beta}$. setting $J'(\beta) = -(1 - \varepsilon)e^{-\beta} + \varepsilon e^{\beta} = 0$ gives $e^{2\beta} = \frac{1 - \varepsilon}{\varepsilon}$, $\beta = \frac12\ln\frac{1 - \varepsilon}{\varepsilon}$. at the optimum $J = 2\sqrt{\varepsilon(1 - \varepsilon)}$: the factor by which one round shrinks the exponential loss.

</details>

2. **the training error identity.** with $f = \sum_m\alpha_mh_m$ (SAMME, $\eta = 1$), show $\frac1n\sum_ie^{-y_if(x_i)/2} = \prod_m2\sqrt{\varepsilon_m(1 - \varepsilon_m)}$.

<details><summary>solution</summary>

with $\beta_m = \alpha_m/2$, the normalized weights before round $m + 1$ are $w_i^{(m+1)} = w_i^{(m)}e^{-\beta_my_ih_m(x_i)}/Z_m$, where $Z_m = \sum_iw_i^{(m)}e^{-\beta_my_ih_m(x_i)} = 2\sqrt{\varepsilon_m(1 - \varepsilon_m)}$ by on paper 1 (and these are the same weights as SAMME's, since SAMME's update differs only by a common factor). unrolling from $w^{(1)} = 1/n$: $w_i^{(M+1)} = \frac1ne^{-y_if(x_i)/2}/\prod_mZ_m$. the left side sums to 1, so $\frac1n\sum_ie^{-y_if(x_i)/2} = \prod_mZ_m$. finally $2\sqrt{\varepsilon(1 - \varepsilon)} = \sqrt{1 - 4\gamma^2} \le e^{-2\gamma^2}$.

</details>

3. **the population minimizer.** for a fixed $x$ with $p = P(y = 1\mid x)$, minimize $E[e^{-yf}\mid x] = pe^{-f} + (1 - p)e^{f}$ over $f$.

<details><summary>solution</summary>

the derivative $-pe^{-f} + (1 - p)e^{f} = 0$ gives $e^{2f} = \frac{p}{1 - p}$, $f^* = \frac12\ln\frac{p}{1 - p}$: half the log-odds. inverting, $p = \sigma(2f^*)$. SAMME's $f$ is $2f^*$, so $p = \sigma(f)$. the logistic loss $\ln(1 + e^{-2yf})$ has the same minimizer; the losses differ in how hard they punish large negative margins (linearly versus exponentially), which is why gradient boosting with the logistic loss (topic 04) is more robust to label noise.

</details>

4. **why stop at $\varepsilon \ge 0.5$?** what is $\alpha$ for $\varepsilon = 0.5$ and for $\varepsilon = 0.6$? what would adding such a learner do?

<details><summary>solution</summary>

$\alpha = \ln1 = 0$ at 0.5: the learner adds nothing and the weights don't change, so the next round would fit the same tree again, forever. at 0.6, $\alpha = \ln(2/3) < 0$: the learner gets a *negative* vote, which for two classes is the same as flipping it, a learner with error 0.4. a weighted tree should never be worse than chance on its own training weights, so in practice $\varepsilon \ge 0.5$ means the weak learner has run out of things to find.

</details>

## experiment

`lab.py` investigates three things with your AdaBoost:

1. training error, test error and the margin distribution over 1,000 rounds, before and after the training error reaches zero.
2. where the weight goes when some labels are wrong, with `learning_rate` 1 and 0.1.
3. what happens to the probabilities $\sigma(f(x))$ as rounds are added.

## check your understanding

1. why must the weak learner accept sample weights? what could you do if it didn't?

<details><summary>answer</summary>

the reweighting is how each round is told what to focus on. without weights, you can **resample**: draw a training set of size $n$ with probabilities $w_i$ (boosting by resampling). it works, with extra noise.

</details>

2. AdaBoost with stumps produces an additive model: a sum of functions of one feature each. why?

<details><summary>answer</summary>

each stump depends on a single feature, so $f(x) = \sum_m\alpha_mh_m(x_{j_m})$ groups into $\sum_jg_j(x_j)$. stumps can't capture interactions; depth-2 trees capture pairwise ones, and so on. the tree depth sets the interaction order of the ensemble.

</details>

3. a random forest's trees are independent; AdaBoost's are not. what does that mean for parallelism, and for the effect of adding a tree?

<details><summary>answer</summary>

forest trees can be grown in parallel and each new tree only reduces variance. boosted trees must be grown one after another, because each depends on the weights left by all previous ones, and each new tree reduces bias, so too many can overfit (slowly, but eventually, with noisy labels).

</details>

4. why does the training error bound need *every* $\varepsilon_m < 1/2$ rather than just the average?

<details><summary>answer</summary>

the bound is a product of factors $2\sqrt{\varepsilon_m(1 - \varepsilon_m)}$, each $\le 1$ with equality at $\varepsilon_m = 1/2$. a round at exactly 1/2 contributes a factor 1, no progress; the guarantee needs a uniform edge $\gamma$ over chance.

</details>

5. why does the training error reach zero, but the exponential loss never does?

<details><summary>answer</summary>

the training error counts signs; once all $y_if(x_i) > 0$ it's 0. the exponential loss $e^{-y_if(x_i)}$ is positive for every finite $f$, so there's always a reason to increase the margins further, which is exactly what keeps AdaBoost improving after zero training error.

</details>

6. AdaBoost's predicted probabilities are often poorly calibrated. why, and how would you fix it?

<details><summary>answer</summary>

$f$ grows with every round (the margins are pushed up), so $\sigma(f)$ drifts toward 0 and 1 even for uncertain points. fix it by calibrating $f$ on held-out data (Platt scaling: fit a logistic regression on $f$), or use a loss with a better-behaved minimizer and shrinkage (gradient boosting with log-loss, topic 04).

</details>

## common bugs

- **labels in $\{0, 1\}$ inside the update.** symptom: errors and margins that make no sense. convert to $\pm1$ before computing $yf$ or $h \ne y$.
- **fitting every round on the original weights.** symptom: the same stump every round. pass the *current* weights as `sample_weight`.
- **forgetting to renormalize.** symptom: weights that shrink to underflow, or a `learning_rate` that seems to do nothing. divide by the sum every round.
- **dividing by a zero error.** symptom: `inf` weights and `nan` predictions. handle $\varepsilon = 0$ before taking the log.
- **`sigmoid(f)` with `np.exp(-f)`.** symptom: overflow warnings and `nan` probabilities after many rounds. use the `tanh` form.
- **yielding one array updated in place.** symptom: every stored stage equals the final one. build a new array each round.

## going deeper

- Freund & Schapire (1997), "A decision-theoretic generalization of on-line learning and an application to boosting", *Journal of Computer and System Sciences* 55(1): 119–139. [doi:10.1006/jcss.1997.1504](https://doi.org/10.1006/jcss.1997.1504)
- Schapire, Freund, Bartlett & Lee (1998), "Boosting the margin: a new explanation for the effectiveness of voting methods", *Annals of Statistics* 26(5): 1651–1686. [doi:10.1214/aos/1024691352](https://doi.org/10.1214/aos/1024691352)
- Friedman, Hastie & Tibshirani (2000), "Additive logistic regression: a statistical view of boosting", *Annals of Statistics* 28(2): 337–407. [doi:10.1214/aos/1016218223](https://doi.org/10.1214/aos/1016218223)
- Zhu, Zou, Rosset & Hastie (2009), "Multi-class AdaBoost", *Statistics and Its Interface* 2(3): 349–360. SAMME. [doi:10.4310/SII.2009.v2.n3.a8](https://doi.org/10.4310/SII.2009.v2.n3.a8)
- Breiman (1999), "Prediction games and arcing algorithms", *Neural Computation* 11(7): 1493–1517. the counterexample to "larger minimum margin is always better". [doi:10.1162/089976699300016106](https://doi.org/10.1162/089976699300016106)
- Viola & Jones (2001), "Rapid object detection using a boosted cascade of simple features", *CVPR*. [doi:10.1109/CVPR.2001.990517](https://doi.org/10.1109/CVPR.2001.990517)
- Schapire & Freund (2012), *Boosting: Foundations and Algorithms*, MIT Press. the whole story, from weak learnability to margins.

## connections

- ← builds on: weighted CART trees (topic 01), bias versus variance and the averaging of topic 02, the logistic function and log loss (ch 07 · topic 03), one-dimensional convex minimization (ch 03, ch 06).
- → used in: gradient boosting, which generalizes "fit the next tree to what's still wrong" to any differentiable loss (topic 04); margins in support vector machines (ch 09 · topic 04) and in generalization theory (ch 13, ch 19); calibration of scores (ch 28).
