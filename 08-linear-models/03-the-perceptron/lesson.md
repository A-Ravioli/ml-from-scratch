# the perceptron

## why this matters

in 1958 Frank Rosenblatt described a machine that learned to classify by correcting its own mistakes, and the press reported it as the embryo of a computer that would walk, talk and be conscious. the machine was a linear classifier with a three-line learning rule. the claims were absurd, but the rule was real, and it's still worth studying: it's the simplest learning algorithm there is, it comes with one of the first theorems of machine learning (if the classes can be separated by a line with some room to spare, it will find one, after a number of mistakes you can bound in advance), and its failure modes are instructive. on data no line can separate, it never stops, and its final answer depends on when you pull the plug. on XOR, it can't succeed at all, which (after Minsky and Papert's 1969 book) helped stall neural network research for over a decade.

you'll build the perceptron and its averaged version as a `Classifier` in `mlfs.linear`, compute the margin and the mistake bound from the convergence theorem, and in the lab measure how tight that bound is, watch the plain perceptron thrash on noisy labels while the averaged one settles, and fix XOR with one extra feature. the update rule you write here (nudge the weights towards the example you got wrong) reappears in stochastic gradient descent, in the SVM's hinge loss (chapter 09), and in every neural network's training loop.

## warm-up: predict first

1. a linear classifier predicts $\operatorname{sign}(w^\top x + b)$. you double $w$ and $b$. what changes?
2. the perceptron gets an example wrong and adds $yx$ to $w$. after the update, is that example necessarily classified correctly?
3. two classes are separable with a wide gap in one dataset and a narrow gap in another, same number of points. which one costs the perceptron more mistakes?
4. on data that no line separates, what does the perceptron's training loop do?
5. can a single linear classifier compute XOR of two binary inputs?

## prerequisites check

- [ ] you know dot products, norms and the Cauchy–Schwarz inequality, and what a hyperplane is (→ ch 01, ch 02 · topic 01)
- [ ] you've built `Classifier`, `check_X_y` and `as_rng` (→ ch 07 · topic 01, ch 00 · topic 05)
- [ ] you know stochastic gradient descent: one example (or batch) per step (→ ch 06 · topic 04)
- [ ] you've seen decision boundaries and how to plot them (→ ch 00 · topic 04)

## core ideas

### idea 1: a mistake-driven update

a linear classifier splits space with a hyperplane: predict class $+1$ if $w^\top x + b > 0$ and $-1$ otherwise. the score $w^\top x + b$ is positive on one side of the boundary $\{x : w^\top x + b = 0\}$, negative on the other, and $w$ points towards the positive side. with labels coded as $y \in \{-1, +1\}$, an example is classified correctly exactly when $y(w^\top x + b) > 0$: the score has the right sign. this product, the **functional margin**, is the quantity everything in this topic revolves around.

the perceptron looks at one example at a time and changes the weights only when it's wrong:

$$\text{if } y_i(w^\top x_i + b) \le 0: \qquad w \leftarrow w + y_i x_i, \qquad b \leftarrow b + y_i.$$

why does that help? after the update, the functional margin on the same example becomes

$$y_i\big((w + y_ix_i)^\top x_i + b + y_i\big) = y_i(w^\top x_i + b) + \|x_i\|^2 + 1,$$

so it grows by $\|x_i\|^2 + 1 > 0$. the update pushes the boundary towards getting $x_i$ right, though not necessarily all the way (warm-up 2), and it may break other examples. geometrically, for a misclassified positive example, adding $x_i$ rotates $w$ towards $x_i$; for a negative one, away.

"$\le 0$" rather than "$< 0$" matters: starting from $w = 0$, every score is 0, so with a strict inequality the perceptron would never make its first update.

> **🔮 predict:** with $w = (1, 1)$, $b = 1$, what does the perceptron do with the example $x = (-1, 0)$, $y = -1$?

> **▶ build: stage 1.** implement `perceptron_step(w, b, x, y, *, fit_intercept=True)`, returning the new `(w, b, mistake)`. return a new array rather than modifying `w` in place.

### idea 2: epochs, order, and stopping

training repeats the step over the data. one pass over all $n$ examples is an **epoch**. the perceptron's behaviour depends on the order of the examples, so the usual practice is to shuffle them at the start of every epoch, with a generator you control (`rng_ = as_rng(rng)`, created once per `fit`, so the same seed gives the same run). stop when a whole epoch passes without a mistake: then every training example is classified correctly and no further update will ever happen. on data that can't be separated that never occurs, so `max_epochs` is the safety net (warm-up 4).

the estimator wraps this up for any two labels. `classes_ = np.unique(y)` must have exactly two entries; `classes_[1]` becomes $+1$ and `classes_[0]` becomes $-1$. `decision_function(X)` returns the raw score $Xw + b$, and `predict` maps positive scores to `classes_[1]`. the perceptron has no notion of probability, so `predict` can't be chapter 07's argmax of `predict_proba`: you override it, and `predict_proba` keeps raising. record the mistakes per epoch in `mistakes_`, the number of epochs in `n_iter_` and the total number of updates in `n_updates_`: you'll need them to see the theory at work.

two facts that are easy to check and worth knowing. scaling $w$ and $b$ by any positive constant changes no prediction (warm-up 1). and so a "learning rate" $\eta$ in $w \leftarrow w + \eta yx$ would change nothing either: from $w = 0$ every iterate is just scaled by $\eta$. the perceptron is gradient descent with step 1 on the loss $\max(0, -y(w^\top x + b))$ (on paper 4), one of the few cases where the step size truly doesn't matter.

> **▶ build: stage 2.** implement `Perceptron(*, max_epochs=100, averaged=False, fit_intercept=True, shuffle=True, rng=None)` with `averaged=False`: `fit` with the stopping rule and the recorded counts, `decision_function` and `predict`.

### idea 3: the averaged perceptron

if the data aren't separable, every epoch contains mistakes, and every mistake moves the weights by a whole data point. the final $w$ is wherever the last few mistakes happened to leave it: run one more epoch and you get a noticeably different classifier. the lab shows the test accuracy of the final weights jumping by 20 points or more from one epoch to the next.

the fix is to **average**. instead of the last iterate, return the mean of all of them:

$$\bar w = \frac1T\sum_{t=1}^T w_t, \qquad \bar b = \frac1T\sum_{t=1}^T b_t,$$

where $w_t$ is the weight vector after the $t$-th visit (mistake or not), over all $T$ visits of all epochs. iterates that survived many visits without a mistake get a big say; ones that were immediately corrected get little. averaging a noisy sequence that bounces around a good region gives a point near the middle of it, which is much more stable and usually generalizes better. it's the practical version of Freund and Schapire's *voted perceptron*, and the same idea as Polyak averaging of SGD iterates in chapter 06. it changes only what `fit` returns, not the training run.

**doing it cheaply.** the obvious implementation keeps a running sum and adds $w$ after every visit: $O(d)$ per visit, fine for dense data. when the examples are sparse (a text document touches a handful of the vocabulary's columns), that's wasteful, since $w$ only changes on mistakes. writing $\Delta_s$ for the update made at visit $s$ (zero if no mistake), $w_t = \sum_{s \le t}\Delta_s$, and

$$\sum_{t=1}^T w_t = \sum_s (T - s + 1)\,\Delta_s \quad\Longrightarrow\quad \bar w = w_T - \frac1T\sum_s (s - 1)\,\Delta_s.$$

so keep a second vector $u$, add $(s - 1)\Delta_s$ to it on each mistake, and finish with $\bar w = w_T - u/T$ (on paper 3). all the work happens on mistakes.

> **▶ build: stage 3.** add `averaged=True` to `Perceptron.fit`: `coef_` and `intercept_` become the average over every visit of every epoch.

### idea 4: the convergence theorem

suppose the data are **linearly separable with margin $\gamma$**: there's a unit vector $w^\star$ ($\|w^\star\| = 1$) with $y_i\,{w^\star}^\top x_i \ge \gamma > 0$ for every $i$, and every $\|x_i\| \le R$. (ignore the intercept for the moment.) then, starting from $w = 0$, the perceptron makes at most

$$\left(\frac{R}{\gamma}\right)^2$$

updates, in any order of the examples, however many examples there are (Novikoff, 1962). the proof is two inequalities (on paper 1). each update increases $w^\top w^\star$ by at least $\gamma$, so after $k$ updates $w^\top w^\star \ge k\gamma$: $w$ moves steadily towards $w^\star$. each update increases $\|w\|^2$ by at most $R^2$ (the cross term is $\le 0$ because it was a mistake), so $\|w\| \le \sqrt k\,R$: $w$ grows slowly. since $w^\top w^\star \le \|w\|$, $k\gamma \le \sqrt kR$, and $k \le (R/\gamma)^2$.

the quantity $\gamma/R$ is the **geometric margin** relative to the data's size: the distance from the hyperplane to the closest point, $\min_i y_i(w^\top x_i + b)/\|w\|$, measured in units of the data radius. a wide gap means few mistakes (warm-up 3), and nothing in the bound depends on the dimension or the number of points. that's a first glimpse of an idea chapter 13 develops: margins, not parameter counts, can control how hard learning is.

**the intercept** is handled by augmentation: the perceptron with the update $b \leftarrow b + y$ is exactly the perceptron without an intercept on the points $\tilde x_i = (x_i, 1)$ with weights $\tilde w = (w, b)$ (on paper 2). so the bound applies with $R = \max_i\|(x_i, 1)\|$ and $\gamma$ computed for $(w, b)$ as a single vector. note what that "1" does: if the data live in a disc of radius 0.1, the augmented radius is dominated by the constant, and learning the intercept becomes the slow part. center your data.

the theorem says nothing good about non-separable data (no $w^\star$, no bound), and that's where idea 3 comes in. and some problems aren't separable at all: no line puts $(0, 0), (1, 1)$ on one side and $(0, 1), (1, 0)$ on the other (XOR, warm-up 5). add the feature $x_1x_2$, though, and they are. choosing, or learning, the right features is the subject of the rest of the course.

> **🔮 predict:** you halve the margin of a separable dataset, keeping $R$ the same. the bound quadruples. does the number of mistakes the perceptron actually makes quadruple too?

> **▶ build: stage 4.** implement `margin(X, y, w, b)` (the geometric margin, negative if a point is misclassified) and `mistake_bound(X, y, w, b)` (Novikoff's $(R/\gamma)^2$ with augmented points, or `np.inf` if $(w, b)$ doesn't separate the data).

## worked example

three points: $x_1 = (1, 1)$ with $y_1 = +1$; $x_2 = (-1, 0)$ and $x_3 = (0, -1)$ with $y = -1$. no shuffling: visit them in order. start from $w = (0, 0)$, $b = 0$.

| visit | example | score $w^\top x + b$ | $y \cdot$ score | mistake? | $w$ after | $b$ after |
|---|---|---|---|---|---|---|
| 1 | $x_1$, $+1$ | 0 | 0 | yes | $(1, 1)$ | 1 |
| 2 | $x_2$, $-1$ | $-1 + 1 = 0$ | 0 | yes | $(2, 1)$ | 0 |
| 3 | $x_3$, $-1$ | $-1$ | 1 | no | $(2, 1)$ | 0 |
| 4 | $x_1$, $+1$ | 3 | 3 | no | $(2, 1)$ | 0 |
| 5 | $x_2$, $-1$ | $-2$ | 2 | no | $(2, 1)$ | 0 |
| 6 | $x_3$, $-1$ | $-1$ | 1 | no | $(2, 1)$ | 0 |

epoch 1 (visits 1–3) has two mistakes, epoch 2 (visits 4–6) none, so training stops: `mistakes_ = [2, 0]`, `n_iter_ = 2`, `n_updates_ = 2`, $w = (2, 1)$, $b = 0$.

**averaged:** $(w, b)$ was $((1, 1), 1)$ after visit 1 and $((2, 1), 0)$ after each of the other five, so $\bar w = \frac{(1, 1) + 5(2, 1)}{6} = (11/6, 1)$ and $\bar b = 1/6$. the trick agrees: the updates were $\Delta_1 = ((1, 1), 1)$ at $s = 1$ and $\Delta_2 = ((1, 0), -1)$ at $s = 2$, so $u = 0\cdot\Delta_1 + 1\cdot\Delta_2 = ((1, 0), -1)$ and $\bar w = (2, 1) - (1, 0)/6 = (11/6, 1)$, $\bar b = 0 + 1/6$.

**margin and bound** for the final $(w, b) = ((2, 1), 0)$: the functional margins are $3, 2, 1$, so the geometric margin is $1/\|w\| = 1/\sqrt5 \approx 0.447$. augmented, the squared norms are $\|(1, 1, 1)\|^2 = 3$ and $\|(-1, 0, 1)\|^2 = \|(0, -1, 1)\|^2 = 2$, so $R^2 = 3$; with $\tilde w = (2, 1, 0)$, $\gamma = 1/\sqrt5$ and the bound is $3 \cdot 5 = 15$. the perceptron made 2 updates.

the tests check the trace, the counts and the averaged weights.

## on paper

1. **Novikoff's theorem.** prove that if $\|x_i\| \le R$ and $y_i{w^\star}^\top x_i \ge \gamma$ for a unit vector $w^\star$, the perceptron without intercept, started at $w = 0$, makes at most $(R/\gamma)^2$ updates.

<details><summary>solution</summary>

let $w_k$ be the weights after $k$ updates, the $k$-th on example $(x, y)$. then $w_k^\top w^\star = w_{k-1}^\top w^\star + y\,x^\top w^\star \ge w_{k-1}^\top w^\star + \gamma$, so by induction $w_k^\top w^\star \ge k\gamma$. and $\|w_k\|^2 = \|w_{k-1}\|^2 + 2y\,w_{k-1}^\top x + \|x\|^2 \le \|w_{k-1}\|^2 + R^2$, because the update happened on a mistake ($y\,w_{k-1}^\top x \le 0$); so $\|w_k\|^2 \le kR^2$. by Cauchy–Schwarz, $k\gamma \le w_k^\top w^\star \le \|w_k\|\|w^\star\| \le \sqrt kR$, hence $k \le R^2/\gamma^2$. the order of the examples never entered the argument.

</details>

2. **the intercept is an extra feature.** show that running the perceptron with $b \leftarrow b + y$ on points $x_i$ gives the same predictions, step by step, as running it without an intercept on $\tilde x_i = (x_i, 1)$.

<details><summary>solution</summary>

with $\tilde w = (w, b)$, the score is $\tilde w^\top\tilde x = w^\top x + b$, so mistakes happen at the same visits, and the update $\tilde w \leftarrow \tilde w + y\tilde x$ is $w \leftarrow w + yx$ together with $b \leftarrow b + y$. by induction, the two runs have identical $(w, b)$ after every visit.

</details>

3. **the averaging trick.** with $w_t = \sum_{s \le t}\Delta_s$ (where $\Delta_s$ is the update at visit $s$, possibly zero), show that $\frac1T\sum_{t=1}^T w_t = w_T - \frac1T\sum_{s=1}^T (s - 1)\Delta_s$.

<details><summary>solution</summary>

$\sum_{t=1}^T w_t = \sum_{t=1}^T\sum_{s \le t}\Delta_s = \sum_{s=1}^T (T - s + 1)\Delta_s$, since $\Delta_s$ appears in $w_s, \dots, w_T$. dividing by $T$: $\sum_s\Delta_s - \frac1T\sum_s(s - 1)\Delta_s = w_T - \frac1T\sum_s (s - 1)\Delta_s$.

</details>

4. **the perceptron is SGD.** show that the perceptron update is one step of stochastic (sub)gradient descent with step size 1 on the loss $\ell(w, b; x, y) = \max(0, -y(w^\top x + b))$. why does the step size not matter when you start from zero?

<details><summary>solution</summary>

if $y(w^\top x + b) < 0$ the loss is $-y(w^\top x + b)$, with gradient $(-yx, -y)$; a step of size 1 against it adds $(yx, y)$. if $y(w^\top x + b) > 0$ the loss is 0 and its gradient is 0. at exactly 0 the loss has a kink, and the perceptron chooses the subgradient $(-yx, -y)$. with a step size $\eta$, every iterate started from 0 is exactly $\eta$ times the step-1 iterate (by induction: the mistake test only depends on the sign of the score, which scaling preserves), so the predictions are identical.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. separable data with a shrinking gap: the number of updates against Novikoff's bound.
2. noisy labels: test accuracy of the final and the averaged weights, epoch by epoch.
3. XOR with two features, and with a product feature added.

## check your understanding

1. the perceptron has stopped with zero training mistakes. is its boundary the "best" separating line?

<details><summary>answer</summary>

no. it's just *some* line that separates the training data, often one that passes close to a few points (the last mistakes pushed it just far enough). a line with the largest margin would be more robust to new points: that's what the support vector machine (chapter 09) finds. the averaged perceptron tends to land closer to the middle of the gap.

</details>

2. you run the perceptron twice on separable data with different shuffling seeds. do you get the same weights? the same number of updates?

<details><summary>answer</summary>

generally neither. the order decides which mistakes happen and when, so both the final separator and the count change. what doesn't change is the guarantee: every order stops after at most $(R/\gamma)^2$ updates.

</details>

3. you multiply every feature by 1,000. what happens to the perceptron's predictions, to its number of updates, and to the bound?

<details><summary>answer</summary>

without an intercept, nothing changes: $R$ and $\gamma$ both scale by 1,000, so the bound is unchanged, and every iterate is scaled by 1,000 with the same signs. with an intercept, the augmented constant 1 becomes negligible next to the features, so the bias moves in tiny steps relative to $w$: when the best separator needs an offset, many more updates may be needed. (the bound changes accordingly, through $\gamma$ of the augmented vector.)

</details>

4. why is "zero mistakes in an epoch" a safe stopping rule, but "zero mistakes in the last 10 visits" isn't?

<details><summary>answer</summary>

after a clean epoch, every training example has been checked against the current weights, which haven't changed since, so none of them can trigger an update. after 10 clean visits, the other $n - 10$ examples could still be misclassified.

</details>

5. the averaged perceptron's weights may misclassify a few training points even on separable data, where the plain perceptron ends with zero errors. why might you prefer it anyway?

<details><summary>answer</summary>

on separable data, the averaged weights are dominated by the late iterates (which survived many visits), so they usually classify the training data correctly too; when they don't, it's by a hair. on new data, the average is typically more central and generalizes better, and on noisy data it is far more stable. training error isn't the goal.

</details>

6. what's the relationship between the perceptron and logistic regression (topic 04)?

<details><summary>answer</summary>

both are linear classifiers trained by gradient steps on a loss of the functional margin $m = y(w^\top x + b)$: the perceptron uses $\max(0, -m)$, logistic regression uses $\log(1 + e^{-m})$, a smooth version that keeps pushing even on correctly classified points and gives probabilities. the SVM's hinge loss $\max(0, 1 - m)$ sits in between.

</details>

## common bugs

- **a strict inequality.** testing `y * score < 0`. symptom: from $w = 0$ nothing ever updates; the classifier predicts one class for everything and "converges" in one epoch.
- **updating `w` in place.** `w += y * x` inside `perceptron_step` changes the caller's array. symptom: the step check fails; averaged weights come out equal to the final ones.
- **labels left as 0/1.** using $y = 0$ for the negative class makes negative examples' updates do nothing. symptom: the boundary only ever moves towards positive points; training never converges on easy separable data.
- **a new random generator every epoch.** `as_rng(self.rng)` inside the epoch loop with an integer seed replays the same permutation each epoch. symptom: the mistakes per epoch don't match the check with `shuffle=True`.
- **averaging only over mistakes, or only over epochs.** the average is over every visit. symptom: the worked example gives something other than $(11/6, 1)$.
- **using `Classifier.predict`.** it calls `predict_proba`, which the perceptron doesn't have. symptom: `NotImplementedError` from `predict`.

## going deeper

- Rosenblatt (1958), "The perceptron: a probabilistic model for information storage and organization in the brain", *Psychological Review* 65(6): 386–408. [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519)
- Novikoff (1962), "On convergence proofs on perceptrons", *Proceedings of the Symposium on the Mathematical Theory of Automata* 12: 615–622, Polytechnic Institute of Brooklyn. the mistake bound.
- Minsky & Papert (1969), *Perceptrons: An Introduction to Computational Geometry*, MIT Press (expanded edition 1988). what single-layer perceptrons can't compute, XOR included.
- Freund & Schapire (1999), "Large margin classification using the perceptron algorithm", *Machine Learning* 37(3): 277–296. the voted and averaged perceptron, with generalization bounds. [doi:10.1023/A:1007662407062](https://doi.org/10.1023/A:1007662407062)
- Collins (2002), "Discriminative training methods for hidden Markov models: theory and experiments with perceptron algorithms", *EMNLP 2002*. the averaged perceptron in practice, with the lazy-update trick. [aclanthology.org/W02-1001](https://aclanthology.org/W02-1001/)
- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning*, Cambridge University Press, section 9.1.2: the perceptron and its bound. [cs.huji.ac.il/~shais/UnderstandingMachineLearning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

## connections

- ← builds on: hyperplanes and Cauchy–Schwarz (ch 01–02), SGD (ch 06), the `Classifier` protocol and seeded randomness (ch 07, ch 00).
- → used in: logistic regression's smooth version of the same update (topic 04), the capstone (a perceptron is a fine baseline for text), SVMs and the max-margin separator, and the kernel perceptron (ch 09), margin-based generalization bounds (ch 13), and the multilayer perceptron, whose neurons are this model with a smooth activation (ch 14).
