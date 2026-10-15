# softmax regression

## why this matters

logistic regression answers yes/no questions. most real classification problems have more than two answers: which digit is this, which of 20 topics is this article about, which of 50,000 words comes next. **softmax regression** (also called multinomial logistic regression) is the natural generalization: one linear score per class, turned into a probability distribution over the classes by the softmax function, trained by the same cross-entropy loss. it's convex, it has the same "prediction minus target" gradient, and it's the exact computation at the top of every neural network classifier and every language model you'll build in this course: the last layer of a transformer *is* a softmax regression on learned features.

it's also a good model in its own right. on MNIST, a softmax regression on raw pixels, with no idea that images have neighbouring pixels, gets around 92% of handwritten digits right, and its weights, plotted as images, look like blurry templates of the digits. you'll build it with both full-batch gradient descent and minibatch SGD, build the "one-vs-rest" alternative to see why a single joint model is better, and measure what batch size does to the speed of learning.

## warm-up: predict first

1. you add 5 to every class's score. what happens to the softmax probabilities?
2. with $k = 2$ classes, softmax regression has two weight vectors. logistic regression has one. are they different models?
3. what's the gradient of the cross-entropy with respect to the scores, if the true class is $c$?
4. one-vs-rest trains $k$ separate binary classifiers. what can go wrong when you put their probabilities together?
5. an epoch of minibatch SGD with batches of 10 and an epoch of full-batch gradient descent both look at every example once. which makes more progress?

## prerequisites check

- [ ] you've built stable `softmax`, `log_softmax` and `one_hot` (→ ch 00 · topics 02–03)
- [ ] you know the categorical distribution and cross-entropy (→ ch 04 · topic 02, ch 05 · topic 05)
- [ ] you've built `SGD`, `Param` and `minibatch_indices`, and know why SGD's gradient is unbiased (→ ch 06 · topic 04)
- [ ] you've built `LogisticRegression` and its gradient $\frac1nA^\top(s - y)$ (→ topic 04)
- [ ] you've built `clone` (→ ch 07 · topic 01)

## core ideas

### idea 1: one score per class, one distribution over classes

with $k$ classes, give each class $c$ its own weights $w_c$ and intercept $b_c$, and score an input by $z_c = w_c^\top x + b_c$. stack the class columns into a matrix $\Theta \in \mathbb R^{p \times k}$ (row 0 the intercepts, the rest the weights) so that, with the design matrix $A = [\mathbf 1, X]$, all the scores are one product: $Z = A\Theta \in \mathbb R^{n \times k}$. the **softmax** turns a row of scores into probabilities:

$$P(y = c \mid x) = \operatorname{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^k e^{z_j}}.$$

every probability is positive and they sum to 1. the largest score gets the largest probability, and the gaps between scores are log-odds: $\log\frac{P(c)}{P(c')} = z_c - z_{c'}$. that's the multiclass version of topic 04's "log-odds are linear". adding the same constant to every score changes nothing (warm-up 1), which you'll exploit for stability and which has consequences for the parameters (idea 2).

**the loss** is the negative log-likelihood of the observed classes, the cross-entropy between the one-hot labels $Y$ and the predicted probabilities $P = \operatorname{softmax}(A\Theta)$ (row by row), plus a ridge penalty on the weights:

$$L(\Theta) = -\frac1n\sum_{i=1}^n\sum_{c=1}^k Y_{ic}\log P_{ic} + \frac12\sum_j\lambda_j\|\Theta_{j,:}\|^2.$$

as in topic 04, $\lambda = (0, \alpha, \dots, \alpha)$ leaves the intercepts unpenalized, so the penalty is $\frac\alpha2\|W\|_F^2$, the squared Frobenius norm of the weight matrix. only the true class's log-probability enters each term, and $\log P_{ic} = z_{ic} - \operatorname{logsumexp}(z_i)$, which your `log_softmax` computes without overflow.

**the gradient**, once more, is prediction minus target. the derivative of $-\log\operatorname{softmax}(z)_c$ with respect to the scores is $\operatorname{softmax}(z) - e_c$ (on paper 1; warm-up 3), so

$$\nabla_\Theta L = \frac1nA^\top(P - Y) + \lambda \odot \Theta,$$

with $\lambda$ broadcast along each row. that's topic 04's formula with matrices in place of vectors: every class's column is pushed up on its own examples and down on everyone else's, in proportion to how much probability it wrongly took.

> **▶ build: stage 1.** implement `softmax_loss(Theta, A, Y, penalty)` and `softmax_gradient(Theta, A, Y, penalty)`, vectorized over examples and classes. use `log_softmax` and `softmax` from `mlfs.utils`.

### idea 2: the estimator, and what it can't tell apart

`SoftmaxRegression(*, alpha=0.0, lr=0.1, max_iters=1000, batch_size=None, tol=1e-06, fit_intercept=True, rng=None)` follows the `Classifier` protocol: `classes_` are the sorted labels, `coef_` is the $(d, k)$ weight matrix (column $c$ for `classes_[c]`), `intercept_` the $(k,)$ intercepts, `decision_function` returns the $(n, k)$ scores and `predict_proba` their softmax. with `batch_size=None`, `fit` minimizes $L$ with your `gradient_descent`, which happily works on a matrix parameter: it only needs `x - lr * g` and a norm.

the loss is convex (it's a log-sum-exp of linear functions plus linear terms and a quadratic), and its curvature is bounded: the Hessian of the cross-entropy with respect to one example's scores is $\operatorname{diag}(p) - pp^\top$, whose eigenvalues are at most $\frac12$. so gradient descent is stable for $\eta < 2/L$ with $L = \frac12\lambda_{\max}(A^\top A/n) + \alpha$. center or standardize your features and $\eta \approx 1$ is safe.

**identifiability.** because adding the same vector $v$ to every class's column of $\Theta$ adds $a_i^\top v$ to every score of example $i$, which softmax ignores, the unpenalized loss can't tell $\Theta$ from $\Theta + v\mathbf 1^\top$: there's a whole line of equivalent solutions (on paper 2). two things settle it in practice. the penalty does: of all the equivalent weight matrices, $\|W\|_F^2$ is smallest when each row sums to zero across classes, so with $\alpha > 0$ the solution is unique. and gradient descent from $\Theta = 0$ does too: each row of the gradient sums to zero across classes (since $\sum_c(P_{ic} - Y_{ic}) = 1 - 1 = 0$), so every iterate keeps rows that sum to zero.

**two classes.** with $k = 2$, the probabilities depend only on the difference of the scores, $z_1 - z_0 = a^\top(\theta_1 - \theta_0)$, so softmax regression with two classes *is* logistic regression with $w = \theta_1 - \theta_0$ (warm-up 2), and at the penalized optimum the two weight columns are $\pm w/2$. the one difference is the penalty: $\theta_0 = -w/2$ and $\theta_1 = w/2$ cost $\frac\alpha2(\|w\|^2/4 + \|w\|^2/4) = \frac{\alpha}{4}\|w\|^2$, so it matches logistic regression with $\alpha/2$ (on paper 3).

> **🔮 predict:** you fit `SoftmaxRegression(alpha=0.2)` on a two-class problem and `LogisticRegression(alpha=0.2)` on the same data. which gives more confident probabilities?

> **▶ build: stage 2.** implement `SoftmaxRegression` with full-batch gradient descent: `fit` (one-hot labels, design matrix, penalty vector, `gradient_descent` from zeros, and `coef_`, `intercept_`, `n_iter_`, `converged_`), `decision_function` and `predict_proba`. for an int `batch_size`, `fit` hands over to `_sgd` (stage 3).

### idea 3: minibatch SGD

every full gradient costs a pass over all $n$ examples. but the gradient is an average over examples, so the average over a random **minibatch** $B$ of size $b$,

$$g_B = \frac1bA_B^\top(P_B - Y_B) + \lambda \odot \Theta,$$

is an unbiased estimate of it (on paper 4), at a cost of $b/n$ of a full pass. **minibatch SGD** takes one step per batch: shuffle the data at the start of each epoch, cut it into batches (your `minibatch_indices`), and step `Theta -= lr * g_B` for each (your `SGD` optimizer, on a `Param` holding $\Theta$). in one epoch it takes $n/b$ steps instead of one (warm-up 5).

why does that help? early in training, the gradient points in roughly the same direction whichever examples you average over: 64 random digits and all 60,000 agree on "make the weights look more like the digits". so a cheap, noisy estimate is nearly as good as the exact one, and you get hundreds of them for the price of one exact step. late in training, the noise matters: with a constant step, SGD doesn't converge but hovers around the minimum, at a distance set by the step size and the batch size. that's fine for prediction (and the subject of chapters 06 and 15 for the rest), but it means "stop when the gradient is small" must use the *full* gradient. here `max_iters` counts epochs, and after each one `_sgd` checks the full gradient norm against `tol`.

> **▶ build: stage 3.** implement `SoftmaxRegression._sgd`: `max_iters` epochs of minibatch SGD with your `SGD`, `Param` and `minibatch_indices`, one generator `rng_ = as_rng(rng)` for all the shuffling, and a full-gradient stopping check after each epoch.

### idea 4: one-vs-rest, the alternative

there's another way to build a multiclass classifier out of binary ones: train $k$ separate classifiers, the $c$-th to separate class $c$ from all the others, and predict the class whose classifier is most confident. it's simple, it works with any binary model (perceptrons, SVMs), and it parallelizes. scikit-learn used it as the default for logistic regression for years.

it has two weaknesses. its "probabilities" aren't a coherent distribution: each comes from a different model, trained on a different question, so they don't sum to 1 (you rescale them, but that doesn't make them calibrated). and each binary problem can be much harder than the multiclass one. if the classes lie along a line, the middle class can't be separated from "everything else" by a single hyperplane, because "everything else" sits on *both* sides of it. its classifier can only learn a weak, nearly flat probability, and the neighbours' classifiers drown it out: the middle class is **masked**. softmax has no such problem, because it never asks a single linear score to do the separating. it compares scores across classes, and the comparison of linear functions can carve out a bump for the middle class (warm-up 4).

> **🔮 predict:** five classes in a row along one axis. one-vs-rest logistic regression and softmax regression are both linear. which classes does one-vs-rest get wrong more often?

> **▶ build: stage 4.** implement `OneVsRest(estimator)`: `fit` trains one clone per class on the labels `y == c`, and `predict_proba` stacks each one's positive-class probability and rescales every row to sum to 1.

## worked example

three points, $x = (-1, 0, 1)$, in classes $0, 1, 2$, with an intercept: $A = \begin{pmatrix}1 & -1\\1 & 0\\1 & 1\end{pmatrix}$, $Y = I_3$, $\Theta \in \mathbb R^{2 \times 3}$, no penalty.

**at $\Theta = 0$.** every score is 0, so every probability is $\tfrac13$ and $L = -\log\tfrac13 = \log 3 \approx 1.0986$.

**the gradient.** $P - Y = \begin{pmatrix}-\tfrac23 & \tfrac13 & \tfrac13\\ \tfrac13 & -\tfrac23 & \tfrac13\\ \tfrac13 & \tfrac13 & -\tfrac23\end{pmatrix}$. the intercept row of $\frac13A^\top(P - Y)$ is $\frac13$ times the column sums, $(0, 0, 0)$. the slope row is $\frac13(-\text{row}_1 + \text{row}_3) = \frac13(1, 0, -1)$. so

$$\nabla L(0) = \begin{pmatrix}0 & 0 & 0\\ \tfrac13 & 0 & -\tfrac13\end{pmatrix}.$$

each row sums to zero (idea 2). the gradient says: lower class 0's slope (its point is on the left), raise class 2's.

**one step** of gradient descent with $\eta = 1$ gives slopes $(-\tfrac13, 0, \tfrac13)$ and intercepts 0. at $x = -1$ the scores become $(\tfrac13, 0, -\tfrac13)$ and the probabilities $\frac{(e^{1/3}, 1, e^{-1/3})}{e^{1/3} + 1 + e^{-1/3}} \approx (0.448, 0.321, 0.230)$: class 0 now leads at its own point.

the tests check the loss and the gradient at 0.

## on paper

1. **the gradient of softmax cross-entropy.** show that $\frac{\partial}{\partial z_j}\left[-\log\operatorname{softmax}(z)_c\right] = \operatorname{softmax}(z)_j - [j = c]$.

<details><summary>solution</summary>

$-\log\operatorname{softmax}(z)_c = -z_c + \log\sum_m e^{z_m}$. the derivative of the first term is $-[j = c]$; of the second, $e^{z_j}/\sum_m e^{z_m} = \operatorname{softmax}(z)_j$. stacking over $j$ gives $p - e_c$; averaging over examples with the chain rule $z_i = \Theta^\top a_i$ gives $\frac1nA^\top(P - Y)$.

</details>

2. **the redundant direction.** show that $L(\Theta + v\mathbf 1^\top) = L(\Theta)$ for any $v \in \mathbb R^p$ when $\alpha = 0$, that each row of $\nabla L$ sums to zero across classes, and that with $\alpha > 0$ the minimizer has weight rows summing to zero.

<details><summary>solution</summary>

$A(\Theta + v\mathbf 1^\top) = A\Theta + (Av)\mathbf 1^\top$ adds $a_i^\top v$ to every score of row $i$, and softmax is invariant to that. the gradient's row sums are $\frac1nA^\top(P - Y)\mathbf 1 = \frac1nA^\top(P\mathbf 1 - Y\mathbf 1) = \frac1nA^\top(\mathbf 1 - \mathbf 1) = 0$ (plus the penalty term, which is zero for $\Theta = 0$ and preserves zero row sums). with $\alpha > 0$, for any weight row $r$, $\|r - \bar r\mathbf 1\|^2 < \|r\|^2$ unless $\bar r = 0$, and the shift doesn't change the data term; so a minimizer must have zero row means.

</details>

3. **two classes.** show that softmax regression with $k = 2$ and penalty $\alpha$ gives the same probabilities as logistic regression with penalty $\alpha/2$.

<details><summary>solution</summary>

the probabilities depend only on $w = \theta_1 - \theta_0$ (weights) and $b = b_1 - b_0$: $P(1) = \sigma(w^\top x + b)$. by the result of on-paper problem 2, the optimal weight columns satisfy $\theta_0 = -\theta_1$, so $\theta_1 = w/2$ and the penalty is $\frac\alpha2(\|w/2\|^2 + \|w/2\|^2) = \frac{\alpha/2}{2}\|w\|^2$. the data terms are identical, so minimizing over $(w, b)$ is logistic regression with $\alpha/2$.

</details>

4. **minibatch gradients are unbiased.** if $B$ is a uniformly random subset of size $b$ of $\{1, \dots, n\}$, show that $E[\frac1b\sum_{i \in B}\nabla\ell_i] = \frac1n\sum_{i=1}^n\nabla\ell_i$.

<details><summary>solution</summary>

$E[\frac1b\sum_{i \in B}\nabla\ell_i] = \frac1b\sum_{i=1}^nP(i \in B)\nabla\ell_i = \frac1b\cdot\frac bn\sum_i\nabla\ell_i$, since each index is in a random $b$-subset with probability $b/n$. (the batches of one shuffled epoch aren't independent of each other, but each one is a uniformly random subset.)

</details>

## experiment

`lab.py` runs three experiments on your code:

1. softmax regression on MNIST pixels: test accuracy with a confidence interval, the most frequent confusions, and the weights drawn as images.
2. softmax versus one-vs-rest logistic regression on five classes in a row.
3. batch size versus progress per epoch, on MNIST.

## check your understanding

1. why is `predict` (argmax of the probabilities) the same as the argmax of `decision_function`?

<details><summary>answer</summary>

softmax is a strictly increasing function of each score relative to the others: the largest score always gets the largest probability. so you never need to compute the softmax just to classify, which matters when $k$ is a 50,000-word vocabulary.

</details>

2. your MNIST model's training accuracy is 93% and test accuracy 92%. is it overfitting? underfitting?

<details><summary>answer</summary>

with a gap of one point, it isn't overfitting much: 7,850 parameters for tens of thousands of images is a small model. it's limited by bias: a single template per digit can't represent the variety of handwriting. more data won't help much; more flexible features (or a network that learns them) will.

</details>

3. a colleague initializes $\Theta$ randomly instead of at zero and trains with $\alpha = 0$. what's different at the end?

<details><summary>answer</summary>

the predictions converge to the same thing (the loss is convex and the minimum value is unique), but the parameters differ: gradient descent never changes the component of $\Theta$ along the redundant direction $v\mathbf 1^\top$, so whatever row means the random start had, they stay. coefficients become hard to interpret; a small $\alpha$ fixes it.

</details>

4. why must `_sgd` check the *full* gradient norm, not the last minibatch's, to decide that it has converged?

<details><summary>answer</summary>

a minibatch gradient is the full gradient plus noise. at the minimum the full gradient is 0, but a batch's gradient isn't (it's the gradient of a different, random objective), so it would never fall below a small `tol`. conversely, a lucky batch could have a tiny gradient far from the minimum.

</details>

5. when would you still choose one-vs-rest?

<details><summary>answer</summary>

when the binary model has no natural multiclass version (an SVM, chapter 09), when classes aren't mutually exclusive (multi-label problems: an article can be about sport *and* politics, so you want $k$ independent yes/no answers, not one distribution), or when you need to add a class later without retraining the others.

</details>

6. how does softmax regression relate to the last layer of a neural network classifier?

<details><summary>answer</summary>

it's identical: a linear map from the last hidden representation to $k$ scores, a softmax, and the cross-entropy loss, with the same $P - Y$ gradient flowing back. the network's job is to learn features in which the classes are linearly separable enough for this layer. chapter 25's linear probes run your `SoftmaxRegression` on frozen learned features for exactly that reason.

</details>

## common bugs

- **softmax over the wrong axis.** normalizing over examples (axis 0) instead of classes. symptom: probabilities that sum to 1 down each column, and nonsense predictions; the gradient check may still pass if the loss uses the same wrong axis, so check that rows sum to 1.
- **labels as integers in the loss.** using `y` where the one-hot `Y` belongs. symptom: shape errors, or a broadcast that silently multiplies by the label value.
- **penalizing the intercept row.** symptom: probabilities pulled towards uniform as $\alpha$ grows, and a mismatch with scikit-learn that grows with $\alpha$.
- **a new generator every epoch.** `as_rng(self.rng)` inside the epoch loop replays the same shuffle each epoch when `rng` is an int. symptom: the plain-SGD check fails after the first epoch.
- **a step size above $2/L$ on raw pixels.** uncentered pixel features have a huge top eigenvalue (the mean image). symptom: the loss oscillates or explodes with full-batch gradient descent, while small batches seem fine for a while.
- **one-vs-rest that fits the original estimator.** forgetting `clone`, so all $k$ "models" are the same object, last fitted on class $k - 1$. symptom: every row predicts the same class.

## going deeper

- Bishop (2006), *Pattern Recognition and Machine Learning*, Springer, section 4.3.4: multiclass logistic regression, its gradient and Hessian. [microsoft.com/en-us/research/people/cmbishop/prml-book](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/)
- LeCun, Bottou, Bengio & Haffner (1998), "Gradient-based learning applied to document recognition", *Proceedings of the IEEE* 86(11): 2278–2324. MNIST and a table of linear and nonlinear baselines on it. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
- Rifkin & Klautau (2004), "In defense of one-vs-all classification", *JMLR* 5: 101–141. when one-vs-rest is as good as anything else (with strong, well-tuned binary classifiers). [jmlr.org/papers/v5/rifkin04a](https://www.jmlr.org/papers/v5/rifkin04a.html)
- Bottou, Curtis & Nocedal (2018), "Optimization methods for large-scale machine learning", *SIAM Review* 60(2): 223–311. why SGD wins on large datasets. [doi:10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, MIT Press, sections 6.2.2.3 (softmax units) and 8.1.3 (minibatch algorithms). [deeplearningbook.org](https://www.deeplearningbook.org/)

## connections

- ← builds on: stable softmax and one-hot encoding (ch 00), the categorical distribution and cross-entropy (ch 04–05), SGD and minibatches (ch 06), `clone` and the classifier protocol (ch 07), logistic regression (topic 04).
- → used in: the capstone (a multiclass text classifier is one `batch_size` away), the softmax output layer of every network (ch 14–15), CNN classifiers on MNIST and CIFAR (ch 16), language models' next-token softmax (ch 17–18), knowledge distillation and calibration with temperatures (ch 15, ch 28), and `ssl.linear_probe` (ch 25), which trains this exact class on frozen representations.
