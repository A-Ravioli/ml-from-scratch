# generalization in deep learning

## why this matters

chapter 13 taught you the classical answer to "why does a model generalize?": because the class of functions it was chosen from is small enough for the training data to pin it down. a hypothesis class with VC dimension $d$, or Rademacher complexity $\mathfrak R_n$, can only overfit so much, and the bounds say how much. then chapter 07's bias–variance picture added the U-shaped curve: test error first falls as the model gets more flexible, then rises again when it starts fitting noise, so you stop at the bottom of the U.

deep networks break both stories. in 2017 Zhang, Bengio, Hardt, Recht and Vinyals took standard image classifiers and trained them on CIFAR-10 with the labels replaced by random ones. the networks reached 100% training accuracy anyway, with the same architecture, optimizer and hyperparameters that generalize well on the real labels. a class that can fit *any* labeling of the training set has the largest possible capacity, and every capacity-based bound for it is vacuous. and when you make networks bigger and bigger, test error typically keeps going *down*, long after the models fit the training data perfectly; plotted against size, the curve can even rise to a peak at the point where the model first interpolates the data and then descend again, which Belkin, Hsu, Ma and Mandal (2019) named **double descent**.

this topic builds the smallest model in which all of this happens exactly and can be computed rather than estimated: a two-layer network with a frozen random first layer, whose second layer is fitted by least squares. you'll corrupt labels, sweep the number of features through the interpolation threshold, and see that gradient descent, among infinitely many perfect fits, picks a particular one. the capstone then reproduces double descent with fully trained networks, where none of this is exact any more.

## warm-up: predict first

1. a network with 370,000 parameters is trained on 1,000 images with completely random labels. can it reach 100% training accuracy? what test accuracy do you expect?
2. a linear model with $p$ parameters is fitted by least squares to $n = 1{,}000$ points. as $p$ goes from 100 to 8,000, sketch the test error you expect. where is it worst?
3. with $p > n$, the equations $Xw = y$ have infinitely many solutions. you run gradient descent from $w = 0$. which one do you end up at?
4. adding a ridge penalty $\lambda\lVert w\rVert^2$ to an interpolating model: does it help or hurt, and does the answer depend on $p$?
5. with 20% of the training labels wrong, does a bigger model generalize better or worse than a smaller one?

## prerequisites check

- [ ] you know VC dimension, Rademacher complexity and why bounds based on them grow with the class's capacity (→ ch 13 · topics 02–04)
- [ ] you know the bias–variance decomposition and the classical U-curve (→ ch 07 · topic 03)
- [ ] you can solve least squares with the normal equations and know what the pseudoinverse is (→ ch 02 · topic 04, ch 08 · topic 01)
- [ ] you know ridge regression and its closed form (→ ch 08 · topic 02)
- [ ] you know random Fourier features approximate a kernel (→ ch 09 · topic 05) and what the NTK of a two-layer network is (→ topic 03)

## core ideas

### idea 1: memorization, and the end of capacity arguments

the random-label experiment has a precise meaning in chapter 13's language. the empirical Rademacher complexity of a class $\mathcal H$ on a sample measures how well the class can correlate with random signs: $\hat{\mathfrak R}_n(\mathcal H) = E_\sigma \sup_{h \in \mathcal H} \frac1n\sum_i \sigma_i h(x_i)$. a network trained to fit random labels is a demonstration that it can match (almost) any sign pattern on the training set, so $\hat{\mathfrak R}_n$ is at its maximum, and the uniform-convergence bound $\text{test error} \le \text{train error} + 2\hat{\mathfrak R}_n + \dots$ allows any test error at all. the same for VC dimension: plug a parameter count of 370,000 into chapter 13's `vc_bound` with $n = 1{,}000$ and the bound on the test error is above 3, for an error rate that can't exceed 1.

the conclusion isn't that networks don't generalize (they do, on real labels) but that **whatever explains it has to depend on the data and on the training algorithm**, not only on the class. the same network, trained the same way, ends at a solution that generalizes for real labels and at one that doesn't for random labels; so the explanation is about *which* solution training finds. three facts from the random-label experiment point the way: random labels take longer to fit than real ones (there is no shared structure to exploit, so every example is memorized individually); with partially corrupted labels, test accuracy degrades gracefully with the noise level, so the network fits the clean examples by learning and the corrupted ones by memorizing; and explicit regularizers (weight decay, dropout, data augmentation) help a little but aren't what makes the difference.

to run the experiment you need a precise corruption. here: choose exactly $\text{round}(fn)$ examples at random without replacement and give each a label drawn uniformly from the *other* classes, so that exactly that fraction of labels is wrong. (Zhang et al. resampled labels uniformly from all classes, so about a tenth of the "corrupted" labels stayed correct by chance; the exact version makes the fraction of wrong labels the knob.)

> **spec.** `corrupt_labels(y, fraction, n_classes, *, rng=None)` returns a new int array like `y` in which exactly `round(fraction * len(y))` positions, chosen uniformly without replacement with `np.random.default_rng(rng)`, hold a class different from the original, drawn uniformly from the other `n_classes - 1`; every other position is unchanged, and `y` isn't modified.

> **🔮 predict:** an MLP with 369 parameters per training image is trained with Adam on 1,000 MNIST digits whose labels are all random. how many epochs do you expect it to need to reach 100% training accuracy, compared with the true labels?

> **▶ build: stage 1.** implement `corrupt_labels`.

### idea 2: a network whose first layer you never train

to study generalization exactly you want a model that is a real two-layer network and still has a closed-form fit. freeze the first layer at random:

$$\phi(x) = \frac{1}{\sqrt p}\,\text{relu}(W^\top x), \qquad W \in \mathbb R^{d \times p},\ W_{ij} \sim \mathcal N(0, 1/d), \qquad f(x) = \phi(x)^\top \beta.$$

only $\beta$ is trained. that is a **random-features model** (Rahimi & Recht, 2007; you built the cosine version in chapter 09). three reasons it's the right laboratory. it's a linear model in $\beta$, so least squares solves it exactly and every quantity in a double-descent curve can be computed in seconds. its number of parameters $p$ is a knob you can turn continuously through $n$. and it is, up to scaling, the lazy-regime network of topic 03 with only the output layer trained: as $p \to \infty$, $\phi(x)^\top\phi(x')$ converges to the output-layer half of the ReLU NTK, so random features are kernel regression's finite-width cousin.

the scaling choices matter for the numbers, not the phenomena: $1/d$ variance keeps $W^\top x$ of order one for inputs of norm about 1, and the $1/\sqrt p$ keeps $\phi(x)^\top\phi(x')$ of order one at every width (so a fixed ridge $\lambda$ means the same thing at every $p$).

> **spec.** `RandomReLUFeatures(n_features, *, rng=None)`: `fit(X)` draws `W_` `(d, n_features)` with entries $\mathcal N(0, 1/d)$ from `np.random.default_rng(self.rng)` and returns `self`; `transform(X)` returns `relu(X @ W_) / sqrt(n_features)`, shape `(n, n_features)`.

> **▶ build: stage 2.** implement `RandomReLUFeatures`.

### idea 3: fitting on both sides of the threshold

with $\Phi \in \mathbb R^{n \times p}$ the features of the training set, least squares minimizes $\lVert\Phi\beta - Y\rVert^2$. what the solution looks like depends on which is bigger.

- **$p < n$ (underparameterized).** generically $\Phi$ has full column rank, the solution is unique, $\beta = (\Phi^\top\Phi)^{-1}\Phi^\top Y$, and it doesn't fit the data exactly.
- **$p > n$ (overparameterized).** generically $\Phi\beta = Y$ has infinitely many solutions: any solution plus any vector in the $(p - n)$-dimensional null space of $\Phi$. the natural choice is the one with the **smallest norm**, $\beta = \Phi^\top(\Phi\Phi^\top)^{-1}Y$, which lies in the row space of $\Phi$.
- **both at once.** the pseudoinverse $\Phi^+$ gives the minimum-norm least-squares solution in every case, including rank-deficient ones (duplicated features, $p = n$ with a singular matrix), where the normal equations break down. `np.linalg.pinv` and `np.linalg.lstsq` compute it from the SVD.

with a ridge penalty, $\min \lVert\Phi\beta - Y\rVert^2 + \lambda\lVert\beta\rVert^2$, the solution is $(\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top Y = \Phi^\top(\Phi\Phi^\top + \lambda I)^{-1}Y$; use the $p \times p$ form when $p \le n$ and the $n \times n$ one when $p > n$. as $\lambda \to 0$ both tend to $\Phi^+ Y$.

> **spec.** `min_norm_least_squares(Phi, Y, *, ridge=0.0)` returns $\beta$ `(p,)` or `(p, k)` minimizing $\lVert\Phi\beta - Y\rVert^2 + \text{ridge}\,\lVert\beta\rVert^2$; for `ridge = 0`, the minimum-norm least-squares solution $\Phi^+ Y$, valid for any shape and rank.

> **▶ build: stage 3.** implement `min_norm_least_squares`.

### idea 4: double descent

sweep $p$ and measure the test error of the ridgeless fit. three regimes appear (Belkin et al., 2019; Advani & Saxe, 2017, saw the peak earlier in linear networks):

1. **$p \ll n$:** the classical regime. more features reduce bias, the fit is stable, test error falls and then starts to rise as $p$ approaches $n$: the left half of the U.
2. **$p \approx n$: the interpolation threshold.** there is now exactly one (or almost exactly one) way to fit all $n$ points, and it is terrible: $\Phi$ is nearly square, its smallest singular value is nearly zero, and the solution has enormous coefficients that amplify every bit of label noise and every quirk of the features. test error spikes. the spike has nothing to do with the classical "too many parameters" story: it comes from the fit having to pass *exactly* through the data with no freedom left (on paper 2).
3. **$p \gg n$:** infinitely many interpolants, and the minimum-norm one gets smoother as $p$ grows (more features to spread the fit over, each contributing a little). the test error descends again, often below the best underparameterized one. in the limit $p \to \infty$ this is ridgeless kernel regression with the random-features kernel.

for linear and random-features models with Gaussian-like data, Hastie, Montanari, Rosset and Tibshirani (2022) and Mei & Montanari (2022) derived these curves exactly as $n, p \to \infty$ with $p/n$ fixed. three practical lessons come with them. **regularization kills the peak:** a well-tuned ridge makes the test error decrease monotonically in $p$ (Nakkiran et al., 2021, "optimal regularization can mitigate double descent"), because the spike is variance from fitting noise exactly. **label noise makes it worse:** the peak is the amplified noise, so noisy labels raise and widen it. and **the threshold is about samples too:** at fixed $p$, adding training data moves $n$ towards $p$, so near the threshold more data can make the test error worse (Nakkiran et al., 2020, "deep double descent", saw this in trained transformers too).

trained deep networks show the same shape in model size, in training time, and in sample size, most clearly with label noise (Nakkiran et al., 2020), though the threshold isn't simply $p = n$: it sits where the model can just barely fit the training set (the capstone measures it).

> **spec.** `double_descent_curve(X_train, Y_train, X_test, y_test, widths, *, ridge=0.0, n_seeds=3)`: `Y_train` one-hot `(n, k)`, `y_test` int labels. for each width $p$ and seed $s \in \{0, \dots, n_{\text{seeds}} - 1\}$: features `RandomReLUFeatures(p, rng=s).fit(X_train)`, $\beta$ = `min_norm_least_squares(features of X_train, Y_train, ridge=ridge)`, predictions $\Phi\beta$. returns a dict of `(len(widths), n_seeds)` arrays: `'test_error'` (fraction of test points whose argmax prediction isn't `y_test`), `'test_mse'` and `'train_mse'` (squared error against the one-hot labels, summed over classes, averaged over points).

> **🔮 predict:** with $n = 1{,}000$ MNIST digits, how much worse is the test error at $p = 1{,}000$ than at $p = 300$? and is $p = 8{,}000$ better or worse than $p = 300$?

> **▶ build: stage 4.** implement `double_descent_curve`.

### idea 5: the solution gradient descent chooses

past the threshold the loss has a whole affine subspace of global minima, all with zero training error. least squares with the pseudoinverse picks the minimum-norm one by fiat. but networks aren't trained with pseudoinverses; they're trained by gradient descent, which minimizes the loss and nothing else. which interpolant does it find?

for least squares the answer is exact. the gradient of $\frac{1}{2n}\lVert Xw - y\rVert^2$ is $\frac1n X^\top(Xw - y)$, a combination of the rows of $X$. so every step changes $w$ only within the row space of $X$, and the component of $w$ in the null space never changes. starting from $w_0 = 0$, the iterates stay in the row space, and the only interpolant in the row space is the minimum-norm one: **gradient descent from zero converges to $X^+ y$**. from any other start, it converges to $X^+ y$ plus the null-space part of $w_0$, which carries the initialization forever (on paper 1).

this is the simplest instance of an **implicit bias**: the optimizer, not the objective, decides among equally good solutions. it is not special to least squares. on linearly separable data, gradient descent on the logistic loss drives $\lVert w\rVert \to \infty$ while its *direction* converges to the hard-margin SVM solution (Soudry et al., 2018). for deep networks the implicit biases of gradient descent are only partly understood (towards low-rank solutions in deep linear networks, Gunasekar et al., 2017; towards flat regions, topic 02), and they are one of the main candidate explanations for the generalization of interpolating networks. the others are data-dependent bounds that do survive the random-label test: Dziugaite & Roy (2017) optimized a PAC-Bayes bound (chapter 13) around a trained MNIST network and got non-vacuous guarantees (test-error bounds of roughly 16–22% on a binary MNIST task, depending on the network), by measuring how much noise the trained weights tolerate.

> **spec.** `gd_least_squares(X, y, *, lr, steps, w0=None)` returns $w$ after `steps` iterations of $w \leftarrow w - \text{lr}\cdot\frac1n X^\top(Xw - y)$ from `w0` (zeros if `None`; `w0` is not modified). `y` is `(n,)`.

> **🔮 predict:** two runs of gradient descent on the same overparameterized problem, one from $w_0 = 0$ and one from a random $w_0$, both reach zero training error. which has the lower test error, and does the gap grow or shrink with the size of $w_0$?

> **▶ build: stage 5.** implement `gd_least_squares`.

## worked example

one equation, two unknowns: $X = \begin{pmatrix}1 & 1\end{pmatrix}$, $y = 2$. every $w$ with $w_1 + w_2 = 2$ fits exactly.

- the minimum-norm solution is $X^+ y = X^\top(XX^\top)^{-1}y = (1, 1)^\top \cdot \tfrac12 \cdot 2 = (1, 1)$, with norm $\sqrt2$.
- gradient descent from $w_0 = 0$: every gradient is a multiple of $(1, 1)$, so $w_t = c_t(1, 1)$ and the limit is $(1, 1)$.
- from $w_0 = (1, -1)$: the null space of $X$ is spanned by $(1, -1)$, and $w_0$ lies entirely in it. that part never changes; the row-space part converges to $(1, 1)$. the limit is $(1, 1) + (1, -1) = (2, 0)$: it fits ($2 + 0 = 2$) with norm 2.

your `gd_least_squares` must reproduce the general version of this (the stage 5 checks: $X^+y$ from zero, $X^+y + (I - X^+X)w_0$ from $w_0$). and for label corruption: with 37 labels and `fraction=0.7`, exactly $\text{round}(25.9) = 26$ labels change, each to a class it wasn't.

## on paper

1. **gradient descent stays in the row space.** for $w_{t+1} = w_t - \frac{\eta}{n}X^\top(Xw_t - y)$, show that $(I - P)w_t = (I - P)w_0$ for all $t$, where $P = X^+X$ is the projection onto the row space. conclude that if the iterates converge to an interpolant, the limit is $X^+y + (I - P)w_0$.

<details><summary>solution</summary>

$(I - P)X^\top = 0$ because the columns of $X^\top$ lie in the row space. so $(I - P)w_{t+1} = (I - P)w_t$, and by induction $(I - P)w_t = (I - P)w_0$. the limit $w_\infty$ interpolates, $Xw_\infty = y$; every interpolant is $X^+y + v$ with $v$ in the null space ($X^+y$ is in the row space), so $Pw_\infty = X^+y$ and $(I - P)w_\infty = (I - P)w_0$. adding: $w_\infty = X^+y + (I - P)w_0$. with $w_0 = 0$ it is the minimum-norm interpolant, since any null-space addition increases the norm (Pythagoras).

</details>

2. **why the peak is at $p = n$.** for $p \ge n$ and $\Phi$ of full row rank, show that the minimum-norm solution has $\lVert\beta\rVert^2 = Y^\top(\Phi\Phi^\top)^{-1}Y \ge \lVert Y\rVert^2/\sigma_{\max}^2$ and can be as large as $\lVert Y\rVert^2/\sigma_{\min}^2$, with $\sigma$ the singular values of $\Phi$. what happens to $\sigma_{\min}$ as $p \downarrow n$?

<details><summary>solution</summary>

$\beta = \Phi^\top(\Phi\Phi^\top)^{-1}Y$, so $\lVert\beta\rVert^2 = Y^\top(\Phi\Phi^\top)^{-1}\Phi\Phi^\top(\Phi\Phi^\top)^{-1}Y = Y^\top(\Phi\Phi^\top)^{-1}Y$, and the eigenvalues of $(\Phi\Phi^\top)^{-1}$ are $1/\sigma_i^2$, which gives the two bounds (the upper one attained when $Y$ is along the weakest direction; noise has a component there). for a random $n \times p$ matrix with independent entries, the smallest singular value behaves like $\sqrt p - \sqrt n$ (times the entry scale): it goes to zero as $p \to n$ (for a square $n \times n$ matrix it's of order $1/\sqrt n$, while the largest is of order $\sqrt n$). so the interpolating solution's norm, and its sensitivity to noise in $Y$, blow up at the threshold, and shrink again as $p$ grows past it. that's the variance spike.

</details>

3. **random labels and Rademacher complexity.** suppose a class $\mathcal H$ of $\pm1$-valued functions can fit every labeling of a sample of size $n$. compute $\hat{\mathfrak R}_n(\mathcal H)$ and explain what it does to the bound $\text{error} \le \widehat{\text{error}} + 2\hat{\mathfrak R}_n + 3\sqrt{\ln(2/\delta)/(2n)}$.

<details><summary>solution</summary>

for every sign vector $\sigma$ some $h$ has $h(x_i) = \sigma_i$, so $\sup_h \frac1n\sum\sigma_i h(x_i) = 1$ and $\hat{\mathfrak R}_n = 1$. the bound becomes $\text{error} \le \widehat{\text{error}} + 2 + \dots$, which is above 1 and says nothing. to get a meaningful bound you must replace $\mathcal H$ by a smaller, data-dependent set of functions that training actually reaches (bounded norm, low compressibility, a posterior around the trained weights), which is the program of norm-based and PAC-Bayes bounds.

</details>

## experiment

the lab trains an MLP with 369 parameters per training image on 1,000 MNIST digits with true, half-corrupted and fully random labels, and compares the time to memorize and the test accuracy with chapter 13's VC bound. then it sweeps a random-features model from 100 to 8,000 features through the interpolation threshold $p = n = 1{,}000$, three feature draws per width, with controls for label noise and a ridge. finally it runs gradient descent to an interpolating solution from zero and from random starts and compares the solutions with the minimum-norm one.

## check your understanding

1. if VC and Rademacher bounds are vacuous for networks, why were they worth learning?

<details><summary>answer</summary>

they are correct statements about worst-case uniform convergence, and they're tight for some classes; what the random-label experiment shows is that "the class a network belongs to" is the wrong object for networks. the same tools, applied to smaller, data-dependent classes (networks with bounded norms reached by training, posteriors around trained weights), give the modern bounds, some of which are non-vacuous (Dziugaite & Roy, 2017).

</details>

2. in the double-descent experiment, every model to the right of the threshold has zero training error. what then distinguishes $p = 1{,}050$ from $p = 8{,}000$?

<details><summary>answer</summary>

the norm of the interpolating solution and how smooth the resulting function is. at $p = 1{,}050$ there are only 50 spare directions, so fitting the data exactly still needs huge coefficients along nearly-degenerate feature combinations; at $p = 8{,}000$ the minimum-norm fit can spread over many directions, each used a little, which averages out noise. training error can't see the difference; the norm can.

</details>

3. why does a tuned ridge remove the peak instead of just lowering it?

<details><summary>answer</summary>

the peak is variance: near the threshold the ridgeless fit amplifies noise along directions with tiny singular values. a ridge $\lambda$ caps the amplification at about $1/\lambda$ instead of $1/\sigma_{\min}^2$, so the near-singular directions stop mattering. with $\lambda$ chosen for each $p$, the bias–variance trade-off is optimized separately at every width and the curve becomes monotone; the lab's 0.001 already flattens it.

</details>

4. Zhang et al. found that random labels take longer to fit than true labels. what does that suggest about early stopping?

<details><summary>answer</summary>

the network fits the shared structure first and memorizes the exceptions later (in the lab: about 23 epochs for true labels, about 80 for random ones). with partly noisy labels, stopping early catches the network after it has learned the clean part and before it has memorized the noise, which is why early stopping is such an effective regularizer under label noise, and why "epoch-wise double descent" exists.

</details>

## common bugs

- **corrupting labels with a class drawn from all classes**, so some "corrupted" labels stay correct; or drawing positions with replacement, so fewer than $fn$ change.
- **the normal equations at $p \ge n$**: $\Phi^\top\Phi$ is singular and `solve` either fails or returns garbage. use `pinv`/`lstsq`, or the dual form with a ridge.
- **the ridge multiplied by $n$** in one form and not the other; the two closed forms must agree.
- **refitting the random features on the test set.** the features are part of the model: draw $W$ once at `fit` and reuse it for every transform.
- **a single feature draw near the threshold.** the test error at $p = n$ varies a lot between draws; use the median of several.
- **forgetting the $1/n$** in the gradient: the learning rate then means something different for every dataset size.

## going deeper

- Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization", *ICLR*. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off", *PNAS* 116(32). [arXiv:1812.11118](https://arxiv.org/abs/1812.11118)
- Nakkiran, Kaplun, Bansal, Yang, Barak & Sutskever (2020), "Deep double descent: where bigger models and more data hurt", *ICLR*. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
- Hastie, Montanari, Rosset & Tibshirani (2022), "Surprises in high-dimensional ridgeless least squares interpolation", *Annals of Statistics* 50(2). [arXiv:1903.08560](https://arxiv.org/abs/1903.08560)
- Mei & Montanari (2022), "The generalization error of random features regression: precise asymptotics and the double descent curve", *Communications on Pure and Applied Mathematics* 75(4). [arXiv:1908.05355](https://arxiv.org/abs/1908.05355)
- Advani & Saxe (2017), "High-dimensional dynamics of generalization error in neural networks". [arXiv:1710.03667](https://arxiv.org/abs/1710.03667)
- Nakkiran, Venkat, Kakade & Ma (2021), "Optimal regularization can mitigate double descent", *ICLR*. [arXiv:2003.01897](https://arxiv.org/abs/2003.01897)
- Bartlett, Long, Lugosi & Tsigler (2020), "Benign overfitting in linear regression", *PNAS* 117(48). [arXiv:1906.11300](https://arxiv.org/abs/1906.11300)
- Soudry, Hoffer, Nacson, Gunasekar & Srebro (2018), "The implicit bias of gradient descent on separable data", *JMLR* 19. [arXiv:1710.10345](https://arxiv.org/abs/1710.10345)
- Gunasekar, Woodworth, Bhojanapalli, Neyshabur & Srebro (2017), "Implicit regularization in matrix factorization", *NeurIPS*. [arXiv:1705.09280](https://arxiv.org/abs/1705.09280)
- Dziugaite & Roy (2017), "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data", *UAI*. [arXiv:1703.11008](https://arxiv.org/abs/1703.11008)
- Rahimi & Recht (2007), "Random features for large-scale kernel machines", *NeurIPS*. [papers.nips.cc](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)

## connections

- ← builds on: VC, Rademacher and PAC-Bayes bounds (ch 13), bias–variance (ch 07 · 03), least squares, the pseudoinverse and ridge (ch 02, ch 08), random features and kernel ridge (ch 09 · 05), the NTK and lazy training (topic 03), flat minima (topic 02).
- → used in: the capstone (double descent with trained networks), topic 05 (why bigger models keep improving), label noise and memorization in self-supervised and large-scale training (ch 25, 27), and the research milestones.
