# rademacher complexity

## why this matters

the VC dimension of topic 02 has two weaknesses you can feel. it's **worst case over data**: it asks whether *some* configuration of points can be shattered, even if your data never looks like that. and it ignores **scale**: a linear classifier in 10,000 dimensions has VC dimension 10,001, whether its weights are huge or tiny, whether the data lies near a line or fills the space. yet regularized linear models in very high dimension generalize fine, and every practitioner knows that the size of the weights matters.

**Rademacher complexity** fixes both. it measures how well a class can correlate with *pure noise* on **your sample**: draw random signs $\sigma_i = \pm1$, one per example, and ask how large $\frac1n\sum_i\sigma_ih(x_i)$ can be made by choosing $h$ from the class. a class that can match any sign pattern on your data has complexity 1, and fitting the training labels proves nothing about it. a class that can barely track noise has complexity near 0, and then fitting the labels is evidence of real structure. the quantity is data-dependent, it sees norms and geometry, and it plugs straight into a generalization bound.

it's also something you can **estimate by simulation**, which is what this topic does. you'll estimate it by Monte Carlo for finite classes and for norm-bounded linear classes, check the estimates against two classical bounds, turn it into a risk bound, and finally measure it the way Zhang et al. (2017) did for deep networks: train a real learner on random labels and see how well it fits them.

## warm-up: predict first

1. a class contains a single hypothesis. what's its Rademacher complexity?
2. a class contains *every* labeling of your $n$ training points. what's its Rademacher complexity, and what does a generalization bound built on it say?
3. you multiply the number of hypotheses in a finite class by 1,000. roughly how much does its Rademacher complexity grow?
4. linear predictors with $\|w\|_2 \le 1$ on data with $\|x\| \le 1$: does their complexity grow with the dimension $d$?
5. a model fits random labels perfectly on your training set. can it still generalize well on the real labels?

## prerequisites check

- [ ] you know the PAC setup, empirical and true risk, and Hoeffding's inequality (→ topic 01)
- [ ] you know shattering and the growth function (→ topic 02)
- [ ] you know vector norms, dual norms and the Cauchy–Schwarz inequality (→ ch 02 · topic 01)
- [ ] you know Jensen's inequality and Monte Carlo standard errors (→ ch 04 · topics 04–05)
- [ ] you've trained logistic regression and $k$-NN (→ ch 08 · topic 04, ch 09 · topic 01)

## core ideas

### idea 1: correlation with noise

let $\mathcal F$ be a class of real-valued functions and $S = (z_1, \dots, z_n)$ a sample. the **empirical Rademacher complexity** of $\mathcal F$ on $S$ is

$$\hat{\mathfrak R}_S(\mathcal F) = \mathbb E_\sigma\Big[\sup_{f \in \mathcal F}\frac1n\sum_{i=1}^n\sigma_if(z_i)\Big],$$

where $\sigma_1, \dots, \sigma_n$ are independent **Rademacher** variables, $+1$ or $-1$ with probability $\frac12$ each, and only $\sigma$ is random ($S$ is fixed). its expectation over samples, $\mathfrak R_n(\mathcal F) = \mathbb E_S\hat{\mathfrak R}_S(\mathcal F)$, is the (distribution-dependent) Rademacher complexity. read the inner expression as a game: nature draws a random sign pattern, you pick the function that agrees with it best, and the score is the average correlation. the definition has no absolute value (Mohri et al., 2018, Definition 3.1); some older texts include one, which changes constants.

two extremes anchor the scale. a single function scores $\mathbb E_\sigma\frac1n\sum_i\sigma_if(z_i) = 0$ (warm-up 1): it can't adapt to the noise. a class of $\pm1$-valued functions that realizes every sign pattern on $S$ always contains $\sigma$ itself and scores exactly $\frac1n\sum_i\sigma_i^2 = 1$ (warm-up 2). everything else is in between, and the more expressive the class *on this sample*, the higher it scores. scaling the class by $c > 0$ scales the complexity by $c$.

for a finite class, represent it by its **prediction matrix** $P$ of shape (hypotheses, $n$), with $P_{hi} = h(x_i)$. then the inner supremum is a max over rows of $P\sigma/n$, and a Monte Carlo estimate draws many sign vectors at once: a $(D, n)$ matrix $\Sigma$ of signs, the product $\Sigma P^\top$ of shape (draws, hypotheses), a max per row, an average. the estimate has Monte Carlo error $\text{sd}/\sqrt D$, where the sd of the per-draw maximum is at most $\max_h\|h\|_2/n$, so a thousand draws pin it down to a few thousandths.

> **spec.** `empirical_rademacher(predictions, *, n_draws=1000, rng=None)`: `predictions` is `(n_hypotheses, n)` (a 1-D array is one hypothesis). with `gen = as_rng(rng)`, draw `n_draws` sign vectors (e.g. `2 * gen.integers(0, 2, size=(n_draws, n)) - 1`, in chunks if you like), and return the mean over draws of $\max_h\frac1n\sum_i\sigma_iP_{hi}$ as a float. no Python loop over draws *and* hypotheses: use a matrix product.

> **▶ build: stage 1.** implement `empirical_rademacher`.

### idea 2: Massart's lemma

for a finite class you can bound the complexity without simulation. **Massart's lemma**: if $A \subset \mathbb R^n$ is a finite set of vectors with $\max_{a \in A}\|a\|_2 \le r$, then

$$\mathbb E_\sigma\Big[\max_{a \in A}\frac1n\sum_i\sigma_ia_i\Big] \le \frac{r\sqrt{2\ln|A|}}{n}.$$

the proof (on paper 2) is the same trick as the union bound in topic 01, done with moment generating functions: each $\sigma^\top a$ is sub-Gaussian with variance proxy $\|a\|^2 \le r^2$, and the maximum of $N$ such variables is at most $r\sqrt{2\ln N}$ in expectation. for $\pm1$-valued hypotheses, $r = \sqrt n$ and the bound is $\sqrt{2\ln N/n}$: complexity grows like the square root of the log of the class size (warm-up 3: a thousand times more hypotheses adds $2\ln 1000/n$ under the square root, roughly doubling a small class's complexity, not multiplying it by 1000).

only *distinct* rows of $P$ count: duplicated hypotheses can't help the maximum. that makes Massart's lemma the bridge to topic 02. the distinct rows of an infinite class's prediction matrix on $S$ are its restriction $\mathcal H_S$, of size at most $\Pi_{\mathcal H}(n) \le (en/d)^d$, so

$$\hat{\mathfrak R}_S(\mathcal H) \le \sqrt{\frac{2d\ln(en/d)}{n}}$$

for a class of VC dimension $d$ (Mohri et al., Corollary 3.8 with 3.18). Rademacher complexity is never worse than the VC analysis, and often much better: Massart's lemma is nearly tight for *random* hypotheses, but a structured class with the same number of labelings, like thresholds, fits noise far worse (the lab measures both).

> **spec.** `massart_bound(predictions)` returns $\max_h\|P_h\|_2\sqrt{2\ln N}/n$ with $N$ the number of **distinct** rows (`np.unique(P, axis=0)`); a single distinct row gives 0.

> **🔮 predict:** on $n = 100$ points, Massart's bound for 10 random $\pm1$ hypotheses is 0.215, and for 10,000 it's 0.429. will the true complexities be close to these bounds, or far below them?

> **▶ build: stage 2.** implement `massart_bound`.

### idea 3: norm-bounded linear classes

for linear predictors with a norm constraint, the supremum inside the expectation has a closed form. by the definition of the **dual norm** $\|v\|_* = \sup_{\|w\| \le 1}w^\top v$,

$$\sup_{\|w\|_p \le B}\frac1n\sum_i\sigma_iw^\top x_i = \sup_{\|w\|_p \le B}\frac1n w^\top\Big(\sum_i\sigma_ix_i\Big) = \frac Bn\Big\|\sum_i\sigma_ix_i\Big\|_q, \qquad \frac1p + \frac1q = 1.$$

the dual of $\ell_2$ is $\ell_2$, the dual of $\ell_1$ is $\ell_\infty$, and the dual of $\ell_\infty$ is $\ell_1$. so the Rademacher complexity of a norm ball is $\frac Bn\mathbb E\|\sum_i\sigma_ix_i\|_q$: one vector norm per draw, no optimization over $w$.

for the $\ell_2$ ball, Jensen's inequality and $\mathbb E\sigma_i\sigma_j = \mathbb 1[i = j]$ give

$$\hat{\mathfrak R}_S \le \frac Bn\sqrt{\mathbb E\Big\|\sum_i\sigma_ix_i\Big\|_2^2} = \frac Bn\sqrt{\sum_i\|x_i\|_2^2} \le \frac{BR}{\sqrt n} \quad\text{if all } \|x_i\|_2 \le R.$$

**there's no dimension in it** (warm-up 4). a linear class in a million dimensions, with bounded weights and bounded data, is no more complex than one in two. this is the theoretical root of margin-based generalization (topic 06) and of why kernel methods, which work in infinite-dimensional feature spaces, can generalize. the price is that the *norm* matters: unregularized weights can be as large as they like. for the $\ell_1$ ball the sup is attained at one of the $2d$ vertices $\pm Be_j$, so Massart's lemma gives $\hat{\mathfrak R}_S \le B\max_i\|x_i\|_\infty\sqrt{2\ln(2d)/n}$: logarithmic in $d$, the reason sparsity-inducing penalties cope with far more features than examples.

> **spec.** `norm_ball_rademacher(X, B, *, norm=2, n_draws=1000, rng=None)`: `X` (n, d), `norm` in `{1, 2, np.inf}`. draw `n_draws` sign vectors as in stage 1, compute $v = \sigma^\top X$ for each, and return $\frac Bn$ times the mean of $\|v\|_q$ with the dual norm $q$ (`np.linalg.norm(..., ord=q, axis=1)`). `l2_rademacher_bound(X, B)` returns $\frac Bn\sqrt{\sum_i\|x_i\|_2^2}$.

> **🔮 predict:** $X$ is the $16 \times 16$ identity and $B = 3$. what are the complexities of the $\ell_2$, $\ell_1$ and $\ell_\infty$ balls? (no simulation needed.)

> **▶ build: stage 3.** implement `norm_ball_rademacher` and `l2_rademacher_bound`.

### idea 4: from complexity to a risk bound

here's the theorem that makes the quantity useful (Mohri et al., 2018, Theorem 3.3; Bartlett & Mendelson, 2002). let $\mathcal G$ be a class of functions with values in $[0, 1]$ (think: losses $g(z) = \ell(h(x), y)$ of hypotheses $h$), and $S$ an iid sample of size $n$. for any $\delta > 0$, with probability at least $1 - \delta$, **every** $g \in \mathcal G$ satisfies

$$\mathbb E[g(z)] \le \frac1n\sum_{i=1}^ng(z_i) + 2\hat{\mathfrak R}_S(\mathcal G) + 3\sqrt{\frac{\ln(2/\delta)}{2n}}.$$

where does it come from? let $\Phi(S) = \sup_g(\mathbb E g - \hat{\mathbb E}_Sg)$, the worst gap between true and empirical risk. (a) changing one example changes $\Phi$ by at most $1/n$, so **McDiarmid's inequality** (a Hoeffding for functions of independent variables) says $\Phi$ is close to its mean. (b) **symmetrization**: replace $\mathbb E g$ by the average on an independent "ghost" sample $S'$; swapping $z_i$ and $z_i'$ doesn't change the distribution, so random signs can be inserted, and $\mathbb E\Phi \le 2\mathfrak R_n(\mathcal G)$. (c) McDiarmid once more replaces $\mathfrak R_n$ by the computable $\hat{\mathfrak R}_S$. each use of McDiarmid costs a $\sqrt{\ln(\cdot/\delta)/(2n)}$ term, which is where the 3 and the 2 inside the log come from.

for **binary classification** with labels and predictions in $\{-1, +1\}$, the 0-1 loss is $\mathbb 1[h(x) \ne y] = (1 - yh(x))/2$. the constant $\frac12$ contributes nothing to the complexity, and $-y_i\sigma_i/2$ has the same distribution as $\sigma_i/2$, so $\hat{\mathfrak R}_S(\text{loss class}) = \frac12\hat{\mathfrak R}_S(\mathcal H)$ (on paper 1). the classification bound is therefore

$$L(h) \le L_S(h) + \hat{\mathfrak R}_S(\mathcal H) + 3\sqrt{\frac{\ln(2/\delta)}{2n}}.$$

warm-up 2's class, which shatters the sample, has $\hat{\mathfrak R}_S = 1$, and the bound is vacuous: it can't be below 1. that's not a weakness of the proof. a class that can fit anything on your sample gives no evidence from fitting it.

> **spec.** `rademacher_bound(emp_risk, rademacher, n, delta)` returns `emp_risk + 2 * rademacher + 3 * sqrt(ln(2 / delta) / (2 * n))`, where `rademacher` is the complexity of the **loss** class (for 0-1 loss, half the classifier class's).

> **▶ build: stage 4.** implement `rademacher_bound`.

### idea 5: measuring complexity with a learner

for a class you can only access through a learning algorithm (a neural network, a kernel machine, $k$-NN), there's no prediction matrix to maximize over. but the supremum in the definition is an optimization problem, and fitting labels is what learners do. for $\pm1$ predictions, $\frac1n\sum_i\sigma_ih(x_i) = 1 - 2\cdot\text{(training error on labels } \sigma)$, so maximizing the correlation with $\sigma$ is the same as minimizing the training error on the labels $\sigma$. if your learner were an exact ERM for the 0-1 loss over $\mathcal H$, then

$$\hat{\mathfrak R}_S(\mathcal H) = \mathbb E_\sigma\Big[\frac1n\sum_i\sigma_i\,\hat h_\sigma(x_i)\Big], \qquad \hat h_\sigma = \text{ERM on } (x_i, \sigma_i).$$

real learners aren't exact 0-1 ERMs (logistic regression minimizes a surrogate, networks are trained by SGD), so the measured correlation is a **lower bound** on the complexity of the class they search, and a bound built from it isn't a certified bound. it's still the most direct evidence there is: Zhang, Bengio, Hardt, Recht & Vinyals (2017) showed that standard image networks fit completely random labels on CIFAR-10 perfectly, so for them $\hat{\mathfrak R}_S \approx 1$ and every bound of this kind is vacuous, even though the same networks generalize well on real labels (warm-up 5). explaining that gap is one of the central problems of deep learning theory (ch 19).

> **spec.** `rademacher_via_erm(fit_predict, X, *, n_draws=50, rng=None)`: with `gen = as_rng(rng)`, repeat `n_draws` times: draw signs `sigma` of shape `(n,)` (floats $\pm1$), call `h = fit_predict(X, sigma)` (it trains on labels `sigma` and returns its $\pm1$ predictions on `X`), and record `mean(sigma * h)`. return the average over draws.

> **🔮 predict:** logistic regression on 1,000 digits projected onto their top $k$ principal components. at roughly which $k$ will it fit random labels almost perfectly: $k = 50$, $k \approx 500$, or only at $k \approx 1000$?

> **▶ build: stage 5.** implement `rademacher_via_erm`.

## worked example

**the smallest nontrivial class.** $\mathcal H = \{\text{all } +1, \text{all } -1\}$ on $n = 2$ points. the four sign patterns give $\max(\sigma_1 + \sigma_2, -\sigma_1 - \sigma_2)/2 = |\sigma_1 + \sigma_2|/2$: values $1, 0, 0, 1$, so $\hat{\mathfrak R}_S = \frac12$. massart: $r = \sqrt2$, $N = 2$, bound $\sqrt2\sqrt{2\ln 2}/2 = \sqrt{\ln 2} = 0.833$. for $n = 10$ the exact value is $\mathbb E|\sum_i\sigma_i|/10 = 0.246$, which your stage 1 must reproduce within Monte Carlo error.

**the identity design.** $X = I_{16}$, $B = 3$: $\sum_i\sigma_ix_i = \sigma$ for every draw, so $\|\sigma\|_2 = 4$, $\|\sigma\|_\infty = 1$, $\|\sigma\|_1 = 16$ exactly, and the three balls have complexities $3 \cdot 4/16 = 0.75$ ($\ell_2$), $3/16 = 0.1875$ ($\ell_1$) and $3$ ($\ell_\infty$), with no Monte Carlo error at all. Jensen's bound $\frac3{16}\sqrt{16} = 0.75$ is exact here.

**a bound.** empirical risk 0.1, loss-class complexity 0.05, $n = 1000$, $\delta = 0.05$: $0.1 + 0.1 + 3\sqrt{\ln 40/2000} = 0.2 + 3 \times 0.04295 = 0.3288$.

## on paper

1. **the 0-1 loss halves the complexity.** for $h$ and $y$ in $\{-1, +1\}$, show $\hat{\mathfrak R}_S(\{(x, y) \mapsto \mathbb 1[h(x) \ne y]\}) = \frac12\hat{\mathfrak R}_{S_X}(\mathcal H)$.

<details><summary>solution</summary>

$\mathbb 1[h(x) \ne y] = (1 - yh(x))/2$. so $\mathbb E_\sigma\sup_h\frac1n\sum_i\sigma_i\frac{1 - y_ih(x_i)}2 = \mathbb E_\sigma\frac1{2n}\sum_i\sigma_i + \frac12\mathbb E_\sigma\sup_h\frac1n\sum_i(-y_i\sigma_i)h(x_i)$. the first term is 0. for fixed $y$, $(-y_i\sigma_i)_i$ has the same distribution as $(\sigma_i)_i$, so the second term is $\frac12\hat{\mathfrak R}_{S_X}(\mathcal H)$.

</details>

2. **Massart's lemma.** prove it. hint: for any $t > 0$, $\exp(t\,\mathbb E\max_a\sigma^\top a) \le \mathbb E\max_a e^{t\sigma^\top a} \le \sum_a\mathbb E e^{t\sigma^\top a}$, and $\mathbb E e^{t\sigma_i a_i} = \cosh(ta_i) \le e^{t^2a_i^2/2}$.

<details><summary>solution</summary>

by Jensen and then bounding the max by the sum, $e^{t\,\mathbb E\max_a\sigma^\top a} \le \sum_a\prod_i\cosh(ta_i) \le \sum_ae^{t^2\|a\|^2/2} \le |A|e^{t^2r^2/2}$. taking logs, $\mathbb E\max_a\sigma^\top a \le \frac{\ln|A|}t + \frac{tr^2}2$. minimizing over $t$ ($t = \sqrt{2\ln|A|}/r$) gives $r\sqrt{2\ln|A|}$; divide by $n$.

</details>

3. **the linear bound.** prove $\hat{\mathfrak R}_S(\{x \mapsto w^\top x : \|w\|_2 \le B\}) \le \frac Bn\sqrt{\sum_i\|x_i\|^2}$.

<details><summary>solution</summary>

by Cauchy–Schwarz the sup of $w^\top v$ over the ball is $B\|v\|$ with $v = \sum_i\sigma_ix_i$. by Jensen ($\sqrt\cdot$ is concave), $\mathbb E\|v\| \le \sqrt{\mathbb E\|v\|^2} = \sqrt{\sum_{i,j}\mathbb E[\sigma_i\sigma_j]x_i^\top x_j} = \sqrt{\sum_i\|x_i\|^2}$.

</details>

4. **when is the bound vacuous?** show that if $\mathcal H$ shatters the sample (as $\pm1$ functions), the classification bound of idea 4 exceeds 1 for every $h$, whatever $n$ is.

<details><summary>solution</summary>

shattering means every sign pattern, including $\sigma$, is in $\mathcal H_S$, so the sup is $\frac1n\sum\sigma_i^2 = 1$ for every draw and $\hat{\mathfrak R}_S(\mathcal H) = 1$. the bound is $L_S(h) + 1 + (\text{positive}) > 1$. a 1-NN classifier shatters any sample of distinct points, so no bound of this form can say anything about it.

</details>

## experiment

`lab.py` investigates, with your code:

1. how the complexity of a finite class grows with the number of hypotheses, against Massart's lemma, and whether structure matters.
2. how the complexities of $\ell_2$- and $\ell_1$-bounded linear classes grow with the dimension.
3. how well logistic regression and 1-NN fit random labels on real digits, and what the resulting bound says next to their real test error.

## check your understanding

1. why is the Rademacher complexity "data-dependent", and why is that an advantage over the VC dimension?

<details><summary>answer</summary>

it's computed on your sample: it sees how the class behaves on the points you have, not on the worst configuration. if your data lies near a low-dimensional subspace, or the class's functions are small on it, the complexity is small even if the VC dimension is huge.

</details>

2. a class is closed under negation ($h \in \mathcal H \Rightarrow -h \in \mathcal H$). show its complexity is $\mathbb E_\sigma\sup_h|\frac1n\sum_i\sigma_ih(x_i)|$. why does that make "with or without absolute value" a minor issue?

<details><summary>answer</summary>

for each $\sigma$, $\sup_h\sum\sigma_ih_i \ge$ both $\sum\sigma_ih_i$ and $-\sum\sigma_ih_i$ for every $h$ (using $-h$), so it equals $\sup_h|\sum\sigma_ih_i|$. most classes in practice (linear classes, norm balls, thresholds with both orientations) are closed under negation, so the two definitions agree; for others they differ by small constants.

</details>

3. Massart's lemma is nearly tight for random hypotheses but loose for thresholds. why?

<details><summary>answer</summary>

$N$ random sign vectors give $N$ nearly independent correlations $\sigma^\top h/n$, and the maximum of $N$ independent Gaussians really is $\approx\sqrt{2\ln N}$ standard deviations. thresholds' labelings are highly correlated with each other (neighbouring thresholds differ on one point), so their correlations with $\sigma$ move together, and the maximum is much smaller. the union bound in the proof treats them as if they were independent.

</details>

4. why does the $\ell_2$ bound $BR/\sqrt n$ not depend on the dimension, while the VC dimension of halfspaces is $d + 1$?

<details><summary>answer</summary>

the VC dimension counts labelings with *unbounded* weights and ignores scale. the Rademacher bound is for real-valued predictions with a norm constraint, so only the geometry ($\|w\|$, $\|x\|$) matters. to use it for classification you compare predictions with a margin (topic 06), which brings the scale back in through $B/\gamma$.

</details>

5. why is the random-label correlation of a learner only a lower bound on the class's complexity?

<details><summary>answer</summary>

the complexity takes the *supremum* over the class. a learner that minimizes a surrogate loss, or stops early, or gets stuck, finds a hypothesis that correlates with $\sigma$ at most as well as the best one in its class. exact 0-1 ERM is NP-hard for halfspaces, so in practice you only ever see lower bounds this way.

</details>

6. in the risk bound, where does the factor 2 in front of $\hat{\mathfrak R}_S$ come from?

<details><summary>answer</summary>

from symmetrization: the gap between the true risk and the empirical risk is bounded by the gap between two empirical risks on independent samples, and that difference, with random signs inserted, is a sum of *two* Rademacher averages (one per sample).

</details>

## common bugs

- **an absolute value inside the expectation.** symptom: a single hypothesis gets complexity $\mathbb E|\frac1n\sum\sigma_ih_i| > 0$ instead of 0.
- **dividing by the number of hypotheses instead of $n$.** symptom: the all-labelings class doesn't score exactly 1.
- **counting duplicate rows in Massart's $N$.** symptom: a bound that grows when you copy hypotheses.
- **the wrong dual norm.** symptom: the $\ell_1$ ball looks more complex than the $\ell_2$ ball. $\ell_1 \leftrightarrow \ell_\infty$, $\ell_2 \leftrightarrow \ell_2$.
- **0/1 labels instead of $\pm1$ signs.** symptom: random-label correlations around 0.5 for learners that ignore the data. use signs, and convert to whatever labels your learner wants inside `fit_predict`.
- **forgetting the factor $\frac12$ for the 0-1 loss.** symptom: classification bounds that are looser than they need to be (still valid).

## going deeper

- Bartlett & Mendelson (2002), "Rademacher and Gaussian complexities: risk bounds and structural results", *JMLR* 3. [jmlr.org/papers/v3/bartlett02a](https://www.jmlr.org/papers/v3/bartlett02a.html)
- Koltchinskii & Panchenko (2002), "Empirical margin distributions and bounding the generalization error of combined classifiers", *Annals of Statistics* 30(1). [doi:10.1214/aos/1015362183](https://doi.org/10.1214/aos/1015362183)
- Kakade, Sridharan & Tewari (2008), "On the complexity of linear prediction: risk bounds, margin bounds, and regularization", *NeurIPS*. complexities of $\ell_p$ balls. [papers.nips.cc/paper/3415](https://papers.nips.cc/paper/3415-on-the-complexity-of-linear-prediction-risk-bounds-margin-bounds-and-regularization)
- Zhang, Bengio, Hardt, Recht & Vinyals (2017), "Understanding deep learning requires rethinking generalization", *ICLR*. [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., ch. 3 (Rademacher complexity, Massart's lemma, the bound) and ch. 5–6 (margins, kernels). [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)
- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning*, ch. 26. [free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)

## connections

- ← builds on: Hoeffding's inequality and the union bound (topic 01), the growth function and Sauer's lemma (topic 02), dual norms and Cauchy–Schwarz (ch 02), Jensen's inequality and Monte Carlo (ch 04), logistic regression and the lasso (ch 08), $k$-NN (ch 09).
- → used in: computing bounds for real models (topic 04), margin bounds, which divide the linear-class complexity by the margin (topic 06), the capstone's generalization report, and the random-label puzzle of deep learning (ch 19 · topic 04).
