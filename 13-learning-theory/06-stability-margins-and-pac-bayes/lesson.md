# stability, margins & PAC-Bayes

## why this matters

topic 04 ended with a verdict: for a linear model on MNIST, the VC bound is vacuous (0.89), the "count the bits" bound is weak, and only the test-set bound says something useful. every uniform bound charged for the whole hypothesis class, as if the learner could have picked any of its members with equal ease. but learners don't do that. regularized logistic regression prefers small weights; SGD starts at a particular point and wanders only so far; a margin-maximizing SVM picks boundaries far from the data. the bounds in this topic charge for **what the learner actually did**, not for everything it could have done.

three ideas do that, each from a different angle. **PAC-Bayes** (McAllester 1999; Seeger 2002; Maurer 2004) puts a prior over hypotheses before seeing data, lets the learner output a *distribution* (a posterior) over hypotheses, and charges the KL divergence between the two: a posterior close to the prior is cheap, however rich the class. **margin bounds** charge for the scale of a classifier relative to how confidently it separates the data. **stability** (Bousquet & Elisseeff 2002) charges for how much the learned function changes when one training point is removed.

the payoff is concrete. Dziugaite & Roy (2017) used PAC-Bayes to compute the first **non-vacuous** generalization bound for a neural network trained on MNIST, where every classical bound had been vacuous by orders of magnitude. in this topic you'll build the same machinery for a linear classifier, compute a bound around 0.2 for its stochastic (Gibbs) version on a problem where the VC bound is 0.89.

## warm-up: predict first

1. you see 0 errors on 1,000 independent test examples. Hoeffding says the true error is below $\sqrt{\ln(1/\delta)/2000} = 0.039$ (at $\delta = 0.05$). can you do better, and by roughly how much?
2. PAC-Bayes bounds a *stochastic* classifier: draw fresh weights from a Gaussian around $w$ for every prediction. will that classifier be better or worse than $w$ itself?
3. the KL divergence between $\mathcal N(w, \sigma^2 I)$ and the prior $\mathcal N(0, \sigma^2 I)$ is $\|w\|^2/(2\sigma^2)$. a 785-dimensional weight vector with norm 9.4: is that KL large compared with $n = 60{,}000$?
4. you try 25 prior widths and report the best bound. is that cheating?
5. ridge regression with a fixed regularization per example: how much does its prediction at a test point change when one of $n$ training points is removed?

## prerequisites check

- [ ] you know the KL divergence and the Gaussian KL formula (→ ch 05 · topic 05)
- [ ] you know the finite-class bound and the test-set bound, and why the latter uses a binomial tail (→ topics 01, 04)
- [ ] you know the Rademacher bound for norm-bounded linear classes, $BR/\sqrt n$ (→ topic 03)
- [ ] you can do bisection on a monotone function (→ ch 01, topic 04's `vc_sample_size`)
- [ ] you know logistic regression, ridge regression and the standard normal CDF $\Phi$ (→ ch 04, ch 08)

## core ideas

### idea 1: the binary KL

how surprising is it to see an error rate of $q$ on $n$ independent examples when the true error rate is $p > q$? the **Chernoff bound** for a binomial says

$$P\big(\text{Binomial}(n, p) \le qn\big) \le e^{-n\,\text{kl}(q\,\|\,p)}, \qquad \text{kl}(q\,\|\,p) = q\ln\frac qp + (1 - q)\ln\frac{1 - q}{1 - p},$$

the KL divergence between a coin with bias $q$ and one with bias $p$, in nats. it's the right measure of distance between error rates: it's 0 iff $q = p$, it grows faster than Hoeffding's $2(q - p)^2$ (that's Pinsker's inequality, $\text{kl} \ge 2(q - p)^2$), and crucially it knows that a coin with tiny bias has tiny variance. telling $p = 0.01$ from $q = 0$ is easy (kl $\approx 0.01$), while telling $0.51$ from $0.50$ is hard (kl $\approx 0.0002$), even though the second gap is the same size.

conventions matter at the edges: $0\ln 0 = 0$, so $\text{kl}(0\,\|\,p) = -\ln(1 - p)$; and $\text{kl}(q\,\|\,p) = \infty$ if $p \in \{0, 1\}$ but $q \ne p$ (you can't see an error from a classifier that never errs).

> **spec.** `kl_bernoulli(q, p)` returns $\text{kl}(q\,\|\,p)$ as a float with the conventions above: terms with a zero coefficient vanish, and `math.inf` when a nonzero coefficient meets $p = 0$ or $p = 1$.

> **▶ build: stage 1.** implement `kl_bernoulli`.

### idea 2: inverting it

a bound of the form "$\text{kl}(\hat L\,\|\,L) \le c$" isn't yet a number for $L$. the **kl-inverse** turns it into one (Langford & Seeger, 2001):

$$\text{kl}^{-1}(q, c) = \max\{p \in [q, 1] : \text{kl}(q\,\|\,p) \le c\}.$$

there's no closed form, but $p \mapsto \text{kl}(q\,\|\,p)$ is continuous and increasing on $[q, 1]$, so **bisection** finds it: keep a bracket $[lo, hi]$ with $\text{kl}(q\,\|\,lo) \le c < \text{kl}(q\,\|\,hi)$ and halve it until it's narrower than a tolerance. since the answer is an upper bound on an error rate, return the **upper** end of the bracket: a coarse tolerance then makes the bound slightly looser, never wrong.

two special cases have closed forms you can check against. if $q = 0$, $\text{kl}(0\,\|\,p) = -\ln(1 - p)$, so $\text{kl}^{-1}(0, c) = 1 - e^{-c} \approx c$ (warm-up 1: with zero errors the slack is *linear* in $c = \ln(1/\delta)/n$, giving $0.003$ instead of Hoeffding's $0.039$, thirteen times tighter). and for $q$ near $\frac12$, Pinsker's inequality is nearly tight and $\text{kl}^{-1}(q, c) \approx q + \sqrt{c/2}$, the Hoeffding rate.

> **spec.** `kl_inverse(q, c, *, tol=1e-10)`: return `q` if `c <= 0`, 1.0 if $\text{kl}(q\,\|\,1) \le c$ (only when $q = 1$); otherwise bisect on $[q, 1]$ until the bracket is narrower than `tol` (with a cap on the number of iterations, 200 is plenty) and return its upper end.

> **▶ build: stage 2.** implement `kl_inverse`.

### idea 3: the PAC-Bayes theorem

fix a **prior** $P$, a distribution over hypotheses, *before* seeing the training data. after seeing the data, the learner outputs a **posterior** $Q$, any distribution over hypotheses it likes (it may depend on the data in any way). the **Gibbs classifier** of $Q$ draws a fresh $h \sim Q$ for every prediction; its true and empirical risks are averages over $Q$, $L(Q) = \mathbb E_{h \sim Q}L(h)$ and $\hat L(Q) = \mathbb E_{h \sim Q}\hat L(h)$.

**theorem** (Seeger 2002, with the constant of Maurer 2004; Langford 2005, Theorem 5.1 states the same bound with $\ln\frac{n+1}\delta$ in place of $\ln\frac{2\sqrt n}\delta$). for losses in $[0, 1]$ and any $\delta \in (0, 1)$, with probability at least $1 - \delta$ over the iid sample, **for all posteriors $Q$ simultaneously**,

$$\text{kl}\big(\hat L(Q)\,\big\|\,L(Q)\big) \le \frac{\text{KL}(Q\,\|\,P) + \ln\frac{2\sqrt n}\delta}{n}, \qquad\text{so}\qquad L(Q) \le \text{kl}^{-1}\Big(\hat L(Q),\ \frac{\text{KL}(Q\,\|\,P) + \ln(2\sqrt n/\delta)}n\Big).$$

two ingredients prove it (on paper 2). **change of measure** (the Donsker–Varadhan variational formula): for any function $\phi$, $\mathbb E_Q\phi \le \text{KL}(Q\,\|\,P) + \ln\mathbb E_Pe^\phi$, which converts a statement about the data-independent $P$ into one about every $Q$, at the price of $\text{KL}(Q\,\|\,P)$. and **Maurer's lemma**: for each fixed $h$, $\mathbb E_S\,e^{n\,\text{kl}(\hat L(h)\,\|\,L(h))} \le 2\sqrt n$, a binomial computation (Maurer proves it for $n \ge 8$; it can be checked numerically to hold for every $n \ge 1$, and for tiny samples Langford's $n + 1$ is the safe choice). Markov's inequality on $\mathbb E_P$ of that quantity, and Jensen to move kl outside the average over $Q$ (kl is jointly convex), finish the proof.

read the result as a generalization of everything so far. with a finite class, a uniform prior, and $Q$ a point mass on one hypothesis, $\text{KL}(Q\,\|\,P) = \ln|\mathcal H|$: Occam's bound of topic 04, with the kl-inverse instead of Hoeffding. with $Q = P$, $\text{KL} = 0$ and it's essentially the test-set bound. in general the "complexity" is the information the data added to the prior, measured in nats. applying Pinsker's inequality to the left side gives the square-root **Pinsker relaxation** of the kl bound, the form McAllester (1999) first proved (with slightly different constants) $L(Q) \le \hat L(Q) + \sqrt{(\text{KL} + \ln(2\sqrt n/\delta))/(2n)}$, valid but looser, especially at small $\hat L$ (lab experiment 2).

> **spec.** `pac_bayes_bound(emp_risk, kl, n, delta)` returns `kl_inverse(emp_risk, (kl + ln(2 sqrt(n) / delta)) / n)`. `mcallester_bound(emp_risk, kl, n, delta)` returns `emp_risk + sqrt((kl + ln(2 sqrt(n) / delta)) / (2n))`.

> **🔮 predict:** empirical risk 0.05, KL 10, $n = 1000$, $\delta = 0.05$. the complexity term is $c = 0.0171$. what does McAllester's bound give? the kl bound?

> **▶ build: stage 3.** implement `pac_bayes_bound` and `mcallester_bound`.

### idea 4: a stochastic linear classifier

for linear classifiers $h_v(x) = \text{sign}(v^\top x)$ with labels $y \in \{-1, +1\}$ (absorb a bias into $x$ as a constant feature), take Gaussian posteriors $Q = \mathcal N(w, \sigma^2 I)$ and priors $P = \mathcal N(\mu_P, \sigma_P^2 I)$. both quantities in the bound then have closed forms.

**the Gibbs risk.** on one example, $y\,v^\top x$ with $v = w + \sigma\varepsilon$ is Gaussian with mean $y\,w^\top x$ and standard deviation $\sigma\|x\|$, so

$$P_{v \sim Q}\big(y\,v^\top x < 0\big) = \Phi\Big(-\frac{y\,w^\top x}{\sigma\|x\|}\Big), \qquad \hat L(Q) = \frac1n\sum_i\Phi\Big(-\frac{y_i\,w^\top x_i}{\sigma\|x_i\|}\Big).$$

the argument of $\Phi$ is the example's **normalized margin** divided by $\sigma$: confidently classified examples contribute almost nothing, examples near the boundary contribute almost $\frac12$. so the Gibbs classifier is worse than $w$ (warm-up 2): it errs on everything $w$ gets wrong, and sometimes on examples near the boundary. it can't be *much* worse, and it does say something about $w$: if $w$ errs on $x$, at least half of $Q$'s mass errs there (the Gaussian is symmetric around $w$), so $L(w) \le 2L(Q)$.

**the KL.** for isotropic Gaussians in $d$ dimensions,

$$\text{KL}(Q\,\|\,P) = \frac d2\Big(\frac{\sigma^2}{\sigma_P^2} - 1 - \ln\frac{\sigma^2}{\sigma_P^2}\Big) + \frac{\|w - \mu_P\|^2}{2\sigma_P^2}.$$

with $\sigma = \sigma_P$ the first term vanishes and the KL is $\|w - \mu_P\|^2/(2\sigma^2)$: the squared distance the data moved the weights from the prior, in units of the noise. that's a norm-based complexity, exactly the kind of quantity topic 03 said should matter, and it's here a direct input to the bound (warm-up 3: with $\|w\| = 9.4$ and $\sigma = 0.18$, the KL is about 1,300 nats, a cost of $0.022$ per example at $n = 60{,}000$).

> **spec.** `gibbs_error(w, sigma, X, y)`: `X` (n, d), `y` in $\{-1, +1\}$; return the mean of $\Phi(-y_i w^\top x_i/(\sigma\|x_i\|))$ (`scipy.special.ndtr` is $\Phi$; a row with $\|x\| = 0$ contributes $\frac12$). `isotropic_gaussian_kl(mu_q, sigma_q, mu_p, sigma_p)` returns the formula above.

> **▶ build: stage 4.** implement `gibbs_error` and `isotropic_gaussian_kl`.

### idea 5: a bound for a trained model

now put it together for a weight vector $w$ you trained, say by logistic regression. you choose the posterior family $Q_\sigma = \mathcal N(w, \sigma^2 I)$; small $\sigma$ keeps the Gibbs risk close to $w$'s but makes the KL large, large $\sigma$ does the opposite. the best $\sigma$ depends on the data, and you may not choose the prior after looking at the data... but you may choose among a **finite grid of priors fixed in advance**, at the price of a union bound: with $k$ candidate priors, run each bound at $\delta/k$ (warm-up 4: not cheating, if you pay for it; with $k = 25$ it costs $\ln 25/n$, nothing).

the prior's *mean* is the other lever. a prior centred at 0 charges for $\|w\|^2$. a prior centred near where the learner will end up charges much less, and it's legal as long as the prior doesn't depend on the data the bound is computed on. the standard trick (Ambroladze, Parrado-Hernández & Shawe-Taylor 2007; Dziugaite & Roy 2017 took it further) is to split the training set: train $w_0$ on one part, use $\mathcal N(w_0, \sigma^2 I)$ as the prior, and compute the bound on the other part, with $n$ the size of that part. the posterior mean $w$ may use all the data.

> **spec.** `pac_bayes_linear(w, X, y, *, sigmas, prior_sigma=None, prior_mean=None, delta=0.05)`: for each $s$ in `sigmas`, posterior $\mathcal N(w, s^2 I)$, prior $\mathcal N(\mu_P, \sigma_P^2 I)$ with $\mu_P$ = `prior_mean` (zeros if `None`) and $\sigma_P$ = `prior_sigma` (if `None`, the same $s$); compute the Gibbs risk on `(X, y)`, the KL, and `pac_bayes_bound(risk, kl, n, delta / len(sigmas))`. return a dict with the best `sigma`, its `emp_risk`, `kl`, `bound`, and the arrays `bounds`, `emp_risks`, `kls` over the grid.

> **▶ build: stage 5.** implement `pac_bayes_linear`.

### idea 6: margins and stability

**margins.** for linear classifiers with $\|w\| \le B$ on data with $\|x\| \le R$, topic 03 bounded the Rademacher complexity of the real-valued outputs by $BR/\sqrt n$. to use it for classification, compare outputs with a margin $\gamma > 0$ through the **ramp loss** $\phi_\gamma(u) = \min(1, \max(0, 1 - u/\gamma))$: 1 for $u \le 0$ (an error), 0 for $u \ge \gamma$ (confidently right), linear in between. it upper-bounds the 0-1 loss and is $1/\gamma$-Lipschitz, so (Talagrand's contraction lemma) its class has Rademacher complexity at most $BR/(\gamma\sqrt n)$. for a fixed $\gamma$, with probability at least $1 - \delta$ (Mohri et al. 2018, Theorem 5.8 with the linear-class bound),

$$L_{0\text{-}1}(w) \le \frac1n\sum_i\phi_\gamma(y_iw^\top x_i) + \frac{2BR}{\gamma\sqrt n} + \sqrt{\frac{\ln(1/\delta)}{2n}}.$$

no dimension appears: only the ratio $BR/\gamma$, the inverse of the normalized margin. a classifier that separates the data with a large margin relative to its scale generalizes, whatever $d$ is. this is the theory behind SVMs (chapter 09) and behind the Gibbs risk's normalized margins above.

**stability.** a learning algorithm $A$ has **uniform stability** $\beta$ if removing any one training point changes the loss of its output on any example by at most $\beta$. Bousquet & Elisseeff (2002, Theorem 12): for a loss bounded by $M$, with probability at least $1 - \delta$,

$$L(A_S) \le \hat L(A_S) + 2\beta + (4n\beta + M)\sqrt{\frac{\ln(1/\delta)}{2n}}.$$

it says nothing about a hypothesis class at all, only about the algorithm, and it's useful only when $\beta = o(1/\sqrt n)$. regularized ERM qualifies: for ridge-type problems with penalty $\lambda\|w\|^2$ per example and Lipschitz losses on bounded data, strong convexity gives $\beta = O(1/(\lambda n))$ (warm-up 5). you can't compute a supremum over every training set, but you can **measure** a lower estimate: retrain without a chosen point and record the largest change in prediction. random removals estimate it; the supremum may be attained by specific points, which random removals can miss (the lab shows a case where that matters a lot).

> **spec.** `linear_margin_bound(margins, gamma, B, R, delta)`: `margins` are $y_iw^\top x_i$ (n,); return the mean ramp loss plus $2BR/(\gamma\sqrt n) + \sqrt{\ln(1/\delta)/(2n)}$. `loo_stability(fit_predict, X, y, X_eval, *, n_remove=20, indices=None, rng=None)`: `fit_predict(X, y, X_eval)` trains and returns predictions on `X_eval`; fit once on all the data, then once without each removed point (the given `indices`, or `n_remove` distinct random ones from `as_rng(rng)`), and return the largest absolute change in any prediction.

> **🔮 predict:** with a fixed regularization per example, how does ridge's leave-one-out change scale with $n$? and 1-nearest-neighbour's?

> **▶ build: stage 6.** implement `linear_margin_bound` and `loo_stability`.

## worked example

**kl and its inverse.** $\text{kl}(0.1\,\|\,0.2) = 0.1\ln 0.5 + 0.9\ln 1.125 = -0.0693 + 0.1060 = 0.0367$. $\text{kl}^{-1}(0, 0.5) = 1 - e^{-0.5} = 0.3935$.

**a PAC-Bayes bound.** empirical Gibbs risk 0.05, KL 10, $n = 1000$, $\delta = 0.05$: $\ln(2\sqrt{1000}/0.05) = \ln 1264.9 = 7.143$, so $c = 17.143/1000 = 0.01714$. McAllester: $0.05 + \sqrt{0.00857} = 0.1426$. kl-inverse: $0.1008$, the $p$ with $0.05\ln\frac{0.05}p + 0.95\ln\frac{0.95}{1 - p} = 0.01714$. the kl form is 30% tighter.

**Occam from PAC-Bayes.** a finite class of 1,000 classifiers, uniform prior, point-mass posterior on one with zero training errors, $n = 1000$, $\delta = 0.05$: $\text{KL} = \ln 1000 = 6.908$, $c = (6.908 + 7.143)/1000 = 0.01405$, and the bound is $\text{kl}^{-1}(0, 0.01405) = 1 - e^{-0.01405} = 0.01395$. (topic 04's Hoeffding-based finite-class bound gives $\sqrt{9.90/2000} = 0.0704$, five times looser.)

**one Gibbs prediction.** $x = (3, 4)$, $y = +1$, $w = (1, 0)$, $\sigma = 1$: the normalized margin is $3/5 = 0.6$, so the Gibbs classifier errs with probability $\Phi(-0.6) = 0.274$, though $w$ is right.

**a margin bound.** margins $(2, 1, 0.5, -0.5, 3, 0.25, 1.5, 0.75)$, $\gamma = 1$: ramp losses $(0, 0, 0.5, 1, 0, 0.75, 0, 0.25)$, mean $0.3125$. with $B = 2$, $R = 1.5$, $\delta = 0.1$, $n = 8$: $0.3125 + 6/\sqrt8 + \sqrt{\ln 10/16} = 0.3125 + 2.121 + 0.379 = 2.81$.

## on paper

1. **the Chernoff bound behind the kl.** for $k \sim \text{Binomial}(n, p)$ and $q < p$, show $P(k \le qn) \le e^{-n\,\text{kl}(q\,\|\,p)}$. hint: $P(k \le qn) \le e^{\lambda qn}\mathbb Ee^{-\lambda k}$ for $\lambda > 0$; optimize over $\lambda$.

<details><summary>solution</summary>

$\mathbb Ee^{-\lambda k} = (1 - p + pe^{-\lambda})^n$, so $P(k \le qn) \le \exp\big(n[\lambda q + \ln(1 - p + pe^{-\lambda})]\big)$. setting the derivative to 0 gives $e^{-\lambda} = \frac{q(1 - p)}{p(1 - q)}$ (positive $\lambda$ since $q < p$). substituting, $\lambda q + \ln(1 - p + pe^{-\lambda}) = q\ln\frac{p(1 - q)}{q(1 - p)} + \ln\frac{1 - p}{1 - q} = -\text{kl}(q\,\|\,p)$.

</details>

2. **change of measure.** prove $\mathbb E_Q\phi \le \text{KL}(Q\,\|\,P) + \ln\mathbb E_Pe^\phi$ for any measurable $\phi$, and explain how it yields "for all $Q$ simultaneously".

<details><summary>solution</summary>

define the Gibbs distribution $G \propto Pe^\phi$, i.e. $\frac{dG}{dP} = e^\phi/\mathbb E_Pe^\phi$. then $0 \le \text{KL}(Q\,\|\,G) = \mathbb E_Q\ln\frac{dQ}{dP} - \mathbb E_Q\phi + \ln\mathbb E_Pe^\phi = \text{KL}(Q\,\|\,P) - \mathbb E_Q\phi + \ln\mathbb E_Pe^\phi$. with $\phi(h) = n\,\text{kl}(\hat L(h)\,\|\,L(h))$, the right side involves $Q$ only through the KL; the random part $\ln\mathbb E_Pe^\phi$ doesn't depend on $Q$ at all, so one high-probability bound on it (Markov's inequality plus Maurer's $\mathbb E_S\mathbb E_Pe^\phi \le 2\sqrt n$) covers every $Q$ at once.

</details>

3. **the Gibbs risk.** derive $P_{v \sim \mathcal N(w, \sigma^2 I)}(y\,v^\top x < 0) = \Phi(-y\,w^\top x/(\sigma\|x\|))$, and show that $L(w) \le 2L(Q)$.

<details><summary>solution</summary>

$y\,v^\top x = y\,w^\top x + \sigma y\,\varepsilon^\top x$ and $y\,\varepsilon^\top x \sim \mathcal N(0, \|x\|^2)$, so $y\,v^\top x \sim \mathcal N(y\,w^\top x, \sigma^2\|x\|^2)$ and $P(\cdot < 0) = \Phi(-y\,w^\top x/(\sigma\|x\|))$. if $w$ errs on $x$, then $y\,w^\top x \le 0$ and this probability is at least $\Phi(0) = \frac12$. so $\mathbb 1[w \text{ errs}] \le 2P_Q(\text{err})$ pointwise; take expectations over $x$.

</details>

4. **ridge is stable.** let $F_S(w) = \frac1n\sum_i(w^\top x_i - y_i)^2 + \lambda\|w\|^2$ with $\|x_i\| \le R$ and residuals bounded by $M$, and use Bousquet & Elisseeff's convention for a removal: $S^{\setminus j}$ keeps the weights $\frac1n$ on the remaining points (equivalently, the $n - 1$ points with weights $\frac1{n-1}$ and penalty $\lambda\frac{n}{n-1}$). show that removing one point moves the minimizer by at most $RM/(\lambda n)$.

<details><summary>solution</summary>

both objectives are $2\lambda$-strongly convex. with the convention above, $F_S - F_{S'}$ is exactly the single term $\frac1n(w^\top x_j - y_j)^2$, whose gradient has norm at most $\frac{2RM}n$ near the two solutions. strong convexity gives $2\lambda\|w_S - w_{S'}\|^2 \le (\nabla F_S(w_{S'}) - \nabla F_S(w_S))^\top(w_{S'} - w_S) = \nabla(F_S - F_{S'})(w_{S'})^\top(w_{S'} - w_S) \le \frac{2RM}n\|w_S - w_{S'}\|$, so $\|w_S - w_{S'}\| \le \frac{RM}{\lambda n}$, and predictions move by at most $R$ times that. (if you instead refit with weights $\frac1{n-1}$ and the same $\lambda$, the effective penalty changes by a factor $\frac{n-1}n$, which moves the minimizer by a further $O(\|w\|/n)$: the same $O(1/n)$ order.)

</details>

## experiment

`lab.py` investigates, with your code:

1. a PAC-Bayes bound for logistic regression on MNIST odd vs even, with a prior at 0 and a data-dependent prior, next to the VC bound, the test-set bound and the measured errors.
2. how the kl-inverse compares with McAllester's square-root bound across empirical risks.
3. the leave-one-out stability of ridge regression and 1-NN on California housing, as $n$ grows.

## check your understanding

1. why must the prior be chosen before seeing the data, while the posterior can depend on it arbitrarily?

<details><summary>answer</summary>

the proof bounds $\mathbb E_P e^{\phi}$ with Markov's inequality over the random sample, which requires $P$ to be fixed independently of that sample. the posterior enters only through the change-of-measure inequality, which holds for every $Q$ deterministically. a data-dependent prior must be built from data the bound doesn't use (sample splitting), or be one of a finite set fixed in advance (union bound).

</details>

2. how does the PAC-Bayes bound recover the finite-class (Occam) bound?

<details><summary>answer</summary>

uniform prior over $|\mathcal H|$ hypotheses, posterior a point mass on the chosen $h$: $\text{KL}(\delta_h\,\|\,P) = \ln\frac{1}{1/|\mathcal H|} = \ln|\mathcal H|$, and the Gibbs classifier is $h$ itself. more generally, a non-uniform prior gives $\ln(1/P(h))$: hypotheses you thought likely are cheaper, which is Occam's razor with a description-length prior.

</details>

3. why is the PAC-Bayes bound non-vacuous for the MNIST linear model while the VC bound isn't?

<details><summary>answer</summary>

the VC bound pays for all 785 dimensions at full strength, with a factor 8 and a $\ln n$. the PAC-Bayes bound pays $\|w - \mu_P\|^2/(2\sigma^2)$: the weights moved a moderate distance relative to how much noise the classifier tolerates, which is a statement about margins on *this* data. it also benefits from the kl-inverse at small error rates.

</details>

4. the bound is for the Gibbs classifier. why not just bound the deterministic $w$?

<details><summary>answer</summary>

the theorem needs a posterior with finite KL to the prior, and a point mass on a continuous weight vector has infinite KL to any Gaussian prior. the Gaussian posterior is the price, and $L(w) \le 2L(Q)$ (or tighter margin-based arguments) converts back.

</details>

5. why is uniform stability useless if $\beta$ only shrinks like $1/\sqrt n$?

<details><summary>answer</summary>

the bound contains $4n\beta\sqrt{\ln(1/\delta)/(2n)} = 4\beta\sqrt{n\ln(1/\delta)/2}$. with $\beta \propto 1/\sqrt n$ that's a constant: the bound doesn't go to 0. with $\beta \propto 1/n$ it decays like $1/\sqrt n$, the usual rate.

</details>

## common bugs

- **`math.log(0)` in the binary KL.** symptom: `ValueError: math domain error` at $q = 0$ or $q = 1$. skip terms with zero coefficient.
- **bisecting on $[0, 1]$ instead of $[q, 1]$.** symptom: an "upper bound" below the empirical risk (there are two solutions of $\text{kl} = c$, one on each side of $q$).
- **returning the lower end of the bracket.** symptom: bounds that are a hair too small when `tol` is coarse.
- **$\Phi$ of the unnormalized margin.** symptom: a Gibbs risk that depends on the scale of $x$; divide by $\sigma\|x\|$.
- **forgetting the union bound over the $\sigma$ grid.** symptom: a bound that's slightly too optimistic; use $\delta/k$.
- **computing the bound on the data that built the prior.** symptom: a bound that's too good to be true, and invalid.
- **stability from too few random removals.** symptom: a model that looks perfectly stable at large $n$ because no removed point mattered (lab experiment 3).

## going deeper

- McAllester (1999), "PAC-Bayesian model averaging", *COLT*. [doi:10.1145/307400.307435](https://doi.org/10.1145/307400.307435)
- Langford & Seeger (2001), "Bounds for averaging classifiers", CMU technical report CMU-CS-01-102. the kl form and its inversion. [cs.cmu.edu/~jcl/papers/averaging/averaging_tech.pdf](https://www.cs.cmu.edu/~jcl/papers/averaging/averaging_tech.pdf)
- Seeger (2002), "PAC-Bayesian generalisation error bounds for Gaussian process classification", *JMLR* 3. [jmlr.org/papers/v3/seeger02a](https://www.jmlr.org/papers/v3/seeger02a.html)
- Maurer (2004), "A note on the PAC Bayesian theorem". the $2\sqrt n$ constant. [arXiv:cs/0411099](https://arxiv.org/abs/cs/0411099)
- Langford & Shawe-Taylor (2002), "PAC-Bayes & margins", *NeurIPS*. Gaussian posteriors for linear classifiers. [papers.nips.cc/paper/2317](https://papers.nips.cc/paper/2317-pac-bayes-margins)
- Bousquet & Elisseeff (2002), "Stability and generalization", *JMLR* 2. [jmlr.org/papers/v2/bousquet02a](https://www.jmlr.org/papers/v2/bousquet02a.html)
- Dziugaite & Roy (2017), "Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data", *UAI*. [arXiv:1703.11008](https://arxiv.org/abs/1703.11008)
- Parrado-Hernández, Ambroladze, Shawe-Taylor & Sun (2012), "PAC-Bayes bounds with data dependent priors", *JMLR* 13. [jmlr.org/papers/v13/parrado12a](https://www.jmlr.org/papers/v13/parrado12a.html)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., ch. 5 (margin theory) and ch. 14 (algorithmic stability). [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)

## connections

- ← builds on: KL divergence and Gaussians (ch 05), the finite-class and test-set bounds (topics 01, 04), Rademacher complexity of norm balls (topic 03), logistic and ridge regression (ch 08), SVM margins (ch 09).
- → used in: the capstone's generalization report, margins in boosting (ch 10 · topic 03), weight decay, dropout and early stopping as stabilizers (ch 15 · topic 03), flat minima and sharpness, which PAC-Bayes ties to weight noise (ch 19 · topic 02), and generalization in deep learning (ch 19 · topic 04).
