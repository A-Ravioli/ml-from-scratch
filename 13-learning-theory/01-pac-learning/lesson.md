# PAC learning

## why this matters

you fit a classifier on 100 labelled examples and it gets every one of them right. a colleague asks the obvious question: how wrong will it be on the next thousand? "zero training errors" can't be the answer by itself. a lookup table that memorizes the 100 examples also gets zero training errors, and on new points it's no better than a coin. what separates a classifier that *learned* from one that *memorized* has to be something about the set of classifiers you were choosing from, and about how many examples you had.

chapter 07 answered the question empirically: hold out data and measure. that works, and it's what you'll do in practice. this chapter asks for something stronger, a guarantee *before* you look at a test set: "with 100 examples, from this family of classifiers, the error is below 10% with probability at least 95%". that kind of statement is a **PAC guarantee** (probably approximately correct, Valiant 1984), and the question "how many examples do I need for one?" is the **sample complexity** of a hypothesis class.

this topic builds the first and simplest version: classes with finitely many hypotheses, where one union bound does all the work. then it confronts the result with the simplest *infinite* class, thresholds on a line, where you can compute the failure probability exactly and measure it by simulation. the comparison already shows the chapter's theme: the bounds are true, they have the right shape, and they can still be off by large factors.

## warm-up: predict first

1. a finite class has a million hypotheses. compared with a class of a thousand, how many more examples does the realizable bound ask for: 1,000 times as many, twice as many, or a few more?
2. thresholds on $[0, 1]$, $h_t(x) = \mathbb 1[x \ge t]$, form an *infinite* class. can a finite number of examples still give a PAC guarantee for them?
3. halving the error tolerance $\varepsilon$ doubles the realizable sample complexity. what does it do in the agnostic (noisy) case?
4. you simulate a learner 1,000 times and it fails 40 times. is its failure probability below $\delta = 0.05$?
5. for a grid of 1,000 thresholds with 10% label noise, the agnostic bound asks for about 2,100 examples at $\varepsilon = \delta = 0.05$. do you think a simulation will agree?

## prerequisites check

- [ ] you know the union bound $P(\bigcup_i A_i) \le \sum_i P(A_i)$ and the inequality $1 - x \le e^{-x}$ (→ ch 01, ch 04 · topic 01)
- [ ] you know Hoeffding's inequality for averages of bounded independent variables (→ ch 05 · topic 01)
- [ ] you know empirical risk, true risk, and why the training error is optimistic (→ ch 07 · topic 01)
- [ ] you can estimate a probability by Monte Carlo and give its standard error $\sqrt{p(1-p)/N}$ (→ ch 04 · topic 05)

## core ideas

### idea 1: the PAC framework and finite classes

here is the setup every topic in this chapter uses. examples $x$ come from an unknown distribution $\mathcal D$ over a domain $\mathcal X$, labels from an unknown rule. you get a sample $S = ((x_1, y_1), \dots, (x_m, y_m))$ of $m$ examples drawn **independently** from the same distribution (iid; everything below breaks without it). a **hypothesis class** $\mathcal H$ is the set of classifiers $h: \mathcal X \to \{0, 1\}$ your learner chooses from. the **true risk** of $h$ is its error rate on fresh data, $L_{\mathcal D}(h) = P_{(x, y) \sim \mathcal D}(h(x) \ne y)$, and its **empirical risk** is its error rate on the sample, $L_S(h) = \frac1m\sum_i \mathbb 1[h(x_i) \ne y_i]$. **empirical risk minimization** (ERM) returns a hypothesis with the smallest $L_S$.

in the **realizable** case some $h^\star \in \mathcal H$ labels everything correctly, $L_{\mathcal D}(h^\star) = 0$. a learner is $(\varepsilon, \delta)$-**PAC** with $m$ examples if, for every distribution and every target in the class,

$$P_{S \sim \mathcal D^m}\big(L_{\mathcal D}(h_S) > \varepsilon\big) \le \delta.$$

two tolerances: $\varepsilon$ is "approximately correct", $\delta$ is "probably". the probability is over the random sample; the guarantee holds for the worst distribution.

**finite classes are PAC learnable by ERM.** call $h$ *bad* if $L_{\mathcal D}(h) > \varepsilon$. a single bad hypothesis survives (makes no mistakes on) one random example with probability less than $1 - \varepsilon$, and on $m$ independent examples with probability less than $(1 - \varepsilon)^m \le e^{-\varepsilon m}$. ERM in the realizable case always returns a hypothesis with $L_S = 0$, so it can only fail if some bad hypothesis survives. by the union bound over at most $|\mathcal H|$ bad hypotheses,

$$P\big(L_{\mathcal D}(h_S) > \varepsilon\big) \le |\mathcal H|\,e^{-\varepsilon m} \le \delta \quad\text{whenever}\quad m \ge \frac{\ln(|\mathcal H|/\delta)}{\varepsilon}.$$

read the formula slowly, because the rest of the chapter refines it. the class size enters through its **logarithm** (warm-up 1: a million hypotheses instead of a thousand costs $\ln 1000/\varepsilon$ more examples, not a thousand times as many). the confidence enters through $\ln(1/\delta)$, so very high confidence is cheap. the tolerance enters as $1/\varepsilon$.

in the **agnostic** case nothing in the class is perfect (labels are noisy, or the truth isn't in $\mathcal H$), and you compare with the best in class instead. Hoeffding's inequality says each fixed $h$ satisfies $P(|L_S(h) - L_{\mathcal D}(h)| > \varepsilon) \le 2e^{-2m\varepsilon^2}$; a union bound makes this hold for every $h \in \mathcal H$ simultaneously (**uniform convergence**) as soon as

$$m \ge \frac{\ln(2|\mathcal H|/\delta)}{2\varepsilon^2}.$$

then ERM's hypothesis $\hat h$ satisfies $L_{\mathcal D}(\hat h) \le L_S(\hat h) + \varepsilon \le L_S(h^\star) + \varepsilon \le L_{\mathcal D}(h^\star) + 2\varepsilon$: within $2\varepsilon$ of the best (on paper 2). the price of noise is the square: $1/\varepsilon^2$ instead of $1/\varepsilon$ (warm-up 3), because you're now estimating error rates near some $p > 0$, where the variance is $p(1 - p)/m$, instead of checking that they're zero. this is the uniform-convergence sample complexity of Shalev-Shwartz & Ben-David (2014), Corollary 4.6; the $\varepsilon$-agnostic-PAC version replaces $\varepsilon$ by $\varepsilon/2$.

> **spec.** `finite_class_sample_complexity(class_size, epsilon, delta, *, realizable=True)` returns the smallest integer $m$ satisfying the bound above: $\lceil \ln(|\mathcal H|/\delta)/\varepsilon\rceil$ (realizable) or $\lceil\ln(2|\mathcal H|/\delta)/(2\varepsilon^2)\rceil$ (agnostic), as a Python `int`. `class_size` may be a huge Python integer (a float32 model is a finite class with $2^{32 \times \text{params}}$ members), so take logarithms with `math.log` and use $\ln(|\mathcal H|/\delta) = \ln|\mathcal H| + \ln(1/\delta)$. raise `ValueError` unless $0 < \varepsilon < 1$ and $0 < \delta < 1$.

> **▶ build: stage 1.** implement `finite_class_sample_complexity`.

### idea 2: ERM over thresholds

the rest of the topic studies one class in detail: **thresholds** on the line, $h_t(x) = \mathbb 1[x \ge t]$ for $t \in \mathbb R \cup \{+\infty\}$ ($t = +\infty$ predicts 0 everywhere). they're the decision stumps of chapter 10 on a single feature, and simple enough that every probability is computable by hand.

the class is infinite, but on a sample only finitely many of its members behave differently. moving $t$ between two consecutive sample points changes no prediction, so it's enough to try $t$ at each distinct sample value and at $+\infty$. a candidate $t$ makes two kinds of mistakes: positives below it (missed) and negatives at or above it (false alarms),

$$\text{errors}(t) = \#\{i : y_i = 1,\ x_i < t\} + \#\{i : y_i = 0,\ x_i \ge t\}.$$

trying every candidate with a loop over the sample costs $O(m^2)$; sorting the positives and the negatives once and counting with binary search (`np.searchsorted`) costs $O(m\log m)$. in the realizable case the minimizers are exactly the consistent thresholds, and the smallest candidate with zero error is the smallest positive example: ERM "hugs" the positives.

> **spec.** `erm_threshold(x, y)`: `x` (n,) floats, `y` (n,) labels in {0, 1} (or bools). candidates are the distinct values of `x` and `np.inf`; return the candidate with the fewest training errors as a `float`, the **smallest** one if several tie. with no positive examples, that's `inf`.

> **▶ build: stage 2.** implement `erm_threshold`.

### idea 3: an infinite class with an exact answer

let $x \sim \text{Uniform}[0, 1]$, labels $y = \mathbb 1[x \ge t^\star]$, optionally flipped independently with probability $\eta < 1/2$ (label noise). two thresholds disagree exactly on the interval between them, so the risk of $h_t$ is

$$L(t) = \eta + (1 - 2\eta)\,\big|\bar t - \bar t^\star\big|,$$

where $\bar t$ is $t$ clipped to $[0, 1]$: on the disagreement interval you're wrong unless the label was flipped, elsewhere only when it was. the best threshold is $t^\star$ itself, with risk $\eta$, so the **excess risk** of $h_t$ is $(1 - 2\eta)|\bar t - \bar t^\star|$.

now the realizable case ($\eta = 0$). ERM returns the smallest positive example, which is always $\ge t^\star$, so its risk is $\hat t - t^\star$. that exceeds $\varepsilon$ exactly when no example lands in $[t^\star, t^\star + \varepsilon)$, an interval of probability $\varepsilon$ (if $t^\star + \varepsilon < 1$). so

$$P\big(L(\hat t) > \varepsilon\big) = (1 - \varepsilon)^m \le e^{-\varepsilon m}, \qquad m \ge \frac{\ln(1/\delta)}{\varepsilon}\ \text{suffices}.$$

compare with idea 1: this is the finite-class bound with $|\mathcal H| = 1$ (warm-up 2). an infinite class can need *fewer* examples than a finite one, because what matters isn't how many hypotheses there are but how many ways they can label a sample. topic 02 turns that observation into the VC dimension.

if $t^\star + \varepsilon \ge 1$, failure is impossible: even $\hat t = +\infty$ errs only on $[t^\star, 1]$, of mass at most $\varepsilon$.

> **spec.** `threshold_risk(t, t_star, noise=0.0)` returns $L(t)$ above (clip both thresholds to $[0, 1]$; `t` may be `inf`). `threshold_failure_probability(m, epsilon, t_star=0.5)` returns the exact $P(L(\hat t) > \varepsilon)$ for your noiseless ERM: $(1 - \varepsilon)^m$, or 0 when $t^\star + \varepsilon \ge 1$.

> **🔮 predict:** with $\varepsilon = 0.1$ and $\delta = 0.05$, the bound $\ln(1/\delta)/\varepsilon$ gives $m = 30$. what's the exact smallest $m$ with $(1 - \varepsilon)^m \le \delta$? how about a finite grid of 1,000 thresholds, using idea 1's bound?

> **▶ build: stage 3.** implement `threshold_risk` and `threshold_failure_probability`.

### idea 4: measuring a failure probability

for most learners and distributions there's no closed form, but you can always **simulate**: draw a sample of size $m$, run the learner, compute (or estimate) the risk of what it returned, and record whether it's more than $\varepsilon$ above the best. repeat $N$ times; the fraction of failures $\hat p$ estimates the failure probability $p$.

$\hat p$ is a binomial proportion, so its standard error is $\sqrt{p(1 - p)/N}$. that's the honest limit of what simulation can tell you about a small $\delta$: with $N = 1000$ trials and $p = 0.05$ the standard error is 0.007, so 40 failures in 1,000 (warm-up 4) is consistent with $p = 0.05$ and with $p = 0.03$. to certify $p < \delta$ you need $N$ well above $1/\delta$; for $\delta = 10^{-6}$ simulation is hopeless and only a bound will do.

with label noise, "failure" means **excess** risk above $\varepsilon$: $L(\hat t) - L(t^\star) > \varepsilon$. comparing the raw risk with $\varepsilon$ would count every trial as a failure as soon as $\eta > \varepsilon$.

> **spec.** `estimate_failure_probability(m, epsilon, *, t_star=0.5, noise=0.0, n_trials=1000, rng=None)`: with `gen = as_rng(rng)`, repeat `n_trials` times: draw `x = gen.random(m)`, labels `(x >= t_star)` flipped where `gen.random(m) < noise`, fit `erm_threshold`, and count a failure when `threshold_risk(t_hat) - threshold_risk(t_star) > epsilon` (same `noise` in both). return failures / n_trials.

> **▶ build: stage 4.** implement `estimate_failure_probability`.

### idea 5: sample complexity, measured

the **sample complexity** $m(\varepsilon, \delta)$ is the smallest $m$ for which the failure probability is at most $\delta$. a theorem gives an upper bound on it; a simulation estimates it directly. scan sample sizes in increasing order and stop at the first one whose estimated failure probability is at most $\delta$. (the estimate is noisy near the answer, so the result is itself a little random; scanning a grid rather than bisecting keeps it simple and predictable.)

this is the experiment that keeps learning theory honest. the realizable bound for thresholds is nearly exact, as you'll see. the agnostic bound is a worst case over *all* distributions with the given noise, and the agnostic rate $1/\varepsilon^2$ comes from distributions where many hypotheses have error rates very close to the best one. whether your distribution is one of them is exactly what the lab measures (warm-up 5).

> **spec.** `empirical_sample_complexity(epsilon, delta, m_grid, *, t_star=0.5, noise=0.0, n_trials=1000, rng=None)`: one generator `as_rng(rng)` shared by every call; go through `sorted(m_grid)` and return the first `m` (as an `int`) with `estimate_failure_probability(m, ...) <= delta`, or `None` if there's none.

> **🔮 predict:** with 10% label noise, $\varepsilon = 0.05$, $\delta = 0.05$, how many examples does ERM over thresholds really need? the agnostic bound for a grid of 1,000 thresholds says about 2,100.

> **▶ build: stage 5.** implement `empirical_sample_complexity`.

## worked example

**the calculator.** $|\mathcal H| = 1000$, $\varepsilon = 0.1$, $\delta = 0.05$. realizable: $\ln(1000/0.05) = \ln 20000 = 9.903$, divided by $0.1$ is $99.03$, so $m = 100$. agnostic: $\ln(2000/0.05) = \ln 40000 = 10.597$, divided by $2 \times 0.01$ is $529.8$, so $m = 530$. a float32 linear model on 784 pixels plus a bias is a finite class of $2^{32 \times 785}$ hypotheses: $\ln|\mathcal H| = 25120\ln 2 = 17411.9$, and the realizable bound with $\varepsilon = \delta = 0.05$ asks for $\lceil(17411.9 + 3.0)/0.05\rceil = 348{,}298$ examples. your stage 1 must reproduce all three.

**ERM by hand.** $x = (0.9, 0.1, 0.4, 0.7, 0.35)$, $y = (1, 0, 1, 1, 0)$. sorted: negatives at 0.1, 0.35; positives at 0.4, 0.7, 0.9. candidate $t = 0.35$ calls 0.35 positive (1 error); $t = 0.4$ makes none; $t = 0.7$ misses 0.4 (1 error). ERM returns $0.4$.

**the exact failure probability.** $m = 20$, $\varepsilon = 0.1$, $t^\star = 0.5$: $0.9^{20} = 0.1216$, while the bound $e^{-2} = 0.1353$ is 11% larger. for $\delta = 0.05$: the bound asks for $\lceil\ln 20/0.1\rceil = 30$ examples, the exact answer is $\lceil\ln 0.05/\ln 0.9\rceil = \lceil 28.43\rceil = 29$.

## on paper

1. **the realizable bound.** prove that a consistent learner over a finite class is $(\varepsilon, \delta)$-PAC with $m \ge \ln(|\mathcal H|/\delta)/\varepsilon$ examples. where exactly did you use independence?

<details><summary>solution</summary>

let $\mathcal H_{\text{bad}} = \{h : L_{\mathcal D}(h) > \varepsilon\}$. the learner fails only if some $h \in \mathcal H_{\text{bad}}$ has $L_S(h) = 0$. for a fixed bad $h$, $P(h(x_i) = y_i) < 1 - \varepsilon$ for each $i$, and **because the examples are independent**, $P(L_S(h) = 0) = \prod_i P(h(x_i) = y_i) < (1 - \varepsilon)^m \le e^{-\varepsilon m}$. the union bound over at most $|\mathcal H|$ bad hypotheses gives failure probability $< |\mathcal H|e^{-\varepsilon m}$, which is $\le \delta$ iff $m \ge \ln(|\mathcal H|/\delta)/\varepsilon$.

</details>

2. **the agnostic bound.** assume $|L_S(h) - L_{\mathcal D}(h)| \le \varepsilon$ for every $h \in \mathcal H$. show that ERM's $\hat h$ satisfies $L_{\mathcal D}(\hat h) \le \min_h L_{\mathcal D}(h) + 2\varepsilon$, and derive the sample size that makes the assumption hold with probability $1 - \delta$.

<details><summary>solution</summary>

let $h^\star$ minimize $L_{\mathcal D}$. then $L_{\mathcal D}(\hat h) \le L_S(\hat h) + \varepsilon \le L_S(h^\star) + \varepsilon \le L_{\mathcal D}(h^\star) + 2\varepsilon$; the middle step is the definition of ERM. Hoeffding for the $m$ iid losses in $[0, 1]$ of a fixed $h$: $P(|L_S(h) - L_{\mathcal D}(h)| > \varepsilon) \le 2e^{-2m\varepsilon^2}$. the union bound over $\mathcal H$ makes the total at most $2|\mathcal H|e^{-2m\varepsilon^2} \le \delta$ iff $m \ge \ln(2|\mathcal H|/\delta)/(2\varepsilon^2)$.

</details>

3. **exact versus bound for thresholds.** solve $(1 - \varepsilon)^m \le \delta$ for $m$ and compare with $\ln(1/\delta)/\varepsilon$. how large is the ratio for $\varepsilon = 0.1$? for $\varepsilon = 0.01$?

<details><summary>solution</summary>

$m \ge \ln(1/\delta)/(-\ln(1 - \varepsilon))$. since $-\ln(1 - \varepsilon) = \varepsilon + \varepsilon^2/2 + \dots$, the ratio of bound to exact is $-\ln(1 - \varepsilon)/\varepsilon \approx 1 + \varepsilon/2$: about 1.054 at $\varepsilon = 0.1$, 1.005 at $\varepsilon = 0.01$. for thresholds the realizable bound is essentially tight.

</details>

4. **how many trials?** you want to tell a failure probability of 0.05 from one of 0.04 by simulation, with the two means separated by at least 4 standard errors. how many trials do you need?

<details><summary>solution</summary>

the standard error at $p \approx 0.045$ is $\sqrt{0.045 \times 0.955/N}$; asking $0.01 \ge 4\sqrt{0.043/N}$ gives $N \ge 16 \times 0.043/10^{-4} \approx 6{,}900$. resolving small differences in small probabilities is expensive, which is why the lab uses thousands of trials per sample size.

</details>

## experiment

`lab.py` investigates, with your code:

1. for noiseless thresholds, how the sample complexity measured by simulation compares with the exact answer, the $\ln(1/\delta)/\varepsilon$ bound, and the finite-class bound for a grid of thresholds, across five values of $\varepsilon$.
2. what label noise does to the measured sample complexity, compared with the agnostic bound, and how it scales with $\varepsilon$.

## check your understanding

1. the PAC guarantee holds "for every distribution". why is that both the strength and the weakness of the framework?

<details><summary>answer</summary>

strength: you don't need to know anything about $\mathcal D$ for the guarantee to hold. weakness: the bound has to cover the worst distribution, so for the distribution you actually have it can be very pessimistic. the lab's noisy experiment shows this directly.

</details>

2. why does $|\mathcal H|$ enter through its logarithm? connect it to bits.

<details><summary>answer</summary>

the union bound multiplies the failure probability by $|\mathcal H|$, and the per-hypothesis failure probability decays exponentially in $m$, so $m$ only needs to grow by $\ln|\mathcal H|/\varepsilon$. $\log_2|\mathcal H|$ is the number of bits needed to name a hypothesis: each extra bit of description costs about $0.69/\varepsilon$ examples.

</details>

3. a finite class bound for a float32 linear model on MNIST asks for 348,298 examples at $\varepsilon = \delta = 0.05$. MNIST has 60,000 training images, yet logistic regression generalizes fine on it. what's wrong?

<details><summary>answer</summary>

nothing is wrong, the bound is just loose: it treats all $2^{25120}$ weight vectors as equally plausible outputs, while the learner effectively explores a tiny, data-dependent part of the class. tighter analyses use the effective complexity (VC dimension, margins, norms, PAC-Bayes priors) instead of the raw count. topics 02–06 do exactly this.

</details>

4. why is the realizable rate $1/\varepsilon$ while the agnostic rate is $1/\varepsilon^2$?

<details><summary>answer</summary>

in the realizable case you only need to rule out hypotheses with error above $\varepsilon$, and each one survives $m$ examples with probability $(1-\varepsilon)^m$, which decays at rate $\varepsilon$ per example. in the agnostic case you need to estimate error rates that are bounded away from 0 to precision $\varepsilon$; an average of $m$ Bernoullis has standard deviation $\sim 1/\sqrt m$, so precision $\varepsilon$ costs $1/\varepsilon^2$.

</details>

5. in idea 3, why is it enough that no example lands in $[t^\star, t^\star + \varepsilon)$, and not some other interval of mass $\varepsilon$?

<details><summary>answer</summary>

ERM returns the smallest positive example, and every positive is at or above $t^\star$. the risk is $\hat t - t^\star$, so it exceeds $\varepsilon$ exactly when the smallest positive is beyond $t^\star + \varepsilon$. a learner that returned the largest negative instead would fail when the interval $(t^\star - \varepsilon, t^\star)$ is empty, with the same probability.

</details>

6. you could get a PAC guarantee for thresholds by discretizing to a grid of 1,000 values and using the finite-class bound. what does the grid cost you, and what does it lose?

<details><summary>answer</summary>

it costs $\ln 1000/\varepsilon \approx 6.9/\varepsilon$ extra examples in the bound, and the target may no longer be in the class exactly (so you're slightly agnostic, with approximation error up to the grid spacing). it loses nothing in practice if the grid is fine, which is why "count the hypotheses after discretizing" is a legitimate, if crude, bounding technique.

</details>

## common bugs

- **`np.log(class_size)` or `class_size / delta` on a huge integer.** symptom: `OverflowError` or `inf` for classes like $2^{25120}$. use `math.log` and add logarithms.
- **rounding to nearest instead of up.** symptom: a sample size that's one too small; the defining inequality fails at your $m$.
- **forgetting the factor 2 or the square in the agnostic formula.** symptom: agnostic sample sizes that are far too small, or equal to the realizable ones.
- **ERM candidates that miss $+\infty$.** symptom: with no positive examples your threshold predicts some point positive and makes one avoidable error.
- **comparing the raw risk with $\varepsilon$ under noise.** symptom: a failure rate of 1.0 whenever the noise rate exceeds $\varepsilon$.
- **a new generator per trial** (`as_rng(rng)` inside the loop with an integer seed). symptom: every trial draws the same sample, so the estimate is 0 or 1.

## going deeper

- Valiant (1984), "A theory of the learnable", *Communications of the ACM* 27(11). the origin of PAC learning. [doi:10.1145/1968.1972](https://doi.org/10.1145/1968.1972)
- Blumer, Ehrenfeucht, Haussler & Warmuth (1987), "Occam's razor", *Information Processing Letters* 24(6). the finite-class bound as a statement about description length. [doi:10.1016/0020-0190(87)90114-1](https://doi.org/10.1016/0020-0190(87)90114-1)
- Shalev-Shwartz & Ben-David (2014), *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, ch. 2–4 (realizable finite classes, the PAC model, uniform convergence). [free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- Mohri, Rostamizadeh & Talwalkar (2018), *Foundations of Machine Learning*, 2nd ed., MIT Press, ch. 2 (the PAC framework, including the threshold and rectangle examples). [cs.nyu.edu/~mohri/mlbook](https://cs.nyu.edu/~mohri/mlbook/)
- Massart & Nédélec (2006), "Risk bounds for statistical learning", *Annals of Statistics* 34(5). when noise is bounded away from 1/2, the agnostic rate improves. [doi:10.1214/009053606000000786](https://doi.org/10.1214/009053606000000786)

## connections

- ← builds on: the union bound and $1 - x \le e^{-x}$ (ch 01), Hoeffding's inequality (ch 05 · topic 01), empirical vs true risk (ch 07 · topic 01), Monte Carlo standard errors (ch 04).
- → used in: the VC dimension, which replaces $\ln|\mathcal H|$ for infinite classes (topic 02), Rademacher complexity (topic 03), computing bounds for real models (topic 04), the online-to-batch view of regret (topic 05), PAC-Bayes, which replaces the uniform $\ln|\mathcal H|$ with a KL divergence to a prior (topic 06), decision stumps (ch 10).
