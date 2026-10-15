# bounds in practice

## why this matters

three topics of theory have given you three ways to say "the true error is at most this". none of them is worth anything until you plug in numbers. this topic does exactly that: it implements the classical bounds with **every constant** spelled out, applies them to models you trained in part II on real data, and puts them next to the error you actually measure on a test set.

the outcome is sobering, and it's important to see it with your own eyes. for a logistic regression with nine parameters, trained on sixteen thousand examples, whose measured train–test gap is below one percentage point, the VC bound promises only that the error is below 37%. for a 784-pixel model it promises nothing at all. bounds that can't rule out a coin flip are called **vacuous**, and the uniform bounds of topics 01–03 are vacuous for most models people actually use.

that doesn't make them useless. they tell you *how* generalization scales (logarithmically in the class size, like $\sqrt{d/n}$ in the VC dimension), they identify which properties of a model matter, and with a little cleverness (counting the bits a model really uses) they can become informative. and one bound is tight in practice: the **test-set bound**, which uses held-out data and an exact binomial tail. knowing exactly what each bound assumes, and what it costs, is the practical skill this topic builds.

## warm-up: predict first

1. a model is stored as 785 float32 numbers. treated as a finite class, how many hypotheses is that, and does the finite-class bound overflow your calculator?
2. logistic regression with 8 features has VC dimension 9. how many training examples does the VC bound need before its complexity term drops below 0.1: about a thousand, about ten thousand, or about a hundred thousand?
3. you measure 3 errors on 50 held-out examples. what's a 95% upper confidence bound on the true error: about 0.10, 0.15, or 0.25?
4. you compare 100 models on the same test set and report the best one's test error with a 95% confidence bound. is the bound still valid?
5. a VC bound and a "count the bits" bound for the same 785-parameter model on 60,000 examples: which one is tighter?

## prerequisites check

- [ ] you know the finite-class and agnostic sample complexities (→ topic 01)
- [ ] you know the VC dimension, the growth function and Sauer's lemma (→ topic 02)
- [ ] you know Hoeffding's inequality and the binomial distribution, and you can use `scipy.stats` (→ ch 04, ch 05 · topic 01)
- [ ] you've trained logistic regression and $k$-NN and know the estimator API (`fit`, `predict`) (→ ch 07 · topic 01, ch 08 · topic 04, ch 09 · topic 01)

## core ideas

### idea 1: the finite-class bound, as a bound

topic 01 asked "how many examples do I need?". the same inequality, solved for $\varepsilon$ instead of $m$, answers "how wrong can I be with the examples I have?". for one fixed hypothesis $h$ with losses in $[0, 1]$, the **one-sided** Hoeffding inequality gives $P(L(h) - L_S(h) > \varepsilon) \le e^{-2n\varepsilon^2}$. a union bound over a finite class, with failure probability $\delta/|\mathcal H|$ for each member, gives: with probability at least $1 - \delta$, **every** $h \in \mathcal H$ satisfies

$$L(h) \le L_S(h) + \sqrt{\frac{\ln|\mathcal H| + \ln(1/\delta)}{2n}}.$$

(topic 01's agnostic calculator used the two-sided version, with $2|\mathcal H|$, because it needed $|L - L_S|$ small in both directions; an upper bound on the true error needs only one side.) because it holds for every $h$ at once, it holds for whatever $h$ your learner picks, however it picks it.

the practical trick is **counting bits**. a model you store as $P$ parameters of $b$ bits each is one of at most $2^{bP}$ possible models, so $\ln|\mathcal H| \le bP\ln 2$, whatever the architecture. for a float32 logistic regression on 784 pixels, $|\mathcal H| = 2^{25120}$ (warm-up 1), a number with 7,562 digits. you can't compute it as a float, but you never need to: Python integers are exact and `math.log` takes their logarithm directly. the bound is then $\sqrt{(17412 + \ln(1/\delta))/(2n)}$: 0.38 at $n = 60{,}000$. the bound is valid only if the hypothesis class really is fixed before you see the data, which matters when you quantize (the lab).

> **spec.** `finite_class_bound(n, class_size, delta, emp_risk=0.0)` returns `emp_risk + sqrt((ln|H| + ln(1/delta)) / (2n))` as a float. `class_size` may be a huge Python int: use `math.log`.

> **▶ build: stage 1.** implement `finite_class_bound`.

### idea 2: the VC bound, with its constants

for an infinite class, topic 02's growth-function argument replaces $|\mathcal H|$ by the number of labelings of a double sample. the form you'll implement is the classical one (Vapnik 1998; Abu-Mostafa, Magdon-Ismail & Lin 2012, eq. 2.12): with probability at least $1 - \delta$, every $h \in \mathcal H$ satisfies

$$L(h) \le L_S(h) + \sqrt{\frac8n\Big(\ln\Pi_{\mathcal H}(2n) + \ln\frac4\delta\Big)}, \qquad \ln\Pi_{\mathcal H}(2n) \le d\ln\frac{2en}{d}\ \ (\text{Sauer, } 2n \ge d).$$

each constant has a source, and it's worth knowing which, because each is a place where the bound gives something away:

- **the ghost sample.** symmetrization compares $S$ with an independent sample $S'$ of the same size: the true risk is replaced by the risk on $S'$, at the cost of halving $\varepsilon$ and counting labelings of $2n$ points instead of $n$.
- **the union bound over labelings.** only $\Pi(2n)$ distinct behaviours exist on $S \cup S'$, so a union bound over them replaces $|\mathcal H|$; Sauer's lemma turns that into $(2en/d)^d$. this is where a polynomial count replaces an infinite one, and also where most of the slack lives, since real classes rarely realize anything like $\Pi(2n)$ labelings *that matter*.
- **Hoeffding on a difference, plus a random swap.** the deviation between the two half-samples, at scale $\varepsilon/2$, gives the $e^{-n\varepsilon^2/8}$, hence the 8; making the symmetrization step rigorous costs the factor 4 in front.

other books state the same theorem with different constants (Bousquet, Boucheron & Lugosi 2004 have $\ln(2/\delta)$; tighter chaining arguments remove the $\ln n$ at the price of unspecified constants). **always say which version you use.** two edge cases: $d = 0$ (a single hypothesis) contributes $\ln\Pi = 0$; and for $2n < d$ Sauer's lemma doesn't apply, but $\Pi(2n) \le 2^{2n}$ always does, which already makes the bound vacuous. since both $\Pi(2n) \le (2en/d)^d$ and $\Pi(2n) \le 2^{2n}$ hold when $2n \ge d$, use the smaller of the two: near $2n \approx d$ Sauer's form is the larger one, and taking the minimum keeps the bound from (absurdly) growing with $n$.

> **spec.** `vc_bound(n, vc_dim, delta, emp_risk=0.0)`: `emp_risk + sqrt((8/n) * (log_growth + ln(4/delta)))` with `log_growth` $= \min(d\ln(2en/d),\ 2n\ln 2)$ if $2n \ge d$, $2n\ln 2$ if $2n < d$, and 0 if $d = 0$. (for $n \gg d$ this is the contract's $d\ln(2en/d)$.)

> **🔮 predict:** with $n = 10{,}000$ and $d = 10$, how big is the complexity term? ($d\ln(2en/d) = 86.0$ and $\ln 80 = 4.38$.)

> **▶ build: stage 2.** implement `vc_bound`.

### idea 3: the test-set bound

the uniform bounds pay for not knowing which hypothesis the learner will choose. if you evaluate **one** classifier, chosen *before* looking at a held-out set of $m$ examples, there's nothing to pay for. its number of test errors $k$ is then exactly $\text{Binomial}(m, L)$, with $L$ its true error, and you can invert the binomial tail exactly (Langford, 2005):

$$\bar L(k, m, \delta) = \max\{p \in [0, 1] : P(\text{Binomial}(m, p) \le k) \ge \delta\}.$$

if the true error were above $\bar L$, seeing as few as $k$ errors would have probability below $\delta$; so with probability at least $1 - \delta$, $L \le \bar L$. this is the one-sided **Clopper–Pearson** upper confidence limit, and because the binomial CDF is a beta tail, $\bar L = $ the $(1 - \delta)$ quantile of $\text{Beta}(k + 1, m - k)$ (for $k < m$; it's 1 when $k = m$). with zero errors it has a closed form, $(1 - \bar L)^m = \delta$. it's always tighter than Hoeffding's $k/m + \sqrt{\ln(1/\delta)/(2m)}$, dramatically so when the error is small: Hoeffding pays for the worst-case variance $\frac14$, the binomial pays for the actual variance $L(1 - L)$.

the test-set bound is the honest workhorse of practical ML, and it has two costs. the held-out examples can't be used for training. and it's valid for one classifier fixed in advance: if you evaluate $k$ classifiers and report the best, you need a union bound, $\bar L(\cdot, m, \delta/k)$ (warm-up 4: no, unless you correct for it). the lab shows what happens if you don't.

> **spec.** `test_set_bound(n_errors, n, delta)` returns $\bar L$: `scipy.stats.beta.ppf(1 - delta, n_errors + 1, n - n_errors)`, or bisection on `scipy.stats.binom.cdf` to within $10^{-7}$; 1.0 when `n_errors == n`.

> **▶ build: stage 3.** implement `test_set_bound`.

### idea 4: how much data would the VC bound need?

turn the VC bound around: how large must $n$ be for the complexity term to fall below a target $\varepsilon$? there's no closed form, because $n$ appears inside the logarithm, but the term is decreasing in $n$ once $n \ge d$ (on paper 3), so a search finds it. a linear scan is hopeless (the answers run into the hundreds of thousands); the standard pattern for monotone searches is **doubling then bisection**: double $n$ until the bound is at most $\varepsilon$, which takes $O(\log n)$ steps, then bisect between the last failure and the first success, another $O(\log n)$ steps.

the answer scales like $\frac{8d}{\varepsilon^2}\ln\frac{n}{d}$, roughly: linear in the VC dimension, quadratic in $1/\varepsilon$. for $d = 9$ (warm-up 2), $\varepsilon = 0.1$ takes about 81,000 examples, and $\varepsilon = 0.05$ about 370,000. compare that with the lab's measured gap, which is under 0.01 from $n = 1000$ on.

> **spec.** `vc_sample_size(vc_dim, epsilon, delta)` returns the smallest integer $n \ge \max(d, 1)$ with `vc_bound(n, vc_dim, delta) <= epsilon` (emp_risk 0): return $\max(d, 1)$ if it already qualifies; otherwise double, then bisect with the invariant "bound(lo) > ε ≥ bound(hi)".

> **▶ build: stage 4.** implement `vc_sample_size`.

### idea 5: a generalization report

a report puts everything for one model side by side: training error, test error, the measured gap, and each bound. two rules make it honest.

**measure the gap; don't estimate it.** the test error on a held-out set *is* the measurement, and the test-set bound turns it into a guarantee. the uniform bounds are statements about every hypothesis in the class, computed from the training data alone; their job in a report is to show how far the guarantee you could get *without* held-out data is from what you measured.

**know what each bound costs in $\delta$.** each bound above holds with probability $1 - \delta$ on its own. if you want all of them to hold *simultaneously*, a union bound says to compute each at $\delta/3$. reporting one bound chosen *after* seeing which is tightest needs the same correction. the report computes each at $\delta$, and says so.

one more convention: a bound of 1 or more is vacuous (an error rate can't exceed 1), and for balanced binary problems anything at or above 0.5 says no more than a coin flip does. where a model's VC dimension is infinite, like $k$-NN's, the VC bound is simply $\infty$ and only the test-set bound has anything to say.

> **spec.** `generalization_report(model, X_train, y_train, X_test, y_test, *, vc_dim, n_params, delta=0.05, bits=32)`: call `model.fit(X_train, y_train)` once, then return a dict with `n_train`, `n_test`, `train_error`, `test_error`, `gap` (test minus train), `vc_bound` (`vc_bound(n_train, vc_dim, delta, train_error)`), `finite_bound` (`finite_class_bound(n_train, 2 ** (bits * n_params), delta, train_error)`), and `test_bound` (`test_set_bound(number of test errors, n_test, delta)`).

> **🔮 predict:** for logistic regression on 16,000 binarized California districts (9 parameters), rank the three bounds from tightest to loosest.

> **▶ build: stage 5.** implement `generalization_report`.

## worked example

**the finite-class bound.** $n = 1000$, $|\mathcal H| = 1000$, $\delta = 0.05$, training error 0.1: $\ln 1000 + \ln 20 = 6.908 + 2.996 = 9.903$; $\sqrt{9.903/2000} = 0.0704$; bound $0.1704$.

**the VC bound.** $n = 10{,}000$, $d = 10$, $\delta = 0.05$: $d\ln(2en/d) = 10\ln 5436.6 = 86.01$, $\ln(4/\delta) = \ln 80 = 4.38$; $\sqrt{8 \times 90.39/10000} = 0.2689$. with $d = 0$ the same $n$ and $\delta$ give $\sqrt{8\ln 80/10^4} = 0.0592$, and `vc_sample_size(0, 0.1, 0.05)` is $\lceil 8\ln 80/0.01\rceil = 3506$.

**the test-set bound.** 0 errors in 100: $(1 - p)^{100} = 0.05$ gives $\bar L = 1 - 0.05^{1/100} = 0.0295$. 3 errors in 50 (warm-up 3): $\bar L = $ `beta.ppf(0.95, 4, 47)` $= 0.148$, while Hoeffding gives $0.06 + \sqrt{\ln 20/100} = 0.233$.

## on paper

1. **one-sided versus two-sided.** derive the finite-class bound of idea 1 from the one-sided Hoeffding inequality, and explain why topic 01's agnostic sample complexity had $\ln(2|\mathcal H|/\delta)$ instead.

<details><summary>solution</summary>

for each $h$, $P(L(h) - L_S(h) > \varepsilon) \le e^{-2n\varepsilon^2}$. the union bound over $\mathcal H$ gives $P(\exists h: L(h) - L_S(h) > \varepsilon) \le |\mathcal H|e^{-2n\varepsilon^2}$; set it to $\delta$ and solve: $\varepsilon = \sqrt{(\ln|\mathcal H| + \ln(1/\delta))/(2n)}$. topic 01 needed $|L - L_S| \le \varepsilon$ in both directions (to compare ERM's choice with the best hypothesis), which doubles each failure probability.

</details>

2. **why the test-set bound is valid.** let $L$ be the true error of a fixed classifier and $K \sim \text{Binomial}(m, L)$ its number of test errors. show that $P(L > \bar L(K, m, \delta)) \le \delta$.

<details><summary>solution</summary>

write $F_p(k) = P(\text{Binomial}(m, p) \le k)$, which is decreasing in $p$. by the definition of $\bar L$ as the largest $p$ with $F_p(K) \ge \delta$, the event $L > \bar L(K)$ implies $F_L(K) < \delta$. now let $k^\star$ be the largest integer with $F_L(k^\star) < \delta$ (if there is none, the event is empty). since $F_L$ is increasing in $k$, $F_L(K) < \delta$ happens exactly when $K \le k^\star$, so $P(F_L(K) < \delta) = P(K \le k^\star) = F_L(k^\star) < \delta$.

</details>

3. **the VC term decreases in $n$.** show that $g(n) = \frac1n\big(d\ln\frac{2en}{d} + c\big)$ with $c > 0$ is decreasing for $n \ge d$.

<details><summary>solution</summary>

$g'(n) = \frac{1}{n^2}\big(d - d\ln\frac{2en}{d} - c\big)$. for $n \ge d$, $\ln\frac{2en}{d} \ge \ln 2e > 1$, so $d - d\ln\frac{2en}{d} < 0$, and with $c > 0$ the derivative is negative. so doubling and bisection find the right answer.

</details>

4. **the price of choosing.** you evaluate $N$ classifiers on the same test set, classifier $j$ making $K_j$ errors. show that reporting $\bar L(K_j, m, \delta/N)$ for every $j$ (hence for the best one) is valid with probability $1 - \delta$. how much does it widen the bound, roughly, when the error is small?

<details><summary>solution</summary>

each $\bar L(K_j, m, \delta/N)$ fails with probability at most $\delta/N$; by the union bound all $N$ hold with probability at least $1 - \delta$. for zero errors, $\bar L = 1 - (\delta/N)^{1/m} \approx \ln(N/\delta)/m$, so the bound grows by $\ln N/m$: additive and logarithmic. comparing 10,000 models costs about $9.2/m$, which for $m = 10{,}000$ is less than 0.1 percentage points.

</details>

## experiment

`lab.py` investigates, with your code and real data:

1. a learning curve for logistic regression on binarized California housing, with the VC bound, the 32-bit counting bound and the test-set bound at every $n$, plus a $k$-NN model whose VC dimension is infinite.
2. MNIST even vs odd with a 785-parameter linear model: what happens to the counting bound and to the measured error when you quantize the weights to fewer bits.
3. how often a "95%" test-set bound fails when the best of $k$ coin-flip classifiers is reported, with and without the $\delta/k$ correction.

## check your understanding

1. the VC bound holds for every hypothesis in the class simultaneously. why is that both why it's valid for any learner and why it's loose?

<details><summary>answer</summary>

valid: it doesn't care how the learner chose $h$, so it covers ERM, regularized fits, early stopping, anything. loose: it must cover the worst hypothesis too, including ones a sensible learner would never pick, and the worst distribution over data.

</details>

2. why does counting bits sometimes beat the VC bound, even though "all float32 weight vectors" is a subset of "all real weight vectors" with the same VC dimension?

<details><summary>answer</summary>

both are valid for the float32 class, but they bound it differently. the counting bound pays $bP\ln 2$, linear in the parameter count with a modest constant; the VC bound pays $8d\ln(2en/d)$, which carries a factor 8 and a $\ln n$. for $n = 60{,}000$, $d = 785$, the counting bound with $b \le 32$ wins. neither is tight; they're two different loose arguments.

</details>

3. the test-set bound is tight. why doesn't everyone use it instead of learning theory?

<details><summary>answer</summary>

everyone does, in practice. but it requires held-out data you can't train on, it's valid only for a classifier fixed before seeing the test set (and degrades with each reuse), and it explains nothing: it tells you *that* a model generalizes, not *why*, or how the error will scale with more data or a different model.

</details>

4. a bound says the error is below 0.37 and the measured test error is 0.156. is the bound "wrong"?

<details><summary>answer</summary>

no. a bound is an upper limit that holds with high probability; it's correct whenever the true error is below it. "loose" isn't "wrong". a bound is wrong only if the true error exceeds it more often than $\delta$ of the time, which is what the lab's coin-flip experiment checks for the test-set bound.

</details>

5. you have 1,000 labelled examples. should you train on all of them and use a uniform bound, or hold out 200 and use the test-set bound?

<details><summary>answer</summary>

for a guarantee you can use, hold out. with 800 training examples a typical model loses a little accuracy, but the test-set bound on 200 examples is within a few percentage points of the measured error, while uniform bounds on 1,000 examples are usually vacuous. (cross-validation gives better estimates still, but no simple bound of this form, because the folds aren't independent.)

</details>

6. in the generalization report, why is each bound computed at $\delta$ and not at $\delta/3$?

<details><summary>answer</summary>

it's a convention: the report shows what each argument gives on its own. if you want to claim all three at once, or report whichever is tightest, you must pay for the choice with a union bound and use $\delta/3$ for each (the values change only slightly, since $\delta$ enters through a logarithm).

</details>

## common bugs

- **`np.log(2 ** (32 * 785))`.** symptom: `OverflowError` (int too large to convert to float). use `math.log`.
- **Sauer's bound for $2n < d$.** symptom: `math domain error` or a negative number under the square root for tiny $n$. switch to $2n\ln 2$.
- **a test-set bound from the training error.** symptom: a "tight" bound that's wrong: the binomial argument needs data the classifier never saw.
- **bisection without an invariant.** symptom: an off-by-one sample size, or an infinite loop when `lo` and `hi` stop moving. keep "bound(lo) > ε ≥ bound(hi)" and stop at `hi - lo == 1`.
- **`beta.ppf(delta, ...)` instead of `beta.ppf(1 - delta, ...)`.** symptom: an "upper" bound below the observed error rate.
- **reporting the best of many models with an uncorrected bound.** symptom: none, until the model is deployed. use $\delta/k$.

## going deeper

- Langford (2005), "Tutorial on practical prediction theory for classification", *JMLR* 6. the test-set bound, Occam bounds, and why bounds should be computed. [jmlr.org/papers/v6/langford05a](https://www.jmlr.org/papers/v6/langford05a.html)
- Vapnik (1998), *Statistical Learning Theory*, Wiley. the VC bounds with constants.
- Abu-Mostafa, Magdon-Ismail & Lin (2012), *Learning From Data*, AMLBook, ch. 2. the VC generalization bound in exactly the form used here, with worked sample sizes. [amlbook.com](https://amlbook.com/)
- Bousquet, Boucheron & Lugosi (2004), "Introduction to statistical learning theory", in *Advanced Lectures on Machine Learning*, Springer LNAI 3176. a compact survey with proofs. [doi:10.1007/978-3-540-28650-9_8](https://doi.org/10.1007/978-3-540-28650-9_8)
- Clopper & Pearson (1934), "The use of confidence or fiducial limits illustrated in the case of the binomial", *Biometrika* 26(4). [doi:10.1093/biomet/26.4.404](https://doi.org/10.1093/biomet/26.4.404)
- Dwork, Feldman, Hardt, Pitassi, Reingold & Roth (2015), "The reusable holdout: preserving validity in adaptive data analysis", *Science* 349(6248). what to do when you can't stop looking at the test set. [doi:10.1126/science.aaa9375](https://doi.org/10.1126/science.aaa9375)

## connections

- ← builds on: the finite-class bound (topic 01), the VC dimension and Sauer's lemma (topic 02), Rademacher bounds (topic 03), held-out evaluation and the estimator API (ch 07), logistic regression (ch 08), $k$-NN (ch 09), binomial confidence intervals (ch 05).
- → used in: PAC-Bayes, which builds non-vacuous bounds from the kl-inverse of the same binomial tail (topic 06), the capstone's generalization report, generalization in deep learning (ch 19 · topic 04), and int8 quantization, where accuracy against bits comes back (ch 26 · topic 06).
