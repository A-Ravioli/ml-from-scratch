# 13 · learning theory · checkpoint

use this to check that you've finished the chapter, or to test out of it. work without notes, then compare with the answers. section 3 is checked by the topics' own tests.

## 1. concept check

1. state the $(\varepsilon, \delta)$-PAC guarantee in words and symbols. what is random, and what must the guarantee hold for?

<details><summary>answer</summary>

with probability at least $1 - \delta$ over an iid sample of size $m$, the learner's hypothesis has true risk at most $\varepsilon$ (realizable) or at most $\varepsilon$ above the best in the class (agnostic): $P_{S \sim \mathcal D^m}(L_{\mathcal D}(h_S) > \varepsilon) \le \delta$. only the sample is random; the guarantee must hold for every distribution $\mathcal D$ (and every target in the class, in the realizable case). (topic 01, idea 1)

</details>

2. why does the realizable sample complexity scale like $1/\varepsilon$ and the agnostic one like $1/\varepsilon^2$?

<details><summary>answer</summary>

realizable: a hypothesis with error above $\varepsilon$ survives $m$ examples with probability $(1 - \varepsilon)^m \approx e^{-\varepsilon m}$. agnostic: you must estimate error rates bounded away from 0 to precision $\varepsilon$, and an average of $m$ Bernoullis fluctuates by $\sim 1/\sqrt m$, so precision $\varepsilon$ takes $1/\varepsilon^2$ examples (Hoeffding: $e^{-2m\varepsilon^2}$). (topic 01, idea 1)

</details>

3. a lab showed thresholds with 10% label noise need only about $1/\varepsilon$ examples. does that contradict the agnostic $1/\varepsilon^2$?

<details><summary>answer</summary>

no: the agnostic bound is a worst case over distributions. thresholds with bounded noise satisfy the Massart condition (excess risk grows linearly away from the best threshold, and labels near it remain informative), under which fast rates hold. the $1/\varepsilon^2$ rate is forced only by distributions with many near-optimal hypotheses that disagree on a lot of mass. (topic 01, lab)

</details>

4. define shattering and the VC dimension. what do you have to show for a lower bound, and for an upper bound?

<details><summary>answer</summary>

$\mathcal H$ shatters $C$ if it realizes all $2^{|C|}$ labelings of $C$; $\text{VCdim}$ is the size of the largest shattered set. lower bound $\text{VCdim} \ge d$: exhibit one shattered $d$-set. upper bound $\text{VCdim} < d + 1$: show that *no* $(d + 1)$-set is shattered (an argument about every configuration, e.g. Radon's theorem for halfspaces). (topic 02, idea 1)

</details>

5. what does Sauer's lemma say, and why is it the key to learnability of infinite classes?

<details><summary>answer</summary>

if $\text{VCdim} = d$, then $\Pi_{\mathcal H}(m) \le \sum_{i \le d}\binom mi \le (em/d)^d$: polynomial, not exponential, in $m$. the uniform convergence argument multiplies a failure probability $e^{-cm\varepsilon^2}$ by the number of labelings of a (double) sample; a polynomial times that goes to 0, $2^m$ times it doesn't. (topic 02, idea 5)

</details>

6. why is "VC dimension = number of parameters" false? give an example each way.

<details><summary>answer</summary>

$\mathbb 1[\sin(\omega x) > 0]$ has one parameter and infinite VC dimension (it encodes labels in the digits of $\omega$). in the other direction, a class with many redundant parameters (say $w_1w_2x$ thresholded) can have a small VC dimension. for halfspaces it happens to match ($d + 1$). (topic 02, lab)

</details>

7. define the empirical Rademacher complexity. what are its values for a single hypothesis and for a class that shatters the sample?

<details><summary>answer</summary>

$\hat{\mathfrak R}_S(\mathcal F) = \mathbb E_\sigma\sup_{f}\frac1n\sum_i\sigma_if(z_i)$ with iid random signs. a single hypothesis: 0. a $\pm1$ class that realizes every labeling of $S$: exactly 1, so any bound built on it is vacuous. (topic 03, idea 1)

</details>

8. why is the $\ell_2$-ball Rademacher bound $BR/\sqrt n$ "dimension-free", and what happens for the $\ell_1$ ball?

<details><summary>answer</summary>

the sup over $\|w\|_2 \le B$ of $w^\top v$ is $B\|v\|_2$, and $\mathbb E\|\sum_i\sigma_ix_i\|_2 \le \sqrt{\sum_i\|x_i\|^2} \le R\sqrt n$, with no coordinates involved. for the $\ell_1$ ball the dual is $\ell_\infty$, a max over $d$ coordinates, which grows like $\sqrt{\ln d}$ (Massart over the $2d$ vertices). (topic 03, idea 3)

</details>

9. state the VC bound you implemented, with its constants, and name where two of the constants come from.

<details><summary>answer</summary>

$L(h) \le L_S(h) + \sqrt{\frac8n\big(d\ln\frac{2en}d + \ln\frac4\delta\big)}$ for every $h$, with probability $1 - \delta$. the $2n$ and the halved $\varepsilon$ come from symmetrization with a ghost sample; the 8 from Hoeffding at scale $\varepsilon/2$ on a difference; the factor 4 from making the symmetrization rigorous; $d\ln(2en/d)$ from Sauer's lemma on $\Pi(2n)$. (topic 04, idea 2)

</details>

10. why is the test-set bound so much tighter than the uniform bounds, and what are its two costs?

<details><summary>answer</summary>

it bounds one classifier fixed before seeing the test set: the number of test errors is exactly binomial, and its tail is inverted exactly, with no union bound and no worst case over hypotheses. costs: the held-out data can't be used for training, and validity is lost if you choose among many classifiers on the same test set (unless you pay $\delta/k$). (topic 04, idea 3)

</details>

11. what is regret, and why measure it against the best *fixed* expert?

<details><summary>answer</summary>

$R_T = \sum_t p_t \cdot \ell_t - \min_i\sum_t\ell_{t,i}$: your total loss minus that of the best single expert in hindsight. against the best expert *per round* no algorithm can have sublinear regret (an adversary punishes whatever you play); a fixed comparator is the strongest benchmark you can still match on average. (topic 05, idea 3)

</details>

12. state Hedge's regret bound, its optimal learning rate, and why $\sqrt T$ can't be beaten.

<details><summary>answer</summary>

$R_T \le \ln N/\eta + \eta T/8$; with $\eta = \sqrt{8\ln N/T}$, $R_T \le \sqrt{(T/2)\ln N}$. lower bound: with independent fair-coin losses every algorithm has expected loss $T/2$, but the best of $N$ experts in hindsight has about $T/2 - c\sqrt{T\ln N}$ by luck. (topic 05, idea 4)

</details>

13. state the PAC-Bayes-kl bound. what must be fixed before seeing the data, and what may depend on it?

<details><summary>answer</summary>

with probability $1 - \delta$, for all posteriors $Q$: $\text{kl}(\hat L(Q)\,\|\,L(Q)) \le (\text{KL}(Q\,\|\,P) + \ln(2\sqrt n/\delta))/n$. the prior $P$ must be independent of the sample the bound uses (or one of a finite grid fixed in advance, with a union bound); the posterior $Q$ may depend on the data arbitrarily. (topic 06, idea 3)

</details>

14. why does a bound need $\beta = o(1/\sqrt n)$ uniform stability to be useful, and which of ridge regression and 1-NN has it?

<details><summary>answer</summary>

the Bousquet–Elisseeff bound has a term $(4n\beta + M)\sqrt{\ln(1/\delta)/(2n)}$, which vanishes only if $n\beta\cdot n^{-1/2} \to 0$. ridge with a fixed per-example regularization has $\beta = O(1/(\lambda n))$; 1-NN has $\beta = O(1)$ (removing a query's nearest neighbour changes its prediction by a full label difference, however large $n$ is). (topic 06, idea 6 and lab)

</details>

## 2. by hand

1. **sample complexity and a VC bound.** (a) a finite class has $|\mathcal H| = 2^{20}$ hypotheses. how many examples guarantee, in the realizable case, error at most $0.05$ with probability $0.99$? (b) a class has VC dimension 3. with $n = 1000$ examples and $\delta = 0.05$, what is the complexity term of the VC bound? use $\ln 1000 \approx 6.91$, $\ln 2e \approx 1.69$, $\ln 80 \approx 4.38$.

<details><summary>answer</summary>

(a) $m \ge (20\ln 2 + \ln 100)/0.05 = (13.86 + 4.61)/0.05 = 369.4$, so $m = 370$. (b) $d\ln(2en/d) = 3(\ln 2e + \ln 1000 - \ln 3) = 3(1.69 + 6.91 - 1.10) = 22.5$; $\sqrt{8(22.5 + 4.38)/1000} = \sqrt{0.215} = 0.464$. even a 3-parameter class gets a weak bound from 1,000 examples.

</details>

2. **a Rademacher complexity and a Hedge update.** (a) compute the empirical Rademacher complexity of the class $\{h_1 = (1, 1),\ h_2 = (1, -1)\}$ on two points exactly, and compare it with Massart's bound. (b) Hedge with 2 experts and $\eta = \ln 3$ sees losses $(1, 0)$ then $(0.5, 1)$. what distribution does it play in round 3?

<details><summary>answer</summary>

(a) the four sign patterns give $\max(\sigma_1 + \sigma_2, \sigma_1 - \sigma_2)/2$: for $(1, 1)$: 1; $(1, -1)$: 1; $(-1, 1)$: 0; $(-1, -1)$: 0. average $\frac12$. massart: $r = \sqrt2$, $N = 2$: $\sqrt2\sqrt{2\ln 2}/2 = \sqrt{\ln 2} = 0.83$. (b) cumulative losses $(1.5, 1)$; weights $e^{-\ln 3 \cdot 1.5} = 3^{-1.5} = 0.192$ and $3^{-1} = 0.333$; normalized $(0.366, 0.634)$.

</details>

3. **a kl-inverse and a PAC-Bayes bound.** (a) show that $\text{kl}^{-1}(0, c) = 1 - e^{-c}$. (b) a finite class of $1024$ hypotheses, a uniform prior, and a learner that outputs (a point mass on) a hypothesis with zero training errors on $n = 500$ examples. compute the PAC-Bayes-kl bound at $\delta = 0.05$, using $\ln 1024 = 6.93$ and $\ln(2\sqrt{500}/0.05) = \ln 894.4 = 6.80$, and compare it with topic 04's Hoeffding-based finite-class bound $\sqrt{(\ln|\mathcal H| + \ln 20)/(2n)}$.

<details><summary>answer</summary>

(a) $\text{kl}(0\,\|\,p) = \ln\frac1{1 - p}$ is increasing in $p$; setting it equal to $c$ gives $p = 1 - e^{-c}$. (b) $\text{KL} = \ln 1024 = 6.93$, $c = (6.93 + 6.80)/500 = 0.0275$, bound $1 - e^{-0.0275} = 0.027$. the Hoeffding bound is $\sqrt{(6.93 + 3.00)/1000} = 0.0997$, almost four times larger: at zero empirical risk the kl form's slack is linear in $c$, not $\sqrt c$.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild in a scratch notebook:

- `finite_class_sample_complexity` (both regimes, huge class sizes) and `finite_class_bound`;
- `vc_bound` (with the $2n < d$ case) and `test_set_bound`;
- `empirical_rademacher` (vectorized Monte Carlo);
- `Hedge` (stable softmax of cumulative losses) and `OnlineGradientDescent` ($\eta/\sqrt t$, projection after the step);
- `kl_bernoulli`, `kl_inverse` (bisection, upper end) and `pac_bayes_bound`.

then run the topics' tests against your file, for example:

```bash
pytest 13-learning-theory/04-bounds-in-practice/test_exercise.py -k "stage1 or stage2 or stage3"
pytest 13-learning-theory/05-online-learning/test_exercise.py -k "stage1 or stage2"
pytest 13-learning-theory/06-stability-margins-and-pac-bayes/test_exercise.py -k "stage1 or stage2 or stage3"
```

(copy your functions into the corresponding `exercise.py`, or point the tests at your scratch file.)

## 4. readiness

move on when you score at least 80% on sections 1–2 (16 of 20 items, counting each lettered by-hand part separately) and the blind rebuild passes its tests. if you're using this as a placement test, the same thresholds apply; review the topics whose questions you missed, and run their labs: in this chapter, the labs are where the numbers live.
