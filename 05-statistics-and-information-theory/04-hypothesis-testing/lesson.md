# hypothesis testing

## why this matters

you add a feature to your model and the validation accuracy goes from 81.2% to 82.0%. you try a new learning-rate schedule and the loss curve looks a bit lower. a colleague reports that their method "significantly" beats the baseline on 3 of 12 benchmarks. in each case the question is the same: **is this a real effect, or could chance alone produce a difference this big?**

hypothesis testing is the formal machinery for that question. it's also one of the most misused tools in science. a p-value answers a narrow, precise question that's easy to misstate, and a researcher who tries several analyses and reports the one that "worked" can produce significant results from pure noise, without ever doing anything that feels like cheating. machine learning is not immune: leaderboards, hyperparameter sweeps and "we tried 20 variants" are multiple testing at industrial scale.

in this topic you'll build the most general test there is, the **permutation test**, which needs almost no assumptions: to ask "could this be chance?", it constructs the world where it *is* chance by shuffling labels, and sees how often that world produces something as extreme. then you'll measure its error rates by simulation, and build the two standard corrections for running many tests at once. in the lab you'll build a p-hacking machine and watch it "discover" effects in noise nearly every time.

## warm-up: predict first

write down an answer to each before reading on. each one comes back later in the lesson.

1. a test gives $p = 0.03$. is the probability that the null hypothesis is true 3%?
2. if there's no effect at all, what's the distribution of the p-value? is $p = 0.8$ more or less likely than $p = 0.03$?
3. you test 20 unrelated hypotheses, all of them actually null, each at $\alpha = 0.05$. what's the chance at least one comes out significant?
4. you check your experiment's p-value after every batch of new data and stop as soon as $p \le 0.05$. is the false-positive rate still 5%?
5. a small study finds a significant effect. is the effect size it reports likely to be accurate, too big, or too small?

## prerequisites check

- [ ] you know conditional probability and why $P(A \mid B) \ne P(B \mid A)$ (→ ch 04 · topic 01)
- [ ] you can simulate many datasets with a shared `rng` (→ topic 01)
- [ ] you know the sampling distribution of a mean and the standard error (→ topic 03)
- [ ] you know what a bootstrap resample is, and why paired data should be resampled in pairs (→ topic 03)
- [ ] you can sort with `np.argsort` and use boolean masks (→ ch 00 · topic 02)

## core ideas

### idea 1: null hypotheses and p-values

a test starts with a **null hypothesis** $H_0$: a precise statement of "nothing is going on", such as "the treatment and control outcomes come from the same distribution" or "the two models are equally accurate". you choose a **test statistic** $T$, a number computed from the data that tends to be large when $H_0$ is false, like the difference in means. you observe $t_{\text{obs}}$. the **p-value** is

$$p = P_{H_0}\big(T \text{ at least as extreme as } t_{\text{obs}}\big):$$

the probability, *computed assuming the null is true*, of a result at least as extreme as the one you saw. "extreme" depends on the **alternative** you care about: $T \ge t_{\text{obs}}$ (one-sided, "greater"), $T \le t_{\text{obs}}$ ("less"), or $|T| \ge |t_{\text{obs}}|$ (two-sided, for statistics whose null distribution is centered at 0).

the two-sided rule $|T| \ge |t_{\text{obs}}|$ only makes sense when $T$ is **centered at 0 under the null**: a difference of means, medians or accuracies. it's always a *valid* test (its size is at most $\alpha$), but for an uncentered statistic it measures the wrong thing. take a ratio of sample variances, $T = s_x^2/s_y^2$, which is near 1 under the null. with 20 points per group and an observed ratio of 0.10, a glaring difference in spread, every shuffle gives $|T| \ge 0.10$, so the two-sided p-value is 1.0. the one-sided 'less' p-value is $1/10001$, the smallest possible with 10,000 shuffles, and the two-sided test on the *centered* statistic $\log(s_x^2/s_y^2)$ gives 0.0002. so center your statistic, or pick the one-sided alternative you mean.

a small p-value says the data would be surprising if $H_0$ were true. it does **not** say $H_0$ is unlikely to be true (warm-up 1). $p$ is $P(\text{data this extreme} \mid H_0)$; "the probability $H_0$ is true" is $P(H_0 \mid \text{data})$, which needs a prior over hypotheses and Bayes' rule (topic 02, and the base-rate fallacy in ch 04 · topic 01). if real effects are rare in your field, most $p = 0.03$ results can be false alarms.

often you can't compute $p$ exactly, but you can draw $B$ samples $t_1, \dots, t_B$ of the statistic under $H_0$. then the **Monte Carlo p-value** is

$$\hat p = \frac{1 + \#\{b : t_b \text{ as extreme as } t_{\text{obs}}\}}{1 + B}.$$

the $+1$s count the observed data as one more draw from the null, which is exactly true when $H_0$ holds. they make the test valid ($P(\hat p \le \alpha) \le \alpha$ under $H_0$, for any $B$) and they stop $\hat p$ from ever being 0: with $B$ draws you can't claim anything more extreme than $1/(1 + B)$. a practical detail: computed statistics that are mathematically equal can differ in the last bit (`0.1 + 0.2 != 0.3`), so compare with a tiny tolerance, or ties will randomly count as less extreme.

> **▶ build: stage 1.** implement `p_value(observed, null_stats, alternative='two-sided')` in `exercise.py` (`marimo edit exercise.py`).

### idea 2: the permutation test

where do null samples come from? suppose you have outcomes $x_1, \dots, x_m$ for treatment and $y_1, \dots, y_k$ for control. if $H_0$ is "both groups come from the same distribution", then the group labels are arbitrary: under $H_0$, every way of splitting the $m + k$ pooled values into a group of $m$ and a group of $k$ was equally likely to be the one you observed. the values are **exchangeable**. so:

1. pool the data,
2. shuffle and split into groups of sizes $m$ and $k$,
3. recompute the statistic,

and repeat. the shuffled statistics are draws from the exact null distribution of $T$, the **permutation distribution**, with no normality assumption, no large-sample approximation, and any statistic you like: difference of means, of medians, of AUCs. with small samples you can even enumerate every split (the worked example does).

**paired data** need a different shuffle. if each unit was measured under both conditions (the same patient before and after, the same test example scored by two models), the null hypothesis is that within each pair the two labels are exchangeable. so the permutation **swaps $x_i$ and $y_i$ within each pair**, independently with probability $1/2$. for the mean difference, that's the same as flipping the sign of each $d_i = x_i - y_i$ at random. shuffling across pairs instead would throw away the pairing, and with it, as topic 03's lab showed, most of your power.

> **▶ build: stage 2.** implement `permutation_null(x, y, statistic=None, *, n_permutations=10000, paired=False, rng=None)`, the shuffled (or sign-flipped) statistics.

### idea 3: errors, size and power

a test makes a decision: reject $H_0$ if $p \le \alpha$. there are two ways to be wrong.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| reject | **type I error** (false positive), probability $\le \alpha$ | correct: probability = **power** |
| don't reject | correct | **type II error** (false negative) |

the **size** of a test is its type I error rate; a valid test has size at most $\alpha$. the **power** is the probability of detecting an effect that's really there. it depends on the effect size, the noise, $n$, and the test. both are probabilities over repeated experiments, so you can measure them like topic 03's coverage: simulate many datasets from a world where you know the truth, run the test on each, and count rejections.

a useful fact (warm-up 2): when $H_0$ is true and the test statistic is continuous, the p-value is **uniformly distributed** on $[0, 1]$. that's just what "$P(p \le \alpha) = \alpha$ for every $\alpha$" means. under the null, $p = 0.8$ is exactly as likely as $p = 0.03$; a p-value of 0.03 isn't "almost significant evidence", it's what one null experiment in 33 gives you.

power has a sting in its tail (warm-up 5). if a study has low power, the only way it can reach significance is by overestimating the effect: only the runs where noise happened to push the estimate up clear the bar. so significant results from underpowered studies are biased upward, sometimes several-fold: the **winner's curse**. it's the same selection effect as ch 04's "best of 1000 equally good models looks better".

> **🔮 predict:** a treatment shifts the outcome by 0.3 standard deviations, and you run studies with 20 people per group. what fraction of studies reach $p \le 0.05$? and among those that do, what's the average estimated effect?

> **▶ build: stage 3.** implement `permutation_test(...)`, which combines stages 1 and 2 and becomes part of `mlfs.stats`, and `rejection_rate(test, simulate, n_sims, alpha=0.05, *, rng=None)`, a size and power simulator.

### idea 4: many tests at once

test $m$ true null hypotheses, each at level $\alpha$, and the expected number of false positives is $m\alpha$. if the tests are independent, the chance of at least one is $1 - (1 - \alpha)^m$: 64% for $m = 20$ (warm-up 3). there are two standard ways to keep this under control, aiming at different targets.

**the family-wise error rate** (FWER) is the probability of *any* false rejection. **Bonferroni** controls it by testing each hypothesis at $\alpha/m$: by the union bound, $P(\text{any false rejection}) \le \sum_i P(p_i \le \alpha/m) \le m \cdot \alpha/m = \alpha$, whatever the dependence between tests. it's simple and always valid, and with many tests it's very conservative: at $m = 1000$ you need $p \le 0.00005$.

**the false discovery rate** (FDR) is the expected *fraction* of rejections that are false (with 0/0 = 0). in a screen of thousands of hypotheses you may be happy if 5% of your discoveries are duds, as long as you find the real ones. the **Benjamini–Hochberg** (BH) procedure controls FDR at level $q$:

1. sort the p-values: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$.
2. find the **largest** $k$ with $p_{(k)} \le \frac{k}{m}q$.
3. reject the hypotheses with the $k$ smallest p-values.

it's a **step-up** procedure: you scan for the largest $k$ that passes, and reject everything below it, even p-values that individually failed their own threshold. for independent tests (and some kinds of positive dependence) its FDR is at most $\frac{m_0}{m}q \le q$, where $m_0$ is the number of true nulls.

multiple testing corrections fix one specific kind of p-hacking: a known number of tests. they can't fix the other kinds (warm-up 4). **optional stopping** (peeking at the p-value and stopping when it dips below 0.05) gives the null hypothesis many chances to cross the line, and since a random walk eventually crosses any line, the false-positive rate climbs toward 1 the longer you're willing to wait. **analytic flexibility** (which outcome, which subgroup, which outliers to drop) multiplies the number of tests without anyone writing them down. the cures are procedural: decide the analysis and sample size in advance, report every test you ran, and use methods designed for sequential monitoring when you must peek.

> **🔮 predict:** in a screen of 1000 hypotheses with 50 real effects, BH at $q = 0.05$ finds some number of discoveries. roughly what fraction of them will be false?

> **▶ build: stage 4.** implement `bonferroni(p_values, alpha=0.05)` and `benjamini_hochberg(p_values, alpha=0.05)`, each returning a boolean reject mask.

## worked example

**a p-value from a null sample.** observed $t = 2$ against the null draws $(-3, -1, 0, 0.5, 2, 2.5, 4)$. "greater": 3 draws are $\ge 2$ (ties count), so $p = (1 + 3)/8 = 0.5$. "less": 5 draws are $\le 2$, $p = 6/8$. "two-sided": $|{-3}|, 2, 2.5, 4$ are $\ge 2$ in absolute value, $p = 5/8$. against 999 null draws that are all 0, an observed 100 gets $p = 1/1000$, never 0.

**a permutation test by enumeration.** treatment $(1.2, 3.4, 2.2, 5.0)$, control $(0.1, 0.5, 1.9)$: the difference in means is $2.95 - 0.833 = 2.117$. there are $\binom74 = 35$ ways to split the 7 pooled values into groups of 4 and 3. exactly two of them give a difference of at least 2.117: the observed split, and the one that swaps 1.2 with 1.9 (giving 2.525). so the exact one-sided p-value is $2/35 = 0.057$, and the two-sided one is $4/35 = 0.114$ (two more splits give differences $\le -2.117$). not significant at 0.05, even though every treatment value but one beats every control value: with 7 points, there just aren't enough ways to be surprising.

**Bonferroni and BH.** p-values $(0.01, 0.04, 0.03, 0.005, 0.20)$ at $\alpha = 0.05$, $m = 5$. Bonferroni rejects $p \le 0.01$: the first and fourth. for BH, sort: $(0.005, 0.01, 0.03, 0.04, 0.20)$ against thresholds $(0.01, 0.02, 0.03, 0.04, 0.05)$. the largest $k$ that passes is 4 ($0.04 \le 0.04$), so BH rejects the four smallest: everything but 0.20. step-up matters: for $(0.04, 0.02, 0.03, 0.035)$, sorted $(0.02, 0.03, 0.035, 0.04)$ against $(0.0125, 0.025, 0.0375, 0.05)$, the *smallest* fails its threshold but $k = 4$ passes, so BH rejects all four.

the tests check all of these.

## on paper

1. **p-values are uniform under the null.** let $T$ have a continuous CDF $F$ under $H_0$, and $p = 1 - F(T)$ (one-sided, "greater"). show $P(p \le u) = u$ for $u \in [0, 1]$.

<details><summary>solution</summary>

$P(p \le u) = P(F(T) \ge 1 - u) = P(T \ge F^{-1}(1 - u)) = 1 - F(F^{-1}(1 - u)) = u$. this is the probability integral transform from ch 04 · topic 02, run backwards. for discrete statistics (like a permutation test with few splits), $P(p \le u) \le u$: the test is valid but conservative.

</details>

2. **Bonferroni via the union bound.** show that testing each of $m$ hypotheses at $\alpha/m$ controls the FWER at $\alpha$, with no assumption about dependence. when is the bound nearly tight, and when is it very loose?

<details><summary>solution</summary>

let $V$ be the set of true nulls. $P(\exists i \in V: p_i \le \alpha/m) \le \sum_{i \in V}P(p_i \le \alpha/m) \le |V|\alpha/m \le \alpha$. it's nearly tight when the tests are independent and $\alpha$ is small ($1 - (1 - \alpha/m)^m \approx \alpha$), and very loose when the tests are strongly positively correlated: 20 copies of the same test have FWER $\alpha/20$ under Bonferroni, not $\alpha$.

</details>

3. **why the permutation p-value is valid.** assume $H_0$ makes the data exchangeable, so the observed statistic $t_0$ and the permuted ones $t_1, \dots, t_B$ are exchangeable random variables. argue that $P(\hat p \le \alpha) \le \alpha$.

<details><summary>solution</summary>

by exchangeability, the rank of $t_0$ among $t_0, t_1, \dots, t_B$ is uniform on $\{1, \dots, B + 1\}$ (with ties broken in the conservative direction). $\hat p = r/(B + 1)$ where $r$ is the number of the $B + 1$ values at least as extreme as $t_0$, i.e. $t_0$'s rank from the extreme end. so $P(\hat p \le \alpha) = P(r \le \alpha(B + 1)) \le \lfloor\alpha(B + 1)\rfloor/(B + 1) \le \alpha$. without the $+1$s, $r$ could be 0 and $P(\hat p \le \alpha)$ slightly exceeds $\alpha$.

</details>

4. **optional stopping.** under $H_0$, the standardized mean difference after $n$ observations behaves like $S_n/\sqrt n$ for a random walk $S_n$. the law of the iterated logarithm says $\limsup_n S_n/\sqrt{2n\log\log n} = 1$ almost surely. explain why a researcher who peeks after every observation, with no limit on $n$, will eventually reach $p \le 0.05$ with probability 1.

<details><summary>solution</summary>

a two-sided test at 0.05 rejects when $|S_n|/\sqrt n > 1.96$. the LIL says $S_n/\sqrt n$ gets as large as $\sqrt{2\log\log n}$ infinitely often, and that grows without bound, so it exceeds 1.96 infinitely often. so eventually some peek rejects, with probability 1. capping the number of looks keeps the inflation finite (about 19% for 10 looks at 0.05 in the lab), and sequential designs (alpha spending, e.g. O'Brien–Fleming boundaries) spend $\alpha$ across the looks on purpose.

</details>

## experiment

`lab.py` runs three experiments with your code:

1. a p-hacking simulator: forking analysis paths applied to studies of a treatment that does nothing.
2. power, the distribution of p-values, and the winner's curse.
3. 1000 tests at once: no correction, Bonferroni and Benjamini–Hochberg.

record your predictions before each run. at least one result surprises almost everyone.

## check your understanding

1. explain in one sentence what $p = 0.01$ means, without the words "probability the null is true".

<details><summary>answer</summary>

if there were no effect, a result at least this extreme would happen in about 1% of experiments like this one.

</details>

2. a non-significant result ($p = 0.4$): does it show there's no effect?

<details><summary>answer</summary>

no. absence of evidence isn't evidence of absence: a low-power test usually fails to reject even when the effect is real. to argue an effect is small, report a confidence interval (topic 03) and show it excludes effects you'd care about, or run an equivalence test.

</details>

3. why does a permutation test need the $+1$ in the numerator and the denominator?

<details><summary>answer</summary>

the observed data are themselves one draw from the null distribution when $H_0$ is true. counting them keeps the test exactly valid for any number of permutations (on paper 3), and prevents the impossible claim $p = 0$.

</details>

4. two classifiers are evaluated on the same 1000 test examples. which permutation test should you use, and what gets permuted?

<details><summary>answer</summary>

the paired test. the units are examples; for each example you have model A's and model B's correctness (or loss). under $H_0$ the two models' results on each example are exchangeable, so you swap them within each example at random. the unpaired test would treat the 2000 results as independent and ignore that the hard examples are hard for both models, losing most of the power. this is the capstone's core test.

</details>

5. why is BH more powerful than Bonferroni, and what do you give up?

<details><summary>answer</summary>

Bonferroni controls the chance of even one false rejection, so its threshold shrinks like $1/m$. BH controls the expected *fraction* of false rejections; its thresholds grow with rank, $kq/m$, so when there are many real effects it can reject many hypotheses. you give up the guarantee that *every* rejection is real: with BH you expect some false discoveries, around $q$ times the number you make.

</details>

6. you tune 50 hyperparameter settings on a validation set and report the best one's validation accuracy. what has gone wrong, and how do you fix it?

<details><summary>answer</summary>

it's a multiple-comparisons problem: the maximum of 50 noisy estimates is biased upward, the winner's curse again. the fix is a held-out test set used once, after the choice is made (ch 07), or nested cross-validation.

</details>

7. a researcher pre-registers one outcome, one sample size, and one analysis. which of the lab's three p-hacking habits does that prevent?

<details><summary>answer</summary>

all three: there's only one outcome to test, no peeking (the sample size is fixed), and no choice of outlier rule after seeing the data. that's the point of pre-registration: it turns an unknown number of implicit tests into one explicit test.

</details>

## common bugs

- **forgetting the $+1$s.** symptom: p-values of exactly 0, and a test slightly anti-conservative for small $B$.
- **`>` instead of `>=`.** ties with the observed value (common with discrete data like accuracies) must count as extreme. symptom: p-values too small for discrete data.
- **resampling with replacement in a permutation test.** that's the bootstrap, a different idea (it estimates variability, not the null distribution). symptom: null values that don't correspond to any split of the data.
- **shuffling across pairs in a paired test.** symptom: no power on data where the pairing matters.
- **a two-sided test with a statistic that isn't centered at 0.** `alternative='two-sided'` counts $|T| \ge |t_{\text{obs}}|$, which is always a valid test but only *means* "as extreme in either direction" when $T$ is centered at 0 under the null (a difference of means, medians or metrics). for a ratio of variances, which sits near 1 under the null, an observed 0.10 is extremely small, yet almost every shuffle has $|T| \ge 0.10$, so the two-sided p-value is 1.0 while the one-sided 'less' p-value is at its floor (idea 1's example). symptom: huge p-values for obvious effects. fix: center the statistic (use the log of the ratio, or a difference), or use a one-sided alternative (for an asymmetric null, two times the smaller one-sided p-value is the standard two-sided recipe).
- **BH as step-down.** stopping at the first sorted p-value that fails its threshold. symptom: BH rejects nothing when the smallest p-value narrowly misses $q/m$, even when many others pass.

## going deeper

- Good (2005), *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, 3rd ed., Springer. permutation tests in depth, with many worked examples.
- Phipson & Smyth (2010), "Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn", *Statistical Applications in Genetics and Molecular Biology* 9(1): 39. the $+1$s. [doi:10.2202/1544-6115.1585](https://doi.org/10.2202/1544-6115.1585)
- Benjamini & Hochberg (1995), "Controlling the false discovery rate: a practical and powerful approach to multiple testing", *Journal of the Royal Statistical Society B* 57(1): 289–300. [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
- Simmons, Nelson & Simonsohn (2011), "False-positive psychology: undisclosed flexibility in data collection and analysis allows presenting anything as significant", *Psychological Science* 22(11): 1359–1366. the p-hacking simulation the lab is modeled on. [doi:10.1177/0956797611417632](https://doi.org/10.1177/0956797611417632)
- Wasserstein & Lazar (2016), "The ASA statement on p-values: context, process, and purpose", *The American Statistician* 70(2): 129–133. [doi:10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108)
- Dror, Baumer, Shlomov & Reichart (2018), "The hitchhiker's guide to testing statistical significance in natural language processing", ACL 2018. which tests to use when comparing ML systems. [aclanthology.org/P18-1128](https://aclanthology.org/P18-1128/)

## connections

- ← builds on: conditional probability and the base-rate fallacy (ch 04 · topic 01), simulation harnesses (topic 01), the bootstrap and paired resampling (topic 03), selection effects (ch 04 · topic 06).
- → used in: the capstone's model-comparison toolkit, comparing models and the dangers of tuning on test data (ch 07), permutation feature importance (ch 10), multi-seed comparisons and ablation tables (`mlfs.research`, ch 30), and evaluating LLMs on many benchmarks at once (ch 27).
