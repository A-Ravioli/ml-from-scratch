# reproducing papers

## why this matters

rebuild a published experiment from the paper alone and there's a real chance it won't come out the same. Raff (2019) independently reproduced about 64% of 255 ML papers; failures were often papers with unspecified details. Gundersen & Kjensmo (2018) found that most of 400 AI papers documented too little to rerun. in deep RL, Henderson et al. (2018) showed that implementation details and seeds move results as much as the differences papers report.

a failed reproduction is only useful if it's informative: if you can say *which* claim didn't hold, how sure you are, and which unreported detail is the likely culprit. a successful one is only useful if "success" meant something more than "my number looked close". both depend on decisions made *before* you run anything, which is what a protocol is for. this topic writes down the protocol that the research milestones R1 (deep learning), R2 (generative models) and R3 (reinforcement learning) follow, and builds the statistics and the report generator behind its last steps.

## warm-up: predict first

1. a paper reports 91.2% test accuracy. you run its method with 3 seeds and get 90.1% ± 1.4% (95% interval). did you reproduce it?
2. you'd consider the reproduction successful if the paper's number lies inside your 95% interval. does running *more* seeds make that criterion easier or harder to meet?
3. a paper claims method A beats B by 4 points. you find A beats B by 0.7 points, with an interval that excludes 0. reproduced, not reproduced, or something else?
4. your seed-to-seed standard deviation is about the size of the margin you'd accept. roughly how many seeds do you need before a correct reproduction is likely to be confirmed: 3, 10, or 30?

## prerequisites check

- [ ] you can read a paper in three passes and write down its claims with evidence locations (→ topic 01)
- [ ] you know the $t$ interval for a mean, and a paired difference (→ ch 05 · topic 03)
- [ ] you know what a one-sided test and a p-value are (→ ch 05 · topic 04)
- [ ] you've read `milestones/R1.md`, which follows this protocol

## core ideas

### idea 1: the protocol, and when a number counts as reproduced

the protocol has nine steps. the milestone briefs summarize it; this is the full version.

1. **read the paper three times** (topic 01): once for the claims (abstract, introduction, figures), once for the method, once for every experimental detail: architecture, initialization, optimizer and schedule, batch size, preprocessing and augmentation, number of runs, how the reported numbers and error bars were computed. record where each detail is.
2. **write the claims table before writing code.** for each claim: the paper's statement, its evidence (figure or table, number of runs, any uncertainty), the experiment you'll run, and a **margin** fixed now: the band the claim predicts, usually the paper's value ± the margin (ideas 1–2 define the verdicts against it). when a scaled-down run can't be expected to match the paper's number and only the direction carries over, the band can be one-sided: "the 90% interval for the difference lies above a stated minimum effect". commit the table, so nobody (you included) can move the goalposts after seeing results.
3. **list the gaps**: every detail the paper doesn't specify, with your choice and a one-line reason. these are the usual suspects when a result doesn't reproduce.
4. **argue the scale-down**: which quantity the claim depends on must be preserved (depth for a degradation claim, data size relative to model size for a data-efficiency claim), and which can shrink (width, epochs, image size), and why the smaller experiment still tests the same claim.
5. **implement from the paper**, with your own `mlfs` modules. read the authors' code only after your first complete run, to diagnose a discrepancy, and record every change it causes.
6. **sanity checks before the real runs**: shapes and parameter counts against the paper's tables, a tiny subset the model must overfit, a run that reproduces bit for bit from its seed, and any baseline you've already measured in the course.
7. **run with a fixed budget and enough seeds**: the briefs set the minimums (R1: at least 5 per configuration, 3 if a single run exceeds a fifth of the budget; R2: at least 3 training seeds, 5 if a run takes under 15 minutes; R3: at least 10, 5 if a single run exceeds the budget), and idea 3 tells you whether *your* claim needs more. log everything needed to redraw every figure.
8. **analyse**: curves with bands over seeds, final metrics with intervals, the paper's comparisons as differences with intervals, and a verdict per claim, judged against the margin you fixed in step 2.
9. **report** (idea 4): summary of verdicts, setup, results next to the paper's, discrepancies with experiments that test them, the reproducibility checklist, and what you'd do with 100× the compute.

the milestone briefs (R1, R2, R3) carry this protocol in summary; they come earlier in the course than this topic, so they state the one rule you need from it up front: fix a margin per claim before running, and call a claim *reproduced* only when its 90% interval lies inside the band the claim predicts.

steps 2 and 8 hide the statistical question: when does a number *count* as reproduced? the tempting criterion, "the paper's value is inside my 95% interval", is backwards. it's a test of "my result differs from the paper's", and *failing* to find a difference isn't evidence of agreement: with 3 noisy seeds the interval is wide, almost any paper value falls inside it, and the criterion is met most easily by the sloppiest reproduction. the lab measures how badly.

the right tool is an **equivalence test** (Schuirmann, 1987; Lakens, 2017). before running, fix a **margin** $\delta$: the largest difference from the paper's value $\theta_p$ that you'd still call the same result. a good margin is tied to the paper's argument (smaller than the gap between the paper's method and its baseline, or than the improvement it claims), never chosen after seeing your numbers. then compute a $(1 - 2\alpha)$ interval for your mean (90% for $\alpha = 0.05$) and compare it with the band $[\theta_p - \delta, \theta_p + \delta]$:

- the interval lies **inside** the band → **reproduced**: you've shown your value is within $\delta$ of the paper's;
- the interval lies **entirely outside** the band → **not reproduced**: you've shown it differs by more than $\delta$;
- otherwise → **inconclusive**: your experiment can't tell. the usual fix is more seeds.

why $1 - 2\alpha$? "inside the band" means two one-sided tests both reject: "my mean is at most $\theta_p - \delta$" and "my mean is at least $\theta_p + \delta$", each at level $\alpha$. each rejects exactly when the corresponding end of a $(1 - 2\alpha)$ interval clears the band's edge (on paper, problem 1). so the probability of declaring "reproduced" when the truth is just outside the margin is at most $\alpha$.

> **spec.** `equivalence_verdict(runs, paper_value, *, margin, alpha=0.05)` → a dict with `mean`, `ci` (a `(lo, hi)` tuple), `n` and `verdict`. the interval is mean $\pm\, t_{1-\alpha,\, n-1}\, s/\sqrt n$ with $s$ using `ddof=1` (a $(1 - 2\alpha)$ $t$ interval). `verdict` is `'reproduced'` if `paper_value - margin <= lo` and `hi <= paper_value + margin`, `'not reproduced'` if `hi < paper_value - margin` or `lo > paper_value + margin`, else `'inconclusive'`. `ValueError` for fewer than 2 runs, a non-finite run, `margin <= 0`, or `alpha` outside $(0, 0.5)$.

> **🔮 predict:** your reproduction is actually 2 margins off the paper's value. with 3 seeds, how often does "the paper's value is inside my 95% interval" call it reproduced?

> **▶ build: stage 1.** implement `equivalence_verdict`.

### idea 2: most claims are comparisons

papers rarely claim a number for its own sake. the claim is a comparison ("dropout reduces test error by 0.4 points"), and reproducing it means reproducing the **difference** $\Delta = \theta_A - \theta_B$. that's good news: absolute numbers depend on details you'll never match exactly (preprocessing, hardware, an unmentioned warmup), but many shift both arms together and cancel in the difference, which is why R1's brief advises reproducing "the gap, not the absolute errors".

a comparison has more possible outcomes than a single number. with the paper's effect $\Delta_p$ and margin $\delta$, and a $(1 - 2\alpha)$ interval $[\ell, u]$ for your difference:

- inside $[\Delta_p - \delta, \Delta_p + \delta]$ → **reproduced**;
- excluding 0 on the paper's side ($\ell > 0$ for a positive $\Delta_p$, $u < 0$ for a negative one) *and* disjoint from the band → **partly reproduced**: the direction holds, and the size is shown to differ (smaller *or* larger);
- excluding 0 on the paper's side but overlapping the band without fitting inside it → **direction reproduced, size inconclusive**: the direction holds, and the experiment can't tell whether the size does. this is *not* "partly": failing to show the size matches isn't evidence that it doesn't, the same mistake idea 1 warns about;
- disjoint from the band, without the direction → **not reproduced**;
- otherwise → **inconclusive**.

the margin must be smaller than $|\Delta_p|$: if the band includes 0, "reproduced" could mean "no effect", and the test can't distinguish the claim from its negation. a claim of *no* difference ($\Delta_p = 0$) has no direction, so it can only be reproduced, not reproduced or inconclusive. (for a one-sided band, a minimum effect $\Delta_{\min}$, the analogue is simpler: reproduced if $\ell > \Delta_{\min}$; the function below handles two-sided bands only.)

when both arms ran on the same seeds (seed $i$ fixed the same data order and subset for both), use the paired differences, as in topic 03 and ch 29: the shared noise cancels. when they didn't, use **Welch's** interval: $\bar a - \bar b \pm t_{1-\alpha,\nu}\sqrt{s_a^2/n_a + s_b^2/n_b}$ with the Welch–Satterthwaite degrees of freedom $\nu = (s_a^2/n_a + s_b^2/n_b)^2 / \big(\tfrac{(s_a^2/n_a)^2}{n_a - 1} + \tfrac{(s_b^2/n_b)^2}{n_b - 1}\big)$, which doesn't assume equal variances.

> **spec.** `effect_verdict(runs, baseline_runs, paper_effect, *, margin, alpha=0.05, paired=True)` → a dict with `diff`, `ci`, `n` and `verdict`, for runs − baseline. paired: `ValueError` unless the lengths match; `diff` and the $(1-2\alpha)$ $t$ interval come from the per-seed differences ($n - 1$ dof); `n` is the number of pairs. unpaired: `diff` = difference of means, Welch's interval as above (if both variances are 0, use $n_a + n_b - 2$ dof); `n` is the smaller group's size. `verdict` is one of the five strings in bold above (`'reproduced'`, `'partly reproduced'`, `'direction reproduced, size inconclusive'`, `'not reproduced'`, `'inconclusive'`), checked in the order listed; "disjoint" means `hi < paper_effect - margin` or `lo > paper_effect + margin`. `ValueError` for fewer than 2 runs on a side, non-finite runs, `margin <= 0`, a nonzero `paper_effect` with `margin >= abs(paper_effect)`, or `alpha` outside $(0, 0.5)$.

> **▶ build: stage 2.** implement `effect_verdict`.

### idea 3: how many seeds before you start

an equivalence test can say "inconclusive" forever if the experiment is too small. if your runs scatter with standard deviation $\sigma$, the $(1-2\alpha)$ interval's half-width is about $t_{1-\alpha, n-1}\,\sigma/\sqrt n$, and the interval can only fit inside a band of half-width $\delta$ if that's smaller than $\delta$. even then, the interval's centre wobbles. suppose your reproduction is exactly right (true mean $= \theta_p$). then $\bar x \sim N(\theta_p, \sigma^2/n)$, and "reproduced" needs $|\bar x - \theta_p| + t\,\sigma/\sqrt n \le \delta$ (treating $s$ as if it were $\sigma$). that has probability $2\Phi\big((\delta - t\sigma/\sqrt n)\sqrt n/\sigma\big) - 1$, which is at least the desired **power** $1 - \beta$ when

$$\big(t_{1-\alpha,\,n-1} + z_{(1 + \text{power})/2}\big)\,\frac{\sigma}{\sqrt n} \le \delta.$$

the smallest such $n$ is how many seeds you need for a correct reproduction to be *confirmed* with that probability. it depends only on $\sigma/\delta$: at $\sigma = \delta$, with $\alpha = 0.05$ and 80% power, it's 10. you get $\sigma$ from a pilot of a few runs (or from the paper's error bars, if it reports how they were computed). the approximation ignores the noise in $s$; the lab checks how much that matters.

this turns the briefs' minimum seed counts into a number for *your* claim, and it belongs in the claims table: if the answer is 60 seeds and you can afford 10, widen the margin with a stated reason, or plan to report an honest "inconclusive".

> **spec.** `seeds_needed(sd, margin, *, alpha=0.05, power=0.8, max_seeds=1000)` → the smallest integer $n \ge 2$ with $(t_{1-\alpha, n-1} + z_{(1+\text{power})/2})\,\text{sd}/\sqrt n \le \text{margin}$ ($z$ from `scipy.stats.norm.ppf`). search $n = 2, 3, \dots$ up to and including `max_seeds`; `ValueError` if none works, or if `sd` or `margin` isn't positive, or `power` isn't in $(0, 1)$.

> **▶ build: stage 3.** implement `seeds_needed`.

### idea 4: the report

the report is the reproduction: code without it is a pile of runs. it leads with a **summary table**: one row per claim, with the paper's value and margin, yours with an interval, the number of seeds, and the verdict. a reader who stops there should know what held. then one short section per claim (statement, where the paper's evidence is, the criterion), the **gaps** table (each unreported detail, your choice, why), and the **reproducibility checklist**: a fixed list of what a rerun needs, adapted from the ML reproducibility checklist of Pineau et al. (2021), with an answer to every item. unanswered items stay visible; a checklist that silently drops the awkward questions isn't one.

written by hand, reports drift: a number from an older run, a verdict with a different margin than the claims table's. generating the report from the claims table and the logged runs removes that class of error.

> **spec.** `reproduction_report(title, claims, results, *, gaps=(), checklist=None, alpha=0.05)` → `(markdown, verdicts)`. each claim is a dict with `id`, `kind` (`'value'` or `'effect'`; default `'value'`), `statement`, `evidence`, `metric`, `paper` (the value or effect), `margin` and optionally `paired` (default `True`, for effects). `results[id]` is a sequence of runs for a value claim, or a pair `(runs, baseline_runs)` for an effect. verdicts come from your stage-1 and stage-2 functions with `alpha`; `verdicts` maps each id to its verdict. the markdown starts with `# {title}`, then `## summary` and a table with the header `| claim | metric | paper | ours (mean, L CI) | n | verdict |` (`L` = `f"{100 * (1 - 2 * alpha):g}%"`) and one row per claim: `| id | metric | {paper:.3g} ± {margin:.3g} | {mean:.3g} [{lo:.3g}, {hi:.3g}] | n | **verdict** |`, with both numeric cells prefixed `Δ = ` for effects (and `diff` in place of `mean`). then `## claims` with a `### {id}: {statement}` section per claim containing `evidence in the paper: {evidence}`; `## gaps` with a table `| detail the paper doesn't specify | our choice | why |` (each gap a dict with non-empty `detail`, `choice`, `reason`), or `none listed.`; and `## reproducibility checklist` with the line `{k} of 10 items answered.` and one line per `CHECKLIST` item (given in the setup cell, in its order): `- [x] {item}: {answer}` or `- [ ] {item}: **unanswered**` (a blank answer counts as unanswered). `ValueError` for duplicate claim ids, a claim without results, an unknown kind, an incomplete gap, or a checklist key that isn't a `CHECKLIST` id.

> **🔮 predict:** a hypothetical paper's claim depends on one hyperparameter it doesn't report. if you sweep five plausible values of it, how many different verdicts will the same claim get?

> **▶ build: stage 4.** implement `reproduction_report`.

## worked example

**a number.** five seeds give $(0.70, 0.72, 0.71, 0.73, 0.69)$: mean 0.71, $s = 0.0158$, $t_{0.95,4} = 2.132$, half-width $2.132 \times 0.0158/\sqrt 5 = 0.0151$, so the 90% interval is $[0.695, 0.725]$. against a paper value of $0.71 \pm 0.02$ (band $[0.69, 0.73]$): reproduced. against $0.75 \pm 0.02$ (band $[0.73, 0.77]$): the interval ends below 0.73, not reproduced. against $0.735 \pm 0.02$ (band $[0.715, 0.755]$): it straddles the edge, inconclusive. with a margin of 0.017 the 90% interval still fits (reproduced), though the 95% one, $[0.690, 0.730]$, wouldn't.

**a comparison.** paired differences $(0.02, 0.03, 0.01, 0.01, 0.03)$: mean 0.020, $s = 0.010$, 90% interval $[0.0105, 0.0295]$. against a claimed $+0.02 \pm 0.012$: reproduced. against $+0.05 \pm 0.015$: the interval is above 0 and entirely below 0.035, partly reproduced (the effect is smaller). against $+0.04 \pm 0.0106$ (band $[0.0294, 0.0506]$): above 0, but its top overlaps the band, so direction reproduced, size inconclusive. against $-0.02 \pm 0.01$: not reproduced.

**seeds.** with $\sigma = \delta$, $\alpha = 0.05$, power 0.8 ($z_{0.9} = 1.282$): $n = 9$ gives $(1.860 + 1.282)/3 = 1.047 > 1$; $n = 10$ gives $(1.833 + 1.282)/\sqrt{10} = 0.985 \le 1$. so 10 seeds. your code must reproduce all of these.

## on paper

1. **TOST and the $1 - 2\alpha$ interval.** the two one-sided $t$ tests reject "$\mu \le \theta_p - \delta$" when $(\bar x - (\theta_p - \delta))/(s/\sqrt n) > t_{1-\alpha, n-1}$, and "$\mu \ge \theta_p + \delta$" when $(\bar x - (\theta_p + \delta))/(s/\sqrt n) < -t_{1-\alpha, n-1}$. show that both reject exactly when the $(1 - 2\alpha)$ interval lies inside the band.

<details><summary>solution</summary>

the first inequality rearranges to $\bar x - t_{1-\alpha,n-1}\, s/\sqrt n > \theta_p - \delta$, i.e. $\ell > \theta_p - \delta$ for the interval's lower end $\ell$. the second to $\bar x + t_{1-\alpha,n-1}\, s/\sqrt n < \theta_p + \delta$, i.e. $u < \theta_p + \delta$. both together: $[\ell, u]$ inside the band. the interval $\bar x \pm t_{1-\alpha,n-1}\, s/\sqrt n$ has coverage $1 - 2\alpha$ (each tail has probability $\alpha$). so a 90% interval gives a 5%-level equivalence test, and using a 95% interval would make it a 2.5% test: valid but needlessly conservative.

</details>

2. **why the naive criterion rewards small experiments.** with true mean $\theta_p + \Delta$ (a reproduction that's off by $\Delta$) and known $\sigma$, the naive rule "the paper's value is inside my 95% interval" succeeds when $|\bar x - \theta_p| \le 1.96\,\sigma/\sqrt n$. write its success probability as a function of $\Delta\sqrt n/\sigma$, and evaluate it for $\Delta = 2\sigma$ at $n = 3$ and $n = 20$.

<details><summary>solution</summary>

$\bar x - \theta_p \sim N(\Delta, \sigma^2/n)$, so with $c = \Delta\sqrt n/\sigma$ the probability is $\Phi(1.96 - c) - \Phi(-1.96 - c)$. at $n = 3$, $c = 3.46$: $\Phi(-1.50) - \Phi(-5.42) \approx 0.067$. at $n = 20$, $c = 8.94$: essentially 0. with $\Delta = \sigma$ (one margin off, when $\sigma = \delta$): $c = 1.73$ at $n = 3$ gives 0.59. using the $t$ quantile instead of 1.96 (as a real 3-seed interval must: $t_{0.975,2} = 4.30$) makes it much worse, which is what the lab measures. the success probability *decreases* with $n$: the criterion is a difference test, and the less data you have, the less able you are to see the difference.

</details>

## experiment

the lab simulates thousands of reproductions with a known truth to compare the naive criterion with your equivalence verdict as the number of seeds grows, and checks by simulation whether `seeds_needed` delivers the power it promises. then it runs a small real reproduction on MNIST: a hypothetical paper's claim about Adam and SGD whose one unreported detail is SGD's learning rate, swept over five values, with the report written by your `reproduction_report`.

## check your understanding

1. why must the margin be fixed in the claims table, before any runs?

<details><summary>answer</summary>

the margin decides the verdict. chosen after seeing the results, it can always be made just wide enough to say "reproduced" (or narrow enough to say "not"). fixing it in advance, with a reason tied to the paper's argument, is what makes the verdict evidence rather than opinion.

</details>

2. your equivalence verdict says "inconclusive" with 5 seeds. a colleague says "so it didn't reproduce". what do you reply?

<details><summary>answer</summary>

inconclusive means the experiment couldn't tell, not that the paper is wrong. run more seeds (`seeds_needed` says how many), or report it as inconclusive with the interval, which still shows the plausible range.

</details>

3. a paper claims A beats B by 4 points, and your claims table fixed a margin of 1.5. you find +0.7 points with an interval of $[+0.2, +1.2]$. what's the verdict, and what should the report do next? what if the interval had been $[+0.2, +3.0]$?

<details><summary>answer</summary>

partly reproduced: the interval excludes 0 and lies entirely below the band $[2.5, 5.5]$, so the direction holds and the size is shown to be smaller. with $[+0.2, +3.0]$ the interval would overlap the band: direction reproduced, size inconclusive, and the next step would be more seeds, not a search for a cause. for the partly-reproduced case the report should look for the cause: which gap (an unreported hyperparameter, a different data split, a shorter schedule from the scale-down) could shrink the effect, and run at least one experiment that tests it, as step 9's discrepancy section asks.

</details>

4. why is reproducing a difference usually easier than reproducing an absolute number?

<details><summary>answer</summary>

many discrepancies (preprocessing, hardware, the data version) shift both arms alike and cancel in the difference, and paired seeds cancel seed noise too.

</details>

5. when would you choose `paired=False` for a comparison?

<details><summary>answer</summary>

when the two arms didn't share seeds in a meaningful way: one arm's numbers come from the paper or another codebase, the arms use different data pipelines so a seed doesn't fix the same data order, or the numbers of runs differ. pairing arbitrary runs pretends to a correlation that isn't there.

</details>

## common bugs

- **the 95% interval for an equivalence test.** valid but at level 2.5%, so you'll say "inconclusive" more than you need to. use $t_{1-\alpha}$, not $t_{1-\alpha/2}$.
- **"not reproduced" only on one side.** a reproduction can overshoot the paper as well as undershoot it; check both band edges.
- **"partly reproduced" for an unresolved size.** an interval that confirms the direction but overlaps the band hasn't shown the size differs; that's "direction reproduced, size inconclusive".
- **"partly reproduced" for the wrong direction.** an interval that excludes 0 on the *opposite* side of the paper's effect is "not reproduced", not "partly". check the sign of the paper's effect.
- **a margin that includes 0.** with $\delta \ge |\Delta_p|$ "no effect" counts as a reproduction. refuse it.

## going deeper

- Pineau et al. (2021), "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)", *JMLR* 22. the ML reproducibility checklist. [arXiv:2003.12206](https://arxiv.org/abs/2003.12206)
- Raff (2019), "A step toward quantifying independently reproducible machine learning research", *NeurIPS*. [arXiv:1909.06674](https://arxiv.org/abs/1909.06674)
- Gundersen & Kjensmo (2018), "State of the art: reproducibility in artificial intelligence", *Proceedings of the AAAI Conference on Artificial Intelligence* 32(1). a survey of 400 papers from IJCAI and AAAI.
- Lakens (2017), "Equivalence tests: a practical primer for t tests, correlations, and meta-analyses", *Social Psychological and Personality Science* 8(4): 355–362. [doi:10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177)
- Schuirmann (1987), "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability", *Journal of Pharmacokinetics and Biopharmaceutics* 15(6): 657–680. [doi:10.1007/BF01068419](https://doi.org/10.1007/BF01068419)
- Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), "Deep reinforcement learning that matters", *AAAI*. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Lucic, Kurach, Michalski, Gelly & Bousquet (2018), "Are GANs created equal? A large-scale study", *NeurIPS*. a reproduction-style study of generative models, useful for R2. [arXiv:1711.10337](https://arxiv.org/abs/1711.10337)
- Bouthillier, Laurent & Vincent (2019), "Unreproducible research is reproducible", *ICML*. [proceedings.mlr.press/v97/bouthillier19a.html](https://proceedings.mlr.press/v97/bouthillier19a.html)

## connections

- ← builds on: topic 01 (three-pass reading, claims with evidence locations), ch 05 · topics 03–04 ($t$ intervals, one-sided tests), ch 29's capstone (paired differences).
- → used in: R1, R2 and R3 follow this protocol; topic 03 builds the general multi-seed tools; topic 04 turns a reproduction's discrepancies into research questions; the thesis capstone reports its own claims with the same discipline.
