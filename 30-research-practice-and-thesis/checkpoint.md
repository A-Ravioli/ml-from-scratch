# 30 · checkpoint: research practice & thesis

pass this to finish the chapter, or to skip its topics and go straight to the thesis. no notes, no code from the chapter, no internet. a pen, paper and a calculator (or a table of $t$ and normal quantiles) are fine for parts 1 and 2.

## 1. concept check

answer each in two or three sentences, then check.

1. what are the three passes of Keshav's method, and what do you have at the end of each?

<details><summary>answer</summary>

pass 1 (5–10 minutes: title, abstract, introduction, headings, conclusion, figures at a glance) gives a decision and the five Cs: category, context, correctness, contributions, clarity. pass 2 (up to an hour, skipping proofs) gives the argument and its evidence, figure by figure, and the references to follow up. pass 3 (hours: reconstructing the work) gives the hidden assumptions, the missing details and the gaps between claims and evidence.

</details>

2. why do raw citation counts favour old papers even in a field where every paper is equally good?

<details><summary>answer</summary>

exposure: an old paper could be cited by every paper published after it, a new one by almost none. and cumulative advantage: cited papers get found and cited more, so early leads snowball. dividing by age doesn't remove the snowball, and PageRank inherits the bias (credit flows down citation chains to the oldest papers). comparing a paper only with its contemporaries does much better.

</details>

3. a reproduction with 3 seeds finds 90.1% ± 1.4% (95% interval) against a paper's 91.2%. a colleague says "reproduced: the paper's number is inside our interval". what's wrong?

<details><summary>answer</summary>

that criterion is a difference test, and failing to find a difference with 3 noisy seeds isn't evidence of agreement: the wider the interval, the more easily it "reproduces". the right test is equivalence: fix a margin before running and check that a $(1 - 2\alpha)$ interval lies *inside* paper ± margin. with 3 seeds and this noise the honest verdict is almost always "inconclusive".

</details>

4. what are the five possible verdicts for a reproduced comparison ("A beats B by $\Delta$"), and what separates "partly reproduced" from "direction reproduced, size inconclusive"?

<details><summary>answer</summary>

reproduced (the interval of your difference lies inside $\Delta \pm$ margin); partly reproduced (the interval excludes 0 on the paper's side *and* is disjoint from the band: the direction holds and the size is shown to differ); direction reproduced, size inconclusive (it excludes 0 on the paper's side but overlaps the band: the size is unresolved, not refuted); not reproduced (disjoint from the band without the direction); and inconclusive (anything else). the margin must be smaller than $|\Delta|$, or "no effect" could count as a reproduction.

</details>

5. why is $t_{0.975, 4} = 2.776$ and not 1.96 the right multiplier for a 95% interval from 5 runs, and what happens if you use 1.96?

<details><summary>answer</summary>

with 5 runs the standard deviation is itself estimated, and $(\bar x - \mu)/(s/\sqrt 5)$ follows a $t$ distribution with 4 degrees of freedom, whose tails are heavier than the normal's. with 1.96 the interval covers the truth only about 88% of the time.

</details>

6. two methods were run on the same 8 seeds, where each seed fixes the data order and subset for both. why does pairing by seed matter, and what's the formula behind it?

<details><summary>answer</summary>

$\operatorname{Var}(a - b) = \operatorname{Var}(a) + \operatorname{Var}(b) - 2\operatorname{Cov}(a, b)$. a shared seed makes $a_i$ and $b_i$ positively correlated, so the per-seed differences are much less noisy than either method alone, and a paired interval can resolve an effect that two overlapping unpaired intervals hide.

</details>

7. what does `expected_best` at $k = 1$ measure, and why can two methods' curves cross?

<details><summary>answer</summary>

$k = 1$ is the average score of a random configuration from the search space: robustness to hyperparameters. the curves cross when one method is easier to tune (better at small $k$) and the other has a higher ceiling (better at large $k$), as SGD and Adam did in topic 03's lab. comparing one at large $k$ with the other at small $k$ measures tuning effort, not methods.

</details>

8. what makes a hypothesis falsifiable, in the sense of topic 04's checker?

<details><summary>answer</summary>

it names a method and a different baseline, a metric and a setting, a direction and a smallest effect that matters, the result that would refute it, and the number of seeds, and it predicts rather than promising to "explore" or "investigate". a statement no possible result could contradict isn't a hypothesis.

</details>

9. why shouldn't the effect you observed in a pilot become the `min_effect` you size the experiment for?

<details><summary>answer</summary>

a small noisy pilot that looks promising is biased upwards (the winner's curse: you noticed it *because* the noise was favourable), so sizing for it leaves the real experiment underpowered. `min_effect` should be the smallest effect that would matter, chosen for what it means.

</details>

10. what is a bar chart's lie factor, and why can line plots be zoomed while bar charts can't?

<details><summary>answer</summary>

the size of an effect as shown divided by its size in the data. a bar encodes its value as a length from the baseline, so truncating the axis inflates the visual ratio (by hundreds for a sub-point difference near 93%); a point or line encodes value as position on a labelled scale, so zooming changes the scale, which the ticks show.

</details>

11. why must a LaTeX `\label` come after `\caption` inside a float?

<details><summary>answer</summary>

`\label` records the counter most recently stepped, and `\caption` is what steps the figure (or table) counter. before the caption, the last stepped counter is usually the section's, so `\ref` silently prints the section number.

</details>

12. in the 2014 NeurIPS experiment, the two committees disagreed on about a quarter of papers with a quarter accepted. why does that imply that about half of each committee's accepted papers were rejected by the other?

<details><summary>answer</summary>

with acceptance rate $r$ and flip rate $f$ (the fraction of A's accepts rejected by B), B must accept as many papers A rejected as A accepted that B rejected, so the disagreements are $2rf$ of all papers. $2 \times 0.25 \times f = 0.25$ gives $f = 0.5$.

</details>

13. a plan of tasks each estimated at their most likely duration is almost never met. give the two reasons.

<details><summary>answer</summary>

right-skew: for tasks whose pessimistic estimate is much further from the likely one than the optimistic is, the mean and median exceed the mode, and along a critical path those gaps add up. merge bias: a task that waits for several parallel tasks starts when the slowest finishes, and the maximum of uncertain durations exceeds any one's likely value.

</details>

14. why resample your own estimate-to-actual ratios rather than multiply the plan by their average?

<details><summary>answer</summary>

a fixed factor shifts the forecast but gives it no spread, so it can't say how likely a date is. resampling keeps the variability, skew and tails of your own overruns, per task, so the forecast inherits realistic tails and merge bias, and gives quantiles of the finish date.

</details>

## 2. by hand (no computer)

1. **an equivalence verdict.** five runs give $(0.70, 0.72, 0.71, 0.73, 0.69)$; $t_{0.95, 4} = 2.132$. compute the 90% interval and the verdict against a paper value of 0.75 with a margin of 0.02, and against 0.72 with a margin of 0.02.

<details><summary>solution</summary>

mean 0.71, $s = 0.0158$, half-width $2.132 \times 0.0158/\sqrt 5 = 0.0151$: interval $[0.695, 0.725]$. against $0.75 \pm 0.02 = [0.73, 0.77]$: the interval lies entirely below, not reproduced. against $0.72 \pm 0.02 = [0.70, 0.74]$: the lower end 0.695 is outside, so it straddles the edge: inconclusive.

</details>

2. **expected best of $k$.** four tuning trials scored $(0.80, 0.85, 0.90, 0.95)$. compute the expected best of $k = 2$ trials drawn with replacement.

<details><summary>solution</summary>

weights $(i/4)^2 - ((i-1)/4)^2$ for $i = 1..4$: $1/16, 3/16, 5/16, 7/16$. expected best $= (0.80 \cdot 1 + 0.85 \cdot 3 + 0.90 \cdot 5 + 0.95 \cdot 7)/16 = (0.80 + 2.55 + 4.50 + 6.65)/16 = 14.50/16 = 0.906$.

</details>

3. **seeds and power.** paired differences have sd 0.6 points. using the normal approximation $n \approx ((z_{1-\alpha} + z_{\text{power}})\,\sigma_d/\Delta)^2$ with $z_{0.95} = 1.645$ and $z_{0.8} = 0.842$, how many seeds detect $\Delta = 0.3$ one-sided with 80% power? why is the exact answer larger?

<details><summary>solution</summary>

$((1.645 + 0.842) \times 2)^2 = 4.974^2 = 24.7$, so about 25. the exact noncentral-$t$ answer is 27: $s_d$ is estimated, so the critical value is a $t$ quantile (larger than 1.645) and the statistic's distribution is wider than normal.

</details>

4. **a critical path.** tasks: A (3 weeks), B (2, after A), C (4, after A), D (1, after B and C). compute each task's earliest finish and slack, the total, and the critical path.

<details><summary>solution</summary>

A 0–3, B 3–5, C 3–7, D 7–8: total 8. latest finishes: D 8, so B and C by 7, A by 3. slack: B $7 - 5 = 2$, everything else 0. critical path A → C → D.

</details>

## 3. blind rebuild

in 60 minutes, with no notes, rebuild in a fresh copy of the topic notebooks:

- `mean_ci`, `compare_runs` and `ablation_table` (topic 03, stages 2–4), and
- `equivalence_verdict` and `effect_verdict` (topic 02, stages 1–2).

check them with the topics' own tests:

```bash
python -m pytest 30-research-practice-and-thesis/03-experimental-rigor/test_exercise.py -k "stage2 or stage3 or stage4"
python -m pytest 30-research-practice-and-thesis/02-reproducing-papers/test_exercise.py -k "stage1 or stage2"
```

## 4. readiness

move on to the thesis when you score at least 80% on parts 1 and 2 and the blind rebuild passes every check. the same checkpoint works as a placement test: if you pass it cold, skim the labs anyway (each has a result worth seeing: a field in which all ten most-cited papers are among the oldest fifth, a "within my error bars" criterion that reproduces wrong results more often with fewer seeds, an unreported learning rate that turns one claim into four verdicts, a single-seed ablation table that's right about half the time, a winning pilot whose true effect is a seventh of what it showed, a 600× lie factor, a review process whose noise matches the 2014 NeurIPS experiment, and a "likely" thesis plan met once in two thousand tries), then start your thesis with the capstone brief.
