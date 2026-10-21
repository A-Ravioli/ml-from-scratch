# writing & reviewing · lab notes

numbers below are from the reference solution on an Apple M-series laptop (CPU, `OMP_NUM_THREADS=1`), real MNIST from the cache. experiment 1's ten training runs take about a second and are deterministic on the same machine; experiments 2 and 3 are deterministic. the lab takes about 8 seconds. offline, experiment 1 runs on a synthetic stand-in and its numbers mean nothing.

## experiment 1: one result, three figures

**what people predict.** a lie factor of 5–20 for the zoomed chart; that the linter flags "the bars" and nothing else.

**what happens.** the narrow MLP (128 hidden units) reaches 92.96% and the wide one (256) 93.63%, a paired difference of **+0.67 points** with a 95% interval of **[−0.32, +1.65]** over 5 seeds: suggestive, not established. the zoomed bar chart's axis starts at 92.81%, so the second bar looks 445% taller for a 0.72% (relative) difference: a **lie factor of about 620**. your linter finds 4 problems in it (no x label, no y label, 7-pt tick labels, bars not starting at 0) and none in the other two figures.

**why.** a bar's length is read as its value, so a truncated axis turns a difference of 0.67 points on bars of 93 into a difference of 0.67 on visible bars of 0.15 and 0.82; the lie factor grows without bound as the axis start approaches the smaller bar. the linter catches the mechanics. it can't catch the two worse flaws: the zoomed chart has no uncertainty, and its title claims a winner that the data (an interval spanning 0) doesn't support. the honest bar chart shows two nearly identical bars with overlapping intervals; that's truthful about the means but hides the pairing, so the reader can't see that 4 of the 5 seeds favour the wider network while one doesn't. the paired-difference plot shows exactly the quantity the claim is about, with its interval and every seed, and makes the right conclusion visible at a glance: probably a small positive effect, not yet resolved; run more seeds (topic 04 says how many).

## experiment 2: how much luck is in an accept decision?

**what people predict.** that reviewers must be extremely noisy (noise several times the spread of true quality) to reproduce the 2014 numbers, or that the experiment's result must reflect something other than noise.

**what happens.** flip rate (accepted by A, rejected by B), mean over 20 simulated conferences of 2,000 papers each, accepting 25%:

| σ (one reviewer) | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|
| flip rate | 0.00 | 0.10 | 0.20 | 0.29 | 0.36 | 0.42 | 0.48 | 0.56 | 0.64 |
| agreement | 1.00 | 0.95 | 0.90 | 0.86 | 0.82 | 0.79 | 0.76 | 0.72 | 0.68 |
| κ | 1.00 | 0.87 | 0.74 | 0.62 | 0.52 | 0.44 | 0.36 | 0.26 | 0.15 |

half of the accepted papers flip at **σ ≈ 1.5**: one reviewer's noise 1.5 times the spread of true quality, which, averaged over 3 reviewers, is a committee-level noise of $1.5/\sqrt 3 \approx 0.87$, *about the same size as the differences between papers*. at that noise the committees agree on 75% of papers with κ ≈ 0.32, matching on-paper problem 2 (1/3). and a paper in the top 10% of true quality is accepted only about 74% of the time (one simulated conference at σ = 1.5), one at the 70th–80th percentile about 44%.

**why.** with a 25% acceptance rate, the threshold sits in the middle of a dense crowd of similar papers. any noise comparable to the gaps between them reshuffles who lands above the line. that reviewers' individual noise must be about 1.5 quality-sds isn't absurd: reviewers differ in expertise, taste and attention, and "quality" is multidimensional. in this model, one reviewer's score has variance $1 + \sigma^2$, of which $\sigma^2/(1 + \sigma^2) = 2.25/3.25 \approx 69\%$ is noise at $\sigma = 1.5$; Cortes & Lawrence (2021) estimate that roughly half of the variance in real reviewers' scores is subjective. the model's figure is higher because it lumps everything into one quality number and has no discussion phase, but the two are the same order of magnitude. the model is crude (one quality number, independent Gaussian noise, no discussion phase), but it shows that the 2014 numbers don't require incompetent reviewers, only realistic disagreement plus a threshold in a crowded region. for your own submissions: the decision on a borderline paper is close to a coin flip, so read the reviews, not the decision; and as a reviewer, specific, located criticisms (stage 3) are what actually help.

## experiment 3: the draft

**what people predict.** 3–5 problems; most people find the typo'd reference and the missing citation.

**what happens.** 8 problems: an unclosed `itemize` (reported twice: `\end{document}` closes `\begin{itemize}`, and `document` is then never closed), two undefined references (`sec:result` for `sec:results`, and `tab:main`, whose label was typed `tab:mian`), a citation missing from the bibliography (`neyshabur2019`), the first figure's `\label` before its `\caption`, and two floats the text never references (the training-curves figure, and the table under its misspelled label).

**why.** LaTeX reports most of these only as warnings in a long log ("there were undefined references"), and two not at all: the label-before-caption figure *resolves*, to the section number (1) instead of the figure number, so "figure 1" happens to print correctly here by coincidence and would silently go wrong as soon as a figure is added before it; and an unreferenced float compiles fine. the misspelled table label shows how one typo produces two symptoms (an undefined reference and an orphaned float), which is typical: fix the cause and both go away. running a checker before every draft you send costs a second and saves a co-author's attention for the content.
