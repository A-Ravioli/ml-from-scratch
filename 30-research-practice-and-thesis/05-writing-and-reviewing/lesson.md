# writing & reviewing

## why this matters

a result nobody can read might as well not exist. your thesis will be judged by people who spend an hour or two on it: they'll read the abstract, look at the figures, skim the introduction for what you claim, and check one or two numbers. if the main figure has no error bars, the axis is zoomed until a 0.3-point difference fills the frame, and "figure 3" turns out to be the training curves because a `\label` was in the wrong place, they will (rightly) trust the rest less.

writing and reviewing are two sides of one skill: making and checking claims. Lipton & Steinhardt (2018) list the failures that reviewers should catch and authors should avoid: claims stronger than their evidence, speculation presented as explanation, gains attributed to the wrong component, and technical language used to impress rather than inform. this topic covers the parts of the craft that can be checked mechanically (figures that don't mislead, LaTeX that resolves, reviews that are specific and consistent), so that your attention goes to the parts that can't. it ends with an uncomfortable measurement: how much of a conference decision is noise.

## warm-up: predict first

1. a bar chart's y-axis starts at 92.8% and shows bars at 93.0% and 93.6%. how many times bigger does the difference look than it is?
2. why is the "jet" rainbow colormap a bad default for a heatmap?
3. in LaTeX, a figure has `\label{fig:a}` *before* its `\caption`. `\ref{fig:a}` prints a number without any warning. what's wrong?
4. two committees review the same papers independently and each accepts 25%. in a well-known experiment, about what fraction of one committee's accepted papers did the other reject?

## prerequisites check

- [ ] you can make matplotlib figures with labels, legends and bands (→ ch 00 · topic 04, `mlfs.utils.plot_curves`)
- [ ] you report results with intervals over seeds (→ topic 03)
- [ ] you can write a pass-3 note on a paper: claims, evidence, assumptions, questions (→ topic 01)
- [ ] you've written a short proposal (→ topic 04), which becomes your paper's introduction

## core ideas

### idea 1: figures that carry the argument

in an ML paper the figures *are* the argument: many readers look at them before (or instead of) reading the text. a good figure makes one point, and it makes it honestly. most of the ways figures fail are well understood:

- **missing labels and units.** an axis without a label ("accuracy (%)", "steps") forces the reader to guess.
- **tiny text.** a figure designed at full-screen size and shrunk to a column width has 5-pt tick labels. anything a reader must read should be at least about 8 pt at its final size.
- **unidentifiable series.** two or more curves need a legend or direct labels.
- **rainbow colormaps.** "jet" and its relatives vary in brightness non-monotonically, so they create visual boundaries where the data has none and hide real ones, and they fail in greyscale and for colour-blind readers (Borland & Taylor, 2007). perceptually uniform maps like viridis were designed to fix this.
- **truncated bars.** a bar's length encodes its value. start the axis at 92.8 and a 0.6-point difference becomes a bar four times taller. Tufte (1983) defines the **lie factor** as the size of the effect shown in the graphic divided by its size in the data; anything far from 1 misleads. (line plots and dot plots don't encode values as lengths, so zooming them is fine, as long as the scale is labelled.)
- **no uncertainty.** a curve or bar from one run, or a mean without an interval, invites the reader to over-read differences. show bands over seeds or intervals (topic 03), and say what they are.

the first five are mechanical: a linter can find them by walking the figure's artists. the last needs judgement, and so does the most important question of all: *does this figure compare what the claim compares?* for paired seeds, a plot of the per-seed differences with their interval shows the claim directly; two bars side by side make the reader do the subtraction, in their head, without the pairing.

> **spec.** `lint_figure(fig, *, min_fontsize=8.0)` → a list of strings `'axes {i}: {rule}: {message}'`, `i` the index in `fig.axes`, skipping colorbar axes (`ax.get_label() == '<colorbar>'`). a *data line* is a `Line2D` in `ax.lines` with `line.get_transform() is ax.transData` whose label doesn't start with `'_nolegend'` (so `axhline`/`axvline` and errorbar parts don't count). an axes is *plotted* if it has data lines, collections, containers or patches (an image alone isn't). rules, each reported at most once per axes: `xlabel`/`ylabel`, a plotted axes with an empty axis label; `fontsize`, any of the axis labels, title, `ax.texts` (those with non-empty text), tick labels or legend texts below `min_fontsize`; `legend`, at least 2 legend handles (`ax.get_legend_handles_labels()`), or at least 2 unlabelled series (data lines and `ErrorbarContainer`s in `ax.containers` whose labels start with `'_'`), and no legend; `colormap`, an image or a collection with data (`get_array()` not `None`) using a colormap in `RAINBOW` (given), once per such colormap name; `bars`, a `BarContainer` whose value axis (y for vertical bars, x for horizontal) doesn't include 0 (its limits are both above or both below 0).

> **🔮 predict:** the same result drawn as a zoomed bar chart, as bars from 0 with intervals, and as paired differences: which one does your linter pass, and which one would you put in the paper?

> **▶ build: stage 1.** implement `lint_figure`.

### idea 2: the paper, and its LaTeX

the standard structure of an ML paper exists because it answers the reader's questions in the order they ask them. the **abstract** says the problem, what you did, the main result *with a number*, and why it matters, in at most about 250 words. the **introduction** motivates the problem and ends with a list of contributions stated as claims a reader could check (Peyton Jones's advice: your contributions are refutable claims, and the rest of the paper is the evidence). **related work** positions you against the closest papers, honestly. the **method** is detailed enough to reimplement; the **experiments** give the setup, the main results with intervals and baselines, and ablations; **limitations** say where the claims stop. Gopen & Swan (1990) add a sentence-level rule: readers expect old information at the start of a sentence and the new, important part at the end, so put it there.

LaTeX adds its own failure modes, most of which compile with at most a warning nobody reads. a `\ref` to a label that doesn't exist prints "??"; a label defined twice silently uses one; a `\cite` key missing from the bibliography prints "[?]". a float's `\label` must come *after* its `\caption`, because `\label` records the most recent counter that was stepped, and the caption is what steps the figure counter: put the label first and `\ref` resolves without error to the number of the enclosing *section*. every figure and table should be referenced from the text (otherwise why is it there?), environments must be closed in order, and the abstract should respect its limit. all of it is regular-expression territory.

> **spec.** `check_latex(source, *, bib_keys=None, max_abstract_words=250)` → a list of strings `'{kind}: {message}'`. first remove comments (`%` to the end of the line, unless it's `\%`). kinds: `env`, for `\begin{x}`/`\end{x}` pairs checked with a stack (an `\end` that doesn't match the innermost open environment, which is then closed anyway; an `\end` with nothing open; each environment left open); `label`, each label defined more than once (naming it); `ref`, each key of `\ref`, `\eqref`, `\autoref`, `\pageref`, `\cref` or `\Cref` (comma-separated lists allowed) that isn't a defined label, named in quotes; `cite`, if `bib_keys` is given, each key of `\cite`, `\citep`, `\citet` (any `\cite…` command, optional `[...]` arguments skipped, comma-separated keys) not in it, named; `float`, for each `figure`, `figure*`, `table` or `table*` environment: no `\caption`, no `\label`, a `\label` before the `\caption`, or a label that no reference command uses (naming the label); `abstract`, an `abstract` environment with more than `max_abstract_words` words, after removing `\commands`.

> **▶ build: stage 2.** implement `check_latex`.

### idea 3: writing a review

reviewing is topic 01's third pass, written down for the authors and the area chair. a useful review has a fixed shape:

- a **summary in your own words** (not the abstract rephrased): it proves you understood the paper, and the authors can see where you didn't;
- **strengths**: what's genuinely good, so it survives revision;
- **weaknesses**, each **specific** (pointing to a location: "figure 4", "table 2", "line 120") and, if it matters for the decision, **actionable** (what experiment or change would address it);
- **questions** whose answers could change your mind;
- **scores** and a **confidence**, consistent with the text: a review that lists two major flaws and recommends a strong accept, or recommends rejection without naming a major flaw, confuses everyone.

a rubric can't judge whether your criticisms are *right*, but it can check this shape, and the shape is most of what separates a useful review from a frustrating one. write the review you'd want to receive: concrete, located, kind about the people and exacting about the claims. this topic's deliverable is one such review of a published paper from your topic 01 reading set, written as if you'd been assigned it, scored with your `score_review`, and then compared with what happened to the paper afterwards (its later versions, the follow-up work, any public reviews).

> **spec.** `score_review(review, *, abstract=None)` → a dict with one bool per name in `CRITERIA` (given) and `'score'`, the number that hold. `summary`: a string of at least 30 words and, if `abstract` is given, at most 25% of its word 4-grams (lowercased runs of `[a-z0-9]`) also occur in the abstract's. `strengths`, `questions`: non-empty lists of non-empty strings. `weaknesses_specific`: `weaknesses` is a non-empty list of dicts, each with a non-empty `text` and a `location` matching `LOCATION` (given). `weaknesses_actionable`: the same non-empty list, and every weakness whose `severity` is `'major'` has a non-empty `suggestion`. `scores_valid`: `scores` has integer (not bool) `soundness`, `presentation`, `contribution` in 1–4, `overall` in 1–10 and `confidence` in 1–5. `consistent`: the scores are valid, and it isn't the case that `overall >= 7` with 2 or more major weaknesses, or `overall <= 3` with none. never modify the review.

> **▶ build: stage 3.** implement `score_review`.

### idea 4: how noisy is review?

in 2014 the NeurIPS organizers ran an experiment: about a tenth of the submissions were reviewed by two independent committees (Cortes & Lawrence, 2021). the committees disagreed on about a quarter of those papers. that sounds tolerable until you notice the acceptance rate was also around a quarter: if one committee accepts 25% of papers and the two disagree on 25% of papers, then (on paper, problem 3) about half of the papers each committee accepted were rejected by the other.

two numbers summarize agreement between committees. the **flip rate** is the fraction of one committee's accepted papers that the other rejects. **Cohen's kappa** corrects raw agreement $p_o$ for the agreement $p_e$ two committees would reach by chance, accepting the same fraction $r$ of papers at random: $p_e = r^2 + (1 - r)^2$ and

$$\kappa = \frac{p_o - p_e}{1 - p_e},$$

which is 1 for identical decisions, 0 for chance, and negative for systematic disagreement. the lab fits a simple model (true quality plus reviewer noise) to the 2014 numbers to see how noisy reviewers must be. the lesson for your own submissions is not cynicism but calibration: a borderline paper's fate is close to a coin flip, so one rejection says less than it feels like, and the reviews matter more than the decision.

> **spec.** `committee_agreement(scores_a, scores_b, accept_rate)` → a dict with `k`, `agree`, `flip_rate` and `kappa`. both committees score the same papers (`ValueError` otherwise); each accepts its `k = round(accept_rate * n)` highest-scored papers, ties broken by order (`np.argsort(-scores, kind='stable')`); `ValueError` unless $0 < k < n$. `agree` is the fraction of papers with the same decision, `flip_rate` the fraction of A's accepted papers that B rejects, `kappa` as above with $r = k/n$.

> **🔮 predict:** in that model, how large must one reviewer's noise be, relative to the spread of true paper quality, for half of one committee's accepted papers to be rejected by the other?

> **▶ build: stage 4.** implement `committee_agreement`.

## worked example

**a figure.** bars at 93.0 and 93.5 with `ylim=(92.5, 94)`, axis labels set, default fonts: one problem, `axes 0: bars: ...`. drop the `ylim` and it passes. two unlabelled `plot` calls on a labelled axes, without a legend: `axes 0: legend: ...`; add an `axhline` and it's still one problem, because reference lines aren't data lines.

**a draft.** `figure~\ref{fig:curve}` when the label is `fig:curves` gives two problems: `ref: 'fig:curve' is referenced but never defined` and `float: figure 1 ('fig:curves') is never referenced in the text`. moving `\label{tab:main}` above `\caption{...}` in a table gives `float: table 1's \label comes before its \caption ...`.

**a review.** a review whose weaknesses are "the evaluation is weak" (no location) and a major one with no suggestion fails `weaknesses_specific` and `weaknesses_actionable`; with `overall = 8` and two major weaknesses it also fails `consistent`.

**two committees.** scores $(5, 4, 3, 2, 1, 0)$ and $(0, 1, 2, 3, 4, 5)$, accepting a third ($k = 2$): A accepts papers 0 and 1, B accepts 4 and 5. they agree only on rejecting papers 2 and 3, so $p_o = 1/3$; every A-accept is a B-reject, flip rate 1; $p_e = (1/3)^2 + (2/3)^2 = 5/9$ and $\kappa = (1/3 - 5/9)/(1 - 5/9) = -0.5$. your code must reproduce all of these.

## on paper

1. **the lie factor.** bars at 93.0 and 93.6 on an axis starting at 92.8. compute the effect in the data (relative difference) and in the graphic (relative difference of the visible bar lengths), and the lie factor.

<details><summary>solution</summary>

data: $(93.6 - 93.0)/93.0 = 0.645\%$. graphic: visible lengths $93.0 - 92.8 = 0.2$ and $93.6 - 92.8 = 0.8$, a relative difference of $(0.8 - 0.2)/0.2 = 300\%$. lie factor $300/0.645 \approx 465$. starting the axis at 0 gives a lie factor of exactly 1, which is why bars must start at 0; if the difference is then invisible, a bar chart is the wrong figure for the claim.

</details>

2. **kappa for the 2014 numbers.** with an acceptance rate of 25% for both committees and disagreement on 25% of papers, compute $p_o$, $p_e$ and $\kappa$.

<details><summary>solution</summary>

$p_o = 0.75$. $p_e = 0.25^2 + 0.75^2 = 0.625$. $\kappa = (0.75 - 0.625)/(1 - 0.625) = 1/3$. raw agreement of 75% sounds good, but most of it is agreeing to reject papers both would reject anyway; kappa says the committees agree about a third of the way from chance to perfect.

</details>

3. **disagreement and flips.** both committees accept a fraction $r$ of $n$ papers, and a fraction $f$ of A's accepted papers are rejected by B. show that the fraction of all papers on which they disagree is $2rf$, and use it with the 2014 numbers.

<details><summary>solution</summary>

A accepts $rn$ papers, of which $frn$ are rejected by B. B also accepts $rn$ papers, so it must accept $frn$ papers that A rejected (its accepted set has the same size as A's, and only $(1 - f)rn$ of them are shared). the disagreements are those $frn + frn = 2frn$ papers, a fraction $2rf$. with $r = 0.25$ and a disagreement fraction of 0.25, $f = 0.25/(2 \times 0.25) = 0.5$: half of each committee's accepted papers.

</details>

## experiment

the lab trains two MNIST models over 5 paired seeds and draws the comparison three ways (a zoomed bar chart, honest bars with intervals, the paired differences), runs your linter on each and computes the zoomed chart's lie factor. then it simulates two reviewing committees whose reviewers see true quality plus noise, uses your `committee_agreement` to find the noise level that reproduces the 2014 NeurIPS numbers, and shows how likely a paper is to be accepted as a function of its true quality. finally, your `check_latex` reads a quickly typed draft.

## check your understanding

1. the linter passes your figure. is it a good figure?

<details><summary>answer</summary>

not necessarily: the linter checks mechanics. it can't tell whether the figure compares what your claim compares, whether the uncertainty shown is the right kind (seeds vs test items), whether the y-range of a line plot hides or exaggerates, or whether the figure is needed at all. passing the linter is the floor, not the bar.

</details>

2. why are line plots allowed a zoomed y-axis but bar charts aren't?

<details><summary>answer</summary>

a bar encodes its value as a length from the baseline, so cutting the baseline changes the encoded ratio between bars. a point or line encodes its value as a position on a labelled scale; zooming changes the scale, which the tick labels show. zoomed line plots can still mislead (exaggerating noise), which is why they need bands.

</details>

3. a reviewer writes "the experiments are not convincing" and recommends rejection. what would make this criticism useful?

<details><summary>answer</summary>

a location (which table or figure), a reason (one seed per configuration; no interval; a baseline tuned less than the method), and a suggestion (5 seeds and intervals; tune the baseline with the same budget). then the authors can fix it or rebut it, and the area chair can weigh it.

</details>

4. your paper is rejected with scores 5, 6 and 4 out of 10. how should the NeurIPS experiment change your reaction?

<details><summary>answer</summary>

scores around the threshold mean the paper is borderline, and borderline decisions are close to a coin flip between committees. the useful signal is in the reviews' specific criticisms: fix what's right, clarify what was misread, and resubmit. it's equally a reason not to over-read an acceptance.

</details>

5. why must a figure's `\label` follow its `\caption`?

<details><summary>answer</summary>

`\label` records the counter most recently stepped with `\refstepcounter`. inside a float, `\caption` steps the figure (or table) counter; before the caption, the most recent step is usually the section counter. so a label placed first makes `\ref` print the section number, with no error or warning.

</details>

## common bugs

- **counting `axhline` as a series.** reference lines live in axes coordinates; check `line.get_transform() is ax.transData`.
- **treating `\%` as a comment.** the regex must not strip after an escaped percent sign: `(?<!\\)%.*`.
- **splitting citation keys without stripping spaces.** `\citep{a, b}` has a key `" b"` until you strip it.
- **counting matches of the vague-word or location regex in the wrong field.** a review's location belongs in each weakness's `location`, not anywhere in the text.
- **kappa without the chance correction.** raw agreement looks high whenever most papers are rejected by both committees.
- **unstable tie-breaking.** `np.argsort(-scores)` without `kind='stable'` can break ties differently between runs or NumPy versions.

## going deeper

- Tufte (1983), *The Visual Display of Quantitative Information*, Graphics Press (2nd ed. 2001). the lie factor and graphical integrity, chapter 2.
- Borland & Taylor (2007), "Rainbow color map (still) considered harmful", *IEEE Computer Graphics and Applications* 27(2): 14–17. [doi:10.1109/MCG.2007.323435](https://doi.org/10.1109/MCG.2007.323435)
- Gopen & Swan (1990), "The science of scientific writing", *American Scientist* 78(6): 550–558.
- Peyton Jones, "How to write a great research paper", a talk (Microsoft Research). [microsoft.com/en-us/research/academic-program/write-great-research-paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- Zobel (2014), *Writing for Computer Science*, 3rd ed., Springer.
- Lipton & Steinhardt (2018), "Troubling trends in machine learning scholarship". [arXiv:1807.03341](https://arxiv.org/abs/1807.03341)
- Cortes & Lawrence (2021), "Inconsistency in conference peer review: revisiting the 2014 NeurIPS experiment". [arXiv:2109.09774](https://arxiv.org/abs/2109.09774)
- Rogers & Augenstein (2020), "What can we do to improve peer review in NLP?", *Findings of EMNLP*. [arXiv:2010.03863](https://arxiv.org/abs/2010.03863)

## connections

- ← builds on: ch 00 · topic 04 (plotting), topic 01 (pass-3 notes are reviews in waiting), topic 03 (intervals in every figure and table), topic 04 (the proposal becomes the introduction).
- → used in: topic 06 (the thesis paper is written alongside the experiments, not after), the thesis capstone (whose rubric checks figures and claims the way this topic does), and every paper you review from now on.
