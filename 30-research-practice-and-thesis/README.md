# 30 · research practice & thesis

> from reading other people's results to producing your own.

the course so far has handed you questions: implement this, measure that, explain what you saw. research starts when nobody hands you the question, and when the answer isn't known to anyone, including the person grading you. that changes what matters. reading becomes triage: which of a thousand papers deserve an hour, and what exactly did they show? reproducing a result becomes a protocol with a verdict decided before the runs. a comparison becomes seeds, intervals and tuning budgets, because a single run is a sample of size one. a question has to be falsifiable and answerable with your compute before you spend a month on it. a paper has to survive reviewers who spend an hour on it, in a process with a lot of luck in it. and a thesis has to finish, which almost every plan underestimates. this chapter turns each of those into a habit and a small tool, then hands you the thesis: original work, a paper, and (if it goes well) a workshop submission.

**you will build:** a BibTeX parser, a linter for structured paper notes, citation graphs with a foundations-first reading order and PageRank reading queues; equivalence and effect verdicts for reproductions, a seed calculator and a reproducibility-report generator; multi-seed runs, $t$ and bootstrap intervals, paired comparisons, ablation tables and expected-best-of-$k$ curves (`mlfs.research`); a falsifiability linter, noncentral-$t$ power, compute and budget planners and a proposal checker; a figure linter, a LaTeX checker, a review scorer and committee-agreement statistics; a critical-path planner, Monte Carlo schedule risk and evidence-based forecasts; and the thesis tooling: a project skeleton, configuration hashes, a run registry, tables from the registry and a report whose numbers are rendered, never typed.
**prerequisites:** chapter 05 (bootstrap intervals, permutation tests, multiple comparisons), chapter 15 (`mlfs.torchnn`'s `MLP`, `fit`, `evaluate`, used by the labs), chapter 19 (`compute_optimal`), chapter 29's capstone (paired comparisons, the sign-flip test's floor, Holm), and the research milestones R1 and R3 (`milestones/`), which follow this chapter's reproduction protocol. a reproduction you've done yourself makes topics 02–04 land much harder.
**time · scaffolding:** L5 · ~50 h for the six topics, plus the thesis (150–250 h over three to five months). the lessons' **spec** boxes and the checks are the contract for the tools; the thesis itself has no checks, only a rubric.
**newly unlocked library components:** after 01, reference managers' BibTeX export and citation-graph services; after 02, `statsmodels`' equivalence tests (`ttost_paired`, `ttost_ind`); after 03, any experiment tracker (MLflow, Weights & Biases) and `scipy.stats.bootstrap`; after 04, `statsmodels.stats.power`; after 05, LaTeX linters such as ChkTeX; after 06, any project-planning tool that does Monte Carlo schedules. (build-before-you-use, see AUTHORING.md §9)

**data.** the labs use real MNIST from the cache (small MLPs, about a tenth of a second per run on a laptop CPU, so experiments can afford hundreds of runs), this repository's own lessons as a citation network, and simulations where a known truth is the point (citation fields with hidden quality, reproductions with a known offset, reviewing committees, thesis schedules). every lab says which is which. **hardware.** every lab runs on a laptop CPU in under a minute.

## topics

each topic directory has a `lesson.md` to read and an `exercise.py` notebook to build in (`marimo edit exercise.py`), then a `lab.py` notebook for experiments.

| # | topic | you build |
|---|---|---|
| 01 | [reading papers](01-reading-papers/) | three-pass reading; a BibTeX parser, a note linter, citation graphs, foundations-first reading orders and PageRank queues; annotated summaries of 10 papers in your area |
| 02 | [reproducing papers](02-reproducing-papers/) | the reproduction protocol used by R1–R3; equivalence and effect verdicts, seeds needed, a reproducibility-report generator |
| 03 | [experimental rigor](03-experimental-rigor/) | seeds, intervals, paired comparisons, ablation tables, tuning budgets (`mlfs.research`) |
| 04 | [finding questions](04-finding-questions/) | from a reproduction gap to a falsifiable hypothesis; power, compute and budget checks; a 2-page proposal |
| 05 | [writing & reviewing](05-writing-and-reviewing/) | paper structure, figures, LaTeX; a figure linter, a LaTeX checker, a review scorer; how noisy peer review is |
| 06 | [thesis project](06-thesis-project/) | the thesis plan: critical paths, Monte Carlo schedule risk, forecasts from your own estimate history |

## capstone

*the thesis*: original work on a question you chose, a paper whose every number is rendered from a run registry, a peer review in each direction, and an optional workshop submission. the checks cover the tooling (project skeleton, configuration hashes, run registry, tables, rendered report); the thesis is judged with the rubric. see [`capstone/brief.md`](capstone/brief.md).

## checkpoint

[`checkpoint.md`](checkpoint.md): pass it to finish the chapter, or to skip its topics and start the thesis.

## notation

| symbol | meaning |
|---|---|
| $n$, $x_i$, $\bar x$, $s$ | number of runs (seeds), one run's score, their mean, their standard deviation (with $n - 1$) |
| $t_{q,\nu}$, $z_q$ | the $q$ quantile of Student's $t$ with $\nu$ degrees of freedom, and of the standard normal |
| $\alpha$, $1 - \beta$ | a test's level and its power |
| $d_i = a_i - b_i$, $\sigma_d$ | per-seed paired difference and its standard deviation |
| $\theta_p$, $\Delta_p$, $\delta$ | a paper's reported value or effect, and the equivalence margin |
| $\lambda = \Delta\sqrt n/\sigma_d$ | noncentrality of the paired $t$ statistic |
| $k$, $v_{(i)}$ | tuning budget (trials), the $i$-th smallest of $n$ trial scores |
| $C$, $N$, $D$, $F$, $u$ | training compute ($\approx 6ND$), parameters, tokens, peak FLOP/s, utilization |
| $\tau$, $\sigma$ | spread of true effects across ideas, and a pilot's noise (winner's curse) |
| $d$, $r_v$, $\mathrm{out}(a)$ | PageRank damping, paper $v$'s score, the number of papers $a$ cites |
| $p_o$, $p_e$, $\kappa$, $r$ | observed and chance agreement, Cohen's kappa, acceptance rate |
| $o$, $m$, $p$ | optimistic, most likely and pessimistic durations of a task |

## how this connects

- builds on: the bootstrap, permutation tests and multiple comparisons (ch 05); honest evaluation (ch 07); random search and experiment logs (ch 15); `mlfs.torchnn` (ch 15); scaling laws and `compute_optimal` (ch 19); the systems view of FLOP/s (ch 26); ch 29's paired-comparison toolkit; R1 and R3, which follow topic 02's protocol.
- used later in: your thesis, R2 (which follows topic 02's protocol), and every project after the course: `mlfs.research` is the last module of the library, and the one you'll use most.

## core references

- Keshav (2007), "How to read a paper", *ACM SIGCOMM Computer Communication Review* 37(3): 83–84. [doi:10.1145/1273445.1273458](https://doi.org/10.1145/1273445.1273458)
- Pineau et al. (2021), "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)", *JMLR* 22. [arXiv:2003.12206](https://arxiv.org/abs/2003.12206)
- Henderson, Islam, Bachman, Pineau, Precup & Meger (2018), "Deep reinforcement learning that matters", *AAAI*. [arXiv:1709.06560](https://arxiv.org/abs/1709.06560)
- Bouthillier et al. (2021), "Accounting for variance in machine learning benchmarks", *MLSys*. [arXiv:2103.03098](https://arxiv.org/abs/2103.03098)
- Dodge, Gururangan, Card, Schwartz & Smith (2019), "Show your work: improved reporting of experimental results", *EMNLP*. [arXiv:1909.03004](https://arxiv.org/abs/1909.03004)
- Lipton & Steinhardt (2018), "Troubling trends in machine learning scholarship". [arXiv:1807.03341](https://arxiv.org/abs/1807.03341)
- Lakens (2017), "Equivalence tests: a practical primer for t tests, correlations, and meta-analyses", *Social Psychological and Personality Science* 8(4): 355–362. [doi:10.1177/1948550617697177](https://doi.org/10.1177/1948550617697177)
- Cortes & Lawrence (2021), "Inconsistency in conference peer review: revisiting the 2014 NeurIPS experiment". [arXiv:2109.09774](https://arxiv.org/abs/2109.09774)
