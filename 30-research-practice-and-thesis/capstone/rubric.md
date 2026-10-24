# 30 · capstone rubric: the thesis

grade each criterion 0–3. a solid thesis scores at least **14 of 18**, with no criterion at 0, and its tooling checks (`test_capstone.py`) all pass. the best way to use this rubric is to have a peer grade your thesis and to grade theirs, then compare: the disagreements are where the paper isn't yet clear.

| criterion | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **question** | no clear question, or one that can't fail ("explore X") | a question, but vague, or unconnected to prior work | a specific, falsifiable hypothesis with a baseline and a smallest effect that matters, motivated by the literature | all of 2, stated before the experiments (the proposal shows it), with a clear account of the gap it came from and why the answer matters either way |
| **evidence** | single runs, no intervals, untuned or missing baselines | several seeds, but intervals missing in places, or the baseline tuned less than the method | intervals everywhere (seeds sized by a power or equivalence calculation), paired comparisons, baselines with the same tuning budget, compute reported | all of 2, plus a result that holds in a second setting (scale, dataset or optimizer), and every claim in the abstract traceable to a registry query |
| **analysis** | results reported without interpretation | an interpretation, but ablations missing or claims beyond the evidence | ablations that identify which part matters, alternative explanations considered, failure cases shown | all of 2, with at least one experiment designed to *distinguish* competing explanations, and negative or inconclusive results reported as such |
| **reproducibility** | numbers typed by hand, code incomplete | code runs but some numbers can't be traced to runs | every number rendered from the registry, the README lists the commands, the reproducibility checklist answered | all of 2, plus a subset that a reader can rerun in under an hour and that regenerates its logged numbers |
| **writing** | hard to follow; figures unlabelled; claims unclear | the structure is there, but the contributions are vague or the figures mislead (truncated axes, no intervals) | clear abstract with numbers, contributions as checkable claims, figures passing `lint_figure`, LaTeX passing `check_latex`, an honest limitations section | all of 2, readable by someone outside your subfield, with every figure making one point and the paper as short as it can be |
| **process** | no plan or log | a plan, but never revisited | a task graph with a Monte Carlo forecast, a weekly log with estimates and actuals, scope cut when the forecast slipped | all of 2, plus a peer review written (scoring ≥ 6 of 7 with `score_review`) and answered point by point, and (optionally) a workshop submission |

## questions a reviewer should ask

use these when grading a peer's thesis (and your own, before anyone else does):

1. can I state the hypothesis in one sentence, and say what result would have refuted it?
2. is the baseline the one a skeptic would choose, and was it tuned as hard as the method?
3. for each number in the abstract: how many seeds, what interval, and which registry query produced it?
4. do the ablations show which part of the method matters, or just that the whole thing works?
5. what's the strongest alternative explanation of the main result, and does any experiment rule it out?
6. what would I need to rerun the main table, and how long would it take?
7. is anything claimed that the evidence doesn't support, in the abstract, the introduction or the conclusion?
8. what does the limitations section leave out?
