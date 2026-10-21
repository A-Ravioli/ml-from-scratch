# 30 · capstone: the thesis

## goal

the last project of the course is a piece of original research: a question you chose, answered by experiments you designed, written up as a paper that someone who has never met you could evaluate, rerun and build on. if it turns out well, you submit it to a workshop.

"original" doesn't mean "a new architecture". most good first theses are one of these:

- **a careful empirical study** of a precise question: *does X still help when Y changes?*, *where does this scaling trend break?*, *which of these three explanations of a known effect survives an ablation?*
- **a reproduction that went somewhere**: a claim that only partly reproduced (topic 02), followed by the experiments that explain why.
- **a negative result**, honestly established: a widely used trick that doesn't help in a well-defined setting, with enough seeds that "doesn't help" means something.
- **a mechanism**: an analysis of *how* a small model does something (training dynamics, representations, a circuit, as in chapter 28).
- **a methodological contribution**: a better evaluation protocol, a fix to a flawed benchmark, a cheaper estimate of something expensive.
- **a new method**, compared against baselines tuned with the same budget, with ablations that show which part matters.

what all of them share is a falsifiable hypothesis stated before the experiments, evidence sized to test it, and a write-up whose claims are no stronger than that evidence.

## the path

the chapter's topics are the stages of the thesis. each produces something you keep.

1. **reading** (topic 01): 10 annotated papers in your area (notes passing `check_note` at pass 2 or better, 3 of them at pass 3), and the gap you found.
2. **the proposal** (topic 04): two pages in the Heilmeier format, `check_proposal(...) == []`, with the hypothesis, the pilot that sized the seeds, and the compute budget. a peer reads it before you start.
3. **the plan** (topic 06): the task graph with three-point estimates, its `schedule_risk` 80% date no later than your deadline (cut scope until it is), and a weekly log with (estimate, actual) pairs. re-forecast with `forecast` every month.
4. **the work**: a project created with your `make_project`; every run logged in your `RunRegistry` (no result exists that isn't in it); seeds sized by `seeds_for_power` (topic 04) or, for an equivalence claim, `seeds_needed` (topic 02); baselines tuned with the same budget as your method, reported with `expected_best` curves if tuning matters (topic 03); ablations with `ablation_table`.
5. **the paper** (topic 05): 6–8 pages plus references and appendix, in a workshop's format. every number rendered from the registry with your `render_report`; every figure passing `lint_figure`, with intervals over seeds; the LaTeX passing `check_latex` (or the Markdown equivalent); a limitations section; and a reproducibility checklist (topic 02's `CHECKLIST`) answered.
6. **review**: swap papers with a peer. write a review that scores at least 6 of 7 with your `score_review`, and answer the review you receive point by point.
7. **(optional) a workshop submission**: find a workshop at a major conference whose topic fits (workshops on negative results, reproducibility, or a specific subfield are good homes for a first paper), check its call for papers *for that year*: page limit, template, anonymity rules, whether it's archival, its dual-submission policy and its deadline. adapt the paper, anonymize it, and submit. reviews you get back are a deliverable too.

## what you build here

`capstone.py` holds the tooling the path relies on, with its checks green. the spec, stage by stage:

- **stage 1 · `make_project(root, title, *, question, author='')`** creates, under `root` (making directories as needed), exactly the files in `SKELETON`: `README.md` (first line `# {title}`, then the question, an author line only if `author` is given, and sections on reproducing and on the layout), `proposal.md` (the Heilmeier headings), `plan.json` (a starter task graph in topic 06's format), `log.md` (a weekly-log template), `paper/paper.md` (a `# {title}` line and one `## ` heading per name in `PAPER_SECTIONS`, in order), and empty `configs/.gitkeep`, `runs/.gitkeep`, `figures/.gitkeep`. it **never overwrites**: existing files are skipped. returns the created files' relative paths (forward slashes), sorted. `ValueError` (before writing anything) for an empty or blank `title` or `question`.
- **stage 2 · `config_hash(config)`**: the first 12 hex digits of the SHA-256 of `json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)` encoded as UTF-8, after converting NumPy scalars to Python numbers and NumPy arrays and tuples to lists (at any depth). key order never matters; `1` and `1.0`, or `True` and `1`, are different configurations. `nan` raises `ValueError` and unserializable objects `TypeError` (as `json.dumps` does).
- **stage 3 · `RunRegistry(path)`**: an append-only JSONL file of runs, loaded on construction if it exists. `log(config, seed, metrics, *, name=None, notes='')` appends one line, a JSON record with `id` (`'{hash}-{seed}'`), `config_hash`, `name` (default: the hash), `config`, `seed`, `metrics` (floats) and `notes`, creates parent directories as needed, and returns the record. `ValueError`, writing nothing, for a seed that isn't an integer (bools included), empty metrics, a metric that isn't a finite number, or a configuration and seed already logged (in any key order). `runs(name=None, config_hash=None)` returns the matching records in logging order (both filters apply together). `values(name, metric)` returns `{seed: value}` sorted by seed for the runs called `name`; `ValueError` if there are none, if they span more than one configuration, or if some lack the metric. `len(registry)` is the number of runs.
- **stage 4 · `results_table(registry, metric, names, *, baseline=None, alpha=0.05, higher_is_better=True, fmt='{:.3f}')`**: your `mlfs.research.ablation_table` over the named configurations' values on the seeds *all* of them share, in seed order, with the other arguments passed through. if some runs were left out, append `\n\n*paired on {k} shared seeds; left out: {name} seeds {s1, s2}; ...*` (names in the order given, only those with left-out seeds). `ValueError` if fewer than 2 seeds are shared.
- **stage 5 · `render_report(template, registry, *, alpha=0.05, fmt='.3f')`** replaces every `{{kind:args}}` placeholder (`kind` a word, `args` without braces): `{{mean:name:metric}}` → `f"{m:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"` from `mean_ci(values, alpha=alpha)`; `{{n:name:metric}}` → the number of runs; `{{diff:name:baseline:metric}}` → `f"{m:+{fmt}} [{lo:+{fmt}}, {hi:+{fmt}}]"` from `mean_ci` of the per-seed differences on the shared seeds (at least 2); `{{table:metric:name1,name2,...}}` → `results_table` with the first name as the baseline, `alpha`, and `fmt` as its format (`'{:' + fmt + '}'`). any other kind, a wrong number of arguments, or an unknown name or metric raises `ValueError`. everything else in the template is left as it is.

## data and compute

any data you can get and document. the course's cached datasets (MNIST, CIFAR-10, tiny Shakespeare, SMS spam, Olivetti faces, California housing; `mlfs.datasets`) cover many good questions; anything else, download once, record its version and checksum in the README, and never from inside a notebook that others will rerun. the compute budget is a laptop (topic 04): plan for at most a few dozen hours of total compute, with every configuration small enough that a run takes minutes and a reader can rerun a representative subset in under an hour. state the hardware, the total compute and the per-run time in the paper.

## configurations

| config | where | what runs | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, offline) | `MLFS_OFFLINE=1` | the tooling checks on synthetic registries in temporary directories; the worked example on a synthetic stand-in, 3 seeds × 4 configurations, 1 epoch on 600 images | a few seconds |
| **full** (the worked example) | default | the worked example on real MNIST, 10 seeds × 4 configurations, 3 epochs on 5,000 images | about 10 seconds |
| **your thesis** | your project directory | your experiments, logged in your registry | your budget |

## a worked example

the reference solution runs a thesis-*shaped* experiment end to end with the tooling, so you can see the pattern: a skeleton in a scratch directory, every run logged, a table and a paragraph rendered from the registry, no number typed by hand. the question comes from topic 04's pilot: *does label smoothing help a small MLP trained briefly on MNIST, and how does the effect depend on its strength?* a $784 \to 128 \to 10$ MLP, SGD with momentum 0.9 and learning rate 0.05, 3 epochs on 5,000 training images, 10 seeds per configuration (each seed fixes the subset, the initialization and the data order), scored on the 10,000 test images. measured with the reference solution on the cached data:

| config | test accuracy, mean ± 95% CI (%) | n | Δ vs no smoothing ± 95% CI (points) |
|---|---|---|---|
| no smoothing | 92.85 ± 0.41 | 10 | (baseline) |
| smoothing 0.05 | 94.13 ± 0.41 | 10 | +1.27 ± 0.54 |
| smoothing 0.1 | 94.27 ± 0.31 | 10 | +1.41 ± 0.41 |
| **smoothing 0.2** | **94.47 ± 0.28** | 10 | +1.62 ± 0.55 |

this is a demonstration, not a thesis: it answers a pilot-sized question in ten seconds. the *thesis* would start where the table ends. a 1.4-point gain from label smoothing after 3 epochs is much larger than what label smoothing is usually credited with; is it a regularization effect, an effect on optimization (smoothed targets cap how confident the network is pushed to be, which keeps the logits, and so the size of later updates, smaller), or an artefact of stopping early? each explanation predicts something different (the gap should close with longer training if it's about early stopping, move with the learning rate if it's about step size, and show up in calibration if it's about overconfidence), and each prediction is a falsifiable hypothesis with a cheap experiment. that's the shape of a good first thesis: a measured surprise, competing explanations, and experiments that tell them apart.

## requirements

the tests check the tooling. the thesis itself is judged with the rubric.

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass (`python -m pytest test_capstone.py`) |
| skeleton | exactly the listed files; nothing is ever overwritten; refuses a blank title or question before writing |
| hashes | canonical (key order free, NumPy-faithful), type-faithful (`1` ≠ `1.0`), match SHA-256 of the canonical JSON |
| registry | appends one JSON line per run, reloads it, refuses repeated runs and bad seeds or metrics without writing |
| tables and reports | identical to `mlfs.research.ablation_table` on shared seeds; every placeholder kind rendered with the right interval; malformed placeholders refused |
| **the thesis** | a solid thesis scores at least **14 of 18** on [`rubric.md`](rubric.md), with no criterion at 0 |

## deliverables

1. **the project directory** (created with `make_project`): the proposal, the plan and weekly log, the code, the registry with every run, and the paper with its placeholders.
2. **the paper**, rendered from the registry, 6–8 pages plus references and an appendix.
3. **the review** you wrote of a peer's thesis, and your response to the review of yours.
4. **the write-up** at the end of `capstone.py`: the five process prompts (question and origin, plan against reality, evidence, what didn't work, reproducibility).
5. **(optional)** the workshop submission and the reviews it received.

## stretch goals

- **a second scale.** rerun your main comparison at a second model or data size (topic 04's compute functions say what fits). a claim that holds at two scales is worth far more than one that holds at one.
- **an equivalence claim.** if your result is "no difference", make it an equivalence claim with a margin and enough seeds (topic 02), not an absence of significance.
- **a reproduction package that runs in an hour.** a script that reruns a representative subset of the registry (two configurations × three seeds, say) and checks the regenerated numbers against the logged ones.
- **pre-register publicly.** post the proposal (hypothesis, margin, seeds, analysis) somewhere timestamped before running the main experiments.
- **the workshop.** submit. a rejection with three thoughtful reviews is a better outcome than a paper nobody outside the course reads.

## self-assessment

grade the thesis with [`rubric.md`](rubric.md); better, have a peer grade it, and grade theirs.

## references

- Heilmeier catechism, DARPA. [darpa.mil/work-with-us/heilmeier-catechism](https://www.darpa.mil/work-with-us/heilmeier-catechism)
- Pineau et al. (2021), "Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program)", *JMLR* 22. [arXiv:2003.12206](https://arxiv.org/abs/2003.12206)
- Lipton & Steinhardt (2018), "Troubling trends in machine learning scholarship". [arXiv:1807.03341](https://arxiv.org/abs/1807.03341)
- Dodge, Gururangan, Card, Schwartz & Smith (2019), "Show your work: improved reporting of experimental results", *EMNLP*. [arXiv:1909.03004](https://arxiv.org/abs/1909.03004)
- Müller, Kornblith & Hinton (2019), "When does label smoothing help?", *NeurIPS*. the worked example's topic, studied properly. [arXiv:1906.02629](https://arxiv.org/abs/1906.02629)
- Szegedy, Vanhoucke, Ioffe, Shlens & Wojna (2016), "Rethinking the Inception architecture for computer vision", *CVPR*. where label smoothing was introduced. [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)
- Peyton Jones, "How to write a great research paper". [microsoft.com/en-us/research/academic-program/write-great-research-paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- Schulman (2020), "An opinionated guide to ML research". [joschu.net/blog/opinionated-guide-ml-research.html](http://joschu.net/blog/opinionated-guide-ml-research.html)
