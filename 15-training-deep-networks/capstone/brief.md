# 15 · capstone: fix the training run

## goal

`capstone.py` contains a complete MNIST training pipeline in PyTorch, assembled from this chapter's pieces: a validation split, input normalization, a kaiming-initialized MLP with dropout, a training step with gradient clipping, your own `StepLR` and `EarlyStopping` (from `mlfs.train`) driving a torch optimizer, an evaluation function, and a final test-set report with a bootstrap interval. it runs from top to bottom without a single error. it also contains **10 bugs**, each of them a mistake that real training code makes all the time.

your job is the one you'll have for the rest of your career whenever a model underperforms: find the bugs, fix them, and explain each one, with a measurement of what it cost. some bugs are loud (`nan` everywhere, accuracy at chance); some cost a point of accuracy; some change nothing you'd notice in this configuration and would bite you in the next one. the checks in `test_capstone.py` tell you *that* something is wrong and what it looks like, never *where*; the docstrings say what each function is supposed to do. everything else is detective work, and every tool you need is from this chapter: the lessons' "common bugs" sections, your `DiagnosticsLogger`, learning-rate and accuracy curves, `activation_stats`, and plain reading.

## data

MNIST (LeCun, Cortes & Burges), loaded with `mlfs.datasets.mnist()` (cached under `~/.cache/mlfs/`): the first 60,000 training images, flattened to 784 pixels and divided by 255, split by the pipeline into 55,000 for training and 5,000 for validation; the 10,000 test images are used once, at the end. offline (and in `verify.py`), a small synthetic stand-in with the same shapes is used; the tests use their own synthetic "digits" with a blank border like MNIST's. your write-up must use the real data.

## configurations

| config | where | pipeline | runtime (laptop CPU) |
|---|---|---|---|
| **small** (tests, `verify.py`, offline) | `MLFS_OFFLINE=1` | 1,800 + 200 synthetic images, MLP (64,), 4 epochs, batch 64 | a few seconds |
| **full** (your write-up) | default | 55,000 + 5,000 real images, MLP (256, 128), dropout 0.2, SGD (lr 0.05, momentum 0.9), batch 128, lr halved every 4 epochs, gradient clipping at 5, early stopping on validation accuracy (patience 3), at most 10 epochs | about 25 s per run with PyTorch on the CPU |

the notebook picks the config from `datasets.is_offline()`.

## the pipeline, as specified

the docstrings are the specification; the code is what's suspect.

| stage | function | what it's supposed to do |
|---|---|---|
| 1 | `split_data(X, y, n_val, rng=None)` | a random split; validation = the first `n_val` indices of one permutation |
| 1 | `normalize(X_train, X_other)` | standardize with the training set's overall pixel mean and std (two scalars) |
| 2 | `build_model(hidden, dropout, seed)` | `Linear → ReLU → Dropout` blocks, kaiming-normal weights (ReLU, fan_in), zero biases, a final `Linear` that outputs logits |
| 3 | `train_one_epoch(model, loader, optimizer, grad_clip, scheduler)` | `zero_grad → forward → cross-entropy → backward → clip → step → (per-step scheduler)`; returns the mean loss per example |
| 4 | `evaluate(model, X, y)` | mean cross-entropy and accuracy, in eval mode under `no_grad`, restoring the mode |
| 4 | `fit(model, ...)` | SGD with momentum; the learning rate halved every `lr_step_epochs` **epochs**; early stopping on validation **accuracy**, with the best weights restored |
| 5 | `run_pipeline` (given) | the stages wired together; the validation and test sets normalized with the training statistics |

## deliverables

1. `capstone.py` with all 10 bugs fixed: every check green, stages 1–5 (`pytest test_capstone.py`).
2. the full config run on real MNIST, with the test accuracy and its 95% bootstrap interval (the notebook's given cells).
3. the write-up at the end of the notebook. for each bug: the **symptom** that led you to it, the **cause**, the **fix**, and its **measured effect**, obtained by re-introducing that one bug into the fixed pipeline and running the full config. then the three closing questions (silent versus loud, the honest number, process).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass; each stage's checks target that stage's bugs |
| fixed pipeline, synthetic digits (stage 5) | $\ge$ 90% test accuracy, median of 3 seeds |
| fixed pipeline, full config, real MNIST | **$\ge$ 0.975 test accuracy**, reported with a 95% bootstrap interval and the majority-class baseline |
| per-bug measurements | all 10, each from a full-config run with only that bug re-introduced |

for reference, the reference solution's full config reaches a test accuracy of **0.981** (95% CI [0.978, 0.984]) with seed 0, and 0.981 and 0.983 with seeds 1 and 2; it trains for all 10 epochs in about 25 s. the majority class scores 0.114. the pipeline as shipped scores **0.098**: its training loss is `nan` from the first step. individually, the ten bugs range from no measurable effect at all to chance-level accuracy; the answer key with every measurement is in `solutions/bug_notes.md` (read it after your write-up).

## stretch goals

- **interactions.** some bugs only matter in the presence of others, or of a different configuration. find one pair whose combined effect is much worse than either alone, and one bug whose effect appears only when you change a hyperparameter (a larger learning rate, a smaller batch, a different dataset).
- **make them loud.** add assertions or logged statistics to the pipeline (a `DiagnosticsLogger` on the torch model via `register_forward_hook`, a check that the initial loss is about $\log 10$, a check that validation and test are normalized with the training statistics) so that each of the 10 bugs, if re-introduced, fails or warns within the first epoch. which bugs resist this?
- **your own bug.** plant an eleventh bug that passes all the checks, and write the check that catches it.
- **seed-to-seed variance.** run the fixed pipeline with 5 seeds and compare the spread with the bootstrap interval, and with the smallest per-bug effect you measured. which of your measured effects are distinguishable from seed noise?

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Karpathy (2019), "A recipe for training neural networks". the habits that catch most of these bugs before they cost a training run. [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/)
- Kaufman, Rosset, Perlich & Stitelman (2012), "Leakage in data mining: formulation, detection, and avoidance", *ACM TKDD* 6(4). [doi:10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579)
- Sculley et al. (2015), "Hidden technical debt in machine learning systems", *NeurIPS*. why pipelines fail silently. [papers.nips.cc/paper/5656](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems)
- LeCun, Bottou, Bengio & Haffner (1998), "Gradient-based learning applied to document recognition", *Proceedings of the IEEE* 86(11): 2278–2324. [doi:10.1109/5.726791](https://doi.org/10.1109/5.726791)
