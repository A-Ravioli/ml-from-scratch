# 16 · capstone: CIFAR-10 with your ResNet

## goal

train **your** ResNet-18 (topic 04) on CIFAR-10 to **at least 90% test accuracy**, with chapter 15's training recipe, and find out what each ingredient of that recipe is worth. the deliverable is a number you can defend (with an interval, a baseline, the runtime and the hardware) and an **ablation table** that measures, rather than asserts, why the recipe works.

the recipe, every piece of which you've built:

| ingredient | from | setting |
|---|---|---|
| network | your `ResNet` (topic 04) | ResNet-18, CIFAR stem, 11.2 M parameters |
| data augmentation | random crop + flip (ch 15 · topic 03) | pad 4 and crop $32 \times 32$, horizontal flip with $p = 0.5$, on whole batches on the device |
| optimizer | SGD with Nesterov momentum (ch 06) | lr 0.2 at batch 256, momentum 0.9 |
| weight decay | (ch 15 · topic 03) | $5 \times 10^{-4}$ on conv and linear weights only, none on biases and batch norm |
| schedule | your `CosineSchedule` (ch 15 · topic 04) | 1 epoch of linear warmup, then cosine to 0, stepped every batch |
| label smoothing | (ch 15 · topic 03) | 0.1 |
| batch norm | (ch 15 · topic 02) | after every convolution |
| training loop | your `fit` (ch 15 · topic 07) | 20 epochs |

## data

CIFAR-10 (Krizhevsky, 2009): 50,000 training and 10,000 test colour images of $32 \times 32$ pixels in 10 balanced classes, loaded with `mlfs.datasets.cifar10()` (cached under `~/.cache/mlfs/`). images are normalized with the training set's per-channel mean and standard deviation. there is no validation split: the recipe is fixed in advance and nothing is tuned, so the test set is used only to report. offline (and in the tests), synthetic stand-ins with the same shapes are used; your write-up must use the real data.

## configurations

| config | where | what | runtime |
|---|---|---|---|
| **small** (tests, offline) | `MLFS_OFFLINE=1` | synthetic data, ResNet with block counts $(1, 1, 1, 1)$ and base width 8, 3 epochs (ablations: 2), CPU | about a minute for the whole notebook |
| **full** (headline number) | default | `RECIPE` above, all 50,000 images, the best device `get_device()` finds | **35 minutes** on an Apple M2 (MPS) |
| **ablation** (table) | default | `RECIPE` with base width 32 and 10 epochs, one ingredient removed per row | 6–10 minutes per row on an M2 |

the notebook picks the config from `datasets.is_offline()`. while you develop, set `MLFS_CAPSTONE_FULL=0` and `MLFS_CAPSTONE_ABLATIONS=0` before starting marimo to skip the long runs.

## what you build

the docstrings in `capstone.py` are the specification.

| stage | functions | what they do |
|---|---|---|
| 1 | `channel_stats(X)`, `to_tensor(X, mean, std)` | per-channel statistics on the $[0, 1]$ scale; uint8 NHWC → standardized float32 NCHW |
| 2 | `augment_batch(x, *, padding=4, flip=True, generator=None)` | a random crop of the zero-padded image and a random horizontal flip, independently per image, as tensor operations on whole batches (no Python loop over images), so augmentation runs on the GPU too |
| 3 | `BatchLoader`, `build_model(config)`, `make_optimizer(model, config)` | batches already on the device, reshuffled and augmented each epoch; your ResNet from a config; SGD with decay on weights only |
| 4 | `predict(model, X)`, `train_and_evaluate(config, data, *, device, seed)` | the recipe end to end with your `fit` and `CosineSchedule` (or a constant lr), returning test accuracy, per-image correctness, history, time and size |
| 5 | `ablation_table(results, baseline)` | each configuration's accuracy with a 95% bootstrap interval, and its difference to the baseline with a **paired** 95% bootstrap interval |

## deliverables

1. `capstone.py` with every check green (`pytest test_capstone.py`).
2. the full run on real CIFAR-10: test accuracy with its 95% interval, the majority-class baseline, runtime and hardware.
3. the ablation table on real CIFAR-10, all six rows.
4. the write-up at the end of the notebook (prompts there; grade it with [`rubric.md`](rubric.md)).

## requirements

| check | threshold |
|---|---|
| tests in `test_capstone.py` | all pass |
| small config, synthetic data (stage 4) | $\ge 70\%$ test accuracy (the reference typically reaches 100%) |
| **full config, real CIFAR-10** | **$\ge 90\%$ test accuracy**, reported with a 95% interval and the majority-class baseline |
| ablation table | all six rows, each with its paired interval; a sentence per row in the write-up |

## reference results

measured with the reference solution on an Apple M2 laptop (MPS, 16 GB), while other jobs were using the CPU:

**full config:** **0.9292** test accuracy (95% bootstrap CI [0.9242, 0.9342]) after 20 epochs in 35.4 minutes; the majority class scores 0.100. the training loss (label-smoothed, so its floor is about 0.5) falls from 1.85 to 0.55. a second run with seed 1 scored **0.9299** ([0.9250, 0.9346]; 44.6 minutes, with the CPU more heavily loaded): the two seeds agree to within 0.1 points, well inside either interval.

**ablations** (base width 32, 10 epochs; every difference is paired over the same 10,000 test images):

| configuration | test accuracy | 95% CI | difference to full recipe (paired 95% CI) |
|---|---|---|---|
| full recipe | 0.8923 | [0.8859, 0.8982] | — |
| no augmentation | 0.8606 | [0.8536, 0.8673] | −0.0317 [−0.0375, −0.0256] |
| no label smoothing | 0.8822 | [0.8757, 0.8886] | −0.0101 [−0.0151, −0.0049] |
| no weight decay | 0.8762 | [0.8698, 0.8826] | −0.0161 [−0.0215, −0.0111] |
| constant lr 0.2 (no warmup, no cosine) | 0.7177 | [0.7088, 0.7264] | −0.1746 [−0.1837, −0.1657] |
| plain network (no residuals) | 0.8816 | [0.8751, 0.8876] | −0.0107 [−0.0159, −0.0053] |

the "full recipe" row retrained with seeds 1 and 2 scored 0.8864 and 0.8860: a seed-to-seed standard deviation of about 0.0035, so the standard deviation of a *difference* between two runs is about 0.005. the paired bootstrap intervals above only capture the test set's sampling noise, not the training run's. and the seed-0 baseline in the table is the luckiest of the three (mean 0.8882), which inflates every difference by about 0.4 points: measured against the three-seed mean, removing label smoothing costs 0.6 points and removing the residual connections 0.7, only 1.2–1.3 standard deviations of a difference from zero, i.e. not distinguishable from seed noise with one run each. weight decay (1.2 points, about 2.4 SD) is probably real; augmentation (2.8 points) and the schedule (16 points) are unambiguous. a careful write-up says so.

## the path past 93%

the full config is a deliberately modest budget (20 epochs on a laptop GPU). known ways to go further, roughly in order of value per hour:

- **a longer schedule.** the same recipe for 50–200 epochs typically reaches 94–95.5% with ResNet-18 on CIFAR-10. on the M2 that's 1.5–6 hours; on a single modern NVIDIA GPU with mixed precision (chapter 26), 200 epochs take well under an hour.
- **test-time augmentation.** averaging the predictions for each image and its mirror image adds a few tenths of a point for free at training time.
- **stronger augmentation and regularization.** cutout, mixup or AutoAugment-style policies, each typically worth half a point to a point at long schedules (and nothing at 20 epochs, where the network is still underfitting).
- **wider networks.** WideResNet-28-10 reaches about 96% but costs roughly 5× the compute.
- **an EMA of the weights** (your `ema_update`, ch 15 · topic 07) for a smoother final model.

## stretch goals

- **seeds.** run the full config with 2–3 seeds and compare the spread with the bootstrap interval. is the test set or the training run the bigger source of noise?
- **one more epoch budget.** run the full recipe for 10 and 40 epochs: how does accuracy scale with compute, and where would you stop?
- **test-time flips.** implement them in `predict` and measure the gain with a paired interval.
- **a different backbone.** swap in a network built from your `DepthwiseSeparableConv` or `ConvNeXtBlock` (topic 05) with a similar parameter budget: accuracy per parameter and per minute.

## self-assessment

grade your write-up with [`rubric.md`](rubric.md).

## references

- Krizhevsky (2009), "Learning multiple layers of features from tiny images", technical report, University of Toronto. the CIFAR-10 dataset. [cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- He, Zhang, Ren & Sun (2016), "Deep residual learning for image recognition", *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- He et al. (2019), "Bag of tricks for image classification with convolutional neural networks", *CVPR*. warmup, cosine decay, label smoothing, no decay on biases and BN, measured one by one. [arXiv:1812.01187](https://arxiv.org/abs/1812.01187)
- Loshchilov & Hutter (2017), "SGDR: stochastic gradient descent with warm restarts", *ICLR*. cosine schedules. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- Zagoruyko & Komodakis (2016), "Wide residual networks", *BMVC*. [arXiv:1605.07146](https://arxiv.org/abs/1605.07146)
