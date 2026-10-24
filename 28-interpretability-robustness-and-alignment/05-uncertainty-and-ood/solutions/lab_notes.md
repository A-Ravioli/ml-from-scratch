# uncertainty & OOD · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, one thread), with CIFAR-10 and MNIST from the cache. the notebook takes about 25 seconds. of the first 3,000 CIFAR-10 test images, the first 1,000 fit the temperature and the other 2,000 are used for everything else, so the conformal experiment never touches the data $T$ was fitted on. the notes also quote training seeds 1 and 2, run with the same code outside the notebook. offline, the notebook runs on synthetic stand-ins and its numbers mean nothing.

## experiment 1: overconfidence, and one number that fixes it

**what people predict.** that a 42%-accurate model is "about 42% confident", or that an overfit model is "a bit" overconfident; and that recalibrating will cost accuracy.

**what happens.** the MLP (3072 → 512 → 512 → 10, inputs standardised) memorises its 5,000 training images (training accuracy 0.999) and gets 0.426 on the first 3,000 test images. on the 2,000 evaluation images:

| | accuracy | NLL | ECE | mean confidence |
|---|---|---|---|---|
| $T = 1$ | 0.436 | 4.345 | **0.427** | 0.863 |
| $T = 5.40$ | 0.436 | **1.684** | **0.034** | 0.439 |

the model is 86% confident on average and right 44% of the time; most of its predictions sit in the top confidence bin, where it's right about half the time. the fitted temperature is 5.40 (seeds 1 and 2: 4.39 and 5.13), and dividing by it brings the mean confidence down to the accuracy, the ECE from 0.43 to 0.03 (seeds 1 and 2: 0.41 → 0.03 and 0.42 → 0.03) and the NLL from 4.3 to 1.7. accuracy is identical.

**why.** cross-entropy on a training set the model can memorise is minimised by pushing every training logit margin to infinity; nothing tells the network that its test inputs deserve less certainty, so its logits on test images are as large as on training images. one temperature fixes most of it because the overconfidence is mostly a global *scale* problem: the logits have the right ranking and roughly the right relative sizes, just too much magnitude. temperature can't fix miscalibration that differs between parts of the input space (well calibrated on easy images, overconfident on hard ones): the reliability curve after scaling still wobbles around the diagonal for that reason.

## experiment 2: does the conformal guarantee hold?

**what people predict.** that a guaranteed 90% method covers at least 90% on every split, or that something is wrong if it doesn't; and that every class gets 90%.

**what happens.** over 1,000 random splits of the 2,000 pool images (500 calibration, 1,500 evaluation images, $\alpha = 0.1$), the mean coverage is **0.9002** against a theoretical 0.9002 ($\lceil 501 \cdot 0.9 \rceil / 501 = 451/501$). **46.0%** of the splits cover less than 90%, the middle 95% of splits range from 0.866 to 0.928, and the histogram lies on top of the beta-binomial curve the theory predicts. the sets contain 5.25 classes on average. pooled over the splits, coverage by true class ranges from 0.82 (horse) and 0.85 (truck) to 0.93 (deer, ship) and 0.95 (frog). (an earlier version of this lab fitted $T$ on half of the same pool; with $T$ fitted on separate data, the guarantee's exchangeability assumption holds exactly, and the numbers barely moved: 0.8997 and 48.6%.)

**why.** the guarantee is about the average over random calibration sets and test points: given one calibration set, coverage is a random variable centred at $k/(n+1)$ with standard deviation about 0.013 for 500 calibration points (on paper 3), so a split below 90% is as common as one above. the histogram's width shrinks only with more calibration data. per-class coverage isn't controlled at all: the score $1 - \hat p_y$ is calibrated against a single threshold, and classes on which the model is systematically less confident (horses, trucks) fall below it more often, balanced by classes on which it's confident. to guarantee coverage per class, calibrate a separate $\hat q$ per class (class-conditional, "Mondrian" conformal prediction), at the cost of needing enough calibration points in every class. the large sets are the model's weakness showing honestly: a 44%-accurate model needs half the classes to be right 90% of the time.

## experiment 3: knowing when an input is unusual

**what people predict.** that photos of cats and trucks will be trivially flagged and that rotated digits are harder, or the reverse; and that an ensemble's uncertainty is always better than one model's.

**what happens.** the five MLPs reach 0.912 accuracy on MNIST as an ensemble. AUROC for telling in-distribution digits from OOD inputs:

| score | vs CIFAR (grey) | vs MNIST rotated 90° |
|---|---|---|
| $1 - \max_c p_c$, one model | 0.728 | 0.809 |
| ensemble total entropy | 0.743 | 0.824 |
| ensemble epistemic (mutual information) | **0.958** | 0.786 |

seeds 1 and 2 gave the same picture: epistemic uncertainty was best on the photos (0.904 and 0.824, against 0.70–0.75 for the confidence-based scores) and no better than total entropy on rotated digits (0.819 and 0.774, against 0.83 and 0.82). on photos, single-model confidence is a poor detector: the histogram shows grey photos getting moderate entropy, overlapping heavily with real digits.

**why.** a single model's confidence on an input far from the data is an extrapolation of its logits, and these MLPs often extrapolate confidently: a grey photo looks like nothing in particular, and the network still picks some digit with moderate certainty. different members extrapolate *differently*, so they disagree on photos, and the mutual information picks up exactly that disagreement while ignoring inputs on which all members are equally unsure. rotated digits are different: they still look like strokes, and a rotated 1 or 0 looks like a digit to every member, so the members agree (low epistemic uncertainty) while each is somewhat unsure (moderate aleatoric uncertainty). there the total entropy is the better signal. no score is a guarantee: an OOD detector is only as good as the kinds of OOD inputs it happened to be tested on.
