# regularization · lab notes

numbers are for real MNIST, measured with the reference solution: a $784 \to 256 \to 256 \to 10$ ReLU network trained with Adam ($10^{-3}$, batch 64) for 30 epochs on the first 1,000 training images, scored on the last 2,000. offline (the synthetic stand-in that `verify.py` uses, with a blanked 3-pixel border so that dead pixels exist) every network reaches 100% validation accuracy, so only the weight norms of experiment 1 and the calibration columns of experiment 3 say anything there.

## experiment 1: Adam + L2 versus AdamW

**what people usually predict:** a tiny λ does almost nothing to any weight; AdamW and Adam + L2 behave about the same with suitably matched λ.

**what happens:**

| optimizer | λ | validation accuracy | RMS, active-pixel weights | RMS, dead-pixel weights |
|---|---|---|---|---|
| Adam + L2 | 0 | 0.922 | 0.0369 | 2.06e-02 |
| Adam + L2 | 1e-4 | 0.922 | 0.0313 | 2.2e-13 |
| Adam + L2 | 1e-3 | 0.922 | 0.0219 | 2.3e-13 |
| Adam + L2 | 1e-2 | 0.918 | 0.0130 | 2.3e-13 |
| Adam + L2 | 1e-1 | 0.751 | 0.0059 | 2.3e-13 |
| AdamW | 0 | 0.922 | 0.0369 | 2.06e-02 |
| AdamW | 1e-2 | 0.922 | 0.0368 | 2.05e-02 |
| AdamW | 1e-1 | 0.922 | 0.0361 | 1.96e-02 |
| AdamW | 1 | 0.925 | 0.0307 | 1.27e-02 |
| AdamW | 10 | 0.899 | 0.0192 | 1.65e-04 |

with Adam + L2, *any* positive λ, even $10^{-4}$, wipes out the 175 dead-pixel weights completely (to $10^{-13}$, numerically zero), while the active weights shrink gradually with λ. with AdamW, both groups shrink together and smoothly; the dead weights shrink faster only because nothing pushes back on them. validation accuracy barely depends on the choice in this small problem: the best AdamW setting (0.925) and the unregularized network (0.922) are within seed noise of each other.

**why:** a dead-pixel weight gets no gradient from the loss (its input is always 0), so under Adam + L2 its whole gradient is $\lambda w$, and Adam normalizes that to $\hat m/\sqrt{\hat v} \approx \operatorname{sign}(w)$: a fixed step of $\eta = 10^{-3}$ towards zero, regardless of λ (on paper 1). from an initial RMS of 0.02 that takes about 20 steps, a bit over one epoch. an active weight's loss gradient dominates $\sqrt{\hat v}$, so the penalty's contribution, $\lambda w/\sqrt{\hat v}$, is small, and its decay depends on how noisy that weight's gradients are. so "L2 under Adam" regularizes hardest exactly the weights the data says nothing about, and weakly the ones it uses. AdamW's decay, $w \leftarrow (1 - \eta\lambda)w$, is applied outside the normalization: every weight loses the same fraction per step ($(1 - 10^{-2})^{480} = 0.008$ for λ = 10 over 480 steps, which matches the dead weights' fall from 0.02 to $1.65 \times 10^{-4}$). its λ is interpretable, which is the real reason it became the default. whether it buys accuracy depends on the problem; Loshchilov & Hutter's gains were on larger image models with long training runs, not on a 30-epoch MLP.

## experiment 2: dropout on a small dataset

**what people usually predict:** that dropout lowers the training loss less but improves validation accuracy by a lot, with $p = 0.5$ best.

**what happens:** (mean of 2 seeds)

| dropout p | training accuracy | validation accuracy | final training loss | best validation loss | final validation loss |
|---|---|---|---|---|---|
| 0 | 1.000 | 0.923 | 0.0025 | 0.296 (epoch 10) | 0.382 |
| 0.2 | 1.000 | 0.929 | 0.0098 | 0.277 (epoch 10) | 0.338 |
| 0.5 | 1.000 | 0.929 | 0.0453 | 0.263 (epoch 19) | 0.286 |

every network memorizes the training set (100% accuracy in eval mode). dropout raises the training loss (×18 for $p = 0.5$), lowers the best and especially the final validation loss, and moves the validation-loss minimum later. validation accuracy improves by about 0.6 points, which is about the size of the seed-to-seed differences in this setting.

**why:** without dropout, once every training image is classified correctly, the only way to keep lowering the training loss is to grow the logits, which makes the network more confident everywhere, including on the 8% of validation images it gets wrong. each of those wrong answers then costs more, so the validation loss rises while accuracy (which ignores confidence) stays flat. dropout trains an ensemble of thinned networks: the noise makes the training loss harder to drive to zero, which limits that runaway confidence (hence the lower final validation loss) and discourages features that only work in combination with specific other units. the accuracy gain is real but small here, and two seeds can't establish it: topic 06's lab shows that seed-to-seed spread is often as large as the effects people report.

## experiment 3: label smoothing, and when to stop

**what people usually predict:** label smoothing gives a less confident, therefore *better calibrated* model, and early stopping on loss or accuracy picks about the same epoch.

**what happens:**

| smoothing | validation accuracy | NLL | confidence when wrong | ECE | stop on val loss | stop on val accuracy |
|---|---|---|---|---|---|---|
| 0 | 0.922 | 0.383 | 0.791 | 0.043 | epoch 15 (best 10) | epoch 21 (best 16) |
| 0.1 | 0.941 | 0.346 | 0.457 | 0.148 | never | epoch 19 (best 14) |
| 0.3 | 0.943 | 0.584 | 0.319 | 0.327 | never | never |

label smoothing improves accuracy by about 2 points here (one seed), and cuts the confidence on wrong answers from 0.79 to 0.46 and 0.32. but its calibration error goes *up*, from 0.04 to 0.15 and 0.33. and the two early stoppers disagree: on the plain model the validation loss bottoms out at epoch 10 and triggers a stop at 15, while the accuracy keeps improving until epoch 16. on the smoothed models the validation loss never triggers at all. (offline: every model is 100% accurate; ECE is 0.00, 0.09 and 0.28, the same pattern of underconfidence.)

**why:** the most confident prediction label smoothing rewards is $1 - s + s/k$: 0.91 for $s = 0.1$, 0.73 for $s = 0.3$ (on paper 3). the smoothed model is right on 94% of validation images but says "91%" or "73%" about almost all of them: it is *under*confident, and ECE measures the gap between confidence and accuracy in either direction. the plain model is overconfident on its errors, but those are few, so its overall ECE is small. so label smoothing fixes overconfidence on the wrong answers by making all answers less confident; whether that's "better calibration" depends on the regime. here accuracy (0.94) is above the confidence label smoothing aims for ($1 - s + s/k$ = 0.91 or 0.73), so the smoothed model ends up underconfident. Müller, Kornblith & Hinton (2019) studied the opposite, more common regime, large overconfident networks, where label smoothing *reduces* ECE; both are the same mechanism (confidence pulled towards $1 - s + s/k$), helping when the model was above it and hurting when its accuracy is. recalibrating with a temperature after training (ch 28) fixes either direction. the early-stopping disagreement is the same story from another angle: once the plain model has memorized the training set, its growing confidence makes the validation *loss* worse (the wrong answers cost more) while the validation *accuracy* still creeps up. if accuracy is what you'll report, stop on accuracy (or on loss with a large patience); the smoothed models, whose confidence is capped, don't have the loss-versus-accuracy split at all.
