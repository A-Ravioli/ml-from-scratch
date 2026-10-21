# adversarial robustness · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, one thread), with MNIST from the cache: LeNets trained for two epochs on the first 12,000 training digits (Adam, lr $2 \times 10^{-3}$, batches of 128), evaluated on the first 1,000 test digits. the notebook takes about 30 seconds. the notes also quote training seeds 1 and 2, run with the same code outside the notebook. offline, the notebook runs on a synthetic stand-in and its numbers mean nothing.

## experiment 1: how small can an attack be?

**what people predict.** that a change you can barely see should cost a few points of accuracy; that FGSM and PGD are about equally strong.

**what happens.** the standard LeNet is 94.7% accurate on clean digits.

| epsilon | 0 | 0.02 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 |
|---|---|---|---|---|---|---|---|
| FGSM | 0.947 | 0.921 | 0.861 | 0.686 | 0.446 | 0.199 | 0.008 |
| PGD-10 | 0.947 | 0.919 | 0.850 | 0.581 | 0.181 | **0.022** | 0.000 |

at $\epsilon = 0.2$ the attacked digits are still obviously the same digits, but covered in a clearly visible grey noise texture over the whole background (each background pixel can rise from 0 to 0.2), and 10-step PGD leaves 2% accuracy (seeds 1 and 2: 4.0% and 2.8%). at $\epsilon = 0.1$, PGD accuracy was 0.58, 0.56 and 0.42 for seeds 0, 1, 2: this short-trained model is not maximally brittle at the smallest budgets, but every model collapses by 0.2. PGD is stronger than FGSM at every budget, by 11 points at 0.1 and 26 points at 0.15. the slider under the curves attacks eight more digits at any $\epsilon$: at 0.1, six of the eight are already misclassified (on seed 0's model).

**why.** for an approximately linear model, an $\ell_\infty$ budget $\epsilon$ moves the logit by up to $\epsilon\|w\|_1$, and with 784 inputs that's a sum of 784 small aligned contributions. the per-pixel change is small; the effect on the output isn't. FGSM takes the whole budget in one step, trusting a linear approximation over a distance where the network isn't linear (ReLUs switch, max-pooling winners change); PGD re-linearises after each small step and follows the loss surface, so it finds higher losses inside the same ball.

## experiment 2: adversarial training

**what people predict.** either "no cost" or "a large cost" in clean accuracy, and robustness that extends to every $\epsilon$; few predict what the gradients will look like.

**what happens.** adversarial training (5-step PGD at $\epsilon = 0.2$, 2 epochs) took 12 seconds against 3 for standard training.

| model | clean | PGD 0.1 | PGD 0.2 | PGD 0.3 |
|---|---|---|---|---|
| standard | 0.947 | 0.581 | 0.022 | 0.000 |
| adversarially trained | 0.910 | 0.752 | **0.578** | 0.121 |

seeds 1 and 2: clean 0.895 and 0.903 (standard 0.936, 0.949), PGD-0.2 accuracy 0.566 and 0.567 (standard 0.040, 0.028), PGD-0.3 accuracy 0.063 and 0.087. robustness costs 3.7–4.6 points of clean accuracy, buys about 54 points at the training budget, and fades quickly beyond it. the input gradients of the standard model are diffuse red and blue speckle; those of the adversarially trained model trace the strokes of the digit.

**why.** the robust model has to be insensitive to every perturbation in the ball, so it gives up the many small, weakly predictive pixel patterns that a standard model combines for its last few points of accuracy (Tsipras et al., 2019; on paper 4). what's left are large-scale features, the strokes themselves, and the gradient of the loss with respect to the input points along them: making the digit more like itself, which is what a human would do. the robustness is specific to the budget it was trained for: at 0.3, perturbations are large enough to genuinely erase strokes, and the model was never asked to resist them.

## experiment 3: does adversarial robustness carry over to natural shifts?

**what people predict.** that a "robust" model is more robust to everything.

**what happens.** accuracy at severities 3 and 5:

| corruption | standard (3 / 5) | adversarially trained (3 / 5) |
|---|---|---|
| Gaussian noise | 0.928 / 0.717 | 0.898 / 0.856 |
| contrast | 0.945 / 0.896 | 0.865 / **0.510** |
| brightness | 0.721 / 0.446 | 0.876 / 0.573 |
| shift | 0.722 / 0.339 | 0.622 / 0.253 |

the one result that held for all three seeds is contrast: at severity 5 the adversarially trained model scored 0.510, 0.517 and 0.676 against the standard model's 0.896, 0.911 and 0.884. the others depend on the seed. at severity 5, noise favoured the robust model in two seeds of three (0.856 vs 0.717, 0.827 vs 0.749, but 0.827 vs 0.845 in seed 1), and brightness in one (0.573 vs 0.446, but 0.502 vs 0.616 and 0.329 vs 0.484). shifted digits were about equally bad for both (0.25–0.30 vs 0.28–0.34).

**why.** an $\ell_\infty$ ball of radius 0.2 contains every small per-pixel change, including all moderate pixel noise, which is why noise robustness tends to improve; it contains no contrast change of severity 5 (that rescales every stroke pixel from about 1 to about 0.35, a change of 0.65 per pixel) and no translation at all. the robust model relies more on large intensity structure, which is exactly what a contrast reduction removes (Yin et al., 2019 saw the same on CIFAR and ImageNet: help on high-frequency corruptions, harm on low-frequency ones like fog and contrast). robustness is a statement about a threat model; outside it, all bets are off, and a defence can make things worse.
