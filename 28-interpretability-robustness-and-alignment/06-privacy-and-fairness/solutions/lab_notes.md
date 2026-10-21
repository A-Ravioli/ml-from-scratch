# privacy & fairness · lab notes

numbers below are from the reference solution on an Apple M2 laptop (CPU, one thread), with MNIST and CIFAR-10 from the cache. the notebook takes about a minute (plus a few seconds per move of the experiment-1 knobs). the notes also quote seeds 1 and 2 (initialisation, batch order and noise), run with the same code outside the notebook. offline, the notebook runs on synthetic stand-ins and its numbers mean nothing. the lending data in experiment 3 is synthetic by design.

## experiment 1: the price of privacy

**what people predict.** that private training costs a lot of accuracy at any meaningful $\epsilon$, and that clipping alone is free.

**what happens.** MLP (784 → 128 → 10), 4 epochs on 12,000 digits, batches of 256 ($q = 0.0213$, 184 steps), $\delta = 10^{-5}$:

| training | $\epsilon$ | test accuracy |
|---|---|---|
| no clipping, no noise (same SGD, lr 0.25) | ∞ | 0.927 |
| DP-SGD, $\sigma = 0$ (clipping only) | ∞ | 0.868 |
| DP-SGD, $\sigma = 0.5$ | 15.95 | 0.868 |
| DP-SGD, $\sigma = 1$ | 2.83 | 0.852 |
| DP-SGD, $\sigma = 2$ | 0.83 | 0.798 |

seeds 1 and 2: 0.932 / 0.872 / 0.867 / 0.857 / 0.803 and 0.925 / 0.868 / 0.871 / 0.853 / 0.799. at the same learning rate, clipping alone costs about 6 points here, the noise at $\epsilon = 2.8$ another 1.5, and at $\epsilon = 0.8$ another 7. the knobs below the table train one more network at a noise multiplier and clip norm of your choice ($\sigma = 1.5$, clip 1: $\epsilon = 1.26$, accuracy 0.840).

**why.** clipping at norm 1 caps the size of every per-example gradient, which slows learning (the clipped mean is biased towards zero and towards examples with small gradients) and acts like a much smaller learning rate on hard examples. (an earlier version of this lab trained the baseline at lr 0.1, which confounded the comparison; now all five runs share lr 0.25.) the noise hurts less than its size suggests because it's added to the *sum* of 256 clipped gradients: with $\sigma = 1$ the noise has norm about $\sigma\sqrt{d} \approx 320$ over the $d \approx 100{,}000$ parameters, against a sum that can have norm up to 256 but is concentrated in the few directions the data agrees on, and averaging over steps cancels much of the rest. the accountant assumes Poisson sampling (each example in each batch independently with probability $q$); shuffled fixed-size batches, used here as in much practical code, give each example exactly one appearance per epoch. the amplification-by-sampling analysis doesn't cover that scheme, so the printed $\epsilon$ is the value for the Poisson version of this training run, not a proven guarantee for the code as written.

## experiment 2: can an attacker tell who was in the training set?

**what people predict.** that a loss threshold is too crude to work, or that any DP training makes the attack useless.

**what happens.** 1,000 CIFAR-10 training images (members) and 1,000 test images (non-members), MLP (3072 → 128 → 10), 30 epochs in batches of 64 ($q = 0.064$, 450 steps):

| model | $\epsilon$ | train acc | test acc | attack AUROC | TPR at 1% FPR | DP bound on TPR |
|---|---|---|---|---|---|---|
| non-private | ∞ | 1.000 | 0.333 | **0.946** | 0.063 | 1 |
| DP-SGD, $\sigma = 1.5$ | 5.8 | 0.439 | 0.294 | 0.600 | 0.019 | 1 |
| DP-SGD, $\sigma = 5$ | 1.4 | 0.289 | 0.250 | 0.542 | 0.008 | 0.040 |

seeds 1 and 2: AUROC 0.934 and 0.940 (non-private), 0.588 and 0.593 ($\sigma = 1.5$), 0.535 and 0.537 ($\sigma = 5$); TPR at 1% FPR 0.13–0.14, 0.011–0.015 and 0.011–0.015. the non-private model's member losses are all near zero while non-member losses spread up to 18.

**why.** the loss-threshold attack works exactly as well as the model memorises: a 67-point gap between training and test accuracy means almost every member has a near-zero loss that few non-members match, so the AUROC is 0.95. its TPR at 1% FPR is still modest (6–14%) because a few non-members are also confidently right, and the strictest thresholds have to sit below them. DP-SGD shrinks the gap (train 0.44 vs test 0.29) and with it the attack, at a large cost in accuracy on this small dataset. the bound $\text{TPR} \le e^\epsilon \text{FPR} + \delta$ is vacuous at $\epsilon = 5.8$ (it allows TPR 1 at FPR 1%) and meaningful at $\epsilon = 1.4$ (4%), where the attack reaches 0.8–1.5%: the bound holds and isn't tight against this weak attack. stronger attacks (per-example calibration with shadow models, Carlini et al., 2022) get much closer to the bound; a loose bound means the guarantee, not the measured attack, is what you can rely on.

## experiment 3: which fairness?

**what people predict.** that an equally accurate score with one threshold for everyone is fair by every criterion, or that some clever threshold can satisfy all criteria at once.

**what happens.** 20,000 synthetic applicants, base rates 60% (A) and 40% (B), the same noisy score in both groups:

| policy | accuracy | DP difference | EO difference | approval A / B | TPR A / B | FPR A / B |
|---|---|---|---|---|---|---|
| one threshold (0.5) | 0.912 | 0.163 | 0.008 | 0.58 / 0.42 | 0.91 / 0.92 | 0.09 / 0.08 |
| equal opportunity (TPR 0.8) | 0.878 | 0.152 | **0.000** | 0.49 / 0.34 | 0.80 / 0.80 | 0.03 / 0.03 |
| demographic parity | 0.880 | **0.000** | 0.184 | 0.53 / 0.53 | 0.85 / 0.98 | 0.05 / 0.23 |

**why.** one threshold on a score that's equally informative in both groups gives (almost) equal error rates, so equalized odds holds; but group A has more repayers, so more of A clears the threshold, and demographic parity fails by 16 points. equalising approval rates means approving some B applicants with lower scores (B's FPR jumps from 0.08 to 0.23) and rejecting some A applicants with higher ones (A's TPR falls to 0.85): error rates become unequal. equal opportunity reaches TPR 0.80 in both groups, with approval rates still 15 points apart. with different base rates, demographic parity and equalized odds can't both hold for any non-trivial classifier (and calibration plus equal error rates can't either; on paper 4). the data can't tell you which disparity is acceptable for loans: that's a decision for the people who bear the consequences, made with the trade-off in plain view.
