# generalization in deep learning · lab notes

numbers below are from the reference solution on a laptop CPU, on the real MNIST (1,000 training digits drawn with seed 0, pooled to 14 × 14; the first 2,000 test digits), with the seeds in the notebook. the lab takes about a minute. offline, the synthetic stand-in is much easier and the double-descent peak is lower, but every pattern below is still there.

## experiment 1: memorizing noise

**what people predict.** many people expect the network to fail on random labels ("there's nothing to learn"), or to need a different learning rate or many more parameters. those who know the Zhang et al. result usually guess that random labels take about as long as real ones.

**what happens.** (MLP 196 → 512 → 512 → 10, 368,650 parameters, Adam at $10^{-3}$, batches of 100, stopped at 100% training accuracy)

| labels randomized | epochs to 100% training accuracy (2 seeds) | test accuracy |
|---|---|---|
| 0% | 22, 24 | 0.874, 0.866 |
| 50% | 75, 83 | 0.515, 0.489 |
| 100% | 84, 83 | 0.018, 0.014 |

every run reaches 100% training accuracy, without changing anything. random labels take about 3.5 times as many epochs as real ones, and the training-accuracy curves (log axis) show why: with real labels the network gets 60–70% right after one epoch and the rest follows quickly; with random labels accuracy climbs slowly from chance, one memorized example at a time. with half the labels corrupted, test accuracy is about 50%, far above the 10% of guessing: the network learned the real digits from the clean half and memorized the corrupted half on top. the fully randomized network is *below* chance on the true test labels (1.4–1.8% against 10%), and that's a feature of the exact corruption, not noise: every one of its training labels is guaranteed to be *wrong*, so the labels still carry information ("this image is not a 7"), and the network learns to avoid each image's true class. Zhang et al.'s labels, drawn from all ten classes, carry none, and their networks sit at chance. chapter 13's VC bound with the parameter count as the VC dimension allows a test error of up to 3.3, i.e. anything.

**why.** a network with 369 parameters per training example can represent a function that takes any prescribed value at 1,000 points, so its capacity on this sample is maximal and every capacity-based bound is vacuous (on paper 3). what differs between the real-label and random-label solutions is not the class but the solution: with real labels, gradient descent finds weights that capture shared structure (a few hundred directions explain most digits), which takes few epochs and transfers to new images; with random labels there is no shared structure to find, so the same optimizer ends at a solution that stores each example separately. any explanation of generalization has to be about which solution the data and the algorithm produce. (Arpit et al., 2017, studied the "learn patterns first, memorize later" dynamics in detail.)

## experiment 2: double descent

**what people predict.** the classical U: the error falls, then rises as the model overfits, and keeps rising (or levels off) as $p$ grows further. some predict a smooth improvement with size and no peak.

**what happens.** (median of 3 feature draws; ridge 0, no label noise)

| p | 100 | 300 | 600 | 800 | 950 | **1,000** | 1,050 | 1,200 | 1,500 | 2,500 | 4,000 | 8,000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test error | 0.218 | 0.161 | 0.194 | 0.297 | 0.545 | **0.861** | 0.515 | 0.278 | 0.185 | 0.128 | 0.117 | **0.105** |
| test MSE | 0.50 | 0.44 | 0.62 | 1.13 | 4.17 | **343** | 3.57 | 1.12 | 0.61 | 0.37 | 0.31 | 0.28 |

the classical regime has its best point near $p = 300$ (16.1% error). at $p = n = 1{,}000$ the error is 86%, worse than chance on some draws, and the MSE is about 800 times its value at $p = 300$. past the threshold the error descends again and keeps descending: at $p = 8{,}000$ it is 10.5%, a third better than the best underparameterized model, with every model there fitting the training labels exactly (train MSE 0.000). the three feature draws agree closely except at the threshold itself.

the controls (same widths, 3 draws, medians):

| setting | error at p = 300 | at p = 1,000 | at p = 8,000 |
|---|---|---|---|
| ridge 0, no noise | 0.161 | 0.861 | 0.105 |
| ridge 0, 20% of training labels wrong | 0.250 | 0.883 | 0.289 |
| ridge 0.001 | 0.152 | 0.125 | 0.120 |
| ridge 0.1 | 0.250 | 0.234 | 0.230 |

a ridge of 0.001 removes the peak entirely (the curve becomes monotone up to noise) and costs a little at the largest widths (0.120 against 0.105); a ridge of 0.1 is too strong everywhere. with 20% label noise the peak widens (the error is above 0.5 from $p = 800$ to $1{,}500$), and the overparameterized models no longer beat the best small one by much: at $p = 8{,}000$ the error is 0.289, and the descent is slow.

**why.** near $p = n$ the feature matrix is nearly square, its smallest singular value nearly zero, and the unique interpolant has huge coefficients that amplify the label noise and every irregularity of the features (on paper 2; with no explicit label noise, the "noise" is the part of the target the random features can't represent, which acts the same way). far past the threshold, the minimum-norm interpolant spreads over thousands of features and the amplification fades; in the limit it is ridgeless kernel regression with a smooth kernel. a ridge caps the amplification, so the peak disappears; label noise increases what gets amplified, so it grows. Hastie et al. (2022) and Mei & Montanari (2022) derived these curves exactly for this kind of model.

## experiment 3: which interpolant does gradient descent pick?

**what people predict.** that all interpolating solutions are equivalent ("they all have zero training error"), or that gradient descent finds some arbitrary one.

**what happens.** (100 digits, 2,000 features, even vs odd, 20,000 steps at learning rate $1/\lambda_{\max}$)

| solution | train MSE | test error | distance to the minimum-norm solution | weight norm |
|---|---|---|---|---|
| gradient descent from $w_0 = 0$ | 4e-9 | 0.1735 | 0.003 | 6.46 |
| gradient descent from $w_0 \sim \mathcal N(0, 0.5^2)$ | 1e-8 | 0.213 | 21.9 | 22.8 |
| minimum norm (`pinv`) | 3e-29 | 0.1735 | 0 | 6.46 |

with $\sigma = 1$ (move the slider) the random start gives 0.259 test error, with $\sigma = 0$ it is identical to the zero start. both runs fit the training labels to within $10^{-8}$; the one from zero *is* the minimum-norm solution, to three decimals, and has the same test error; the one from a random start differs from it by exactly the null-space part of its initialization (the distance, 21.9, is the norm of that part) and is 4 points worse on test data.

**why.** every gradient of the squared loss is a combination of the training feature vectors, so gradient descent can only move $w$ within their span; the part of $w_0$ orthogonal to the span is never touched (on paper 1). from zero, the iterates stay in the span, where the only interpolant is the minimum-norm one. the random component added by a random start is invisible on the training set (it's orthogonal to every training feature vector) but not on test points, where it adds noise to every prediction. this is an *implicit bias*: the algorithm chooses among equally good minima. it's also a warning about initialization scale: lazy, large-initialization networks carry their random initial function $f_0$ (topic 03) the same way.
