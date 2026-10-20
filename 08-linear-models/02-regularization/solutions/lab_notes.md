# regularization · lab notes

## experiment 1: ridge paths and lasso paths

**what people usually predict:** ridge coefficients reach zero "at some large $\alpha$", and lasso coefficients appear in a jumble, maybe several at once.

**what happens:** across 60 values of $\alpha$ from $10^{-2}$ to $10^5$, **none** of ridge's 600 coefficients is exactly zero: they shrink smoothly towards 0 and get arbitrarily close without arriving. along the lasso path the coefficients enter one at a time, in the order [0, 1, 2, 5, 9, 6, 7, 3, 4, 8]: the three real features first, then the noise. at the slider's default ($\alpha \approx 0.084$) exactly the three real features are nonzero. a few of ridge's irrelevant coefficients first move *away* from zero as $\alpha$ grows, before shrinking.

**why:** ridge's solution is $(X^\top X + \alpha I)^{-1}X^\top y$, a smooth function of $\alpha$ that's zero only in the limit. its penalty's gradient $2\alpha w_j$ vanishes as $w_j \to 0$, so it never has the strength to hold a coefficient at zero. the lasso's coordinate update is $S(\rho_j, \alpha)/z_j$: a coefficient is exactly zero whenever its feature's correlation with the current residual, $|\rho_j|$, is below $\alpha$. as $\alpha$ decreases from $\alpha_{\max}$, the threshold drops past each feature's correlation in turn, so features enter one by one, most useful first. the noise features enter at small $\alpha$ because their correlation with the residual is small but not zero: with finite data, noise always looks a little useful, and the lasso at small $\alpha$ is close to least squares, which uses everything. the irrelevant ridge coefficients that grow at first are compensating for the shrinkage of the correlated real features (features are correlated 0.3 here): the penalty shortens the vector as a whole, not each coordinate.

## experiment 2: two features that say the same thing

**what people usually predict:** both models land near the truth $(1, 1)$ on average, with the lasso a bit noisier.

**what happens** (40 bootstrap resamples):

| model | resamples with $w_1$ or $w_2$ exactly 0 | std of $w_1$ | mean of $w_1 + w_2$ |
|---|---|---|---|
| lasso ($\alpha = 0.1$) | **42%** | **0.60** | 2.00 |
| ridge ($\alpha = 10$) | 0% | 0.07 | 1.99 |
| elastic net ($\alpha = 0.1$, ratio 0.5) | 0% | 0.10 | 1.99 |

the lasso's points are smeared along the line $w_1 + w_2 = 2$, from $(0, 2)$ to $(2, 0)$, with a pile-up at $(2, 0)$: it often keeps $x_1$ alone, occasionally $x_2$ alone, and otherwise some arbitrary split. ridge and the elastic net cluster tightly around $(1, 1)$.

**why:** for predictions, only $w_1 + w_2$ matters (the features are nearly the same), and all three models get it right. the penalties disagree about how to split it. $|w_1| + |w_2|$ is the same for every same-sign split, so the lasso is nearly indifferent, and tiny differences in each resample (which twin happens to correlate a bit better with the residual) decide the split, often pushing it to a corner. ridge's $w_1^2 + w_2^2$ is smallest at the even split, a strong, stable preference. the elastic net's ridge part restores that preference while keeping the $\ell_1$ part's ability to zero out the five irrelevant features. the lesson: a lasso coefficient of 0 does *not* mean "this feature doesn't matter", only "this feature wasn't needed given the others". if you're asked which features matter, check the stability of the selection under resampling (this is the idea behind *stability selection*, Meinshausen and Bühlmann, 2010) or use the elastic net.

## experiment 3: the Bayesian view, measured

**what people usually predict:** theory and practice "roughly agree", maybe within a factor of 10; the best $\alpha$ should shrink as data grow.

**what happens** (60 draws, $n = 40$, $d = 30$, $\sigma = 1$):

| $\tau$ | $\sigma^2/\tau^2$ | best $\alpha$ on average | CV's median choice (middle half) | test MSE: least squares · $\sigma^2/\tau^2$ · CV · per-draw best |
|---|---|---|---|---|
| 0.1 | 100 | 100 | 100 (42 – 3,800) | 4.52 · 1.21 · 1.28 · 1.21 |
| 0.3 | 11.1 | 10 | 11.7 (5.6 – 17.8) | 4.52 · 1.91 · 1.99 · 1.84 |
| 1.0 | 1 | 1 | 1.0 (0.70 – 1.44) | 4.52 · 3.41 · 3.44 · 3.08 |

the minimum of the average test error sits exactly on the dotted line $\sigma^2/\tau^2$ for every $\tau$ (to the grid's resolution). cross-validation's choices scatter over about half an order of magnitude around it (more for $\tau = 0.1$, where the error curve is very flat on the right), but its test error is within about 6% of the theoretical optimum (1.28 against 1.21 for $\tau = 0.1$, under 5% for the others). least squares, with 30 parameters for 40 rows, has a test MSE of 4.52 whatever $\tau$ is: 4.5 times the noise.

**why:** the posterior mean minimizes the expected squared error over everything you're uncertain about, here $w$ drawn from the prior and the noise. so when the data really come from the prior, ridge with $\alpha = \sigma^2/\tau^2$ is the best possible estimator *on average*, for any $n$ (the optimal $\alpha$ doesn't depend on $n$ in this scaling, even though the optimal amount of shrinkage does, because $X^\top X$ grows with $n$ while $\alpha$ doesn't). least squares is bad here because with 30 parameters and 40 rows its variance is huge: on average its error is $\sigma^2 d/(n - d - 1) = 30/9 \approx 3.3$ above the noise floor for Gaussian features, whatever $w$ is. cross-validation doesn't know $\sigma$ or $\tau$, but it estimates the test error of each $\alpha$ directly, which is all it needs; each estimate is noisy (8 validation rows per fold), so its pick wanders, but near the optimum the error curve is flat, so wandering costs little. the per-draw best is lower than any fixed $\alpha$ because it's chosen by looking at the test set: that's chapter 07's winner's curse, not a method you could use.
