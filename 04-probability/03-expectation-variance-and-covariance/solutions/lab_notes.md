# expectation, variance & covariance · lab notes

## experiment 1: when does the textbook formula break?

**what people usually predict:** float64 has 16 digits, so the shortcut formula should be fine until the offset is astronomically large, maybe $10^{15}$. float32 a bit earlier. negative variances seem impossible ("it's a sum of squares").

**what happens:** the float64 shortcut passes 1% error around $\mu \approx 10^7$ and returns its first negative variance, $-128$, at $\mu = 10^9$. in float32 it's far worse: 1% error at $\mu \approx 300$, and a negative variance by $\mu = 10^4$. by $\mu = 10^5$ the float32 "variance" is 1024. two-pass and `RunningStats` stay accurate to about $10^{-15}$ for small offsets and still have about 7 correct digits at $\mu = 10^{12}$.

**why:** the shortcut subtracts two numbers of size $\mu^2$ that each carry a rounding error of about $\varepsilon\mu^2$, to get an answer of size $\sigma^2$. so its relative error is about $\varepsilon\mu^2/\sigma^2$. setting that to 0.01 gives $\mu \approx \sqrt{0.01/\varepsilon}$: about $10^7$ for float64 ($\varepsilon \approx 10^{-16}$) and about 400 for float32 ($\varepsilon \approx 6 \times 10^{-8}$), matching the plot. once the error exceeds $\sigma^2$ itself, the result's sign is a coin flip. data at offsets like these are everywhere: Unix timestamps ($1.7 \times 10^9$), prices in cents, pixel sums, sensor counts, and any activation in a float32 network that has drifted away from zero. the stable methods only ever square *deviations*, so they lose nothing to cancellation. their small error at the far right isn't the algorithm's fault: at $\mu = 10^{12}$, storing $\mu + z$ in float64 already rounds $z$ to the nearest $10^{-4}$, so the data themselves have lost information about the variance.

## experiment 2: can you trust a Monte Carlo error bar?

**what people usually predict:** about 95% coverage, maybe 93–94% for small $n$. the estimator is unbiased, so estimates should be above and below the truth equally often.

**what happens:** the RMS error does shrink like $1/\sqrt n$. but with $c = 2$ and $n = 100$, only about 74% of the "95%" intervals contain the truth, and about 67% of the estimates are *below* it. even at $n = 3000$ coverage is only about 89%. with $c = 1$ it's much better (91% at $n = 100$, 94% at $n = 3000$).

**why:** $e^{2Z}$ is extremely right-skewed: its mean $e^2 \approx 7.4$ is pulled up by rare huge values (one draw in a thousand exceeds $e^{6.2} \approx 480$). a sample of 100 usually contains none of those rare values, so its mean is too low, *and* its sample standard deviation is too low, so the interval is both off-center and too short. the occasional sample that does catch a huge value overshoots by a lot, which is how the average stays unbiased. the $\pm 1.96$ SE recipe relies on the CLT (topic 05) having kicked in, and for heavy-tailed quantities that takes a very large $n$. in practice: be suspicious of mean ± 2 SE for skewed metrics (revenue per user, importance weights, rewards in RL, per-example losses with outliers). look at the distribution, use more samples, or use the bootstrap (ch 05 · topic 03), which at least shows the asymmetry.

## experiment 3: the eigenvalues of a sample covariance

**what people usually predict:** eigenvalues clustered tightly around 1, perhaps from 0.9 to 1.1, since 200 samples feels like plenty.

**what happens:** with $n = 200$ and $d = 100$, the eigenvalues spread from about 0.10 to 2.8, following the Marchenko–Pastur curve almost exactly. with $n = 120$ the smallest is 0.01 and the largest 3.7. even with $n = 2000$ (20 samples per dimension) they range from 0.62 to 1.48. the *mean* eigenvalue is 1.00 every time.

**why:** each entry of $\hat\Sigma$ is a good estimate, with error about $1/\sqrt n$, but there are $d^2$ of them, and the eigenvalues feel all those errors at once. random matrix theory gives the limit: when $n, d \to \infty$ with $d/n = q$, the eigenvalues of the sample covariance of isotropic data fill the interval $[(1 - \sqrt q)^2, (1 + \sqrt q)^2]$ with the Marchenko–Pastur density. the spread depends on $d/n$, not on $n$ alone, so "samples per dimension" is the number to watch. the mean stays right because the trace (the sum of eigenvalues) is the sum of the diagonal variances, each estimated without bias. the consequences are everywhere in ML: PCA on this pure noise would "find" a top component with 2.8 times the average variance, the smallest eigenvalues make $\hat\Sigma^{-1}$ (used in Gaussian classifiers and Mahalanobis distances) badly overestimated in some directions, and shrinkage estimators like $\hat\Sigma + \lambda I$ (ch 08, ch 11) exist precisely to pull the spectrum back in.

reference: Marchenko & Pastur (1967), "Distribution of eigenvalues for some sets of random matrices", *Mathematics of the USSR-Sbornik* 1(4): 457–483. [doi:10.1070/SM1967v001n04ABEH001994](https://doi.org/10.1070/SM1967v001n04ABEH001994)
