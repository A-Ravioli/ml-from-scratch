# generalized linear models · lab notes

## experiment 1: modelling count data

**what people usually predict:** a few negative predictions at most, and similar errors: "with 300 days, a linear model is fine for averages".

**what happens** (300 training days, 2,000 test days):

| model | negative predictions | test RMSE | mean Poisson deviance | $D^2$ |
|---|---|---|---|---|
| linear regression | **7.9%** | 3.99 | 2.104 | 0.579 |
| Poisson GLM | 0.0% | **2.92** | **1.099** | **0.780** |
| the true means | 0.0% | 2.88 | 1.072 | 0.786 |

the Poisson GLM is almost as good as knowing the true means; least squares predicts an impossible negative count on one test day in 13, and its deviance is twice the GLM's. the GLM's coefficients are $(0.62, -0.77, 0.32)$ against a truth of $(0.6, -0.8, 0.3)$: rain multiplies rentals by 0.46 (truth 0.45). on the fit plot, the linear model's rain and no-rain lines are parallel and cross zero on cold days, while the GLM's curves follow the truth: rain costs about 1 rental on a cold day and about 18 on a hot one.

**why:** the data are multiplicative: $\mu = \exp(\dots)$. a linear model can only add a constant for rain, so it averages the effect over temperatures, overstates it on cold days (going negative) and understates it on hot ones; it also can't bend to follow the exponential growth with temperature. the residual plot shows the variance growing with the mean (a Poisson count's standard deviation is $\sqrt\mu$), which least squares ignores, weighting a miss of 5 at $\mu = 30$ the same as at $\mu = 2$. the GLM's IRLS weights account for exactly that. the diagonal edge of blue points at the lower left is days with 0 rentals: their residual is $0 - \hat y$, a line of slope $-1$, and when $\hat y$ is negative the residual is positive: the model undershot, predicting an impossible count below the 0 that was observed.

## experiment 2: when the variance is wrong

**what people usually predict:** overdispersion biases the slope; the intervals are "a bit too narrow", maybe 90% coverage.

**what happens** (300 datasets of 200 points, true slope 0.5):

| data | mean slope | actual spread | model-based se | coverage of the "95%" CI | median dispersion | coverage with se × √dispersion |
|---|---|---|---|---|---|---|
| Poisson counts | 0.496 | 0.039 | 0.041 | 95% | 0.99 | 95% |
| overdispersed counts | 0.491 | 0.092 | 0.041 | **57%** | 3.92 | 93% |

with overdispersed counts, the slope is still centered on the truth, but it really varies more than twice as much as the Poisson standard error says, and the textbook interval misses the truth in 43% of datasets. the Pearson dispersion is about 4 (the variance $\mu + \mu^2$ averages about 4 times $\mu$ over these data), and multiplying the standard errors by $\sqrt{\hat\phi} \approx 2$ restores coverage to 93%.

**why:** the Poisson GLM's estimating equation, $A^\top(y - \mu) = 0$, only uses the mean of $y$; if $\log\mu$ is really linear in $x$, it's unbiased (on average) whatever the variance is. the standard error, though, comes from the Fisher information, $(A^\top\operatorname{diag}(\mu)A)^{-1}$, which is the inverse variance of the estimate *only if* $\operatorname{Var}(y) = \mu$. with a variance $\phi$ times larger, the estimate's variance is $\phi$ times larger too (on paper 4), so the interval is $\sqrt\phi$ times too narrow. the Pearson dispersion measures $\phi$ from the residuals, which is the quasi-Poisson fix. it's only exactly right when the variance is proportional to $\mu$ (here it's $\mu + \mu^2$, so the correction is approximate, hence 93% rather than 95%); a negative binomial GLM models the extra variance directly, and sandwich standard errors or the bootstrap need no variance model at all. the practical lesson: a Poisson model that reports tiny p-values on real count data is usually overconfident. always check the dispersion.
