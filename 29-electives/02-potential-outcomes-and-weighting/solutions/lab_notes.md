# potential outcomes & weighting · lab notes

numbers below are from the reference solution with the lab's seeds (all data are simulated, so offline and online runs agree). your code gives the same numbers if your logistic regression from ch 08 converges to the same fit, and very close ones otherwise.

## experiment 1: when the weights explode

**what people usually predict:** the spread grows a bit as overlap worsens, maybe doubling; clipping is a harmless fix; AIPW is immune because it's "doubly robust".

**what happens:** bias / sd over 200 datasets of $n = 1{,}000$:

| $\gamma$ | Horvitz–Thompson | Hájek | clipped at 0.05 | AIPW | largest weight's share | ESS of ~500 treated |
|---|---|---|---|---|---|---|
| 0.5 | −0.00 / 0.07 | +0.00 / 0.07 | −0.00 / 0.07 | −0.00 / 0.07 | 0.3% | 459 |
| 1 | −0.00 / 0.10 | +0.00 / 0.11 | +0.00 / 0.10 | +0.00 / 0.08 | 0.8% | 362 |
| 2 | +0.03 / 0.22 | +0.05 / 0.26 | +0.10 / 0.12 | +0.01 / 0.15 | 2.9% | 167 |
| 3 | +0.09 / 0.55 | +0.18 / 0.41 | +0.15 / 0.13 | +0.05 / 0.42 | 4.8% | 93 |
| 4 | +0.06 / **1.30** | +0.22 / 0.65 | +0.14 / 0.13 | −0.02 / 0.41 | 6.7% | 75 |

the Horvitz–Thompson sd grows **18-fold** from $\gamma = 0.5$ to $\gamma = 4$ while the sample size stays at 1,000. the histogram at $\gamma = 4$ shows why: most datasets land near 2, a few land at −8 or +12. clipping keeps the sd near 0.13 at every $\gamma$, but its bias settles at +0.14 to +0.15. AIPW's sd also grows sixfold, much less than IPW's but far from immune.

**why:** at $\gamma = 4$ a median dataset has one unit carrying 6.7% of all the weight, and ~500 treated units are worth about 75 equally weighted ones. a weighted mean has variance roughly $\sigma^2/\mathrm{ESS}$, so the spread follows the ESS, and the occasional dataset with a treated unit at $e \approx 0.001$ produces the outliers. clipping caps the weights, so it caps the variance, but it treats those units as if they were more common than they are; that's harmless only if their outcomes look like everyone else's. here the effect is $2 + x_0$ and units with extreme scores have extreme $x_0$, so clipping systematically under-weights the people with the largest (and smallest) effects, and the bias doesn't go away with more data. AIPW divides only the *residuals* $y - \hat\mu$ by the scores; with a correct outcome model those residuals are pure noise with sd 1, much smaller than the outcomes themselves, so the explosion is damped but not removed.

## experiment 2: doubly robust

**what people usually predict:** AIPW is unbiased in every row except "both wrong", where it lands somewhere between IPW and regression.

**what happens:** bias and sd over 100 datasets of $n = 2{,}000$:

| estimator | propensity model | outcome model | bias | sd |
|---|---|---|---|---|
| AIPW | right | right | +0.005 | 0.078 |
| AIPW | right | wrong | +0.108 | 0.934 |
| AIPW | wrong | right | −0.000 | 0.049 |
| AIPW | wrong | wrong | **+3.017** | 0.221 |
| IPW (Hájek) | right | — | +0.139 | 0.693 |
| IPW (Hájek) | wrong | — | +2.651 | 0.190 |
| regression | — | right | −0.000 | 0.049 |
| regression | — | wrong | +2.257 | 0.132 |

with one model right, AIPW is unbiased up to noise (the +0.108 with the right propensity model is about 1.2 standard errors of the mean of 100 estimates, $0.934/\sqrt{100} = 0.093$). with both wrong it's the **worst** of all the estimators: +3.0, against +2.7 for IPW and +2.3 for regression.

**why:** the bias of AIPW is the product $\mathbb E[(e - \hat e)(\mu - \hat\mu)/\hat e]$ (on paper 3). if either factor is zero, so is the bias. when both are wrong, nothing makes the product small: here the missing $x_0^2$ term pushes both errors in the same direction in the same region (large $|x_0|$: more likely treated *and* higher outcomes), so they compound instead of cancelling. this is the Kang & Schafer (2007) lesson: "doubly robust" means two chances to be right, not protection against being wrong twice. note also the price of the right propensity model: with $x_0^2$ in the logit, scores approach 1 for large $|x_0|$, and the right-propensity rows have sd 0.7–0.9, fifteen to twenty times the right regression row's 0.049. the "wrong propensity, right outcome" AIPW row is as precise as regression, because when the outcome model is right the weighted residuals are small whatever the weights.

## experiment 3: do the error bars tell the truth?

**what people usually predict:** close to 95% everywhere ("it's a valid CI"), or a collapse to something like 50% under poor overlap.

**what happens:** 400 datasets per row:

| $n$ | $\gamma$ | coverage | median reported se | actual sd of estimates | datasets off by > 1 |
|---|---|---|---|---|---|
| 200 | 0.5 | 95.0% ± 1.1% | 0.163 | 0.165 | 0.0% |
| 200 | 2 | 92.8% ± 1.3% | 0.207 | 0.536 | 1.8% |
| 200 | 4 | **86.0% ± 1.7%** | 0.204 | 0.541 | 4.2% |
| 1,000 | 0.5 | 96.0% ± 1.0% | 0.073 | 0.071 | 0.0% |
| 1,000 | 2 | 95.2% ± 1.1% | 0.111 | 0.171 | 0.5% |
| 1,000 | 4 | 92.2% ± 1.3% | 0.155 | 0.449 | 2.2% |

with good overlap the reported se matches the real spread and coverage is nominal. with poor overlap and $n = 200$, the "95%" interval covers 86% of the time, and the actual sd of the estimates is 2.5–3× the typical reported se. more data helps (92% at $n = 1{,}000$), but slowly.

**why:** the se is the sample sd of the scores $\psi_i$ divided by $\sqrt n$. with poor overlap the $\psi_i$ are heavy-tailed: their variance is dominated by rare units with tiny $\hat e$. most datasets contain none of those units, so their sample sd *underestimates* the true variance, and the interval is too narrow; the few datasets that do contain one get a huge estimate error and a huge se at the same time. coverage averages over both, which is why it's "only" 86% while the sd mismatch looks much worse. the CLT's promise is asymptotic, and how large $n$ must be depends on how heavy the tails are, which is exactly what overlap controls. practical responses: report the propensity distribution and the ESS alongside every weighted estimate; prefer estimands that need less overlap (the ATT needs only $e < 1$ among controls, the overlap-weighted effect needs neither tail); or trim and say clearly that the estimand changed.
