# classical forecasting

## why this matters

a grid operator has to commit generating capacity for tomorrow by noon today. a warehouse has to order stock a week before customers want it. a web team has to provision servers for next Monday's traffic. every one of these is a **forecast**: a statement about the future made from the past of the same series, often with nothing else to go on. it's also a setting where machine-learning habits mislead in specific ways. the observations aren't independent, so random train/test splits leak the future into the past. the series isn't stationary, so a model fit on last year describes a process that no longer exists. and the strongest competitor is often embarrassingly simple: "the same as this hour last week" beats many sophisticated models, and forecasting competitions have shown this over and over (Makridakis et al., 2018).

this topic builds the classical toolkit that every time-series method, including the state-space models later in this block, is measured against: what stationarity means and how differencing gets you there, how autocorrelations reveal a series' memory, the ARIMA family and how to fit it, how forecast uncertainty grows with the horizon, and how to backtest honestly. in the lab you'll find out how hard the seasonal-naive forecast is to beat on realistic synthetic demand data, and when ARIMA's intervals can be trusted.

## warm-up: predict first

1. a series is a random walk: $y_t = y_{t-1} + e_t$. you compute its sample autocorrelation at lag 1 from 1,000 observations. roughly what do you get?
2. you fit an AR(1) to hourly data and forecast 1,000 hours ahead. what does the forecast look like at that range?
3. the standard deviation of a random walk's forecast error one step ahead is $\sigma$. what is it 100 steps ahead?
4. you have five years of daily sales. you shuffle the days, do 5-fold cross-validation of a model that uses the previous 7 days as features, and get a great score. what went wrong?
5. on hourly data with a weekly pattern, a forecast that repeats the value from exactly one week ago is a baseline. how much better, in percent, do you think a carefully fitted ARIMA does?

## prerequisites check

- [ ] you can fit least squares and read a design matrix (→ ch 08 · topic 01)
- [ ] you know the multivariate Gaussian and can compute the variance of a sum of independent terms (→ ch 04 · topics 03–04)
- [ ] you know what a Markov chain and a stationary distribution are (→ ch 04 · topic 07)
- [ ] you know why a test set must not influence training, and how to report uncertainty across seeds (→ ch 07 · topics 02–03)
- [ ] you can minimize a smooth function with `scipy.optimize.minimize` (→ ch 06)

## core ideas

### idea 1: stationarity, differencing, and autocorrelation

a time series $y_1, \dots, y_n$ is one draw from a stochastic process, and you never see a second draw. to learn anything you have to assume that something about the process stays the same over time. the standard assumption is **weak stationarity**: the mean $\mathbb E[y_t] = \mu$ doesn't depend on $t$, and the covariance $\operatorname{Cov}(y_t, y_{t+k}) = \gamma_k$ depends only on the lag $k$. then every pair of observations $k$ apart is a sample of the same relationship, and averaging over time estimates it.

real series usually violate this: they trend, they have seasons, their level shifts. the classical move is to transform the series until it looks stationary. the workhorse is **differencing**. writing $B$ for the backshift operator ($By_t = y_{t-1}$), the first difference $(1 - B)y_t = y_t - y_{t-1}$ removes a linear trend (and turns a random walk into white noise); $(1 - B)^2$ removes a quadratic one; and the **seasonal difference** $(1 - B^m)y_t = y_t - y_{t-m}$ removes any pattern that repeats every $m$ steps, whatever its shape. a seasonal difference also removes a linear trend's slope, leaving a constant $m \times$ slope.

once a series is (roughly) stationary, its **autocorrelation function** (ACF) summarizes its memory:

$$\hat\gamma_k = \frac1n\sum_{t=1}^{n-k}(y_t - \bar y)(y_{t+k} - \bar y), \qquad r_k = \hat\gamma_k / \hat\gamma_0 .$$

note the $1/n$ for every lag, even though only $n - k$ products enter the sum. it biases $r_k$ toward zero at large lags, but it guarantees that the estimated autocovariances form a positive semidefinite sequence (a valid covariance), which the methods below rely on. for white noise, $r_k$ is approximately $\mathcal N(0, 1/n)$, which gives the $\pm 2/\sqrt n$ bands you see on ACF plots. at the other extreme, a random walk's sample ACF starts near 1 (about 0.99 at lag 1 for $n = 1000$) and decays only slowly and roughly linearly: the classic sign of a non-stationary series that needs a difference.

the ACF mixes direct and indirect dependence: in an AR(1) $y_t = \phi y_{t-1} + e_t$, $y_t$ depends on $y_{t-2}$ only through $y_{t-1}$, yet $r_2 = \phi^2 \ne 0$. the **partial autocorrelation** at lag $k$ removes the intermediate lags: it is the last coefficient $\phi_{kk}$ of the best linear predictor of $y_t$ from $y_{t-1}, \dots, y_{t-k}$. those coefficients solve the **Yule–Walker equations** $R_k\phi = r_{1:k}$, where $R_k$ is the $k \times k$ Toeplitz matrix of autocorrelations, and the **Durbin–Levinson recursion** computes all of them in $O(k^2)$ by reusing the order-$(k-1)$ solution:

$$\phi_{kk} = \frac{r_k - \sum_{j=1}^{k-1}\phi_{k-1,j}\,r_{k-j}}{1 - \sum_{j=1}^{k-1}\phi_{k-1,j}\,r_j}, \qquad \phi_{k,j} = \phi_{k-1,j} - \phi_{kk}\,\phi_{k-1,k-j}.$$

> **spec.** `difference(x, d=1, *, lag=1)` applies $(1 - B^{\text{lag}})^d$ and returns a float array of length `len(x) - d * lag` (`d=0` returns the series unchanged). `acf(x, nlags)` returns $r_0, \dots, r_{\text{nlags}}$ (so `nlags + 1` values, $r_0 = 1$) with the $1/n$ autocovariance above, deviations taken from the overall mean. `pacf(x, nlags)` returns $\phi_{00} = 1, \phi_{11}, \dots, \phi_{\text{nlags},\text{nlags}}$ from Durbin–Levinson on `acf(x, nlags)`.

> **🔮 predict:** for an AR(2), which of ACF and PACF will be exactly zero (up to noise) beyond lag 2? for an MA(2)?

> **▶ build: stage 1.** implement `difference`, `acf` and `pacf`.

### idea 2: autoregressive models

an **AR(p)** model says the series is a linear function of its own past plus a fresh shock:

$$y_t = c + \phi_1y_{t-1} + \dots + \phi_py_{t-p} + e_t, \qquad e_t \overset{\text{iid}}{\sim} \mathcal N(0, \sigma^2).$$

in operator form, $\phi(B)y_t = c + e_t$ with $\phi(B) = 1 - \phi_1B - \dots - \phi_pB^p$. it is stationary when every root of the polynomial $\phi(z)$ lies outside the unit circle (for an AR(1): $|\phi_1| < 1$), and then its mean is $\mu = c / (1 - \sum\phi_i)$. the AR(p) is literally a Markov chain on the state $(y_{t-1}, \dots, y_{t-p})$, which is why its PACF cuts off after lag $p$: given the last $p$ values, older ones add nothing.

fitting it is just regression. condition on the first $p$ observations and regress $y_t$ on $(1, y_{t-1}, \dots, y_{t-p})$ for $t = p+1, \dots, n$. this **conditional least squares** estimate is the Gaussian maximum likelihood estimate conditional on the first $p$ values; it differs from the exact MLE only through those $p$ start-up terms, which matter little once $n \gg p$. the innovation variance estimate is the mean squared residual.

to simulate an ARMA (idea 3 adds the MA part), run the recursion forward from zeros and throw away a **burn-in**: the zero start isn't a draw from the stationary distribution, but its influence dies off geometrically.

> **spec.** `simulate_arma(n, phi=(), theta=(), *, c=0.0, sigma=1.0, burn=200, seed=None)`: draw all shocks at once, `e = np.random.default_rng(seed).normal(0, sigma, n + burn)`, run $y_t = c + \sum_i\phi_iy_{t-i} + e_t + \sum_j\theta_je_{t-j}$ for $t = 0, \dots, n + \text{burn} - 1$ with $y$ and $e$ taken as zero before $t = 0$, and return the last $n$ values. `fit_ar(x, p)` returns `(c, phi, sigma2)`: the least-squares coefficients of $x_t$ on $(1, x_{t-1}, \dots, x_{t-p})$ over $t = p, \dots, n - 1$ (`phi[0]` is the lag-1 coefficient, an array of length $p$) and the mean squared residual over those $n - p$ points. `p = 0` fits the mean.

> **▶ build: stage 2.** implement `simulate_arma` and `fit_ar`.

### idea 3: moving averages and ARMA

some memory is short and sharp rather than long and decaying: a shock that affects this hour and the next, then vanishes. that's a **moving average** term. an **ARMA(p, q)** model is

$$y_t = c + \sum_{i=1}^p\phi_iy_{t-i} + e_t + \sum_{j=1}^q\theta_je_{t-j}, \qquad \text{i.e.}\quad \phi(B)y_t = c + \theta(B)e_t .$$

an MA(q) has an ACF that cuts off after lag $q$ (shocks more than $q$ apart share nothing) while its PACF decays, the mirror image of an AR (on paper 2). mixing both gives parsimonious models for series whose ACF and PACF both decay.

the MA part makes fitting harder because the shocks $e_t$ aren't observed. the **conditional sum of squares** (CSS) method reconstructs them recursively: given parameters, set the residuals before $t = p$ to zero and compute

$$\hat e_t = y_t - c - \sum_i\phi_iy_{t-i} - \sum_j\theta_j\hat e_{t-j}, \qquad t = p, \dots, n - 1,$$

then choose the parameters that minimize $\frac{1}{n-p}\sum_t\hat e_t^2$. the recursion is a linear filter, $\hat e = \theta(B)^{-1}(\text{AR prediction errors})$, so it's stable when $\theta(z)$ has its roots outside the unit circle (**invertibility**); with $|\theta| \ge 1$ the zero start-up never washes out. the objective isn't quadratic in $\theta$, so you need a numerical optimizer and a good starting point. the classic start is **Hannan–Rissanen** (1982): fit a long AR to estimate the shocks, then regress $y_t$ on lagged $y$ and lagged estimated shocks. with $q = 0$, CSS *is* least squares, so `fit_arma(x, p, 0)` must agree with `fit_ar`.

> **spec.** `arma_residuals(x, c, phi=(), theta=())` returns an array the length of `x`: zeros for $t < p$, then the recursion above. `fit_arma(x, p, q)` returns `(c, phi, theta, sigma2)` minimizing the mean squared conditional residual over $t \ge p$, with `sigma2` that minimum. how you optimize is up to you (`scipy.optimize.minimize` from a Hannan–Rissanen start works well).

> **🔮 predict:** simulate $y = (1 + 0.4B)e$ and fit it with the sign convention flipped ($\hat e_t = y_t + \theta\hat e_{t-1}$). what $\hat\theta$ will the optimizer report?

> **▶ build: stage 3.** implement `arma_residuals` and `fit_arma`.

### idea 4: ARIMA, forecasting, and why uncertainty grows

put the two ideas together: difference the series until it's stationary, then model the differences as an ARMA. that's **ARIMA(p, d, q)**: $\phi(B)(1 - B)^dy_t = c + \theta(B)e_t$. with an additional seasonal difference you get ARIMA$(p, d, q)(0, 1, 0)_m$, which this topic writes as `ARIMA((p, d, q), seasonal_period=m)`:

$$\phi(B)\,(1 - B)^d\,(1 - B^m)\,y_t = c + \theta(B)\,e_t .$$

(the full seasonal ARIMA of Box & Jenkins also has seasonal AR and MA polynomials in $B^m$; you won't need them here.) note what the constant means after differencing: with $d = 1$ it's a drift (the forecast trends), with a seasonal difference it's a year-on-year (or week-on-week) growth.

**the point forecast.** multiply out $\Phi^*(B) = \phi(B)(1 - B)^d(1 - B^m) = 1 - a_1B - a_2B^2 - \dots$ and the model is a (non-stationary) ARMA in levels: $y_t = c + \sum_ia_iy_{t-i} + e_t + \sum_j\theta_je_{t-j}$. the minimum-mean-squared-error forecast replaces every future shock by its mean, zero, and every future $y$ by its own forecast, and uses the in-sample residuals $\hat e_t$ for past shocks:

$$\hat y_{n+h} = c + \sum_ia_i\,\tilde y_{n+h-i} + \sum_{j \ge h}\theta_j\,\hat e_{n+h-j}, \qquad \tilde y_s = \begin{cases}y_s & s \le n\\ \hat y_s & s > n.\end{cases}$$

stationary models forecast a return to the mean (an AR(1) decays toward $\mu$ like $\phi^h$); a random walk forecasts its last value forever; a seasonal random walk repeats its last season.

**the uncertainty.** write the model as an infinite moving average, $y_t = \text{(deterministic part)} + \sum_{j\ge0}\psi_je_{t-j}$, where the **psi weights** are the coefficients of $\theta(B)/\Phi^*(B)$, computed by the recursion $\psi_0 = 1$, $\psi_j = \theta_j + \sum_{i=1}^{j}a_i\psi_{j-i}$ (with $\theta_j = 0$ for $j > q$). the $h$-step forecast error is the sum of the shocks that haven't happened yet, $\sum_{j<h}\psi_je_{n+h-j}$, so

$$\operatorname{Var}(y_{n+h} - \hat y_{n+h}) = \sigma^2\sum_{j=0}^{h-1}\psi_j^2 .$$

for a stationary AR(1), $\psi_j = \phi^j$ and the variance levels off at the process variance $\sigma^2/(1 - \phi^2)$. for a random walk, $\psi_j = 1$ and the standard deviation grows like $\sqrt h$ without limit. a seasonal random walk has $\psi_j = 1$ only at multiples of $m$, so its uncertainty steps up once per season. building the psi weights from $\phi$ alone, forgetting the differencing, gives intervals that are far too narrow for every integrated model. (these intervals also ignore the uncertainty in the estimated parameters, which is usually small next to $\sigma^2$ when $n$ is in the hundreds.)

> **spec.** `ARIMA(order=(1, 0, 0), *, seasonal_period=None)`. `fit(y)` takes the seasonal difference at lag `seasonal_period` (once, if given), then `d` regular differences, fits `fit_arma(w, p, q)` to the result, stores `c_`, `phi_`, `theta_`, `sigma2_`, and returns `self`; refitting replaces everything. `forecast(h, *, return_std=False)` returns the length-`h` mean forecast of $y$ after the end of the fitted series, or `(mean, std)` with the psi-weight standard deviations above. forecasting must not change the fitted state.

> **▶ build: stage 4.** implement `ARIMA`.

### idea 5: baselines and honest backtesting

**the baselines.** the **naive** forecast repeats the last value; the **seasonal naive** forecast repeats the value from one season ago, $\hat y_{n+h} = y_{n+h-km}$ with $k = \lceil h/m\rceil$. they're the forecasts of a random walk and a seasonal random walk, and they're the first thing to beat: any method that can't beat them on a backtest isn't worth its complexity. Hyndman & Athanasopoulos (2021, §5.2) recommend reporting them alongside every model.

**the metric.** errors in raw units can't be compared across series of different scales, and percentage errors explode near zero. the **mean absolute scaled error** (Hyndman & Koehler, 2006) scales the forecast's MAE by the in-sample MAE of the one-step seasonal-naive forecast:

$$\text{MASE} = \frac{\frac1h\sum_{j=1}^h|y_{n+j} - \hat y_{n+j}|}{\frac{1}{n-m}\sum_{t=m+1}^n|y_t - y_{t-m}|}.$$

below 1 means better (out of sample) than seasonal naive was in sample.

**the protocol.** cross-validation by random folds is wrong for time series: a fold's test points have training neighbours on both sides, so the model interpolates using the future, and autocorrelation makes those neighbours near-copies of the test points. **rolling-origin evaluation** (also called time-series cross-validation or backtesting) replays history: for each origin $t$, fit on $y_1, \dots, y_t$ only, forecast $y_{t+1}, \dots, y_{t+h}$, record the errors, move the origin forward. it measures exactly what you'll experience in deployment, at the cost of one refit per origin.

> **spec.** `seasonal_naive(y, h, m=1)` returns `y[n - m + (j % m)]` for `j = 0..h-1` (the last observed value in the same season). `mase(actual, forecast, insample, m=1)` is the formula above. `rolling_origin(y, forecaster, *, initial, horizon, step=1)` calls `forecaster(history, horizon)` at origins `t = initial, initial + step, …` for every `t` with `t + horizon <= len(y)`, passing `history` as a copy of `y[:t]`, and returns `(origins, forecasts, actuals)`: an integer array of origins and two `(n_origins, horizon)` arrays, `actuals[i] = y[t_i : t_i + horizon]`.

> **🔮 predict:** at each origin the forecaster sees `y[:t]`. which index is the first one it must forecast, and what goes wrong if you pass `y[:t + 1]`?

> **▶ build: stage 5.** implement `seasonal_naive`, `mase` and `rolling_origin`.

## worked example

**autocorrelation by hand.** $y = (1, 2, 3, 4)$: $\bar y = 2.5$, deviations $(-1.5, -0.5, 0.5, 1.5)$, $n\hat\gamma_0 = 5$. lag 1: $(-1.5)(-0.5) + (-0.5)(0.5) + (0.5)(1.5) = 1.25$, so $r_1 = 0.25$. lag 2: $(-1.5)(0.5) + (-0.5)(1.5) = -1.5$, $r_2 = -0.3$. lag 3: $(-1.5)(1.5)$, $r_3 = -0.45$. (dividing lag $k$ by $n - k$ instead would give $r_1 = 0.333$: the check tells you which you did.)

**a seasonal random walk.** $(1 - B^4)y_t = c + e_t$. the forecast for the next 9 steps repeats the last four observed values, adding $c$ once per season passed: $\hat y_{n+j} = y_{n-3+((j-1) \bmod 4)} + c\,(1 + \lfloor (j-1)/4\rfloor)$ (with $y_1, \dots, y_n$ observed, so $\hat y_{n+1}$ repeats $y_{n-3}$). its psi weights are $1$ at lags $0, 4, 8$ and $0$ elsewhere, so the forecast standard deviations are $\sigma\cdot(1,1,1,1,\sqrt2,\sqrt2,\sqrt2,\sqrt2,\sqrt3)$.

**MASE.** in-sample $(1, 3, 2, 4, 3, 5)$; one-step naive errors $2, 1, 2, 1, 2$, mean $1.6$. forecasting $(5, 5)$ when the truth is $(6, 4)$ gives MAE 1, so MASE $= 1/1.6 = 0.625$. with $m = 2$ the in-sample errors are $|2 - 1|, |4 - 3|, |3 - 2|, |5 - 4|$, mean 1, and MASE $= 1$.

your code must reproduce all three.

## on paper

1. **AR(1) moments.** for $y_t = c + \phi y_{t-1} + e_t$ with $|\phi| < 1$, stationary, derive $\mu$, $\gamma_0$ and $r_k$.

<details><summary>solution</summary>

taking expectations, $\mu = c + \phi\mu$, so $\mu = c/(1 - \phi)$. with $z_t = y_t - \mu = \phi z_{t-1} + e_t$ and $z_{t-1}$ independent of $e_t$: $\gamma_0 = \phi^2\gamma_0 + \sigma^2$, so $\gamma_0 = \sigma^2/(1 - \phi^2)$. multiplying by $z_{t-k}$ and taking expectations: $\gamma_k = \phi\gamma_{k-1}$, so $r_k = \phi^k$.

</details>

2. **MA(1) memory.** for $y_t = e_t + \theta e_{t-1}$, compute $r_1$ and $r_k$ for $k \ge 2$. why can't $|r_1|$ exceed $1/2$? and why do $\theta$ and $1/\theta$ give the same ACF?

<details><summary>solution</summary>

$\gamma_0 = (1 + \theta^2)\sigma^2$, $\gamma_1 = \theta\sigma^2$, $\gamma_k = 0$ for $k \ge 2$ (no shared shocks). so $r_1 = \theta/(1 + \theta^2)$, which is maximized at $\theta = 1$ with value $1/2$. replacing $\theta$ by $1/\theta$: $\frac{1/\theta}{1 + 1/\theta^2} = \frac{\theta}{\theta^2 + 1}$, identical. the two models are indistinguishable from second moments; the invertible one ($|\theta| < 1$) is chosen by convention, and it's the one whose CSS residual recursion is stable.

</details>

3. **psi weights.** derive the psi weights of ARIMA(0, 1, 1), $(1 - B)y_t = (1 + \theta B)e_t$, and the variance of the $h$-step forecast error.

<details><summary>solution</summary>

$\Phi^*(B) = 1 - B$, so $a_1 = 1$. $\psi_0 = 1$; $\psi_1 = \theta + a_1\psi_0 = 1 + \theta$; $\psi_j = 0 + \psi_{j-1} = 1 + \theta$ for $j \ge 2$. the variance is $\sigma^2(1 + (h - 1)(1 + \theta)^2)$: it still grows linearly in $h$, more slowly than a random walk's when $\theta < 0$. (with $\theta = -1$ the series is white noise plus a constant, and the variance stays at $\sigma^2$.)

</details>

4. **leakage, quantified.** you predict $y_t$ of an AR(1) with $\phi = 0.9$ from its two neighbours $y_{t-1}$ and $y_{t+1}$ (which random-fold CV allows): first by their plain average, then by the best linear combination of them. compare the mean squared errors with the honest one-step forecast $\phi y_{t-1}$.

<details><summary>solution</summary>

honest: the error is $e_t$, variance $\sigma^2$. two-sided, with $\gamma_0 = \sigma^2/(1-\phi^2)$, $\gamma_k = \phi^k\gamma_0$: the plain average has error variance $\gamma_0(\tfrac32 + \tfrac12\phi^2 - 2\phi) = 5.263\cdot0.105\,\sigma^2 = 0.553\sigma^2$; the best linear interpolator, $\frac{\phi}{1 + \phi^2}(y_{t-1} + y_{t+1})$, has $\sigma^2/(1 + \phi^2) = 0.552\sigma^2$. either way, about $0.55\sigma^2$: random-fold CV can report almost half the honest error even with a model that, at forecast time, could never see $y_{t+1}$.

</details>

## experiment

the lab works on synthetic hourly demand series (trend, daily and weekly seasonality, AR noise and level shifts), simulated because no real time series is cached for this course and because a simulator tells you the truth. experiment 1 reads ACFs and PACFs of simulated ARMAs and of the demand series. experiment 2 backtests your ARIMAs against naive and seasonal-naive forecasts by rolling origin on ten series. experiment 3 checks whether your forecast intervals cover the truth as often as they claim.

## check your understanding

1. why does a seasonal difference at lag $m$ remove *any* periodic pattern of period $m$, while a regression on a few sine waves removes only the shapes you included?

<details><summary>answer</summary>

if $s_t = s_{t-m}$ for all $t$, then $(1 - B^m)s_t = 0$ identically, whatever the shape of $s$. sines remove only the harmonics you put in. the price of differencing: it adds noise (the difference of two noisy values has twice the noise variance if the noise is white) and throws away the first $m$ observations.

</details>

2. why use $1/n$ rather than $1/(n - k)$ in the autocovariance at lag $k$?

<details><summary>answer</summary>

with $1/n$ the matrix of autocovariances is guaranteed positive semidefinite, so it's a valid covariance and Yule–Walker / Durbin–Levinson give stationary AR fits. with $1/(n - k)$ the large-lag estimates have huge variance (few products) and the matrix can be indefinite. the price is a bias toward zero at large lags, of order $k/n$.

</details>

3. an AR(1) with $\phi = 0.95$ fitted to hourly data: what's the forecast at a 1-week horizon, and how wide is its interval compared with the 1-hour one?

<details><summary>answer</summary>

$0.95^{168} \approx 2\times10^{-4}$, so the forecast is the mean. the standard deviation goes from $\sigma$ to $\sigma/\sqrt{1 - 0.95^2} \approx 3.2\sigma$, the unconditional standard deviation. a stationary model forgets; at long horizons it can only offer the mean and the marginal spread.

</details>

4. two ARIMA models have almost identical in-sample $\hat\sigma^2$. can their forecast intervals differ a lot?

<details><summary>answer</summary>

yes. the in-sample residual variance is about one-step errors. the $h$-step variance multiplies it by $\sum\psi_j^2$, which depends on the differencing and the AR roots: an ARIMA(1, 0, 0) with $\phi = 0.98$ and an ARIMA(0, 1, 0) fit the same data almost equally well one step ahead, but the first's intervals level off and the second's grow like $\sqrt h$ forever.

</details>

5. why is the seasonal-naive forecast so hard to beat on series with strong seasonality?

<details><summary>answer</summary>

it captures the seasonal shape exactly (whatever it is), adapts instantly to level shifts (after one season), and has no parameters to estimate or misspecify. a model has to beat it by predicting the deviations from last season, and those are often close to unpredictable beyond a few steps. at long horizons, a stationary model's forecast of the deviations returns to zero, which *is* the seasonal-naive forecast.

</details>

6. you backtest with origins every 24 hours, always at midnight. what does that measure, and what doesn't it?

<details><summary>answer</summary>

it measures day-ahead forecasts issued at midnight, which is right if that's how the forecast will be used. it doesn't measure forecasts issued at other hours, which can behave very differently: the lab's daily-difference model fails specifically at the start of certain days. choose the origins to match the decision the forecast feeds.

</details>

7. MASE below 1 on a backtest. is your model better than seasonal naive?

<details><summary>answer</summary>

not necessarily: the denominator is the seasonal-naive *in-sample one-step* error. at a 24-step horizon seasonal naive itself can score above or below 1. compare against the seasonal-naive forecast's own out-of-sample MASE, on the same origins, and report the paired difference across series with its uncertainty.

</details>

## common bugs

- **dividing the lag-$k$ autocovariance by $n - k$.** symptom: ACF values a bit too large at long lags, PACF doesn't match the Yule–Walker oracle, the hand example gives $r_1 = 0.333$. fix: one denominator, $\sum(y_t - \bar y)^2$, for every lag.
- **psi weights from $\phi$ alone.** symptom: intervals for integrated models are too narrow and, for $p \ge 1$, level off when they should grow (for $p = 0$ there is no $\phi$ at all, so the recursion degenerates to $\psi_j = \theta_j$ and the random-walk check fails). fix: multiply out $\phi(B)(1 - B)^d(1 - B^m)$ first, by polynomial multiplication (`np.convolve`), not by adding $-1$ at lag 1.
- **forgetting the MA terms when forecasting.** symptom: the 1-step forecast of an MA(1) is just $c$. fix: use the in-sample residuals for past shocks; only future shocks are zero.
- **a residual recursion that starts at $t = 0$ with zero-padded $y$.** symptom: residuals don't match the oracle in the first few positions, and CSS estimates are slightly off. fix: condition on the first $p$ observations; their residuals are zero.
- **an off-by-one in rolling origin.** symptom: the last valid origin is missing (`t + horizon < n`), or the forecaster sees `y[t]` (peeking). fix: origins run while `t + horizon <= n`, and the forecaster gets exactly `y[:t]`.
- **a seasonal-naive forecast that doesn't wrap.** symptom: an index error, or nonsense, when `h > m`. fix: index with `j % m`.

## going deeper

- Hyndman & Athanasopoulos (2021), *Forecasting: Principles and Practice*, 3rd ed., OTexts. ch 5 (the forecaster's toolbox, baselines, time-series cross-validation), ch 9 (ARIMA). [otexts.com/fpp3](https://otexts.com/fpp3/)
- Box, Jenkins, Reinsel & Ljung (2015), *Time Series Analysis: Forecasting and Control*, 5th ed., Wiley. the original identification–estimation–diagnosis cycle; ch 5 on forecasting and psi weights.
- Brockwell & Davis (2016), *Introduction to Time Series and Forecasting*, 3rd ed., Springer. §2.5 (Durbin–Levinson), §5 (ARMA estimation). [doi:10.1007/978-3-319-29854-2](https://doi.org/10.1007/978-3-319-29854-2)
- Hannan & Rissanen (1982), "Recursive estimation of mixed autoregressive-moving average order", *Biometrika* 69(1). [doi:10.1093/biomet/69.1.81](https://doi.org/10.1093/biomet/69.1.81)
- Hyndman & Koehler (2006), "Another look at measures of forecast accuracy", *International Journal of Forecasting* 22(4). MASE. [doi:10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001)
- Makridakis, Spiliotis & Assimakopoulos (2018), "Statistical and Machine Learning forecasting methods: Concerns and ways forward", *PLOS ONE* 13(3). simple statistical methods against ML models on the M3 data. [doi:10.1371/journal.pone.0194889](https://doi.org/10.1371/journal.pone.0194889)
- Tashman (2000), "Out-of-sample tests of forecasting accuracy: an analysis and review", *International Journal of Forecasting* 16(4). rolling-origin evaluation. [doi:10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0)

## connections

- ← builds on: least squares (ch 08), the multivariate Gaussian and variances of sums (ch 04), Markov chains (ch 04 · topic 07), honest evaluation with baselines (ch 07), numerical optimization (ch 06).
- → used in: topic 06 rewrites ARIMA as a linear-Gaussian state-space model, and shows that the Kalman filter for a random walk observed in noise *is* exponential smoothing, an ARIMA(0, 1, 1) forecast; topics 07–08 build deep sequence models out of the same linear recurrences, and the block's project measures them against simple baselines in the same spirit as this topic's seasonal naive.
