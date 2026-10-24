# classical forecasting · lab notes

numbers are from the reference solution. **every series in this lab is synthetic** (`demand_series` in the lab's setup cell: trend, a two-harmonic daily cycle, lower weekends, a slowly drifting seasonal amplitude, AR(1) noise with $\phi = 0.85$ per hour, and a Poisson(2) number of permanent level shifts of size $\mathcal N(0, 8^2)$). no real time series is cached for the course. the backtests use ten independent series (seeds 0–9), each ten weeks of hourly data, with 28 day-ahead origins at midnight over the last four weeks. the whole lab runs in about 10 seconds on one CPU core; nothing in it is random beyond the fixed seeds.

## experiment 1: reading the memory of a series

**what people usually predict:** that both plots decay for both models, or the reverse assignment (ACF cuts off for the AR).

**what happens:**

| series | ACF lags 1–4 | PACF lags 1–4 |
|---|---|---|
| AR(2), $\phi = (0.6, 0.25)$ | 0.81, 0.73, 0.64, 0.56 | 0.81, 0.24, −0.01, −0.02 |
| MA(2), $\theta = (0.7, 0.4)$ | 0.59, 0.22, −0.03, −0.05 | 0.59, −0.18, −0.12, 0.10 |
| demand, raw | 0.95, 0.85, 0.71, 0.54 | 0.95, −0.53, −0.26, −0.10 |
| demand, $(1 - B^{168})y$ | 0.86, 0.74, 0.63, 0.53 | 0.86, −0.03, 0.00, −0.02 |

the AR(2)'s PACF cuts off after lag 2 and its ACF decays geometrically; the MA(2)'s ACF cuts off after lag 2 and its PACF decays with alternating signs. (the theoretical MA(2) values are $r_1 = (\theta_1 + \theta_1\theta_2)/(1 + \theta_1^2 + \theta_2^2) = 0.594$ and $r_2 = \theta_2/(1 + \theta_1^2 + \theta_2^2) = 0.242$.) the raw demand ACF oscillates with period 24 and stays large out to lag 200, with the biggest peak near lag 168: the seasonality dominates everything. after the weekly difference, the PACF has one big spike (0.86) and nothing else: the differenced series looks like an AR(1), which is what the simulator put in (AR(1) noise with 0.85). the ACF of the differenced series also has a negative dip around lag 168, the signature of differencing the noise as well as the season (a seasonal MA term with $\theta \approx -1$ in Box–Jenkins terms, which this topic's ARIMA doesn't include).

**why:** an AR(p) is a Markov chain on its last $p$ values, so given them, older values add nothing (PACF zero beyond $p$), but each value is still correlated with all earlier ones through the chain (ACF decays). an MA(q) is a sum of the last $q + 1$ shocks, so values more than $q$ apart share nothing (ACF zero beyond $q$), but writing it as an infinite AR (when invertible) needs every lag (PACF decays).

## experiment 2: the backtest

**what people usually predict:** the ARIMAs all beat the naive baselines comfortably, often by 30–50%; few people expect any ARIMA to lose to "repeat yesterday".

**what happens:** (MASE relative to the in-sample daily seasonal-naive error; mean ± standard error over the 10 series)

| method | MASE | range over series |
|---|---|---|
| naive | 2.250 ± 0.060 | 1.97–2.49 |
| seasonal naive, m = 24 | 0.991 ± 0.020 | 0.89–1.08 |
| seasonal naive, m = 168 | 0.789 ± 0.059 | 0.63–1.14 |
| ARIMA(2, 1, 1) | 3.279 ± 0.095 | 2.80–3.67 |
| ARIMA(2, 0, 1), daily difference | 1.062 ± 0.028 | 0.95–1.23 |
| ARIMA(2, 0, 1), weekly difference | 0.695 ± 0.039 | 0.57–0.97 |

paired over the 10 series, the weekly-differenced ARIMA minus weekly seasonal naive is −0.094 MASE (standard error 0.027; ARIMA better on 9 of 10 series): a 12% gain. the error-by-horizon plot shows where it comes from: at 1 hour ahead the ARIMA's MAE is about 2.2 against seasonal naive's 5.3, by 12 hours the gap is under 1, and from about hour 17 on the two curves lie on top of each other.

**why:** ARIMA(2, 1, 1) has no idea there is a daily cycle. after first differencing, its forecast quickly flattens to "the current level plus a drift", which misses the whole daily swing: worse than the naive forecast, which at least is right at the current hour. the daily-difference ARIMA knows the daily shape but not the weekend, and its AR part (with $\phi_1 + \phi_2$ near 1) *carries the weekend's deviation forward*: on Saturday it learns "today is 10 below yesterday" and on Sunday it predicts that again, when Sunday is actually like Saturday. that is worse than the constant-zero deviation that pure "repeat yesterday" assumes, so a model that contains the baseline as a special case loses to it out of sample: the CSS fit chose the parameters that minimize *one-step* errors, which say nothing about 24-step behaviour across a day boundary. the weekly-difference ARIMA has the right seasonal structure, and its AR part predicts the deviation from last week; with $\phi \approx 0.85$ per hour, that knowledge is worth a lot at hour 1 and nearly nothing at hour 24 ($0.85^{24} \approx 0.02$), where the forecast collapses onto seasonal naive. the level shifts hurt every method equally for a week (until the shifted level enters "last week") and help explain the large spread across series (seasonal naive ranges from 0.63 to 1.14).

if the forecast feeds a day-ahead decision made at midnight, the honest summary is "the right ARIMA is 12% better than repeating last week, and all of that is in the first half of the day". a team without the time to identify the weekly structure should ship seasonal naive (m = 168), not a generic ARIMA.

## experiment 3: are the intervals honest?

**what people usually predict:** roughly 95% everywhere, or a little below because parameters are estimated; some predict that intervals get worse with the horizon.

**what happens:**

| setting | 1 h ahead | 24 h ahead | all horizons |
|---|---|---|---|
| simulated ARIMA(1,1,1), ARIMA(1,1,1) fit | 0.97 | 0.94 | 0.94 |
| demand, AR(2) after weekly difference | 0.96 | 0.94 | 0.96 |
| demand, ARIMA(2,0,1) after daily difference | 0.47 | 0.93 | 0.88 |

(280 origins per setting, so each fraction has a binomial standard error of about 0.013 near 0.95.) the correctly specified models are close to nominal at every horizon. the daily-difference model's 1-hour intervals contain the truth less than half the time. broken down by the weekday being forecast: Mon 0.03, Tue 0.15, Wed 0.93, Thu 0.95, Fri 0.97, Sat 0.28, Sun 0.03.

**why:** with more than a thousand observations per fit, parameter estimates are precise and the psi-weight formula, which treats them as known, is accurate when the model is right. the daily-difference model is wrong in a specific way: it treats the weekend drop as a persistent shock. its one-step errors are small *within* a day, and those dominate the in-sample $\hat\sigma^2$, so the interval is narrow; but at midnight on Saturday (the drop starts), Sunday (it doesn't continue: the model predicts another drop), Monday (it reverses) and Tuesday (the model predicts another rise), the forecast is off by about 10, several standard deviations. the in-sample variance can't warn you because those boundary hours are a small fraction of the training errors; only a backtest at the decision time reveals it. by 24 hours ahead the interval has widened enough ($\sum\psi_j^2$ has grown) to cover the error most of the time.
