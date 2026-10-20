# kalman filters

## why this matters

a radar reports an aircraft's position once a second, each report off by tens of metres. the aircraft's velocity isn't reported at all. a GPS receiver in a tunnel gets no fix for thirty seconds. a sensor on a bridge measures strain that is mostly noise around a slowly drifting true value. in each case there's a hidden quantity that evolves over time according to known physics, and noisy, partial, sometimes missing measurements of it. you want the best estimate of the hidden state *now*, updated in real time as each measurement arrives, together with an honest statement of how uncertain it is.

the Kalman filter (Kalman, 1960) is the exact answer when the dynamics are linear and the noise is Gaussian, and a remarkably good approximate answer far beyond that. it put Apollo on the Moon, runs in every phone's navigation stack, and is the backbone of econometric state-space modelling. it's also the simplest member of the family this block is about: a linear recurrence on a hidden state, $x_t = Fx_{t-1} + \dots$. topics 07 and 08 build deep sequence models out of exactly that recurrence. here you'll derive the filter from Gaussian conditioning, build it with the smoother that looks back, learn the noise levels from data, and find out what happens when the noise model is wrong.

## warm-up: predict first

1. you measure a constant with noise variance 4, ten times. what is the variance of your best estimate? now the "constant" drifts a little between measurements: will the best estimate still weight all ten measurements equally?
2. a tracking filter's measurement-noise variance is set 100× too large. does it react to a sudden turn faster or slower than it should? what about 100× too small?
3. the filter's estimate at time $t$ uses measurements up to $t$. if you're allowed to wait and use measurements after $t$ too, how much better can you do?
4. you never measure velocity, only position. can the filter estimate the velocity anyway?
5. exponential smoothing, $m_t = m_{t-1} + \alpha(y_t - m_{t-1})$, is a heuristic from the 1950s. is it optimal for anything?

## prerequisites check

- [ ] you can condition a multivariate Gaussian: $x \mid y$ for jointly Gaussian $(x, y)$ (→ ch 04 · topic 04)
- [ ] you know the log density of a multivariate Gaussian and why we use `slogdet` and `solve` instead of `det` and `inv` (→ ch 04 · topic 04; ch 02 · topic 08)
- [ ] you know AR and ARIMA models and their forecasts (→ topic 05)
- [ ] you can maximize a likelihood numerically and know why to optimize log-parameters (→ ch 05, ch 06)
- [ ] you've seen hidden Markov models: a hidden chain with noisy emissions, and the forward algorithm (→ ch 12)

## core ideas

### idea 1: linear-Gaussian state-space models

a **state-space model** has a hidden state $x_t \in \mathbb R^n$ that evolves by itself and an observation $y_t \in \mathbb R^m$ generated from the current state. in the linear-Gaussian case:

$$x_0 \sim \mathcal N(m_0, P_0), \qquad x_t = Fx_{t-1} + w_t, \qquad y_t = Hx_t + v_t, \qquad w_t \sim \mathcal N(0, Q),\ v_t \sim \mathcal N(0, R),$$

with all the noise terms independent. it's a hidden Markov model whose hidden state is continuous and Gaussian rather than discrete. a few examples show how much fits this form:

- **local level** (a random walk observed in noise): $n = m = 1$, $F = H = 1$. the "true" level wanders with variance $q$ per step, and you see it with variance $r$ added.
- **constant velocity** in one dimension: $x = (\text{position}, \text{velocity})$, $F = \begin{pmatrix}1 & \Delta t\\0 & 1\end{pmatrix}$, $H = (1\ \ 0)$. random accelerations of intensity $q$ give $Q = q\begin{pmatrix}\Delta t^3/3 & \Delta t^2/2\\ \Delta t^2/2 & \Delta t\end{pmatrix}$ (integrate a white-noise acceleration over one step). in two dimensions, take one copy per axis.
- **any ARMA model**: an AR(p) is a state-space model with state $(y_t, \dots, y_{t-p+1})$, $F$ the companion matrix, $H = (1, 0, \dots, 0)$ and $R = 0$; MA terms add shocks to the state. so topic 05's ARIMA is a special case, and the Kalman filter computes its exact likelihood (rather than the conditional one), handles missing values, and gives the same forecasts.

the state is where the memory lives: given $x_t$, the past and future of the series are independent. that's what makes recursive estimation possible.

> **spec.** `simulate_lgssm(F, H, Q, R, m0, P0, T, *, seed=None)` returns `(x, y)` with shapes `(T, n)` and `(T, m)` (inputs are 2-D arrays, `m0` 1-D). with `rng = np.random.default_rng(seed)`, draw in this order: `x[0] = rng.multivariate_normal(m0, P0)`; for `t = 1..T-1`, `x[t] = F @ x[t-1] + rng.multivariate_normal(zeros(n), Q)`; then the observation noise in one call, `rng.multivariate_normal(zeros(m), R, size=T)`, and `y = x @ H.T + noise`.

> **▶ build: stage 1.** implement `simulate_lgssm`.

### idea 2: the filter is Gaussian conditioning, one step at a time

**filtering** means computing $p(x_t \mid y_{0:t})$. because everything is linear and Gaussian, the joint distribution of all states and observations is one big Gaussian, and $p(x_t \mid y_{0:t})$ is Gaussian too. you could compute it by building the $(n + m)T$-dimensional joint and conditioning (the tests do exactly that on small problems), but that costs $O(T^3)$ and can't run online. the Kalman filter gets the same answer in $O(T)$ by keeping only the current Gaussian, $\mathcal N(m_t, P_t)$, and updating it in two steps.

**predict.** if $x_{t-1} \mid y_{0:t-1} \sim \mathcal N(m_{t-1}, P_{t-1})$, then $x_t = Fx_{t-1} + w_t$ is a linear map plus independent noise:

$$m_t^- = Fm_{t-1}, \qquad P_t^- = FP_{t-1}F^\top + Q.$$

(at $t = 0$ there is no previous step: the prediction is the prior, $m_0^- = m_0$, $P_0^- = P_0$.)

**update.** given the prediction, $(x_t, y_t)$ is jointly Gaussian with $\mathbb E[y_t] = Hm_t^-$, $\operatorname{Cov}(y_t) = S_t = HP_t^-H^\top + R$, and $\operatorname{Cov}(x_t, y_t) = P_t^-H^\top$. conditioning on the observed $y_t$ (ch 04's formula) gives

$$K_t = P_t^-H^\top S_t^{-1}, \qquad m_t = m_t^- + K_t\,(y_t - Hm_t^-), \qquad P_t = P_t^- - K_tS_tK_t^\top .$$

read the update as a correction: the **innovation** $\nu_t = y_t - Hm_t^-$ is the part of the observation the prediction didn't expect, and the **Kalman gain** $K_t$ decides how much of it to believe. a large $R$ (noisy sensor) makes $S_t$ large and $K_t$ small: trust the prediction. a large $P_t^-$ (uncertain prediction) makes $K_t$ large: trust the measurement. in one dimension with $H = 1$, $K = P^-/(P^- + r)$, a precision-weighted average.

**the likelihood comes for free.** by the chain rule, $p(y_{0:T-1}) = \prod_tp(y_t \mid y_{0:t-1})$, and each factor is the Gaussian density of the innovation: $y_t \mid y_{0:t-1} \sim \mathcal N(Hm_t^-, S_t)$. so

$$\log p(y_{0:T-1}) = -\frac12\sum_t\Big(m\log 2\pi + \log\det S_t + \nu_t^\top S_t^{-1}\nu_t\Big).$$

this **prediction-error decomposition** is how you fit state-space models by maximum likelihood (idea 5).

**missing data cost nothing.** if $y_t$ is missing, there's nothing to condition on: skip the update ($m_t = m_t^-$, $P_t = P_t^-$) and add nothing to the likelihood. the filter coasts on its dynamics and its uncertainty grows.

> **spec.** `kalman_filter(y, F, H, Q, R, m0, P0)` takes `y` of shape `(T, m)` (a 1-D `y` means $m = 1$) and returns a dict with `"means"` `(T, n)` and `"covs"` `(T, n, n)` (filtered: $m_t$, $P_t$), `"pred_means"` and `"pred_covs"` (predicted: $m_t^-$, $P_t^-$, the prior at $t = 0$), and `"loglik"` (a float). a row of `y` containing any NaN is missing: skip its update and its likelihood term. don't modify the inputs.

> **🔮 predict:** in the scalar random walk with $q = r = 1$ and prior $\mathcal N(0, 1)$, you observe $y_0 = 1$, $y_1 = 2$. what are $m_0$ and $m_1$? (the worked example has the answer.)

> **▶ build: stage 2.** implement `kalman_filter`.

### idea 3: smoothing looks back

the filter answers "where is it now?". offline, you can ask "where was it then, given everything?": $p(x_t \mid y_{0:T-1})$, the **smoothing** distribution. the **Rauch–Tung–Striebel** (RTS) smoother (Rauch, Tung & Striebel, 1965) gets it with one backward pass over the filter's output. start at the end, where filtering and smoothing coincide, and go back:

$$G_t = P_tF^\top(P_{t+1}^-)^{-1}, \qquad m_t^s = m_t + G_t\,(m_{t+1}^s - m_{t+1}^-), \qquad P_t^s = P_t + G_t\,(P_{t+1}^s - P_{t+1}^-)\,G_t^\top .$$

the logic (on paper 2): given $x_{t+1}$, the future observations tell you nothing more about $x_t$ (Markov property), so you only need $p(x_t \mid x_{t+1}, y_{0:t})$, which is Gaussian conditioning of the filtered $x_t$ on the predicted $x_{t+1}$ (the gain $G_t$ is that regression coefficient). then average that over what you now know about $x_{t+1}$. if the smoothed $x_{t+1}$ differs from what was predicted at $t$, the correction propagates back.

the smoother is the right tool whenever you're not making decisions in real time: reconstructing a trajectory after the fact, filling gaps in a record, or the E-step of the EM algorithm for learning $F$, $Q$, $R$ (Shumway & Stoffer, 1982). its covariances are never larger than the filter's (more conditioning can only shrink a Gaussian's covariance), and through a gap in the data it's least certain in the middle, not at the end.

> **spec.** `rts_smoother(filt, F)` takes `kalman_filter`'s dict and returns `(means, covs)` of shapes `(T, n)` and `(T, n, n)`: the smoothed moments, equal to the filtered ones at $t = T - 1$. don't modify `filt`.

> **▶ build: stage 3.** implement `rts_smoother`.

### idea 4: the steady state, and checking a filter without ground truth

look at the covariance recursion: $P_t^-$ and $P_t$ never depend on the observations, only on $F$, $H$, $Q$, $R$ and $P_0$. for a time-invariant model that is observable (the measurements can, over time, see every direction of the state) and whose noise excites every unstable direction, $P_t^-$ converges to a fixed point $P$ of the **Riccati recursion**

$$P = F\big(P - PH^\top(HPH^\top + R)^{-1}HP\big)F^\top + Q,$$

the discrete algebraic Riccati equation, and the gain converges to $K = PH^\top(HPH^\top + R)^{-1}$. the **steady-state filter** uses this constant gain from the start: no covariance bookkeeping at all, and after a short transient it is exactly as good. if the model isn't detectable (say an unstable state that the measurements can't see), $P_t^-$ grows forever, and the iteration must give up rather than run forever.

for the local level model the fixed point is a quadratic: $P^2 - qP - qr = 0$, so $P = \frac{q + \sqrt{q^2 + 4qr}}{2}$ and $K = \frac{P}{P + r}$. the steady-state filter is then $m_t = m_{t-1} + K(y_t - m_{t-1})$: **simple exponential smoothing** with $\alpha = K$. the 1950s heuristic is the optimal filter for a random walk in noise, and the right amount of smoothing is set by the signal-to-noise ratio $q/r$ (lab experiment 4 connects it to topic 05's ARIMA).

**consistency.** in practice you never know the true state, so how do you check that a filter's uncertainty is honest? use the innovations. if the model is right, $\nu_t \sim \mathcal N(0, S_t)$ independently over time, so the **normalized innovation squared**

$$\text{NIS}_t = \nu_t^\top S_t^{-1}\nu_t \sim \chi^2_m,$$

with mean $m$. a time-average of NIS far above $m$ means the filter is overconfident (its $S_t$ is too small: $R$ or $Q$ too small); far below means underconfident. this is the standard consistency test in tracking (Bar-Shalom, Li & Kirubarajan, 2001, §5.4) and needs only the data.

> **spec.** `steady_state_gain(F, H, Q, R, *, tol=1e-10, max_iter=10_000)` iterates the Riccati recursion from any positive-definite start until successive $P$'s agree to `tol` (relative), and returns `(K, P)` with `K` of shape `(n, m)` and `P` the steady-state **predicted** covariance; it raises `RuntimeError` if it hasn't converged after `max_iter` iterations. `nis(y, filt, H, R)` returns the `(T,)` array of $\nu_t^\top S_t^{-1}\nu_t$ from the filter's predicted moments, NaN where `y` is missing.

> **🔮 predict:** with $R$ set 20× too small, will the mean NIS be about 20× too large, or less? (the filter adapts its $P$ too.)

> **▶ build: stage 4.** implement `steady_state_gain` and `nis`.

### idea 5: learning the noise

where do $Q$ and $R$ come from? sometimes from a datasheet (the sensor's noise) and physics (how hard the target can accelerate). often you have a structure but not the scale: you trust the shape of $Q$ from the constant-velocity model, but not its intensity. then fit the scales by maximum likelihood: with multipliers $q$, $r > 0$,

$$(\hat q, \hat r) = \arg\max_{q, r}\ \log p(y_{0:T-1} \mid F, H, qQ, rR, m_0, P_0),$$

using the filter's prediction-error decomposition as the objective. two scalars don't need gradients: a derivative-free optimizer on $(\log q, \log r)$ (which keeps them positive and puts both on comparable scales) converges in a hundred or so filter runs. for more parameters you would differentiate through the filter, or use EM with the smoother.

why can the data tell $q$ from $r$ at all, when you never see the state? because they leave different fingerprints on the observations. in the local level model, the differences $y_t - y_{t-1} = w_t + v_t - v_{t-1}$ have variance $q + 2r$ and lag-1 autocovariance $-r$: measurement noise makes successive differences *negatively* correlated, process noise doesn't. the likelihood uses exactly this information (and more).

> **spec.** `fit_noise_scales(y, F, H, Q, R, m0, P0)` returns `(q, r)`, positive floats that maximize `kalman_filter(y, F, H, q * Q, r * R, m0, P0)["loglik"]` (to within a few percent in each multiplier). missing rows are allowed. how you optimize is up to you.

> **▶ build: stage 5.** implement `fit_noise_scales`.

## worked example

**two steps by hand.** scalar random walk, $F = H = 1$, $q = r = 1$, prior $\mathcal N(0, 1)$, observations $y_0 = 1$, $y_1 = 2$.

- $t = 0$: no predict step, so $m_0^- = 0$, $P_0^- = 1$. $S_0 = 1 + 1 = 2$, $K_0 = 1/2$, $m_0 = 0 + \frac12(1 - 0) = 0.5$, $P_0 = 1 - \frac12\cdot2\cdot\frac12 = 0.5$.
- $t = 1$: predict $m_1^- = 0.5$, $P_1^- = 0.5 + 1 = 1.5$. $S_1 = 2.5$, $K_1 = 0.6$, $m_1 = 0.5 + 0.6\cdot1.5 = 1.4$, $P_1 = 1.5 - 0.6\cdot2.5\cdot0.6 = 0.6$.
- log-likelihood: $\log\mathcal N(1; 0, 2) + \log\mathcal N(2; 0.5, 2.5) = -1.5155 - 1.8271 = -3.3426$.
- NIS: $1^2/2 = 0.5$ and $1.5^2/2.5 = 0.9$.
- smoothing back to $t = 0$: $G_0 = 0.5/1.5 = 1/3$, $m_0^s = 0.5 + \frac13(1.4 - 0.5) = 0.8$, $P_0^s = 0.5 + \frac19(0.6 - 1.5) = 0.4$.
- steady state for $q = r = 1$: $P = (1 + \sqrt5)/2 = 1.618$ and $K = 1.618/2.618 = 0.618$, the golden ratio's reciprocal. the filter's $K_t$ ($0.5, 0.6, 0.615, \dots$) converges to it within a few steps.

your code must reproduce the filter, likelihood, NIS and steady-state numbers.

## on paper

1. **the update from conditioning.** starting from the joint Gaussian of $(x_t, y_t)$ given $y_{0:t-1}$, derive $m_t$, $P_t$ and $K_t$.

<details><summary>solution</summary>

$\begin{pmatrix}x_t\\y_t\end{pmatrix} \sim \mathcal N\left(\begin{pmatrix}m^-\\Hm^-\end{pmatrix}, \begin{pmatrix}P^- & P^-H^\top\\HP^- & HP^-H^\top + R\end{pmatrix}\right)$, since $\operatorname{Cov}(x, Hx + v) = P^-H^\top$ ($v$ independent of $x$). the conditioning formula $\mu_x + \Sigma_{xy}\Sigma_{yy}^{-1}(y - \mu_y)$, $\Sigma_{xx} - \Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}$ gives $m = m^- + P^-H^\top S^{-1}(y - Hm^-)$ and $P = P^- - P^-H^\top S^{-1}HP^-$. with $K = P^-H^\top S^{-1}$, the last term is $KSK^\top$ ($KSK^\top = P^-H^\top S^{-1}SS^{-1}HP^-$).

</details>

2. **the RTS step.** using $p(x_t \mid x_{t+1}, y_{0:T-1}) = p(x_t \mid x_{t+1}, y_{0:t})$, derive the smoother's mean and covariance recursion.

<details><summary>solution</summary>

given $y_{0:t}$, $(x_t, x_{t+1})$ is jointly Gaussian with $\operatorname{Cov}(x_t, x_{t+1}) = P_tF^\top$ and $\operatorname{Cov}(x_{t+1}) = P_{t+1}^-$. conditioning: $x_t \mid x_{t+1} \sim \mathcal N(m_t + G(x_{t+1} - m_{t+1}^-),\ P_t - GP_{t+1}^-G^\top)$ with $G = P_tF^\top(P_{t+1}^-)^{-1}$. now average over $x_{t+1} \sim \mathcal N(m_{t+1}^s, P_{t+1}^s)$: the mean becomes $m_t + G(m_{t+1}^s - m_{t+1}^-)$, and by the law of total variance the covariance is $P_t - GP_{t+1}^-G^\top + GP_{t+1}^sG^\top$.

</details>

3. **exponential smoothing is ARIMA(0, 1, 1).** in the steady-state local-level filter, write $y_t = m_{t-1} + \epsilon_t$ (the innovation). show that $y_t - y_{t-1} = \epsilon_t - (1 - K)\epsilon_{t-1}$.

<details><summary>solution</summary>

the filter is $m_t = m_{t-1} + K\epsilon_t$. then $y_t - y_{t-1} = (m_{t-1} + \epsilon_t) - (m_{t-2} + \epsilon_{t-1}) = K\epsilon_{t-1} + \epsilon_t - \epsilon_{t-1} = \epsilon_t - (1 - K)\epsilon_{t-1}$: an ARIMA(0, 1, 1) with $\theta = -(1 - K)$, whose innovations are the filter's. heavy smoothing (small $K$) means $\theta$ close to $-1$: the differenced series is nearly pure noise.

</details>

4. **what identifies the noise.** for the local level model, compute the variance and lag-1 autocovariance of $\Delta y_t = y_t - y_{t-1}$, and show how to recover $q$ and $r$ from them.

<details><summary>solution</summary>

$\Delta y_t = w_t + v_t - v_{t-1}$. variance $q + 2r$; covariance with $\Delta y_{t-1} = w_{t-1} + v_{t-1} - v_{t-2}$ is $-r$ (the shared $v_{t-1}$, with opposite signs); higher lags zero. so $r = -\gamma_1$ and $q = \gamma_0 + 2\gamma_1$. it's a method-of-moments estimator; maximum likelihood uses the same information more efficiently.

</details>

## experiment

the lab tracks a simulated object moving in the plane with random accelerations, measured in position only. experiment 1 compares raw measurements, your filter and your smoother, on position and on the never-measured velocity. experiment 2 gives the filter wrong noise levels, from 100× too small to 100× too large, and measures both the error and whether the filter's uncertainty is honest. experiment 3 puts a 40-step gap in the data and learns the noise levels from a single trajectory. experiment 4 fits a local level model and your topic 05 ARIMA(0, 1, 1) to the same series and compares them.

## check your understanding

1. why doesn't the covariance recursion depend on the observed values, and what does that let you do offline?

<details><summary>answer</summary>

$P_t^-$ and $P_t$ depend only on $F$, $H$, $Q$, $R$, $P_0$ and which steps are observed, because Gaussian conditioning's covariance doesn't depend on the conditioning value. you can precompute the gains (or use the steady-state gain) before any data arrive, check the filter's achievable accuracy at design time, and run the filter with one matrix-vector product per step.

</details>

2. you scale $Q$, $R$ and $P_0$ all by the same factor $s$. what happens to the gain, the estimates, and the covariances?

<details><summary>answer</summary>

every covariance in the recursion scales by $s$ ($P_t^-$, $P_t$, $S_t$), so $K_t = P_t^-H^\top S_t^{-1}$ is unchanged and so are all the means. only the reported uncertainty changes, by $s$. the estimates depend only on the *ratios* of the noise levels; the uncertainty depends on their scale. this is why lab experiment 2's heatmap has a nearly constant diagonal for the error but not for the coverage (with $P_0$ fixed, the diagonal is only approximately constant).

</details>

3. the smoother's uncertainty in the middle of a gap is smaller than the filter's at the end of the gap. why?

<details><summary>answer</summary>

the filter at the end of the gap has extrapolated for the whole gap from one side. the smoother in the middle has data on both sides, each only half a gap away, and interpolating between two anchors is far more certain than extrapolating from one. (for a random-walk state the variance grows linearly with distance from an anchor; for constant velocity, cubically, which makes the contrast dramatic.)

</details>

4. why is an overconfident filter (too-small $R$ or $Q$) dangerous even when its estimates are decent?

<details><summary>answer</summary>

anything downstream that uses the covariance (gating which measurement belongs to which target, sensor fusion weighting, collision checks, decisions with a safety margin) trusts a number that's wrong. an overconfident tracker rejects valid measurements as outliers and can lose the target entirely; a fusion system overweights it. NIS catches this without ground truth.

</details>

5. topic 05's ARIMA fitting conditioned on the first $p$ observations. what does the Kalman filter do differently, and when does it matter?

<details><summary>answer</summary>

written as a state-space model with the stationary distribution as the prior, the Kalman filter's prediction-error decomposition gives the *exact* likelihood, including the first observations, and handles missing values. it matters for short series, for MA parameters near the non-invertible boundary (where the conditional residuals' zero start-up washes out slowly), and for gappy data. for long complete series the two estimates nearly coincide.

</details>

6. the Kalman filter is optimal among all estimators for linear-Gaussian models. what does it lose if the noise is not Gaussian?

<details><summary>answer</summary>

it stays the best *linear* estimator (minimum mean squared error among linear functions of the data) for any noise with the given covariances. a nonlinear estimator can do better: with heavy-tailed measurement noise, a single outlier drags the Kalman estimate by $K\nu_t$, while a robust filter would discount it. the reported covariance is still the right linear-estimator error covariance, but probabilities computed from a Gaussian with that covariance are no longer exact.

</details>

## common bugs

- **a predict step before the first observation.** symptom: the hand example's $m_0$ comes out as 2/3 instead of 0.5, and the filter disagrees with brute-force conditioning at $t = 0$. fix: the prior *is* the prediction for $x_0$.
- **`inv(S)` with a transposed gain.** symptom: correct in one dimension, wrong for vector observations. fix: $K = P^-H^\top S^{-1}$, computed as `solve(S, H @ P).T`.
- **treating NaN observations as zeros.** symptom: the estimate dives toward zero through a gap. fix: skip the update when any element of the row is NaN.
- **the smoother gain with the filtered covariance at $t + 1$.** symptom: smoothed means that don't match conditioning. fix: $G_t$ uses the *predicted* $P_{t+1}^-$, and the correction compares the smoothed $m_{t+1}^s$ with the *predicted* $m_{t+1}^-$.
- **an uncapped Riccati loop.** symptom: `steady_state_gain` never returns on an undetectable model. fix: `max_iter` and raise.
- **optimizing $q$ and $r$ directly.** symptom: the optimizer steps to a negative variance and the filter returns NaN. fix: optimize $\log q$ and $\log r$.

## going deeper

- Kalman (1960), "A new approach to linear filtering and prediction problems", *Journal of Basic Engineering* 82(1). [doi:10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- Rauch, Tung & Striebel (1965), "Maximum likelihood estimates of linear dynamic systems", *AIAA Journal* 3(8). the RTS smoother. [doi:10.2514/3.3166](https://doi.org/10.2514/3.3166)
- Särkkä & Svensson (2023), *Bayesian Filtering and Smoothing*, 2nd ed., Cambridge University Press. ch 6 (Kalman filter), ch 12 (RTS smoother), ch 16 (parameter estimation); a free PDF of the first edition is on the author's site. [doi:10.1017/9781108917407](https://doi.org/10.1017/9781108917407)
- Durbin & Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press. state-space forms of ARIMA, exact likelihoods, the local level model (ch 2). [doi:10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001)
- Bar-Shalom, Li & Kirubarajan (2001), *Estimation with Applications to Tracking and Navigation*, Wiley. NIS/NEES consistency tests (§5.4), the white-noise acceleration model (§6.2). [doi:10.1002/0471221279](https://doi.org/10.1002/0471221279)
- Shumway & Stoffer (1982), "An approach to time series smoothing and forecasting using the EM algorithm", *Journal of Time Series Analysis* 3(4). learning $F$, $Q$, $R$ with the smoother. [doi:10.1111/j.1467-9892.1982.tb00349.x](https://doi.org/10.1111/j.1467-9892.1982.tb00349.x)
- Muth (1960), "Optimal properties of exponentially weighted forecasts", *Journal of the American Statistical Association* 55(290). exponential smoothing is optimal for a random walk plus noise. [doi:10.1080/01621459.1960.10482053](https://doi.org/10.1080/01621459.1960.10482053)

## connections

- ← builds on: Gaussian conditioning (ch 04), hidden Markov models and the forward algorithm (ch 12), maximum likelihood (ch 05), ARIMA and its forecasts (topic 05).
- → used in: topic 07 keeps the linear recurrence $x_t = \bar Ax_{t-1} + \bar Bu_t$ but drops the noise and the probabilistic reading: the state becomes a learned memory, the input drives it, and the "filter" is trained end to end; topic 08 makes the recurrence's matrices depend on the input, which a Kalman filter with time-varying $F_t$ and $Q_t$ already allows.
