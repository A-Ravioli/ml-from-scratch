# kalman filters · lab notes

numbers are from the reference solution. every trajectory is simulated (constant velocity in the plane, white-noise acceleration $q = 0.05$, position measured with $R = 4I$, prior $\mathcal N(0, \operatorname{diag}(25, 25, 1, 1))$, 200 steps), with fixed seeds, so the numbers are reproducible exactly. the lab runs in about 20 seconds on one CPU core.

## experiment 1: how much does filtering buy?

**what people usually predict:** a modest gain on position (10–30%), smoothing a little better than filtering, and "no way" to estimate velocity well without measuring it.

**what happens:** (RMS error, mean ± standard error over 20 trajectories)

| estimate | RMS error |
|---|---|
| measurements | 2.83 ± 0.02 |
| filter | 1.73 ± 0.02 |
| smoother | 1.00 ± 0.02 |
| velocity: differenced measurements | 3.98 ± 0.03 |
| velocity: filter | 0.63 ± 0.01 |
| velocity: smoother | 0.34 ± 0.00 |

the filter cuts the position error by 39%, the smoother by 65%. the velocity, never measured, is estimated with RMS error 0.63 by the filter and 0.34 by the smoother, while differencing the measurements gives 3.98, worse than the typical speed change over many steps.

**why:** the steady-state gain sets an exponentially decaying weighting of past measurements, with an effective memory of a few steps here (the position gain settles near 0.4). the filter can only look backwards, so its estimate lags slightly behind any change of velocity; the smoother looks both ways, and velocity is a *difference* of positions, which benefits most from having data on both sides. differencing measurements has error variance $2R$ per axis (two independent noises), RMS $\sqrt{2\cdot 2\cdot 4} = 4$ for two axes: the filter instead fits a smooth trajectory through many measurements.

## experiment 2: what bad noise models do

**what people usually predict:** errors grow symmetrically with misspecification in both directions, and scaling everything up "just makes the filter more cautious".

**what happens:** (10 trajectories per cell; rows $s_Q$, columns $s_R$)

RMS position error:

| $s_Q$ \ $s_R$ | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| 100 | 2.76 | 2.62 | 2.32 | 1.98 | 2.05 |
| 10 | 2.62 | 2.32 | 1.93 | 1.77 | 2.73 |
| 1 | 2.32 | 1.93 | **1.69** | 2.30 | 5.15 |
| 0.1 | 1.94 | 1.69 | 2.20 | 4.72 | 10.98 |
| 0.01 | 1.70 | 2.20 | 4.66 | 10.55 | 22.64 |

coverage of the filter's own 95% ellipse:

| $s_Q$ \ $s_R$ | 0.01 | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|---|
| 100 | 0.03 | 0.28 | 0.97 | 1.00 | 1.00 |
| 10 | 0.04 | 0.29 | 0.98 | 1.00 | 1.00 |
| 1 | 0.03 | 0.31 | **0.96** | 1.00 | 1.00 |
| 0.1 | 0.03 | 0.27 | 0.69 | 0.76 | 0.78 |
| 0.01 | 0.03 | 0.11 | 0.24 | 0.35 | 0.41 |

two things stand out. first, the error is (nearly) constant along every diagonal: $s_Q = s_R = 0.01$ gives 1.70, the same as the true model's 1.69, and $s_Q = s_R = 10$ gives 1.77. the estimate depends on the ratio $Q/R$, not on the scale. the small drift toward the top-right corner (2.05 at 100/100) comes from the prior $P_0$, which wasn't scaled: relative to a 100× noise model it's very confident, so the start-up transient is slow. second, the coverage tells the opposite story: along the same diagonal it runs from 0.03 (everything 100× too small, wildly overconfident, with the *best possible* error) to 1.00 (everything 100× too large).

the error is asymmetric too: $R$ 100× too small (2.32) costs much less than $R$ 100× too large (5.15), and $Q$ too small is worst of all (4.66 at $s_Q = 0.01$, rising to 22.6 with $R$ also too large). a filter that trusts its measurements too much becomes a slightly smoothed copy of them (error close to the raw 2.83); a filter that trusts its dynamics too much ignores the measurements and lags behind every random acceleration, which compounds over time.

**why:** scaling $Q$, $R$ (and $P_0$) by $s$ scales every covariance in the recursion by $s$, so the gain $K = P^-H^\top S^{-1}$ is unchanged and so are the means. the covariances themselves are $s$ times the true ones, which is exactly what the coverage measures. you can't detect a wrong scale from the error, only from consistency: the NIS in the slider panel averages about 2 (its expected value, $m = 2$) for the true model, and far above 2 when $R$ is too small. that's a check you can run on real data with no ground truth.

## experiment 3: gaps, and learning the noise from data

**what people usually predict:** the smoother is least certain at the end of the gap (like the filter), and the learned multipliers are rough, maybe within a factor of 2.

**what happens:** the filter's position standard deviation grows from about 1.2 to about 37 through the 40-step gap (cubically, as you'd expect from integrating random accelerations twice), then collapses at the first measurement. the smoother's is a symmetric bulge that peaks in the middle of the gap at step 100, at about 5.5. the filter's error drifts to about 15 by the end of the gap; the smoother's stays within its ±2 std band. from a single trajectory with the gap, starting from guesses 10× off in opposite directions, the learned multipliers are $q = 11.01$, $r = 0.091$ (truth: 10 and 0.1), i.e. the learned model has 1.10× the true $Q$ and 0.91× the true $R$. with it, the RMS position error on observed steps is 1.75 and the 95% ellipse covers the truth 97% of the time.

**why:** at step 100 the smoother has measurements 20 steps behind and 20 steps ahead; interpolating between two anchors pins the trajectory down far better than extrapolating from one. $q$ and $r$ are identifiable because they leave different fingerprints in the observations: measurement noise is uncorrelated from step to step (it makes successive differences negatively correlated), while process noise accumulates (it makes the path wander on long time scales). 190 observed steps are enough to separate them to about 10%.

## experiment 4: the Kalman filter you already know

**what people usually predict:** no particular relation, or $\theta = K$.

**what happens:** (2,000 steps per row; the topic 05 ARIMA is fitted by conditional sum of squares)

| true q / r | learned (q, r) | K (true model) | K (learned) | ARIMA(0,1,1) θ̂ | −(1 − K learned) |
|---|---|---|---|---|---|
| 0.05 | 0.045, 1.033 | 0.200 | 0.187 | −0.814 | −0.813 |
| 0.5 | 0.532, 1.003 | 0.500 | 0.510 | −0.490 | −0.490 |
| 5 | 5.418, 0.748 | 0.854 | 0.891 | −0.108 | −0.109 |

$\hat\theta = -(1 - K)$ to within 0.001 in every row, even though the two numbers come from different models fitted by different criteria.

**why:** the steady-state local-level filter is $m_t = m_{t-1} + K\epsilon_t$ with innovation $\epsilon_t = y_t - m_{t-1}$, and differencing $y_t = m_{t-1} + \epsilon_t$ gives $\Delta y_t = \epsilon_t - (1 - K)\epsilon_{t-1}$ (on paper 3). the local level model and the ARIMA(0, 1, 1) with $\theta \in (-1, 0)$ are the same model for the observations, in two parametrizations, and both fits maximize (essentially) the same Gaussian likelihood; they differ only in how they treat the start of the series, which hardly matters over 2,000 steps. the steady-state filter is simple exponential smoothing with $\alpha = K$: with a low signal-to-noise ratio ($q/r = 0.05$) the optimal $\alpha$ is about 0.2, heavy smoothing; with a high one ($q/r = 5$) it's about 0.85, almost no smoothing. "how much to smooth" is a statement about $q/r$.
