# kalman filters · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`rng = np.random.default_rng(seed)`; allocate `x = np.zeros((T, n))`; `x[0] = rng.multivariate_normal(m0, P0)`; loop `t = 1..T-1` adding `rng.multivariate_normal(np.zeros(n), Q)`; only after the loop draw `noise = rng.multivariate_normal(np.zeros(m), R, size=T)` and return `x, x @ H.T + noise`. the order of the draws is part of the spec, because the checks reproduce your random numbers.

</details>

## stage 2

<details><summary>hint</summary>

keep `m, P` (copies of `m0, P0`). in the loop: if `t > 0`, predict (`m = F @ m`, `P = F @ P @ F.T + Q`); store the prediction; if the row has no NaN, compute `v = y[t] - H @ m`, `S = H @ P @ H.T + R`, `K = np.linalg.solve(S, H @ P).T`, update `m = m + K @ v`, `P = P - K @ S @ K.T` (symmetrize with `0.5 * (P + P.T)`), and add `-0.5 * (m_dim * log(2 pi) + slogdet(S)[1] + v @ solve(S, v))` to the log-likelihood; store the filtered moments either way. reshape a 1-D `y` to `(T, 1)` first.

</details>

## stage 3

<details><summary>hint</summary>

copy the filtered means and covariances; loop `t` from `T - 2` down to `0`: `G = np.linalg.solve(pred_covs[t + 1], F @ covs[t]).T` (that's $P_tF^\top(P^-_{t+1})^{-1}$, using symmetry), then `ms[t] = means[t] + G @ (ms[t + 1] - pred_means[t + 1])` and `Ps[t] = covs[t] + G @ (Ps[t + 1] - pred_covs[t + 1]) @ G.T`.

</details>

## stage 4

<details><summary>hint</summary>

`steady_state_gain`: start from something positive definite (`Q + I`), and in a `for _ in range(max_iter)` loop compute `S`, `K` and `P_new = F @ (P - K @ S @ K.T) @ F.T + Q`; return when `max|P_new - P| < tol * max(1, max|P|)`, recomputing `K` from the final `P`; raise `RuntimeError` after the loop. `nis`: for each observed row, `v = y[t] - H @ pred_means[t]`, `S = H @ pred_covs[t] @ H.T + R`, value `v @ solve(S, v)`; start from `np.full(T, np.nan)`.

</details>

## stage 5

<details><summary>hint</summary>

define `nll(v) = -kalman_filter(y, F, H, exp(v[0]) * Q, exp(v[1]) * R, m0, P0)["loglik"]` and minimize it from `v = (0, 0)` with `scipy.optimize.minimize(..., method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})`; return `exp` of the solution. if your filter is slow, profile it: each likelihood evaluation runs the whole filter.

</details>
