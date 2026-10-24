# classical forecasting · hints

one hint per stage. open it only after you've been stuck for a while.

## stage 1

<details><summary>hint</summary>

`difference` is a loop of `d` passes of `out = out[lag:] - out[:-lag]`. for `acf`, center once (`z = x - x.mean()`), compute `denom = z @ z`, and for each lag `z[:n - k] @ z[k:] / denom`. for `pacf`, keep the current coefficient vector `phi` (length $k - 1$) and, at each $k$, compute `a = (r[k] - phi @ r[1:k][::-1]) / (1 - phi @ r[1:k])`, then `phi = concatenate([phi - a * phi[::-1], [a]])`; `a` is the lag-$k$ partial autocorrelation. check it against `scipy.linalg.solve_toeplitz(r[:k], r[1:k + 1])[-1]`.

</details>

## stage 2

<details><summary>hint</summary>

for `simulate_arma`, a plain Python loop over $t$ with `if t - i >= 0` guards is fine and easy to get right; `scipy.signal.lfilter` is the fast version: `u = c + lfilter(r_[1, theta], [1], e)` then `x = lfilter([1], r_[1, -phi], u)`, both starting from zeros. drop `x[:burn]`. for `fit_ar`, build the design matrix with columns `ones`, `x[p-1:n-1]`, `x[p-2:n-2]`, …, `x[0:n-p]` (lag 1 first) and use `np.linalg.lstsq` against `x[p:]`.

</details>

## stage 3

<details><summary>hint</summary>

residuals: first the AR prediction errors `w[t] = x[t] - c - sum_i phi[i-1] * x[t-i]` for `t >= p`, then the MA part is the filter `e = lfilter([1], r_[1, theta], w[p:])` (that's $\hat e_t = w_t - \sum_j\theta_j\hat e_{t-j}$ started from zeros); put zeros in front. for the fit: if `q == 0` return `fit_ar`. otherwise get a start from Hannan–Rissanen (a long AR fit, its residuals as stand-ins for the shocks, then least squares of `x[t]` on `1`, lagged `x`, lagged residuals), and hand `lambda v: mean(arma_residuals(x, v[0], v[1:1+p], v[1+p:])[p:] ** 2)` to `scipy.optimize.minimize`. bounding each $\theta_j$ to $(-0.999, 0.999)$ with `method="L-BFGS-B"` keeps the optimizer out of the non-invertible region.

</details>

## stage 4

<details><summary>hint</summary>

in `fit`, difference with `difference(y, 1, lag=m)` (if seasonal) and then `difference(w, d)`, call `fit_arma`, and keep `y` and the residuals of the differenced series. in `forecast`, get the levels polynomial with `np.convolve`: start from `r_[1, -phi]`, convolve `d` times with `[1, -1]` and once with `[1, 0, …, 0, -1]` (length `m + 1`); then `a = -poly[1:]`. extend `y` by `h` slots and the residuals (aligned to `y`'s indices: the first `len(y) - len(resid)` are zero) by `h` zeros, and run `y[t] = c + sum_i a[i-1] y[t-i] + sum_j theta[j-1] e[t-j]` for the new slots. psi: `psi[0] = 1`, `psi[j] = theta_j + sum_{i=1..j} a_i psi[j-i]`, std `= sqrt(sigma2 * cumsum(psi**2))`.

</details>

## stage 5

<details><summary>hint</summary>

`seasonal_naive`: `[y[n - m + (j % m)] for j in range(h)]`. `mase`: the scale is `mean(abs(insample[m:] - insample[:-m]))`. `rolling_origin`: `origins = np.arange(initial, len(y) - horizon + 1, step)` (the `+ 1` keeps the last origin whose window ends exactly at the end of the series), and for each origin call `forecaster(y[:t].copy(), horizon)`.

</details>
