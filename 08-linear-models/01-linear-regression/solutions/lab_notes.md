# linear regression · lab notes

## experiment 1: polynomial bases overfit

**what people usually predict:** the best degree is "around 3 to 5"; at degree 14 the training error is zero and the test error is "bad", maybe twice the best.

**what happens** (15 training points, 2,000 test points, noise $\sigma = 0.25$):

| model | test RMSE |
|---|---|
| predict the training mean | 0.832 |
| degree 3 (the best) | **0.295** |
| degree 9 | about 0.5 |
| degree 14 | **10,761** |

the training RMSE falls steadily, and at degree 14 it's $2.2 \times 10^{-4}$. the largest coefficient grows from 40 at degree 3 to $5.2 \times 10^{12}$ at degree 14. the degree-14 curve passes through the training points and then shoots off by thousands near the ends of the interval and in the gaps between points.

**why:** 15 parameters (14 coefficients plus the intercept) for 15 points means the polynomial can interpolate any targets, noise included, so the training error has nothing left to measure. to bend through every noisy point, the fit adds huge, nearly cancelling multiples of $x^9, \dots, x^{14}$, which are almost parallel on $[0, 1]$: the cancellations hold at the training points and nowhere else. away from them, and especially near the ends where high powers grow fastest and fewer points pin the curve down, the uncancelled pieces dominate. that's the variance term from chapter 07, topic 04, in its most extreme form. the training RMSE isn't exactly 0 because the design matrix (with its ones column) has a condition number of about $5 \times 10^{14}$, and with coefficients of size $10^{12}$ the rounding errors in the fitted values are no longer negligible, even with QR. the cure is fewer or better-behaved features (try the RBF basis) or a penalty on large coefficients, which is topic 02.

## experiment 2: normal equations versus QR

**what people usually predict:** both lose accuracy at the same rate, maybe with the normal equations a bit worse; "double precision is plenty".

**what happens:** the condition number of the design matrix grows about 5.6× per degree, from 4 at degree 1 to $2.3 \times 10^{10}$ at degree 14. the QR error tracks the dotted $\kappa\epsilon$ line and stays below $10^{-7}$ throughout: about 8 correct digits even at degree 14. the normal equations track the dashed $\kappa^2\epsilon$ line: $10^{-7}$ at degree 8, $10^{-2}$ at degree 10, and errors of order 1 (no correct digits) from degree 12. Cholesky didn't fail on these matrices, but with 200 equally spaced points instead of 100 it raises `LinAlgError` from degree 12 on: the rounded $X^\top X$ is no longer positive definite.

**why:** forming $X^\top X$ squares the condition number, and that is done *before* any clever solver gets to work, so the information is lost at that step. QR works on $X$ directly with orthogonal transformations, which don't amplify rounding errors, so the error is governed by $\kappa(X)$. once the normal equations' relative error reaches 1, the computed coefficients are unrelated to the true ones; they can still give reasonable *predictions* on the training points, because the ill-conditioned directions are exactly those that barely change $Xw$. scaling the columns barely helps here: the problem isn't that the columns have different sizes, it's that they point in nearly the same direction. a better basis does help, which is why numerical analysts use orthogonal polynomials (Legendre, Chebyshev) for high-degree fits.

## experiment 3: gradient descent and the shape of the bowl

**what people usually predict:** scaling a feature by 10 makes gradient descent "10 times slower", or doesn't matter because the step size is tuned to $1/L$.

**what happens** (tolerance $10^{-8}$ on the gradient norm, $\eta = 1/L$):

| scale $s$ | condition number $\kappa$ | GD steps (raw) | GD steps (standardized) |
|---|---|---|---|
| 1 | 1 | 9 | 6 |
| 3 | 8 | 142 | 6 |
| 10 | 91 | 1,664 | 6 |
| 30 | 815 | 15,049 | 6 |

the steps grow with $\kappa$, which grows like $s^2$: going from $s = 3$ to $s = 30$ multiplies the condition number by 100 and the step count by 106. after standardizing, every row takes 6 steps.

**why:** with $\eta = 1/L$ the slowest error component shrinks by $1 - 1/\kappa$ per step, so reaching tolerance takes about $\kappa\ln(\|g_0\|/\text{tol})$ steps. here $\ln(\|g_0\|/10^{-8}) \approx 18$, and indeed every row has steps $\approx 18\kappa$ (142/8, 1,664/91, 15,049/815 are all about 18). multiplying a feature by $s$ multiplies its variance, and so its eigenvalue of $X^\top X/n$, by $s^2$. the step size has to respect the steepest direction, so the flat direction crawls. standardizing gives both features unit variance; with independent features $H$ is then nearly the identity, $\kappa \approx 1$, and a step of $1/L$ jumps almost straight to the minimum. the exact solvers are indifferent to $s$ (in exact arithmetic the coefficient just shrinks by $s$), because they don't move through the landscape at all. this is the conditioning argument behind standardizing features before any gradient-based training, and behind preconditioning and adaptive optimizers (chapter 06).
