# logistic regression · lab notes

## experiment 1: when a line separates the classes perfectly

**what people usually predict:** IRLS converges to "the" answer, perhaps with large weights; tightening `tol` changes the last few digits; gradient descent gets close to the same answer.

**what happens** (100 separable points):

| fit | steps | converged_ | $\|w\|$ | most extreme probability |
|---|---|---|---|---|
| IRLS, $\alpha = 0$, tol $10^{-4}$ | 10 | True | **56.3** | $10^{-56}$ |
| IRLS, $\alpha = 0$, tol $10^{-8}$ | 19 | True | **161.9** | $10^{-161}$ |
| IRLS, $\alpha = 0$, tol $10^{-12}$ | 28 | True | **269.9** | $10^{-268}$ |
| IRLS, $\alpha = 0.01$ | 6 | True | 3.8 | $1.7 \times 10^{-4}$ |

every unpenalized run reports success, and the three answers differ by a factor of 5. each extra 4 digits of tolerance costs 9 more steps and adds about 110 to $\|w\|$. gradient descent with $\alpha = 0$ reaches $\|w\| = 5.4$ after 100 steps, 12.4 after 1,000, 26.0 after 10,000 and 46.8 after 100,000, and is still climbing. with $\alpha = 0.01$ both solvers settle at $\|w\| = 3.8$ within a few hundred GD steps (6 IRLS steps).

**why:** if $w$ separates the data, so does $2w$, with every margin doubled and every loss term smaller: the loss has no minimum, only an infimum of 0 at infinity. the gradient, $\frac1nA^\top(s - y)$, is made of terms of size $e^{-\text{margin}}$, so it shrinks exponentially as the weights grow, and eventually drops below any `tol`; "converged" then only means "the gradient is small", and where that happens is set by `tol`. IRLS heads off to infinity at a steady pace (each Newton step adds roughly a constant to the margins, so $\|w\|$ grows linearly in the step count, and the stopping point moves linearly with $\log(1/\text{tol})$). gradient descent's steps are proportional to the gradient, which is exponentially small in $\|w\|$, so it needs exponentially many steps for each further unit: $\|w\|$ grows only logarithmically in the long run (Soudry et al., 2018 show that its *direction* converges to the max-margin separator). the probabilities are the real damage: $10^{-161}$ is a claim no finite dataset can support. the penalty $\frac\alpha2\|w\|^2$ eventually outgrows any decrease of the loss, so there's a unique finite optimum, both solvers find it, and the probabilities stay sane.

## experiment 2: Newton versus gradient descent

**what people usually predict:** IRLS is "a bit faster", maybe 5× fewer steps.

**what happens** (500 points, 5 features correlated 0.8): IRLS's gradient norms are $0.6, 0.2, 5\times10^{-2}, 8\times10^{-3}, 3\times10^{-4}, 7\times10^{-7}, 3\times10^{-12}$: below $10^{-8}$ after **6** steps, with the exponent roughly doubling over the last three. gradient descent with $\eta = 1/L \approx 1.0$ needs **905** steps. on the plot's logarithmic step axis its curve stays nearly flat for the first hundred steps and then bends down: that's a constant factor per step (linear convergence), just a slow one.

**why:** Newton's method minimizes a local quadratic model; near the optimum the model is excellent, and the error after a step is proportional to the *square* of the error before it (quadratic convergence), so once you're close, 3 steps take you from $10^{-3}$ to machine precision. gradient descent's error shrinks by a factor of about $1 - \mu/L$ per step, where $\mu$ and $L$ are the smallest and largest curvatures: correlated features make the Hessian badly conditioned (a long narrow valley along the direction where the features disagree), so $\mu/L$ is small and progress is slow. with $p = 6$ parameters, a Newton step costs barely more than a gradient step, so IRLS wins outright. with $p = 10^5$ (a bag of words), a Hessian has $10^{10}$ entries: then quasi-Newton (L-BFGS) or first-order methods are the only option, which is why scikit-learn defaults to L-BFGS.

## experiment 3: least squares as a classifier

**what people usually predict:** points that are already on the correct side can't hurt; maybe they help a little.

**what happens** (accuracy on the original 200 points):

| training set | least squares | logistic |
|---|---|---|
| original 200 points | 0.855 | 0.850 |
| plus 40 easy points | **0.785** | **0.850** |

on the original data the two boundaries nearly coincide. adding 40 points that both models already classify correctly tilts the least-squares boundary by about 45 degrees and costs it 7 points of accuracy; the logistic boundary doesn't visibly move.

**why:** least squares wants the fitted value to *equal* 1 at the far points, which sit about 8 units to the right and 3 up. on the original line the fit there would be far above 1, and the squared loss charges for overshooting exactly as for undershooting. to reduce that, it tilts the plane so that the fit grows less in the direction of the far cluster, and the 0.5 contour, the boundary, swings with it. the cross-entropy charges $\log(1 + e^{-m})$ for a point with margin $m$: for the far points $m$ is large, the loss and its gradient are essentially zero, and they exert no pull at all. the general lesson: pick a loss that matches what you care about. for classification, being *more* right than necessary should never be penalized.
