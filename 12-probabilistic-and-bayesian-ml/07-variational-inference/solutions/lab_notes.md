# variational inference · lab notes

## experiment 1: how wide is the mean-field posterior?

**what people usually predict:** that mean-field VI gets the variances roughly right and only loses the correlation (the ellipse becomes an axis-aligned ellipse with the same marginal widths), so sds near 1.

**what happens:** (3,000 Adam steps at learning rate 0.01, shipped seeds) the fitted standard deviations follow $\sqrt{1 - \rho^2}$ almost exactly: 1.005/0.966 at $\rho = 0$, 0.873/0.839 at 0.5, 0.606/0.583 at 0.8, **0.440/0.424 at 0.9** (theory 0.436), 0.315/0.305 at 0.95 and 0.200/0.197 at 0.98 (theory 0.199). the true marginal sd is 1 throughout. the means are right. the full-covariance fit (dotted, left panel) lies on top of the target's contours.

**why:** the reverse KL $\mathbb E_q[\log q - \log p]$ is large wherever $q$ has mass and $p$ doesn't. an axis-aligned $q$ as wide as the marginals would put lots of mass in the two empty corners of the diagonal ellipse, where $\log p$ is very negative; it's cheaper to shrink until $q$ fits *inside* the ellipse. the exact optimum (on paper 4) sets each variance to $1/\Lambda_{ii}$, the variance of $z_i$ given all the others, which for this target is $1 - \rho^2$. the effect is not small: at $\rho = 0.9$ the reported 95% interval is 2.3 times too narrow. anything that depends on posterior spread is affected: credible intervals, posterior predictive uncertainty, decisions under risk, and the evidence estimate (the ELBO gap, 0.51 nats at $\rho = 0.8$). the mean is unbiased here because the target is Gaussian and symmetric; with skewed or multimodal posteriors it isn't. a full-covariance family removes this particular problem (it contains the target), at a cost of $O(d^2)$ parameters.

## experiment 2: which way round is the KL?

**what people usually predict:** that VI's Gaussian, like an average, sits between the two modes or spreads to cover both.

**what happens:** with the modes 7 apart (shipped seeds, 3,000 steps), all six VI runs land on the heavier mode: means between $+3.46$ and $+3.57$ (the mode is at $+3.5$), sds 0.95–1.06 (the mode's sd is 1). their $-\mathrm{KL}(q\|p)$ is about $-0.50$ to $-0.51$, which matches the value for a Gaussian sitting exactly on the 60% mode, $\log 0.6 = -0.511$ (the small differences are Monte Carlo noise). the moment-matched Gaussian has mean $+0.70$ and sd 3.57: it sits in the valley between the modes, where the target has almost no mass, and covers both. **but not every seed does this**: about 1 run in 12 at $d = 7$ ends in a broad Gaussian straddling both modes (mean $\approx 0.55$, sd $\approx 3.06$, $-\mathrm{KL} = -1.31$), and stays there however long you run it.

that straddling fit is not a slow transient: it is a genuine **local maximum** of the ELBO. minimizing $\mathrm{KL}(q\|p)$ over $(m, \log s)$ by quadrature (the KL of a 1-D Gaussian against this mixture can be computed on a fine grid) gives this landscape:

| separation $d$ | broad fit (mean, sd, $-\mathrm{KL}$) | fit on the heavier mode ($-\mathrm{KL}$) | which is better |
|---|---|---|---|
| 3 | 0.26, 1.69, $-0.08$ | (doesn't exist) | broad: the only optimum |
| 4 | 0.33, 2.02, $-0.23$ | (doesn't exist) | broad: the only optimum |
| 5 | 0.40, 2.36, $-0.486$ | 2.43, 1.09, $-0.492$ | broad, barely |
| 6 | 0.47, 2.70, $-0.85$ | 2.99, 1.02, $-0.51$ | the mode |
| 7 | 0.55, 3.06, $-1.31$ | 3.50, 1.00, $-0.51$ | the mode |
| 9 | 0.76, 3.77, $-2.54$ | 4.50, 1.00, $-0.51$ | the mode |

(the lighter mode is a third local optimum, with $-\mathrm{KL} \approx \log 0.4 = -0.92$, once $d \gtrsim 6$.) so reverse KL does **not** always mode-seek: for moderate separations ($d \lesssim 5$) the best single Gaussian *is* a broad one covering a bimodal target, because the valley is shallow enough that the mass $q$ puts there costs little. only once the valley is deep ($d \gtrsim 6$) does sitting on one mode win, and even then the broad solution survives as a local optimum that an optimizer started in the middle (mean 0, sd 1) can fall into.

**why:** the reverse KL is an expectation under $q$: a $q$ that puts mass in the valley pays $\log q - \log p$ there, while a $q$ that ignores the smaller mode pays nothing for it (the expectation never looks there). for a deep valley the first cost dominates, so the best $q$ is a single mode, matched closely, with the weight of the ignored mode showing up only as the constant $\log 0.6$ in the ELBO: that's "mode-seeking". for a shallow valley the cost of covering it is small and the broad fit wins. the forward KL $\mathbb E_p[\log p - \log q]$ is an expectation under $p$: missing a mode is infinitely bad, mass in the valley is cheap, so its optimum always covers everything ("mass-covering"), and for a Gaussian family it simply matches the mean and variance. why do most seeds pick the heavier mode? $q$ starts in the valley with sd 1, and early in training its samples feel the larger mode's pull more; a seed whose early noise widens $q$ quickly can instead settle into the broad local optimum, and the ELBO (here $-1.31$ against $-0.51$) is how you tell which run to keep. for posteriors over parameters, mode-seeking gives one internally consistent explanation but hides the alternatives; for generative models of images, mass-covering (maximum likelihood) avoids missing modes of the data but produces blurry samples in between, the classic VAE-versus-GAN trade-off of ch 21.

## experiment 3: CAVI's many answers

**what people usually predict:** that a coordinate-ascent algorithm with a monotone objective always finds the same, right answer on clean, well-separated data.

**what happens:** (shipped seed, 6 clusters of 60 points, 4 apart) **23 of 30** random starts find all six clusters. the other 7 end at clearly worse fixed points (final ELBOs around $-1{,}275$ and $-1{,}301$, against $-1{,}155$ for the good ones): in the worst one, two components share the leftmost cluster while a single component straddles the two clusters at $+6$ and $+10$. every run's ELBO increased monotonically. the best fit's means are within 0.2 of the truth, each with a posterior sd of about 0.13 ($\approx 1/\sqrt{60}$).

**why:** CAVI changes one factor at a time to its conditional optimum, so it can only climb; but the ELBO of a mixture has many local maxima (and $K!$ copies of each global one, from relabelling). when two components start in the same cluster, moving one of them to the empty cluster far away would require passing through configurations where it explains no points well, which lowers the ELBO; no single coordinate update can make that jump. the remedy is the same as for EM: several random starts (or smarter initializations like k-means++), keeping the run with the highest ELBO, which here picks a correct solution every time. the posterior sds of the means describe uncertainty *within* one local optimum; they say nothing about the uncertainty over which clustering is right, or how many clusters there are, which the mean-field posterior simply can't express.
