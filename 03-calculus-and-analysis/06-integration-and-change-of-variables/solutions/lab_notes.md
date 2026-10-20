# integration & change of variables · lab notes

## experiment 1: does a transformed Gaussian still integrate to 1?

**what people usually predict:** without the Jacobian factor the integral is wrong for every map, probably by different amounts; with it, 1.

**what happens:** with the Jacobian term all three maps give **1.0000** (0.9998 for the radial map, whose density has a sharp peak at the origin that a 300 × 300 grid resolves only to about $10^{-4}$). without it: **2.0000** for the linear "scale" map, **1.0000** for the banana, and about **8.76** for the radial map.

**why:** leaving out the Jacobian means integrating $p_Z(T^{-1}(x))$; substituting $x = T(z)$ turns that into $\int p_Z(z)\,|\det J_T(z)|\,dz = \mathbb E[|\det J_T(Z)|]$, the average volume stretch under $p_Z$. for the scale map, $\det J_T = 2$ everywhere, so the wrong integral is exactly 2. the banana map is a shear, $\det J_T = 1$ everywhere (on paper 3), so the density simply moves: forgetting the correction does no harm, and the samples bend into a banana while the total mass stays put. the radial map $z(1 + \|z\|)$ has $\det J_T = (1 + r)(1 + 2r)$ in 2-D, whose average under a standard Gaussian is about 8.8. this is why flow architectures are built from pieces with known, cheap determinants: shears and coupling layers (determinant 1, or a product of diagonal scales), autoregressive maps (triangular Jacobians). the "integrates to 1" check is also a superb unit test for any density code: it catches a missing or wrong-signed Jacobian immediately.

## experiment 2: grids versus random points as the dimension grows

**what people usually predict:** Monte Carlo is a "rough" method, so grids win until some large dimension, maybe 10 or 20. or the opposite: since Monte Carlo is dimension-free, it should win everywhere.

**what happens:** with about 65,536 evaluations, the midpoint rule's relative error is about $10^{-11}$ in 1-D, $10^{-6}$ in 2-D, $8 \times 10^{-5}$ in 3-D, $6.5 \times 10^{-4}$ in 4-D, then $2.6 \times 10^{-3}$, $7 \times 10^{-3}$, $2 \times 10^{-2}$ by 7–8-D. Monte Carlo's error stays around $10^{-3}$ throughout (its standard error grows slowly, from $1.1 \times 10^{-3}$ to $3.7 \times 10^{-3}$, because $e^{x_1 + \dots + x_d}$ gets more variable in more dimensions). the curves cross around **$d = 4$–5**; by $d = 8$ the grid, with only 4 cells per axis, is ten times worse. the Monte Carlo errors scatter around their standard errors, sometimes above and sometimes below, as an honest error bar should.

**why:** a grid with $N$ points has $N^{1/d}$ cells per axis, so its $O(h^2)$ error becomes $O(N^{-2/d})$; Monte Carlo's is $\sigma N^{-1/2}$ in any dimension. the exponents match at $d = 4$, and the constants decide the exact crossover. two further lessons. first, the grid gives no error estimate at all, while Monte Carlo reports its own uncertainty for free. second, the grid also suffers from granularity: in 7-D, $5^7 = 78{,}125$ already exceeds the budget, so it can only use $4^7 = 16{,}384$ points. this is why posterior expectations in Bayesian ML, partition functions, and expectations over data are computed by sampling (chapter 12), and why so much effort goes into reducing $\sigma$ (importance sampling, control variates, MCMC).

## experiment 3: where is the peak of a transformed density?

**what people usually predict:** the peak of $X = e^Z$ is at $e^0 = 1$: the most likely $Z$ maps to the most likely $X$.

**what happens:** with $\sigma = 1$ the peak of your `pushforward_log_density` sits at about **0.37**, matching $e^{-\sigma^2} = 0.368$, and the histogram agrees. the median of the samples is about **0.99** ($e^0 = 1$, up to sampling noise). as $\sigma$ grows the mode slides towards 0 ($e^{-2.25} \approx 0.11$ at $\sigma = 1.5$); as $\sigma \to 0$ it approaches 1. the density integrates to 1.0000.

**why:** $\log p_X(x) = \log p_Z(\log x) - \log x$, and the Jacobian term $-\log x$ is largest for small $x$: the exponential map squeezes the negative half-line of $Z$ into the interval $(0, 1)$, piling density up there, and stretches the positive half-line over $(1, \infty)$, thinning it out. maximizing gives $\log x = -\sigma^2$ (on paper 2). quantiles transform cleanly under a monotone map ($P(X \le e^q) = P(Z \le q)$), so the median moves to $e^0 = 1$, but the mode is a property of the density, which depends on the units of measurement. the practical consequence: maximum a posteriori estimates change when you reparameterize (variance versus log-variance, probability versus logit), while posterior probabilities of events don't. the same Jacobian bookkeeping is needed whenever you sample in an unconstrained space and want densities in a constrained one, e.g. the log-transform for positive parameters in HMC and variational inference (chapter 12).
