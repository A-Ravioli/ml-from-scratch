# function spaces · lab notes

## experiment 1: the Gibbs phenomenon

**what people usually predict:** the overshoot is a low-$K$ artifact; with enough terms it disappears, and the approximation converges everywhere.

**what happens:** the peak of the partial sum near the jump is **1.186** at $K = 5$, **1.1795** at $K = 25$, and **1.1790** at both $K = 101$ and $K = 201$: it converges to about 1.179, not to 1. meanwhile the $L^2$ error falls steadily, **0.65, 0.31, 0.16, 0.11**, roughly like $1/\sqrt K$. on the zoomed plot, the overshoot gets narrower and moves towards the jump as $K$ grows, but it never gets shorter.

**why:** the partial sum is the convolution of $f$ with the Dirichlet kernel, whose side lobes have a fixed shape that just gets compressed as $K$ grows. at a jump, that fixed shape produces a fixed overshoot: about $0.0895$ times the jump of 2, i.e. $\frac2\pi\int_0^\pi\frac{\sin t}{t}dt \approx 1.1790$. its width is about $\pi/K$, so its contribution to the squared error is about (overshoot)² × width, which goes to zero: $L^2$ convergence. the sup-norm error stays around 0.18 (and right *at* the jump, the series converges to the midpoint 0, so the error there is 1). this is the clearest example of idea 1's point: in function spaces, "converges" depends on the norm. it also explains ringing artifacts next to sharp edges in compressed images and audio, and why spectral methods for problems with shocks need filtering or different bases.

## experiment 2: smoothness and the decay of coefficients

**what people usually predict:** the smooth functions (the line, $e^{\cos x}$) decay fast and the rough ones slowly, so the straight line should rank with $e^{\cos x}$, or at least beat the triangle.

**what happens:** fitted log-log slopes for $5 \le k \le 40$: square wave **$-1.00$**, triangle wave **$-2.00$**, straight line **$-1.00$**, and $e^{\cos x}$ falls off a cliff (below $10^{-12}$ by $k = 12$; its "slope" of about $-23$ really reflects exponential decay). the straight line's coefficients are the slowest of all in absolute size: $|b_k| = 2/k$. its Legendre coefficients, by contrast, are **$[0, 1, 0, 0, \dots]$**: $x = P_1$ exactly.

**why:** a Fourier series represents the periodic extension of $f$. for the line, that extension is a sawtooth with a jump of $2\pi$ at $x = \pm\pi$, so by on paper 4 the coefficients decay like $1/k$, exactly as for the square wave's jumps. the triangle wave's periodic extension is continuous with a kink, giving $1/k^2$, and $e^{\cos x}$ is smooth and periodic, giving faster than any power (its coefficients are $2I_k(1)$, modified Bessel functions, which decay like $1/(2^kk!)$). smoothness *of the function on the interval* isn't what matters; smoothness *of the object the basis sees* is. Legendre polynomials don't assume periodicity, so the line costs them a single coefficient, and smooth functions on an interval get fast Legendre (or Chebyshev) convergence. the practical rules: use Fourier bases for genuinely periodic things (angles, time of day, positions on a circle), polynomial bases or windowing for things on an interval; and if you see $1/k$ decay from a function that looks smooth, suspect a boundary mismatch.

## experiment 3: projection versus interpolation

**what people usually predict:** both improve with $K$; interpolation, which hits the function *exactly* at more and more points, might even be better.

**what happens:** the maximum error of the Legendre projection falls steadily: **0.33, 0.15, 0.068, 0.032, 0.016, 0.0078** for $K = 4, 8, \dots, 24$, roughly halving every 4 degrees. interpolation at equispaced points gets *worse*: **0.44, 1.05, 3.7, 14, 60, 257**. the plot shows the interpolant matching the function in the middle and oscillating wildly near $\pm1$.

**why:** this is the Runge phenomenon. Runge's function $1/(1 + 25x^2)$ has complex poles at $\pm i/5$, very close to the real interval, which limits how well polynomials can approximate it (the same reason as topic 03's experiment 1). the projection only promises the best *average* (L²) error, and it delivers geometric convergence at the rate the poles allow. equispaced interpolation demands exact agreement at points that are evenly spread, and the polynomial that achieves that has to swing wildly between the outer points: the error amplification (the Lebesgue constant of equispaced points) grows like $2^K$, overwhelming any convergence. interpolating at Chebyshev points (clustered towards the ends) fixes it; so does least squares with many more points than coefficients. the ML lesson: fitting training points exactly with a high-capacity model can make it worse between the points; constraining *how* the fit uses its capacity (here, minimizing the average error over the whole interval) matters more than the capacity itself. you'll meet versions of this in overfitting (chapter 07), regularization (chapter 08), and, with a twist, double descent (chapter 19).
