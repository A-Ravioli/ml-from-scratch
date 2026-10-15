# kernels & RKHS · lab notes

numbers from the reference solution with the lab's seeds. timings are wall-clock on one shared laptop and vary between runs; memory, feature counts and eigenvalues don't.

## experiment 1: explicit features versus the kernel trick

**what people usually predict:** that the explicit route is a bit slower, maybe 2–10×, and that the two might differ numerically.

**what happens:** the two Gram matrices agree to round-off at every degree (largest difference $6 \times 10^{-13}$ at $p = 4$). the explicit feature matrix grows as $(d + 1)^p$ numbers per point: 11, 121, 1,331, 14,641, so **70 MB** for 600 points at degree 4, against 10 numbers per point for the kernel. continued, degree 8 would need $2 \times 10^8$ numbers per point, about 1 TB for 600 points. the timings are less clean than the memory. at degrees 1–3 both routes take a few milliseconds, and the kernel can even measure slower (at small $D$ the explicit route is one fast BLAS product, while the kernel does an extra elementwise power). at degree 4 the explicit route is roughly 10–30× slower (11× to 31× across runs and machines). the operation counts say why it only gets worse: $\Phi\Phi^\top$ costs $n^2(d+1)^p$ multiply-adds, $5.3 \times 10^9$ at degree 4 against $3.6 \times 10^6$ for $XX^\top$, and the ratio grows by 11 with each extra degree.

**why:** the explicit route costs $O(n(d+1)^p)$ memory and $O(n^2(d+1)^p)$ time for $\Phi\Phi^\top$; the kernel costs $O(n^2 d)$ for $XX^\top$ plus an elementwise power, whatever $p$ is. (the tensor-power map is deliberately naive: it stores every monomial once per ordering. the de-duplicated map with $\binom{d+p}{p}$ multinomially weighted monomials is smaller, 1,001 at $d = 10$, $p = 4$, but still grows polynomially in $d$ and exponentially in $p$.) for the RBF kernel, "explicit features" don't exist at all: its feature space is infinite-dimensional (idea 2), so the kernel is the only way to compute with it. the flip side, which topic 05 is about: kernels cost $O(n^2)$ memory and up to $O(n^3)$ time in the number of examples, which explicit features don't.

## experiment 2: which similarities are kernels?

**what people usually predict:** that anything symmetric, positive and peaked at $x = y$ is a kernel, so only the sigmoid (which can be negative) fails; and that the dimension doesn't matter.

**what happens (150 points):**

| similarity | 1-D | 2-D |
|---|---|---|
| RBF $\exp(-\|x-y\|^2)$ | ✓ | ✓ |
| Laplacian $\exp(-\|x-y\|_1)$ | ✓ | ✓ |
| exponential $\exp(-\|x-y\|)$ | ✓ | ✓ |
| Cauchy $1/(1+\|x-y\|^2)$ | ✓ | ✓ |
| triangle $\max(0, 1-\|x-y\|/2)$ | ✓ | ✗ (smallest/largest eigenvalue $-2.9\times10^{-3}$) |
| $\exp(-\|x-y\|^4)$ | ✗ ($-0.086$) | ✗ ($-0.072$) |
| $\cos\|x-y\|$ | ✓ | ✗ ($-0.51$) |
| sigmoid $\tanh(x^\top y - 1)$ | ✗ ($-1.0$) | ✗ ($-1.0$) |

**why:** PSD is a global property of the function, not a local one about its shape near 0. by Bochner's theorem, a translation-invariant $k(x - y)$ is a kernel exactly when it's the Fourier transform of a nonnegative measure (a "spectral density"). $\exp(-r^2)$ has a Gaussian spectrum and $\exp(-r)$ a Cauchy-like one, both positive; $\exp(-r^4)$ has a Fourier transform that dips below zero, so some combination of points exposes a negative eigenvalue, though it looks like a slightly sharper Gaussian. the triangle function's 1-D Fourier transform is a $\mathrm{sinc}^2$, nonnegative, so it's a kernel on the line; the radial function $\max(0, 1 - \|x\|/2)$ in 2-D has a 2-D Fourier transform that goes negative. $\cos(x - y) = \cos x\cos y + \sin x\sin y$ is a two-feature inner product on the line, but $\cos\|x - y\|$ in the plane isn't of that form at all. the sigmoid is negative for many pairs and, with $K_{ii} = \tanh(\|x_i\|^2 - 1)$ sometimes negative on the diagonal, can't be PSD. finally, passing on a sample proves nothing: with only 60 points, the triangle in 2-D fails on just 9 of 20 random samples (checked separately), and passes on the rest.

## experiment 3: interpolating with RBF bumps

**what people usually predict:** that the smoothest-looking interpolant (small $\gamma$) has the smallest norm, since "small RKHS norm means smooth".

**what happens:** every interpolant goes through all 12 points, but they look nothing alike. with $\gamma = 10$, a row of narrow spikes that drops to 0 between the points. with $\gamma = 0.01$, a smooth curve that shoots past ±4 as soon as it leaves the range of the data. the RKHS norm is *largest* for small $\gamma$ ($2.9 \times 10^5$ at $\gamma = 0.003$, with $\mathrm{cond}(K) = 4 \times 10^{17}$, i.e. numerically singular; the plateau of the computed norm below $\gamma \approx 0.03$ is a truncation artefact of least squares on a singular $K$, and the true norm keeps growing without bound as $\gamma \to 0$) and *smallest* for the spikes (2.63 at $\gamma = 31.6$, just below $\|y\| = 2.71$, with $\mathrm{cond}(K) = 10.8$). at the slider's default, $\gamma = 0.316$: norm 95.8, condition number $2.4 \times 10^7$.

**why:** a small norm means smooth *as measured by that kernel*. each $\gamma$ defines a different RKHS with a different norm, so comparing norms across $\gamma$ compares different rulers. as $\gamma \to \infty$, $K \to I$, $\alpha \to y$ and $\|f\|^2 = y^\top K^{-1}y \to \|y\|^2$: in that RKHS, isolated spikes are cheap. as $\gamma \to 0$, all bumps look alike, $K$ approaches the rank-one matrix $\mathbf 1\mathbf 1^\top$, and hitting 12 noisy targets exactly requires huge, nearly cancelling coefficients: large norm, wild swings where the cancellation fails (outside the data), and a condition number at the limit of float64. none of these curves recovers $\sin x$, because exact interpolation forces the function through the noise. the fix is to stop interpolating and trade data fit against norm, $\min \sum_i (f(x_i) - y_i)^2 + \lambda\|f\|^2$: kernel ridge regression, topic 05.
