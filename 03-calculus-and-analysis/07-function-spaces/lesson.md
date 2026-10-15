# function spaces

## why this matters

a lot of machine learning is about choosing a function: a regression curve, a decision boundary, a value function, a density. once you start comparing functions ("which one fits better?", "how far apart are these two?", "what's the closest simple function to this complicated one?") you're treating functions as **points in a space**, with distances, lengths and angles. that one shift of perspective explains an enormous amount. least squares regression is a projection. kernel methods (chapter 09) work in a space of functions where inner products are cheap. Gaussian processes put a probability distribution on functions. positional encodings in transformers and Fourier features for neural networks represent inputs in a basis of sines and cosines. and the double-descent and spectral-bias phenomena in deep learning (chapter 19) are statements about which basis functions a model learns first.

this topic makes the analogy concrete. you'll define the inner product of two functions, build an orthogonal basis of polynomials (Legendre) and use the classical one of sines and cosines (Fourier), and **project** functions onto them, exactly the way you projected vectors onto subspaces with QR in chapter 02. along the way you'll meet the ways infinite dimensions differ from finite ones: two reasonable notions of "close" can disagree, smooth functions can have slowly converging expansions, and fitting more points exactly can make an approximation worse.

## warm-up: predict first

1. what would it mean for two functions on $[-1, 1]$, like $x$ and $x^2$, to be "orthogonal"? are they?
2. in $\mathbb{R}^n$, projecting a vector onto an orthonormal basis $q_1, \dots, q_k$ gives coefficients $q_i^\top v$. what's the function version?
3. you approximate a square wave by more and more sines. does the maximum error go to zero?
4. which function's Fourier series converges faster on $[-\pi, \pi]$: $|x|$ or $x$?
5. fitting a degree-20 polynomial through 21 equally spaced samples of a smooth function: good idea?

## prerequisites check

- [ ] you know inner products, norms, orthogonality and Cauchy–Schwarz in $\mathbb{R}^n$ (→ ch 02 · topic 04)
- [ ] you can project onto a subspace and know Gram–Schmidt (→ ch 02 · topic 04)
- [ ] you can integrate numerically with the midpoint rule and know its $O(h^2)$ error (→ topic 06)
- [ ] you know what convergence of a sequence means, and that different sequences converge at different rates (→ topic 01)
- [ ] you remember Taylor series and their radius of convergence (→ topic 03)

## core ideas

### idea 1: functions as vectors

functions on an interval $[a, b]$ can be added and scaled, and the results are functions again: they form a **vector space**, just one whose "vectors" have infinitely many coordinates (one value per point). to do geometry you need an inner product. the natural one replaces the sum in $u^\top v = \sum_iu_iv_i$ by an integral:

$$\langle f, g\rangle = \int_a^b f(x)\,g(x)\,dx,\qquad \|f\| = \sqrt{\langle f, f\rangle} = \Big(\int_a^b f(x)^2\,dx\Big)^{1/2}.$$

this is the **$L^2$ inner product**, and $\|f - g\|$ is the root-mean-square distance between $f$ and $g$, scaled by the interval length. everything you know from $\mathbb{R}^n$ survives: linearity, Cauchy–Schwarz $|\langle f, g\rangle| \le \|f\|\|g\|$, the triangle inequality, and **orthogonality**, $\langle f, g\rangle = 0$. warm-up 1: $\langle x, x^2\rangle = \int_{-1}^1x^3\,dx = 0$, so yes: any odd function is orthogonal to any even function on a symmetric interval. on $[-\pi, \pi]$, $\sin$ and $\cos$ are orthogonal, and so are $\sin(jx)$ and $\sin(kx)$ for $j \ne k$.

$L^2$ isn't the only way to measure functions. the **sup norm** $\|f\|_\infty = \max_x|f(x)|$ measures the worst point instead of the average. a vector space with any norm is a **normed space**; one whose norm comes from an inner product is an **inner-product space**. in $\mathbb{R}^n$ all norms are equivalent (each is within a constant factor of the others), so "converges" means the same thing in any of them. for functions it doesn't: a sequence can converge to zero in $L^2$ (the average squared error vanishes) while its sup norm stays large (a narrowing spike that never gets shorter). warm-up 3 and experiment 1 are exactly this.

on a computer, inner products are integrals, so you compute them with the quadrature you built in topic 06: a midpoint rule with a few thousand cells is plenty for smooth functions.

> **▶ build: stage 1.** implement `l2_inner(f, g, a, b, *, n=4096)` (midpoint rule with $n$ cells, `f` and `g` vectorized) and `l2_norm(f, a, b, *, n=4096)`.

### idea 2: an orthogonal basis of polynomials

the monomials $1, x, x^2, x^3, \dots$ span the polynomials, but they're a terrible basis: on $[0, 1]$, $x^{10}$ and $x^{11}$ are nearly parallel (their angle is under 3°), which is why the Vandermonde matrices of high-degree polynomial fits are so ill-conditioned (chapter 02 · topic 08). the fix is the same as for vectors: **Gram–Schmidt**. orthogonalize $1, x, x^2, \dots$ on $[-1, 1]$ in the $L^2$ inner product and scale each result so it equals 1 at $x = 1$. you get the **Legendre polynomials**:

$$P_0 = 1,\quad P_1 = x,\quad P_2 = \tfrac12(3x^2 - 1),\quad P_3 = \tfrac12(5x^3 - 3x),\ \dots$$

with $\langle P_j, P_k\rangle = 0$ for $j \ne k$ and $\|P_k\|^2 = 2/(2k + 1)$. you don't need to run Gram–Schmidt every time: orthogonal polynomials always satisfy a **three-term recurrence**, because $xP_j$ is a polynomial of degree $j + 1$ that's automatically orthogonal to everything of degree below $j - 1$ (on paper 1 shows the first step). for Legendre,

$$(j + 1)P_{j+1}(x) = (2j + 1)\,x\,P_j(x) - j\,P_{j-1}(x).$$

it's fast ($O(k)$ operations per point), and numerically stable on $[-1, 1]$, where every $|P_k| \le 1$. other weights and intervals give other families (Chebyshev, Hermite, Laguerre), the same idea with a different inner product.

> **▶ build: stage 2.** implement `legendre(k, x)` with the recurrence, vectorized over `x`.

### idea 3: projection and best approximation

in $\mathbb{R}^n$, the closest point to $v$ in the span of orthogonal vectors $q_0, \dots, q_K$ is $\sum_k\frac{\langle v, q_k\rangle}{\langle q_k, q_k\rangle}q_k$, and the residual is orthogonal to the span. the same argument, word for word, works for functions (warm-up 2). the best $L^2$ approximation of $f$ on $[-1, 1]$ by a polynomial of degree $K$ is

$$f_K = \sum_{k=0}^Kc_kP_k,\qquad c_k = \frac{\langle f, P_k\rangle}{\langle P_k, P_k\rangle} = \frac{2k + 1}{2}\int_{-1}^1f(x)P_k(x)\,dx.$$

each coefficient is computed independently of the others (that's what orthogonality buys you), and adding a term never changes the earlier ones. by Pythagoras, $\|f\|^2 = \|f_K\|^2 + \|f - f_K\|^2$, which gives **Bessel's inequality**, $\sum_kc_k^2\|P_k\|^2 \le \|f\|^2$, and shows that the error can only shrink as $K$ grows.

**does it shrink to zero?** for the Legendre polynomials on $[-1, 1]$ (and the Fourier basis on $[-\pi, \pi]$), yes, for every $f$ with $\|f\| < \infty$: the basis is **complete**, and Bessel's inequality becomes **Parseval's identity**, an equality. making this precise needs one more ingredient. $L^2$ is a **Hilbert space**: an inner-product space that's complete in the sense of topic 01, every Cauchy sequence converges. (the continuous functions alone aren't complete in this norm: smooth functions can converge in $L^2$ to a function with a jump, so the space has to include such limits.) completeness is what lets you talk about the limit $\sum_{k=0}^\infty c_kP_k$ at all. this is the same structure, infinite-dimensional linear algebra, that underlies reproducing kernel Hilbert spaces in chapter 09.

> **🔮 predict:** the Legendre coefficients of $f(x) = x^3$: how many are nonzero, and what are they?

> **▶ build: stage 3.** implement `legendre_coefficients(f, K, *, n=4096)` (using your `l2_inner` and `legendre`) and `legendre_series(c, x)`.

### idea 4: the Fourier basis

on $[-\pi, \pi]$, the functions $1, \cos x, \sin x, \cos 2x, \sin 2x, \dots$ are mutually orthogonal, with $\|\cos kx\|^2 = \|\sin kx\|^2 = \pi$ for $k \ge 1$ and $\|1\|^2 = 2\pi$. projecting gives the **Fourier series**

$$f(x) \approx \frac{a_0}{2} + \sum_{k=1}^K\big(a_k\cos kx + b_k\sin kx\big),\qquad a_k = \frac1\pi\int_{-\pi}^{\pi}f(x)\cos kx\,dx,\quad b_k = \frac1\pi\int_{-\pi}^{\pi}f(x)\sin kx\,dx.$$

(the $a_0/2$ convention lets one formula cover all the $a_k$: $a_0/2$ is the mean of $f$.) a Fourier coefficient measures how much of frequency $k$ is in $f$. for periodic functions the midpoint rule on a uniform grid is extraordinarily accurate (it's exact for trigonometric polynomials of degree below $n$), so a few thousand points give coefficients to machine precision for smooth $f$.

**smoothness controls decay.** integrate $a_k$ by parts: every time the periodic extension of $f$ has one more continuous derivative, the coefficients pick up another factor of $1/k$. a jump gives coefficients of size $1/k$ (the square wave: $b_k = 4/(\pi k)$ for odd $k$); a kink, like the triangle wave $|x|$, gives $1/k^2$; an infinitely smooth periodic function gives faster than any power, typically $e^{-ck}$. the catch is the word *periodic*: the Fourier series sees $f$ on $[-\pi, \pi]$ as one period of a repeating function. the straight line $f(x) = x$ is perfectly smooth, but its periodic extension is a sawtooth that jumps at $\pm\pi$, so its coefficients decay only like $1/k$ (warm-up 4: $|x|$ converges faster). Legendre polynomials don't assume periodicity, so they have no such problem.

**the Gibbs phenomenon.** near a jump, Fourier partial sums overshoot. the overshoot doesn't shrink as $K$ grows; it gets narrower, squeezed against the jump, but its height converges to about 9% of the jump. so the $L^2$ error goes to zero while the maximum error doesn't (idea 1's two norms disagreeing, in the most famous example there is).

> **🔮 predict:** for the square wave $\operatorname{sign}(x)$, the partial sums overshoot $+1$ just to the right of 0. as $K$ goes from 25 to 200, does the peak move closer to 1?

> **▶ build: stage 4.** implement `fourier_coefficients(f, K, *, n=4096)`, returning `(a, b)` with $K + 1$ entries each ($b_0 = 0$), and `fourier_series(a, b, x)`.

## worked example

**a polynomial in the Legendre basis.** project $f(x) = x^3$ onto $P_0, \dots, P_5$. $x^3$ is odd, so $c_0 = c_2 = c_4 = 0$. then

$$c_1 = \tfrac32\int_{-1}^1x^3\cdot x\,dx = \tfrac32\cdot\tfrac25 = \tfrac35,\qquad c_3 = \tfrac72\int_{-1}^1x^3\cdot\tfrac{5x^3 - 3x}{2}\,dx = \tfrac74\Big(\tfrac{10}{7} - \tfrac65\Big) = \tfrac25,$$

and $c_5 = 0$ because $x^3$ is a combination of $P_1$ and $P_3$ only. check: $\tfrac35x + \tfrac25\cdot\tfrac{5x^3 - 3x}{2} = \tfrac35x + x^3 - \tfrac35x = x^3$. ✓ a degree-$K$ polynomial is always exactly its own degree-$K$ projection.

**a jump in the Fourier basis.** for $\operatorname{sign}(x)$ on $[-\pi, \pi]$: it's odd, so every $a_k = 0$, and

$$b_k = \frac1\pi\int_{-\pi}^{\pi}\operatorname{sign}(x)\sin kx\,dx = \frac2\pi\int_0^\pi\sin kx\,dx = \frac{2(1 - \cos k\pi)}{\pi k} = \begin{cases}4/(\pi k) & k\text{ odd}\\ 0 & k\text{ even}.\end{cases}$$

so $\operatorname{sign}(x) \approx \frac4\pi\big(\sin x + \tfrac13\sin 3x + \tfrac15\sin 5x + \dots\big)$: coefficients decaying like $1/k$, as a jump demands. your code must reproduce both sets of coefficients (to the accuracy of the midpoint rule), and the Gibbs peak of about 1.179 for $K = 101$.

## on paper

1. **Gram–Schmidt for $P_2$.** orthogonalize $x^2$ against $P_0 = 1$ and $P_1 = x$ on $[-1, 1]$, then scale so the result is 1 at $x = 1$. check the recurrence with $j = 1$.

<details><summary>solution</summary>

$\langle x^2, 1\rangle = 2/3$ and $\|1\|^2 = 2$, so subtract $\tfrac13$; $\langle x^2, x\rangle = 0$. that leaves $x^2 - \tfrac13$, which is $\tfrac23$ at $x = 1$, so $P_2 = \tfrac32(x^2 - \tfrac13) = \tfrac12(3x^2 - 1)$. the recurrence with $j = 1$: $2P_2 = 3x\cdot x - 1\cdot1$, so $P_2 = \tfrac12(3x^2 - 1)$. ✓

</details>

2. **the error of a projection.** show that $f - f_K$ is orthogonal to every $P_j$ with $j \le K$, and deduce $\|f - f_K\|^2 = \|f\|^2 - \sum_{k\le K}c_k^2\|P_k\|^2$. why does this make $f_K$ the best degree-$K$ approximation?

<details><summary>solution</summary>

$\langle f - f_K, P_j\rangle = \langle f, P_j\rangle - c_j\|P_j\|^2 = 0$ by the choice of $c_j$ (the other terms vanish by orthogonality). so $f = f_K + (f - f_K)$ with the two parts orthogonal, and Pythagoras gives $\|f\|^2 = \|f_K\|^2 + \|f - f_K\|^2$, with $\|f_K\|^2 = \sum c_k^2\|P_k\|^2$. for any other degree-$K$ polynomial $q$, $f - q = (f - f_K) + (f_K - q)$ with $f_K - q$ in the span, so $\|f - q\|^2 = \|f - f_K\|^2 + \|f_K - q\|^2 \ge \|f - f_K\|^2$.

</details>

3. **Parseval for the square wave.** use the coefficients above and Parseval's identity, $\frac1\pi\int_{-\pi}^\pi f^2 = \frac{a_0^2}{2} + \sum_{k\ge1}(a_k^2 + b_k^2)$, to compute $\sum_{m\ge0}\frac{1}{(2m + 1)^2}$.

<details><summary>solution</summary>

the left side is $\frac1\pi\cdot2\pi = 2$. the right side is $\sum_{k\text{ odd}}\frac{16}{\pi^2k^2}$. so $\sum_{k\text{ odd}}\frac1{k^2} = \frac{2\pi^2}{16} = \frac{\pi^2}{8}$. projections onto a complete orthogonal basis preserve length, the infinite-dimensional Pythagoras.

</details>

4. **why kinks decay faster than jumps.** for a function $f$ whose periodic extension is continuous with a piecewise-continuous derivative, integrate $b_k = \frac1\pi\int f\sin kx$ by parts to show $b_k = \frac1k a_k[f']$, where $a_k[f']$ is a cosine coefficient of $f'$. conclude that the triangle wave's coefficients are $O(1/k^2)$.

<details><summary>solution</summary>

$\int_{-\pi}^\pi f\sin kx\,dx = \big[-f\cos kx/k\big]_{-\pi}^\pi + \frac1k\int f'\cos kx\,dx$. the boundary term vanishes when the periodic extension is continuous ($f(\pi) = f(-\pi)$). so $b_k = \frac1k a_k[f']$ (and similarly $a_k = -\frac1kb_k[f']$). for $|x|$, $f' = \operatorname{sign}(x)$, whose coefficients are $O(1/k)$, so $|x|$'s are $O(1/k^2)$. for $f(x) = x$, the boundary term is $-2\pi\cos(k\pi)/k \ne 0$: the jump of the periodic extension leaves the coefficients at $O(1/k)$.

</details>

## experiment

`lab.py` investigates three things with your code:

1. the Fourier partial sums of a square wave near its jump, as the number of frequencies grows.
2. how fast the Fourier coefficients of four functions decay, including a perfectly smooth straight line, and the Legendre coefficients of the same line.
3. Legendre projection against polynomial interpolation at equally spaced points, for Runge's function $1/(1 + 25x^2)$.

## check your understanding

1. on $[0, 1]$, are $1$ and $x$ orthogonal? what about on $[-1, 1]$?

<details><summary>answer</summary>

on $[0, 1]$, $\langle 1, x\rangle = \tfrac12 \ne 0$: not orthogonal. on $[-1, 1]$ it's 0. orthogonality depends on the interval (and the weight), which is why each orthogonal-polynomial family comes with its own interval.

</details>

2. why is the monomial basis a bad choice for high-degree polynomial fits, and what does the Legendre basis fix?

<details><summary>answer</summary>

high powers of $x$ are nearly parallel functions, so the least-squares matrix in that basis is badly conditioned and its coefficients are huge and cancel. an orthogonal basis makes the "normal equations" diagonal: each coefficient is a separate, well-conditioned inner product. the approximating polynomial is the same; only the coordinates change.

</details>

3. (the predict callout in idea 3.) how many Legendre coefficients of $x^3$ are nonzero?

<details><summary>answer</summary>

two: $c_1 = 3/5$ and $c_3 = 2/5$ (the worked example). odd functions have only odd-degree coefficients, and a cubic needs nothing beyond degree 3.

</details>

4. (the predict callout in idea 4.) does the Gibbs peak move closer to 1 as $K$ grows?

<details><summary>answer</summary>

no: it stays at about 1.179, roughly 9% of the jump of 2 above the top (experiment 1). it moves *closer to the jump* and gets narrower, so its contribution to the $L^2$ error vanishes, but the maximum error doesn't.

</details>

5. a signal is compressed by keeping only its 50 lowest-frequency Fourier coefficients. where will the reconstruction look worst, and why?

<details><summary>answer</summary>

next to sharp edges and discontinuities, where it will ring (Gibbs), and at the ends of the window if the signal isn't periodic (the implicit jump of the periodic extension). smooth regions are captured well because their energy is in low frequencies.

</details>

6. why is completeness of $L^2$ (a Hilbert space) needed to make sense of an infinite Fourier series?

<details><summary>answer</summary>

the partial sums $f_K$ form a Cauchy sequence in the $L^2$ norm (by Bessel's inequality, the tail sums of $c_k^2\|\phi_k\|^2$ go to zero). completeness guarantees that a Cauchy sequence has a limit inside the space. without it, the series might "converge" to something that isn't a function in your space, as with the rationals and $\sqrt2$ in topic 01.

</details>

7. (warm-up 5.) why might fitting a degree-20 polynomial through 21 equispaced samples be a bad idea, even with exact data?

<details><summary>answer</summary>

the interpolant must pass through every point, and for many smooth functions (Runge's function) that forces wild oscillations between the points near the ends of the interval, with errors that *grow* with the degree (experiment 3). projection, which minimizes the average error, or interpolation at clustered (Chebyshev) points, avoids this. it's an early example of how fitting training points exactly can generalize badly.

</details>

## common bugs

- **forgetting the cell width in the inner product.** symptom: inner products scale with $n$. the midpoint rule is $\sum f(x_i)g(x_i)\,h$.
- **an off-by-one in the recurrence.** symptom: `legendre(k, x)` returns $P_{k+1}$ or $P_{k-1}$. check $P_2$ and $P_3$ against their formulas.
- **coefficients without the normalization.** symptom: Legendre coefficients too small by $2/(2k + 1)$. divide $\langle f, P_k\rangle$ by $\|P_k\|^2$.
- **the $a_0$ convention.** symptom: the constant term is half or twice what it should be. with $f \approx a_0/2 + \dots$, $a_0 = \frac1\pi\int f$ is twice the mean.
- **$1/(2\pi)$ instead of $1/\pi$.** symptom: every Fourier coefficient is half the right size, and the series reconstructs $f/2$.
- **expecting pointwise convergence at a jump.** symptom: "my Fourier code is wrong, it overshoots by 9%." that's the Gibbs phenomenon, not a bug.

## going deeper

- Kreyszig (1989), *Introductory Functional Analysis with Applications*, Wiley, chapters 2–3. normed spaces, inner-product spaces, Hilbert spaces and orthonormal bases, at an accessible level.
- Trefethen (2019), *Approximation Theory and Approximation Practice*, extended ed., SIAM, chapters 3, 7–9, 13 and 17 (Chebyshev series, convergence for differentiable and analytic functions, the Gibbs phenomenon, equispaced points and the Runge phenomenon, orthogonal polynomials). [doi:10.1137/1.9781611975949](https://doi.org/10.1137/1.9781611975949)
- Stein & Shakarchi (2003), *Fourier Analysis: An Introduction*, Princeton University Press, chapters 2–3. convergence of Fourier series and Parseval's identity.
- Rahimi & Recht (2007), "Random features for large-scale kernel machines", *Advances in Neural Information Processing Systems 20*. random Fourier features: kernels as expectations over a Fourier basis (chapter 09). [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
- Tancik et al. (2020), "Fourier features let networks learn high frequency functions in low dimensional domains", *NeurIPS*. a modern use of the Fourier basis in deep learning. [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)

## connections

- ← builds on: inner products, projections and Gram–Schmidt (ch 02 · topic 04), conditioning of Vandermonde matrices (ch 02 · topic 08), quadrature (topic 06), completeness and convergence (topic 01), radius of convergence (topic 03).
- → used in: least squares and polynomial features (ch 08), reproducing kernel Hilbert spaces, random Fourier features and Gaussian processes (ch 09), PCA as projection (ch 11), positional encodings (ch 18), spectral bias and kernel views of neural networks (ch 19), spectral graph convolutions (ch 20).
