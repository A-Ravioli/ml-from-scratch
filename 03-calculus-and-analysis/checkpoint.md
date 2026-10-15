# 03 · calculus & analysis · checkpoint

use this to finish the chapter, or to test out of it before starting. sections 1 and 2 take about an hour with pen and paper. section 3 takes 60 minutes at a computer.

## 1. concept check

1. state the ε–N definition of $a_n \to L$. why is "the first index where $|a_n - L| < \varepsilon$" not the $N$ the definition asks for?

<details><summary>answer</summary>

for every $\varepsilon > 0$ there is an $N$ such that $|a_n - L| < \varepsilon$ for **all** $n \ge N$. a sequence can enter the band and leave it again; $N$ must come after the last time it's outside. (topic 01, idea 1)

</details>

2. the consecutive differences $a_{n+1} - a_n$ of a sequence go to zero. does it converge? what's the right "no limit needed" test?

<details><summary>answer</summary>

not necessarily: the harmonic partial sums have gaps $1/(n+1) \to 0$ but diverge like $\log n$. the right test is the Cauchy condition, $|a_m - a_n| < \varepsilon$ for *all* $m, n \ge N$; in $\mathbb{R}^n$ (complete) it's equivalent to convergence. (topic 01, idea 2)

</details>

3. errors $10^{-1}, 10^{-2}, 10^{-4}, 10^{-8}$: what order and rate? how do you estimate the order from three consecutive errors without knowing the rate?

<details><summary>answer</summary>

quadratic ($q = 2$, $C = 1$). $q_k = \log(e_{k+2}/e_{k+1})/\log(e_{k+1}/e_k)$, which cancels $C$. (topic 01, idea 3)

</details>

4. why is Newton's method only linear at a double root, and with what rate?

<details><summary>answer</summary>

near a root of multiplicity $m$, $f/f' \approx (x - r)/m$, so each step removes only a fraction $1/m$ of the error: linear with rate $(m - 1)/m$, i.e. $\tfrac12$ for a double root. quadratic convergence needs $f'(x^*) \ne 0$. (topic 01, on paper 2)

</details>

5. what is the directional derivative of $f$ at $x$ along $v$, and which unit direction maximizes it?

<details><summary>answer</summary>

$D_vf = \nabla f(x)\cdot v$; by Cauchy–Schwarz it's largest for $v = \nabla f/\|\nabla f\|$, with value $\|\nabla f\|$. the gradient is perpendicular to the level sets. (topic 02, idea 1)

</details>

6. a map $F: \mathbb{R}^4 \to \mathbb{R}^2$ is composed with $G: \mathbb{R}^2 \to \mathbb{R}^3$. what are the shapes of $J_F$, $J_G$ and $J_{G\circ F}$, and how are they related?

<details><summary>answer</summary>

$2 \times 4$, $3 \times 2$, and $J_{G\circ F} = J_G(F(x))\,J_F(x)$, $3 \times 4$. the chain rule is matrix multiplication. (topic 02, idea 2)

</details>

7. why does a numerical Hessian by second differences use $h \approx 10^{-4}$ rather than the gradient's $10^{-6}$?

<details><summary>answer</summary>

rounding error in a second difference is about $\varepsilon|f|/h^2$ (you divide by $h^2$), truncation is $O(h^2)$; they balance at $h \approx \varepsilon^{1/4} \approx 10^{-4}$, with error near $10^{-8}$. at $h = 10^{-6}$ rounding gives errors near $10^{-4}$. (topic 02, idea 3)

</details>

8. at a critical point the Hessian has eigenvalues $4, 0, -1$. what kind of point is it? and with eigenvalues $4, 0, 1$?

<details><summary>answer</summary>

the first is a saddle (mixed signs decide it, whatever the zero does). the second is degenerate: the second-derivative test can't decide, and higher-order terms do. (topic 02, idea 4)

</details>

9. what does Taylor's theorem say about the error of a degree-$K$ Taylor polynomial, and why can a smooth function's Taylor series still diverge?

<details><summary>answer</summary>

$|f(x) - T_K(x)| \le M|x - x_0|^{K+1}/(K + 1)!$ with $M$ bounding $|f^{(K+1)}|$ between $x_0$ and $x$. if $M$ grows like $K!$, the bound only shrinks for $|x - x_0| < $ some radius; that radius is the distance to the nearest complex singularity ($\pm i$ for $1/(1 + x^2)$). (topic 03, idea 2)

</details>

10. what is the agreement ratio $\rho$ of a step, and what do $\rho \approx 1$, $\rho = 0.3$ and $\rho < 0$ tell you?

<details><summary>answer</summary>

actual reduction / predicted reduction. $\approx 1$: the model was right, trust it further; $0.3$: some progress but the step was too long for the model; $< 0$: $f$ went up, reject and shrink. (topic 03, idea 5)

</details>

11. why does evaluating $f$ on the dual number $x + \varepsilon$ (with $\varepsilon^2 = 0$) produce $f'(x)$ in the dual part? how do you get a second derivative the same way?

<details><summary>answer</summary>

Taylor: $f(x + \varepsilon) = f(x) + f'(x)\varepsilon + \tfrac12f''\varepsilon^2 + \dots$ and every term past the second vanishes; each elementary operation applies the chain rule exactly. nesting, $x + \varepsilon_1 + \varepsilon_2$ with $\varepsilon_1\varepsilon_2 \ne 0$, puts $f''(x)$ in the $\varepsilon_1\varepsilon_2$ coefficient. (topic 04, ideas 1–2)

</details>

12. how many forward passes does forward mode need for the gradient of $f: \mathbb{R}^n \to \mathbb{R}$? for the Jacobian of $F: \mathbb{R} \to \mathbb{R}^m$? for one Jacobian-vector product?

<details><summary>answer</summary>

$n$; one; one. forward mode costs one pass per input direction, which is why scalar losses with many parameters need reverse mode (chapter 14). (topic 04, idea 4)

</details>

13. what does $L$-smoothness guarantee, and why does it imply that gradient descent with step $\eta < 2/L$ decreases $f$?

<details><summary>answer</summary>

$\|\nabla f(x) - \nabla f(y)\| \le L\|x - y\|$, which gives the descent lemma $f(y) \le f(x) + \nabla f^\top(y - x) + \tfrac L2\|y - x\|^2$. plugging in $y = x - \eta\nabla f$ gives $f(y) \le f(x) - \eta(1 - L\eta/2)\|\nabla f\|^2$, a decrease for $0 < \eta < 2/L$. (topic 05, ideas 3–4)

</details>

14. if $X = T(Z)$ for an invertible $T$, what is $p_X$? why do volume-preserving maps not need a correction?

<details><summary>answer</summary>

$p_X(x) = p_Z(T^{-1}(x))\,|\det J_{T^{-1}}(x)|$. density is probability per unit volume; if $|\det J_T| = 1$ everywhere, volumes don't change and the density just moves. (topic 06, idea 4)

</details>

15. why do the Fourier coefficients of the smooth function $f(x) = x$ on $[-\pi, \pi]$ decay only like $1/k$?

<details><summary>answer</summary>

the Fourier series represents the periodic extension, which for $f(x) = x$ is a sawtooth that jumps at $\pm\pi$; a jump gives $1/k$ coefficients (and Gibbs overshoot). Legendre polynomials, which don't assume periodicity, represent $x$ with one coefficient. (topic 07, idea 4)

</details>

## 2. by hand (no computer)

1. **Newton and its order.** (a) take two Newton steps for $f(x) = x^2 - 3$ from $x_0 = 2$. (b) the errors of some method are $0.2, 0.04, 0.0016$; estimate its order. (c) how many bisection midpoints are needed to bracket a root to half-width $10^{-6}$ starting from $[0, 1]$?

<details><summary>solution</summary>

(a) $x_{k+1} = \tfrac12(x_k + 3/x_k)$: $x_1 = \tfrac12(2 + 1.5) = 1.75$, $x_2 = \tfrac12(1.75 + 3/1.75) = \tfrac12(1.75 + 1.714286) = 1.732143$ (true $\sqrt3 = 1.732051$, error $9 \times 10^{-5}$).

(b) $q = \log(0.0016/0.04)/\log(0.04/0.2) = \log(0.04)/\log(0.2) = (-3.219)/(-1.609) = 2$.

(c) the half-width after $j$ midpoints is $1/2^j$; $2^j \ge 10^6$ needs $j = 20$.

</details>

2. **a Taylor model and a Hessian.** for $f(x, y) = x^2y + e^{y}$ at $(1, 0)$: (a) compute $\nabla f$ and $H$; (b) write the quadratic model $m(1 + d_1, d_2)$; (c) is $(1, 0)$ a critical point? find the Newton step $-H^{-1}\nabla f$ and say whether it's a descent direction.

<details><summary>solution</summary>

(a) $\nabla f = (2xy,\ x^2 + e^y) = (0, 2)$. $H = \begin{pmatrix}2y & 2x\\ 2x & e^y\end{pmatrix} = \begin{pmatrix}0 & 2\\ 2 & 1\end{pmatrix}$.

(b) $f(1, 0) = 1$, so $m = 1 + 2d_2 + \tfrac12(4d_1d_2 + d_2^2) = 1 + 2d_2 + 2d_1d_2 + \tfrac12d_2^2$.

(c) no: $\nabla f \ne 0$. $H^{-1} = \frac{1}{-4}\begin{pmatrix}1 & -2\\ -2 & 0\end{pmatrix}$, so $-H^{-1}\nabla f = \tfrac14\begin{pmatrix}1 & -2\\ -2 & 0\end{pmatrix}\begin{pmatrix}0\\ 2\end{pmatrix} = (-1, 0)$. its slope is $\nabla f\cdot(-1, 0) = 0$: not a descent direction. $H$ is indefinite ($\det H = -4$), so the Newton step heads for the model's saddle; a damped Newton method would modify $H$ first.

</details>

3. **dual numbers and a density.** (a) evaluate $f(x) = x\sin x + 1/x$ on $2 + \varepsilon$ symbolically and read off $f'(2)$. (b) $Z \sim \mathcal N(0, 1)$ and $X = 3Z + 1$. write $p_X(x)$ using the change-of-variables formula and check it's the $\mathcal N(1, 9)$ density.

<details><summary>solution</summary>

(a) $\sin(2 + \varepsilon) = \sin 2 + \varepsilon\cos 2$, so $(2 + \varepsilon)\sin(2 + \varepsilon) = 2\sin 2 + \varepsilon(\sin 2 + 2\cos 2)$, and $1/(2 + \varepsilon) = \tfrac12 - \tfrac14\varepsilon$. so $f'(2) = \sin 2 + 2\cos 2 - \tfrac14 \approx 0.9093 - 0.8323 - 0.25 = -0.173$.

(b) $T^{-1}(x) = (x - 1)/3$ and $|dT^{-1}/dx| = \tfrac13$, so $p_X(x) = \varphi\big(\tfrac{x - 1}{3}\big)\cdot\tfrac13 = \frac{1}{3\sqrt{2\pi}}e^{-(x-1)^2/18}$, the $\mathcal N(1, 9)$ density. ✓

</details>

## 3. blind rebuild

with no notes and no looking at your old code, in 60 minutes: rebuild **forward-mode autodiff**, the core of `mlfs.autodiff`. work in a scratch directory, starting from the *untouched* stub notebook and the topic's checks, so your own solved `exercise.py` stays where it is. from the repo root:

```bash
T=03-calculus-and-analysis/04-forward-mode-autodiff
mkdir -p /tmp/rebuild-0304
git show HEAD:$T/exercise.py > /tmp/rebuild-0304/exercise.py   # the stubs as shipped
cp $T/test_exercise.py /tmp/rebuild-0304/
```

(if you've committed your own solutions, regenerate the stubs instead: copy `$T/solutions/` to `/tmp/rebuild-0304/solutions/` and run `python scripts/stubify.py /tmp/rebuild-0304`.) open `/tmp/rebuild-0304/exercise.py` with `marimo edit`, implement the `Dual` class (arithmetic, elementary functions, powers, nesting-safe), `dual_parts`, `derivative`, `value_and_grad`, `grad`, `jvp` and `jacobian` from memory, and run the checks against your copy:

```bash
pytest /tmp/rebuild-0304/test_exercise.py
```

you pass if every check is green within the hour.

**harder variant** (for testing out of the chapter with confidence): in a further 60 minutes, rebuild `numerical_jacobian` and `numerical_hessian` (topic 02, stages 2–3) and the capstone's `newton_root`, `hessian` and `newton_minimize`, each in its own scratch directory, and pass:

```bash
pytest /tmp/rebuild-0302/test_exercise.py -k "stage2 or stage3"
pytest /tmp/rebuild-cap/test_capstone.py -k "stage1 or stage2 or stage3"
```

(run them from the repo root: the capstone imports `mlfs.autodiff`, which the tests resolve to the reference solution, so only the functions you rebuilt are being tested.)

## 4. readiness

move on if you got at least 80% of sections 1–2 (15 concept questions + 3 problems, counting each lettered part separately) and passed section 3. otherwise, revisit the topics behind the questions you missed: each answer above names the topic and idea it comes from.
