# sequences, limits & convergence

## why this matters

almost every algorithm in this course is a loop that produces a sequence: gradient descent produces weights $w_0, w_1, w_2, \dots$, EM produces likelihoods, power iteration produces eigenvector estimates, Newton's method produces roots. three questions come with every such loop, and you will be asking them for the rest of the course. **does it converge? how fast? and when do I stop?**

the answers are not obvious, and intuition fails in specific ways. "the loss barely changed in the last step" sounds like a good stopping rule, but the partial sums of $1 + \tfrac12 + \tfrac13 + \dots$ change by less and less forever and never settle. "this solver halves the error each step" and "this solver squares the error each step" sound similar, but starting from an error of $0.1$ the first needs about 30 steps to reach $10^{-10}$ and the second needs 4. and Newton's method, famous for being "quadratic", quietly becomes as slow as bisection on a perfectly innocent equation.

this topic gives you the language to make those statements precise (limits, Cauchy sequences, orders and rates of convergence) and a tool that reads the order and rate of convergence straight off a list of errors. you'll point that tool at three root finders you build yourself, and later at every optimizer in chapter 06.

## warm-up: predict first

1. the sequence $a_n = 2^{-n}$ is within $0.001$ of 0 from some index on. which one? the sequence $0.5, 0.0001, 0.4, 0.2, 0.01, \dots$ first gets within $0.001$ of 0 at its second term. is that the right answer for it?
2. the gaps between consecutive terms of a sequence go to zero. must the sequence converge?
3. method A multiplies the error by $\tfrac12$ each step; method B squares it. starting from an error of $0.1$, how many steps does each need to reach $10^{-12}$?
4. Newton's method is "quadratically convergent". do you expect it to be quadratic when solving $x^2 = 0$?
5. the secant method replaces Newton's derivative with the slope through the last two points. what order of convergence would you guess: 1, 2, or something in between?

## prerequisites check

- [ ] you can compute and manipulate logarithms, including $\log(a/b)$ and $\log a^q$ (→ ch 01 · topic 01)
- [ ] you know what a derivative is and can differentiate polynomials (→ ch 01 · topic 02)
- [ ] you can vectorize simple operations in NumPy (`cumsum`, slicing, boolean masks) (→ ch 00 · topic 02)
- [ ] you know machine epsilon, $\varepsilon \approx 2.2 \times 10^{-16}$, and why subtracting nearly equal numbers loses digits (→ ch 00 · topic 03)

## core ideas

### idea 1: what "converges" means

a sequence $a_0, a_1, a_2, \dots$ **converges to $L$** if its terms eventually stay as close to $L$ as you like. "as close as you like" is a challenge: someone names a tolerance $\varepsilon > 0$. "eventually stay" is your answer: an index $N$ after which *every* term is within $\varepsilon$:

$$a_n \to L \quad\iff\quad \text{for every } \varepsilon > 0 \text{ there is an } N \text{ with } |a_n - L| < \varepsilon \text{ for all } n \ge N.$$

the order of the quantifiers is the whole content. $N$ is allowed to depend on $\varepsilon$ (smaller tolerances need later indices), but it must work for every term after it, not just one. a sequence that touches $L$ once and wanders off hasn't converged (warm-up 1). the definition says nothing about any particular term: change the first million terms and the limit doesn't move.

**a proof in this style.** $a_n = 1 + (-1)^n/n$ converges to 1: $|a_n - 1| = 1/n$, and $1/n < \varepsilon$ whenever $n > 1/\varepsilon$, so $N = \lfloor 1/\varepsilon\rfloor + 1$ works. the proof is a recipe for $N$ given $\varepsilon$. the function $\varepsilon \mapsto N(\varepsilon)$ is already a measure of speed: here $N \approx 1/\varepsilon$, so each extra digit of accuracy costs ten times more terms. for $a_n = 2^{-n}$, $N \approx \log_2(1/\varepsilon)$: each extra digit costs about 3.3 terms. idea 3 makes this comparison systematic.

a computer can't check "for all $n$" or "for every $\varepsilon$". what it can do is take the first $K$ terms and a tolerance and report the $N$ that works *within those terms*. that's a useful diagnostic, and it's exactly as trustworthy as your belief that nothing strange happens after term $K$.

> **▶ build: stage 1.** implement `tail_index(a, limit, eps)`: the smallest $N$ such that $|a_n - L| < \varepsilon$ for every $n \ge N$ in the array. return $K$ (the length) if even the last term is outside the band. the trap: the *first* index inside the band is the wrong answer.

### idea 2: converging without knowing the limit

the definition needs $L$. but an algorithm doesn't know its own limit (if it did, it would be finished). we want a test that uses only the terms themselves.

a sequence is **Cauchy** if its terms eventually stay close *to each other*:

$$\text{for every } \varepsilon > 0 \text{ there is an } N \text{ with } |a_m - a_n| < \varepsilon \text{ for all } m, n \ge N.$$

equivalently, the **diameter** of the tail, $d_N = \sup_{m,n \ge N}|a_m - a_n|$, goes to 0. every convergent sequence is Cauchy: if all terms after $N$ are within $\varepsilon/2$ of $L$, any two of them are within $\varepsilon$ of each other. the converse is a property of the number system called **completeness**: in $\mathbb{R}$ (and in $\mathbb{R}^n$), every Cauchy sequence converges. that's what lets you prove a sequence converges without first guessing its limit. in the rationals it fails: Newton's iterates for $\sqrt2$ are rational and Cauchy, but their limit isn't rational. (chapter 03 · topic 07 meets completeness again, for spaces of functions.)

the condition is about **all pairs** $m, n \ge N$, not consecutive ones (warm-up 2). the partial sums of the harmonic series, $H_n = \sum_{k=1}^n 1/k$, have consecutive gaps $1/(n+1) \to 0$, but $H_m - H_n \approx \log(m/n)$, which is as large as you like if $m$ is large enough. the harmonic series diverges, slowly and smoothly, while every individual step looks like convergence. this is why "the change in the last step was tiny" is never, by itself, a proof of convergence. it's a heuristic that works when you also know the *rate* (idea 3).

for $K$ computed terms, the diameter of the tail from $N$ is just $\max(a_{N:}) - \min(a_{N:})$, and all $K$ diameters come from running maxima and minima taken from the right: one pass, no nested loops.

> **▶ build: stage 2.** implement `cauchy_diameters(a)`, returning $d_N = \max(a_{N:}) - \min(a_{N:})$ for every $N$, vectorized with `np.maximum.accumulate` and `np.minimum.accumulate` on the reversed array.

### idea 3: how fast: order and rate

let $e_k = |x_k - x^*|$ be the error of an iterative method after $k$ steps. two numbers describe how fast it shrinks: the **order** $q$ and the **rate** $C$, defined by

$$e_{k+1} \approx C\,e_k^{\,q}\quad\text{for large } k.$$

- **linear** convergence: $q = 1$ with $0 < C < 1$. each step multiplies the error by $C$, so you gain a fixed number of digits per step, $\log_{10}(1/C)$. bisection has $C = \tfrac12$ (one bit per step); gradient descent on a quadratic has $C = (\kappa - 1)/(\kappa + 1)$ with $\kappa$ the condition number (chapter 06). on a semilog plot the error is a straight line.
- **superlinear**: $e_{k+1}/e_k \to 0$. **quadratic** is the case $q = 2$: the number of correct digits roughly *doubles* each step. on a semilog plot the error curve bends downward like a cliff.
- **sublinear**: $e_{k+1}/e_k \to 1$, as for $e_k = 1/k$. the error still goes to zero, but each digit costs ten times more steps than the last. stochastic gradient descent lives here.

warm-up 3: halving from $0.1$ needs $k$ with $0.1 \cdot 2^{-k} \le 10^{-12}$, so $k = 37$. squaring gives $10^{-2}, 10^{-4}, 10^{-8}, 10^{-16}$: 4 steps. order matters far more than rate.

**reading $q$ and $C$ off the errors.** take logs of $e_{k+1} = Ce_k^q$ and $e_{k+2} = Ce_{k+1}^q$ and subtract: $\log(e_{k+2}/e_{k+1}) = q\log(e_{k+1}/e_k)$. so each triple of consecutive errors gives an estimate

$$q_k = \frac{\log(e_{k+2}/e_{k+1})}{\log(e_{k+1}/e_k)},$$

which doesn't need $C$ at all. once you trust a value of $q$, each pair gives a rate estimate $C_k = e_{k+1}/e_k^{\,q}$. both are *asymptotic*: early estimates can be far off, and the last few, once errors reach about $10^{-15}$, are polluted by rounding. look for the stretch in the middle where the estimates settle.

the errors need $x^*$. in a test you usually know it (you built the problem). in the wild, a common substitute is the last iterate, or the step lengths $|x_{k+1} - x_k|$, which for a superlinear method approximate $e_k$ well.

> **🔮 predict:** you run $q_k$ on errors $e_k = 1/k$, which converge sublinearly. what values of $q_k$ do you expect to see, and why can't $q$ alone tell sublinear from linear?

> **▶ build: stage 3.** implement `order_estimates(errors)` (length $n - 2$, raising `ValueError` for non-positive errors) and `rate_estimates(errors, q)` (length $n - 1$).

### idea 4: three root finders and their rates

to find $x^*$ with $f(x^*) = 0$:

**bisection.** if $f(a)$ and $f(b)$ have opposite signs and $f$ is continuous, there's a root in between (the intermediate value theorem, which, like idea 2, rests on completeness). evaluate $f$ at the midpoint $m$, keep whichever half still has a sign change, repeat. after $k$ midpoints the bracket has width $(b - a)/2^k$, so the error *bound* halves exactly each step: linear, $C = \tfrac12$. it needs one function evaluation per step and nothing else, and it can't fail. the actual error $|m_k - x^*|$ jumps around under that bound, because the root can sit anywhere in the bracket.

**Newton's method.** replace $f$ by its tangent line at $x_k$ and solve that instead:

$$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}.$$

why it's quadratic: Taylor's theorem around $x_k$ gives $0 = f(x^*) = f(x_k) + f'(x_k)(x^* - x_k) + \tfrac12f''(\xi)(x^* - x_k)^2$ for some $\xi$ between them. divide by $f'(x_k)$ and rearrange:

$$x_{k+1} - x^* = \frac{f''(\xi)}{2f'(x_k)}(x_k - x^*)^2, \qquad\text{so}\qquad e_{k+1} \approx \left|\frac{f''(x^*)}{2f'(x^*)}\right|e_k^2.$$

quadratic, with rate $|f''/2f'|$ at the root, *provided* $f'(x^*) \ne 0$ and you start close enough. both conditions matter. far from the root Newton can overshoot wildly or cycle; at a root where $f'(x^*) = 0$ the formula above breaks down (warm-up 4, and experiment 2).

**the secant method.** Newton needs $f'$. replace it with the slope through the last two iterates:

$$x_{k+1} = x_k - f(x_k)\,\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}.$$

one new evaluation of $f$ per step, no derivative. its error satisfies $e_{k+1} \approx C\,e_ke_{k-1}$: a product of the last two errors rather than a square. guess $e_{k+1} \approx Ae_k^q$ and substitute to find which $q$ is consistent with that recursion (on paper 3).

> **🔮 predict:** before you do on paper 3, guess the secant method's order. then: is it faster or slower than Newton *per function evaluation*, counting $f'$ as one evaluation?

> **▶ build: stage 4.** implement `bisection` (one new evaluation of $f$ per step; return the midpoints), `newton_1d` and `secant` (return all iterates, starting with the initial point(s)), each stopping at the first iterate with $|f| \le$ `tol`.

## worked example

Newton's method for $f(x) = x^2 - 2$, from $x_0 = 1$. the step is $x_{k+1} = x_k - (x_k^2 - 2)/(2x_k) = \tfrac12(x_k + 2/x_k)$, the Babylonian square root.

| $k$ | $x_k$ | $e_k = \lvert x_k - \sqrt2\rvert$ | $q_{k-2}$ | $e_k/e_{k-1}^2$ |
|---|---|---|---|---|
| 0 | $1$ | $4.142 \times 10^{-1}$ | | |
| 1 | $3/2 = 1.5$ | $8.579 \times 10^{-2}$ | | $0.5000$ |
| 2 | $17/12 = 1.41\overline{6}$ | $2.453 \times 10^{-3}$ | $2.2575$ | $0.3333$ |
| 3 | $577/408 = 1.4142157$ | $2.124 \times 10^{-6}$ | $1.9839$ | $0.3529$ |
| 4 | $665857/470832 = 1.41421356237469$ | $1.595 \times 10^{-12}$ | $1.9998$ | $0.3535$ |

the first order estimate is $\log(2.453 \times 10^{-3}/8.579 \times 10^{-2})/\log(8.579 \times 10^{-2}/4.142 \times 10^{-1}) = (-3.554)/(-1.575) = 2.2575$. the estimates settle at 2, and the rate estimates settle at $|f''/(2f')| = 2/(2 \cdot 2\sqrt2) = 1/(2\sqrt2) \approx 0.3536$, exactly as idea 4 predicts. the correct digits go 0, 1, 2, 5, 11: doubling. the tests check these iterates, these order estimates, and that last rate.

## on paper

1. **an ε–N proof.** show that $a_n = \dfrac{3n + 1}{n + 2} \to 3$: find an explicit $N(\varepsilon)$.

<details><summary>solution</summary>

$|a_n - 3| = \left|\dfrac{3n + 1 - 3n - 6}{n + 2}\right| = \dfrac{5}{n + 2}$. this is $< \varepsilon$ when $n + 2 > 5/\varepsilon$, so $N = \lceil 5/\varepsilon\rceil$ works (any larger $N$ works too). like $1/n$, the error is about $5/n$: each digit costs ten times more terms, which is sublinear convergence.

</details>

2. **Newton at a double root.** let $f(x) = (x - r)^2g(x)$ with $g(r) \ne 0$. show that Newton's error satisfies $e_{k+1} \approx \tfrac12e_k$ near $r$, and more generally $e_{k+1} \approx \tfrac{m-1}{m}e_k$ for a root of multiplicity $m$. what modified step restores quadratic convergence if you know $m$?

<details><summary>solution</summary>

for $f = (x - r)^mg$, $f' = m(x - r)^{m-1}g + (x - r)^mg'$, so

$$\frac{f}{f'} = \frac{(x - r)g}{mg + (x - r)g'} = \frac{x - r}{m} + O((x - r)^2).$$

then $x_{k+1} - r = (x_k - r) - f/f' = (x_k - r)(1 - \tfrac1m) + O((x_k - r)^2)$: linear with rate $(m - 1)/m$, which is $\tfrac12$ for a double root, as slow as bisection. the step $x_{k+1} = x_k - m\,f(x_k)/f'(x_k)$ cancels the leading term and is quadratic again.

</details>

3. **the secant method's order.** assume $e_{k+1} = Ce_ke_{k-1}$ and look for a solution with $e_{k+1} = Ae_k^q$. show that $q$ must satisfy $q^2 = q + 1$, so $q = (1 + \sqrt5)/2 \approx 1.618$.

<details><summary>solution</summary>

from $e_k = Ae_{k-1}^q$ we get $e_{k-1} = (e_k/A)^{1/q}$. substitute into the recursion: $Ae_k^q = Ce_k(e_k/A)^{1/q}$. matching powers of $e_k$ gives $q = 1 + 1/q$, i.e. $q^2 - q - 1 = 0$, whose positive root is the golden ratio $\varphi \approx 1.618$. per function evaluation, Newton (two evaluations, $f$ and $f'$) gains a factor $2^{1/2} \approx 1.41$ in digits per evaluation, the secant method $1.618$: the secant method is usually faster in practice.

</details>

4. **how many bisection steps?** on $[a, b]$ with tolerance $\tau$, how many midpoints does your `bisection` record (assuming no exact hit)? check it against the stage 4 test: $[1, 3]$ and $\tau = 10^{-3}$.

<details><summary>solution</summary>

after the $j$-th midpoint the half-width is $(b - a)/2^j$. we stop at the first $j$ with $(b - a)/2^j \le \tau$, i.e. $j = \lceil\log_2((b - a)/\tau)\rceil$. for width 2 and $\tau = 10^{-3}$: $\log_2 2000 = 10.97$, so 11 midpoints. the count depends only on the interval and the tolerance, never on $f$: bisection's cost is completely predictable.

</details>

## experiment

`lab.py` runs three experiments on your code:

1. a race between your `bisection`, `newton_1d` and `secant` on one cubic, with your `order_estimates` as the referee.
2. Newton's method at a root of multiplicity $m$, with $f$ computed in factored and in expanded form.
3. whether your `cauchy_diameters` can tell a convergent series from a divergent one, as the number of computed terms grows.

## check your understanding

1. why is the order of quantifiers in the ε–N definition ("for every ε there is an N") essential? what would "there is an N such that for every ε…" mean?

<details><summary>answer</summary>

the second version would require a single $N$ after which $|a_n - L| < \varepsilon$ for every $\varepsilon > 0$, i.e. $a_n = L$ exactly for all $n \ge N$: an eventually constant sequence. the real definition lets $N$ grow as $\varepsilon$ shrinks.

</details>

2. why can't a program prove that a sequence converges by looking at its first $K$ terms?

<details><summary>answer</summary>

convergence is a statement about every term, and terms $K + 1, K + 2, \dots$ are unconstrained by the first $K$. the harmonic partial sums look convergent at any finite window (experiment 3). a program can only provide evidence; a proof needs structure, such as a contraction rate or a monotone bounded sequence.

</details>

3. a method's errors are $10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}$. what order and rate? and for $10^{-2}, 10^{-4}, 10^{-8}, 10^{-16}$?

<details><summary>answer</summary>

the first is linear ($q = 1$) with rate $0.1$: one digit per step. the second is quadratic ($q = 2$) with rate $1$: the digits double.

</details>

4. (the predict callout in idea 3.) what does `order_estimates` return for $e_k = 1/k$?

<details><summary>answer</summary>

$q_k = \log\frac{k+1}{k+2}\big/\log\frac{k}{k+1} \to 1$. the order estimate can't distinguish sublinear from linear: both give $q \approx 1$. the rate does: for sublinear convergence $e_{k+1}/e_k \to 1$, while linear convergence has a rate strictly below 1.

</details>

5. the last two order estimates in your Newton run are 1.2 and $-3$. is Newton broken?

<details><summary>answer</summary>

almost certainly not: once the error reaches about $10^{-15}$, it's dominated by rounding, and ratios of rounding noise are meaningless. drop errors below about $10^{-14}$ (relative to the size of $x^*$) before estimating.

</details>

6. when would you choose bisection over Newton, despite its slower rate?

<details><summary>answer</summary>

when you have a sign-change bracket and need a guarantee: bisection always converges, with a known number of steps, no derivative, and no risk of jumping out of the region. a common hybrid (Brent's method) uses fast secant or interpolation steps while keeping a bisection bracket as a safety net.

</details>

7. why does completeness matter for an algorithm designer?

<details><summary>answer</summary>

it's what guarantees that a sequence whose terms bunch together actually has a limit in your space. for example, a contraction (each step shrinks distances by a factor $C < 1$) produces a Cauchy sequence, and completeness turns that into existence of a fixed point. that's the Banach fixed-point theorem, behind the convergence proofs of value iteration (chapter 23) and many others.

</details>

## common bugs

- **returning the first index inside the band.** symptom: `tail_index` passes on monotone sequences and fails on oscillating ones. look for the last index *outside* the band and add one.
- **adjacent gaps instead of tail diameters.** symptom: the harmonic series looks Cauchy. the diameter takes the max over all pairs in the tail.
- **upside-down order estimates.** symptom: a quadratic method reports $q = 0.5$. the *newer* ratio goes in the numerator.
- **trusting estimates at the rounding floor.** symptom: wild order estimates (negative, or huge) at the end of a run. filter out errors near machine precision first.
- **re-evaluating $f$ at the bracket ends.** symptom: bisection uses twice the function evaluations it should. keep $f(a)$ around and update it along with $a$.
- **one step too many.** symptom: Newton returns a duplicated last iterate, or takes a step after $|f| \le$ tol. check the stopping rule before stepping.

## going deeper

- Abbott (2015), *Understanding Analysis*, 2nd ed., Springer, chapters 1–2. a friendly, rigorous treatment of limits, Cauchy sequences and completeness. [doi:10.1007/978-1-4939-2712-8](https://doi.org/10.1007/978-1-4939-2712-8)
- Süli & Mayers (2003), *An Introduction to Numerical Analysis*, Cambridge University Press, chapter 1. bisection, Newton and the secant method with their convergence proofs.
- Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., Springer, §A.2 (rates of convergence). the definitions used for every optimizer in chapter 06. [doi:10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5)
- Brent (1973), *Algorithms for Minimization Without Derivatives*, Prentice-Hall, chapter 4. the bisection–secant hybrid behind `scipy.optimize.brentq`.

## connections

- ← builds on: logs and growth rates (ch 01 · topic 01), derivatives (ch 01 · topic 02), floating point (ch 00 · topic 03).
- → used in: Taylor's theorem and Newton's error (topic 03), forward-mode autodiff for exact derivatives (topic 04), step sizes and linear rates (topic 05), the Newton capstone, convergence rates of every optimizer (ch 06), limit theorems for random sequences (ch 04 · topic 05), and fixed-point arguments in reinforcement learning (ch 23).
