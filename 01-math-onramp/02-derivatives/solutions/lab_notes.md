# derivatives · lab notes

## experiment 1: how small should the step be?

**what people usually predict:** the smaller the better, since the derivative is a limit as $h \to 0$. at worst, very small steps stop improving things.

**what happens:** every curve is a U (on these axes, with $h$ shrinking to the right, a V). for $\sin$ at $x = 1$: the forward difference is best near **$h \approx 10^{-8}$–$10^{-9}$**, with an error around $3 \times 10^{-9}$. the central difference is best near **$h \approx 3 \times 10^{-6}$**, with an error around $3 \times 10^{-13}$, ten thousand times better. the second derivative is best near **$h \approx 10^{-4}$**, with an error around $10^{-9}$. at $h = 10^{-15}$ they are all bad: the forward and central differences are off by about $0.04$ and $0.01$ (several percent), and the second derivative is off by $0.84$, which is 100% of the answer (it comes out as exactly 0). for $e^x$ at $x = 2$, at $h = 10^{-15}$ the second derivative is off by about $10^{15}$. (for $x^3$ some steps give an error of exactly 0: with $x = 1.5$ and steps like $10^{-1}$, the arithmetic happens to be exact in floating point. ignore those lucky dips and look at the overall shape.)

**why:** two errors compete. truncation error, from using a finite step, shrinks as $h$ shrinks: in proportion to $h$ for forward differences (slope 1 on the log-log plot) and to $h^2$ for central and second differences (slope 2). rounding error grows as $h$ shrinks: the function values are only known to about $\varepsilon|f| \approx 10^{-16}$, their difference cancels the shared digits and keeps the noise, and dividing by $h$ (or by $h^2$ for the second derivative) amplifies it. that's slope $-1$ for the first derivatives and $-2$ for the second derivative on the left side of the plot. balancing the two gives the best steps: $\sqrt\varepsilon \approx 10^{-8}$ (forward), $\varepsilon^{1/3} \approx 5 \times 10^{-6}$ (central), $\varepsilon^{1/4} \approx 10^{-4}$ (second). that's exactly why `derivative` defaults to central with $h = 10^{-5}$ and `second_derivative` to $h = 10^{-4}$. you'll see the same U-curve again in chapter 02 for gradients with many variables, and it's why gradient checks are always done in double precision.

## experiment 2: the slope at a corner

**what people usually predict:** that all three methods say something like "undefined" or give an error, or that they all give $1$ (or all $-1$). and that just to the right of the corner, at $x = 0.000003$, the answer is $1$.

**what happens:** at $x = 0$ the forward difference says $+1$, the backward difference says $-1$, and the central difference says **exactly $0$**, with complete confidence. at $x = 0.000003$, where $|x|$ is smooth with slope exactly $1$, the central difference with $h = 10^{-5}$ says **$0.3$** (and the backward difference says $-0.4$). only beyond $x = h = 10^{-5}$ do all three give the right answer, $1$. the plot shows the central estimate as a ramp from $-1$ to $1$ across the whole interval $[-h, h]$, instead of a jump.

**why:** a finite difference doesn't look at the point $x$. it looks at the points $x \pm h$. at the corner, the central difference averages the slope on the left ($-1$) and the slope on the right ($+1$) and gets $0$. that's not a derivative (none exists there), it's a compromise. at $x = 3 \times 10^{-6}$ the left point $x - h$ is on the *other side* of the corner, so the estimate mixes the two slopes: $\frac{|x + h| - |x - h|}{2h} = \frac{2x}{2h} = 0.3$. so a finite difference is only trustworthy if the function is smooth on the whole interval $[x - h, x + h]$, not just at $x$. in ML this bites with ReLU, $\max(0, x)$, and anything with $|\cdot|$ or $\max$: a derivative check at a point where some input to a ReLU lies within $h$ of zero can "fail" even though your derivative is right. the fix is to choose check points away from the corners.

## experiment 3: can a derivative check miss a bug?

**what people usually predict:** a check is a check: both bugs get caught at every set of points, or at least at most of them.

**what happens:**

| check points | forgot inner: $\cos(x^2)$ | outer at $x$: $\cos(x)\cdot 2x$ | your `chain_rule` |
|---|---|---|---|
| (a) $\{0, 1\}$ | caught ($1.0$) | **missed** ($2 \times 10^{-10}$) | $2 \times 10^{-10}$ |
| (b) $\{0.5, \sqrt{\pi/2}\}$ | **missed** ($2 \times 10^{-10}$) | caught ($0.78$) | $2 \times 10^{-10}$ |
| (c) $\{0\}$ | caught ($1.0$) | **missed** ($0$) | $0$ |
| (d) 5 random points | caught ($1.5$) | caught ($0.78$) | $3 \times 10^{-10}$ |

each special set misses one of the two bugs. only the random points catch both.

**why:** a check only tests the formula at the points you give it, and each bug happens to equal the true derivative $2x\cos(x^2)$ at some points. "outer at $x$" gives $2x\cos x$, which equals $2x\cos(x^2)$ when $x = 0$ (both zero) or $x = 1$ (then $x^2 = x$). "forgot inner" gives $\cos(x^2)$, which equals $2x\cos(x^2)$ when $2x = 1$ ($x = 0.5$) or when $\cos(x^2) = 0$ ($x = \sqrt{\pi/2} \approx 1.2533$). special points like $0$ and $1$ are exactly where formulas simplify, which is exactly where bugs hide. the rules for the rest of the course: check at several random, "ugly" points (never only $0$, $1$, or points where something is symmetric), and in more dimensions use random non-symmetric matrices with different sizes. every test file in this repo follows those rules. look back at `test_exercise.py`: its check points are $-1.7, -0.6, 0.3, 0.9, 2.2$.
